//! Issue 884 T2b — the fork-config utilization CUDA source (the v6-q8 GEMM
//! generation), opt-in `prefill_mmq_v2`. Kept as its own file (not a third
//! string in `gemm_ternary_i8_mma_cuda_raw.rs`) to keep that file from
//! growing past its already-large size; the const is consumed only there.
//!
//! Split into a second NVRTC module (like `GEMM_Q8_CUDA_SRC`) so it stays
//! self-contained: its own `mma` helper + the cp.async helpers. No quantize
//! kernels here — the v6 arm reuses the `prefill_q8_act` module's
//! `quantize_rows_i8_q8_cuda_*` output (the same scratch) verbatim.

/// The v6-q8 GEMM (Issue 884 T2b): the single-plane q8 math of
/// `GEMM_Q8_CUDA_SRC`'s `gemm_i8_mma_tm128v4_q8_*` under the fork-config
/// scheduling class (the A2 lever — the pinned opponent's per-arch tile /
/// multi-stage / swizzled-smem family). The arithmetic SEQUENCE per output
/// element is v4-q8's exactly — same mma k-order, same per-group ascending
/// Fused/Strict fold, same `s_t` epilogue — so v6 ≡ v4-q8 BITWISE by
/// construction (the unit gate asserts 0 bit-diffs); only scheduling and
/// memory layout change:
///
/// 1. **Warp tile 32×32** (v4: 32×16): 4 nfrags per mfrag — every A smem
///    fragment load now feeds 8 `mma.m16n8k32.s8` per k-step instead of 4,
///    halving A-side smem traffic per MAC. 8 warps (4 row-bands × 2
///    token-warps), 256 threads, same 128×64 block/grid as v4.
/// 2. **XOR-perm swizzled A smem** (v4: stride-132 byte rows): the A
///    fragment read at `(row, word cw)` lands at physical word
///    `row*32 + ((cw & 3) | ((((cw >> 2) + row) & 7) << 2))`. v4's reads
///    banked as `(row + 8*ks + t) mod 32` — the `g + t` anti-diagonal, ~3.2
///    conflicts per load; the perm spreads the 32 lanes over all 32 banks
///    (`t | (((2ks + g) & 7) << 2)` is a bijection over the warp). A row
///    stride becomes exactly 128 B.
/// 3. **cp.async B stage** (`cp.async.cg.shared.global`, 16 B): the B slab
///    is raw q8 words — the one operand cp.async can serve directly; the
///    register round-trip (LDG + ST per chunk) becomes one fire-and-forget
///    LDGSTS issued at group start, overlapped with the group's mma work.
///    A keeps its in-register raw prefetch: the weights need the SWAR
///    bitplane expansion before the mma can consume them, which cp.async
///    cannot skip.
/// 4. **2 blocks/SM**: A is SINGLE-buffered (the SWAR store for group g+1
///    happens after a barrier post-compute, from the register prefetch) and
///    B stays double-buffered — 34,816 B dynamic smem, under the 48 KB
///    default, so `__launch_bounds__(256, 2)` holds 2 blocks/SM (16 warps,
///    v4's occupancy) without a smem opt-in.
/// 5. **LUT staging — MEASURED NEGATIVE, not shipped**: replacing the
///    per-word shift/mul SWAR chain with a 256-entry nibble-pair smem LUT
///    (1 LDS + 1 ST per word) ran 1.175x vs the SWAR form's 1.261x at
///    ffn_gate — the per-lane divergent LUT indices bank-conflict on the LSU
///    pipe (already loaded with fragment reads) while the SWAR chain rides
///    the idle ALU pipe. The record lives in the `V6_A_STORE` comment.
///
/// What is deliberately NOT here: per-arch
/// tile tables (one sm_89 config; the fleet is 4090s), and an anchor (hi/lo)
/// twin — B878 gate-verified the q8 numerics and the anchor path is the
/// env-off default, untouched.
///
/// **LDM variants (T2b continuation, Bench 880)** — `ldmatrix` fragment
/// loads, the fork-paired partner of the swizzle: `LDM=1` (v6tl) loads each
/// A fragment (m16n8k32: 4×u32/lane) with ONE `ldmatrix.m8n8.x4.shared.b16`
/// instead of 4 `LDS.32` — 32 LDS/group/warp → 8 ldmatrix, and the per-lane
/// address math collapses to one row address per (mf, ks). `LDM=2` (v6tb)
/// additionally loads each B fragment (2×u32/lane) with one
/// `ldmatrix.m8n8.x2.shared.b16` (2 LDS → 1). Both ride the SAME swizzle:
/// `V6_A_WORD` with a 16-B-aligned base yields 4 consecutive physical words
/// (the XOR perm is 16-B-run-preserving) and the 8 row addresses per matrix
/// hit 8 distinct 16-B blocks (`(ks*2 + r) & 7` over r = 0..7 —
/// conflict-free), which is exactly the layout contract `ldmatrix` needs.
/// Register→operand order is the hardware fragment layout, so the mma text
/// is untouched and bit-identity is by construction (the unit gate pins it).
/// MEASURED (interleaved, ffn_gate m=17408 n=5120 p=2048): v6tl 1.135×,
/// **v6tb 1.156× — the winner, now the `RIIR_PREFILL_MMQ_V2` arm route**
/// (v6t 1.107×, v6d 1.065×, plain v6 0.963×). Real but NOT the fork's
/// 47–53% class: 32.3% of int8 peak — the residual wall moves to the 4×
/// smem read amplification of BOTH operands at the 32×32 warp tile (see
/// B880 §next wall).
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "prefill_mmq_v2",
    not(target_os = "macos")
))]
pub(crate) const GEMM_MMQ_V2_CUDA_SRC: &str = r#"
// ---------------------------------------------------------------------------
// helpers — mma (the q8 module's text verbatim) + the cp.async trio (pure-PTX
// cvta; NVRTC-safe).
// ---------------------------------------------------------------------------

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

__device__ __forceinline__ unsigned int v6_smem_u32addr(const void* p)
{
    unsigned int addr;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
        : "=r"(addr) : "l"(p));
    return addr;
}

__device__ __forceinline__ void v6_cp_async16(void* smem_dst, const void* gmem_src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(v6_smem_u32addr(smem_dst)), "l"(gmem_src));
}

// 4-B variant for scalar slabs (group scales): .cg is 16-B-only; .ca carries
// 4/8/16. Sector cost is the containing 32-B sector either way — the win is
// issuing each address ONCE per group per block (v10) instead of per warp.
__device__ __forceinline__ void v6_cp_async4(void* smem_dst, const void* gmem_src)
{
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
        :: "r"(v6_smem_u32addr(smem_dst)), "l"(gmem_src));
}

__device__ __forceinline__ void v6_cp_async_commit()
{
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void v6_cp_async_wait_all()
{
    asm volatile("cp.async.wait_group 0;");
}

// ldmatrix fragment loads (LDM>=1): one instruction per mma operand
// fragment instead of per-word LDS. The address is the 16-B-aligned start
// of one fragment row (4 consecutive physical words under the V6_A_WORD
// perm — verified 16-B-aligned in both layouts: A rows are 128-B strides,
// B rows 144-B = 9×16). x4: lanes 0-7/8-15/16-23/24-31 address matrices
// 0/1/2/3; register r_i is matrix i in the hardware mma-fragment order, so
// the mma call text is unchanged. x2: lanes 0-15 address matrices 0/1,
// lanes 16-31 ignored.
__device__ __forceinline__ void v6_ldmatrix_x4(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    const unsigned int* p)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(v6_smem_u32addr(p)));
}

__device__ __forceinline__ void v6_ldmatrix_x2(
    unsigned int& r0, unsigned int& r1, const unsigned int* p)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(r0), "=r"(r1)
        : "r"(v6_smem_u32addr(p)));
}

// A-swizzle: logical word cw (0..31; each word = 4 k-bytes) of row r (0..127)
// -> physical u32 index in the 128x32-word single-buffer stage. Read lanes
// (quad row g, byte-lane t: cw = 8ks + t) bank as
//   t | (((2ks + g) & 7) << 2)
// — 32 distinct banks across the warp (v4's layout banked as
// (r + 8ks + t) mod 32, the ~3.2-way g+t anti-diagonal). Staging writes
// (row r, word w4, sub-word q: cw = 8*w4 + q) share the bijection.
#define V6_A_WORD(row, cw)                                                     \
    ((row) * 32 + (((cw) & 3) | (((((cw) >> 2) + (row)) & 7) << 2)))

// SWAR bitplane -> i8 bytes for one (row, word-column-block) slot: words
// pw/nw (raw pos/neg planes) expand to 8 u32 at logical cw = w4*8 + q — the
// q8 module's masked-SWAR math verbatim (t0/u0 disjoint-plane masking means
// the (1,1) bit case lands on 0x00), retargeted at the swizzled layout.
// Params are parenthesized at every use: the callers pass `pw0 & ~nw0`-
// style expressions and `>>` binds tighter than `&`.
// MEASURED ALTERNATIVE (kept as the record): a 256-entry nibble-pair LUT
// (1 LDS + 1 ST per word) ran 1.175x vs this form's 1.261x at ffn_gate —
// the per-lane divergent LUT indices bank-conflict on the LSU pipe (already
// loaded with fragment reads) while the shift/mul chain rides the idle ALU
// pipe. ALU expansion wins on sm_89 at this mix; do not re-try the LUT
// without changing the index distribution.
#define V6_A_STORE(buf, r, w4, tn, un)                                         \
    _Pragma("unroll")                                                          \
    for (int q = 0; q < 8; ++q) {                                              \
        unsigned int tp = ((((tn)) >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
        unsigned int up = ((((un)) >> (4 * q)) & 0xFu) * 0x00204081u & 0x01010101u;\
        up |= up << 1; up |= up << 2; up |= up << 4;                           \
        (buf)[V6_A_WORD((r), (w4) * 8 + q)] = tp | up;                         \
    }

// ---------------------------------------------------------------------------
// gemm_i8_mma_tm128v6_q8_{fused,strict} — the fork-config q8 GEMM family.
// Two tile geometries over the same body (the v5 GEMM_BODY_V5 parameterization
// precedent) + the LDM fragment-load selector (0 = per-word LDS, 1 = ldmatrix
// A, 2 = ldmatrix A+B):
//   TOKS=64  (v6):   256 thr, 8 warps (4 row-bands x 2 token-warps of 32),
//                    2 blocks/SM, B slab 64 toks — 34,816 B smem.
//   TOKS=128 (v6t):  512 thr, 16 warps (4 row-bands x 4 token-warps of 32),
//                    1 block/SM (the 2-block 64-reg cap would spill), B slab
//                    128 toks — 53,248 B smem (Rust-side opt-in).
// Warp tile is 32x32 in both: every A smem fragment load feeds 8 mma per
// k-step. v6t halves the block count — half the per-token A global staging
// traffic and epilogue passes — at the same A-dup (4x vs 2x smem reads per
// token, a wash as v5t measured). smem: a_buf 128*32 u32 (16 KB, single) +
// b_buf 2*TOKS*36 u32 (double). FUSED selects the fold: fmaf vs strict
// mul+add (the FoldMode selector).
// ---------------------------------------------------------------------------
#define GEMM_BODY_V6_Q8(NAME, FUSED, TOKS, NWARP, MINB, ADBL, LDM)             \
extern "C" __global__ void __launch_bounds__((NWARP) * 32, (MINB)) NAME(        \
    const unsigned int* __restrict__ pos_bits,   /* [m * words_per_row] */     \
    const unsigned int* __restrict__ neg_bits,                                 \
    const float* __restrict__ group_scale,       /* [m * groups] */            \
    const unsigned int* __restrict__ q_w,        /* [p * (n/4)] */             \
    const float* __restrict__ s_t,               /* [p] */                     \
    float* __restrict__ out,                     /* [p * m] */                 \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char v6_smem_raw[];               \
    unsigned int* const a_buf0 = (unsigned int*)v6_smem_raw;     /* 128*32 */  \
    unsigned int* const a_buf1 = a_buf0 + ((ADBL) ? 128 * 32 : 0);             \
    unsigned int* const b_buf0 = a_buf0 + ((ADBL) ? 2 * 128 * 32 : 128 * 32);  \
    unsigned int* const b_buf1 = b_buf0 + (TOKS) * 36;                         \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                                                  \
    const int g_id = lane >> 2;                                                \
    const int t_id = lane & 3;                                                 \
    const int warp_row = (wid / ((TOKS) / 32)) * 32; /* 4 row-bands of 32 */   \
    const int warp_n = wid % ((TOKS) / 32);      /* TOKS/32 tok-warps x 32 */  \
    const int row_base = blockIdx.x * 128 + warp_row;                          \
    const int blk_rows = blockIdx.x * 128;                                     \
    const int qwpr = n >> 2;                                                   \
                                                                               \
    /* staging: A = 512 (row, word-block) slots at 16/NWARP per thread;        \
       B = TOKS*8 16-byte chunks at TOKS/(NWARP*4) per thread. */              \
    const int a_r0 = tid >> 2;                                                 \
    const int a_w4 = tid & 3;                                                  \
    const int b_w4 = (tid & 7) << 2;                                           \
    const int b_t0 = tid >> 3;                                                 \
    unsigned int pw[16 / (NWARP)];                                             \
    unsigned int nw[16 / (NWARP)];                                             \
                                                                               \
    /* accumulators: 2 mfrag x 4 nfrag x 4 regs (single plane). */             \
    int dh[2][4][4];                                                           \
    float o[2][4][4];                                                          \
    _Pragma("unroll")                                                          \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                      \
        for (int nf = 0; nf < 4; ++nf) {                                       \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; ++c) { o[mf][nf][c] = 0.0f; }               \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* prologue: stage group 0 (A swizzled + B via reg/store), fetch A(1). */  \
    _Pragma("unroll")                                                          \
    for (int si = 0; si < (16 / (NWARP)); ++si) {                              \
        const int row = a_r0 + si * ((NWARP) * 8);                             \
        const long a_off = (long)((blk_rows + row < m) ? blk_rows + row : m - 1)\
            * words_per_row + a_w4;                                            \
        pw[si] = pos_bits[a_off];                                              \
        nw[si] = neg_bits[a_off];                                              \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int si = 0; si < (16 / (NWARP)); ++si) {                              \
        const int row = a_r0 + si * ((NWARP) * 8);                             \
        V6_A_STORE(a_buf0, row, a_w4, pw[si] & ~nw[si], nw[si] & ~pw[si]);     \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {                    \
        const int b_tk = b_t0 + si * ((NWARP) * 4);                            \
        int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                          \
        b_tr = b_tr < p ? b_tr : p - 1;                                        \
        *(uint4*)(b_buf0 + (unsigned)b_tk * 36u + (unsigned)b_w4) =            \
            *(const uint4*)(q_w + (long)b_tr * qwpr + b_w4);                   \
    }                                                                          \
    if (1 < groups_per_row) {                                                  \
        _Pragma("unroll")                                                      \
        for (int si = 0; si < (16 / (NWARP)); ++si) {                          \
            const int row = a_r0 + si * ((NWARP) * 8);                         \
            const long a_off = (long)((blk_rows + row < m) ? blk_rows + row : m - 1)\
                * words_per_row + a_w4;                                        \
            pw[si] = pos_bits[a_off + 4];                                      \
            nw[si] = neg_bits[a_off + 4];                                      \
        }                                                                      \
    }                                                                          \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        if (grp > 0) { v6_cp_async_wait_all(); }                               \
        __syncthreads();   /* A(grp) staged & B(grp) landed & g-1 done */      \
        unsigned int* const abuf =                                             \
            (ADBL) ? ((grp & 1) ? a_buf1 : a_buf0) : a_buf0;                   \
        unsigned int* const bbuf = (grp & 1) ? b_buf1 : b_buf0;                \
        unsigned int* const nbbuf = (grp & 1) ? b_buf0 : b_buf1;               \
        const bool has_next = (grp + 1) < groups_per_row;                      \
        if (has_next) {                                                        \
            /* B(grp+1): fire-and-forget LDGSTS, overlapped with this group.*/ \
            _Pragma("unroll")                                                  \
            for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {            \
                const int b_tk = b_t0 + si * ((NWARP) * 4);                    \
                int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                  \
                b_tr = b_tr < p ? b_tr : p - 1;                                \
                v6_cp_async16(nbbuf + (unsigned)b_tk * 36u + (unsigned)b_w4,   \
                    q_w + (long)b_tr * qwpr + b_w4 + ((long)(grp + 1) << 5));  \
            }                                                                  \
            v6_cp_async_commit();                                              \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 4; ++c) { dh[mf][nf][c] = 0; }             \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < 4; ++ks) {                                       \
            unsigned int aw[2][4];                                             \
            if ((LDM) >= 1) {                                                  \
                /* ldmatrix.x4 per (mf, ks): lane l addresses fragment row */ \
                /* (l&7) + 8·((l>>3)&1), word col ks*8 + 4·((l>>4)&1) — the */ \
                /* x4 register order IS the m16n8k32 A-fragment order.      */ \
                _Pragma("unroll")                                              \
                for (int mf = 0; mf < 2; ++mf) {                               \
                    const int la = warp_row + mf * 16 + (lane & 7)             \
                        + ((lane >> 3) & 1) * 8;                               \
                    const int lc = ks * 8 + ((lane >> 4) & 1) * 4;             \
                    v6_ldmatrix_x4(aw[mf][0], aw[mf][1], aw[mf][2],            \
                        aw[mf][3], &abuf[V6_A_WORD(la, lc)]);                  \
                }                                                              \
            } else {                                                           \
                _Pragma("unroll")                                              \
                for (int mf = 0; mf < 2; ++mf) {                               \
                    const int r0 = warp_row + mf * 16 + g_id;                  \
                    aw[mf][0] = abuf[V6_A_WORD(r0, ks * 8 + t_id)];            \
                    aw[mf][1] = abuf[V6_A_WORD(r0 + 8, ks * 8 + t_id)];        \
                    aw[mf][2] = abuf[V6_A_WORD(r0, ks * 8 + 4 + t_id)];        \
                    aw[mf][3] = abuf[V6_A_WORD(r0 + 8, ks * 8 + 4 + t_id)];    \
                }                                                              \
            }                                                                  \
            unsigned int bw[4][2];                                             \
            if ((LDM) == 2) {                                                  \
                /* ldmatrix.x2 per (nf, ks): lanes 0-7 address the B rows   */ \
                /* (n = warp_n*32 + nf*8 + lane&7, words ks*8..), lanes 8-15 */ \
                /* the +4-word matrix — r0/r1 = b0/b1 directly.             */ \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    const unsigned int bn =                                    \
                        (unsigned int)(warp_n * 32 + nf * 8 + (lane & 7)) * 36u\
                        + (unsigned int)(ks * 8 + ((lane >> 3) & 1) * 4);      \
                    v6_ldmatrix_x2(bw[nf][0], bw[nf][1], &bbuf[bn]);           \
                }                                                              \
            } else {                                                           \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    const unsigned int bb =                                    \
                        (unsigned int)(warp_n * 32 + nf * 8 + g_id) * 36u      \
                        + (unsigned int)(ks * 8 + t_id);                       \
                    bw[nf][0] = bbuf[bb];          bw[nf][1] = bbuf[bb + 4u];  \
                }                                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    mma_s8_m16n8k32(dh[mf][nf][0], dh[mf][nf][1],              \
                        dh[mf][nf][2], dh[mf][nf][3],                          \
                        aw[mf][0], aw[mf][1], aw[mf][2], aw[mf][3],            \
                        bw[nf][0], bw[nf][1]);                                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* A(grp+1) store: ADBL=1 → into the OTHER buffer (v4's schedule —    \
           the ks reads above hit abuf, so no barrier needed); ADBL=0 → into \
           the SINGLE buffer, after a barrier (every warp's ks loop is done   \
           reading a_buf). Regs still hold grp+1's words either way. */       \
        if (ADBL) {                                                            \
            if (has_next) {                                                    \
                unsigned int* const nabuf = (grp & 1) ? a_buf0 : a_buf1;       \
                _Pragma("unroll")                                              \
                for (int si = 0; si < (16 / (NWARP)); ++si) {                  \
                    const int row = a_r0 + si * ((NWARP) * 8);                 \
                    V6_A_STORE(nabuf, row, a_w4, pw[si] & ~nw[si],             \
                        nw[si] & ~pw[si]);                                     \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            __syncthreads();                                                   \
            if (has_next) {                                                    \
                _Pragma("unroll")                                              \
                for (int si = 0; si < (16 / (NWARP)); ++si) {                  \
                    const int row = a_r0 + si * ((NWARP) * 8);                 \
                    V6_A_STORE(a_buf0, row, a_w4, pw[si] & ~nw[si],            \
                        nw[si] & ~pw[si]);                                     \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* prefetch A(grp+2) AFTER the store (single pf register set); the     \
           group of compute ahead covers the LDG latency. */                   \
        if ((grp + 2) < groups_per_row) {                                      \
            const long g4 = (long)(grp + 2) * 4;                               \
            _Pragma("unroll")                                                  \
            for (int si = 0; si < (16 / (NWARP)); ++si) {                      \
                const int row = a_r0 + si * ((NWARP) * 8);                     \
                const long a_off = (long)((blk_rows + row < m) ? blk_rows + row : m - 1)\
                    * words_per_row + a_w4;                                    \
                pw[si] = pos_bits[a_off + g4];                                 \
                nw[si] = neg_bits[a_off + g4];                                 \
            }                                                                  \
        }                                                                      \
        /* fold — registers only; overlaps the other warps' A stores. The      \
           per-element sequence (ascending groups, fmaf vs mul+add) is the     \
           v4-q8 text verbatim — the bit-identity contract. */                 \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            const int r0 = row_base + mf * 16 + g_id;                          \
            const int rc0f = r0 < m ? r0 : m - 1;                              \
            const int rc8f = (r0 + 8) < m ? (r0 + 8) : m - 1;                  \
            const float sw0 = group_scale[(long)rc0f * groups_per_row + grp];  \
            const float sw8 = group_scale[(long)rc8f * groups_per_row + grp];  \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 4; ++nf) {                                   \
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
    /* epilogue: the v4 text verbatim (same 32-row bands), NWARP*32 stride. */ \
    float* stg = (float*)a_buf0;          /* [32][TOKS] floats reused */       \
    const int tok_blk = blockIdx.y * (TOKS);                                   \
    _Pragma("unroll")                                                          \
    for (int ch = 0; ch < 4; ++ch) {                                           \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            if ((warp_row + mf * 16) / 32 == ch) {                             \
                const int r_in = warp_row + mf * 16 + g_id - ch * 32;          \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    const int tk0 = warp_n * 32 + nf * 8 + t_id * 2;           \
                    stg[r_in * (TOKS) + tk0]     = o[mf][nf][0];               \
                    stg[r_in * (TOKS) + tk0 + 1] = o[mf][nf][1];               \
                    stg[(r_in + 8) * (TOKS) + tk0]     = o[mf][nf][2];         \
                    stg[(r_in + 8) * (TOKS) + tk0 + 1] = o[mf][nf][3];         \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const int row0 = blockIdx.x * 128 + ch * 32;                           \
        for (int idx = tid; idx < 32 * (TOKS); idx += (NWARP) * 32) {          \
            const int r = idx / (TOKS);                                        \
            const int tk = idx % (TOKS);                                       \
            const int tok = tok_blk + tk;                                      \
            if (tok < p && row0 + r < m) {                                     \
                out[(long)tok * m + row0 + r] = stg[r * (TOKS) + tk] * s_t[tok];\
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// TOKS=64: 256 threads (8 warps), 2 blocks/SM — 34,816 B smem (A single).
GEMM_BODY_V6_Q8(gemm_i8_mma_tm128v6_q8_fused, 1, 64, 8, 2, 0, 0)
GEMM_BODY_V6_Q8(gemm_i8_mma_tm128v6_q8_strict, 0, 64, 8, 2, 0, 0)
// TOKS=128 (v6t): 512 threads (16 warps), 1 block/SM — 53,248 B smem
// (A single; opt-in on the Rust side).
GEMM_BODY_V6_Q8(gemm_i8_mma_tm128v6t_q8_fused, 1, 128, 16, 1, 0, 0)
GEMM_BODY_V6_Q8(gemm_i8_mma_tm128v6t_q8_strict, 0, 128, 16, 1, 0, 0)
// TOKS=128 + double-A (v6d): v4's 1-barrier pipeline — A(g+1) lands in the
// other buffer during g's compute. 69,632 B smem (opt-in on the Rust side).
GEMM_BODY_V6_Q8(gemm_i8_mma_tm128v6d_q8_fused, 1, 128, 16, 1, 1, 0)
GEMM_BODY_V6_Q8(gemm_i8_mma_tm128v6d_q8_strict, 0, 128, 16, 1, 1, 0)
// v6tl (T2b continuation, Bench 880): v6t + ldmatrix.x4 A fragments — the
// named wall (32 LDS.32/group/warp -> 8 ldmatrix; the fork pairs its
// swizzle with ldmatrix).
GEMM_BODY_V6_Q8(gemm_i8_mma_tm128v6tl_q8_fused, 1, 128, 16, 1, 0, 1)
GEMM_BODY_V6_Q8(gemm_i8_mma_tm128v6tl_q8_strict, 0, 128, 16, 1, 0, 1)
// v6tb: + ldmatrix.x2 B fragments (2 LDS -> 1 per (nf,ks) per lane).
GEMM_BODY_V6_Q8(gemm_i8_mma_tm128v6tb_q8_fused, 1, 128, 16, 1, 0, 2)
GEMM_BODY_V6_Q8(gemm_i8_mma_tm128v6tb_q8_strict, 0, 128, 16, 1, 0, 2)

// ---------------------------------------------------------------------------
// gemm_i8_mma_tm128v7_q8_{fused,strict} — the fork-style GLOBAL-A family
// (Issue 884 T2b rung 3, Bench 881). The pinned opponent's A-path: NO A smem
// stage at all — each warp reads its fragment's mask words straight from
// GLOBAL per k-slice, expands them in registers with the same nibble->byte
// SWAR the staging used, and feeds the mma directly. What it deletes: the
// 16 KB A stage (halving smem -> 2 blocks/SM fits at TOKS=64), the A STS
// traffic, the A ldmatrix loads, and the second per-group barrier. What it
// pays: the expansion is duplicated per token-warp (4x the ALU work the
// staged form amortized), and the mask LDGs are consumed in the same k-step
// (a per-ks register pipeline hides one k-step of latency).
//
// Bit-identity by construction: the register expansion is the staging math
// verbatim (tp = nibble*0x00204081 & 0x01010101; up likewise then *= 0xFF —
// byte-identical to the 6-op shift-broadcast chain since 0x01*0xFF = 0xFF,
// 0x00*0xFF = 0x00 with no inter-byte carries), same tn/un disjoint masking
// (the (1,1) bit case lands 0x00), same mma operands, same fold.
//
// Pipeline: `cur`/`nxt` register banks hold the 8 mask words a lane needs
// per k-step (2 mf x {row g, row g+8} x {pos,neg}); the ks+1 loads issue
// before the current ks's ldmatrix+expand+mma, and grp+1's ks=0 words are
// fetched before the fold (the group_scale reads give them cover).
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int v7_expand(unsigned int tn, unsigned int un)
{
    // 4 mask bits (nibble) -> 4 expanded i8 bytes packed in a u32:
    // pos bit -> 0x01, neg bit -> 0xFF, unset -> 0x00. Identical output to
    // V6_A_STORE's form (the *= 0xFF replaces the shift-broadcast chain).
    unsigned int tp = (tn * 0x00204081u) & 0x01010101u;
    unsigned int up = (un * 0x00204081u) & 0x01010101u;
    up *= 0xFFu;
    return tp | up;
}

#define GEMM_BODY_V7_Q8(NAME, FUSED, TOKS, NWARP, MINB)                        \
extern "C" __global__ void __launch_bounds__((NWARP) * 32, (MINB)) NAME(        \
    const unsigned int* __restrict__ pos_bits,                                 \
    const unsigned int* __restrict__ neg_bits,                                 \
    const float* __restrict__ group_scale,       /* [m * groups] */            \
    const unsigned int* __restrict__ q_w,        /* [p * (n/4)] */             \
    const float* __restrict__ s_t,               /* [p] */                     \
    float* __restrict__ out,                     /* [p * m] */                 \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char v7_smem_raw[];               \
    unsigned int* const b_buf0 = (unsigned int*)v7_smem_raw;                   \
    unsigned int* const b_buf1 = b_buf0 + (TOKS) * 36;                         \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                                                  \
    const int g_id = lane >> 2;                                                \
    const int t_id = lane & 3;                                                 \
    const int warp_row = (wid / ((TOKS) / 32)) * 32; /* 4 row-bands of 32 */   \
    const int warp_n = wid % ((TOKS) / 32);      /* TOKS/32 tok-warps x 32 */  \
    const int row_base = blockIdx.x * 128 + warp_row;                          \
    const int blk_rows = blockIdx.x * 128;                                     \
    const int qwpr = n >> 2;                                                   \
                                                                               \
    /* B staging: TOKS*8 16-byte chunks at TOKS/(NWARP*4) per thread. */       \
    const int b_w4 = (tid & 7) << 2;                                           \
    const int b_t0 = tid >> 3;                                                 \
                                                                               \
    /* per-lane A mask words: [mf][row 0/1][plane 0/1], two ks banks. */       \
    unsigned int acur[2][2][2];                                                \
    unsigned int anxt[2][2][2];                                                \
                                                                               \
    /* accumulators + fold regs (v6 verbatim). */                              \
    int dh[2][4][4];                                                           \
    float o[2][4][4];                                                          \
    _Pragma("unroll")                                                          \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                      \
        for (int nf = 0; nf < 4; ++nf) {                                       \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; ++c) { o[mf][nf][c] = 0.0f; }               \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* B prologue: stage group 0 (plain LDG/STS like v6). */                   \
    _Pragma("unroll")                                                          \
    for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {                    \
        const int b_tk = b_t0 + si * ((NWARP) * 4);                            \
        int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                          \
        b_tr = b_tr < p ? b_tr : p - 1;                                        \
        *(uint4*)(b_buf0 + (unsigned)b_tk * 36u + (unsigned)b_w4) =            \
            *(const uint4*)(q_w + (long)b_tr * qwpr + b_w4);                   \
    }                                                                          \
                                                                               \
    /* A prologue: load (grp 0, ks 0). The v7_expand lane rows: */             \
    const int arow[2][2] = {                                                   \
        { (blk_rows + warp_row + g_id), (blk_rows + warp_row + g_id + 8) },    \
        { (blk_rows + warp_row + 16 + g_id), (blk_rows + warp_row + 16 + g_id + 8) } \
    };                                                                         \
    _Pragma("unroll")                                                          \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                      \
        for (int rp = 0; rp < 2; ++rp) {                                       \
            const int rr = arow[mf][rp] < m ? arow[mf][rp] : m - 1;            \
            const long off = (long)rr * words_per_row;                         \
            acur[mf][rp][0] = pos_bits[off];                                   \
            acur[mf][rp][1] = neg_bits[off];                                   \
        }                                                                      \
    }                                                                          \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        if (grp > 0) { v6_cp_async_wait_all(); }                               \
        __syncthreads();   /* B(grp) landed (and grp-1 compute done) */        \
        unsigned int* const bbuf = (grp & 1) ? b_buf1 : b_buf0;                \
        unsigned int* const nbbuf = (grp & 1) ? b_buf0 : b_buf1;               \
        const bool has_next = (grp + 1) < groups_per_row;                      \
        if (has_next) {                                                        \
            _Pragma("unroll")                                                  \
            for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {            \
                const int b_tk = b_t0 + si * ((NWARP) * 4);                    \
                int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                  \
                b_tr = b_tr < p ? b_tr : p - 1;                                \
                v6_cp_async16(nbbuf + (unsigned)b_tk * 36u + (unsigned)b_w4,   \
                    q_w + (long)b_tr * qwpr + b_w4 + ((long)(grp + 1) << 5));  \
            }                                                                  \
            v6_cp_async_commit();                                              \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                _Pragma("unroll")                                              \
                for (int c = 0; c < 4; ++c) { dh[mf][nf][c] = 0; }             \
            }                                                                  \
        }                                                                      \
        const long gbase = (long)grp * 4;                                      \
        _Pragma("unroll")                                                      \
        for (int ks = 0; ks < 4; ++ks) {                                       \
            if (ks < 3) {                                                      \
                /* issue A(grp, ks+1) before this k-step's work. */            \
                _Pragma("unroll")                                              \
                for (int mf = 0; mf < 2; ++mf) {                               \
                    _Pragma("unroll")                                          \
                    for (int rp = 0; rp < 2; ++rp) {                           \
                        const int rr = arow[mf][rp] < m ? arow[mf][rp] : m - 1;\
                        const long off = (long)rr * words_per_row + gbase + ks + 1; \
                        anxt[mf][rp][0] = pos_bits[off];                       \
                        anxt[mf][rp][1] = neg_bits[off];                       \
                    }                                                          \
                }                                                              \
            }                                                                  \
            unsigned int bw[4][2];                                             \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                const unsigned int bn =                                        \
                    (unsigned int)(warp_n * 32 + nf * 8 + (lane & 7)) * 36u    \
                    + (unsigned int)(ks * 8 + ((lane >> 3) & 1) * 4);          \
                v6_ldmatrix_x2(bw[nf][0], bw[nf][1], &bbuf[bn]);               \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                /* a0 = k-slice bytes t_id*4..+4 (nibble t_id), a2 = +16    */ \
                /* (nibble t_id+4); a1/a3 the same on row+8. The nibble     */ \
                /* extraction + expand reproduces V6_A_STORE's byte packing.*/ \
                const unsigned int tp0 = (acur[mf][0][0] >> (t_id * 4)) & 0xFu;\
                const unsigned int tn0 = (acur[mf][0][1] >> (t_id * 4)) & 0xFu;\
                const unsigned int tp4 = (acur[mf][0][0] >> (16 + t_id * 4)) & 0xFu; \
                const unsigned int tn4 = (acur[mf][0][1] >> (16 + t_id * 4)) & 0xFu; \
                const unsigned int u0 = tp0 & ~tn0, v0 = tn0 & ~tp0;           \
                const unsigned int u4 = tp4 & ~tn4, v4 = tn4 & ~tp4;           \
                const unsigned int a0 = v7_expand(u0, v0);                     \
                const unsigned int a2 = v7_expand(u4, v4);                     \
                const unsigned int tp1 = (acur[mf][1][0] >> (t_id * 4)) & 0xFu;\
                const unsigned int tn1 = (acur[mf][1][1] >> (t_id * 4)) & 0xFu;\
                const unsigned int tp5 = (acur[mf][1][0] >> (16 + t_id * 4)) & 0xFu; \
                const unsigned int tn5 = (acur[mf][1][1] >> (16 + t_id * 4)) & 0xFu; \
                const unsigned int u1 = tp1 & ~tn1, v1 = tn1 & ~tp1;           \
                const unsigned int u5 = tp5 & ~tn5, v5 = tn5 & ~tp5;           \
                const unsigned int a1 = v7_expand(u1, v1);                     \
                const unsigned int a3 = v7_expand(u5, v5);                     \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    mma_s8_m16n8k32(dh[mf][nf][0], dh[mf][nf][1],              \
                        dh[mf][nf][2], dh[mf][nf][3],                          \
                        a0, a1, a2, a3, bw[nf][0], bw[nf][1]);                 \
                }                                                              \
            }                                                                  \
            if (ks < 3) {                                                      \
                _Pragma("unroll")                                              \
                for (int mf = 0; mf < 2; ++mf) {                               \
                    _Pragma("unroll")                                          \
                    for (int rp = 0; rp < 2; ++rp) {                           \
                        acur[mf][rp][0] = anxt[mf][rp][0];                     \
                        acur[mf][rp][1] = anxt[mf][rp][1];                     \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* A(grp+1, ks=0): issued before the fold — the group_scale reads   */ \
        /* give it latency cover; consumed by the next group's first ks.    */ \
        if (has_next) {                                                        \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                _Pragma("unroll")                                              \
                for (int rp = 0; rp < 2; ++rp) {                               \
                    const int rr = arow[mf][rp] < m ? arow[mf][rp] : m - 1;    \
                    const long off = (long)rr * words_per_row + gbase + 4;     \
                    acur[mf][rp][0] = pos_bits[off];                           \
                    acur[mf][rp][1] = neg_bits[off];                           \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* fold — v6 text verbatim. */                                          \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            const int r0 = row_base + mf * 16 + g_id;                          \
            const int rc0f = r0 < m ? r0 : m - 1;                              \
            const int rc8f = (r0 + 8) < m ? (r0 + 8) : m - 1;                  \
            const float sw0 = group_scale[(long)rc0f * groups_per_row + grp];  \
            const float sw8 = group_scale[(long)rc8f * groups_per_row + grp];  \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 4; ++nf) {                                   \
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
    /* epilogue: v6's schedule over the B slab reused as staging (b_buf0 is */ \
    /* TOKS*36 words >= 32*TOKS floats; the loop's leading __syncthreads    */ \
    /* separates the last group's B reads from the stg writes).             */ \
    float* stg = (float*)b_buf0;           /* [32][TOKS] floats reused */       \
    const int tok_blk = blockIdx.y * (TOKS);                                   \
    _Pragma("unroll")                                                          \
    for (int ch = 0; ch < 4; ++ch) {                                           \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            if ((warp_row + mf * 16) / 32 == ch) {                             \
                const int r_in = warp_row + mf * 16 + g_id - ch * 32;          \
                _Pragma("unroll")                                              \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    const int tk0 = warp_n * 32 + nf * 8 + t_id * 2;           \
                    stg[r_in * (TOKS) + tk0]     = o[mf][nf][0];               \
                    stg[r_in * (TOKS) + tk0 + 1] = o[mf][nf][1];               \
                    stg[(r_in + 8) * (TOKS) + tk0]     = o[mf][nf][2];         \
                    stg[(r_in + 8) * (TOKS) + tk0 + 1] = o[mf][nf][3];         \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const int row0 = blockIdx.x * 128 + ch * 32;                           \
        for (int idx = tid; idx < 32 * (TOKS); idx += (NWARP) * 32) {          \
            const int r = idx / (TOKS);                                        \
            const int tk = idx % (TOKS);                                       \
            const int tok = tok_blk + tk;                                      \
            if (tok < p && row0 + r < m) {                                     \
                out[(long)tok * m + row0 + r] = stg[r * (TOKS) + tk] * s_t[tok];\
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// v7: TOKS=64, 256 threads (8 warps), 2 blocks/SM — 18,432 B smem (B only).
GEMM_BODY_V7_Q8(gemm_i8_mma_tm128v7_q8_fused, 1, 64, 8, 2)
GEMM_BODY_V7_Q8(gemm_i8_mma_tm128v7_q8_strict, 0, 64, 8, 2)
// v7t: TOKS=128, 512 threads (16 warps), 1 block/SM — 36,864 B (under the
// 48 KB default; no opt-in needed).
GEMM_BODY_V7_Q8(gemm_i8_mma_tm128v7t_q8_fused, 1, 128, 16, 1)
GEMM_BODY_V7_Q8(gemm_i8_mma_tm128v7t_q8_strict, 0, 128, 16, 1)

// ---------------------------------------------------------------------------
// gemm_i8_mma_tm128v8_q8_{fused,strict} — the FORMAT-RUNG family (Issue 884
// T2b rung 4, Plan 572): v7's global-A structure with the A operand stored as
// PACKED-2-BIT Q2_0-CODE words instead of dual bitplanes. This is the fork's
// actual lever (B881): the decode becomes 8 ALU-pipe ops (prmt/and/shr — the
// LOP3 class; 9 pre-903-T2a — the w-mask folds into t1/t2 by composition)
// per 8 weights from the same two LDG.32 the v7 pipeline already
// issues, vs the bitplane SWAR's ~15 ops per 4 weights that saturated the INT
// pipe at 2x the mma rate.
//
// Rest format (Plan 572): `packed_w: u32[m * n/16]`, k-contiguous LSB-first —
// the 2-bit code of weight k lives at bits [2k mod 32, +2) of word k/16.
// Codes are the Q2_0 table (riir-infer-core q2_0.rs, the bridge's inverse):
// 00 -> -1, 01 -> 0, 10 -> +1; code 11 is never emitted by the transform (the
// (pos,neg)=(1,1) bit case folds to zero, the SAME fold the masked SWAR
// applies — the unit gate pins it on non-disjoint planes).
//
// Decode per (row, ks), lane t: L0 = packed[row, grp*8 + ks*2] carries the
// a0 code byte at byte lane t; L2 = packed[row, grp*8 + ks*2 + 1] the a2
// code byte. 8 ops (Issue 903 T2a: was 9 — the w=&0xFFFF mask folds into
// the t1/t2 masks by composition, w&M2 == P&(M1&M2) bit-for-bit):
//   P  = prmt(L0, L2, SEL)   SEL = 0x40 + t*0x11 — the two code bytes
//                            adjacent: a0's in byte 0, a2's in byte 1
//   T1 = P & 0x00003333      even codes at nibble pitch (the folded
//                            0x0000FFFF & 0x33333333; selector s2/s3 = 0,
//                            so prmt stays in default byte-index mode)
//   E  = prmt(Q2T, 0, T1)    Q2T = 0x000100FF (bytes [0xFF,0x00,0x01,0x00]):
//                            code 0 -> 0xFF (-1), 1 -> 0x00, 2 -> 0x01 (+1);
//                            byte 3 = 0x00 is the defensive fold
//   T2 = (P >> 2) & 0x00003333   odd codes (the folded shifted mask)
//   O  = prmt(Q2T, 0, T2)
//   a0 = prmt(E, O, 0x5140)  constant-selector byte merges (pure shuffle):
//                            [E0 O0 E1 O1] = k-ascending a0
//   a2 = prmt(E, O, 0x7362)  [E2 O2 E3 O3] = k-ascending a2
// Bit-identity by construction: the table reproduces V6_A_STORE/v7_expand's
// byte packing for every reachable code; the merges restore k order; the mma
// operands, fold and epilogue are the v7 text verbatim. Pipeline registers
// HALVE vs v7 (one packed pair per (mf,row,ks) bank vs a pos/neg pair).
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int v8_prmt(
    unsigned int a, unsigned int b, unsigned int s)
{
    unsigned int r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
    return r;
}

// One packed word pair (one row, one k-step) -> the row's two A-fragment
// registers (a0 = k [t*4, t*4+4), a2 = k [t*4+16, +4) of the group window),
// each 4 i8 bytes in {0x00, 0x01, 0xFF} — the exact packing v7_expand and
// V6_A_STORE produce.
__device__ __forceinline__ void v8_decode_pair(
    unsigned int l0, unsigned int l2, unsigned int sel,
    unsigned int& a0, unsigned int& a2)
{
    const unsigned int p = v8_prmt(l0, l2, sel);
    const unsigned int t1 = p & 0x00003333u;
    const unsigned int e = v8_prmt(0x000100FFu, 0u, t1);
    const unsigned int t2 = (p >> 2) & 0x00003333u;
    const unsigned int o = v8_prmt(0x000100FFu, 0u, t2);
    a0 = v8_prmt(e, o, 0x5140u);
    a2 = v8_prmt(e, o, 0x7362u);
}

#define GEMM_BODY_V8_Q8(NAME, FUSED, TOKS, NWARP, MINB)                        \
extern "C" __global__ void __launch_bounds__((NWARP) * 32, (MINB)) NAME(        \
    const unsigned int* __restrict__ packed_w,   /* [m * (n/16)] Q2_0 codes */  \
    const float* __restrict__ group_scale,       /* [m * groups] */            \
    const unsigned int* __restrict__ q_w,        /* [p * (n/4)] */             \
    const float* __restrict__ s_t,               /* [p] */                     \
    float* __restrict__ out,                     /* [p * m] */                 \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char v8_smem_raw[];               \
    unsigned int* const b_buf0 = (unsigned int*)v8_smem_raw;                   \
    unsigned int* const b_buf1 = b_buf0 + (TOKS) * 36;                         \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                                                  \
    const int g_id = lane >> 2;                                                \
    const int t_id = lane & 3;                                                 \
    const int warp_row = (wid / ((TOKS) / 32)) * 32; /* 4 row-bands of 32 */   \
    const int warp_n = wid % ((TOKS) / 32);      /* TOKS/32 tok-warps x 32 */  \
    const int row_base = blockIdx.x * 128 + warp_row;                          \
    const int blk_rows = blockIdx.x * 128;                                     \
    const int qwpr = n >> 2;                                                   \
                                                                               \
    /* B staging: TOKS*8 16-byte chunks at TOKS/(NWARP*4) per thread. */       \
    const int b_w4 = (tid & 7) << 2;                                           \
    const int b_t0 = tid >> 3;                                                 \
                                                                               \
    /* per-lane A packed word pair: [mf][row 0/1][word 0/1], two ks banks. */  \
    const unsigned int sel = 0x40u + (unsigned int)t_id * 0x11u;               \
    unsigned int acur[2][2][2];                                                \
    unsigned int anxt[2][2][2];                                                \
                                                                               \
    /* accumulators + fold regs (v7 verbatim). */                              \
    int dh[2][4][4];                                                           \
    float o[2][4][4];                                                          \
    _Pragma("unroll")                                                      \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                  \
        for (int nf = 0; nf < 4; ++nf) {                                       \
            _Pragma("unroll")                                              \
            for (int c = 0; c < 4; ++c) { o[mf][nf][c] = 0.0f; }               \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* B prologue: stage group 0 (plain LDG/STS like v7). */                   \
    _Pragma("unroll")                                                      \
    for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {                    \
        const int b_tk = b_t0 + si * ((NWARP) * 4);                            \
        int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                          \
        b_tr = b_tr < p ? b_tr : p - 1;                                        \
        *(uint4*)(b_buf0 + (unsigned)b_tk * 36u + (unsigned)b_w4) =            \
            *(const uint4*)(q_w + (long)b_tr * qwpr + b_w4);                   \
    }                                                                          \
                                                                               \
    /* A prologue: load (grp 0, ks 0) — one packed word pair per (mf, row). */ \
    const int arow[2][2] = {                                                   \
        { (blk_rows + warp_row + g_id), (blk_rows + warp_row + g_id + 8) },    \
        { (blk_rows + warp_row + 16 + g_id), (blk_rows + warp_row + 16 + g_id + 8) } \
    };                                                                         \
    _Pragma("unroll")                                                      \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                  \
        for (int rp = 0; rp < 2; ++rp) {                                       \
            const int rr = arow[mf][rp] < m ? arow[mf][rp] : m - 1;            \
            const long off = (long)rr * words_per_row;                         \
            acur[mf][rp][0] = packed_w[off];                                   \
            acur[mf][rp][1] = packed_w[off + 1];                               \
        }                                                                      \
    }                                                                          \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        if (grp > 0) { v6_cp_async_wait_all(); }                               \
        __syncthreads();   /* B(grp) landed (and grp-1 compute done) */        \
        unsigned int* const bbuf = (grp & 1) ? b_buf1 : b_buf0;                \
        unsigned int* const nbbuf = (grp & 1) ? b_buf0 : b_buf1;               \
        const bool has_next = (grp + 1) < groups_per_row;                      \
        if (has_next) {                                                        \
            _Pragma("unroll")                                              \
            for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {            \
                const int b_tk = b_t0 + si * ((NWARP) * 4);                    \
                int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                  \
                b_tr = b_tr < p ? b_tr : p - 1;                                \
                v6_cp_async16(nbbuf + (unsigned)b_tk * 36u + (unsigned)b_w4,   \
                    q_w + (long)b_tr * qwpr + b_w4 + ((long)(grp + 1) << 5));  \
            }                                                                  \
            v6_cp_async_commit();                                              \
        }                                                                      \
        _Pragma("unroll")                                                  \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            _Pragma("unroll")                                              \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                _Pragma("unroll")                                          \
                for (int c = 0; c < 4; ++c) { dh[mf][nf][c] = 0; }             \
            }                                                                  \
        }                                                                      \
        /* packed words per row: 8 per 128-k group (2 per k-step). */          \
        const long gbase = (long)grp * 8;                                      \
        _Pragma("unroll")                                                  \
        for (int ks = 0; ks < 4; ++ks) {                                       \
            if (ks < 3) {                                                      \
                /* issue A(grp, ks+1) before this k-step's work. */            \
                _Pragma("unroll")                                          \
                for (int mf = 0; mf < 2; ++mf) {                               \
                    _Pragma("unroll")                                      \
                    for (int rp = 0; rp < 2; ++rp) {                           \
                        const int rr = arow[mf][rp] < m ? arow[mf][rp] : m - 1;\
                        const long off = (long)rr * words_per_row + gbase      \
                            + (long)(ks * 2 + 2);                              \
                        anxt[mf][rp][0] = packed_w[off];                       \
                        anxt[mf][rp][1] = packed_w[off + 1];                   \
                    }                                                          \
                }                                                              \
            }                                                                  \
            unsigned int bw[4][2];                                             \
            _Pragma("unroll")                                              \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                const unsigned int bn =                                        \
                    (unsigned int)(warp_n * 32 + nf * 8 + (lane & 7)) * 36u    \
                    + (unsigned int)(ks * 8 + ((lane >> 3) & 1) * 4);          \
                v6_ldmatrix_x2(bw[nf][0], bw[nf][1], &bbuf[bn]);               \
            }                                                                  \
            _Pragma("unroll")                                              \
            for (int mf = 0; mf < 2; ++mf) {                               \
                /* 8 ALU-pipe ops per row pair — the LOP3-class decode. */ \
                unsigned int a0, a1, a2, a3;                                   \
                v8_decode_pair(acur[mf][0][0], acur[mf][0][1], sel, a0, a2);   \
                v8_decode_pair(acur[mf][1][0], acur[mf][1][1], sel, a1, a3);   \
                _Pragma("unroll")                                          \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    mma_s8_m16n8k32(dh[mf][nf][0], dh[mf][nf][1],              \
                        dh[mf][nf][2], dh[mf][nf][3],                          \
                        a0, a1, a2, a3, bw[nf][0], bw[nf][1]);                 \
                }                                                              \
            }                                                                  \
            if (ks < 3) {                                                      \
                _Pragma("unroll")                                          \
                for (int mf = 0; mf < 2; ++mf) {                               \
                    _Pragma("unroll")                                      \
                    for (int rp = 0; rp < 2; ++rp) {                           \
                        acur[mf][rp][0] = anxt[mf][rp][0];                     \
                        acur[mf][rp][1] = anxt[mf][rp][1];                     \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* A(grp+1, ks=0): issued before the fold — v7's schedule verbatim. */ \
        if (has_next) {                                                        \
            _Pragma("unroll")                                              \
            for (int mf = 0; mf < 2; ++mf) {                               \
                _Pragma("unroll")                                          \
                for (int rp = 0; rp < 2; ++rp) {                           \
                    const int rr = arow[mf][rp] < m ? arow[mf][rp] : m - 1;    \
                    const long off = (long)rr * words_per_row + gbase + 8;     \
                    acur[mf][rp][0] = packed_w[off];                           \
                    acur[mf][rp][1] = packed_w[off + 1];                       \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* fold — v7 text verbatim (the bit-identity contract). */             \
        _Pragma("unroll")                                                  \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            const int r0 = row_base + mf * 16 + g_id;                          \
            const int rc0f = r0 < m ? r0 : m - 1;                              \
            const int rc8f = (r0 + 8) < m ? (r0 + 8) : m - 1;                  \
            const float sw0 = group_scale[(long)rc0f * groups_per_row + grp];  \
            const float sw8 = group_scale[(long)rc8f * groups_per_row + grp];  \
            _Pragma("unroll")                                              \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                _Pragma("unroll")                                          \
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
    /* epilogue: v7's schedule over the B slab reused as staging. */           \
    float* stg = (float*)b_buf0;           /* [32][TOKS] floats reused */       \
    const int tok_blk = blockIdx.y * (TOKS);                                   \
    _Pragma("unroll")                                                      \
    for (int ch = 0; ch < 4; ++ch) {                                           \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                  \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            if ((warp_row + mf * 16) / 32 == ch) {                             \
                const int r_in = warp_row + mf * 16 + g_id - ch * 32;          \
                _Pragma("unroll")                                          \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    const int tk0 = warp_n * 32 + nf * 8 + t_id * 2;           \
                    stg[r_in * (TOKS) + tk0]     = o[mf][nf][0];               \
                    stg[r_in * (TOKS) + tk0 + 1] = o[mf][nf][1];               \
                    stg[(r_in + 8) * (TOKS) + tk0]     = o[mf][nf][2];         \
                    stg[(r_in + 8) * (TOKS) + tk0 + 1] = o[mf][nf][3];         \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const int row0 = blockIdx.x * 128 + ch * 32;                           \
        for (int idx = tid; idx < 32 * (TOKS); idx += (NWARP) * 32) {          \
            const int r = idx / (TOKS);                                        \
            const int tk = idx % (TOKS);                                       \
            const int tok = tok_blk + tk;                                      \
            if (tok < p && row0 + r < m) {                                     \
                out[(long)tok * m + row0 + r] = stg[r * (TOKS) + tk] * s_t[tok];\
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// v8: TOKS=64, 256 threads (8 warps), 2 blocks/SM — 18,432 B smem (B only).
GEMM_BODY_V8_Q8(gemm_i8_mma_tm128v8_q8_fused, 1, 64, 8, 2)
GEMM_BODY_V8_Q8(gemm_i8_mma_tm128v8_q8_strict, 0, 64, 8, 2)
// v8t: TOKS=128, 512 threads (16 warps), 1 block/SM — 36,864 B (the primary
// arm; v7t's geometry, which beat v7's by 10%).
GEMM_BODY_V8_Q8(gemm_i8_mma_tm128v8t_q8_fused, 1, 128, 16, 1)
GEMM_BODY_V8_Q8(gemm_i8_mma_tm128v8t_q8_strict, 0, 128, 16, 1)

// ---------------------------------------------------------------------------
// gemm_i8_mma_tm128v9_q8_{fused,strict} — the STAGED-PACKED rung (Issue 884
// T2b rung 5, Plan 573): the v6tb staged structure over the v8 packed format.
// B882 isolated the wall — two formats at v8's GLOBAL-A structure lose the
// same way (per-lane 8 LDG.32/lane/k-step, every row's words re-read by every
// token-warp, register-walled pipeline at 128 regs) — while v6tb's one-pass A
// stage pays a SWAR round trip (LDG → decode → STS) plus a second barrier per
// group. The packed format is the thing that makes A cp.async-SERVABLE (raw
// code words; the bitplane format never was), so v9 takes both sides:
//
// 1. A staged ONCE per block via cp.async: 256 x 16-B chunks per k-group (the
//    whole 4 KB of code data), double-buffered — v8's per-lane global stream
//    collapses ~30x in instruction count and the token-warp re-read dies; ONE
//    barrier per group (v6tb: two — its A store had to wait out the reads).
// 2. Per-use fragment decode via v8_decode_pair (B882 measured the decode ALU
//    ~free at HIGHER amplification than v9's 4 token-warps).
// 3. Fragment loads = LDS.32 quad-broadcast: the 4 lanes of a mma quad need
//    different BYTES of the SAME code word (byte lane t of a0/a2 — see the
//    v8 decode note), and ldmatrix distributes WHOLE words to lane columns,
//    so it cannot serve this granularity. The broadcast is conflict-free at
//    the 12-word row stride (12·g mod 32 over g=0..7 = {0,12,24,4,16,28,8,20}
//    — 8 distinct banks) and 48-B rows keep every cp.async dst 16-B aligned.
//    Pad words 8..12 are never written and never read.
// 4. Register relief: v8's 16-deep acur/anxt pipeline becomes 8 transient LDS
//    words — v7 AND v8 sat at exactly 128 regs; v9 should sit under it.
//
// smem: acode 2 x 128 x 12 u32 (12,288 B) + b_buf 2 x TOKS x 36 u32 — v9
// (TOKS=64): 30,720 B under the default, 2 blocks/SM; v9t (TOKS=128):
// 49,152 B, 1 block/SM (Rust-side opt-in, as v6t). Same packed_w input,
// same decode, same mma/fold/epilogue sequence — bit-identity by construction
// (the unit gate pins it, incl. the non-disjoint-plane fold).
// ---------------------------------------------------------------------------
#define GEMM_BODY_V9_Q8(NAME, FUSED, TOKS, NWARP, MINB)                        \
extern "C" __global__ void __launch_bounds__((NWARP) * 32, (MINB)) NAME(        \
    const unsigned int* __restrict__ packed_w,   /* [m * (n/16)] Q2_0 codes */  \
    const float* __restrict__ group_scale,       /* [m * groups] */            \
    const unsigned int* __restrict__ q_w,        /* [p * (n/4)] */             \
    const float* __restrict__ s_t,               /* [p] */                     \
    float* __restrict__ out,                     /* [p * m] */                 \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char v9_smem_raw[];               \
    unsigned int* const acode0 = (unsigned int*)v9_smem_raw; /* 128*12 */      \
    unsigned int* const acode1 = acode0 + 128 * 12;                            \
    unsigned int* const b_buf0 = acode0 + 2 * 128 * 12;                        \
    unsigned int* const b_buf1 = b_buf0 + (TOKS) * 36;                         \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                                                  \
    const int g_id = lane >> 2;                                                \
    const int t_id = lane & 3;                                                 \
    const int warp_row = (wid / ((TOKS) / 32)) * 32; /* 4 row-bands of 32 */   \
    const int warp_n = wid % ((TOKS) / 32);      /* TOKS/32 tok-warps x 32 */  \
    const int row_base = blockIdx.x * 128 + warp_row;                          \
    const int blk_rows = blockIdx.x * 128;                                     \
    const int qwpr = n >> 2;                                                   \
                                                                               \
    /* B staging: TOKS*8 16-byte chunks at TOKS/(NWARP*4) per thread (v8). */  \
    const int b_w4 = (tid & 7) << 2;                                           \
    const int b_t0 = tid >> 3;                                                 \
    const unsigned int sel = 0x40u + (unsigned int)t_id * 0x11u;               \
                                                                               \
    /* accumulators + fold regs (v8 verbatim). */                              \
    int dh[2][4][4];                                                           \
    float o[2][4][4];                                                          \
    _Pragma("unroll")                                                      \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                  \
        for (int nf = 0; nf < 4; ++nf) {                                       \
            _Pragma("unroll")                                              \
            for (int c = 0; c < 4; ++c) { o[mf][nf][c] = 0.0f; }               \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* prologue: B(0) plain LDG/STS (v8); A(0) via cp.async, 128 rows x     */ \
    /* 2 16-B chunks = 256 chunks (1 per thread at both geometries).        */ \
    _Pragma("unroll")                                                      \
    for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {                    \
        const int b_tk = b_t0 + si * ((NWARP) * 4);                            \
        int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                          \
        b_tr = b_tr < p ? b_tr : p - 1;                                        \
        *(uint4*)(b_buf0 + (unsigned)b_tk * 36u + (unsigned)b_w4) =            \
            *(const uint4*)(q_w + (long)b_tr * qwpr + b_w4);                   \
    }                                                                          \
    for (int c = tid; c < 256; c += (NWARP) * 32) {                            \
        const int row = c >> 1;                                                \
        const int half = c & 1;                                                \
        const int gr = (blk_rows + row < m) ? blk_rows + row : m - 1;          \
        v6_cp_async16(acode0 + row * 12 + half * 4,                            \
            packed_w + (long)gr * words_per_row + half * 4);                   \
    }                                                                          \
    v6_cp_async_commit();                                                      \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        /* A(0) is in flight at grp 0 — the wait is unconditional (v8 could */ \
        /* skip it: its B(0) rode plain STS, no commit group outstanding). */  \
        v6_cp_async_wait_all();                                                \
        __syncthreads();   /* A(grp)+B(grp) landed & grp-1 compute done */     \
        unsigned int* const abuf = (grp & 1) ? acode1 : acode0;                \
        unsigned int* const nabuf = (grp & 1) ? acode0 : acode1;               \
        unsigned int* const bbuf = (grp & 1) ? b_buf1 : b_buf0;                \
        unsigned int* const nbbuf = (grp & 1) ? b_buf0 : b_buf1;               \
        const bool has_next = (grp + 1) < groups_per_row;                      \
        if (has_next) {                                                        \
            /* B(grp+1) then A(grp+1) — both in flight across this group. */  \
            _Pragma("unroll")                                              \
            for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {            \
                const int b_tk = b_t0 + si * ((NWARP) * 4);                    \
                int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                  \
                b_tr = b_tr < p ? b_tr : p - 1;                                \
                v6_cp_async16(nbbuf + (unsigned)b_tk * 36u + (unsigned)b_w4,   \
                    q_w + (long)b_tr * qwpr + b_w4 + ((long)(grp + 1) << 5));  \
            }                                                                  \
            v6_cp_async_commit();                                              \
            for (int c = tid; c < 256; c += (NWARP) * 32) {                    \
                const int row = c >> 1;                                        \
                const int half = c & 1;                                        \
                const int gr =                                                 \
                    (blk_rows + row < m) ? blk_rows + row : m - 1;             \
                v6_cp_async16(nabuf + row * 12 + half * 4,                     \
                    packed_w + (long)gr * words_per_row                        \
                        + (long)(grp + 1) * 8 + half * 4);                     \
            }                                                                  \
            v6_cp_async_commit();                                              \
        }                                                                      \
        _Pragma("unroll")                                                  \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            _Pragma("unroll")                                              \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                _Pragma("unroll")                                          \
                for (int c = 0; c < 4; ++c) { dh[mf][nf][c] = 0; }             \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                  \
        for (int ks = 0; ks < 4; ++ks) {                                       \
            /* A code words: 8 LDS.32, quad-broadcast (conflict-free at    */ \
            /* the 12-word stride). Slots are 1:1 with block rows (the     */ \
            /* stage already clamped out-of-range rows to row m-1).        */ \
            unsigned int aw[2][2][2];                                          \
            _Pragma("unroll")                                              \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                _Pragma("unroll")                                          \
                for (int rp = 0; rp < 2; ++rp) {                               \
                    const int rs = warp_row + mf * 16 + rp * 8 + g_id;         \
                    aw[mf][rp][0] = abuf[rs * 12 + ks * 2];                    \
                    aw[mf][rp][1] = abuf[rs * 12 + ks * 2 + 1];                \
                }                                                              \
            }                                                                  \
            unsigned int bw[4][2];                                             \
            _Pragma("unroll")                                              \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                const unsigned int bn =                                        \
                    (unsigned int)(warp_n * 32 + nf * 8 + (lane & 7)) * 36u    \
                    + (unsigned int)(ks * 8 + ((lane >> 3) & 1) * 4);          \
                v6_ldmatrix_x2(bw[nf][0], bw[nf][1], &bbuf[bn]);               \
            }                                                                  \
            _Pragma("unroll")                                              \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                /* 8 ALU-pipe ops per row pair — the LOP3-class decode. */ \
                unsigned int a0, a1, a2, a3;                                   \
                v8_decode_pair(aw[mf][0][0], aw[mf][0][1], sel, a0, a2);       \
                v8_decode_pair(aw[mf][1][0], aw[mf][1][1], sel, a1, a3);       \
                _Pragma("unroll")                                          \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    mma_s8_m16n8k32(dh[mf][nf][0], dh[mf][nf][1],              \
                        dh[mf][nf][2], dh[mf][nf][3],                          \
                        a0, a1, a2, a3, bw[nf][0], bw[nf][1]);                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* fold — v8 text verbatim (the bit-identity contract). */             \
        _Pragma("unroll")                                                  \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            const int r0 = row_base + mf * 16 + g_id;                          \
            const int rc0f = r0 < m ? r0 : m - 1;                              \
            const int rc8f = (r0 + 8) < m ? (r0 + 8) : m - 1;                  \
            const float sw0 = group_scale[(long)rc0f * groups_per_row + grp];  \
            const float sw8 = group_scale[(long)rc8f * groups_per_row + grp];  \
            _Pragma("unroll")                                              \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                _Pragma("unroll")                                          \
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
    /* epilogue: v8's schedule over the B slab reused as staging. */           \
    float* stg = (float*)b_buf0;           /* [32][TOKS] floats reused */       \
    const int tok_blk = blockIdx.y * (TOKS);                                   \
    _Pragma("unroll")                                                      \
    for (int ch = 0; ch < 4; ++ch) {                                           \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                  \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            if ((warp_row + mf * 16) / 32 == ch) {                             \
                const int r_in = warp_row + mf * 16 + g_id - ch * 32;          \
                _Pragma("unroll")                                          \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    const int tk0 = warp_n * 32 + nf * 8 + t_id * 2;           \
                    stg[r_in * (TOKS) + tk0]     = o[mf][nf][0];               \
                    stg[r_in * (TOKS) + tk0 + 1] = o[mf][nf][1];               \
                    stg[(r_in + 8) * (TOKS) + tk0]     = o[mf][nf][2];         \
                    stg[(r_in + 8) * (TOKS) + tk0 + 1] = o[mf][nf][3];         \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const int row0 = blockIdx.x * 128 + ch * 32;                           \
        for (int idx = tid; idx < 32 * (TOKS); idx += (NWARP) * 32) {          \
            const int r = idx / (TOKS);                                        \
            const int tk = idx % (TOKS);                                       \
            const int tok = tok_blk + tk;                                      \
            if (tok < p && row0 + r < m) {                                     \
                out[(long)tok * m + row0 + r] = stg[r * (TOKS) + tk] * s_t[tok];\
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \

// v9: TOKS=64, 256 threads (8 warps), 2 blocks/SM — 30,720 B smem (A codes
// double + B double), under the 48 KB default.
GEMM_BODY_V9_Q8(gemm_i8_mma_tm128v9_q8_fused, 1, 64, 8, 2)
GEMM_BODY_V9_Q8(gemm_i8_mma_tm128v9_q8_strict, 0, 64, 8, 2)
// v9t: TOKS=128, 512 threads (16 warps), 1 block/SM — 49,152 B (the primary
// arm; v8t/v6tb's geometry).
GEMM_BODY_V9_Q8(gemm_i8_mma_tm128v9t_q8_fused, 1, 128, 16, 1)
GEMM_BODY_V9_Q8(gemm_i8_mma_tm128v9t_q8_strict, 0, 128, 16, 1)

// ---------------------------------------------------------------------------
// v10 (T4 rung, ncu-guided — Plan 574): v9's body + the two L2-traffic
// repairs the Bench 884 profile named. The slab is L2-bandwidth-bound (L2
// 76-90% SOL, tensor ~34%, DRAM ~8%): (a) the fold's group scales rode 4
// scattered ld.global.f32 per warp per group (160-B row stride) = ~16x
// sector amplification — v10 cp.async's them once per group per block into
// double-buffered 128-float stages (+1 KB smem) and the fold reads LDS
// quad-broadcast (same VALUES, same order: bit-identical by construction);
// (b) v9's epilogue mapped consecutive lanes to consecutive TOKENS, so
// every warp store hit 32 different out rows 69,632 B apart = 8.0x write
// amplification (1.14 GB vs 142.6 MB useful at the slab) — v10 stages
// through a transposed [TOKS][33] tile (33 ≡ 1 mod 32: conflict-free
// writers) and the readers store 128-B contiguous per warp. +256 words
// smem total: v10 31,744 B (under the default), v10t 50,176 B (opt-in).
// ---------------------------------------------------------------------------
#define GEMM_BODY_V10_Q8(NAME, FUSED, TOKS, NWARP, MINB, DECODE)               \
extern "C" __global__ void __launch_bounds__((NWARP) * 32, (MINB)) NAME(        \
    const unsigned int* __restrict__ packed_w,   /* [m * (n/16)] Q2_0 codes */  \
    const float* __restrict__ group_scale,       /* [m * groups] */            \
    const unsigned int* __restrict__ q_w,        /* [p * (n/4)] */             \
    const float* __restrict__ s_t,               /* [p] */                     \
    float* __restrict__ out,                     /* [p * m] */                 \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char v10_smem_raw[];              \
    unsigned int* const acode0 = (unsigned int*)v10_smem_raw; /* 128*12 */     \
    unsigned int* const acode1 = acode0 + 128 * 12;                            \
    unsigned int* const b_buf0 = acode0 + 2 * 128 * 12;                        \
    unsigned int* const b_buf1 = b_buf0 + (TOKS) * 36;                         \
    float* const scl0 = (float*)(b_buf1 + (TOKS) * 36);   /* 128 floats */     \
    float* const scl1 = scl0 + 128;                                            \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                                                  \
    const int g_id = lane >> 2;                                                \
    const int t_id = lane & 3;                                                 \
    const int warp_row = (wid / ((TOKS) / 32)) * 32; /* 4 row-bands of 32 */   \
    const int warp_n = wid % ((TOKS) / 32);      /* TOKS/32 tok-warps x 32 */  \
    const int row_base = blockIdx.x * 128 + warp_row;                          \
    const int blk_rows = blockIdx.x * 128;                                     \
    const int qwpr = n >> 2;                                                   \
                                                                               \
    /* B staging: TOKS*8 16-byte chunks at TOKS/(NWARP*4) per thread (v9). */  \
    const int b_w4 = (tid & 7) << 2;                                           \
    const int b_t0 = tid >> 3;                                                 \
    const unsigned int sel = 0x40u + (unsigned int)t_id * 0x11u;               \
                                                                               \
    /* accumulators + fold regs (v9 verbatim). */                              \
    int dh[2][4][4];                                                           \
    float o[2][4][4];                                                          \
    _Pragma("unroll")                                                      \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                  \
        for (int nf = 0; nf < 4; ++nf) {                                       \
            _Pragma("unroll")                                              \
            for (int c = 0; c < 4; ++c) { o[mf][nf][c] = 0.0f; }               \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* prologue: B(0) plain LDG/STS; A(0) cp.async 16-B; S(0) cp.async 4-B. */ \
    _Pragma("unroll")                                                      \
    for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {                    \
        const int b_tk = b_t0 + si * ((NWARP) * 4);                            \
        int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                          \
        b_tr = b_tr < p ? b_tr : p - 1;                                        \
        *(uint4*)(b_buf0 + (unsigned)b_tk * 36u + (unsigned)b_w4) =            \
            *(const uint4*)(q_w + (long)b_tr * qwpr + b_w4);                   \
    }                                                                          \
    for (int c = tid; c < 256; c += (NWARP) * 32) {                            \
        const int row = c >> 1;                                                \
        const int half = c & 1;                                                \
        const int gr = (blk_rows + row < m) ? blk_rows + row : m - 1;          \
        v6_cp_async16(acode0 + row * 12 + half * 4,                            \
            packed_w + (long)gr * words_per_row + half * 4);                   \
    }                                                                          \
    for (int c = tid; c < 128; c += (NWARP) * 32) {                            \
        const int gr = (blk_rows + c < m) ? blk_rows + c : m - 1;              \
        v6_cp_async4(scl0 + c, group_scale + (long)gr * groups_per_row + 0);   \
    }                                                                          \
    v6_cp_async_commit();                                                      \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        /* A(0)/S(0) are in flight at grp 0 — the wait is unconditional. */    \
        v6_cp_async_wait_all();                                                \
        __syncthreads();   /* A/S(grp)+B(grp) landed & grp-1 compute done */   \
        unsigned int* const abuf = (grp & 1) ? acode1 : acode0;                \
        unsigned int* const nabuf = (grp & 1) ? acode0 : acode1;               \
        unsigned int* const bbuf = (grp & 1) ? b_buf1 : b_buf0;                \
        unsigned int* const nbbuf = (grp & 1) ? b_buf0 : b_buf1;               \
        float* const sbuf = (grp & 1) ? scl1 : scl0;                           \
        float* const nsbuf = (grp & 1) ? scl0 : scl1;                          \
        const bool has_next = (grp + 1) < groups_per_row;                      \
        if (has_next) {                                                        \
            /* B(grp+1), A(grp+1), S(grp+1) — all in flight across this. */    \
            _Pragma("unroll")                                              \
            for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {            \
                const int b_tk = b_t0 + si * ((NWARP) * 4);                    \
                int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                  \
                b_tr = b_tr < p ? b_tr : p - 1;                                \
                v6_cp_async16(nbbuf + (unsigned)b_tk * 36u + (unsigned)b_w4,   \
                    q_w + (long)b_tr * qwpr + b_w4 + ((long)(grp + 1) << 5));  \
            }                                                                  \
            v6_cp_async_commit();                                              \
            for (int c = tid; c < 256; c += (NWARP) * 32) {                    \
                const int row = c >> 1;                                        \
                const int half = c & 1;                                        \
                const int gr =                                                 \
                    (blk_rows + row < m) ? blk_rows + row : m - 1;             \
                v6_cp_async16(nabuf + row * 12 + half * 4,                     \
                    packed_w + (long)gr * words_per_row                        \
                        + (long)(grp + 1) * 8 + half * 4);                     \
            }                                                                  \
            for (int c = tid; c < 128; c += (NWARP) * 32) {                    \
                const int gr = (blk_rows + c < m) ? blk_rows + c : m - 1;      \
                v6_cp_async4(nsbuf + c,                                        \
                    group_scale + (long)gr * groups_per_row + grp + 1);        \
            }                                                                  \
            v6_cp_async_commit();                                              \
        }                                                                      \
        _Pragma("unroll")                                                  \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            _Pragma("unroll")                                              \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                _Pragma("unroll")                                          \
                for (int c = 0; c < 4; ++c) { dh[mf][nf][c] = 0; }             \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                  \
        for (int ks = 0; ks < 4; ++ks) {                                       \
            /* A code words: 8 LDS.32, quad-broadcast (conflict-free at    */ \
            /* the 12-word stride). Slots are 1:1 with block rows (the     */ \
            /* stage already clamped out-of-range rows to row m-1).        */ \
            unsigned int aw[2][2][2];                                          \
            _Pragma("unroll")                                              \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                _Pragma("unroll")                                          \
                for (int rp = 0; rp < 2; ++rp) {                               \
                    const int rs = warp_row + mf * 16 + rp * 8 + g_id;         \
                    aw[mf][rp][0] = abuf[rs * 12 + ks * 2];                    \
                    aw[mf][rp][1] = abuf[rs * 12 + ks * 2 + 1];                \
                }                                                              \
            }                                                                  \
            unsigned int bw[4][2];                                             \
            _Pragma("unroll")                                              \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                const unsigned int bn =                                        \
                    (unsigned int)(warp_n * 32 + nf * 8 + (lane & 7)) * 36u    \
                    + (unsigned int)(ks * 8 + ((lane >> 3) & 1) * 4);          \
                v6_ldmatrix_x2(bw[nf][0], bw[nf][1], &bbuf[bn]);               \
            }                                                                  \
            _Pragma("unroll")                                              \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                /* 8 ALU-pipe ops per row pair — the LOP3-class decode. */ \
                unsigned int a0, a1, a2, a3;                                   \
                if (DECODE) {                                                  \
                    v8_decode_pair(aw[mf][0][0], aw[mf][0][1], sel, a0, a2);   \
                    v8_decode_pair(aw[mf][1][0], aw[mf][1][1], sel, a1, a3);   \
                } else {                                                       \
                    /* Issue-903 T1 nodecode timing probe: the decode ALU  */ \
                    /* ops removed (raw code words feed the mma; VALUES are */ \
                    /* garbage BY DESIGN - timing-only, never a launchable  */ \
                    /* production arm).                                     */ \
                    a0 = aw[mf][0][0]; a1 = aw[mf][1][0];                      \
                    a2 = aw[mf][0][1]; a3 = aw[mf][1][1];                      \
                }                                                              \
                _Pragma("unroll")                                          \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    mma_s8_m16n8k32(dh[mf][nf][0], dh[mf][nf][1],              \
                        dh[mf][nf][2], dh[mf][nf][3],                          \
                        a0, a1, a2, a3, bw[nf][0], bw[nf][1]);                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* fold — v9 op order; the scale VALUES come from the smem stage   */ \
        /* (the stage clamped rows exactly like the A stage, so the local  */ \
        /* row indexes the same clamped scale the global load would).      */ \
        _Pragma("unroll")                                                  \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            const float sw0 = sbuf[warp_row + mf * 16 + g_id];                 \
            const float sw8 = sbuf[warp_row + mf * 16 + 8 + g_id];             \
            _Pragma("unroll")                                              \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                _Pragma("unroll")                                          \
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
    /* epilogue: transposed [TOKS][33] stage over the B slab (freed). The  */ \
    /* writers bank-shift on 33 ≡ 1 mod 32; the readers map consecutive    */ \
    /* lanes to consecutive out COLUMNS so each warp store is 128-B        */ \
    /* contiguous (v9's token-major mapping wrote 32 sectors per 128 B).   */ \
    float* stg2 = (float*)b_buf0;                                              \
    const int tok_blk = blockIdx.y * (TOKS);                                   \
    _Pragma("unroll")                                                      \
    for (int ch = 0; ch < 4; ++ch) {                                           \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                  \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            if ((warp_row + mf * 16) / 32 == ch) {                             \
                const int r_in = warp_row + mf * 16 + g_id - ch * 32;          \
                _Pragma("unroll")                                          \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    const int tk0 = warp_n * 32 + nf * 8 + t_id * 2;           \
                    stg2[tk0 * 33 + r_in]             = o[mf][nf][0];          \
                    stg2[(tk0 + 1) * 33 + r_in]       = o[mf][nf][1];          \
                    stg2[tk0 * 33 + r_in + 8]         = o[mf][nf][2];          \
                    stg2[(tk0 + 1) * 33 + r_in + 8]   = o[mf][nf][3];          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const int row0 = blockIdx.x * 128 + ch * 32;                           \
        for (int idx = tid; idx < 32 * (TOKS); idx += (NWARP) * 32) {          \
            const int r = idx & 31;                                            \
            const int tk = idx >> 5;                                           \
            const int tok = tok_blk + tk;                                      \
            if (tok < p && row0 + r < m) {                                     \
                out[(long)tok * m + row0 + r] = stg2[tk * 33 + r] * s_t[tok];  \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// v10: TOKS=64, 256 threads (8 warps), 2 blocks/SM — 31,744 B smem (A codes
// double + B double + scale double), under the 48 KB default.
GEMM_BODY_V10_Q8(gemm_i8_mma_tm128v10_q8_fused, 1, 64, 8, 2, 1)
GEMM_BODY_V10_Q8(gemm_i8_mma_tm128v10_q8_strict, 0, 64, 8, 2, 1)
// v10t: TOKS=128, 512 threads (16 warps), 1 block/SM — 50,176 B (the primary
// arm; v9t's geometry + the scale stage).
GEMM_BODY_V10_Q8(gemm_i8_mma_tm128v10t_q8_fused, 1, 128, 16, 1, 1)
GEMM_BODY_V10_Q8(gemm_i8_mma_tm128v10t_q8_strict, 0, 128, 16, 1, 1)
// Issue 903 T1 - the nodecode timing probe (v10t with the decode ALU ops
// stripped; VALUES GARBAGE BY DESIGN - prices the decode issue-slot share
// as time(v10t) - time(nodecode); never dispatched by any production path).
GEMM_BODY_V10_Q8(gemm_i8_mma_tm128v10t_q8_nodecode, 1, 128, 16, 1, 0)
// Plan 597 rung A (Issue 918, the owner-GO'd lever's cheapest arm): the v10
// body at MINB=3 - 24 warps/SM (3 blocks x 8), the __launch_bounds__(256, 3)
// reg cap at 85. Same per-block smem as v10 (31,744 B; 3 x 31,744 = 95,232 B
// fits the ~100 KB SM budget). Kernel-level probe ONLY - no production
// dispatch; the S1.e gate (>= +10% TF at G1 bit-identity, spill-free)
// decides whether a tile-shrunk production variant follows.
GEMM_BODY_V10_Q8(gemm_i8_mma_tm128v10o3_q8_fused, 1, 64, 8, 3, 1)
GEMM_BODY_V10_Q8(gemm_i8_mma_tm128v10o3_q8_strict, 0, 64, 8, 3, 1)
// Plan 597 S1 ceiling probe (Bench 933 follow-on): the same MINB=3 geometry
// with the decode stripped (DECODE=0, VALUES GARBAGE BY DESIGN - the
// Issue-903 nodecode convention). time(nodecode-o3) is the measured CEILING
// of the 24-warp class on this body: spill-free + ~65%-of-peak => more warps
// buy nothing and the lever's ladder collapses to physics; spill-free +
// >=75% => the tile-shrunk production variant (v12) has a proven target.
// Never dispatched by any production path.
GEMM_BODY_V10_Q8(gemm_i8_mma_tm128v10o3_q8_nodecode, 1, 64, 8, 3, 0)

// ---------------------------------------------------------------------------
// v12 (Plan 597 — the tile-shrunk body; Bench 933 §5/§6's ONE remaining
// arm): 64-row blocks x 16-tok warps. The warp tile halves (32x32 -> 32x16,
// nf 4->2) so the accumulators halve (dh[2][2][4]+o[2][2][4] = 32 regs) —
// the ONLY route the register file leaves open to 24 warps/SM (3 x 8-warp
// blocks at the __launch_bounds__(256,3) 85-reg cap; every 85-reg attempt on
// the 64-acc tile spilled — Bench 933 §4/§6, the B895 law x4). Per-block smem
// at TOKS=64: 2*64*12*4 (A) + 2*64*36*4 (B) + 2*64*4 (S) = 25,088 B (under
// the 48 KB default; 3 blocks = 75,264 B fits the ~100 KB SM budget).
// Bit-identity: per-output op sequence is v10t verbatim (group int accumulate
// + group-ordered FMA fold + x s_t) — the geometry change touches no
// arithmetic. x-grid doubles (m/64) and each weight slab is re-read by 2x
// the y-blocks of v10t's TOKS=128 — the priced L2 trade.
// ---------------------------------------------------------------------------
#define GEMM_BODY_V12_Q8(NAME, FUSED, TOKS, NWARP, MINB, DECODE)               \
extern "C" __global__ void __launch_bounds__((NWARP) * 32, (MINB)) NAME(        \
    const unsigned int* __restrict__ packed_w,   /* [m * (n/16)] Q2_0 codes */  \
    const float* __restrict__ group_scale,       /* [m * groups] */            \
    const unsigned int* __restrict__ q_w,        /* [p * (n/4)] */             \
    const float* __restrict__ s_t,               /* [p] */                     \
    float* __restrict__ out,                     /* [p * m] */                 \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char v12_smem_raw[];              \
    unsigned int* const acode0 = (unsigned int*)v12_smem_raw; /* 64*12 */      \
    unsigned int* const acode1 = acode0 + 64 * 12;                             \
    unsigned int* const b_buf0 = acode0 + 2 * 64 * 12;                         \
    unsigned int* const b_buf1 = b_buf0 + (TOKS) * 36;                         \
    float* const scl0 = (float*)(b_buf1 + (TOKS) * 36);   /* 64 floats */      \
    float* const scl1 = scl0 + 64;                                             \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                                                  \
    const int g_id = lane >> 2;                                                \
    const int t_id = lane & 3;                                                 \
    const int warp_row = (wid / ((TOKS) / 16)) * 32; /* 2 row-bands of 32 */   \
    const int warp_n = wid % ((TOKS) / 16);      /* TOKS/16 tok-warps x 16 */  \
    const int row_base = blockIdx.x * 64 + warp_row;                           \
    const int blk_rows = blockIdx.x * 64;                                       \
    const int qwpr = n >> 2;                                                    \
                                                                               \
    /* B staging: TOKS*8 16-byte chunks at TOKS/(NWARP*4) per thread (v9). */  \
    const int b_w4 = (tid & 7) << 2;                                           \
    const int b_t0 = tid >> 3;                                                 \
    const unsigned int sel = 0x40u + (unsigned int)t_id * 0x11u;               \
                                                                               \
    /* accumulators + fold regs (halved tile: nf=2). */                        \
    int dh[2][2][4];                                                           \
    float o[2][2][4];                                                          \
    _Pragma("unroll")                                                      \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                  \
        for (int nf = 0; nf < 2; ++nf) {                                       \
            _Pragma("unroll")                                              \
            for (int c = 0; c < 4; ++c) { o[mf][nf][c] = 0.0f; }              \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* prologue: B(0) plain LDG/STS; A(0) cp.async 16-B; S(0) cp.async 4-B. */ \
    _Pragma("unroll")                                                      \
    for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {                    \
        const int b_tk = b_t0 + si * ((NWARP) * 4);                            \
        int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                          \
        b_tr = b_tr < p ? b_tr : p - 1;                                        \
        *(uint4*)(b_buf0 + (unsigned)b_tk * 36u + (unsigned)b_w4) =            \
            *(const uint4*)(q_w + (long)b_tr * qwpr + b_w4);                   \
    }                                                                          \
    for (int c = tid; c < 128; c += (NWARP) * 32) {                            \
        const int row = c >> 1;                                                \
        const int half = c & 1;                                                \
        const int gr = (blk_rows + row < m) ? blk_rows + row : m - 1;          \
        v6_cp_async16(acode0 + row * 12 + half * 4,                            \
            packed_w + (long)gr * words_per_row + half * 4);                   \
    }                                                                          \
    for (int c = tid; c < 64; c += (NWARP) * 32) {                             \
        const int gr = (blk_rows + c < m) ? blk_rows + c : m - 1;              \
        v6_cp_async4(scl0 + c, group_scale + (long)gr * groups_per_row + 0);   \
    }                                                                          \
    v6_cp_async_commit();                                                      \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        /* A(0)/S(0) are in flight at grp 0 — the wait is unconditional. */    \
        v6_cp_async_wait_all();                                                \
        __syncthreads();   /* A/S(grp)+B(grp) landed & grp-1 compute done */   \
        unsigned int* const abuf = (grp & 1) ? acode1 : acode0;                \
        unsigned int* const nabuf = (grp & 1) ? acode0 : acode1;                \
        unsigned int* const bbuf = (grp & 1) ? b_buf1 : b_buf0;                \
        unsigned int* const nbbuf = (grp & 1) ? b_buf0 : b_buf1;                \
        float* const sbuf = (grp & 1) ? scl1 : scl0;                           \
        float* const nsbuf = (grp & 1) ? scl0 : scl1;                          \
        const bool has_next = (grp + 1) < groups_per_row;                      \
        if (has_next) {                                                        \
            /* B(grp+1), A(grp+1), S(grp+1) — all in flight across this. */    \
            _Pragma("unroll")                                              \
            for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {            \
                const int b_tk = b_t0 + si * ((NWARP) * 4);                    \
                int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                  \
                b_tr = b_tr < p ? b_tr : p - 1;                                \
                v6_cp_async16(nbbuf + (unsigned)b_tk * 36u + (unsigned)b_w4,   \
                    q_w + (long)b_tr * qwpr + b_w4 + ((long)(grp + 1) << 5));  \
            }                                                                  \
            v6_cp_async_commit();                                              \
            for (int c = tid; c < 128; c += (NWARP) * 32) {                    \
                const int row = c >> 1;                                        \
                const int half = c & 1;                                        \
                const int gr =                                                 \
                    (blk_rows + row < m) ? blk_rows + row : m - 1;             \
                v6_cp_async16(nabuf + row * 12 + half * 4,                     \
                    packed_w + (long)gr * words_per_row                        \
                        + (long)(grp + 1) * 8 + half * 4);                     \
            }                                                                  \
            for (int c = tid; c < 64; c += (NWARP) * 32) {                     \
                const int gr = (blk_rows + c < m) ? blk_rows + c : m - 1;      \
                v6_cp_async4(nsbuf + c,                                        \
                    group_scale + (long)gr * groups_per_row + grp + 1);        \
            }                                                                  \
            v6_cp_async_commit();                                              \
        }                                                                      \
        _Pragma("unroll")                                                  \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            _Pragma("unroll")                                              \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                _Pragma("unroll")                                          \
                for (int c = 0; c < 4; ++c) { dh[mf][nf][c] = 0; }             \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                  \
        for (int ks = 0; ks < 4; ++ks) {                                       \
            unsigned int aw[2][2][2];                                          \
            _Pragma("unroll")                                              \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                _Pragma("unroll")                                          \
                for (int rp = 0; rp < 2; ++rp) {                               \
                    const int rs = warp_row + mf * 16 + rp * 8 + g_id;         \
                    aw[mf][rp][0] = abuf[rs * 12 + ks * 2];                    \
                    aw[mf][rp][1] = abuf[rs * 12 + ks * 2 + 1];                \
                }                                                              \
            }                                                                  \
            unsigned int bw[2][2];                                             \
            _Pragma("unroll")                                              \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                const unsigned int bn =                                        \
                    (unsigned int)(warp_n * 16 + nf * 8 + (lane & 7)) * 36u    \
                    + (unsigned int)(ks * 8 + ((lane >> 3) & 1) * 4);          \
                v6_ldmatrix_x2(bw[nf][0], bw[nf][1], &bbuf[bn]);               \
            }                                                                  \
            _Pragma("unroll")                                              \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                unsigned int a0, a1, a2, a3;                                   \
                if (DECODE) {                                                  \
                    v8_decode_pair(aw[mf][0][0], aw[mf][0][1], sel, a0, a2);   \
                    v8_decode_pair(aw[mf][1][0], aw[mf][1][1], sel, a1, a3);   \
                } else {                                                       \
                    a0 = aw[mf][0][0]; a1 = aw[mf][1][0];                      \
                    a2 = aw[mf][0][1]; a3 = aw[mf][1][1];                      \
                }                                                              \
                _Pragma("unroll")                                          \
                for (int nf = 0; nf < 2; ++nf) {                               \
                    mma_s8_m16n8k32(dh[mf][nf][0], dh[mf][nf][1],              \
                        dh[mf][nf][2], dh[mf][nf][3],                          \
                        a0, a1, a2, a3, bw[nf][0], bw[nf][1]);                  \
                }                                                              \
            }                                                                  \
        }                                                                      \
        _Pragma("unroll")                                                  \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            const float sw0 = sbuf[warp_row + mf * 16 + g_id];                 \
            const float sw8 = sbuf[warp_row + mf * 16 + 8 + g_id];             \
            _Pragma("unroll")                                              \
            for (int nf = 0; nf < 2; ++nf) {                                   \
                _Pragma("unroll")                                          \
                for (int c = 0; c < 4; ++c) {                                  \
                    const float hi = (float)dh[mf][nf][c];                     \
                    if (FUSED) {                                               \
                        o[mf][nf][c] = __fmaf_rn(c >= 2 ? sw8 : sw0, hi, o[mf][nf][c]);\
                    } else {                                                   \
                        const float t3 = __fmul_rn(c >= 2 ? sw8 : sw0, hi);     \
                        o[mf][nf][c] = __fadd_rn(o[mf][nf][c], t3);            \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* epilogue: transposed [TOKS][33] stage over the B slab (freed); 2      */ \
    /* row-bands of 32 now (ch < 2). */                                        \
    float* stg2 = (float*)b_buf0;                                              \
    const int tok_blk = blockIdx.y * (TOKS);                                   \
    _Pragma("unroll")                                                      \
    for (int ch = 0; ch < 2; ++ch) {                                           \
        __syncthreads();                                                       \
        _Pragma("unroll")                                                  \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            if ((warp_row + mf * 16) / 32 == ch) {                             \
                const int r_in = warp_row + mf * 16 + g_id - ch * 32;          \
                _Pragma("unroll")                                          \
                for (int nf = 0; nf < 2; ++nf) {                               \
                    const int tk0 = warp_n * 16 + nf * 8 + t_id * 2;           \
                    stg2[tk0 * 33 + r_in]             = o[mf][nf][0];          \
                    stg2[(tk0 + 1) * 33 + r_in]       = o[mf][nf][1];          \
                    stg2[tk0 * 33 + r_in + 8]         = o[mf][nf][2];          \
                    stg2[(tk0 + 1) * 33 + r_in + 8]   = o[mf][nf][3];          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        const int row0 = blockIdx.x * 64 + ch * 32;                            \
        for (int idx = tid; idx < 32 * (TOKS); idx += (NWARP) * 32) {          \
            const int r = idx & 31;                                            \
            const int tk = idx >> 5;                                           \
            const int tok = tok_blk + tk;                                      \
            if (tok < p && row0 + r < m) {                                     \
                out[(long)tok * m + row0 + r] = stg2[tk * 33 + r] * s_t[tok];  \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// v12 probe arms (Plan 597, Bench 933 §6's one live arm): TOKS=64, 8 warps,
// 3 blocks/SM = 24 warps at the 85-reg cap. fused+strict for the G1 gate;
// nodecode per the Issue-903 convention (timing-only, garbage values).
GEMM_BODY_V12_Q8(gemm_i8_mma_tm64v12_q8_fused, 1, 64, 8, 3, 1)
GEMM_BODY_V12_Q8(gemm_i8_mma_tm64v12_q8_strict, 0, 64, 8, 3, 1)
GEMM_BODY_V12_Q8(gemm_i8_mma_tm64v12_q8_nodecode, 1, 64, 8, 3, 0)

// ---------------------------------------------------------------------------
// v11gu/v11gut (Issue 902 T1 — the fused gate+up pair): the FFN family's two
// same-shape GEMMs (ffn_gate, ffn_up — both [m, n], same input activations)
// launched as ONE kernel. The v10 body's B (activation) tile is the re-read
// traffic pot: per token-block column the activations were requested by every
// m/128 row-block TWICE (once per launch) — the fused body stages each B tile
// ONCE per (row-block, tok-block) and runs the ks-slice MMA twice (gate into
// dhg, up into dhu), so the ldmatrix count per useful MMA halves too. A stages
// and scale stages double (one pair per matrix); the epilogue runs twice
// (gate slab into out0, up slab into out1) over the same stg2 buffer.
// Bit-identity: per-output-element op sequence is v10t verbatim (int dh
// accumulation within group; group-ordered FMA fold; × s_t at store) — the
// interleaving only reorders INDEPENDENT gate/up accumulator updates.
// smem: 4*128*12*4 (A codes, 2 mats x 2 bufs) + 2*TOKS*36*4 (B) + 4*128*4
// (scales, 2 mats x 2 bufs) = 45,056 B at TOKS=64 / 63,488 B at TOKS=128.
// ---------------------------------------------------------------------------
#define V11GU_EPILOGUE_ONE(OMAT, OUTP, TOKS_, NWARP_)                          \
    {                                                                          \
        float* stg2 = (float*)b_buf0;                                          \
        const int tok_blk = blockIdx.y * (TOKS_);                              \
        _Pragma("unroll")                                                      \
        for (int ch = 0; ch < 4; ++ch) {                                       \
            __syncthreads();                                                   \
            _Pragma("unroll")                                                  \
            for (int mf = 0; mf < 2; ++mf) {                                   \
                if ((warp_row + mf * 16) / 32 == ch) {                         \
                    const int r_in = warp_row + mf * 16 + g_id - ch * 32;       \
                    _Pragma("unroll")                                          \
                    for (int nf = 0; nf < 4; ++nf) {                           \
                        const int tk0 = warp_n * 32 + nf * 8 + t_id * 2;        \
                        stg2[tk0 * 33 + r_in]             = OMAT[mf][nf][0];  \
                        stg2[(tk0 + 1) * 33 + r_in]       = OMAT[mf][nf][1];  \
                        stg2[tk0 * 33 + r_in + 8]         = OMAT[mf][nf][2];  \
                        stg2[(tk0 + 1) * 33 + r_in + 8]   = OMAT[mf][nf][3];  \
                    }                                                          \
                }                                                              \
            }                                                                  \
            __syncthreads();                                                   \
            const int row0 = blockIdx.x * 128 + ch * 32;                       \
            for (int idx = tid; idx < 32 * (TOKS_); idx += (NWARP_) * 32) {    \
                const int r = idx & 31;                                        \
                const int tk = idx >> 5;                                       \
                const int tok = tok_blk + tk;                                  \
                if (tok < p && row0 + r < m) {                                 \
                    OUTP[(long)tok * m + row0 + r] = stg2[tk * 33 + r] * s_t[tok]; \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

#define V11GU_KS_MAT(ABUF, DH)                                                \
    _Pragma("unroll")                                                          \
    for (int ks = 0; ks < 4; ++ks) {                                           \
        unsigned int bw[4][2];                                                 \
        _Pragma("unroll")                                                      \
        for (int nf = 0; nf < 4; ++nf) {                                       \
            const unsigned int bn =                                            \
                (unsigned int)(warp_n * 32 + nf * 8 + (lane & 7)) * 36u        \
                + (unsigned int)(ks * 8 + ((lane >> 3) & 1) * 4);              \
            v6_ldmatrix_x2(bw[nf][0], bw[nf][1], &bbuf[bn]);                   \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int mf = 0; mf < 2; ++mf) {                                       \
            unsigned int aw[2][2][2];                                          \
            _Pragma("unroll")                                                  \
            for (int rp = 0; rp < 2; ++rp) {                                   \
                const int rs = warp_row + mf * 16 + rp * 8 + g_id;             \
                aw[mf][rp][0] = ABUF[rs * 12 + ks * 2];                        \
                aw[mf][rp][1] = ABUF[rs * 12 + ks * 2 + 1];                    \
            }                                                                  \
            unsigned int a0, a1, a2, a3;                                       \
            v8_decode_pair(aw[mf][0][0], aw[mf][0][1], sel, a0, a2);           \
            v8_decode_pair(aw[mf][1][0], aw[mf][1][1], sel, a1, a3);           \
            _Pragma("unroll")                                                  \
            for (int nf = 0; nf < 4; ++nf) {                                   \
                mma_s8_m16n8k32(DH[mf][nf][0], DH[mf][nf][1],                  \
                    DH[mf][nf][2], DH[mf][nf][3],                              \
                    a0, a1, a2, a3, bw[nf][0], bw[nf][1]);                      \
            }                                                                  \
        }                                                                      \
    }

#define V11GU_ZERO_DH(DH)                                                      \
    _Pragma("unroll")                                                          \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                      \
        for (int nf = 0; nf < 4; ++nf) {                                       \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; ++c) { DH[mf][nf][c] = 0; }                 \
        }                                                                      \
    }

#define V11GU_FOLD_MAT(DH, OMAT, SBUF, FUSED_)                                 \
    _Pragma("unroll")                                                          \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        const float sw0 = SBUF[warp_row + mf * 16 + g_id];                     \
        const float sw8 = SBUF[warp_row + mf * 16 + 8 + g_id];                 \
        _Pragma("unroll")                                                      \
        for (int nf = 0; nf < 4; ++nf) {                                       \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; ++c) {                                      \
                const float hi = (float)DH[mf][nf][c];                         \
                if (FUSED_) {                                                  \
                    OMAT[mf][nf][c] = __fmaf_rn(c >= 2 ? sw8 : sw0, hi, OMAT[mf][nf][c]); \
                } else {                                                       \
                    const float t3 = __fmul_rn(c >= 2 ? sw8 : sw0, hi);        \
                    OMAT[mf][nf][c] = __fadd_rn(OMAT[mf][nf][c], t3);          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

#define GEMM_BODY_V11GU_Q8(NAME, FUSED, TOKS, NWARP, MINB, SEQ)               \
extern "C" __global__ void __launch_bounds__((NWARP) * 32, (MINB)) NAME(        \
    const unsigned int* __restrict__ packed_w0,  /* [m * (n/16)] mat0 codes */ \
    const float* __restrict__ group_scale0,      /* [m * groups] mat0 */       \
    const unsigned int* __restrict__ packed_w1,  /* [m * (n/16)] mat1 codes */ \
    const float* __restrict__ group_scale1,      /* [m * groups] mat1 */       \
    const unsigned int* __restrict__ q_w,        /* [p * (n/4)] */             \
    const float* __restrict__ s_t,               /* [p] */                     \
    float* __restrict__ out0,                    /* [p * m] mat0 slab */       \
    float* __restrict__ out1,                    /* [p * m] mat1 slab */       \
    int m, int n, int p,                                                       \
    int words_per_row, int groups_per_row)                                     \
{                                                                              \
    extern __shared__ __align__(16) unsigned char v11_smem_raw[];              \
    unsigned int* const ag0 = (unsigned int*)v11_smem_raw; /* 128*12 mat0 */    \
    unsigned int* const ag1 = ag0 + 128 * 12;                                  \
    unsigned int* const au0 = ag1 + 128 * 12;             /* 128*12 mat1 */    \
    unsigned int* const au1 = au0 + 128 * 12;                                  \
    unsigned int* const b_buf0 = au1 + 128 * 12;                               \
    unsigned int* const b_buf1 = b_buf0 + (TOKS) * 36;                         \
    float* const sg0 = (float*)(b_buf1 + (TOKS) * 36);   /* 128 floats mat0 */ \
    float* const sg1 = sg0 + 128;                                              \
    float* const su0 = sg1 + 128;                        /* 128 floats mat1 */ \
    float* const su1 = su0 + 128;                                              \
                                                                               \
    const int tid = threadIdx.x;                                               \
    const int lane = tid & 31;                                                 \
    const int wid = tid >> 5;                                                  \
    const int g_id = lane >> 2;                                                \
    const int t_id = lane & 3;                                                 \
    const int warp_row = (wid / ((TOKS) / 32)) * 32; /* 4 row-bands of 32 */   \
    const int warp_n = wid % ((TOKS) / 32);      /* TOKS/32 tok-warps x 32 */  \
    const int blk_rows = blockIdx.x * 128;                                     \
    const int qwpr = n >> 2;                                                   \
                                                                               \
    /* B staging: TOKS*8 16-byte chunks at TOKS/(NWARP*4) per thread (v9). */  \
    const int b_w4 = (tid & 7) << 2;                                           \
    const int b_t0 = tid >> 3;                                                 \
    const unsigned int sel = 0x40u + (unsigned int)t_id * 0x11u;               \
                                                                               \
    /* accumulators: two mat sets (gate/up) + the per-group int dh pair. */     \
    int dhg[2][4][4];                                                          \
    int dhu[2][4][4];                                                          \
    float og[2][4][4];                                                         \
    float ou[2][4][4];                                                         \
    _Pragma("unroll")                                                      \
    for (int mf = 0; mf < 2; ++mf) {                                           \
        _Pragma("unroll")                                                  \
        for (int nf = 0; nf < 4; ++nf) {                                       \
            _Pragma("unroll")                                              \
            for (int c = 0; c < 4; ++c) {                                      \
                og[mf][nf][c] = 0.0f;                                          \
                ou[mf][nf][c] = 0.0f;                                          \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* prologue: B(0) plain LDG/STS; A(0) x2 + S(0) x2 cp.async. */            \
    _Pragma("unroll")                                                      \
    for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {                    \
        const int b_tk = b_t0 + si * ((NWARP) * 4);                            \
        int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                          \
        b_tr = b_tr < p ? b_tr : p - 1;                                        \
        *(uint4*)(b_buf0 + (unsigned)b_tk * 36u + (unsigned)b_w4) =            \
            *(const uint4*)(q_w + (long)b_tr * qwpr + b_w4);                   \
    }                                                                          \
    for (int c = tid; c < 256; c += (NWARP) * 32) {                            \
        const int row = c >> 1;                                                \
        const int half = c & 1;                                                \
        const int gr = (blk_rows + row < m) ? blk_rows + row : m - 1;          \
        v6_cp_async16(ag0 + row * 12 + half * 4,                               \
            packed_w0 + (long)gr * words_per_row + half * 4);                  \
        v6_cp_async16(au0 + row * 12 + half * 4,                               \
            packed_w1 + (long)gr * words_per_row + half * 4);                  \
    }                                                                          \
    for (int c = tid; c < 128; c += (NWARP) * 32) {                            \
        const int gr = (blk_rows + c < m) ? blk_rows + c : m - 1;              \
        v6_cp_async4(sg0 + c, group_scale0 + (long)gr * groups_per_row + 0);   \
        v6_cp_async4(su0 + c, group_scale1 + (long)gr * groups_per_row + 0);   \
    }                                                                          \
    v6_cp_async_commit();                                                      \
                                                                               \
    for (int grp = 0; grp < groups_per_row; ++grp) {                           \
        v6_cp_async_wait_all();                                                \
        __syncthreads();   /* A/S(grp)+B(grp) landed & grp-1 compute done */   \
        unsigned int* const abg = (grp & 1) ? ag1 : ag0;                       \
        unsigned int* const abu = (grp & 1) ? au1 : au0;                       \
        unsigned int* const bbuf = (grp & 1) ? b_buf1 : b_buf0;                \
        float* const sgb = (grp & 1) ? sg1 : sg0;                              \
        float* const sub = (grp & 1) ? su1 : su0;                              \
        const bool has_next = (grp + 1) < groups_per_row;                      \
        if (has_next) {                                                        \
            /* B(grp+1) — ONE stage for both mats (the fusion win). */         \
            unsigned int* const nbbuf = (grp & 1) ? b_buf0 : b_buf1;           \
            unsigned int* const nabg = (grp & 1) ? ag0 : ag1;                  \
            unsigned int* const nabu = (grp & 1) ? au0 : au1;                  \
            float* const nsgb = (grp & 1) ? sg0 : sg1;                         \
            float* const nsub = (grp & 1) ? su0 : su1;                         \
            _Pragma("unroll")                                              \
            for (int si = 0; si < ((TOKS) / ((NWARP) * 4)); ++si) {            \
                const int b_tk = b_t0 + si * ((NWARP) * 4);                    \
                int b_tr = (int)(blockIdx.y * (TOKS)) + b_tk;                  \
                b_tr = b_tr < p ? b_tr : p - 1;                                \
                v6_cp_async16(nbbuf + (unsigned)b_tk * 36u + (unsigned)b_w4,   \
                    q_w + (long)b_tr * qwpr + b_w4 + ((long)(grp + 1) << 5));  \
            }                                                                  \
            v6_cp_async_commit();                                              \
            for (int c = tid; c < 256; c += (NWARP) * 32) {                    \
                const int row = c >> 1;                                        \
                const int half = c & 1;                                        \
                const int gr =                                                 \
                    (blk_rows + row < m) ? blk_rows + row : m - 1;             \
                v6_cp_async16(nabg + row * 12 + half * 4,                      \
                    packed_w0 + (long)gr * words_per_row                        \
                        + (long)(grp + 1) * 8 + half * 4);                     \
                v6_cp_async16(nabu + row * 12 + half * 4,                      \
                    packed_w1 + (long)gr * words_per_row                        \
                        + (long)(grp + 1) * 8 + half * 4);                     \
            }                                                                  \
            for (int c = tid; c < 128; c += (NWARP) * 32) {                    \
                const int gr = (blk_rows + c < m) ? blk_rows + c : m - 1;      \
                v6_cp_async4(nsgb + c,                                         \
                    group_scale0 + (long)gr * groups_per_row + grp + 1);       \
                v6_cp_async4(nsub + c,                                         \
                    group_scale1 + (long)gr * groups_per_row + grp + 1);       \
            }                                                                  \
            v6_cp_async_commit();                                              \
        }                                                                      \
        if (SEQ) {                                                             \
            /* Sequential shared-dh form: ONE dh set lives at a time           \
               (-16 int regs vs interleaved) at the cost of a second           \
               B-fragment ldmatrix pass per group (v10t's per-mat LDS           \
               count; the cp.async B STAGE stays shared — the L2 request        \
               halving is the surviving win). Per-output op order is           \
               unchanged: dh zeroed, ks 0..3, fold — exactly v10t. */           \
            V11GU_ZERO_DH(dhg)                                                 \
            V11GU_KS_MAT(abg, dhg)                                             \
            V11GU_FOLD_MAT(dhg, og, sgb, FUSED)                                \
            V11GU_ZERO_DH(dhg)                                                 \
            V11GU_KS_MAT(abu, dhg)                                             \
            V11GU_FOLD_MAT(dhg, ou, sub, FUSED)                                \
        } else {                                                               \
            /* Interleaved form: ONE B-fragment load per ks feeds BOTH mats'   \
               MMA sets (per-MMA ldmatrix count halves), but BOTH dh sets      \
               live across the ks loop (+16 int regs) — needs the 255-reg       \
               budget (MINB=1 at 256 threads) to avoid spilling. */            \
            V11GU_ZERO_DH(dhg)                                                 \
            V11GU_ZERO_DH(dhu)                                                 \
            _Pragma("unroll")                                              \
            for (int ks = 0; ks < 4; ++ks) {                                   \
                unsigned int bw[4][2];                                         \
                _Pragma("unroll")                                          \
                for (int nf = 0; nf < 4; ++nf) {                               \
                    const unsigned int bn =                                    \
                        (unsigned int)(warp_n * 32 + nf * 8 + (lane & 7)) * 36u \
                        + (unsigned int)(ks * 8 + ((lane >> 3) & 1) * 4);       \
                    v6_ldmatrix_x2(bw[nf][0], bw[nf][1], &bbuf[bn]);           \
                }                                                              \
                _Pragma("unroll")                                          \
                for (int mf = 0; mf < 2; ++mf) {                               \
                    unsigned int aw[2][2][2];                                  \
                    _Pragma("unroll")                                      \
                    for (int rp = 0; rp < 2; ++rp) {                           \
                        const int rs = warp_row + mf * 16 + rp * 8 + g_id;     \
                        aw[mf][rp][0] = abg[rs * 12 + ks * 2];                 \
                        aw[mf][rp][1] = abg[rs * 12 + ks * 2 + 1];             \
                    }                                                          \
                    unsigned int a0, a1, a2, a3;                               \
                    v8_decode_pair(aw[mf][0][0], aw[mf][0][1], sel, a0, a2);   \
                    v8_decode_pair(aw[mf][1][0], aw[mf][1][1], sel, a1, a3);   \
                    _Pragma("unroll")                                      \
                    for (int nf = 0; nf < 4; ++nf) {                           \
                        mma_s8_m16n8k32(dhg[mf][nf][0], dhg[mf][nf][1],        \
                            dhg[mf][nf][2], dhg[mf][nf][3],                    \
                            a0, a1, a2, a3, bw[nf][0], bw[nf][1]);              \
                    }                                                          \
                }                                                              \
                _Pragma("unroll")                                          \
                for (int mf = 0; mf < 2; ++mf) {                               \
                    unsigned int aw[2][2][2];                                  \
                    _Pragma("unroll")                                      \
                    for (int rp = 0; rp < 2; ++rp) {                           \
                        const int rs = warp_row + mf * 16 + rp * 8 + g_id;     \
                        aw[mf][rp][0] = abu[rs * 12 + ks * 2];                 \
                        aw[mf][rp][1] = abu[rs * 12 + ks * 2 + 1];             \
                    }                                                          \
                    unsigned int a0, a1, a2, a3;                               \
                    v8_decode_pair(aw[mf][0][0], aw[mf][0][1], sel, a0, a2);   \
                    v8_decode_pair(aw[mf][1][0], aw[mf][1][1], sel, a1, a3);   \
                    _Pragma("unroll")                                      \
                    for (int nf = 0; nf < 4; ++nf) {                           \
                        mma_s8_m16n8k32(dhu[mf][nf][0], dhu[mf][nf][1],        \
                            dhu[mf][nf][2], dhu[mf][nf][3],                    \
                            a0, a1, a2, a3, bw[nf][0], bw[nf][1]);              \
                    }                                                          \
                }                                                              \
            }                                                                  \
            V11GU_FOLD_MAT(dhg, og, sgb, FUSED)                                \
            V11GU_FOLD_MAT(dhu, ou, sub, FUSED)                                \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* epilogue: the v10 transposed [TOKS][33] stage, run per matrix over  */   \
    /* the same stg2 slab (the leading __syncthreads separates the two).   */   \
    V11GU_EPILOGUE_ONE(og, out0, (TOKS), (NWARP))                              \
    V11GU_EPILOGUE_ONE(ou, out1, (TOKS), (NWARP))                              \
}

// v11gu: TOKS=64, 256 threads (8 warps), 2 blocks/SM — 45,056 B smem (the
// occupancy arm; 2 x 45,056 = 90,112 B/SM fits the 100 KB budget).
// MEASURED-REFUTED (Bench 895): the interleaved body needs ~160 regs; the
// MINB=2 launch-bounds cap (128) forced a 192 B local spill — 0.330x/0.359x.
// Kept for the record; the repaired rungs are v11gs/v11gq below.
GEMM_BODY_V11GU_Q8(gemm_i8_mma_tm128v11gu_q8_fused, 1, 64, 8, 2, 0)
GEMM_BODY_V11GU_Q8(gemm_i8_mma_tm128v11gu_q8_strict, 0, 64, 8, 2, 0)
// v11gut: TOKS=128, 512 threads (16 warps), 1 block/SM — 63,488 B (v10t's
// geometry + the doubled A/scale stages). MEASURED-REFUTED (Bench 895): the
// 512-thread reg ceiling is 128 by construction (the full reg file) — the
// interleaved body cannot fit at any launch-bounds setting. 0.337x/0.371x.
GEMM_BODY_V11GU_Q8(gemm_i8_mma_tm128v11gut_q8_fused, 1, 128, 16, 1, 0)
GEMM_BODY_V11GU_Q8(gemm_i8_mma_tm128v11gut_q8_strict, 0, 128, 16, 1, 0)
// v11gs (Bench 895 repair rung 1): SEQ=1 — the shared-dh sequential body
// (~+16 regs vs v10t, one dh set live at a time) at TOKS=64/256thr/MINB=2 —
// 16 warps/SM kept; the B cp.async STAGE stays shared (the L2-request
// halving), at v10t's per-mat ldmatrix count.
GEMM_BODY_V11GU_Q8(gemm_i8_mma_tm128v11gs_q8_fused, 1, 64, 8, 2, 1)
GEMM_BODY_V11GU_Q8(gemm_i8_mma_tm128v11gs_q8_strict, 0, 64, 8, 2, 1)
// v11gq (Bench 895 repair rung 2): SEQ=0 — the interleaved body (both dh
// sets + shared bw loads) at TOKS=64/256thr/MINB=1: the 1-block launch
// bounds raise the reg ceiling to 255 (256 thr x 255 = 65,280 <= 64K),
// eliminating the spill at the cost of 8 warps/SM.
GEMM_BODY_V11GU_Q8(gemm_i8_mma_tm128v11gq_q8_fused, 1, 64, 8, 1, 0)
GEMM_BODY_V11GU_Q8(gemm_i8_mma_tm128v11gq_q8_strict, 0, 64, 8, 1, 0)
"#;
