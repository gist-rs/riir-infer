//! Issue 898 + Issue 899 — the head-ganged kv_group prefill-attention
//! kernel family (`att_pf_mq8g[_dp]` and the split-gang occupancy ladder
//! `att_pf_mq8g3/g2[_dp]`), source for the NVRTC module.
//!
//! One 256-thread block per (kv_group, q-tile, sub) serves GH heads of the
//! GQA group (`ATTN_GANG_HEADS = 6` total, the production n_head=24 / n_kv=4
//! shape): `GH*8` staged q-rows, the K row loaded ONCE per tile (float4, as
//! `att_pf_mq8v`) and the V row loaded ONCE per pp serving all staged weight
//! rows. The vec arm's 6 head-blocks per (kv_group, q-tile) each fetched the
//! same K/V slice independently — the gang cuts that KV L2→SM traffic 6×
//! (the residual structural redundancy Bench 891 re-priced).
//!
//! ## Issue 899 race (c) — the deferred `s_rmx` write (a shipped-g6 fix)
//!
//! The softmax scalars (`rmax`, `rsum`, per-tile `tmax`) live in smem
//! scalar arrays — written by THREAD 0 ONLY, at the tile merge. The
//! Issue-898 form wrote `s_rmx[r] = new_max` in the all-threads eprev
//! loop: a cross-warp read-then-write hazard ("same-value writes" are
//! benign only for PURE stores — never when the read feeds the
//! computation). A lagging warp's `rmax = s_rmx[r]` load lands after a
//! leading warp's same-tile store, so it computes
//! `eprev = expf(new_max − new_max) = 1.0` instead of the rescale → a
//! corrupted `racc` merge, ~1e-2-scale output error, tile-monotone
//! (warp skew accumulates per tile). Fired by the split-gang's warp
//! scheduling at p=2048 (G1), latent in the shipped g6 since Issue 898
//! — the Bench-892 bug-#2 fix covered the `s_rsm` RMW but missed this
//! sibling hazard. Fix (both bodies): the eprev loop is PURE READS;
//! Phase 3 recomputes `nm = max(tmax, old_rmax)` from the unwritten
//! `s_rmx` (same operands → same bits); the merge's tid0 stores BOTH
//! `s_rsm` and the deferred `s_rmx`, published by the existing
//! end-of-tile `__syncthreads`. Bit-identical to the intended chain by
//! construction (max/expf/fmaf over unchanged operands).
//!
//! ## Issue 899 — the occupancy ladder (the `GH` template)
//!
//! The GH=6 full gang is smem-bound at 1 block/SM (98,880 B dynamic smem +
//! 256 threads = 8 resident warps); per-SM latency hiding is the measured
//! wall, not traffic. The ladder splits the 48 rows across
//! `BPG = 6/GH` co-resident blocks instead:
//!
//! | variant | GH | rows/block | smem/block | blocks/SM | warps | KV L2× |
//! |---|---|---|---|---|---|---|
//! | `att_pf_mq8g` (898, shipped) | 6 | 48 | 98,880 B | 1 | 8 | 1× |
//! | `att_pf_mq8g3` (899) | 3 | 24 | 49,440 B | 2 | 16 | 2× |
//! | `att_pf_mq8g2` (899) | 2 | 16 | 32,960 B | 3 | 24 | 3× |
//!
//! The per-SM smem total is invariant (~99 KB of row arrays either way) —
//! the ladder trades KV L2 re-reads (still 3×/2× better than the vec arm's
//! 6×) for resident warps and DECOUPLED barriers (two co-resident blocks
//! hide each other's `__syncthreads` stalls; a fat block cannot). Register
//! arrays scale with rows/block (peak ≈ 210 → ≈ 120 → ≈ 90), and
//! `__launch_bounds__(256, BPG)` makes ptxas cap registers at 128/85 so the
//! co-residency is real, not nominal. The sub-block index varies FASTEST in
//! the grid (consecutive blocks share the same K/V slice AND q rows — L2
//! locality for the redundant fetch).
//!
//! BIT-IDENTICALITY is by the same construction as the full gang: every
//! per-row chain (P1 d-ascending `fmaf(s_q[r][d], k[d], score[r])`, the
//! identical stride max-tree, P4 pp-ascending `tsum += w;
//! racc = fmaf(w, v, racc)` with one V element serving all rows, the
//! deferred-fma merge, the final divide + sigmoid gate) sees the vec
//! kernel's operands in the vec kernel's order — a row's arithmetic does
//! not depend on WHICH block owns it. The GH=6 instantiation is the shipped
//! 898 kernel verbatim (BPG folds to 1).
//!
//! cp.async KV double-buffering (the issue's other lever) is structurally
//! infeasible here and NOT implemented: one K (or V) tile is
//! 256 pos × 256 ch × 4 B = 256 KB — 2.6× the 99 KB sm_89 per-block max —
//! and the K row is consumed per-thread from registers; cp.async only moves
//! global→smem. Measured in Bench 893 as a documented negative.

/// The launcher-side byte count for the full-gang smem opt-in (shared with
/// the `CudaAttnKernels::new` attribute set): 48·256·2 array floats + 3·48
/// scalar floats.
pub(crate) const ATTENTION_GANG_SMEM_BYTES: usize =
    (48 * 256 * 2 + 3 * 48) * core::mem::size_of::<f32>();

/// Issue 899 — smem for the GH=3 half-gang (24 rows): 24·256·2 array
/// floats, plus 3·24 scalar floats = 49,440 B (two of these with 2×1 KB
/// block reservation fit the 100 KB sm_89 per-SM budget → 2 blocks/SM).
pub(crate) const ATTENTION_GANG3_SMEM_BYTES: usize =
    (24 * 256 * 2 + 3 * 24) * core::mem::size_of::<f32>();

/// Issue 899 — smem for the GH=2 third-gang (16 rows): 16·256·2 array
/// floats, plus 3·16 scalar floats = 32,960 B (three of these with 3×1 KB
/// fit → 3 blocks/SM).
pub(crate) const ATTENTION_GANG2_SMEM_BYTES: usize =
    (16 * 256 * 2 + 3 * 16) * core::mem::size_of::<f32>();

pub(crate) const ATTENTION_PREFILL_GANG_CUDA_SRC: &str = r#"
// ===========================================================================
// Issue 898 + 899 — att_pf_mq8g{,3,2}: the head-ganged kv_group block and
// its occupancy ladder. One 256-thread block per (kv_group, q-tile, sub)
// serves GH heads — GH*8 staged q-rows, K row loaded once per tile, V row
// once per pp. Per-row chains are VERBATIM att_pf_mq8v (bit-identical by
// construction — see the Rust-side module doc for the register/smem
// accounting and the ladder table).
// ===========================================================================
#define ATTN_GANG_HEADS 6
#define ATTN_GANG_ROWS (ATTN_GANG_HEADS * ATTN_MQ) /* 48 */

__device__ __forceinline__ void att_pf_mq8g_body(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    float* __restrict__ attn_out,
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset)
{
    extern __shared__ float gang_smem[];
    float* s_w = gang_smem;                    /* [48][256] scores → weights */
    float* s_q = s_w + ATTN_GANG_ROWS * 256;   /* [48][256] staged q rows */
    float* s_rmx = s_q + ATTN_GANG_ROWS * 256; /* [48] running max (all-thread-identical) */
    float* s_rsm = s_rmx + ATTN_GANG_ROWS;     /* [48] running sum */
    float* s_tmx = s_rsm + ATTN_GANG_ROWS;     /* [48] per-tile max */

/* Dynamic-smem 2D indexing (float* rows — the static-smem kernels' real
 * [r][c] arrays don't exist here). */
#define GANG_SW(r, c) s_w[(r) * 256 + (c)]
#define GANG_SQ(r, c) s_q[(r) * 256 + (c)]

    const int tid = (int)threadIdx.x;
    const int tiles_per_head = (p + ATTN_MQ - 1) / ATTN_MQ;
    const int kv_group = (int)(blockIdx.x / tiles_per_head);
    const int q_base = (int)(blockIdx.x % tiles_per_head) * ATTN_MQ;
    const int q_count = (p - q_base) < ATTN_MQ ? (p - q_base) : ATTN_MQ;
    const int hpg = n_head / n_kv; /* == ATTN_GANG_HEADS (launcher-guarded) */
    const int head0 = kv_group * hpg;

    const int kv_head_off = kv_group * head_dim;
    const int q_stride = n_head * head_dim;
    const int kv_stride = n_kv * head_dim;

    /* Stage the 48 q rows (exact copies); init the softmax scalars
     * (same-value writes by every thread — benign). */
    for (int r = 0; r < ATTN_GANG_ROWS; r++) {
        const int qi8 = r & (ATTN_MQ - 1);
        const int h = r / ATTN_MQ;
        if (qi8 < q_count) {
            GANG_SQ(r, tid) = query[
                (long)(q_base + qi8) * q_stride + (head0 + h) * head_dim + tid];
        }
        s_rmx[r] = -1e30f;
        s_rsm[r] = 0.0f;
    }
    __syncthreads();

    float racc[ATTN_GANG_ROWS];
    for (int r = 0; r < ATTN_GANG_ROWS; r++) racc[r] = 0.0f;

    const int max_n_pos = q_offset + q_base + q_count;
    const int n_tiles = (max_n_pos + head_dim - 1) / head_dim;

    for (int tile = 0; tile < n_tiles; tile++) {
        const int tile_base = tile * head_dim;
        const int pos = tile_base + tid;

        /* L2 prefetch of tile t+1's K+V — VERBATIM from att_pf_mq8v. */
        if (tile + 1 < n_tiles) {
            const int pf_base = (tile + 1) * head_dim;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const int line = j * 256 + tid;
                const int half = line >> 11;      /* 0 = K, 1 = V */
                const int l = line & 2047;
                const int r = l >> 3;             /* row within tile */
                const int d = (l & 7) << 5;       /* 32-float line offset */
                const int ppos = pf_base + r;
                if (ppos < max_n_pos) {
                    const char* addr = (const char*)(
                        (half == 0 ? key : value) +
                        (long)ppos * kv_stride + kv_head_off + d);
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(addr)
                                 : "memory");
                }
            }
        }

        /* Phase 1: this thread's position's score for every staged row —
         * the K row loads ONCE (64 float4s, unroll-8 quad loop, as mq8v)
         * and feeds all 48 rows. Per-row chain d-ascending, operands
         * (s_q[r][d], k[d]) in mq8v's order. */
        float score[ATTN_GANG_ROWS];
        for (int r = 0; r < ATTN_GANG_ROWS; r++) score[r] = 0.0f;
        if (pos < max_n_pos) {
            const float4* k4 = reinterpret_cast<const float4*>(
                key + (long)pos * kv_stride + kv_head_off);
#pragma unroll 8
            for (int d4 = 0; d4 < head_dim / 4; d4++) {
                const float4 kv = k4[d4];
                const float ka[4] = { kv.x, kv.y, kv.z, kv.w };
                for (int c = 0; c < 4; c++) {
                    for (int r = 0; r < ATTN_GANG_ROWS; r++) {
                        score[r] = __fmaf_rn(GANG_SQ(r, d4 * 4 + c), ka[c],
                                             score[r]);
                    }
                }
            }
            for (int r = 0; r < ATTN_GANG_ROWS; r++) {
                score[r] = score[r] * scale;
            }
        }
        for (int r = 0; r < ATTN_GANG_ROWS; r++) {
            const int qi8 = r & (ATTN_MQ - 1);
            const bool vp = qi8 < q_count && pos < (q_offset + q_base + qi8 + 1);
            GANG_SW(r, tid) = (pos < max_n_pos && vp) ? score[r] : -1e30f;
        }
        __syncthreads();

        /* Phase 2: the max tree per row (identical stride structure). */
        if (tid < 128) {
            for (int r = 0; r < ATTN_GANG_ROWS; r++) {
                if (GANG_SW(r, tid + 128) > GANG_SW(r, tid)) {
                    GANG_SW(r, tid) = GANG_SW(r, tid + 128);
                }
            }
        }
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
                for (int r = 0; r < ATTN_GANG_ROWS; r++) {
                    if (GANG_SW(r, tid + stride) > GANG_SW(r, tid)) {
                        GANG_SW(r, tid) = GANG_SW(r, tid + stride);
                    }
                }
            }
            __syncthreads();
        }
        /* Tile max → s_tmx (tid 0, after the tree completes; the sync
         * below publishes it — replaces mq8v's per-thread tmax[qi] reg
         * read, saving 48 regs). */
        if (tid == 0) {
            for (int r = 0; r < ATTN_GANG_ROWS; r++) s_tmx[r] = GANG_SW(r, 0);
        }
        __syncthreads(); /* Issue 715 race (a) + s_tmx publication */

        /* Online softmax update (deferred-fma merge — the settled form).
         * rmax lives in smem; eprev in regs; etile is NOT held — Phase 3
         * recomputes it from identical operands (expf deterministic →
         * identical bits).
         *
         * Issue 899 race (c) — the s_rmx update is DEFERRED to the tid0
         * merge: an in-loop `s_rmx[r] = new_max` by ALL threads is a
         * cross-warp read-then-write hazard — warp B's `rmax = s_rmx[r]`
         * load can land AFTER warp A's same-tile store, so B computes
         * eprev = expf(new_max − new_max) = 1.0 instead of the rescale
         * (corrupted racc merge, ~1e-2-scale output error — fired by the
         * split-gang's warp scheduling at p=2048, tile-monotone; latent
         * in the shipped g6 since Issue 898 — the bug-#2 fix covered the
         * s_rsm RMW but missed this one). Reads here are PURE (no smem
         * writes this tile); new_max is recomputed identically wherever
         * needed (max over unchanged operands → identical bits). */
        float eprev[ATTN_GANG_ROWS];
        for (int r = 0; r < ATTN_GANG_ROWS; r++) {
            const float tmax = s_tmx[r];
            const float rmax = s_rmx[r];
            const float new_max = tmax > rmax ? tmax : rmax;
            eprev[r] = __expf(rmax - new_max);
        }

        /* Phase 3: weights (Mul form; 0 for invalid). tmax from s_tmx,
         * etile recomputed — `nm` recomputes new_max from the UNWRITTEN
         * s_rmx (same operands → same bits as the pre-fix s_rmx read-back);
         * the score comes from the REGISTER copy — the Phase-2 tree has
         * partially overwritten the s_w score columns (the mq8v register
         * form, Bench-892 bug #1). */
        for (int r = 0; r < ATTN_GANG_ROWS; r++) {
            const int qi8 = r & (ATTN_MQ - 1);
            const bool vp = qi8 < q_count && pos < (q_offset + q_base + qi8 + 1);
            const float tmax = s_tmx[r];
            const float nm = tmax > s_rmx[r] ? tmax : s_rmx[r];
            GANG_SW(r, tid) = vp
                ? __expf(tmax - nm) * __expf(score[r] - s_tmx[r])
                : 0.0f;
        }
        __syncthreads();

        /* Phase 4: serial accumulation ascending pp; the V row (one load
         * per pp) now serves all 48 rows (the 6× cut — mq8v's 6
         * head-blocks each re-loaded the same element). The causal bound
         * depends only on qi8, so the validity branch hoists out of the
         * head loop; each row's chain stays pp-ascending with unchanged
         * operand order. */
        float tsum[ATTN_GANG_ROWS], tacc[ATTN_GANG_ROWS];
        for (int r = 0; r < ATTN_GANG_ROWS; r++) {
            tsum[r] = 0.0f;
            tacc[r] = 0.0f;
        }
#pragma unroll 4
        for (int pp = 0; pp < head_dim; pp++) {
            const int kv_pos = tile_base + pp;
            if (kv_pos < max_n_pos) {
                const float v =
                    value[(long)kv_pos * kv_stride + kv_head_off + tid];
                for (int qi8 = 0; qi8 < ATTN_MQ; qi8++) {
                    if (qi8 < q_count && kv_pos < (q_offset + q_base + qi8 + 1)) {
                        for (int h = 0; h < ATTN_GANG_HEADS; h++) {
                            const int r = h * ATTN_MQ + qi8;
                            const float w = GANG_SW(r, pp);
                            tsum[r] = tsum[r] + w;
                            tacc[r] = __fmaf_rn(w, v, tacc[r]);
                        }
                    }
                }
            }
        }

        /* Merge (RESF=1: both fold into the accumulation adds). Thread 0
         * alone owns BOTH smem scalar updates (the Bench-892 bug-#2 fix
         * for s_rsm, extended by Issue 899 race (c): s_rmx's deferred
         * store lands here too — same value tid0 computes from unchanged
         * operands; the __syncthreads below publishes both before the
         * next tile's reads). tsum/eprev are all-thread-identical so
         * tid0's inputs are exact; racc stays in per-thread registers
         * (the vec-kernel form). */
        if (tid == 0) {
            for (int r = 0; r < ATTN_GANG_ROWS; r++) {
                const float tmax = s_tmx[r];
                const float rmax = s_rmx[r];
                const float new_max = tmax > rmax ? tmax : rmax;
                s_rsm[r] = __fmaf_rn(s_rsm[r], eprev[r], tsum[r]);
                s_rmx[r] = new_max;
            }
        }
        for (int r = 0; r < ATTN_GANG_ROWS; r++) {
            racc[r] = __fmaf_rn(racc[r], eprev[r], tacc[r]);
        }
        __syncthreads(); /* Issue 715 race (b) */
    }

    /* Final: 1/sum (div.full) + sigmoid gate. */
    for (int r = 0; r < ATTN_GANG_ROWS; r++) {
        const int qi8 = r & (ATTN_MQ - 1);
        const int h = r / ATTN_MQ;
        if (qi8 < q_count) {
            const float inv_sum = at_div_full(1.0f, s_rsm[r]);
            const float raw = racc[r] * inv_sum;
            const long off = (long)(q_base + qi8) * q_stride +
                             (long)(head0 + h) * head_dim + tid;
            const float g = gate[off];
            const float sig =
                at_div_full(1.0f, 1.0f + at_exp_fast(0.0f - g));
            attn_out[off] = raw * sig;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 1) att_pf_mq8g(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ gate,    /* [p, n_head, hd] */
    float* __restrict__ attn_out,      /* [p, n_head, hd] */
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset)
{
    att_pf_mq8g_body(query, key, value, gate, attn_out, head_dim, n_head,
                     n_kv, p, scale, q_offset);
}

// CUDA-graph twin of att_pf_mq8g (q_offset from the device buffer).
extern "C" __global__ void __launch_bounds__(256, 1) att_pf_mq8g_dp(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ gate,    /* [p, n_head, hd] */
    float* __restrict__ attn_out,      /* [p, n_head, hd] */
    int head_dim, int n_head, int n_kv, int p, float scale,
    const int* __restrict__ q_offset_dev)
{
    const int q_offset = *q_offset_dev;
    att_pf_mq8g_body(query, key, value, gate, attn_out, head_dim, n_head,
                     n_kv, p, scale, q_offset);
}

// ===========================================================================
// Issue 899 — the split-gang occupancy ladder. The Issue-898 full gang is
// smem-bound at 1 block/SM (8 resident warps); the ladder serves the 48
// rows from BPG = 6/GH co-resident blocks (16 / 24 warps at GH=3 / GH=2)
// at the same per-SM smem total, trading KV L2 re-reads (×BPG, still
// 6/BPG× better than the vec arm) for latency hiding and DECOUPLED
// barriers. The sub index varies FASTEST in the grid: consecutive blocks
// share the same (kv_group, q-tile) K/V slice AND q rows — the redundant
// fetches are L2-local. Per-row chains are the 898 body VERBATIM — a
// row's arithmetic never depends on which block owns it, so the ladder is
// bit-identical to att_pf_mq8v by the same construction (G1-pinned).
// __launch_bounds__(256, BPG) makes ptxas cap registers (≤128 at GH=3,
// ≤85 at GH=2) so the co-residency is real; the row register arrays
// scale down with GH, so the caps hold without spills.
// ===========================================================================
template <int GH>
__device__ __forceinline__ void att_pf_mq8gN_body(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    float* __restrict__ attn_out,
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset)
{
    constexpr int GROWS = GH * ATTN_MQ;        /* staged rows per block */
    constexpr int BPG = ATTN_GANG_HEADS / GH;  /* blocks per (kv_group, q-tile) */

    extern __shared__ float gang_smem[];
    float* s_w = gang_smem;                   /* [GROWS][256] scores → weights */
    float* s_q = s_w + GROWS * 256;           /* [GROWS][256] staged q rows */
    float* s_rmx = s_q + GROWS * 256;         /* [GROWS] running max (all-thread-identical) */
    float* s_rsm = s_rmx + GROWS;             /* [GROWS] running sum */
    float* s_tmx = s_rsm + GROWS;             /* [GROWS] per-tile max */

    const int tid = (int)threadIdx.x;
    const int tiles_per_head = (p + ATTN_MQ - 1) / ATTN_MQ;
    /* sub fastest: consecutive blocks share K/V + q rows (L2-local). */
    const int sub = (int)(blockIdx.x % BPG);
    const int tix = (int)(blockIdx.x / BPG);
    const int kv_group = tix / tiles_per_head;
    const int q_base = (tix % tiles_per_head) * ATTN_MQ;
    const int q_count = (p - q_base) < ATTN_MQ ? (p - q_base) : ATTN_MQ;
    const int hpg = n_head / n_kv; /* == ATTN_GANG_HEADS (launcher-guarded) */
    const int head0 = kv_group * hpg + sub * GH;

    const int kv_head_off = kv_group * head_dim;
    const int q_stride = n_head * head_dim;
    const int kv_stride = n_kv * head_dim;

    /* Stage this block's q rows (exact copies); init the softmax scalars
     * (same-value writes by every thread — benign). */
    for (int r = 0; r < GROWS; r++) {
        const int qi8 = r & (ATTN_MQ - 1);
        const int h = r / ATTN_MQ;
        if (qi8 < q_count) {
            GANG_SQ(r, tid) = query[
                (long)(q_base + qi8) * q_stride + (head0 + h) * head_dim + tid];
        }
        s_rmx[r] = -1e30f;
        s_rsm[r] = 0.0f;
    }
    __syncthreads();

    float racc[GROWS];
    for (int r = 0; r < GROWS; r++) racc[r] = 0.0f;

    const int max_n_pos = q_offset + q_base + q_count;
    const int n_tiles = (max_n_pos + head_dim - 1) / head_dim;

    for (int tile = 0; tile < n_tiles; tile++) {
        const int tile_base = tile * head_dim;
        const int pos = tile_base + tid;

        /* L2 prefetch of tile t+1's K+V — VERBATIM from att_pf_mq8v (the
         * BPG sibling blocks issue the same hints; L2 dedupes). */
        if (tile + 1 < n_tiles) {
            const int pf_base = (tile + 1) * head_dim;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const int line = j * 256 + tid;
                const int half = line >> 11;      /* 0 = K, 1 = V */
                const int l = line & 2047;
                const int r = l >> 3;             /* row within tile */
                const int d = (l & 7) << 5;       /* 32-float line offset */
                const int ppos = pf_base + r;
                if (ppos < max_n_pos) {
                    const char* addr = (const char*)(
                        (half == 0 ? key : value) +
                        (long)ppos * kv_stride + kv_head_off + d);
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(addr)
                                 : "memory");
                }
            }
        }

        /* Phase 1: this thread's position's score for every staged row —
         * the K row loads ONCE (float4, unroll-8 quad loop, as mq8v) and
         * feeds all GROWS rows. Per-row chain d-ascending, operands
         * (s_q[r][d], k[d]) in mq8v's order. */
        float score[GROWS];
        for (int r = 0; r < GROWS; r++) score[r] = 0.0f;
        if (pos < max_n_pos) {
            const float4* k4 = reinterpret_cast<const float4*>(
                key + (long)pos * kv_stride + kv_head_off);
#pragma unroll 8
            for (int d4 = 0; d4 < head_dim / 4; d4++) {
                const float4 kv = k4[d4];
                const float ka[4] = { kv.x, kv.y, kv.z, kv.w };
                for (int c = 0; c < 4; c++) {
                    for (int r = 0; r < GROWS; r++) {
                        score[r] = __fmaf_rn(GANG_SQ(r, d4 * 4 + c), ka[c],
                                             score[r]);
                    }
                }
            }
            for (int r = 0; r < GROWS; r++) {
                score[r] = score[r] * scale;
            }
        }
        for (int r = 0; r < GROWS; r++) {
            const int qi8 = r & (ATTN_MQ - 1);
            const bool vp = qi8 < q_count && pos < (q_offset + q_base + qi8 + 1);
            GANG_SW(r, tid) = (pos < max_n_pos && vp) ? score[r] : -1e30f;
        }
        __syncthreads();

        /* Phase 2: the max tree per row (identical stride structure). */
        if (tid < 128) {
            for (int r = 0; r < GROWS; r++) {
                if (GANG_SW(r, tid + 128) > GANG_SW(r, tid)) {
                    GANG_SW(r, tid) = GANG_SW(r, tid + 128);
                }
            }
        }
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
                for (int r = 0; r < GROWS; r++) {
                    if (GANG_SW(r, tid + stride) > GANG_SW(r, tid)) {
                        GANG_SW(r, tid) = GANG_SW(r, tid + stride);
                    }
                }
            }
            __syncthreads();
        }
        /* Tile max → s_tmx (tid 0, after the tree completes; the sync
         * below publishes it). */
        if (tid == 0) {
            for (int r = 0; r < GROWS; r++) s_tmx[r] = GANG_SW(r, 0);
        }
        __syncthreads(); /* Issue 715 race (a) + s_tmx publication */

        /* Online softmax update (deferred-fma merge — the settled form;
         * s_rmx reads are PURE this tile — the Issue-899 race-(c) fix,
         * see the g6 body comment). */
        float eprev[GROWS];
        for (int r = 0; r < GROWS; r++) {
            const float tmax = s_tmx[r];
            const float rmax = s_rmx[r];
            const float new_max = tmax > rmax ? tmax : rmax;
            eprev[r] = __expf(rmax - new_max);
        }

        /* Phase 3: weights (Mul form; 0 for invalid) — `nm` recomputes
         * new_max from the UNWRITTEN s_rmx (same operands → same bits);
         * the score comes from the REGISTER copy (Bench-892 bug #1). */
        for (int r = 0; r < GROWS; r++) {
            const int qi8 = r & (ATTN_MQ - 1);
            const bool vp = qi8 < q_count && pos < (q_offset + q_base + qi8 + 1);
            const float tmax = s_tmx[r];
            const float nm = tmax > s_rmx[r] ? tmax : s_rmx[r];
            GANG_SW(r, tid) = vp
                ? __expf(tmax - nm) * __expf(score[r] - s_tmx[r])
                : 0.0f;
        }
        __syncthreads();

        /* Phase 4: serial accumulation ascending pp — this block's GH
         * heads only (the row set splits across the BPG sibling blocks;
         * each row's chain is pp-ascending with unchanged operand
         * order). */
        float tsum[GROWS], tacc[GROWS];
        for (int r = 0; r < GROWS; r++) {
            tsum[r] = 0.0f;
            tacc[r] = 0.0f;
        }
#pragma unroll 4
        for (int pp = 0; pp < head_dim; pp++) {
            const int kv_pos = tile_base + pp;
            if (kv_pos < max_n_pos) {
                const float v =
                    value[(long)kv_pos * kv_stride + kv_head_off + tid];
                for (int qi8 = 0; qi8 < ATTN_MQ; qi8++) {
                    if (qi8 < q_count && kv_pos < (q_offset + q_base + qi8 + 1)) {
                        for (int h = 0; h < GH; h++) {
                            const int r = h * ATTN_MQ + qi8;
                            const float w = GANG_SW(r, pp);
                            tsum[r] = tsum[r] + w;
                            tacc[r] = __fmaf_rn(w, v, tacc[r]);
                        }
                    }
                }
            }
        }

        /* Merge (RESF=1) — tid0-owned s_rsm AND the deferred s_rmx store
         * (Issue 899 race (c)); racc per thread. */
        if (tid == 0) {
            for (int r = 0; r < GROWS; r++) {
                const float tmax = s_tmx[r];
                const float rmax = s_rmx[r];
                const float new_max = tmax > rmax ? tmax : rmax;
                s_rsm[r] = __fmaf_rn(s_rsm[r], eprev[r], tsum[r]);
                s_rmx[r] = new_max;
            }
        }
        for (int r = 0; r < GROWS; r++) {
            racc[r] = __fmaf_rn(racc[r], eprev[r], tacc[r]);
        }
        __syncthreads(); /* Issue 715 race (b) */
    }

    /* Final: 1/sum (div.full) + sigmoid gate — this block's rows only. */
    for (int r = 0; r < GROWS; r++) {
        const int qi8 = r & (ATTN_MQ - 1);
        const int h = r / ATTN_MQ;
        if (qi8 < q_count) {
            const float inv_sum = at_div_full(1.0f, s_rsm[r]);
            const float raw = racc[r] * inv_sum;
            const long off = (long)(q_base + qi8) * q_stride +
                             (long)(head0 + h) * head_dim + tid;
            const float g = gate[off];
            const float sig =
                at_div_full(1.0f, 1.0f + at_exp_fast(0.0f - g));
            attn_out[off] = raw * sig;
        }
    }
}

#define ATT_PF_MQ8G_SPLIT_ENTRY(SUF, GH)                                     \
extern "C" __global__ void __launch_bounds__(256, ATTN_GANG_HEADS / (GH))    \
att_pf_mq8##SUF(                                                            \
    const float* __restrict__ query,   /* [p, n_head, hd] */                 \
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */        \
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */        \
    const float* __restrict__ gate,    /* [p, n_head, hd] */                 \
    float* __restrict__ attn_out,      /* [p, n_head, hd] */                 \
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset)    \
{                                                                            \
    att_pf_mq8gN_body<GH>(query, key, value, gate, attn_out, head_dim,       \
                          n_head, n_kv, p, scale, q_offset);                 \
}                                                                            \
extern "C" __global__ void __launch_bounds__(256, ATTN_GANG_HEADS / (GH))    \
att_pf_mq8##SUF##_dp(                                                       \
    const float* __restrict__ query,   /* [p, n_head, hd] */                 \
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */        \
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */        \
    const float* __restrict__ gate,    /* [p, n_head, hd] */                 \
    float* __restrict__ attn_out,      /* [p, n_head, hd] */                 \
    int head_dim, int n_head, int n_kv, int p, float scale,                  \
    const int* __restrict__ q_offset_dev)                                    \
{                                                                            \
    const int q_offset = *q_offset_dev;                                      \
    att_pf_mq8gN_body<GH>(query, key, value, gate, attn_out, head_dim,       \
                          n_head, n_kv, p, scale, q_offset);                 \
}

/* Issue 899 ladder: GH=3 (24 rows, 2 blocks/SM) and GH=2 (16 rows,
 * 3 blocks/SM). GH=6 stays the Issue-898 att_pf_mq8g above, verbatim. */
ATT_PF_MQ8G_SPLIT_ENTRY(g3, 3)
ATT_PF_MQ8G_SPLIT_ENTRY(g2, 2)
"#;
