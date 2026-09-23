//! Plan 551 / Issue 773 T2 — the tensor-core attention score phase (verify
//! C-axis lever 2), COMPLETED from the 2026-08-28 CPU-window scaffold after
//! the G0 gate PASSED (`probe_551_g0_score_share`: the score phase is
//! 8.0–10.3 ms of the 70.63 ms verify chunk @20K, x16 attention layers;
//! pre-declared gate >= 8 ms).
//!
//! ## What this is
//!
//! A drop-in sibling of `attention_decode_splitgqa_partial_rows_qg_f32`
//! (the default verify attention at long ctx) whose SCORE phase runs on
//! tensor cores via **3xtf32** (`mma.sync.aligned.m16n8k8.row.col
//! .f32.tf32.tf32.f32` + the f32→tf32 hi/lo split, 3 mma per k-step:
//! hi·hi + hi·lo + lo·hi). Per-element error ~2⁻²¹-class vs f32's 2⁻²⁴ —
//! the numerics arm chosen because the dotma lesson (Bench 759: a
//! fold-order change in this exact dot flipped the chat lane's pos-96
//! near-tie) makes plain f16/bf16 mma a likely G1 failure. The f16 arm is
//! deliberately NOT authored (Plan 551: if 3xtf32 fails G1, record the
//! stop — never force the lossier arm).
//!
//! Everything EXCEPT the score phase is VERBATIM the SIMT body: the K-tile
//! staging (transposed), the P·V phase, the partial writes, and the
//! `part_m/part_l/part_out` contract the unchanged rows-combine consumes.
//! The online-softmax structure per task row is identical to the SIMT
//! body's (tile max → new_max → exp_prev/exp_tile → p=exp_tile·expf(s−m)
//! → run_sum rescale) — only the dot itself changes representation.
//!
//! ## Geometry (measured, not the plan's guess — Plan 551 T0)
//!
//! The dbirks 27B config is **n_head = 24, n_kv_head = 4 → g = 6**
//! (the plan's "n_embd/256 = 20 heads, g ∈ {4,5}" guess was wrong; the
//! attention Q/K/V width is 6144 on an n_embd=5120 GDN-hybrid model), and
//! full attention runs every 4th layer (`full_attention_interval = 4`).
//! grid = (n_kv_head, n_chunks, ceil(p/16)); block = 1024 (32 warps).
//! Per z-block: M = 16·g = 96 tasks (6 m16 fragments), N = 32 keys
//! (ATT_SPLIT_GQA_TILE), K = 256 dims.
//!
//! ## Tiling (Plan 551 T-a)
//!
//! Warp w ∈ [0, g) owns m-frag mf = w (ONE warp per m-frag — the scaffold
//! header's "warp < 4g, mf = w/4" mapping was internally inconsistent and
//! is corrected here: an m16n8k8 mma is warp-wide, so g warps × 4 n-frags
//! × 32 k-steps × 3 mma = 384 mma/warp/tile covers M=16g × N=32 exactly
//! once). The row max/sum reductions stay warp-local (quad = lanes with
//! equal groupID, `__shfl_xor` 1/2); the other 32−g warps skip the score
//! phase and keep the staging/PV/partial phases fed. T-b (2-tile N=64) is
//! NOT authored — G2 decides on T-a first (the plan's record-the-loser
//! rule applies to measured arms only).
//!
//! ## The gates (Plan 551, pre-declared — binding)
//!
//! G1: loop-vs-greedy token identity 0 mismatches ×3, BOTH payloads (doc
//! @20K + chat) — model-level, pending the heavy-harness window. G2:
//! verify-chunk ms @20K, verdict-stable ×2. G3: opt-in via
//! `QWEN38_VERIFY_ATTN_MMA` (default OFF; BROKEN — see the header below);
//! the decode path + existing kernels bit-untouched.
//!
//! # KNOWN-BROKEN (2026-08-28, the model-level G1 FAIL root-cause) — do NOT enable.
//!
//! The model-level gate (Plan 551's binding G1: loop-vs-greedy @20K, both
//! arms) FAILED catastrophically — 133/256 argmax flips, 504/512 loop
//! stream divergence — and the kernel-level root-cause chain (the fixtures
//! in `tests/qwen38_verify_attn_mma_g1.rs`) localized a HARD
//! fragment-assembly bug in the score phase: with tf32-EXACT random data
//! (lo=0, the split inactive) part_m is still 100% wrong (maxd 6.4 on a
//! ±1.2 score range); the tagged readout shows task-row cross-talk (odd
//! heads' partials never written; even heads' scores pair q with ~2×
//! out-of-range K rows; dim tags from OTHER (row,head) tasks). The original
//! ±0.01 random gate was VACUOUS: near-uniform softmax makes any score
//! error invisible on the consumed surface (max_rel 5.6e-6 while 97% of
//! output elements already differed bitwise). The T-a6/T-a24 tilings agree
//! bit-identically with each other (shared assembly bug). Fix = a follow-up
//! unit against the PTX ISA m16n8k8 fragment tables; until then
//! `QWEN38_VERIFY_ATTN_MMA` must stay unset (the default-off wiring is the
//! only guard — production decode/verify paths are untouched without it).
//! kernels bit-untouched. G4: the launcher reuses the caller's verify
//! buffers (no new device allocs per chunk).

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaStream};
use cudarc::driver::{LaunchConfig, PushKernelArg};

/// The mma-arm kernel source. `kv_elt`/`KV_RD` are redefined locally (the
/// f16-KV hatch composes by prepending `#define KV_F16 1` the same way the
/// main module does — same boundary-only conversion contract). The combine
/// kernel is copied VERBATIM from `ATTENTION_CUDA_SRC`
/// (`attention_decode_split_combine_rows_f32`) so this module is
/// self-contained (G3: no production-module changes).
pub(crate) const ATTENTION_SCORE_MMA_CUDA_SRC: &str = r#"
#ifdef KV_F16
typedef unsigned short kv_elt;
__device__ __forceinline__ float kv_f16_to_f32(unsigned short h)
{
    const unsigned int sign = ((unsigned int)h & 0x8000u) << 16;
    const unsigned int exp = ((unsigned int)h >> 10) & 0x1fu;
    unsigned int mant = (unsigned int)h & 0x3ffu;
    if (exp == 0x1fu) {
        return __uint_as_float(sign | 0x7f800000u
                               | (mant ? (0x400000u | (mant << 13)) : 0u));
    }
    if (exp == 0) {
        if (mant == 0) return __uint_as_float(sign);
        int s = 0;
        while (!(mant & 0x400u)) { mant <<= 1; s++; }
        return __uint_as_float(sign | ((unsigned int)(113 - s) << 23)
                               | ((mant & 0x3ffu) << 13));
    }
    return __uint_as_float(sign | (exp << 23) | (mant << 13));
}
#define KV_RD(p) (kv_f16_to_f32(*(p)))
#else
typedef float kv_elt;
#define KV_RD(p) (*(p))
#endif

#define ATT_MMA_TILE 32   // keys per tile (mirrors ATT_SPLIT_GQA_TILE)

// ── 3xtf32 primitives (PTX ISA m16n8k8, row.col, f32 accumulate) ──
__device__ __forceinline__ unsigned cvt_tf32(float x)
{
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}
// mma.sync m16n8k8 tf32: A 4x b32, B 2x b32, C/D 4x f32. NOT volatile:
// the "+f"/"r" constraints carry the data dependencies, and volatile
// forbids the scheduler from moving the next k-step's fragment loads
// past the mma batch — every k-step then pays full L1/L2 load latency
// serially (measured as a total G2 wash before this fix).
__device__ __forceinline__ void mma_m16n8k8_tf32(
    float* __restrict__ d,
    const unsigned* __restrict__ a,
    const unsigned* __restrict__ b)
{
    asm(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ void attention_score_mma_body(
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
    const int p)
{
    const int n_positions = base_pos + p;
    const int kv_group = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int n_chunks = gridDim.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp = tid >> 8;
    const int dim = tid & 255;
    const int g = n_head / n_kv_head;
    // Issue 754 T6 grid.z row-groups (the mma arm builds on the T6 shape):
    // block z owns the 16-row slice [z*16, z*16+16).
    const int row_off = blockIdx.z * 16;
    const int row_lim = p - row_off;

    float* k_smem = smem;                                        // hd*(T+1)
    float* p_smem = k_smem + head_dim * (ATT_MMA_TILE + 1);     // 16*g*T
    float* st_smem = p_smem + 16 * g * ATT_MMA_TILE;             // 16*g

    const int chunk_start = chunk_id * chunk_len;
    if (chunk_start >= n_positions) {
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

    // PV accumulators (VERBATIM the SIMT body).
    float acc[4][8];
#pragma unroll
    for (int u = 0; u < 4; u++)
#pragma unroll
        for (int h = 0; h < 8; h++) acc[u][h] = 0.0f;

    // Plan 551 T-a: warp w < g owns m-frag mf = w and ALL 4 n-frags.
    const int mf = warp;
    const bool score_warp = warp < g;
    // Online-softmax state per row-half (rows gid and gid+8 of the
    // m-frag); every lane of the quad holds the same value after the
    // xor-reductions, so no broadcast is needed.
    float run_max[2];
    float run_sum[2];
    run_max[0] = -1e30f; run_max[1] = -1e30f;
    run_sum[0] = 0.0f;   run_sum[1] = 0.0f;

    const int tile0 = chunk_start / ATT_MMA_TILE;
    const int tile1 = (chunk_end + ATT_MMA_TILE - 1) / ATT_MMA_TILE;
    const int ks = ATT_MMA_TILE + 1;
    for (int tile = tile0; tile < tile1; tile++) {
        const int tile_base = tile * ATT_MMA_TILE;

        // (A) Stage K tile — VERBATIM the SIMT body (transposed stage).
        for (int i = tid; i < ATT_MMA_TILE * head_dim; i += blockDim.x) {
            const int r_ = i / head_dim;
            const int d = i % head_dim;
            k_smem[d * ks + r_] = KV_RD(&key_cache[(tile_base + r_) * kv_stride + kv_off + d]);
        }
        __syncthreads();

        // (B) Score phase — the 3xtf32 mma arm (the ONLY changed phase).
        if (score_warp) {
            // D fragments: 4 n-frags x 4 f32 per lane.
            // c0=(gid,tig*2) c1=(gid,tig*2+1) c2=(gid+8,tig*2) c3=(gid+8,tig*2+1)
            float d_acc[4][4];
#pragma unroll
            for (int kf = 0; kf < 4; kf++)
#pragma unroll
                for (int e = 0; e < 4; e++) d_acc[kf][e] = 0.0f;

            const int gid = lane >> 2;
            const int tig = lane & 3;
            // k8 loop: head_dim/8 = 32 steps (launcher contract). No
            // unroll pragma — measured: unroll-8 blows the 64-reg
            // launch_bounds(1024) budget and spills to local memory (G2
            // 0.683x, WORSE than no-pragma); non-volatile mma + the float2
            // A-loads alone let the scheduler overlap the next k-step's
            // loads with the current mma batch.
            for (int k8 = 0; k8 < head_dim; k8 += 8) {
                // A fragments (this warp's m-frag): task rows t16 =
                // mf*16 + {gid, gid+8}, k = k8 + 2*tig + {0,1} — ONE float2
                // load per half (the two k-elements are consecutive).
                unsigned a_hi[4], a_lo[4];
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int t16 = mf * 16 + half * 8 + gid;
                    const int r = row_off + t16 / g;
                    const int h = t16 % g;
                    if (r < p) {
                        const float2 qv = *reinterpret_cast<const float2*>(
                            query + (long)r * q_stride
                            + (kv_group * g + h) * head_dim + k8 + 2 * tig);
                        const float q0_hi = __uint_as_float(cvt_tf32(qv.x));
                        const float q0_lo = qv.x - q0_hi;
                        const float q1_hi = __uint_as_float(cvt_tf32(qv.y));
                        const float q1_lo = qv.y - q1_hi;
                        a_hi[half * 2] = cvt_tf32(q0_hi);
                        a_lo[half * 2] = cvt_tf32(q0_lo);
                        a_hi[half * 2 + 1] = cvt_tf32(q1_hi);
                        a_lo[half * 2 + 1] = cvt_tf32(q1_lo);
                    } else {
                        a_hi[half * 2] = 0u;
                        a_lo[half * 2] = 0u;
                        a_hi[half * 2 + 1] = 0u;
                        a_lo[half * 2 + 1] = 0u;
                    }
                }
                // B fragments: 4 n-frags, 2 k-elems per lane each
                // (b0 = B[tig*2][gid], b1 = B[tig*2+1][gid]; the transposed
                // k_smem gives K[key][dim] at [dim*ks + key]).
#pragma unroll
                for (int kf = 0; kf < 4; kf++) {
                    const float k0 = k_smem[(k8 + 2 * tig) * ks + (kf * 8 + gid)];
                    const float k1 = k_smem[(k8 + 2 * tig + 1) * ks + (kf * 8 + gid)];
                    const float k0_hi = __uint_as_float(cvt_tf32(k0));
                    const float k0_lo = k0 - k0_hi;
                    const float k1_hi = __uint_as_float(cvt_tf32(k1));
                    const float k1_lo = k1 - k1_hi;
                    const unsigned b_hi[2] = {cvt_tf32(k0_hi), cvt_tf32(k1_hi)};
                    const unsigned b_lo[2] = {cvt_tf32(k0_lo), cvt_tf32(k1_lo)};
                    // 3xtf32: d += a_hi*b_hi + a_hi*b_lo + a_lo*b_hi
                    // (the a_lo*b_lo term is dropped — 2^-22-squared class).
                    mma_m16n8k8_tf32(d_acc[kf], a_hi, b_hi);
                    mma_m16n8k8_tf32(d_acc[kf], a_hi, b_lo);
                    mma_m16n8k8_tf32(d_acc[kf], a_lo, b_hi);
                }
            }

            // Scale + causal mask in place (the SIMT body applies scale to
            // the dot before the tile max; invalid keys score -1e30).
#pragma unroll
            for (int kf = 0; kf < 4; kf++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int half = e >> 1;
                    const int t16 = mf * 16 + half * 8 + gid;
                    const int r_rel = t16 / g;
                    const int col = kf * 8 + tig * 2 + (e & 1);
                    const int pos = tile_base + col;
                    const bool valid = (r_rel < row_lim)
                        && (pos < chunk_end)
                        && (pos <= base_pos + row_off + r_rel);
                    float s = d_acc[kf][e] * scale;
                    d_acc[kf][e] = valid ? s : -1e30f;
                }
            }

            // Online-softmax update per row-half — the SIMT contract
            // (tile max -> new_max -> exp_prev/exp_tile -> p -> rescale).
            float st_write[2];
#pragma unroll
            for (int half = 0; half < 2; half++) {
                float m = fmaxf(d_acc[0][half * 2], d_acc[0][half * 2 + 1]);
#pragma unroll
                for (int kf = 1; kf < 4; kf++) {
                    m = fmaxf(m, d_acc[kf][half * 2]);
                    m = fmaxf(m, d_acc[kf][half * 2 + 1]);
                }
                // Quad reduce (lanes with equal groupID hold the row).
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                const float m_tile = m;
                const float new_max = (m_tile > run_max[half]) ? m_tile : run_max[half];
                const float exp_prev = expf(run_max[half] - new_max);
                const float exp_tile = expf(m_tile - new_max);
                float s = 0.0f;
#pragma unroll
                for (int kf = 0; kf < 4; kf++) {
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        const int idx = half * 2 + e;
                        const int t16 = mf * 16 + half * 8 + gid;
                        const int r_rel = t16 / g;
                        const int col = kf * 8 + tig * 2 + (e & 1);
                        const int pos = tile_base + col;
                        const bool valid = (r_rel < row_lim)
                            && (pos < chunk_end)
                            && (pos <= base_pos + row_off + r_rel);
                        float pv = 0.0f;
                        if (valid) pv = exp_tile * expf(d_acc[kf][idx] - m_tile);
                        d_acc[kf][idx] = pv;
                        s += pv;
                    }
                }
                s += __shfl_xor_sync(0xffffffffu, s, 1);
                s += __shfl_xor_sync(0xffffffffu, s, 2);
                run_sum[half] = run_sum[half] * exp_prev + s;
                run_max[half] = new_max;
                st_write[half] = exp_prev;
            }

            // p_smem / st_smem handoff (the PV phase reads both).
#pragma unroll
            for (int kf = 0; kf < 4; kf++) {
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int t16 = mf * 16 + half * 8 + gid;
                    p_smem[t16 * ATT_MMA_TILE + kf * 8 + tig * 2] = d_acc[kf][half * 2];
                    p_smem[t16 * ATT_MMA_TILE + kf * 8 + tig * 2 + 1] =
                        d_acc[kf][half * 2 + 1];
                }
            }
            if (tig == 0) {
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int t16 = mf * 16 + half * 8 + gid;
                    st_smem[t16] = st_write[half];
                }
            }
        }
        __syncthreads();

        // (C) P.V phase — VERBATIM the SIMT body.
        const int l_valid = (chunk_end - tile_base < ATT_MMA_TILE)
                                ? (chunk_end - tile_base)
                                : ATT_MMA_TILE;
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int r = grp * 4 + u;
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
                const int r = grp * 4 + u;
                if (r < row_lim) {
#pragma unroll
                    for (int h = 0; h < 8; h++) {
                        if (h < g) {
                            acc[u][h] +=
                                p_smem[(r * g + h) * ATT_MMA_TILE + l] * v;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // part_out partial writes — VERBATIM the SIMT body (all threads).
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
    // part_m / part_l writes — the MMA mapping of the SIMT tail: score
    // warp mf covers rows mf*16 + {gid, gid+8} (tig == 0 lane writes;
    // every lane holds the reduced stats).
    if (score_warp && (lane & 3) == 0) {
#pragma unroll
        for (int half = 0; half < 2; half++) {
            const int t16 = mf * 16 + half * 8 + (lane >> 2);
            const int r_rel = t16 / g;
            if (r_rel < row_lim) {
                const int p_idx =
                    ((row_off + r_rel) * n_head + kv_group * g + (t16 % g)) * n_chunks
                    + chunk_id;
                part_m[p_idx] = run_max[half];
                part_l[p_idx] = run_sum[half];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(1024)
attention_decode_splitgqa_partial_rows_qg_mma_f32(
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
    const int base_pos,
    const int p)
{
    extern __shared__ float smem[];
    attention_score_mma_body(query, key_cache, value_cache, part_m, part_l,
                             part_out, smem, scale, head_dim, n_head, n_kv_head,
                             chunk_len, base_pos, p);
}

// Plan 551 T-a24 — the second pre-declared tiling: warp-per-(m-frag x
// n-frag), 4g active warps (24 at g=6) instead of g (6). The measured
// motivation: the T-a6 arm's score window (0.71 ms) was SLOWER than SIMT
// (0.53 ms) despite the ~50-100x FLOP-rate tensor cores — only 6/32
// warps issued loads during the score window (memory-level parallelism
// collapsed on a latency-bound kernel). T-a24 keeps 4x the warps issuing,
// at the cost of a smem exchange for the cross-n-frag row max/sum (the
// row max needs all 4 n-frags = 4 warps; two uniform barriers per tile).
// Extra smem: red_max/red_sum [g][2][8][4] each (768 floats at g=6).
static __device__ void attention_score_mma24_body(
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
    const int p)
{
    const int n_positions = base_pos + p;
    const int kv_group = blockIdx.x;
    const int chunk_id = blockIdx.y;
    const int n_chunks = gridDim.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp = tid >> 8;
    const int dim = tid & 255;
    const int g = n_head / n_kv_head;
    const int row_off = blockIdx.z * 16;
    const int row_lim = p - row_off;

    float* k_smem = smem;
    float* p_smem = k_smem + head_dim * (ATT_MMA_TILE + 1);
    float* st_smem = p_smem + 16 * g * ATT_MMA_TILE;
    float* red_max = st_smem + 16 * g;            // [g][2][8][4]
    float* red_sum = red_max + g * 2 * 8 * 4;     // [g][2][8][4]

    const int chunk_start = chunk_id * chunk_len;
    if (chunk_start >= n_positions) {
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

    float acc[4][8];
#pragma unroll
    for (int u = 0; u < 4; u++)
#pragma unroll
        for (int h = 0; h < 8; h++) acc[u][h] = 0.0f;

    // T-a24 ownership: warp w < 4g owns ONE (m-frag, n-frag) tile.
    const int mf = warp >> 2;
    const int kf_own = warp & 3;
    const bool score_warp = warp < 4 * g;
    float run_max[2];
    float run_sum[2];
    run_max[0] = -1e30f; run_max[1] = -1e30f;
    run_sum[0] = 0.0f;   run_sum[1] = 0.0f;

    const int tile0 = chunk_start / ATT_MMA_TILE;
    const int tile1 = (chunk_end + ATT_MMA_TILE - 1) / ATT_MMA_TILE;
    const int ks = ATT_MMA_TILE + 1;
    for (int tile = tile0; tile < tile1; tile++) {
        const int tile_base = tile * ATT_MMA_TILE;

        // (A) Stage K tile — VERBATIM.
        for (int i = tid; i < ATT_MMA_TILE * head_dim; i += blockDim.x) {
            const int r_ = i / head_dim;
            const int d = i % head_dim;
            k_smem[d * ks + r_] = KV_RD(&key_cache[(tile_base + r_) * kv_stride + kv_off + d]);
        }
        __syncthreads();

        // (B) Score phase, step 1: the mma product for this warp's tile.
        // d_acc/sc live at the TILE-LOOP level (they carry the scores and
        // the softmax scalars across the two intra-score uniform barriers).
        float d_acc[4];
        float sc[4];
#pragma unroll
        for (int e = 0; e < 4; e++) d_acc[e] = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; e++) sc[e] = 0.0f;
        if (score_warp) {
            const int gid = lane >> 2;
            const int tig = lane & 3;
            for (int k8 = 0; k8 < head_dim; k8 += 8) {
                unsigned a_hi[4], a_lo[4];
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int t16 = mf * 16 + half * 8 + gid;
                    const int r = row_off + t16 / g;
                    const int h = t16 % g;
                    if (r < p) {
                        const float2 qv = *reinterpret_cast<const float2*>(
                            query + (long)r * q_stride
                            + (kv_group * g + h) * head_dim + k8 + 2 * tig);
                        const float q0_hi = __uint_as_float(cvt_tf32(qv.x));
                        const float q0_lo = qv.x - q0_hi;
                        const float q1_hi = __uint_as_float(cvt_tf32(qv.y));
                        const float q1_lo = qv.y - q1_hi;
                        a_hi[half * 2] = cvt_tf32(q0_hi);
                        a_lo[half * 2] = cvt_tf32(q0_lo);
                        a_hi[half * 2 + 1] = cvt_tf32(q1_hi);
                        a_lo[half * 2 + 1] = cvt_tf32(q1_lo);
                    } else {
                        a_hi[half * 2] = 0u;
                        a_lo[half * 2] = 0u;
                        a_hi[half * 2 + 1] = 0u;
                        a_lo[half * 2 + 1] = 0u;
                    }
                }
                // B fragment: this warp's OWN n-frag only.
                const float k0 = k_smem[(k8 + 2 * tig) * ks + (kf_own * 8 + gid)];
                const float k1 = k_smem[(k8 + 2 * tig + 1) * ks + (kf_own * 8 + gid)];
                const float k0_hi = __uint_as_float(cvt_tf32(k0));
                const float k0_lo = k0 - k0_hi;
                const float k1_hi = __uint_as_float(cvt_tf32(k1));
                const float k1_lo = k1 - k1_hi;
                const unsigned b_hi[2] = {cvt_tf32(k0_hi), cvt_tf32(k1_hi)};
                const unsigned b_lo[2] = {cvt_tf32(k0_lo), cvt_tf32(k1_lo)};
                mma_m16n8k8_tf32(d_acc, a_hi, b_hi);
                mma_m16n8k8_tf32(d_acc, a_hi, b_lo);
                mma_m16n8k8_tf32(d_acc, a_lo, b_hi);
            }

            // Scale + causal mask in place; quad-max per row-half (this
            // n-frag's contribution) -> red_max exchange.
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int half = e >> 1;
                const int t16 = mf * 16 + half * 8 + gid;
                const int r_rel = t16 / g;
                const int col = kf_own * 8 + tig * 2 + (e & 1);
                const int pos = tile_base + col;
                const bool valid = (r_rel < row_lim)
                    && (pos < chunk_end)
                    && (pos <= base_pos + row_off + r_rel);
                float s = d_acc[e] * scale;
                d_acc[e] = valid ? s : -1e30f;
            }
            // Quad reduce on ALL lanes (the 0xffffffff shuffle mask
            // requires warp-uniform participation — guarding the shuffle
            // itself by tig==0 deadlocks); only the red_max write is
            // tig==0-guarded.
#pragma unroll
            for (int half = 0; half < 2; half++) {
                float m = fmaxf(d_acc[half * 2], d_acc[half * 2 + 1]);
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                if (tig == 0) {
                    red_max[((mf * 2 + half) * 8 + gid) * 4 + kf_own] = m;
                }
            }
        }
        __syncthreads();  // uniform: red_max visible to the 4 n-frag warps

        // (B) step 2: pv for own 4 scores; quad-sum -> red_sum exchange.
        // (d_acc + sc carry state across this region's entry barrier.)
        if (score_warp) {
            const int gid = lane >> 2;
            const int tig = lane & 3;
#pragma unroll
            for (int half = 0; half < 2; half++) {
                float m_tile = -1e30f;
#pragma unroll
                for (int kf = 0; kf < 4; kf++) {
                    m_tile = fmaxf(
                        m_tile, red_max[((mf * 2 + half) * 8 + gid) * 4 + kf]);
                }
                const float new_max =
                    (m_tile > run_max[half]) ? m_tile : run_max[half];
                const float exp_prev = expf(run_max[half] - new_max);
                const float exp_tile = expf(m_tile - new_max);
                float s = 0.0f;
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int idx = half * 2 + e;
                    const int t16 = mf * 16 + half * 8 + gid;
                    const int r_rel = t16 / g;
                    const int col = kf_own * 8 + tig * 2 + (e & 1);
                    const int pos = tile_base + col;
                    const bool valid = (r_rel < row_lim)
                        && (pos < chunk_end)
                        && (pos <= base_pos + row_off + r_rel);
                    float pv = 0.0f;
                    if (valid) pv = exp_tile * expf(d_acc[idx] - m_tile);
                    d_acc[idx] = pv;
                    s += pv;
                }
                s += __shfl_xor_sync(0xffffffffu, s, 1);
                s += __shfl_xor_sync(0xffffffffu, s, 2);
                sc[half * 2] = new_max;
                sc[half * 2 + 1] = exp_prev;
                if (tig == 0) {
                    red_sum[((mf * 2 + half) * 8 + gid) * 4 + kf_own] = s;
                }
            }
        }
        __syncthreads();  // uniform: red_sum visible

        // (B) step 3: read the row's 4 subtotals, rescale the running
        // stats, write p_smem/st_smem.
        if (score_warp) {
            const int gid = lane >> 2;
            const int tig = lane & 3;
#pragma unroll
            for (int half = 0; half < 2; half++) {
                float s_tile = 0.0f;
#pragma unroll
                for (int kf = 0; kf < 4; kf++) {
                    s_tile += red_sum[((mf * 2 + half) * 8 + gid) * 4 + kf];
                }
                run_sum[half] = run_sum[half] * sc[half * 2 + 1] + s_tile;
                run_max[half] = sc[half * 2];
            }
#pragma unroll
            for (int half = 0; half < 2; half++) {
                const int t16 = mf * 16 + half * 8 + gid;
                p_smem[t16 * ATT_MMA_TILE + kf_own * 8 + tig * 2] = d_acc[half * 2];
                p_smem[t16 * ATT_MMA_TILE + kf_own * 8 + tig * 2 + 1] =
                    d_acc[half * 2 + 1];
            }
            if (kf_own == 0 && tig == 0) {
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int t16 = mf * 16 + half * 8 + gid;
                    st_smem[t16] = sc[half * 2 + 1];
                }
            }
        }
        __syncthreads();

        // (C) P.V phase — VERBATIM the SIMT body.
        const int l_valid = (chunk_end - tile_base < ATT_MMA_TILE)
                                ? (chunk_end - tile_base)
                                : ATT_MMA_TILE;
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const int r = grp * 4 + u;
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
                const int r = grp * 4 + u;
                if (r < row_lim) {
#pragma unroll
                    for (int h = 0; h < 8; h++) {
                        if (h < g) {
                            acc[u][h] +=
                                p_smem[(r * g + h) * ATT_MMA_TILE + l] * v;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // part_out partial writes — VERBATIM.
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
    // part_m / part_l — the kf_own == 0 warp of each m-frag writes.
    if (score_warp && kf_own == 0 && (lane & 3) == 0) {
#pragma unroll
        for (int half = 0; half < 2; half++) {
            const int t16 = mf * 16 + half * 8 + (lane >> 2);
            const int r_rel = t16 / g;
            if (r_rel < row_lim) {
                const int p_idx =
                    ((row_off + r_rel) * n_head + kv_group * g + (t16 % g)) * n_chunks
                    + chunk_id;
                part_m[p_idx] = run_max[half];
                part_l[p_idx] = run_sum[half];
            }
        }
    }
}

// The graph-capture twin: base_pos read from a device buffer at kernel
// runtime (the T9.12 contract; the launcher pins the FIXED max chunk
// count and dead chunks write neutral partials).
extern "C" __global__ void __launch_bounds__(1024)
attention_decode_splitgqa_partial_rows_qg_mma_f32_devpos(
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
    const int* __restrict__ pos_dev,
    const int p)
{
    extern __shared__ float smem[];
    const int base_pos = *pos_dev;
    attention_score_mma_body(query, key_cache, value_cache, part_m, part_l,
                             part_out, smem, scale, head_dim, n_head, n_kv_head,
                             chunk_len, base_pos, p);
}

// T-a24 entry points (the second pre-declared tiling).
extern "C" __global__ void __launch_bounds__(1024)
attention_decode_splitgqa_partial_rows_qg_mma24_f32(
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
    const int base_pos,
    const int p)
{
    extern __shared__ float smem[];
    attention_score_mma24_body(query, key_cache, value_cache, part_m, part_l,
                               part_out, smem, scale, head_dim, n_head, n_kv_head,
                               chunk_len, base_pos, p);
}

extern "C" __global__ void __launch_bounds__(1024)
attention_decode_splitgqa_partial_rows_qg_mma24_f32_devpos(
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
    const int* __restrict__ pos_dev,
    const int p)
{
    extern __shared__ float smem[];
    const int base_pos = *pos_dev;
    attention_score_mma24_body(query, key_cache, value_cache, part_m, part_l,
                               part_out, smem, scale, head_dim, n_head, n_kv_head,
                               chunk_len, base_pos, p);
}

// VERBATIM copy of `attention_decode_split_combine_rows_f32` from
// ATTENTION_CUDA_SRC — the unchanged combine this module's partials feed
// (self-contained module; no production-source coupling).
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
"#;

/// Plan 551 T1: the compiled mma score-phase kernels (scalar + devpos
/// twins + the verbatim combine), loaded by the env-gated wiring in
/// `qwen38_dense_cudarc` when `QWEN38_VERIFY_ATTN_MMA=1`.
pub struct AttentionScoreMmaKernels {
    score_mma: CudaFunction,
    score_mma_devpos: CudaFunction,
    score_mma24: CudaFunction,
    score_mma24_devpos: CudaFunction,
    combine: CudaFunction,
    _module: Arc<cudarc::driver::safe::CudaModule>,
}

impl AttentionScoreMmaKernels {
    pub fn new(ctx: Arc<CudaContext>, kv_f16: bool) -> Result<Self, super::CudarcKernelError> {
        let src = if kv_f16 {
            format!("#define KV_F16 1\n{ATTENTION_SCORE_MMA_CUDA_SRC}")
        } else {
            ATTENTION_SCORE_MMA_CUDA_SRC.to_string()
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
        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| super::CudarcKernelError::Compile(format!("{e}")))
        };
        let score_mma = load("attention_decode_splitgqa_partial_rows_qg_mma_f32")?;
        let score_mma_devpos =
            load("attention_decode_splitgqa_partial_rows_qg_mma_f32_devpos")?;
        let score_mma24 = load("attention_decode_splitgqa_partial_rows_qg_mma24_f32")?;
        let score_mma24_devpos =
            load("attention_decode_splitgqa_partial_rows_qg_mma24_f32_devpos")?;
        let combine = load("attention_decode_split_combine_rows_f32")?;
        // smem opt-in (the production qg pattern): the T-a6 pair's
        // k[256][33] + p[16*g][32] + st[16*g] reaches 50,688 B at g=8; the
        // T-a24 pair adds red_max/red_sum ([g][2][8][4] each) reaching
        // 54,784 B at g=8 — one ceiling covers both (under the 99 KB
        // sm_89 per-block max).
        let qg_smem =
            (256 * 33 + 16 * 8 * 32 + 16 * 8 + 2 * (8 * 2 * 8 * 4))
                * core::mem::size_of::<f32>() as i32;
        for f in [
            &score_mma,
            &score_mma_devpos,
            &score_mma24,
            &score_mma24_devpos,
        ] {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                qg_smem,
            )
            .map_err(|e| {
                super::CudarcKernelError::Launch(format!("score mma smem opt-in: {e}"))
            })?;
        }
        Ok(Self {
            score_mma,
            score_mma_devpos,
            score_mma24,
            score_mma24_devpos,
            combine,
            _module: module,
        })
    }

    fn validate(head_dim: usize, n_head: usize, n_kv_head: usize, p: usize) -> Result<(), super::CudarcKernelError> {
        if head_dim != 256 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "score mma: head_dim must be 256 (got {head_dim})"
            )));
        }
        if !n_head.is_multiple_of(n_kv_head) || n_head / n_kv_head == 0 || n_head / n_kv_head > 8 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "score mma: GQA group {g} outside [1, 8]",
                g = n_head / n_kv_head
            )));
        }
        if p == 0 || p > 64 {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "score mma: p must be in 1..=64 (got {p})"
            )));
        }
        Ok(())
    }

    /// The mma twin of `launch_attention_splitgqa_rows_qg` — same buffers,
    /// grid, smem plan, and the unchanged rows-combine. G4: reuses the
    /// caller's verify buffers.
    ///
    /// # Safety
    ///
    /// Same shape contracts as the production qg launcher.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_splitgqa_rows_qg_mma(
        &self,
        arm24: bool,
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
        Self::validate(head_dim, n_head, n_kv_head, p)?;
        if !chunk_len.is_multiple_of(32) {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "score mma: chunk_len {chunk_len} not a multiple of the 32-position tile"
            )));
        }
        let g = n_head / n_kv_head;
        // T-a24 adds red_max/red_sum ([g][2][8][4] each).
        let red = if arm24 { 2 * (g * 2 * 8 * 4) } else { 0 };
        let smem_floats = head_dim * 33 + 16 * g * 32 + 16 * g + red;
        let func = if arm24 { &self.score_mma24 } else { &self.score_mma };
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
                .launch_builder(func)
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
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        self.launch_combine(stream, part_m, part_l, part_out, attn_out, head_dim, n_head, n_chunks, p)?;
        Ok(())
    }

    /// The graph-capture twin: `base_pos` read from `pos_dev` at kernel
    /// runtime; the caller pins the FIXED max chunk count (the T9.12
    /// contract — dead chunks write neutral partials).
    ///
    /// # Safety
    ///
    /// Same shape contracts as the production qg devpos launcher.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_splitgqa_rows_qg_mma_devpos(
        &self,
        arm24: bool,
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
        Self::validate(head_dim, n_head, n_kv_head, p)?;
        if !chunk_len.is_multiple_of(32) {
            return Err(super::CudarcKernelError::InvalidArg(format!(
                "score mma devpos: chunk_len {chunk_len} not a multiple of the 32-position tile"
            )));
        }
        let g = n_head / n_kv_head;
        let red = if arm24 { 2 * (g * 2 * 8 * 4) } else { 0 };
        let smem_floats = head_dim * 33 + 16 * g * 32 + 16 * g + red;
        let func = if arm24 {
            &self.score_mma24_devpos
        } else {
            &self.score_mma_devpos
        };
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
                .launch_builder(func)
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
                .launch(cfg)
                .map_err(|e| super::CudarcKernelError::Launch(e.to_string()))?;
        }
        self.launch_combine(stream, part_m, part_l, part_out, attn_out, head_dim, n_head, n_chunks, p)?;
        Ok(())
    }

    /// The unchanged rows-combine (verbatim kernel, same launch as the
    /// production qg launcher's tail).
    #[allow(clippy::too_many_arguments)]
    fn launch_combine(
        &self,
        stream: &CudaStream,
        part_m: &cudarc::driver::safe::CudaSlice<f32>,
        part_l: &cudarc::driver::safe::CudaSlice<f32>,
        part_out: &cudarc::driver::safe::CudaSlice<f32>,
        attn_out: &cudarc::driver::safe::CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_chunks: usize,
        p: usize,
    ) -> Result<(), super::CudarcKernelError> {
        let cfg2 = LaunchConfig {
            grid_dim: ((p * n_head) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (hd_i, nh_i, nc_i, p_i) =
            (head_dim as i32, n_head as i32, n_chunks as i32, p as i32);
        unsafe {
            stream
                .launch_builder(&self.combine)
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
