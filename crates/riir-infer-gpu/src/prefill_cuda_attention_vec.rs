//! Issue 896 — the vectorized-MLP mq8 prefill-attention kernels
//! (`att_pf_mq8v[_dp]`), source for the NVRTC module.
//!
//! Pivoted from the issue's literal "deeper L2 prefetch" mechanism: at
//! prefill shapes (bp=0) each block's KV slice is L2-resident (≤16 MB vs
//! 72 MB L2), so prefetching *further ahead* only deepens the resident-block
//! working-set thrash (512 blocks × 512 KB active windows ≈ 268 MB > L2).
//! The in-kernel lever that adds no traffic: widen per-thread memory-level
//! parallelism in the two row-scan phases —
//! - Phase 1: the K row (256 floats) loads as 64 `float4`s, `#pragma
//!   unroll 8` on the quad loop (8 × 16 B in flight per thread vs 4 × 4 B);
//! - Phase 4: `#pragma unroll 4` on the pp loop so the V-row loads pipeline
//!   across row boundaries.
//!
//! The fma chains stay d/pp-ascending with the same operand order — the
//! T1.6 bit-identity precedent (load reordering that preserves chain order
//! never changes bits); the G1 pins assert to_bits equality anyway.
//!
//! The 1-tile L2 prefetch block is kept VERBATIM from `att_pf_mq8p`: it is
//! inert at prefill shapes (L2-resident) and earns its keep in the
//! DRAM-resident verify regime (1.37× at bp=24576, Bench 732 sweep).
//!
//! Everything else — phases 2/3, the online-softmax merge, the final
//! divide + sigmoid gate, `__launch_bounds__` — is byte-identical to
//! `att_pf_mq8p` (prefill_cuda_attention.rs).

pub(crate) const ATTENTION_PREFILL_VEC_CUDA_SRC: &str = r#"
// ===========================================================================
// Issue 896 — att_pf_mq8v: the vectorized-MLP twin of att_pf_mq8p.
// VERBATIM serial arithmetic (bit-identical by construction — the only
// deltas are load WIDTH (float4) and issue DEPTH (unroll), which change
// neither the operand order nor the accumulator chains). See the Rust-side
// module doc for the mechanism accounting (L2-thrash pivot, Bench 890).
// ===========================================================================
__device__ __forceinline__ void att_pf_mq8v_body(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    float* __restrict__ attn_out,
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset)
{
    __shared__ float s_w[ATTN_MQ][256]; /* scores → tree → weights per qi */
    __shared__ float s_q[ATTN_MQ][256]; /* the staged q rows */

    const int tid = (int)threadIdx.x;
    const int tiles_per_head = (p + ATTN_MQ - 1) / ATTN_MQ;
    const int head_idx = blockIdx.x / tiles_per_head;
    const int q_base = (int)(blockIdx.x % tiles_per_head) * ATTN_MQ;
    const int q_count = (p - q_base) < ATTN_MQ ? (p - q_base) : ATTN_MQ;

    const int kv_group = head_idx * n_kv / n_head;
    const int kv_head_off = kv_group * head_dim;
    const int q_stride = n_head * head_dim;
    const int kv_stride = n_kv * head_dim;

    /* Stage the q rows (exact copies). */
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        if (qi < q_count) {
            s_q[qi][tid] =
                query[(long)(q_base + qi) * q_stride + head_idx * head_dim + tid];
        }
    }
    __syncthreads();

    float rmax[ATTN_MQ], rsum[ATTN_MQ], racc[ATTN_MQ];
#pragma unroll
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        rmax[qi] = -1e30f;
        rsum[qi] = 0.0f;
        racc[qi] = 0.0f;
    }

    const int max_n_pos = q_offset + q_base + q_count;
    const int n_tiles = (max_n_pos + head_dim - 1) / head_dim;

    for (int tile = 0; tile < n_tiles; tile++) {
        const int tile_base = tile * head_dim;
        const int pos = tile_base + tid;

        /* L2 prefetch of tile t+1's K+V — VERBATIM from att_pf_mq8p. */
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

        /* Phase 1: this thread's position's score for every qi — the K
         * row loads as 64 float4s (16 B per request; 8 in flight via the
         * unroll-8 quad loop). d order preserved: (d4, c) enumerates
         * d = d4*4 + c ascending, so each per-qi fma chain sees the same
         * operands in the same order as the scalar kernel. */
        float score[ATTN_MQ];
        if (pos < max_n_pos) {
            const float4* k4 = reinterpret_cast<const float4*>(
                key + (long)pos * kv_stride + kv_head_off);
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) score[qi] = 0.0f;
#pragma unroll 8
            for (int d4 = 0; d4 < head_dim / 4; d4++) {
                const float4 kv = k4[d4];
                const float ka[4] = { kv.x, kv.y, kv.z, kv.w };
#pragma unroll
                for (int c = 0; c < 4; c++) {
#pragma unroll
                    for (int qi = 0; qi < ATTN_MQ; qi++) {
                        score[qi] = __fmaf_rn(s_q[qi][d4 * 4 + c], ka[c],
                                              score[qi]);
                    }
                }
            }
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) score[qi] = score[qi] * scale;
        }
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const bool vp = qi < q_count && pos < (q_offset + q_base + qi + 1);
            s_w[qi][tid] = (pos < max_n_pos && vp) ? score[qi] : -1e30f;
        }
        __syncthreads();

        /* Phase 2: the max tree per qi (identical stride structure). */
        if (tid < 128) {
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) {
                if (s_w[qi][tid + 128] > s_w[qi][tid]) {
                    s_w[qi][tid] = s_w[qi][tid + 128];
                }
            }
        }
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    if (s_w[qi][tid + stride] > s_w[qi][tid]) {
                        s_w[qi][tid] = s_w[qi][tid + stride];
                    }
                }
            }
            __syncthreads();
        }
        float tmax[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) tmax[qi] = s_w[qi][0];
        __syncthreads(); /* Issue 715 race (a) */

        /* Online softmax update (deferred-fma merge — the settled form). */
        float eprev[ATTN_MQ], etile[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const float new_max = tmax[qi] > rmax[qi] ? tmax[qi] : rmax[qi];
            eprev[qi] = __expf(rmax[qi] - new_max);
            etile[qi] = __expf(tmax[qi] - new_max);
            rmax[qi] = new_max;
        }

        /* Phase 3: weights (Mul form; 0 for invalid). */
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const bool vp = qi < q_count && pos < (q_offset + q_base + qi + 1);
            s_w[qi][tid] =
                vp ? etile[qi] * __expf(score[qi] - tmax[qi]) : 0.0f;
        }
        __syncthreads();

        /* Phase 4: serial accumulation ascending pp; the V row (one load
         * per pp) serves all qi. `#pragma unroll 4` lets the per-thread V
         * loads pipeline 4 rows ahead of the fma chains (each pp iteration
         * is an independent address; the accumulator chains stay
         * pp-ascending). */
        float tsum[ATTN_MQ], tacc[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            tsum[qi] = 0.0f;
            tacc[qi] = 0.0f;
        }
#pragma unroll 4
        for (int pp = 0; pp < head_dim; pp++) {
            const int kv_pos = tile_base + pp;
            if (kv_pos < max_n_pos) {
                const float v =
                    value[(long)kv_pos * kv_stride + kv_head_off + tid];
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    if (qi < q_count && kv_pos < (q_offset + q_base + qi + 1)) {
                        const float w = s_w[qi][pp];
                        tsum[qi] = tsum[qi] + w;
                        tacc[qi] = __fmaf_rn(w, v, tacc[qi]);
                    }
                }
            }
        }

        /* Merge (RESF=1: both fold into the accumulation adds). */
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            rsum[qi] = __fmaf_rn(rsum[qi], eprev[qi], tsum[qi]);
            racc[qi] = __fmaf_rn(racc[qi], eprev[qi], tacc[qi]);
        }
        __syncthreads(); /* Issue 715 race (b) */
    }

    /* Final: 1/sum (div.full) + sigmoid gate. */
#pragma unroll
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        if (qi < q_count) {
            const float inv_sum = at_div_full(1.0f, rsum[qi]);
            const float raw = racc[qi] * inv_sum;
            const long off =
                (long)(q_base + qi) * q_stride + head_idx * head_dim + tid;
            const float g = gate[off];
            const float sig =
                at_div_full(1.0f, 1.0f + at_exp_fast(0.0f - g));
            attn_out[off] = raw * sig;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 4) att_pf_mq8v(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ gate,    /* [p, n_head, hd] */
    float* __restrict__ attn_out,      /* [p, n_head, hd] */
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset)
{
    att_pf_mq8v_body(query, key, value, gate, attn_out, head_dim, n_head,
                     n_kv, p, scale, q_offset);
}

// CUDA-graph twin of att_pf_mq8v (q_offset from the device buffer).
extern "C" __global__ void __launch_bounds__(256, 4) att_pf_mq8v_dp(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ gate,    /* [p, n_head, hd] */
    float* __restrict__ attn_out,      /* [p, n_head, hd] */
    int head_dim, int n_head, int n_kv, int p, float scale,
    const int* __restrict__ q_offset_dev)
{
    const int q_offset = *q_offset_dev;
    att_pf_mq8v_body(query, key, value, gate, attn_out, head_dim, n_head,
                     n_kv, p, scale, q_offset);
}
"#;
