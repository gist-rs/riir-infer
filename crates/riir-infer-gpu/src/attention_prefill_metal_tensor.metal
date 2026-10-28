// attention_prefill_metal_tensor.metal
// Issue 1146 ladder 4(c) — causal, gated, GQA flash-attention prefill on the
// Metal 4 tensor ops (`mpp::tensor_ops::matmul2d`), modelled on upstream
// llama.cpp's `kernel_flash_attn_ext_tensor` (fa_tensor.metal, dk256 shape:
// Q=32 queries per threadgroup, C=64 cache rows per block, 8 simdgroups).
//
// Semantics are those of `QwenAttentionPrefillTiledM32CubeCL` exactly:
//   q    [p][n_head][256]        f32 (this chunk's queries)
//   k, v [rows][n_kv][256]       f32 (the cache, rows = base_pos + p)
//   gate [p][n_head][256]        f32
//   out  [p][n_head][256]        f32 = softmax(q·kᵀ·scale, causal) · v · σ(gate)
//   causal: query at chunk row i (absolute base_pos + i) sees cache rows
//   0..=base_pos + i; GQA group = head * n_kv / n_head.
//
// Precision (a TOLERANCE arm, never bit-identical to the f32 CubeCL arms):
// Q is staged as half with the softmax scale folded in, P is half (the lazy
// rescale keeps exp(s - M) <= e^8, which half holds), K/V are read as f32
// straight from the cache, scores and O accumulate in f32.
//
// Partial blocks: the host only routes this kernel when rows >= C, so every
// block is read IN BOUNDS — a block that would run past `kv_end` is shifted
// back to end at `rows`, and the positions it re-reads (< ic) are masked as
// already consumed. No dynamic tensor extents are needed.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

#define FOR_UNROLL _Pragma("clang loop unroll(full)") for

// Lazy-rescale threshold (upstream: 8 — e^8 fits in half). Overridable for
// diagnosis via the host's RIIR_ATTN_TENSOR_DEFINES.
#ifndef ATTN_RESCALE_THRESH
#define ATTN_RESCALE_THRESH 8.0f
#endif

constant constexpr short DK = 256;  // head dim (the Bonsai/Qwen3.x shape)
constant constexpr short DV = 256;
constant constexpr short Q = 32;    // queries per threadgroup
constant constexpr short C = 64;    // cache rows per block
constant constexpr short NSG = 8;   // simdgroups per threadgroup
constant constexpr short NW = 32;   // simdgroup width
constant constexpr short NT = NW * NSG;
constant constexpr short NQ = Q / NSG;  // query rows per simdgroup
constant constexpr short NC = C / NW;   // score columns per lane

static_assert(Q % NSG == 0, "Q must be divisible by NSG");
static_assert(C % NW == 0, "C must be divisible by NW");

// Threadgroup memory layout (bytes): sq half[Q*DK] | ss float[Q*C] |
// sp half[Q*C] | sr float[Q] | sf int[1]. The host sizes it from the same
// constants (`THREADGROUP_BYTES` in attention_prefill_metal_tensor.rs).

kernel void attn_prefill_tensor_f32kv(
        device const float * query  [[buffer(0)]],
        device       float * key    [[buffer(1)]],
        device       float * value  [[buffer(2)]],
        device const float * gate   [[buffer(3)]],
        device       float * out    [[buffer(4)]],
        constant uint & n_head      [[buffer(5)]],
        constant uint & n_kv        [[buffer(6)]],
        constant uint & p           [[buffer(7)]],
        constant uint & base_pos    [[buffer(8)]],
        constant float & scale      [[buffer(9)]],
        threadgroup char * shmem    [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int q0 = tgpig[0] * Q;  // first chunk-row of this tile
    const int head = tgpig[1];
    const short tiitg = sgitg * NW + tiisg;

    const int q_stride = int(n_head) * DK;
    const int kv_stride = int(n_kv) * DK;
    const int kv_group = head * int(n_kv) / int(n_head);
    const int rows = int(base_pos) + int(p);

    threadgroup half  * sq = (threadgroup half  *) shmem;      // [Q][DK]
    threadgroup float * ss = (threadgroup float *) (sq + Q*DK); // [Q][C]
    threadgroup half  * sp = (threadgroup half  *) (ss + Q*C);  // [Q][C]
    threadgroup float * sr = (threadgroup float *) (sp + Q*C);  // [Q]
    threadgroup int   * sf = (threadgroup int   *) (sr + Q);    // [1]

    // Stage Q (scale folded in); rows past p are zero.
    device const float * qh = query + head * DK;
    for (int i = tiitg; i < Q*DK/4; i += NT) {
        const int j = i / (DK/4);
        float4 q4 = 0.0f;
        if (q0 + j < int(p)) {
            q4 = ((device const float4 *) (qh + (q0 + j) * q_stride))[i % (DK/4)];
        }
        ((threadgroup half4 *) sq)[i] = (half4) (q4 * scale);
    }

    float M[NQ];
    float S[NQ];
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        M[jj] = -FLT_MAX/2;
        S[jj] = 0.0f;
    }

    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C, Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C, Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    device float * kh = key + kv_group * DK;
    device float * vh = value + kv_group * DK;

    auto tv0 = tensor(vh, dextents<int32_t, 2>(DV, C), array<int, 2>({1, kv_stride}));
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    if (tiitg == 0) {
        sf[0] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Causal upper bound for the whole tile: the last active row's position.
    const int q_last = min(q0 + Q, int(p)) - 1;
    const int kv_end = int(base_pos) + q_last + 1;

    for (int ic0 = 0, ic = 0; ic < kv_end; ++ic0, ic += C) {
        // In-bounds block start: shift a block that would run past `rows`
        // back so it ends at `rows` (the host guarantees rows >= C).
        const int s0 = min(ic, rows - C);

        // S = Q · Kᵀ for cache rows [s0, s0 + C).
        {
            auto tk = tensor(kh + (uint64_t) s0 * kv_stride,
                             dextents<int32_t, 2>(DK, C), array<int, 2>({1, kv_stride}));
            mm_qk.run(tq, tk, ts);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax with the causal + already-consumed mask.
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const int q_abs = int(base_pos) + q0 + j;

            float s[NC];
            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                const int pos = s0 + ii * NW + tiisg;
                const bool ok = pos >= ic && pos <= q_abs;
                s[ii] = ok ? ss[j*C + ii*NW + tiisg] : -1e30f;
            }

            float m = M[jj];
            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                m = max(m, s[ii]);
            }
            m = simd_max(m);

            // Lazy rescale: move the running max only when it grows by more
            // than 8 (e^8 fits in half).
            float ms = 1.0f;
            if (m > M[jj] + ATTN_RESCALE_THRESH) {
                ms = exp(M[jj] - m);
                M[jj] = m;
                if (tiisg == 0) {
                    sf[0] = ic0 + 1;
                }
            }

            float sum = 0.0f;
            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                // The sum uses the same rounded values as P·V.
                const half pv = (half) exp(s[ii] - M[jj]);
                sp[j*C + ii*NW + tiisg] = pv;
                sum += (float) pv;
            }
            S[jj] = S[jj] * ms + simd_sum(sum);
            if (tiisg == 0) {
                sr[j] = ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(ms) · O + P · V.
        if (sf[0] == ic0 + 1) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }
        {
            auto tv = tensor(vh + (uint64_t) s0 * kv_stride,
                             dextents<int32_t, 2>(DV, C), array<int, 2>({1, kv_stride}));
            mm_pv.run(tp, tv, co);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalise; every active row saw cache row 0, so S > 0 there.
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        if (tiisg == 0) {
            sr[j] = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O · (1/S) · σ(gate), in the destination's own layout.
    device const float * gh = gate + head * DK;
    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            const auto idx = co.get_multidimensional_index(i);  // [dv, row]
            const int row = idx[1];
            float g = 0.0f;
            if (q0 + row < int(p)) {
                g = gh[(q0 + row) * q_stride + idx[0]];
            }
            co[i] *= sr[row] / (1.0f + exp(-g));
        }
    }

    // Store; rows past p are clipped by the tensor extents.
    device float * pdst = out + q0 * q_stride + head * DK;
    auto td = tensor(pdst, dextents<int32_t, 2>(DV, int(p) - q0), array<int, 2>({1, q_stride}));
    co.store(td);
}
