// gemm_q4k_metal_tensor.metal
// Plan 614 Phase-3 task 1 — the Q4_K-dequant matmul2d GEMM (the dense lane's
// serving-byte tensor kernel: ~4.5 bpw weights consumed in place, no f16
// resident copy).
//
// The ternary tensor kernel's staged-dequant structure (the proven skeleton:
// threadgroup `half` A tile → barrier → matmul2d.run per K-tile) with the
// Q4_K dequant grafted into the staging phase. Each 16-k staging work item
// covers HALF a Q4_K sub-block (sub-blocks are 32 elements; k_pos is
// 16-aligned so a 16-wide window never crosses a sub-block boundary or its
// 256-element block), so the per-item dequant is: one scale/min pair decode
// + one d/dmin pair + 16 qs bytes, one nibble each.
//
// Byte-exactness contract (the G1 band): the staged half values are
// `d * sc * q - dmin * mn` computed in f32 with the SAME expression order as
// `dequantize_row_q4_k` (src/quant/q4k.rs), rounded to `half` once — the
// reference CPU GEMM consumes f16-rounded dequanted weights, so the band
// measures summation order only. `d`/`dmin` arrive as f32 (the Q4KHandle
// pre-converts the f16 block scales at build — no f16 decode in-kernel).
//
// get_scale_min_k4 ported verbatim from src/quant/q4k.rs (u8 bit math —
// uchar loads, no signedness hazards).
//
// Geometry: the fork's proven tile (64×128, 4 simdgroups, K_TILE 32 =
// SZ_SIMD 16 × NK_CHUNK 2) — the same as gemm_ternary_metal_tensor.metal and
// gemm_dense_f16_metal_tensor.metal; the host constants mirror the SWEEP_*
// defaults by construction.
//
// Staging traffic note (the Bench-1163 intensity thesis): Q4_K carries
// ~3.4× the f16 path's arithmetic per loaded byte (0.56 B/weight vs 2 B +
// the dequant ALU work), which is the load-latency-hiding mechanism the
// measured 148→451 GB/s intensity ordering rests on.
#ifndef SWEEP_SZ_SIMD
#define SWEEP_SZ_SIMD 16
#endif
#ifndef SWEEP_NK_CHUNK
#define SWEEP_NK_CHUNK 2
#endif
#ifndef SWEEP_NRA
#define SWEEP_NRA 64
#endif
#ifndef SWEEP_NRB
#define SWEEP_NRB 128
#endif
#ifndef SWEEP_NSG
#define SWEEP_NSG 4
#endif
#ifndef SWEEP_RELAXED
#define SWEEP_RELAXED false
#endif

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

constant constexpr int SZ_SIMD     = SWEEP_SZ_SIMD;   // 16
constant constexpr int NK_CHUNK    = SWEEP_NK_CHUNK;  // 2
constant constexpr int K_TILE      = SZ_SIMD * NK_CHUNK;  // 32
constant constexpr int NRA         = SWEEP_NRA;       // M_tile (features)
constant constexpr int NRB         = SWEEP_NRB;       // N_tile (tokens)
constant constexpr int NSG         = SWEEP_NSG;       // simdgroups per workgroup
constant constexpr int NUM_THREADS = 32 * NSG;        // 128
constant constexpr int A_WORK_ITEMS = NRA * NK_CHUNK; // 128 (16 elements each)
constant constexpr int QK_K        = 256;
constant constexpr int BLOCK_BYTES = 144;             // d(2)+dmin(2)+scales(12)+qs(128)

// get_scale_min_k4 — the verbatim port of src/quant/q4k.rs.
static inline void get_scale_min_k4_msl(
    uint j, device const uchar* s, thread uint& sc, thread uint& mn) {
    if (j < 4u) {
        sc = s[j] & 63u;
        mn = s[j + 4u] & 63u;
    } else {
        sc = (s[j + 4u] & 0x0Fu) | ((s[j - 4u] >> 6) << 4);
        mn = (s[j + 4u] >> 4) | ((s[j] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// Main kernel — matmul2d Q4_K-dequant GEMM
//
// Computes Y[P × M] = dequant_q4k(W[M × K]) @ X[P × K](f32)^T
//
// Buffers:
//   0: weight_q4k — the packed Q4_K payloads [M × bpr × 144 B], byte-addressed
//      (the Q4KHandle's 36-u32-per-block words, addressed as bytes; block b of
//      row r at (r * bpr + b) * 144; d/dmin LIVE SEPARATELY, see buffer 1).
//   1: d_dmin — [M × bpr × 2] f32 (the handle's pre-converted block scales:
//      element (r, b) at (r * bpr + b) * 2 + {0, 1}).
//   2: input_batch  X [P × K] row-major f32
//   3: output_batch Y [P × M] row-major f32 (DEDICATED buffer + blit — the
//      cooperative-store aliasing cure)
//   4: k_input_dim, 5: m_features, 6: p_tokens
//
// Dispatch: grid = ((P + NRB - 1)/NRB, (M + NRA - 1)/NRA, 1),
// 32*NSG threads/workgroup, threadgroup memory 0: K_TILE * NRA * 2 bytes
// (4 KB at the fork geometry).
// ---------------------------------------------------------------------------

kernel void gemm_q4k_tensor(
    device const uchar* weight_q4k    [[buffer(0)]],
    device const float* d_dmin        [[buffer(1)]],
    device const float* input_batch   [[buffer(2)]],
    device float*       output_batch  [[buffer(3)]],
    constant uint&      k_input_dim   [[buffer(4)]],
    constant uint&      m_features    [[buffer(5)]],
    constant uint&      p_tokens      [[buffer(6)]],
    threadgroup char*   shmem         [[threadgroup(0)]],
    uint3               tgpig         [[threadgroup_position_in_grid]],
    ushort              tiitg         [[thread_index_in_threadgroup]]
) {
    const int ra = tgpig.y * NRA;   // feature row offset
    const int rb = tgpig.x * NRB;   // token col offset

    // ---- Threadgroup staging buffer for dequanted A tile ----
    // Column-major (the tensor view's default): element (k, row) at
    // sa[k + row * K_TILE] — half4 writes legal (offsets are 32 B aligned).
    threadgroup half* sa = (threadgroup half*)(shmem);
    auto tA = tensor(sa, dextents<int32_t, 2>(K_TILE, NRA));

    // ---- B (activations): device (K × P) view, strides {1, K} ----
    device float* input_rw = (device float*)(input_batch);
    auto tB = tensor(input_rw,
                     dextents<int32_t, 2>(k_input_dim, p_tokens),
                     array<int, 2>({1, (int)k_input_dim}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, K_TILE, false, true, SWEEP_RELAXED,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    const uint bpr = k_input_dim / QK_K;  // blocks per weight row
    const uint words_per_block = 36u;     // 144 B / 4

    // ---- K-dimension accumulation loop ----
    for (int loop_k = 0; loop_k < (int)k_input_dim; loop_k += K_TILE) {

        // === PHASE 1: dequant Q4_K into the threadgroup half tile ===
        // Each work item = one (row, 16-k) unit = exactly ONE Q4_K sub-block:
        // k_pos is a multiple of 16, so e = k_pos % 256 is a multiple of 16 —
        // the 16-element window sits inside sub-block j = e / 16 of block b.
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row_local = work / NK_CHUNK;      // [0, NRA)
            const int k_chunk   = work % NK_CHUNK;      // [0, NK_CHUNK)
            const int k_base    = k_chunk * SZ_SIMD;    // [0, 32) step 16
            const int k_pos     = loop_k + k_base;      // absolute K position
            const int row_global = ra + row_local;

            half4 v0, v1, v2, v3;
            bool ok = row_global < (int)m_features && k_pos + SZ_SIMD <= (int)k_input_dim;
            if (ok) {
                const uint rg = (uint)row_global;
                const uint e     = (uint)k_pos % QK_K;   // element offset in block
                const uint b     = (uint)k_pos / QK_K;   // block index in the row
                const uint blk   = rg * bpr + b;         // global block index
                const uint blk_off = blk * BLOCK_BYTES;
                device const uchar* sc_p = weight_q4k + blk_off + 4u;   // scales[12]
                device const uchar* qs_p = weight_q4k + blk_off + 16u;  // qs[128]

                uint j = e >> 5;                          // sub-block [0, 8): 32 elements each
                uint sc, mn;
                get_scale_min_k4_msl(j, sc_p, sc, mn);
                const uint pair = j >> 1;                 // which 64-element pair
                // Nibble stream (the dequantize_row_q4_k layout): pair p's
                // qs bytes [32p, 32p+32) carry sub-block 2p in their LOW
                // nibbles (element 64p + t = byte 32p + t low) and sub-block
                // 2p+1 in their HIGH nibbles (element 64p + 32 + t = byte
                // 32p + t high). One byte feeds ONE element per stream, so
                // our 16-element window (16-aligned, never straddles the
                // 32-element stream split) consumes 16 consecutive bytes,
                // one nibble each.
                const uint pos   = e & 63u;
                const bool low   = (e & 32u) == 0u;       // sub-block j even
                const uint byte0 = 32u * pair + (low ? pos : (pos - 32u));

                const float d    = d_dmin[blk * 2u];
                const float dmin = d_dmin[blk * 2u + 1u];
                const float d_sc = d * (float)sc;
                const float m_t  = dmin * (float)mn;

                #pragma clang loop unroll(full)
                for (uint t = 0; t < 16u; ++t) {
                    const uint byte_v = qs_p[byte0 + t];
                    const uint nib = low ? (byte_v & 0x0Fu) : (byte_v >> 4);
                    const float w = d_sc * (float)nib - m_t;
                    const half h = (half)w;
                    if (t < 4u) {
                        v0[t] = h;
                    } else if (t < 8u) {
                        v1[t - 4u] = h;
                    } else if (t < 12u) {
                        v2[t - 8u] = h;
                    } else {
                        v3[t - 12u] = h;
                    }
                }
            } else {
                v0 = half4(0.0h);
                v1 = half4(0.0h);
                v2 = half4(0.0h);
                v3 = half4(0.0h);
            }

            threadgroup half4* dst = (threadgroup half4*)(
                sa + (size_t)k_base + (size_t)row_local * K_TILE);
            dst[0] = v0;
            dst[1] = v1;
            dst[2] = v2;
            dst[3] = v3;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: cooperative tensor matmul ===
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Store result tile ===
    auto tD = tensor(output_batch,
                     dextents<int32_t, 2>(m_features, p_tokens),
                     array<int, 2>({1, (int)m_features}));
    cT.store(tD.slice(ra, rb));
}
