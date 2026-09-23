// gemm_ternary_metal_tensor.metal
// Issue 656 — raw MSL ternary GEMM using metal::tensor / mpp::tensor_ops::matmul2d.
//
// Phase 2: hardware-accelerated matmul2d kernel.
// Replaces the Phase 1 scalar reference with the same cooperative-tensor
// matmul API that llama.cpp's kernel_mul_mm uses (64×128 output tiles,
// 4 simdgroups, 32-element K-tile).
//
// Layout (matching our bit-plane format + the llama.cpp dispatch pattern):
//   W: weights [M_features × K_input_dim]  (ternary bit-plane: pos_u32, neg_u32, scale_f32)
//   X: activations [P_tokens × K_input_dim]  (f32, row-major)
//   Y: output [P_tokens × M_features] = dequant(W) @ X^T
//
// GEMM mapping (llama.cpp convention):
//   A = dequanted weights, staged in threadgroup `half` memory
//   B = activations, read directly from device memory via tensor strides
//   C = output
//   descriptor(NRB=P_tile, NRA=M_tile, K_tile, transpose_left=false,
//              transpose_right=true, relaxed_precision=true, MAC)
//   mm.run(mB, mA, cT)  — note the swapped (B, A) arg order, matching llama.cpp

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// ---------------------------------------------------------------------------
// Tile constants — match llama.cpp's kernel_mul_mm (ggml-metal-impl.h)
// ---------------------------------------------------------------------------
// SZ_SIMDGROUP = 16 (Metal simdgroup width for matmul2d)
// K-tile = 32 (N_MM_NK_TOTAL = SZ_SIMDGROUP * N_MM_NK = 16 * 2)
// NRA = 64 (output rows / features per workgroup)
// NRB = 128 (output cols / tokens per workgroup)
// 4 simdgroups = 128 threads per workgroup
// ---------------------------------------------------------------------------
constant constexpr int SZ_SIMD     = 16;
constant constexpr int NK_CHUNK    = 2;                          // N_MM_NK
constant constexpr int K_TILE      = SZ_SIMD * NK_CHUNK;         // 32
constant constexpr int NRA         = 64;                          // M_tile (features)
constant constexpr int NRB         = 128;                         // N_tile (tokens)
constant constexpr int NSG         = 4;                           // simdgroups per workgroup
constant constexpr int NUM_THREADS = 32 * NSG;                    // 128
constant constexpr int A_WORK_ITEMS = NRA * NK_CHUNK;             // 128 (16 elements each)

// ---------------------------------------------------------------------------
// Main kernel — matmul2d ternary GEMM
//
// Computes Y[P × M] = dequant_ternary(W[M × K]) @ X[P × K]^T
// using mpp::tensor_ops::matmul2d with 64×128 output tiles.
//
// Dispatch: grid = ((P + 127)/128, (M + 63)/64, 1), 128 threads/workgroup.
// ---------------------------------------------------------------------------

kernel void gemm_ternary_tensor(
    device const uint*  pos_bits_u32     [[buffer(0)]],
    device const uint*  neg_bits_u32     [[buffer(1)]],
    device const float* group_scale_f32  [[buffer(2)]],
    device const float* input_batch      [[buffer(3)]],   // X[P × K] row-major
    device float*       output_batch     [[buffer(4)]],   // Y[P × M] row-major (written)
    constant uint&      blocks64         [[buffer(5)]],   // 64-element blocks per weight row
    constant uint&      groups_per_row   [[buffer(6)]],   // 128-element scale groups per row
    constant uint&      k_input_dim      [[buffer(7)]],   // K (input dimension)
    constant uint&      m_features       [[buffer(8)]],   // M (output features)
    constant uint&      p_tokens         [[buffer(9)]],   // P (number of tokens)
    threadgroup char*   shmem            [[threadgroup(0)]],
    uint3               tgpig            [[threadgroup_position_in_grid]],
    ushort              tiitg            [[thread_index_in_threadgroup]]
) {
    // Tile origin in the full output matrix
    const int ra = tgpig.y * NRA;   // feature row offset
    const int rb = tgpig.x * NRB;   // token col offset

    // ---- Threadgroup staging buffer for dequanted A tile ----
    // Layout: sa[K_TILE][NRA] = sa[32][64], contiguous in NRA (last dim).
    // Element (k, row) at sa[k * NRA + row].
    threadgroup half* sa = (threadgroup half*)(shmem);

    // ---- Wrap A tile as a metal::tensor (threadgroup memory) ----
    auto tA = tensor(sa, dextents<int32_t, 2>(K_TILE, NRA));

    // ---- Wrap B (activations) as a metal::tensor (device memory) ----
    // X is stored [P × K] row-major. We want B[k][p] = X[p][k], achieved via
    // strides {1, K}: element (k, p) at ptrX[k + p*K] = X[p][k].
    // Note: matmul2d's static_assert rejects `const float` element type, so we
    // cast to a non-const device pointer (the resource is Read-only on the
    // host side; this is purely a type-system workaround).
    device float* input_rw = (device float*)(input_batch);
    auto tB = tensor(input_rw,
                     dextents<int32_t, 2>(k_input_dim, p_tokens),
                     array<int, 2>({1, (int)k_input_dim}));

    // ---- Configure the matmul2d operation ----
    // descriptor(NRB, NRA, K_TILE, transpose_left=false, transpose_right=true,
    //            relaxed_precision=false, mode=multiply_accumulate)
    // relaxed_precision=false for max accuracy (we need bit-level G1 correctness).
    // Matching llama.cpp's dispatch geometry with stricter precision.
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, K_TILE, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm;

    // ---- Destination cooperative tensor (accumulates across K-loop) ----
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    // Note: llama.cpp's kernel_mul_mm does NOT manually zero-initialize cT.
    // The cooperative_tensor destination is initialized by the first mm.run()
    // call (multiply mode on the first K-tile, accumulate on subsequent tiles).
    // This matches the proven llama.cpp dispatch pattern.

    // ---- Ternary format constants ----
    const uint words_per_row = blocks64 * 2u;  // u32 words per weight row

    // ---- K-dimension accumulation loop ----
    for (int loop_k = 0; loop_k < (int)k_input_dim; loop_k += K_TILE) {

        // === PHASE 1: Dequant ternary weights into threadgroup `half` buffer ===
        // Each thread handles one (row, k_chunk) unit = 16 K-positions.
        // A_WORK_ITEMS = NRA * NK_CHUNK = 64 * 2 = 128 = NUM_THREADS.
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row_local = work / NK_CHUNK;       // [0, NRA)
            const int k_chunk   = work % NK_CHUNK;       // [0, NK_CHUNK)
            const int k_base    = k_chunk * SZ_SIMD;     // [0, 32) in steps of 16
            const int k_pos     = loop_k + k_base;       // absolute K position

            const int row_global = ra + row_local;

            #pragma clang loop unroll(full)
            for (short i = 0; i < SZ_SIMD; ++i) {
                const int col = k_pos + i;
                half val = 0.0h;
                if (row_global < (int)m_features && col < (int)k_input_dim) {
                    const uint word_idx = (uint)row_global * words_per_row + (uint)(col / 32);
                    const uint bit_pos  = (uint)(col % 32);
                    const uint pos_bit  = (pos_bits_u32[word_idx] >> bit_pos) & 1u;
                    const uint neg_bit  = (neg_bits_u32[word_idx] >> bit_pos) & 1u;
                    const float sign    = (float)pos_bit - (float)neg_bit;
                    const float scale   = group_scale_f32[(uint)row_global * groups_per_row + (uint)(col / 128)];
                    val = (half)(sign * scale);
                }
                // metal::tensor default layout is COLUMN-MAJOR (strides {1, extent[0]}).
                // tA = tensor(sa, dextents(K_TILE, NRA)) → element (k, row) at sa[k + row*K_TILE].
                // This matches llama.cpp's sa[row * N_MM_NK_TOTAL + k] indexing.
                sa[(size_t)(k_base + i) + (size_t)row_local * K_TILE] = val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: Cooperative tensor matmul ===
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Store result tile to output ===
    // Y is [P × M] row-major: element (p, m) at output_batch[p * M + m].
    // tD wraps output as [M, P] with strides {1, M}: element (m, p) at dst[m + p*M].
    auto tD = tensor(output_batch,
                     dextents<int32_t, 2>(m_features, p_tokens),
                     array<int, 2>({1, (int)m_features}));
    cT.store(tD.slice(ra, rb));
}
