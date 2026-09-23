// gemm_ternary_metal_wgpu.metal
// Issue 657 — wgpu MSL passthrough variant of the metal::tensor matmul2d kernel.
//
// Same compute logic as gemm_ternary_metal_tensor.metal, but adapted for wgpu
// passthrough dispatch:
//   - All 5 scalar params packed into a single uniform struct buffer (binding 5)
//   - Threadgroup memory explicitly sized via the dispatch (not kernel attribute)
//
// This kernel is compiled via wgpu::Device::create_shader_module_passthrough
// and dispatched on the same wgpu queue as CubeCL, sharing CubeCL-managed
// buffers directly (zero-copy interop).

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// ---------------------------------------------------------------------------
// Tile constants — match llama.cpp's kernel_mul_mm (ggml-metal-impl.h)
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
// Uniform struct — all scalar params in one buffer (binding 5)
// ---------------------------------------------------------------------------

struct GemmParams {
    uint blocks64;
    uint groups_per_row;
    uint k_input_dim;
    uint m_features;
    uint p_tokens;
};

// ---------------------------------------------------------------------------
// Main kernel — matmul2d ternary GEMM (wgpu passthrough variant)
// ---------------------------------------------------------------------------

kernel void gemm_ternary_tensor_wgpu(
    device const uint*  pos_bits_u32     [[buffer(0)]],
    device const uint*  neg_bits_u32     [[buffer(1)]],
    device const float* group_scale_f32  [[buffer(2)]],
    device const float* input_batch      [[buffer(3)]],   // X[P × K] row-major
    device float*       output_batch     [[buffer(4)]],   // Y[P × M] row-major (written)
    constant GemmParams& params          [[buffer(5)]],   // scalar params struct
    uint3               tgpig            [[threadgroup_position_in_grid]],
    ushort              tiitg            [[thread_index_in_threadgroup]]
) {
    const uint blocks64       = params.blocks64;
    const uint groups_per_row = params.groups_per_row;
    const uint k_input_dim    = params.k_input_dim;
    const uint m_features     = params.m_features;
    const uint p_tokens       = params.p_tokens;

    // Tile origin in the full output matrix
    const int ra = tgpig.y * NRA;   // feature row offset
    const int rb = tgpig.x * NRB;   // token col offset

    // ---- Threadgroup staging buffer for dequanted A tile ----
    // Fixed-size declaration (K_TILE × NRA × sizeof(half) = 32 × 64 × 2 = 4096 bytes).
    // wgpu passthrough mode doesn't support runtime-sized threadgroup memory
    // bindings ([[threadgroup(N)]]); we declare a fixed-size array instead.
    threadgroup half sa[K_TILE * NRA];

    // ---- Wrap A tile as a metal::tensor (threadgroup memory) ----
    auto tA = tensor(sa, dextents<int32_t, 2>(K_TILE, NRA));

    // ---- Wrap B (activations) as a metal::tensor (device memory) ----
    device float* input_rw = (device float*)(input_batch);
    auto tB = tensor(input_rw,
                     dextents<int32_t, 2>(k_input_dim, p_tokens),
                     array<int, 2>({1, (int)k_input_dim}));

    // ---- Configure the matmul2d operation ----
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, K_TILE, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm;

    // ---- Destination cooperative tensor ----
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    // ---- Ternary format constants ----
    const uint words_per_row = blocks64 * 2u;

    // ---- K-dimension accumulation loop ----
    for (int loop_k = 0; loop_k < (int)k_input_dim; loop_k += K_TILE) {

        // === PHASE 1: Dequant ternary weights into threadgroup `half` buffer ===
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row_local = work / NK_CHUNK;
            const int k_chunk   = work % NK_CHUNK;
            const int k_base    = k_chunk * SZ_SIMD;
            const int k_pos     = loop_k + k_base;

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
    auto tD = tensor(output_batch,
                     dextents<int32_t, 2>(m_features, p_tokens),
                     array<int, 2>({1, (int)m_features}));
    cT.store(tD.slice(ra, rb));
}
