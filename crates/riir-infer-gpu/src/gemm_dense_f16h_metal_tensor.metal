// gemm_dense_f16h_metal_tensor.metal
// Issue 1170 probe arm — the half×half matmul2d variant of the dense GEMM.
//
// IDENTICAL to gemm_dense_f16_metal_tensor.metal except the B operand
// (activations) is `device half` instead of `device float` — the fork's
// kernel_mul_mm sees half×half (it stages/converts B into threadgroup
// memory); ours reads the activation as a device f16 VIEW (the caller
// stages the conversion once per chunk). The question this kernel answers:
// does the half×half MMA schedule beat the mixed half×float device path's
// 41–53 TFLOPS (bench 940) materially? PROBE-ONLY until the A/B says yes —
// no route wiring, no numerics surface.
//
// Numerics class (if promoted): activations rounded f16 — lossy class, the
// bench-923/940 discipline applies (pre-registered band + walk agreement).

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

constant constexpr int SZ_SIMD  = SWEEP_SZ_SIMD;
constant constexpr int NK_CHUNK = SWEEP_NK_CHUNK;  // N_MM_NK
constant constexpr int K_TILE   = SZ_SIMD * NK_CHUNK;  // 32
constant constexpr int NRA      = SWEEP_NRA;      // M_tile (features)
constant constexpr int NRB      = SWEEP_NRB;      // N_tile (tokens)
constant constexpr int NSG      = SWEEP_NSG;      // simdgroups per workgroup

kernel void gemm_dense_f16_tensor(
    device const half* weights_maj   [[buffer(0)]],   // W [M × K] row-major f16
    device const half* input_batch   [[buffer(1)]],   // X [P × K] row-major f16 (staged view)
    device float*       output_batch  [[buffer(2)]],   // Y [P × M] row-major f32
    constant uint&      k_input_dim   [[buffer(3)]],   // K (input dimension)
    constant uint&      m_features    [[buffer(4)]],   // M (output features)
    constant uint&      p_tokens      [[buffer(5)]],   // P (number of tokens)
    uint3               tgpig         [[threadgroup_position_in_grid]]
) {
    const int ra = tgpig.y * NRA;   // feature row offset
    const int rb = tgpig.x * NRB;   // token col offset

    // A (weights): (K × M) view, strides {1, K} — element (k, m) at w[m*K + k].
    device half* w_rw = (device half*)(weights_maj);
    auto tA = tensor(w_rw,
                     dextents<int32_t, 2>(k_input_dim, m_features),
                     array<int, 2>({1, (int)k_input_dim}));

    // B (activations, f16): (K × P) view, strides {1, K} — element (k, p) at x[p*K + k].
    device half* input_rw = (device half*)(input_batch);
    auto tB = tensor(input_rw,
                     dextents<int32_t, 2>(k_input_dim, p_tokens),
                     array<int, 2>({1, (int)k_input_dim}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, K_TILE, false, true, SWEEP_RELAXED,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm;

    // Destination cooperative tensor (accumulates across the K loop; the
    // first mm.run() initializes in multiply mode — the llama.cpp dispatch
    // pattern, no manual zero-init, identical to the dense kernel).
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (int loop_k = 0; loop_k < (int)k_input_dim; loop_k += K_TILE) {
        auto mA = tA.slice(loop_k, ra);
        auto mB = tB.slice(loop_k, rb);
        mm.run(mB, mA, cT);
    }

    // Store: Y [P × M] row-major via an (M × P) view, strides {1, M} —
    // element (m, p) at y[p*M + m]; the tile slice lands at (ra, rb).
    auto tD = tensor(output_batch,
                     dextents<int32_t, 2>(m_features, p_tokens),
                     array<int, 2>({1, (int)m_features}));
    cT.store(tD.slice(ra, rb));
}
