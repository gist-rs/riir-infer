// gemm_dense_f16_metal_tensor.metal
// Issue 1150 follow-up — the dense-f16 matmul2d GEMM (the Metal prefill speed
// unlock for the dense-qwen3.8 lane).
//
// The dense twin of gemm_ternary_metal_tensor.metal: the SAME
// mpp::tensor_ops::matmul2d cooperative-tensor path (64×128 output tiles, 4
// simdgroups, 32-element K-tile — the llama.cpp kernel_mul_mm geometry), with
// the ternary bit-plane dequant phase DELETED — the A operand (weights) is
// read directly from device f16 memory as a strided tensor view, so the
// kernel has no threadgroup staging, no dequant arithmetic, and no staging
// barrier in the K loop.
//
// Layout (matching the ternary kernel's proven B/tD conventions):
//   W: weights [M_features × K_input_dim]  (f16, row-major, device memory)
//   X: activations [P_tokens × K_input_dim]  (f32, row-major, device memory)
//   Y: output [P_tokens × M_features] = W @ X^T  (f32, row-major, written)
//
// GEMM mapping (the ternary kernel's):
//   A = W, wrapped as a (K × M) tensor view with strides {1, K} — element
//       (k, m) at w[m*K + k]; sliced per tile at (loop_k, ra).
//   B = X, wrapped as a (K × P) tensor view with strides {1, K} — element
//       (k, p) at x[p*K + k]; sliced per tile at (loop_k, rb).
//   C = Y, wrapped as an (M × P) tensor view with strides {1, M} — element
//       (m, p) at y[p*M + m]; sliced per tile at (ra, rb).
//   descriptor(NRB, NRA, K_TILE, transpose_left=false, transpose_right=true,
//              relaxed_precision=SWEEP_RELAXED, multiply_accumulate)
//   mm.run(mB, mA, cT) — the swapped (B, A) arg order, matching llama.cpp.
//
// The mixed half(A) × float(B) operand pair is the proven in-crate shape:
// the ternary kernel stages its dequanted A tile as `half` threadgroup
// memory and reads its B tile straight from device `float`, with a float
// cooperative destination. Here A is `device half` instead of staged
// `threadgroup half` — same element type, device memory space, which the
// B/tD device views already prove matmul2d accepts.
//
// Edge tiles: cooperative-tensor load/store bounds-check against the tensor
// VIEW extents, and every view here spans the FULL matrix (K × M, K × P,
// M × P) — a slice past the edge clamps inside the cooperative unit exactly
// as the ternary kernel's tB/tD slices do (measured there: edge shapes in
// the G1 gate).
//
// relaxed_precision: pinned FALSE by default (bit-tightest accumulate, the
// incumbent G1 posture — matches the ternary kernel's default).
// `SWEEP_RELAXED true` (the Issue-1146 rung-2b selector's define) is accepted
// for A/B cells but is NOT wired by the dense host yet.
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

// ---------------------------------------------------------------------------
// Decode GEMV — plain (non-cooperative) kernel, no tile constraints.
//
// The matmul2d path cannot serve decode: a (K × 1) activation view leaves
// the cooperative B tile's token dimension at extent 1 and the load/stores
// silently produce nothing (measured, the G1 gate's p=1 cell). Decode is
// bandwidth-bound — one simdgroup per output row, lanes striding K with
// coalesced f16 reads and one simd_sum reduction, matches the incumbent
// GemvCubeCL shape with the f16 weight reader.
//
// Dispatch: grid = (ceil(M/4), 1, 1), 128 threads/workgroup = 4 simdgroups
// = 4 rows per workgroup, 32 lanes per row.
// ---------------------------------------------------------------------------

kernel void gemv_dense_f16(
    device const half*  weights  [[buffer(0)]],   // W [M × K] row-major f16
    device const float* input    [[buffer(1)]],   // x [K] f32
    device float*       output   [[buffer(2)]],   // y [M] f32
    constant uint&      k_dim    [[buffer(3)]],
    constant uint&      m_dim    [[buffer(4)]],
    uint3               tgpig    [[threadgroup_position_in_grid]],
    ushort              tiitg    [[thread_index_in_threadgroup]],
    ushort              sit      [[thread_index_in_simdgroup]],
    ushort              sptg     [[simdgroups_per_threadgroup]])
{
    const uint row = tgpig.x * sptg + (tiitg / 32);
    if (row >= m_dim) { return; }
    float acc = 0.0f;
    for (uint k = sit; k < k_dim; k += 32u) {
        acc += float(weights[row * k_dim + k]) * input[k];
    }
    output[row] = simd_sum(acc);
}

// ---------------------------------------------------------------------------
// Main kernel — matmul2d dense-f16 GEMM
//
// Computes Y[P × M] = W[M × K](f16) @ X[P × K](f32)^T
//
// Dispatch: grid = ((P + NRB - 1)/NRB, (M + NRA - 1)/NRA, 1), 32*NSG
// threads/workgroup. No threadgroup memory.
// ---------------------------------------------------------------------------

kernel void gemm_dense_f16_tensor(
    device const half*  weights_maj   [[buffer(0)]],   // W [M × K] row-major f16
    device const float* input_batch   [[buffer(1)]],   // X [P × K] row-major f32
    device float*       output_batch  [[buffer(2)]],   // Y [P × M] row-major f32
    constant uint&      k_input_dim   [[buffer(3)]],   // K (input dimension)
    constant uint&      m_features    [[buffer(4)]],   // M (output features)
    constant uint&      p_tokens      [[buffer(5)]],   // P (number of tokens)
    uint3               tgpig         [[threadgroup_position_in_grid]]
) {
    const int ra = tgpig.y * NRA;   // feature row offset
    const int rb = tgpig.x * NRB;   // token col offset

    // A (weights): (K × M) view, strides {1, K} — element (k, m) at w[m*K + k].
    // matmul2d's static_assert rejects `const float` element types; the same
    // type-system workaround the ternary kernel applies to its B view.
    device half* w_rw = (device half*)(weights_maj);
    auto tA = tensor(w_rw,
                     dextents<int32_t, 2>(k_input_dim, m_features),
                     array<int, 2>({1, (int)k_input_dim}));

    // B (activations): (K × P) view, strides {1, K} — element (k, p) at x[p*K + k].
    device float* input_rw = (device float*)(input_batch);
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
    // pattern, no manual zero-init, identical to the ternary kernel).
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
