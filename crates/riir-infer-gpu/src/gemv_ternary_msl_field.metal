// gemv_ternary_msl_field.metal
// Issue 1138 — the hand-MSL FFN decode-GEMV lane (i1137 reopen lever (a),
// adjudicated GO 2026-10-27). Raw MSL outside CubeCL codegen, dispatched via
// metal-rs (`gemv_ternary_msl_field.rs`) — the Issue-656/663 pattern.
//
// The GGML PR-26980 algebra (`kernel_mul_mv_tq2_0_f32_impl`, the vendored
// fork's native ternary kernel) mapped onto OUR pos/neg plane format:
//
// - Field packing: byte v holds four 2-bit fields c = (p − n) + 2 ∈ {1,2,3},
//   weight w = c − 2. Byte b covers values [4b .. 4b+3] SEQUENTIALLY (our
//   repack choice — the fork's qs order interleaves against its y layout;
//   ours is trivially coalesced: byte b's y-float4 is y4[b]).
// - Per-byte dot identity (exact float-floor extraction, no per-bit integer
//   work): with f0 = v, f1 = ⌊v/4⌋, f2 = ⌊v/16⌋, f3 = ⌊v/64⌋ and
//   coef = (a0, a1 − 4a0, a2 − 4a1, a3 − 4a2) for the byte's four y values
//   a0..a3, the identity Σ coef·f = Σ c_k·a_k holds exactly (the 4a/16a/64a
//   terms cancel telescopically), so
//       Σ (c_k − 2)·a_k = dot(coef, f) − 2·(a0+a1+a2+a3).
//   (The fork subtracts sumy ONCE because their code is w = q − 1; ours is
//   w = c − 2, hence the factor 2.)
// - Block = 256 values = TWO of our 128-value scale groups (halves h0 ∈
//   {0,1}), each half scaled by its own f16 group scale. The fork has ONE
//   f16 d per 256-block; we have two per block.
// - Geometry = the fork's: 8 rows/threadgroup (4 rows/simdgroup × 2
//   simdgroups, 64 threads), NBLOCK = 4 blocks strided across the simdgroup
//   (blk = tiisg/8), coef[4] + sumy hoisted per (block, half) and shared
//   across the 4 rows, float4 y loads.
//
// Dispatch gates (enforced host-side; refusal → the incumbent fallback, the
// Issue-1124 posture-gate shape): cols % 256 == 0, rows % 8 == 0.
//
// G1 class: NOT to_bits — the per-half scale placement and the accumulation
// order differ from the incumbent f16scale kernel. This falls under the
// pre-authorized Bench-742 law (unit max-abs band + loop-stream identity).

#include <metal_stdlib>
using namespace metal;

kernel void gemv_ternary_field_f32(
    device const uchar* qs,      // [rows * cols/4] row-major packed field bytes
    device const half*  gscales, // [rows * cols/128] row-major f16 group scales
    device const float* y,       // [cols] the input vector
    device float*       out,     // [rows]
    constant uint& rows,
    constant uint& cols,
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg  [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    const uint first_row = tgpig.x * 8u + sgitg * 4u;
    const uint blk = tiisg / 8u; // 0..3, the NBLOCK stride lane
    const uint htg = tiisg % 8u; // 0..7, thread within the block

    const uint nb = cols / 256u;      // 256-value blocks per row
    const uint groups = cols / 128u;  // our f16 scale groups per row
    const uint row_bytes = cols / 4u; // packed bytes per row

    device const float4* y4 = (device const float4*)y;

    float sumf[4] = {0.f, 0.f, 0.f, 0.f};

    for (uint ib = blk; ib < nb; ib += 4u) {
        for (uint h0 = 0; h0 < 2u; ++h0) {
            // This thread's 4 bytes of the half: byte columns
            // 64*ib + 32*h0 + 4*htg + {0,1,2,3}. Byte b covers values
            // [4b..4b+3], so its y-float4 is y4[b] — contiguous 4 loads.
            const uint base = (ib << 6) + (h0 << 5) + (htg << 2);
            const float4 Y0 = y4[base + 0];
            const float4 Y1 = y4[base + 1];
            const float4 Y2 = y4[base + 2];
            const float4 Y3 = y4[base + 3];

            // Hoisted per-byte coefficients + y-sum (row-independent: y is
            // shared across the 4 rows of the simdgroup).
            const float4 coef0 = float4(Y0.x, Y0.y - 4.0f * Y0.x, Y0.z - 4.0f * Y0.y, Y0.w - 4.0f * Y0.z);
            const float4 coef1 = float4(Y1.x, Y1.y - 4.0f * Y1.x, Y1.z - 4.0f * Y1.y, Y1.w - 4.0f * Y1.z);
            const float4 coef2 = float4(Y2.x, Y2.y - 4.0f * Y2.x, Y2.z - 4.0f * Y2.y, Y2.w - 4.0f * Y2.z);
            const float4 coef3 = float4(Y3.x, Y3.y - 4.0f * Y3.x, Y3.z - 4.0f * Y3.y, Y3.w - 4.0f * Y3.z);
            const float sumy = (Y0.x + Y0.y + Y0.z + Y0.w) + (Y1.x + Y1.y + Y1.z + Y1.w)
                             + (Y2.x + Y2.y + Y2.z + Y2.w) + (Y3.x + Y3.y + Y3.z + Y3.w);
            const float two_sumy = 2.0f * sumy;

            const uint qcol = base; // byte column == y float4 index (sequential pack)
            const uint gcol = (ib << 1) + h0;

            for (uint row = 0; row < 4u; ++row) {
                const uint r = first_row + row;
                device const uchar* q = qs + (ulong)r * (ulong)row_bytes + qcol;
                float sum = 0.f;
                {
                    const float v = (float)q[0];
                    sum += dot(coef0, float4(v, floor(v * 0.25f), floor(v * 0.0625f), floor(v * 0.015625f)));
                }
                {
                    const float v = (float)q[1];
                    sum += dot(coef1, float4(v, floor(v * 0.25f), floor(v * 0.0625f), floor(v * 0.015625f)));
                }
                {
                    const float v = (float)q[2];
                    sum += dot(coef2, float4(v, floor(v * 0.25f), floor(v * 0.0625f), floor(v * 0.015625f)));
                }
                {
                    const float v = (float)q[3];
                    sum += dot(coef3, float4(v, floor(v * 0.25f), floor(v * 0.0625f), floor(v * 0.015625f)));
                }
                const float d = (float)gscales[(ulong)r * (ulong)groups + gcol];
                sumf[row] += d * (sum - two_sumy);
            }
        }
    }

    for (uint row = 0; row < 4u; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }
    if (tiisg == 0u) {
        for (uint row = 0; row < 4u; ++row) {
            out[first_row + row] = sumf[row];
        }
    }
}
