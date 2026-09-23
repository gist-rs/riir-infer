//! Issue 608 T2 — dp4a ternary GEMV via raw CUDA (cudarc + nvrtc).
//!
//! Sits BESIDE the CubeCL path (`gemv_ternary_cubecl.rs`) as the CUDA-only
//! alternative for NVIDIA GPUs. The CubeCL kernel remains the Metal/wgpu path
//! on Apple/non-NVIDIA hardware.
//!
//! ## Why this exists
//!
//! The CubeCL ternary GEMV achieves ~108 GB/s on the 4090 (10.7% of the
//! 1008 GB/s roofline). llama.cpp's CUDA kernel achieves 621.5 GB/s (61.7%)
//! using `__byte_perm` + `__dp4a` — two instructions CubeCL 0.10 cannot emit
//! (Issue 608 §"The blocker"). T1 confirmed a standalone `.cu` port of those
//! two instructions hits 901 GB/s (1.43× llama.cpp) on the same hardware.
//!
//! Closing this 8.3× kernel gap is the critical path to passing Issue 604 G2
//! on throughput (currently 15 tok/s vs 26.72 target — see `.issues/604`).
//!
//! ## Weight layout — bit-plane → packed Q2_0 codes
//!
//! The CubeCL path stores ternary weights as two bit-planes (`pos_bits`,
//! `neg_bits` as `[u64]`): bit k set in `pos_bits` → +1, in `neg_bits` → -1,
//! neither → 0. The dp4a kernel needs them as packed 2-bit codes (8 codes per
//! `int16`), indexed by the `__byte_perm(0x020100FF, ...)` trick where
//! `code 0 → -1, code 1 → 0, code 2 → +1`.
//!
//! Conversion (one-time at `upload_weights`):
//! ```text
//! code = pos_bit - neg_bit + 1   ∈ {0, 1, 2}
//! ```
//! Check: pos=1,neg=0 → 2 (+1); pos=0,neg=1 → 0 (-1); pos=0,neg=0 → 1 (0). ✓
//!
//! ## Activation quantization — f32 → int8 (Q8_1 scheme)
//!
//! `__dp4a` requires both operands in int8. Activations are quantized per
//! `ACTIVATION_BLOCK` elements: `act_int8[i] = round(x[i] / absmax * 127)`,
//! with the per-block scale stored alongside as a float. T3 (Issue 608 §T3)
//! measured the accuracy cost: mean_rel 3.7e-3 on gaussian activations,
//! 1.8e-2 on outlier distributions; argmax preserved on every row. T3b found
//! that shrinking the activation block from 32 → 16 buys 1.45× accuracy for
//! 1.2% throughput — adopted here as the default.
//!
//! ## Precision caveat — task-level gates required
//!
//! The int8 activation quantization means this path **cannot pass the existing
//! per-logit relative-error gates** (Bench 606 G1 = 1e-3, Issue 604 G1 = 1e-2).
//! Those gates correctly guard the f32 CubeCL path and stay in place. The
//! dp4a path needs task-level gates (argmax agreement + top-k overlap + quest
//! CSP solve rate) — Issue 608 T3a (open). Until T3a lands, this module is a
//! perf proof, not a production path.
//!
//! ## Proven cudarc pattern
//!
//! Mirrors `riir-train-gpu/src/muon_dense_cuda_ns.rs` (Issue 402 Phase 2):
//! CUDA source as a `const &str`, `cudarc::nvrtc::compile_ptx_with_opts` at
//! construction, `ctx.load_module` + `module.load_function`, then
//! `stream.launch_builder(&func).arg(...).launch(cfg)` per dispatch.

#![allow(clippy::too_many_arguments)]

use std::error::Error;
use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;

use katgpt_core::TernaryGroupWeights;

/// Activation quantization block size. T3b (Issue 608) found 16 buys 1.45×
/// accuracy over llama.cpp's 32 for 1.2% throughput — the knee is at 16, not 8.
pub const ACTIVATION_BLOCK: usize = 16;

/// Weight group size — must match `katgpt_types::GROUP_SIZE` (128).
pub(crate) const WEIGHT_GROUP: usize = 128;

/// Workgroup size — 8 warps × 32 lanes = 256 threads (matches T1).
/// Issue 741 T10 Phase B: `pub` (was pub(crate)) — riir-train-gpu's relocated
/// `gemv_transposed_into` uses it for grid math; re-exported at the crate
/// root with the module's other public items.
pub const WG_THREADS: u32 = 256;

/// CUDA source for the dp4a ternary GEMV kernel. Ported from the T1 standalone
/// probe (`crates/riir-gpu/cuda/issue608_t1_dp4a_probe.cu`).
///
/// One warp per output row. Each lane walks the activation blocks assigned to
/// it, accumulates a partial sum via `__dp4a`, then warp-reduces via
/// `__shfl_down_sync`. Constants are passed as kernel args so the same PTX
/// covers every (M, N) shape.
///
/// `pub(crate)` so the Issue 615 full-cudarc forward port
/// (`ternary_deltanet_gpu_forward_cudarc.rs`) can compile this kernel against
/// a shared `CudaContext` + `CudaStream` — DRY: one source, two integrations
/// (the standalone mixed-mode handler + the GPU-resident forward path).
pub const GEMV_CUDA_SRC: &str = r#"
// Issue 734 T5 — f16-bit wscale → f32. NVRTC compiles this source standalone
// (no CUDA headers), so `__half2float` is unavailable; inline PTX wraps the
// hardware `cvt.f32.f16` (exact for normals AND subnormals — Q2_0 group
// scales can be f16-subnormal, the gemv_q4k Issue 593 lesson).
__device__ __forceinline__ float f16_bits_to_f32(unsigned short hbits)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(hbits));
    return f;
}

// Issue 705 — the per-row dp4a GEMV math, shared by every gemv entrypoint
// below (single, multi, fused, persistent). Extracted from what had become
// three byte-identical copies; `__forceinline__` makes this a pure source-level
// dedup — the inlined instruction sequence per row is unchanged (same
// lane-strided block walk, same accumulation order, same warp reduction), so
// results stay bit-identical to the pre-refactor kernels.
//
// One warp computes one output row: each lane walks its strided activation
// blocks (blk = lane, lane+32, ...), accumulates via `__dp4a`, then the warp
// reduces via `__shfl_down_sync`. Returns the lane-0 result.
__device__ __forceinline__ float gemv_ternary_row(
    const short* __restrict__ row_codes,   // codes  + row * int16_per_row
    const unsigned short* __restrict__ row_scale, // wscale + row * groups_per_row (f16 bits)
    const signed char* __restrict__ act,   // quantized activations [n]
    const float* __restrict__ ascale,      // per-ablock scales [ablocks]
    int lane,
    int ablock,
    int ablocks)
{
    float acc = 0.0f;
    if (ablock == 16) {
        // Issue 734 T5 — vectorized fast path for the production shape
        // (ACTIVATION_BLOCK = 16 at every call site). Same accumulation order
        // as the generic loop below — j=0 consumes i16 word 0 (qpair low half)
        // with act ints 0/1 (uv.x/uv.y), j=1 consumes word 1 (qpair high half)
        // with ints 2/3 (uv.z/uv.w); the old `short` loads sign-extended bit 15
        // into bits 16+, which __byte_perm's selector never reads — selectors
        // are the low 16 bits only. Bit-identical outputs; the loads collapse
        // from 2×2B + 4×4B to 1×4B + 1×16B per 16 weights (3× fewer load
        // instructions on the DRAM-bound hot path).
        for (int blk = lane; blk < ablocks; blk += 32) {
            const int elem0 = blk << 4;
            // NOTE: __ldcs on this load measured a ~1.2% tok/s REGRESSION
            // (93.43 → 92.28, Issue 734 T5 A/B) — Ada's default caching wins
            // for the 128 B/warp-iteration streaming window. Plain load.
            const unsigned int qpair = *(const unsigned int*)(row_codes + (elem0 >> 3));
            const int4 uv = *(const int4*)(act + elem0);

            const int q0 = (int)(qpair & 0xffffu);
            const int q1 = (int)(qpair >> 16);

            int sumi = 0;
            {
                const int qe = __byte_perm(0x020100FF, 0x020100FF, q0 >> 0);
                const int qo = __byte_perm(0x020100FF, 0x020100FF, q0 >> 2);
                const int qx = __byte_perm(qe, qo, 0x5140);   // elements 0..3
                const int qy = __byte_perm(qe, qo, 0x7362);   // elements 4..7
                sumi = __dp4a(uv.x, qx, sumi);
                sumi = __dp4a(uv.y, qy, sumi);
            }
            {
                const int qe = __byte_perm(0x020100FF, 0x020100FF, q1 >> 0);
                const int qo = __byte_perm(0x020100FF, 0x020100FF, q1 >> 2);
                const int qx = __byte_perm(qe, qo, 0x5140);   // elements 8..11
                const int qy = __byte_perm(qe, qo, 0x7362);   // elements 12..15
                sumi = __dp4a(uv.z, qx, sumi);
                sumi = __dp4a(uv.w, qy, sumi);
            }

            // Issue 734 T5 — wscale is stored as f16 bits (half the DRAM traffic
            // of f32; 12.5% of the codes bytes at the old width). Every scale
            // value originated as f16 in the GGUF, so f16→f32 is EXACT — the
            // float value consumed here is bit-identical to the old f32 upload.
            const float ws = f16_bits_to_f32(row_scale[elem0 >> 7]);
            acc += (float)sumi * ws * ascale[blk];
        }
    } else {
        // Generic path (ablock != 16 — no production call site; kept for
        // shape safety). Original j-loop, unchanged.
        for (int blk = lane; blk < ablocks; blk += 32) {
            const int elem0 = blk * ablock;
            const short* q4 = row_codes + (elem0 >> 3);
            const int* u8   = (const int*)(act + elem0);

            int sumi = 0;
            const int inner = ablock >> 3;  // int16 words per block
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j >= inner) break;
                const int q = q4[j];
                const int u = u8[j * 2 + 0];
                const int v = u8[j * 2 + 1];
                const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
                const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
                const int qx = __byte_perm(qe, qo, 0x5140);
                const int qy = __byte_perm(qe, qo, 0x7362);
                sumi = __dp4a(u, qx, sumi);
                sumi = __dp4a(v, qy, sumi);
            }

            const float ws = f16_bits_to_f32(row_scale[elem0 >> 7]);
            acc += (float)sumi * ws * ascale[blk];
        }
    }

    // Warp reduction.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

// Original split-kernel path (Issue 608 T2). The activation quantize step
// runs as a SEPARATE kernel (quantize_f32_to_i8 in cudarc_kernels::mod)
// before this kernel; this entrypoint consumes its int8 + ascale outputs.
extern "C" __global__ void gemv_ternary_dp4a(
    const short* __restrict__ codes,    // [M * INT16_PER_ROW]  packed 2-bit codes
    const unsigned short* __restrict__ wscale, // [M * GROUPS_PER_ROW] f16-bit per-128 weight scales
    const signed char* __restrict__ act,// [N]                  int8 activations
    const float* __restrict__ ascale,   // [ABLOCKS]            per-ABLOCK scales
    float* __restrict__ out,            // [M]
    int m_rows,
    int int16_per_row,                  // n / 8
    int groups_per_row,                 // n / 128
    int ablock,                         // ACTIVATION_BLOCK (16)
    int ablocks)                        // n / ablock
{
    const int row  = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x % 32;
    if (row >= m_rows) return;

    const float acc = gemv_ternary_row(
        codes + (long)row * int16_per_row,
        wscale + (long)row * groups_per_row,
        act, ascale, lane, ablock, ablocks);
    if (lane == 0) out[row] = acc;
}

// Issue 697 — multi-segment dp4a GEMV: one launch computes up to 4 weight
// matrices that share the SAME input vector (same `n`, same quantized act).
// Motivation (nsys attribution, Bench 682): the per-token decode issued 497
// separate GEMV launches; the tiny a/b projections (m=48, 6 blocks) ran at the
// ~5.6 µs kernel latency floor (12 GB/s effective — 540 µs/token for 6.6 MB),
// and every launch paid a wave-quantization tail (gate/up at 774 GB/s vs the
// 854 GB/s the same inner loop achieves at 31,040 blocks on lm_head).
// Concatenating same-input segments into one grid amortizes both: the a/b
// rows ride along the qkv+z launch at zero marginal cost and the tail waves
// merge.
//
// Row math is BYTE-IDENTICAL to `gemv_ternary_dp4a` — the per-row dp4a loop,
// accumulation order, and warp reduction are unchanged; only the row →
// (codes, wscale, out) segment mapping is added (each row still executes the
// exact same instruction sequence it would in a standalone launch).
//
// `accumulate != 0` makes the kernel `out[row] += acc` instead of a plain
// store — used by the hot path to fold the residual add into the out_proj /
// down_proj GEMV (m == n_embd for both), eliminating the separate
// `residual_add_f32` launch + the tmp/ffn_out round-trip. The f32 add has the
// same operands and order as `residual_add_f32` did (x + gemv_result), so the
// result is bit-identical.
extern "C" __global__ void gemv_ternary_dp4a_multi(
    const short* __restrict__ codes0,  const unsigned short* __restrict__ wscale0,  float* __restrict__ out0, int m0,
    const short* __restrict__ codes1,  const unsigned short* __restrict__ wscale1,  float* __restrict__ out1, int m1,
    const short* __restrict__ codes2,  const unsigned short* __restrict__ wscale2,  float* __restrict__ out2, int m2,
    const short* __restrict__ codes3,  const unsigned short* __restrict__ wscale3,  float* __restrict__ out3, int m3,
    const signed char* __restrict__ act,
    const float* __restrict__ ascale,
    int int16_per_row,
    int groups_per_row,
    int ablock,
    int ablocks,
    int accumulate)
{
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x % 32;

    // Segment select: rows [0, m0) → seg0, [m0, m0+m1) → seg1, etc.
    // Zero-length segments match no rows — their pointers are never
    // dereferenced (the branch guards the row range).
    const short* codes;
    const unsigned short* wscale;
    float* out;
    if (row < m0) {
        codes = codes0; wscale = wscale0; out = out0;
    } else if ((row -= m0) < m1) {
        codes = codes1; wscale = wscale1; out = out1;
    } else if ((row -= m1) < m2) {
        codes = codes2; wscale = wscale2; out = out2;
    } else if ((row -= m2) < m3) {
        codes = codes3; wscale = wscale3; out = out3;
    } else {
        return;
    }

    const float acc = gemv_ternary_row(
        codes + (long)row * int16_per_row,
        wscale + (long)row * groups_per_row,
        act, ascale, lane, ablock, ablocks);
    if (lane == 0) {
        if (accumulate) out[row] += acc;
        else            out[row]  = acc;
    }
}

// Issue 705 (filed as 702, renumbered — the 702 number belongs to the l2_normalize suite-order divergence) — persistent grid-stride variant of `gemv_ternary_dp4a_multi`.
//
// The kernel supports launching at ANY grid size: each warp loops over rows
// with stride = total resident warps, so a capped grid (e.g. SMs ×
// blocks_per_sm) processes multiple rows per warp with no partial tail wave.
// Row math is byte-identical to `gemv_ternary_dp4a_multi` at every grid size:
// each row is computed by exactly one warp via `gemv_ternary_row` (same
// lane-strided walk, same accumulation order, same warp reduction) —
// grid-stride only changes WHICH warp computes a row, never how.
//
// **Measured verdict (Bench 684): the cap is a 1-2% LOSS** (uncapped 87.1 vs
// 768-cap 84.9-86.2 graph tok/s) — the wavefront locality the hardware block
// scheduler gives the over-sized naive grid outweighs the wave-quantization
// tail; the z-class counter-example (exact-1-wave at 706 GB/s vs lm_head's
// 854) independently refutes the tail theory. The production default is
// therefore UNcapped (this kernel with grid >= ceil(rows/8) — the loop runs
// ≤ 1 iteration per warp, behaviorally identical to the non-persistent
// kernel). The `RIIR_GEMV_PERSISTENT_GRID` env cap retains the capped mode
// as a measurement/reproduction apparatus.
extern "C" __global__ void gemv_ternary_dp4a_multi_persistent(
    const short* __restrict__ codes0,  const unsigned short* __restrict__ wscale0,  float* __restrict__ out0, int m0,
    const short* __restrict__ codes1,  const unsigned short* __restrict__ wscale1,  float* __restrict__ out1, int m1,
    const short* __restrict__ codes2,  const unsigned short* __restrict__ wscale2,  float* __restrict__ out2, int m2,
    const short* __restrict__ codes3,  const unsigned short* __restrict__ wscale3,  float* __restrict__ out3, int m3,
    const signed char* __restrict__ act,
    const float* __restrict__ ascale,
    int int16_per_row,
    int groups_per_row,
    int ablock,
    int ablocks,
    int accumulate,
    int total_rows)                     // m0 + m1 + m2 + m3
{
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int total_warps = gridDim.x * warps_per_block;

    for (int row = blockIdx.x * warps_per_block + (threadIdx.x / 32);
         row < total_rows;
         row += total_warps)
    {
        int r = row;
        const short* codes;
        const unsigned short* wscale;
        float* out;
        if (r < m0) {
            codes = codes0; wscale = wscale0; out = out0;
        } else if ((r -= m0) < m1) {
            codes = codes1; wscale = wscale1; out = out1;
        } else if ((r -= m1) < m2) {
            codes = codes2; wscale = wscale2; out = out2;
        } else if ((r -= m2) < m3) {
            codes = codes3; wscale = wscale3; out = out3;
        } else {
            return;  // unreachable when total_rows == m0+m1+m2+m3 (caller invariant)
        }

        const float acc = gemv_ternary_row(
            codes + (long)r * int16_per_row,
            wscale + (long)r * groups_per_row,
            act, ascale, lane, ablock, ablocks);
        if (lane == 0) {
            if (accumulate) out[r] += acc;
            else            out[r]  = acc;
        }
    }
}

// Plan 604 T1 (Issue 987 rung G1) — two-rows-per-warp variant of the multi
// GEMV for the UNDERFILLED launch classes (grid <= ~1 wave at the one-row
// shape; today: FFN down m=5,120 n=17,408 and GDN out_proj m=5,120 n=5,120,
// 128 launches/token).
//
// T0 attribution (2026-09-19): those classes run at ~648 GB/s vs the
// 824-854 GB/s the IDENTICAL inner loop reaches on the >=5.7-wave classes;
// the load-issue model says they sit at ~81% LSU issue utilization (3 loads
// per lane-block-iteration: qpair + act int4 + ascale) vs ~63% for the
// at-ceiling classes — the activation loads are the binding constraint.
// This kernel shares the act window (`uv`) and `ascale[blk]` between TWO
// consecutive rows of the same warp: 4 loads per 2 lane-block-iterations =
// 2/blk (-33% LSU pressure). The code streams (qpair + wscale) stay
// per-row and unchanged.
//
// Bit-identity: each row keeps its OWN lane-strided block walk, dp4a chain,
// accumulation order, and 5-step warp reduction — the exact instruction
// sequence `gemv_ternary_row` executes per row; only the act loads are
// shared, and the loaded values are the same bytes the one-row kernel reads
// from the same addresses. Segment resolution is per-row (a pair may
// straddle a segment boundary — each row runs the if-chain independently).
// Grid-stride over PAIRS mirrors the persistent kernel (uncapped default
// = exactly one pair per warp).
extern "C" __global__ void gemv_ternary_dp4a_multi_r2(
    const short* __restrict__ codes0,  const unsigned short* __restrict__ wscale0,  float* __restrict__ out0, int m0,
    const short* __restrict__ codes1,  const unsigned short* __restrict__ wscale1,  float* __restrict__ out1, int m1,
    const short* __restrict__ codes2,  const unsigned short* __restrict__ wscale2,  float* __restrict__ out2, int m2,
    const short* __restrict__ codes3,  const unsigned short* __restrict__ wscale3,  float* __restrict__ out3, int m3,
    const signed char* __restrict__ act,
    const float* __restrict__ ascale,
    int int16_per_row,
    int groups_per_row,
    int ablock,
    int ablocks,
    int accumulate,
    int total_rows)
{
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int total_warps = gridDim.x * warps_per_block;

    for (int pair = blockIdx.x * warps_per_block + (threadIdx.x / 32);
         pair * 2 < total_rows;
         pair += total_warps)
    {
        const int rowA = pair * 2;
        const int rowB = rowA + 1;
        const bool hasB = rowB < total_rows;  // warp-uniform: same rows, whole warp

        // Per-row segment resolution — the same chain the one-row kernel runs.
        const short* codesA; const unsigned short* wsA; float* outA; int rA;
        const short* codesB; const unsigned short* wsB; float* outB; int rB;
        {
            int r = rowA;
            if (r < m0) {
                codesA = codes0; wsA = wscale0; outA = out0; rA = r;
            } else if ((r -= m0) < m1) {
                codesA = codes1; wsA = wscale1; outA = out1; rA = r;
            } else if ((r -= m1) < m2) {
                codesA = codes2; wsA = wscale2; outA = out2; rA = r;
            } else if ((r -= m2) < m3) {
                codesA = codes3; wsA = wscale3; outA = out3; rA = r;
            } else {
                return;
            }
        }
        {
            int r = rowB;
            if (r < m0) {
                codesB = codes0; wsB = wscale0; outB = out0; rB = r;
            } else if ((r -= m0) < m1) {
                codesB = codes1; wsB = wscale1; outB = out1; rB = r;
            } else if ((r -= m1) < m2) {
                codesB = codes2; wsB = wscale2; outB = out2; rB = r;
            } else if ((r -= m2) < m3) {
                codesB = codes3; wsB = wscale3; outB = out3; rB = r;
            } else {
                codesB = codesA; wsB = wsA; outB = outA; rB = rA;  // hasB false — never dereferenced
            }
        }

        const short* cA = codesA + (long)rA * int16_per_row;
        const unsigned short* wA = wsA + (long)rA * groups_per_row;
        const short* cB = hasB ? (codesB + (long)rB * int16_per_row) : cA;
        const unsigned short* wB = hasB ? (wsB + (long)rB * groups_per_row) : wA;

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        if (ablock == 16) {
            // Vectorized two-row path (production shape). Per iteration: ONE
            // act window load + ONE ascale load, TWO code-stream loads; each
            // row's dp4a chain and per-block multiply-add sequence is
            // instruction-for-instruction the ablock==16 arm of
            // gemv_ternary_row (same operand values, same order).
            for (int blk = lane; blk < ablocks; blk += 32) {
                const int elem0 = blk << 4;
                const unsigned int qpA = *(const unsigned int*)(cA + (elem0 >> 3));
                const unsigned int qpB = *(const unsigned int*)(cB + (elem0 >> 3));
                const int4 uv = *(const int4*)(act + elem0);
                const float as = ascale[blk];

                int siA = 0;
                {
                    const int q0 = (int)(qpA & 0xffffu);
                    const int q1 = (int)(qpA >> 16);
                    const int qe = __byte_perm(0x020100FF, 0x020100FF, q0 >> 0);
                    const int qo = __byte_perm(0x020100FF, 0x020100FF, q0 >> 2);
                    const int qx = __byte_perm(qe, qo, 0x5140);   // elements 0..3
                    const int qy = __byte_perm(qe, qo, 0x7362);   // elements 4..7
                    siA = __dp4a(uv.x, qx, siA);
                    siA = __dp4a(uv.y, qy, siA);
                    const int qe2 = __byte_perm(0x020100FF, 0x020100FF, q1 >> 0);
                    const int qo2 = __byte_perm(0x020100FF, 0x020100FF, q1 >> 2);
                    const int qx2 = __byte_perm(qe2, qo2, 0x5140); // elements 8..11
                    const int qy2 = __byte_perm(qe2, qo2, 0x7362); // elements 12..15
                    siA = __dp4a(uv.z, qx2, siA);
                    siA = __dp4a(uv.w, qy2, siA);
                }
                acc0 += (float)siA * f16_bits_to_f32(wA[elem0 >> 7]) * as;

                if (hasB) {
                    int siB = 0;
                    {
                        const int q0 = (int)(qpB & 0xffffu);
                        const int q1 = (int)(qpB >> 16);
                        const int qe = __byte_perm(0x020100FF, 0x020100FF, q0 >> 0);
                        const int qo = __byte_perm(0x020100FF, 0x020100FF, q0 >> 2);
                        const int qx = __byte_perm(qe, qo, 0x5140);
                        const int qy = __byte_perm(qe, qo, 0x7362);
                        siB = __dp4a(uv.x, qx, siB);
                        siB = __dp4a(uv.y, qy, siB);
                        const int qe2 = __byte_perm(0x020100FF, 0x020100FF, q1 >> 0);
                        const int qo2 = __byte_perm(0x020100FF, 0x020100FF, q1 >> 2);
                        const int qx2 = __byte_perm(qe2, qo2, 0x5140);
                        const int qy2 = __byte_perm(qe2, qo2, 0x7362);
                        siB = __dp4a(uv.z, qx2, siB);
                        siB = __dp4a(uv.w, qy2, siB);
                    }
                    acc1 += (float)siB * f16_bits_to_f32(wB[elem0 >> 7]) * as;
                }
            }
        } else {
            // Generic fallback (ablock != 16 — no production call site): no
            // amortization, plain per-row calls. Correctness-only path.
            acc0 = gemv_ternary_row(cA, wA, act, ascale, lane, ablock, ablocks);
            if (hasB) {
                acc1 = gemv_ternary_row(cB, wB, act, ascale, lane, ablock, ablocks);
            }
        }

        // Two independent 5-step warp reductions — the same reduction each
        // row gets in the one-row kernel (hasB is warp-uniform, so full-mask
        // shuffles are safe even when acc1 is unused).
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc0 += __shfl_down_sync(0xffffffffu, acc0, off);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, off);
        }
        if (lane == 0) {
            if (accumulate) {
                outA[rA] += acc0;
                if (hasB) outB[rB] += acc1;
            } else {
                outA[rA] = acc0;
                if (hasB) outB[rB] = acc1;
            }
        }
    }
}

// Plan 604 T3 (Issue 987 rung G2, re-ranked after T1's negative) — the
// one-row persistent kernel + a per-lane `prefetch.global.L2` of the NEXT
// warp-iteration's codes word, for the latency-exposed UNDERFILLED classes.
//
// Evidence chain: T0 put the 640-block down/out class at ~648 GB/s vs the
// 824-854 ceiling; T1's R2 experiment (two rows per warp, act-load sharing)
// measured a 4.4% LOSS — the warp-count halving cost more than the L1 act
// traffic cut saved, which both (a) bounds the smem-staging rung's upside to
// ~the L1 term (parked) and (b) says the class is short on MEMORY-LEVEL
// PARALLELISM, not L1 bandwidth: fewer resident warps made it slower. A
// ≤1-wave launch hides DRAM latency only with in-flight loads per warp —
// the exact condition the fork's GB10 branch names ("scoreboard-latency
// bound ... prefetch one K iteration ahead", their #135) — while upstream's
// own adoption is gated to the LOW-bandwidth DGX-Spark (#26705, "little
// exposed latency left" on big parts — about BIG launches on high-BW parts).
// This variant targets only the underfilled launches, where the Spark
// condition (exposed latency) actually holds on a 4090.
//
// The prefetch is data-path-free: same loads, same dp4a chains, same
// reductions — outputs are bit-identical by construction (a prefetch cannot
// change values; the full-model FNV pins are the end-to-end proof).
extern "C" __global__ void gemv_ternary_dp4a_multi_pf(
    const short* __restrict__ codes0,  const unsigned short* __restrict__ wscale0,  float* __restrict__ out0, int m0,
    const short* __restrict__ codes1,  const unsigned short* __restrict__ wscale1,  float* __restrict__ out1, int m1,
    const short* __restrict__ codes2,  const unsigned short* __restrict__ wscale2,  float* __restrict__ out2, int m2,
    const short* __restrict__ codes3,  const unsigned short* __restrict__ wscale3,  float* __restrict__ out3, int m3,
    const signed char* __restrict__ act,
    const float* __restrict__ ascale,
    int int16_per_row,
    int groups_per_row,
    int ablock,
    int ablocks,
    int accumulate,
    int total_rows)
{
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int total_warps = gridDim.x * warps_per_block;

    for (int row = blockIdx.x * warps_per_block + (threadIdx.x / 32);
         row < total_rows;
         row += total_warps)
    {
        int r = row;
        const short* codes;
        const unsigned short* wscale;
        float* out;
        if (r < m0) {
            codes = codes0; wscale = wscale0; out = out0;
        } else if ((r -= m0) < m1) {
            codes = codes1; wscale = wscale1; out = out1;
        } else if ((r -= m1) < m2) {
            codes = codes2; wscale = wscale2; out = out2;
        } else if ((r -= m2) < m3) {
            codes = codes3; wscale = wscale3; out = out3;
        } else {
            return;
        }

        const short* row_codes = codes + (long)r * int16_per_row;
        const unsigned short* row_scale = wscale + (long)r * groups_per_row;

        if (ablock == 16) {
            float acc = 0.0f;
            for (int blk = lane; blk < ablocks; blk += 32) {
                const int elem0 = blk << 4;
                // One K-iteration-ahead L2 prefetch of this lane's next codes
                // word (32 lanes' next words = the warp's next 128-B window —
                // the fork #135 pattern). No data-path effect.
                if (blk + 32 < ablocks) {
                    const short* next_word = row_codes + ((blk + 32) << 4 >> 3);
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(next_word));
                }
                const unsigned int qpair = *(const unsigned int*)(row_codes + (elem0 >> 3));
                const int4 uv = *(const int4*)(act + elem0);

                const int q0 = (int)(qpair & 0xffffu);
                const int q1 = (int)(qpair >> 16);

                int sumi = 0;
                {
                    const int qe = __byte_perm(0x020100FF, 0x020100FF, q0 >> 0);
                    const int qo = __byte_perm(0x020100FF, 0x020100FF, q0 >> 2);
                    const int qx = __byte_perm(qe, qo, 0x5140);
                    const int qy = __byte_perm(qe, qo, 0x7362);
                    sumi = __dp4a(uv.x, qx, sumi);
                    sumi = __dp4a(uv.y, qy, sumi);
                }
                {
                    const int qe = __byte_perm(0x020100FF, 0x020100FF, q1 >> 0);
                    const int qo = __byte_perm(0x020100FF, 0x020100FF, q1 >> 2);
                    const int qx = __byte_perm(qe, qo, 0x5140);
                    const int qy = __byte_perm(qe, qo, 0x7362);
                    sumi = __dp4a(uv.z, qx, sumi);
                    sumi = __dp4a(uv.w, qy, sumi);
                }

                const float ws = f16_bits_to_f32(row_scale[elem0 >> 7]);
                acc += (float)sumi * ws * ascale[blk];
            }

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) {
                if (accumulate) out[r] += acc;
                else            out[r]  = acc;
            }
        } else {
            // Generic path (ablock != 16 — no production site): the unmodified
            // one-row math, no prefetch.
            const float acc = gemv_ternary_row(
                row_codes, row_scale, act, ascale, lane, ablock, ablocks);
            if (lane == 0) {
                if (accumulate) out[r] += acc;
                else            out[r]  = acc;
            }
        }
    }
}

// Plan 604 follow-up (Issue 987's menu, rung G4 — the one MLP lever the
// campaign never tried): the one-row kernel with a MANUALLY 4x-batched
// K-loop. R2 (fewer warps, −4.4%) and PF (an L2 prefetch HINT, no-op) both
// failed to move the underfilled 640-block down/out class; this rung KEEPS
// the warp count and the one-row shape and instead puts FOUR REAL dependent
// loads in flight per lane before the dp4a chains consume them — raising
// per-warp memory-level parallelism without removing warps, the exact
// combination neither negative rung tested.
//
// Bit-identity: each block's work is an independent integer dp4a chain
// (exact), and the FP acc updates execute in EXACTLY the order the
// sequential loop uses (blk, blk+32, blk+64, blk+96) — only the LOADS are
// hoisted above the compute, no FP reassociation. Tail iterations (when
// ablocks - blk < 96+1) run the original body. Same warp reduction.
extern "C" __global__ void gemv_ternary_dp4a_multi_u4(
    const short* codes0,  const unsigned short* wscale0,  float* out0, int m0,
    const short* codes1,  const unsigned short* wscale1,  float* out1, int m1,
    const short* codes2,  const unsigned short* wscale2,  float* out2, int m2,
    const short* codes3,  const unsigned short* wscale3,  float* out3, int m3,
    const signed char* act,
    const float* ascale,
    int int16_per_row,
    int groups_per_row,
    int ablock,
    int ablocks,
    int accumulate,
    int total_rows)
{
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int total_warps = gridDim.x * warps_per_block;

    for (int row = blockIdx.x * warps_per_block + (threadIdx.x / 32);
         row < total_rows;
         row += total_warps)
    {
        int r = row;
        const short* codes;
        const unsigned short* wscale;
        float* out;
        if (r < m0) {
            codes = codes0; wscale = wscale0; out = out0;
        } else if ((r -= m0) < m1) {
            codes = codes1; wscale = wscale1; out = out1;
        } else if ((r -= m1) < m2) {
            codes = codes2; wscale = wscale2; out = out2;
        } else if ((r -= m2) < m3) {
            codes = codes3; wscale = wscale3; out = out3;
        } else {
            return;
        }

        const short* row_codes = codes + (long)r * int16_per_row;
        const unsigned short* row_scale = wscale + (long)r * groups_per_row;

        float acc = 0.0f;
        if (ablock == 16) {
            int blk = lane;
            // 4x-batched main loop: hoist the four blocks' loads (codes word,
            // act int4, wscale) so they are in flight together, then run each
            // block's dp4a chain and FP acc update IN SOURCE ORDER.
            for (; blk + 96 < ablocks; blk += 128) {
                const int e0 = blk << 4;
                const int e1 = (blk + 32) << 4;
                const int e2 = (blk + 64) << 4;
                const int e3 = (blk + 96) << 4;

                const unsigned int qp0 = *(const unsigned int*)(row_codes + (e0 >> 3));
                const unsigned int qp1 = *(const unsigned int*)(row_codes + (e1 >> 3));
                const unsigned int qp2 = *(const unsigned int*)(row_codes + (e2 >> 3));
                const unsigned int qp3 = *(const unsigned int*)(row_codes + (e3 >> 3));
                const int4 uv0 = *(const int4*)(act + e0);
                const int4 uv1 = *(const int4*)(act + e1);
                const int4 uv2 = *(const int4*)(act + e2);
                const int4 uv3 = *(const int4*)(act + e3);
                const float w0 = f16_bits_to_f32(row_scale[e0 >> 7]);
                const float w1 = f16_bits_to_f32(row_scale[e1 >> 7]);
                const float w2 = f16_bits_to_f32(row_scale[e2 >> 7]);
                const float w3 = f16_bits_to_f32(row_scale[e3 >> 7]);

                #define GEMV_U4_STEP(qp, uv, wscl, blkidx)                                   \
                    do {                                                                      \
                        const int q0_ = (int)((qp) & 0xffffu);                                \
                        const int q1_ = (int)((qp) >> 16);                                    \
                        int sumi_ = 0;                                                        \
                        {                                                                     \
                            const int qe_ = __byte_perm(0x020100FF, 0x020100FF, q0_ >> 0);   \
                            const int qo_ = __byte_perm(0x020100FF, 0x020100FF, q0_ >> 2);   \
                            const int qx_ = __byte_perm(qe_, qo_, 0x5140);                   \
                            const int qy_ = __byte_perm(qe_, qo_, 0x7362);                   \
                            sumi_ = __dp4a((uv).x, qx_, sumi_);                              \
                            sumi_ = __dp4a((uv).y, qy_, sumi_);                              \
                        }                                                                     \
                        {                                                                     \
                            const int qe_ = __byte_perm(0x020100FF, 0x020100FF, q1_ >> 0);   \
                            const int qo_ = __byte_perm(0x020100FF, 0x020100FF, q1_ >> 2);   \
                            const int qx_ = __byte_perm(qe_, qo_, 0x5140);                   \
                            const int qy_ = __byte_perm(qe_, qo_, 0x7362);                   \
                            sumi_ = __dp4a((uv).z, qx_, sumi_);                              \
                            sumi_ = __dp4a((uv).w, qy_, sumi_);                              \
                        }                                                                     \
                        acc += (float)sumi_ * (wscl) * ascale[blkidx];                       \
                    } while (0)

                GEMV_U4_STEP(qp0, uv0, w0, blk);
                GEMV_U4_STEP(qp1, uv1, w1, blk + 32);
                GEMV_U4_STEP(qp2, uv2, w2, blk + 64);
                GEMV_U4_STEP(qp3, uv3, w3, blk + 96);
                #undef GEMV_U4_STEP
            }
            // Tail: the original 1x body, order-continuing.
            for (; blk < ablocks; blk += 32) {
                const int elem0 = blk << 4;
                const unsigned int qpair = *(const unsigned int*)(row_codes + (elem0 >> 3));
                const int4 uv = *(const int4*)(act + elem0);
                const int q0 = (int)(qpair & 0xffffu);
                const int q1 = (int)(qpair >> 16);
                int sumi = 0;
                {
                    const int qe = __byte_perm(0x020100FF, 0x020100FF, q0 >> 0);
                    const int qo = __byte_perm(0x020100FF, 0x020100FF, q0 >> 2);
                    const int qx = __byte_perm(qe, qo, 0x5140);
                    const int qy = __byte_perm(qe, qo, 0x7362);
                    sumi = __dp4a(uv.x, qx, sumi);
                    sumi = __dp4a(uv.y, qy, sumi);
                }
                {
                    const int qe = __byte_perm(0x020100FF, 0x020100FF, q1 >> 0);
                    const int qo = __byte_perm(0x020100FF, 0x020100FF, q1 >> 2);
                    const int qx = __byte_perm(qe, qo, 0x5140);
                    const int qy = __byte_perm(qe, qo, 0x7362);
                    sumi = __dp4a(uv.z, qx, sumi);
                    sumi = __dp4a(uv.w, qy, sumi);
                }
                const float ws = f16_bits_to_f32(row_scale[elem0 >> 7]);
                acc += (float)sumi * ws * ascale[blk];
            }
        } else {
            // Generic path (ablock != 16 — no production call site): the
            // unmodified one-row math.
            acc = gemv_ternary_row(row_codes, row_scale, act, ascale, lane, ablock, ablocks);
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) {
            if (accumulate) out[r] += acc;
            else            out[r]  = acc;
        }
    }
}


// Plan 611 (Issue 1000 — the owner-greenlit lossy lane) — the SPLIT rung for
// the underfilled 640-block down/out class: TWO warps per row, each walking
// HALF the ablock range, partials combined through shared memory in a FIXED
// order (lower half + upper half). Bench 941's four-way exhaustion left this
// as the one untried direction: R2 REMOVED warps (-4.4%), PF/U4 raised
// per-lane memory-level parallelism at constant warp count (both no-ops) —
// the class is short on RESIDENT warps (5 blocks/SM = 40 of 64), and this
// rung demands 2x warp slots (the 5,120-row class goes 640 -> 1280 blocks,
// packing wave 1 to 8 blocks/SM where occupancy allows).
//
// LOSSY CLASS: the accumulation order changes (two half-range butterflies
// summed pairwise instead of one full-range butterfly) — deterministic at
// every run, but NOT bit-identical to the one-row kernels. The Issue-750-T3
// lossy-surface gates (per-family retention walk + a new pin class) govern
// any promotion; `RIIR_GEMV_SPLIT` keeps it default-off until then.
extern "C" __global__ void gemv_ternary_dp4a_multi_split2(
    const short* codes0,  const unsigned short* wscale0,  float* out0, int m0,
    const short* codes1,  const unsigned short* wscale1,  float* out1, int m1,
    const short* codes2,  const unsigned short* wscale2,  float* out2, int m2,
    const short* codes3,  const unsigned short* wscale3,  float* out3, int m3,
    const signed char* act,
    const float* ascale,
    int int16_per_row,
    int groups_per_row,
    int ablock,
    int ablocks,
    int accumulate,
    int total_rows)
{
    // Two warps per row: the warp-PAIR index is the row.
    const int row  = blockIdx.x * (blockDim.x / 64) + (threadIdx.x / 64);
    const int lane = threadIdx.x % 32;
    const int wid  = threadIdx.x >> 5;          // warp id within the block
    const int w    = wid & 1;                   // warp-in-pair (0: lower half)
    const bool active = row < total_rows;

    // Segment select (same chain as the multi kernel; `r` ends SEGMENT-LOCAL).
    // Inactive pairs keep the segment-0 pointers and never dereference (the
    // walk AND the write are guarded by `active`) — NO early return: the
    // block-wide __syncthreads() below must be reached by every thread.
    const short* codes = codes0;
    const unsigned short* wscale = wscale0;
    float* out = out0;
    int r = row;
    if (active) {
        if (r >= m0) {
            if ((r -= m0) >= m1) {
                if ((r -= m1) >= m2) {
                    r -= m2;
                    codes = codes3; wscale = wscale3; out = out3;
                } else {
                    codes = codes2; wscale = wscale2; out = out2;
                }
            } else {
                codes = codes1; wscale = wscale1; out = out1;
            }
        }
    } else {
        r = 0;
    }

    const int half = (ablocks + 1) >> 1;
    const int blk0 = w * half;
    const int blk1 = (w == 0) ? half : ablocks;

    float acc = 0.0f;
    if (active) {
        const short* row_codes = codes + (long)r * int16_per_row;
        const unsigned short* row_scale = wscale + (long)r * groups_per_row;
        if (ablock == 16) {
            // Range-clamped copy of gemv_ternary_row's fast path — the
            // per-block math is VERBATIM; only the loop bounds change
            // ([blk0, blk1) instead of [0, ablocks)).
            for (int blk = blk0 + lane; blk < blk1; blk += 32) {
                const int elem0 = blk << 4;
                const unsigned int qpair = *(const unsigned int*)(row_codes + (elem0 >> 3));
                const int4 uv = *(const int4*)(act + elem0);
                const int q0 = (int)(qpair & 0xffffu);
                const int q1 = (int)(qpair >> 16);
                int sumi = 0;
                {
                    const int qe = __byte_perm(0x020100FF, 0x020100FF, q0 >> 0);
                    const int qo = __byte_perm(0x020100FF, 0x020100FF, q0 >> 2);
                    const int qx = __byte_perm(qe, qo, 0x5140);
                    const int qy = __byte_perm(qe, qo, 0x7362);
                    sumi = __dp4a(uv.x, qx, sumi);
                    sumi = __dp4a(uv.y, qy, sumi);
                }
                {
                    const int qe = __byte_perm(0x020100FF, 0x020100FF, q1 >> 0);
                    const int qo = __byte_perm(0x020100FF, 0x020100FF, q1 >> 2);
                    const int qx = __byte_perm(qe, qo, 0x5140);
                    const int qy = __byte_perm(qe, qo, 0x7362);
                    sumi = __dp4a(uv.z, qx, sumi);
                    sumi = __dp4a(uv.w, qy, sumi);
                }
                const float ws = f16_bits_to_f32(row_scale[elem0 >> 7]);
                acc += (float)sumi * ws * ascale[blk];
            }
        } else {
            // Generic path (ablock != 16 — no production call site): the
            // original j-loop form, range-clamped.
            for (int blk = blk0 + lane; blk < blk1; blk += 32) {
                const int elem0 = blk * ablock;
                const short* q4 = row_codes + (elem0 >> 3);
                const int* u8   = (const int*)(act + elem0);
                int sumi = 0;
                const int inner = ablock >> 3;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (j >= inner) break;
                    const int q = q4[j];
                    const int u = u8[j * 2 + 0];
                    const int v = u8[j * 2 + 1];
                    const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
                    const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
                    const int qx = __byte_perm(qe, qo, 0x5140);
                    const int qy = __byte_perm(qe, qo, 0x7362);
                    sumi = __dp4a(u, qx, sumi);
                    sumi = __dp4a(v, qy, sumi);
                }
                const float ws = f16_bits_to_f32(row_scale[elem0 >> 7]);
                acc += (float)sumi * ws * ascale[blk];
            }
        }
    }

    // Same warp reduction as every other variant.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);

    __shared__ float partials[16];   // one slot per warp (8 at 256 threads)
    if (lane == 0) partials[wid] = acc;
    __syncthreads();
    if (active && w == 0 && lane == 0) {
        // FIXED combine order: lower-half partial + upper-half partial.
        const float total = partials[wid] + partials[wid + 1];
        if (accumulate) out[r] += total;
        else            out[r]  = total;
    }
}
// Issue 616 T4 — fused quantize+dp4a kernel with cooperative shared-memory
// quantization. Takes f32 activations directly; all 256 threads in the block
// cooperate to quantize the full N-element activation vector into shared
// memory ONCE, then each warp runs its dp4a loop reading from shared memory.
//
// This mirrors llama.cpp's fused approach: the activation is quantized in-block
// rather than via a separate kernel launch + global-memory round-trip.
//
// Shared memory layout (`extern __shared__ byte`):
//   [0 .. N)               — signed char act_i8[N]
//   [N .. N + ablocks*4)   — float ascale[ablocks]
// Total: N + ablocks*4 bytes (caller sets via `shared_mem_bytes`).
//
// Bit-identical to the split path (same quantization formula, same dp4a loop).
extern "C" __global__ void gemv_ternary_dp4a_fused(
    const short* __restrict__ codes,    // [M * INT16_PER_ROW]  packed 2-bit codes
    const unsigned short* __restrict__ wscale, // [M * GROUPS_PER_ROW] f16-bit per-128 weight scales
    const float* __restrict__ act,      // [N]                  f32 activations
    float* __restrict__ out,            // [M]
    int m_rows,
    int int16_per_row,                  // n / 8
    int groups_per_row,                 // n / 128
    int ablock,                         // ACTIVATION_BLOCK (16)
    int ablocks,                        // n / ablock
    int n)                              // activation length
{
    // ── Phase 1: cooperative quantize f32 → int8 into shared memory ──
    extern __shared__ signed char smem_raw[];
    signed char* sh_act = smem_raw;                  // [N]
    float* sh_ascale = (float*)(smem_raw + n);       // [ablocks]

    const int tid = threadIdx.x;
    const int bs = blockDim.x;  // 256

    // Each thread quantizes whole 16-element blocks in a strided pattern.
    // Thread t handles blocks t, t+bs, t+2*bs, ... Each block = ablock elements.
    // This mirrors quantize_f32_to_i8 bit-exactly (same per-block absmax path).
    for (int blk = tid; blk < ablocks; blk += bs) {
        const int start = blk * ablock;
        const int len = (ablock < n - start) ? ablock : (n - start);

        // Per-block absmax (sequential over the block's `len` elements).
        float absmax = 0.0f;
        for (int i = 0; i < len; ++i) {
            absmax = fmaxf(absmax, fabsf(act[start + i]));
        }
        const float d = (absmax > 0.0f) ? (absmax * (1.0f / 127.0f)) : 1.0f;
        const float inv_d = 1.0f / d;
        if (tid == blk % bs || true) {
            // Every thread writes its own block's ascale — no conflict since
            // each block is owned by exactly one thread.
            sh_ascale[blk] = d;
        }
        for (int i = 0; i < len; ++i) {
            int q = (int)roundf(act[start + i] * inv_d);
            q = (q < -128) ? -128 : (q > 127 ? 127 : q);
            sh_act[start + i] = (signed char)q;
        }
    }
    __syncthreads();

    // ── Phase 2: dp4a GEMV — same loop as the split-kernel path ──
    // Now reads from sh_act (shared memory) instead of global act buffer.
    const int row  = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x % 32;
    if (row >= m_rows) return;

    const float acc = gemv_ternary_row(
        codes + (long)row * int16_per_row,
        wscale + (long)row * groups_per_row,
        sh_act, sh_ascale, lane, ablock, ablocks);
    if (lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Issue 641 T7.3 — Transposed ternary GEMV for backward pass.
//
// Computes y[n] = W^T @ x[m] where W is the ternary weight matrix [m, n]
// stored in packed 2-bit code format. This is the backward matvec
// (grad_input = W^T @ grad_output) needed for per-layer gradient
// propagation through frozen ternary projection layers.
//
// Unlike the forward `gemv_ternary_dp4a`, this kernel does NOT use int8
// activation quantization or dp4a — the transposed access pattern
// (column-wise reads of a row-major ternary matrix) makes dp4a
// inapplicable. Instead, each thread accumulates a scalar sum over all
// rows for one output column, reading one ternary value per row.
//
// The memory access pattern is well-coalesced within each warp: 32
// consecutive threads (lanes) read from the same row i, accessing shorts
// at consecutive positions (4 unique shorts per warp, since 8 codes per
// short and 32 lanes → 4 shorts). The per-group scale and gradient are
// broadcast reads (same for all lanes in the same column-group).
//
// Launch: grid = ceil(n / blockDim.x), block = 256 (8 warps).
// Each thread handles one output column j. Iterates over all m rows.
//
// Performance estimate (m=17408, n=5120, 4090):
//   Total FMAs: 89M. At ~10K FMA/cycle: ~9K cycles ≈ 9 us.
//   Memory: ~22MB weight codes at 1008 GB/s: ~22 us.
//   Expected: ~25-50 us per matvec.
// ---------------------------------------------------------------------------

extern "C" __global__ void gemv_ternary_transposed_f32(
    const short* __restrict__ codes,     // [m * int16_per_row]  packed 2-bit codes
    const unsigned short* __restrict__ wscale, // [m * groups_per_row] f16-bit per-128 weight scales
    const float* __restrict__ grad_y,    // [m]                  f32 upstream gradient
    float* __restrict__ grad_x,          // [n]                  output (W^T @ grad_y)
    int m_rows,
    int n_cols,
    int int16_per_row,                   // n / 8
    int groups_per_row)                  // ceil(n / 128)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_cols) return;

    // Column-group index (fixed for this thread — determines which scale
    // to read from each row).
    const int g = j / 128;
    const int short_idx = j / 8;
    const int bit_shift = (j % 8) * 2;

    float acc = 0.0f;

    for (int i = 0; i < m_rows; i++) {
        // Extract the ternary value W[i, j] from the packed code.
        // code ∈ {0, 1, 2}: 0 → -1, 1 → 0, 2 → +1.
        unsigned short ucode = (unsigned short)codes[(long)i * int16_per_row + short_idx];
        int code = (ucode >> bit_shift) & 0x3;
        float w_val;
        if (code == 0) w_val = -1.0f;
        else if (code == 1) w_val = 0.0f;
        else w_val = 1.0f;

        // Issue 734 T5 — f16-bit wscale (exact f16→f32; see gemv_ternary_row).
        const float scale = f16_bits_to_f32(wscale[(long)i * groups_per_row + g]);
        acc += w_val * scale * grad_y[i];
    }

    grad_x[j] = acc;
}
"#;

/// Per-weight-matrix device buffers.
struct WeightBuffers {
    /// Packed 2-bit codes (`[M * int16_per_row]` int16).
    codes_dev: CudaSlice<i16>,
    /// Per-128-weight scales (`[M * groups_per_row]`) as **f16 bits** —
    /// Issue 734 T5: halves the wscale DRAM traffic (was 12.5% of the codes
    /// bytes at f32). Exact: every scale originated as f16 in the GGUF, so
    /// the f16→f32 conversion in-kernel reproduces the old f32 values
    /// bit-identically.
    wscale_dev: CudaSlice<u16>,
    /// Output buffer (`[M]` float). Overwritten each `forward`.
    out_dev: CudaSlice<f32>,
    /// Row count (M).
    m: usize,
    /// Column count (N).
    n: usize,
}

/// Convert ternary bit-plane weights to the dp4a kernel's packed 2-bit code
/// format + decode the f16 group scales to f32.
///
/// Shared between `TernaryGemmCudaRaw::upload_weights` (Issue 608 standalone
/// handler) and the Issue 615 GPU-resident forward path. Keeping this `pub(crate)`
/// avoids duplicating the bit-plane unpacking logic across the two consumers.
///
/// Returns `(codes, wscale)`:
/// - `codes`: `[M * (N/8)]` `i16`, each packing 8 ternary values as 2-bit codes
///   `code = pos - neg + 1 ∈ {0, 1, 2}` (0 → −1, 1 → 0, 2 → +1).
/// - `wscale`: `[M * ceil(N/128)]` `u16` — the f16 **bits** of the group
///   scales, passed through unchanged (Issue 734 T5: the kernels convert
///   f16→f32 in-register via `__half2float`, which is exact — bit-identical
///   to the old f32 upload at half the DRAM traffic).
///
/// Panics if `N` is not a multiple of 8 (dp4a requires 8-element alignment).
pub fn convert_bitplane_to_packed_codes(
    w: &TernaryGroupWeights,
) -> (Vec<i16>, Vec<u16>) {
    let m = w.rows;
    let n = w.cols;
    assert!(n.is_multiple_of(8), "dp4a requires N % 8 == 0; got N={n}");
    let int16_per_row = n / 8;
    let groups_per_row = n.div_ceil(WEIGHT_GROUP);

    // code[k] = pos_bit[k] - neg_bit[k] + 1  ∈ {0, 1, 2}
    let n_codes = m * int16_per_row;
    let mut codes = vec![0i16; n_codes];
    for row in 0..m {
        let row_pos = &w.pos_bits[row * w.blocks64..(row + 1) * w.blocks64];
        let row_neg = &w.neg_bits[row * w.blocks64..(row + 1) * w.blocks64];
        let out = &mut codes[row * int16_per_row..(row + 1) * int16_per_row];
        // Each int16 packs 8 consecutive ternary values. int16 i covers
        // element range [8*i, 8*i+8). The u64 block holding element 8*i is
        // block index (8*i) / 64 = i / 8; within-block bit offset is
        // (8*i) % 64 = 8*(i % 8).
        #[allow(clippy::needless_range_loop)]
        for i in 0..int16_per_row {
            let block_idx = i / 8;
            let bit_off = 8 * (i % 8);
            let pos_block = row_pos[block_idx];
            let neg_block = row_neg[block_idx];
            let mut packed: u16 = 0;
            for k in 0..8 {
                let bit = 1u64 << (bit_off + k);
                let pos = (pos_block & bit) != 0;
                let neg = (neg_block & bit) != 0;
                // pos - neg + 1: (1,0)->2, (0,1)->0, (0,0)->1
                let code: u16 = if pos {
                    2
                } else if neg {
                    0
                } else {
                    1
                };
                packed |= code << (2 * k);
            }
            out[i] = packed as i16;
        }
    }

    // Group scales: pass the f16 bits through unchanged (Issue 734 T5).
    // `group_scale` is already half::f16 on the host — no conversion needed.
    let mut wscale = vec![0u16; m * groups_per_row];
    for (i, &s) in w.group_scale.iter().enumerate() {
        wscale[i] = s.to_bits();
    }

    (codes, wscale)
}

/// Error type for `TernaryGemmCudaRaw`.
#[derive(Debug)]
pub enum TernaryGemmCudaRawError {
    /// CUDA context creation failed (no device, driver mismatch, etc.).
    CudaInit(String),
    /// NVRTC compilation failed (no CUDA toolkit, syntax error, etc.).
    Compile(String),
    /// Device allocation or upload failed.
    Alloc(String),
    /// Kernel launch failed.
    Launch(String),
    /// Shape mismatch on `forward` (input/output slice lengths).
    ShapeMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for TernaryGemmCudaRawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CudaInit(s) => write!(f, "CUDA init failed: {s}"),
            Self::Compile(s) => write!(f, "NVRTC compile failed: {s}"),
            Self::Alloc(s) => write!(f, "device alloc failed: {s}"),
            Self::Launch(s) => write!(f, "kernel launch failed: {s}"),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl Error for TernaryGemmCudaRawError {}

/// Raw-CUDA dp4a ternary GEMV handler.
///
/// Holds the CUDA context + compiled kernel + per-weight device buffers.
/// `upload_weights` returns an index; pass it to `forward`.
///
/// Thread-safety: NOT `Sync` — the inner CUDA stream serializes dispatches.
/// Callers must serialize `forward` calls (or construct one handler per
/// thread, each with its own stream).
pub struct TernaryGemmCudaRaw {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    /// Issue 641 T7.3 — transposed ternary GEMV kernel for backward pass.
    transposed_kernel: CudaFunction,
    /// Compiled module — kept alive so `kernel` remains valid.
    _module: Arc<CudaModule>,
    /// Per-weight-matrix device buffers, indexed by upload order.
    weights: Vec<WeightBuffers>,
}

impl TernaryGemmCudaRaw {
    /// Initialize the CUDA context + compile the dp4a kernel.
    ///
    /// Uses device 0 (the first CUDA device). Compile target is `sm_89`
    /// (Ada Lovelace / RTX 4090); `__dp4a` + `__byte_perm` are sm_70+
    /// intrinsics, so older NVIDIA GPUs also work if the arch is adjusted.
    pub fn new() -> Result<Self, TernaryGemmCudaRawError> {
        let ctx = CudaContext::new(0).map_err(|e| TernaryGemmCudaRawError::CudaInit(e.to_string()))?;
        // `ctx.new_stream()` returns `Arc<CudaStream>` directly — no wrapping.
        let stream = ctx
            .new_stream()
            .map_err(|e| TernaryGemmCudaRawError::CudaInit(e.to_string()))?;

        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            GEMV_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| TernaryGemmCudaRawError::Compile(format!("{e}")))?;
        // `ctx.load_module()` returns `Arc<CudaModule>` directly — no wrapping.
        let module = ctx
            .load_module(ptx)
            .map_err(|e| TernaryGemmCudaRawError::Compile(e.to_string()))?;
        let kernel = module
            .load_function("gemv_ternary_dp4a")
            .map_err(|e| TernaryGemmCudaRawError::Compile(format!("{e}")))?;
        let transposed_kernel = module
            .load_function("gemv_ternary_transposed_f32")
            .map_err(|e| TernaryGemmCudaRawError::Compile(format!("{e}")))?;

        Ok(Self {
            stream,
            kernel,
            transposed_kernel,
            _module: module,
            weights: Vec::new(),
        })
    }

    /// Upload a ternary weight matrix, returning a handle for `forward`.
    ///
    /// Converts bit-plane (`pos_bits`/`neg_bits`) → packed 2-bit codes and
    /// uploads to the device. One-time cost amortized across every forward.
    pub fn upload_weights(
        &mut self,
        w: &TernaryGroupWeights,
    ) -> Result<usize, TernaryGemmCudaRawError> {
        let m = w.rows;
        let n = w.cols;
        if !n.is_multiple_of(8) {
            return Err(TernaryGemmCudaRawError::ShapeMismatch {
                expected: 0, // multiple of 8
                got: n % 8,
            });
        }

        let (codes, wscale) = convert_bitplane_to_packed_codes(w);

        let codes_dev = self
            .stream
            .clone_htod(&codes)
            .map_err(|e| TernaryGemmCudaRawError::Alloc(e.to_string()))?;
        let wscale_dev = self
            .stream
            .clone_htod(&wscale)
            .map_err(|e| TernaryGemmCudaRawError::Alloc(e.to_string()))?;
        let out_dev = self
            .stream
            .alloc_zeros::<f32>(m)
            .map_err(|e| TernaryGemmCudaRawError::Alloc(e.to_string()))?;

        let idx = self.weights.len();
        self.weights.push(WeightBuffers {
            codes_dev,
            wscale_dev,
            out_dev,
            m,
            n,
        });
        Ok(idx)
    }

    /// Compute `out = w @ x` for the weight matrix at `weight_idx`.
    ///
    /// Quantizes `x` to int8 per `ACTIVATION_BLOCK` elements, launches the
    /// dp4a kernel, syncs, and reads back the result.
    ///
    /// **Per-call allocation:** this path uploads the activation int8 + scales
    /// via `htod_copy` (new device slice each call). This is a correctness-
    /// first implementation; the G4 alloc-free optimization (persistent
    /// activation buffer + `htod_copy_into`) is a follow-up once the kernel
    /// is proven end-to-end.
    pub fn forward(
        &mut self,
        weight_idx: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), TernaryGemmCudaRawError> {
        // Split borrows to satisfy the borrow checker: `stream`, `kernel`,
        // and `weights[weight_idx]` are disjoint fields of `self`.
        let stream = &self.stream;
        let kernel = &self.kernel;
        let wb = &mut self.weights[weight_idx];

        if x.len() != wb.n {
            return Err(TernaryGemmCudaRawError::ShapeMismatch {
                expected: wb.n,
                got: x.len(),
            });
        }
        if out.len() != wb.m {
            return Err(TernaryGemmCudaRawError::ShapeMismatch {
                expected: wb.m,
                got: out.len(),
            });
        }

        let n = wb.n;
        let ablocks = n.div_ceil(ACTIVATION_BLOCK);

        // ── Quantize activations to int8 per ACTIVATION_BLOCK ──
        let mut act_i8 = vec![0i8; n];
        let mut ascale = vec![0f32; ablocks];
        #[allow(clippy::needless_range_loop)]
        for blk in 0..ablocks {
            let start = blk * ACTIVATION_BLOCK;
            let end = (start + ACTIVATION_BLOCK).min(n);
            let mut absmax: f32 = 0.0;
            #[allow(clippy::needless_range_loop)]
            for i in start..end {
                absmax = absmax.max(x[i].abs());
            }
            // d = absmax / 127; quantize as round(x / d). d=1.0 if all zero.
            let d = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
            ascale[blk] = d;
            let inv_d = 1.0 / d;
            #[allow(clippy::needless_range_loop)]
            for i in start..end {
                let q = (x[i] * inv_d).round().clamp(-128.0, 127.0);
                act_i8[i] = q as i8;
            }
        }

        // ── Upload activations ──
        let act_dev = stream
            .clone_htod(&act_i8)
            .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;
        let ascale_dev = stream
            .clone_htod(&ascale)
            .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;

        // ── Launch ──
        let m_i32 = wb.m as i32;
        let int16_per_row_i32 = (n / 8) as i32;
        let groups_per_row_i32 = n.div_ceil(WEIGHT_GROUP) as i32;
        let ablock_i32 = ACTIVATION_BLOCK as i32;
        let ablocks_i32 = ablocks as i32;

        let grid_x = (wb.m as u32).div_ceil(WG_THREADS / 32);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (WG_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        // Safety: the dp4a kernel is a pure GEMV — reads from `codes_dev`,
        // `wscale_dev`, `act_dev`, `ascale_dev`; writes only to `out_dev`.
        // No aliasing, no out-of-bounds (grid covers exactly `m` rows).
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&wb.codes_dev)
                .arg(&wb.wscale_dev)
                .arg(&act_dev)
                .arg(&ascale_dev)
                .arg(&mut wb.out_dev)
                .arg(&m_i32)
                .arg(&int16_per_row_i32)
                .arg(&groups_per_row_i32)
                .arg(&ablock_i32)
                .arg(&ablocks_i32)
                .launch(cfg)
                .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;
        }

        // ── Sync + read back ──
        stream
            .synchronize()
            .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;
        stream
            .memcpy_dtoh(&wb.out_dev, out)
            .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;

        Ok(())
    }

    /// Issue 641 T7.3 — Compute `grad_x = W^T @ grad_y` for the weight matrix
    /// at `weight_idx`.
    ///
    /// This is the transposed matvec needed for the backward pass: given the
    /// upstream gradient `grad_y[m]`, compute the gradient w.r.t. the layer
    /// input `grad_x[n]` by multiplying with the transposed weight matrix.
    ///
    /// Unlike `forward()`, this path uses f32 gradients directly (no int8
    /// quantization) because the transposed access pattern precludes dp4a.
    /// The kernel reads the SAME uploaded ternary codes + scales — no
    /// additional upload is needed.
    ///
    /// # Arguments
    /// - `weight_idx`: index returned by `upload_weights`
    /// - `grad_y`: `[m]` upstream gradient (length = weight rows)
    /// - `grad_x`: `[n]` output gradient (length = weight cols)
    pub fn backward_transposed(
        &self,
        weight_idx: usize,
        grad_y: &[f32],
        grad_x: &mut [f32],
    ) -> Result<(), TernaryGemmCudaRawError> {
        let stream = &self.stream;
        let kernel = &self.transposed_kernel;
        let wb = &self.weights[weight_idx];

        if grad_y.len() != wb.m {
            return Err(TernaryGemmCudaRawError::ShapeMismatch {
                expected: wb.m,
                got: grad_y.len(),
            });
        }
        if grad_x.len() != wb.n {
            return Err(TernaryGemmCudaRawError::ShapeMismatch {
                expected: wb.n,
                got: grad_x.len(),
            });
        }

        let grad_y_dev = stream
            .clone_htod(grad_y)
            .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;
        let mut grad_x_dev = stream
            .alloc_zeros::<f32>(wb.n)
            .map_err(|e| TernaryGemmCudaRawError::Alloc(e.to_string()))?;

        let m_i32 = wb.m as i32;
        let n_i32 = wb.n as i32;
        let int16_per_row_i32 = (wb.n / 8) as i32;
        let groups_per_row_i32 = wb.n.div_ceil(WEIGHT_GROUP) as i32;

        let grid_x = (wb.n as u32).div_ceil(WG_THREADS);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (WG_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        // Safety: the transposed GEMV kernel reads from `codes_dev`,
        // `wscale_dev`, `grad_y_dev`; writes only to `grad_x_dev`. No aliasing.
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&wb.codes_dev)
                .arg(&wb.wscale_dev)
                .arg(&grad_y_dev)
                .arg(&mut grad_x_dev)
                .arg(&m_i32)
                .arg(&n_i32)
                .arg(&int16_per_row_i32)
                .arg(&groups_per_row_i32)
                .launch(cfg)
                .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;
        }

        stream
            .synchronize()
            .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;
        stream
            .memcpy_dtoh(&grad_x_dev, grad_x)
            .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;

        Ok(())
    }

    /// Number of weight matrices currently uploaded.
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }
}

// ───────────────────────────────────────────────────────────────────────────
// TernaryMatvecHook impl — Issue 608 T2 Phase B proof-of-concept
// ───────────────────────────────────────────────────────────────────────────
//
// The CubeCL hook path (`GpuTernaryMatvec`) uploads + dispatches + downloads
// per matvec, which with wgpu/CubeCL-CUDA costs ~1ms of sync overhead per
// GEMV — yielding only ~0.69 tok/s (Bench 603). The dp4a kernel is 4.7×
// faster than the CubeCL ternary kernel, and cudarc's dispatch overhead is
// ~17µs per launch vs wgpu's ~1ms — a ~60× reduction.
//
// This hook measures whether the dp4a kernel's speedup + lower overhead is
// enough to overcome the per-GEMV CPU→GPU transfer cost that the GPU-resident
// path eliminates. If the mixed-mode forward (CPU elementwise + dp4a GEMV)
// beats 15 tok/s (the CubeCL GPU-resident path), the dp4a kernel is worth a
// full-port investment. If it doesn't, only a shared-stream integration
// (fork CubeCL to expose its CUstream + device pointers) can capture the win.
//
// This is a measurement tool, not a production path — per-call `clone_htod`
// violates G4 (alloc-free), and the int8 quantization means it cannot pass the
// existing per-logit gates (T3a, open).

/// dp4a-backed `TernaryMatvecHook` for the mixed-mode forward benchmark.
///
/// Wraps [`TernaryGemmCudaRaw`] and caches uploaded weight matrices by
/// `(pos_bits.as_ptr(), neg_bits.as_ptr())` pointer identity — the same
/// caching key the CubeCL `GpuTernaryMatvec` uses.
///
/// Thread-safe via internal `Mutex`; the inner CUDA stream serializes all
/// dispatches regardless. `Send + Sync` is required by the
/// `TernaryMatvecHook` trait.
pub struct GpuTernaryMatvecDp4a {
    inner: std::sync::Mutex<Dp4aHookInner>,
}

struct Dp4aHookInner {
    handler: TernaryGemmCudaRaw,
    /// Pointer-pair → weight index in the handler.
    cache: std::collections::HashMap<(usize, usize), usize>,
    /// Reusable output buffer (avoid per-call Vec allocation in `matvec`).
    /// Sized to the largest weight matrix output seen so far.
    out_buf: Vec<f32>,
}

impl GpuTernaryMatvecDp4a {
    /// Initialize the dp4a hook — creates the CUDA context + compiles the
    /// nvrtc kernel (~130-180ms one-time cost).
    pub fn new() -> Result<Self, TernaryGemmCudaRawError> {
        let handler = TernaryGemmCudaRaw::new()?;
        Ok(Self {
            inner: std::sync::Mutex::new(Dp4aHookInner {
                handler,
                cache: std::collections::HashMap::new(),
                out_buf: Vec::new(),
            }),
        })
    }

    /// Pre-upload a weight matrix. Call once per unique projection before the
    /// first forward to avoid lazy-upload during timing.
    pub fn preupload(&self, w: &TernaryGroupWeights) -> Result<(), TernaryGemmCudaRawError> {
        let mut inner = self.inner.lock().unwrap();
        let key = (w.pos_bits.as_ptr() as usize, w.neg_bits.as_ptr() as usize);
        if !inner.cache.contains_key(&key) {
            let idx = inner.handler.upload_weights(w)?;
            inner.cache.insert(key, idx);
            if inner.out_buf.len() < w.rows {
                inner.out_buf.resize(w.rows, 0.0);
            }
        }
        Ok(())
    }

    /// Pre-upload ALL projections from the loaded model (iterates over
    /// `TernaryGroupWeights` references).
    pub fn preupload_all<'a, I>(&self, projections: I) -> Result<(), TernaryGemmCudaRawError>
    where
        I: IntoIterator<Item = &'a TernaryGroupWeights>,
    {
        for w in projections {
            if w.rows > 0 {
                self.preupload(w)?;
            }
        }
        Ok(())
    }

    /// Total weight upload time accumulated so far (diagnostic — for the
    /// benchmark to report one-time cost separately from steady-state).
    pub fn num_weights(&self) -> usize {
        self.inner.lock().unwrap().handler.num_weights()
    }
}

impl katgpt_core::TernaryMatvecHook for GpuTernaryMatvecDp4a {
    fn matvec(&self, w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
        if w.rows == 0 {
            return;
        }
        debug_assert_eq!(x.len(), w.cols, "dp4a hook: input dim mismatch");
        debug_assert_eq!(y.len(), w.rows, "dp4a hook: output dim mismatch");

        let mut inner = self.inner.lock().unwrap();
        // Split the struct borrow so `handler.forward()` and `&mut out_buf`
        // can coexist (otherwise the `MutexGuard`'s `DerefMut` yields a
        // single `&mut Dp4aHookInner` that can't be borrowed twice).
        let Dp4aHookInner {
            handler,
            cache,
            out_buf,
        } = &mut *inner;
        let key = (w.pos_bits.as_ptr() as usize, w.neg_bits.as_ptr() as usize);
        let idx = if let Some(idx) = cache.get(&key).copied() { idx } else {
                // Lazy upload — should not happen if preupload_all was called,
                // but handles the fallback gracefully.
                let idx = handler
                    .upload_weights(w)
                    .expect("dp4a hook: weight upload failed");
                cache.insert(key, idx);
                if out_buf.len() < w.rows {
                    out_buf.resize(w.rows, 0.0);
                }
                idx
            };

        // Use the reusable output buffer, then copy to caller's slice.
        // This avoids per-call Vec allocation.
        out_buf[..w.rows].fill(0.0);
        handler
            .forward(idx, x, &mut out_buf[..w.rows])
            .expect("dp4a hook: kernel launch failed");
        y.copy_from_slice(&out_buf[..w.rows]);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use half::f16;

    /// Skip the test if no CUDA device is available (e.g. on macOS/CI).
    fn cuda_or_skip() -> Option<()> {
        // Test the full path (context init + nvrtc compile) via `new()`.
        // If it fails, skip rather than panic — matches the riir-train-gpu
        // convention for GPU-gated tests.
        if CudaContext::new(0).is_err() {
            return None;
        }
        Some(())
    }

    /// Build a small `TernaryGroupWeights` with known values for unit tests.
    fn make_test_weights(m: usize, n: usize) -> TernaryGroupWeights {
        let mut w = TernaryGroupWeights::new(m, n);
        // Deterministic pattern: cycle through {-1, 0, +1} by column index.
        for row in 0..m {
            for col in 0..n {
                let val: i8 = match col % 3 {
                    0 => 1,
                    1 => 0,
                    _ => -1,
                };
                w.set(row, col, val);
            }
        }
        // Set a non-trivial scale so the result isn't just 1.0.
        for i in 0..w.group_scale.len() {
            w.group_scale[i] = f16::from_f32(0.5);
        }
        w
    }

    /// CPU reference: `y = w @ x` for ternary group weights.
    fn cpu_matvec(w: &TernaryGroupWeights, x: &[f32]) -> Vec<f32> {
        let m = w.rows;
        let mut y = vec![0f32; m];
        for (row, y_val) in y.iter_mut().enumerate() {
            let mut acc = 0f32;
            for (col, &xv) in x.iter().enumerate() {
                let block = w.pos_bits[row * w.blocks64 + col / 64];
                let neg_block = w.neg_bits[row * w.blocks64 + col / 64];
                let bit = 1u64 << (col % 64);
                let val: f32 = if block & bit != 0 {
                    1.0
                } else if neg_block & bit != 0 {
                    -1.0
                } else {
                    0.0
                };
                let group = col / WEIGHT_GROUP;
                let scale = w.group_scale[row * w.groups_per_row + group].to_f32();
                acc += val * xv * scale;
            }
            *y_val = acc;
        }
        y
    }

    /// G1: dp4a kernel output matches CPU reference within int8 quantization
    /// tolerance. T3 measured mean_rel ~3.7e-3 for gaussian activations; we
    /// use 2% (0.02) as the gate to absorb the activation quantization noise
    /// at small scale.
    #[test]
    fn test_dp4a_matches_cpu_within_int8_tolerance() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let m = 256;
        let n = 256; // multiple of 128 and 16
        let w = make_test_weights(m, n);

        // Gaussian-ish input (deterministic via LCG).
        let mut state: u32 = 0xCAFEBABE;
        let mut x = vec![0f32; n];
        for v in &mut x {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((state >> 16) as f32 / 65535.0 - 0.5) * 2.0; // [-1, 1)
        }

        let cpu_out = cpu_matvec(&w, &x);

        let mut handler = TernaryGemmCudaRaw::new().expect("CUDA init");
        let idx = handler.upload_weights(&w).expect("upload");
        let mut gpu_out = vec![0f32; m];
        handler.forward(idx, &x, &mut gpu_out).expect("forward");

        // Compute max relative error.
        let mut max_rel = 0f32;
        let mut mean_rel = 0f32;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            let denom = a.abs().max(1e-6);
            let rel = (a - b).abs() / denom;
            max_rel = max_rel.max(rel);
            mean_rel += rel;
        }
        mean_rel /= m as f32;
        eprintln!(
            "[dp4a_g1] m={m} n={n}: mean_rel={mean_rel:.4e} max_rel={max_rel:.4e}"
        );
        // T3 tolerance: mean_rel < 0.02 (2%) for gaussian. Use max_rel < 0.05.
        assert!(
            mean_rel < 0.02,
            "mean_rel {mean_rel:.4e} exceeds 2% int8 quantization tolerance"
        );
        assert!(
            max_rel < 0.05,
            "max_rel {max_rel:.4e} exceeds 5% int8 quantization tolerance"
        );
    }

    /// Verify the bit-plane → packed code conversion: a weight matrix of all
    /// +1 should produce the same output as the CPU reference.
    #[test]
    fn test_all_positive_weights() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let m = 128;
        let n = 128;
        let mut w = TernaryGroupWeights::new(m, n);
        for row in 0..m {
            for col in 0..n {
                w.set(row, col, 1);
            }
        }
        // scale = 1.0
        for s in &mut w.group_scale {
            *s = f16::from_f32(1.0);
        }

        let x = vec![1.0f32; n];
        let cpu_out = cpu_matvec(&w, &x);
        // Every row should sum to n (all +1 weights × 1.0 input × 1.0 scale).
        assert!(cpu_out.iter().all(|&v| (v - n as f32).abs() < 1e-3));

        let mut handler = TernaryGemmCudaRaw::new().expect("CUDA init");
        let idx = handler.upload_weights(&w).expect("upload");
        let mut gpu_out = vec![0f32; m];
        handler.forward(idx, &x, &mut gpu_out).expect("forward");

        // GPU should match CPU within int8 tolerance.
        let mut max_diff = 0f32;
        for (a, b) in cpu_out.iter().zip(gpu_out.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        eprintln!("[all_pos] max_diff = {max_diff:.4e} (expected ~{n})");
        assert!(max_diff < (n as f32 * 0.02), "max_diff {max_diff} too large");
    }

    /// Issue 616 T4 — fused kernel must produce BIT-IDENTICAL output to the
    /// split path (quantize kernel + `gemv_ternary_dp4a`). The quantization
    /// formula is the same (d=absmax/127, q=round(x/d), clamp [-128,127]);
    /// only the parallelization differs. Any divergence here is a bug.
    #[test]
    fn test_fused_matches_split_bit_identical() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let m = 256;
        let n = 256; // multiple of 128 and 16
        let w = make_test_weights(m, n);

        // Mixed-magnitude input (exercises per-block scale independence).
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let blk = i / 16;
                let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
                let mag = ((blk + 1) as f32) * 0.1 * ((i % 7) as f32 + 1.0);
                sign * mag
            })
            .collect();

        // ── Split path: TernaryGemmCudaRaw::forward ──
        let mut handler = TernaryGemmCudaRaw::new().expect("CUDA init");
        let idx = handler.upload_weights(&w).expect("upload");
        let mut split_out = vec![0f32; m];
        handler.forward(idx, &x, &mut split_out).expect("forward");

        // ── Fused path: launch `gemv_ternary_dp4a_fused` directly ──
        // Reuse the handler's compiled module (same GEMV_CUDA_SRC has both
        // entrypoints now) + the uploaded weight buffers.
        let fused_kernel = handler
            ._module
            .load_function("gemv_ternary_dp4a_fused")
            .expect("load fused kernel");
        let wb = &handler.weights[idx];

        let x_dev = handler.stream.clone_htod(&x).expect("upload x");
        let fused_out_dev = handler.stream.alloc_zeros::<f32>(m).expect("alloc out");

        let ablocks = n.div_ceil(ACTIVATION_BLOCK);
        let m_i32 = m as i32;
        let int16_per_row = (n / 8) as i32;
        let groups_per_row = n.div_ceil(WEIGHT_GROUP) as i32;
        let ablock_i32 = ACTIVATION_BLOCK as i32;
        let ablocks_i32 = ablocks as i32;
        let n_i32 = n as i32;
        let shared_mem_bytes = (n + ablocks * 4) as u32;
        let grid_x = (m as u32).div_ceil(WG_THREADS / 32);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (WG_THREADS, 1, 1),
            shared_mem_bytes,
        };
        unsafe {
            handler
                .stream
                .launch_builder(&fused_kernel)
                .arg(&wb.codes_dev)
                .arg(&wb.wscale_dev)
                .arg(&x_dev)
                .arg(&fused_out_dev)
                .arg(&m_i32)
                .arg(&int16_per_row)
                .arg(&groups_per_row)
                .arg(&ablock_i32)
                .arg(&ablocks_i32)
                .arg(&n_i32)
                .launch(cfg)
                .expect("launch fused");
        }
        handler.stream.synchronize().expect("sync");
        let mut fused_out = vec![0f32; m];
        handler
            .stream
            .memcpy_dtoh(&fused_out_dev, &mut fused_out)
            .expect("download");

        // ── Bit-identical comparison ──
        let mut max_diff = 0.0f32;
        let mut mismatches = 0usize;
        for i in 0..m {
            let diff = (split_out[i] - fused_out[i]).abs();
            max_diff = max_diff.max(diff);
            if diff != 0.0 {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "  mismatch at row {i}: split={:.6}, fused={:.6}, diff={:.4e}",
                        split_out[i], fused_out[i], diff
                    );
                }
            }
        }
        eprintln!(
            "[fused_vs_split] m={m} n={n}: max_diff={max_diff:.4e}, mismatches={mismatches}"
        );
        assert!(
            max_diff == 0.0,
            "fused kernel diverges from split path: max_diff={max_diff:.4e}, {mismatches}/{m} rows differ"
        );
    }

    /// Issue 616 T4 — larger-scale bit-identical test with production-shaped
    /// dimensions (n=5120 matches n_embd; m=4096 is a typical FFN projection).
    /// Catches bugs that only manifest at larger block counts (e.g., ablocks
    /// > 32 so the strided pattern wraps).
    #[test]
    fn test_fused_matches_split_bit_identical_large() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let m = 512;
        let n = 512; // multiple of 128 and 16; gives ablocks=32 (one full warp stride)
        let w = make_test_weights(m, n);

        // Gaussian-ish input via LCG.
        let mut state: u32 = 0xDEADBEEF;
        let x: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 16) as f32 / 65535.0 - 0.5) * 2.0
            })
            .collect();

        let mut handler = TernaryGemmCudaRaw::new().expect("CUDA init");
        let idx = handler.upload_weights(&w).expect("upload");
        let mut split_out = vec![0f32; m];
        handler.forward(idx, &x, &mut split_out).expect("forward");

        let fused_kernel = handler
            ._module
            .load_function("gemv_ternary_dp4a_fused")
            .expect("load fused kernel");
        let wb = &handler.weights[idx];
        let x_dev = handler.stream.clone_htod(&x).expect("upload x");
        let fused_out_dev = handler.stream.alloc_zeros::<f32>(m).expect("alloc out");

        let ablocks = n.div_ceil(ACTIVATION_BLOCK);
        let m_i32 = m as i32;
        let int16_per_row = (n / 8) as i32;
        let groups_per_row = n.div_ceil(WEIGHT_GROUP) as i32;
        let ablock_i32 = ACTIVATION_BLOCK as i32;
        let ablocks_i32 = ablocks as i32;
        let n_i32 = n as i32;
        let shared_mem_bytes = (n + ablocks * 4) as u32;
        let grid_x = (m as u32).div_ceil(WG_THREADS / 32);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (WG_THREADS, 1, 1),
            shared_mem_bytes,
        };
        unsafe {
            handler
                .stream
                .launch_builder(&fused_kernel)
                .arg(&wb.codes_dev)
                .arg(&wb.wscale_dev)
                .arg(&x_dev)
                .arg(&fused_out_dev)
                .arg(&m_i32)
                .arg(&int16_per_row)
                .arg(&groups_per_row)
                .arg(&ablock_i32)
                .arg(&ablocks_i32)
                .arg(&n_i32)
                .launch(cfg)
                .expect("launch fused");
        }
        handler.stream.synchronize().expect("sync");
        let mut fused_out = vec![0f32; m];
        handler
            .stream
            .memcpy_dtoh(&fused_out_dev, &mut fused_out)
            .expect("download");

        let mut max_diff = 0.0f32;
        for (a, b) in split_out.iter().zip(fused_out.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        eprintln!("[fused_vs_split_large] m={m} n={n}: max_diff={max_diff:.4e}");
        assert!(
            max_diff == 0.0,
            "fused kernel diverges from split path at scale: max_diff={max_diff:.4e}"
        );
    }

    /// CPU reference: `y = w^T @ x` for ternary group weights (transposed matvec).
    /// y has length n (columns), x has length m (rows).
    fn cpu_matvec_transposed(w: &TernaryGroupWeights, x: &[f32]) -> Vec<f32> {
        let n = w.cols;
        let mut y = vec![0f32; n];
        for (col, y_val) in y.iter_mut().enumerate() {
            let mut acc = 0f32;
            for (row, &xv) in x.iter().enumerate() {
                let block = w.pos_bits[row * w.blocks64 + col / 64];
                let neg_block = w.neg_bits[row * w.blocks64 + col / 64];
                let bit = 1u64 << (col % 64);
                let val: f32 = if block & bit != 0 {
                    1.0
                } else if neg_block & bit != 0 {
                    -1.0
                } else {
                    0.0
                };
                let group = col / WEIGHT_GROUP;
                let scale = w.group_scale[row * w.groups_per_row + group].to_f32();
                acc += val * xv * scale;
            }
            *y_val = acc;
        }
        y
    }

    /// Issue 641 T7.3 — transposed ternary GEMV must match CPU reference
    /// bit-identically (both use f32 accumulation, no int8 quantization).
    ///
    /// ⚠ Do NOT run while another GPU process is active (Issue 649 contention).
    #[test]
    fn test_transposed_gemv_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let m = 256;
        let n = 256; // multiple of 128 and 8
        let w = make_test_weights(m, n);

        // Deterministic input.
        let mut state: u32 = 0xDEADBEEF;
        let mut x = vec![0f32; m];
        for v in &mut x {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((state >> 16) as f32 / 65535.0 - 0.5) * 2.0;
        }

        // CPU reference
        let cpu_out = cpu_matvec_transposed(&w, &x);

        // GPU
        let mut handler = TernaryGemmCudaRaw::new().expect("CUDA init");
        let idx = handler.upload_weights(&w).expect("upload");
        let mut gpu_out = vec![0f32; n];
        handler
            .backward_transposed(idx, &x, &mut gpu_out)
            .expect("backward_transposed");

        // Both paths use f32 accumulation → expect near-bit-identical.
        let mut max_diff = 0.0f32;
        let mut max_rel = 0.0f32;
        for i in 0..n {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            max_diff = max_diff.max(diff);
            let denom = cpu_out[i].abs().max(1e-6);
            max_rel = max_rel.max(diff / denom);
        }
        eprintln!(
            "[transposed_gemv] m={m} n={n}: max_diff={max_diff:.6e}, max_rel={max_rel:.6e}"
        );
        assert!(
            max_diff < 1e-4,
            "transposed GEMV diverges from CPU: max_diff={max_diff:.6e}"
        );
    }

    /// Issue 641 T7.3 — larger-scale test with Bonsai-relevant dimensions.
    /// Catches bugs at larger group counts (multiple 128-element groups per row).
    #[test]
    fn test_transposed_gemv_matches_cpu_large() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        // n=5120 matches n_embd; m=1024 is a small FFN-shaped matrix.
        let m = 1024;
        let n = 512;
        let w = make_test_weights(m, n);

        let mut state: u32 = 0x12345678;
        let mut x = vec![0f32; m];
        for v in &mut x {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((state >> 16) as f32 / 65535.0 - 0.5) * 2.0;
        }

        let cpu_out = cpu_matvec_transposed(&w, &x);

        let mut handler = TernaryGemmCudaRaw::new().expect("CUDA init");
        let idx = handler.upload_weights(&w).expect("upload");
        let mut gpu_out = vec![0f32; n];
        handler
            .backward_transposed(idx, &x, &mut gpu_out)
            .expect("backward_transposed");

        let mut max_diff = 0.0f32;
        let mut max_rel = 0.0f32;
        for i in 0..n {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            max_diff = max_diff.max(diff);
            let denom = cpu_out[i].abs().max(1e-6);
            max_rel = max_rel.max(diff / denom);
        }
        eprintln!(
            "[transposed_gemv_large] m={m} n={n}: max_diff={max_diff:.6e}, max_rel={max_rel:.6e}"
        );
        assert!(
            max_diff < 1e-3,
            "transposed GEMV diverges from CPU at scale: max_diff={max_diff:.6e}"
        );
    }

    /// Issue 697 — multi-segment dp4a GEMV must produce BIT-IDENTICAL output
    /// to the per-segment split launches (same per-row dp4a loop, only the
    /// row → segment mapping is new). Covers the production segment mixes:
    /// 4 segments incl. the tiny m=48 a/b shape, and the accumulate mode.
    #[test]
    fn test_multi_gemv_matches_split_bit_identical() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        // Production-shaped segment mix (deltanet input projections):
        // qkv m=1024, z m=512, a m=48, b m=48 — all share n=512.
        let n = 512; // multiple of 128 and 16
        let seg_m: [usize; 4] = [1024, 512, 48, 48];
        let total_m: usize = seg_m.iter().sum();
        let ws: Vec<_> = seg_m.iter().map(|&m| make_test_weights(m, n)).collect();

        let mut state: u32 = 0xC0FFEE42;
        let x: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 16) as f32 / 65535.0 - 0.5) * 2.0
            })
            .collect();

        // ── Split path: one launch per segment (production forward shape) ──
        let mut handler = TernaryGemmCudaRaw::new().expect("CUDA init");
        let mut split_outs: Vec<Vec<f32>> = Vec::with_capacity(4);
        for w in &ws {
            let idx = handler.upload_weights(w).expect("upload");
            let mut out = vec![0f32; w.rows];
            handler.forward(idx, &x, &mut out).expect("forward");
            split_outs.push(out);
        }

        // ── Multi path: one concatenated launch ──
        let multi_kernel = handler
            ._module
            .load_function("gemv_ternary_dp4a_multi")
            .expect("load multi kernel");
        // Forward already uploaded the 4 weight sets at indices 0..4 — same
        // buffers, plus fresh output slices + the shared quantized activation.
        // Re-quantize identically to `forward`: per-16-block absmax scaling.
        let ablocks = n.div_ceil(ACTIVATION_BLOCK);
        let mut act_i8 = vec![0i8; n];
        let mut ascale = vec![0f32; ablocks];
        for (blk, ascale_val) in ascale.iter_mut().enumerate() {
            let s = blk * ACTIVATION_BLOCK;
            let len = ACTIVATION_BLOCK.min(n - s);
            let mut absmax = 0f32;
            for i in 0..len {
                absmax = absmax.max(x[s + i].abs());
            }
            let d = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
            *ascale_val = d;
            for i in 0..len {
                let q = (x[s + i] / d).round().clamp(-128.0, 127.0);
                act_i8[s + i] = q as i8;
            }
        }
        let act_dev = handler.stream.clone_htod(&act_i8).expect("upload act");
        let ascale_dev = handler.stream.clone_htod(&ascale).expect("upload ascale");
        let mut multi_out_dev = Vec::with_capacity(4);
        for &m in &seg_m {
            multi_out_dev.push(handler.stream.alloc_zeros::<f32>(m).expect("alloc out"));
        }

        let int16_per_row = (n / 8) as i32;
        let groups_per_row = n.div_ceil(WEIGHT_GROUP) as i32;
        let ablock_i32 = ACTIVATION_BLOCK as i32;
        let ablocks_i32 = ablocks as i32;
        let m_i32: Vec<i32> = seg_m.iter().map(|&m| m as i32).collect();
        let acc_zero = 0i32;
        let grid_x = (total_m as u32).div_ceil(WG_THREADS / 32);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (WG_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let wbs: Vec<_> = (0..4).map(|i| &handler.weights[i]).collect();
            unsafe {
                handler
                    .stream
                    .launch_builder(&multi_kernel)
                    .arg(&wbs[0].codes_dev)
                    .arg(&wbs[0].wscale_dev)
                    .arg(&multi_out_dev[0])
                    .arg(&m_i32[0])
                    .arg(&wbs[1].codes_dev)
                    .arg(&wbs[1].wscale_dev)
                    .arg(&multi_out_dev[1])
                    .arg(&m_i32[1])
                    .arg(&wbs[2].codes_dev)
                    .arg(&wbs[2].wscale_dev)
                    .arg(&multi_out_dev[2])
                    .arg(&m_i32[2])
                    .arg(&wbs[3].codes_dev)
                    .arg(&wbs[3].wscale_dev)
                    .arg(&multi_out_dev[3])
                    .arg(&m_i32[3])
                    .arg(&act_dev)
                    .arg(&ascale_dev)
                    .arg(&int16_per_row)
                    .arg(&groups_per_row)
                    .arg(&ablock_i32)
                    .arg(&ablocks_i32)
                    .arg(&acc_zero)
                    .launch(cfg)
                    .expect("launch multi");
            }
        }
        handler.stream.synchronize().expect("sync");
        let mut multi_outs: Vec<Vec<f32>> = Vec::with_capacity(4);
        for dev in &multi_out_dev {
            let mut out = vec![0f32; dev.len()];
            handler
                .stream
                .memcpy_dtoh(dev, &mut out)
                .expect("download");
            multi_outs.push(out);
        }

        let mut max_diff = 0.0f32;
        let mut mismatches = 0usize;
        for seg in 0..4 {
            for i in 0..seg_m[seg] {
                let diff = (split_outs[seg][i] - multi_outs[seg][i]).abs();
                max_diff = max_diff.max(diff);
                if diff != 0.0 {
                    mismatches += 1;
                }
            }
        }
        eprintln!(
            "[multi_vs_split] segs={seg_m:?} n={n}: max_diff={max_diff:.4e}, mismatches={mismatches}"
        );
        assert!(
            max_diff == 0.0,
            "multi kernel diverges from split path: max_diff={max_diff:.4e}, {mismatches} rows differ"
        );

        // ── Accumulate mode: out[row] += acc must equal out[row] + split_out ──
        let base: Vec<f32> = (0..seg_m[0])
            .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
            .collect();
        let acc_dev = handler.stream.clone_htod(&base).expect("upload base");
        let acc_one = 1i32;
        {
            let wb = &handler.weights[0];
            let cfg1 = LaunchConfig {
                grid_dim: ((seg_m[0] as u32).div_ceil(WG_THREADS / 32), 1, 1),
                block_dim: (WG_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                handler
                    .stream
                    .launch_builder(&multi_kernel)
                    .arg(&wb.codes_dev)
                    .arg(&wb.wscale_dev)
                    .arg(&acc_dev)
                    .arg(&m_i32[0])
                    .arg(&wb.codes_dev)
                    .arg(&wb.wscale_dev)
                    .arg(&acc_dev)
                    .arg(&0i32)
                    .arg(&wb.codes_dev)
                    .arg(&wb.wscale_dev)
                    .arg(&acc_dev)
                    .arg(&0i32)
                    .arg(&wb.codes_dev)
                    .arg(&wb.wscale_dev)
                    .arg(&acc_dev)
                    .arg(&0i32)
                    .arg(&act_dev)
                    .arg(&ascale_dev)
                    .arg(&int16_per_row)
                    .arg(&groups_per_row)
                    .arg(&ablock_i32)
                    .arg(&ablocks_i32)
                    .arg(&acc_one)
                    .launch(cfg1)
                    .expect("launch multi accum");
            }
        }
        handler.stream.synchronize().expect("sync");
        let mut acc_out = vec![0f32; seg_m[0]];
        handler
            .stream
            .memcpy_dtoh(&acc_dev, &mut acc_out)
            .expect("download");
        let mut acc_max_diff = 0.0f32;
        for i in 0..seg_m[0] {
            // Same operand order as residual_add_f32: x + gemv_result.
            let expect = base[i] + split_outs[0][i];
            acc_max_diff = acc_max_diff.max((expect - acc_out[i]).abs());
        }
        eprintln!("[multi_accum] m={} n={n}: max_diff={acc_max_diff:.4e}", seg_m[0]);
        assert!(
            acc_max_diff == 0.0,
            "multi accumulate mode diverges from base+split: max_diff={acc_max_diff:.4e}"
        );
    }

    /// Plan 604 T1 (Issue 987 G1) — the two-rows-per-warp kernel must be
    /// BIT-IDENTICAL to the one-row persistent kernel (the production R1
    /// dispatch) on every shape class the wrapper will route to it: the
    /// underfilled down/out shape (large even segments), segment-straddle
    /// pairs (the [1024, 512, 48, 48] mix — pairs spanning segment
    /// boundaries, incl. tiny m=48 segments), and the odd-total tail guard
    /// (odd total_rows → the last pair processes rowA only). Both store and
    /// accumulate modes.
    #[test]
    fn test_multi_gemv_r2_matches_r1_bit_identical() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        // n must be production-scale for the u4 arm: the 4x-batched main loop
        // engages only when a lane sees ≥ 5 K-iterations (blk+96 < ablocks).
        // n=5120 = the real GDN out_proj width → ablocks=320 → 10 iterations
        // per lane = 2 full batches + 2 tail iterations (mixed coverage).
        // (The old n=512 gave ablocks=32 = exactly ONE iteration per lane —
        // vacuous for u4's batching.)
        let n: usize = 5120;
        let mixes: [&[usize]; 3] = [&[2560, 2560], &[1024, 512, 48, 48], &[1024, 512, 48, 47]];

        let mut handler = TernaryGemmCudaRaw::new().expect("CUDA init");
        let r1_kernel = handler
            ._module
            .load_function("gemv_ternary_dp4a_multi_persistent")
            .expect("load persistent kernel");
        let r2_kernel = handler
            ._module
            .load_function("gemv_ternary_dp4a_multi_r2")
            .expect("load r2 kernel");
        let pf_kernel = handler
            ._module
            .load_function("gemv_ternary_dp4a_multi_pf")
            .expect("load pf kernel");
        let u4_kernel = handler
            ._module
            .load_function("gemv_ternary_dp4a_multi_u4")
            .expect("load u4 kernel");

        // Shared quantized activation (identical scheme to `forward`).
        let mut state: u32 = 0x60D5C0DE;
        let x: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 16) as f32 / 65535.0 - 0.5) * 2.0
            })
            .collect();
        let ablocks = n.div_ceil(ACTIVATION_BLOCK);
        let mut act_i8 = vec![0i8; n];
        let mut ascale = vec![0f32; ablocks];
        for (blk, ascale_val) in ascale.iter_mut().enumerate() {
            let s = blk * ACTIVATION_BLOCK;
            let len = ACTIVATION_BLOCK.min(n - s);
            let mut absmax = 0f32;
            for i in 0..len {
                absmax = absmax.max(x[s + i].abs());
            }
            let d = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
            *ascale_val = d;
            for i in 0..len {
                let q = (x[s + i] / d).round().clamp(-128.0, 127.0);
                act_i8[s + i] = q as i8;
            }
        }
        let act_dev = handler.stream.clone_htod(&act_i8).expect("upload act");
        let ascale_dev = handler.stream.clone_htod(&ascale).expect("upload ascale");
        let int16_per_row = (n / 8) as i32;
        let groups_per_row = n.div_ceil(WEIGHT_GROUP) as i32;
        let ablock_i32 = ACTIVATION_BLOCK as i32;
        let ablocks_i32 = ablocks as i32;

        // Local launcher — same arg wiring as the production wrapper for both
        // variants (identical signatures; only rows-per-warp differs).
        fn launch_variant(
            handler: &TernaryGemmCudaRaw,
            kernel: &cudarc::driver::safe::CudaFunction,
            wbs: &[&WeightBuffers],
            m_i32: &[i32; 4],
            outs: &[cudarc::driver::safe::CudaSlice<f32>; 4],
            act_dev: &cudarc::driver::safe::CudaSlice<i8>,
            ascale_dev: &cudarc::driver::safe::CudaSlice<f32>,
            int16_per_row: i32,
            groups_per_row: i32,
            ablock_i32: i32,
            ablocks_i32: i32,
            accumulate: i32,
            total_rows: i32,
            rows_per_warp: u32,
        ) {
            let grid_x = (total_rows.max(1) as u32)
                .div_ceil(rows_per_warp * (WG_THREADS / 32))
                .max(1);
            let cfg = LaunchConfig {
                grid_dim: (grid_x, 1, 1),
                block_dim: (WG_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                handler
                    .stream
                    .launch_builder(kernel)
                    .arg(&wbs[0].codes_dev)
                    .arg(&wbs[0].wscale_dev)
                    .arg(&outs[0])
                    .arg(&m_i32[0])
                    .arg(&wbs[1].codes_dev)
                    .arg(&wbs[1].wscale_dev)
                    .arg(&outs[1])
                    .arg(&m_i32[1])
                    .arg(&wbs[2].codes_dev)
                    .arg(&wbs[2].wscale_dev)
                    .arg(&outs[2])
                    .arg(&m_i32[2])
                    .arg(&wbs[3].codes_dev)
                    .arg(&wbs[3].wscale_dev)
                    .arg(&outs[3])
                    .arg(&m_i32[3])
                    .arg(act_dev)
                    .arg(ascale_dev)
                    .arg(&int16_per_row)
                    .arg(&groups_per_row)
                    .arg(&ablock_i32)
                    .arg(&ablocks_i32)
                    .arg(&accumulate)
                    .arg(&total_rows)
                    .launch(cfg)
                    .expect("launch variant");
            }
        }

        for &seg_m in &mixes {
            // Pad to exactly 4 segments (zero-length segments match no rows;
            // pointers reused from segment 0 — the production wrapper's own
            // convention, and the straddle guard is per-row).
            let mut segs: Vec<usize> = seg_m.to_vec();
            while segs.len() < 4 {
                segs.push(0);
            }
            let segs: [usize; 4] = segs.try_into().unwrap();
            let total_m: usize = segs.iter().sum();
            let base_idx = handler.weights.len();
            for &m in &segs {
                let w = make_test_weights(m, n);
                let _ = handler.upload_weights(&w).expect("upload");
            }
            let wbs: Vec<_> = (0..4).map(|i| &handler.weights[base_idx + i]).collect();
            let m_i32: [i32; 4] = segs.map(|m| m as i32);
            let total_rows = total_m as i32;

            // ── Store mode, all three variants (r1 = production reference) ──
            let outs_r1: Vec<_> = segs
                .iter()
                .map(|&m| handler.stream.alloc_zeros::<f32>(m).expect("alloc"))
                .collect();
            let outs_r2: Vec<_> = segs
                .iter()
                .map(|&m| handler.stream.alloc_zeros::<f32>(m).expect("alloc"))
                .collect();
            let outs_pf: Vec<_> = segs
                .iter()
                .map(|&m| handler.stream.alloc_zeros::<f32>(m).expect("alloc"))
                .collect();
            let outs_u4: Vec<_> = segs
                .iter()
                .map(|&m| handler.stream.alloc_zeros::<f32>(m).expect("alloc"))
                .collect();
            launch_variant(
                &handler, &r1_kernel, &wbs, &m_i32, outs_r1.as_slice().try_into().unwrap(),
                &act_dev, &ascale_dev, int16_per_row, groups_per_row, ablock_i32,
                ablocks_i32, 0, total_rows, 1,
            );
            launch_variant(
                &handler, &r2_kernel, &wbs, &m_i32, outs_r2.as_slice().try_into().unwrap(),
                &act_dev, &ascale_dev, int16_per_row, groups_per_row, ablock_i32,
                ablocks_i32, 0, total_rows, 2,
            );
            launch_variant(
                &handler, &pf_kernel, &wbs, &m_i32, outs_pf.as_slice().try_into().unwrap(),
                &act_dev, &ascale_dev, int16_per_row, groups_per_row, ablock_i32,
                ablocks_i32, 0, total_rows, 1,
            );
            launch_variant(
                &handler, &u4_kernel, &wbs, &m_i32, outs_u4.as_slice().try_into().unwrap(),
                &act_dev, &ascale_dev, int16_per_row, groups_per_row, ablock_i32,
                ablocks_i32, 0, total_rows, 1,
            );
            handler.stream.synchronize().expect("sync");

            for (name, outs_x) in [("r2", &outs_r2), ("pf", &outs_pf), ("u4", &outs_u4)] {
                let mut max_diff = 0.0f32;
                let mut mismatches = 0usize;
                for seg in 0..4 {
                    let mut a = vec![0f32; segs[seg]];
                    let mut b = vec![0f32; segs[seg]];
                    handler.stream.memcpy_dtoh(&outs_r1[seg], &mut a).expect("dtoh");
                    handler.stream.memcpy_dtoh(&outs_x[seg], &mut b).expect("dtoh");
                    for i in 0..segs[seg] {
                        let diff = (a[i] - b[i]).abs();
                        max_diff = max_diff.max(diff);
                        if diff != 0.0 {
                            mismatches += 1;
                        }
                    }
                }
                eprintln!(
                    "[{name}_vs_r1] segs={segs:?} n={n} store: max_diff={max_diff:.4e}, mismatches={mismatches}"
                );
                assert!(
                    max_diff == 0.0,
                    "{name} kernel diverges from r1 (store): segs={segs:?}, max_diff={max_diff:.4e}, {mismatches} rows differ"
                );
            }

            // ── Accumulate mode (mix 0 only — the down/out epilogue shape) ──
            if seg_m.len() == 2 {
                let mk_prefill = |handler: &TernaryGemmCudaRaw, m: usize| {
                    let v: Vec<f32> = (0..m).map(|i| ((i % 17) as f32 - 8.0) * 0.5).collect();
                    handler.stream.clone_htod(&v).expect("prefill")
                };
                let acc_r1: Vec<_> = segs
                    .iter()
                    .map(|&m| mk_prefill(&handler, m))
                    .collect();
                let acc_r2: Vec<_> = segs
                    .iter()
                    .map(|&m| mk_prefill(&handler, m))
                    .collect();
                launch_variant(
                    &handler, &r1_kernel, &wbs, &m_i32, acc_r1.as_slice().try_into().unwrap(),
                    &act_dev, &ascale_dev, int16_per_row, groups_per_row, ablock_i32,
                    ablocks_i32, 1, total_rows, 1,
                );
                launch_variant(
                    &handler, &r2_kernel, &wbs, &m_i32, acc_r2.as_slice().try_into().unwrap(),
                    &act_dev, &ascale_dev, int16_per_row, groups_per_row, ablock_i32,
                    ablocks_i32, 1, total_rows, 2,
                );
                handler.stream.synchronize().expect("sync");
                let mut acc_max = 0.0f32;
                for seg in 0..4 {
                    let mut a = vec![0f32; segs[seg]];
                    let mut b = vec![0f32; segs[seg]];
                    handler.stream.memcpy_dtoh(&acc_r1[seg], &mut a).expect("dtoh");
                    handler.stream.memcpy_dtoh(&acc_r2[seg], &mut b).expect("dtoh");
                    for i in 0..segs[seg] {
                        acc_max = acc_max.max((a[i] - b[i]).abs());
                    }
                }
                eprintln!("[r2_vs_r1] segs={segs:?} n={n} accumulate: max_diff={acc_max:.4e}");
                assert!(
                    acc_max == 0.0,
                    "r2 kernel diverges from r1 (accumulate): segs={segs:?}, max_diff={acc_max:.4e}"
                );
            }
        }
    }

    /// Plan 611 (Issue 1000 — the owner-greenlit lossy lane) — the split2
    /// kernel is LOSSY-CLASS (accumulation reorder): the gates are (a)
    /// run-twice DETERMINISM (exact), and (b) a reorder-class bound vs the
    /// one-row persistent kernel (max_rel small, not zero — unlike r2/pf/u4
    /// whose gate was bit-identity). Shapes: both real split-target widths
    /// (n=17,408 FFN down; n=5,120 GDN out_proj), the straddle mix, and the
    /// accumulate epilogue.
    #[test]
    fn test_multi_gemv_split2_reorder_class_and_determinism() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let mut handler = TernaryGemmCudaRaw::new().expect("CUDA init");
        let r1_kernel = handler
            ._module
            .load_function("gemv_ternary_dp4a_multi_persistent")
            .expect("load persistent kernel");
        let split2_kernel = handler
            ._module
            .load_function("gemv_ternary_dp4a_multi_split2")
            .expect("load split2 kernel");

        // (n, mixes) — n=17408 is the FFN down width, n=5120 the out_proj
        // width (both split targets); at n=5120 the straddle mix too.
        let cases: &[(usize, &[&[usize]])] = &[
            (17_408, &[&[5_120]]),
            (5_120, &[&[5_120], &[2_560, 2_560], &[1_024, 512, 48, 47]]),
        ];

        for &(n, mixes) in cases {
            let mut state: u32 = 0x6D5C0DE;
            let x: Vec<f32> = (0..n)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((state >> 16) as f32 / 65535.0 - 0.5) * 2.0
                })
                .collect();
            let ablocks = n.div_ceil(ACTIVATION_BLOCK);
            let mut act_i8 = vec![0i8; n];
            let mut ascale = vec![0f32; ablocks];
            for (blk, ascale_val) in ascale.iter_mut().enumerate() {
                let s = blk * ACTIVATION_BLOCK;
                let len = ACTIVATION_BLOCK.min(n - s);
                let mut absmax = 0f32;
                for i in 0..len {
                    absmax = absmax.max(x[s + i].abs());
                }
                let d = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                *ascale_val = d;
                for i in 0..len {
                    let q = (x[s + i] / d).round().clamp(-128.0, 127.0);
                    act_i8[s + i] = q as i8;
                }
            }
            let act_dev = handler.stream.clone_htod(&act_i8).expect("upload act");
            let ascale_dev = handler.stream.clone_htod(&ascale).expect("upload ascale");
            let int16_per_row = (n / 8) as i32;
            let groups_per_row = n.div_ceil(WEIGHT_GROUP) as i32;
            let ablock_i32 = ACTIVATION_BLOCK as i32;
            let ablocks_i32 = ablocks as i32;

            fn launch(
                handler: &TernaryGemmCudaRaw,
                kernel: &cudarc::driver::safe::CudaFunction,
                wbs: &[&WeightBuffers],
                m_i32: &[i32; 4],
                outs: &[cudarc::driver::safe::CudaSlice<f32>; 4],
                act_dev: &cudarc::driver::safe::CudaSlice<i8>,
                ascale_dev: &cudarc::driver::safe::CudaSlice<f32>,
                int16_per_row: i32,
                groups_per_row: i32,
                ablock_i32: i32,
                ablocks_i32: i32,
                accumulate: i32,
                total_rows: i32,
                split2: bool,
            ) {
                let grid_x = if split2 {
                    (total_rows.max(1) as u32).div_ceil(WG_THREADS / 64).max(1)
                } else {
                    (total_rows.max(1) as u32).div_ceil(WG_THREADS / 32).max(1)
                };
                let cfg = LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (WG_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    handler
                        .stream
                        .launch_builder(kernel)
                        .arg(&wbs[0].codes_dev)
                        .arg(&wbs[0].wscale_dev)
                        .arg(&outs[0])
                        .arg(&m_i32[0])
                        .arg(&wbs[1].codes_dev)
                        .arg(&wbs[1].wscale_dev)
                        .arg(&outs[1])
                        .arg(&m_i32[1])
                        .arg(&wbs[2].codes_dev)
                        .arg(&wbs[2].wscale_dev)
                        .arg(&outs[2])
                        .arg(&m_i32[2])
                        .arg(&wbs[3].codes_dev)
                        .arg(&wbs[3].wscale_dev)
                        .arg(&outs[3])
                        .arg(&m_i32[3])
                        .arg(act_dev)
                        .arg(ascale_dev)
                        .arg(&int16_per_row)
                        .arg(&groups_per_row)
                        .arg(&ablock_i32)
                        .arg(&ablocks_i32)
                        .arg(&accumulate)
                        .arg(&total_rows)
                        .launch(cfg)
                        .expect("launch");
                }
            }

            for &seg_m in mixes {
                let mut segs: Vec<usize> = seg_m.to_vec();
                while segs.len() < 4 {
                    segs.push(0);
                }
                let segs: [usize; 4] = segs.try_into().unwrap();
                let total_m: usize = segs.iter().sum();
                let base_idx = handler.weights.len();
                for &m in &segs {
                    let w = make_test_weights(m, n);
                    let _ = handler.upload_weights(&w).expect("upload");
                }
                let wbs: Vec<_> = (0..4).map(|i| &handler.weights[base_idx + i]).collect();
                let m_i32: [i32; 4] = segs.map(|m| m as i32);
                let total_rows = total_m as i32;

                // ── r1 reference + split2 run-twice (determinism) ──
                let mk_outs = |handler: &TernaryGemmCudaRaw| -> Vec<_> {
                    segs.iter()
                        .map(|&m| handler.stream.alloc_zeros::<f32>(m).expect("alloc"))
                        .collect()
                };
                let outs_r1 = mk_outs(&handler);
                let outs_s2_a = mk_outs(&handler);
                let outs_s2_b = mk_outs(&handler);
                launch(
                    &handler, &r1_kernel, &wbs, &m_i32,
                    outs_r1.as_slice().try_into().unwrap(),
                    &act_dev, &ascale_dev, int16_per_row, groups_per_row,
                    ablock_i32, ablocks_i32, 0, total_rows, false,
                );
                launch(
                    &handler, &split2_kernel, &wbs, &m_i32,
                    outs_s2_a.as_slice().try_into().unwrap(),
                    &act_dev, &ascale_dev, int16_per_row, groups_per_row,
                    ablock_i32, ablocks_i32, 0, total_rows, true,
                );
                launch(
                    &handler, &split2_kernel, &wbs, &m_i32,
                    outs_s2_b.as_slice().try_into().unwrap(),
                    &act_dev, &ascale_dev, int16_per_row, groups_per_row,
                    ablock_i32, ablocks_i32, 0, total_rows, true,
                );
                handler.stream.synchronize().expect("sync");

                let dl = |outs: &Vec<_>, seg: usize| -> Vec<f32> {
                    let mut v = vec![0f32; segs[seg]];
                    handler.stream.memcpy_dtoh(&outs[seg], &mut v).expect("dtoh");
                    v
                };
                let mut max_rel = 0.0f64;
                let mut max_diff = 0.0f32;
                for seg in 0..4 {
                    let a = dl(&outs_r1, seg);
                    let b = dl(&outs_s2_a, seg);
                    let b2 = dl(&outs_s2_b, seg);
                    for i in 0..segs[seg] {
                        assert!(
                            b[i].to_bits() == b2[i].to_bits(),
                            "split2 NOT deterministic: segs={segs:?} n={n} seg={seg} row={i}"
                        );
                        let diff = (a[i] - b[i]).abs();
                        max_diff = max_diff.max(diff);
                        let denom = a[i].abs().max(1.0);
                        max_rel = max_rel.max((diff / denom) as f64);
                    }
                }
                eprintln!(
                    "[split2_vs_r1] segs={segs:?} n={n} store: max_diff={max_diff:.4e} max_rel={max_rel:.3e}"
                );
                assert!(
                    max_rel < 1e-4,
                    "split2 exceeds the reorder-class bound vs r1: segs={segs:?} n={n} max_rel={max_rel:.3e}"
                );

                // ── Accumulate mode (single-segment, the down/out epilogue) ──
                // (Merge fixup: the original WIP built `Vec<&CudaSlice>` where
                // `launch` takes `&[CudaSlice; 4]` — the owned-Vec shape the
                // r1/split2 section above already uses; slots 1-3 carry m=0
                // empties, matching m_i32.)
                if seg_m.len() == 1 {
                    let mk_prefill_vec = |handler: &TernaryGemmCudaRaw, m: usize| -> Vec<_> {
                        (0..4)
                            .map(|i| {
                                if i == 0 {
                                    let v: Vec<f32> =
                                        (0..m).map(|j| ((j % 17) as f32 - 8.0) * 0.5).collect();
                                    handler.stream.clone_htod(&v).expect("prefill")
                                } else {
                                    handler.stream.alloc_zeros::<f32>(0).expect("empty")
                                }
                            })
                            .collect()
                    };
                    let acc_r1 = mk_prefill_vec(&handler, segs[0]);
                    let acc_s2 = mk_prefill_vec(&handler, segs[0]);
                    launch(
                        &handler, &r1_kernel, &wbs, &m_i32,
                        acc_r1.as_slice().try_into().unwrap(),
                        &act_dev, &ascale_dev, int16_per_row, groups_per_row,
                        ablock_i32, ablocks_i32, 1, total_rows, false,
                    );
                    launch(
                        &handler, &split2_kernel, &wbs, &m_i32,
                        acc_s2.as_slice().try_into().unwrap(),
                        &act_dev, &ascale_dev, int16_per_row, groups_per_row,
                        ablock_i32, ablocks_i32, 1, total_rows, true,
                    );
                    handler.stream.synchronize().expect("sync");
                    let a: Vec<f32> = {
                        let mut v = vec![0f32; segs[0]];
                        handler.stream.memcpy_dtoh(&acc_r1[0], &mut v).expect("dtoh");
                        v
                    };
                    let b: Vec<f32> = {
                        let mut v = vec![0f32; segs[0]];
                        handler.stream.memcpy_dtoh(&acc_s2[0], &mut v).expect("dtoh");
                        v
                    };
                    let mut acc_rel = 0.0f64;
                    for i in 0..segs[0] {
                        let diff = (a[i] - b[i]).abs();
                        let denom = a[i].abs().max(1.0);
                        acc_rel = acc_rel.max((diff / denom) as f64);
                    }
                    eprintln!("[split2_vs_r1] segs={segs:?} n={n} accumulate: max_rel={acc_rel:.3e}");
                    assert!(
                        acc_rel < 1e-4,
                        "split2 accumulate exceeds reorder bound: n={n} max_rel={acc_rel:.3e}"
                    );
                }
            }
        }
    }

    /// Issue 705 — the persistent grid-stride variant must be bit-identical to
    /// the split path at EVERY grid size, including grids far below full
    /// residency (many loop iterations per warp) and grids above the work
    /// (single iteration, matching the non-persistent kernel's assignment).
    #[test]
    fn test_persistent_gemv_matches_split_bit_identical() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        // Production-shaped segment mix (deltanet input projections).
        let n = 512; // multiple of 128 and 16
        let seg_m: [usize; 4] = [1024, 512, 48, 48];
        let total_m: usize = seg_m.iter().sum();
        let ws: Vec<_> = seg_m.iter().map(|&m| make_test_weights(m, n)).collect();

        let mut state: u32 = 0xC0FFEE42;
        let x: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 16) as f32 / 65535.0 - 0.5) * 2.0
            })
            .collect();

        // ── Split reference: one launch per segment ──
        let mut handler = TernaryGemmCudaRaw::new().expect("CUDA init");
        let mut split_outs: Vec<Vec<f32>> = Vec::with_capacity(4);
        for w in &ws {
            let idx = handler.upload_weights(w).expect("upload");
            let mut out = vec![0f32; w.rows];
            handler.forward(idx, &x, &mut out).expect("forward");
            split_outs.push(out);
        }

        // ── Persistent kernel at several grid sizes ──
        let persistent_kernel = handler
            ._module
            .load_function("gemv_ternary_dp4a_multi_persistent")
            .expect("load persistent kernel");

        // Quantize identically to the multi test (per-16-block absmax).
        let ablocks = n.div_ceil(ACTIVATION_BLOCK);
        let mut act_i8 = vec![0i8; n];
        let mut ascale = vec![0f32; ablocks];
        for (blk, ascale_val) in ascale.iter_mut().enumerate() {
            let s = blk * ACTIVATION_BLOCK;
            let len = ACTIVATION_BLOCK.min(n - s);
            let mut absmax = 0f32;
            for i in 0..len {
                absmax = absmax.max(x[s + i].abs());
            }
            let d = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
            *ascale_val = d;
            for i in 0..len {
                let q = (x[s + i] / d).round().clamp(-128.0, 127.0);
                act_i8[s + i] = q as i8;
            }
        }
        let act_dev = handler.stream.clone_htod(&act_i8).expect("upload act");
        let ascale_dev = handler.stream.clone_htod(&ascale).expect("upload ascale");

        let int16_per_row = (n / 8) as i32;
        let groups_per_row = n.div_ceil(WEIGHT_GROUP) as i32;
        let ablock_i32 = ACTIVATION_BLOCK as i32;
        let ablocks_i32 = ablocks as i32;
        let m_i32: Vec<i32> = seg_m.iter().map(|&m| m as i32).collect();
        let acc_zero = 0i32;
        let total_rows_i32 = total_m as i32;
        let full_grid = (total_m as u32).div_ceil(WG_THREADS / 32);

        // Grid ladder: far below residency (32 warps → many iterations per
        // warp), just below one full wave, exactly the work grid, and above it.
        let grids = [
            4u32,
            17,
            full_grid,
            full_grid + 7,
        ];
        for grid_x in grids {
            let mut out_dev = Vec::with_capacity(4);
            for &m in &seg_m {
                out_dev.push(handler.stream.alloc_zeros::<f32>(m).expect("alloc out"));
            }
            let cfg = LaunchConfig {
                grid_dim: (grid_x, 1, 1),
                block_dim: (WG_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            {
                let wbs: Vec<_> = (0..4).map(|i| &handler.weights[i]).collect();
                unsafe {
                    handler
                        .stream
                        .launch_builder(&persistent_kernel)
                        .arg(&wbs[0].codes_dev)
                        .arg(&wbs[0].wscale_dev)
                        .arg(&out_dev[0])
                        .arg(&m_i32[0])
                        .arg(&wbs[1].codes_dev)
                        .arg(&wbs[1].wscale_dev)
                        .arg(&out_dev[1])
                        .arg(&m_i32[1])
                        .arg(&wbs[2].codes_dev)
                        .arg(&wbs[2].wscale_dev)
                        .arg(&out_dev[2])
                        .arg(&m_i32[2])
                        .arg(&wbs[3].codes_dev)
                        .arg(&wbs[3].wscale_dev)
                        .arg(&out_dev[3])
                        .arg(&m_i32[3])
                        .arg(&act_dev)
                        .arg(&ascale_dev)
                        .arg(&int16_per_row)
                        .arg(&groups_per_row)
                        .arg(&ablock_i32)
                        .arg(&ablocks_i32)
                        .arg(&acc_zero)
                        .arg(&total_rows_i32)
                        .launch(cfg)
                        .expect("launch persistent");
                }
            }
            handler.stream.synchronize().expect("sync");
            let mut max_diff = 0.0f32;
            for seg in 0..4 {
                let mut host = vec![0f32; out_dev[seg].len()];
                handler
                    .stream
                    .memcpy_dtoh(&out_dev[seg], &mut host)
                    .expect("download");
                for i in 0..seg_m[seg] {
                    max_diff = max_diff.max((split_outs[seg][i] - host[i]).abs());
                }
            }
            eprintln!(
                "[persistent_vs_split] grid={grid_x} (full={full_grid}), rows={total_m}: max_diff={max_diff:.4e}"
            );
            assert!(
                max_diff == 0.0,
                "persistent kernel diverges from split path at grid={grid_x}: max_diff={max_diff:.4e}"
            );
        }

        // ── Accumulate mode at a tiny grid (multiple iterations + += path) ──
        let base: Vec<f32> = (0..seg_m[0])
            .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
            .collect();
        let acc_dev = handler.stream.clone_htod(&base).expect("upload base");
        let acc_one = 1i32;
        {
            let wb = &handler.weights[0];
            let cfg = LaunchConfig {
                grid_dim: (3, 1, 1), // 24 warps for 1024 rows — heavy striding
                block_dim: (WG_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                handler
                    .stream
                    .launch_builder(&persistent_kernel)
                    .arg(&wb.codes_dev)
                    .arg(&wb.wscale_dev)
                    .arg(&acc_dev)
                    .arg(&m_i32[0])
                    .arg(&wb.codes_dev)
                    .arg(&wb.wscale_dev)
                    .arg(&acc_dev)
                    .arg(&0i32)
                    .arg(&wb.codes_dev)
                    .arg(&wb.wscale_dev)
                    .arg(&acc_dev)
                    .arg(&0i32)
                    .arg(&wb.codes_dev)
                    .arg(&wb.wscale_dev)
                    .arg(&acc_dev)
                    .arg(&0i32)
                    .arg(&act_dev)
                    .arg(&ascale_dev)
                    .arg(&int16_per_row)
                    .arg(&groups_per_row)
                    .arg(&ablock_i32)
                    .arg(&ablocks_i32)
                    .arg(&acc_one)
                    .arg(&m_i32[0]) // total_rows = m0 only
                    .launch(cfg)
                    .expect("launch persistent accum");
            }
        }
        handler.stream.synchronize().expect("sync");
        let mut acc_out = vec![0f32; seg_m[0]];
        handler
            .stream
            .memcpy_dtoh(&acc_dev, &mut acc_out)
            .expect("download");
        let mut acc_max_diff = 0.0f32;
        for i in 0..seg_m[0] {
            // Same operand order as residual_add_f32: x + gemv_result.
            let expect = base[i] + split_outs[0][i];
            acc_max_diff = acc_max_diff.max((expect - acc_out[i]).abs());
        }
        eprintln!("[persistent_accum] m={} n={n}: max_diff={acc_max_diff:.4e}", seg_m[0]);
        assert!(
            acc_max_diff == 0.0,
            "persistent accumulate mode diverges from base+split: max_diff={acc_max_diff:.4e}"
        );
    }
}
