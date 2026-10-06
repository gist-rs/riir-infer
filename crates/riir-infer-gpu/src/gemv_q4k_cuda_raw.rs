//! Plan 618 item 5 (Issue 028 T4) — the cudarc Q4_K GEMV: the GPU prefill
//! arm of the dual-PTQ disaggregated container.
//!
//! The deltanet cudarc decode lane (ternary dp4a, Issues 608/705) had no q4
//! weight path: a Q4_K prefill copy loaded through `ProjWeights::Q4K` refused
//! loud at the GPU hooks (S2's fail-closed posture). This module lands the
//! CUDA kernel + upload + launch wrappers so a q4-carrying layer runs the
//! same fused quantize→GEMV chain as the ternary one.
//!
//! ## Kernel math — the CPU reference shape, activation-quantized
//!
//! Weight side: standard Q4_K super-blocks (`BlockQ4K`, 256 values / 144
//! bytes): f16 `d`/`dmin` header, 12 bytes of packed 6-bit (scale, min) pairs
//! over 8 sub-blocks of 32, 128 bytes of 4-bit quants. Nibble layout (the CPU
//! `dequantize_row_q4_k` verbatim): sub-block `j` owns elements `[32j, 32j+32)`
//! and element `32j+i` is nibble `qs[32·(j/2) + i]` — LOW nibble for even `j`,
//! HIGH for odd `j`. Nibble alternation is at SUB-BLOCK granularity (NOT
//! byte-internal like q4_0).
//!
//! Activation side: the SAME int8 buffers the ternary dp4a path already
//! produces (per-16-element absmax/127 scaling + `ascale`) — the quantize
//! step is format-independent, so a q4 layer costs zero extra launches on
//! the quantize side. The min-offset term folds as a dot with ones:
//!
//! ```text
//! y_r = Σ_blk ascale[blk] · [ d·sc(sb)·Σ q4·x8  −  dmin·m(sb)·Σ x8 ]
//! ```
//!
//! where `sb = (blk/2) & 7` is the sub-block owning the 16-element
//! activation block. Both int partials ride `__dp4a` (the `Σ x8` leg against
//! a 0x01010101 operand); scales convert once per block. q ∈ [0,15] and
//! x8 ∈ [-128,127] keep every int partial inside i32 (max 15·127·16 ≈ 30k).
//!
//! Accuracy class: the activation quantization dominates (the dp4a lane's
//! measured 3.7e-3 mean-rel class, Issue 608 T3) — the same tolerance class
//! as the ternary production path, NOT the f32 CubeCL lane's 1e-3. Task-level
//! gates (argmax agreement + per-logit bound) pin the parity test.
//!
//! ## Layout
//!
//! One warp per output row (the ternary kernel's shape); each lane walks
//! activation blocks lane-strided (`blk = lane, lane+32, …`), so per-lane
//! loads stay 4/16-byte aligned: the block base is 16-byte aligned (144 = 16·9)
//! and the in-block qs offset `(blk & 15) · 8` keeps every `uint` load
//! aligned. Because a sub-block's 32 elements read consecutive bytes at ONE
//! nibble class, the dp4a lanes consume the unpacked bytes sequentially — the
//! `__byte_perm` interleave the ternary kernel needs does not apply here.
//!
//! Provenance: the GGUF wire layout is llama.cpp's `block_q4_K`; the decode
//! math mirrors this repo's CPU reference (`riir-infer-core` `quant::q4k`),
//! which is gate-tested against the llama.cpp pattern.

/// CUDA source for the Q4_K GEMV kernels (single + multi-persistent).
///
/// NVRTC-compiled standalone (no CUDA headers — the ternary `GEMV_CUDA_SRC`
/// pattern: inline PTX for the f16 convert; `__dp4a`/`__byte_perm`/
/// `__shfl_down_sync` are builtins).
pub const Q4K_CUDA_SRC: &str = r#"
// f16 bits → f32. NVRTC compiles this source standalone (no CUDA headers),
// so `__half2float` is unavailable; inline PTX wraps the hardware
// `cvt.f32.f16` (exact for normals AND subnormals — Q4_K d/dmin are f16 in
// the GGUF wire, the gemv_q4k Issue 593 lesson).
__device__ __forceinline__ float q4k_f16_bits_to_f32(unsigned short hbits)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(hbits));
    return f;
}

// The 6-bit (scale, min) decoder — `quant::q4k::get_scale_min_k4` verbatim.
__device__ __forceinline__ void q4k_get_scale_min(int j, const unsigned char* __restrict__ q,
                                                  unsigned char* __restrict__ sc,
                                                  unsigned char* __restrict__ m)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// One warp computes one output row: each lane walks its strided activation
// blocks (blk = lane, lane+32, ...), accumulates via `__dp4a` (int dot for
// the quants, ones-dot for the min term), then the warp reduces via
// `__shfl_down_sync`. Returns the lane-0 result.
//
//   y_r = Σ_blk ascale[blk] · [ d·sc(sb)·Σ q4·x8 − dmin·m(sb)·Σ x8 ]
//
// Q4_K nibble layout (the CPU dequantizer's, verbatim): sub-block j owns
// elements [32j, 32j+32) and element 32j+i is nibble `qs[32·(j/2) + i]` —
// the LOW nibble for even j, the HIGH nibble for odd j. Nibble alternation
// is at SUB-BLOCK granularity (NOT byte-internal like q4_0), so within one
// activation block every byte contributes the same nibble class and the
// dp4a lanes consume the unpacked bytes SEQUENTIALLY — no byte_perm.
__device__ __forceinline__ float gemv_q4k_row(
    const unsigned char* __restrict__ row,  // row base: bpr blocks × 144 bytes
    const signed char* __restrict__ act,    // quantized activations [n]
    const float* __restrict__ ascale,       // per-16-block scales [n/16]
    int lane,
    int n,
    int bpr)                                // super-blocks per row = n / 256
{
    float acc = 0.0f;
    const int ablocks = n >> 4;             // n % 256 == 0 ⇒ n % 16 == 0
    for (int blk = lane; blk < ablocks; blk += 32) {
        const int bi = blk >> 4;            // super-block index within the row
        const unsigned char* blk_p = row + (long)bi * 144;
        const float d    = q4k_f16_bits_to_f32(*(const unsigned short*)(blk_p));
        const float dmin = q4k_f16_bits_to_f32(*(const unsigned short*)(blk_p + 2));
        const int j = (blk >> 1) & 7;       // sub-block owning this act-block
        unsigned char sc, m;
        q4k_get_scale_min(j, blk_p + 4, &sc, &m);

        // This act-block = 16 consecutive elements of sub-block j, starting
        // at i0 = 16·(blk & 1): qs bytes [32·(j/2) + i0, +16), all read at
        // nibble shift (j odd → high). Byte offset stays 16-aligned (i0 is
        // 0 or 16, 32·(j/2) is a multiple of 32), so uint loads are legal.
        const unsigned char* qs = blk_p + 16 + 32 * (j >> 1) + ((blk & 1) << 4);
        const int shift = (j & 1) << 2;
        const signed char* x8 = act + (blk << 4);

        int sumi = 0;
        int sumx = 0;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            // One qs word (4 elements) pairs with ONE x8 word (the same 4
            // elements — sub-block elements are byte-sequential).
            const unsigned int q4 = *(const unsigned int*)(qs + (w << 2));
            const int xv = *(const int*)(x8 + (w << 2));
            const unsigned int mask = 0x0F0F0F0Fu;
            sumi = __dp4a((int)((q4 >> shift) & mask), xv, sumi);
            sumx = __dp4a(xv, (int)0x01010101u, sumx);
        }

        const float asb = ascale[blk];
        acc += (d * (float)sc * (float)sumi - dmin * (float)m * (float)sumx) * asb;
    }

    // Warp reduction.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

// Single-GEMV entrypoint: consumes the same pre-quantized int8 + ascale
// buffers the ternary `gemv_ternary_dp4a` consumes (the quantize step is
// format-independent and runs as its own kernel before this one).
extern "C" __global__ void gemv_q4k_dp4a(
    const unsigned char* __restrict__ blocks, // [m * bpr * 144] raw BlockQ4K bytes
    const signed char* __restrict__ act,      // [n] int8
    const float* __restrict__ ascale,         // [n/16]
    float* __restrict__ out,                  // [m]
    int m_rows,
    int bpr,                                  // n / 256
    int n)
{
    const int row  = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x % 32;
    if (row >= m_rows) return;

    const float acc = gemv_q4k_row(
        blocks + (long)row * bpr * 144, act, ascale, lane, n, bpr);
    if (lane == 0) out[row] = acc;
}

// Multi-segment persistent variant — mirrors `gemv_ternary_dp4a_multi_persistent`
// exactly (same argument layout, same grid-stride row walk, same accumulate
// contract): up to 4 q4 segments sharing one quantized input in ONE launch.
// Row math is byte-identical to `gemv_q4k_dp4a` (each row runs the same
// `gemv_q4k_row`); only the row → segment mapping is added.
extern "C" __global__ void gemv_q4k_dp4a_multi_persistent(
    const unsigned char* __restrict__ b0, float* __restrict__ out0, int m0,
    const unsigned char* __restrict__ b1, float* __restrict__ out1, int m1,
    const unsigned char* __restrict__ b2, float* __restrict__ out2, int m2,
    const unsigned char* __restrict__ b3, float* __restrict__ out3, int m3,
    const signed char* __restrict__ act,
    const float* __restrict__ ascale,
    int bpr,
    int n,
    int accumulate,
    int total_rows)                           // m0 + m1 + m2 + m3
{
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int total_warps = gridDim.x * warps_per_block;

    for (int row = blockIdx.x * warps_per_block + (threadIdx.x / 32);
         row < total_rows;
         row += total_warps)
    {
        const unsigned char* blocks;
        float* out;
        if (row < m0) {
            blocks = b0; out = out0;
        } else if ((row -= m0) < m1) {
            blocks = b1; out = out1;
        } else if ((row -= m1) < m2) {
            blocks = b2; out = out2;
        } else if ((row -= m2) < m3) {
            blocks = b3; out = out3;
        } else {
            return;
        }

        const float acc = gemv_q4k_row(
            blocks + (long)row * bpr * 144, act, ascale, lane, n, bpr);
        if (lane == 0) {
            if (accumulate) out[row] += acc;
            else            out[row]  = acc;
        }
    }
}
"#;

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use riir_infer_core::quant::q4k::BlockQ4K;

use crate::cudarc_kernels::CudarcKernelError;
use crate::gemv_ternary_cuda_raw::WG_THREADS;

/// Compiled Q4_K GEMV kernels against a shared [`CudaContext`] (one NVRTC
/// module; the ternary lane's compile pattern).
pub struct Q4KGemvKernels {
    /// `gemv_q4k_dp4a` — single (weights, out) pair.
    pub single: CudaFunction,
    /// `gemv_q4k_dp4a_multi_persistent` — up to 4 segments, one launch.
    pub multi: CudaFunction,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
}

impl Q4KGemvKernels {
    /// NVRTC-compile the Q4_K GEMV module and load both entrypoints.
    pub fn compile(ctx: &Arc<CudaContext>) -> Result<Self, CudarcKernelError> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            Q4K_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let single = module
            .load_function("gemv_q4k_dp4a")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let multi = module
            .load_function("gemv_q4k_dp4a_multi_persistent")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        Ok(Self {
            single,
            multi,
            module,
        })
    }
}

/// Launch `gemv_q4k_dp4a`: `out = W_q4k @ (x8 ⊙ ascale-dequant)`.
///
/// `q4_blocks` is the row-major raw `BlockQ4K` byte payload (`m · n/256 · 144`
/// bytes); `quant_i8_buf`/`ascale_buf` are the SAME buffers the ternary path
/// quantizes into (their first `n` / `n/16` elements are consumed).
pub fn launch_gemv_q4k(
    stream: &Arc<CudaStream>,
    kernels: &Q4KGemvKernels,
    q4_blocks: &CudaSlice<u8>,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    out: &CudaSlice<f32>,
    m: usize,
    n: usize,
) -> Result<(), CudarcKernelError> {
    debug_assert!(n.is_multiple_of(256), "q4 GEMV requires n % 256 == 0");
    if m == 0 {
        return Ok(());
    }
    let m_i32 = m as i32;
    let bpr_i32 = (n / 256) as i32;
    let n_i32 = n as i32;

    let grid_x = (m as u32).div_ceil(WG_THREADS / 32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (WG_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernels.single)
            .arg(q4_blocks)
            .arg(quant_i8_buf)
            .arg(ascale_buf)
            .arg(out)
            .arg(&m_i32)
            .arg(&bpr_i32)
            .arg(&n_i32)
            .launch(cfg)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Launch `gemv_q4k_dp4a_multi_persistent`: up to 4 `(blocks, out, m)` segments
/// sharing one quantized input, one launch. Mirrors the ternary multi's
/// segment contract (same `.n` per caller invariant; zero-length segments
/// match no rows and reuse segment 0's pointers).
pub fn launch_gemv_q4k_multi(
    stream: &Arc<CudaStream>,
    kernels: &Q4KGemvKernels,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    segs: &[(&CudaSlice<u8>, &CudaSlice<f32>, usize)], // (blocks, out, m)
    accumulate: bool,
    n: usize,
) -> Result<(), CudarcKernelError> {
    debug_assert!(n.is_multiple_of(256), "q4 GEMV requires n % 256 == 0");
    debug_assert!(
        segs.len() <= 4,
        "q4 multi-GEMV supports up to 4 segments (mirrors the ternary multi)"
    );
    if segs.is_empty() {
        return Ok(());
    }
    let total_m: usize = segs.iter().map(|(_, _, m)| *m).sum();
    if total_m == 0 {
        return Ok(());
    }

    // Pad to exactly 4 segments (zero-length segments match no rows; their
    // pointers are never dereferenced — reuse segment 0's).
    let mut padded: Vec<(&CudaSlice<u8>, &CudaSlice<f32>, usize)> = Vec::with_capacity(4);
    padded.extend_from_slice(segs);
    while padded.len() < 4 {
        padded.push(padded[0]);
    }
    let [s0, s1, s2, s3] = padded.as_slice() else {
        unreachable!()
    };

    let m: [i32; 4] = padded
        .iter()
        .map(|(_, _, m)| *m as i32)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let bpr_i32 = (n / 256) as i32;
    let n_i32 = n as i32;
    let acc_i32 = accumulate as i32;
    let total_rows_i32 = total_m as i32;

    // Uncapped one-warp-per-row grid — the ternary lane's measured-best
    // default (Bench 684); the grid-stride loop degenerates to ≤ 1 iteration
    // per warp.
    let grid_x = ((total_m as u32).div_ceil(WG_THREADS / 32)).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (WG_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&kernels.multi)
            .arg(s0.0)
            .arg(s0.1)
            .arg(&m[0])
            .arg(s1.0)
            .arg(s1.1)
            .arg(&m[1])
            .arg(s2.0)
            .arg(s2.1)
            .arg(&m[2])
            .arg(s3.0)
            .arg(s3.1)
            .arg(&m[3])
            .arg(quant_i8_buf)
            .arg(ascale_buf)
            .arg(&bpr_i32)
            .arg(&n_i32)
            .arg(&acc_i32)
            .arg(&total_rows_i32)
            .launch(cfg)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// The raw little-endian byte payload of a `BlockQ4K` slice (the GPU upload
/// form; `BlockQ4K` is `#[repr(C)]` Pod, 144 bytes).
pub fn q4k_blocks_as_bytes(blocks: &[BlockQ4K]) -> &[u8] {
    bytemuck::cast_slice(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use riir_infer_core::quant::q4k::{
        dequantize_row_q4_k, gemv_q4_k_row, get_scale_min_k4, quantize_row_q4_k, QK_K,
    };

    fn cuda_or_skip() -> Option<Arc<CudaContext>> {
        CudaContext::new(0).ok()
    }

    /// Host f16-bits → f32 (the kernel's PTX convert; scales are normal-range
    /// f16 in practice but the helper stays correct for subnormals).
    fn f16_bits_to_f32(h: u16) -> f32 {
        let sign = ((h >> 15) as u32) << 31;
        let exp = ((h >> 10) & 0x1f) as u32;
        let frac = (h & 0x3ff) as u32;
        let bits = if exp == 0 {
            if frac == 0 {
                sign
            } else {
                let e = frac.leading_zeros() - 21;
                sign | ((127 - 15 + 1 - e as i32) as u32) << 23 | ((frac << e) & 0x3ff) << 13
            }
        } else {
            sign | (exp + 112) << 23 | frac << 13
        };
        f32::from_bits(bits)
    }

    /// Deterministic LCG source (no RNG dep; spread magnitudes so every
    /// sub-block's amax differs).
    fn lcg_src(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut lcg = seed;
        (0..len)
            .map(|_| {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((lcg >> 33) as f64) / (1u64 << 31) as f64;
                (u - 0.5) as f32 * scale
            })
            .collect()
    }

    /// The production quantize formula (cudarc_kernels `quantize_f32_to_i8`):
    /// per-16 absmax/127 scale, round, clamp [-128, 127].
    fn quantize_like_gpu(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
        let ablocks = x.len() / 16;
        let mut x8 = vec![0i8; x.len()];
        let mut ascale = vec![1.0f32; ablocks];
        for b in 0..ablocks {
            let absmax = x[b * 16..(b + 1) * 16]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            let d = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
            ascale[b] = d;
            for (i, v) in x[b * 16..(b + 1) * 16].iter().enumerate() {
                x8[b * 16 + i] = (v / d).round().clamp(-128.0, 127.0) as i8;
            }
        }
        (x8, ascale)
    }

    /// The exact host mirror of the kernel's block walk — the structural
    /// reference. An indexing or nibble-order bug in the CUDA moves the
    /// output far outside f32 association noise.
    fn host_mirror(blocks: &[BlockQ4K], x8: &[i8], ascale: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (blk, &asb) in ascale.iter().enumerate() {
            let bi = blk / 16;
            let j = (blk / 2) % 8; // sub-block owning this act-block
            let block = &blocks[bi];
            let (sc, m) = get_scale_min_k4(j, &block.scales);
            // Element 32j+i is nibble qs[32·(j/2)+i]: LOW for even j, HIGH
            // for odd j. This act-block starts at i0 = 16·(blk & 1).
            let byte_base = 32 * (j / 2) + ((blk & 1) << 4);
            let shift = if j & 1 == 1 { 4 } else { 0 };
            let base = blk * 16;
            let mut sumi = 0i64;
            let mut sumx = 0i64;
            for i in 0..16 {
                let e = base + i;
                let q = (block.qs[byte_base + i] >> shift) & 0x0F;
                sumi += i64::from(q) * i64::from(x8[e]);
                sumx += i64::from(x8[e]);
            }
            let d = f16_bits_to_f32(block.d);
            let dmin = f16_bits_to_f32(block.dmin);
            acc += (d * f32::from(sc) * sumi as f32 - dmin * f32::from(m) * sumx as f32) * asb;
        }
        acc
    }

    /// The q4 GEMV kernel matches (a) its own exact host math and (b) the
    /// CPU f32 reference within the activation-quantization class. 40 rows ×
    /// 512 cols (2 super-blocks/row, 32 activation blocks — lane-strided walk
    /// with a ragged 40-row grid tail over 8-warp blocks).
    #[test]
    fn q4k_gemv_matches_host_mirror_and_cpu_reference() {
        let Some(ctx) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let stream: Arc<CudaStream> = ctx.default_stream();
        let kernels = Q4KGemvKernels::compile(&ctx).expect("compile q4 kernels");

        let (m, n) = (40usize, 512usize);
        let nb = n / QK_K;
        let mut blocks = vec![BlockQ4K::zeroed(); m * nb];
        let src = lcg_src(m * n, 0x9E37_79B9_7F4A_7C15, 4.0);
        for r in 0..m {
            quantize_row_q4_k(&src[r * n..(r + 1) * n], &mut blocks[r * nb..(r + 1) * nb]);
        }

        let x: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 3.0).collect();
        let (x8, ascale) = quantize_like_gpu(&x);

        // ── (a0) the mirror's layout reading pinned to the CPU dequantizer ──
        // Σ dequant(W)·(x8·ascale) is the TRUE quantized-activation dot; the
        // mirror must reproduce it exactly (same int operands, f32 assoc
        // within noise). This leg fails WITHOUT the GPU when the nibble/
        // byte mapping drifts from the dequantizer's.
        let mut dense = vec![0.0f32; n];
        for r in 0..m.min(4) {
            dequantize_row_q4_k(&blocks[r * nb..(r + 1) * nb], &mut dense);
            let dequant_dot: f32 = dense
                .iter()
                .enumerate()
                .map(|(e, w)| w * (x8[e] as f32) * ascale[e / 16])
                .sum();
            let want = host_mirror(&blocks[r * nb..(r + 1) * nb], &x8, &ascale);
            let rel = (dequant_dot - want).abs() / dequant_dot.abs().max(1e-9);
            assert!(
                rel < 1e-4,
                "row {r}: mirror {want:.4} vs dequant-dot {dequant_dot:.4} (rel {rel:.2e}) — \
                 the host layout reading drifted from the CPU dequantizer"
            );
        }

        // ── (a) vs the exact host mirror (structural truth) ──
        let mirror: Vec<f32> = (0..m)
            .map(|r| host_mirror(&blocks[r * nb..(r + 1) * nb], &x8, &ascale))
            .collect();
        // ── (b) the CPU f32 dequant-dot reference (quant-dominated class) ──
        let cpu_ref: Vec<f32> = (0..m)
            .map(|r| gemv_q4_k_row(&blocks[r * nb..(r + 1) * nb], &x))
            .collect();

        let blocks_dev = stream
            .clone_htod(q4k_blocks_as_bytes(&blocks))
            .expect("htod blocks");
        let x8_dev = stream.clone_htod(&x8).expect("htod x8");
        let ascale_dev = stream.clone_htod(&ascale).expect("htod ascale");
        let out_dev = stream.alloc_zeros::<f32>(m).expect("alloc out");

        launch_gemv_q4k(
            &stream,
            &kernels,
            &blocks_dev,
            &x8_dev,
            &ascale_dev,
            &out_dev,
            m,
            n,
        )
        .expect("launch");
        stream.synchronize().expect("sync");
        let mut got = vec![0.0f32; m];
        stream.memcpy_dtoh(&out_dev, &mut got).expect("dtoh");

        for r in 0..m {
            let rel = (got[r] - mirror[r]).abs() / mirror[r].abs().max(1e-9);
            assert!(
                rel < 1e-5,
                "row {r}: GPU {} vs mirror {} (rel {rel:.2e})",
                got[r],
                mirror[r]
            );
        }
        // Reference class: the activation quantization dominates. The bound
        // is magnitude-referenced (|Δ| relative to Σ|w·x|), NOT value-
        // referenced — the S2 fixture lesson: a cancellation row's dot sits
        // near zero and a value-referenced ratio blows up on CORRECT math.
        let mut max_rel = 0.0f64;
        let mut dense = vec![0.0f32; n];
        for r in 0..m {
            dequantize_row_q4_k(&blocks[r * nb..(r + 1) * nb], &mut dense);
            let scale_ref: f64 = dense
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| (f64::from(*w) * f64::from(*xv)).abs())
                .sum();
            let rel = (f64::from(got[r]) - f64::from(cpu_ref[r])).abs() / scale_ref.max(1e-9);
            max_rel = max_rel.max(rel);
        }
        eprintln!("[q4k parity] max |Δ| vs CPU f32 ref (magnitude-referenced): {max_rel:.3e}");
        // Q4_K quantization (≈0.7% RMS per element) × activation int8 (≈0.4%
        // RMS) over a cancellation-heavy random row: 2% of the row magnitude
        // is the honest class bound, measured margin ~3×.
        assert!(max_rel < 0.02, "activation-quant class blown: {max_rel:.3e}");
    }

    /// The multi kernel: 3 segments (2 + 1 + 1 rows) in one launch ==
    /// three single launches. Exercises the segment select + the padded
    /// 4th slot (zero rows, never dereferenced).
    #[test]
    fn q4k_multi_matches_singles() {
        let Some(ctx) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let stream: Arc<CudaStream> = ctx.default_stream();
        let kernels = Q4KGemvKernels::compile(&ctx).expect("compile q4 kernels");

        let n = 256usize;
        let nb = n / QK_K;
        let seg_ms = [2usize, 1, 1];
        let total: usize = seg_ms.iter().sum();

        let mut blocks = vec![BlockQ4K::zeroed(); total * nb];
        let src = lcg_src(total * n, 0xDEAD_BEEF_CAFE_F00D, 2.0);
        for r in 0..total {
            quantize_row_q4_k(&src[r * n..(r + 1) * n], &mut blocks[r * nb..(r + 1) * nb]);
        }
        let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 / 17.0 - 0.5) * 2.5).collect();
        let (x8, ascale) = quantize_like_gpu(&x);
        let expected: Vec<f32> = (0..total)
            .map(|r| gemv_q4_k_row(&blocks[r * nb..(r + 1) * nb], &x))
            .collect();

        let x8_dev = stream.clone_htod(&x8).expect("htod");
        let ascale_dev = stream.clone_htod(&ascale).expect("htod");

        // Separate device uploads per segment — the multi kernel takes raw
        // segment pointers, so contiguity is not required (row-block offsets
        // within a segment are multiples of 144 bytes — 16-aligned, so the
        // kernel's uint loads stay aligned). Owned handles outlive `segs`.
        let outs: Vec<_> = seg_ms
            .iter()
            .map(|&sm| stream.alloc_zeros::<f32>(sm).expect("alloc"))
            .collect();
        let mut off_rows = 0usize;
        let mut devs: Vec<CudaSlice<u8>> = Vec::with_capacity(seg_ms.len());
        for &sm in seg_ms.iter() {
            let lo = off_rows * nb;
            let hi = (off_rows + sm) * nb;
            devs.push(
                stream
                    .clone_htod(q4k_blocks_as_bytes(&blocks[lo..hi]))
                    .expect("htod segment"),
            );
            off_rows += sm;
        }
        let segs: Vec<(&CudaSlice<u8>, &CudaSlice<f32>, usize)> = devs
            .iter()
            .zip(outs.iter())
            .zip(seg_ms.iter())
            .map(|((d, o), &sm)| (d, o, sm))
            .collect();
        launch_gemv_q4k_multi(&stream, &kernels, &x8_dev, &ascale_dev, &segs, false, n)
            .expect("multi launch");
        stream.synchronize().expect("sync");

        let mut row = 0usize;
        for (s, out) in outs.iter().enumerate() {
            let mut got = vec![0.0f32; seg_ms[s]];
            stream.memcpy_dtoh(out, &mut got).expect("dtoh");
            for (i, g) in got.iter().enumerate() {
                let want = expected[row + i];
                let rel = (f64::from(*g) - f64::from(want)).abs() / f64::from(want).abs().max(1e-9);
                assert!(
                    rel < 0.05,
                    "seg {s} row {i}: {g} vs {want} (rel {rel:.2e})"
                );
            }
            row += seg_ms[s];
        }
    }

    /// The accumulate contract: `out[row] += acc` — the second launch adds
    /// to the first launch's result (the residual-fold shape).
    #[test]
    fn q4k_accumulate_adds_instead_of_overwriting() {
        let Some(ctx) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let stream: Arc<CudaStream> = ctx.default_stream();
        let kernels = Q4KGemvKernels::compile(&ctx).expect("compile q4 kernels");

        let (m, n) = (1usize, 256usize);
        let nb = n / QK_K;
        let mut blocks = vec![BlockQ4K::zeroed(); m * nb];
        let src = lcg_src(m * n, 0x0BAD_C0DE_0123_4567, 2.0);
        quantize_row_q4_k(&src, &mut blocks);
        let x: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 / 11.0 - 0.5) * 2.0).collect();
        let (x8, ascale) = quantize_like_gpu(&x);
        let want = gemv_q4_k_row(&blocks, &x);

        let blocks_dev = stream.clone_htod(q4k_blocks_as_bytes(&blocks)).expect("htod");
        let x8_dev = stream.clone_htod(&x8).expect("htod");
        let ascale_dev = stream.clone_htod(&ascale).expect("htod");
        let out_dev = stream.alloc_zeros::<f32>(m).expect("alloc");

        let segs = [(&blocks_dev, &out_dev, m)];
        launch_gemv_q4k_multi(&stream, &kernels, &x8_dev, &ascale_dev, &segs, false, n)
            .expect("first launch");
        // Second launch with the SAME segment, accumulate=true: out doubles.
        launch_gemv_q4k_multi(&stream, &kernels, &x8_dev, &ascale_dev, &segs, true, n)
            .expect("second launch");
        stream.synchronize().expect("sync");

        let mut got = vec![0.0f32; m];
        stream.memcpy_dtoh(&out_dev, &mut got).expect("dtoh");
        let rel = (f64::from(got[0]) - 2.0 * f64::from(want)).abs()
            / f64::from(want).abs().max(1e-9);
        assert!(rel < 0.05, "accumulate: {} vs 2×{} (rel {rel:.2e})", got[0], want);
    }
}
