//! Plan 618 S3 (Issue 028 T4) — the cudarc Q6_K GEMV: the matched-storage
//! single-checkpoint control arm's GPU path (the S2.5 q4 twin).
//!
//! Weight side: standard `Q6_K` super-blocks (`BlockQ6K`, 256 values / 210
//! bytes): f16 `d` header, 16 signed int8 sub-block scales (16 values each),
//! 128 bytes of low-nibble quants (`ql`) + 64 bytes of high-2-bit quants
//! (`qh`). No min offset (unlike Q4_K) — the value model is `w = d·sc·(q−32)`
//! with `q ∈ [0, 63]`.
//!
//! Activation side: the SAME int8 + ascale buffers the ternary/q4 paths
//! quantize into — the quantize step is format-independent.
//!
//! ```text
//! y_r = Σ_blk ascale[blk] · d·sc(sb) · Σ q6·x8      (q6 = q − 32)
//! ```
//!
//! where `sb = 8·(blk/128) + (blk%8)` is the sub-block owning the
//! 16-element activation block (= the element index / 16 — Q6_K has NO
//! interleaved scale decode). The signed `d·sc` product carries the block
//! sign (the reference encoder's sign dance), so both factors multiply as
//! signed floats. `q6 ∈ [−32, 31]`, `x8 ∈ [−128, 127]` keep every int
//! partial inside i32 (max 32·127·16 ≈ 65k).
//!
//! ## Layout (the CPU dequantizer's, verbatim)
//!
//! Within a 128-element half `h = blk/8` (elements `[128h, 128h+128)`), an
//! activation block starts at half-position `p0 = (blk%8)·16`:
//! - `ql` byte: `64h + (p0 & 31) + (p0 & 32)` — 16 CONSECUTIVE bytes, all
//!   the same nibble class (low for `p0 < 64`, high for `p0 ≥ 64`), so the
//!   unpacked bytes feed the dp4a lanes sequentially (the q4 shape).
//! - `qh` byte: `32h + (p0 & 31)` — 2 consecutive bytes; 2-bit field `g =
//!   p0/32` for every element.
//! - `sc` index: `8h + p0/16`.
//!
//! Every byte offset stays 4-byte aligned (the 16-byte granularity of `p0`),
//! so uint loads are legal.

/// CUDA source for the Q6_K GEMV kernels (single + multi-persistent) —
/// NVRTC-compiled standalone (the q4 module's pattern).
pub const Q6K_CUDA_SRC: &str = r#"
// f16 bits -> f32 (the q4 module's PTX convert; `cvt.f32.f16` is exact for
// normals and subnormals).
__device__ __forceinline__ float q6k_f16_bits_to_f32(unsigned short hbits)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(hbits));
    return f;
}

// One warp computes one output row: each lane walks its strided activation
// blocks (blk = lane, lane+32, ...), unpacks the 6-bit quants in registers,
// accumulates via `__dp4a`, then the warp reduces via `__shfl_down_sync`.
__device__ __forceinline__ float gemv_q6k_row(
    const unsigned char* __restrict__ row,  // row base: bpr blocks x 210 bytes
    const signed char* __restrict__ act,    // quantized activations [n]
    const float* __restrict__ ascale,       // per-16-block scales [n/16]
    int lane,
    int n,
    int bpr)                                // super-blocks per row = n / 256
{
    float acc = 0.0f;
    const int ablocks = n >> 4;
    for (int blk = lane; blk < ablocks; blk += 32) {
        const int bi  = blk >> 4;           // super-block within the row
        const int ab  = blk & 15;           // act-block within the super-block
        const unsigned char* blk_p = row + (long)bi * 210;
        // BlockQ6K layout: ql[128], qh[64], scales[16], d[2] — ql FIRST
        // (unlike BlockQ4K, whose f16 headers lead).
        const float d = q6k_f16_bits_to_f32(*(const unsigned short*)(blk_p + 208));
        const int sc_idx = ab;              // sub-block = element index / 16

        const int h  = ab >> 3;             // 128-half
        const int p0 = (ab & 7) << 4;       // half-position of the act-block
        const int shift = (p0 & 64) >> 4;   // low nibble below 64, high above
        const int g     = p0 >> 5;          // 2-bit qh field (0..3)
        const unsigned char* ql = blk_p + 64 * h + (p0 & 31) + (p0 & 32);
        const unsigned char* qh = blk_p + 128 + 32 * h + (p0 & 31);
        const signed char* x8 = act + (blk << 4);

        // 16 ql bytes + 16 qh bytes -> 16 signed q6 bytes packed as 4 u32s.
        unsigned int qp0 = 0, qp1 = 0, qp2 = 0, qp3 = 0;
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            const unsigned int nib = ((unsigned int)ql[u] >> shift) & 0x0Fu;
            const unsigned int hi  = (((unsigned int)qh[u] >> (2 * g)) & 3u) << 4;
            const unsigned int q6  = ((nib | hi) + 224u) & 0xFFu;  // q - 32 (mod 256)
            if (u < 4)       qp0 |= q6 << (8 * u);
            else if (u < 8)  qp1 |= q6 << (8 * (u - 4));
            else if (u < 12) qp2 |= q6 << (8 * (u - 8));
            else             qp3 |= q6 << (8 * (u - 12));
        }

        int sumi = 0;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const unsigned int qp = (w == 0) ? qp0 : (w == 1) ? qp1 : (w == 2) ? qp2 : qp3;
            const int xv = *(const int*)(x8 + (w << 2));
            sumi = __dp4a((int)qp, xv, sumi);
        }

        // Signed int8 scale (the reference encoder's sign dance) - plain
        // signed multiply, no decoder. scales at byte 192.
        const float sc = (float)(int)(*(const signed char*)(blk_p + 192 + sc_idx));
        const float asb = ascale[blk];
        acc += (d * sc * (float)sumi) * asb;
    }

    // Warp reduction.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

// Single-GEMV entrypoint (the q4 module's shape, verbatim contract).
extern "C" __global__ void gemv_q6k_dp4a(
    const unsigned char* __restrict__ blocks, // [m * bpr * 210] raw BlockQ6K bytes
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

    const float acc = gemv_q6k_row(
        blocks + (long)row * bpr * 210, act, ascale, lane, n, bpr);
    if (lane == 0) out[row] = acc;
}

// Multi-segment persistent variant — mirrors `gemv_q4k_dp4a_multi_persistent`
// exactly (same argument layout, same grid-stride row walk, same accumulate
// contract): up to 4 q6 segments sharing one quantized input in ONE launch.
extern "C" __global__ void gemv_q6k_dp4a_multi_persistent(
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

        const float acc = gemv_q6k_row(
            blocks + (long)row * bpr * 210, act, ascale, lane, n, bpr);
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

use riir_infer_core::quant::q6k::BlockQ6K;

use crate::cudarc_kernels::CudarcKernelError;
use crate::gemv_ternary_cuda_raw::WG_THREADS;

/// Compiled Q6_K GEMV kernels against a shared [`CudaContext`] (one NVRTC
/// module; the q4 lane's compile pattern).
pub struct Q6KGemvKernels {
    /// `gemv_q6k_dp4a` — single (weights, out) pair.
    pub single: CudaFunction,
    /// `gemv_q6k_dp4a_multi_persistent` — up to 4 segments, one launch.
    pub multi: CudaFunction,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
}

impl Q6KGemvKernels {
    /// NVRTC-compile the Q6_K GEMV module and load both entrypoints.
    pub fn compile(ctx: &Arc<CudaContext>) -> Result<Self, CudarcKernelError> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            Q6K_CUDA_SRC,
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
            .load_function("gemv_q6k_dp4a")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let multi = module
            .load_function("gemv_q6k_dp4a_multi_persistent")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        Ok(Self { single, multi, module })
    }
}

/// Launch `gemv_q6k_dp4a`: `out = W_q6k @ (x8 ⊙ ascale-dequant)`.
pub fn launch_gemv_q6k(
    stream: &Arc<CudaStream>,
    kernels: &Q6KGemvKernels,
    q6_blocks: &CudaSlice<u8>,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    out: &CudaSlice<f32>,
    m: usize,
    n: usize,
) -> Result<(), CudarcKernelError> {
    debug_assert!(n.is_multiple_of(256), "q6 GEMV requires n % 256 == 0");
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
            .arg(q6_blocks)
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

/// Launch `gemv_q6k_dp4a_multi_persistent`: up to 4 `(blocks, out, m)`
/// segments sharing one quantized input, one launch (the q4 multi's
/// contract, verbatim).
pub fn launch_gemv_q6k_multi(
    stream: &Arc<CudaStream>,
    kernels: &Q6KGemvKernels,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    segs: &[(&CudaSlice<u8>, &CudaSlice<f32>, usize)], // (blocks, out, m)
    accumulate: bool,
    n: usize,
) -> Result<(), CudarcKernelError> {
    debug_assert!(n.is_multiple_of(256), "q6 GEMV requires n % 256 == 0");
    debug_assert!(
        segs.len() <= 4,
        "q6 multi-GEMV supports up to 4 segments (mirrors the ternary multi)"
    );
    if segs.is_empty() {
        return Ok(());
    }
    let total_m: usize = segs.iter().map(|(_, _, m)| *m).sum();
    if total_m == 0 {
        return Ok(());
    }

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

/// The raw little-endian byte payload of a `BlockQ6K` slice (the GPU upload
/// form; `BlockQ6K` is `#[repr(C)]` Pod, 210 bytes).
pub fn q6k_blocks_as_bytes(blocks: &[BlockQ6K]) -> &[u8] {
    bytemuck::cast_slice(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use riir_infer_core::quant::q6k::{dequantize_row_q6_k, gemv_q6_k_row, quantize_row_q6_k, QK_K};

    fn cuda_or_skip() -> Option<Arc<CudaContext>> {
        CudaContext::new(0).ok()
    }

    /// Host f16-bits → f32 (the kernel's PTX convert).
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

    /// Deterministic LCG source (spread magnitudes so every sub-block's amax
    /// differs).
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

    /// The production quantize formula (per-16 absmax/127).
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
    /// reference for the 6-bit extraction layout.
    fn host_mirror(blocks: &[BlockQ6K], x8: &[i8], ascale: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (blk, &asb) in ascale.iter().enumerate() {
            let bi = blk / 16;
            let ab = blk % 16;
            let block = &blocks[bi];
            let d = f16_bits_to_f32(block.d);
            let sc = f32::from(block.scales[ab]);
            let h = ab >> 3;
            let p0 = (ab & 7) << 4;
            let shift = (p0 & 64) >> 4;
            let g = p0 >> 5;
            let ql = &block.ql[64 * h + (p0 & 31) + (p0 & 32)..];
            let qh = &block.qh[32 * h + (p0 & 31)..];
            let base = blk * 16;
            let mut sumi = 0i64;
            for i in 0..16 {
                let nib = ((u32::from(ql[i]) >> shift) & 0x0F) as i32;
                let hi = ((i32::from(qh[i]) >> (2 * g)) & 3) << 4;
                let q6 = (nib | hi) - 32;
                sumi += i64::from(q6) * i64::from(x8[base + i]);
            }
            acc += (d * sc * sumi as f32) * asb;
        }
        acc
    }

    /// The q6 GEMV kernel matches (a) its own exact host math, (b) the CPU
    /// dequantizer's layout reading, and (c) the CPU f32 reference within the
    /// activation-quantization class. 40 rows × 512 cols (ragged grid tail).
    #[test]
    fn q6k_gemv_matches_host_mirror_and_cpu_reference() {
        let Some(ctx) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let stream: Arc<CudaStream> = ctx.default_stream();
        let kernels = Q6KGemvKernels::compile(&ctx).expect("compile q6 kernels");

        let (m, n) = (40usize, 512usize);
        let nb = n / QK_K;
        let mut blocks = vec![BlockQ6K::zeroed(); m * nb];
        let src = lcg_src(m * n, 0x9E37_79B9_7F4A_7C15, 4.0);
        for r in 0..m {
            quantize_row_q6_k(&src[r * n..(r + 1) * n], &mut blocks[r * nb..(r + 1) * nb]);
        }

        let x: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 / 23.0 - 0.5) * 3.0).collect();
        let (x8, ascale) = quantize_like_gpu(&x);

        // ── (a0) the mirror's layout reading pinned to the CPU dequantizer ──
        let mut dense = vec![0.0f32; n];
        for r in 0..m.min(4) {
            dequantize_row_q6_k(&blocks[r * nb..(r + 1) * nb], &mut dense);
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
            .map(|r| gemv_q6_k_row(&blocks[r * nb..(r + 1) * nb], &x))
            .collect();

        let blocks_dev = stream
            .clone_htod(q6k_blocks_as_bytes(&blocks))
            .expect("htod blocks");
        let x8_dev = stream.clone_htod(&x8).expect("htod x8");
        let ascale_dev = stream.clone_htod(&ascale).expect("htod ascale");
        let out_dev = stream.alloc_zeros::<f32>(m).expect("alloc out");

        launch_gemv_q6k(
            &stream, &kernels, &blocks_dev, &x8_dev, &ascale_dev, &out_dev, m, n,
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
        // Magnitude-referenced bound (the S2 fixture lesson: a cancellation
        // row's dot sits near zero; reference the row's Σ|w·x| instead).
        let mut max_rel = 0.0f64;
        let mut dense = vec![0.0f32; n];
        for r in 0..m {
            dequantize_row_q6_k(&blocks[r * nb..(r + 1) * nb], &mut dense);
            let scale_ref: f64 = dense
                .iter()
                .zip(x.iter())
                .map(|(w, xv)| (f64::from(*w) * f64::from(*xv)).abs())
                .sum();
            let rel = (f64::from(got[r]) - f64::from(cpu_ref[r])).abs() / scale_ref.max(1e-9);
            max_rel = max_rel.max(rel);
        }
        eprintln!("[q6k parity] max |Δ| vs CPU f32 ref (magnitude-referenced): {max_rel:.3e}");
        // Q6_K quantization (≈0.2% RMS) × activation int8 (≈0.4% RMS): 2% of
        // the row magnitude is the honest class bound.
        assert!(max_rel < 0.02, "activation-quant class blown: {max_rel:.3e}");
    }

    /// The multi kernel: 3 segments (2 + 1 + 1 rows) in one launch ==
    /// three single launches (the segment select + the padded 4th slot).
    #[test]
    fn q6k_multi_matches_singles() {
        let Some(ctx) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let stream: Arc<CudaStream> = ctx.default_stream();
        let kernels = Q6KGemvKernels::compile(&ctx).expect("compile q6 kernels");

        let n = 256usize;
        let nb = n / QK_K;
        let src = lcg_src(4 * n, 0x7777_AAAAu64, 2.5);
        let mut blocks = vec![BlockQ6K::zeroed(); 4 * nb];
        for r in 0..4 {
            quantize_row_q6_k(&src[r * n..(r + 1) * n], &mut blocks[r * nb..(r + 1) * nb]);
        }
        let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 / 17.0 - 0.5) * 4.0).collect();
        let (x8, ascale) = quantize_like_gpu(&x);

        let x8_dev = stream.clone_htod(&x8).expect("htod x8");
        let ascale_dev = stream.clone_htod(&ascale).expect("htod ascale");

        // Singles.
        let mut want = [0.0f32; 4];
        for r in 0..4 {
            let blocks_dev = stream
                .clone_htod(q6k_blocks_as_bytes(&blocks[r * nb..(r + 1) * nb]))
                .expect("htod blocks");
            let out_dev = stream.alloc_zeros::<f32>(1).expect("alloc");
            launch_gemv_q6k(
                &stream,
                &kernels,
                &blocks_dev,
                &x8_dev,
                &ascale_dev,
                &out_dev,
                1,
                n,
            )
            .expect("launch single");
            stream.synchronize().expect("sync");
            stream.memcpy_dtoh(&out_dev, &mut want[r..r + 1]).expect("dtoh");
        }

        // Multi: segments (rows 0-1), (row 2), (row 3) — separate device
        // uploads per segment (the multi kernel takes raw segment pointers;
        // contiguity is not required — row-block offsets stay 210-byte
        // multiples and the kernel's ql/qh loads are 4-byte aligned within
        // each block).
        let seg_ms = [2usize, 1, 1];
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
                    .clone_htod(q6k_blocks_as_bytes(&blocks[lo..hi]))
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
        launch_gemv_q6k_multi(&stream, &kernels, &x8_dev, &ascale_dev, &segs, false, n)
            .expect("launch multi");
        stream.synchronize().expect("sync");

        let mut got = [0.0f32; 4];
        let mut row = 0usize;
        for (s, out) in outs.iter().enumerate() {
            stream.memcpy_dtoh(out, &mut got[row..row + seg_ms[s]]).expect("dtoh");
            row += seg_ms[s];
        }

        for r in 0..4 {
            let rel = (got[r] - want[r]).abs() / want[r].abs().max(1e-9);
            assert!(rel < 1e-5, "row {r}: multi {} vs single {} (rel {rel:.2e})", got[r], want[r]);
        }
    }
}
