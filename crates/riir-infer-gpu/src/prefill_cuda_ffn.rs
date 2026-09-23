//! Issue 734 Arm 7 (Bench 721) — the cudarc-side **FFN-block migration**,
//! the structural follow-up to Bench 720's G2 refutation (the per-GEMM host
//! round trip measured 0.095-0.234×; 83% of the arm was CubeCL staging +
//! pipeline serialization, not PCIe).
//!
//! ## The shape
//!
//! Instead of round-tripping EVERY projection (Bench 720: 5 reads + 5 writes
//! per GDN layer), the whole **FFN block** runs on the cudarc stream with
//! ONE read + ONE write per layer:
//!
//! ```text
//! read x_b [p×n] → rmsnorm → quantize → gate GEMM ┐
//!                            quantize → up   GEMM ┤→ swiglu → quantize → down GEMM
//!                                                  → residual(x + ffnout) → write x_b'
//! ```
//!
//! gate+up share ONE quantize pass (identical bits — the quantize is a pure
//! function of the same input). All three GEMMs ride Bench 719's
//! `mma.sync.m16n8k32.s8` kernel with the bit-identity selectors proven on
//! real activations in Bench 720 (`QuantDiv::Full` + `FoldMode::Fused` +
//! `MmaTile::Tm64`).
//!
//! ## The numerics gate (this unit's FIRST deliverable)
//!
//! The elementwise/reduction ops crossing runtimes — `rmsnorm_batched_f32`,
//! `deltanet_gating_f32` (SwiGLU), `residual_add_f32` — must be
//! **bit-identical** to the CubeCL originals or the full-model Bench-710 FNV
//! pins break. The CUDA replicas reproduce the CubeCL kernels' exact
//! arithmetic structure (256-thread strided accumulation + the same shared-
//! memory tree order for rmsnorm; the same `g·sigmoid(g)·up` op order for
//! SwiGLU), with **variant families** for the driver-dependent transcendental
//! forms — the Bench-719 lesson generalized:
//!
//! - the NVIDIA SPIR-V consumer's `OpFDiv` lowering is NOT `div.rn`
//!   (div.full / div.approx matched bit-for-bit there);
//! - the Vulkan driver FMA-contracts mul→add chains (the outer-fold finding);
//! - `OpSqrt` / `OpExp` (GLSL.std.450 Sqrt/Exp) lowering forms are UNKNOWN
//!   for this driver — the probe sweeps (div × sqrt) for rmsnorm and
//!   (div × exp) for SwiGLU and selects the exact forms empirically
//!   (`bench_734_ffn_cuda_bitidentity`).
//!
//! ## Knob
//!
//! `RIIR_PREFILL_CUDA_FFN` env ("1"/"true"/"on") or
//! [`set_prefill_use_cuda_ffn`] — explicit env wins over the setter; DEFAULT
//! OFF. Gated `p ≤ 4096` + `n % 128 == 0` + `mlp % 128 == 0` (the mma
//! GROUP_COLS contract on both GEMM input dims). Any init/read/alloc/launch
//! failure falls through to the CubeCL FFN path (bit-safe: both compute the
//! same values).

use std::sync::{Arc, Mutex, OnceLock};

use cubecl::prelude::*;
use cubecl::server::Handle;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;

use crate::cubecl_runtime::ActiveRuntime;
use crate::gemm_ternary_i8_mma_cuda_raw::{GemmI8MmaScratch, GemmTernaryI8MmaCuda};
use crate::gemv_ternary_cubecl::TernaryHandle;

// ---------------------------------------------------------------------------
// Knob
// ---------------------------------------------------------------------------

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_CUDA_FFN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn prefill_use_cuda_ffn() -> bool {
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env = *ENV.get_or_init(|| {
        std::env::var("RIIR_PREFILL_CUDA_FFN")
            .ok()
            .map(|s| matches!(s.trim(), "1" | "true" | "on"))
    });
    env.unwrap_or_else(|| PREFILL_CUDA_FFN.load(std::sync::atomic::Ordering::Relaxed))
}

/// Force-enable/disable the cudarc FFN-block prefill arm (overrides the
/// DEFAULT-OFF state; an explicit `RIIR_PREFILL_CUDA_FFN` env value wins over
/// both). Issue 734 Arm 7 A/B knob — public for the e2e benches.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_use_cuda_ffn(on: bool) {
    PREFILL_CUDA_FFN.store(on, std::sync::atomic::Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Numerics-form variants (the probe selects; see module doc)
// ---------------------------------------------------------------------------

/// rmsnorm Phase-1 accumulation form: `partial += x*x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnAccum {
    /// `__fmaf_rn(x, x, partial)` — matches a driver that FMA-contracts
    /// (the Bench-719 outer-fold finding generalized).
    Fma,
    /// Separate `__fmul_rn` + `__fadd_rn`.
    MulAdd,
}

/// rmsnorm Phase-3 mean form: `sum * inv_dim + eps`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnMean {
    /// `__fmaf_rn(sum, inv_dim, eps)` — contracted.
    Fma,
    /// Separate mul + add.
    MulAdd,
}

/// rmsnorm Phase-3 reciprocal-sqrt form: `1 / (mean + eps).sqrt()` — the
/// `OpFDiv(1, OpSqrt(x))` lowering (9 plausible driver forms).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnInvSqrt {
    DivRnSqrtRn,
    DivFullSqrtRn,
    DivApproxSqrtRn,
    DivRnSqrtApprox,
    DivFullSqrtApprox,
    DivApproxSqrtApprox,
    /// Merged `rsqrt.approx.f32` (the driver pattern-matching div+sqrt).
    RsqrtApprox,
    /// `rcp.approx(sqrt.rn(x))`.
    RcpSqrtRn,
    /// `rcp.approx(sqrt.approx(x))`.
    RcpSqrtApprox,
}

/// SwiGLU exp form — GLSL.std.450 `Exp` lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnExp {
    /// CUDA `expf` (~2 ulp accurate).
    Accurate,
    /// `__expf` (fast path).
    Fast,
    /// Explicit `ex2.approx.f32(x * log2e)`.
    Ex2,
}

/// SwiGLU reciprocal form: `1 / (1 + e)` — the `OpFDiv(1, add)` lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnRecip {
    Rn,
    Full,
    Approx,
    /// `rcp.approx.f32(1 + e)` — the driver rewriting div-by-1 as rcp.
    Rcp,
}

impl FfnInvSqrt {
    const COUNT: usize = 9;
    fn index(self) -> usize {
        self as usize
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

pub(crate) const FFN_CUDA_SRC: &str = r#"
// ---------------------------------------------------------------------------
// helpers (the Bench-719 forms)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float ffn_div_rn(float a, float b) { return a / b; }
__device__ __forceinline__ float ffn_div_full(float a, float b)
{
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float ffn_div_approx(float a, float b)
{
    float r;
    asm("div.approx.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float ffn_rcp_approx(float a)
{
    float r;
    asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float ffn_sqrt_rn(float a) { return __fsqrt_rn(a); }
__device__ __forceinline__ float ffn_sqrt_approx(float a)
{
    float r;
    asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float ffn_rsqrt_approx(float a)
{
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float ffn_exp_acc(float a) { return expf(a); }
__device__ __forceinline__ float ffn_exp_fast(float a) { return __expf(a); }
__device__ __forceinline__ float ffn_exp_ex2(float a)
{
    float l2 = a * 1.4426950408889634f;
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(l2));
    return r;
}

// accumulation forms: partial += x*x
#define ACC_FMA(x, p) __fmaf_rn(x, x, p)
#define ACC_MUL(x, p) ((x) * (x)) + (p)
// mean forms: sum * inv_dim + eps
#define MEAN_FMA(s, id, e) __fmaf_rn(s, id, e)
#define MEAN_MUL(s, id, e) ((s) * (id)) + (e)
// 1/sqrt forms: given e = mean+eps combined value
#define INV_DRN_SRN(e) ffn_div_rn(1.0f, ffn_sqrt_rn(e))
#define INV_DFL_SRN(e) ffn_div_full(1.0f, ffn_sqrt_rn(e))
#define INV_DAP_SRN(e) ffn_div_approx(1.0f, ffn_sqrt_rn(e))
#define INV_DRN_SAP(e) ffn_div_rn(1.0f, ffn_sqrt_approx(e))
#define INV_DFL_SAP(e) ffn_div_full(1.0f, ffn_sqrt_approx(e))
#define INV_DAP_SAP(e) ffn_div_approx(1.0f, ffn_sqrt_approx(e))
#define INV_RSQ(e) ffn_rsqrt_approx(e)
#define INV_RCP_SRN(e) ffn_rcp_approx(ffn_sqrt_rn(e))
#define INV_RCP_SAP(e) ffn_rcp_approx(ffn_sqrt_approx(e))

// ---------------------------------------------------------------------------
// Batched RMSNorm — an EXACT structural replica of `rmsnorm_batched_f32`
// (norms_cubecl.rs): 256 threads, one block per row, strided Phase-1
// accumulation (i = tid, tid+256, ... ascending), the same shared-memory
// tree order (smem[tid] = smem[tid] + smem[tid+step], step 128..1), the same
// broadcast, and Phase-4 `(x * inv_rms) * g`. Only the ACCUM / MEAN / INVSQ
// forms vary.
// ---------------------------------------------------------------------------
#define RMSNORM(NAME, ACC, MEANF, INVSQ)                                     \
extern "C" __global__ void NAME(                                             \
    const float* __restrict__ input,   /* [rows * dim] */                    \
    const float* __restrict__ gamma,   /* [dim] */                           \
    float* __restrict__ output,        /* [rows * dim] */                    \
    float inv_dim,                                                           \
    float eps,                                                               \
    int dim)                                                                 \
{                                                                            \
    __shared__ float smem[256];                                              \
    const unsigned int tid = threadIdx.x;                                    \
    const long base = (long)blockIdx.x * (long)dim;                          \
    float partial = 0.0f;                                                    \
    for (int i = (int)tid; i < dim; i += 256) {                              \
        float x = input[base + i];                                           \
        partial = ACC(x, partial);                                           \
    }                                                                        \
    smem[tid] = partial;                                                     \
    __syncthreads();                                                         \
    if (tid < 128u) smem[tid] = smem[tid] + smem[tid + 128u];                \
    __syncthreads();                                                         \
    if (tid < 64u) smem[tid] = smem[tid] + smem[tid + 64u];                  \
    __syncthreads();                                                         \
    if (tid < 32u) smem[tid] = smem[tid] + smem[tid + 32u];                  \
    __syncthreads();                                                         \
    if (tid < 16u) smem[tid] = smem[tid] + smem[tid + 16u];                  \
    __syncthreads();                                                         \
    if (tid < 8u) smem[tid] = smem[tid] + smem[tid + 8u];                    \
    __syncthreads();                                                         \
    if (tid < 4u) smem[tid] = smem[tid] + smem[tid + 4u];                    \
    __syncthreads();                                                         \
    if (tid < 2u) smem[tid] = smem[tid] + smem[tid + 2u];                    \
    __syncthreads();                                                         \
    if (tid < 1u) smem[0] = smem[0] + smem[1];                               \
    __syncthreads();                                                         \
    if (tid == 0u) {                                                         \
        float e = MEANF(smem[0], inv_dim, eps);                              \
        smem[0] = INVSQ(e);                                                  \
    }                                                                        \
    __syncthreads();                                                         \
    const float inv_rms = smem[0];                                           \
    for (int j = (int)tid; j < dim; j += 256) {                              \
        float x = input[base + j];                                           \
        float g = gamma[j];                                                  \
        output[base + j] = (x * inv_rms) * g;                                \
    }                                                                        \
}

#define X9(NAME, ACC, MEANF)                                                 \
RMSNORM(NAME##_i0, ACC, MEANF, INV_DRN_SRN)                                  \
RMSNORM(NAME##_i1, ACC, MEANF, INV_DFL_SRN)                                  \
RMSNORM(NAME##_i2, ACC, MEANF, INV_DAP_SRN)                                  \
RMSNORM(NAME##_i3, ACC, MEANF, INV_DRN_SAP)                                  \
RMSNORM(NAME##_i4, ACC, MEANF, INV_DFL_SAP)                                  \
RMSNORM(NAME##_i5, ACC, MEANF, INV_DAP_SAP)                                  \
RMSNORM(NAME##_i6, ACC, MEANF, INV_RSQ)                                      \
RMSNORM(NAME##_i7, ACC, MEANF, INV_RCP_SRN)                                  \
RMSNORM(NAME##_i8, ACC, MEANF, INV_RCP_SAP)

X9(rmsnorm_ffn_a0m0, ACC_FMA, MEAN_FMA)
X9(rmsnorm_ffn_a0m1, ACC_FMA, MEAN_MUL)
X9(rmsnorm_ffn_a1m0, ACC_MUL, MEAN_FMA)
X9(rmsnorm_ffn_a1m1, ACC_MUL, MEAN_MUL)

// ---------------------------------------------------------------------------
// SwiGLU — an EXACT replica of `deltanet_gating_f32` (deltanet_cubecl.rs):
//   neg = 0.0 - g;  sig = 1 / (1 + exp(neg));  out = (g * sig) * up
// Only the exp + recip forms vary.
// ---------------------------------------------------------------------------
#define SWIGLU(NAME, EXPF, DIVF)                                             \
extern "C" __global__ void NAME(                                             \
    const float* __restrict__ gate,                                          \
    const float* __restrict__ up,                                            \
    float* __restrict__ output,                                              \
    int n)                                                                   \
{                                                                            \
    const long idx = (long)blockIdx.x * (long)blockDim.x + threadIdx.x;      \
    if (idx >= (long)n) return;                                              \
    const float g = gate[idx];                                               \
    const float neg_g = 0.0f - g;                                            \
    const float e = EXPF(neg_g);                                             \
    const float sig = DIVF(1.0f + e);                                        \
    output[idx] = (g * sig) * up[idx];                                       \
}

#define DIVRN(x) ffn_div_rn(1.0f, x)
#define DIVFL(x) ffn_div_full(1.0f, x)
#define DIVAP(x) ffn_div_approx(1.0f, x)
#define DIVRCP(x) ffn_rcp_approx(x)

#define X4S(NAME, EXPF)                                                      \
SWIGLU(NAME##_d0, EXPF, DIVRN)                                               \
SWIGLU(NAME##_d1, EXPF, DIVFL)                                               \
SWIGLU(NAME##_d2, EXPF, DIVAP)                                               \
SWIGLU(NAME##_d3, EXPF, DIVRCP)

X4S(swiglu_ffn_e0, ffn_exp_acc)
X4S(swiglu_ffn_e1, ffn_exp_fast)
X4S(swiglu_ffn_e2, ffn_exp_ex2)

// ---------------------------------------------------------------------------
// Residual add — replica of `residual_add_f32`: out[i] = a[i] + b[i].
// No variants (a single add has no form ambiguity).
// ---------------------------------------------------------------------------
extern "C" __global__ void residual_add_ffn(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int n)
{
    const long idx = (long)blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)n) return;
    output[idx] = a[idx] + b[idx];
}

// ---------------------------------------------------------------------------
// Issue 742 T1.8 — per-row argmax over a [rows * n] row-major logits block,
// one row per blockIdx.y. The reduction body + tie-break are VERBATIM
// `argmax_first_f32` (cudarc_kernels, Issue 697): indices visited in
// increasing order with strict `>` (first index wins); cross-block merge via
// the ordered-float-key atomicMax packing (higher logit wins, equal logit
// prefers the SMALLER index; +/-0.0 normalized to +0.0 before keying).
// `results[row]` must be zeroed before launch (stream memset).
// Read back as: index = !(packed as u32).
// ---------------------------------------------------------------------------
extern "C" __global__ void argmax_first_rows_f32(
    const float* __restrict__ values,   // [rows * n] row-major
    int n,
    int rows,
    unsigned long long* __restrict__ results)  // [rows], zeroed before launch
{
    const int row = blockIdx.y;
    if (row >= rows) return;
    const float* __restrict__ row_values =
        values + (long long)row * (long long)n;
    const int tid = threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // -INFINITY via bit pattern (nvrtc has no INFINITY macro in default mode;
    // the Issue-697 convention).
    float m = __int_as_float(0xff800000);
    int mi = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        const float v = row_values[i];
        if (v > m) { m = v; mi = i; }
    }

    __shared__ float sm[256];
    __shared__ int si[256];
    sm[tid] = m;
    si[tid] = mi;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sm[tid + s] > sm[tid] ||
                (sm[tid + s] == sm[tid] && si[tid + s] < si[tid])) {
                sm[tid] = sm[tid + s];
                si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        unsigned int bits = __float_as_uint(sm[0]);
        if ((bits & 0x7FFFFFFFu) == 0u) bits = 0u;
        const unsigned int key =
            (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        const unsigned long long packed =
            ((unsigned long long)key << 32) | (unsigned int)(~(unsigned int)si[0]);
        atomicMax(&results[row], packed);
    }
}

// Issue 884 T2a — per-row NLL gather over `[rows * n]` logits: one f32
// pair per row into `out` (`out[2r] = logsumexp(row)`, `out[2r+1] =
// row[targets[r]]`; NLL = lse - target). SINGLE pass, deterministic:
// online-softmax accumulation in fixed index order (thread t reads t,
// t+256, ...; pairwise tree combine with exact rescale) — no atomics, no
// cross-block reduction, bit-reproducible run to run. grid (rows), block
// 256.
extern "C" __global__ void nll_rows_f32(
    const float* __restrict__ values,   // [rows * n] row-major
    const int* __restrict__ targets,    // [rows]
    int n,
    int rows,
    float* __restrict__ out)            // [2 * rows]
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* __restrict__ rv = values + (long long)row * (long long)n;
    const int tid = threadIdx.x;

    // Online max+sumexp (single read): running max m, running sum acc;
    // rescale acc by exp(m_old - m_new) when the max grows.
    float m = __int_as_float(0xff800000);
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = rv[i];
        if (v <= m) {
            acc = __fadd_rn(acc, expf(v - m));
        } else {
            acc = __fadd_rn(expf(m - v), acc);
            m = v;
        }
    }
    __shared__ float sm[256];
    __shared__ float sa[256];
    sm[tid] = m;
    sa[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float ma = sm[tid];
            const float mb = sm[tid + s];
            const float aa = sa[tid];
            const float ab = sa[tid + s];
            if (mb > ma) {
                sm[tid] = mb;
                sa[tid] = __fadd_rn(expf(ma - mb) * aa, ab);
            } else {
                sa[tid] = __fadd_rn(aa, expf(mb - ma) * ab);
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[2 * row] = __fadd_rn(logf(sa[0]), sm[0]);
        out[2 * row + 1] = rv[targets[row]];
    }
}
"#;

// ---------------------------------------------------------------------------
// Kernel wrapper
// ---------------------------------------------------------------------------

/// The cudarc-side FFN elementwise/reduction kernels (all variants compiled;
/// the canonical forms selected by the probe — see module doc).
pub struct CudaFfnKernels {
    rmsnorm: Vec<CudaFunction>,
    swiglu: Vec<CudaFunction>,
    residual: CudaFunction,
    /// Issue 742 T1.8 — per-row argmax over `[rows * n]` logits (the verify
    /// tail's device-side token selection; Issue-697 tie-break semantics).
    argmax_rows: CudaFunction,
    /// Issue 884 T2a — per-row logsumexp + target gather (the NLL tail).
    nll_rows: CudaFunction,
    _module: Arc<CudaModule>,
}

fn idx_rmsnorm(a: FfnAccum, m: FfnMean, s: FfnInvSqrt) -> usize {
    let am = (a == FfnAccum::MulAdd) as usize;
    let mm = (m == FfnMean::MulAdd) as usize;
    (am * 2 + mm) * FfnInvSqrt::COUNT + s.index()
}

fn idx_swiglu(e: FfnExp, r: FfnRecip) -> usize {
    e as usize * 4 + r as usize
}

impl CudaFfnKernels {
    /// Own context + stream (the `GemmTernaryI8MmaCuda::new_standalone`
    /// precedent) — lets integration tests construct the kernels without
    /// importing cudarc types (cudarc is not a dev-dependency).
    pub fn new_standalone() -> Result<(Self, Arc<CudaStream>), String> {
        let ctx = CudaContext::new(0).map_err(|e| e.to_string())?;
        let stream = ctx.new_stream().map_err(|e| e.to_string())?;
        let kernels = Self::new(ctx)?;
        Ok((kernels, stream))
    }

    /// Compile (NVRTC, sm_89) + load all variants against a shared context.
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, String> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            FFN_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;
        let mut rmsnorm = Vec::with_capacity(36);
        for a in [FfnAccum::Fma, FfnAccum::MulAdd] {
            for m in [FfnMean::Fma, FfnMean::MulAdd] {
                for i in 0..FfnInvSqrt::COUNT {
                    let name = format!(
                        "rmsnorm_ffn_a{}m{}_i{i}",
                        (a == FfnAccum::MulAdd) as u8,
                        (m == FfnMean::MulAdd) as u8
                    );
                    rmsnorm.push(
                        module
                            .load_function(&name)
                            .map_err(|e| format!("{name}: {e}"))?,
                    );
                }
            }
        }
        let mut swiglu = Vec::with_capacity(12);
        for e in 0..3u8 {
            for d in 0..4u8 {
                let name = format!("swiglu_ffn_e{e}_d{d}");
                swiglu.push(
                    module
                        .load_function(&name)
                        .map_err(|e| format!("{name}: {e}"))?,
                );
            }
        }
        let residual = module
            .load_function("residual_add_ffn")
            .map_err(|e| format!("residual_add_ffn: {e}"))?;
        let argmax_rows = module
            .load_function("argmax_first_rows_f32")
            .map_err(|e| format!("argmax_first_rows_f32: {e}"))?;
        let nll_rows = module
            .load_function("nll_rows_f32")
            .map_err(|e| format!("nll_rows_f32: {e}"))?;
        Ok(Self {
            rmsnorm,
            swiglu,
            residual,
            argmax_rows,
            nll_rows,
            _module: module,
        })
    }

    /// Launch one rmsnorm variant. `input`/`output` are `[rows * dim]`;
    /// `gamma` is `[dim]`. `inv_dim` must be computed host-side exactly like
    /// the CubeCL launcher (`1.0f32 / dim as f32`).
    ///
    /// # Safety
    ///
    /// Caller guarantees the buffers cover `rows * dim` (`input`/`output`) and
    /// `dim` (`gamma`) f32 elements.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_rmsnorm(
        &self,
        stream: &CudaStream,
        a: FfnAccum,
        m: FfnMean,
        s: FfnInvSqrt,
        input: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        let inv_dim = 1.0f32 / dim as f32;
        let func = &self.rmsnorm[idx_rmsnorm(a, m, s)];
        let (rows_u, dim_i) = (rows as u32, dim as i32);
        let cfg = LaunchConfig {
            grid_dim: (rows_u, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(input)
                .arg(gamma)
                .arg(output)
                .arg(&inv_dim)
                .arg(&eps)
                .arg(&dim_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Launch one SwiGLU variant over `n` elements.
    ///
    /// # Safety
    ///
    /// Caller guarantees `gate`/`up`/`output` cover `n` f32 elements.
    pub unsafe fn launch_swiglu(
        &self,
        stream: &CudaStream,
        e: FfnExp,
        r: FfnRecip,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), String> {
        let func = &self.swiglu[idx_swiglu(e, r)];
        let (n_i, grid) = (n as i32, n.div_ceil(256) as u32);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(gate)
                .arg(up)
                .arg(output)
                .arg(&n_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Launch the residual add over `n` elements. In-place `a === output` is
    /// safe (same-index elementwise read/write).
    ///
    /// # Safety
    ///
    /// Caller guarantees `a`/`b`/`output` cover `n` f32 elements.
    pub unsafe fn launch_residual(
        &self,
        stream: &CudaStream,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), String> {
        let (n_i, grid) = (n as i32, n.div_ceil(256) as u32);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.residual)
                .arg(a)
                .arg(b)
                .arg(output)
                .arg(&n_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 742 T1.8 — per-row argmax over `values` (`[rows * n]` row-major,
    /// row `r` occupying `[r*n, (r+1)*n)`), writing one packed u64 per row
    /// into `results` (`[rows]`). Tie-break semantics are the Issue-697
    /// CPU-exact first-index convention (see the kernel doc).
    ///
    /// `results` MUST be zeroed before each launch (a stream memset is
    /// graph-capturable). Read back as: `index = !(packed as u32)`.
    ///
    /// Dispatch: `(ceil(n/256) capped at 4096, rows, 1)` blocks × 256 threads
    /// — each `blockIdx.y` row is an independent Issue-697 argmax.
    ///
    /// # Safety
    ///
    /// Caller guarantees `values` covers `rows * n` f32 elements and
    /// `results` covers `rows` u64 elements.
    pub unsafe fn launch_argmax_rows(
        &self,
        stream: &CudaStream,
        values: &CudaSlice<f32>,
        n: usize,
        rows: usize,
        results: &CudaSlice<u64>,
    ) -> Result<(), String> {
        let (n_i, rows_i) = (n as i32, rows as i32);
        let grid_x = (n as u32).div_ceil(256).min(4096);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, rows as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * (core::mem::size_of::<f32>() + core::mem::size_of::<i32>())
                as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.argmax_rows)
                .arg(values)
                .arg(&n_i)
                .arg(&rows_i)
                .arg(results)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 884 T2a — per-row logsumexp + target gather over `values`
    /// (`[rows * n]` row-major; the verify tail's device-side NLL). Writes
    /// `out[2r] = logsumexp(row r)`, `out[2r+1] = row r's target logit`;
    /// NLL = lse − target, computed by the caller. Deterministic (no
    /// atomics; fixed-order accumulation — see the kernel doc).
    ///
    /// Dispatch: `rows` blocks × 256 threads (one block per row — the
    /// whole row reduces inside one block; the logits read is the same
    /// memory the argmax pass just streamed, so it lands warm in L2 for
    /// the vocab-sized rows that fit).
    ///
    /// # Safety
    ///
    /// Caller guarantees `values` covers `rows * n` f32 elements, `targets`
 /// covers `rows` i32 elements (each < n), and `out` covers `2 * rows`
    /// f32 elements.
    pub unsafe fn launch_nll_rows(
        &self,
        stream: &CudaStream,
        values: &CudaSlice<f32>,
        targets: &CudaSlice<i32>,
        n: usize,
        rows: usize,
        out: &CudaSlice<f32>,
    ) -> Result<(), String> {
        let (n_i, rows_i) = (n as i32, rows as i32);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 2 * 256 * core::mem::size_of::<f32>() as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.nll_rows)
                .arg(values)
                .arg(targets)
                .arg(&n_i)
                .arg(&rows_i)
                .arg(out)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The FFN-block stack (context + stream + kernels + grow-only buffers)
// ---------------------------------------------------------------------------

struct FfnBufs {
    x: Option<CudaSlice<f32>>,       // [p*n] — input, then residual out (in-place)
    normx: Option<CudaSlice<f32>>,   // [p*n]
    gate: Option<CudaSlice<f32>>,    // [p*mlp]
    up: Option<CudaSlice<f32>>,      // [p*mlp]
    hid: Option<CudaSlice<f32>>,     // [p*mlp]
    ffnout: Option<CudaSlice<f32>>,  // [p*n]
    gamma: Option<CudaSlice<f32>>,   // [n]
    /// `(dim, words)` the scratch was sized for — `p` rides `words`.
    scratch_n: Option<(usize, usize, GemmI8MmaScratch)>,
    scratch_mlp: Option<(usize, usize, GemmI8MmaScratch)>,
    out_host: Vec<f32>,
}

struct FfnStack {
    mma: GemmTernaryI8MmaCuda,
    ffn: CudaFfnKernels,
    stream: Arc<CudaStream>,
    bufs: Mutex<FfnBufs>,
}

static FFN_STACK: OnceLock<Option<Arc<FfnStack>>> = OnceLock::new();

fn ffn_stack() -> Option<Arc<FfnStack>> {
    FFN_STACK
        .get_or_init(|| match build_ffn_stack() {
            Ok(s) => Some(Arc::new(s)),
            Err(e) => {
                eprintln!(
                    "[734-arm7] CUDA FFN stack init failed ({e}) — prefill stays on CubeCL"
                );
                None
            }
        })
        .clone()
}

fn build_ffn_stack() -> Result<FfnStack, String> {
    let ctx = CudaContext::new(0).map_err(|e| e.to_string())?;
    let stream = ctx.new_stream().map_err(|e| e.to_string())?;
    let mma = GemmTernaryI8MmaCuda::new(ctx.clone()).map_err(|e| e.to_string())?;
    let ffn = CudaFfnKernels::new(ctx)?;
    Ok(FfnStack {
        mma,
        ffn,
        stream,
        bufs: Mutex::new(FfnBufs {
            x: None,
            normx: None,
            gate: None,
            up: None,
            hid: None,
            ffnout: None,
            gamma: None,
            scratch_n: None,
            scratch_mlp: None,
            out_host: Vec::new(),
        }),
    })
}

/// The probe-selected canonical numerics forms for the FFN block (Bench 721
/// `bench_734_ffn_cuda_bitidentity` verdict, measured 2026-08-23):
/// the NVIDIA Vulkan driver lowers `OpFDiv(1, OpSqrt)` to a MERGED
/// `rsqrt.approx.f32` — every div∘sqrt composition differs (3.1-3.7M
/// bit-diffs @2048×5120); the accum/mean forms coincide (NVRTC's default
/// fmad contracts both spellings to the same fma.rn the driver emits).
pub fn canonical_rmsnorm_forms() -> (FfnAccum, FfnMean, FfnInvSqrt) {
    (FfnAccum::Fma, FfnMean::Fma, FfnInvSqrt::RsqrtApprox)
}

/// The probe-selected canonical SwiGLU forms (Bench 721): GLSL `Exp` →
/// `__expf`/`ex2.approx(x·log2e)` (the accurate `expf` differs on 35% of
/// production slots); `OpFDiv(1, add)` → the non-rn division family
/// (`div.full` — consistent with the quantize's `QuantDiv::Full`).
pub fn canonical_swiglu_forms() -> (FfnExp, FfnRecip) {
    (FfnExp::Fast, FfnRecip::Full)
}

// ---------------------------------------------------------------------------
// FFN-block dispatch
// ---------------------------------------------------------------------------

/// Per-phase timing trace (`RIIR_PREFILL_CUDA_FFN_TRACE=1`) — the Bench-721
/// breakdown probe: read (queue sync + DMA) / upload / block (norm+GEMMs+
/// swiglu+residual on CUDA) / dtoh / writeback per layer-dispatch.
fn trace_enabled() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| {
        std::env::var("RIIR_PREFILL_CUDA_FFN_TRACE").is_ok_and(|s| matches!(s.trim(), "1" | "true" | "on"))
    })
}

/// Run the whole FFN block (norm → gate/up → swiglu → down → residual) on the
/// cudarc stack: ONE `read` (x_b) + ONE `write` (x_out) per layer — the
/// Bench-720 G2 structural answer. Returns `false` (→ the caller falls
/// through to the CubeCL FFN path) on any failure — degradation is bit-safe:
/// both paths compute the same values.
///
/// Requires `p <= 4096`, `n % 128 == 0`, `mlp % 128 == 0` (checked by the
/// caller — the `prefill_tokens_chunk` arm).
#[allow(clippy::too_many_lines)]
pub fn dispatch_ffn_block(
    client: &ComputeClient<ActiveRuntime>,
    gate_w: &TernaryHandle,
    up_w: &TernaryHandle,
    down_w: &TernaryHandle,
    gamma_handle: &Handle,
    x_in: &Handle,
    x_out: &Handle,
    p: usize,
    n: usize,
    mlp: usize,
    eps: f32,
) -> bool {
    if p == 0 {
        return true;
    }
    let Some(stack) = ffn_stack() else { return false };
    let stream = &stack.stream;
    let trace = trace_enabled();
    let t0 = std::time::Instant::now();

    // Weight mirrors — lazy, once per handle (shared with the arm-6 path).
    let mirror = |w: &TernaryHandle| -> Option<Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>> {
        w.cuda_mma_cache
            .get_or_init(|| {
                match crate::prefill_cuda_mma::build_weight_cache(client, stream, w, true) {
                    Ok(c) => Some(Arc::new(c)),
                    Err(e) => {
                        eprintln!(
                            "[734-arm7] weight mirror failed for m={} n={} ({e}) — \
                             FFN block stays on CubeCL",
                            w.m, w.n
                        );
                        None
                    }
                }
            })
            .clone()
    };
    let Some(gate_cache) = mirror(gate_w) else { return false };
    let Some(up_cache) = mirror(up_w) else { return false };
    let Some(down_cache) = mirror(down_w) else { return false };

    // 1) Read gamma + x_b back to host (gamma rides the same queue drain).
    let Ok(gamma_bytes) = client.read_one(gamma_handle.clone()) else {
        return false;
    };
    let Ok(x_bytes) = client.read_one(x_in.clone()) else {
        return false;
    };
    let t_read = t0.elapsed();
    let gamma_f32 = f32::from_bytes(&gamma_bytes);
    let x_f32 = f32::from_bytes(&x_bytes);
    debug_assert_eq!(gamma_f32.len(), n, "post_attn_norm gamma shape");
    debug_assert_eq!(x_f32.len(), n * p, "ffn input shape");

    let pn = p * n;
    let pm = p * mlp;
    let words_n = p * (n / 4);
    let words_m = p * (mlp / 4);

    let Ok(mut bufs) = stack.bufs.lock() else { return false };

    // 2) Grow-only staging (prefix views keep kernels inside [0..len)).
    macro_rules! grow {
        ($field:ident, $len:expr) => {
            if bufs.$field.as_ref().is_none_or(|s| s.len() < $len) {
                match stream.alloc_zeros::<f32>($len) {
                    Ok(s) => bufs.$field = Some(s),
                    Err(_) => return false,
                }
            }
        };
    }
    grow!(x, pn);
    grow!(normx, pn);
    grow!(gate, pm);
    grow!(up, pm);
    grow!(hid, pm);
    grow!(ffnout, pn);
    grow!(gamma, n);
    if bufs
        .scratch_n
        .as_ref()
        .is_none_or(|(_, cw, _)| *cw < words_n)
    {
        match stack.mma.alloc_scratch(stream, n, p) {
            Ok(s) => bufs.scratch_n = Some((n, words_n, s)),
            Err(_) => return false,
        }
    }
    if bufs
        .scratch_mlp
        .as_ref()
        .is_none_or(|(cd, cw, _)| *cd < mlp || *cw < words_m)
    {
        match stack.mma.alloc_scratch(stream, mlp, p) {
            Ok(s) => bufs.scratch_mlp = Some((mlp, words_m, s)),
            Err(_) => return false,
        }
    }
    if bufs.out_host.len() < pn {
        bufs.out_host.resize(pn, 0.0);
    }

    let FfnBufs {
        x,
        normx,
        gate,
        up,
        hid,
        ffnout,
        gamma,
        scratch_n,
        scratch_mlp,
        out_host,
    } = &mut *bufs;
    let (Some(x), Some(normx), Some(gate), Some(up), Some(hid), Some(ffnout), Some(gamma)) =
        (x, normx, gate, up, hid, ffnout, gamma)
    else {
        return false;
    };
    let (Some((_, _, scr_n)), Some((_, _, scr_m))) = (scratch_n, scratch_mlp) else {
        return false;
    };

    // 3) Upload gamma + x.
    {
        let Some(mut g_view) = gamma.try_slice_mut(0..n) else { return false };
        if stream.memcpy_htod(gamma_f32, &mut g_view).is_err() {
            return false;
        }
    }
    {
        let Some(mut x_view) = x.try_slice_mut(0..pn) else { return false };
        if stream.memcpy_htod(x_f32, &mut x_view).is_err() {
            return false;
        }
    }
    let t_up = t0.elapsed();

    // 4) The block — all on the cudarc stream.
    let (ra, rm, rs) = canonical_rmsnorm_forms();
    let (se, sr) = canonical_swiglu_forms();
    let run = || -> Result<(), String> {
        unsafe {
            stack.ffn.launch_rmsnorm(stream, ra, rm, rs, x, gamma, normx, p, n, eps)?;
            stack
                .mma
                .launch_prefill_quantize(stream, normx, scr_n, n, p)
                .map_err(|e| e.to_string())?;
            crate::prefill_cuda_mma::launch_prefill_gemm_pair_cached(
                &stack.mma,
                stream,
                &gate_cache,
                &up_cache,
                scr_n,
                gate,
                up,
                mlp,
                n,
                p,
            )
            .map_err(|e| e.to_string())?;
            stack.ffn.launch_swiglu(stream, se, sr, gate, up, hid, pm)?;
            stack
                .mma
                .launch_prefill_quantize(stream, hid, scr_m, mlp, p)
                .map_err(|e| e.to_string())?;
            crate::prefill_cuda_mma::launch_prefill_gemm_cached(
                &stack.mma,
                stream,
                &down_cache,
                scr_m,
                ffnout,
                n,
                mlp,
                p,
            )
            .map_err(|e| e.to_string())?;
            // In-place residual: x = x + ffnout (same-index elementwise).
            stack.ffn.launch_residual(stream, x, ffnout, x, pn)?;
        }
        Ok(())
    };
    if run().is_err() {
        return false;
    }
    let t_block = t0.elapsed();

    // 5) Read back + write into the CubeCL output handle.
    let Some(x_view) = x.try_slice(0..pn) else { return false };
    if stream.memcpy_dtoh(&x_view, &mut out_host[..pn]).is_err() {
        return false;
    }
    let t_down = t0.elapsed();
    client.write(
        x_out,
        cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(&out_host[..pn]).to_vec()),
    );
    if trace {
        eprintln!(
            "[721-trace] p={p} n={n} mlp={mlp}: read {}us up {}us(+{}) block {}us(+{}) down {}us(+{}) write {}us",
            t_read.as_micros(),
            t_up.as_micros(),
            (t_up - t_read).as_micros(),
            t_block.as_micros(),
            (t_block - t_up).as_micros(),
            t_down.as_micros(),
            (t_down - t_block).as_micros(),
            t0.elapsed().as_micros(),
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue 742 T1.8 — the per-row argmax kernel (the verify tail's device
    /// token selection): GPU rows result vs CPU first-index argmax on
    /// synthetic data, including ties (the Issue-697 first-index convention)
    /// and NaN-free negative/positive mixes. Validates the NVRTC compile +
    /// the packed-u64 readback convention without the 7.1 GB model.
    #[test]
    fn argmax_rows_matches_cpu_first_index() {
        let (kernels, stream) = CudaFfnKernels::new_standalone().expect("cuda init");
        let mut rng: u32 = 0x742_718;
        // xorshift32 — deterministic, no deps. Returns the RAW u32 so callers
        // can shape the distribution (f32::from_bits(raw) would be huge/NaN).
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };

        // n = 1024 → grid_x = 4 blocks/row; rows = 5 (odd on purpose).
        let (n, rows) = (1024usize, 5usize);
        let mut values = vec![0.0f32; rows * n];
        for v in values.iter_mut() {
            *v = ((next() & 0xFFFF) as f32 / 65_535.0) - 0.5;
        }
        // Inject exact ties in every row: duplicate the max at a HIGHER
        // index (first-index must win) and at a LOWER index.
        for r in 0..rows {
            let lo = r * n + 100;
            let hi = r * n + 900;
            values[lo] = 42.0;
            values[hi] = 42.0;
        }

        let values_dev = stream
            .clone_htod(values.as_slice())
            .expect("htod values");
        let mut results_dev = stream.alloc_zeros::<u64>(rows).expect("alloc results");
        assert!(stream.memset_zeros(&mut results_dev).is_ok());
        unsafe {
            kernels
                .launch_argmax_rows(&stream, &values_dev, n, rows, &results_dev)
                .expect("launch argmax rows");
        }
        let mut host = vec![0u64; rows];
        {
            let view = results_dev.try_slice(0..rows).expect("slice results");
            stream.memcpy_dtoh(&view, &mut host).expect("dtoh results");
        }

        for r in 0..rows {
            let cpu = {
                let row = &values[r * n..(r + 1) * n];
                let mut best = 0usize;
                for (i, &x) in row.iter().enumerate() {
                    if x > row[best] {
                        best = i;
                    }
                }
                best
            };
            let gpu = !(host[r] as u32) as usize;
            assert_eq!(gpu, cpu, "row {r}: gpu {gpu} != cpu first-index {cpu}");
            assert_eq!(cpu, 100, "row {r}: tie must resolve to the first index");
        }
    }
}
