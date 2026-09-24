//! The CUDA backend of the riir forward (feature `laya-riir-cuda`,
//! non-macOS) — the `.issues/002` lane: the same op semantics as the CPU
//! lane over CUDA kernels compiled at construction via NVRTC (cudarc's
//! `compile_ptx_with_opts`, arch `sm_89` = Ada Lovelace / the 4090 box).
//!
//! Architecture — the Metal backend's contract, ported verbatim:
//! - ONE forward body: the forward code calls the [`Backend`] methods and
//!   nothing else; op order lives in `encoder` / `head` exactly once.
//! - **Device slots**: agent-owned weight slices get a PERMANENT
//!   `(ptr, len)`-keyed device copy ([`Self::weight_buf`]); activations
//!   and per-forward host-authored inputs get an EPOCH-keyed
//!   `(ptr, len, gen)` slot ([`Self::chain_buf`] for reads — miss ⇒ upload
//!   host bytes, hit ⇒ device-current by the write-first discipline;
//!   [`Self::chain_slot_for`] for destinations — miss ⇒ zero-alloc, hit ⇒
//!   the existing buffer). `begin_pass` syncs, bumps the epoch and drops
//!   the previous pass's slots, so a recycled heap address can never be
//!   served a stale epoch's bytes (the consumer repo's issue-015 class).
//!   Slots are [`Arc<CudaSlice<f32>>`] — `CudaSlice::clone()` is a
//!   device-to-device COPY in cudarc, not a refcount bump, so every cache
//!   hit hands out an `Arc` clone and every launch binds `slice.as_ref()`.
//! - **Lazy submission**: kernel launches are async on ONE stream (serial
//!   stream ordering gives the ordering the Metal lane needed a command
//!   buffer for); [`Backend::download_into`] is the host read barrier —
//!   sync the stream, then `memcpy_dtoh` the slot that an op WROTE this
//!   epoch (matched by base pointer + sufficient extent, newest epoch —
//!   the CLS row is a leading prefix of the hidden slot, so the copy takes
//!   a `CudaView` of the slot's first `src.len()` elements).
//! - **The GEMM**: ONE strided batched kernel — `C[z] = A[z] @ B[z]` with
//!   runtime strides `(a_rs, a_cs, b_rs, b_cs)` + per-batch element
//!   offsets `(a_bs, b_bs, c_bs)` and base element offsets, grid
//!   `(⌈n/64⌉, ⌈m/64⌉, batch)`, 64×64×32 tiles staged in shared memory,
//!   512 threads, fp32 FMA accumulate (NVRTC's default `fmad=true`).
//!   Every backend matmul shape (plain / weight / kt / batched-heads) is
//!   the SAME kernel at different stride tables. The weight operand binds
//!   row-major `[n, k]` DIRECTLY as the `[k, n]` operand (strides
//!   `b_rs=1, b_cs=k`) — the B-tile loader maps consecutive lanes along
//!   whichever stride is 1, so no device-side transpose cache exists
//!   (unlike Metal's simdgroup layout constraint, `weight_t_buf`).
//! - **Attention v1**: the trait DEFAULT op sequence (split → rope →
//!   scale → batched kt-scores → mask broadcast → softmax → batched value
//!   mix → merge), every step a device kernel — attention is <6 % of
//!   forward FLOPs at the pinned geometries (the projections dominate:
//!   ~217 GFLOP/forward at seq 317), so a fused flash kernel is a measured
//!   follow-up rung, not the landing gate. `needs_window_mask` stays
//!   `true` (the default path consumes the mask tensor).
//!
//! Numerics: fp32 throughout; `erff` / `expf` / `sqrtf` are CUDA's precise
//! device intrinsics (≤2 ulp — the same class as Metal's `precise::exp`
//! and the A&S erf the MSL lane transcribed, both inside the G5 drift
//! budget); LayerNorm mirrors the CPU op order exactly (f64-rounded `1/d`
//! mean scale, division by `sqrtf(var + eps)`); softmax normalizes by
//! multiply-with-reciprocal (the Metal lane's form); gelu is the CPU
//! lane's exact multiply order `(erff(v/√2) + 1) · 0.5 · v`. The G5 gate
//! at the cuda posture (top-1 ≥ 99.9 %, p-drift ≤ 1e-3) is the standing
//! correctness authority — no number from this lane is published before
//! it is green at this posture.
//!
//! Debug: `LAYA_CUDA_TRACE=1` logs every chain-cache miss; the contract
//! notes of `metal.rs`'s module doc (write-first audit, the weights-cache
//! agent-lifetime addressing discipline) carry over unchanged.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;

use super::backend::Backend;
use crate::laya::LayaError;

/// The sgemm tile: 64×64 output per block, BK 32, 512 threads. Staging =
/// A [64][33] + B [32][65] = 4192 floats = 16 768 B — under the 48 KB
/// dynamic-smem default (no `cudaFuncAttributeMaxDynamicSharedMemorySize`
/// opt-in needed).
const BM: u32 = 64;
const BN: u32 = 64;
const BK: u32 = 32;
const SGEMM_THREADS: u32 = 512;
const SGEMM_SMEM_BYTES: u32 = (BM * 33 + BK * 65) * 4;

/// The row kernels (LN / softmax) run 256 threads per row.
const ROW_THREADS: u32 = 256;

/// Elementwise default block size.
const EW_THREADS: u32 = 256;

const KERNELS: &[&str] = &[
    "sgemm",
    "add",
    "copy_f",
    "add_bias_row",
    "scale",
    "relu",
    "gelu_erf",
    "glu_gelu_gate",
    "ln_rows",
    "softmax_rows",
    "rope",
    "split_heads",
    "merge_heads",
    "gather_rows",
    "add_mask_bct",
];

/// The CUDA C source. Self-contained (NVRTC — no host headers). Sizes fit
/// u32 (every pinned extent < 2³¹); every op is the CPU lane's semantics.
/// `extern "C"` keeps the PTX entry names exactly as [`KERNELS`] lists
/// them (no C++ mangling to fight at `load_function`).
const CUDA_SRC: &str = r#"
#define BM 64
#define BN 64
#define BK 32

// ── sgemm: C[z] = A[z] @ B[z], runtime strides + base/batch offsets. ─────
// One 64×64 tile per block, 512 threads (16 warps in a 4×4 grid); each
// thread owns a 2×4 output fragment (rows tr / tr+8 inside its warp's
// 16×16 tile, cols tc..tc+3). Staging: A [64][33] + B [32][65].
// The B loader maps consecutive lanes along whichever B stride is 1, so a
// row-major [n, k] weight binds DIRECTLY as the [k, n] operand (strides
// b_rs=1, b_cs=k) with fully coalesced staging — no device transpose.
extern "C" __global__ void sgemm(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    const unsigned int a_off,
    const unsigned int b_off,
    const unsigned int c_off,
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const unsigned int a_rs,
    const unsigned int a_cs,
    const unsigned int b_rs,
    const unsigned int b_cs,
    const unsigned int a_bs,
    const unsigned int b_bs,
    const unsigned int c_bs)
{
    __shared__ float ta[BM * 33];
    __shared__ float tb[BK * 65];

    const unsigned int m0 = blockIdx.y * BM;
    const unsigned int n0 = blockIdx.x * BN;
    const float* A = a + a_off + (size_t)blockIdx.z * a_bs;
    const float* B = b + b_off + (size_t)blockIdx.z * b_bs;
    float* C = c + c_off + (size_t)blockIdx.z * c_bs;

    const unsigned int tid = threadIdx.x;
    // Warp grid 4×4: each warp owns a 16×16 sub-tile; within a warp,
    // 8 lane-rows × 4 lane-col-quads — thread rows tr / tr+8, cols
    // tc..tc+3 (8 accumulators per thread × 512 threads = 64×64).
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int wr = warp >> 2;          // warp row 0..3
    const unsigned int wc = warp & 3u;          // warp col 0..3
    const unsigned int tr = lane >> 2;          // 0..7
    const unsigned int tc = (lane & 3u) * 4u;   // 0, 4, 8, 12
    const unsigned int row0 = wr * 16u + tr;
    const unsigned int col0 = wc * 16u + tc;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (unsigned int t = 0; t < k; t += BK) {
        __syncthreads();
        // Stage A [64][32]: 4 elements per thread (2048 / 512), lanes
        // along the column (A row-major — a_cs is 1 at every call site).
        for (unsigned int q = 0; q < 4; ++q) {
            const unsigned int idx = tid + q * 512u;
            const unsigned int r = idx >> 5;
            const unsigned int col = idx & 31u;
            const unsigned int gr = m0 + r;
            const unsigned int gc = t + col;
            ta[r * 33u + col] =
                (gr < m && gc < k) ? A[(size_t)gr * a_rs + gc * a_cs] : 0.0f;
        }
        // Stage B [32][64]: 4 per thread. b_cs==1 (row-major [k][n]):
        // lanes along n — contiguous. b_rs==1 (kt / W [n,k] bound as
        // [k,n]): lanes along k — contiguous in the weight's memory.
        if (b_cs == 1u) {
            for (unsigned int q = 0; q < 4; ++q) {
                const unsigned int idx = tid + q * 512u;
                const unsigned int kk = idx >> 6;
                const unsigned int col = idx & 63u;
                const unsigned int gk = t + kk;
                const unsigned int gn = n0 + col;
                tb[kk * 65u + col] =
                    (gk < k && gn < n) ? B[(size_t)gk * b_rs + gn] : 0.0f;
            }
        } else {
            for (unsigned int q = 0; q < 4; ++q) {
                const unsigned int idx = tid + q * 512u;
                const unsigned int kk = idx & 31u;
                const unsigned int col = idx >> 5;
                const unsigned int gk = t + kk;
                const unsigned int gn = n0 + col;
                tb[kk * 65u + col] =
                    (gk < k && gn < n) ? B[(size_t)gn * b_cs + gk * b_rs] : 0.0f;
            }
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int kk = 0; kk < BK; ++kk) {
            const float a0 = ta[row0 * 33u + kk];
            const float a1 = ta[(row0 + 8u) * 33u + kk];
            const float b0 = tb[kk * 65u + col0];
            const float b1 = tb[kk * 65u + col0 + 1u];
            const float b2 = tb[kk * 65u + col0 + 2u];
            const float b3 = tb[kk * 65u + col0 + 3u];
            acc0 += a0 * b0; acc1 += a0 * b1; acc2 += a0 * b2; acc3 += a0 * b3;
            acc4 += a1 * b0; acc5 += a1 * b1; acc6 += a1 * b2; acc7 += a1 * b3;
        }
    }
    const unsigned int gr0 = m0 + row0;
    const unsigned int gc0 = n0 + col0;
    if (gr0 < m) {
        if (gc0 + 0u < n) C[(size_t)gr0 * n + gc0 + 0u] = acc0;
        if (gc0 + 1u < n) C[(size_t)gr0 * n + gc0 + 1u] = acc1;
        if (gc0 + 2u < n) C[(size_t)gr0 * n + gc0 + 2u] = acc2;
        if (gc0 + 3u < n) C[(size_t)gr0 * n + gc0 + 3u] = acc3;
    }
    if (gr0 + 8u < m) {
        if (gc0 + 0u < n) C[(size_t)(gr0 + 8u) * n + gc0 + 0u] = acc4;
        if (gc0 + 1u < n) C[(size_t)(gr0 + 8u) * n + gc0 + 1u] = acc5;
        if (gc0 + 2u < n) C[(size_t)(gr0 + 8u) * n + gc0 + 2u] = acc6;
        if (gc0 + 3u < n) C[(size_t)(gr0 + 8u) * n + gc0 + 3u] = acc7;
    }
}

// ── block reductions (canonical: full-warp shuffles, warp-0 second
// stage, smem broadcast). scratch holds 8 partials + the broadcast slot.

__device__ __forceinline__ float warp_sum(float v)
{
    #pragma unroll
    for (unsigned int o = 16u; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_sum_bcast(float v, float* scratch)
{
    v = warp_sum(v);
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    if (lane == 0u) scratch[warp] = v;
    __syncthreads();
    if (warp == 0u) {
        v = (lane < 8u) ? scratch[lane] : 0.0f;
        v = warp_sum(v); // lanes 8..31 hold 0 — the tree still covers 0..7
        if (lane == 0u) scratch[8] = v;
    }
    __syncthreads();
    return scratch[8];
}

// ── elementwise tail — the MSL_TAIL semantics, verbatim ──────────────────

// x[x_off + i] += y[y_off + i].
extern "C" __global__ void add(
    float* __restrict__ x, const unsigned int x_off,
    const float* __restrict__ y, const unsigned int y_off,
    const unsigned int len)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) x[x_off + i] += y[y_off + i];
}

// dst[dst_off + i] = src[src_off + i]  (`copy` is a reserved word class
// in some toolchains' auto-generated headers — spelled `copy_f`).
extern "C" __global__ void copy_f(
    float* __restrict__ dst, const unsigned int dst_off,
    const float* __restrict__ src, const unsigned int src_off,
    const unsigned int len)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) dst[dst_off + i] = src[src_off + i];
}

extern "C" __global__ void add_bias_row(
    float* __restrict__ x, const float* __restrict__ bias,
    const unsigned int len, const unsigned int d)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) x[i] += bias[i % d];
}

extern "C" __global__ void scale(
    float* __restrict__ x, const unsigned int len, const float s)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) x[i] *= s;
}

extern "C" __global__ void relu(
    float* __restrict__ x, const unsigned int len)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len && x[i] < 0.0f) x[i] = 0.0f;
}

// The CPU lane's exact op order: (erff(v/√2) + 1) · 0.5 · v.
extern "C" __global__ void gelu_erf(
    float* __restrict__ x, const unsigned int len)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        const float e = erff(x[i] * 0.70710678118654752440f);
        x[i] = (e + 1.0f) * 0.5f * x[i];
    }
}

// out[r, j] = gelu_erf(fused[r, j]) · fused[r, I + j].
extern "C" __global__ void glu_gelu_gate(
    const float* __restrict__ fused, float* __restrict__ out,
    const unsigned int rows, const unsigned int i_sz)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * i_sz) return;
    const unsigned int r = i / i_sz;
    const unsigned int j = i % i_sz;
    const unsigned int base = r * 2u * i_sz;
    const float e = erff(fused[base + j] * 0.70710678118654752440f);
    const float act = (e + 1.0f) * 0.5f * fused[base + j];
    out[i] = act * fused[base + i_sz + j];
}

// One block per row: mean → centered² mean → 1/sqrt(var+eps) →
// (v−mean)·inv·w — the CPU op order; inv_d is the f64-rounded 1/d.
extern "C" __global__ void ln_rows(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ out,
    const unsigned int rows, const unsigned int d,
    const float inv_d, const float eps)
{
    __shared__ float scratch[9];
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * d;
    float* orow = out + (size_t)row * d;
    const unsigned int t = threadIdx.x;

    float acc = 0.0f;
    for (unsigned int i = t; i < d; i += 256u) acc += xr[i];
    const float mean = block_sum_bcast(acc, scratch) * inv_d;

    float vacc = 0.0f;
    for (unsigned int i = t; i < d; i += 256u) {
        const float c = xr[i] - mean;
        vacc += c * c;
    }
    const float var = block_sum_bcast(vacc, scratch) * inv_d;
    const float inv = 1.0f / sqrtf(var + eps);
    for (unsigned int i = t; i < d; i += 256u)
        orow[i] = (xr[i] - mean) * inv * w[i];
}

// One block per row: max → exp → sum → multiply by 1/sum (the Metal
// lane's reciprocal-multiply normalize).
extern "C" __global__ void softmax_rows(
    float* __restrict__ x, const unsigned int rows, const unsigned int n)
{
    __shared__ float scratch[9];
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    float* xr = x + (size_t)row * n;
    const unsigned int t = threadIdx.x;

    float mx = -3.402823466e+38f;
    for (unsigned int i = t; i < n; i += 256u) mx = fmaxf(mx, xr[i]);
    #pragma unroll
    for (unsigned int o = 16u; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, o));
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    if (lane == 0u) scratch[warp] = mx;
    __syncthreads();
    if (warp == 0u) {
        float v = (lane < 8u) ? scratch[lane] : -3.402823466e+38f;
        #pragma unroll
        for (unsigned int o = 4u; o > 0; o >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
        if (lane == 0u) scratch[8] = v;
    }
    __syncthreads();
    mx = scratch[8];

    float sum = 0.0f;
    for (unsigned int i = t; i < n; i += 256u) {
        const float e = expf(xr[i] - mx);
        xr[i] = e;
        sum += e;
    }
    const float inv = 1.0f / block_sum_bcast(sum, scratch);
    for (unsigned int i = t; i < n; i += 256u) xr[i] *= inv;
}

// Rotate-half RoPE on [heads, seq, hd]: o[j] = q1·c − q2·s,
// o[j+half] = q2·c + q1·s — one thread per (h·seq+pos, j) pair.
extern "C" __global__ void rope(
    float* __restrict__ q,
    const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    const unsigned int seq, const unsigned int heads, const unsigned int hd)
{
    const unsigned int hf = hd / 2u;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= heads * seq * hf) return;
    const unsigned int hps = gid / hf;
    const unsigned int j = gid % hf;
    const unsigned int base = hps * hd;
    const unsigned int prow = (hps % seq) * hd;
    const float c = cos_t[prow + j];
    const float s = sin_t[prow + j];
    const float q1 = q[base + j];
    const float q2 = q[base + hf + j];
    q[base + j] = q1 * c - q2 * s;
    q[base + hf + j] = q2 * c + q1 * s;
}

extern "C" __global__ void split_heads(
    const float* __restrict__ src, float* __restrict__ out,
    const unsigned int row_stride, const unsigned int off,
    const unsigned int seq, const unsigned int heads, const unsigned int hd)
{
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= heads * seq * hd) return;
    const unsigned int h = gid / (seq * hd);
    const unsigned int r = gid % (seq * hd);
    const unsigned int s = r / hd;
    const unsigned int i = r % hd;
    out[gid] = src[(size_t)s * row_stride + off + h * hd + i];
}

extern "C" __global__ void merge_heads(
    const float* __restrict__ src, float* __restrict__ out,
    const unsigned int seq, const unsigned int heads, const unsigned int hd)
{
    const unsigned int d = heads * hd;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= seq * d) return;
    const unsigned int s = gid / d;
    const unsigned int rem = gid % d;
    const unsigned int h = rem / hd;
    const unsigned int i = rem % hd;
    out[gid] = src[(size_t)(h * seq + s) * hd + i];
}

extern "C" __global__ void gather_rows(
    const float* __restrict__ x, const unsigned int* __restrict__ rows,
    float* __restrict__ out,
    const unsigned int d, const unsigned int nrows)
{
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nrows * d) return;
    const unsigned int r = gid / d;
    const unsigned int i = gid % d;
    out[gid] = x[(size_t)rows[r] * d + i];
}

// scores[r] += mask[r % mlen] — the sliding-window mask broadcast over
// every head's slab of the scores parent in ONE dispatch.
extern "C" __global__ void add_mask_bct(
    float* __restrict__ x, const float* __restrict__ mask,
    const unsigned int len, const unsigned int mlen)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) x[i] += mask[i % mlen];
}
"#;

fn rt(msg: impl Into<String>) -> LayaError {
    LayaError::Runtime(msg.into())
}

/// The permanent `(ptr, len)`-keyed weight cache.
type WeightMap = HashMap<(usize, usize), Arc<CudaSlice<f32>>>;
/// The epoch-keyed `(ptr, len, gen)` activation slot cache.
type ChainMap = HashMap<(usize, usize, u64), Arc<CudaSlice<f32>>>;

fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LAYA_CUDA_TRACE").as_deref() == Ok("1"))
}

fn next_trace_instance() -> usize {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The CUDA backend: one context + stream, the compiled-once kernel set,
/// the permanent weight cache, and the generation-keyed activation cache.
pub struct Cuda {
    stream: Arc<CudaStream>,
    kernels: HashMap<&'static str, CudaFunction>,
    /// `(ptr, len)` → device copy for the agent-OWNED weight slices
    /// (stable addresses and contents for the agent's lifetime — every
    /// `matmul_w` weight, LN scales, biases; the agent-lifetime addressing
    /// contract of `metal.rs` carries over unchanged).
    weights: Mutex<WeightMap>,
    /// `(ptr, len, gen)` → device slot for activations and per-forward
    /// host-authored inputs; the generation (bumped at every pass) makes a
    /// recycled heap address MISS instead of serving a stale epoch's bytes.
    chain: Mutex<ChainMap>,
    /// The pass generation (how many `begin_pass` calls have run).
    epoch: AtomicU64,
    /// Debug-trace instance id.
    trace_id: usize,
}

impl Cuda {
    /// Build the backend — fails loud when no CUDA device exists or NVRTC
    /// cannot compile the source (never a silent CPU fallback; the
    /// `LAYA_DEVICE=metal` precedent).
    pub fn new() -> Result<Self, LayaError> {
        let ctx = CudaContext::new(0).map_err(|e| rt(format!("CUDA init: {e}")))?;
        let stream = ctx
            .new_stream()
            .map_err(|e| rt(format!("CUDA stream: {e}")))?;
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| rt(format!("CUDA nvrtc compile: {e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| rt(format!("CUDA module load: {e}")))?;
        let mut kernels = HashMap::with_capacity(KERNELS.len());
        for name in KERNELS {
            let f = module
                .load_function(name)
                .map_err(|e| rt(format!("CUDA kernel {name}: {e}")))?;
            kernels.insert(*name, f);
        }
        Ok(Self {
            stream,
            kernels,
            weights: Mutex::new(WeightMap::new()),
            chain: Mutex::new(ChainMap::new()),
            epoch: AtomicU64::new(0),
            trace_id: next_trace_instance(),
        })
    }

    fn kernel(&self, name: &'static str) -> CudaFunction {
        self.kernels.get(name).cloned().unwrap_or_else(|| {
            panic!("cuda kernel {name} missing — constructor loaded every KERNELS entry")
        })
    }

    /// Device-resident copy of an agent-owned weight slice — permanent
    /// cache, first-miss upload, never invalidated.
    fn weight_buf(&self, data: &[f32]) -> Arc<CudaSlice<f32>> {
        let key = (data.as_ptr() as usize, data.len());
        let mut map = self.weights.lock().expect("weight cache poison");
        if let Some(b) = map.get(&key) {
            return Arc::clone(b);
        }
        let b = self
            .stream
            .clone_htod(data)
            .unwrap_or_else(|e| panic!("cuda weight upload: {e}"));
        let b = Arc::new(b);
        map.insert(key, Arc::clone(&b));
        b
    }

    /// Activation / per-forward input: hit within the current epoch → the
    /// device copy is current (no copy); miss → create + upload host bytes.
    fn chain_buf(&self, data: &[f32]) -> Arc<CudaSlice<f32>> {
        let epoch = self.epoch.load(Ordering::Relaxed);
        let key = (data.as_ptr() as usize, data.len(), epoch);
        let mut map = self.chain.lock().expect("chain cache poison");
        if let Some(b) = map.get(&key) {
            return Arc::clone(b);
        }
        let b = self
            .stream
            .clone_htod(data)
            .unwrap_or_else(|e| panic!("cuda chain upload: {e}"));
        let b = Arc::new(b);
        map.insert(key, Arc::clone(&b));
        if trace_enabled() {
            eprintln!(
                "[trace] cuda chain MISS inst {} ptr {:p} len {} epoch {epoch}",
                self.trace_id,
                data.as_ptr(),
                data.len()
            );
        }
        b
    }

    /// A device destination slot for this epoch's `(ptr, len)` — write
    /// first, so a hit reuses the existing buffer.
    fn chain_slot_for(&self, dst: &[f32]) -> Arc<CudaSlice<f32>> {
        let epoch = self.epoch.load(Ordering::Relaxed);
        let key = (dst.as_ptr() as usize, dst.len(), epoch);
        let mut map = self.chain.lock().expect("chain cache poison");
        if let Some(b) = map.get(&key) {
            return Arc::clone(b);
        }
        let b = self
            .stream
            .alloc_zeros::<f32>(dst.len().max(1))
            .unwrap_or_else(|e| panic!("cuda slot alloc: {e}"));
        let b = Arc::new(b);
        map.insert(key, Arc::clone(&b));
        b
    }

    /// One forward is beginning: drain outstanding GPU work (a dropped
    /// slot's device memory must never be freed under a live kernel),
    /// bump the pass epoch, and drop the previous pass's slots
    /// (host-authored buffers are rebuilt per forward, often at recycled
    /// heap addresses — last pass's keys must never hit).
    fn begin_pass_impl(&self) {
        self.stream
            .synchronize()
            .unwrap_or_else(|e| panic!("cuda begin_pass sync: {e}"));
        self.epoch.fetch_add(1, Ordering::Relaxed);
        self.chain.lock().expect("chain cache poison").clear();
    }

    /// The batched GEMM dispatch — `uargs` is the Metal lane's table
    /// `[m, n, k, a_rs, a_cs, b_rs, b_cs, a_bs, b_bs, c_bs]` (element
    /// strides, then per-batch element strides; batch-1 callers pass
    /// zeros). Trait-method element offsets ride the explicit off args.
    #[allow(clippy::too_many_arguments)]
    fn run_sgemm(
        &self,
        a: &CudaSlice<f32>,
        a_off: u32,
        b: &CudaSlice<f32>,
        b_off: u32,
        out: &CudaSlice<f32>,
        c_off: u32,
        uargs: &[u32; 10],
        batch: u32,
    ) {
        let f = self.kernel("sgemm");
        let (m, n) = (uargs[0], uargs[1]);
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(BN), m.div_ceil(BM), batch.max(1)),
            block_dim: (SGEMM_THREADS, 1, 1),
            shared_mem_bytes: SGEMM_SMEM_BYTES,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(a)
                .arg(b)
                .arg(out)
                .arg(&a_off)
                .arg(&b_off)
                .arg(&c_off)
                .arg(&uargs[0])
                .arg(&uargs[1])
                .arg(&uargs[2])
                .arg(&uargs[3])
                .arg(&uargs[4])
                .arg(&uargs[5])
                .arg(&uargs[6])
                .arg(&uargs[7])
                .arg(&uargs[8])
                .arg(&uargs[9])
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda sgemm launch: {e}"));
        }
    }
}

impl Backend for Cuda {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn matmul(
        &self,
        a: &[f32],
        a_off: usize,
        m: usize,
        k: usize,
        b: &[f32],
        b_off: usize,
        n: usize,
        dst: &mut [f32],
        dst_off: usize,
    ) {
        assert!(a.len() >= m * k + a_off, "lhs extent");
        assert!(b.len() >= k * n + b_off, "rhs extent");
        assert!(dst.len() >= m * n + dst_off, "dst extent");
        let ab = self.chain_buf(a);
        let bb = self.chain_buf(b);
        let ob = self.chain_slot_for(dst);
        self.run_sgemm(
            ab.as_ref(),
            a_off as u32,
            bb.as_ref(),
            b_off as u32,
            ob.as_ref(),
            dst_off as u32,
            &[m as u32, n as u32, k as u32, k as u32, 1, n as u32, 1, 0, 0, 0],
            1,
        );
    }

    fn matmul_w(&self, a: &[f32], m: usize, k: usize, w: &[f32], n: usize, dst: &mut [f32]) {
        assert_eq!(a.len(), m * k, "lhs extent");
        assert_eq!(w.len(), n * k, "weight extent");
        assert_eq!(dst.len(), m * n, "dst extent");
        let ab = self.chain_buf(a);
        // W binds row-major [n, k] DIRECTLY as the [k, n] operand (strides
        // b_rs=1, b_cs=k) — the B-tile loader's coalesced `b_rs == 1` path.
        // No device transpose (the simdgroup layout constraint is Metal's).
        let wb = self.weight_buf(w);
        let ob = self.chain_slot_for(dst);
        self.run_sgemm(
            ab.as_ref(),
            0,
            wb.as_ref(),
            0,
            ob.as_ref(),
            0,
            &[m as u32, n as u32, k as u32, k as u32, 1, 1, k as u32, 0, 0, 0],
            1,
        );
    }

    fn matmul_kt(
        &self,
        q: &[f32],
        q_off: usize,
        m: usize,
        hd: usize,
        k: &[f32],
        k_off: usize,
        dst: &mut [f32],
        dst_off: usize,
    ) {
        assert!(q.len() >= m * hd + q_off, "q extent");
        assert!(k.len() >= m * hd + k_off, "k extent");
        assert!(dst.len() >= m * m + dst_off, "dst extent");
        let qb = self.chain_buf(q);
        let kb = self.chain_buf(k);
        let ob = self.chain_slot_for(dst);
        self.run_sgemm(
            qb.as_ref(),
            q_off as u32,
            kb.as_ref(),
            k_off as u32,
            ob.as_ref(),
            dst_off as u32,
            &[
                m as u32,
                m as u32,
                hd as u32,
                hd as u32, // a_rs
                1,         // a_cs
                1,         // b_rs — B = Kᵀ, K row-major [m, hd]
                hd as u32, // b_cs
                0,
                0,
                0,
            ],
            1,
        );
    }

    fn matmul_kt_heads(
        &self,
        q: &[f32],
        k: &[f32],
        heads: usize,
        m: usize,
        hd: usize,
        dst: &mut [f32],
    ) {
        assert_eq!(q.len(), heads * m * hd, "q extent");
        assert_eq!(k.len(), heads * m * hd, "k extent");
        assert_eq!(dst.len(), heads * m * m, "dst extent");
        let qb = self.chain_buf(q);
        let kb = self.chain_buf(k);
        let ob = self.chain_slot_for(dst);
        self.run_sgemm(
            qb.as_ref(),
            0,
            kb.as_ref(),
            0,
            ob.as_ref(),
            0,
            &[
                m as u32,
                m as u32,
                hd as u32,
                hd as u32,       // a_rs
                1,               // a_cs
                1,               // b_rs
                hd as u32,       // b_cs
                (m * hd) as u32, // a_bs
                (m * hd) as u32, // b_bs
                (m * m) as u32,  // c_bs
            ],
            heads as u32,
        );
    }

    fn matmul_heads(
        &self,
        a: &[f32],
        b: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
        dst: &mut [f32],
    ) {
        assert_eq!(a.len(), heads * m * k, "a extent");
        assert_eq!(b.len(), heads * k * n, "b extent");
        assert_eq!(dst.len(), heads * m * n, "dst extent");
        let ab = self.chain_buf(a);
        let bb = self.chain_buf(b);
        let ob = self.chain_slot_for(dst);
        self.run_sgemm(
            ab.as_ref(),
            0,
            bb.as_ref(),
            0,
            ob.as_ref(),
            0,
            &[
                m as u32,
                n as u32,
                k as u32,
                k as u32,       // a_rs
                1,              // a_cs
                n as u32,       // b_rs
                1,              // b_cs
                (m * k) as u32, // a_bs
                (k * n) as u32, // b_bs
                (m * n) as u32, // c_bs
            ],
            heads as u32,
        );
    }

    // attention_forward: the trait DEFAULT op sequence (split → rope →
    // scale → scores → mask → softmax → mix → merge) runs entirely
    // device-side through the ops above — the `.issues/002` v1 posture.
    // A fused flash kernel is a measured follow-up rung.

    fn add(&self, x: &mut [f32], x_off: usize, y: &[f32], y_off: usize, len: usize) {
        assert!(x.len() >= len + x_off, "add x extent");
        assert!(y.len() >= len + y_off, "add y extent");
        let xb = self.chain_buf(x);
        let yb = self.chain_buf(y);
        let f = self.kernel("add");
        let groups = (len as u32).div_ceil(EW_THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (EW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(xb.as_ref())
                .arg(&(x_off as u32))
                .arg(yb.as_ref())
                .arg(&(y_off as u32))
                .arg(&(len as u32))
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda add launch: {e}"));
        }
    }

    fn add_mask_broadcast(&self, x: &mut [f32], mask: &[f32], heads: usize) {
        let mlen = mask.len();
        assert!(heads > 0, "heads");
        assert_eq!(x.len(), heads * mlen, "mask broadcast extent");
        let xb = self.chain_buf(x);
        let mb = self.chain_buf(mask);
        let len = x.len();
        let f = self.kernel("add_mask_bct");
        let groups = (len as u32).div_ceil(EW_THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (EW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(xb.as_ref())
                .arg(mb.as_ref())
                .arg(&(len as u32))
                .arg(&(mlen as u32))
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda add_mask_bct launch: {e}"));
        }
    }

    fn add_bias_row(&self, x: &mut [f32], d: usize, bias: &[f32]) {
        assert_eq!(bias.len(), d, "bias extent");
        assert_eq!(x.len() % d, 0, "row extent");
        let xb = self.chain_buf(x);
        let bb = self.weight_buf(bias);
        let f = self.kernel("add_bias_row");
        let len = x.len() as u32;
        let groups = len.div_ceil(EW_THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (EW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(xb.as_ref())
                .arg(bb.as_ref())
                .arg(&len)
                .arg(&(d as u32))
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda add_bias_row launch: {e}"));
        }
    }

    fn scale(&self, x: &mut [f32], s: f32) {
        let xb = self.chain_buf(x);
        let f = self.kernel("scale");
        let len = x.len() as u32;
        let groups = len.div_ceil(EW_THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (EW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(xb.as_ref())
                .arg(&len)
                .arg(&s)
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda scale launch: {e}"));
        }
    }

    fn layer_norm_nobias_into(
        &self,
        x: &[f32],
        w: &[f32],
        eps: f32,
        d: usize,
        _sq: &mut Vec<f32>,
        out: &mut [f32],
    ) {
        assert_eq!(w.len(), d, "norm extent");
        assert_eq!(x.len(), out.len(), "ln extent");
        assert_eq!(x.len() % d, 0, "row extent");
        let inv_d = (1f64 / d as f64) as f32;
        let rows = x.len() / d;
        let xb = self.chain_buf(x);
        let wb = self.weight_buf(w);
        let ob = self.chain_slot_for(out);
        let f = self.kernel("ln_rows");
        let cfg = LaunchConfig {
            grid_dim: (rows.max(1) as u32, 1, 1),
            block_dim: (ROW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(xb.as_ref())
                .arg(wb.as_ref())
                .arg(ob.as_ref())
                .arg(&(rows as u32))
                .arg(&(d as u32))
                .arg(&inv_d)
                .arg(&eps)
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda ln_rows launch: {e}"));
        }
    }

    fn softmax_rows(&self, x: &mut [f32], n: usize) {
        assert_eq!(x.len() % n, 0, "softmax row extent");
        let rows = x.len() / n;
        let xb = self.chain_buf(x);
        let f = self.kernel("softmax_rows");
        let cfg = LaunchConfig {
            grid_dim: (rows.max(1) as u32, 1, 1),
            block_dim: (ROW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(xb.as_ref())
                .arg(&(rows as u32))
                .arg(&(n as u32))
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda softmax_rows launch: {e}"));
        }
    }

    fn relu(&self, x: &mut [f32]) {
        let xb = self.chain_buf(x);
        let f = self.kernel("relu");
        let len = x.len() as u32;
        let groups = len.div_ceil(EW_THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (EW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(xb.as_ref())
                .arg(&len)
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda relu launch: {e}"));
        }
    }

    fn gelu_erf(&self, x: &mut [f32]) {
        let xb = self.chain_buf(x);
        let f = self.kernel("gelu_erf");
        let len = x.len() as u32;
        let groups = len.div_ceil(EW_THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (EW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(xb.as_ref())
                .arg(&len)
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda gelu_erf launch: {e}"));
        }
    }

    fn glu_gelu_gate(&self, fused: &[f32], rows: usize, i_sz: usize, out: &mut [f32]) {
        assert_eq!(fused.len(), rows * 2 * i_sz, "fused extent");
        assert_eq!(out.len(), rows * i_sz, "glu out extent");
        let fb = self.chain_buf(fused);
        let ob = self.chain_slot_for(out);
        let f = self.kernel("glu_gelu_gate");
        let total = (rows * i_sz) as u32;
        let groups = total.div_ceil(EW_THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (EW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(fb.as_ref())
                .arg(ob.as_ref())
                .arg(&(rows as u32))
                .arg(&(i_sz as u32))
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda glu_gelu_gate launch: {e}"));
        }
    }

    fn apply_rope(
        &self,
        q: &mut [f32],
        seq: usize,
        heads: usize,
        hd: usize,
        cos: &[f32],
        sin: &[f32],
    ) {
        let half = hd / 2;
        assert_eq!(q.len(), heads * seq * hd, "q extent");
        assert_eq!(cos.len(), seq * hd, "cos extent");
        assert_eq!(sin.len(), seq * hd, "sin extent");
        let qb = self.chain_buf(q);
        let cb = self.chain_buf(cos);
        let sb = self.chain_buf(sin);
        let f = self.kernel("rope");
        let total = (heads * seq * half) as u32;
        let groups = total.div_ceil(EW_THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (EW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(qb.as_ref())
                .arg(cb.as_ref())
                .arg(sb.as_ref())
                .arg(&(seq as u32))
                .arg(&(heads as u32))
                .arg(&(hd as u32))
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda rope launch: {e}"));
        }
    }

    fn split_heads(
        &self,
        src: &[f32],
        row_stride: usize,
        off: usize,
        seq: usize,
        heads: usize,
        hd: usize,
        out: &mut [f32],
    ) {
        assert_eq!(out.len(), heads * seq * hd, "split extent");
        let sb = self.chain_buf(src);
        let ob = self.chain_slot_for(out);
        let f = self.kernel("split_heads");
        let total = (heads * seq * hd) as u32;
        let groups = total.div_ceil(EW_THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (EW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(sb.as_ref())
                .arg(ob.as_ref())
                .arg(&(row_stride as u32))
                .arg(&(off as u32))
                .arg(&(seq as u32))
                .arg(&(heads as u32))
                .arg(&(hd as u32))
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda split_heads launch: {e}"));
        }
    }

    fn merge_heads(&self, src: &[f32], seq: usize, heads: usize, hd: usize, out: &mut [f32]) {
        let d = heads * hd;
        assert_eq!(out.len(), seq * d, "merge extent");
        let sb = self.chain_buf(src);
        let ob = self.chain_slot_for(out);
        let f = self.kernel("merge_heads");
        let total = (seq * d) as u32;
        let groups = total.div_ceil(EW_THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (EW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(sb.as_ref())
                .arg(ob.as_ref())
                .arg(&(seq as u32))
                .arg(&(heads as u32))
                .arg(&(hd as u32))
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda merge_heads launch: {e}"));
        }
    }

    fn gather_rows(&self, x: &[f32], d: usize, rows: &[usize], out: &mut [f32]) {
        assert_eq!(out.len(), rows.len() * d, "gather extent");
        let xb = self.chain_buf(x);
        let u32s: Vec<u32> = rows.iter().map(|&r| r as u32).collect();
        let rb = self
            .stream
            .clone_htod(&u32s)
            .unwrap_or_else(|e| panic!("cuda gather rows upload: {e}"));
        let ob = self.chain_slot_for(out);
        let f = self.kernel("gather_rows");
        let total = out.len() as u32;
        let groups = total.div_ceil(EW_THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (EW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(xb.as_ref())
                .arg(&rb)
                .arg(ob.as_ref())
                .arg(&(d as u32))
                .arg(&(rows.len() as u32))
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda gather_rows launch: {e}"));
        }
    }

    fn copy_into(&self, src: &[f32], dst: &mut [f32]) {
        assert_eq!(src.len(), dst.len(), "copy extent");
        let sb = self.chain_buf(src);
        let db = self.chain_slot_for(dst);
        let f = self.kernel("copy_f");
        let len = dst.len() as u32;
        let groups = len.div_ceil(EW_THREADS).max(1);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (EW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(db.as_ref())
                .arg(&0u32)
                .arg(sb.as_ref())
                .arg(&0u32)
                .arg(&len)
                .launch(cfg)
                .unwrap_or_else(|e| panic!("cuda copy_f launch: {e}"));
        }
    }

    fn begin_pass(&self) {
        self.begin_pass_impl();
    }

    fn download_into(&self, src: &[f32], out: &mut [f32]) {
        assert!(src.len() <= out.len(), "download extent");
        self.stream
            .synchronize()
            .unwrap_or_else(|e| panic!("cuda download sync: {e}"));
        let ptr = src.as_ptr() as usize;
        let map = self.chain.lock().expect("chain cache poison");
        // The src may be a PREFIX of the written buffer (the CLS row is the
        // leading d of the whole hidden slot), so match by base pointer and
        // sufficient extent, taking the newest epoch — the Metal rule.
        let slot = map
            .iter()
            .filter(|(k, _)| k.0 == ptr && k.1 >= src.len())
            .max_by_key(|(k, _)| k.2)
            .map(|(_, b)| Arc::clone(b));
        drop(map);
        let Some(b) = slot else {
            panic!(
                "download_into: no device buffer for this slice — host reads \
                 require a backend-produced buffer (the lazy-sync contract)"
            );
        };
        // The slot may be LONGER than the read (prefix download) — take a
        // view of exactly the requested extent.
        let view = b.slice(..src.len());
        self.stream
            .memcpy_dtoh(&view, out)
            .unwrap_or_else(|e| panic!("cuda download: {e}"));
    }

    fn warm_weight(&self, data: &[f32]) {
        if data.is_empty() {
            return;
        }
        let _ = self.weight_buf(data);
    }

    fn warm_weight_2d(&self, data: &[f32], _n: usize, _k: usize) {
        // The weight binds row-major [n, k] directly (no transpose), so
        // this is just `warm_weight` — the shape args are information the
        // transpose-carrying backends need, not this one.
        self.warm_weight(data);
    }
}
