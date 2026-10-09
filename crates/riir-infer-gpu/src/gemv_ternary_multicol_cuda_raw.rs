//! Issue 038 — the n=3-8 decode band on Ada: multi-column dp4a ternary GEMV.
//!
//! Port of PrismML-Eng/llama.cpp `2a42998c5` (PR #306, "cuda: PQ2_0 mat-vec
//! kernel for 3-8 columns on Ada", distill Lead 12) to OUR packed-code weight
//! format and OUR 16-element activation block. Upstream measured on
//! Ternary-Bonsai-2-27B-PQ2_0 (our league model) on an RTX 4070 (sm_89): batched
//! decode +17/+25/+29% at 4/6/8 seq; kernel cold-L2 1.96×/1.75× at n=4/8.
//!
//! ## The mechanism (upstream atoms, ported)
//!
//! 1. **Permuted activation layout**: within each 16-element group, position
//!    `k*4 + m` holds element `4m + k`. With that, `(code_word >> 2k) &
//!    0x03030303` pairs the RAW 2-bit codes `{0,1,2}` element-wise with the
//!    activation bytes at `k*4` — **dp4a needs no per-weight decode** (one
//!    shift+mask yields four dp4a-ready bytes; the n=1 kernel instead spends 8
//!    `__byte_perm`s per 16 weights to decode to signed digits).
//! 2. **Digit-bias one-subtraction**: `sum(q*(c-1)) = sum(q*c) - sum(q)` — the
//!    ternary offset rides the activation block's stored int sum; no
//!    per-weight `(c-1)` decode.
//! 3. **Warp = 4 rows** (`MC_ROWS`), all 32 lanes walking different chunks:
//!    each lane loads its per-column activation int4 ONCE and reuses it across
//!    the 4 rows in registers — the activation loads amortize 4× per row.
//!    `__launch_bounds__(128, 3)` caps registers at 3 blocks/SM (upstream's
//!    occupancy point on the same Ada class).
//!
//! ## Port deltas vs upstream (deliberate)
//!
//! - **16-element chunks, not 32**: upstream quantizes activations per
//!   QK8_1=32; our production posture is the 16-block (Issue 608 T3b: 1.45×
//!   accuracy for 1.2% throughput). Chunk = one uint32 code word + one int4
//!   activation quad; the permutation is block-local, so the port is the
//!   layout + the four-instruction inner product.
//! - **Separate scale/sum buffers** (`ascale` f32, `actsum` i16) instead of
//!   ggml's interleaved half2 — our storage is ours to lay out.
//! - **Host-side quantize+permute** (correctness-first, mirroring
//!   [`super::gemv_ternary_cuda_raw::TernaryGemmCudaRaw::forward`]'s host
//!   quantizer exactly). The GPU-resident quantize variant lands with the
//!   dispatch wiring, not before.
//!
//! ## Numerics parity with the n=1 kernel (asserted by test)
//!
//! Per chunk the integer dot is order-exact (dp4a is int math; the raw-minus-
//! bias identity holds exactly), the per-lane chunk walk is the same
//! lane-strided order, the per-block float expression has the same shape
//! (`(float)sumi * ws * ascale`), and the final reduction is the same
//! full-warp `__shfl_down_sync` tree. With identical quantized activations
//! (both host quantizers are formula-identical), `forward_multicol` output is
//! **bit-identical** to `forward()` — the parity test asserts `to_bits()`
//! equality. A mismatch there means FMA-contraction divergence between the
//! two compilation units: fix by matching the expression shape, never by
//! weakening the assert.
//!
//! ## Admission + owed gates
//!
//! Tokens (columns) ∈ `[3, 8]` — outside that band the caller keeps the n=1
//! GEMV (1 column) or the MMA path (prefill / big batches). `n % 16 == 0`
//! (exact chunks, same precondition class as the n=1 fast path). sm_89 target.
//!
//! ⛔ **Not wired to any production path yet** — opt-in by construction. The
//! dispatch arm (tree-verify driver + batched decode at n∈[3,8]) lands only
//! after, on a free 4090 lane: (a) these unit gates (currently skipping for
//! GPU exclusivity beside the plan437 trainer), (b) task-level argmax
//! agreement at the verify posture (Issue 608 T3a class), (c) the paired
//! bench vs `gemv_ternary_cuda_raw` and `gemm_ternary_i8_mma` at n∈{3..8}
//! (plan 612 G2 runbook pairing), (d) GOAT promotion per the feature-flag
//! discipline. Upstream's accumulation-order drift face (their pplx byte
//! moved 6.3776→6.3820) does not apply here — this port is bit-identical to
//! the incumbent n=1 kernel by construction — but the bench still reads the
//! league gates before any promotion.

#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig,
};

use katgpt_core::TernaryGroupWeights;

use super::gemv_ternary_cuda_raw::{
    ACTIVATION_BLOCK, TernaryGemmCudaRawError, WEIGHT_GROUP, convert_bitplane_to_packed_codes,
};

/// Rows per warp — upstream `PQ2_0_MC_ROWS`. The activation slice is loaded
/// once per lane and reused across these rows in registers.
pub const MC_ROWS: u32 = 4;

/// Warps per block — upstream `PQ2_0_MC_WARPS`. Block = 128 threads with
/// `__launch_bounds__(MC_THREADS, 3)` (upstream's Ada occupancy point).
pub const MC_WARPS: u32 = 4;

/// Threads per block.
pub const MC_THREADS: u32 = MC_WARPS * 32;

/// Supported token band — the issue's n∈[3,8]; outside it the caller keeps
/// the n=1 GEMV or the MMA path.
pub const MC_TOKENS_MIN: usize = 3;
pub const MC_TOKENS_MAX: usize = 8;

/// CUDA source for the multi-column dp4a ternary GEMV. NVRTC compiles this
/// standalone (no CUDA headers) — same discipline as [`super::gemv_ternary_cuda_raw::GEMV_CUDA_SRC`].
pub const MULTICOL_GEMV_CUDA_SRC: &str = r#"
// Rust-side consts don't cross the NVRTC boundary — define them here.
// (MC_ROWS_V = MC_ROWS, MC_THREADS = MC_WARPS * 32.)
#define MC_ROWS_V 4
#define MC_THREADS 128

// Same standalone f16 conversion as the n=1 source (no CUDA headers under
// NVRTC — inline PTX wraps the hardware cvt.f32.f16, exact for normals AND
// subnormals).
__device__ __forceinline__ float mc_f16_bits_to_f32(unsigned short hbits)
{
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(hbits));
    return f;
}

__device__ __forceinline__ int mc_imin(int a, int b) { return a < b ? a : b; }

// One warp computes MC_ROWS output rows for all NCOLS tokens. All 32 lanes
// walk DIFFERENT chunks (c = lane, lane+32, ...): each chunk's weight codes
// are loaded exactly once per row, and each lane's per-token activation int4
// is loaded once and reused across the 4 rows in registers.
//
// Chunk = 16 weights (OUR activation block, not upstream's 32): one uint32
// code word per row + one int4 per token. Shift+mask pairs the raw {0,1,2}
// codes with the permuted activation bytes element-wise; the ternary digit
// bias is the single int subtraction against the block's stored act sum.
//
// Per-block float expression matches the n=1 kernel's shape exactly —
// `(float)sumi * ws * ascale` in the same accumulation order — so results
// are bit-identical to gemv_ternary_dp4a given identical quantized
// activations (asserted by the Rust-side parity test).
template <int NCOLS>
__device__ __forceinline__ void gemv_ternary_multicol_body(
    const short* __restrict__ codes,           // [m * int16_per_row]
    const unsigned short* __restrict__ wscale, // [m * groups_per_row] f16 bits
    const signed char* __restrict__ act,       // [NCOLS * n] permuted
    const float* __restrict__ ascale,          // [NCOLS * nchunk]
    const short* __restrict__ actsum,          // [NCOLS * nchunk] raw int sums
    float* __restrict__ out,                   // [NCOLS * m] token-major
    int m_rows,
    int int16_per_row,                         // n / 8
    int groups_per_row,                        // n / 128
    int nchunk)                                // n / 16
{
    const int lane = threadIdx.x & 31;
    const int row0 = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * MC_ROWS_V;
    if (row0 >= m_rows) {
        return;
    }

    const long col_elems = (long)nchunk << 4; // n

    float acc[MC_ROWS_V][NCOLS] = {};

    for (int c = lane; c < nchunk; c += 32) {
        unsigned int w0[MC_ROWS_V];
        float ws[MC_ROWS_V];
        #pragma unroll
        for (int r = 0; r < MC_ROWS_V; ++r) {
            // Clamped rows are computed but not stored (upstream's tight-warp
            // trick — no per-row early exit inside the chunk loop).
            const int row = mc_imin(row0 + r, m_rows - 1);
            w0[r] = (unsigned int)*(const unsigned int*)(codes + (long)row * int16_per_row + (c << 1));
            ws[r] = mc_f16_bits_to_f32(wscale[(long)row * groups_per_row + (c >> 3)]);
        }

        int4 a[NCOLS];
        float d8[NCOLS];
        int qsum[NCOLS];
        #pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            a[j] = *(const int4*)(act + (long)j * col_elems + (c << 4));
            d8[j] = ascale[(long)j * nchunk + c];
            qsum[j] = (int)actsum[(long)j * nchunk + c];
        }

        #pragma unroll
        for (int r = 0; r < MC_ROWS_V; ++r) {
            int t[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                t[k] = (int)((w0[r] >> (2 * k)) & 0x03030303u);
            }
            #pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                int sumi = 0;
                sumi = __dp4a(t[0], a[j].x, sumi);
                sumi = __dp4a(t[1], a[j].y, sumi);
                sumi = __dp4a(t[2], a[j].z, sumi);
                sumi = __dp4a(t[3], a[j].w, sumi);
                acc[r][j] += (float)(sumi - qsum[j]) * ws[r] * d8[j];
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < MC_ROWS_V; ++r) {
        #pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = acc[r][j];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            const int row = row0 + r;
            if (lane == 0 && row < m_rows) {
                out[(long)j * m_rows + row] = v;
            }
        }
    }
}

// Per-column-count entrypoints (3..=8): unmangled names for cudarc
// load_function. The macro keeps the six wrappers byte-consistent.
#define MC_ENTRY(NCOLS_V)                                                        \
extern "C" __global__ void __launch_bounds__(MC_THREADS, 3)                      \
gemv_ternary_dp4a_multicol_n##NCOLS_V(                                           \
    const short* __restrict__ codes,                                             \
    const unsigned short* __restrict__ wscale,                                   \
    const signed char* __restrict__ act,                                         \
    const float* __restrict__ ascale,                                            \
    const short* __restrict__ actsum,                                            \
    float* __restrict__ out,                                                     \
    int m_rows, int int16_per_row, int groups_per_row, int nchunk)               \
{                                                                                \
    gemv_ternary_multicol_body<NCOLS_V>(codes, wscale, act, ascale, actsum, out, \
        m_rows, int16_per_row, groups_per_row, nchunk);                          \
}

MC_ENTRY(3)
MC_ENTRY(4)
MC_ENTRY(5)
MC_ENTRY(6)
MC_ENTRY(7)
MC_ENTRY(8)
"#;

/// Compiled-kernel names by token count (index = tokens − 3).
const MC_KERNEL_NAMES: [&str; 6] = [
    "gemv_ternary_dp4a_multicol_n3",
    "gemv_ternary_dp4a_multicol_n4",
    "gemv_ternary_dp4a_multicol_n5",
    "gemv_ternary_dp4a_multicol_n6",
    "gemv_ternary_dp4a_multicol_n7",
    "gemv_ternary_dp4a_multicol_n8",
];

struct MultiColWeightBuffers {
    codes_dev: CudaSlice<i16>,
    wscale_dev: CudaSlice<u16>,
    m: usize,
    n: usize,
}

/// Raw-CUDA multi-column dp4a ternary GEMV handler (Issue 038).
///
/// Same lifecycle as [`super::gemv_ternary_cuda_raw::TernaryGemmCudaRaw`]:
/// upload weights once, then `forward_multicol` per token batch. NOT `Sync`
/// — the inner stream serializes dispatches.
///
/// Weight encoding is shared with the n=1 handler ([`convert_bitplane_to_packed_codes`])
/// — the same uploaded (codes, wscale) semantics, so a consumer can upload
/// once into either handler family and stay byte-compatible.
pub struct TernaryGemmMultiColCudaRaw {
    stream: Arc<CudaStream>,
    /// One compiled entrypoint per supported token count (index = tokens − 3).
    kernels: Vec<CudaFunction>,
    /// Compiled module — kept alive so `kernels` remain valid.
    _module: Arc<CudaModule>,
    weights: Vec<MultiColWeightBuffers>,
}

/// Host-side quantize + permute: mirrors `TernaryGemmCudaRaw::forward`'s
/// per-block quantizer EXACTLY (same `d`, same int8 values), then writes the
/// PQ2 permuted layout — position `k*4 + m` of each 16-block holds element
/// `4m + k` — plus the per-block scales and raw int sums.
///
/// Returns `(act_perm [tokens*n] i8, ascale [tokens*nchunk] f32, actsum [tokens*nchunk] i16)`.
fn quantize_permute_activations(
    xs: &[f32],
    tokens: usize,
    n: usize,
) -> (Vec<i8>, Vec<f32>, Vec<i16>) {
    let nchunk = n / ACTIVATION_BLOCK;
    let mut act_perm = vec![0i8; tokens * n];
    let mut ascale = vec![0f32; tokens * nchunk];
    let mut actsum = vec![0i16; tokens * nchunk];

    for tok in 0..tokens {
        let x_row = &xs[tok * n..(tok + 1) * n];
        let base = tok * n;
        for chunk in 0..nchunk {
            let start = chunk * ACTIVATION_BLOCK;
            let mut absmax: f32 = 0.0;
            for &v in &x_row[start..start + ACTIVATION_BLOCK] {
                absmax = absmax.max(v.abs());
            }
            // Identical to the n=1 host quantizer: d = absmax/127 (1.0 if
            // all zero), q = round(x/d) clamped.
            let d = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
            let inv_d = 1.0 / d;

            let mut q = [0i8; ACTIVATION_BLOCK];
            let mut sum: i32 = 0;
            for (j, &v) in x_row[start..start + ACTIVATION_BLOCK].iter().enumerate() {
                let qi = (v * inv_d).round().clamp(-128.0, 127.0);
                q[j] = qi as i8;
                sum += q[j] as i32;
            }

            // Permuted store: pos (k*4 + m) = element (4m + k).
            for k in 0..4 {
                for m in 0..4 {
                    act_perm[base + start + k * 4 + m] = q[4 * m + k];
                }
            }
            ascale[tok * nchunk + chunk] = d;
            actsum[tok * nchunk + chunk] = sum as i16;
        }
    }

    (act_perm, ascale, actsum)
}

impl TernaryGemmMultiColCudaRaw {
    /// Initialize the CUDA context + compile the multi-column kernels.
    ///
    /// sm_89 (Ada) — same target as the n=1 handler; the launch-bounds
    /// occupancy point is an Ada measurement.
    pub fn new() -> Result<Self, TernaryGemmCudaRawError> {
        let ctx =
            CudaContext::new(0).map_err(|e| TernaryGemmCudaRawError::CudaInit(e.to_string()))?;
        let stream = ctx
            .new_stream()
            .map_err(|e| TernaryGemmCudaRawError::CudaInit(e.to_string()))?;

        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            MULTICOL_GEMV_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| TernaryGemmCudaRawError::Compile(format!("{e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| TernaryGemmCudaRawError::Compile(e.to_string()))?;
        let kernels = MC_KERNEL_NAMES
            .iter()
            .map(|name| {
                module
                    .load_function(name)
                    .map_err(|e| TernaryGemmCudaRawError::Compile(format!("{name}: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            stream,
            kernels,
            _module: module,
            weights: Vec::new(),
        })
    }

    /// Upload a ternary weight matrix, returning a handle for
    /// `forward_multicol`.
    ///
    /// Requires `n % 16 == 0` (exact 16-element chunks — stricter than the
    /// n=1 handler's `% 8`: a half chunk has no code word to shift).
    pub fn upload_weights(
        &mut self,
        w: &TernaryGroupWeights,
    ) -> Result<usize, TernaryGemmCudaRawError> {
        let m = w.rows;
        let n = w.cols;
        if !n.is_multiple_of(ACTIVATION_BLOCK) {
            return Err(TernaryGemmCudaRawError::ShapeMismatch {
                expected: 0, // n must be a multiple of ACTIVATION_BLOCK (16)
                got: n % ACTIVATION_BLOCK,
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

        let idx = self.weights.len();
        self.weights.push(MultiColWeightBuffers {
            codes_dev,
            wscale_dev,
            m,
            n,
        });
        Ok(idx)
    }

    /// Compute `out[tokens × m] = xs[tokens × n] @ Wᵀ` — token-major output
    /// (`out[tok*m + row]`), one kernel for the whole batch.
    ///
    /// `tokens` must be in `[3, 8]`; outside the band use the n=1 handler
    /// (1 token) or the MMA path (prefill / larger batches).
    pub fn forward_multicol(
        &mut self,
        weight_idx: usize,
        xs: &[f32],
        tokens: usize,
        out: &mut [f32],
    ) -> Result<(), TernaryGemmCudaRawError> {
        if !(MC_TOKENS_MIN..=MC_TOKENS_MAX).contains(&tokens) {
            return Err(TernaryGemmCudaRawError::ShapeMismatch {
                expected: 0, // tokens must be in [3, 8]
                got: tokens,
            });
        }

        let wb = &mut self.weights[weight_idx];
        let n = wb.n;
        let m = wb.m;
        if xs.len() != tokens * n {
            return Err(TernaryGemmCudaRawError::ShapeMismatch {
                expected: tokens * n,
                got: xs.len(),
            });
        }
        if out.len() != tokens * m {
            return Err(TernaryGemmCudaRawError::ShapeMismatch {
                expected: tokens * m,
                got: out.len(),
            });
        }
        if m == 0 {
            return Ok(());
        }

        let nchunk = n / ACTIVATION_BLOCK;
        let (act_perm, ascale, actsum) = quantize_permute_activations(xs, tokens, n);

        let act_dev = self
            .stream
            .clone_htod(&act_perm)
            .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;
        let ascale_dev = self
            .stream
            .clone_htod(&ascale)
            .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;
        let actsum_dev = self
            .stream
            .clone_htod(&actsum)
            .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;
        let mut out_dev = self
            .stream
            .alloc_zeros::<f32>(tokens * m)
            .map_err(|e| TernaryGemmCudaRawError::Alloc(e.to_string()))?;

        let kernel = &self.kernels[tokens - MC_TOKENS_MIN];
        let m_i32 = m as i32;
        let int16_per_row_i32 = (n / 8) as i32;
        let groups_per_row_i32 = n.div_ceil(WEIGHT_GROUP) as i32;
        let nchunk_i32 = nchunk as i32;

        let warps_needed = (m as u32).div_ceil(MC_ROWS);
        let grid_x = warps_needed.div_ceil(MC_WARPS);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (MC_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        // Safety: pure GEMV — reads codes/wscale/act/ascale/actsum, writes
        // out_dev [tokens*m]; grid covers ceil(m/4) warps and every store is
        // row-guarded. No aliasing (out_dev is per-call).
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(&wb.codes_dev)
                .arg(&wb.wscale_dev)
                .arg(&act_dev)
                .arg(&ascale_dev)
                .arg(&actsum_dev)
                .arg(&mut out_dev)
                .arg(&m_i32)
                .arg(&int16_per_row_i32)
                .arg(&groups_per_row_i32)
                .arg(&nchunk_i32)
                .launch(cfg)
                .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;
        self.stream
            .memcpy_dtoh(&out_dev, out)
            .map_err(|e| TernaryGemmCudaRawError::Launch(e.to_string()))?;

        Ok(())
    }

    /// Number of weight matrices currently uploaded.
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// Probe for a usable CUDA device — skip loud when absent (same posture
    /// as the n=1 module's tests).
    fn cuda_or_skip() -> Option<()> {
        if CudaContext::new(0).is_err() {
            return None;
        }
        Some(())
    }

    /// Mirrors the n=1 module's `make_test_weights` (local copy — those
    /// helpers are private to that module's test mod).
    fn make_test_weights(m: usize, n: usize) -> TernaryGroupWeights {
        let mut w = TernaryGroupWeights::new(m, n);
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
        for s in &mut w.group_scale {
            *s = f16::from_f32(0.5);
        }
        w
    }

    /// LCG-deterministic input block (same generator as the n=1 tests).
    fn lcg_inputs(seed: u32, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 16) as f32 / 65535.0 - 0.5) * 2.0
            })
            .collect()
    }

    /// CPU reference (float weights × float x, per-row group scale) — same
    /// shape as the n=1 module's.
    fn cpu_matvec(w: &TernaryGroupWeights, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; w.rows];
        for (row, y_val) in y.iter_mut().enumerate() {
            let mut acc = 0f32;
            for (col, &xv) in x.iter().enumerate() {
                let pos_block = w.pos_bits[row * w.blocks64 + col / 64];
                let neg_block = w.neg_bits[row * w.blocks64 + col / 64];
                let bit = 1u64 << (col % 64);
                let val: f32 = if pos_block & bit != 0 {
                    1.0
                } else if neg_block & bit != 0 {
                    -1.0
                } else {
                    0.0
                };
                let scale = w.group_scale[row * w.groups_per_row + col / WEIGHT_GROUP].to_f32();
                acc += val * xv * scale;
            }
            *y_val = acc;
        }
        y
    }

    /// NVRTC compilation gate — CPU-ONLY (no CUDA context, no GPU): proves
    /// the template/macro source compiles to PTX and every entrypoint symbol
    /// is present. Runs even under GPU exclusivity (the trainer carve) —
    /// nvrtc is a host-side compiler.
    #[test]
    fn test_nvrtc_compiles_multicol_src() {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            MULTICOL_GEMV_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .expect("NVRTC compile of MULTICOL_GEMV_CUDA_SRC failed");
        let ptx = ptx.to_src();
        assert!(
            ptx.len() > 1000,
            "suspiciously small PTX: {} bytes",
            ptx.len()
        );
        for name in MC_KERNEL_NAMES {
            assert!(ptx.contains(name), "PTX missing entrypoint {name}");
        }
    }

    /// G1: multi-column output matches the CPU reference within the same
    /// int8-quantization tolerance the n=1 gate uses (mean_rel < 2%,
    /// max_rel < 5%).
    #[test]
    fn test_multicol_matches_cpu_within_int8_tolerance() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let m = 256;
        let n = 256; // multiple of 128 and 16
        let tokens = 4;
        let w = make_test_weights(m, n);
        let xs = lcg_inputs(0xCAFE0001, tokens * n);

        let mut handler = TernaryGemmMultiColCudaRaw::new().expect("CUDA init");
        let idx = handler.upload_weights(&w).expect("upload");
        let mut gpu_out = vec![0f32; tokens * m];
        handler
            .forward_multicol(idx, &xs, tokens, &mut gpu_out)
            .expect("forward");

        let mut max_rel = 0f32;
        let mut mean_rel = 0f32;
        let mut n_cmp = 0usize;
        for tok in 0..tokens {
            let cpu_out = cpu_matvec(&w, &xs[tok * n..(tok + 1) * n]);
            for (a, b) in cpu_out.iter().zip(&gpu_out[tok * m..(tok + 1) * m]) {
                let rel = (a - b).abs() / a.abs().max(1e-6);
                max_rel = max_rel.max(rel);
                mean_rel += rel;
                n_cmp += 1;
            }
        }
        mean_rel /= n_cmp as f32;
        eprintln!(
            "[mc_g1] m={m} n={n} tokens={tokens}: mean_rel={mean_rel:.4e} max_rel={max_rel:.4e}"
        );
        assert!(
            mean_rel < 0.02,
            "mean_rel {mean_rel:.4e} exceeds 2% int8 quantization tolerance"
        );
        assert!(
            max_rel < 0.05,
            "max_rel {max_rel:.4e} exceeds 5% int8 quantization tolerance"
        );
    }

    /// The parity gate: with identical host quantization, `forward_multicol`
    /// is BIT-IDENTICAL to the incumbent n=1 `forward` per token — the integer
    /// dot is order-exact, the per-block float expression shape matches, and
    /// the lane walk + full-warp reduce tree are the same. A `to_bits` mismatch
    /// means FMA contraction diverged between the two NVRTC units: fix by
    /// matching the expression shape, never by weakening this assert.
    #[test]
    fn test_multicol_bit_parity_with_n1_kernel() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let m = 64;
        let n = 256;
        let w = make_test_weights(m, n);

        let mut h1 =
            super::super::gemv_ternary_cuda_raw::TernaryGemmCudaRaw::new().expect("n1 CUDA init");
        let idx1 = h1.upload_weights(&w).expect("n1 upload");
        let mut hmc = TernaryGemmMultiColCudaRaw::new().expect("mc CUDA init");
        let idxmc = hmc.upload_weights(&w).expect("mc upload");

        // Every entrypoint executes at least once (issue 038 review 2026-10-10:
        // n5/n6/n7 were compiled-but-never-run under the old {3,4,8} loop).
        for tokens in MC_TOKENS_MIN..=MC_TOKENS_MAX {
            let xs = lcg_inputs(0xBEEF0000 ^ tokens as u32, tokens * n);
            let mut ref_out = vec![0f32; tokens * m];
            for tok in 0..tokens {
                h1.forward(
                    idx1,
                    &xs[tok * n..(tok + 1) * n],
                    &mut ref_out[tok * m..(tok + 1) * m],
                )
                .expect("n1 forward");
            }
            let mut mc_out = vec![0f32; tokens * m];
            hmc.forward_multicol(idxmc, &xs, tokens, &mut mc_out)
                .expect("mc forward");

            for (i, (a, b)) in ref_out.iter().zip(&mc_out).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "bit parity broken at tokens={tokens} element {i}: {a} vs {b}"
                );
            }
        }
    }

    /// Shape discipline: the token band, the input/output lengths, and the
    /// `% 16` upload precondition all refuse loudly.
    #[test]
    fn test_multicol_rejects_bad_shapes() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let m = 16;
        let n = 256;
        let w = make_test_weights(m, n);

        let mut handler = TernaryGemmMultiColCudaRaw::new().expect("CUDA init");
        let idx = handler.upload_weights(&w).expect("upload");

        let xs = vec![0f32; 4 * n];
        let mut out = vec![0f32; 2 * m];
        // Below the band (1-2 tokens: the n=1 kernel's job).
        assert!(matches!(
            handler.forward_multicol(idx, &xs[..2 * n], 2, &mut out[..2 * m]),
            Err(TernaryGemmCudaRawError::ShapeMismatch { .. })
        ));
        // Above the band (9 > 8: the MMA path's job).
        let xs9 = vec![0f32; 9 * n];
        let mut out9 = vec![0f32; 9 * m];
        assert!(matches!(
            handler.forward_multicol(idx, &xs9, 9, &mut out9),
            Err(TernaryGemmCudaRawError::ShapeMismatch { .. })
        ));
        // Input length mismatch.
        assert!(matches!(
            handler.forward_multicol(idx, &xs[..3 * n - 1], 4, &mut out),
            Err(TernaryGemmCudaRawError::ShapeMismatch { .. })
        ));
        // Output length mismatch.
        assert!(matches!(
            handler.forward_multicol(idx, &xs, 4, &mut out[..4 * m - 1]),
            Err(TernaryGemmCudaRawError::ShapeMismatch { .. })
        ));

        // Upload precondition: n % 16 != 0 refuses.
        let w8 = make_test_weights(m, 8);
        assert!(matches!(
            handler.upload_weights(&w8),
            Err(TernaryGemmCudaRawError::ShapeMismatch { .. })
        ));
    }

    /// The host quantizer+permuter is invertible per 16-block: unpermuting
    /// recovers exactly the n=1 host quantizer's int8 values (same d, same
    /// q). Runs on CPU — no CUDA needed, always executes.
    #[test]
    fn test_quantize_permute_roundtrip_matches_n1_quantizer() {
        let n = 64;
        let tokens = 3;
        let xs = lcg_inputs(0xDEAD0042, tokens * n);

        let (act_perm, ascale, actsum) = quantize_permute_activations(&xs, tokens, n);
        let nchunk = n / ACTIVATION_BLOCK;

        for tok in 0..tokens {
            for chunk in 0..nchunk {
                let start = chunk * ACTIVATION_BLOCK;
                // Independent reference: the n=1 host quantizer's exact formula.
                let mut absmax = 0f32;
                for &v in &xs[tok * n + start..tok * n + start + ACTIVATION_BLOCK] {
                    absmax = absmax.max(v.abs());
                }
                let d = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                assert_eq!(ascale[tok * nchunk + chunk], d);

                let mut sum = 0i32;
                for j in 0..ACTIVATION_BLOCK {
                    let expect = (xs[tok * n + start + j] * (1.0 / d))
                        .round()
                        .clamp(-128.0, 127.0) as i8;
                    sum += expect as i32;
                    // permuted pos k*4+m holds element 4m+k → element j sits at
                    // pos (j%4)*4 + j/4.
                    let pos = start + (j % 4) * 4 + j / 4;
                    assert_eq!(
                        act_perm[tok * n + pos],
                        expect,
                        "tok={tok} chunk={chunk} j={j}"
                    );
                }
                assert_eq!(actsum[tok * nchunk + chunk] as i32, sum);
            }
        }
    }
}
