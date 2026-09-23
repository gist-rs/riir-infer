//! Issue 734 Arm 12 — the **GDN chunked recurrence on the cudarc stack**
//! (the llama.cpp prefill shape, owner-approved numerics-policy change).
//!
//! The T4 record (Bench 705) implemented the correct Delta-rule chunkwise
//! solve on CubeCL (`deltanet_delta_rule_chunked.rs`) and proved the algebra
//! kernel-exact (≤8.3e-7 vs sequential on synthetic regimes incl. production
//! shapes) — but its e2e bit-identity gate failed (max_rel 1.55/2.78
//! @P=128/64, argmax stable) because a different summation order diverges
//! ~4e-3/layer and compounds through 48 layers. llama.cpp's GDN prefill
//! (`delta-net-base.cpp`, CS=64) IS this algorithm and is not bit-identical
//! to any serial scan — chunking is their reference math. Per the owner
//! sign-off (2026-08-23) this arm ships that numerics class, gated on
//! argmax-stability + distributional agreement + generation agreement
//! instead of the Bench-710 FNV pin.
//!
//! # Algorithm (per GDN layer, chunk size C=64)
//!
//! For each 64-token chunk with boundary state S₀ (layout [h, d_v, d_k],
//! q/k replicated per v-head by the expand stage):
//!
//! ```text
//!   γ_i = ∏_{m≤i} α_m                (log-space, floor −60)
//!   X[i][j]   = β_i·(γ_i/γ_j)·(k_i·k_j)      j < i
//!   QKR[i][j] = (γ_i/γ_j)·(q_i·k_j)          j ≤ i
//!   RHS[i]    = β_i·(v_i − γ_i·(S₀·k_i))
//!   QS0[i]    = γ_i·(S₀·q_i)
//!   T = (I+X)⁻¹                    (explicit unit-lower-tri inverse)
//!   U = T·RHS                      (matvec — no serial solve)
//!   O_i      = (QS0[i] + Σ_{j≤i} QKR[i][j]·U_j) / √d
//!   S_end     = γ_end·S₀ + Σ_i (γ_end/γ_i)·U_i ⊗ k_i
//! ```
//!
//! # Why the explicit inverse (the v1→v3 lesson)
//!
//! v1 did the forward substitution inside the serial kernel with
//! warp-butterfly reductions: **503M SHFL warp-instructions per layer**
//! (256 butterflies × 5 shfl × 6144 warps × 64 chunks) — at the measured
//! rate that alone is ~8 ms/layer (chunk_seq measured 10.55 ms ≈ the rowpar
//! kernel it replaced). v2 killed phase-A/C butterflies but read k/q rows
//! lane-per-token from GLOBAL — 32 separate token rows per warp-load,
//! ~100 GB/layer of sector traffic → 70 ms (worse). **v3 removes EVERY
//! butterfly**: the inverse T = (I+X)⁻¹ is state-independent, so it is
//! computed in the PARALLEL phase (thread-per-column recurrence,
//! batched over all (head, chunk)) and the serial kernel becomes pure
//! matvecs — lane-per-token dots over smem-staged tiles (cooperative
//! coalesced staging in two 32-token halves, conflict-free padded rows).
//!
//! # Dispatch (4 kernels/layer — zero host crossings)
//!
//! 1. `gdn_pf_decay` — one thread per (head, chunk): log-γ + ratios.
//! 2. `gdn_pf_gram` — one block per (head, chunk): X/QKR from a
//!    smem-staged k tile + register k_i/q_i (the v1 thread-per-(i,j) form
//!    read 32 token rows per warp-load — fully uncoalesced).
//! 3. `gdn_pf_tinv` — one block per (head, chunk), thread-per-column:
//!    `Y[i][j] = −X[i][j] − Σ_{m=j+1}^{i−1} X[i][m]·Y[m][j]` (strict
//!    lower parts of (I+X)·(I+Y)=I).
//! 4. `gdn_pf_chunk_seq` — grid (n_v × 16 row-tiles) × 8 warps, ONE warp
//!    per state row held in 4 registers/lane across ALL chunks. Per chunk:
//!    stage half → rhs/qs0 (lane=token serial dots, smem broadcast state
//!    row) → stage half2 → rhs/qs0 → U = T·RHS (lane=token matvec) →
//!    O = qs0 + QKR·u (lane=token matvec) → state rank-64 update
//!    (lane-owns-cols, coalesced global k). ZERO shuffles.
//!
//! # Numerics
//!
//! Deliberately NOT bit-identical to the rowpar kernel (different summation
//! order — see the module doc). The explicit inverse replaces the direct
//! triangular solve (classic stability tradeoff — measured in the G1 probe:
//! tensor-scaled ≤ ~1e-5 on synthetic regimes; the e2e distributional gates
//! judge the real-data class). `__logf`/`__expf` fast forms accepted. The
//! log-γ floor at −60 keeps underflowed α (=0.0f32) from poisoning ratios.

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;

/// Inner chunk size (the llama.cpp `delta-net-base.cpp` CS for non-KDA —
/// also the T4 production-shape C).
pub const GDN_CHUNK: usize = 64;

pub(crate) const GDN_CHUNKED_CUDA_SRC: &str = r#"
// warp butterfly sum (kept for reference forms; v3 does not use it on the
// serial path)
__device__ __forceinline__ float ps_treeup(float v)
{
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    return v;
}

// ---------------------------------------------------------------------------
// 1) cumulative decay per (head, chunk): log_gamma, gamma, decay_to_end,
//    total_decay. Layouts: [n_v * n_chunks * 64] (head-major), [n_v*n_chunks].
// ---------------------------------------------------------------------------
extern "C" __global__ void gdn_pf_decay(
    const float* __restrict__ decay,     // [p * n_v] token-major
    float* __restrict__ log_gamma,       // [hc * 64]
    float* __restrict__ gamma,           // [hc * 64]
    float* __restrict__ dte,             // [hc * 64]
    float* __restrict__ total_decay,     // [hc]
    int n_v, int n_chunks, int p)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int hc = n_v * n_chunks;
    if (idx >= hc) return;
    const int h = idx / n_chunks;
    const int c = idx % n_chunks;
    const int t0 = c * 64;
    const int clen = min(64, p - t0);
    float* lg = log_gamma + idx * 64;
    float* ga = gamma + idx * 64;
    float acc = 0.0f;
    for (int i = 0; i < clen; i++) {
        const float a = decay[(t0 + i) * n_v + h];
        acc += __logf(a);
        if (acc < -60.0f) acc = -60.0f;
        lg[i] = acc;
        ga[i] = __expf(acc);
    }
    const float lg_end = lg[clen - 1];
    total_decay[idx] = __expf(lg_end);
    float* dt = dte + idx * 64;
    for (int j = 0; j < clen; j++) {
        dt[j] = __expf(lg_end - lg[j]);
    }
}

// ---------------------------------------------------------------------------
// 2) X/QKR interaction matrices — one block per (head, chunk); the chunk's k
//    rows staged in smem (row stride 129: odd stride = conflict-free
//    lane-per-token column walks), thread i owns row i with k_i/q_i in
//    registers; the j-loop dots regs×smem.
// ---------------------------------------------------------------------------
#define GRS 129
extern "C" __global__ void __launch_bounds__(64, 2) gdn_pf_gram(
    const float* __restrict__ qkvx,       // [p * 3 * v_dim]
    const float* __restrict__ beta,       // [p * n_v] token-major
    const float* __restrict__ log_gamma,  // [hc * 64]
    float* __restrict__ x_mat,            // [hc * 64 * 64] strict-lower
    float* __restrict__ qkr_mat,          // [hc * 64 * 64] lower-incl-diag
    int n_v, int n_chunks, int p, int v_dim)
{
    __shared__ float ks[64][GRS];
    const int hc_idx = blockIdx.x;
    const int h = hc_idx / n_chunks;
    const int c = hc_idx % n_chunks;
    const int t0 = c * 64;
    const int clen = min(64, p - t0);
    const int stride = 3 * v_dim;
    const int kbase = v_dim + h * 128;

    for (int r = threadIdx.x; r < clen; r += 64) {
        const float* kr = qkvx + (size_t)(t0 + r) * stride + kbase;
        #pragma unroll
        for (int m = 0; m < 128; m++) {
            ks[r][m] = kr[m];
        }
    }
    __syncthreads();
    if (threadIdx.x >= clen) return;
    const int i = threadIdx.x;
    const int ti = t0 + i;
    const float* ki = qkvx + (size_t)ti * stride + kbase;
    const float* qi = qkvx + (size_t)ti * stride + h * 128;
    float4 ki4[32];
    float4 qi4[32];
    #pragma unroll
    for (int m4 = 0; m4 < 32; m4++) {
        ki4[m4] = *reinterpret_cast<const float4*>(ki + m4 * 4);
        qi4[m4] = *reinterpret_cast<const float4*>(qi + m4 * 4);
    }
    const float b = beta[(size_t)ti * n_v + h];
    const float* lg = log_gamma + hc_idx * 64;
    const float lgi = lg[i];
    float* xr = x_mat + (size_t)hc_idx * 4096 + i * 64;
    float* qr = qkr_mat + (size_t)hc_idx * 4096 + i * 64;
    for (int j = 0; j <= i; j++) {
        const float ratio = __expf(lgi - lg[j]);
        const float* kj = ks[j];
        float kk = 0.0f;
        float qk = 0.0f;
        #pragma unroll 4
        for (int m4 = 0; m4 < 32; m4++) {
            const float k0 = kj[m4 * 4];
            const float k1 = kj[m4 * 4 + 1];
            const float k2 = kj[m4 * 4 + 2];
            const float k3 = kj[m4 * 4 + 3];
            kk = __fmaf_rn(ki4[m4].x, k0, kk);
            kk = __fmaf_rn(ki4[m4].y, k1, kk);
            kk = __fmaf_rn(ki4[m4].z, k2, kk);
            kk = __fmaf_rn(ki4[m4].w, k3, kk);
            qk = __fmaf_rn(qi4[m4].x, k0, qk);
            qk = __fmaf_rn(qi4[m4].y, k1, qk);
            qk = __fmaf_rn(qi4[m4].z, k2, qk);
            qk = __fmaf_rn(qi4[m4].w, k3, qk);
        }
        if (j < i) {
            xr[j] = b * ratio * kk;
        }
        qr[j] = ratio * qk;
    }
}

// ---------------------------------------------------------------------------
// 3) T-inverse: strict-lower Y of (I+X)(I+Y) = I, one block per (head,
//    chunk), thread-per-column serial row recurrence (state-independent —
//    this deletes the serial kernel's forward-substitution butterflies).
//      Y[i][j] = -X[i][j] - sum_{m=j+1}^{i-1} X[i][m] * Y[m][j]   (i > j)
//    Reads X (L2-hot from gram) + own column's earlier Y (L1-hot, self-
//    written); writes t_mat strict-lower.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(64, 4) gdn_pf_tinv(
    const float* __restrict__ x_mat,      // [hc * 64 * 64] strict-lower
    float* __restrict__ t_mat,            // [hc * 64 * 64] strict-lower out
    int n_chunks, int p)
{
    const int hc_idx = blockIdx.x;
    const int c = hc_idx % n_chunks;
    const int clen = min(64, p - c * 64);
    if (threadIdx.x >= clen) return;
    const int j = threadIdx.x;
    const float* xcol_base = x_mat + (size_t)hc_idx * 4096;
    float* tcol_base = t_mat + (size_t)hc_idx * 4096;
    for (int i = j + 1; i < clen; i++) {
        const float* xr = xcol_base + i * 64;
        float acc = 0.0f;
        for (int m = j + 1; m < i; m++) {
            acc = __fmaf_rn(xr[m], tcol_base[m * 64 + j], acc);
        }
        tcol_base[i * 64 + j] = __fsub_rn(__fsub_rn(0.0f, xr[j]), acc);
    }
}

// ---------------------------------------------------------------------------
// 4) the serial half — one warp per state row, ALL chunks in one dispatch,
//    ZERO shuffles. grid (n_v, 16 row-tiles), block 256 = 8 warps; warp w
//    owns row blockIdx.y*8 + w. Lane L owns k-cols {L, L+32, L+64, L+96}
//    (state registers + phase-D updates) and tokens {L, L+32} (phase-A/C
//    matvecs). k/q tiles staged cooperatively in two 32-token halves
//    (row stride 129 — conflict-free lane-per-token column walks).
// ---------------------------------------------------------------------------
#define TRS 130
extern "C" __global__ void __launch_bounds__(256, 2) gdn_pf_chunk_seq(
    const float* __restrict__ qkvx,        // [p * 3 * v_dim]
    const float* __restrict__ beta,        // [p * n_v] token-major
    const float* __restrict__ gamma,       // [hc * 64]
    const float* __restrict__ dte,         // [hc * 64]
    const float* __restrict__ total_decay, // [hc]
    const float* __restrict__ t_mat,       // [hc * 64 * 64] strict-lower
    const float* __restrict__ qkr_mat,     // [hc * 64 * 64]
    float* __restrict__ state,             // [n_v * 128 * 128] in-place
    float* __restrict__ output,            // [p * v_dim] token-major
    int n_v, int n_chunks, int p, int v_dim)
{
    const int h = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.y << 3) + warp;
    const int c0 = lane;
    const int c1 = lane + 32;
    const int c2 = lane + 64;
    const int c3 = lane + 96;
    const int srow = (h * 128 + row) * 128;
    float s0 = state[srow + c0];
    float s1 = state[srow + c1];
    float s2 = state[srow + c2];
    float s3 = state[srow + c3];
    const float scale = rsqrtf(128.0f);
    const int stride = 3 * v_dim;
    const int hoff = h * 128;
    const int kbase = v_dim + hoff;
    const int vbase = 2 * v_dim + hoff;

    // 33280 B staging union: phases A (k/q half-tiles [32][130] x2) then
    // B/C (T/QKR full tiles [64][65] x2) never coexist, so they overlay.
    __shared__ float sm[8320];
    __shared__ float sh_rhs[8][64];
    __shared__ float sh_qs0[8][64];
    __shared__ float sh_u[8][64];
    __shared__ float sh_srow[8][128];
#define KHALF(half) (reinterpret_cast<float(*)[TRS]>(&sm[0]))      /* region A */
#define QHALF(half) (reinterpret_cast<float(*)[TRS]>(&sm[4160]))    /* region B */
#define TTILE   (reinterpret_cast<float(*)[65]>(&sm[0]))
#define QKRTILE (reinterpret_cast<float(*)[65]>(&sm[4160]))

    // cooperative half-stage: 256 threads copy 32 rows × 128 cols from
    // `sec` (0=k, 1=q) of tokens [t0 + half*32, +32) into the union tile.
    // Tokens beyond p (short final chunk) write 0 (their dots are skipped).
    // Every thread of the block participates (block-wide __syncthreads).
#define STAGE_HALF(sec, half)                                                  \
    do {                                                                       \
        float(*const dst)[TRS] = (sec) ? QHALF(half) : KHALF(half);            \
        for (int e = threadIdx.x; e < 32 * 128; e += 256) {                    \
            const int r = e >> 7;                                              \
            const int m = e & 127;                                             \
            const int tok = t0 + (half) * 32 + r;                              \
            dst[r][m] = tok < p                                                \
                ? qkvx[(size_t)tok * stride + ((sec) ? hoff : kbase) + m]      \
                : 0.0f;                                                        \
        }                                                                      \
        __syncthreads();                                                       \
    } while (0)

    for (int ch = 0; ch < n_chunks; ch++) {
        const int t0 = ch * 64;
        const int clen = min(64, p - t0);
        const int hc = h * n_chunks + ch;
        const float* gam = gamma + hc * 64;
        const float* dtec = dte + hc * 64;

        // stage this warp's state row (lane writes its 4 cols)
        sh_srow[warp][c0] = s0;
        sh_srow[warp][c1] = s1;
        sh_srow[warp][c2] = s2;
        sh_srow[warp][c3] = s3;
        __syncwarp();

        // Phase A — rhs/qs0 via lane=token serial dots over the staged
        // tiles + the broadcast state row. Half 0: k first, then q.
        STAGE_HALF(0, 0);
        {
            const int i = lane;
            if (i < clen) {
                float rk = 0.0f;
                const float* kr = KHALF(0)[i];
                #pragma unroll 8
                for (int m = 0; m < 128; m++) {
                    rk = __fmaf_rn(sh_srow[warp][m], kr[m], rk);
                }
                const int t = t0 + i;
                const float b = beta[t * n_v + h];
                const float g = gam[i];
                const float vv = qkvx[(size_t)t * stride + vbase + row];
                sh_rhs[warp][i] = b * __fsub_rn(vv, g * rk);
            }
        }
        STAGE_HALF(1, 0);   // q half 0 (k half 0 no longer needed)
        {
            const int i = lane;
            if (i < clen) {
                float rq = 0.0f;
                const float* qr = QHALF(0)[i];
                #pragma unroll 8
                for (int m = 0; m < 128; m++) {
                    rq = __fmaf_rn(sh_srow[warp][m], qr[m], rq);
                }
                sh_qs0[warp][i] = gam[i] * rq;
            }
        }
        // k half 0 tile reused for half 1
        STAGE_HALF(0, 1);
        {
            const int i = lane + 32;
            if (i < clen) {
                float rk = 0.0f;
                const float* kr = KHALF(1)[i - 32];
                #pragma unroll 8
                for (int m = 0; m < 128; m++) {
                    rk = __fmaf_rn(sh_srow[warp][m], kr[m], rk);
                }
                const int t = t0 + i;
                const float b = beta[t * n_v + h];
                const float g = gam[i];
                const float vv = qkvx[(size_t)t * stride + vbase + row];
                sh_rhs[warp][i] = b * __fsub_rn(vv, g * rk);
            }
        }
        STAGE_HALF(1, 1);
        {
            const int i = lane + 32;
            if (i < clen) {
                float rq = 0.0f;
                const float* qr = QHALF(1)[i - 32];
                #pragma unroll 8
                for (int m = 0; m < 128; m++) {
                    rq = __fmaf_rn(sh_srow[warp][m], qr[m], rq);
                }
                sh_qs0[warp][i] = gam[i] * rq;
            }
        }

        // Stage T + QKR tiles (overlay the k/q tiles). The pre-barrier is
        // load-bearing: region B is still being read by slower warps' q1
        // dots (the same race class fires only at high block counts —
        // h=48/768 blocks — where warp scheduling diverges).
        __syncthreads();
        do {
            for (int e = threadIdx.x; e < 64 * 64; e += 256) {
                const int i = e >> 6;
                const int j = e & 63;
                TTILE[i][j] = t_mat[(size_t)hc * 4096 + e];
                QKRTILE[i][j] = qkr_mat[(size_t)hc * 4096 + e];
            }
            __syncthreads();
        } while (0);
        __syncwarp();

        // Phase B — U = T·RHS: lane owns u_i for i ∈ {lane, lane+32};
        // serial j<i dot over the T row (smem tile) + rhs (smem).
        {
            const int i0 = lane;
            if (i0 < clen) {
                const float* tr = TTILE[i0];
                float acc = sh_rhs[warp][i0];
                for (int j = 0; j < i0; j++) {
                    acc = __fmaf_rn(tr[j], sh_rhs[warp][j], acc);
                }
                sh_u[warp][i0] = acc;
            }
            const int i1 = lane + 32;
            if (i1 < clen) {
                const float* tr = TTILE[i1];
                float acc = sh_rhs[warp][i1];
                for (int j = 0; j < i1; j++) {
                    acc = __fmaf_rn(tr[j], sh_rhs[warp][j], acc);
                }
                sh_u[warp][i1] = acc;
            }
        }
        __syncwarp();

        // Phase C — O_i = (qs0_i + sum_{j<=i} QKR[i][j]·u_j)·scale.
        {
            const int i0 = lane;
            if (i0 < clen) {
                const float* qr = QKRTILE[i0];
                float acc = sh_qs0[warp][i0];
                for (int j = 0; j <= i0; j++) {
                    acc = __fmaf_rn(qr[j], sh_u[warp][j], acc);
                }
                output[(t0 + i0) * v_dim + hoff + row] = acc * scale;
            }
            const int i1 = lane + 32;
            if (i1 < clen) {
                const float* qr = QKRTILE[i1];
                float acc = sh_qs0[warp][i1];
                for (int j = 0; j <= i1; j++) {
                    acc = __fmaf_rn(qr[j], sh_u[warp][j], acc);
                }
                output[(t0 + i1) * v_dim + hoff + row] = acc * scale;
            }
        }
        __syncwarp();

        // Phase D — state: s = gamma_end·s + sum_i dte_i·u_i·k_i
        // (lane-owns-cols; k from global, coalesced consecutive-c lanes).
        {
            const float g_end = total_decay[hc];
            s0 *= g_end;
            s1 *= g_end;
            s2 *= g_end;
            s3 *= g_end;
            for (int i = 0; i < clen; i++) {
                const float coef = dtec[i] * sh_u[warp][i];
                const int tr = (t0 + i) * stride;
                const float k0 = qkvx[tr + kbase + c0];
                const float k1 = qkvx[tr + kbase + c1];
                const float k2 = qkvx[tr + kbase + c2];
                const float k3 = qkvx[tr + kbase + c3];
                s0 = __fmaf_rn(coef, k0, s0);
                s1 = __fmaf_rn(coef, k1, s1);
                s2 = __fmaf_rn(coef, k2, s2);
                s3 = __fmaf_rn(coef, k3, s3);
            }
        }
        // End-of-chunk block barrier: without it a fast warp's next-chunk
        // STAGE(0,0) writes region A (= TTILE) while a slow warp is still
        // in Phase B reading it.
        __syncthreads();
    }
    state[srow + c0] = s0;
    state[srow + c1] = s1;
    state[srow + c2] = s2;
    state[srow + c3] = s3;
}
"#;

/// The cudarc-side GDN chunked-recurrence kernels (Arm 12, v3).
pub struct CudaGdnChunkedKernels {
    decay: CudaFunction,
    gram: CudaFunction,
    tinv: CudaFunction,
    chunk_seq: CudaFunction,
    _module: Arc<CudaModule>,
}

impl CudaGdnChunkedKernels {
    /// Own context + stream (the `CudaDeltanetKernels::new_standalone`
    /// precedent) — lets integration tests construct the kernels without
    /// importing cudarc types (cudarc is not a dev-dependency).
    pub fn new_standalone() -> Result<(Self, Arc<CudaStream>), String> {
        let ctx = CudaContext::new(0).map_err(|e| e.to_string())?;
        let stream = ctx.new_stream().map_err(|e| e.to_string())?;
        let kernels = Self::new(ctx)?;
        Ok((kernels, stream))
    }

    /// Compile (NVRTC, sm_89) + load.
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, String> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            GDN_CHUNKED_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;
        let decay = module
            .load_function("gdn_pf_decay")
            .map_err(|e| format!("gdn_pf_decay: {e}"))?;
        let gram = module
            .load_function("gdn_pf_gram")
            .map_err(|e| format!("gdn_pf_gram: {e}"))?;
        let tinv = module
            .load_function("gdn_pf_tinv")
            .map_err(|e| format!("gdn_pf_tinv: {e}"))?;
        let chunk_seq = module
            .load_function("gdn_pf_chunk_seq")
            .map_err(|e| format!("gdn_pf_chunk_seq: {e}"))?;
        Ok(Self {
            decay,
            gram,
            tinv,
            chunk_seq,
            _module: module,
        })
    }

    /// Number of inner chunks for a prompt length.
    pub fn n_chunks(p: usize) -> usize {
        p.div_ceil(GDN_CHUNK)
    }

    /// Launch the whole chunked recurrence (4 kernels) for one layer.
    ///
    /// # Safety
    ///
    /// Caller guarantees `qkvx` covers `p * 3 * v_dim` f32 (q at
    /// `[t, 0..v_dim)`, k at `[t, v_dim..2·v_dim)`, v at
    /// `[t, 2·v_dim..3·v_dim)`, each head-major over `v_dim = n_head * 128`),
    /// `beta`/`decay` cover `p * n_head` (token-major), `state` covers
    /// `n_head * 128 * 128`, `output` covers `p * v_dim`, and the scratch
    /// buffers cover `n_head * n_chunks * 64` (lg/gamma/dte),
    /// `n_head * n_chunks` (total_decay), `n_head * n_chunks * 4096`
    /// (x/t/qkr). Requires `head_dim == 128` and `1 <= p`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_chunked(
        &self,
        stream: &CudaStream,
        qkvx: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        decay: &CudaSlice<f32>,
        state: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        lg: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        dte: &CudaSlice<f32>,
        total_decay: &CudaSlice<f32>,
        x_mat: &CudaSlice<f32>,
        t_mat: &CudaSlice<f32>,
        qkr_mat: &CudaSlice<f32>,
        n_head: usize,
        p: usize,
        v_dim: usize,
    ) -> Result<(), String> {
        debug_assert_eq!(v_dim, n_head * 128, "head_dim == 128 required");
        let n_chunks = Self::n_chunks(p);

        unsafe {
            self.launch_decay(stream, decay, lg, gamma, dte, total_decay, n_head, p)?;
            self.launch_gram(stream, qkvx, beta, lg, x_mat, qkr_mat, n_head, n_chunks, p, v_dim)?;
            self.launch_tinv(stream, x_mat, t_mat, n_chunks, p)?;
            self.launch_chunk_seq(
                stream, qkvx, beta, gamma, dte, total_decay, t_mat, qkr_mat, state, output,
                n_head, n_chunks, p, v_dim,
            )?;
        }
        Ok(())
    }

    /// Kernel 1 — cumulative decay per (head, chunk).
    ///
    /// # Safety
    ///
    /// `decay` covers `p * n_head` (token-major); `lg`/`gamma`/`dte` cover
    /// `n_head * n_chunks * 64`; `total_decay` covers `n_head * n_chunks`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_decay(
        &self,
        stream: &CudaStream,
        decay: &CudaSlice<f32>,
        lg: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        dte: &CudaSlice<f32>,
        total_decay: &CudaSlice<f32>,
        n_head: usize,
        p: usize,
    ) -> Result<(), String> {
        let n_chunks = Self::n_chunks(p);
        let hc = n_head * n_chunks;
        let (nv_i, nc_i, p_i) = (n_head as i32, n_chunks as i32, p as i32);
        let grid = hc.div_ceil(256) as u32;
        unsafe {
            stream
                .launch_builder(&self.decay)
                .arg(decay)
                .arg(lg)
                .arg(gamma)
                .arg(dte)
                .arg(total_decay)
                .arg(&nv_i)
                .arg(&nc_i)
                .arg(&p_i)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Kernel 2 — the X/QKR interaction matrices for ALL chunks
    /// (state-independent).
    ///
    /// # Safety
    ///
    /// `qkvx` covers `p * 3 * v_dim`; `beta` covers `p * n_head`; `lg`
    /// covers `n_head * n_chunks * 64`; `x_mat`/`qkr_mat` cover
    /// `n_head * n_chunks * 4096`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_gram(
        &self,
        stream: &CudaStream,
        qkvx: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        lg: &CudaSlice<f32>,
        x_mat: &CudaSlice<f32>,
        qkr_mat: &CudaSlice<f32>,
        n_head: usize,
        n_chunks: usize,
        p: usize,
        v_dim: usize,
    ) -> Result<(), String> {
        let hc = n_head * n_chunks;
        let (nv_i, nc_i, p_i, vd_i) = (n_head as i32, n_chunks as i32, p as i32, v_dim as i32);
        unsafe {
            stream
                .launch_builder(&self.gram)
                .arg(qkvx)
                .arg(beta)
                .arg(lg)
                .arg(x_mat)
                .arg(qkr_mat)
                .arg(&nv_i)
                .arg(&nc_i)
                .arg(&p_i)
                .arg(&vd_i)
                .launch(LaunchConfig {
                    grid_dim: (hc as u32, 1, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0, // ks[64][129] is STATIC smem in-kernel
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Kernel 3 — the T-inverse (strict-lower Y of (I+X)⁻¹).
    ///
    /// # Safety
    ///
    /// `x_mat` was filled by [`Self::launch_gram`]; `t_mat` covers
    /// `n_head * n_chunks * 4096`.
    pub unsafe fn launch_tinv(
        &self,
        stream: &CudaStream,
        x_mat: &CudaSlice<f32>,
        t_mat: &CudaSlice<f32>,
        n_chunks: usize,
        p: usize,
    ) -> Result<(), String> {
        let nc_i = n_chunks as i32;
        let p_i = p as i32;
        // grid = hc blocks (caller's hc == x_mat.len()/4096)
        let hc = (x_mat.len() / 4096) as u32;
        unsafe {
            stream
                .launch_builder(&self.tinv)
                .arg(x_mat)
                .arg(t_mat)
                .arg(&nc_i)
                .arg(&p_i)
                .launch(LaunchConfig {
                    grid_dim: (hc, 1, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Kernel 4 — the serial half (one warp per state row, all chunks in one
    /// dispatch, zero shuffles).
    ///
    /// # Safety
    ///
    /// Same buffer contract as [`Self::launch_chunked`]; the gram + tinv
    /// outputs must already be populated.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_chunk_seq(
        &self,
        stream: &CudaStream,
        qkvx: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        dte: &CudaSlice<f32>,
        total_decay: &CudaSlice<f32>,
        t_mat: &CudaSlice<f32>,
        qkr_mat: &CudaSlice<f32>,
        state: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        n_head: usize,
        n_chunks: usize,
        p: usize,
        v_dim: usize,
    ) -> Result<(), String> {
        let (nv_i, nc_i, p_i, vd_i) = (n_head as i32, n_chunks as i32, p as i32, v_dim as i32);
        unsafe {
            stream
                .launch_builder(&self.chunk_seq)
                .arg(qkvx)
                .arg(beta)
                .arg(gamma)
                .arg(dte)
                .arg(total_decay)
                .arg(t_mat)
                .arg(qkr_mat)
                .arg(state)
                .arg(output)
                .arg(&nv_i)
                .arg(&nc_i)
                .arg(&p_i)
                .arg(&vd_i)
                .launch(LaunchConfig {
                    grid_dim: (n_head as u32, 16, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0, // tiles are STATIC smem in-kernel
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
