//! Plan 621 T3 — the GDN **L8 eager-prefill recurrence** (the Yukon
//! lane-geometry class; riir-ai Issue 971 item 4, Research 284 candidate #1).
//!
//! K rows per warp × WPC warps per block, grid `(n_head, hd/(K*WPC))`; the
//! warp's K rows are CONSECUTIVE state rows of one head (their k/q/beta/decay
//! operands are per-(token, head) — shared across the warp's rows exactly like
//! the `recmr` ladder). The L8 part is the lane PERMUTATION + the register
//! fold that replaces the first butterfly steps of the plane-sum reduction.
//!
//! ## The numerics family (the finding that shaped this port — plan 621
//! addendum-d)
//!
//! The production prefill rowpar canonical form is
//! `(dot4_f1, UPD_FMA, ps_treeup, SC_RSQ)` (Bench 719/721/734; the recmr
//! r4w4 default inherits it). `ps_treeup` is the **ASCENDING** xor butterfly
//! (offsets 1, 2, 4, 8, 16 — adjacent-pairs-first): a DIFFERENT binary
//! reduction tree from the descending butterfly (16…1) of the
//! `cudarc_kernels::deltanet` family (`recurrence_f32_fused_rows_hd128`).
//! The plan-621 T2 oracle proved the plan's §4 STRIDED map (group-lane n
//! holds chains n + 8m) tree-identical to the DESCENDING tree — which would
//! FAIL the production FNV pin. THIS module therefore ports the **ADJACENT
//! map**: group-lane `n` holds incumbent chains `K*n + m` (m = 0..K-1).
//!
//! Under the adjacent map, `ps_treeup`'s first log2(K) steps live entirely in
//! registers — chain L = K·n + m pairs with L ⊕ 2^j inside the same lane for
//! 2^j < K — and every remaining step pairs lanes `n` and `n ⊕ off` with
//! `off < G = 32/K` (never crossing the row-group):
//!
//! ```text
//! K=4: u = (p0 + p1) + (p2 + p3)      // ps_treeup steps 1+2 (child order
//!                                     // self-first, identical)
//!      u += shfl_xor(u, 1|2|4)        // steps 4/8/16 — lane offsets < G=8
//!      3 shuffles per reduction vs the incumbent's 5
//! K=2: u = p0 + p1; u += shfl_xor(u, 1|2|4|8)   // 4 shuffles vs 5
//! ```
//!
//! Same tree, same leaves, same child order at every node → **bit-identical
//! to the incumbent rowpar path by construction** (the bytewise-FNV G1
//! class; `tests/l8_fold_tree_equality_spec.rs` pins the tree mechanically).
//! Per-warp shuffle cost at K=4: 4 rows × 2 reductions × 3 = 24 shuffles per
//! token vs the recmr r4 shape's 40 — the eager rung the prefill ladder was
//! missing. Gates: `recl8_bit_identity_*` below (all six geometries,
//! bit-exact vs the production dispatch); the kernel-only bench is
//! `#[ignore]` (`recl8_kernel_only_bench`).

use std::sync::Arc;

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig,
};

/// The prefill-band admission floor (the Yukon `GDN_L8_MIN_TOKENS`
/// equivalence): below 17 tokens the eager lane stays incumbent — the 8-16
/// band belongs to the verify lane. Pinned beside the knob so the dispatch
/// and the gates read one constant.
pub const GDN_L8_MIN_TOKENS: usize = 17;

/// Issue 904's geometry-axis lesson applied: rows-per-warp × warps-per-block
/// priced independently. Index = arm for [`CudaDeltanetL8Kernels::
/// launch_recurrence_l8`]. Every `(K, WPC)` divides `head_dim == 128`.
/// `(4, 1)` is the plan's canonical geometry (grid `(n_head, 32)`, block 32);
/// the WPC variants exist so the kernel-only bench can separate the fold win
/// from the occupancy win on the 4090 (the recmr ladder measured WPC as a
/// first-class axis — Bench 896).
pub const RECL8_GEOMETRIES: &[(u32, u32)] = &[(4, 1), (4, 2), (4, 4), (2, 1), (2, 2), (2, 4)];

/// The shipped default: `(K=4, WPC=4)` — the T5 kernel-only bench's winner
/// at every p ≥ 17 on the 4090 (0.66-0.81× the incumbent across the band;
/// the plan's canonical `(4,1)` measured 0.68-0.97×). Bit-identity is
/// arm-independent (the T4 gate covers all six), so this is a perf-only
/// pick, exactly the plan's "T5 picks on perf alone" rule.
pub const RECL8_DEFAULT_ARM: usize = 2;

const DELTANET_L8_CUDA_SRC: &str = r#"
// Helpers — byte-identical SHAPES to the prefill canonical forms (each cudarc
// module compiles its own source string; the l8_ prefix keeps the two modules'
// NVRTC symbols disjoint).
__device__ __forceinline__ float l8_rsqrt_approx(float a)
{
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float l8_dot4_f1(float s0, float s1, float s2, float s3,
                                            float k0, float k1, float k2, float k3)
{
    float t = __fmul_rn(s1, k1);
    t = __fmaf_rn(s0, k0, t);
    t = __fmaf_rn(s2, k2, t);
    t = __fmaf_rn(s3, k3, t);
    return t;
}

// ---------------------------------------------------------------------------
// Plan 621 T3 — the L8 adjacent-map recurrence. K=4 below (the canonical
// geometry): 8 lanes per row-group, 4 chains per lane, float4 loads (the
// lane's 4 chains are CONTIGUOUS floats at every leaf block — the adjacent
// map's layout bonus).
//
// PORT LAW (bit-identity class): per-chain arithmetic verbatim (l8_dot4_f1 /
// UPD_FMA / __fmul_rn decay / __fmul_rn(beta, sub(v, kv)) / rsqrt.approx
// scale); fold child order exact ((p0+p1)+(p2+p3)); `+=` self-first on every
// shuffle; never a tolerance band.
// ---------------------------------------------------------------------------
#define RECL8_K4(NAME, WPC)                                                    \
extern "C" __global__ void __launch_bounds__(32 * WPC, 1) NAME(                \
    const float* __restrict__ qkvx,      /* [p, 3*v_dim] */                    \
    const float* __restrict__ beta,      /* [p, n_head] */                     \
    const float* __restrict__ decay,     /* [p, n_head] */                     \
    float* __restrict__ state,           /* [n_head*hd*hd] in-place */         \
    float* __restrict__ output,          /* [p, v_dim] */                      \
    int head_dim, int n_head, int p, int v_dim)                                \
{                                                                              \
    const int head = blockIdx.x;                                               \
    const int lane = threadIdx.x & 31;                                         \
    const int warp = threadIdx.x >> 5;                                         \
    const int g = lane >> 3;        /* row within the warp's 4 rows */         \
    const int n = lane & 7;         /* group-lane: chains 4n..4n+3 */          \
    const int row = blockIdx.y * (4 * WPC) + warp * 4 + g;                     \
    const long qkvx_stride = 3L * v_dim;                                       \
    const long head_off = (long)head * head_dim;                               \
    const long k_base = v_dim + head_off;                                      \
    const long v_base = 2L * v_dim + head_off;                                 \
    const long row_off = (long)head * head_dim * head_dim                      \
                       + (long)row * head_dim;                                 \
    const int cbase = 4 * n;        /* first column of the lane's quad */      \
    float s[4][4];                                                             \
    _Pragma("unroll")                                                          \
    for (int c = 0; c < 4; c++) {                                              \
        const float4 v = *reinterpret_cast<const float4*>(                     \
            state + row_off + cbase + 32 * c);                                 \
        s[0][c] = v.x;                                                         \
        s[1][c] = v.y;                                                         \
        s[2][c] = v.z;                                                         \
        s[3][c] = v.w;                                                         \
    }                                                                          \
    const float scale = l8_rsqrt_approx((float)head_dim);                      \
    for (int t = 0; t < p; t++) {                                              \
        const long t_row = (long)t * qkvx_stride;                              \
        const long sc = (long)t * n_head + head;                               \
        const float beta_val = beta[sc];                                       \
        const float decay_val = decay[sc];                                     \
        float k[4][4];                                                         \
        float q[4][4];                                                         \
        _Pragma("unroll")                                                      \
        for (int c = 0; c < 4; c++) {                                          \
            const float4 kv = *reinterpret_cast<const float4*>(                \
                qkvx + t_row + k_base + cbase + 32 * c);                       \
            const float4 qv = *reinterpret_cast<const float4*>(                \
                qkvx + t_row + head_off + cbase + 32 * c);                     \
            k[0][c] = kv.x;                                                    \
            k[1][c] = kv.y;                                                    \
            k[2][c] = kv.z;                                                    \
            k[3][c] = kv.w;                                                    \
            q[0][c] = qv.x;                                                    \
            q[1][c] = qv.y;                                                    \
            q[2][c] = qv.z;                                                    \
            q[3][c] = qv.w;                                                    \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int m = 0; m < 4; m++) {                                          \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; c++) {                                      \
                s[m][c] = __fmul_rn(s[m][c], decay_val);                       \
            }                                                                  \
        }                                                                      \
        const float p0 = l8_dot4_f1(s[0][0], s[0][1], s[0][2], s[0][3],        \
                                    k[0][0], k[0][1], k[0][2], k[0][3]);       \
        const float p1 = l8_dot4_f1(s[1][0], s[1][1], s[1][2], s[1][3],        \
                                    k[1][0], k[1][1], k[1][2], k[1][3]);       \
        const float p2 = l8_dot4_f1(s[2][0], s[2][1], s[2][2], s[2][3],        \
                                    k[2][0], k[2][1], k[2][2], k[2][3]);       \
        const float p3 = l8_dot4_f1(s[3][0], s[3][1], s[3][2], s[3][3],        \
                                    k[3][0], k[3][1], k[3][2], k[3][3]);       \
        /* ps_treeup steps 1+2 as the register fold (adjacent map). */         \
        float u = __fadd_rn(__fadd_rn(p0, p1), __fadd_rn(p2, p3));             \
        u += __shfl_xor_sync(0xffffffffu, u, 1);                               \
        u += __shfl_xor_sync(0xffffffffu, u, 2);                               \
        u += __shfl_xor_sync(0xffffffffu, u, 4);                               \
        const float v_row = qkvx[t_row + v_base + row];                        \
        const float delta_row = __fmul_rn(beta_val,                            \
                                          __fsub_rn(v_row, u));                \
        _Pragma("unroll")                                                      \
        for (int m = 0; m < 4; m++) {                                          \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; c++) {                                      \
                s[m][c] = __fmaf_rn(k[m][c], delta_row, s[m][c]);              \
            }                                                                  \
        }                                                                      \
        const float a0 = l8_dot4_f1(s[0][0], s[0][1], s[0][2], s[0][3],        \
                                    q[0][0], q[0][1], q[0][2], q[0][3]);       \
        const float a1 = l8_dot4_f1(s[1][0], s[1][1], s[1][2], s[1][3],        \
                                    q[1][0], q[1][1], q[1][2], q[1][3]);       \
        const float a2 = l8_dot4_f1(s[2][0], s[2][1], s[2][2], s[2][3],        \
                                    q[2][0], q[2][1], q[2][2], q[2][3]);       \
        const float a3 = l8_dot4_f1(s[3][0], s[3][1], s[3][2], s[3][3],        \
                                    q[3][0], q[3][1], q[3][2], q[3][3]);       \
        float dot = __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));           \
        dot += __shfl_xor_sync(0xffffffffu, dot, 1);                           \
        dot += __shfl_xor_sync(0xffffffffu, dot, 2);                           \
        dot += __shfl_xor_sync(0xffffffffu, dot, 4);                           \
        if (n == 0) {                                                          \
            output[(long)t * v_dim + head_off + row] =                         \
                __fmul_rn(dot, scale);                                         \
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int c = 0; c < 4; c++) {                                              \
        *reinterpret_cast<float4*>(state + row_off + cbase + 32 * c) =         \
            make_float4(s[0][c], s[1][c], s[2][c], s[3][c]);                   \
    }                                                                          \
}

// K=2: 16 lanes per row-group, 2 chains per lane, scalar loads (the lane's
// chains are 2 floats at every leaf block — float2 would need .z/.w guards;
// the fold is p0+p1, then lane xor 1/2/4/8 — offsets < G=16).
#define RECL8_K2(NAME, WPC)                                                    \
extern "C" __global__ void __launch_bounds__(32 * WPC, 1) NAME(                \
    const float* __restrict__ qkvx,                                            \
    const float* __restrict__ beta,                                            \
    const float* __restrict__ decay,                                           \
    float* __restrict__ state,                                                 \
    float* __restrict__ output,                                                \
    int head_dim, int n_head, int p, int v_dim)                                \
{                                                                              \
    const int head = blockIdx.x;                                               \
    const int lane = threadIdx.x & 31;                                         \
    const int warp = threadIdx.x >> 5;                                         \
    const int g = lane >> 4;                                                   \
    const int n = lane & 15;                                                   \
    const int row = blockIdx.y * (2 * WPC) + warp * 2 + g;                     \
    const long qkvx_stride = 3L * v_dim;                                       \
    const long head_off = (long)head * head_dim;                               \
    const long k_base = v_dim + head_off;                                      \
    const long v_base = 2L * v_dim + head_off;                                 \
    const long row_off = (long)head * head_dim * head_dim                      \
                       + (long)row * head_dim;                                 \
    const int cbase = 2 * n;                                                   \
    float s[2][4];                                                             \
    _Pragma("unroll")                                                          \
    for (int c = 0; c < 4; c++) {                                              \
        s[0][c] = state[row_off + cbase + 32 * c];                             \
        s[1][c] = state[row_off + cbase + 1 + 32 * c];                         \
    }                                                                          \
    const float scale = l8_rsqrt_approx((float)head_dim);                      \
    for (int t = 0; t < p; t++) {                                              \
        const long t_row = (long)t * qkvx_stride;                              \
        const long sc = (long)t * n_head + head;                               \
        const float beta_val = beta[sc];                                       \
        const float decay_val = decay[sc];                                     \
        float k[2][4];                                                         \
        float q[2][4];                                                         \
        _Pragma("unroll")                                                      \
        for (int c = 0; c < 4; c++) {                                          \
            k[0][c] = qkvx[t_row + k_base + cbase + 32 * c];                   \
            k[1][c] = qkvx[t_row + k_base + cbase + 1 + 32 * c];               \
            q[0][c] = qkvx[t_row + head_off + cbase + 32 * c];                 \
            q[1][c] = qkvx[t_row + head_off + cbase + 1 + 32 * c];             \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int m = 0; m < 2; m++) {                                          \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; c++) {                                      \
                s[m][c] = __fmul_rn(s[m][c], decay_val);                       \
            }                                                                  \
        }                                                                      \
        const float p0 = l8_dot4_f1(s[0][0], s[0][1], s[0][2], s[0][3],        \
                                    k[0][0], k[0][1], k[0][2], k[0][3]);       \
        const float p1 = l8_dot4_f1(s[1][0], s[1][1], s[1][2], s[1][3],        \
                                    k[1][0], k[1][1], k[1][2], k[1][3]);       \
        float u = __fadd_rn(p0, p1);                                           \
        u += __shfl_xor_sync(0xffffffffu, u, 1);                               \
        u += __shfl_xor_sync(0xffffffffu, u, 2);                               \
        u += __shfl_xor_sync(0xffffffffu, u, 4);                               \
        u += __shfl_xor_sync(0xffffffffu, u, 8);                               \
        const float v_row = qkvx[t_row + v_base + row];                        \
        const float delta_row = __fmul_rn(beta_val,                            \
                                          __fsub_rn(v_row, u));                \
        _Pragma("unroll")                                                      \
        for (int m = 0; m < 2; m++) {                                          \
            _Pragma("unroll")                                                  \
            for (int c = 0; c < 4; c++) {                                      \
                s[m][c] = __fmaf_rn(k[m][c], delta_row, s[m][c]);              \
            }                                                                  \
        }                                                                      \
        const float a0 = l8_dot4_f1(s[0][0], s[0][1], s[0][2], s[0][3],        \
                                    q[0][0], q[0][1], q[0][2], q[0][3]);       \
        const float a1 = l8_dot4_f1(s[1][0], s[1][1], s[1][2], s[1][3],        \
                                    q[1][0], q[1][1], q[1][2], q[1][3]);       \
        float dot = __fadd_rn(a0, a1);                                         \
        dot += __shfl_xor_sync(0xffffffffu, dot, 1);                           \
        dot += __shfl_xor_sync(0xffffffffu, dot, 2);                           \
        dot += __shfl_xor_sync(0xffffffffu, dot, 4);                           \
        dot += __shfl_xor_sync(0xffffffffu, dot, 8);                           \
        if (n == 0) {                                                          \
            output[(long)t * v_dim + head_off + row] =                         \
                __fmul_rn(dot, scale);                                         \
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int c = 0; c < 4; c++) {                                              \
        state[row_off + cbase + 32 * c] = s[0][c];                             \
        state[row_off + cbase + 1 + 32 * c] = s[1][c];                         \
    }                                                                          \
}

RECL8_K4(recl8_pf_k4w1, 1)
RECL8_K4(recl8_pf_k4w2, 2)
RECL8_K4(recl8_pf_k4w4, 4)
RECL8_K2(recl8_pf_k2w1, 1)
RECL8_K2(recl8_pf_k2w2, 2)
RECL8_K2(recl8_pf_k2w4, 4)
"#;

/// Holds the compiled L8 recurrence kernels (the plan-621 eager prefill arm).
pub struct CudaDeltanetL8Kernels {
    /// Index = arm into [`RECL8_GEOMETRIES`].
    recurrence_l8: Vec<CudaFunction>,
    _module: Arc<CudaModule>,
}

/// Launches dispatched through the L8 recurrence (the vacuous guard — the
/// Bench-768 lesson: bit-identical outputs mean only the counter proves the
/// arm actually ran).
static RECL8_LAUNCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Launches dispatched through the L8 kernel family so far (diagnostic).
pub fn recl8_launches() -> usize {
    RECL8_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

impl CudaDeltanetL8Kernels {
    /// Own context + stream (the standalone precedent) — lets tests construct
    /// the kernels without importing cudarc types.
    pub fn new_standalone() -> Result<(Self, Arc<CudaStream>), String> {
        let ctx = CudaContext::new(0).map_err(|e| e.to_string())?;
        let stream = ctx.new_stream().map_err(|e| e.to_string())?;
        let kernels = Self::new(ctx)?;
        Ok((kernels, stream))
    }

    /// Compile (NVRTC, sm_89) + load all L8 geometry arms against a shared
    /// context.
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, String> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            DELTANET_L8_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;

        let mut recurrence_l8 = Vec::with_capacity(RECL8_GEOMETRIES.len());
        for &(k, wpc) in RECL8_GEOMETRIES {
            let name = format!("recl8_pf_k{k}w{wpc}");
            recurrence_l8.push(
                module
                    .load_function(&name)
                    .map_err(|e| format!("{name}: {e}"))?,
            );
        }

        Ok(Self {
            recurrence_l8,
            _module: module,
        })
    }

    /// Plan 621 T3 — the L8 eager recurrence: K rows per warp (the warp's
    /// rows are CONSECUTIVE state rows of one head), WPC warps per block.
    /// Bit-identical to [`CudaDeltanetKernels::launch_recurrence`]'s rowpar
    /// path by construction (the adjacent-map fold IS the ps_treeup tree —
    /// the module doc carries the derivation; the tests pin it on-GPU).
    ///
    /// # Safety
    ///
    /// Caller guarantees `qkvx` covers `p * 3 * v_dim`, `beta`/`decay` cover
    /// `p * n_head`, `state` covers `n_head * head_dim * head_dim`, `output`
    /// covers `p * v_dim`, `head_dim == 128`, `v_dim % 128 == 0` (the
    /// float4-load alignment contract), and `arm` indexes
    /// [`RECL8_GEOMETRIES`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_recurrence_l8(
        &self,
        stream: &CudaStream,
        arm: usize,
        qkvx: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        decay: &CudaSlice<f32>,
        state: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        p: usize,
        v_dim: usize,
    ) -> Result<(), String> {
        let Some(&(k, wpc)) = RECL8_GEOMETRIES.get(arm) else {
            return Err(format!(
                "recl8 arm {arm} out of range (0..{})",
                RECL8_GEOMETRIES.len()
            ));
        };
        debug_assert_eq!(head_dim, 128, "L8 recurrence: head_dim == 128");
        debug_assert_eq!(
            head_dim % (k * wpc) as usize,
            0,
            "L8 recurrence: head_dim % (K*WPC) == 0"
        );
        debug_assert_eq!(
            v_dim % head_dim,
            0,
            "L8 recurrence: v_dim % head_dim == 0 (float4 alignment)"
        );
        let func = &self.recurrence_l8[arm];
        let (hd_i, nh_i, p_i, vd_i) = (head_dim as i32, n_head as i32, p as i32, v_dim as i32);
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, (head_dim / (k * wpc) as usize) as u32, 1),
            block_dim: (32 * wpc, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(qkvx)
                .arg(beta)
                .arg(decay)
                .arg(state)
                .arg(output)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&p_i)
                .arg(&vd_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        RECL8_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefill_cuda_deltanet::{CudaDeltanetKernels, PlaneSum, RecDot, RecScale, RecUpd};

    fn cuda_or_skip() -> Option<()> {
        match CudaContext::new(0) {
            Ok(_) => Some(()),
            Err(e) => {
                eprintln!("[skip] no CUDA device: {e}");
                None
            }
        }
    }

    /// Seeded LCG in [-3, 3) — the established test-input shape.
    fn lcg_vec(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                lo + (hi - lo) * ((x >> 40) as f32 / (1u64 << 24) as f32)
            })
            .collect()
    }

    const HD: usize = 128;

    /// Plan 621 T4 — the on-GPU FNV-class gate: every L8 geometry arm must be
    /// BIT-identical to the production rowpar dispatch (launch_recurrence →
    /// recmr r4w4 default), state AND output, across the league shape (48
    /// value heads) and small odd shapes, at p in the 17+ admission band.
    /// Requires `RIIR_PREFILL_REC_MR` unset (the production default is the
    /// incumbent under test). GPU lane: run serially (`--test-threads 1`) or
    /// alone — the sibling GPU lanes share the box and the process-global
    /// launch counter.
    #[test]
    fn recl8_bit_identity_vs_incumbent_all_geometries() {
        let Some(_) = cuda_or_skip() else {
            return;
        };
        let (incumbent, stream) = CudaDeltanetKernels::new_standalone().expect("incumbent compile");
        let (l8, l8_stream) = CudaDeltanetL8Kernels::new_standalone().expect("l8 compile");

        // (n_head, v_dim): the Bonsai-2 league GDN shape + small odd shapes.
        const SHAPES: &[(usize, usize)] = &[(48, 48 * HD), (2, 2 * HD), (1, HD)];
        const P_ROWS: &[usize] = &[17, 64, 512];

        for &(n_head, v_dim) in SHAPES {
            for &p in P_ROWS {
                let seed = 0x6210_0000_0000_0001u64 ^ ((n_head as u64) << 24) ^ (p as u64);
                let qkvx = lcg_vec(seed, p * 3 * v_dim, -3.0, 3.0);
                let beta = lcg_vec(seed ^ 0xB, p * n_head, 0.01, 1.0);
                let decay = lcg_vec(seed ^ 0xD, p * n_head, 0.01, 1.0);
                let state_init = lcg_vec(seed ^ 0x5, n_head * HD * HD, -2.0, 2.0);

                let run = |kernels: &CudaDeltanetKernels,
                           stream: &Arc<CudaStream>|
                 -> (Vec<f32>, Vec<f32>) {
                    let qkvx_dev = stream.clone_htod(&qkvx).unwrap();
                    let beta_dev = stream.clone_htod(&beta).unwrap();
                    let decay_dev = stream.clone_htod(&decay).unwrap();
                    let state_dev = stream.clone_htod(&state_init).unwrap();
                    let out_dev = stream.clone_htod(&vec![0f32; p * v_dim]).unwrap();
                    unsafe {
                        kernels
                            .launch_recurrence(
                                stream,
                                RecDot::F1,
                                RecUpd::Fma,
                                PlaneSum::TreeUp,
                                RecScale::Rsqrt,
                                &qkvx_dev,
                                &beta_dev,
                                &decay_dev,
                                &state_dev,
                                &out_dev,
                                HD,
                                n_head,
                                p,
                                v_dim,
                            )
                            .expect("incumbent launch");
                    }
                    stream.synchronize().expect("sync");
                    let mut state_out = vec![0f32; n_head * HD * HD];
                    let mut out = vec![0f32; p * v_dim];
                    stream.memcpy_dtoh(&state_dev, &mut state_out).unwrap();
                    stream.memcpy_dtoh(&out_dev, &mut out).unwrap();
                    (state_out, out)
                };

                let run_l8 = |arm: usize| -> (Vec<f32>, Vec<f32>) {
                    let qkvx_dev = l8_stream.clone_htod(&qkvx).unwrap();
                    let beta_dev = l8_stream.clone_htod(&beta).unwrap();
                    let decay_dev = l8_stream.clone_htod(&decay).unwrap();
                    let state_dev = l8_stream.clone_htod(&state_init).unwrap();
                    let out_dev = l8_stream.clone_htod(&vec![0f32; p * v_dim]).unwrap();
                    unsafe {
                        l8.launch_recurrence_l8(
                            &l8_stream, arm, &qkvx_dev, &beta_dev, &decay_dev, &state_dev,
                            &out_dev, HD, n_head, p, v_dim,
                        )
                        .expect("l8 launch");
                    }
                    l8_stream.synchronize().expect("sync");
                    let mut state_out = vec![0f32; n_head * HD * HD];
                    let mut out = vec![0f32; p * v_dim];
                    l8_stream.memcpy_dtoh(&state_dev, &mut state_out).unwrap();
                    l8_stream.memcpy_dtoh(&out_dev, &mut out).unwrap();
                    (state_out, out)
                };

                let launches_before = recl8_launches();
                let (ref_state, ref_out) = run(&incumbent, &stream);

                for (arm, &geom) in RECL8_GEOMETRIES.iter().enumerate() {
                    let (l8_state, l8_out) = run_l8(arm);
                    let state_mismatches: Vec<usize> = ref_state
                        .iter()
                        .zip(l8_state.iter())
                        .enumerate()
                        .filter(|(_, (x, y))| x.to_bits() != y.to_bits())
                        .map(|(i, _)| i)
                        .collect();
                    assert!(
                        state_mismatches.is_empty(),
                        "L8 arm {arm} (geom {geom:?}) n_head={n_head} p={p}: {} state bit mismatches, first at {:?} (incumbent={:?} l8={:?})",
                        state_mismatches.len(),
                        state_mismatches.first(),
                        state_mismatches.first().map(|&i| ref_state[i]),
                        state_mismatches.first().map(|&i| l8_state[i]),
                    );
                    let out_mismatches: Vec<usize> = ref_out
                        .iter()
                        .zip(l8_out.iter())
                        .enumerate()
                        .filter(|(_, (x, y))| x.to_bits() != y.to_bits())
                        .map(|(i, _)| i)
                        .collect();
                    assert!(
                        out_mismatches.is_empty(),
                        "L8 arm {arm:?} (geom {:?}) n_head={n_head} p={p}: {} output bit mismatches, first at {:?}",
                        RECL8_GEOMETRIES[arm],
                        out_mismatches.len(),
                        out_mismatches.first(),
                    );
                }
                assert!(
                    recl8_launches() - launches_before >= RECL8_GEOMETRIES.len(),
                    "the vacuous guard: all six arms must have launched"
                );
                // (>= not ==: the counter is PROCESS-GLOBAL — a sibling GPU
                // test running concurrently in this binary (the e2e lane's
                // L8-armed rounds) increments it too. Measured: the == form
                // failed under parallel libtest threads and passed serially,
                // 3/3 — a counter race, never a numerics one.)
            }
        }
    }

    /// Plan 621 T5 — the kernel-only interleaved-pairs bench (the H200
    /// comparability read: recurrence-only µs/token at the Bonsai-2 league
    /// GDN shape, 48 value heads). Incumbent (recmr r4w4 default) vs every
    /// L8 geometry arm; per-pair ratios, median. PROVENANCE beside the
    /// numbers.
    ///
    /// ```bash
    /// cargo test -p riir-infer-gpu --release --features ternary_gemv_cuda_raw,ternary_gemm_batched,cubecl_runtime \
    ///     --lib prefill_cuda_deltanet_l8::tests::recl8_kernel_only_bench -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "GPU timing lane — run explicitly on an idle 4090"]
    fn recl8_kernel_only_bench() {
        let Some(_) = cuda_or_skip() else {
            return;
        };
        let (incumbent, stream) = CudaDeltanetKernels::new_standalone().expect("incumbent compile");
        let (l8, _) = CudaDeltanetL8Kernels::new_standalone().expect("l8 compile");

        const N_HEAD: usize = 48;
        const V_DIM: usize = N_HEAD * HD;
        const P_ROWS: &[usize] = &[17, 64, 512, 2048, 4096];
        const PAIRS: usize = 7;

        let prov = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                "--format=csv,noheader",
            ])
            .output()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .replace('\n', " | ")
            })
            .unwrap_or_else(|_| "unavailable".into());
        println!(
            "PROVENANCE: gpu[util%,mem,°C]={prov} · pairs={PAIRS} (interleaved, order alternating)"
        );

        for &p in P_ROWS {
            let seed = 0x6215_0000_0000_0000u64 ^ (p as u64);
            let qkvx = lcg_vec(seed, p * 3 * V_DIM, -3.0, 3.0);
            let beta = lcg_vec(seed ^ 0xB, p * N_HEAD, 0.01, 1.0);
            let decay = lcg_vec(seed ^ 0xD, p * N_HEAD, 0.01, 1.0);
            let state_init = lcg_vec(seed ^ 0x5, N_HEAD * HD * HD, -2.0, 2.0);

            let qkvx_dev = stream.clone_htod(&qkvx).unwrap();
            let beta_dev = stream.clone_htod(&beta).unwrap();
            let decay_dev = stream.clone_htod(&decay).unwrap();
            let out_dev = stream.clone_htod(&vec![0f32; p * V_DIM]).unwrap();

            // One device state buffer per arm slot (incumbent + 6 arms),
            // re-uploaded before every timed run (state is in-place).
            let mut states: Vec<CudaSlice<f32>> = (0..=RECL8_GEOMETRIES.len())
                .map(|_| stream.clone_htod(&state_init).unwrap())
                .collect();

            let mut run_arm = |arm: usize| -> f64 {
                stream.memcpy_htod(&state_init, &mut states[arm]).unwrap();
                let t = std::time::Instant::now();
                if arm == 0 {
                    unsafe {
                        incumbent
                            .launch_recurrence(
                                &stream,
                                RecDot::F1,
                                RecUpd::Fma,
                                PlaneSum::TreeUp,
                                RecScale::Rsqrt,
                                &qkvx_dev,
                                &beta_dev,
                                &decay_dev,
                                &states[arm],
                                &out_dev,
                                HD,
                                N_HEAD,
                                p,
                                V_DIM,
                            )
                            .expect("incumbent launch");
                    }
                } else {
                    unsafe {
                        l8.launch_recurrence_l8(
                            &stream,
                            arm - 1,
                            &qkvx_dev,
                            &beta_dev,
                            &decay_dev,
                            &states[arm],
                            &out_dev,
                            HD,
                            N_HEAD,
                            p,
                            V_DIM,
                        )
                        .expect("l8 launch");
                    }
                }
                stream.synchronize().expect("sync");
                t.elapsed().as_secs_f64() * 1e6
            };

            // Warm-up (untimed) — every arm once.
            for arm in 0..=RECL8_GEOMETRIES.len() {
                run_arm(arm);
            }

            // Interleaved pairs, order alternating per pair (the Bench-723
            // sequential-A/B lesson: the ratio lands on both arms of a pair,
            // so box drift cannot dress up as an arm win).
            let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); RECL8_GEOMETRIES.len()];
            for pair in 0..PAIRS {
                for arm in 1..=RECL8_GEOMETRIES.len() {
                    let (t_a, t_b) = if pair % 2 == 0 {
                        (run_arm(0), run_arm(arm))
                    } else {
                        (run_arm(arm), run_arm(0))
                    };
                    let (t_inc, t_l8) = if pair % 2 == 0 {
                        (t_a, t_b)
                    } else {
                        (t_b, t_a)
                    };
                    ratios[arm - 1].push(t_l8 / t_inc);
                }
            }

            for (arm, &(k, wpc)) in RECL8_GEOMETRIES.iter().enumerate() {
                let mut rs = ratios[arm].clone();
                rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = rs[rs.len() / 2];
                println!(
                    "P={p:>5} recl8_k{k}w{wpc}: median ratio {med:.4}x vs incumbent ({}) over {} pairs",
                    if med < 1.0 { "FASTER" } else { "slower" },
                    rs.len(),
                );
            }
            let inc_us = run_arm(0);
            println!(
                "P={p:>5} incumbent recmr r4w4: {:.1} µs total ({:.2} µs/token)",
                inc_us,
                inc_us / p as f64
            );
        }
    }

    /// Plan 621 T5 — the e2e prefill A/B on the LEAGUE model (Bonsai-2-27B
    /// PQ2_0): the whole-prefill cudarc stack at the league cell (p=2048
    /// chunks), knob OFF vs ON, INTERLEAVED rounds (reset_state between
    /// rounds — the GDN recurrent state carries in place), FNV of the final
    /// logits asserted EQUAL (bit-identity e2e — the bytewise-G1 class's
    /// whole-model proof) + tok/s beside each round.
    ///
    /// The L8 win is kernel-only 1.26× at p=2048 (recl8_kernel_only_bench);
    /// the recurrence is one stage of the GDN layer, so this lane measures
    /// the honest e2e delta (expected single-digit %).
    ///
    /// ```bash
    /// RIIR_PREFILL_CUDA_GRAPHS=0 cargo test -p riir-infer-gpu --release \
    ///   --features ternary_gemv_cuda_raw,ternary_gemm_batched,cubecl_runtime \
    ///   --lib prefill_cuda_deltanet_l8::tests::recl8_e2e_league_prefill_ab \
    ///   -- --ignored --nocapture
    /// ```
    ///
    /// Env: `BONSAI_GGUF` (default the sibling riir-train data path).
    /// `RIIR_PREFILL_CUDA_GRAPHS=0` REQUIRED (the knob must be consulted per
    /// launch — a captured graph would freeze one arm's kernel sequence; the
    /// D5 law). Requires `RIIR_PREFILL_GDN_L8` UNSET in the invocation env
    /// (the test drives the knob via the setter; the env would win over it).
    #[test]
    #[ignore = "league-model GPU lane — run explicitly on an idle 4090 (7 min incl. load)"]
    fn recl8_e2e_league_prefill_ab() {
        let Some(_) = cuda_or_skip() else {
            return;
        };
        if std::env::var("RIIR_PREFILL_CUDA_GRAPHS").ok().as_deref() != Some("0") {
            panic!(
                "run with RIIR_PREFILL_CUDA_GRAPHS=0 — a captured graph would freeze one arm's kernel sequence (the D5 law)"
            );
        }
        if std::env::var("RIIR_PREFILL_GDN_L8").is_ok() {
            panic!(
                "RIIR_PREFILL_GDN_L8 must be UNSET — the test drives the knob via the setter and the env wins over it"
            );
        }
        let model_path = std::env::var("BONSAI_GGUF").unwrap_or_else(|_| {
            "../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".to_string()
        });
        let model_path = std::path::PathBuf::from(model_path);

        eprintln!("[621-l8-e2e] loading {} …", model_path.display());
        let t0 = std::time::Instant::now();
        let (mut config, weights) =
            riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf(&model_path)
                .map_err(|e| format!("load: {e}"))
                .unwrap();
        let gguf = riir_infer_core::gguf_loader::GgufFile::open(&model_path)
            .map_err(|e| format!("reopen: {e}"))
            .unwrap();
        let tok = riir_infer_core::tokenizer::BpeTokenizer::from_gguf(&gguf)
            .map_err(|e| format!("tokenizer: {e}"))
            .unwrap();
        let bos = config.bos_token;
        // Headroom: the league cell p=2048 (+64 slack) — the DQ-lane clamp.
        config.block_size = config.block_size.min(2048 + 64);
        let ctx = crate::CubeCLContext::new()
            .map_err(|e| format!("GPU init: {e}"))
            .unwrap();
        let mut fwd = crate::TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
        eprintln!(
            "[621-l8-e2e] forward ready ({:.0}s) · n_layer={} · n_v heads={}",
            t0.elapsed().as_secs_f32(),
            config.n_layer,
            config.deltanet_linear_n_value_heads,
        );

        // The shipping knob posture (the bench_947/948 pin set) + the CUDA
        // whole-prefill arm (the DQ-lane recipe).
        use crate::prefill_cuda_full::set_prefill_gdn_l8;
        use crate::ternary_deltanet_gpu_forward as tdf;
        tdf::set_prefill_use_gemv(false);
        tdf::set_prefill_seq_rmsnorm(false);
        tdf::set_prefill_zero_scratch(false);
        tdf::set_prefill_batch_elementwise(true);
        tdf::set_prefill_use_cmma16(true);
        tdf::set_prefill_use_cmma_i8(true);
        tdf::set_prefill_cmma_i8_psplit(true);
        crate::set_prefill_use_cuda_mma(false);
        crate::set_prefill_use_cuda_ffn(false);
        tdf::set_prefill_chunk_max(4096);
        crate::prefill_cuda_full::set_prefill_use_cuda(true);

        // One deterministic ~2048-token prompt (the BPE fill loop, the
        // g1_032 prompt() shape without the fixture pin — this test compares
        // ARMS, not a stored pin).
        let words: Vec<&str> = (0..4096)
            .map(|i| {
                const STEMS: [&str; 8] = [
                    "the", "river", "bends", "quiet", "under", "stone", "light", "again",
                ];
                STEMS[i % 8]
            })
            .collect();
        let mut tokens = tok.encode(&words.join(" "));
        if tokens.first() != Some(&bos) {
            tokens.insert(0, bos);
        }
        tokens.truncate(2048);
        eprintln!("[621-l8-e2e] prompt tokens={}", tokens.len());

        fn logits_fnv(v: &[f32]) -> u64 {
            v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
                x.to_bits().to_le_bytes().iter().fold(h, |h, b| {
                    (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
                })
            })
        }

        let rounds = 4usize;
        let mut fnv_off: Vec<u64> = Vec::new();
        let mut fnv_on: Vec<u64> = Vec::new();
        let mut secs_off: Vec<f64> = Vec::new();
        let mut secs_on: Vec<f64> = Vec::new();

        for round in 0..rounds {
            let off = round % 2 == 0;
            set_prefill_gdn_l8(!off);
            fwd.reset_state();
            let t = std::time::Instant::now();
            let logits = fwd.prefill(&tokens);
            let dt = t.elapsed().as_secs_f64();
            let f = logits_fnv(&logits);
            let argmax = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            eprintln!(
                "[621-l8-e2e] round {round} L8={} : {:.1} tok/s ({dt:.3}s) · argmax {argmax} · fnv {f:016x}",
                !off,
                tokens.len() as f64 / dt
            );
            if off {
                fnv_off.push(f);
                secs_off.push(dt);
            } else {
                fnv_on.push(f);
                secs_on.push(dt);
            }
        }
        set_prefill_gdn_l8(false);

        // BIT-IDENTITY e2e: every round's FNV equal across arms.
        for (i, (&a, &b)) in fnv_off.iter().zip(fnv_on.iter()).enumerate() {
            assert_eq!(
                a, b,
                "e2e FNV mismatch round pair {i}: L8 changed the prefill logits — the bit-identity claim is FALSE at the model level"
            );
        }
        assert_eq!(
            fnv_off.first(),
            fnv_off.last(),
            "arm-off rounds disagree (box drift) — rerun on a quiet box"
        );

        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (mo, mn) = (med(secs_off), med(secs_on));
        println!(
            "[621-l8-e2e] VERDICT: FNV identical {}/{} pairs · e2e L8/rowpar = {:.4}x ({:.1} vs {:.1} tok/s median)",
            fnv_on.len(),
            fnv_on.len(),
            mn / mo,
            tokens.len() as f64 / mo,
            tokens.len() as f64 / mn
        );
    }
}
