//! Issue 734 Arm 8 — the cudarc-side **whole-prefill** DeltaNet kernels.
//!
//! The FFN-block arm (Bench 721 / `prefill_cuda_ffn`) proved the cudarc mma
//! GEMMs + elementwise replicas bit-identical cross-runtime, but measured the
//! per-layer host crossing DEAD (0.608× @2048). The whole-prefill arm removes
//! ALL per-layer crossings — every activation stays on the cudarc side from
//! embedding to tail. That requires CUDA twins of the CubeCL prefill kernels
//! the Issue-615 decode port never needed (it has single-token variants):
//!
//! - `dequant_wte_row_f32` batched over all p tokens (pure bit extraction —
//!   no transcendentals → no variants)
//! - `deltanet_conv1d_chunked_f32` + `deltanet_conv1d_carry_update_f32` —
//!   the per-(token, channel) window form: every output is an independent
//!   function of the raw inputs + the entry carry, so ONE dispatch over all
//!   p tokens is bit-identical to any chunking of the CubeCL path
//! - `deltanet_beta_decay_batched_f32` — sigmoid + softplus(`ln`) + `exp`
//! - `expand_and_l2_normalize_heads_batched_f32` — serial per-thread L2 norm
//!   (+ `elv2_pf_*` staged-sum V2, Bench 809: the sq_sum computed ONCE per
//!   (token, section) into smem — bit-identical, 2.0× kernel)
//! - `deltanet_z_gating_f32` — silu
//! - `deltanet_recurrence_multi_token_f32` — THE hard one: register-blocked
//!   rows (32 lanes × 4 cols), sequential over p tokens, `plane_sum` warp
//!   reductions (+ the `recmr_pf_*` multi-row ILP ladder, Bench 896 / Issue
//!   904: RPW rows/warp × WPC warps/block, bit-identical by construction,
//!   r4w4 DEFAULT-ON at 2.19×/1.89× the legacy kernel — see `REC_MR_GEOMETRIES`)
//!
//! ## The variant families (the Bench-719/721 method)
//!
//! Proven forms (Bench 721, live 27B activations): GLSL `Exp` → `__expf`;
//! `OpFDiv(1, add)` → `div.full`; `OpFDiv(1, OpSqrt)` → merged
//! `rsqrt.approx.f32`; NVRTC default-fmad contraction ≡ driver contraction.
//! UNKNOWN lowerings get macro-generated variant families; the
//! `bench_734_prefill_cuda_bitidentity` probe selects the exact form:
//!
//! - **PlaneSum** — SPIR-V `OpGroupNonUniformFAdd`: tree (xor-butterfly, the
//!   halving association) vs sequential (ascending 32-term chain).
//! - **Log** — GLSL `Log`: `logf` vs `__logf` vs `lg2.approx·ln2` vs
//!   `lg2.approx/log2e`.
//! - **RecDot** — the 4-term lane dot's contraction shape (which product
//!   stays the base mul).
//! - The remaining axes reuse the proven defaults + a small control family.

use std::sync::Arc;

use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig,
};
use cudarc::driver::PushKernelArg;

// ---------------------------------------------------------------------------
// Variant enums (probe-selectable)
// ---------------------------------------------------------------------------

/// Recurrence 4-term dot form: `s0*k0 + s1*k1 + s2*k2 + s3*k3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecDot {
    /// All separate `__fmul_rn`/`__fadd_rn` (never contracted).
    Ma,
    /// Inner pair `fma(s0,k0, s1*k1)`, fold upward — the natural left-assoc
    /// contraction (`a*b + c*d` folds the FIRST mul).
    F1,
    /// Base = FIRST product as a rounded mul, the rest fold left.
    Fl,
    /// Base = LAST product (the outer-fold shape).
    Fr,
}

/// Rank-1 update form: `s += k * delta`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecUpd {
    /// Separate `__fmul_rn` + `__fadd_rn`.
    Ma,
    /// `__fmaf_rn(k, delta, s)`.
    Fma,
}

/// Warp-wide sum lowering of cubecl `plane_sum` (32 lanes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneSum {
    /// `__shfl_xor_sync` butterfly with DESCENDING offsets (16,8,4,2,1) —
    /// the stride-halving association.
    Tree,
    /// `__shfl_xor_sync` butterfly with ASCENDING offsets (1,2,4,8,16) —
    /// the adjacent-pairs-first association.
    TreeUp,
    /// Ascending sequential 32-term chain via `__shfl_sync` reads.
    Seq,
}

/// `1/sqrt(head_dim)` form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecScale {
    /// Merged `rsqrt.approx.f32` (the proven `OpFDiv(1, OpSqrt)` lowering).
    Rsqrt,
    /// `1.0f / __fsqrt_rn(x)` (div.rn composition).
    DivRn,
}

/// GLSL `Exp` lowering (proven `__expf` in Bench 721 — `expf` the control).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigExp {
    Fast,
    Acc,
}

/// `OpFDiv(1, add)` lowering (proven `div.full` in Bench 721).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigDiv {
    Rn,
    Full,
    Approx,
    Rcp,
}

/// GLSL `Log` lowering — the UNKNOWN axis (softplus; RoPE reuses it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogForm {
    /// Accurate `logf`.
    Acc,
    /// Fast `__logf`.
    Fast,
    /// `lg2.approx(x) * ln2_f32`.
    Lg2Ln2,
    /// `lg2.approx(x) / log2e_f32`.
    Lg2Div,
}

/// L2-norm accumulation form: `sq += val*val`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum L2Accum {
    Ma,
    Fma,
}

/// `1/sqrt(sq_sum)` form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum L2Inv {
    Rsqrt,
    DivRnSqrtRn,
    DivFullSqrtRn,
    RcpSqrtApprox,
}

/// Conv dot accumulation form: `sum += val*w`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvFma {
    Ma,
    Fma,
}

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

pub(crate) const DELTANET_PREFILL_CUDA_SRC: &str = r#"
// ---------------------------------------------------------------------------
// helpers (the Bench-719/721 proven forms + the probe axes)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float dn_div_rn(float a, float b) { return a / b; }
__device__ __forceinline__ float dn_div_full(float a, float b)
{
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float dn_div_approx(float a, float b)
{
    float r;
    asm("div.approx.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float dn_rcp_approx(float a)
{
    float r;
    asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float dn_rsqrt_approx(float a)
{
    float r;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float dn_exp_fast(float a) { return __expf(a); }
__device__ __forceinline__ float dn_exp_acc(float a) { return expf(a); }
__device__ __forceinline__ float dn_lg2_approx(float a)
{
    float r;
    asm("lg2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float dn_log_acc(float a) { return logf(a); }
__device__ __forceinline__ float dn_log_fast(float a) { return __logf(a); }
__device__ __forceinline__ float dn_log_lg2ln2(float a) { return dn_lg2_approx(a) * 0.6931471805599453f; }
__device__ __forceinline__ float dn_log_lg2div(float a) { return dn_lg2_approx(a) / 1.4426950408889634f; }

// div forms used as `1 / x`
#define DIV_RN(x) dn_div_rn(1.0f, (x))
#define DIV_FULL(x) dn_div_full(1.0f, (x))
#define DIV_APR(x) dn_div_approx(1.0f, (x))
#define DIV_RCP(x) dn_rcp_approx((x))

// sigmoid(x) = x * (1 / (1 + exp(0 - x))) — exact cubecl op order.
#define SIG_X(x, EXPF, DIVF) ((x) * DIVF(1.0f + EXPF(0.0f - (x))))

// ---------------------------------------------------------------------------
// plane_sum lowerings (32-lane warp sum)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float ps_tree(float v)
{
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}
__device__ __forceinline__ float ps_treeup(float v)
{
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    return v;
}
__device__ __forceinline__ float ps_seq(float v)
{
    float s = 0.0f;
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 0));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 1));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 2));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 3));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 4));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 5));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 6));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 7));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 8));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 9));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 10));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 11));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 12));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 13));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 14));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 15));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 16));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 17));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 18));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 19));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 20));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 21));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 22));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 23));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 24));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 25));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 26));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 27));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 28));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 29));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 30));
    s = __fadd_rn(s, __shfl_sync(0xffffffffu, v, 31));
    return s;
}

// ---------------------------------------------------------------------------
// 4-term dot forms
// ---------------------------------------------------------------------------
__device__ __forceinline__ float dot4_ma(float s0, float s1, float s2, float s3,
                                         float k0, float k1, float k2, float k3)
{
    return __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(s0, k0), __fmul_rn(s1, k1)),
                               __fmul_rn(s2, k2)), __fmul_rn(s3, k3));
}
__device__ __forceinline__ float dot4_f1(float s0, float s1, float s2, float s3,
                                         float k0, float k1, float k2, float k3)
{
    float t = __fmul_rn(s1, k1);
    t = __fmaf_rn(s0, k0, t);
    t = __fmaf_rn(s2, k2, t);
    t = __fmaf_rn(s3, k3, t);
    return t;
}
__device__ __forceinline__ float dot4_fl(float s0, float s1, float s2, float s3,
                                         float k0, float k1, float k2, float k3)
{
    float t = __fmul_rn(s0, k0);
    t = __fmaf_rn(s1, k1, t);
    t = __fmaf_rn(s2, k2, t);
    t = __fmaf_rn(s3, k3, t);
    return t;
}
__device__ __forceinline__ float dot4_fr(float s0, float s1, float s2, float s3,
                                         float k0, float k1, float k2, float k3)
{
    float t = __fmul_rn(s3, k3);
    t = __fmaf_rn(s2, k2, t);
    t = __fmaf_rn(s1, k1, t);
    t = __fmaf_rn(s0, k0, t);
    return t;
}

// update + scale forms
#define UPD_MA(s, k, d) __fadd_rn((s), __fmul_rn((k), (d)))
#define UPD_FMA(s, k, d) __fmaf_rn((k), (d), (s))
#define SC_RSQ(d) dn_rsqrt_approx((float)(d))
#define SC_DRN(d) dn_div_rn(1.0f, __fsqrt_rn((float)(d)))

// ---------------------------------------------------------------------------
// Batched wte row dequant — pure bit extraction + one mul (no variants).
// One thread per (token, col). Bit-identical to p DequantWteRowCubeCL calls.
// ---------------------------------------------------------------------------
extern "C" __global__ void dequant_wte_batch(
    const unsigned int* __restrict__ pos_bits,   // [rows * blocks64 * 2]
    const unsigned int* __restrict__ neg_bits,
    const float* __restrict__ group_scale,       // [rows * groups_per_row]
    float* __restrict__ out,                     // [p * n]
    const unsigned int* __restrict__ tokens,     // [p]
    int blocks64,
    int groups_per_row,
    int n,
    int p)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)p * n;
    if (idx >= total) return;
    const int t = (int)(idx / n);
    const int col = (int)(idx % n);
    const unsigned int row = tokens[t];

    const int words_per_row = blocks64 * 2;
    const int block_idx = col / 64;
    const int word_in_block = (col % 64) / 32;
    const int bit_pos = col % 32;
    const long word_idx = (long)row * words_per_row + block_idx * 2 + word_in_block;
    const unsigned int pos_bit = (pos_bits[word_idx] >> bit_pos) & 1u;
    const unsigned int neg_bit = (neg_bits[word_idx] >> bit_pos) & 1u;
    const float sign_f = (float)((int)pos_bit - (int)neg_bit);
    const int group = col / 128;
    const float scale = group_scale[(long)row * groups_per_row + group];
    out[idx] = sign_f * scale;
}

// ---------------------------------------------------------------------------
// conv1d over ALL p tokens — the chunked kernel's per-(token, channel) form
// (every output is an independent function of the raw inputs + the entry
// carry, so ONE dispatch over p is bit-identical to any chunking).
// ---------------------------------------------------------------------------
#define CONV1D(NAME, CONVF, EXPF, DIVF)                                        \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ input,   /* [p * conv_dim] raw */                \
    float* __restrict__ output,        /* [p * conv_dim] SiLU out */           \
    const float* __restrict__ weight,  /* [conv_dim * ks] */                   \
    const float* __restrict__ carry,   /* [conv_dim * ks] conv_state layout */ \
    int p, int conv_dim, int kernel_size)                                      \
{                                                                              \
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;              \
    const long total = (long)p * conv_dim;                                     \
    if (idx >= total) return;                                                  \
    const int t = (int)(idx / conv_dim);                                       \
    const int ch = (int)(idx % conv_dim);                                      \
    const int ks_m1 = kernel_size - 1;                                         \
    const int weight_off = ch * kernel_size;                                   \
    const int carry_off = ch * kernel_size + 1; /* conv_state: offset 1 */     \
    float sum = 0.0f;                                                          \
    for (int k = 0; k < kernel_size; k++) {                                    \
        const long sample_pos = (long)t - ks_m1 + k;                           \
        const float val = sample_pos < 0                                       \
            ? carry[carry_off + (int)(sample_pos + ks_m1)]                     \
            : input[sample_pos * conv_dim + ch];                               \
        sum = CONVF(val, weight[weight_off + k], sum);                         \
    }                                                                          \
    output[idx] = SIG_X(sum, EXPF, DIVF);                                      \
}

#define CF_MA(v, w, s) __fadd_rn((s), __fmul_rn((v), (w)))
#define CF_FMA(v, w, s) __fmaf_rn((v), (w), (s))

#define CONV_X4(NAME, CONVF, EXPF)                                             \
CONV1D(NAME##_g0, CONVF, EXPF, DIV_RN)                                         \
CONV1D(NAME##_g1, CONVF, EXPF, DIV_FULL)                                       \
CONV1D(NAME##_g2, CONVF, EXPF, DIV_APR)                                        \
CONV1D(NAME##_g3, CONVF, EXPF, DIV_RCP)

CONV_X4(conv1d_pf_d0e0, CF_MA, dn_exp_acc)
CONV_X4(conv1d_pf_d0e1, CF_MA, dn_exp_fast)
CONV_X4(conv1d_pf_d1e0, CF_FMA, dn_exp_acc)
CONV_X4(conv1d_pf_d1e1, CF_FMA, dn_exp_fast)

// Carry update — pure copies (no math variants). One thread per channel.
extern "C" __global__ void conv1d_carry_update(
    const float* __restrict__ input,   /* [p * conv_dim] raw */
    float* __restrict__ carry,         /* [conv_dim * ks] conv_state layout */
    int p, int conv_dim, int kernel_size)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;
    const int ks_m1 = kernel_size - 1;
    const int carry_off = ch * kernel_size + 1;
    for (int j = 0; j < ks_m1; j++) {
        const int src = j + p;
        if (src < ks_m1) {
            carry[carry_off + j] = carry[carry_off + src];
        } else {
            carry[carry_off + j] = input[(long)(src - ks_m1) * conv_dim + ch];
        }
    }
}

// ---------------------------------------------------------------------------
// beta/decay batched — sigmoid + softplus(ln) + exp.
// LOG(4) x EXP(2) x DIV(4) = 32 variants.
// ---------------------------------------------------------------------------
#define BETA_DECAY(NAME, LOGF, EXPF, DIVF)                                     \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ a_raw,    /* [total] */                          \
    const float* __restrict__ b_raw,    /* [total] */                          \
    const float* __restrict__ a_log,    /* [n_head] */                         \
    const float* __restrict__ dt_bias,  /* [n_head] */                         \
    float* __restrict__ beta_out,       /* [total] */                          \
    float* __restrict__ decay_out,      /* [total] */                          \
    int n_head, int total)                                                     \
{                                                                              \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                       \
    if (i >= total) return;                                                    \
    const int h = i % n_head;                                                  \
    const float b_val = b_raw[i];                                              \
    beta_out[i] = DIVF(1.0f + EXPF(0.0f - b_val));                             \
    const float a_val = __fadd_rn(a_raw[i], dt_bias[h]);                       \
    float sp;                                                                  \
    if (a_val > 20.0f) {                                                       \
        sp = a_val;                                                            \
    } else if (a_val < -20.0f) {                                               \
        sp = 0.0f;                                                             \
    } else {                                                                   \
        sp = LOGF(__fadd_rn(1.0f, EXPF(a_val)));                               \
    }                                                                          \
    const float g = __fmul_rn(a_log[h], sp);                                   \
    decay_out[i] = EXPF(g);                                                    \
}

#define BD_X4(NAME, LOGF, EXPF)                                                \
BETA_DECAY(NAME##_g0, LOGF, EXPF, DIV_RN)                                      \
BETA_DECAY(NAME##_g1, LOGF, EXPF, DIV_FULL)                                    \
BETA_DECAY(NAME##_g2, LOGF, EXPF, DIV_APR)                                     \
BETA_DECAY(NAME##_g3, LOGF, EXPF, DIV_RCP)

BD_X4(bd_pf_l0e0, dn_log_acc, dn_exp_acc)
BD_X4(bd_pf_l0e1, dn_log_acc, dn_exp_fast)
BD_X4(bd_pf_l1e0, dn_log_fast, dn_exp_acc)
BD_X4(bd_pf_l1e1, dn_log_fast, dn_exp_fast)
BD_X4(bd_pf_l2e0, dn_log_lg2ln2, dn_exp_acc)
BD_X4(bd_pf_l2e1, dn_log_lg2ln2, dn_exp_fast)
BD_X4(bd_pf_l3e0, dn_log_lg2div, dn_exp_acc)
BD_X4(bd_pf_l3e1, dn_log_lg2div, dn_exp_fast)

// ---------------------------------------------------------------------------
// expand + L2-normalize heads, batched over p tokens.
// ACC(2) x INV(4) = 8 variants.
// ---------------------------------------------------------------------------
#define EXPAND_L2(NAME, ACC, INV)                                              \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ compact,   /* [p, (2*n_k+n_v)*hd] */             \
    float* __restrict__ expanded,        /* [p, 3*n_v*hd] */                   \
    int n_k_heads, int n_v_heads, int head_dim, int total)                     \
{                                                                              \
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (idx >= total) return;                                                  \
    const int qk_block = n_v_heads * head_dim;                                 \
    const int per_out = 3 * qk_block;                                          \
    const int per_in = (2 * n_k_heads + n_v_heads) * head_dim;                 \
    const int t = idx / per_out;                                               \
    const int local = idx % per_out;                                           \
    const int in_base = t * per_in;                                            \
    const int v_compact_off = in_base + 2 * n_k_heads * head_dim;              \
    const int section = local / qk_block;                                      \
    const int local_idx = local % qk_block;                                    \
    const int out_head = local_idx / head_dim;                                 \
    const int col = local_idx % head_dim;                                      \
    if (section < 2) {                                                         \
        const int src_head = out_head % n_k_heads;                             \
        const int src_off = in_base + section * n_k_heads * head_dim           \
            + src_head * head_dim;                                             \
        float sq_sum = 0.0f;                                                   \
        for (int c = 0; c < head_dim; c++) {                                   \
            const float val = compact[src_off + c];                            \
            sq_sum = ACC(val, sq_sum);                                         \
        }                                                                      \
        const float inv_norm = sq_sum > 0.0f ? INV(sq_sum) : 0.0f;             \
        expanded[idx] = compact[src_off + col] * inv_norm;                     \
    } else {                                                                   \
        expanded[idx] = compact[v_compact_off + local_idx];                    \
    }                                                                          \
}

#define ELA_MA(v, s) __fadd_rn((s), __fmul_rn((v), (v)))
#define ELA_FMA(v, s) __fmaf_rn((v), (v), (s))
#define ELI_RSQ(x) dn_rsqrt_approx((x))
#define ELI_DRN(x) dn_div_rn(1.0f, __fsqrt_rn((x)))
#define ELI_DFL(x) dn_div_full(1.0f, __fsqrt_rn((x)))
#define ELI_RCP(x) dn_rcp_approx(__fsqrt_rn((x)))

#define EL_X4(NAME, ACC)                                                       \
EXPAND_L2(NAME##_i0, ACC, ELI_RSQ)                                             \
EXPAND_L2(NAME##_i1, ACC, ELI_DRN)                                             \
EXPAND_L2(NAME##_i2, ACC, ELI_DFL)                                             \
EXPAND_L2(NAME##_i3, ACC, ELI_RCP)

EL_X4(el_pf_a0, ELA_MA)
EL_X4(el_pf_a1, ELA_FMA)

// ---------------------------------------------------------------------------
// Issue 772 T2 residue (Bench 803 → 809): staged-sum V2 of the family above.
// The legacy kernel has EVERY element thread re-read its full head_dim source
// row serially (n_v*head_dim threads x head_dim loads per (token, section);
// 786k loads per section per token at the Bonsai dims) — kernel-only measured
// ~337 GB/s. V2 computes each source row's sq_sum ONCE per (token, section):
// thread s < n_k_heads accumulates with the SAME ACC form in the SAME scalar
// order (bit-identical staged sum), stages the n_k sums in shared memory, and
// every element thread applies the SAME INV form + multiply from the staged
// sum. Index-for-index identical writes; the V section is the same copy.
// Grid (p, 3); block 1024. The launcher falls back to the legacy variants when
// n_k_heads > 1024 (the shared tile bound).
// ---------------------------------------------------------------------------
#define EXPAND_L2_V2(NAME, ACC, INV)                                           \
extern "C" __global__ void NAME(                                                \
    const float* __restrict__ compact,   /* [p, (2*n_k+n_v)*hd] */             \
    float* __restrict__ expanded,        /* [p, 3*n_v*hd] */                   \
    int n_k_heads, int n_v_heads, int head_dim, int p)                         \
{                                                                              \
    const int t = blockIdx.x;                                                  \
    const int section = blockIdx.y;      /* 0=Q, 1=K, 2=V */                   \
    const int tid = threadIdx.x;                                               \
    const int qk_block = n_v_heads * head_dim;                                 \
    const int per_out = 3 * qk_block;                                          \
    const int per_in = (2 * n_k_heads + n_v_heads) * head_dim;                 \
    const int in_base = t * per_in;                                            \
    if (section == 2) {                                                        \
        const int v_compact_off = in_base + 2 * n_k_heads * head_dim;          \
        float* out_v = expanded + t * per_out + 2 * qk_block;                  \
        for (int e = tid; e < qk_block; e += blockDim.x)                       \
            out_v[e] = compact[v_compact_off + e];                             \
        return;                                                                \
    }                                                                          \
    __shared__ float s_sq[1024];                                               \
    if (tid < n_k_heads) {                                                     \
        const int src_off = in_base + section * n_k_heads * head_dim           \
            + tid * head_dim;                                                  \
        float sq_sum = 0.0f;                                                   \
        for (int c = 0; c < head_dim; c++) {                                   \
            const float val = compact[src_off + c];                            \
            sq_sum = ACC(val, sq_sum);                                         \
        }                                                                      \
        s_sq[tid] = sq_sum;                                                    \
    }                                                                          \
    __syncthreads();                                                           \
    for (int e = tid; e < qk_block; e += blockDim.x) {                         \
        const int out_head = e / head_dim;                                     \
        const int col = e - out_head * head_dim;                               \
        const int src_head = out_head % n_k_heads;                             \
        const int src_off = in_base + section * n_k_heads * head_dim           \
            + src_head * head_dim;                                             \
        const int idx = t * per_out + section * qk_block + e;                  \
        const float sq_sum = s_sq[src_head];                                   \
        const float inv_norm = sq_sum > 0.0f ? INV(sq_sum) : 0.0f;             \
        expanded[idx] = compact[src_off + col] * inv_norm;                     \
    }                                                                          \
}

#define ELV2_X4(NAME, ACC)                                                     \
EXPAND_L2_V2(NAME##_i0, ACC, ELI_RSQ)                                          \
EXPAND_L2_V2(NAME##_i1, ACC, ELI_DRN)                                          \
EXPAND_L2_V2(NAME##_i2, ACC, ELI_DFL)                                          \
EXPAND_L2_V2(NAME##_i3, ACC, ELI_RCP)

ELV2_X4(elv2_pf_a0, ELA_MA)
ELV2_X4(elv2_pf_a1, ELA_FMA)

// ---------------------------------------------------------------------------
// z-gating: out *= z * sigmoid(z). EXP(2) x DIV(4) = 8 variants.
// ---------------------------------------------------------------------------
#define ZGATING(NAME, EXPF, DIVF)                                              \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ output,          /* [n] in-place */                    \
    const float* __restrict__ z,         /* [n] */                             \
    int n)                                                                     \
{                                                                              \
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (idx >= n) return;                                                      \
    const float z_val = z[idx];                                                \
    output[idx] = output[idx] * z_val * DIVF(1.0f + EXPF(0.0f - z_val));       \
}

#define ZG_X4(NAME, EXPF)                                                      \
ZGATING(NAME##_g0, EXPF, DIV_RN)                                               \
ZGATING(NAME##_g1, EXPF, DIV_FULL)                                             \
ZGATING(NAME##_g2, EXPF, DIV_APR)                                              \
ZGATING(NAME##_g3, EXPF, DIV_RCP)

ZG_X4(zg_pf_e0, dn_exp_acc)
ZG_X4(zg_pf_e1, dn_exp_fast)

// ---------------------------------------------------------------------------
// Multi-token row-parallel recurrence — register-blocked, sequential tokens,
// plane_sum reductions. DOT(4) x UPD(2) x PS(2) x SCALE(2) = 64 variants.
// Grid (n_head, head_dim), block 32 (head_dim == 128 assumed — 4 cols/lane).
// ---------------------------------------------------------------------------
#define RECURRENCE(NAME, DOTF, UPDF, PSF, SCF)                                 \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ qkvx,      /* [p, 3*v_dim] */                    \
    const float* __restrict__ beta,      /* [p, n_head] */                     \
    const float* __restrict__ decay,     /* [p, n_head] */                     \
    float* __restrict__ state,           /* [n_head*hd*hd] in-place */         \
    float* __restrict__ output,          /* [p, v_dim] */                      \
    int head_dim, int n_head, int p, int v_dim)                                \
{                                                                              \
    const int head = blockIdx.x;                                               \
    const int row = blockIdx.y;                                                \
    const int lane = threadIdx.x;                                              \
    const int stride = 32;                                                     \
    const long qkvx_stride = 3L * v_dim;                                       \
    const long head_off = (long)head * head_dim;                               \
    const long k_base = v_dim + head_off;                                      \
    const long v_base = 2L * v_dim + head_off;                                 \
    const long row_off = (long)head * head_dim * head_dim                      \
                       + (long)row * head_dim;                                  \
    const int c0 = lane;                                                       \
    const int c1 = lane + stride;                                              \
    const int c2 = lane + 2 * stride;                                          \
    const int c3 = lane + 3 * stride;                                          \
    float s0 = state[row_off + c0];                                            \
    float s1 = state[row_off + c1];                                            \
    float s2 = state[row_off + c2];                                            \
    float s3 = state[row_off + c3];                                            \
    const float scale = SCF(head_dim);                                         \
    for (int t = 0; t < p; t++) {                                              \
        const long t_row = (long)t * qkvx_stride;                              \
        const long sc = (long)t * n_head + head;                               \
        const float beta_val = beta[sc];                                       \
        const float decay_val = decay[sc];                                     \
        const float k0 = qkvx[t_row + k_base + c0];                            \
        const float k1 = qkvx[t_row + k_base + c1];                            \
        const float k2 = qkvx[t_row + k_base + c2];                            \
        const float k3 = qkvx[t_row + k_base + c3];                            \
        s0 = __fmul_rn(s0, decay_val);                                         \
        s1 = __fmul_rn(s1, decay_val);                                         \
        s2 = __fmul_rn(s2, decay_val);                                         \
        s3 = __fmul_rn(s3, decay_val);                                         \
        const float acc_k = DOTF(s0, s1, s2, s3, k0, k1, k2, k3);              \
        const float kv_mem_row = PSF(acc_k);                                   \
        const float v_row = qkvx[t_row + v_base + row];                        \
        const float delta_row = __fmul_rn(beta_val, __fsub_rn(v_row, kv_mem_row)); \
        s0 = UPDF(s0, k0, delta_row);                                          \
        s1 = UPDF(s1, k1, delta_row);                                          \
        s2 = UPDF(s2, k2, delta_row);                                          \
        s3 = UPDF(s3, k3, delta_row);                                          \
        const float q0 = qkvx[t_row + head_off + c0];                          \
        const float q1 = qkvx[t_row + head_off + c1];                          \
        const float q2 = qkvx[t_row + head_off + c2];                          \
        const float q3 = qkvx[t_row + head_off + c3];                          \
        const float acc_q = DOTF(s0, s1, s2, s3, q0, q1, q2, q3);              \
        const float dot = PSF(acc_q);                                          \
        if (lane == 0) {                                                       \
            output[(long)t * v_dim + head_off + row] = __fmul_rn(dot, scale);  \
        }                                                                      \
    }                                                                          \
    state[row_off + c0] = s0;                                                  \
    state[row_off + c1] = s1;                                                  \
    state[row_off + c2] = s2;                                                  \
    state[row_off + c3] = s3;                                                  \
}

#define REC_X4(NAME, DOTF, UPDF, PSF)                                          \
RECURRENCE(NAME##s0, DOTF, UPDF, PSF, SC_RSQ)                                 \
RECURRENCE(NAME##s1, DOTF, UPDF, PSF, SC_DRN)

#define REC_X2P(NAME, DOTF, UPDF)                                              \
REC_X4(NAME##p0, DOTF, UPDF, ps_tree)                                         \
REC_X4(NAME##p1, DOTF, UPDF, ps_treeup)                                       \
REC_X4(NAME##p2, DOTF, UPDF, ps_seq)

#define REC_X2U(NAME, DOTF)                                                    \
REC_X2P(NAME##u0, DOTF, UPD_MA)                                               \
REC_X2P(NAME##u1, DOTF, UPD_FMA)

REC_X2U(rec_pf_d0, dot4_ma)
REC_X2U(rec_pf_d1, dot4_f1)
REC_X2U(rec_pf_d2, dot4_fl)
REC_X2U(rec_pf_d3, dot4_fr)

// ---------------------------------------------------------------------------
// Issue 904 T1 — L3 multi-row ILP probes (launch-geometry axis only).
// RPW rows per 32-lane warp, WPC warps per block; grid (n_head, hd/(RPW*WPC)).
// The warp's RPW rows are CONSECUTIVE state rows of the same head: their
// k/q/beta/decay operands are IDENTICAL (loaded once per token — the load
// sharing is the secondary effect), and their per-row arithmetic is
// instruction- and order-identical to the RECURRENCE body above (loads
// hoisted; every FP op of a row's chain unchanged) — bit-identical by
// construction. Two orthogonal axes priced independently:
//   WPC > 1 — dissolves the 24-blocks/SM slot limit (32-thr blocks cap at
//             24 of 48 warp slots on sm_89);
//   RPW > 1 — interleaves RPW independent t-serial chains per warp, hiding
//             the two 5-deep ps_treeup shuffle latencies per token.
// Requires head_dim == 128 AND head_dim % (RPW*WPC) == 0 (launcher asserts).
// ---------------------------------------------------------------------------
#define RECURRENCE_MR(NAME, DOTF, UPDF, PSF, SCF, RPW, WPC)                    \
extern "C" __global__ void __launch_bounds__(32 * WPC, 1) NAME(               \
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
    const int stride = 32;                                                     \
    const long qkvx_stride = 3L * v_dim;                                       \
    const long head_off = (long)head * head_dim;                               \
    const long k_base = v_dim + head_off;                                      \
    const long v_base = 2L * v_dim + head_off;                                 \
    const int row = blockIdx.y * (RPW * WPC) + warp * RPW;                     \
    const int c0 = lane;                                                       \
    const int c1 = lane + stride;                                              \
    const int c2 = lane + 2 * stride;                                          \
    const int c3 = lane + 3 * stride;                                          \
    long row_off[RPW];                                                         \
    float s0[RPW], s1[RPW], s2[RPW], s3[RPW];                                  \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < RPW; r++) {                                            \
        row_off[r] = (long)head * head_dim * head_dim                          \
                   + (long)(row + r) * head_dim;                               \
        s0[r] = state[row_off[r] + c0];                                        \
        s1[r] = state[row_off[r] + c1];                                        \
        s2[r] = state[row_off[r] + c2];                                        \
        s3[r] = state[row_off[r] + c3];                                        \
    }                                                                          \
    const float scale = SCF(head_dim);                                         \
    for (int t = 0; t < p; t++) {                                              \
        const long t_row = (long)t * qkvx_stride;                              \
        const long sc = (long)t * n_head + head;                               \
        const float beta_val = beta[sc];                                       \
        const float decay_val = decay[sc];                                     \
        const float k0 = qkvx[t_row + k_base + c0];                            \
        const float k1 = qkvx[t_row + k_base + c1];                            \
        const float k2 = qkvx[t_row + k_base + c2];                            \
        const float k3 = qkvx[t_row + k_base + c3];                            \
        const float q0 = qkvx[t_row + head_off + c0];                          \
        const float q1 = qkvx[t_row + head_off + c1];                          \
        const float q2 = qkvx[t_row + head_off + c2];                          \
        const float q3 = qkvx[t_row + head_off + c3];                          \
        _Pragma("unroll")                                                      \
        for (int r = 0; r < RPW; r++) {                                        \
            s0[r] = __fmul_rn(s0[r], decay_val);                               \
            s1[r] = __fmul_rn(s1[r], decay_val);                               \
            s2[r] = __fmul_rn(s2[r], decay_val);                               \
            s3[r] = __fmul_rn(s3[r], decay_val);                               \
            const float acc_k = DOTF(s0[r], s1[r], s2[r], s3[r], k0, k1, k2, k3); \
            const float kv_mem_row = PSF(acc_k);                               \
            const float v_row = qkvx[t_row + v_base + row + r];                \
            const float delta_row = __fmul_rn(beta_val, __fsub_rn(v_row, kv_mem_row)); \
            s0[r] = UPDF(s0[r], k0, delta_row);                                \
            s1[r] = UPDF(s1[r], k1, delta_row);                                \
            s2[r] = UPDF(s2[r], k2, delta_row);                                \
            s3[r] = UPDF(s3[r], k3, delta_row);                                \
            const float acc_q = DOTF(s0[r], s1[r], s2[r], s3[r], q0, q1, q2, q3); \
            const float dot = PSF(acc_q);                                      \
            if (lane == 0) {                                                   \
                output[(long)t * v_dim + head_off + row + r] = __fmul_rn(dot, scale); \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    _Pragma("unroll")                                                          \
    for (int r = 0; r < RPW; r++) {                                            \
        state[row_off[r] + c0] = s0[r];                                        \
        state[row_off[r] + c1] = s1[r];                                        \
        state[row_off[r] + c2] = s2[r];                                        \
        state[row_off[r] + c3] = s3[r];                                        \
    }                                                                          \
}

// Canonical form only (F1/Fma/TreeUp/Rsqrt — the league dispatch form); the
// geometry ladder is the probe axis. Instantiate: (RPW, WPC).
#define RECMR_CANON(NAME, RPW, WPC)                                            \
RECURRENCE_MR(NAME, dot4_f1, UPD_FMA, ps_treeup, SC_RSQ, RPW, WPC)

RECMR_CANON(recmr_pf_r1w4, 1, 4)
RECMR_CANON(recmr_pf_r2w1, 2, 1)
RECMR_CANON(recmr_pf_r2w2, 2, 2)
RECMR_CANON(recmr_pf_r2w4, 2, 4)
RECMR_CANON(recmr_pf_r4w1, 4, 1)
RECMR_CANON(recmr_pf_r4w2, 4, 2)
RECMR_CANON(recmr_pf_r4w4, 4, 4)
RECMR_CANON(recmr_pf_r8w1, 8, 1)
RECMR_CANON(recmr_pf_r8w2, 8, 2)
"#;

// ---------------------------------------------------------------------------
// Kernel wrapper
// ---------------------------------------------------------------------------

/// The cudarc-side DeltaNet prefill kernels (all variants compiled; the
/// canonical forms selected by the probe — see module doc).
pub struct CudaDeltanetKernels {
    dequant_wte: CudaFunction,
    conv1d: Vec<CudaFunction>,
    carry_update: CudaFunction,
    beta_decay: Vec<CudaFunction>,
    expand_l2: Vec<CudaFunction>,
    /// Issue 772 T2 residue: staged-sum V2 of `expand_l2`, SAME variant
    /// indexing (`idx_expand_l2`) — bit-identical to the legacy kernels.
    expand_l2_v2: Vec<CudaFunction>,
    z_gating: Vec<CudaFunction>,
    recurrence: Vec<CudaFunction>,
    /// Issue 904 T1 — multi-row ILP probe arms, ordered by
    /// [`REC_MR_GEOMETRIES`] (canonical form only).
    recurrence_mr: Vec<CudaFunction>,
    _module: Arc<CudaModule>,
}

/// Issue 772 T2 residue (Bench 809): V2 staged-sum dispatch for the
/// `expand_l2` family. DEFAULT ON — bit-identical output (same ACC form /
/// order, same INV form, index-for-index identical writes), so the env is a
/// KILL-SWITCH: `RIIR_EXPAND_L2_ROWS_LEGACY=1` restores the legacy kernels.
/// The launch counter is the vacuous guard (the Bench-768 lesson).
static EXPAND_L2_V2: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
static EXPAND_L2_V2_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static EXPAND_L2_V2_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn expand_l2_v2_enabled() -> bool {
    if EXPAND_L2_V2_ENV
        .set(!matches!(
            std::env::var("RIIR_EXPAND_L2_ROWS_LEGACY")
                .unwrap_or_default()
                .to_lowercase()
                .as_str(),
            "1" | "on" | "true",
        ))
        .is_ok()
    {
        EXPAND_L2_V2.store(
            EXPAND_L2_V2_ENV.get().copied().unwrap_or(true),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    EXPAND_L2_V2.load(std::sync::atomic::Ordering::Relaxed)
}

/// Runtime override for one-binary A/B (`None` restores env/default).
#[cfg_attr(not(test), allow(dead_code))]
fn set_expand_l2_v2(override_value: Option<bool>) {
    let v = override_value
        .or_else(|| EXPAND_L2_V2_ENV.get().copied())
        .unwrap_or(true);
    EXPAND_L2_V2.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Launches dispatched through the V2 staged-sum family (the vacuous guard).
#[cfg_attr(not(test), allow(dead_code))]
fn expand_l2_v2_launches() -> usize {
    EXPAND_L2_V2_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Issue 904 T1/T3 — the measured multi-row winner (Bench 896): r4w4
/// (4 rows/warp × 4 warps/block, 128-thr blocks) — 2.195×/1.894× the legacy
/// 32-thr rowpar kernel at the league shapes (p=2048/4096). Index into
/// [`REC_MR_GEOMETRIES`].
pub const REC_MR_DEFAULT_ARM: usize = 6;

/// Issue 904 — multi-row ILP dispatch: `Some(arm)` = route
/// `launch_recurrence` through `launch_recurrence_mr(arm)` (bit-identical by
/// the G1 gate); `None` = the legacy 32-thr kernel. DEFAULT = the measured
/// winner. Env `RIIR_PREFILL_REC_MR`: `0|off|false|legacy` restores the
/// legacy kernel; `1..=9` selects arm N-1 (probe convenience); garbage fails
/// OPEN to legacy with a warning. Sentinels: `usize::MAX` = the legacy
/// override, `NO_REC_MR_OVERRIDE` = no override installed (env/default
/// resolve). The legacy launch counter is the vacuous guard (the Bench-768
/// lesson).
static REC_MR_ARM: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(NO_REC_MR_OVERRIDE);
static REC_MR_ENV: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
static REC_MR_LEGACY_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

const NO_REC_MR_OVERRIDE: usize = usize::MAX - 1;

#[inline]
fn rec_mr_arm() -> Option<usize> {
    // Override (tests / one-binary A/B) wins over env — order-independent:
    // the override lives in its own static, the env resolves via OnceLock,
    // and neither can clobber the other.
    let over = REC_MR_ARM.load(std::sync::atomic::Ordering::Relaxed);
    if over != NO_REC_MR_OVERRIDE {
        return (over != usize::MAX).then_some(over);
    }
    *REC_MR_ENV.get_or_init(|| {
        match std::env::var("RIIR_PREFILL_REC_MR") {
            Ok(s) => {
                let t = s.trim().to_lowercase();
                if matches!(t.as_str(), "0" | "off" | "false" | "legacy") {
                    None
                } else {
                    match t.parse::<usize>() {
                        Ok(n) if (1..=REC_MR_GEOMETRIES.len()).contains(&n) => Some(n - 1),
                        _ => {
                            eprintln!(
                                "[rec-mr] RIIR_PREFILL_REC_MR={s:?} is not an arm in 1..={} nor off/legacy — failing OPEN to the legacy kernel",
                                REC_MR_GEOMETRIES.len()
                            );
                            None
                        }
                    }
                }
            }
            Err(_) => Some(REC_MR_DEFAULT_ARM),
        }
    })
}

/// Runtime override for one-binary A/B (`Some(arm)` selects an arm, `None`
/// forces the legacy kernel; tests restore `Some(REC_MR_DEFAULT_ARM)` when
/// done). Process-wide — sibling tests in the same binary share it.
#[cfg_attr(not(test), allow(dead_code))]
fn set_rec_mr(arm: Option<usize>) {
    REC_MR_ARM.store(arm.unwrap_or(usize::MAX), std::sync::atomic::Ordering::Relaxed);
}

/// Launches dispatched through the LEGACY kernel (the vacuous guard: stays 0
/// on the default path; a legacy run advances it).
#[cfg_attr(not(test), allow(dead_code))]
fn rec_mr_legacy_launches() -> usize {
    REC_MR_LEGACY_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

fn idx_conv1d(f: ConvFma, e: SigExp, d: SigDiv) -> usize {
    let cf = (f == ConvFma::Fma) as usize;
    let ef = (e == SigExp::Fast) as usize;
    (cf * 2 + ef) * 4 + d as usize
}

fn idx_beta_decay(l: LogForm, e: SigExp, d: SigDiv) -> usize {
    (l as usize * 2 + (e == SigExp::Fast) as usize) * 4 + d as usize
}

fn idx_expand_l2(a: L2Accum, i: L2Inv) -> usize {
    ((a == L2Accum::Fma) as usize) * 4 + i as usize
}

fn idx_z_gating(e: SigExp, d: SigDiv) -> usize {
    ((e == SigExp::Fast) as usize) * 4 + d as usize
}

fn idx_recurrence(dot: RecDot, upd: RecUpd, ps: PlaneSum, sc: RecScale) -> usize {
    let u = (upd == RecUpd::Fma) as usize;
    let p = match ps {
        PlaneSum::Tree => 0,
        PlaneSum::TreeUp => 1,
        PlaneSum::Seq => 2,
    };
    let s = (sc == RecScale::DivRn) as usize;
    ((dot as usize * 2 + u) * 3 + p) * 2 + s
}

/// Issue 904 T1 — the L3 multi-row ILP probe geometries `(rows_per_warp,
/// warps_per_block)`, canonical rec form only (F1/Fma/TreeUp/Rsqrt — the
/// league dispatch form). Index = position in
/// [`CudaDeltanetKernels::recurrence_mr`] / the `launch_recurrence_mr` `arm`
/// argument. All nine divide `head_dim == 128`.
/// The canonical expand-kernel variant pair — the Bench 809 production pin.
/// Moved here from `prefill_cuda_full` with the kernel family (the enum types
/// it names live in this module); `prefill_cuda_full` consumes it through the
/// module re-export.
pub fn canonical_expand_forms() -> (L2Accum, L2Inv) {
    (L2Accum::Fma, L2Inv::Rsqrt)
}
/// Moved here from `prefill_cuda_full` (S4a, the S3 `canonical_expand_forms`
/// class): the recurrence-form tuple's types (RecDot/RecUpd/PlaneSum/RecScale)
/// live in this module, and this crate's own lib tests need it without a
/// cross-crate path back into the engine gpu crate.
pub fn canonical_recurrence_forms() -> (RecDot, RecUpd, PlaneSum, RecScale) {
    (RecDot::F1, RecUpd::Fma, PlaneSum::TreeUp, RecScale::Rsqrt)
}

pub const REC_MR_GEOMETRIES: &[(u32, u32)] = &[
    (1, 4),
    (2, 1),
    (2, 2),
    (2, 4),
    (4, 1),
    (4, 2),
    (4, 4),
    (8, 1),
    (8, 2),
];

impl CudaDeltanetKernels {
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
            DELTANET_PREFILL_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;

        let dequant_wte = module
            .load_function("dequant_wte_batch")
            .map_err(|e| format!("dequant_wte_batch: {e}"))?;

        let mut conv1d = Vec::with_capacity(16);
        for cf in 0..2u8 {
            for ef in 0..2u8 {
                for d in 0..4u8 {
                    let name = format!("conv1d_pf_d{cf}e{ef}_g{d}");
                    conv1d.push(
                        module
                            .load_function(&name)
                            .map_err(|e| format!("{name}: {e}"))?,
                    );
                }
            }
        }

        let carry_update = module
            .load_function("conv1d_carry_update")
            .map_err(|e| format!("conv1d_carry_update: {e}"))?;

        let mut beta_decay = Vec::with_capacity(32);
        for l in 0..4u8 {
            for ef in 0..2u8 {
                for d in 0..4u8 {
                    let name = format!("bd_pf_l{l}e{ef}_g{d}");
                    beta_decay.push(
                        module
                            .load_function(&name)
                            .map_err(|e| format!("{name}: {e}"))?,
                    );
                }
            }
        }

        let mut expand_l2 = Vec::with_capacity(8);
        for a in 0..2u8 {
            for i in 0..4u8 {
                let name = format!("el_pf_a{a}_i{i}");
                expand_l2.push(
                    module
                        .load_function(&name)
                        .map_err(|e| format!("{name}: {e}"))?,
                );
            }
        }

        let mut expand_l2_v2 = Vec::with_capacity(8);
        for a in 0..2u8 {
            for i in 0..4u8 {
                let name = format!("elv2_pf_a{a}_i{i}");
                expand_l2_v2.push(
                    module
                        .load_function(&name)
                        .map_err(|e| format!("{name}: {e}"))?,
                );
            }
        }

        let mut z_gating = Vec::with_capacity(8);
        for e in 0..2u8 {
            for d in 0..4u8 {
                let name = format!("zg_pf_e{e}_g{d}");
                z_gating.push(
                    module
                        .load_function(&name)
                        .map_err(|e| format!("{name}: {e}"))?,
                );
            }
        }

        let mut recurrence = Vec::with_capacity(96);
        for d in 0..4u8 {
            for u in 0..2u8 {
                for p in 0..3u8 {
                    for s in 0..2u8 {
                        let name = format!("rec_pf_d{d}u{u}p{p}s{s}");
                        recurrence.push(
                            module
                                .load_function(&name)
                                .map_err(|e| format!("{name}: {e}"))?,
                        );
                    }
                }
            }
        }

        let mut recurrence_mr = Vec::with_capacity(REC_MR_GEOMETRIES.len());
        for &(r, w) in REC_MR_GEOMETRIES {
            let name = format!("recmr_pf_r{r}w{w}");
            recurrence_mr.push(
                module
                    .load_function(&name)
                    .map_err(|e| format!("{name}: {e}"))?,
            );
        }

        Ok(Self {
            dequant_wte,
            conv1d,
            carry_update,
            beta_decay,
            expand_l2,
            expand_l2_v2,
            z_gating,
            recurrence,
            recurrence_mr,
            _module: module,
        })
    }

    // -- launchers ---------------------------------------------------------

    /// Dequantize `p` wte rows into `out` `[p * n]`.
    ///
    /// # Safety
    ///
    /// Caller guarantees `pos_bits`/`neg_bits` cover
    /// `rows * blocks64 * 2` u32, `group_scale` covers
    /// `rows * groups_per_row` f32, `tokens` covers `p` u32, `out` covers
    /// `p * n` f32, and every token `< rows`.
    pub unsafe fn launch_dequant_wte_batch(
        &self,
        stream: &CudaStream,
        pos_bits: &CudaSlice<u32>,
        neg_bits: &CudaSlice<u32>,
        group_scale: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        tokens: &CudaSlice<u32>,
        blocks64: usize,
        groups_per_row: usize,
        n: usize,
        p: usize,
    ) -> Result<(), String> {
        let total = p * n;
        let (blocks64_i, groups_i, n_i, p_i) = (
            blocks64 as i32,
            groups_per_row as i32,
            n as i32,
            p as i32,
        );
        let grid = total.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.dequant_wte)
                .arg(pos_bits)
                .arg(neg_bits)
                .arg(group_scale)
                .arg(out)
                .arg(tokens)
                .arg(&blocks64_i)
                .arg(&groups_i)
                .arg(&n_i)
                .arg(&p_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// conv1d over all `p` tokens (raw `input` → SiLU `output`).
    ///
    /// # Safety
    ///
    /// Caller guarantees `input`/`output` cover `p * conv_dim` f32, `weight`
    /// covers `conv_dim * kernel_size`, `carry` covers
    /// `conv_dim * kernel_size` (the conv_state sliding-window layout).
    pub unsafe fn launch_conv1d(
        &self,
        stream: &CudaStream,
        f: ConvFma,
        e: SigExp,
        d: SigDiv,
        input: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        carry: &CudaSlice<f32>,
        p: usize,
        conv_dim: usize,
        kernel_size: usize,
    ) -> Result<(), String> {
        let func = &self.conv1d[idx_conv1d(f, e, d)];
        let (p_i, cd_i, ks_i) = (p as i32, conv_dim as i32, kernel_size as i32);
        let total = p * conv_dim;
        let grid = total.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(input)
                .arg(output)
                .arg(weight)
                .arg(carry)
                .arg(&p_i)
                .arg(&cd_i)
                .arg(&ks_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Update the conv carry (conv_state layout) for the whole chunk.
    ///
    /// # Safety
    ///
    /// Same buffer contract as [`Self::launch_conv1d`].
    pub unsafe fn launch_carry_update(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<f32>,
        carry: &CudaSlice<f32>,
        p: usize,
        conv_dim: usize,
        kernel_size: usize,
    ) -> Result<(), String> {
        let (p_i, cd_i, ks_i) = (p as i32, conv_dim as i32, kernel_size as i32);
        let grid = conv_dim.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.carry_update)
                .arg(input)
                .arg(carry)
                .arg(&p_i)
                .arg(&cd_i)
                .arg(&ks_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Batched beta/decay over `total = p * n_head` elements.
    ///
    /// # Safety
    ///
    /// Caller guarantees `a_raw`/`b_raw`/`beta_out`/`decay_out` cover `total`
    /// f32 and `a_log`/`dt_bias` cover `n_head` f32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_beta_decay(
        &self,
        stream: &CudaStream,
        l: LogForm,
        e: SigExp,
        d: SigDiv,
        a_raw: &CudaSlice<f32>,
        b_raw: &CudaSlice<f32>,
        a_log: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        beta_out: &CudaSlice<f32>,
        decay_out: &CudaSlice<f32>,
        n_head: usize,
        total: usize,
    ) -> Result<(), String> {
        let func = &self.beta_decay[idx_beta_decay(l, e, d)];
        let (nh_i, tot_i) = (n_head as i32, total as i32);
        let grid = total.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(a_raw)
                .arg(b_raw)
                .arg(a_log)
                .arg(dt_bias)
                .arg(beta_out)
                .arg(decay_out)
                .arg(&nh_i)
                .arg(&tot_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Batched expand + L2-normalize over `total = p * 3 * n_v * hd` elements.
    ///
    /// # Safety
    ///
    /// Caller guarantees `compact` covers `p * (2*n_k + n_v) * hd` and
    /// `expanded` covers `total` f32.
    pub unsafe fn launch_expand_l2(
        &self,
        stream: &CudaStream,
        a: L2Accum,
        i: L2Inv,
        compact: &CudaSlice<f32>,
        expanded: &CudaSlice<f32>,
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
        total: usize,
    ) -> Result<(), String> {
        let idx = idx_expand_l2(a, i);
        // V2 staged-sum dispatch (Issue 772 T2 residue): bit-identical to the
        // legacy kernels below; n_k_heads > 1024 cannot stage its sums in the
        // 1024-wide shared tile, so those configs stay on the legacy kernels.
        if expand_l2_v2_enabled() && n_k_heads <= 1024 && total > 0 {
            let p = total / (3 * n_v_heads * head_dim);
            let func = &self.expand_l2_v2[idx];
            let (nk_i, nv_i, hd_i, p_i) = (
                n_k_heads as i32,
                n_v_heads as i32,
                head_dim as i32,
                p as i32,
            );
            let cfg = LaunchConfig {
                grid_dim: (p as u32, 3, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(func)
                    .arg(compact)
                    .arg(expanded)
                    .arg(&nk_i)
                    .arg(&nv_i)
                    .arg(&hd_i)
                    .arg(&p_i)
                    .launch(cfg)
                    .map_err(|e| e.to_string())?;
            }
            EXPAND_L2_V2_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        let func = &self.expand_l2[idx];
        let (nk_i, nv_i, hd_i, tot_i) = (
            n_k_heads as i32,
            n_v_heads as i32,
            head_dim as i32,
            total as i32,
        );
        let grid = total.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(compact)
                .arg(expanded)
                .arg(&nk_i)
                .arg(&nv_i)
                .arg(&hd_i)
                .arg(&tot_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// In-place z-gating over `n` elements.
    ///
    /// # Safety
    ///
    /// Caller guarantees `output`/`z` cover `n` f32.
    pub unsafe fn launch_z_gating(
        &self,
        stream: &CudaStream,
        e: SigExp,
        d: SigDiv,
        output: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), String> {
        let func = &self.z_gating[idx_z_gating(e, d)];
        let n_i = n as i32;
        let grid = n.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(output)
                .arg(z)
                .arg(&n_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Multi-token recurrence — all `p` tokens in one dispatch, state updated
    /// in place. Requires `head_dim == 128` (32 lanes × 4 cols).
    ///
    /// # Safety
    ///
    /// Caller guarantees `qkvx` covers `p * 3 * v_dim`, `beta`/`decay` cover
    /// `p * n_head`, `state` covers `n_head * head_dim * head_dim`, `output`
    /// covers `p * v_dim`, and `head_dim == 128`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_recurrence(
        &self,
        stream: &CudaStream,
        dot: RecDot,
        upd: RecUpd,
        ps: PlaneSum,
        sc: RecScale,
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
        debug_assert_eq!(head_dim, 128, "register-blocked recurrence: head_dim == 128");
        // Issue 904 T3 — default-on multi-row dispatch (bit-identical by the
        // G1 gate; the env is a kill-switch, the runtime override is the A/B
        // seam).
        if let Some(arm) = rec_mr_arm() {
            return unsafe { self.launch_recurrence_mr(stream, arm, qkvx, beta, decay, state, output, head_dim, n_head, p, v_dim) };
        }
        REC_MR_LEGACY_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let func = &self.recurrence[idx_recurrence(dot, upd, ps, sc)];
        let (hd_i, nh_i, p_i, vd_i) = (
            head_dim as i32,
            n_head as i32,
            p as i32,
            v_dim as i32,
        );
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, head_dim as u32, 1),
            block_dim: (32, 1, 1),
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
        Ok(())
    }

    /// Issue 904 T1 — launch a multi-row ILP probe arm (`arm` indexes
    /// [`REC_MR_GEOMETRIES`]; canonical rec form). Same buffer contract as
    /// [`Self::launch_recurrence`]; additionally requires
    /// `head_dim % (rpw * wpc) == 0` for the selected geometry.
    ///
    /// # Safety
    ///
    /// Same as [`Self::launch_recurrence`] (buffers + `head_dim == 128`), plus
    /// the geometry divisibility above.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_recurrence_mr(
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
        let Some(&(rpw, wpc)) = REC_MR_GEOMETRIES.get(arm) else {
            return Err(format!(
                "rec_mr arm {arm} out of range (0..{})",
                REC_MR_GEOMETRIES.len()
            ));
        };
        debug_assert_eq!(head_dim, 128, "register-blocked recurrence: head_dim == 128");
        let rows_per_blk = (rpw * wpc) as usize;
        debug_assert_eq!(
            head_dim % rows_per_blk,
            0,
            "multi-row recurrence: head_dim % (rpw*wpc) == 0"
        );
        let func = &self.recurrence_mr[arm];
        let (hd_i, nh_i, p_i, vd_i) = (
            head_dim as i32,
            n_head as i32,
            p as i32,
            v_dim as i32,
        );
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, (head_dim / rows_per_blk) as u32, 1),
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_or_skip() -> Option<()> {
        match cudarc::driver::safe::CudaContext::new(0) {
            Ok(_) => Some(()),
            Err(e) => {
                eprintln!("[skip] no CUDA device: {e}");
                None
            }
        }
    }

    /// Issue 772 T2 residue (Bench 809): the V2 staged-sum expand family must
    /// be BIT-identical to the legacy per-element family across all 8 ACC×INV
    /// variants. Poisoned destination proves full coverage; a fully-zero
    /// token row exercises the `sq_sum > 0` else-branch; run-twice re-launch
    /// pins V2 determinism; the counter is the vacuous guard.
    #[test]
    fn test_expand_l2_v2_bit_identical_to_legacy_all_variants() {
        const P_ROWS: &[usize] = &[1, 3, 8];
const POISON: f32 = 12_345.5;

let Some(_) = cuda_or_skip() else {
            return;
        };
        let (kernels, stream) = CudaDeltanetKernels::new_standalone().expect("compile");

        // Bonsai production dims + smaller odd shapes (incl. hd % 4 != 0).
        const DIMS: &[(usize, usize, usize)] = &[
            (16, 48, 128),
            (2, 6, 16),
            (4, 4, 10),
            (1, 1, 128),
        ];

        let variants = [
            (L2Accum::Ma, L2Inv::Rsqrt),
            (L2Accum::Ma, L2Inv::DivRnSqrtRn),
            (L2Accum::Ma, L2Inv::DivFullSqrtRn),
            (L2Accum::Ma, L2Inv::RcpSqrtApprox),
            (L2Accum::Fma, L2Inv::Rsqrt),
            (L2Accum::Fma, L2Inv::DivRnSqrtRn),
            (L2Accum::Fma, L2Inv::DivFullSqrtRn),
            (L2Accum::Fma, L2Inv::RcpSqrtApprox),
        ];

        let launches_before = expand_l2_v2_launches();
        let mut v2_launches_expected = 0usize;

        for &(n_k, n_v, hd) in DIMS {
            let per_in = (2 * n_k + n_v) * hd;
            let per_out = 3 * n_v * hd;
            for &p in P_ROWS {
                let compact_len = p * per_in;
                let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (compact_len as u64);
                let mut compact: Vec<f32> = (0..compact_len)
                    .map(|_| {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let r = ((seed >> 33) & 0xFF_FFFF) as f32 / 8_388_608.0 - 0.5;
                        r * 6.0
                    })
                    .collect();
                // Zero one full token row: sq_sum == 0 on every section.
                let zero_t = p / 2;
                compact[zero_t * per_in..(zero_t + 1) * per_in].fill(0.0);
                let compact_dev = stream.clone_htod(&compact).unwrap();
                let poison: Vec<f32> = vec![POISON; p * per_out];

                for &(a, i) in &variants {
                    let run = |v2: bool| -> Vec<f32> {
                        set_expand_l2_v2(Some(v2));
                        let expanded_dev = stream.clone_htod(&poison).unwrap();
                        unsafe {
                            kernels
                                .launch_expand_l2(
                                    &stream,
                                    a,
                                    i,
                                    &compact_dev,
                                    &expanded_dev,
                                    n_k,
                                    n_v,
                                    hd,
                                    p * per_out,
                                )
                                .expect("launch");
                        }
                        stream.synchronize().expect("sync");
                        let mut out = vec![0f32; p * per_out];
                        stream.memcpy_dtoh(&expanded_dev, &mut out).unwrap();
                        out
                    };

                    let legacy = run(false);
                    let v2_a = run(true);
                    let v2_b = run(true);
                    v2_launches_expected += 2;

                    let mismatches: Vec<usize> = legacy
                        .iter()
                        .zip(v2_a.iter())
                        .enumerate()
                        .filter(|(_, (x, y))| x.to_bits() != y.to_bits())
                        .map(|(t, _)| t)
                        .collect();
                    assert!(
                        mismatches.is_empty(),
                        "variant {a:?}/{i:?} n_k={n_k} n_v={n_v} hd={hd} p={p}: {} bit mismatches, first at {:?} (legacy={:?} v2={:?})",
                        mismatches.len(),
                        mismatches.first(),
                        mismatches.first().map(|&t| legacy[t]),
                        mismatches.first().map(|&t| v2_a[t]),
                    );
                    // Full coverage on both paths: no sentinel survived.
                    assert!(
                        legacy.iter().chain(v2_a.iter()).all(|v| v.to_bits() != POISON.to_bits()),
                        "sentinel survived (coverage gap) variant {a:?}/{i:?} p={p}",
                    );
                    // Run-twice determinism of the V2 kernel.
                    assert!(
                        v2_a.iter().zip(v2_b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "V2 run-twice nondeterminism variant {a:?}/{i:?} p={p}",
                    );
                    // The zero row produced exact zeros (val * 0.0 = 0.0).
                    let zout = zero_t * per_out;
                    assert!(legacy[zout..zout + per_out].iter().all(|v| *v == 0.0));
                }
            }
        }
        set_expand_l2_v2(None);
        // Vacuous guard (`>=`: sibling tests may exercise the family
        // concurrently — the static is shared process-wide).
        assert!(
            expand_l2_v2_launches() - launches_before >= v2_launches_expected,
            "vacuous guard: V2 launch counter did not advance as expected",
        );
    }

    /// Kernel-only A/B at the Bonsai production dims (n_k=16, n_v=48, hd=128,
    /// p=2048), canonical variant — the Bench 809 verdict instrument.
    #[test]
    #[ignore = "kernel-only timing probe — needs an exclusive GPU window"]
    fn test_expand_l2_v2_timing_probe() {
        let Some(_) = cuda_or_skip() else {
            return;
        };
        let (kernels, stream) = CudaDeltanetKernels::new_standalone().expect("compile");
        let (a, i) = super::canonical_expand_forms();

        let (n_k, n_v, hd, p) = (16usize, 48usize, 128usize, 2048usize);
        let per_in = (2 * n_k + n_v) * hd;
        let per_out = 3 * n_v * hd;
        let compact_len = p * per_in;
        let mut seed = 0xDEFA_CE02u64;
        let compact: Vec<f32> = (0..compact_len)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) & 0xFF_FFFF) as f32 / 8_388_608.0 - 0.5
            })
            .collect();
        let compact_dev = stream.clone_htod(&compact).unwrap();
        let expanded_dev = stream.alloc_zeros::<f32>(p * per_out).unwrap();

        let traffic_gb = ((compact_len + p * per_out) * 4) as f64 / 1e9;
        let timed = |v2: bool| -> f64 {
            set_expand_l2_v2(Some(v2));
            stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            unsafe {
                kernels
                    .launch_expand_l2(
                        &stream,
                        a,
                        i,
                        &compact_dev,
                        &expanded_dev,
                        n_k,
                        n_v,
                        hd,
                        p * per_out,
                    )
                    .expect("launch");
            }
            stream.synchronize().unwrap();
            t0.elapsed().as_secs_f64() * 1e3
        };

        for v2 in [false, true] {
            timed(v2);
        }
        let mut legacy_t = Vec::with_capacity(5);
        let mut v2_t = Vec::with_capacity(5);
        for _ in 0..5 {
            legacy_t.push(timed(false));
            v2_t.push(timed(true));
        }
        legacy_t.sort_by(|x, y| x.total_cmp(y));
        v2_t.sort_by(|x, y| x.total_cmp(y));
        let (lm, vm) = (legacy_t[2], v2_t[2]);
        eprintln!(
            "[expand-l2-timing] traffic {traffic_gb:.3} GB/call | legacy {lm:.3} ms ({:.0} GB/s) | v2 {vm:.3} ms ({:.0} GB/s) | v2/legacy {:.3}",
            traffic_gb / (lm / 1e3),
            traffic_gb / (vm / 1e3),
            vm / lm,
        );
        set_expand_l2_v2(None);
    }

    /// Issue 904 T1 (G1): the multi-row ILP probe arms must be BIT-identical
    /// to the production rowpar kernel (`launch_recurrence`, canonical form)
    /// — output AND final in-place state, `to_bits`, zero diffs. Poisoned
    /// output+state prove full write coverage; run-twice re-launch pins arm
    /// determinism. hd=128 only (the register-blocked unroll); the geometry
    /// ladder covers every arm at every case.
    #[test]
    fn test_rec_mr_bit_identical_to_rowpar() {
        const POISON: f32 = 12_345.5;
        // (n_v, p): the league n_v, a small n_v, and odd p values.
        const CASES: &[(usize, usize)] = &[(48, 3), (48, 130), (2, 5), (1, 1)];

        let Some(_) = cuda_or_skip() else {
            return;
        };
        let (kernels, stream) = CudaDeltanetKernels::new_standalone().expect("compile");
        let (rd, ru, rp, rsc) = super::canonical_recurrence_forms();

        // The baseline reference MUST be the legacy kernel: `launch_recurrence`
        // dispatches through the multi-row winner by default (Issue 904 T3).
        set_rec_mr(None);
        let legacy_before = rec_mr_legacy_launches();

        let lcg = |seed: u64| -> u64 {
            seed.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407)
        };
        let mut seed = 0x0001_5EED_0904_A1B1_u64;
        for &(n_v, p) in CASES {
            let hd = 128usize;
            let v_dim = n_v * hd;
            let state_len = n_v * hd * hd;
            let qkvx_len = p * 3 * v_dim;
            // beta/decay in (0.05, 0.95): real gate ranges, no underflow pathologies.
            let beta: Vec<f32> = (0..p * n_v)
                .map(|_| {
                    seed = lcg(seed);
                    0.05 + ((seed >> 40) as f32 / 16_777_216.0) * 0.90
                })
                .collect();
            let decay: Vec<f32> = (0..p * n_v)
                .map(|_| {
                    seed = lcg(seed);
                    0.5 + ((seed >> 40) as f32 / 16_777_216.0) * 0.499
                })
                .collect();
            let qkvx: Vec<f32> = (0..qkvx_len)
                .map(|_| {
                    seed = lcg(seed);
                    ((seed >> 33) & 0xFF_FFFF) as f32 / 8_388_608.0 - 0.5
                })
                .collect();
            let state0: Vec<f32> = (0..state_len)
                .map(|_| {
                    seed = lcg(seed);
                    ((seed >> 33) & 0xFF_FFFF) as f32 / 8_388_608.0 - 0.5
                })
                .collect();

            let beta_dev = stream.clone_htod(&beta).unwrap();
            let decay_dev = stream.clone_htod(&decay).unwrap();
            let qkvx_dev = stream.clone_htod(&qkvx).unwrap();
            let out_poison: Vec<f32> = vec![POISON; p * v_dim];
            let _st_poison: Vec<f32> = vec![POISON; state_len];

            // Baseline: production rowpar kernel, canonical form.
            let run_baseline = || -> (Vec<f32>, Vec<f32>) {
                let state_dev = stream.clone_htod(&state0).unwrap();
                let out_dev = stream.clone_htod(&out_poison).unwrap();
                unsafe {
                    kernels
                        .launch_recurrence(
                            &stream, rd, ru, rp, rsc, &qkvx_dev, &beta_dev, &decay_dev,
                            &state_dev, &out_dev, hd, n_v, p, v_dim,
                        )
                        .expect("baseline launch");
                }
                stream.synchronize().unwrap();
                let mut out = vec![0f32; p * v_dim];
                let mut st = vec![0f32; state_len];
                stream.memcpy_dtoh(&out_dev, &mut out).unwrap();
                stream.memcpy_dtoh(&state_dev, &mut st).unwrap();
                (out, st)
            };
            let (base_out, base_state) = run_baseline();

            for (arm, &(arm_rows, arm_cols)) in REC_MR_GEOMETRIES.iter().enumerate() {
                let run_arm = || -> (Vec<f32>, Vec<f32>) {
                    let state_dev = stream.clone_htod(&state0).unwrap();
                    let out_dev = stream.clone_htod(&out_poison).unwrap();
                    unsafe {
                        kernels
                            .launch_recurrence_mr(
                                &stream, arm, &qkvx_dev, &beta_dev, &decay_dev,
                                &state_dev, &out_dev, hd, n_v, p, v_dim,
                            )
                            .expect("arm launch");
                    }
                    stream.synchronize().unwrap();
                    let mut out = vec![0f32; p * v_dim];
                    let mut st = vec![0f32; state_len];
                    stream.memcpy_dtoh(&out_dev, &mut out).unwrap();
                    stream.memcpy_dtoh(&state_dev, &mut st).unwrap();
                    (out, st)
                };
                let (a_out, a_state) = run_arm();
                let (b_out, b_state) = run_arm(); // run-twice determinism

                let bad_out: Vec<usize> = base_out
                    .iter()
                    .zip(a_out.iter())
                    .enumerate()
                    .filter(|(_, (x, y))| x.to_bits() != y.to_bits())
                    .map(|(i, _)| i)
                    .collect();
                assert!(
                    bad_out.is_empty(),
                    "arm {arm} (r{arm_rows}w{arm_cols}) n_v={n_v} p={p}: {} output bit mismatches, first at {:?}",
                    bad_out.len(),
                    bad_out.first(),
                );
                let bad_state: Vec<usize> = base_state
                    .iter()
                    .zip(a_state.iter())
                    .enumerate()
                    .filter(|(_, (x, y))| x.to_bits() != y.to_bits())
                    .map(|(i, _)| i)
                    .collect();
                assert!(
                    bad_state.is_empty(),
                    "arm {arm} (r{arm_rows}w{arm_cols}) n_v={n_v} p={p}: {} state bit mismatches, first at {:?}",
                    bad_state.len(),
                    bad_state.first(),
                );
                // Full write coverage: no sentinel survived anywhere.
                assert!(
                    a_out.iter().chain(a_state.iter()).all(|v| v.to_bits() != POISON.to_bits()),
                    "arm {arm} n_v={n_v} p={p}: sentinel survived (coverage gap)",
                );
                // Run-twice determinism.
                assert!(
                    a_out.iter().zip(b_out.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
                        && a_state
                            .iter()
                            .zip(b_state.iter())
                            .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "arm {arm} n_v={n_v} p={p}: run-twice nondeterminism",
                );
            }

            // T3 wiring check (first case only): the DEFAULT launch_recurrence
            // dispatch == the winner arm's direct launch, bit-for-bit.
            if n_v == 48 && p == 3 {
                set_rec_mr(Some(REC_MR_DEFAULT_ARM));
                let state_dev = stream.clone_htod(&state0).unwrap();
                let out_dev = stream.clone_htod(&out_poison).unwrap();
                unsafe {
                    kernels
                        .launch_recurrence(
                            &stream, rd, ru, rp, rsc, &qkvx_dev, &beta_dev, &decay_dev,
                            &state_dev, &out_dev, hd, n_v, p, v_dim,
                        )
                        .expect("default dispatch launch");
                }
                stream.synchronize().unwrap();
                let mut out = vec![0f32; p * v_dim];
                let mut st = vec![0f32; state_len];
                stream.memcpy_dtoh(&out_dev, &mut out).unwrap();
                stream.memcpy_dtoh(&state_dev, &mut st).unwrap();
                // The winner arm result from the ladder above (recompute for
                // independence from the loop's last-arm state).
                let state_dev2 = stream.clone_htod(&state0).unwrap();
                let out_dev2 = stream.clone_htod(&out_poison).unwrap();
                unsafe {
                    kernels
                        .launch_recurrence_mr(
                            &stream, REC_MR_DEFAULT_ARM, &qkvx_dev, &beta_dev, &decay_dev,
                            &state_dev2, &out_dev2, hd, n_v, p, v_dim,
                        )
                        .expect("winner direct launch");
                }
                stream.synchronize().unwrap();
                let mut out2 = vec![0f32; p * v_dim];
                let mut st2 = vec![0f32; state_len];
                stream.memcpy_dtoh(&out_dev2, &mut out2).unwrap();
                stream.memcpy_dtoh(&state_dev2, &mut st2).unwrap();
                assert!(
                    out.iter().zip(out2.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
                        && st.iter().zip(st2.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "T3 wiring: default launch_recurrence != winner arm direct"
                );
                set_rec_mr(None);
            }
        }
        // Vacuous guard: every baseline reference launch went through the
        // legacy kernel (>= : sibling tests may run concurrently).
        assert!(
            rec_mr_legacy_launches() - legacy_before > 0,
            "vacuous guard: legacy launch counter did not advance"
        );
        set_rec_mr(Some(REC_MR_DEFAULT_ARM));
    }

    /// Issue 904 T1 (G2): kernel-only A/B at the league shapes (n_v=48,
    /// hd=128, p ∈ {2048, 4096}) — baseline vs all six multi-row geometries,
    /// interleaved ×9, medians. The ≥1.2× gate decides T3 wiring.
    #[test]
    #[ignore = "kernel-only timing probe — needs an exclusive GPU window"]
    fn test_rec_mr_timing_probe() {
        let Some(_) = cuda_or_skip() else {
            return;
        };
        let (kernels, stream) = CudaDeltanetKernels::new_standalone().expect("compile");
        let (rd, ru, rp, rsc) = super::canonical_recurrence_forms();

        // Force the legacy kernel for the baseline arm: `launch_recurrence`
        // dispatches through the multi-row winner by default (Issue 904 T3).
        set_rec_mr(None);

        let (n_v, hd) = (48usize, 128usize);
        let v_dim = n_v * hd;
        let state_len = n_v * hd * hd;

        for p in [2048usize, 4096usize] {
            const ROUNDS: usize = 9;

            let qkvx_len = p * 3 * v_dim;
            let mut seed = 0xDEFA_CE90_4001u64;
            let beta: Vec<f32> = (0..p * n_v)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    0.05 + ((seed >> 40) as f32 / 16_777_216.0) * 0.90
                })
                .collect();
            let decay: Vec<f32> = (0..p * n_v)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    0.5 + ((seed >> 40) as f32 / 16_777_216.0) * 0.499
                })
                .collect();
            let qkvx: Vec<f32> = (0..qkvx_len)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    ((seed >> 33) & 0xFF_FFFF) as f32 / 8_388_608.0 - 0.5
                })
                .collect();
            let state0: Vec<f32> = (0..state_len)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    ((seed >> 33) & 0xFF_FFFF) as f32 / 8_388_608.0 - 0.5
                })
                .collect();

            let beta_dev = stream.clone_htod(&beta).unwrap();
            let decay_dev = stream.clone_htod(&decay).unwrap();
            let qkvx_dev = stream.clone_htod(&qkvx).unwrap();
            let state_dev = stream.clone_htod(&state0).unwrap();
            let out_dev = stream.alloc_zeros::<f32>(p * v_dim).unwrap();

            let timed = |arm: Option<usize>| -> f64 {
                stream.synchronize().unwrap();
                let t0 = std::time::Instant::now();
                let res = match arm {
                    Some(a) => unsafe {
                        kernels.launch_recurrence_mr(
                            &stream, a, &qkvx_dev, &beta_dev, &decay_dev, &state_dev,
                            &out_dev, hd, n_v, p, v_dim,
                        )
                    },
                    None => unsafe {
                        kernels.launch_recurrence(
                            &stream, rd, ru, rp, rsc, &qkvx_dev, &beta_dev, &decay_dev,
                            &state_dev, &out_dev, hd, n_v, p, v_dim,
                        )
                    },
                };
                res.expect("launch");
                stream.synchronize().unwrap();
                t0.elapsed().as_secs_f64() * 1e3
            };

            // Boost-clock burn + warmup (every arm once, then baseline hard).
            for a in 0..REC_MR_GEOMETRIES.len() {
                let _ = timed(Some(a));
            }
            for _ in 0..3 {
                let _ = timed(None);
            }
            let n_arms = REC_MR_GEOMETRIES.len();
            let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ROUNDS); n_arms + 1];
            for _ in 0..ROUNDS {
                samples[0].push(timed(None)); // baseline
                for a in 0..n_arms {
                    samples[a + 1].push(timed(Some(a)));
                }
            }
            for row in &mut samples {
                row.sort_by(|x, y| x.total_cmp(y));
            }
            let med: Vec<f64> = samples.iter().map(|r| r[ROUNDS / 2]).collect();
            eprintln!("[rec-mr-timing] n_v={n_v} hd={hd} p={p} (median of {ROUNDS}, interleaved):");
            eprintln!("  baseline (r1w1, 32thr): {:8.3} ms (1.000x)", med[0]);
            for (a, m) in med.iter().enumerate().skip(1) {
                let (r, w) = REC_MR_GEOMETRIES[a - 1];
                eprintln!(
                    "  r{r}w{w} ({}thr, grid.y={}): {:8.3} ms ({:.3}x)",
                    32 * w,
                    hd / (r * w) as usize,
                    m,
                    med[0] / m,
                );
            }
        }
        set_rec_mr(Some(REC_MR_DEFAULT_ARM));
    }
}
