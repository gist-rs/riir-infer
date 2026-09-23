//! Issue 734 Arm 8 — the cudarc-side **whole-prefill** attention kernels.
//!
//! CUDA twins of the CubeCL batched prefill attention chain (Issue 653
//! kernels): RoPE partial batched (per-token absolute positions), the QG/KV
//! splits, the KV-cache fill, and the causal gated flash-attention prefill
//! kernel (online softmax, 256-thread cube per (head, q_pos), the
//! Issue-715-barrier smem max tree, serial Phase-4 accumulation — NO softcap).
//!
//! ## Variant families
//!
//! - **RoPE** — `inv_freq = exp(ln(theta_base) · (0 − exponent))` then
//!   `cos/sin` per element: the `Log` axis (4), the `Exp` axis (2), the
//!   `Sin/Cos` axis (accurate `sinf/cosf` vs fast `__sinf/__cosf`), and the
//!   rotation contraction (`q0·c − q1·s` as mul/sub vs fma).
//! - **Attention** — the serial QK dot + weighted-value accumulation
//!   contraction (MA vs FMA), the online-softmax `Exp` form, and the final
//!   `1/running_sum` division (the `OpFDiv(1, x)` family). The output gate's
//!   sigmoid uses the PROVEN forms (`__expf` + `div.full`, Bench 721).
//! - The splits and the KV fill are pure copies — no variants.

use std::sync::Arc;

use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig,
};
use cudarc::driver::PushKernelArg;

use crate::prefill_cuda_deltanet::{LogForm, SigExp};

// ---------------------------------------------------------------------------
// Variant enums
// ---------------------------------------------------------------------------

/// Sin/Cos lowering — the UNKNOWN axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinCosForm {
    /// Accurate `sinf` / `cosf`.
    Acc,
    /// Fast `__sinf` / `__cosf` (`sin.approx.f32`).
    Fast,
}

/// RoPE rotation form: `q0*cos − q1*sin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeRot {
    /// Separate `__fmul_rn` + `__fsub_rn`/`__fadd_rn`.
    Ma,
    /// `fma(q0, c, −(q1*s))` / `fma(q0, s, q1*c)` — the contraction.
    Fma,
}

/// Attention serial-dot accumulation form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnDot {
    Ma,
    Fma,
}

/// Attention weighted-value accumulation form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnAcc {
    Ma,
    Fma,
}

/// Phase-3 weight form: `exp_tile * exp(my - tile_max)` vs the exp-product
/// FUSION `exp(my - new_max)` (algebraically equal, different rounding —
//  single-tile they coincide because exp_tile = exp(0) = 1.0 exactly).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnPh3 {
    Mul,
    Fused,
}

/// Online-softmax rescale form: separate muls then adds vs the contracted
/// `fma(running, exp_prev, tile_sum)` (the driver may fold the thread-local
/// mul+add across the smem barrier — the Bench-719 outer-fold class). The
/// mixed forms contract one accumulator and not the other (the `if
/// valid_dim` guard can inhibit the out-path contraction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnRes {
    /// Both separate: `r = r * ep; r = r + acc`.
    Mul,
    /// Both contracted: `r = fma(r, ep, acc)`.
    Fma,
    /// sum contracted, out separate.
    FmaSum,
    /// out contracted, sum separate.
    FmaOut,
}

/// Final `1/running_sum` lowering (`OpFDiv(1, x)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnInv {
    Rn,
    Full,
    Approx,
    Rcp,
}

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

pub(crate) const ATTENTION_PREFILL_CUDA_SRC: &str = r#"
// ---------------------------------------------------------------------------
// helpers (shared with the deltanet module's forms — duplicated here so each
// module compiles standalone)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float at_div_rn(float a, float b) { return a / b; }
__device__ __forceinline__ float at_div_full(float a, float b)
{
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float at_div_approx(float a, float b)
{
    float r;
    asm("div.approx.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float at_rcp_approx(float a)
{
    float r;
    asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float at_exp_fast(float a) { return __expf(a); }
__device__ __forceinline__ float at_exp_acc(float a) { return expf(a); }
__device__ __forceinline__ float at_lg2_approx(float a)
{
    float r;
    asm("lg2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float at_log_acc(float a) { return logf(a); }
__device__ __forceinline__ float at_log_fast(float a) { return __logf(a); }
__device__ __forceinline__ float at_log_lg2ln2(float a) { return at_lg2_approx(a) * 0.6931471805599453f; }
__device__ __forceinline__ float at_log_lg2div(float a) { return at_lg2_approx(a) / 1.4426950408889634f; }

#define AT_DIV_RN(x) at_div_rn(1.0f, (x))
#define AT_DIV_FULL(x) at_div_full(1.0f, (x))
#define AT_DIV_APR(x) at_div_approx(1.0f, (x))
#define AT_DIV_RCP(x) at_rcp_approx((x))

// ---------------------------------------------------------------------------
// RoPE partial batched — LOG(4) x EXP(2) x SC(2) x ROT(2) = 32 variants.
// ---------------------------------------------------------------------------
#define ROPE(NAME, LOGF, EXPF, SINF, COSF, ROT)                                \
extern "C" __global__ void NAME(                                               \
    float* __restrict__ q,               /* [p, n_head, head_dim] */           \
    float* __restrict__ k,               /* [p, n_kv, head_dim] */             \
    int rotary_pairs, float theta_base, int head_dim, int n_head, int n_kv,    \
    int p, int base_pos)                                                       \
{                                                                              \
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;              \
    const long total_q = (long)p * n_head * rotary_pairs;                      \
    if (idx >= total_q) return;                                                \
    const int q_stride = n_head * rotary_pairs;                                \
    const int token = (int)(idx / q_stride);                                   \
    const int within = (int)(idx % q_stride);                                  \
    const int head = within / rotary_pairs;                                    \
    const int pair = within % rotary_pairs;                                    \
    const float pos = (float)(base_pos + token);                               \
    const float rd = (float)(2 * rotary_pairs);                                \
    const float expo = (float)(2 * pair) / rd;                                 \
    const float log_base = LOGF(theta_base);                                   \
    const float inv_freq = EXPF(log_base * (0.0f - expo));                     \
    const float theta = pos * inv_freq;                                        \
    const float cos_t = COSF(theta);                                           \
    const float sin_t = SINF(theta);                                           \
    const int q_row_stride = n_head * head_dim;                                \
    const int q_head_off = token * q_row_stride + head * head_dim;             \
    const int q_i0 = q_head_off + pair;                                        \
    const int q_i1 = q_i0 + rotary_pairs;                                      \
    const float q0 = q[q_i0];                                                  \
    const float q1 = q[q_i1];                                                  \
    ROT(q, q_i0, q_i1, q0, q1, cos_t, sin_t);                                  \
    const long total_k = (long)p * n_kv * rotary_pairs;                        \
    if (idx < total_k) {                                                       \
        const int k_stride = n_kv * rotary_pairs;                              \
        const int k_token = (int)(idx / k_stride);                             \
        const int k_within = (int)(idx % k_stride);                            \
        const int k_head = k_within / rotary_pairs;                            \
        const int k_pair = k_within % rotary_pairs;                            \
        const float k_pos = (float)(base_pos + k_token);                       \
        const float k_expo = (float)(2 * k_pair) / rd;                         \
        const float k_inv_freq = EXPF(log_base * (0.0f - k_expo));             \
        const float k_theta = k_pos * k_inv_freq;                              \
        const float k_cos = COSF(k_theta);                                     \
        const float k_sin = SINF(k_theta);                                     \
        const int k_row_stride = n_kv * head_dim;                              \
        const int k_head_off = k_token * k_row_stride + k_head * head_dim;     \
        const int k_i0 = k_head_off + k_pair;                                  \
        const int k_i1 = k_i0 + rotary_pairs;                                  \
        const float k0 = k[k_i0];                                              \
        const float k1 = k[k_i1];                                              \
        ROT(k, k_i0, k_i1, k0, k1, k_cos, k_sin);                              \
    }                                                                          \
}

#define ROT_MA(buf, i0, i1, a, b, c, s)                                        \
    buf[i0] = __fsub_rn(__fmul_rn((a), (c)), __fmul_rn((b), (s)));             \
    buf[i1] = __fadd_rn(__fmul_rn((a), (s)), __fmul_rn((b), (c)));
#define ROT_FMA(buf, i0, i1, a, b, c, s)                                       \
    buf[i0] = __fmaf_rn((a), (c), -__fmul_rn((b), (s)));                       \
    buf[i1] = __fmaf_rn((a), (s), __fmul_rn((b), (c)));

// Issue 742 T1.5 — the CUDA-graph twin: `base_pos` read from a 1-element
// device buffer at kernel runtime (the decode path's pos_dev pattern). Same
// body, same grids — only the scalar's source changes, so the twin is
// bit-identical to the scalar kernel at equal `*base_pos_dev == base_pos`.
#define ROPE_DP(NAME, LOGF, EXPF, SINF, COSF, ROT)                             \
extern "C" __global__ void NAME##_dp(                                          \
    float* __restrict__ q,               /* [p, n_head, head_dim] */           \
    float* __restrict__ k,               /* [p, n_kv, head_dim] */             \
    int rotary_pairs, float theta_base, int head_dim, int n_head, int n_kv,    \
    int p, const int* __restrict__ base_pos_dev)                                \
{                                                                              \
    const int base_pos = *base_pos_dev;                                         \
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;              \
    const long total_q = (long)p * n_head * rotary_pairs;                      \
    if (idx >= total_q) return;                                                \
    const int q_stride = n_head * rotary_pairs;                                \
    const int token = (int)(idx / q_stride);                                   \
    const int within = (int)(idx % q_stride);                                  \
    const int head = within / rotary_pairs;                                    \
    const int pair = within % rotary_pairs;                                    \
    const float pos = (float)(base_pos + token);                               \
    const float rd = (float)(2 * rotary_pairs);                                \
    const float expo = (float)(2 * pair) / rd;                                 \
    const float log_base = LOGF(theta_base);                                   \
    const float inv_freq = EXPF(log_base * (0.0f - expo));                     \
    const float theta = pos * inv_freq;                                        \
    const float cos_t = COSF(theta);                                           \
    const float sin_t = SINF(theta);                                           \
    const int q_row_stride = n_head * head_dim;                                \
    const int q_head_off = token * q_row_stride + head * head_dim;             \
    const int q_i0 = q_head_off + pair;                                        \
    const int q_i1 = q_i0 + rotary_pairs;                                      \
    const float q0 = q[q_i0];                                                  \
    const float q1 = q[q_i1];                                                  \
    ROT(q, q_i0, q_i1, q0, q1, cos_t, sin_t);                                  \
    const long total_k = (long)p * n_kv * rotary_pairs;                        \
    if (idx < total_k) {                                                       \
        const int k_stride = n_kv * rotary_pairs;                              \
        const int k_token = (int)(idx / k_stride);                             \
        const int k_within = (int)(idx % k_stride);                            \
        const int k_head = k_within / rotary_pairs;                            \
        const int k_pair = k_within % rotary_pairs;                            \
        const float k_pos = (float)(base_pos + k_token);                       \
        const float k_expo = (float)(2 * k_pair) / rd;                         \
        const float k_inv_freq = EXPF(log_base * (0.0f - k_expo));             \
        const float k_theta = k_pos * k_inv_freq;                              \
        const float k_cos = COSF(k_theta);                                     \
        const float k_sin = SINF(k_theta);                                     \
        const int k_row_stride = n_kv * head_dim;                              \
        const int k_head_off = k_token * k_row_stride + k_head * head_dim;     \
        const int k_i0 = k_head_off + k_pair;                                  \
        const int k_i1 = k_i0 + rotary_pairs;                                  \
        const float k0 = k[k_i0];                                              \
        const float k1 = k[k_i1];                                              \
        ROT(k, k_i0, k_i1, k0, k1, k_cos, k_sin);                              \
    }                                                                          \
}

#define ROPE_SC2(NAME, LOGF, EXPF, ROT)                                        \
ROPE(NAME##_c0, LOGF, EXPF, sinf, cosf, ROT)                                   \
ROPE(NAME##_c1, LOGF, EXPF, __sinf, __cosf, ROT)                               \
ROPE_DP(NAME##_c0, LOGF, EXPF, sinf, cosf, ROT)                                \
ROPE_DP(NAME##_c1, LOGF, EXPF, __sinf, __cosf, ROT)

#define ROPE_EX2(NAME, LOGF, ROT)                                              \
ROPE_SC2(NAME##_e0, LOGF, at_exp_acc, ROT)                                    \
ROPE_SC2(NAME##_e1, LOGF, at_exp_fast, ROT)

#define ROPE_LG2(NAME, ROT)                                                    \
ROPE_EX2(NAME##_l0, at_log_acc, ROT)                                          \
ROPE_EX2(NAME##_l1, at_log_fast, ROT)                                         \
ROPE_EX2(NAME##_l2, at_log_lg2ln2, ROT)                                       \
ROPE_EX2(NAME##_l3, at_log_lg2div, ROT)

ROPE_LG2(rope_pf_r0, ROT_MA)
ROPE_LG2(rope_pf_r1, ROT_FMA)

// ---------------------------------------------------------------------------
// QG split (batched) — pure copy.
// ---------------------------------------------------------------------------
extern "C" __global__ void split_qg_batched(
    const float* __restrict__ qg,   /* [p, n_head, 2*hd] */
    float* __restrict__ q,          /* [p, n_head, hd] */
    float* __restrict__ gate,       /* [p, n_head, hd] */
    int head_dim, int n_head, int p)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int per_token = n_head * head_dim;
    const long total = (long)p * per_token;
    if (idx >= total) return;
    const int token = (int)(idx / per_token);
    const int within = (int)(idx % per_token);
    const int head = within / head_dim;
    const int dim = within % head_dim;
    const int qg_stride = n_head * 2 * head_dim;
    const long src = (long)token * qg_stride + head * 2 * head_dim + dim;
    q[idx] = qg[src];
    gate[idx] = qg[src + head_dim];
}

// ---------------------------------------------------------------------------
// KV split (batched) — pure copy.
// ---------------------------------------------------------------------------
extern "C" __global__ void split_kv_batched(
    const float* __restrict__ kv,   /* [p, 2*kvd] */
    float* __restrict__ k,          /* [p, kvd] */
    float* __restrict__ v,          /* [p, kvd] */
    int kvd, int p)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)p * kvd;
    if (idx >= total) return;
    const int token = (int)(idx / kvd);
    const int dim = (int)(idx % kvd);
    const long kv_off = (long)token * 2 * kvd + dim;
    k[idx] = kv[kv_off];
    v[idx] = kv[kv_off + kvd];
}

// ---------------------------------------------------------------------------
// KV cache fill (split) — pure copy at absolute rows.
// ---------------------------------------------------------------------------
extern "C" __global__ void kv_cache_fill_split(
    const float* __restrict__ k,         /* [p, kvd] chunk-local */
    const float* __restrict__ v,         /* [p, kvd] chunk-local */
    float* __restrict__ key_cache,       /* [rows, kvd] absolute */
    float* __restrict__ value_cache,     /* [rows, kvd] absolute */
    int kvd, int p, long base_pos)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)p * kvd;
    if (idx >= total) return;
    const long cache_off = base_pos * kvd + idx;
    key_cache[cache_off] = k[idx];
    value_cache[cache_off] = v[idx];
}

// Issue 742 T1.5 — the graph twin: `base_pos` from the device pos buffer.
extern "C" __global__ void kv_cache_fill_split_dp(
    const float* __restrict__ k,         /* [p, kvd] chunk-local */
    const float* __restrict__ v,         /* [p, kvd] chunk-local */
    float* __restrict__ key_cache,       /* [rows, kvd] absolute */
    float* __restrict__ value_cache,     /* [rows, kvd] absolute */
    int kvd, int p, const int* __restrict__ base_pos_dev)
{
    const long base_pos = (long)(*base_pos_dev);
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)p * kvd;
    if (idx >= total) return;
    const long cache_off = base_pos * kvd + idx;
    key_cache[cache_off] = k[idx];
    value_cache[cache_off] = v[idx];
}

// ---------------------------------------------------------------------------
// Causal gated flash attention (prefill) — DOT(2) x ACC(2) x EXP(2) x PH3(2)
// x INV(4) = 128 variants. Grid (n_head * p), block head_dim (256), smem 256.
// Replicates qwen_attention_prefill_gated_f32 exactly: per-thread serial QK
// dot, smem max tree (stride 128→1, `>`), online softmax rescale, Phase-3
// exp weights, serial Phase-4 accumulation, final 1/sum + sigmoid gate.
// ---------------------------------------------------------------------------
#define ATTENTION(NAME, DOTF, ACCF, EXPF, PH3, RESF, INVF)                      \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ query,   /* [p, n_head, hd] */                   \
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */          \
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */          \
    const float* __restrict__ gate,    /* [p, n_head, hd] */                   \
    float* __restrict__ attn_out,      /* [p, n_head, hd] */                   \
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset)      \
{                                                                              \
    __shared__ float smem[256];                                                \
    const int cube_id = blockIdx.x;                                            \
    const int head_idx = cube_id / p;                                          \
    const int q_pos = cube_id % p;                                             \
    const int q_pos_abs = q_pos + q_offset;                                    \
    const int tid = (int)threadIdx.x;                                          \
    const int kv_group = head_idx * n_kv / n_head;                             \
    const int kv_head_off = kv_group * head_dim;                               \
    const int q_stride = n_head * head_dim;                                    \
    const int kv_stride = n_kv * head_dim;                                     \
    const int q_token_off = q_pos * q_stride + head_idx * head_dim;            \
    const bool valid_dim = tid < head_dim;                                     \
    float running_max = -1e30f;                                                \
    float running_sum = 0.0f;                                                  \
    float running_out = 0.0f;                                                  \
    const int n_positions = q_pos_abs + 1;                                     \
    const int n_tiles = (n_positions + head_dim - 1) / head_dim;               \
    for (int tile = 0; tile < n_tiles; tile++) {                               \
        const int tile_base = tile * head_dim;                                 \
        const int pos = tile_base + tid;                                       \
        const bool valid_pos = pos < n_positions;                              \
        float my_score = -1e30f;                                               \
        if (valid_pos) {                                                       \
            const long k_base = (long)pos * kv_stride + kv_head_off;           \
            float dot = 0.0f;                                                  \
            for (int d = 0; d < head_dim; d++) {                               \
                dot = DOTF(query[q_token_off + d], key[k_base + d], dot);      \
            }                                                                  \
            my_score = dot * scale;                                            \
        }                                                                      \
        smem[tid] = my_score;                                                  \
        __syncthreads();                                                       \
        if (tid < 128 && smem[tid + 128] > smem[tid]) {                        \
            smem[tid] = smem[tid + 128];                                       \
        }                                                                      \
        __syncthreads();                                                       \
        for (int stride = 64; stride > 0; stride >>= 1) {                      \
            if (tid < stride && smem[tid + stride] > smem[tid]) {              \
                smem[tid] = smem[tid + stride];                                \
            }                                                                  \
            __syncthreads();                                                   \
        }                                                                      \
        const float tile_max = smem[0];                                        \
        __syncthreads(); /* Issue 715 race (a) */                              \
        const float new_max = tile_max > running_max ? tile_max : running_max; \
        const float exp_prev = EXPF(running_max - new_max);                    \
        const float exp_tile = EXPF(tile_max - new_max);                       \
        running_max = new_max;                                                 \
        if (RESF == 0) {                                                          \
            running_sum = running_sum * exp_prev;                              \
            running_out = running_out * exp_prev;                              \
        } else if (RESF == 1) {                                                 \
            /* deferred: both fold into the accumulation adds */             \
        } else if (RESF == 2) {                                                 \
            running_out = running_out * exp_prev;                              \
        } else {                                                               \
            running_sum = running_sum * exp_prev;                              \
        }                                                                      \
        if (valid_pos) {                                                       \
            if (PH3) {                                                          \
                smem[tid] = EXPF(my_score - new_max);                           \
            } else {                                                            \
                smem[tid] = exp_tile * EXPF(my_score - tile_max);               \
            }                                                                  \
        } else {                                                               \
            smem[tid] = 0.0f;                                                   \
        }                                                                      \
        __syncthreads();                                                       \
        float tile_sum = 0.0f;                                                 \
        float acc = 0.0f;                                                      \
        for (int pp = 0; pp < head_dim; pp++) {                                \
            const int kv_pos = tile_base + pp;                                 \
            if (kv_pos < n_positions) {                                        \
                const float weight = smem[pp];                                 \
                tile_sum = tile_sum + weight;                                  \
                if (valid_dim) {                                               \
                    const long v_idx = (long)kv_pos * kv_stride + kv_head_off  \
                        + tid;                                                 \
                    acc = ACCF(weight, value[v_idx], acc);                     \
                }                                                              \
            }                                                                  \
        }                                                                      \
        if (RESF == 1) {                                                       \
            running_sum = __fmaf_rn(running_sum, exp_prev, tile_sum);          \
            if (valid_dim) {                                                   \
                running_out = __fmaf_rn(running_out, exp_prev, acc);           \
            }                                                                  \
        } else if (RESF == 2) {                                                \
            running_sum = __fmaf_rn(running_sum, exp_prev, tile_sum);          \
            if (valid_dim) {                                                   \
                running_out = running_out + acc;                               \
            }                                                                  \
        } else if (RESF == 3) {                                                \
            running_sum = running_sum + tile_sum;                              \
            if (valid_dim) {                                                   \
                running_out = __fmaf_rn(running_out, exp_prev, acc);           \
            }                                                                  \
        } else {                                                               \
            if (valid_dim) {                                                   \
                running_out = running_out + acc;                               \
            }                                                                  \
            running_sum = running_sum + tile_sum;                              \
        }                                                                      \
        __syncthreads(); /* Issue 715 race (b) */                              \
    }                                                                          \
    if (valid_dim) {                                                           \
        const float inv_sum = INVF(running_sum);                               \
        const float raw = running_out * inv_sum;                               \
        const float g = gate[(long)q_token_off + tid];                         \
        const float sig = at_div_full(1.0f, 1.0f + at_exp_fast(0.0f - g));     \
        attn_out[(long)q_token_off + tid] = raw * sig;                         \
    }                                                                          \
}

#define AD_MA(q, k, d) __fadd_rn((d), __fmul_rn((q), (k)))
#define AD_FMA(q, k, d) __fmaf_rn((q), (k), (d))
#define AA_MA(w, v, a) __fadd_rn((a), __fmul_rn((w), (v)))
#define AA_FMA(w, v, a) __fmaf_rn((w), (v), (a))
#define AI_RN(x) AT_DIV_RN(x)
#define AI_FULL(x) AT_DIV_FULL(x)
#define AI_APR(x) AT_DIV_APR(x)
#define AI_RCP(x) AT_DIV_RCP(x)

#define ATT_INV4(NAME, DOTF, ACCF, EXPF, PH3, RESF)                            \
ATTENTION(NAME##_v0, DOTF, ACCF, EXPF, PH3, RESF, AI_RN)                       \
ATTENTION(NAME##_v1, DOTF, ACCF, EXPF, PH3, RESF, AI_FULL)                     \
ATTENTION(NAME##_v2, DOTF, ACCF, EXPF, PH3, RESF, AI_APR)                     \
ATTENTION(NAME##_v3, DOTF, ACCF, EXPF, PH3, RESF, AI_RCP)

#define ATT_EXP2(NAME, DOTF, ACCF, PH3, RESF)                                  \
ATT_INV4(NAME##_e0, DOTF, ACCF, at_exp_acc, PH3, RESF)                        \
ATT_INV4(NAME##_e1, DOTF, ACCF, at_exp_fast, PH3, RESF)

#define ATT_PH2(NAME, DOTF, ACCF, RESF)                                        \
ATT_EXP2(NAME##_h0, DOTF, ACCF, 0, RESF)                                      \
ATT_EXP2(NAME##_h1, DOTF, ACCF, 1, RESF)

#define ATT_RES4(NAME, DOTF, ACCF)                                             \
ATT_PH2(NAME##_r0, DOTF, ACCF, 0)                                             \
ATT_PH2(NAME##_r1, DOTF, ACCF, 1)                                             \
ATT_PH2(NAME##_r2, DOTF, ACCF, 2)                                             \
ATT_PH2(NAME##_r3, DOTF, ACCF, 3)

#define ATT_ACC2(NAME, DOTF)                                                   \
ATT_RES4(NAME##_a0, DOTF, AA_MA)                                              \
ATT_RES4(NAME##_a1, DOTF, AA_FMA)

ATT_ACC2(att_pf_d0, AD_MA)
ATT_ACC2(att_pf_d1, AD_FMA)

// ---------------------------------------------------------------------------
// Multi-q causal gated attention (Issue 734 Arm 9) — the L2-traffic cut.
//
// One block (256 threads = head_dim) processes ATTN_Q=8 consecutive q
// positions of one head. The 8 q rows are staged in smem ONCE; every K
// element loaded feeds 8 dots; every V row loaded (once, into a register)
// serves all 8 q's Phase-4 accumulations → K/V L2 traffic /8.
//
// BIT-IDENTITY to qwen_attention_prefill_gated_f32 (and to the arm-8
// single-q kernel — the probe-settled canonical forms are HARDCODED):
// - dot: `__fmaf_rn(q, k, dot)` ascending d — the contracted form per q.
// - max tree: identical stride structure (128 → 1, `>`) per qi; the max
//   VALUE is exact for any tree shape (finite scores).
// - online update: `new_max = tmax > rmax ? tmax : rmax`; `exp_prev =
//   __expf(rmax - new_max)`; `exp_tile = __expf(tmax - new_max)`.
// - Phase 3 (Mul form): `exp_tile * __expf(score - tile_max)`.
// - Phase 4: per-qi guarded (kv_pos < q's n_positions) ascending-pp
//   accumulation — the reference's exact (pp, cube) set: `tsum + w` plain
//   add, `__fmaf_rn(w, v, tacc)`.
// - Merge (the settled RESF=1 deferred form): `rsum = fma(rsum, exp_prev,
//   tsum)`; `racc = fma(racc, exp_prev, tacc)`.
// - Final: `div.full(1, rsum)`, `raw = racc * inv`, sigmoid gate via
//   `div.full(1, 1 + __expf(-g))`.
// Tiles beyond a shorter q's causal range are EXACT no-ops (all-invalid
// scores → weights 0 / guarded-out adds; `fma(x, 1.0f, +0.0f) == x` and
// `__expf(0) == 1.0f` exactly), so the block loops to the LONGEST q's
// n_tiles and every qi reproduces its own cube's arithmetic.
// ---------------------------------------------------------------------------
#define ATTN_MQ 8

extern "C" __global__ void __launch_bounds__(256, 4) att_pf_mq8(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ gate,    /* [p, n_head, hd] */
    float* __restrict__ attn_out,      /* [p, n_head, hd] */
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset)
{
    __shared__ float s_w[ATTN_MQ][256]; /* scores → tree → weights per qi */
    __shared__ float s_q[ATTN_MQ][256]; /* the staged q rows */

    const int tid = (int)threadIdx.x;
    const int tiles_per_head = (p + ATTN_MQ - 1) / ATTN_MQ;
    const int head_idx = blockIdx.x / tiles_per_head;
    const int q_base = (int)(blockIdx.x % tiles_per_head) * ATTN_MQ;
    const int q_count = (p - q_base) < ATTN_MQ ? (p - q_base) : ATTN_MQ;

    const int kv_group = head_idx * n_kv / n_head;
    const int kv_head_off = kv_group * head_dim;
    const int q_stride = n_head * head_dim;
    const int kv_stride = n_kv * head_dim;

    /* Stage the q rows (exact copies). */
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        if (qi < q_count) {
            s_q[qi][tid] =
                query[(long)(q_base + qi) * q_stride + head_idx * head_dim + tid];
        }
    }
    __syncthreads();

    float rmax[ATTN_MQ], rsum[ATTN_MQ], racc[ATTN_MQ];
#pragma unroll
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        rmax[qi] = -1e30f;
        rsum[qi] = 0.0f;
        racc[qi] = 0.0f;
    }

    /* The block's longest causal range (also the K/V buffer bound). */
    const int max_n_pos = q_offset + q_base + q_count;
    const int n_tiles = (max_n_pos + head_dim - 1) / head_dim;

    for (int tile = 0; tile < n_tiles; tile++) {
        const int tile_base = tile * head_dim;
        const int pos = tile_base + tid;

        /* Phase 1: this thread's position's score for every qi (one K
         * stream serves all 8 dots). */
        float score[ATTN_MQ];
        if (pos < max_n_pos) {
            const long k_base = (long)pos * kv_stride + kv_head_off;
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) score[qi] = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                const float k = key[k_base + d];
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    score[qi] = __fmaf_rn(s_q[qi][d], k, score[qi]);
                }
            }
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) score[qi] = score[qi] * scale;
        }
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const bool vp = qi < q_count && pos < (q_offset + q_base + qi + 1);
            s_w[qi][tid] = (pos < max_n_pos && vp) ? score[qi] : -1e30f;
        }
        __syncthreads();

        /* Phase 2: the max tree per qi (identical stride structure). */
        if (tid < 128) {
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) {
                if (s_w[qi][tid + 128] > s_w[qi][tid]) {
                    s_w[qi][tid] = s_w[qi][tid + 128];
                }
            }
        }
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    if (s_w[qi][tid + stride] > s_w[qi][tid]) {
                        s_w[qi][tid] = s_w[qi][tid + stride];
                    }
                }
            }
            __syncthreads();
        }
        float tmax[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) tmax[qi] = s_w[qi][0];
        __syncthreads(); /* Issue 715 race (a) */

        /* Online softmax update (deferred-fma merge — the settled form). */
        float eprev[ATTN_MQ], etile[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const float new_max = tmax[qi] > rmax[qi] ? tmax[qi] : rmax[qi];
            eprev[qi] = __expf(rmax[qi] - new_max);
            etile[qi] = __expf(tmax[qi] - new_max);
            rmax[qi] = new_max;
        }

        /* Phase 3: weights (Mul form; 0 for invalid). */
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const bool vp = qi < q_count && pos < (q_offset + q_base + qi + 1);
            s_w[qi][tid] =
                vp ? etile[qi] * __expf(score[qi] - tmax[qi]) : 0.0f;
        }
        __syncthreads();

        /* Phase 4: serial accumulation ascending pp; the V row (one load)
         * serves all qi. */
        float tsum[ATTN_MQ], tacc[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            tsum[qi] = 0.0f;
            tacc[qi] = 0.0f;
        }
        for (int pp = 0; pp < head_dim; pp++) {
            const int kv_pos = tile_base + pp;
            if (kv_pos < max_n_pos) {
                const float v =
                    value[(long)kv_pos * kv_stride + kv_head_off + tid];
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    if (qi < q_count && kv_pos < (q_offset + q_base + qi + 1)) {
                        const float w = s_w[qi][pp];
                        tsum[qi] = tsum[qi] + w;
                        tacc[qi] = __fmaf_rn(w, v, tacc[qi]);
                    }
                }
            }
        }

        /* Merge (RESF=1: both fold into the accumulation adds). */
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            rsum[qi] = __fmaf_rn(rsum[qi], eprev[qi], tsum[qi]);
            racc[qi] = __fmaf_rn(racc[qi], eprev[qi], tacc[qi]);
        }
        __syncthreads(); /* Issue 715 race (b) */
    }

    /* Final: 1/sum (div.full) + sigmoid gate. */
#pragma unroll
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        if (qi < q_count) {
            const float inv_sum = at_div_full(1.0f, rsum[qi]);
            const float raw = racc[qi] * inv_sum;
            const long off =
                (long)(q_base + qi) * q_stride + head_idx * head_dim + tid;
            const float g = gate[off];
            const float sig =
                at_div_full(1.0f, 1.0f + at_exp_fast(0.0f - g));
            attn_out[off] = raw * sig;
        }
    }
}

// Issue 742 T1.5 — the CUDA-graph twin of att_pf_mq8: `q_offset` read from
// the 1-element device pos buffer at kernel runtime. Body VERBATIM (the
// prologue binds the same identifier) — bit-identical at equal values.
extern "C" __global__ void __launch_bounds__(256, 4) att_pf_mq8_dp(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ gate,    /* [p, n_head, hd] */
    float* __restrict__ attn_out,      /* [p, n_head, hd] */
    int head_dim, int n_head, int n_kv, int p, float scale,
    const int* __restrict__ q_offset_dev)
{
    const int q_offset = *q_offset_dev;
    __shared__ float s_w[ATTN_MQ][256]; /* scores → tree → weights per qi */
    __shared__ float s_q[ATTN_MQ][256]; /* the staged q rows */

    const int tid = (int)threadIdx.x;
    const int tiles_per_head = (p + ATTN_MQ - 1) / ATTN_MQ;
    const int head_idx = blockIdx.x / tiles_per_head;
    const int q_base = (int)(blockIdx.x % tiles_per_head) * ATTN_MQ;
    const int q_count = (p - q_base) < ATTN_MQ ? (p - q_base) : ATTN_MQ;

    const int kv_group = head_idx * n_kv / n_head;
    const int kv_head_off = kv_group * head_dim;
    const int q_stride = n_head * head_dim;
    const int kv_stride = n_kv * head_dim;

    /* Stage the q rows (exact copies). */
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        if (qi < q_count) {
            s_q[qi][tid] =
                query[(long)(q_base + qi) * q_stride + head_idx * head_dim + tid];
        }
    }
    __syncthreads();

    float rmax[ATTN_MQ], rsum[ATTN_MQ], racc[ATTN_MQ];
#pragma unroll
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        rmax[qi] = -1e30f;
        rsum[qi] = 0.0f;
        racc[qi] = 0.0f;
    }

    /* The block's longest causal range (also the K/V buffer bound). */
    const int max_n_pos = q_offset + q_base + q_count;
    const int n_tiles = (max_n_pos + head_dim - 1) / head_dim;

    for (int tile = 0; tile < n_tiles; tile++) {
        const int tile_base = tile * head_dim;
        const int pos = tile_base + tid;

        /* Phase 1: this thread's position's score for every qi (one K
         * stream serves all 8 dots). */
        float score[ATTN_MQ];
        if (pos < max_n_pos) {
            const long k_base = (long)pos * kv_stride + kv_head_off;
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) score[qi] = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                const float k = key[k_base + d];
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    score[qi] = __fmaf_rn(s_q[qi][d], k, score[qi]);
                }
            }
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) score[qi] = score[qi] * scale;
        }
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const bool vp = qi < q_count && pos < (q_offset + q_base + qi + 1);
            s_w[qi][tid] = (pos < max_n_pos && vp) ? score[qi] : -1e30f;
        }
        __syncthreads();

        /* Phase 2: the max tree per qi (identical stride structure). */
        if (tid < 128) {
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) {
                if (s_w[qi][tid + 128] > s_w[qi][tid]) {
                    s_w[qi][tid] = s_w[qi][tid + 128];
                }
            }
        }
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    if (s_w[qi][tid + stride] > s_w[qi][tid]) {
                        s_w[qi][tid] = s_w[qi][tid + stride];
                    }
                }
            }
            __syncthreads();
        }
        float tmax[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) tmax[qi] = s_w[qi][0];
        __syncthreads(); /* Issue 715 race (a) */

        /* Online softmax update (deferred-fma merge — the settled form). */
        float eprev[ATTN_MQ], etile[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const float new_max = tmax[qi] > rmax[qi] ? tmax[qi] : rmax[qi];
            eprev[qi] = __expf(rmax[qi] - new_max);
            etile[qi] = __expf(tmax[qi] - new_max);
            rmax[qi] = new_max;
        }

        /* Phase 3: weights (Mul form; 0 for invalid). */
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const bool vp = qi < q_count && pos < (q_offset + q_base + qi + 1);
            s_w[qi][tid] =
                vp ? etile[qi] * __expf(score[qi] - tmax[qi]) : 0.0f;
        }
        __syncthreads();

        /* Phase 4: serial accumulation ascending pp; the V row (one load)
         * serves all qi. */
        float tsum[ATTN_MQ], tacc[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            tsum[qi] = 0.0f;
            tacc[qi] = 0.0f;
        }
        for (int pp = 0; pp < head_dim; pp++) {
            const int kv_pos = tile_base + pp;
            if (kv_pos < max_n_pos) {
                const float v =
                    value[(long)kv_pos * kv_stride + kv_head_off + tid];
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    if (qi < q_count && kv_pos < (q_offset + q_base + qi + 1)) {
                        const float w = s_w[qi][pp];
                        tsum[qi] = tsum[qi] + w;
                        tacc[qi] = __fmaf_rn(w, v, tacc[qi]);
                    }
                }
            }
        }

        /* Merge (RESF=1: both fold into the accumulation adds). */
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            rsum[qi] = __fmaf_rn(rsum[qi], eprev[qi], tsum[qi]);
            racc[qi] = __fmaf_rn(racc[qi], eprev[qi], tacc[qi]);
        }
        __syncthreads(); /* Issue 715 race (b) */
    }

    /* Final: 1/sum (div.full) + sigmoid gate. */
#pragma unroll
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        if (qi < q_count) {
            const float inv_sum = at_div_full(1.0f, rsum[qi]);
            const float raw = racc[qi] * inv_sum;
            const long off =
                (long)(q_base + qi) * q_stride + head_idx * head_dim + tid;
            const float g = gate[off];
            const float sig =
                at_div_full(1.0f, 1.0f + at_exp_fast(0.0f - g));
            attn_out[off] = raw * sig;
        }
    }
}

// Issue 742 T1.6 — the K-staged mq8 body (shared by the host-pos and
// devpos entry points). The serial kernel's phase 1 reads K with
// thread=position mapping: each lane streams its own 256-float row, so a
// warp's scalar loads touch 32 rows 4 KB apart — 32 wavefronts per load
// instruction, ~32x L1/L2 transaction amplification (the measured ~107
// GB/s effective at 2K ctx). This variant replaces ONLY phase 1: the tile's
// 256 K rows are loaded in 8 sub-stages of ATTN_STAGE rows, each stage
// loading COALESCED (consecutive threads read consecutive dims of one row)
// into transposed dynamic smem k_smem[d][r] (+1 pad — bank-conflict-free
// both directions), then the dots run with the (qi, r) mapping: thread
// qi*32+r computes ONE (qi, position) dot as the same ascending-d fma
// chain over the same operand VALUES (smem holds exact copies), so every
// raw score is BIT-IDENTICAL to the serial kernel's. The raw scores land
// in s_w[qi][pos] exactly as the serial phase 1 wrote them; they are
// copied to s_w2 before the (in-place) max tree so phase 3 can re-read
// them (the serial kernel kept them in registers). Phases 2/4, the online
// softmax update and the merge are VERBATIM — the output is bit-identical
// to att_pf_mq8 by construction.
//
// All data loops are bounded by the compile-time MQS_HD (the launcher
// rejects head_dim != 256; the runtime arg is guarded) — the Batch-49 /
// Issue-706 lesson: a runtime-arg loop bound blocks unrolling, and the
// 1-chain-per-thread dot loop MUST unroll to pipeline its smem loads
// ahead of the dependent fma chain (v1 with the runtime bound measured
// 0.8x — smem-load-latency-exposed per iteration).
#define ATTN_STAGE 32
#define MQS_HD 256

__device__ __forceinline__ void att_pf_mq8s_body(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    float* __restrict__ attn_out,
    float* __restrict__ k_smem,   /* MQS_HD * (ATTN_STAGE + 1) floats */
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset)
{
    if (head_dim != MQS_HD) return; /* launcher contract; defensive */
    __shared__ float s_w[ATTN_MQ][MQS_HD]; /* raw scores -> tree -> weights */
    __shared__ float s_q[ATTN_MQ][MQS_HD]; /* the staged q rows; DEAD after
                                            * the dots — s_w2 overlays it */

    const int tid = (int)threadIdx.x;
    const int tiles_per_head = (p + ATTN_MQ - 1) / ATTN_MQ;
    const int head_idx = blockIdx.x / tiles_per_head;
    const int q_base = (int)(blockIdx.x % tiles_per_head) * ATTN_MQ;
    const int q_count = (p - q_base) < ATTN_MQ ? (p - q_base) : ATTN_MQ;

    const int kv_group = head_idx * n_kv / n_head;
    const int kv_head_off = kv_group * head_dim;
    const int q_stride = n_head * head_dim;
    const int kv_stride = n_kv * head_dim;

    /* Stage the q rows (exact copies). */
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        if (qi < q_count) {
            s_q[qi][tid] =
                query[(long)(q_base + qi) * q_stride + head_idx * head_dim + tid];
        }
    }
    __syncthreads();

    float rmax[ATTN_MQ], rsum[ATTN_MQ], racc[ATTN_MQ];
#pragma unroll
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        rmax[qi] = -1e30f;
        rsum[qi] = 0.0f;
        racc[qi] = 0.0f;
    }

    /* The block's longest causal range (also the K/V buffer bound). */
    const int max_n_pos = q_offset + q_base + q_count;
    const int n_tiles = (max_n_pos + head_dim - 1) / head_dim;

    /* Phase-1 (staged) thread mapping: thread -> (qi, r). */
    const int dot_qi = tid >> 5;
    const int dot_r = tid & 31;

    for (int tile = 0; tile < n_tiles; tile++) {
        const int tile_base = tile * head_dim;

        /* Re-stage the q rows: the previous tile's raw-score copy
         * OVERLAID s_q (s_q is dead after the dots — the overlay saves an
         * 8 KB buffer); the q values themselves are unchanged (const
         * input), so this is an exact re-copy. */
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            if (qi < q_count) {
                s_q[qi][tid] =
                    query[(long)(q_base + qi) * q_stride + head_idx * head_dim + tid];
            }
        }
        __syncthreads();

        /* Phase 1 (staged): coalesced K row load + the (qi, r) dots. */
#pragma unroll
        for (int stage = 0; stage < MQS_HD / ATTN_STAGE; stage++) {
            const int stage_base = stage * ATTN_STAGE;
            for (int i = tid; i < ATTN_STAGE * MQS_HD; i += 256) {
                const int r = i / MQS_HD;
                const int d = i % MQS_HD;
                const int pos = tile_base + stage_base + r;
                k_smem[d * (ATTN_STAGE + 1) + r] =
                    (pos < max_n_pos)
                        ? key[(long)pos * kv_stride + kv_head_off + d]
                        : 0.0f;
            }
            __syncthreads();
            {
                const int pos = stage_base + dot_r; /* tile-relative */
                float sc = 0.0f;
#pragma unroll 8
                for (int d = 0; d < MQS_HD; d++) {
                    sc = __fmaf_rn(s_q[dot_qi][d],
                                   k_smem[d * (ATTN_STAGE + 1) + dot_r], sc);
                }
                sc = sc * scale;
                const int apos = tile_base + pos;
                const bool vp = dot_qi < q_count &&
                                apos < (q_offset + q_base + dot_qi + 1);
                s_w[dot_qi][pos] = (apos < max_n_pos && vp) ? sc : -1e30f;
            }
            __syncthreads();
        }

        /* Keep the raw scores: the max tree reduces s_w in place, and
         * phase 3 needs each thread's own raw score (the serial kernel
         * kept it in a register). s_q is dead after the dots — overlay. */
        {
            float* w = &s_w[0][0];
            float* w2 = &s_q[0][0];
            for (int i = tid; i < ATTN_MQ * MQS_HD; i += 256) {
                w2[i] = w[i];
            }
        }
        __syncthreads();

        /* Phase 2: the max tree per qi (identical stride structure). */
        if (tid < 128) {
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) {
                if (s_w[qi][tid + 128] > s_w[qi][tid]) {
                    s_w[qi][tid] = s_w[qi][tid + 128];
                }
            }
        }
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    if (s_w[qi][tid + stride] > s_w[qi][tid]) {
                        s_w[qi][tid] = s_w[qi][tid + stride];
                    }
                }
            }
            __syncthreads();
        }
        float tmax[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) tmax[qi] = s_w[qi][0];
        __syncthreads(); /* Issue 715 race (a) */

        /* Online softmax update (deferred-fma merge — the settled form). */
        float eprev[ATTN_MQ], etile[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const float new_max = tmax[qi] > rmax[qi] ? tmax[qi] : rmax[qi];
            eprev[qi] = __expf(rmax[qi] - new_max);
            etile[qi] = __expf(tmax[qi] - new_max);
            rmax[qi] = new_max;
        }

        /* Phase 3: weights (Mul form; 0 for invalid) — the raw score is
         * re-read from the s_q overlay (bit-identical to the serial
         * register). */
        const int pos = tile_base + tid;
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const bool vp = qi < q_count && pos < (q_offset + q_base + qi + 1);
            s_w[qi][tid] =
                vp ? etile[qi] * __expf(s_q[qi][tid] - tmax[qi]) : 0.0f;
        }
        __syncthreads();

        /* Phase 4: serial accumulation ascending pp; the V row (one load)
         * serves all qi. */
        float tsum[ATTN_MQ], tacc[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            tsum[qi] = 0.0f;
            tacc[qi] = 0.0f;
        }
        for (int pp = 0; pp < head_dim; pp++) {
            const int kv_pos = tile_base + pp;
            if (kv_pos < max_n_pos) {
                const float v =
                    value[(long)kv_pos * kv_stride + kv_head_off + tid];
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    if (qi < q_count && kv_pos < (q_offset + q_base + qi + 1)) {
                        const float w = s_w[qi][pp];
                        tsum[qi] = tsum[qi] + w;
                        tacc[qi] = __fmaf_rn(w, v, tacc[qi]);
                    }
                }
            }
        }

        /* Merge (RESF=1: both fold into the accumulation adds). */
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            rsum[qi] = __fmaf_rn(rsum[qi], eprev[qi], tsum[qi]);
            racc[qi] = __fmaf_rn(racc[qi], eprev[qi], tacc[qi]);
        }
        __syncthreads(); /* Issue 715 race (b) */
    }

    /* Final: 1/sum (div.full) + sigmoid gate. */
#pragma unroll
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        if (qi < q_count) {
            const float inv_sum = at_div_full(1.0f, rsum[qi]);
            const float raw = racc[qi] * inv_sum;
            const long off =
                (long)(q_base + qi) * q_stride + head_idx * head_dim + tid;
            const float g = gate[off];
            const float sig =
                at_div_full(1.0f, 1.0f + at_exp_fast(0.0f - g));
            attn_out[off] = raw * sig;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) att_pf_mq8s(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ gate,    /* [p, n_head, hd] */
    float* __restrict__ attn_out,      /* [p, n_head, hd] */
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset)
{
    extern __shared__ float k_smem[];
    att_pf_mq8s_body(query, key, value, gate, attn_out, k_smem, head_dim,
                     n_head, n_kv, p, scale, q_offset);
}

// Issue 742 T1.5/T1.6 — the CUDA-graph twin of att_pf_mq8s: `q_offset`
// read from the 1-element device pos buffer at kernel runtime.
extern "C" __global__ void __launch_bounds__(256, 2) att_pf_mq8s_dp(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ gate,    /* [p, n_head, hd] */
    float* __restrict__ attn_out,      /* [p, n_head, hd] */
    int head_dim, int n_head, int n_kv, int p, float scale,
    const int* __restrict__ q_offset_dev)
{
    extern __shared__ float k_smem[];
    const int q_offset = *q_offset_dev;
    att_pf_mq8s_body(query, key, value, gate, attn_out, k_smem, head_dim,
                     n_head, n_kv, p, scale, q_offset);
}

// Issue 742 T1.6 — the L2-prefetch mq8 body: VERBATIM serial arithmetic
// (bit-identical by construction — prefetch is semantically inert: it
// touches only cache state, never values or ordering), plus at the top of
// each tile iteration a cooperative prefetch of tile t+1's K and V rows
// into L2. Rationale (measured): the serial kernel at long ctx is
// DRAM-LATENCY-bound — per block ~6 GB/s effective (4.2 MB / 697 us at
// bp=2032) with only 48 blocks x 8 warps of dependent-chain loops keeping
// a handful of loads in flight; the K-staging restructure (att_pf_mq8s)
// did NOT fix this (measured 0.89x) because smem staging does not add
// memory-level parallelism. Warming L2 one tile ahead overlaps the miss
// latency with the current tile's compute. `#pragma unroll 4` on the
// phase-1 d-loop (partial unroll is legal on runtime bounds and keeps
// each per-qi chain d-ascending — same operands, same order, same bits)
// lets the loads issue 4-deep ahead of the fma chains.
__device__ __forceinline__ void att_pf_mq8p_body(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    float* __restrict__ attn_out,
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset)
{
    __shared__ float s_w[ATTN_MQ][256]; /* scores → tree → weights per qi */
    __shared__ float s_q[ATTN_MQ][256]; /* the staged q rows */

    const int tid = (int)threadIdx.x;
    const int tiles_per_head = (p + ATTN_MQ - 1) / ATTN_MQ;
    const int head_idx = blockIdx.x / tiles_per_head;
    const int q_base = (int)(blockIdx.x % tiles_per_head) * ATTN_MQ;
    const int q_count = (p - q_base) < ATTN_MQ ? (p - q_base) : ATTN_MQ;

    const int kv_group = head_idx * n_kv / n_head;
    const int kv_head_off = kv_group * head_dim;
    const int q_stride = n_head * head_dim;
    const int kv_stride = n_kv * head_dim;

    /* Stage the q rows (exact copies). */
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        if (qi < q_count) {
            s_q[qi][tid] =
                query[(long)(q_base + qi) * q_stride + head_idx * head_dim + tid];
        }
    }
    __syncthreads();

    float rmax[ATTN_MQ], rsum[ATTN_MQ], racc[ATTN_MQ];
#pragma unroll
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        rmax[qi] = -1e30f;
        rsum[qi] = 0.0f;
        racc[qi] = 0.0f;
    }

    const int max_n_pos = q_offset + q_base + q_count;
    const int n_tiles = (max_n_pos + head_dim - 1) / head_dim;

    for (int tile = 0; tile < n_tiles; tile++) {
        const int tile_base = tile * head_dim;
        const int pos = tile_base + tid;

        /* L2 prefetch of tile t+1's K+V (2 tensors x 256 rows x 8 x 128B
         * lines = 4096 lines / 256 threads = 16 per thread). Prefetches
         * never fault and carry no data dependency. */
        if (tile + 1 < n_tiles) {
            const int pf_base = (tile + 1) * head_dim;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                const int line = j * 256 + tid;
                const int half = line >> 11;      /* 0 = K, 1 = V */
                const int l = line & 2047;
                const int r = l >> 3;             /* row within tile */
                const int d = (l & 7) << 5;       /* 32-float line offset */
                const int ppos = pf_base + r;
                if (ppos < max_n_pos) {
                    const char* addr = (const char*)(
                        (half == 0 ? key : value) +
                        (long)ppos * kv_stride + kv_head_off + d);
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(addr)
                                 : "memory");
                }
            }
        }

        /* Phase 1: this thread's position's score for every qi (one K
         * stream serves all 8 dots). */
        float score[ATTN_MQ];
        if (pos < max_n_pos) {
            const long k_base = (long)pos * kv_stride + kv_head_off;
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) score[qi] = 0.0f;
#pragma unroll 4
            for (int d = 0; d < head_dim; d++) {
                const float k = key[k_base + d];
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    score[qi] = __fmaf_rn(s_q[qi][d], k, score[qi]);
                }
            }
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) score[qi] = score[qi] * scale;
        }
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const bool vp = qi < q_count && pos < (q_offset + q_base + qi + 1);
            s_w[qi][tid] = (pos < max_n_pos && vp) ? score[qi] : -1e30f;
        }
        __syncthreads();

        /* Phase 2: the max tree per qi (identical stride structure). */
        if (tid < 128) {
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) {
                if (s_w[qi][tid + 128] > s_w[qi][tid]) {
                    s_w[qi][tid] = s_w[qi][tid + 128];
                }
            }
        }
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    if (s_w[qi][tid + stride] > s_w[qi][tid]) {
                        s_w[qi][tid] = s_w[qi][tid + stride];
                    }
                }
            }
            __syncthreads();
        }
        float tmax[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) tmax[qi] = s_w[qi][0];
        __syncthreads(); /* Issue 715 race (a) */

        /* Online softmax update (deferred-fma merge — the settled form). */
        float eprev[ATTN_MQ], etile[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const float new_max = tmax[qi] > rmax[qi] ? tmax[qi] : rmax[qi];
            eprev[qi] = __expf(rmax[qi] - new_max);
            etile[qi] = __expf(tmax[qi] - new_max);
            rmax[qi] = new_max;
        }

        /* Phase 3: weights (Mul form; 0 for invalid). */
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const bool vp = qi < q_count && pos < (q_offset + q_base + qi + 1);
            s_w[qi][tid] =
                vp ? etile[qi] * __expf(score[qi] - tmax[qi]) : 0.0f;
        }
        __syncthreads();

        /* Phase 4: serial accumulation ascending pp; the V row (one load)
         * serves all qi. */
        float tsum[ATTN_MQ], tacc[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            tsum[qi] = 0.0f;
            tacc[qi] = 0.0f;
        }
        for (int pp = 0; pp < head_dim; pp++) {
            const int kv_pos = tile_base + pp;
            if (kv_pos < max_n_pos) {
                const float v =
                    value[(long)kv_pos * kv_stride + kv_head_off + tid];
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    if (qi < q_count && kv_pos < (q_offset + q_base + qi + 1)) {
                        const float w = s_w[qi][pp];
                        tsum[qi] = tsum[qi] + w;
                        tacc[qi] = __fmaf_rn(w, v, tacc[qi]);
                    }
                }
            }
        }

        /* Merge (RESF=1: both fold into the accumulation adds). */
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            rsum[qi] = __fmaf_rn(rsum[qi], eprev[qi], tsum[qi]);
            racc[qi] = __fmaf_rn(racc[qi], eprev[qi], tacc[qi]);
        }
        __syncthreads(); /* Issue 715 race (b) */
    }

    /* Final: 1/sum (div.full) + sigmoid gate. */
#pragma unroll
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        if (qi < q_count) {
            const float inv_sum = at_div_full(1.0f, rsum[qi]);
            const float raw = racc[qi] * inv_sum;
            const long off =
                (long)(q_base + qi) * q_stride + head_idx * head_dim + tid;
            const float g = gate[off];
            const float sig =
                at_div_full(1.0f, 1.0f + at_exp_fast(0.0f - g));
            attn_out[off] = raw * sig;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 4) att_pf_mq8p(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ gate,    /* [p, n_head, hd] */
    float* __restrict__ attn_out,      /* [p, n_head, hd] */
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset)
{
    att_pf_mq8p_body(query, key, value, gate, attn_out, head_dim, n_head,
                     n_kv, p, scale, q_offset);
}

// Issue 742 T1.6 — the CUDA-graph twin of att_pf_mq8p.
extern "C" __global__ void __launch_bounds__(256, 4) att_pf_mq8p_dp(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ gate,    /* [p, n_head, hd] */
    float* __restrict__ attn_out,      /* [p, n_head, hd] */
    int head_dim, int n_head, int n_kv, int p, float scale,
    const int* __restrict__ q_offset_dev)
{
    const int q_offset = *q_offset_dev;
    att_pf_mq8p_body(query, key, value, gate, attn_out, head_dim, n_head,
                     n_kv, p, scale, q_offset);
}

// ===========================================================================
// Issue 742 T1.7 — split-KV mq8 attention (partial + combine): the
// Bench-732/733 flash-decoding pattern applied to the prefill verify chunk.
//
// The T1.6 re-diagnosis: at p=16 the serial/prefetch kernel runs
// n_head*ceil(p/8) = 48 blocks (24/128 SMs), each walking its whole causal
// KV range as a SERIAL chain of 256-position tiles — latency-bound with too
// little memory-level parallelism (~6 GB/s per block effective at bp=2032;
// ~15 ms growth per 2K positions). The fix: split the KV range into
// chunk_len-sized chunks; grid (n_head*tiles_per_head, n_chunks) blocks
// each compute an UNNORMALIZED partial (m, l, out[8][256]) over its chunk —
// the per-chunk tile loop is the serial kernel VERBATIM (fresh running
// state at the chunk boundary) — then a barrier-free combine merges the
// partials (M = max_c m_c; l = sum l_c*exp(m_c-M); out = sum
// out_c*exp(m_c-M); final out/l + sigmoid gate = the serial final phase
// verbatim).
//
// TOLERANCE-CLASS by construction: the cross-chunk merge reassociates the
// serial cascade's per-tile rescale rounding — bit-identity vs the serial
// kernel is impossible (gate: max_rel <= 1e-5, the decode-split precedent
// measured 6.0e-7 on the same class). Dead chunks (beyond the block's
// causal range) write the NEUTRAL partial (m=-1e30, l=0, out=0):
// exp(-1e30-M) underflows to exactly 0, so live-grid (eager) and max-grid
// (graph-capture) merges are BIT-IDENTICAL — test-pinned (the decode-split
// class).
//
// CUDA-graph contract (`_dp` twin): grid FIXED at (n_head*tiles_per_head,
// n_chunks_max); q_offset read from the 1-element device pos buffer; every
// scratch slot the combine reads is rewritten every launch (live or
// neutral) — rollback-safe under replay.
//
// Scratch (caller-owned, one set serves all attn layers — single-stream
// ordering): part_m/part_l [qtile * 8 * n_chunks], part_out
// [qtile * 8 * n_chunks * 256], where qtile = blockIdx.x =
// head_idx*tiles_per_head + q_tile and n_chunks is the STRIDE both kernels
// agree on per launch (live count in eager, max in graph).
// Launcher contracts: head_dim == 256, chunk_len a nonzero multiple of 256
// (tile alignment — a tile never straddles a chunk),
// n_head*tiles_per_head <= 128 (the small-p regime the split serves).
// ===========================================================================
__device__ __forceinline__ void att_pf_mq8kv_body(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    float* __restrict__ part_m,        /* [qtile][8][n_chunks] */
    float* __restrict__ part_l,        /* [qtile][8][n_chunks] */
    float* __restrict__ part_out,      /* [qtile][8][n_chunks][256] */
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset,
    int chunk_len, int n_chunks_stride)
{
    __shared__ float s_w[ATTN_MQ][256]; /* scores -> tree -> weights per qi */
    __shared__ float s_q[ATTN_MQ][256]; /* the staged q rows */

    const int tid = (int)threadIdx.x;
    const int tiles_per_head = (p + ATTN_MQ - 1) / ATTN_MQ;
    const int head_idx = blockIdx.x / tiles_per_head;
    const int q_base = (int)(blockIdx.x % tiles_per_head) * ATTN_MQ;
    const int q_count = (p - q_base) < ATTN_MQ ? (p - q_base) : ATTN_MQ;
    const int chunk_id = blockIdx.y;
    const int qtile = blockIdx.x;

    const int kv_group = head_idx * n_kv / n_head;
    const int kv_head_off = kv_group * head_dim;
    const int q_stride = n_head * head_dim;
    const int kv_stride = n_kv * head_dim;

    /* Partial-slot bases. m/l layout: [qtile][qi][c]; out:
     * [qtile][qi][c][256] (coalesced in d for the combine). */
    const long ml_base = (long)qtile * ATTN_MQ * n_chunks_stride;
    const long po_base = (ml_base + (long)chunk_id) * head_dim;

    /* The block's longest causal range (serial semantics). */
    const int max_n_pos = q_offset + q_base + q_count;
    const int chunk_start = chunk_id * chunk_len;

    /* Dead chunk — neutral partial (uniform whole-block early exit, taken
     * before any barrier). Neutral = the identity element of the combine:
     * exp(-1e30 - M) == 0 and l/out == 0, contributing exact zeros. */
    if (chunk_start >= max_n_pos) {
        if (tid == 0) {
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) {
                part_m[ml_base + (long)qi * n_chunks_stride + chunk_id] =
                    -1e30f;
                part_l[ml_base + (long)qi * n_chunks_stride + chunk_id] =
                    0.0f;
            }
        }
        if (tid < head_dim) {
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) {
                part_out[po_base + (long)qi * n_chunks_stride * head_dim +
                         tid] = 0.0f;
            }
        }
        return;
    }

    /* Stage the q rows (exact copies — serial semantics). */
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        if (qi < q_count) {
            s_q[qi][tid] =
                query[(long)(q_base + qi) * q_stride + head_idx * head_dim + tid];
        }
    }
    __syncthreads();

    float rmax[ATTN_MQ], rsum[ATTN_MQ], racc[ATTN_MQ];
#pragma unroll
    for (int qi = 0; qi < ATTN_MQ; qi++) {
        rmax[qi] = -1e30f;
        rsum[qi] = 0.0f;
        racc[qi] = 0.0f;
    }

    /* This chunk's tile range covering [chunk_start,
     * min(chunk_start+chunk_len, max_n_pos)). chunk_len % head_dim == 0
     * (launcher contract) so tiles never straddle a chunk boundary and
     * tile t covers exactly [t*256, (t+1)*256). */
    const int chunk_end_raw = chunk_start + chunk_len;
    const int chunk_end =
        chunk_end_raw < max_n_pos ? chunk_end_raw : max_n_pos;
    const int t0 = chunk_start / head_dim;
    const int t1 = (chunk_end + head_dim - 1) / head_dim;

    for (int tile = t0; tile < t1; tile++) {
        const int tile_base = tile * head_dim;
        const int pos = tile_base + tid;

        /* Phase 1: this thread's position's score for every qi (one K
         * stream serves all 8 dots). */
        float score[ATTN_MQ];
        if (pos < max_n_pos) {
            const long k_base = (long)pos * kv_stride + kv_head_off;
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) score[qi] = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                const float k = key[k_base + d];
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    score[qi] = __fmaf_rn(s_q[qi][d], k, score[qi]);
                }
            }
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) score[qi] = score[qi] * scale;
        }
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const bool vp = qi < q_count && pos < (q_offset + q_base + qi + 1);
            s_w[qi][tid] = (pos < max_n_pos && vp) ? score[qi] : -1e30f;
        }
        __syncthreads();

        /* Phase 2: the max tree per qi (identical stride structure). */
        if (tid < 128) {
#pragma unroll
            for (int qi = 0; qi < ATTN_MQ; qi++) {
                if (s_w[qi][tid + 128] > s_w[qi][tid]) {
                    s_w[qi][tid] = s_w[qi][tid + 128];
                }
            }
        }
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    if (s_w[qi][tid + stride] > s_w[qi][tid]) {
                        s_w[qi][tid] = s_w[qi][tid + stride];
                    }
                }
            }
            __syncthreads();
        }
        float tmax[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) tmax[qi] = s_w[qi][0];
        __syncthreads(); /* Issue 715 race (a) */

        /* Online softmax update (deferred-fma merge — the settled form). */
        float eprev[ATTN_MQ], etile[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const float new_max = tmax[qi] > rmax[qi] ? tmax[qi] : rmax[qi];
            eprev[qi] = __expf(rmax[qi] - new_max);
            etile[qi] = __expf(tmax[qi] - new_max);
            rmax[qi] = new_max;
        }

        /* Phase 3: weights (Mul form; 0 for invalid). */
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            const bool vp = qi < q_count && pos < (q_offset + q_base + qi + 1);
            s_w[qi][tid] =
                vp ? etile[qi] * __expf(score[qi] - tmax[qi]) : 0.0f;
        }
        __syncthreads();

        /* Phase 4: serial accumulation ascending pp; the V row (one load)
         * serves all qi. */
        float tsum[ATTN_MQ], tacc[ATTN_MQ];
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            tsum[qi] = 0.0f;
            tacc[qi] = 0.0f;
        }
        for (int pp = 0; pp < head_dim; pp++) {
            const int kv_pos = tile_base + pp;
            if (kv_pos < max_n_pos) {
                const float v =
                    value[(long)kv_pos * kv_stride + kv_head_off + tid];
#pragma unroll
                for (int qi = 0; qi < ATTN_MQ; qi++) {
                    if (qi < q_count && kv_pos < (q_offset + q_base + qi + 1)) {
                        const float w = s_w[qi][pp];
                        tsum[qi] = tsum[qi] + w;
                        tacc[qi] = __fmaf_rn(w, v, tacc[qi]);
                    }
                }
            }
        }

        /* Merge (RESF=1: both fold into the accumulation adds). */
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            rsum[qi] = __fmaf_rn(rsum[qi], eprev[qi], tsum[qi]);
            racc[qi] = __fmaf_rn(racc[qi], eprev[qi], tacc[qi]);
        }
        __syncthreads(); /* Issue 715 race (b) */
    }

    /* The UNNORMALIZED partial (the combine owns 1/l + the gate). */
    if (tid == 0) {
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            part_m[ml_base + (long)qi * n_chunks_stride + chunk_id] = rmax[qi];
            part_l[ml_base + (long)qi * n_chunks_stride + chunk_id] = rsum[qi];
        }
    }
    if (tid < head_dim) {
#pragma unroll
        for (int qi = 0; qi < ATTN_MQ; qi++) {
            part_out[po_base + (long)qi * n_chunks_stride * head_dim + tid] =
                racc[qi];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 4) att_pf_mq8kv(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    int head_dim, int n_head, int n_kv, int p, float scale, int q_offset,
    int chunk_len, int n_chunks_stride)
{
    att_pf_mq8kv_body(query, key, value, part_m, part_l, part_out, head_dim,
                      n_head, n_kv, p, scale, q_offset, chunk_len,
                      n_chunks_stride);
}

// Issue 742 T1.7 — the CUDA-graph twin: q_offset read from the 1-element
// device pos buffer at kernel runtime; grid FIXED at (n_head*tph,
// n_chunks_max) — dead chunks write the neutral partial (every scratch
// slot rewritten every launch = rollback-safe under replay).
extern "C" __global__ void __launch_bounds__(256, 4) att_pf_mq8kv_dp(
    const float* __restrict__ query,   /* [p, n_head, hd] */
    const float* __restrict__ key,     /* [(base_pos+p), n_kv, hd] */
    const float* __restrict__ value,   /* [(base_pos+p), n_kv, hd] */
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    float* __restrict__ part_out,
    int head_dim, int n_head, int n_kv, int p, float scale,
    const int* __restrict__ q_offset_dev,
    int chunk_len, int n_chunks_stride)
{
    const int q_offset = *q_offset_dev;
    att_pf_mq8kv_body(query, key, value, part_m, part_l, part_out, head_dim,
                      n_head, n_kv, p, scale, q_offset, chunk_len,
                      n_chunks_stride);
}

// Issue 742 T1.7 — the merge: for each (qtile, qi), M = max_c m_c;
// l = sum_c l_c*exp(m_c-M); out[d] = sum_c out_c[d]*exp(m_c-M); then the
// serial final phase VERBATIM (div.full 1/l, sigmoid gate via div.full).
// One block per qtile, one thread per dim; the per-chunk scalars are
// broadcast reads (barrier-free, race-free by construction — the
// Issue-715 class cannot arise; deterministic ascending-c order).
// Dead chunks (neutral partials) contribute exp(-1e30-M)*0 = 0 exactly, so
// live-grid and max-grid merges are BIT-IDENTICAL (test-pinned). All
// params are capture-fixed (q_offset only affects the partial) — no devpos
// twin needed.
extern "C" __global__ void __launch_bounds__(256, 4) att_pf_mq8kv_combine(
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    const float* __restrict__ part_out,
    const float* __restrict__ gate,    /* [p, n_head, hd] */
    float* __restrict__ attn_out,      /* [p, n_head, hd] */
    int head_dim, int n_head, int p, int n_chunks, int tiles_per_head)
{
    const int tid = (int)threadIdx.x;
    const int qtile = blockIdx.x;
    const int head_idx = qtile / tiles_per_head;
    const int q_base = (int)(qtile % tiles_per_head) * ATTN_MQ;
    const int q_count = (p - q_base) < ATTN_MQ ? (p - q_base) : ATTN_MQ;

    const long ml_base = (long)qtile * ATTN_MQ * n_chunks;
    const long po_base = ml_base * head_dim;
    const int q_stride = n_head * head_dim;

    for (int qi = 0; qi < q_count; qi++) {
        const long ml = ml_base + (long)qi * n_chunks;
        float m_max = -1e30f;
        for (int c = 0; c < n_chunks; c++) {
            if (part_m[ml + c] > m_max) m_max = part_m[ml + c];
        }
        float lsum = 0.0f;
        float out_acc = 0.0f;
        const long po = po_base + (long)qi * n_chunks * head_dim;
        for (int c = 0; c < n_chunks; c++) {
            const float w = at_exp_fast(part_m[ml + c] - m_max);
            lsum += part_l[ml + c] * w;
            out_acc += w * part_out[po + (long)c * head_dim + tid];
        }
        /* Final phase — VERBATIM the serial kernel's. */
        const float inv_sum = at_div_full(1.0f, lsum);
        const float raw = out_acc * inv_sum;
        const long off =
            (long)(q_base + qi) * q_stride + head_idx * head_dim + tid;
        const float g = gate[off];
        const float sig =
            at_div_full(1.0f, 1.0f + at_exp_fast(0.0f - g));
        attn_out[off] = raw * sig;
    }
}
"#;

// ---------------------------------------------------------------------------
// Kernel wrapper
// ---------------------------------------------------------------------------

/// The cudarc-side attention prefill kernels (all variants compiled; the
/// canonical forms selected by the probe — see module doc).
pub struct CudaAttnKernels {
    rope: Vec<CudaFunction>,
    /// Issue 742 T1.5 — the 32 CUDA-graph twins (base_pos from a device
    /// buffer; same idx mapping as `rope`).
    rope_dp: Vec<CudaFunction>,
    split_qg: CudaFunction,
    split_kv: CudaFunction,
    kv_fill: CudaFunction,
    /// Issue 742 T1.5 — the kv_fill graph twin.
    kv_fill_dp: CudaFunction,
    attention: Vec<CudaFunction>,
    attention_mq8: CudaFunction,
    /// Issue 742 T1.5 — the mq8 graph twin.
    attention_mq8_dp: CudaFunction,
    /// Issue 742 T1.6 — the K-staged mq8 (bit-identical, coalesced K reads).
    attention_mq8s: CudaFunction,
    /// Issue 742 T1.6 — the K-staged mq8 graph twin.
    attention_mq8s_dp: CudaFunction,
    /// Issue 742 T1.6 — the L2-prefetch mq8 (bit-identical, latency fix).
    attention_mq8p: CudaFunction,
    /// Issue 742 T1.6 — the L2-prefetch mq8 graph twin.
    attention_mq8p_dp: CudaFunction,
    /// Issue 742 T1.7 — the split-KV mq8 partial (tolerance-class).
    attention_mq8kv: CudaFunction,
    /// Issue 742 T1.7 — the split-KV mq8 partial graph twin (fixed grid,
    /// neutral dead chunks).
    attention_mq8kv_dp: CudaFunction,
    /// Issue 742 T1.7 — the split-KV mq8 partial merge (capture-fixed
    /// params — no devpos twin needed).
    attention_mq8kv_combine: CudaFunction,
    /// Issue 896 — the vectorized-MLP mq8 (bit-identical: float4 row loads
    /// + deeper unroll; no ordering change).
    attention_mq8v: CudaFunction,
    /// Issue 896 — the vectorized-MLP mq8 graph twin.
    attention_mq8v_dp: CudaFunction,
    /// Issue 898 — the head-ganged kv_group mq8 (one block per (kv_group,
    /// q-tile) serving all 6 heads; 98,880 B dynamic smem, bit-identical).
    attention_mq8g: CudaFunction,
    /// Issue 898 — the head-ganged mq8 graph twin.
    attention_mq8g_dp: CudaFunction,
    /// Issue 899 — the GH=3 half-gang occupancy rung (24 rows/block,
    /// 2 blocks per (kv_group, q-tile); 49,440 B dynamic smem, 2
    /// blocks/SM; bit-identical by the same construction).
    attention_mq8g3: CudaFunction,
    /// Issue 899 — the GH=3 half-gang graph twin.
    attention_mq8g3_dp: CudaFunction,
    /// Issue 899 — the GH=2 third-gang occupancy rung (16 rows/block,
    /// 3 blocks per (kv_group, q-tile); 32,960 B dynamic smem, 3
    /// blocks/SM; bit-identical by the same construction).
    attention_mq8g2: CudaFunction,
    /// Issue 899 — the GH=2 third-gang graph twin.
    attention_mq8g2_dp: CudaFunction,
    /// Plan 605 T1 — the FA-class mma prefill attention (tolerance-class:
    /// f16 operands, f32 KQ acc, f16 VKQ acc — the opponent's production
    /// class; see `prefill_cuda_attention_fa`).
    attention_fa: CudaFunction,
    /// Plan 605 T1 — the FA-class graph twin (q_offset from pos_dev).
    attention_fa_dp: CudaFunction,
    /// Plan 605 T1 — the f32->f16 KV conversion pass (eager).
    kv_f32_to_f16: CudaFunction,
    /// Plan 605 T1 — the f32->f16 KV conversion pass (devpos twin).
    kv_f32_to_f16_dp: CudaFunction,
    _module: Arc<CudaModule>,
}

fn idx_rope(l: LogForm, e: SigExp, sc: SinCosForm, r: RopeRot) -> usize {
    ((l as usize * 2 + (e == SigExp::Fast) as usize) * 2
        + (sc == SinCosForm::Fast) as usize)
        * 2
        + (r == RopeRot::Fma) as usize
}

fn idx_attention(
    d: AttnDot,
    a: AttnAcc,
    e: SigExp,
    ph: AttnPh3,
    rs: AttnRes,
    i: AttnInv,
) -> usize {
    let di = (d == AttnDot::Fma) as usize;
    let ai = (a == AttnAcc::Fma) as usize;
    let ri = rs as usize;
    let hi = (ph == AttnPh3::Fused) as usize;
    let ei = (e == SigExp::Fast) as usize;
    ((((di * 2 + ai) * 4 + ri) * 2 + hi) * 2 + ei) * 4 + i as usize
}

impl CudaAttnKernels {
    /// Own context + stream (the `new_standalone` precedent).
    pub fn new_standalone() -> Result<(Self, Arc<CudaStream>), String> {
        let ctx = CudaContext::new(0).map_err(|e| e.to_string())?;
        let stream = ctx.new_stream().map_err(|e| e.to_string())?;
        let kernels = Self::new(ctx)?;
        Ok((kernels, stream))
    }

    /// Compile (NVRTC, sm_89) + load all variants against a shared context.
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, String> {
        // Issue 896 — the vec kernels live in their own source file (keeps
        // both .rs files under the line budget); concatenated at compile
        // time AFTER the main source (the shared helpers must come first).
        static COMBINED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        let src = COMBINED.get_or_init(|| {
            let mut s = String::with_capacity(
                ATTENTION_PREFILL_CUDA_SRC.len()
                    + crate::prefill_cuda_attention_vec::ATTENTION_PREFILL_VEC_CUDA_SRC.len()
                    + crate::prefill_cuda_attention_gang::ATTENTION_PREFILL_GANG_CUDA_SRC.len()
                    + crate::prefill_cuda_attention_fa::ATTENTION_PREFILL_FA_CUDA_SRC.len(),
            );
            s.push_str(ATTENTION_PREFILL_CUDA_SRC);
            s.push_str(crate::prefill_cuda_attention_vec::ATTENTION_PREFILL_VEC_CUDA_SRC);
            s.push_str(crate::prefill_cuda_attention_gang::ATTENTION_PREFILL_GANG_CUDA_SRC);
            s.push_str(crate::prefill_cuda_attention_fa::ATTENTION_PREFILL_FA_CUDA_SRC);
            s
        });
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            src.as_str(),
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;

        let mut rope = Vec::with_capacity(32);
        let mut rope_dp = Vec::with_capacity(32);
        for l in 0..4u8 {
            for ef in 0..2u8 {
                for c in 0..2u8 {
                    for r in 0..2u8 {
                        let name = format!("rope_pf_r{r}_l{l}_e{ef}_c{c}");
                        rope.push(
                            module
                                .load_function(&name)
                                .map_err(|e| format!("{name}: {e}"))?,
                        );
                        let name_dp = format!("rope_pf_r{r}_l{l}_e{ef}_c{c}_dp");
                        rope_dp.push(
                            module
                                .load_function(&name_dp)
                                .map_err(|e| format!("{name_dp}: {e}"))?,
                        );
                    }
                }
            }
        }

        let split_qg = module
            .load_function("split_qg_batched")
            .map_err(|e| format!("split_qg_batched: {e}"))?;
        let split_kv = module
            .load_function("split_kv_batched")
            .map_err(|e| format!("split_kv_batched: {e}"))?;
        let kv_fill = module
            .load_function("kv_cache_fill_split")
            .map_err(|e| format!("kv_cache_fill_split: {e}"))?;
        let kv_fill_dp = module
            .load_function("kv_cache_fill_split_dp")
            .map_err(|e| format!("kv_cache_fill_split_dp: {e}"))?;

        let mut attention = Vec::with_capacity(512);
        for d in 0..2u8 {
            for a in 0..2u8 {
                for r in 0..4u8 {
                    for h in 0..2u8 {
                        for e in 0..2u8 {
                            for v in 0..4u8 {
                                let name = format!("att_pf_d{d}_a{a}_r{r}_h{h}_e{e}_v{v}");
                                attention.push(
                                    module
                                        .load_function(&name)
                                        .map_err(|e| format!("{name}: {e}"))?,
                                );
                            }
                        }
                    }
                }
            }
        }

        let attention_mq8 = module
            .load_function("att_pf_mq8")
            .map_err(|e| format!("att_pf_mq8: {e}"))?;
        let attention_mq8_dp = module
            .load_function("att_pf_mq8_dp")
            .map_err(|e| format!("att_pf_mq8_dp: {e}"))?;
        let attention_mq8s = module
            .load_function("att_pf_mq8s")
            .map_err(|e| format!("att_pf_mq8s: {e}"))?;
        let attention_mq8s_dp = module
            .load_function("att_pf_mq8s_dp")
            .map_err(|e| format!("att_pf_mq8s_dp: {e}"))?;
        let attention_mq8p = module
            .load_function("att_pf_mq8p")
            .map_err(|e| format!("att_pf_mq8p: {e}"))?;
        let attention_mq8p_dp = module
            .load_function("att_pf_mq8p_dp")
            .map_err(|e| format!("att_pf_mq8p_dp: {e}"))?;
        let attention_mq8kv = module
            .load_function("att_pf_mq8kv")
            .map_err(|e| format!("att_pf_mq8kv: {e}"))?;
        let attention_mq8kv_dp = module
            .load_function("att_pf_mq8kv_dp")
            .map_err(|e| format!("att_pf_mq8kv_dp: {e}"))?;
        let attention_mq8kv_combine = module
            .load_function("att_pf_mq8kv_combine")
            .map_err(|e| format!("att_pf_mq8kv_combine: {e}"))?;
        let attention_mq8v = module
            .load_function("att_pf_mq8v")
            .map_err(|e| format!("att_pf_mq8v: {e}"))?;
        let attention_mq8v_dp = module
            .load_function("att_pf_mq8v_dp")
            .map_err(|e| format!("att_pf_mq8v_dp: {e}"))?;
        let attention_mq8g = module
            .load_function("att_pf_mq8g")
            .map_err(|e| format!("att_pf_mq8g: {e}"))?;
        let attention_mq8g_dp = module
            .load_function("att_pf_mq8g_dp")
            .map_err(|e| format!("att_pf_mq8g_dp: {e}"))?;
        // Issue 899 — the split-gang occupancy ladder (GH=3/GH=2).
        let attention_mq8g3 = module
            .load_function("att_pf_mq8g3")
            .map_err(|e| format!("att_pf_mq8g3: {e}"))?;
        let attention_mq8g3_dp = module
            .load_function("att_pf_mq8g3_dp")
            .map_err(|e| format!("att_pf_mq8g3_dp: {e}"))?;
        let attention_mq8g2 = module
            .load_function("att_pf_mq8g2")
            .map_err(|e| format!("att_pf_mq8g2: {e}"))?;
        let attention_mq8g2_dp = module
            .load_function("att_pf_mq8g2_dp")
            .map_err(|e| format!("att_pf_mq8g2_dp: {e}"))?;
        // K-staged mq8 smem opt-in: 33,792 B dynamic + 24 KB static = 57.8 KB
        // total per block — over the 48 KB default combined budget (under
        // the 99 KB sm_89 per-block max). Same pattern as the mma GEMMs.
        let mq8s_smem = 256 * 33 * core::mem::size_of::<f32>() as i32;
        for f in [&attention_mq8s, &attention_mq8s_dp] {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                mq8s_smem,
            )
            .map_err(|e| format!("att_pf_mq8s smem opt-in: {e}"))?;
        }
        // Head-ganged mq8 smem opt-in (Issue 898): 48·256·2 array floats +
        // 3·48 scalar floats = 98,880 B dynamic — over the 48 KB default
        // combined budget, under the 99 KB sm_89 per-block max. Same
        // pattern as mq8s.
        let mq8g_smem = crate::prefill_cuda_attention_gang::ATTENTION_GANG_SMEM_BYTES as i32;
        for f in [&attention_mq8g, &attention_mq8g_dp] {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                mq8g_smem,
            )
            .map_err(|e| format!("att_pf_mq8g smem opt-in: {e}"))?;
        }
        // Split-gang ladder smem opt-ins (Issue 899): 49,440 B (GH=3, over
        // the 48 KB default) and 32,960 B (GH=2, under it — set anyway for a
        // uniform contract). Co-residency is enforced by the kernels' own
        // __launch_bounds__(256, 6/GH) register caps + the sm_89 per-SM
        // budget (2×49,440 / 3×32,960 + per-block reservations ≈ 99 KB).
        let mq8g3_smem = crate::prefill_cuda_attention_gang::ATTENTION_GANG3_SMEM_BYTES as i32;
        for f in [&attention_mq8g3, &attention_mq8g3_dp] {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                mq8g3_smem,
            )
            .map_err(|e| format!("att_pf_mq8g3 smem opt-in: {e}"))?;
        }
        let mq8g2_smem = crate::prefill_cuda_attention_gang::ATTENTION_GANG2_SMEM_BYTES as i32;
        for f in [&attention_mq8g2, &attention_mq8g2_dp] {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                mq8g2_smem,
            )
            .map_err(|e| format!("att_pf_mq8g2 smem opt-in: {e}"))?;
        }

        // Plan 605 T1 — the FA-class kernels (33,792 B dynamic smem — under
        // the 48 KB default budget, no attribute opt-in needed).
        let attention_fa = module
            .load_function("att_pf_fa")
            .map_err(|e| format!("att_pf_fa: {e}"))?;
        let attention_fa_dp = module
            .load_function("att_pf_fa_dp")
            .map_err(|e| format!("att_pf_fa_dp: {e}"))?;
        let kv_f32_to_f16 = module
            .load_function("kv_f32_to_f16")
            .map_err(|e| format!("kv_f32_to_f16: {e}"))?;
        let kv_f32_to_f16_dp = module
            .load_function("kv_f32_to_f16_dp")
            .map_err(|e| format!("kv_f32_to_f16_dp: {e}"))?;

        Ok(Self {
            rope,
            rope_dp,
            split_qg,
            split_kv,
            kv_fill,
            kv_fill_dp,
            attention,
            attention_mq8,
            attention_mq8_dp,
            attention_mq8s,
            attention_mq8s_dp,
            attention_mq8p,
            attention_mq8p_dp,
            attention_mq8kv,
            attention_mq8kv_dp,
            attention_mq8kv_combine,
            attention_mq8v,
            attention_mq8v_dp,
            attention_mq8g,
            attention_mq8g_dp,
            attention_mq8g3,
            attention_mq8g3_dp,
            attention_mq8g2,
            attention_mq8g2_dp,
            attention_fa,
            attention_fa_dp,
            kv_f32_to_f16,
            kv_f32_to_f16_dp,
            _module: module,
        })
    }

    // -- launchers ---------------------------------------------------------

    /// Batched partial RoPE for all `p` tokens (absolute positions
    /// `base_pos + t`), in-place on `q` and `k`.
    ///
    /// # Safety
    ///
    /// Caller guarantees `q` covers `p * n_head * head_dim`, `k` covers
    /// `p * n_kv * head_dim`, and `rotary_pairs * 2 <= head_dim`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_rope(
        &self,
        stream: &CudaStream,
        l: LogForm,
        e: SigExp,
        sc: SinCosForm,
        r: RopeRot,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        rotary_pairs: usize,
        theta_base: f32,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        let func = &self.rope[idx_rope(l, e, sc, r)];
        let (rp_i, hd_i, nh_i, nk_i, p_i, bp_i) = (
            rotary_pairs as i32,
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            p as i32,
            base_pos as i32,
        );
        let total = p * n_head * rotary_pairs;
        let grid = total.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(q)
                .arg(k)
                .arg(&rp_i)
                .arg(&theta_base)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&bp_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Split the interleaved QG buffer for all `p` tokens.
    ///
    /// # Safety
    ///
    /// Caller guarantees `qg` covers `p * 2 * n_head * head_dim`, `q`/`gate`
    /// cover `p * n_head * head_dim` each.
    pub unsafe fn launch_split_qg(
        &self,
        stream: &CudaStream,
        qg: &CudaSlice<f32>,
        q: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        p: usize,
    ) -> Result<(), String> {
        let (hd_i, nh_i, p_i) = (head_dim as i32, n_head as i32, p as i32);
        let total = p * n_head * head_dim;
        let grid = total.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.split_qg)
                .arg(qg)
                .arg(q)
                .arg(gate)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&p_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Split the combined KV buffer for all `p` tokens.
    ///
    /// # Safety
    ///
    /// Caller guarantees `kv` covers `p * 2 * kvd`, `k`/`v` cover `p * kvd`.
    pub unsafe fn launch_split_kv(
        &self,
        stream: &CudaStream,
        kv: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        kvd: usize,
        p: usize,
    ) -> Result<(), String> {
        let (kvd_i, p_i) = (kvd as i32, p as i32);
        let total = p * kvd;
        let grid = total.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.split_kv)
                .arg(kv)
                .arg(k)
                .arg(v)
                .arg(&kvd_i)
                .arg(&p_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Fill KV cache rows `[base_pos, base_pos + p)` from chunk-local K/V.
    ///
    /// # Safety
    ///
    /// Caller guarantees `k`/`v` cover `p * kvd` and the caches cover
    /// `(base_pos + p) * kvd`.
    pub unsafe fn launch_kv_fill(
        &self,
        stream: &CudaStream,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        key_cache: &CudaSlice<f32>,
        value_cache: &CudaSlice<f32>,
        kvd: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        let (kvd_i, p_i) = (kvd as i32, p as i32);
        let bp_l = base_pos as i64;
        let total = p * kvd;
        let grid = total.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.kv_fill)
                .arg(k)
                .arg(v)
                .arg(key_cache)
                .arg(value_cache)
                .arg(&kvd_i)
                .arg(&p_i)
                .arg(&bp_l)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Causal gated flash attention over all `p` query tokens (absolute
    /// causal range via `q_offset = base_pos`).
    ///
    /// # Safety
    ///
    /// Caller guarantees `query`/`gate`/`attn_out` cover
    /// `p * n_head * head_dim`, `key`/`value` cover
    /// `(base_pos + p) * n_kv * head_dim`, and `head_dim <= 256`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention(
        &self,
        stream: &CudaStream,
        d: AttnDot,
        a: AttnAcc,
        e: SigExp,
        ph: AttnPh3,
        rs: AttnRes,
        i: AttnInv,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        // Host-computed scale, exactly like the CubeCL launcher.
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let func = &self.attention[idx_attention(d, a, e, ph, rs, i)];
        let (hd_i, nh_i, nk_i, p_i, qo_i) = (
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            p as i32,
            base_pos as i32,
        );
        let cfg = LaunchConfig {
            grid_dim: ((n_head * p) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(&qo_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Multi-q causal gated attention (Issue 734 Arm 9 — the L2-traffic
    /// cut): one 256-thread block processes `ATTN_MQ=8` consecutive q
    /// positions of one head. Bit-identical to [`Self::launch_attention`]
    /// with the canonical (probe-settled) forms — every arithmetic sequence
    /// per q position is unchanged; only the K/V load sharing differs.
    ///
    /// # Safety
    ///
    /// Caller guarantees `head_dim == 256` (the block/tile width) plus the
    /// [`Self::launch_attention`] buffer contracts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        debug_assert_eq!(head_dim, 256, "mq8 requires head_dim == 256");
        if head_dim != 256 {
            return Err("att_pf_mq8: head_dim != 256".into());
        }
        // Host-computed scale, exactly like the CubeCL launcher.
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (nh_i, nk_i, p_i, qo_i) = (
            n_head as i32,
            n_kv_head as i32,
            p as i32,
            base_pos as i32,
        );
        let grid = (n_head * p.div_ceil(8)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_mq8)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&256i32)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(&qo_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 742 T1.5 — CUDA-graph twin of [`Self::launch_rope`]: `base_pos`
    /// read from `pos_dev` (1-element device buffer) at kernel runtime. Same
    /// grids/args otherwise — bit-identical at `*pos_dev == base_pos`.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_rope`]; `pos_dev` covers 1 element.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_rope_devpos(
        &self,
        stream: &CudaStream,
        l: LogForm,
        e: SigExp,
        sc: SinCosForm,
        r: RopeRot,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        rotary_pairs: usize,
        theta_base: f32,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        pos_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        let func = &self.rope_dp[idx_rope(l, e, sc, r)];
        let (rp_i, hd_i, nh_i, nk_i, p_i) = (
            rotary_pairs as i32,
            head_dim as i32,
            n_head as i32,
            n_kv_head as i32,
            p as i32,
        );
        let total = p * n_head * rotary_pairs;
        let grid = total.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(func)
                .arg(q)
                .arg(k)
                .arg(&rp_i)
                .arg(&theta_base)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 742 T1.5 — CUDA-graph twin of [`Self::launch_kv_fill`].
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_kv_fill`]; `pos_dev` covers 1 element.
    pub unsafe fn launch_kv_fill_devpos(
        &self,
        stream: &CudaStream,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        key_cache: &CudaSlice<f32>,
        value_cache: &CudaSlice<f32>,
        kvd: usize,
        p: usize,
        pos_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        let (kvd_i, p_i) = (kvd as i32, p as i32);
        let total = p * kvd;
        let grid = total.div_ceil(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.kv_fill_dp)
                .arg(k)
                .arg(v)
                .arg(key_cache)
                .arg(value_cache)
                .arg(&kvd_i)
                .arg(&p_i)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 742 T1.5 — CUDA-graph twin of [`Self::launch_attention_mq8`].
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8`]; `pos_dev` covers 1
    /// element.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_devpos(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        pos_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        debug_assert_eq!(head_dim, 256, "mq8 requires head_dim == 256");
        if head_dim != 256 {
            return Err("att_pf_mq8_dp: head_dim != 256".into());
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (nh_i, nk_i, p_i) = (n_head as i32, n_kv_head as i32, p as i32);
        let grid = (n_head * p.div_ceil(8)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_mq8_dp)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&256i32)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 742 T1.6 — K-staged [`Self::launch_attention_mq8`]: phase-1 K
    /// reads go through transposed dynamic smem via COALESCED row loads (the
    /// serial kernel's thread=position mapping costs ~32x L1 wavefront
    /// amplification at long ctx). Every fma chain is unchanged — the output
    /// is BIT-IDENTICAL to `att_pf_mq8` by construction (gate-pinned).
    ///
    /// Dynamic smem = `head_dim * (ATTN_STAGE + 1)` floats = 33,792 B at 256
    /// (under the 48 KB default — no attribute opt-in needed).
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_staged(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        debug_assert_eq!(head_dim, 256, "mq8s requires head_dim == 256");
        if head_dim != 256 {
            return Err("att_pf_mq8s: head_dim != 256".into());
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (nh_i, nk_i, p_i, qo_i) = (
            n_head as i32,
            n_kv_head as i32,
            p as i32,
            base_pos as i32,
        );
        let grid = (n_head * p.div_ceil(8)) as u32;
        let smem = (head_dim * 33 * core::mem::size_of::<f32>()) as u32; // ATTN_STAGE+1 floats, BYTES
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_mq8s)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&256i32)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(&qo_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 742 T1.6 — CUDA-graph twin of
    /// [`Self::launch_attention_mq8_staged`] (`base_pos` from `pos_dev` at
    /// kernel runtime). Bit-identical to `att_pf_mq8_dp` at equal values.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8_devpos`]; `pos_dev`
    /// covers 1 element.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_staged_devpos(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        pos_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        debug_assert_eq!(head_dim, 256, "mq8s requires head_dim == 256");
        if head_dim != 256 {
            return Err("att_pf_mq8s_dp: head_dim != 256".into());
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (nh_i, nk_i, p_i) = (n_head as i32, n_kv_head as i32, p as i32);
        let grid = (n_head * p.div_ceil(8)) as u32;
        let smem = (head_dim * 33 * core::mem::size_of::<f32>()) as u32; // ATTN_STAGE+1 floats, BYTES
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_mq8s_dp)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&256i32)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 742 T1.6 — L2-prefetch [`Self::launch_attention_mq8`]: VERBATIM
    /// serial arithmetic (bit-identical by construction — the prefetch is
    /// semantically inert) + a cooperative prefetch of tile t+1's K/V into
    /// L2 at each tile top (the serial kernel is DRAM-latency-bound at long
    /// ctx: ~6 GB/s per block measured) + `#pragma unroll 4` on the phase-1
    /// d-loop (issue-ahead; each per-qi chain stays d-ascending).
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_prefetch(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        debug_assert_eq!(head_dim, 256, "mq8p requires head_dim == 256");
        if head_dim != 256 {
            return Err("att_pf_mq8p: head_dim != 256".into());
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (nh_i, nk_i, p_i, qo_i) = (
            n_head as i32,
            n_kv_head as i32,
            p as i32,
            base_pos as i32,
        );
        let grid = (n_head * p.div_ceil(8)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_mq8p)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&256i32)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(&qo_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 742 T1.6 — CUDA-graph twin of
    /// [`Self::launch_attention_mq8_prefetch`].
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8_devpos`]; `pos_dev`
    /// covers 1 element.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_prefetch_devpos(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        pos_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        debug_assert_eq!(head_dim, 256, "mq8p requires head_dim == 256");
        if head_dim != 256 {
            return Err("att_pf_mq8p_dp: head_dim != 256".into());
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (nh_i, nk_i, p_i) = (n_head as i32, n_kv_head as i32, p as i32);
        let grid = (n_head * p.div_ceil(8)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_mq8p_dp)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&256i32)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 896 — the vectorized-MLP mq8 (float4 K-row loads + deeper
    /// unroll in the two row-scan phases; the 1-tile L2 prefetch block kept
    /// verbatim from [`Self::launch_attention_mq8_prefetch`]). BIT-IDENTICAL
    /// to [`Self::launch_attention_mq8`] by construction — the fma chains
    /// stay d/pp-ascending with unchanged operand order (the T1.6
    /// load-reorder precedent; pinned to_bits by `bench_896_attn_vec_g1`).
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_vec(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        debug_assert_eq!(head_dim, 256, "mq8v requires head_dim == 256");
        if head_dim != 256 {
            return Err("att_pf_mq8v: head_dim != 256".into());
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (nh_i, nk_i, p_i, qo_i) = (
            n_head as i32,
            n_kv_head as i32,
            p as i32,
            base_pos as i32,
        );
        let grid = (n_head * p.div_ceil(8)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_mq8v)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&256i32)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(&qo_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 896 — CUDA-graph twin of [`Self::launch_attention_mq8_vec`].
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8_devpos`]; `pos_dev`
    /// covers 1 element.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_vec_devpos(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        pos_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        debug_assert_eq!(head_dim, 256, "mq8v requires head_dim == 256");
        if head_dim != 256 {
            return Err("att_pf_mq8v_dp: head_dim != 256".into());
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (nh_i, nk_i, p_i) = (n_head as i32, n_kv_head as i32, p as i32);
        let grid = (n_head * p.div_ceil(8)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_mq8v_dp)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&256i32)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 898 — the head-ganged kv_group mq8: one 256-thread block per
    /// (kv_group, q-tile) serves ALL heads of the GQA group (requires
    /// `n_head/n_kv == 6` — the 48-row gang; other shapes stay on the vec
    /// arm). 48 staged q-rows, K row once per tile (float4, as mq8v), V row
    /// once per pp serving 48 rows — 6× less KV re-reading than the vec
    /// arm's one-block-per-head layout. 98,880 B dynamic smem (opt-in set
    /// in [`Self::new`]); smem-bound at 1 block/SM. BIT-IDENTICAL to
    /// [`Self::launch_attention_mq8_vec`] by construction — every per-row
    /// chain keeps the vec kernel's operand order (pinned to_bits by
    /// `bench_898_attn_gang_g1`).
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_gang(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        if head_dim != 256 {
            return Err("att_pf_mq8g: head_dim != 256".into());
        }
        if n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) || n_head / n_kv_head != 6 {
            return Err(format!(
                "att_pf_mq8g: requires n_head/n_kv == 6 (48-row gang), got \
                 {n_head}/{n_kv_head}"
            ));
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (nh_i, nk_i, p_i, qo_i) = (
            n_head as i32,
            n_kv_head as i32,
            p as i32,
            base_pos as i32,
        );
        let grid = (n_kv_head * p.div_ceil(8)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: crate::prefill_cuda_attention_gang::ATTENTION_GANG_SMEM_BYTES
                as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_mq8g)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&256i32)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(&qo_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 898 — CUDA-graph twin of [`Self::launch_attention_mq8_gang`].
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8_devpos`]; `pos_dev`
    /// covers 1 element.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_gang_devpos(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        pos_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        if head_dim != 256 {
            return Err("att_pf_mq8g_dp: head_dim != 256".into());
        }
        if n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) || n_head / n_kv_head != 6 {
            return Err(format!(
                "att_pf_mq8g_dp: requires n_head/n_kv == 6 (48-row gang), got \
                 {n_head}/{n_kv_head}"
            ));
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (nh_i, nk_i, p_i) = (n_head as i32, n_kv_head as i32, p as i32);
        let grid = (n_kv_head * p.div_ceil(8)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: crate::prefill_cuda_attention_gang::ATTENTION_GANG_SMEM_BYTES
                as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_mq8g_dp)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&256i32)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 899 — shared eager launcher for the split-gang occupancy
    /// ladder: grid `n_kv·bpg·ceil(p/8)` (bpg = blocks per (kv_group,
    /// q-tile), sub-fastest so consecutive blocks share K/V + q rows), 256
    /// threads, `grows = bpg-block rows` staged. Same guards as the full
    /// gang (head_dim 256, n_head/n_kv == 6).
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_attention_mq8_gang_split_eager(
        &self,
        f: &CudaFunction,
        smem: u32,
        bpg: usize,
        tag: &str,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        if head_dim != 256 {
            return Err(format!("{tag}: head_dim != 256"));
        }
        if n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) || n_head / n_kv_head != 6 {
            return Err(format!(
                "{tag}: requires n_head/n_kv == 6, got {n_head}/{n_kv_head}"
            ));
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (nh_i, nk_i, p_i, qo_i) = (
            n_head as i32,
            n_kv_head as i32,
            p as i32,
            base_pos as i32,
        );
        let grid = (n_kv_head * bpg * p.div_ceil(8)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(f)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&256i32)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(&qo_i)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 899 — CUDA-graph twin of the split-gang eager helper.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8_devpos`]; `pos_dev`
    /// covers 1 element.
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_attention_mq8_gang_split_devpos(
        &self,
        f: &CudaFunction,
        smem: u32,
        bpg: usize,
        tag: &str,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        pos_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        if head_dim != 256 {
            return Err(format!("{tag}: head_dim != 256"));
        }
        if n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) || n_head / n_kv_head != 6 {
            return Err(format!(
                "{tag}: requires n_head/n_kv == 6, got {n_head}/{n_kv_head}"
            ));
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let (nh_i, nk_i, p_i) = (n_head as i32, n_kv_head as i32, p as i32);
        let grid = (n_kv_head * bpg * p.div_ceil(8)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe {
            stream
                .launch_builder(f)
                .arg(query)
                .arg(key)
                .arg(value)
                .arg(gate)
                .arg(attn_out)
                .arg(&256i32)
                .arg(&nh_i)
                .arg(&nk_i)
                .arg(&p_i)
                .arg(&scale)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Issue 899 — the GH=3 half-gang occupancy rung: 2 blocks per
    /// (kv_group, q-tile), 24 staged rows each (49,440 B dynamic smem,
    /// 2 blocks/SM = 16 resident warps, KV L2 re-reads ×2). BIT-IDENTICAL
    /// to [`Self::launch_attention_mq8_vec`] by construction — every
    /// per-row chain keeps the vec kernel's operand order (pinned to_bits
    /// by `bench_898_attn_gang_g1`).
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_gang3(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        // Safety: same contract as [`Self::launch_attention_mq8`]; the
        // helper is the single unsafe seam (guards + launch).
        unsafe {
            self.launch_attention_mq8_gang_split_eager(
                &self.attention_mq8g3,
                crate::prefill_cuda_attention_gang::ATTENTION_GANG3_SMEM_BYTES as u32,
                2,
                "att_pf_mq8g3",
                stream,
                query,
                key,
                value,
                gate,
                attn_out,
                head_dim,
                n_head,
                n_kv_head,
                p,
                base_pos,
            )
        }
    }

    /// Issue 899 — CUDA-graph twin of [`Self::launch_attention_mq8_gang3`].
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8_devpos`]; `pos_dev`
    /// covers 1 element.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_gang3_devpos(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        pos_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        // Safety: same contract as [`Self::launch_attention_mq8_devpos`];
        // the helper is the single unsafe seam (guards + launch).
        unsafe {
            self.launch_attention_mq8_gang_split_devpos(
                &self.attention_mq8g3_dp,
                crate::prefill_cuda_attention_gang::ATTENTION_GANG3_SMEM_BYTES as u32,
                2,
                "att_pf_mq8g3_dp",
                stream,
                query,
                key,
                value,
                gate,
                attn_out,
                head_dim,
                n_head,
                n_kv_head,
                p,
                pos_dev,
            )
        }
    }

    /// Issue 899 — the GH=2 third-gang occupancy rung: 3 blocks per
    /// (kv_group, q-tile), 16 staged rows each (32,960 B dynamic smem,
    /// 3 blocks/SM = 24 resident warps, KV L2 re-reads ×3). BIT-IDENTICAL
    /// to [`Self::launch_attention_mq8_vec`] by construction — every
    /// per-row chain keeps the vec kernel's operand order (pinned to_bits
    /// by `bench_898_attn_gang_g1`).
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_gang2(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        // Safety: same contract as [`Self::launch_attention_mq8`]; the
        // helper is the single unsafe seam (guards + launch).
        unsafe {
            self.launch_attention_mq8_gang_split_eager(
                &self.attention_mq8g2,
                crate::prefill_cuda_attention_gang::ATTENTION_GANG2_SMEM_BYTES as u32,
                3,
                "att_pf_mq8g2",
                stream,
                query,
                key,
                value,
                gate,
                attn_out,
                head_dim,
                n_head,
                n_kv_head,
                p,
                base_pos,
            )
        }
    }

    /// Issue 899 — CUDA-graph twin of [`Self::launch_attention_mq8_gang2`].
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8_devpos`]; `pos_dev`
    /// covers 1 element.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_gang2_devpos(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        pos_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        // Safety: same contract as [`Self::launch_attention_mq8_devpos`];
        // the helper is the single unsafe seam (guards + launch).
        unsafe {
            self.launch_attention_mq8_gang_split_devpos(
                &self.attention_mq8g2_dp,
                crate::prefill_cuda_attention_gang::ATTENTION_GANG2_SMEM_BYTES as u32,
                3,
                "att_pf_mq8g2_dp",
                stream,
                query,
                key,
                value,
                gate,
                attn_out,
                head_dim,
                n_head,
                n_kv_head,
                p,
                pos_dev,
            )
        }
    }

    /// Plan 605 T1 — the f32→f16 KV conversion pass (eager). Converts
    /// rows `[0, base_pos + p)` of `k`/`v` into the f16 scratch
    /// `kh`/`vh` and ZEROES the pad tail `[kv_len, padded_rows)` (the
    /// causal mask makes the zeros inert; the kernel's cp.async tile
    /// loads stay in bounds).
    ///
    /// Scratch contract: `kh`/`vh` each cover `padded_rows * n_kv_head *
    /// 256` f16 elements, `padded_rows >= ceil((base_pos + p)/32) * 32`
    /// ([`fa_scratch_rows`]).
    ///
    /// # Safety
    ///
    /// Caller guarantees the buffers cover the stated ranges; `k`/`v`
    /// cover `(base_pos + p) * n_kv_head * 256` f32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_kv_f32_to_f16(
        &self,
        stream: &CudaStream,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        kh: &CudaSlice<u16>,
        vh: &CudaSlice<u16>,
        kv_len: usize,
        padded_rows: usize,
        n_kv_head: usize,
    ) -> Result<(), String> {
        let per = padded_rows * n_kv_head * 256;
        let total = 2 * per;
        let threads = 256u32;
        let blocks = total.div_ceil(256).min(65535) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.kv_f32_to_f16)
                .arg(k)
                .arg(v)
                .arg(kh)
                .arg(vh)
                .arg(&(kv_len as i32))
                .arg(&(padded_rows as i32))
                .arg(&(n_kv_head as i32))
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Plan 605 T1 — the f32→f16 KV conversion pass (devpos twin: kv_len =
    /// `*pos_dev + p` at kernel runtime — graphs-safe, fixed grid).
    ///
    /// # Safety
    ///
    /// Same as [`Self::launch_kv_f32_to_f16`]; `pos_dev` covers 1 element;
    /// `padded_rows` is the CAPTURE-time maximum `base_pos + p`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_kv_f32_to_f16_devpos(
        &self,
        stream: &CudaStream,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        kh: &CudaSlice<u16>,
        vh: &CudaSlice<u16>,
        p: usize,
        padded_rows: usize,
        n_kv_head: usize,
        pos_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        let per = padded_rows * n_kv_head * 256;
        let total = 2 * per;
        let threads = 256u32;
        let blocks = total.div_ceil(256).min(65535) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.kv_f32_to_f16_dp)
                .arg(k)
                .arg(v)
                .arg(kh)
                .arg(vh)
                .arg(&(p as i32))
                .arg(&(padded_rows as i32))
                .arg(&(n_kv_head as i32))
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Plan 605 T1 — the FA-class mma prefill attention (eager). Grid
    /// `(ceil(p/4), n_kv)` x 128 threads; `kh`/`vh` are the f16 scratch
    /// from [`Self::launch_kv_f32_to_f16`] (rows `>= fa_scratch_rows(base_pos +
    /// p)` zero-padded).
    ///
    /// TOLERANCE-CLASS vs [`Self::launch_attention_mq8`] — f16 operands +
    /// mma accumulation (the opponent's production class; `bench_946`).
    ///
    /// # Safety
    ///
    /// Caller guarantees: `query`/`gate`/`attn_out` cover `p * n_head *
    /// head_dim`; `kh`/`vh` cover `padded_rows * n_kv_head * head_dim`
    /// f16; `head_dim == 256`; `n_head / n_kv_head == 6`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_fa(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        kh: &CudaSlice<u16>,
        vh: &CudaSlice<u16>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) -> Result<(), String> {
        self.fa_launch_check("att_pf_fa", head_dim, n_head, n_kv_head)?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let cfg = LaunchConfig {
            grid_dim: (p.div_ceil(4) as u32, n_kv_head as u32, 1),
            block_dim: (32, 4, 1), /* (warp_size, nwarps) — threadIdx.y IS the warp index */
            shared_mem_bytes: crate::prefill_cuda_attention_fa::ATTENTION_FA_SMEM_BYTES as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_fa)
                .arg(query)
                .arg(kh)
                .arg(vh)
                .arg(gate)
                .arg(attn_out)
                .arg(&(n_head as i32))
                .arg(&(n_kv_head as i32))
                .arg(&(p as i32))
                .arg(&scale)
                .arg(&(base_pos as i32))
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Plan 605 T1 — the FA-class graph twin (q_offset from `pos_dev` at
    /// kernel runtime).
    ///
    /// # Safety
    ///
    /// Same as [`Self::launch_attention_fa`]; `pos_dev` covers 1 element;
    /// `padded_rows` sized at the CAPTURE-time max `base_pos + p`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_fa_devpos(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        kh: &CudaSlice<u16>,
        vh: &CudaSlice<u16>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        pos_dev: &CudaSlice<i32>,
    ) -> Result<(), String> {
        self.fa_launch_check("att_pf_fa_dp", head_dim, n_head, n_kv_head)?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let cfg = LaunchConfig {
            grid_dim: (p.div_ceil(4) as u32, n_kv_head as u32, 1),
            block_dim: (32, 4, 1), /* (warp_size, nwarps) */
            shared_mem_bytes: crate::prefill_cuda_attention_fa::ATTENTION_FA_SMEM_BYTES as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_fa_dp)
                .arg(query)
                .arg(kh)
                .arg(vh)
                .arg(gate)
                .arg(attn_out)
                .arg(&(n_head as i32))
                .arg(&(n_kv_head as i32))
                .arg(&(p as i32))
                .arg(&scale)
                .arg(pos_dev)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn fa_launch_check(&self, tag: &str, head_dim: usize, n_head: usize, n_kv: usize) -> Result<(), String> {
        if head_dim != 256 {
            return Err(format!("{tag}: head_dim != 256"));
        }
        if n_kv == 0 || !n_head.is_multiple_of(n_kv) || n_head / n_kv != 6 {
            return Err(format!(
                "{tag}: requires n_head/n_kv == 6, got {n_head}/{n_kv}"
            ));
        }
        Ok(())
    }

    /// Issue 742 T1.7 — split-KV mq8 attention (partial + combine), eager
    /// form: grid `(n_head * ceil(p/8), n_chunks)`, `n_chunks` = the LIVE
    /// chunk count `ceil((base_pos+p)/chunk_len)` (used as BOTH grid.y and
    /// the scratch stride — the two always agree per launch).
    ///
    /// TOLERANCE-CLASS vs [`Self::launch_attention_mq8`] (the cross-chunk
    /// merge reassociates the serial cascade's rescale rounding — gate
    /// `max_rel <= 1e-5`, see `bench_742_t1_attn_splitkv_g1`). Live-grid and
    /// max-grid merges are BIT-IDENTICAL (dead chunks contribute exact
    /// zeros — test-pinned).
    ///
    /// Scratch contract: `part_m`/`part_l` cover
    /// `n_head * ceil(p/8) * 8 * n_chunks`, `part_out` covers `... * 256`.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8`]; the caller owns the
    /// three partial buffers (read by the combine in the same stream
    /// order).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_split(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        part_m: &CudaSlice<f32>,
        part_l: &CudaSlice<f32>,
        part_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
        chunk_len: usize,
        n_chunks: usize,
    ) -> Result<(), String> {
        unsafe {
            self.launch_attention_mq8_split_inner(
                stream,
                None,
                query,
                key,
                value,
                gate,
                attn_out,
                part_m,
                part_l,
                part_out,
                head_dim,
                n_head,
                n_kv_head,
                p,
                base_pos,
                chunk_len,
                n_chunks,
            )
        }
    }

    /// Issue 742 T1.7 — CUDA-graph twin of
    /// [`Self::launch_attention_mq8_split`]: grid FIXED at
    /// `(n_head * ceil(p/8), n_chunks_max)`, `q_offset` from the device pos
    /// buffer; dead chunks write the NEUTRAL partial so every scratch slot
    /// is rewritten every launch (rollback-safe under replay). `n_chunks`
    /// is both grid.y and the stride (max-grid).
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8_split`]; `pos_dev`
    /// covers 1 element.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_attention_mq8_split_devpos(
        &self,
        stream: &CudaStream,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        part_m: &CudaSlice<f32>,
        part_l: &CudaSlice<f32>,
        part_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        pos_dev: &CudaSlice<i32>,
        chunk_len: usize,
        n_chunks: usize,
    ) -> Result<(), String> {
        unsafe {
            self.launch_attention_mq8_split_inner(
                stream,
                Some(pos_dev),
                query,
                key,
                value,
                gate,
                attn_out,
                part_m,
                part_l,
                part_out,
                head_dim,
                n_head,
                n_kv_head,
                p,
                0, // q_offset comes from pos_dev in this arm
                chunk_len,
                n_chunks,
            )
        }
    }

    /// The shared partial+combine launch (`devpos = Some` selects the graph
    /// twin: q_offset read from the device buffer instead of the host
    /// scalar).
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::launch_attention_mq8_split`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_attention_mq8_split_inner(
        &self,
        stream: &CudaStream,
        devpos: Option<&CudaSlice<i32>>,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        attn_out: &CudaSlice<f32>,
        part_m: &CudaSlice<f32>,
        part_l: &CudaSlice<f32>,
        part_out: &CudaSlice<f32>,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
        chunk_len: usize,
        n_chunks: usize,
    ) -> Result<(), String> {
        if head_dim != 256 {
            return Err("att_pf_mq8kv: head_dim != 256".into());
        }
        if chunk_len == 0 || !chunk_len.is_multiple_of(256) {
            return Err("att_pf_mq8kv: chunk_len must be a nonzero multiple of 256".into());
        }
        if n_chunks == 0 {
            return Err("att_pf_mq8kv: n_chunks must be nonzero".into());
        }
        let qtiles = n_head * p.div_ceil(8);
        let need_ml = qtiles * 8 * n_chunks;
        if part_m.len() < need_ml || part_l.len() < need_ml || part_out.len() < need_ml * 256 {
            return Err(format!(
                "att_pf_mq8kv: scratch too small (need m/l {need_ml}, out {})",
                need_ml * 256
            ));
        }
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let hd_i = 256i32;
        let (nh_i, nk_i, p_i, ch_i, nc_i, tph_i, qo_i) = (
            n_head as i32,
            n_kv_head as i32,
            p as i32,
            chunk_len as i32,
            n_chunks as i32,
            p.div_ceil(8) as i32,
            base_pos as i32,
        );
        let func = if devpos.is_some() {
            &self.attention_mq8kv_dp
        } else {
            &self.attention_mq8kv
        };
        let cfg = LaunchConfig {
            grid_dim: (qtiles as u32, n_chunks as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            if let Some(pd) = devpos {
                stream
                    .launch_builder(func)
                    .arg(query)
                    .arg(key)
                    .arg(value)
                    .arg(part_m)
                    .arg(part_l)
                    .arg(part_out)
                    .arg(&hd_i)
                    .arg(&nh_i)
                    .arg(&nk_i)
                    .arg(&p_i)
                    .arg(&scale)
                    .arg(pd)
                    .arg(&ch_i)
                    .arg(&nc_i)
                    .launch(cfg)
                    .map_err(|e| e.to_string())?;
            } else {
                stream
                    .launch_builder(func)
                    .arg(query)
                    .arg(key)
                    .arg(value)
                    .arg(part_m)
                    .arg(part_l)
                    .arg(part_out)
                    .arg(&hd_i)
                    .arg(&nh_i)
                    .arg(&nk_i)
                    .arg(&p_i)
                    .arg(&scale)
                    .arg(&qo_i)
                    .arg(&ch_i)
                    .arg(&nc_i)
                    .launch(cfg)
                    .map_err(|e| e.to_string())?;
            }
        }
        let cfg_c = LaunchConfig {
            grid_dim: (qtiles as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.attention_mq8kv_combine)
                .arg(part_m)
                .arg(part_l)
                .arg(part_out)
                .arg(gate)
                .arg(attn_out)
                .arg(&hd_i)
                .arg(&nh_i)
                .arg(&p_i)
                .arg(&nc_i)
                .arg(&tph_i)
                .launch(cfg_c)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
