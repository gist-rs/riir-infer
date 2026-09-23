//! Issue 615 T7 — Full cudarc GPU-resident forward for ternary Qwen3.5 DeltaNet.
//!
//! Mirrors [`TernaryDeltanetGpuForward`] (the CubeCL path) but dispatches every
//! op through cudarc on a shared `Arc<CudaStream>`, capturing the dp4a GEMV
//! kernel's 4.7× speedup end-to-end (Issue 608 T1).
//!
//! ## Why this exists alongside the CubeCL forward
//!
//! The CubeCL path uses `GemvTernaryCubeCL` which achieves ~192 GB/s (~10.7%
//! of the RTX 4090's roofline). The dp4a kernel (`gemv_ternary_dp4a`, Issue 608)
//! achieves 88.9% roofline (901 GB/s) — a 4.7× kernel speedup. But CubeCL and
//! cudarc cannot share device buffers or streams through their public APIs
//! (Issue 615 §"Architectural blocker"), so the only way to capture the dp4a
//! speedup is a full forward port where ALL ops (including elementwise) run on
//! the same cudarc stream.
//!
//! Issue 608 T2 Phase B measured the mixed-mode alternative (CPU elementwise +
//! dp4a GEMV with per-GEMV upload/download) at 14.42 tok/s — **slower** than
//! the CubeCL baseline (15.00 tok/s) because the per-GEMV transfer overhead
//! cancels the kernel speedup. Only a shared-stream GPU-resident path can win.
//!
//! ## Architecture
//!
//! - **One context, one stream**: All kernel sets (`ElementwiseKernels`,
//!   `AttentionKernels`, `DeltanetKernels`, `EmbeddingDequantKernels`, and the
//!   dp4a GEMV `CudaFunction`) are compiled against one `Arc<CudaContext>` and
//!   dispatched on one `Arc<CudaStream>`.
//! - **Pre-uploaded weights**: Per-layer ternary weights are converted to
//!   packed 2-bit codes (`convert_bitplane_to_packed_codes`) and uploaded once.
//! - **Persistent activations**: All activation buffers are allocated once in
//!   [`Self::new`] and reused across every token — zero per-token allocation.
//! - **GPU-side quantize**: The dp4a kernel takes int8 activations. A
//!   `quantize_f32_to_i8` CUDA kernel runs on the shared stream into persistent
//!   scratch buffers. No CPU round-trip.
//! - **Zero sync between layers**: All 64 layers chain as GPU dispatches. The
//!   only CPU↔GPU transfer per token is the logits download at the very end.
//!
//! ## Borrow structure
//!
//! The struct is split into [`ForwardInfraCudarc`] (read-only: ctx, stream,
//! kernels, weights, config) and [`ActivationsCudarc`] (mutable per token).
//! This split lets the borrow checker see that weight reads and activation
//! writes are disjoint — no `&mut self` that locks the whole struct.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use std::sync::Arc;

use cudarc::driver::safe::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice,
};
#[cfg(feature = "cuda_graphs_forward")]
use cudarc::driver::safe::CudaGraph;
use cudarc::driver::PushKernelArg;

use crate::cudarc_kernels::{
    AttentionKernels, CudarcKernelError, DeltanetKernels, ElementwiseKernels,
    EmbeddingDequantKernels, HalfStateFmt, LoraDecodeKernels, QvLoraGpuCudarc,
};
use crate::gemv_ternary_cuda_raw::{convert_bitplane_to_packed_codes, GEMV_CUDA_SRC, WG_THREADS};

use riir_infer_core::deltanet::ternary_weights::{
    DeltaNetTernaryLayerWeights, GateProjWeights, QwenDeltaNetTernaryWeights,
};
use riir_infer_core::types::{Config, DeltaNetLayerType};

use riir_infer_core::deltanet::minimal_activation_cache::{
    MinimalActivationCache, MinimalLayerActivations,
};

// ───────────────────────────────────────────────────────────────────────────
// Weight buffers
// ───────────────────────────────────────────────────────────────────────────

/// Per-weight-matrix device buffers for the dp4a GEMV kernel.
///
/// No `out_dev` — the GEMV output goes directly into the caller-provided
/// activation buffer. This is the key difference that makes the forward path
/// alloc-free.
///
/// Issue 741 T10 Phase B D4 widening: `pub` (+ its 4 fields) so
/// riir-train-gpu's relocated backward section can name the type + read the
/// dims/codes in `gemv_transposed_into`. Reversible at Proposal 041 Phase 2.
pub struct WeightBuffersCudarc {
    pub codes: CudaSlice<i16>,
    /// f16 bits of the per-128-weight group scales (Issue 734 T5 — halves the
    /// wscale DRAM traffic; exact f16→f32 conversion in-kernel).
    pub wscale: CudaSlice<u16>,
    pub m: usize,
    pub n: usize,
}

impl WeightBuffersCudarc {
    fn upload(stream: &Arc<CudaStream>, w: &katgpt_core::TernaryGroupWeights) -> Self {
        let m = w.rows;
        let n = w.cols;
        if m == 0 || n == 0 {
            return Self {
                codes: stream.alloc_zeros::<i16>(0).expect("alloc empty codes"),
                wscale: stream.alloc_zeros::<u16>(0).expect("alloc empty wscale"),
                m: 0,
                n: 0,
            };
        }
        let (codes, wscale) = convert_bitplane_to_packed_codes(w);
        Self {
            codes: stream.clone_htod(&codes).expect("upload ternary codes"),
            wscale: stream.clone_htod(&wscale).expect("upload ternary wscale"),
            m,
            n,
        }
    }
}

/// Per-layer pre-uploaded GPU weight handles.
///
/// Issue 741 T10 Phase B D4 widening: `pub` (+ all fields) so
/// riir-train-gpu's relocated backward section can read the per-layer
/// weight buffers via `fwd.infra.layers[idx]`. Reversible at Proposal 041
/// Phase 2.
pub struct GpuLayerWeightsCudarc {
    // DeltaNet projections (None for Attention layers)
    pub in_proj_qkv: Option<WeightBuffersCudarc>,
    pub in_proj_z: Option<WeightBuffersCudarc>,
    pub in_proj_a: Option<WeightBuffersCudarc>,
    pub in_proj_b: Option<WeightBuffersCudarc>,
    pub out_proj: Option<WeightBuffersCudarc>,
    // Attention projections (None for DeltaNet layers)
    pub attn_wq: Option<WeightBuffersCudarc>,
    pub attn_wk: Option<WeightBuffersCudarc>,
    pub attn_wv: Option<WeightBuffersCudarc>,
    pub attn_wo: Option<WeightBuffersCudarc>,
    // FFN projections (both layer types)
    pub gate_proj: WeightBuffersCudarc,
    pub up_proj: WeightBuffersCudarc,
    pub down_proj: WeightBuffersCudarc,
    // Dense weights (f32 on GPU)
    pub input_norm: CudaSlice<f32>,
    pub post_attn_norm: CudaSlice<f32>,
    /// Issue 980 T4 — the Bonsai-2 dense `ssm_alpha`/`ssm_beta` escape set
    /// (f32, `[n_v_heads × n_embd]`). `None` for ternary a/b (pre-rotation
    /// files) and attention layers; consumed by `gemv_dense_f32` on the
    /// PRIMAL normed input.
    pub dense_a: Option<CudaSlice<f32>>,
    pub dense_b: Option<CudaSlice<f32>>,
    pub conv1d_weight: Option<CudaSlice<f32>>,
    pub a_log: Option<CudaSlice<f32>>,
    pub dt_bias: Option<CudaSlice<f32>>,
    pub linear_norm: Option<CudaSlice<f32>>,
    pub attn_q_norm: Option<CudaSlice<f32>>,
    pub attn_k_norm: Option<CudaSlice<f32>>,
}

/// Issue 705 (filed as 702, renumbered) — multi-GEMV launcher: the compiled
/// `gemv_ternary_dp4a_multi_persistent` function plus the grid cap it is
/// launched with (`min(ceil(total_rows/8), cap)` blocks).
///
/// **Measured default: UNcapped** (one-warp-per-row, the pre-702 shape).
/// The occupancy-derived cap (SMs × blocks/SM = 768 on the 4090) was
/// measured a 1-2% LOSS at every tested size (Bench 684 sweep: uncapped 87.1
/// vs 768-cap 84.9-86.2 vs 1024-cap 82.1 vs 512-cap 75.6 graph tok/s) —
/// the wavefront locality the hardware block scheduler gives the naive
/// over-sized grid outweighs the wave-quantization tail it pays. The wave-tail
/// theory is additionally refuted by the z-class counter-example (768-block
/// exact-1-wave launch measured 706 GB/s vs lm_head's 854 — tails were never
/// the differentiator). The kernel + `RIIR_GEMV_PERSISTENT_GRID` override are
/// retained as the measurement apparatus; the cap is opt-in only.
struct GemvMultiPersistent {
    function: CudaFunction,
    /// Grid cap. `u32::MAX` = uncapped (one-warp-per-row — the measured-best
    /// default). A finite value caps the launch at full-residency-style grids;
    /// the kernel's grid-stride loop then processes multiple rows per warp
    /// (bit-identical at any cap — each row is still computed by exactly one
    /// warp via `gemv_ternary_row`).
    grid: u32,
    /// Plan 604 T1 (Issue 987 rung G1) — two-rows-per-warp variant for
    /// UNDERFILLED launches (see `r2_max_rows`). Same argument signature as
    /// `function`, so the wrapper only swaps the fn handle + grid math.
    /// **Measured −4.4% (Plan 604 T1 A/B, 2026-09-20): REFUTED** — kept as the
    /// negative-result apparatus (RIIR_GEMV_R2=1), never default.
    function_r2: CudaFunction,
    /// `RIIR_GEMV_R2` (default OFF — the A/B knob; the negative result is
    /// recorded, the default will not flip).
    r2_enabled: bool,
    /// Plan 604 T3 (Issue 987 rung G2) — the one-row kernel + one-iteration-
    /// ahead `prefetch.global.L2`, for the same underfilled classes. R2's
    /// negative re-ranked this rung FIRST: the ≤1-wave launches are short on
    /// memory-level parallelism (latency-exposed — the fork's GB10 diagnosis),
    /// which is the condition prefetch addresses.
    function_pf: CudaFunction,
    /// `RIIR_GEMV_PF` (default OFF — A/B first).
    pf_enabled: bool,
    /// Plan 604 follow-up (Issue 987's menu, rung G4 — the untried MLP lever):
    /// the one-row kernel with a MANUALLY 4x-batched K-loop — four REAL
    /// dependent loads in flight per lane (vs PF's hint, vs R2's warp
    /// halving). Same signature + one-row grid as `function`.
    function_u4: CudaFunction,
    /// `RIIR_GEMV_U4` (default OFF — A/B first).
    u4_enabled: bool,
    /// The underfilled threshold: launches whose total row count would map
    /// to ≤ `grid` blocks at the one-row shape (i.e. ≤ occupancy_grid × 8
    /// rows = 6,144 on the 4090) dispatch to `function_r2`/`function_pf`/
    /// `function_u4` when
    /// enabled. T0 (Plan 604): today that population is EXACTLY the FFN down
    /// (m=5,120 n=17,408) + GDN out_proj (m=5,120 n=5,120) classes — the
    /// 640-block grid bucket at ~648 GB/s vs the 824-854 ceiling. Bigger
    /// classes are DRAM-bound (LSU ~63%) and stay one-row-per-warp.
    r2_max_rows: usize,
}

/// Read-only infrastructure: context, stream, compiled kernels, weights, config.
///
/// Separated from [`ActivationsCudarc`] so the forward pass can hold an
/// immutable `&ForwardInfraCudarc` alongside a mutable `&mut ActivationsCudarc`
/// without borrow-checker conflicts.
///
/// Issue 741 T10 Phase B D4 widening: `pub` + the backward-reached fields
/// (`ctx`, `stream`, `layers`, `layer_types`, `gemv_transposed`, `deltanet`,
/// `config`, `attention`) so riir-train-gpu's relocated backward section can
/// reach the forward's infrastructure through `fwd.infra`. The `backward`
/// field itself moved OUT with the Proposal 041 carve: the gradient kernels
/// are a training surface and stay engine-side (riir-gpu's residue module
/// `cudarc_backward_kernels`); the training side holds its own kernel set
/// beside the forward and constructs it from `infra.ctx`.
pub struct ForwardInfraCudarc {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    elementwise: ElementwiseKernels,
    pub attention: AttentionKernels,
    pub deltanet: DeltanetKernels,
    embedding: EmbeddingDequantKernels,
    /// Split-path dp4a kernel (takes pre-quantized int8 + ascale).
    /// Production path. The fused kernel (`gemv_fused`) is kept for
    /// comparison as a documented negative result (Issue 616 T4).
    gemv: CudaFunction,
    /// Issue 697 / Issue 705 — multi-segment dp4a GEMV: one launch for up to 4
    /// weight matrices sharing the same quantized input vector (qkv+z+a+b,
    /// gate+up, q+k+v). Eliminates the m=48 latency-floor a/b launches and
    /// merges wave-quantization tails. Byte-identical row math to `gemv`.
    /// Issue 705 ships the persistent grid-stride variant: the launch grid is
    /// capped at full residency so big launches never pay a partial tail wave.
    gemv_multi: GemvMultiPersistent,
    /// Issue 616 T4 — fused quantize+dp4a kernel (takes f32 activations
    /// directly, quantizes on-the-fly per lane). Eliminates the separate
    /// `quantize_f32_to_i8` launch + the int8/ascale global round-trip.
    ///
    /// **NEGATIVE RESULT** — kept compiled for reproduction but NOT used in
    /// the production forward. See `gemv_fused_into` doc-comment for the
    /// measurements (20.85 / 36.94 / 48.4 tok/s for per-lane / cooperative /
    /// split-path respectively).
    #[allow(dead_code)]
    gemv_fused: CudaFunction,
    /// Issue 641 T7.5 — transposed ternary GEMV for the backward pass.
    /// Loaded from the same `_gemv_module` as the forward `gemv`.
    /// (Consumed cross-crate by riir-train-gpu's backward section since
    /// Issue 741 T10 Phase B.)
    pub gemv_transposed: CudaFunction,
    _gemv_module: Arc<CudaModule>,
    pub config: Config,
    pub layer_types: Vec<DeltaNetLayerType>,
    pub layers: Vec<GpuLayerWeightsCudarc>,
    final_norm: CudaSlice<f32>,
    lm_head: WeightBuffersCudarc,
    wte_pos_bits: CudaSlice<u32>,
    wte_neg_bits: CudaSlice<u32>,
    wte_scale: CudaSlice<f32>,
    wte_blocks64: usize,
    wte_groups_per_row: usize,
    /// Issue 980 T4 — Bonsai-2 Hadamard rotation tables + kernels. `None`
    /// for pre-rotation files (the guard refuses folded models before this
    /// could be built without the feature — the loader owns that gate).
    pub rotation: Option<crate::deltanet_rotation_cudarc::RotationTables>,
}

/// Mutable per-token state: all activation buffers + per-layer recurrent state.
struct ActivationsCudarc {
    x: CudaSlice<f32>,
    tmp: CudaSlice<f32>,
    ffn_out: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    qkv_expanded: CudaSlice<f32>,
    z_buf: CudaSlice<f32>,
    a_raw: CudaSlice<f32>,
    b_raw: CudaSlice<f32>,
    beta_buf: CudaSlice<f32>,
    decay_buf: CudaSlice<f32>,
    recurrent_out: CudaSlice<f32>,
    ffn_gate: CudaSlice<f32>,
    ffn_up: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    attn_qg: CudaSlice<f32>,
    attn_q: CudaSlice<f32>,
    attn_gate: CudaSlice<f32>,
    attn_k: CudaSlice<f32>,
    attn_v: CudaSlice<f32>,
    attn_out: CudaSlice<f32>,
    /// Split-path quantize scratch.
    quant_i8_buf: CudaSlice<i8>,
    ascale_buf: CudaSlice<f32>,
    /// Issue 634 — side buffer for the final RMSNorm f32 output. Only written
    /// by `forward_token_with_final_hidden` (via `rmsnorm_quantize_with_norm_x_f32`);
    /// unused on the hot path (`forward_token` uses the fused `rmsnorm_quantize_f32`
    /// which doesn't emit a side buffer). 8 KiB at n_embd=5120.
    final_norm_x: CudaSlice<f32>,
    /// Issue 980 T4 — the PRIMAL normed input (f32). The split rotation path
    /// norms into this buffer once per layer; the folded qkv/z (and gate/up,
    /// and lm_head) projections consume rotations of it, the dense escape-set
    /// a/b consume it directly.
    norm_x: CudaSlice<f32>,
    /// Issue 980 T4 — rotated-input scratch (f32, n_embd): the sign+FWHT copy
    /// of `norm_x` the folded input projections quantize from. Never aliases
    /// the residual stream (`x` stays primal for the out_proj/down accumulate).
    rot_scratch: CudaSlice<f32>,
    /// Issue 980 T4 — f32 SwiGLU hidden (f32, mlp_hidden): the split down path
    /// swiglus into this, rotates it in place, then quantizes for the GEMV.
    ffn_hidden: CudaSlice<f32>,
    /// Issue 980 T4 — permute scratch for the `gdn_v_grouped` head gather
    /// (f32, v_dim). Written by a device-to-device copy right before the
    /// permute kernel gathers from it.
    permute_tmp: CudaSlice<f32>,
    layer_states: Vec<LayerStateCudarc>,
    pos: usize,
    /// Issue 618 — device-side position buffer for CUDA Graph capture.
    /// The `_devpos` kernel variants read `pos` from this buffer (1 i32)
    /// at kernel runtime. Updated per-token via `memcpy_htod` BEFORE
    /// `graph.launch()` so the captured graph processes the right position.
    #[cfg_attr(not(feature = "cuda_graphs_forward"), allow(dead_code))]
    pos_dev_buf: CudaSlice<i32>,
    /// Issue 618 — device-side token-id buffer for CUDA Graph capture.
    /// The embedding `_devpos` kernel reads `row_idx` from this buffer.
    #[cfg_attr(not(feature = "cuda_graphs_forward"), allow(dead_code))]
    token_dev_buf: CudaSlice<i32>,
    /// Issue 618 — host-side staging buffer for async memcpy_htod of pos.
    /// cuMemcpyHtoDAsync_v2 is ASYNC — the host pointer must remain valid
    /// until the GPU completes the copy. A local stack variable would be
    /// dropped before the async op reads it. This persistent buffer keeps
    /// the data alive.
    #[cfg_attr(not(feature = "cuda_graphs_forward"), allow(dead_code))]
    pos_host: i32,
    /// Issue 618 — host-side staging buffer for async memcpy_htod of token.
    #[cfg_attr(not(feature = "cuda_graphs_forward"), allow(dead_code))]
    token_host: i32,
    /// Issue 697 — GPU-side argmax over the final logits (packed key|~idx).
    /// Written by `argmax_first_f32` at the end of every forward; read via
    /// `last_argmax()`.
    argmax_buf: CudaSlice<u64>,
    /// Issue 984 T2(a) — pinned (write-combined) transfer buffers for the
    /// shipping decode path's per-token host↔device traffic. `feed_pinned`
    /// `[token, pos]` is the htod source (WC is the canonical upload pattern;
    /// the driver skips its pageable staging pass); `argmax_pinned` is the
    /// 8-byte argmax landing (direct DMA instead of pageable staging; the
    /// single uncached u64 read WC costs is ~100 ns — noise). cudarc's
    /// `alloc_pinned` is WC — the right flavor for BOTH directions here
    /// (the Issue-984 "custom cached-pinned path" note was over-cautious for
    /// an 8-byte read; it matters for vocab-sized downloads, not this).
    #[cfg_attr(not(feature = "cuda_graphs_forward"), allow(dead_code))]
    feed_pinned: PinnedHostSlice<i32>,
    /// Issue 984 T2(a) — the argmax dtoh landing half of the pinned pair.
    #[cfg_attr(not(feature = "cuda_graphs_forward"), allow(dead_code))]
    argmax_pinned: PinnedHostSlice<u64>,
}

/// Plan 603 R2 — the persistent recurrent state's RESIDENT dtype. F32 is
/// the canonical lane (registered pins); `Half` stores 16-bit half bits
/// (f16/bf16 — half the per-token state DRAM traffic, the Issue-734-T5
/// wscale-residency pattern applied to the state). Every accessor matches
/// on the arm, so no code path can silently read a stale representation
/// (compile-time completeness over runtime flags).
enum RecStateBuf {
    F32(CudaSlice<f32>),
    Half(CudaSlice<u16>),
}

impl RecStateBuf {
    /// The half-lane format this forward was constructed with (None = the
    /// canonical f32 lane). Read once from `RIIR_GDN_STATE_HALF`
    /// (`f16` | `bf16`; unset/anything else = off) — the opt-in posture
    /// until the Plan-603 gates pass, the `=0`-style kill-switch after.
    fn half_fmt_from_env() -> Option<HalfStateFmt> {
        static FMT: std::sync::OnceLock<Option<HalfStateFmt>> = std::sync::OnceLock::new();
        *FMT.get_or_init(|| match std::env::var("RIIR_GDN_STATE_HALF").as_deref() {
            Ok("f16") => Some(HalfStateFmt::F16),
            Ok("bf16") => Some(HalfStateFmt::Bf16),
            _ => None,
        })
    }

    fn alloc(
        stream: &Arc<CudaStream>,
        state_dim: usize,
        fmt: Option<HalfStateFmt>,
    ) -> Result<Self, cudarc::driver::DriverError> {
        match fmt {
            None => Ok(Self::F32(stream.alloc_zeros::<f32>(state_dim)?)),
            Some(HalfStateFmt::F16) | Some(HalfStateFmt::Bf16) => {
                Ok(Self::Half(stream.alloc_zeros::<u16>(state_dim)?))
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(s) => s.len(),
            Self::Half(s) => s.len(),
        }
    }

    fn memset_zeros(&mut self, stream: &Arc<CudaStream>) -> Result<(), cudarc::driver::DriverError> {
        match self {
            Self::F32(s) => stream.memset_zeros(s),
            Self::Half(s) => stream.memset_zeros(s),
        }
    }

    /// Same-dtype device-to-device copy (backups/rollback — the pair is
    /// same-armed by construction: both allocated from the same env).
    #[cfg(feature = "speculative_decode")]
    fn copy_dtod(
        stream: &Arc<CudaStream>,
        src: &Self,
        dst: &mut Self,
    ) -> Result<(), cudarc::driver::DriverError> {
        match (src, dst) {
            (Self::F32(s), Self::F32(d)) => stream.memcpy_dtod(s, d),
            (Self::Half(s), Self::Half(d)) => stream.memcpy_dtod(s, d),
            // A cross-arm pair is a construction bug (both lanes read the
            // same env OnceLock) — refuse loudly rather than convert.
            (Self::F32(_), Self::Half(_)) | (Self::Half(_), Self::F32(_)) => {
                unreachable!("RecStateBuf cross-dtype dtod (f32/half lanes mixed)")
            }
        }
    }

    /// The state as f32 bits on the HOST (the hybrid-cache boundary).
    /// `Half` widens via [`f16_bits_to_f32`]/[`bf16_bits_to_f32`] — the exact
    /// IEEE widen the hardware `cvt.f32.f16`/`cvt.f32.bf16` performs (normals
    /// AND subnormals — the gemv_q4k Issue-593 lesson). Hand-rolled bit
    /// math, not the `half` crate: `half` is gated behind `cubecl_runtime`
    /// and this lane must build under bare `default`.
    fn download_to_f32(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<f32>, cudarc::driver::DriverError> {
        match self {
            Self::F32(s) => {
                let mut out = vec![0f32; s.len()];
                stream.memcpy_dtoh(s, &mut out)?;
                Ok(out)
            }
            Self::Half(s) => {
                let mut bits = vec![0u16; s.len()];
                stream.memcpy_dtoh(s, &mut bits)?;
                Ok(match Self::half_fmt_from_env() {
                    Some(HalfStateFmt::Bf16) => {
                        bits.into_iter().map(bf16_bits_to_f32).collect()
                    }
                    _ => bits.into_iter().map(f16_bits_to_f32).collect(),
                })
            }
        }
    }

    /// Host f32 bits → the resident dtype (the upload boundary). `Half`
    /// narrows with the SAME rounding the device `cvt.rn` produces (RNE)
    /// + the kernel's saturating clamp for f16 (±65504).
    fn upload_from_f32(
        &mut self,
        stream: &Arc<CudaStream>,
        src: &[f32],
    ) -> Result<(), cudarc::driver::DriverError> {
        match self {
            Self::F32(d) => stream.memcpy_htod(src, d),
            Self::Half(d) => {
                let bits: Vec<u16> = match Self::half_fmt_from_env() {
                    Some(HalfStateFmt::Bf16) => {
                        src.iter().map(|&x| f32_to_bf16_bits_rn(x)).collect()
                    }
                    _ => src
                        .iter()
                        .map(|&x| f32_to_f16_bits_rn(x.clamp(-65504.0, 65504.0)))
                        .collect(),
                };
                stream.memcpy_htod(&bits, d)
            }
        }
    }
}

// ── Plan 603 R2 — host-side half bit converters (exact IEEE, PTX-matched) ──

/// f16 bits → f32 (exact widen; normals AND subnormals).
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign // ±0
            } else {
                // Subnormal: normalize. f16 subnormal value = frac · 2^-24;
                // shift k so the leading 1 sits at bit 10, then the f32
                // exponent field = 113 - k (= 127 - 24 + (10 - k)).
                let mut k = 0u32;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    k += 1;
                }
                let mant10 = f & 0x3ff; // drop the leading 1
                sign | ((113 - k) << 23) | (mant10 << 13)
            }
        }
        0x1f => {
            if frac == 0 {
                sign | 0x7f80_0000 // ±inf
            } else {
                sign | 0x7f80_0000 | (frac << 13) // NaN (payload preserved)
            }
        }
        e => sign | ((e + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// f32 → f16 bits, round-to-nearest-even (matches PTX `cvt.rn.f16.f32`).
/// The caller clamps to ±65504 (the kernel's saturation); the overflow →
/// inf arm is still exact for direct use.
fn f32_to_f16_bits_rn(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let frac = b & 0x007f_ffff;
    if exp == 0xff {
        // inf / NaN (quiet the NaN)
        return sign | 0x7c00 | u16::from(frac != 0) << 9;
    }
    let unbiased = exp - 127;
    if unbiased >= 16 {
        // Overflow (|x| ≥ 65536-ish rounds past f16 max) → ±inf under RN.
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        // Normal f16 window: round the 23-bit mantissa to 10 bits, RNE.
        let round_bit = 1u32 << 12; // first dropped bit (23 - 10 - 1)
        let rem = frac & (round_bit | (round_bit - 1));
        let mut f = frac >> 13;
        let mut e = (unbiased + 15) as u32;
        if rem > round_bit || (rem == round_bit && (f & 1) == 1) {
            f += 1;
            if f == 0x400 {
                f = 0;
                e += 1;
            }
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
        return sign | ((e << 10) as u16) | f as u16;
    }
    // Subnormal / underflow window: target k = mant · 2^(unbiased+1), RNE.
    let mant = (1u32 << 23) | frac; // 24-bit significand (f32 subnorms: frac-only, far below the window)
    let shift = (-(unbiased + 1)) as u32; // 1..=24 keeps the subnormal window
    if shift > 24 {
        return sign; // strictly below the half-tip → ±0
    }
    let keep = mant >> shift;
    let rem = mant & ((1u32 << shift) - 1);
    let half = 1u32 << (shift - 1);
    let mut k = keep;
    if rem > half || (rem == half && (k & 1) == 1) {
        k += 1; // may carry into 0x400 = the smallest NORMAL — correct
    }
    sign | k as u16
}

/// bf16 bits → f32 (exact widen — bf16 is the top 16 bits of f32).
fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// f32 → bf16 bits, round-to-nearest-even on the truncated bits (matches
/// PTX `cvt.rn.bf16.f32` for finite inputs; NaN quiet bit preserved).
fn f32_to_bf16_bits_rn(x: f32) -> u16 {
    let b = x.to_bits();
    if b & 0x7f80_0000 == 0x7f80_0000 && b & 0x007f_ffff != 0 {
        // NaN in → quiet-NaN bf16 out (payload top bit forced; matches the
        // hardware's NaN-preserving cvt for our finite-input lane).
        return ((b >> 16) as u16) | 0x0040;
    }
    let lsb = (b >> 16) & 1;
    let round_bias = match lsb {
        0 => 0x7fff, // tie → even (down when low half < 0x8000)
        _ => 0x8000, // tie → even (up when low half ≥ 0x8000)
    };
    let rounded = b.wrapping_add(round_bias);
    (rounded >> 16) as u16
}

/// Per-layer persistent recurrent state (DeltaNet) or KV cache (Attention).
struct LayerStateCudarc {
    deltanet_state: Option<RecStateBuf>,
    conv_state: Option<CudaSlice<f32>>,
    // Issue 746 — KV cache contract (APPEND-ONLY + FULL-LENGTH): `key_cache`
    // and `value_cache` are pre-allocated at `with_shared()` for the full
    // `kv_cache_dim = block_size * kvd_attn` (max_seq = config.block_size)
    // and are only ever APPENDED at `pos` — never evicted/trimmed/rotated/
    // capped. `rollback_speculative_gpu` relies on this write-before-read
    // regime for phantom-token immunity (the same one the
    // `LayerStateBackupCudarc` doc above records: decode at position p reads
    // exactly `[0..=p]`, and the append for p overwrites the stale entry
    // before the decode kernel runs — re-execution alone repairs the cache
    // after a rejected speculative verify, no trim needed).
    // Adding ANY evict/trim/rotate/cap API here WITHOUT extending the
    // speculative rollback contract reintroduces the oMLX phantom-token
    // hazard (a rotated ring cannot trim — rejected verify tokens become
    // phantoms; oMLX `mlx_lm_mtp/cache_rollback.py`, DeepSeek-V4-Flash
    // sliding_window=128). See Issue 746.
    key_cache: Option<CudaSlice<f32>>,
    value_cache: Option<CudaSlice<f32>>,
}

/// Issue 717 G3b — max drafts per speculative verify cycle. The rotated
/// logits/argmax pool is pre-allocated at this size (8 × ~1 MB logits +
/// 8 × 8 B argmax at Bonsai vocab — trivial) so the verify path never calls
/// `alloc_zeros` mid-cycle: `cuMemAlloc` can implicitly synchronize the
/// device, which would drain the pipeline between the K verify forwards and
/// break the no-intermediate-sync contract that makes the seam useful.
#[cfg(feature = "speculative_decode")]
const SPEC_MAX_K: usize = 8;

/// Issue 717 G3b — GPU-side backup of one layer's speculative-checkpoint
/// state (DeltaNet layers only; attention KV needs no backup — decode at
/// position `p` reads `[0..=p]` and the append for `p` overwrites the stale
/// entry before the decode kernel runs).
#[cfg(feature = "speculative_decode")]
struct LayerStateBackupCudarc {
    deltanet_state: Option<RecStateBuf>,
    conv_state: Option<CudaSlice<f32>>,
}

/// Issue 717 G3b — speculative-decode seam state: pre-allocated checkpoint
/// backups + the K-buffer logits/argmax rotation pool.
#[cfg(feature = "speculative_decode")]
struct SpecStateCudarc {
    state_backups: Vec<LayerStateBackupCudarc>,
    logits_pool: Vec<CudaSlice<f32>>,
    argmax_pool: Vec<CudaSlice<u64>>,
}

// ───────────────────────────────────────────────────────────────────────────
// Forward struct
// ───────────────────────────────────────────────────────────────────────────

/// GPU-resident forward pass for ternary Qwen3.5 DeltaNet (cudarc path).
///
/// All activations stay on GPU between layers. The forward processes a single
/// decode token and returns the logits. See the [module docs](self) for the
/// full architecture rationale.
pub struct TernaryDeltanetGpuForwardCudarc {
    /// Issue 741 T10 Phase B D4 widening: `pub` so riir-train-gpu's
    /// `TernaryDeltanetBackwardTraining` extension trait can reach the
    /// forward's read-only infrastructure (stream, kernels, weights, config).
    /// Read-only by convention — the backward never mutates infra.
    pub infra: ForwardInfraCudarc,
    acts: ActivationsCudarc,
    /// Issue 618 — lazily-captured CUDA Graph for the forward path.
    /// None until the first `forward_token_graph()` call captures it.
    /// Subsequent calls just launch the graph + sync + dtoh.
    #[cfg(feature = "cuda_graphs_forward")]
    graph: Option<CudaGraph>,
    /// Issue 666/781 — per-layer Q+V LoRA adapters for decode. The L4 BCL4
    /// checkpoint carries ONE adapter per DeltaNet layer, so a single
    /// (adapter, target-layer) pair cannot serve it. Index = layer_idx;
    /// `None` = frozen at that layer. Each slot uploads its 4 weight
    /// matrices + rank scratch at attach time; the shared `lora_norm_x`
    /// side buffer serves every slot (layers run sequentially, so one
    /// buffer is never contended).
    lora_layers: Vec<Option<LoraLayerSlot>>,
    lora_kernels: LoraDecodeKernels,
    /// Side buffer for the LoRA target layer's post-RMSNorm `norm_x`.
    /// Separate from `acts.final_norm_x` to avoid aliasing with the
    /// `forward_token_with_final_hidden` path.
    lora_norm_x: CudaSlice<f32>,
    /// Issue 504 T1 (Plan 370 confirm) — master switch for LoRA application in
    /// ANY forward. Decode defaults to `true` (unchanged behavior — attached
    /// slots apply). The closed-loop TRAINING path flips this off around the
    /// frozen prompt/eval forwards and on for the target window, so an
    /// attached adapter can never leak into a forward that must run frozen
    /// (the prompt feeds the recurrence state the CPU closed-loop path also
    /// builds frozen). When `false`, `build_lora_ctx_enabled` yields `None`
    /// and every forward — decode and training — runs the frozen backbone
    /// regardless of attached slots.
    lora_forward_enabled: bool,
    /// Issue 717 G3b — speculative-decode seam state (checkpoint backups +
    /// the verify rotation pool). See the speculative section in the impl.
    #[cfg(feature = "speculative_decode")]
    spec: SpecStateCudarc,
}

/// Host-side stage timings for one `forward_token_graph` call (µs).
///
/// Issue 696 follow-up — the stage breakdown of the GRAPH decode path (the
/// graph-path analogue of Issue 633's eager-path section breakdown). All
/// timings are host-side `Instant` deltas:
///   - `htod_us` — the two 4-byte memcpy_htod calls (token + pos)
///   - `launch_us` — the cudaGraphLaunch driver call (submission only;
///     async — does NOT include GPU execution)
///   - `sync_us` — stream synchronize (drains the GPU: includes replay
///     execution + everything queued before it)
///   - `dtoh_us` — the vocab-sized logits download
///
/// The caller's own work (argmax, sampling, decode loop) is NOT included —
/// it's the gap between `sum()` and the measured tok/s.
#[cfg(feature = "cuda_graphs_forward")]
#[derive(Clone, Copy, Debug, Default)]
pub struct GraphStageTimings {
    pub htod_us: f64,
    pub launch_us: f64,
    pub sync_us: f64,
    pub dtoh_us: f64,
}

impl TernaryDeltanetGpuForwardCudarc {
    /// Create a GPU-resident cudarc forward pass from loaded ternary weights.
    ///
    /// Pre-uploads ALL weights to GPU (~2-3s for 27B model) and allocates all
    /// persistent activation buffers.
    pub fn new(
        config: &Config,
        weights: &QwenDeltaNetTernaryWeights,
    ) -> Result<Self, CudarcKernelError> {
        let ctx = CudaContext::new(0).map_err(|e| CudarcKernelError::CudaInit(e.to_string()))?;
        let stream = ctx.default_stream();
        Self::with_shared(ctx, stream, config, weights)
    }

    /// Issue 618/696 — graph-capture-ready construction.
    ///
    /// Same as [`Self::new`] but configures the context for CUDA Graph
    /// capture (`forward_token_graph`):
    ///   1. Disables cudarc event tracking — REQUIRED before any allocation,
    ///      otherwise cudarc attaches sync events to every `CudaSlice` which
    ///      break stream capture with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.
    ///   2. Uses a dedicated non-blocking stream instead of the default stream.
    ///
    /// SAFETY contract (same as the Issue 618 GOAT bench): single-stream
    /// usage only — disabling event tracking removes cudarc's cross-stream
    /// use-after-free guard. Every forward on the returned instance (eager
    /// `forward_token` included) runs on this one stream; eager forwards on
    /// this instance were verified argmax-identical to the `Self::new` path
    /// in the Issue 618 sanity block (Bench 667).
    pub fn new_graph_ready(
        config: &Config,
        weights: &QwenDeltaNetTernaryWeights,
    ) -> Result<Self, CudarcKernelError> {
        // Issue 980 T4.5 — the CUDA-graph (devpos) lane now carries the
        // rotation launcher twins (in-graph embedding inverse, the attention
        // devpos twin, FFN + final-tail branches in forward_from_x_devpos);
        // folded models capture + replay correctly. The eager path (Self::new)
        // remains the reference lane.
        let ctx = CudaContext::new(0).map_err(|e| CudarcKernelError::CudaInit(e.to_string()))?;
        // SAFETY: single-stream usage (see doc above); cross-stream sync not needed.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx
            .new_stream()
            .map_err(|e| CudarcKernelError::CudaInit(e.to_string()))?;
        Self::with_shared(ctx, stream, config, weights)
    }

    /// Build the forward using a caller-provided context + stream.
    pub fn with_shared(
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        config: &Config,
        weights: &QwenDeltaNetTernaryWeights,
    ) -> Result<Self, CudarcKernelError> {
        // Issue 980 T4 — Bonsai-2 Hadamard rotation. The eager path serves
        // folded models (rotation kernels built below); the CUDA-graph lane
        // still REFUSES them (the devpos launcher twins are not wired —
        // enforced in `new_graph_ready` and `invalidate_graph`'s capture
        // paths). Dense escape-set a/b upload below (gemv_dense_f32).
        let rotation = match &weights.rotation {
            Some(cfg) => Some(crate::deltanet_rotation_cudarc::RotationTables::build(
                &ctx, &stream, cfg,
            )?),
            None => None,
        };
        // ── Compile all kernel sets against the shared context ──
        let elementwise = ElementwiseKernels::new(ctx.clone())?;
        // Issue 734 T5 — multi-block rmsnorm scratch, allocated on the forward's
        // stream BEFORE any launch/graph capture (addresses bake into graphs).
        // 512 leaf slots cover dim ≤ 8192 (block_size = next_pow2(dim/16)); the
        // launcher falls back to the 1-block kernel for larger dims.
        elementwise.init_rmsnorm_mb_scratch(&stream, 512)?;
        let attention = AttentionKernels::new(ctx.clone())?;
        let deltanet = DeltanetKernels::new(ctx.clone())?;
        let embedding = EmbeddingDequantKernels::new(ctx.clone())?;

        // Compile the dp4a GEMV kernel against the shared context.
        let gemv_ptx = cudarc::nvrtc::compile_ptx_with_opts(
            GEMV_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let gemv_module = ctx
            .load_module(gemv_ptx)
            .map_err(|e| CudarcKernelError::Compile(e.to_string()))?;
        let gemv = gemv_module
            .load_function("gemv_ternary_dp4a")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        // Issue 697 / Issue 705 — multi-segment dp4a GEMV (same module). The
        // persistent grid-stride variant replaces the wave-quantized launch:
        // grid = min(ceil(rows/8), SMs × blocks_per_sm).
        let gemv_multi_persistent = gemv_module
            .load_function("gemv_ternary_dp4a_multi_persistent")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        // Issue 705 — full-residency grid, queried ONCE here (never on the
        // launch path). The occupancy call needs the context current on this
        // thread; `attribute` is a pure device query and needs no binding.
        ctx.bind_to_thread()
            .map_err(|e| CudarcKernelError::CudaInit(format!("bind ctx: {e}")))?;
        let sm_count = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| CudarcKernelError::CudaInit(format!("SM count: {e}")))? as u32;
        let blocks_per_sm = gemv_multi_persistent
            .occupancy_max_active_blocks_per_multiprocessor(WG_THREADS, 0, None)
            .map_err(|e| {
                CudarcKernelError::CudaInit(format!("gemv occupancy: {e}"))
            })?
            .max(1);
        // Issue 705 (filed as 702, renumbered) — grid cap, resolved ONCE here (never on the launch path).
        //
        // Measured verdict (Bench 684): the occupancy-derived full-residency
        // cap is a 1-2% LOSS — uncapped one-warp-per-row ships as the default.
        // `RIIR_GEMV_PERSISTENT_GRID` overrides the cap for tuning/repro:
        //   unset | "auto"  -> u32::MAX (uncapped, measured best, the default)
        //   <number>         -> that cap (768 = occupancy grid on the 4090)
        let occupancy_grid = sm_count * blocks_per_sm;
        let grid = match std::env::var("RIIR_GEMV_PERSISTENT_GRID") {
            Ok(v) if v.eq_ignore_ascii_case("auto") || v.is_empty() => u32::MAX,
            Ok(v) => v.parse::<u32>().unwrap_or(u32::MAX).max(1),
            Err(_) => u32::MAX,
        };
        // Plan 604 T1 (Issue 987 G1) — the two-rows-per-warp variant, loaded
        // from the SAME module (one NVRTC compile). Env-gated A/B knob:
        //   unset | "0" | "off" | "false" -> disabled (default, the pre-604 path)
        //   "1" | "on" | "true"          -> underfilled launches dispatch to r2
        let r2_enabled = match std::env::var("RIIR_GEMV_R2") {
            Ok(v) => matches!(v.to_ascii_lowercase().as_str(), "1" | "on" | "true"),
            Err(_) => false,
        };
        let pf_enabled = match std::env::var("RIIR_GEMV_PF") {
            Ok(v) => matches!(v.to_ascii_lowercase().as_str(), "1" | "on" | "true"),
            Err(_) => false,
        };
        let u4_enabled = match std::env::var("RIIR_GEMV_U4") {
            Ok(v) => matches!(v.to_ascii_lowercase().as_str(), "1" | "on" | "true"),
            Err(_) => false,
        };
        let r2_max_rows = occupancy_grid as usize * (WG_THREADS as usize / 32);
        let gemv_multi_r2_fn = gemv_module
            .load_function("gemv_ternary_dp4a_multi_r2")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let gemv_multi_pf_fn = gemv_module
            .load_function("gemv_ternary_dp4a_multi_pf")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let gemv_multi_u4_fn = gemv_module
            .load_function("gemv_ternary_dp4a_multi_u4")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let gemv_multi = GemvMultiPersistent {
            function: gemv_multi_persistent,
            grid,
            function_r2: gemv_multi_r2_fn,
            r2_enabled,
            function_pf: gemv_multi_pf_fn,
            pf_enabled,
            function_u4: gemv_multi_u4_fn,
            u4_enabled,
            r2_max_rows,
        };
        // Construction-time diagnostic (once per handler) — makes the grid
        // decision observable without a profiler.
        eprintln!(
            "[issue702] gemv grid cap: SMs={sm_count} blocks/SM={blocks_per_sm} occupancy_grid={occupancy_grid} -> cap={} ({})",
            gemv_multi.grid,
            if gemv_multi.grid == u32::MAX { "uncapped default" } else { "env override" }
        );
        eprintln!(
            "[plan604] gemv r2: enabled={r2_enabled} pf: enabled={pf_enabled} u4: enabled={u4_enabled} underfilled_max_rows={r2_max_rows}{}{}{}",
            if r2_enabled { " (RIIR_GEMV_R2 set)" } else { "" },
            if pf_enabled { " (RIIR_GEMV_PF set)" } else { "" },
            if u4_enabled { " (RIIR_GEMV_U4 set)" } else { "" }
        );
        // Issue 616 T4 — fused quantize+dp4a kernel (same module, second entry).
        let gemv_fused = gemv_module
            .load_function("gemv_ternary_dp4a_fused")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        // Issue 641 T7.5 — transposed ternary GEMV for backward (same module).
        let gemv_transposed = gemv_module
            .load_function("gemv_ternary_transposed_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;

        // ── Resolve dimensions ──
        let n = config.n_embd;
        let n_v_heads = config.deltanet_linear_n_value_heads;
        let head_dim = config.deltanet_linear_head_dim;
        let n_k_heads = config.deltanet_linear_n_heads;
        let q_dim = n_k_heads * head_dim;
        let v_dim = n_v_heads * head_dim;
        let qkv_dim = 2 * q_dim + v_dim;
        let z_dim = v_dim;
        let conv_dim = qkv_dim;
        let kernel_size = config.deltanet_conv_kernel_size;
        let mlp = config.mlp_hidden;
        let vocab = config.vocab_size;

        let attn_n_head = config.n_head;
        let attn_n_kv = config.n_kv_head;
        let attn_hd = config.head_dim;
        let q_dim_attn = attn_n_head * attn_hd;
        let kvd_attn = attn_n_kv * attn_hd;

        let layer_types = if weights.layer_types.is_empty() {
            vec![DeltaNetLayerType::Attention; config.n_layer]
        } else {
            weights.layer_types.clone()
        };

        // ── Upload per-layer weights ──
        let layers: Vec<GpuLayerWeightsCudarc> = weights
            .layers
            .iter()
            .map(|l| upload_layer_weights_cudarc(&stream, l))
            .collect();

        // ── Upload global weights ──
        let final_norm = upload_f32_slice(&stream, &weights.final_norm);
        let lm_head = WeightBuffersCudarc::upload(&stream, &weights.lm_head);

        // Embedding table bit-planes.
        let wte_pos_u32 = crate::gemv_ternary_cubecl::cast_u64_to_u32(&weights.wte.pos_bits);
        let wte_neg_u32 = crate::gemv_ternary_cubecl::cast_u64_to_u32(&weights.wte.neg_bits);
        let wte_scale_f32 =
            crate::gemv_ternary_cubecl::prepare_group_scale_f32(&weights.wte.group_scale);
        let wte_blocks64 = weights.wte.blocks64;
        let wte_groups_per_row = weights.wte.groups_per_row;
        let wte_pos_bits = stream
            .clone_htod(&wte_pos_u32)
            .map_err(|e| CudarcKernelError::CudaInit(e.to_string()))?;
        let wte_neg_bits = stream
            .clone_htod(&wte_neg_u32)
            .map_err(|e| CudarcKernelError::CudaInit(e.to_string()))?;
        let wte_scale = stream
            .clone_htod(&wte_scale_f32)
            .map_err(|e| CudarcKernelError::CudaInit(e.to_string()))?;

        // ── Allocate persistent activation buffers ──
        let alloc = |len: usize| -> Result<CudaSlice<f32>, CudarcKernelError> {
            stream.alloc_zeros::<f32>(len).map_err(alloc_err)
        };

        let x = alloc(n)?;
        let tmp = alloc(n)?;
        let ffn_out = alloc(n)?;
        let qkv = alloc(qkv_dim)?;
        let qkv_expanded = alloc(3 * n_v_heads * head_dim)?;
        let z_buf = alloc(z_dim)?;
        let a_raw = alloc(n_v_heads)?;
        let b_raw = alloc(n_v_heads)?;
        let beta_buf = alloc(n_v_heads)?;
        let decay_buf = alloc(n_v_heads)?;
        let recurrent_out = alloc(n_v_heads * head_dim)?;
        let ffn_gate = alloc(mlp)?;
        let ffn_up = alloc(mlp)?;
        let logits = alloc(vocab)?;
        // Issue 634 — f32 side buffer for the final RMSNorm output, written by
        // `rmsnorm_quantize_with_norm_x_f32`. 8 KiB at n_embd=5120. Unused on the
        // hot path; only `forward_token_with_final_hidden` reads it.
        let final_norm_x = alloc(n)?;
        // Issue 980 T4 — rotation-path scratch: the PRIMAL normed input, the
        // sign+FWHT copy the folded projections quantize from, the f32 SwiGLU
        // hidden, and the head-permute gather source. ~180 KB total at
        // Bonsai-2 shapes; allocated unconditionally so the struct stays
        // non-generic over the rotation state.
        let norm_x = alloc(n)?;
        let rot_scratch = alloc(n)?;
        let ffn_hidden = alloc(mlp)?;
        let permute_tmp = alloc(n_v_heads * head_dim)?;
        let attn_qg = alloc(2 * q_dim_attn)?;
        let attn_q = alloc(q_dim_attn)?;
        let attn_gate = alloc(q_dim_attn)?;
        let attn_k = alloc(kvd_attn)?;
        let attn_v = alloc(kvd_attn)?;
        let attn_out = alloc(q_dim_attn)?;

        // Quantize scratch — sized to the largest activation dim.
        let max_n = n.max(mlp).max(q_dim_attn).max(2 * q_dim_attn).max(kvd_attn);
        let max_ablocks = max_n.div_ceil(16);
        let quant_i8_buf = stream.alloc_zeros::<i8>(max_n).map_err(alloc_err)?;
        let ascale_buf = alloc(max_ablocks)?;

        // Issue 618 — device-side pos/token buffers for CUDA Graph capture.
        // 1 i32 each. Updated per-token via memcpy_htod before graph.launch().
        let pos_dev_buf = stream.alloc_zeros::<i32>(1).map_err(alloc_err)?;
        let token_dev_buf = stream.alloc_zeros::<i32>(1).map_err(alloc_err)?;
        // Issue 697 — GPU-side argmax result (packed ordered-float key << 32 |
        // ~index). Zeroed by a capturable stream memset at the end of every
        // forward; read back via `last_argmax()` (8 bytes instead of a
        // vocab-sized logits download).
        let argmax_buf = stream.alloc_zeros::<u64>(1).map_err(alloc_err)?;
        // Issue 984 T2(a) — pinned transfer pair (see the field docs). The
        // context owns the allocation lifetime; `PinnedHostSlice` self-frees.
        let mut feed_pinned = unsafe { ctx.alloc_pinned::<i32>(2) }.map_err(alloc_err)?;
        let mut argmax_pinned = unsafe { ctx.alloc_pinned::<u64>(1) }.map_err(alloc_err)?;
        // Initialize both to valid zero bits (alloc_pinned leaves them unset).
        {
            let s = feed_pinned.as_mut_slice().map_err(alloc_err)?;
            s.fill(0);
        }
        {
            let s = argmax_pinned.as_mut_slice().map_err(alloc_err)?;
            s.fill(0);
        }

        // ── Allocate per-layer persistent state ──
        let state_dim = n_v_heads * head_dim * head_dim;
        let conv_state_dim = conv_dim * kernel_size;
        let max_seq = config.block_size;
        // Issue 746: full-length by contract — append-only KV; see the
        // contract note on `LayerStateCudarc::key_cache`.
        let kv_cache_dim = max_seq * kvd_attn;
        // Plan 603 R2 — the recurrent state's resident dtype (f32 canonical;
        // half under RIIR_GDN_STATE_HALF). head_dim 128 only for the half
        // lane (the kernel is hd128-specialized, the Issue-706 lesson);
        // anything else stays on f32 regardless of the env.
        let state_half_fmt = if head_dim == 128 {
            RecStateBuf::half_fmt_from_env()
        } else {
            None
        };

        let mut layer_states = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let lt = layer_types[i];
            let (deltanet_state, conv_state) = if lt == DeltaNetLayerType::DeltaNet {
                (
                    Some(
                        RecStateBuf::alloc(&stream, state_dim, state_half_fmt)
                            .map_err(alloc_err)?,
                    ),
                    Some(stream.alloc_zeros::<f32>(conv_state_dim).map_err(alloc_err)?),
                )
            } else {
                (None, None)
            };
            let (key_cache, value_cache) = if lt == DeltaNetLayerType::Attention {
                (
                    Some(stream.alloc_zeros::<f32>(kv_cache_dim).map_err(alloc_err)?),
                    Some(stream.alloc_zeros::<f32>(kv_cache_dim).map_err(alloc_err)?),
                )
            } else {
                (None, None)
            };
            layer_states.push(LayerStateCudarc {
                deltanet_state,
                conv_state,
                key_cache,
                value_cache,
            });
        }

        // ── Issue 717 G3b — speculative-decode seam buffers ──
        // Pre-allocated at construction so the verify/checkpoint paths never
        // allocate mid-cycle (cuMemAlloc can implicitly synchronize the
        // device — the exact thing the no-intermediate-sync contract forbids).
        #[cfg(feature = "speculative_decode")]
        let spec = {
            let mut state_backups = Vec::with_capacity(config.n_layer);
            for i in 0..config.n_layer {
                let (ds, cs) = if layer_types[i] == DeltaNetLayerType::DeltaNet {
                    (
                        Some(
                            RecStateBuf::alloc(&stream, state_dim, state_half_fmt)
                                .map_err(alloc_err)?,
                        ),
                        Some(
                            stream
                                .alloc_zeros::<f32>(conv_state_dim)
                                .map_err(alloc_err)?,
                        ),
                    )
                } else {
                    (None, None)
                };
                state_backups.push(LayerStateBackupCudarc {
                    deltanet_state: ds,
                    conv_state: cs,
                });
            }
            let logits_pool = (0..SPEC_MAX_K)
                .map(|_| {
                    stream
                        .alloc_zeros::<f32>(config.vocab_size)
                        .map_err(alloc_err)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let argmax_pool = (0..SPEC_MAX_K)
                .map(|_| stream.alloc_zeros::<u64>(1).map_err(alloc_err))
                .collect::<Result<Vec<_>, _>>()?;
            SpecStateCudarc {
                state_backups,
                logits_pool,
                argmax_pool,
            }
        };

        let infra = ForwardInfraCudarc {
            ctx,
            stream,
            elementwise,
            attention,
            deltanet,
            embedding,
            gemv,
            gemv_multi,
            gemv_fused,
            gemv_transposed,
                        _gemv_module: gemv_module,
            config: config.clone(),
            layer_types,
            layers,
            final_norm,
            lm_head,
            wte_pos_bits,
            wte_neg_bits,
            wte_scale,
            wte_blocks64,
            wte_groups_per_row,
            rotation,
        };
        let acts = ActivationsCudarc {
            x,
            tmp,
            ffn_out,
            qkv,
            qkv_expanded,
            z_buf,
            a_raw,
            b_raw,
            beta_buf,
            decay_buf,
            recurrent_out,
            ffn_gate,
            ffn_up,
            logits,
            attn_qg,
            attn_q,
            attn_gate,
            attn_k,
            attn_v,
            attn_out,
            quant_i8_buf,
            ascale_buf,
            final_norm_x,
            norm_x,
            rot_scratch,
            ffn_hidden,
            permute_tmp,
            layer_states,
            pos: 0,
            pos_dev_buf,
            token_dev_buf,
            pos_host: 0,
            token_host: 0,
            argmax_buf,
            feed_pinned,
            argmax_pinned,
        };
        // Issue 666 — LoRA decode kernel infrastructure. Compiled once at
        // construction (~1ms nvrtc); slots start empty and are populated via
        // `set_lora` (single) / `attach_lora_layer` (per-layer, Issue 781).
        let lora_kernels = LoraDecodeKernels::new(infra.ctx.clone())?;
        let lora_norm_x = infra
            .stream
            .alloc_zeros::<f32>(infra.config.n_embd)
            .map_err(alloc_err)?;
        let lora_layers: Vec<Option<LoraLayerSlot>> =
            (0..infra.config.n_layer).map(|_| None).collect();
        Ok(Self {
            infra,
            acts,
            #[cfg(feature = "cuda_graphs_forward")]
            graph: None,
            lora_layers,
            lora_kernels,
            lora_norm_x,
            lora_forward_enabled: true,
            #[cfg(feature = "speculative_decode")]
            spec,
        })
    }

    /// Access the shared CUDA context.
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.infra.ctx
    }

    /// Access the shared CUDA stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.infra.stream
    }

    /// Issue 666 — Attach a Q+V LoRA adapter for decode at `layer_idx`.
    ///
    /// Single-adapter overwrite semantics (detach everything, then attach this
    /// one) so existing single-layer callers are unchanged. Multi-adapter
    /// callers (the L4 BCL4 checkpoint, one adapter per DeltaNet layer) use
    /// [`Self::attach_lora_layer`].
    ///
    /// The adapter is a **decode-time** correction — it does NOT affect the
    /// training forward (`forward_token_training`). Pass the same `QvLora`
    /// that was trained (e.g. loaded from a Plan 334 / Issue 641 T6 checkpoint).
    ///
    /// # Errors
    ///
    /// Returns [`CudarcKernelError::InvalidArg`] when `layer_idx` is out of
    /// range or the target layer is not a DeltaNet layer (LoRA on attention
    /// layers is not supported — the Q+V projections are DeltaNet-specific).
    /// Issue 722 H2: these checks were `debug_assert!`-only, so a release
    /// caller paid adapter upload + rank-scratch alloc + graph invalidation
    /// for a LoRA that could never fire (`apply_lora` runs only inside
    /// DeltaNet layers) — wrong-model output with zero diagnostic.
    pub fn set_lora(
        &mut self,
        layer_idx: usize,
        lora: &riir_infer_core::deltanet::qv_lora::QvLora,
    ) -> Result<(), CudarcKernelError> {
        self.validate_lora_target(layer_idx)?;
        self.clear_lora();
        self.attach_lora_layer(layer_idx, lora)
    }

    /// Issue 781 — attach ANOTHER per-layer Q+V LoRA adapter without
    /// disturbing previously attached layers.
    ///
    /// The L4 fixer's BCL4 checkpoint adapts every DeltaNet layer, so the
    /// serving path attaches all of them BEFORE the first decode (and before
    /// any graph capture — each attach invalidates a captured graph, since the
    /// graph bakes in whether the LoRA kernels run AND their buffer
    /// addresses).
    ///
    /// Same target validation and decode-time-only contract as
    /// [`Self::set_lora`]. Attaching over an occupied slot drops the old
    /// adapter (freeing its device buffers — which is why the graph is
    /// invalidated here too).
    pub fn attach_lora_layer(
        &mut self,
        layer_idx: usize,
        lora: &riir_infer_core::deltanet::qv_lora::QvLora,
    ) -> Result<(), CudarcKernelError> {
        self.validate_lora_target(layer_idx)?;
        let rank = lora.rank;
        // Upload the weights BEFORE allocating scratch so a failed upload
        // leaves no partial slot behind.
        let gpu = QvLoraGpuCudarc::from_qv_lora(&self.infra.stream, lora)?;
        let slot = LoraLayerSlot {
            lora: gpu,
            ax_q: self
                .infra
                .stream
                .alloc_zeros::<f32>(rank)
                .map_err(alloc_err)?,
            ax_v: self
                .infra
                .stream
                .alloc_zeros::<f32>(rank)
                .map_err(alloc_err)?,
        };
        self.lora_layers[layer_idx] = Some(slot);
        // Issue 696 — a captured graph bakes in whether the LoRA kernels run
        // AND their weight/scratch buffer addresses. Dropping an occupied slot
        // frees device buffers; a fresh slot's buffers are new allocations
        // either way — the previously captured graph is stale. Drop it so the
        // next forward_token_graph re-captures WITH the current adapter set.
        #[cfg(feature = "cuda_graphs_forward")]
        {
            self.graph = None;
        }
        Ok(())
    }

    /// Issue 722 H2 validation, shared by [`Self::set_lora`] and
    /// [`Self::attach_lora_layer`].
    fn validate_lora_target(&self, layer_idx: usize) -> Result<(), CudarcKernelError> {
        if layer_idx >= self.infra.config.n_layer {
            return Err(CudarcKernelError::InvalidArg(format!(
                "set_lora: layer_idx {layer_idx} >= n_layer {}",
                self.infra.config.n_layer
            )));
        }
        if self.infra.layer_types.get(layer_idx) != Some(&DeltaNetLayerType::DeltaNet) {
            return Err(CudarcKernelError::InvalidArg(format!(
                "set_lora: target layer {layer_idx} is not a DeltaNet layer \
                 (LoRA on attention layers is not supported — the Q+V \
                 projections are DeltaNet-specific)"
            )));
        }
        Ok(())
    }

    /// Issue 504 T1 (Plan 370 confirm) — closed-loop training weight refresh:
    /// update the adapter at `layer_idx` to `lora`'s CURRENT weights, without
    /// dropping the slot (in-place `memcpy_htod` into the existing device
    /// buffers — no realloc, no graph invalidation, ~650 KB PCIe per layer).
    /// Attaches a fresh slot when none exists yet (first training step).
    ///
    /// This is what makes [`forward_token_training`] adapter-aware: the
    /// training driver calls it for every target layer right before the
    /// target window, while `lora_forward_enabled` gates WHEN any forward
    /// applies the adapters (off during the frozen prompt/eval windows).
    ///
    /// # Errors
    ///
    /// Propagates [`Self::validate_lora_target`] and
    /// `QvLoraGpuCudarc::update_from` (shape-mismatch → `InvalidArg`).
    pub fn update_lora_layer_weights(
        &mut self,
        layer_idx: usize,
        lora: &riir_infer_core::deltanet::qv_lora::QvLora,
    ) -> Result<(), CudarcKernelError> {
        self.validate_lora_target(layer_idx)?;
        match self.lora_layers[layer_idx].as_mut() {
            Some(slot) => slot.lora.update_from(&self.infra.stream, lora),
            None => self.attach_lora_layer(layer_idx, lora),
        }
    }

    /// Issue 666 — Detach ALL LoRA adapters. Subsequent forwards run the
    /// frozen backbone only (no LoRA correction).
    pub fn clear_lora(&mut self) {
        for slot in self.lora_layers.iter_mut() {
            *slot = None;
        }
        // Issue 696 — dropping the adapters also frees their device buffers; a
        // captured graph referencing them would replay against freed pointers.
        #[cfg(feature = "cuda_graphs_forward")]
        {
            self.graph = None;
        }
    }

    /// Issue 666 — Returns true if any LoRA adapter is currently attached.
    pub fn has_lora(&self) -> bool {
        self.lora_layers.iter().any(Option::is_some)
    }

    /// Issue 504 T1 — master switch for LoRA application in every forward
    /// (decode AND training). Default `true` (decode behavior unchanged).
    /// The closed-loop training driver turns this OFF around frozen forwards
    /// (prompt/eval) and ON for the target window; attached slots then apply
    /// only where the training math wants them.
    pub fn set_lora_forward_enabled(&mut self, enabled: bool) {
        // A captured decode graph bakes whether the LoRA kernels run (Issue
        // 696); flipping the switch changes that predicate, so drop the graph.
        #[cfg(feature = "cuda_graphs_forward")]
        if self.lora_forward_enabled != enabled {
            self.graph = None;
        }
        self.lora_forward_enabled = enabled;
    }

    /// Issue 504 T1 — whether forwards currently apply attached adapters.
    pub fn lora_forward_enabled(&self) -> bool {
        self.lora_forward_enabled
    }

    /// Issue 781 — number of attached adapters (diagnostic for multi-layer
    /// attach sites: the BCL4 serving path asserts it matches the adapter
    /// set it built).
    pub fn attached_lora_layers(&self) -> usize {
        self.lora_layers.iter().filter(|s| s.is_some()).count()
    }

    /// Set the input hidden state from a token ID via GPU-side dequant.
    ///
    /// Zero CPU allocation, zero GPU allocation per token.
    pub fn set_input_token(&mut self, token: usize) -> Result<(), CudarcKernelError> {
        let n = self.infra.config.n_embd;
        self.infra.embedding.launch_dequant_row(
            &self.infra.stream,
            &self.infra.wte_pos_bits,
            &self.infra.wte_neg_bits,
            &self.infra.wte_scale,
            &self.acts.x,
            token,
            self.infra.wte_blocks64,
            self.infra.wte_groups_per_row,
            n,
        )?;
        // Issue 980 T4 — a Hadamard-latent embedding table stores rotated
        // rows; restore the primal basis right after the lookup (Hadamard
        // first, sign second — the CPU `rotate_inverse_inplace` twin).
        if let Some(rot) = &self.infra.rotation
            && rot.inverse_embedding
        {
            let signs = rot.signs_for_width(n);
            rot.kernels.fwht_rotate_inverse(
                &self.infra.stream,
                &self.acts.x,
                signs,
                n,
                rot.block_size,
            )?;
        }
        Ok(())
    }

    /// Forward pass for a single decode token.
    ///
    /// Assumes `set_input_token` was called. Returns the logits (downloaded
    /// once at the end — the only CPU↔GPU sync point per token).
    pub fn forward_token(&mut self) -> Result<Vec<f32>, CudarcKernelError> {
        // Issue 666 — split disjoint borrows: infra (shared), acts (mutable),
        // and the LoRA fields (shared + mutable scratch) are disjoint struct
        // fields, so Rust allows borrowing them simultaneously.
        let infra = &self.infra;
        let acts = &mut self.acts;
        let lora_kernels = &self.lora_kernels;
        let lora_norm_x = &self.lora_norm_x;
        let mut lora_ctx = build_lora_ctx_enabled(
            lora_kernels,
            self.lora_layers.as_mut_slice(),
            lora_norm_x,
            self.lora_forward_enabled,
        );
        forward_from_x_with_lora(infra, acts, lora_ctx.as_mut())?;

        // Sync + download logits.
        infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let mut logits = vec![0.0f32; infra.config.vocab_size];
        infra
            .stream
            .memcpy_dtoh(&acts.logits, &mut logits)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        Ok(logits)
    }

    /// Issue 634 — Forward pass that returns both the logits AND the final
    /// normed hidden state `norm_x` (the input to the lm_head projection).
    ///
    /// Mirrors [`TernaryDeltanetGpuForward::forward_token_with_final_hidden`]
    /// on the CubeCL path. Used by the lm_head LoRA training pipeline
    /// (Plan 334), which needs `norm_x` to precompute the per-token
    /// `(base_logits, norm_x, target)` cache without re-running the backbone
    /// per gradient step.
    ///
    /// The per-layer forward (layers 0..n_layer) is identical to
    /// [`forward_token`](Self::forward_token) — same kernel sequence, same
    /// fusion choices (Issue 624 pre-attn norm fusion, Issue 623 final-norm
    /// fusion, Issue 625 SwiGLU fusion, Issue 626 gate fusion). The ONLY
    /// difference is at the final RMSNorm+lm_head dispatch: this method uses
    /// `rmsnorm_quantize_and_gemv_batch_with_norm_x`, which is the same as the
    /// hot-path `rmsnorm_quantize_and_gemv_batch` PLUS a side-buffer write of
    /// the f32 `norm_x`. Same launch count, one extra elementwise store per
    /// active thread.
    ///
    /// Returns `(logits, final_norm_x)` where `final_norm_x.len() == n_embd`.
    /// One stream sync + two dtoh transfers at the end (still the only CPU↔GPU
    /// sync point per token).
    pub fn forward_token_with_final_hidden(
        &mut self,
    ) -> Result<(Vec<f32>, Vec<f32>), CudarcKernelError> {
        guard_no_rotation(&self.infra, "forward_token_with_final_hidden")?;
        let n_embd = self.infra.config.n_embd;
        let eps = self.infra.config.rms_norm_eps as f32;

        // ── Run layers 0..n_layer-1, stopping before the final norm+lm_head. ──
        // We re-run `forward_from_x`'s layer loop inline because the final
        // RMSNorm+lm_head dispatch needs the sided variant. The layer loop
        // body is identical to `forward_from_x` (same fusion choices).
        {
            let infra = &self.infra;
            let acts = &mut self.acts;
            // Issue 666/781 — build LoRA ctx from disjoint struct fields.
            let lora_kernels = &self.lora_kernels;
            let lora_norm_x = &self.lora_norm_x;
            let mut lora_ctx = build_lora_ctx_enabled(
                lora_kernels,
                self.lora_layers.as_mut_slice(),
                lora_norm_x,
                self.lora_forward_enabled,
            );
            forward_layers_with_lora(infra, acts, lora_ctx.as_mut())?;

            // ── Final RMSNorm + lm_head (fused, with norm_x side write) ──
            // Issue 634 — same as the hot path's `rmsnorm_quantize_and_gemv_batch`
            // call, but with the extra `norm_x_out` write so the LoRA precompute
            // can read the final normed hidden state.
            rmsnorm_quantize_and_gemv_batch_with_norm_x(
                &infra.elementwise,
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &acts.x,
                &infra.final_norm,
                &acts.final_norm_x,
                eps,
                n_embd,
                &[(&infra.lm_head, &acts.logits)],
            )?;

            // Issue 722 H4 — same argmax tail the graph path captures and
            // `forward_from_x_with_lora` runs eagerly ("same sequence as the
            // graph path so `last_argmax()` is valid after every forward").
            // The `with_norm_x` tail previously skipped it, so `last_argmax()`
            // returned the PREVIOUS token's argmax after this variant.
            infra
                .stream
                .memset_zeros(&mut acts.argmax_buf)
                .map_err(alloc_err)?;
            infra.elementwise.launch_argmax_first(
                &infra.stream,
                &acts.logits,
                infra.config.vocab_size,
                &acts.argmax_buf,
            )?;

            acts.pos += 1;
        }

        // Sync + download logits AND final_norm_x.
        self.infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let mut logits = vec![0.0f32; self.infra.config.vocab_size];
        let mut norm_x = vec![0.0f32; n_embd];
        self.infra
            .stream
            .memcpy_dtoh(&self.acts.logits, &mut logits)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        self.infra
            .stream
            .memcpy_dtoh(&self.acts.final_norm_x, &mut norm_x)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        Ok((logits, norm_x))
    }

    /// **Per-layer residual-stream capture** (Issue 717 G2-full — the DSpark
    /// drafter-feature extraction hook).
    ///
    /// Mirrors [`TernaryDeltanetGpuForward::forward_token_with_layer_capture`]
    /// on the CubeCL path: after each layer listed in `capture_layers` completes,
    /// a copy of the post-layer residual stream (`acts.x[0..n_embd]` — the same
    /// tensor the PrismML fork taps as `l_out`, i.e. the INPUT of the next
    /// layer) is downloaded into `capture_out[i][..n_embd]` (indexed by position
    /// in `capture_layers`, which must be strictly ascending).
    ///
    /// The layer loop is re-run inline (same kernel sequence as
    /// [`forward_token`](Self::forward_token) — same fusion choices) because the
    /// capture points sit BETWEEN layers. Each capture adds a stream sync + a
    /// 20 KB dtoh; with the 5 DSpark target layers that is 5 syncs/token —
    /// diagnostic-only cost, never use on a hot path.
    ///
    /// **Issue 749 T5 verdict — do NOT re-file "port the all-positions capture
    /// here".** This tap is decode-shaped and already minimal: `acts.x` is the
    /// single-token residual `[n_embd]` and the dtoh copies exactly the `n`
    /// floats it keeps — there is no read-everything-keep-one-row waste to
    /// remove (that pattern was the *prefill* tap's, fixed on the CubeCL path
    /// by `prefill_with_all_positions_capture`). Capturing P positions here is
    /// P decode forwards = O(P) by construction. This file has no prefill to
    /// attach an all-positions capture to; batching would be new kernel work.
    /// Consumers wanting per-layer states at every position (e.g. the Maglev
    /// teacher, riir-train Plan 343) use the CubeCL prefill path — which on the
    /// 4090 runs through wgpu/Vulkan, not this raw-CUDA file.
    ///
    /// Returns the logits `[vocab_size]` (downloaded once at the end).
    pub fn forward_token_with_layer_capture(
        &mut self,
        capture_layers: &[usize],
        capture_out: &mut [Vec<f32>],
    ) -> Result<Vec<f32>, CudarcKernelError> {
        guard_no_rotation(&self.infra, "forward_token_with_layer_capture")?;
        let n = self.infra.config.n_embd;
        let eps = self.infra.config.rms_norm_eps as f32;
        assert_eq!(
            capture_layers.len(),
            capture_out.len(),
            "capture_out must have one buffer per capture layer"
        );
        for w in capture_out.iter() {
            assert!(w.len() >= n, "capture buffers must be >= n_embd");
        }
        for w in capture_layers.windows(2) {
            assert!(w[0] < w[1], "capture_layers must be strictly ascending");
        }

        let mut cap_i = 0usize;
        {
            let infra = &self.infra;
            let acts = &mut self.acts;
            let lora_kernels = &self.lora_kernels;
            let lora_norm_x = &self.lora_norm_x;
            let mut lora_ctx = build_lora_ctx_enabled(
                lora_kernels,
                self.lora_layers.as_mut_slice(),
                lora_norm_x,
                self.lora_forward_enabled,
            );

            debug_assert!(
                acts.pos < infra.config.block_size,
                "position {} exceeds KV cache block_size {}",
                acts.pos,
                infra.config.block_size,
            );

            for layer_idx in 0..infra.config.n_layer {
                let layer_w = &infra.layers[layer_idx];
                let is_deltanet =
                    infra.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;
                if is_deltanet {
                    forward_deltanet_layer(
                        infra,
                        acts,
                        layer_idx,
                        layer_w,
                        eps,
                        lora_ctx.as_mut(),
                        true,
                    )?;
                } else {
                    forward_attention_layer(infra, acts, layer_idx, layer_w, eps, true)?;
                }
                rmsnorm_quantize_and_gemv_batch(
                    &infra.elementwise,
                    &infra.stream,
                    &infra.gemv_multi,
                    &acts.quant_i8_buf,
                    &acts.ascale_buf,
                    &acts.x,
                    &layer_w.post_attn_norm,
                    eps,
                    n,
                    &[
                        (&layer_w.gate_proj, &acts.ffn_gate),
                        (&layer_w.up_proj, &acts.ffn_up),
                    ],
                )?;
                swiglu_quantize_and_gemv_accum(
                    &infra.elementwise,
                    &infra.stream,
                    &infra.gemv_multi,
                    &acts.quant_i8_buf,
                    &acts.ascale_buf,
                    &acts.ffn_gate,
                    &acts.ffn_up,
                    infra.config.mlp_hidden,
                    &layer_w.down_proj,
                    &acts.x,
                )?;

                // Capture point: x now holds layer `layer_idx`'s output =
                // layer (layer_idx+1)'s input. Sync + download.
                if cap_i < capture_layers.len() && layer_idx == capture_layers[cap_i] {
                    infra
                        .stream
                        .synchronize()
                        .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
                    infra
                        .stream
                        .memcpy_dtoh(&acts.x, &mut capture_out[cap_i][..n])
                        .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
                    cap_i += 1;
                }
            }

            // Standard final tail (identical to `forward_from_x_with_lora`).
            rmsnorm_quantize_and_gemv_batch(
                &infra.elementwise,
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &acts.x,
                &infra.final_norm,
                eps,
                n,
                &[(&infra.lm_head, &acts.logits)],
            )?;
            infra
                .stream
                .memset_zeros(&mut acts.argmax_buf)
                .map_err(alloc_err)?;
            infra.elementwise.launch_argmax_first(
                &infra.stream,
                &acts.logits,
                infra.config.vocab_size,
                &acts.argmax_buf,
            )?;
            acts.pos += 1;
        }

        self.infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let mut logits = vec![0.0f32; self.infra.config.vocab_size];
        self.infra
            .stream
            .memcpy_dtoh(&self.acts.logits, &mut logits)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        Ok(logits)
    }

    /// **Per-layer residual-stream patch** (riir-train Plan 402 P1 — the
    /// context-read boundary sweep's injection half; the capture twin is
    /// [`Self::forward_token_with_layer_capture`]).
    ///
    /// Before running layer `patch_layer`, the residual stream `acts.x` is
    /// OVERWRITTEN with `donor_x` (a host-side capture from a contrastive
    /// run at the same boundary — `forward_token_with_layer_capture` on the
    /// donor). Everything below `patch_layer` belongs to THIS run; everything
    /// at/above restarts from the donor's residual while the per-layer KV /
    /// recurrent states stay this run's — the decode-shape sufficiency probe:
    /// if the final logits flip toward the donor's answer, the
    /// context-distinguishing information that decides the answer was already
    /// carried by the residual at that boundary; if not, downstream layers
    /// re-read the context through their own routes (attention KV / recurrence).
    ///
    /// Same inline layer loop as [`Self::forward_token_with_layer_capture`]
    /// (same fusion choices); adds ONE sync + 20 KB htod at the boundary.
    /// Sweep-cadence diagnostic — never use on a hot path.
    ///
    /// Returns the logits `[vocab_size]` (downloaded once at the end).
    pub fn forward_token_with_layer_patch(
        &mut self,
        patch_layer: usize,
        donor_x: &[f32],
    ) -> Result<Vec<f32>, CudarcKernelError> {
        guard_no_rotation(&self.infra, "forward_token_with_layer_patch")?;
        let n = self.infra.config.n_embd;
        let eps = self.infra.config.rms_norm_eps as f32;
        assert_eq!(donor_x.len(), n, "donor_x must be exactly n_embd");
        assert!(patch_layer < self.infra.config.n_layer, "patch_layer out of range");

        {
            let infra = &self.infra;
            let acts = &mut self.acts;
            let lora_kernels = &self.lora_kernels;
            let lora_norm_x = &self.lora_norm_x;
            let mut lora_ctx = build_lora_ctx_enabled(
                lora_kernels,
                self.lora_layers.as_mut_slice(),
                lora_norm_x,
                self.lora_forward_enabled,
            );

            debug_assert!(
                acts.pos < infra.config.block_size,
                "position {} exceeds KV cache block_size {}",
                acts.pos,
                infra.config.block_size,
            );

            for layer_idx in 0..infra.config.n_layer {
                // Patch point: replace this run's residual with the donor's
                // BEFORE layer `patch_layer` consumes it.
                if layer_idx == patch_layer {
                    infra
                        .stream
                        .synchronize()
                        .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
                    infra
                        .stream
                        .memcpy_htod(donor_x, &mut acts.x)
                        .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
                }
                let layer_w = &infra.layers[layer_idx];
                let is_deltanet =
                    infra.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;
                if is_deltanet {
                    forward_deltanet_layer(
                        infra,
                        acts,
                        layer_idx,
                        layer_w,
                        eps,
                        lora_ctx.as_mut(),
                        true,
                    )?;
                } else {
                    forward_attention_layer(infra, acts, layer_idx, layer_w, eps, true)?;
                }
                rmsnorm_quantize_and_gemv_batch(
                    &infra.elementwise,
                    &infra.stream,
                    &infra.gemv_multi,
                    &acts.quant_i8_buf,
                    &acts.ascale_buf,
                    &acts.x,
                    &layer_w.post_attn_norm,
                    eps,
                    n,
                    &[
                        (&layer_w.gate_proj, &acts.ffn_gate),
                        (&layer_w.up_proj, &acts.ffn_up),
                    ],
                )?;
                swiglu_quantize_and_gemv_accum(
                    &infra.elementwise,
                    &infra.stream,
                    &infra.gemv_multi,
                    &acts.quant_i8_buf,
                    &acts.ascale_buf,
                    &acts.ffn_gate,
                    &acts.ffn_up,
                    infra.config.mlp_hidden,
                    &layer_w.down_proj,
                    &acts.x,
                )?;
            }

            // Standard final tail (identical to `forward_from_x_with_lora`).
            rmsnorm_quantize_and_gemv_batch(
                &infra.elementwise,
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &acts.x,
                &infra.final_norm,
                eps,
                n,
                &[(&infra.lm_head, &acts.logits)],
            )?;
            infra
                .stream
                .memset_zeros(&mut acts.argmax_buf)
                .map_err(alloc_err)?;
            infra.elementwise.launch_argmax_first(
                &infra.stream,
                &acts.logits,
                infra.config.vocab_size,
                &acts.argmax_buf,
            )?;
            acts.pos += 1;
        }

        self.infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let mut logits = vec![0.0f32; self.infra.config.vocab_size];
        self.infra
            .stream
            .memcpy_dtoh(&self.acts.logits, &mut logits)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        Ok(logits)
    }

    /// Returns the model dimensions needed to construct a
    /// [`MinimalActivationCache`] (Issue 641 T2).
    ///
    /// Convenience method so callers don't need to count DeltaNet layers
    /// manually from `layer_types`.
    pub fn training_dims(&self) -> (usize, usize, usize, usize) {
        let n_deltanet = self
            .infra
            .layer_types
            .iter()
            .filter(|&&t| t == DeltaNetLayerType::DeltaNet)
            .count();
        (
            n_deltanet,
            self.infra.config.deltanet_linear_n_value_heads,
            self.infra.config.deltanet_linear_head_dim,
            self.infra.config.n_embd,
        )
    }

    /// **Minimal-cache training forward** (Issue 641 T2) — saves `x_in` +
    /// `norm_x` for ALL layers, plus `qkv_expanded` + `beta` + `decay` for
    /// DeltaNet layers, into a [`MinimalActivationCache`].
    ///
    /// Mirrors [`forward_token_with_final_hidden`](Self::forward_token_with_final_hidden)
    /// but inserts GPU sync points per layer to download `x_in` (the pre-RMSNorm
    /// hidden state) + `norm_x` (post-RMSNorm) + DeltaNet extras.
    ///
    /// # Cost
    ///
    /// ~2 GPU syncs per layer per token (download `x_in` + `norm_x`). For the
    /// 64-layer model: ~128 syncs/token. Acceptable for training precompute;
    /// never use on the inference path.
    ///
    /// # Backward compatibility
    ///
    /// The existing hot-path forward (`forward_token`, `forward_token_with_final_hidden`)
    /// is unchanged. This is a NEW method for the recomputation backward
    /// (Issue 641 Path A).
    pub fn forward_token_training(
        &mut self,
        cache: &mut MinimalActivationCache,
    ) -> Result<(Vec<f32>, Vec<f32>), CudarcKernelError> {
        self.forward_token_training_impl(cache, None)
    }

    /// **Issue 492 T1 GPU lane - probed training forward.** Identical math
    /// to [`forward_token_training`](Self::forward_token_training) plus one
    /// capture event: after the probed attention layer's decode kernel, the
    /// stream syncs and the probed head's window (`q_post_rope`, `k/v` cache
    /// rows `0..=pos`) and the kernel's output slice are downloaded into
    /// `capture`. Passive monitor: no kernel sequence or math change, so
    /// logits stay bit-identical (G3); the cost is one sync + a dtoh of
    /// `((pos + 1) * kvd * 2 + 2 * q_dim)` floats - a diagnostic-cadence
    /// tap, never per-token on a hot path.
    pub fn forward_token_training_with_attn_probe(
        &mut self,
        cache: &mut MinimalActivationCache,
        probe: &AttentionProbeSpec,
        capture: &mut AttentionProbeCapture,
    ) -> Result<(Vec<f32>, Vec<f32>), CudarcKernelError> {
        let cfg = &self.infra.config;
        if probe.layer >= cfg.n_layer {
            return Err(CudarcKernelError::InvalidArg(format!(
                "attention probe layer {} out of range (n_layer {})",
                probe.layer, cfg.n_layer
            )));
        }
        if self.infra.layer_types[probe.layer] != DeltaNetLayerType::Attention {
            return Err(CudarcKernelError::InvalidArg(format!(
                "attention probe layer {} is not an Attention layer",
                probe.layer
            )));
        }
        if probe.head >= cfg.n_head {
            return Err(CudarcKernelError::InvalidArg(format!(
                "attention probe head {} out of range (n_head {})",
                probe.head, cfg.n_head
            )));
        }
        self.forward_token_training_impl(cache, Some((probe, capture)))
    }

    fn forward_token_training_impl(
        &mut self,
        cache: &mut MinimalActivationCache,
        probe: Option<(&AttentionProbeSpec, &mut AttentionProbeCapture)>,
    ) -> Result<(Vec<f32>, Vec<f32>), CudarcKernelError> {
        guard_no_rotation(&self.infra, "forward_token_training")?;
        let n = self.infra.config.n_embd;
        let eps = self.infra.config.rms_norm_eps as f32;
        let n_layer = self.infra.config.n_layer;
        let mlp = self.infra.config.mlp_hidden;
        let vocab = self.infra.config.vocab_size;

        cache.begin_token(n_layer);

        let (probe_spec, mut probe_cap) = match probe {
            Some((spec, cap)) => (Some(spec), Some(cap)),
            None => (None, None),
        };

        // Per-layer download scratch.
        let mut x_in_buf = vec![0.0f32; n];
        let mut norm_x_buf = vec![0.0f32; n];

        {
            let infra = &self.infra;
            let acts = &mut self.acts;
            // Issue 504 T1 (Plan 370 confirm) — build the LoRA ctx from the
            // same disjoint struct fields the decode path uses. Attached
            // adapters now apply in the TRAINING forward too (the closed-loop
            // fix for riir-train Issue 729 option 1), so the loss and the
            // downstream LoRA gradients see the adapter's true forward effect.
            // The frozen-forward windows (prompt/eval) hold
            // `lora_forward_enabled == false`, which yields `None` here and
            // keeps the exact frozen launch sequence.
            let lora_kernels = &self.lora_kernels;
            let lora_norm_x = &self.lora_norm_x;
            let mut lora_ctx = build_lora_ctx_enabled(
                lora_kernels,
                self.lora_layers.as_mut_slice(),
                lora_norm_x,
                self.lora_forward_enabled,
            );

            // ── Layer loop (with per-layer x_in + norm_x download) ──
            for layer_idx in 0..n_layer {
                let layer_w = &infra.layers[layer_idx];
                let is_deltanet =
                    infra.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

                // Sync + download x_in (the hidden state BEFORE input RMSNorm).
                infra
                    .stream
                    .synchronize()
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
                infra
                    .stream
                    .memcpy_dtoh(&acts.x, &mut x_in_buf)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

                // Run layer with norm_x side-buffer (writes to acts.final_norm_x).
                if is_deltanet {
                    forward_deltanet_layer_training(
                        infra,
                        acts,
                        layer_idx,
                        layer_w,
                        eps,
                        // `lora_ctx` is the OWNED Option<LoraFwdCtx> from
                        // build_lora_ctx_enabled, so this is as_mut(), matching the
                        // four sibling call sites (1022/1081/1217/1751).
                        // `as_deref_mut()` needs LoraFwdCtx: DerefMut and is correct
                        // only for the Option<&mut LoraFwdCtx> PARAMETER shape.
                        lora_ctx.as_mut(),
                    )?;
                } else {
                    // Issue 492 T1: reborrow the capture &mut per iteration.
                    let attn_probe = match (probe_spec, probe_cap.as_deref_mut()) {
                        (Some(spec), Some(cap)) if spec.layer == layer_idx => Some((spec, cap)),
                        _ => None,
                    };
                    forward_attention_layer_training(
                        infra,
                        acts,
                        layer_idx,
                        layer_w,
                        eps,
                        attn_probe,
                    )?;
                }

                // Download norm_x from the side buffer.
                infra
                    .stream
                    .synchronize()
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
                infra
                    .stream
                    .memcpy_dtoh(&acts.final_norm_x, &mut norm_x_buf)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

                if is_deltanet {
                    // Download DeltaNet extras.
                    let n_v_heads = infra.config.deltanet_linear_n_value_heads;
                    let head_dim = infra.config.deltanet_linear_head_dim;
                    let qkv_len = 3 * n_v_heads * head_dim;
                    let mut qkv = vec![0.0f32; qkv_len];
                    let mut beta = vec![0.0f32; n_v_heads];
                    let mut decay = vec![0.0f32; n_v_heads];
                    infra
                        .stream
                        .memcpy_dtoh(&acts.qkv_expanded, &mut qkv)
                        .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
                    infra
                        .stream
                        .memcpy_dtoh(&acts.beta_buf, &mut beta)
                        .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
                    infra
                        .stream
                        .memcpy_dtoh(&acts.decay_buf, &mut decay)
                        .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

                    cache.push_layer_activation(MinimalLayerActivations::deltanet(
                        x_in_buf.clone(),
                        norm_x_buf.clone(),
                        qkv,
                        beta,
                        decay,
                    ));
                } else {
                    cache.push_layer_activation(MinimalLayerActivations::attention(
                        x_in_buf.clone(),
                        norm_x_buf.clone(),
                    ));
                }

                // ── Residual add: x = x + tmp ──
                infra.elementwise.launch_residual_add(
                    &infra.stream, &acts.x, &acts.tmp, &acts.x, n,
                )?;

                // ── Post-attention RMSNorm + FFN gate/up (fused) ──
                rmsnorm_quantize_and_gemv_batch(
                    &infra.elementwise, &infra.stream, &infra.gemv_multi,
                    &acts.quant_i8_buf, &acts.ascale_buf,
                    &acts.x, &layer_w.post_attn_norm, eps, n,
                    &[
                        (&layer_w.gate_proj, &acts.ffn_gate),
                        (&layer_w.up_proj, &acts.ffn_up),
                    ],
                )?;
                swiglu_quantize_and_gemv(
                    &infra.elementwise, &infra.stream, &infra.gemv,
                    &acts.quant_i8_buf, &acts.ascale_buf,
                    &acts.ffn_gate, &acts.ffn_up, mlp,
                    &layer_w.down_proj, &acts.ffn_out,
                )?;

                // ── Residual add: x = x + ffn_out ──
                infra.elementwise.launch_residual_add(
                    &infra.stream, &acts.x, &acts.ffn_out, &acts.x, n,
                )?;
            }

            // Save x_pre_finalnorm (the hidden state before the final RMSNorm).
            infra
                .stream
                .synchronize()
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            let mut x_final = vec![0.0f32; n];
            infra
                .stream
                .memcpy_dtoh(&acts.x, &mut x_final)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            cache.x_pre_finalnorm = x_final;

            // ── Final RMSNorm + lm_head (fused, with norm_x side write) ──
            rmsnorm_quantize_and_gemv_batch_with_norm_x(
                &infra.elementwise, &infra.stream, &infra.gemv_multi,
                &acts.quant_i8_buf, &acts.ascale_buf,
                &acts.x, &infra.final_norm, &acts.final_norm_x, eps, n,
                &[(&infra.lm_head, &acts.logits)],
            )?;

            // Issue 722 H4 — same argmax tail as every other forward variant
            // (the `last_argmax()` family contract: valid after ANY
            // `forward_token*` call — this variant previously skipped it).
            infra
                .stream
                .memset_zeros(&mut acts.argmax_buf)
                .map_err(alloc_err)?;
            infra.elementwise.launch_argmax_first(
                &infra.stream,
                &acts.logits,
                infra.config.vocab_size,
                &acts.argmax_buf,
            )?;

            acts.pos += 1;
        }

        // Sync + download logits AND final_norm_x.
        self.infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let mut logits = vec![0.0f32; vocab];
        let mut norm_x = vec![0.0f32; n];
        self.infra
            .stream
            .memcpy_dtoh(&self.acts.logits, &mut logits)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        self.infra
            .stream
            .memcpy_dtoh(&self.acts.final_norm_x, &mut norm_x)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        Ok((logits, norm_x))
    }

    /// Dispatch-only forward: run all kernel launches but DO NOT sync or
    /// download logits. Used by Issue 618 CUDA Graphs capture — sync/dtoh
    /// must happen AFTER the captured region so they aren't recorded into
    /// the graph (they aren't capturable into a graph).
    ///
    /// After calling this, callers must `stream.synchronize()` and read
    /// `logits_buffer()` to retrieve the output.
    pub fn forward_dispatch_only(&mut self) -> Result<(), CudarcKernelError> {
        let infra = &self.infra;
        let acts = &mut self.acts;
        forward_from_x(infra, acts)
    }

    /// Read-only access to the logits buffer (post-forward).
    /// Used by Issue 618 CUDA Graphs path — caller downloads logits manually
    /// after `graph.launch()` + `synchronize()`.
    pub fn logits_buffer(&self) -> &cudarc::driver::safe::CudaSlice<f32> {
        &self.acts.logits
    }

    /// Issue 618 — Forward one token via CUDA Graph replay.
    ///
    /// Replaces `set_input_token(t); forward_token()` with a single call that:
    ///   1. Writes token_id and pos to device buffers (memcpy_htod, 4 bytes each).
    ///   2. On first call: captures the forward path into a CudaGraph via
    ///      stream capture. Subsequent calls just launch the graph.
    ///   3. Launches the graph, syncs, downloads logits.
    ///
    /// Issue 696 — an attached LoRA adapter (`set_lora`) is captured into the
    /// graph: the LoRA kernels record alongside the backbone kernels. Calling
    /// `set_lora`/`clear_lora` invalidates the captured graph so the next call
    /// re-captures with/without the adapter.
    ///
    /// ## Requirements
    ///
    /// - The forward MUST be created on a non-default stream (`ctx.new_stream()`),
    ///   because the default (null) stream is NOT capturable.
    /// - Event tracking MUST be disabled before construction
    ///   (`unsafe { ctx.disable_event_tracking(); }`) — otherwise cross-stream
    ///   event waits break capture isolation.
    ///
    /// ## Returns
    ///
    /// The logits (downloaded once at the end — same as `forward_token`).
    #[cfg(feature = "cuda_graphs_forward")]
    pub fn forward_token_graph(
        &mut self,
        token_id: usize,
    ) -> Result<Vec<f32>, CudarcKernelError> {
        // Write to PERSISTENT host buffers (not stack locals) — cuMemcpyHtoDAsync_v2
        // is async, so the host pointer must remain valid until the GPU completes
        // the copy. Stack locals would be dropped before the async op reads them.
        {
            // Scoped mutable borrow for the assignment.
            let acts = &mut self.acts;
            acts.token_host = token_id as i32;
            acts.pos_host = acts.pos as i32;
        }

        // Split borrows: take separate refs to the stream + acts fields so the
        // borrow checker sees them as disjoint (stream from infra, acts from self).
        let stream = &self.infra.stream;
        let acts = &mut self.acts;
        let token_host_ref: &i32 = &acts.token_host;
        let pos_host_ref: &i32 = &acts.pos_host;
        let token_dev_ref: &mut CudaSlice<i32> = &mut acts.token_dev_buf;
        let pos_dev_ref: &mut CudaSlice<i32> = &mut acts.pos_dev_buf;

        // Write new pos and token to the device buffers. These memcpys are
        // stream-ordered BEFORE the graph launch (same stream).
        stream
            .memcpy_htod(std::slice::from_ref(token_host_ref), token_dev_ref)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        stream
            .memcpy_htod(std::slice::from_ref(pos_host_ref), pos_dev_ref)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        // The acts borrow ends here (NLL). No explicit drop needed.

        if self.graph.is_none() {
            // ── First call: capture the forward into a graph ──
            // The capture records every kernel launch in `forward_from_x_devpos`
            // into a replayable graph. Buffer addresses are baked in (constant
            // per the persistent allocations in `with_shared`). Issue 722 H12 —
            // the capture body is shared with the argmax/timed variants
            // (`ensure_graph_captured`; the H3 abort contract lives there).
            self.ensure_graph_captured()?;
        }

        // ── Launch the graph (captured or replayed) ──
        let graph = self.graph.as_ref().expect("graph was just captured");
        graph
            .launch()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        // Sync + download logits (outside the graph — sync isn't capturable).
        self.infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let mut logits = vec![0.0f32; self.infra.config.vocab_size];
        self.infra
            .stream
            .memcpy_dtoh(&self.acts.logits, &mut logits)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        // Increment pos AFTER the graph launch (the graph itself doesn't run
        // host-side code, so forward_from_x_devpos no longer increments pos).
        self.acts.pos += 1;
        Ok(logits)
    }

    /// Issue 722 H12 — the shared capture-region body for the three
    /// `forward_token_graph*` entry points (previously triplicated inline;
    /// the file's own "keep them in sync" comment is now enforced by
    /// construction).
    ///
    /// Captures `forward_from_x_devpos` into a graph on first call (or after
    /// `invalidate_graph` / a speculative run — the rotation invalidates the
    /// baked buffer addresses), uploads it, and leaves the stream synced.
    /// The launch itself stays in each caller (the timed variant marks it).
    ///
    /// Issue 722 H3 — if any launch inside the capture region fails, the
    /// stream is returned to non-capturing state BEFORE the error propagates:
    /// `end_capture` on an invalidated region reports the invalidation (which
    /// we swallow — the original launch error is the actionable one) and, per
    /// CUDA semantics, ALWAYS transitions the stream out of capture mode.
    /// Without this, one failed launch (e.g. an OOM mid-capture) would wedge
    /// every later operation on the stream.
    #[cfg(feature = "cuda_graphs_forward")]
    fn ensure_graph_captured(&mut self) -> Result<(), CudarcKernelError> {
        if self.graph.is_some() {
            return Ok(());
        }
        self.infra
            .stream
            .begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        // Run the forward — every kernel launch gets captured. Issue 696 —
        // build the LoRA ctx from disjoint struct fields (same pattern as
        // `forward_token`) so the LoRA kernels are captured into the graph
        // when an adapter is attached.
        let infra = &self.infra;
        let acts = &mut self.acts;
        let lora_kernels = &self.lora_kernels;
        let lora_norm_x = &self.lora_norm_x;
        let mut lora_ctx = build_lora_ctx_enabled(
            lora_kernels,
            self.lora_layers.as_mut_slice(),
            lora_norm_x,
            self.lora_forward_enabled,
        );
        if let Err(e) = forward_from_x_devpos(infra, acts, lora_ctx.as_mut()) {
            let _ = self.infra.stream.end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            return Err(e);
        }
        let graph = self
            .infra
            .stream
            .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let graph = graph
            .ok_or_else(|| CudarcKernelError::Launch("end_capture returned None (no graph captured)".into()))?;
        // Pre-upload to absorb first-launch setup overhead.
        graph
            .upload()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        self.infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        self.graph = Some(graph);
        Ok(())
    }

    /// Issue 618 — Drop the captured CUDA Graph (if any) so the next
    /// `forward_token_graph` call re-captures. Since Issue 696 this is purely
    /// defensive: `reset_state` zeroes in place (addresses stable) and
    /// `set_lora`/`clear_lora` invalidate the graph themselves, so callers
    /// normally never need this.
    #[cfg(feature = "cuda_graphs_forward")]
    pub fn invalidate_graph(&mut self) {
        self.graph = None;
    }

    /// Issue 697 — greedy-decode variant of [`forward_token_graph`]: instead of
    /// downloading the vocab-sized logits vector (~1 MB), downloads the 8-byte
    /// GPU-computed argmax (`argmax_first_f32` is captured into the graph, with
    /// its zeroing memset).
    ///
    /// Tie-break semantics are CPU-exact (first index among equal maxima), so
    /// the token sequence is IDENTICAL to argmaxing the downloaded logits.
    /// Saves ~380 µs/token of dtoh + host argmax (Bench 674 stage profile).
    ///
    /// Issue 984 T2(a) — the per-token transfers are PINNED (WC): the feed pair
    /// uploads from `feed_pinned` (no pageable staging) and the argmax lands in
    /// `argmax_pinned` via direct DMA; the return read is fenced by the pinned
    /// slice's event (recorded after the dtoh in stream order), which replaces
    /// the old second `synchronize` — one fewer driver call per token.
    #[cfg(feature = "cuda_graphs_forward")]
    pub fn forward_token_graph_argmax(
        &mut self,
        token_id: usize,
    ) -> Result<usize, CudarcKernelError> {
        {
            let acts = &mut self.acts;
            acts.token_host = token_id as i32;
            acts.pos_host = acts.pos as i32;
        }
        let stream = &self.infra.stream;
        let acts = &mut self.acts;
        // Issue 984 T2(a) — stage the feed pair into the pinned WC source (the
        // event fence inside `as_mut_slice` is a no-op here: the previous
        // token's htod completed before the last argmax read returned).
        {
            let feed = acts
                .feed_pinned
                .as_mut_slice()
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            feed[0] = acts.token_host;
            feed[1] = acts.pos_host;
        }
        let feed: &[i32] = acts
            .feed_pinned
            .as_slice()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        stream
            .memcpy_htod(&feed[0..1], &mut acts.token_dev_buf)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        stream
            .memcpy_htod(&feed[1..2], &mut acts.pos_dev_buf)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        if self.graph.is_none() {
            // Issue 722 H12 — shared capture body (`ensure_graph_captured`).
            self.ensure_graph_captured()?;
        }

        let graph = self.graph.as_ref().expect("graph was just captured");
        graph
            .launch()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        // Pinned 8-byte dtoh (async; the copy's completion event is recorded
        // in stream order). The read below is fenced by that event — it waits
        // for the graph AND the DMA, replacing the explicit second sync.
        let stream = &self.infra.stream;
        stream
            .memcpy_dtoh(&self.acts.argmax_buf, &mut self.acts.argmax_pinned)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let packed = unsafe {
            *self
                .acts
                .argmax_pinned
                .as_ptr()
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?
        };
        self.acts.pos += 1;
        Ok((!(packed as u32)) as usize)
    }

    /// Issue 697 — decode the GPU-computed argmax from `argmax_buf` (the
    /// stream must already be synchronized — every `forward_token*` variant
    /// syncs before returning).
    ///
    /// Valid after any `forward_token`, `forward_token_graph*` call on this
    /// instance. Stale only if NO forward has run yet.
    pub fn last_argmax(&self) -> Result<usize, CudarcKernelError> {
        let mut packed = [0u64; 1];
        self.infra
            .stream
            .memcpy_dtoh(&self.acts.argmax_buf, &mut packed)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        Ok((!(packed[0] as u32)) as usize)
    }

    /// Issue 696 follow-up — `forward_token_graph` with host-side stage
    /// timings (see [`GraphStageTimings`]). Identical behavior to
    /// [`forward_token_graph`]; use for profiling only.
    #[cfg(feature = "cuda_graphs_forward")]
    pub fn forward_token_graph_timed(
        &mut self,
        token_id: usize,
    ) -> Result<(Vec<f32>, GraphStageTimings), CudarcKernelError> {
        let mut t = GraphStageTimings::default();
        let mut tick = std::time::Instant::now();
        let mut mark = || {
            let us = tick.elapsed().as_secs_f64() * 1e6;
            tick = std::time::Instant::now();
            us
        };
        // Re-implement the forward_token_graph body with timing marks — the
        // non-timed variant is the production path; keep them in sync.
        {
            let acts = &mut self.acts;
            acts.token_host = token_id as i32;
            acts.pos_host = acts.pos as i32;
        }
        let stream = &self.infra.stream;
        let acts = &mut self.acts;
        stream
            .memcpy_htod(std::slice::from_ref(&acts.token_host), &mut acts.token_dev_buf)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        stream
            .memcpy_htod(std::slice::from_ref(&acts.pos_host), &mut acts.pos_dev_buf)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        t.htod_us += mark();

        if self.graph.is_none() {
            // Issue 722 H12 — shared capture body (`ensure_graph_captured`).
            // On first call this captures + uploads + syncs; on replay it's a
            // no-op fast path (the launch_us mark below includes capture time
            // on the first call, same as the pre-extraction behavior).
            self.ensure_graph_captured()?;
        }

        let graph = self.graph.as_ref().expect("graph was just captured");
        graph
            .launch()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        t.launch_us += mark();

        self.infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        t.sync_us += mark();

        let mut logits = vec![0.0f32; self.infra.config.vocab_size];
        self.infra
            .stream
            .memcpy_dtoh(&self.acts.logits, &mut logits)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        t.dtoh_us += mark();

        self.acts.pos += 1;
        Ok((logits, t))
    }

    /// Issue 984 T1 — stage-timed variant of the SHIPPING decode path
    /// ([`forward_token_graph_argmax`]): same stage marks as
    /// [`forward_token_graph_timed`] except `dtoh_us` carries the
    /// argmax-decode cost (the 8-byte pinned dtoh + its event-fenced read —
    /// Issue 984 T2(a); the shipping path no longer pays the second
    /// `synchronize`). The gap between this method's per-stage sum and ARM 1's
    /// measured wall is the loop + bookkeeping residue.
    #[cfg(feature = "cuda_graphs_forward")]
    pub fn forward_token_graph_argmax_timed(
        &mut self,
        token_id: usize,
    ) -> Result<(usize, GraphStageTimings), CudarcKernelError> {
        let mut t = GraphStageTimings::default();
        let mut tick = std::time::Instant::now();
        let mut mark = || {
            let us = tick.elapsed().as_secs_f64() * 1e6;
            tick = std::time::Instant::now();
            us
        };
        {
            let acts = &mut self.acts;
            acts.token_host = token_id as i32;
            acts.pos_host = acts.pos as i32;
        }
        let stream = &self.infra.stream;
        let acts = &mut self.acts;
        {
            let feed = acts
                .feed_pinned
                .as_mut_slice()
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            feed[0] = acts.token_host;
            feed[1] = acts.pos_host;
        }
        let feed: &[i32] = acts
            .feed_pinned
            .as_slice()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        stream
            .memcpy_htod(&feed[0..1], &mut acts.token_dev_buf)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        stream
            .memcpy_htod(&feed[1..2], &mut acts.pos_dev_buf)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        t.htod_us += mark();

        if self.graph.is_none() {
            self.ensure_graph_captured()?;
        }

        let graph = self.graph.as_ref().expect("graph was just captured");
        graph
            .launch()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        t.launch_us += mark();

        // The shipping path fences on the dtoh's completion event instead of
        // an explicit sync; the timed variant keeps the explicit sync so the
        // sync stage stays comparable with ARM 2 / the T1 record.
        self.infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        t.sync_us += mark();

        self.infra
            .stream
            .memcpy_dtoh(&self.acts.argmax_buf, &mut self.acts.argmax_pinned)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let packed = unsafe {
            *self
                .acts
                .argmax_pinned
                .as_ptr()
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?
        };
        t.dtoh_us += mark();

        self.acts.pos += 1;
        Ok(((!(packed as u32)) as usize, t))
    }

    /// Reset all per-layer state for a new sequence.
    pub fn reset_state(&mut self) -> Result<(), CudarcKernelError> {
        let infra = &self.infra;
        let acts = &mut self.acts;
        reset_state(infra, acts)
    }

    /// Current decode position (number of tokens processed).
    /// Needed by the speculative seam: [`checkpoint_speculative_gpu`] does not
    /// carry `pos` (it is host-side bookkeeping) — the caller snapshots it
    /// via this accessor and passes it to [`rollback_speculative_gpu`].
    pub fn position(&self) -> usize {
        self.acts.pos
    }

    /// Synchronize the stream (drain all queued GPU work). Surfaces async
    /// launch errors at a chosen point — useful to isolate which queued
    /// stage faulted (CUDA reports async faults at the NEXT sync, which
    /// otherwise obscures the culprit).
    pub fn synchronize(&self) -> Result<(), CudarcKernelError> {
        self.infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))
    }

    // ────────────────────────────────────────────────────────────────────
    // Issue 717 G3b: speculative decoding — the cudarc port of the Issue 665
    // seam (see `TernaryDeltanetGpuForward` on the cubecl path for the M3
    // twin). Same contract, two 4090-specific upgrades:
    //
    //   1. The rotation also swaps the GPU-side argmax buffers (Issue 697),
    //      so `forward_speculative_verify_argmax` reads K+1 × 8 bytes per
    //      cycle instead of K+1 × vocab×4 bytes of logits — the
    //      device-side-argmax-minimal-download rule applied to verify.
    //   2. Checkpoint/rollback are device-to-device `memcpy_dtod` copies into
    //      pre-allocated backups — fully async, zero CPU↔GPU sync (the
    //      cubecl `_gpu` variant's contract, on CUDA).
    //
    // ## The rotation (K=2)
    //
    //   swap[0]: acts.logits ↔ pool[0]   (pool[0] := original L0)
    //   forward 0 → writes the pool[0] allocation (now `acts.logits`)
    //   swap[1]: acts.logits ↔ pool[1]   (pool[1] := fwd0's buffer)
    //   forward 1 → writes the pool[1] allocation
    //
    // Reads: `[pool[0], pool[1], acts.logits]` = `[pre-spec, fwd0, fwd1]` —
    // K+1 position-aligned results with out[j] verifying `draft[j]` and
    // out[K] the post-draft bonus position. Identical semantics to the
    // cubecl swap chain. The argmax pool rotates in lockstep. The field KEEPS
    // the last forward's buffer between cycles (never un-rotated mid-run) —
    // that is what makes the next cycle's out[0] the live pre-speculation
    // prediction.
    //
    // End-of-run contract (Issue 722 H1 — an earlier doc referenced a
    // `spec_restore_buffers`/`spec_unrotate` pair that never shipped): there
    // is NO un-rotation. Eager forwards need nothing — they always write
    // through the current field handles. A previously captured CUDA Graph is
    // invalidated BY the rotation itself (`spec_verify_dispatch` drops it):
    // graphs bake buffer addresses at capture time, and a replay against the
    // pre-rotation allocations would write buffers the fields no longer point
    // at (`forward_token_graph*` would silently download the wrong logits).
    // The next graph call re-captures with the post-rotation addresses — a
    // one-time cost per speculative run.
    // ────────────────────────────────────────────────────────────────────

    /// Run the K verify forwards WITHOUT intermediate syncs (core rotation).
    /// After this returns, the K+1 result buffers hold `[pre-spec, fwd0, ..fwd(K-1)]`
    /// (logits in `spec.logits_pool[0..k]` + `acts.logits`; argmax in
    /// `spec.argmax_pool[0..k]` + `acts.argmax_buf`) and the stream is still
    /// in flight — callers read results through the pool + field handles (as
    /// the two `forward_speculative_verify*` wrappers do). No un-rotation
    /// exists or is needed (the rotation persists by design; see the contract
    /// above). Any captured CUDA Graph is invalidated here (see above).
    #[cfg(feature = "speculative_decode")]
    fn spec_verify_dispatch(
        &mut self,
        draft_tokens: &[usize],
    ) -> Result<(), CudarcKernelError> {
        let k = draft_tokens.len();
        assert!(
            (1..=SPEC_MAX_K).contains(&k),
            "speculative verify supports 1..={SPEC_MAX_K} drafts, got {k}"
        );
        // Issue 746 tripwire — the whole verify chunk must fit the
        // full-length KV cache, checked up front (before any dispatch work);
        // complements the per-forward Issue 641 guard in
        // `forward_layers_with_lora`, which only sees the CURRENT position
        // mid-run and would fire on the second verify forward at best.
        debug_assert!(
            self.acts.pos + k <= self.infra.config.block_size,
            "spec verify would overflow the full-length KV cache \
             (pos {} + k {k} > block_size {}) — Issue 746 tripwire on the \
             append-only KV contract",
            self.acts.pos,
            self.infra.config.block_size
        );
        // Issue 722 H1 — the rotation below swaps the logits/argmax field
        // handles, so any graph captured BEFORE this run bakes addresses a
        // replay would write while the fields point at pool buffers. The
        // rotation owns the invalidation (the real mitigation; the doc
        // previously pointed at a nonexistent `spec_restore_buffers`).
        // Re-capture is one-time per spec run; the eager path needs nothing.
        #[cfg(feature = "cuda_graphs_forward")]
        {
            self.graph = None;
        }
        for (i, &tok) in draft_tokens.iter().enumerate() {
            assert!(
                tok < self.infra.config.vocab_size,
                "speculative draft token {tok} ≥ vocab {} — the drafter is \
                 responsible for in-vocab proposals (embedding row read would be \
                 out of bounds)",
                self.infra.config.vocab_size
            );
            std::mem::swap(&mut self.acts.logits, &mut self.spec.logits_pool[i]);
            std::mem::swap(&mut self.acts.argmax_buf, &mut self.spec.argmax_pool[i]);
            self.set_input_token(tok)?;
            self.forward_dispatch_only()?;
        }
        Ok(())
    }

    /// Read one packed-argmax buffer (must already be synchronized).
    #[cfg(feature = "speculative_decode")]
    fn read_argmax_buf(
        &self,
        buf: &CudaSlice<u64>,
    ) -> Result<usize, CudarcKernelError> {
        let mut packed = [0u64; 1];
        self.infra
            .stream
            .memcpy_dtoh(buf, &mut packed)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        Ok((!(packed[0] as u32)) as usize)
    }

    /// K verify forwards + ONE sync + K+1 argmax reads (8 bytes each).
    ///
    /// Returns K+1 token ids, position-aligned exactly like
    /// [`forward_speculative_verify`]: `out[j]` is the model's greedy argmax
    /// for `draft[j]`'s position (so `out[j] == draft[j]` accepts draft j),
    /// and `out[K]` is the bonus position after the last draft — the token
    /// the model itself would emit next. This is the hot-path variant: total
    /// download per cycle is (K+1) × 8 bytes (vs (K+1) × vocab × 4 for the
    /// logits variant).
    ///
    /// Advances `acts.pos` by K; on any rejection the caller MUST
    /// [`rollback_speculative_gpu`] (see below).
    #[cfg(feature = "speculative_decode")]
    pub fn forward_speculative_verify_argmax(
        &mut self,
        draft_tokens: &[usize],
    ) -> Result<Vec<usize>, CudarcKernelError> {
        let k = draft_tokens.len();
        self.spec_verify_dispatch(draft_tokens)?;
        self.infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let mut out = Vec::with_capacity(k + 1);
        for i in 0..k {
            out.push(self.read_argmax_buf(&self.spec.argmax_pool[i])?);
        }
        out.push(self.read_argmax_buf(&self.acts.argmax_buf)?);
        Ok(out)
    }

    /// K verify forwards + ONE sync + K+1 full-logits downloads.
    ///
    /// The logits variant (mirrors the cubecl `forward_speculative_verify`
    /// contract verbatim) for consumers that need distributions — sampled
    /// verification, acceptance-rate estimation. Returns K+1 logits vectors,
    /// `out[0]` = pre-speculation (verifies draft[0]), `out[i]` = after
    /// processing draft[i-1], `out[K]` = after the last draft (bonus seed).
    #[cfg(feature = "speculative_decode")]
    pub fn forward_speculative_verify(
        &mut self,
        draft_tokens: &[usize],
    ) -> Result<Vec<Vec<f32>>, CudarcKernelError> {
        let k = draft_tokens.len();
        self.spec_verify_dispatch(draft_tokens)?;
        self.infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        let vocab = self.infra.config.vocab_size;
        let mut out = Vec::with_capacity(k + 1);
        for i in 0..k {
            let mut v = vec![0.0f32; vocab];
            self.infra
                .stream
                .memcpy_dtoh(&self.spec.logits_pool[i], &mut v)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            out.push(v);
        }
        let mut v = vec![0.0f32; vocab];
        self.infra
            .stream
            .memcpy_dtoh(&self.acts.logits, &mut v)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        out.push(v);
        Ok(out)
    }

    /// GPU-side speculative checkpoint: async dtod copies of every DeltaNet
    /// layer's recurrent + conv state into pre-allocated backups.
    ///
    /// ZERO CPU↔GPU sync — the copies enqueue on the stream and the FIFO
    /// order guarantees they complete before any subsequent verify forward
    /// reads the state. The attention KV cache needs no backup: decode at
    /// position `p` reads `[0..=p]` and the append for `p` overwrites the
    /// stale speculative entry before the decode kernel runs (same argument
    /// as the cubecl path).
    ///
    /// The caller snapshots `pos` separately (`self.position()`) and passes
    /// it to [`rollback_speculative_gpu`].
    #[cfg(feature = "speculative_decode")]
    pub fn checkpoint_speculative_gpu(&mut self) -> Result<(), CudarcKernelError> {
        let stream = &self.infra.stream;
        let states = &self.acts.layer_states;
        let backups = &mut self.spec.state_backups;
        for (st, bk) in states.iter().zip(backups.iter_mut()) {
            if let (Some(src), Some(dst)) = (&st.deltanet_state, &mut bk.deltanet_state) {
                RecStateBuf::copy_dtod(stream, src, dst)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            }
            if let (Some(src), Some(dst)) = (&st.conv_state, &mut bk.conv_state) {
                stream
                    .memcpy_dtod(src, dst)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            }
        }
        Ok(())
    }

    /// GPU-side speculative rollback: async dtod copies backup → state +
    /// restore the host-side position.
    ///
    /// Call ONLY when speculation was rejected. After this the state is
    /// exactly the pre-speculation state; the caller re-applies the committed
    /// prefix (accepted drafts + correction) with
    /// `set_input_token` + `forward_dispatch_only` — those forwards are also
    /// sync-free; the next verify cycle's single read sync drains everything.
    #[cfg(feature = "speculative_decode")]
    pub fn rollback_speculative_gpu(
        &mut self,
        checkpoint_pos: usize,
    ) -> Result<(), CudarcKernelError> {
        self.acts.pos = checkpoint_pos;
        let stream = &self.infra.stream;
        let backups = &self.spec.state_backups;
        let states = &mut self.acts.layer_states;
        for (bk, st) in backups.iter().zip(states.iter_mut()) {
            if let (Some(src), Some(dst)) = (&bk.deltanet_state, &mut st.deltanet_state) {
                RecStateBuf::copy_dtod(stream, src, dst)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            }
            if let (Some(src), Some(dst)) = (&bk.conv_state, &mut st.conv_state) {
                stream
                    .memcpy_dtod(src, dst)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Forward one token with per-section GPU timing via CUDA events.
    ///
    /// Returns `(logits, SectionProfile)` where `SectionProfile` contains the
    /// GPU-side elapsed time for each major forward section (GEMV, elementwise,
    /// attention-specific). Used by Issue 616 launch-overhead investigation.
    #[allow(clippy::type_complexity)]
    pub fn forward_token_profiled(
        &mut self,
    ) -> Result<(Vec<f32>, SectionProfile), CudarcKernelError> {
        guard_no_rotation(&self.infra, "forward_token_profiled")?;
        let infra = &self.infra;
        let acts = &mut self.acts;

        // Create CUDA events for section timing.
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let mk_evt = || {
            infra
                .ctx
                .new_event(flags)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))
        };

        // We record events at three boundaries per layer:
        //   A: before pre-attn norm (start of layer)
        //   B: after layer internals (deltanet/attention done, before FFN GEMVs)
        //   C: after FFN down_proj GEMV (end of GEMV work)
        //   D: after residual add (end of layer)
        // This gives us: (A→B) = norm + layer internals,
        //                (B→C) = FFN GEMVs + swiglu,
        //                (C→D) = residual add.
        // Plus initial (start→A₀) and final (D₆₃→logits).
        let n_layer = infra.config.n_layer;
        // 4 events per layer + 1 start + 1 end = 4*64 + 2 = 258 events.
        let mut events_a: Vec<cudarc::driver::safe::CudaEvent> = Vec::with_capacity(n_layer);
        let mut events_b: Vec<cudarc::driver::safe::CudaEvent> = Vec::with_capacity(n_layer);
        let mut events_c: Vec<cudarc::driver::safe::CudaEvent> = Vec::with_capacity(n_layer);
        let mut events_d: Vec<cudarc::driver::safe::CudaEvent> = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            events_a.push(mk_evt()?);
            events_b.push(mk_evt()?);
            events_c.push(mk_evt()?);
            events_d.push(mk_evt()?);
        }
        let evt_start = mk_evt()?;
        let evt_end = mk_evt()?;

        // Record start event.
        evt_start
            .record(&infra.stream)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        let n = infra.config.n_embd;
        let eps = infra.config.rms_norm_eps as f32;

        let mut gemv_count = 0u32;

        for layer_idx in 0..n_layer {
            // Record event A (start of layer).
            events_a[layer_idx]
                .record(&infra.stream)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

            let layer_w = &infra.layers[layer_idx];
            let is_deltanet = infra.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

            // ── Pre-attn norm is fused into the layer functions' input
            // projection quantize (Issue 624). No separate launch_rmsnorm here.
            if is_deltanet {
                forward_deltanet_layer(infra, acts, layer_idx, layer_w, eps, None, false)?;
            } else {
                forward_attention_layer(infra, acts, layer_idx, layer_w, eps, false)?;
            }
            infra.elementwise.launch_residual_add(
                &infra.stream, &acts.x, &acts.tmp, &acts.x, n,
            )?;
            // Note: post-attn RMSNorm is fused into the FFN block below (Issue 623).

            // Record event B (before FFN GEMVs).
            events_b[layer_idx]
                .record(&infra.stream)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

            // ── FFN block (fused RMSNorm + quantize + gate/up + swiglu + down) ──
            // Issue 623 — fuse post-attn RMSNorm + quantize into one kernel.
            // Note: the fused norm+quantize is counted in the FFN section
            // (event B → C), not pre_gemv. The timing boundary shifts by one
            // kernel launch (~5 µs) — negligible for section analysis.
            rmsnorm_quantize_and_gemv_batch(
                &infra.elementwise, &infra.stream, &infra.gemv_multi,
                &acts.quant_i8_buf, &acts.ascale_buf,
                &acts.x, &layer_w.post_attn_norm, eps, n,
                &[
                    (&layer_w.gate_proj, &acts.ffn_gate),
                    (&layer_w.up_proj, &acts.ffn_up),
                ],
            )?;
            gemv_count += 2;
            // Issue 625 — fuse SwiGLU + quantize into one kernel, then dispatch
            // down_proj GEMV. Saves 1 launch + intermediate ffn_hidden traffic.
            swiglu_quantize_and_gemv(
                &infra.elementwise, &infra.stream, &infra.gemv,
                &acts.quant_i8_buf, &acts.ascale_buf,
                &acts.ffn_gate, &acts.ffn_up, infra.config.mlp_hidden,
                &layer_w.down_proj, &acts.ffn_out,
            )?;
            gemv_count += 1;

            // Record event C (after FFN GEMVs).
            events_c[layer_idx]
                .record(&infra.stream)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

            // ── Residual add ──
            infra.elementwise.launch_residual_add(
                &infra.stream, &acts.x, &acts.ffn_out, &acts.x, n,
            )?;

            // Record event D (end of layer).
            events_d[layer_idx]
                .record(&infra.stream)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }

        // ── Final norm + lm_head (fused) ──
        // Issue 623 — fuse RMSNorm + quantize + lm_head GEMV.
        rmsnorm_quantize_and_gemv_batch(
            &infra.elementwise, &infra.stream, &infra.gemv_multi,
            &acts.quant_i8_buf, &acts.ascale_buf,
            &acts.x, &infra.final_norm, eps, n,
            &[(&infra.lm_head, &acts.logits)],
        )?;
        gemv_count += 1;

        // Record end event + sync.
        evt_end
            .record(&infra.stream)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        // ── Compute section timings ──
        let total_ms = evt_start
            .elapsed_ms(&evt_end)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        // A→B: norm + layer internals (deltanet/attention) + residual + norm
        // B→C: FFN GEMVs (3 per layer) + swiglu
        // C→D: residual add
        // D→A(next): nothing (immediately next event)
        // start→A₀: nothing (immediately first event)
        // D_last→end: final norm + lm_head GEMV
        let mut pre_gemv_ms = 0.0f32; // A→B across all layers
        let mut ffn_gemv_ms = 0.0f32; // B→C across all layers
        let mut residual_ms = 0.0f32; // C→D across all layers

        for i in 0..n_layer {
            pre_gemv_ms += events_a[i]
                .elapsed_ms(&events_b[i])
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            ffn_gemv_ms += events_b[i]
                .elapsed_ms(&events_c[i])
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            residual_ms += events_c[i]
                .elapsed_ms(&events_d[i])
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        let final_ms = events_d[n_layer - 1]
            .elapsed_ms(&evt_end)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        let section = SectionProfile {
            total_ms,
            pre_gemv_ms, // norms + layer internals (deltanet/attention)
            ffn_gemv_ms, // FFN GEMVs + swiglu
            residual_ms, // residual adds (negligible)
            final_ms,    // final norm + lm_head
            gemv_count,
        };

        // Download logits.
        let mut logits = vec![0.0f32; infra.config.vocab_size];
        infra
            .stream
            .memcpy_dtoh(&acts.logits, &mut logits)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        acts.pos += 1;
        Ok((logits, section))
    }

    /// Download all persistent state (DeltaNet recurrent + conv, Attention KV
    /// cache) from GPU buffers into a CPU [`HybridCache`]. Used by Plan 334
    /// hybrid training: GPU forward for prompt tokens (frozen, no LoRA) → state
    /// transfer → CPU forward for target tokens (with LoRA + activation saving
    /// for backward).
    ///
    /// Mirrors `TernaryDeltanetGpuForward::download_state_to_hybrid_cache` (the
    /// CubeCL version, commit `8e6eab7f`) for the cudarc backend. Unblocks arm
    /// B/C training on the 4090 via the hybrid GPU-prompt + CPU-target path.
    ///
    /// The `cache` MUST be pre-initialized with
    /// [`HybridCache::with_layer_types`] using the same `config` + `layer_types`
    /// as this forward struct. Sizes are checked via `debug_assert_eq!`.
    ///
    /// Only the first `self.acts.pos` positions of the KV caches are valid
    /// (positions `0..self.acts.pos` were written by the prompt forward). The
    /// DeltaNet recurrent + conv states are overwritten wholesale.
        /// Issue 879 T3 — diagnostic KV fake-quant hook: download one attention
    /// layer's KV-cache rows `[row0, row0+rows)` (`[rows][kvd]` each), hand
    /// them to the caller's transform, upload the result back. The T3
    /// KV-quant NLL arms quantize each chunk's own rows at chunk boundaries
    /// so every FUTURE chunk reads quantized values. Diagnostic only (the
    /// dump_kv_row precedent on the dense path): 2 syncs per call, never
    /// called on the serving path.
    pub fn transform_kv_rows(
        &mut self,
        layer_idx: usize,
        row0: usize,
        rows: usize,
        f: &mut dyn FnMut(&mut [f32], &mut [f32]),
    ) -> Result<(), CudarcKernelError> {
        let kvd = self.infra.config.n_kv_head * self.infra.config.head_dim;
        let err = |e: cudarc::driver::DriverError| CudarcKernelError::Launch(e.to_string());
        let (ks, vs) = {
            let st = &mut self.acts.layer_states[layer_idx];
            (
                st.key_cache
                    .as_mut()
                    .expect("transform_kv_rows: layer has no key_cache (not an attention layer)"),
                st.value_cache
                    .as_mut()
                    .expect("transform_kv_rows: layer has no value_cache (not an attention layer)"),
            )
        };
        let (start, end) = (row0 * kvd, (row0 + rows) * kvd);
        debug_assert!(end <= ks.len(), "transform_kv_rows: rows beyond cache");
        let mut k = vec![0.0f32; end - start];
        let mut v = vec![0.0f32; end - start];
        self.infra
            .stream
            .memcpy_dtoh(&ks.slice(start..end), &mut k)
            .map_err(err)?;
        self.infra
            .stream
            .memcpy_dtoh(&vs.slice(start..end), &mut v)
            .map_err(err)?;
        f(&mut k, &mut v);
        self.infra
            .stream
            .memcpy_htod(&k, &mut ks.slice_mut(start..end))
            .map_err(err)?;
        self.infra
            .stream
            .memcpy_htod(&v, &mut vs.slice_mut(start..end))
            .map_err(err)?;
        Ok(())
    }

    pub fn download_state_to_hybrid_cache(
        &self,
        cache: &mut riir_infer_core::deltanet::forward::HybridCache,
    ) -> Result<(), CudarcKernelError> {
        let infra = &self.infra;
        let acts = &self.acts;

        // Sync — all kernel launches must complete before downloading.
        infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        // 1. DeltaNet recurrent + conv states.
        // Download to temp buffers first (cudarc memcpy_dtoh requires
        // dst.len() >= src.len(); the GPU allocation size is the source of
        // truth). Then copy into the CPU HybridCache with size checks.
        for i in 0..infra.config.n_layer {
            if infra.layer_types[i] != DeltaNetLayerType::DeltaNet {
                continue;
            }
            if let Some(ref handle) = acts.layer_states[i].deltanet_state {
                let n = handle.len();
                // Plan 603 R2 — `Half` widens on the host (the `half` crate
                // matches the hardware cvt exactly); the cache stays f32.
                let tmp = handle
                    .download_to_f32(&infra.stream)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
                let dst = &mut cache.deltanet_state.recurrent_states[i];
                debug_assert_eq!(
                    dst.len(),
                    n,
                    "recurrent state size mismatch at layer {}: GPU={}, CPU={}",
                    i,
                    n,
                    dst.len()
                );
                dst.copy_from_slice(&tmp);
            }
            if let Some(ref handle) = acts.layer_states[i].conv_state {
                let n = handle.len();
                let mut tmp = vec![0.0f32; n];
                infra
                    .stream
                    .memcpy_dtoh(handle, &mut tmp)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
                let dst = &mut cache.deltanet_state.conv_states[i];
                debug_assert_eq!(
                    dst.len(), n,
                    "conv state size mismatch at layer {}: GPU={}, CPU={}",
                    i, n, dst.len()
                );
                dst.copy_from_slice(&tmp);
            }
        }

        // 2. Attention KV caches.
        //
        // GPU layout: `[block_size, kvd]` row-major (one row per position).
        // CPU layout: same. Copy only the first `pos` positions.
        let kvd = infra.config.n_kv_head * infra.config.head_dim;
        let n_positions = acts.pos;
        let n_floats = n_positions * kvd;

        for i in 0..infra.config.n_layer {
            if infra.layer_types[i] != DeltaNetLayerType::Attention {
                continue;
            }
            let layer = &mut cache.kv_cache.layers[i];
            debug_assert_eq!(
                layer.key.len(),
                infra.config.block_size * kvd,
                "key cache size mismatch at layer {i}"
            );
            if n_floats == 0 {
                continue;
            }
            if let Some(ref kh) = acts.layer_states[i].key_cache {
                // Download the full GPU buffer into a temp, then copy the valid
                // prefix. (cudarc memcpy_dtoh requires dst len == src len.)
                let mut tmp = vec![0.0f32; infra.config.block_size * kvd];
                infra
                    .stream
                    .memcpy_dtoh(kh, &mut tmp)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
                layer.key[..n_floats].copy_from_slice(&tmp[..n_floats]);
            }
            if let Some(ref vh) = acts.layer_states[i].value_cache {
                let mut tmp = vec![0.0f32; infra.config.block_size * kvd];
                infra
                    .stream
                    .memcpy_dtoh(vh, &mut tmp)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
                layer.value[..n_floats].copy_from_slice(&tmp[..n_floats]);
            }
        }

        Ok(())
    }

    /// **State restore** — the upload twin of
    /// [`Self::download_state_to_hybrid_cache`] (riir-train Plan 402 P1: the
    /// boundary sweep re-runs the query forward from an end-of-context
    /// checkpoint instead of re-prefilling ~40 tokens per patched layer —
    /// the 40× per-pair cost collapse).
    ///
    /// Uploads the CPU `HybridCache` (produced by the download method on the
    /// SAME forward, same `config` + `layer_types`) back into the GPU state
    /// buffers and restores `acts.pos` to the cached position count
    /// (`cache.kv_cache` position bookkeeping — callers pass `pos` explicitly
    /// because the CPU cache does not carry it).
    ///
    /// Only the first `pos` positions of the KV caches are uploaded (the rest
    /// is stale and will be overwritten by later forwards). Diagnostic /
    /// sweep-cadence — the htod copies make this too slow for a hot path.
    pub fn upload_state_from_hybrid_cache(
        &mut self,
        cache: &riir_infer_core::deltanet::forward::HybridCache,
        pos: usize,
    ) -> Result<(), CudarcKernelError> {
        let infra = &self.infra;
        let acts = &mut self.acts;

        infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        // 1. DeltaNet recurrent + conv states (wholesale).
        for i in 0..infra.config.n_layer {
            if infra.layer_types[i] != DeltaNetLayerType::DeltaNet {
                continue;
            }
            if let Some(ref mut handle) = acts.layer_states[i].deltanet_state {
                let src = &cache.deltanet_state.recurrent_states[i];
                debug_assert_eq!(
                    src.len(),
                    handle.len(),
                    "recurrent state size mismatch at layer {i}"
                );
                // Plan 603 R2 — `Half` narrows on the host (the `half` crate's
                // RN conversion + the kernel's saturating clamp for f16).
                handle
                    .upload_from_f32(&infra.stream, src)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            }
            if let Some(ref mut handle) = acts.layer_states[i].conv_state {
                let src = &cache.deltanet_state.conv_states[i];
                debug_assert_eq!(src.len(), handle.len(), "conv state size mismatch at layer {i}");
                infra
                    .stream
                    .memcpy_htod(src, handle)
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            }
        }

        // 2. Attention KV caches — first `pos` positions only.
        let kvd = infra.config.n_kv_head * infra.config.head_dim;
        let n_floats = pos * kvd;
        for i in 0..infra.config.n_layer {
            if infra.layer_types[i] != DeltaNetLayerType::Attention {
                continue;
            }
            let layer = &cache.kv_cache.layers[i];
            debug_assert_eq!(
                layer.key.len(),
                infra.config.block_size * kvd,
                "key cache size mismatch at layer {i}"
            );
            if n_floats == 0 {
                continue;
            }
            if let Some(ref mut kh) = acts.layer_states[i].key_cache {
                // cudarc memcpy_htod uploads exactly src.len() floats — upload
                // only the valid prefix via a slice view of the CPU buffer.
                infra
                    .stream
                    .memcpy_htod(&layer.key[..n_floats], &mut kh.slice_mut(..n_floats))
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            }
            if let Some(ref mut vh) = acts.layer_states[i].value_cache {
                infra
                    .stream
                    .memcpy_htod(&layer.value[..n_floats], &mut vh.slice_mut(..n_floats))
                    .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
            }
        }

        acts.pos = pos;
        Ok(())
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Free functions: the actual forward pass (split-borrow pattern)
// ───────────────────────────────────────────────────────────────────────────

// Issue 666 — LoRA forward context. Bundles references to the LoRA kernel,
// the per-layer adapter slots, and the norm_x side buffer so they can be
// threaded through `forward_layers` → `forward_deltanet_layer` without
// disturbing the existing borrow split between `infra` and `acts`.

/// Issue 781 — one attached per-layer adapter: the uploaded weights + its
/// rank scratch, kept together so attach/detach is a single slot swap
/// (scratch allocated at attach time, reused per token — GOAT G4).
struct LoraLayerSlot {
    lora: QvLoraGpuCudarc,
    ax_q: CudaSlice<f32>,
    ax_v: CudaSlice<f32>,
}

struct LoraFwdCtx<'a> {
    kernels: &'a LoraDecodeKernels,
    /// Per-layer adapter slots (len n_layer). `forward_deltanet_layer`
    /// resolves its own layer's slot; every frozen layer sees `None`.
    slots: &'a mut [Option<LoraLayerSlot>],
    norm_x: &'a CudaSlice<f32>,
}

/// Issue 781 — one layer's resolved adapter inside `forward_deltanet_layer`:
/// the shared side buffer (read), the rank scratch (written by the
/// down-projection kernel), and the uploaded weights (read). The borrows are
/// split from the `&mut LoraFwdCtx` field-wise so the kernel call can take
/// `&mut` scratch while reading the shared buffer + weights.
struct LoraLayerApply<'a> {
    kernels: &'a LoraDecodeKernels,
    norm_x: &'a CudaSlice<f32>,
    ax_q: &'a mut CudaSlice<f32>,
    ax_v: &'a mut CudaSlice<f32>,
    lora: &'a QvLoraGpuCudarc,
}

/// Issue 504 T1 — enabled-gated builder: when `enabled` is false (the
/// closed-loop training driver's frozen-forward windows), yields `None` even
/// with slots attached, so EVERY forward — decode and training — runs the
/// frozen backbone.
fn build_lora_ctx_enabled<'a>(
    kernels: &'a LoraDecodeKernels,
    slots: &'a mut [Option<LoraLayerSlot>],
    norm_x: &'a CudaSlice<f32>,
    enabled: bool,
) -> Option<LoraFwdCtx<'a>> {
    if enabled && slots.iter().any(Option::is_some) {
        Some(LoraFwdCtx {
            kernels,
            slots,
            norm_x,
        })
    } else {
        None
    }
}

fn forward_from_x(
    infra: &ForwardInfraCudarc,
    acts: &mut ActivationsCudarc,
) -> Result<(), CudarcKernelError> {
    forward_from_x_with_lora(infra, acts, None)
}

fn forward_from_x_with_lora(
    infra: &ForwardInfraCudarc,
    acts: &mut ActivationsCudarc,
    lora_ctx: Option<&mut LoraFwdCtx<'_>>,
) -> Result<(), CudarcKernelError> {
    let n = infra.config.n_embd;
    let eps = infra.config.rms_norm_eps as f32;

    forward_layers_with_lora(infra, acts, lora_ctx)?;

    // ── Final RMSNorm + lm_head (fused) ──
    // Issue 623 — fuse the final RMSNorm + quantize, then dispatch lm_head GEMV.
    //
    // Issue 980 T4 — on a folded model the split path applies: norm → f32
    // into `norm_x`, rotate IN PLACE there (`norm_x` is the lm_head's only
    // consumer on this path), quantize + GEMV. Logits come out primal.
    //
    // Issue 980 T4 escalation — K1 (3→1).
    if let Some(rot) = &infra.rotation {
        if rot.fused_geometry_ok(n) {
            rot.kernels.rmsnorm_rotate_quantize(
                &infra.stream,
                &acts.x,
                &infra.final_norm,
                rot.signs_for_width(n),
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                n,
                eps,
                rot.block_size,
            )?;
            gemv_prequantized_multi(
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &[(&infra.lm_head, &acts.logits)],
                false,
            )?;
        } else {
            infra.elementwise.launch_rmsnorm(
                &infra.stream,
                &acts.x,
                &infra.final_norm,
                &acts.norm_x,
                n,
                eps,
            )?;
            rot.kernels.fwht_rotate_forward(
                &infra.stream,
                &acts.norm_x,
                rot.signs_for_width(n),
                n,
                rot.block_size,
            )?;
            quantize_and_gemv_batch(
                &infra.elementwise,
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &acts.norm_x,
                &[(&infra.lm_head, &acts.logits)],
            )?;
        }
    } else {
        rmsnorm_quantize_and_gemv_batch(
            &infra.elementwise,
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &acts.x,
            &infra.final_norm,
            eps,
            n,
            &[(&infra.lm_head, &acts.logits)],
        )?;
    };

    // ── Issue 697 — GPU-side argmax over the final logits ──
    // (Eager path: same sequence as the graph path so `last_argmax()` is valid
    // after every forward. ~5 µs for vocab=248K.)
    infra
        .stream
        .memset_zeros(&mut acts.argmax_buf)
        .map_err(alloc_err)?;
    infra.elementwise.launch_argmax_first(
        &infra.stream,
        &acts.logits,
        infra.config.vocab_size,
        &acts.argmax_buf,
    )?;

    acts.pos += 1;
    Ok(())
}

/// Issue 634 — Runs the per-layer forward loop (layers 0..n_layer) but NOT the
/// final RMSNorm + lm_head. Extracted from `forward_from_x` so that
/// `forward_token_with_final_hidden` can reuse the exact same per-layer kernel
/// sequence while substituting its own sided final-norm dispatch.
///
/// Borrows are split (`infra` read-only, `acts` mutable) to satisfy the
/// borrow checker when callers need to also read `infra.final_norm` /
/// `infra.lm_head` after this returns.
/// Back-compat wrapper — delegates to [`forward_layers_with_lora`] with no
/// LoRA. Kept for callers that don't need LoRA (profiled path, devpos path).
#[allow(dead_code)]
fn forward_layers(
    infra: &ForwardInfraCudarc,
    acts: &mut ActivationsCudarc,
) -> Result<(), CudarcKernelError> {
    forward_layers_with_lora(infra, acts, None)
}

#[allow(dead_code)]
fn guard_no_rotation(
    infra: &ForwardInfraCudarc,
    entry: &str,
) -> Result<(), CudarcKernelError> {
    if infra.rotation.is_some() {
        return Err(CudarcKernelError::InvalidArg(format!(
            "{entry}: the Hadamard-folded (Bonsai-2) runtime is eager-path only (Issue 980 T4) — use forward_token()"
        )));
    }
    Ok(())
}

fn forward_layers_with_lora(
    infra: &ForwardInfraCudarc,
    acts: &mut ActivationsCudarc,
    mut lora_ctx: Option<&mut LoraFwdCtx<'_>>,
) -> Result<(), CudarcKernelError> {
    let n = infra.config.n_embd;
    let eps = infra.config.rms_norm_eps as f32;

    // Issue 641 — guard against KV cache overflow. Attention layers append at
    // `pos` and decode over `pos + 1` entries in a buffer of size
    // `block_size * kvd_attn`. When `pos >= block_size`, both the append and
    // the decode read/write out of bounds — silently corrupting attention
    // output (or crashing with CUDA_ERROR_ILLEGAL_ADDRESS). This debug_assert
    // catches the misconfiguration early in debug builds; the release path
    // trusts the caller to set block_size ≥ max sequence length.
    debug_assert!(
        acts.pos < infra.config.block_size,
        "position {} exceeds KV cache block_size {} — increase config.block_size",
        acts.pos,
        infra.config.block_size,
    );

    for layer_idx in 0..infra.config.n_layer {
        let layer_w = &infra.layers[layer_idx];
        let is_deltanet = infra.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

        // ── Pre-attention RMSNorm is fused into the layer functions' input
        // projection quantize (Issue 624). The layer functions call
        // rmsnorm_quantize_and_gemv_batch directly, saving 1 launch/layer.
        //
        // Issue 697 — the layer's out_proj GEMV accumulates directly into the
        // residual stream, so there is NO separate residual_add after the layer.
        if is_deltanet {
            forward_deltanet_layer(infra, acts, layer_idx, layer_w, eps, lora_ctx.as_deref_mut(), true)?;
        } else {
            forward_attention_layer(infra, acts, layer_idx, layer_w, eps, true)?;
        }

        // ── Post-attention RMSNorm + FFN gate/up (fused) ──
        // Issue 623 — fuse RMSNorm + quantize into one kernel, then dispatch
        // gate + up GEMVs from the same int8 buffer. Saves 1 launch + the
        // intermediate norm_x memory traffic.
        // (The post-attn residual was folded into the out_proj GEMV above.)
        //
        // Issue 980 T4 — on a folded model the split path applies: norm →
        // f32 (`norm_x`), rotate a copy, quantize + GEMV gate/up from it;
        // SwiGLU into `ffn_hidden`, rotate it in place (folded down input),
        // quantize + accumulate-GEMV down. The residual stream `x` stays
        // primal throughout (the down GEMV accumulates into it).
        //
        // Issue 980 T4 escalation — K1 + K5 restore the Issue-623/625 fusion
        // shapes with the rotation folded in (4→1 + 3→1 launches).
        if let Some(rot) = &infra.rotation {
            if rot.fused_geometry_ok(n) {
                rot.kernels.rmsnorm_rotate_quantize(
                    &infra.stream,
                    &acts.x,
                    &layer_w.post_attn_norm,
                    rot.signs_for_width(n),
                    &acts.quant_i8_buf,
                    &acts.ascale_buf,
                    n,
                    eps,
                    rot.block_size,
                )?;
                gemv_prequantized_multi(
                    &infra.stream,
                    &infra.gemv_multi,
                    &acts.quant_i8_buf,
                    &acts.ascale_buf,
                    &[
                        (&layer_w.gate_proj, &acts.ffn_gate),
                        (&layer_w.up_proj, &acts.ffn_up),
                    ],
                    false,
                )?;
                rot.kernels.swiglu_rotate_quantize(
                    &infra.stream,
                    &acts.ffn_gate,
                    &acts.ffn_up,
                    rot.signs_for_width(infra.config.mlp_hidden),
                    &acts.quant_i8_buf,
                    &acts.ascale_buf,
                    infra.config.mlp_hidden,
                    rot.block_size,
                )?;
                gemv_prequantized_multi(
                    &infra.stream,
                    &infra.gemv_multi,
                    &acts.quant_i8_buf,
                    &acts.ascale_buf,
                    &[(&layer_w.down_proj, &acts.x)],
                    true,
                )?;
                continue;
            }
            infra.elementwise.launch_rmsnorm(
                &infra.stream,
                &acts.x,
                &layer_w.post_attn_norm,
                &acts.norm_x,
                n,
                eps,
            )?;
            infra
                .stream
                .memcpy_dtod(&acts.norm_x, &mut acts.rot_scratch)
                .map_err(alloc_err)?;
            rot.kernels.fwht_rotate_forward(
                &infra.stream,
                &acts.rot_scratch,
                rot.signs_for_width(n),
                n,
                rot.block_size,
            )?;
            quantize_and_gemv_batch(
                &infra.elementwise,
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &acts.rot_scratch,
                &[
                    (&layer_w.gate_proj, &acts.ffn_gate),
                    (&layer_w.up_proj, &acts.ffn_up),
                ],
            )?;
            infra.elementwise.launch_swiglu(
                &infra.stream,
                &acts.ffn_gate,
                &acts.ffn_up,
                &acts.ffn_hidden,
                infra.config.mlp_hidden,
            )?;
            rot.kernels.fwht_rotate_forward(
                &infra.stream,
                &acts.ffn_hidden,
                rot.signs_for_width(infra.config.mlp_hidden),
                infra.config.mlp_hidden,
                rot.block_size,
            )?;
            infra.elementwise.launch_quantize(
                &infra.stream,
                &acts.ffn_hidden,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                infra.config.mlp_hidden,
            )?;
            gemv_prequantized_multi(
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &[(&layer_w.down_proj, &acts.x)],
                true,
            )?;
            continue;
        }
        rmsnorm_quantize_and_gemv_batch(
            &infra.elementwise,
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &acts.x,
            &layer_w.post_attn_norm,
            eps,
            n,
            &[
                (&layer_w.gate_proj, &acts.ffn_gate),
                (&layer_w.up_proj, &acts.ffn_up),
            ],
        )?;
        // Issue 625 — fuse SwiGLU + quantize into one kernel, then dispatch
        // down_proj GEMV. Saves 1 launch + intermediate ffn_hidden traffic.
        // Issue 697 — accumulate variant: down_proj adds directly into x, so
        // there is NO separate residual_add after the FFN.
        swiglu_quantize_and_gemv_accum(
            &infra.elementwise,
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &acts.ffn_gate,
            &acts.ffn_up,
            infra.config.mlp_hidden,
            &layer_w.down_proj,
            &acts.x,
        )?;
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────────
// Profiled forward (Issue 616): same kernels, with CUDA events per section.
// ───────────────────────────────────────────────────────────────────────────

/// Per-section GPU timing for one forward_token call.
/// All times in milliseconds. Categories are mutually exclusive (no overlap).
#[derive(Debug, Default, Clone)]
pub struct SectionProfile {
    /// Total GPU time from first kernel to logits sync.
    pub total_ms: f32,
    /// Pre-GEMV per layer: 2× RMSNorm + layer internals (deltanet/attention) + residual.
    pub pre_gemv_ms: f32,
    /// FFN GEMVs (gate_proj + up_proj + down_proj + SwiGLU) per layer.
    pub ffn_gemv_ms: f32,
    /// Residual add after FFN (negligible).
    pub residual_ms: f32,
    /// Final RMSNorm + lm_head GEMV.
    pub final_ms: f32,
    /// Count of GEMV dispatches (for per-launch average).
    pub gemv_count: u32,
}

/// Plan 603 R1 — the ternary/bonsai decode recurrence dispatch flag.
/// Issue 742's fused single-pass kernel (1R+1W over the persistent state vs
/// the row-parallel kernel's 3R+2W) is the DEFAULT, mirroring the qwen38
/// lane's `QWEN38_DN_FUSED` posture; `RIIR_BONSAI_DN_FUSED=0` restores the
/// row-parallel kernel as the A/B hatch. Both are bit-identical (same
/// per-element op order + same __shfl_xor butterfly — the Issue-742
/// bit-identity argument, verified on the real model by the league pins).
fn dn_recurrence_fused_enabled() -> bool {
    static FUSED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FUSED.get_or_init(|| {
        std::env::var("RIIR_BONSAI_DN_FUSED").map(|v| v != "0").unwrap_or(true)
    })
}

fn forward_deltanet_layer(
    infra: &ForwardInfraCudarc,
    acts: &mut ActivationsCudarc,
    layer_idx: usize,
    layer_w: &GpuLayerWeightsCudarc,
    eps: f32,
    lora_ctx: Option<&mut LoraFwdCtx<'_>>,
    accumulate_out: bool,
) -> Result<(), CudarcKernelError> {
    let n_v_heads = infra.config.deltanet_linear_n_value_heads;
    let head_dim = infra.config.deltanet_linear_head_dim;
    let n_k_heads = infra.config.deltanet_linear_n_heads;
    let q_dim = n_k_heads * head_dim;
    let z_dim = n_v_heads * head_dim;
    let conv_dim = 2 * q_dim + z_dim;
    let kernel_size = infra.config.deltanet_conv_kernel_size;

    let qkv_w = layer_w.in_proj_qkv.as_ref().expect("in_proj_qkv");
    let z_w = layer_w.in_proj_z.as_ref().expect("in_proj_z");
    let out_w = layer_w.out_proj.as_ref().expect("out_proj");
    let conv1d_w = layer_w.conv1d_weight.as_ref().expect("conv1d_weight");
    let a_log = layer_w.a_log.as_ref().expect("a_log");
    let dt_bias = layer_w.dt_bias.as_ref().expect("dt_bias");
    let linear_norm = layer_w.linear_norm.as_ref().expect("linear_norm");

    // Issue 666/781 — when THIS layer has an attached adapter, use the
    // `_with_norm_x` RMSNorm variant (writes the f32 post-norm hidden state to
    // a side buffer) so the LoRA kernel can read `norm_x` for its
    // down-projection. Then apply the LoRA delta to the Q+V slices of
    // `acts.qkv` BEFORE conv1d. Field-wise borrow split through the `&mut`
    // ctx: `slots` (mutable — the rank scratch is kernel-written) vs
    // `norm_x`/`kernels` (shared) are disjoint.
    let layer_lora: Option<LoraLayerApply<'_>> = match lora_ctx {
        Some(ctx) => ctx.slots[layer_idx].as_mut().map(|slot| LoraLayerApply {
                kernels: ctx.kernels,
                norm_x: ctx.norm_x,
                ax_q: &mut slot.ax_q,
                ax_v: &mut slot.ax_v,
                lora: &slot.lora,
            }),
        None => None,
    };

    // 1-4. Input projections: qkv, z, a, b — fused pre-attn RMSNorm + quantize.
    // Issue 624 — fuse the pre-attention RMSNorm + quantize into one kernel,
    // then dispatch all 4 GEMVs from the same int8 buffer. Saves 1 launch
    // per layer + eliminates the intermediate f32 norm_x memory traffic.
    // (Previously: separate launch_rmsnorm in forward_from_x + quantize here.)
    //
    // Issue 980 T4 — on a folded model the split path applies: norm → f32
    // (`norm_x`), a rotated COPY feeds the folded qkv/z GEMVs, and the dense
    // escape-set a/b run as fp32 GEMVs on the PRIMAL normed input. LoRA
    // refuses here (the adapter path is primal-basis-shaped and this branch
    // replaces its fused kernel).
    if let Some(rot) = &infra.rotation {
        if layer_lora.is_some() {
            return Err(CudarcKernelError::InvalidArg(
                "LoRA on a Hadamard-folded (Bonsai-2) DeltaNet layer is not supported (Issue 980 T4)".into(),
            ));
        }
        let n = infra.config.n_embd;
        let d_a = layer_w.dense_a.as_ref().expect(
            "folded model: dense ssm_alpha missing (loader contract guarantees the escape set)",
        );
        let d_b = layer_w.dense_b.as_ref().expect(
            "folded model: dense ssm_beta missing (loader contract guarantees the escape set)",
        );
        // Issue 980 T4 escalation — K2, the whole input stage in ONE
        // heterogeneous launch (Bench 940: the split path's 6 launches/layer
        // at the ~3 µs in-graph dispatch floor was the +1.90 ms/token cost).
        // The a/b GEMV blocks fold the normalization INLINE (each re-runs the
        // full-row reduction redundantly — bit-identical inv_rms, no
        // cross-block dependency); the q/kv/z blocks write rotated int8.
        if rot.fused_geometry_ok(n) {
            rot.kernels.gdn_input_fused(
                &infra.stream,
                &acts.x,
                &layer_w.input_norm,
                rot.signs_for_width(n),
                d_a,
                d_b,
                &acts.a_raw,
                &acts.b_raw,
                n_v_heads,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                n,
                eps,
                rot.block_size,
            )?;
            gemv_prequantized_multi(
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &[(qkv_w, &acts.qkv), (z_w, &acts.z_buf)],
                false,
            )?;
        } else {
            // Split fallback (odd Hadamard geometry — not Bonsai-2's 1024).
            infra.elementwise.launch_rmsnorm(
                &infra.stream,
                &acts.x,
                &layer_w.input_norm,
                &acts.norm_x,
                n,
                eps,
            )?;
            infra
                .stream
                .memcpy_dtod(&acts.norm_x, &mut acts.rot_scratch)
                .map_err(alloc_err)?;
            rot.kernels.fwht_rotate_forward(
                &infra.stream,
                &acts.rot_scratch,
                rot.signs_for_width(n),
                n,
                rot.block_size,
            )?;
            quantize_and_gemv_batch(
                &infra.elementwise,
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &acts.rot_scratch,
                &[(qkv_w, &acts.qkv), (z_w, &acts.z_buf)],
            )?;
            rot.kernels.gemv_dense(&infra.stream, d_a, &acts.norm_x, &acts.a_raw, n_v_heads, n)?;
            rot.kernels.gemv_dense(&infra.stream, d_b, &acts.norm_x, &acts.b_raw, n_v_heads, n)?;
        }
    } else if let Some(la) = layer_lora {
        // Ternary a/b exist only on pre-rotation files (the folded branch uses
        // dense_a/dense_b above) — resolve them lazily per branch (Issue 980 T4).
        let a_w = layer_w.in_proj_a.as_ref().expect("in_proj_a");
        let b_w = layer_w.in_proj_b.as_ref().expect("in_proj_b");
        rmsnorm_quantize_and_gemv_batch_with_norm_x(
            &infra.elementwise, &infra.stream, &infra.gemv_multi,
            &acts.quant_i8_buf, &acts.ascale_buf,
            &acts.x, &layer_w.input_norm, la.norm_x, eps, infra.config.n_embd,
            &[
                (qkv_w, &acts.qkv),
                (z_w, &acts.z_buf),
                (a_w, &acts.a_raw),
                (b_w, &acts.b_raw),
            ],
        )?;
        // Apply the Q+V LoRA correction to `acts.qkv` in-place.
        let k_dim = q_dim; // DeltaNet: k_dim == q_dim
        la.kernels.launch_qv_apply(
            &infra.stream,
            la.norm_x,
            &mut acts.qkv,
            la.ax_q,
            la.ax_v,
            &la.lora.a_q, &la.lora.b_q, &la.lora.a_v, &la.lora.b_v,
            la.lora.n_embd, la.lora.q_dim, k_dim, la.lora.v_dim,
            la.lora.rank, la.lora.scale,
        )?;
    } else {
        let a_w = layer_w.in_proj_a.as_ref().expect("in_proj_a");
        let b_w = layer_w.in_proj_b.as_ref().expect("in_proj_b");
        rmsnorm_quantize_and_gemv_batch(
            &infra.elementwise, &infra.stream, &infra.gemv_multi,
            &acts.quant_i8_buf, &acts.ascale_buf,
            &acts.x, &layer_w.input_norm, eps, infra.config.n_embd,
            &[
                (qkv_w, &acts.qkv),
                (z_w, &acts.z_buf),
                (a_w, &acts.a_raw),
                (b_w, &acts.b_raw),
            ],
        )?;
    }

    // 5. Conv1d + SiLU (in-place on qkv, updates conv_state).
    let conv_state = acts.layer_states[layer_idx]
        .conv_state
        .as_ref()
        .expect("conv_state");
    infra.deltanet.launch_conv1d(
        &infra.stream,
        &acts.qkv,
        conv1d_w,
        conv_state,
        conv_dim,
        kernel_size,
    )?;

    // 6. Beta/decay.
    infra.deltanet.launch_beta_decay(
        &infra.stream,
        &acts.a_raw,
        &acts.b_raw,
        a_log,
        dt_bias,
        &acts.beta_buf,
        &acts.decay_buf,
        n_v_heads,
    )?;

    // 7. Expand Q/K + L2-norm + copy V.
    infra.deltanet.launch_expand_and_l2_normalize(
        &infra.stream,
        &acts.qkv,
        &acts.qkv_expanded,
        n_k_heads,
        n_v_heads,
        head_dim,
    )?;

    // 8. Recurrence (Plan 603 R1 — Issue 742's fused single-pass, 1R+1W over
    // the state, DEFAULT here as in the qwen38 lane; bit-identical to the
    // parallel kernel — same per-element op order + same __shfl_xor
    // butterfly; the `RIIR_BONSAI_DN_FUSED=0` hatch restores the row-parallel
    // kernel, non-{64,128,256} head_dims always fall back to it).
    // Plan 603 R2 — the `Half` state arm dispatches the half-residency twin
    // (same op order, f32 compute; only the state STORE rounds — see the
    // kernel comment in cudarc_kernels/deltanet.rs).
    let state = acts.layer_states[layer_idx]
        .deltanet_state
        .as_ref()
        .expect("deltanet_state");
    match state {
        RecStateBuf::Half(h) => {
            infra.deltanet.launch_recurrence_fused_half_hd128(
                &infra.stream,
                &acts.qkv_expanded,
                &acts.beta_buf,
                &acts.decay_buf,
                h,
                &acts.recurrent_out,
                n_v_heads,
                RecStateBuf::half_fmt_from_env().unwrap_or(HalfStateFmt::F16),
            )?;
        }
        RecStateBuf::F32(state_f32) if dn_recurrence_fused_enabled() && matches!(head_dim, 64 | 128 | 256) => {
            infra.deltanet.launch_recurrence_fused(
                &infra.stream,
                &acts.qkv_expanded,
                &acts.beta_buf,
                &acts.decay_buf,
                state_f32,
                &acts.recurrent_out,
                head_dim,
                n_v_heads,
            )?;
        }
        RecStateBuf::F32(state_f32) if cfg!(feature = "deltanet_recurrence_parallel") => {
            infra.deltanet.launch_recurrence_parallel(
                &infra.stream,
                &acts.qkv_expanded,
                &acts.beta_buf,
                &acts.decay_buf,
                state_f32,
                &acts.recurrent_out,
                head_dim,
                n_v_heads,
            )?;
        }
        RecStateBuf::F32(state_f32) => {
            infra.deltanet.launch_recurrence(
                &infra.stream,
                &acts.qkv_expanded,
                &acts.beta_buf,
                &acts.decay_buf,
                state_f32,
                &acts.recurrent_out,
                head_dim,
                n_v_heads,
            )?;
        }
    }

    // 9–11. Per-head RMSNorm + z gating + output projection (ALL FUSED with
    // quantize). Issue 627 — fuse the per-head RMSNorm + `silu(z) *` + int8
    // quantize into one kernel, then dispatch the dp4a out_proj GEMV. Saves
    // 1 launch vs Issue 626 (which had separate RMSNorm + gate_silu_quantize)
    // + eliminates the intermediate f32 normalized `recurrent_out` round-trip
    // through HBM. The RMSNorm reduction tree is identical to
    // `rmsnorm_batched_f32`, producing bit-identical inv_rms.
    //
    // Issue 697 — with `accumulate_out`, the out_proj GEMV accumulates
    // directly into the residual stream `x` (out[row] += acc), eliminating
    // the separate residual_add launch + the tmp round-trip. The f32 add is
    // identical to what residual_add_f32 computed (same operands, same order).
    //
    // Issue 980 T4 — on a folded model the rotation splits this fusion:
    // per-head norm and silu gate run standalone (they precede the rotation
    // and the gate pairing is pre-permute), then the grouped-head permute +
    // sign+FWHT run in place on `recurrent_out`, then quantize + GEMV
    // (accum or tmp, same as the fused arms).
    if let Some(rot) = &infra.rotation {
        let v_dim = n_v_heads * head_dim;
        // Issue 980 T4 escalation — K3, the whole output chain in ONE kernel
        // (per-head norm + silu(z) gate + tiled→grouped permute + sign + FWHT
        // + quantize; 6 launches → 1). rep=1 degenerates the head mapping to
        // identity for ungrouped geometries.
        if rot.gdn_out_fused_ok(v_dim, head_dim) {
            let rep = if rot.gdn_v_grouped {
                n_v_heads / rot.gdn_k_groups
            } else {
                1
            };
            rot.kernels.gdn_out_fused(
                &infra.stream,
                &acts.recurrent_out,
                &acts.z_buf,
                linear_norm,
                rot.signs_for_width(v_dim),
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                v_dim,
                head_dim,
                rot.gdn_k_groups,
                rep,
                eps,
                rot.block_size,
            )?;
        } else {
            // Split fallback (odd geometry — not Bonsai-2's shapes).
            infra.attention.launch_rmsnorm_batched(
                &infra.stream,
                &acts.recurrent_out,
                linear_norm,
                &acts.recurrent_out,
                n_v_heads,
                head_dim,
                eps,
            )?;
            rot.kernels
                .gate_silu(&infra.stream, &acts.recurrent_out, &acts.z_buf, v_dim)?;
            if rot.gdn_v_grouped {
                infra
                    .stream
                    .memcpy_dtod(&acts.recurrent_out, &mut acts.permute_tmp)
                    .map_err(alloc_err)?;
                rot.kernels.gdn_v_permute(
                    &infra.stream,
                    &acts.permute_tmp,
                    &acts.recurrent_out,
                    v_dim,
                    rot.gdn_v_heads,
                    rot.gdn_k_groups,
                )?;
            }
            rot.kernels.fwht_rotate_forward(
                &infra.stream,
                &acts.recurrent_out,
                rot.signs_for_width(v_dim),
                v_dim,
                rot.block_size,
            )?;
            infra.elementwise.launch_quantize(
                &infra.stream,
                &acts.recurrent_out,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                v_dim,
            )?;
        }
        gemv_prequantized_multi(
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &[(out_w, if accumulate_out { &acts.x } else { &acts.tmp })],
            accumulate_out,
        )?;
    } else if accumulate_out {
        rmsnorm_gate_silu_quantize_and_gemv_accum(
            &infra.elementwise,
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &acts.recurrent_out,
            &acts.z_buf,
            linear_norm,
            n_v_heads,
            head_dim,
            eps,
            out_w,
            &acts.x,
        )?;
    } else {
        rmsnorm_gate_silu_quantize_and_gemv(
            &infra.elementwise,
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &acts.recurrent_out,
            &acts.z_buf,
            linear_norm,
            n_v_heads,
            head_dim,
            eps,
            out_w,
            &acts.tmp,
        )?;
    }
    Ok(())
}

/// Issue 492 T1 GPU lane - which attention (layer, head) the capture tap
/// samples. The position is whatever `acts.pos` the training forward has
/// reached when the tap fires (the caller owns the sampling cadence).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionProbeSpec {
    /// Layer index - must be an `Attention` layer (validated on entry).
    pub layer: usize,
    /// Head index within the layer (`0..n_head`).
    pub head: usize,
}

/// Issue 492 T1 GPU lane - one captured attention window plus the
/// training kernel's output for the probed head, downloaded host-side
/// for the numeric-drift probe (riir-train-engine `numeric_drift`).
///
/// `k_cache` / `v_cache` carry the position-ordered rows `0..=position`
/// at stride `kvd` - the exact shape contract the probe's
/// `AttentionHeadWindow` reads - and `kv_off` is the probed head's GQA
/// group offset within one cache row, so the consumer never needs the
/// model config.
#[derive(Debug, Default, Clone)]
pub struct AttentionProbeCapture {
    /// Token position the window was captured at.
    pub position: usize,
    /// Per-head dimension (`config.head_dim`).
    pub head_dim: usize,
    /// KV cache row stride (`n_kv_head * head_dim`).
    pub kvd: usize,
    /// The probed head's KV group offset within one cache row.
    pub kv_off: usize,
    /// `[head_dim]` - post-RoPE query at the probed position.
    pub q_head: Vec<f32>,
    /// `[(position + 1) * kvd]` - key cache rows `0..=position`.
    pub k_cache: Vec<f32>,
    /// `[(position + 1) * kvd]` - value cache rows `0..=position`.
    pub v_cache: Vec<f32>,
    /// `[head_dim]` - the attention decode kernel's output for the
    /// probed head (the kernel-under-test arm).
    pub attn_out_head: Vec<f32>,
}

fn forward_attention_layer(
    infra: &ForwardInfraCudarc,
    acts: &mut ActivationsCudarc,
    layer_idx: usize,
    layer_w: &GpuLayerWeightsCudarc,
    eps: f32,
    accumulate_out: bool,
) -> Result<(), CudarcKernelError> {
    let n_head = infra.config.n_head;
    let n_kv = infra.config.n_kv_head;
    let hd = infra.config.head_dim;
    let q_dim = n_head * hd;
    let kvd = n_kv * hd;
    let rotary_dim = if infra.config.rope_dimension_count > 0 {
        infra.config.rope_dimension_count
    } else {
        hd
    };
    let theta_base = infra.config.rope_theta;
    let pos = acts.pos;

    let wq = layer_w.attn_wq.as_ref().expect("attn_wq");
    let wk = layer_w.attn_wk.as_ref().expect("attn_wk");
    let wv = layer_w.attn_wv.as_ref().expect("attn_wv");
    let wo = layer_w.attn_wo.as_ref().expect("attn_wo");
    let q_norm = layer_w.attn_q_norm.as_ref().expect("attn_q_norm");
    let k_norm = layer_w.attn_k_norm.as_ref().expect("attn_k_norm");

    // 1. Q (gated), K, V projections — fused pre-attn RMSNorm + quantize.
    // Issue 624 — fuse the pre-attention RMSNorm + quantize into one kernel,
    // then dispatch all 3 GEMVs from the same int8 buffer. Saves 1 launch
    // per layer + eliminates the intermediate f32 norm_x memory traffic.
    //
    // Issue 980 T4 — on a folded model the split path applies: norm → f32,
    // rotate a copy, quantize + GEMV the folded q/k/v from it. (Attention
    // layers carry no a/b — LoRA is DeltaNet-only by `set_lora`'s check.)
    if let Some(rot) = &infra.rotation {
        let n = infra.config.n_embd;
        // Issue 980 T4 escalation — K1 (rmsnorm+sign+FWHT+quantize, 4→1).
        if rot.fused_geometry_ok(n) {
            rot.kernels.rmsnorm_rotate_quantize(
                &infra.stream,
                &acts.x,
                &layer_w.input_norm,
                rot.signs_for_width(n),
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                n,
                eps,
                rot.block_size,
            )?;
            gemv_prequantized_multi(
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &[
                    (wq, &acts.attn_qg),
                    (wk, &acts.attn_k),
                    (wv, &acts.attn_v),
                ],
                false,
            )?;
        } else {
            // Split fallback (odd Hadamard geometry — not Bonsai-2's 1024).
            infra.elementwise.launch_rmsnorm(
                &infra.stream,
                &acts.x,
                &layer_w.input_norm,
                &acts.norm_x,
                n,
                eps,
            )?;
            infra
                .stream
                .memcpy_dtod(&acts.norm_x, &mut acts.rot_scratch)
                .map_err(alloc_err)?;
            rot.kernels.fwht_rotate_forward(
                &infra.stream,
                &acts.rot_scratch,
                rot.signs_for_width(n),
                n,
                rot.block_size,
            )?;
            quantize_and_gemv_batch(
                &infra.elementwise,
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &acts.rot_scratch,
                &[
                    (wq, &acts.attn_qg),
                    (wk, &acts.attn_k),
                    (wv, &acts.attn_v),
                ],
            )?;
        }
    } else {
        rmsnorm_quantize_and_gemv_batch(
            &infra.elementwise, &infra.stream, &infra.gemv_multi,
            &acts.quant_i8_buf, &acts.ascale_buf,
            &acts.x, &layer_w.input_norm, eps, infra.config.n_embd,
            &[
                (wq, &acts.attn_qg),
                (wk, &acts.attn_k),
                (wv, &acts.attn_v),
            ],
        )?;
    }

    // 2. Split QG into Q and gate.
    infra.attention.launch_split_qg(
        &infra.stream,
        &acts.attn_qg,
        &acts.attn_q,
        &acts.attn_gate,
        hd,
        n_head,
    )?;

    // 3. Per-head RMSNorm on Q and K (in-place).
    infra.attention.launch_rmsnorm_batched(
        &infra.stream,
        &acts.attn_q,
        q_norm,
        &acts.attn_q,
        n_head,
        hd,
        eps,
    )?;
    infra.attention.launch_rmsnorm_batched(
        &infra.stream,
        &acts.attn_k,
        k_norm,
        &acts.attn_k,
        n_kv,
        hd,
        eps,
    )?;

    // 4. Partial RoPE on Q and K (in-place).
    infra.attention.launch_rope(
        &infra.stream,
        &acts.attn_q,
        &acts.attn_k,
        rotary_dim,
        hd,
        n_head,
        n_kv,
        pos,
        theta_base,
    )?;

    // 5. Append K, V to KV cache.
    let key_cache = acts.layer_states[layer_idx]
        .key_cache
        .as_ref()
        .expect("key_cache");
    let value_cache = acts.layer_states[layer_idx]
        .value_cache
        .as_ref()
        .expect("value_cache");
    infra.attention.launch_kv_cache_append(
        &infra.stream,
        &acts.attn_k,
        &acts.attn_v,
        key_cache,
        value_cache,
        kvd,
        pos,
    )?;

    // 6. Flash attention decode.
    let n_positions = pos + 1;
    infra.attention.launch_attention_decode(
        &infra.stream,
        &acts.attn_q,
        key_cache,
        value_cache,
        &acts.attn_out,
        hd,
        n_head,
        n_kv,
        n_positions,
    )?;

    // 7–8. Output gating + output projection (FUSED with quantize).
    // Issue 626 — fuse `sigmoid(gate) * attn_out` + int8 quantize into one
    // kernel, then dispatch the dp4a wo GEMV. Saves 1 launch + the
    // intermediate f32 gated attn_out round-trip through HBM.
    // Issue 697 — accumulate variant folds the residual add into the GEMV.
    //
    // Issue 980 T4 — on a folded model the gate splits out (pre-rotation
    // pairing), the rotation runs in place on `attn_out` (consumed only
    // here; the output is primal), then quantize + GEMV as usual.
    //
    // Issue 980 T4 escalation — K4 fuses gate+sign+FWHT+quantize (3→1).
    if let Some(rot) = &infra.rotation {
        if rot.fused_geometry_ok(q_dim) {
            rot.kernels.gate_sigmoid_rotate_quantize(
                &infra.stream,
                &acts.attn_out,
                &acts.attn_gate,
                rot.signs_for_width(q_dim),
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                q_dim,
                rot.block_size,
            )?;
        } else {
            rot.kernels
                .gate_sigmoid(&infra.stream, &acts.attn_out, &acts.attn_gate, q_dim)?;
            rot.kernels.fwht_rotate_forward(
                &infra.stream,
                &acts.attn_out,
                rot.signs_for_width(q_dim),
                q_dim,
                rot.block_size,
            )?;
            infra.elementwise.launch_quantize(
                &infra.stream,
                &acts.attn_out,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                q_dim,
            )?;
        }
        gemv_prequantized_multi(
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &[(wo, if accumulate_out { &acts.x } else { &acts.tmp })],
            accumulate_out,
        )?;
    } else if accumulate_out {
        gate_sigmoid_quantize_and_gemv_accum(
            &infra.elementwise,
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &acts.attn_out,
            &acts.attn_gate,
            q_dim,
            wo,
            &acts.x,
        )?;
    } else {
        gate_sigmoid_quantize_and_gemv(
            &infra.elementwise,
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &acts.attn_out,
            &acts.attn_gate,
            q_dim,
            wo,
            &acts.tmp,
        )?;
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────────
// Training-mode layer functions (Issue 641 T2)
// ───────────────────────────────────────────────────────────────────────────
//
// Identical to the hot-path layer functions EXCEPT the input RMSNorm uses
// `rmsnorm_quantize_and_gemv_batch_with_norm_x` (writes f32 norm_x to
// `acts.final_norm_x` side buffer) instead of the un-sided variant. This lets
// `forward_token_training` download the GPU-computed norm_x for the minimal
// activation cache. Everything else (conv1d, recurrence, RoPE, KV cache,
// output proj, etc.) is identical to the hot path — only the input-norm step
// differs.

/// Training-mode DeltaNet layer forward — same as [`forward_deltanet_layer`]
/// but writes `norm_x` to `acts.final_norm_x` during the input RMSNorm step.
#[allow(clippy::too_many_arguments)]
fn forward_deltanet_layer_training(
    infra: &ForwardInfraCudarc,
    acts: &mut ActivationsCudarc,
    layer_idx: usize,
    layer_w: &GpuLayerWeightsCudarc,
    eps: f32,
    // Issue 504 T1 (riir-train Issue 729 option 1) — attached-adapter context.
    // When this layer has a slot AND the master switch is on, the Q+V LoRA
    // correction applies here exactly as the decode path applies it in
    // `forward_deltanet_layer`: added to `acts.qkv` BEFORE conv1d, from the
    // post-RMSNorm `norm_x` this training variant already writes to
    // `acts.final_norm_x` (the activation cache's per-layer norm_x — the same
    // buffer `compute_lora_gradients` consumes host-side). The LoRA-grad math
    // downstream then differentiates the TRUE forward: dY is the post-adapter
    // upstream gradient, dA/dB use the pre-adapter norm_x. The one term NOT
    // modeled is the adapter's contribution to the upstream dX (the backward
    // propagates through the frozen weights only) — identical first-order
    // semantics to the CPU closed-loop path, which also computes LoRA grads
    // downstream of a frozen-weight backward (`qwen_deltanet_model_backward`).
    lora_ctx: Option<&mut LoraFwdCtx<'_>>,
) -> Result<(), CudarcKernelError> {
    let n_v_heads = infra.config.deltanet_linear_n_value_heads;
    let head_dim = infra.config.deltanet_linear_head_dim;
    let n_k_heads = infra.config.deltanet_linear_n_heads;
    let q_dim = n_k_heads * head_dim;
    let z_dim = n_v_heads * head_dim;
    let conv_dim = 2 * q_dim + z_dim;
    let kernel_size = infra.config.deltanet_conv_kernel_size;

    let qkv_w = layer_w.in_proj_qkv.as_ref().expect("in_proj_qkv");
    let z_w = layer_w.in_proj_z.as_ref().expect("in_proj_z");
    let a_w = layer_w.in_proj_a.as_ref().expect("in_proj_a");
    let b_w = layer_w.in_proj_b.as_ref().expect("in_proj_b");
    let out_w = layer_w.out_proj.as_ref().expect("out_proj");
    let conv1d_w = layer_w.conv1d_weight.as_ref().expect("conv1d_weight");
    let a_log = layer_w.a_log.as_ref().expect("a_log");
    let dt_bias = layer_w.dt_bias.as_ref().expect("dt_bias");
    let linear_norm = layer_w.linear_norm.as_ref().expect("linear_norm");

    // 1-4. Input projections with norm_x side-buffer write (training mode).
    rmsnorm_quantize_and_gemv_batch_with_norm_x(
        &infra.elementwise, &infra.stream, &infra.gemv_multi,
        &acts.quant_i8_buf, &acts.ascale_buf,
        &acts.x, &layer_w.input_norm, &acts.final_norm_x, eps, infra.config.n_embd,
        &[
            (qkv_w, &acts.qkv),
            (z_w, &acts.z_buf),
            (a_w, &acts.a_raw),
            (b_w, &acts.b_raw),
        ],
    )?;

    // Issue 504 T1 — apply the Q+V LoRA correction to `acts.qkv` in place,
    // BEFORE conv1d (same insertion point as the decode path). The norm_x
    // source is `acts.final_norm_x` — this training variant's with_norm_x
    // destination, which the activation cache downloads right after the
    // layer returns (so the cached norm_x stays the PRE-adapter projection
    // input, exactly what the host-side LoRA-grad math consumes).
    if let Some(ctx) = lora_ctx
        && let Some(slot) = ctx.slots[layer_idx].as_mut() {
            let k_dim = q_dim; // DeltaNet: k_dim == q_dim
            ctx.kernels.launch_qv_apply(
                &infra.stream,
                &acts.final_norm_x,
                &mut acts.qkv,
                &mut slot.ax_q,
                &mut slot.ax_v,
                &slot.lora.a_q, &slot.lora.b_q, &slot.lora.a_v, &slot.lora.b_v,
                slot.lora.n_embd, slot.lora.q_dim, k_dim, slot.lora.v_dim,
                slot.lora.rank, slot.lora.scale,
            )?;
        }

    // 5. Conv1d + SiLU
    let conv_state = acts.layer_states[layer_idx]
        .conv_state.as_ref().expect("conv_state");
    infra.deltanet.launch_conv1d(
        &infra.stream, &acts.qkv, conv1d_w, conv_state, conv_dim, kernel_size,
    )?;

    // 6. Beta/decay
    infra.deltanet.launch_beta_decay(
        &infra.stream, &acts.a_raw, &acts.b_raw, a_log, dt_bias,
        &acts.beta_buf, &acts.decay_buf, n_v_heads,
    )?;

    // 7. Expand Q/K + L2-norm + copy V
    infra.deltanet.launch_expand_and_l2_normalize(
        &infra.stream, &acts.qkv, &acts.qkv_expanded, n_k_heads, n_v_heads, head_dim,
    )?;

    // 8. Recurrence (Plan 603 R1 — fused single-pass default, the same
    // dispatch as the decode path above). Plan 603 R2 — the training
    // variant REFUSES the half-state lane: gradient training against
    // half-resident state silently changes training numerics; the lane is
    // a decode-surface lever.
    let state = acts.layer_states[layer_idx]
        .deltanet_state
        .as_ref()
        .expect("deltanet_state");
    let state_f32 = match state {
        RecStateBuf::F32(s) => s,
        RecStateBuf::Half(_) => {
            return Err(CudarcKernelError::InvalidArg(
                "RIIR_GDN_STATE_HALF is a decode lane; training forwards require f32 state (unset it)".into(),
            ));
        }
    };
    if dn_recurrence_fused_enabled() && matches!(head_dim, 64 | 128 | 256) {
        infra.deltanet.launch_recurrence_fused(
            &infra.stream, &acts.qkv_expanded, &acts.beta_buf, &acts.decay_buf,
            state_f32, &acts.recurrent_out, head_dim, n_v_heads,
        )?;
    } else if cfg!(feature = "deltanet_recurrence_parallel") {
        infra.deltanet.launch_recurrence_parallel(
            &infra.stream, &acts.qkv_expanded, &acts.beta_buf, &acts.decay_buf,
            state_f32, &acts.recurrent_out, head_dim, n_v_heads,
        )?;
    } else {
        infra.deltanet.launch_recurrence(
            &infra.stream, &acts.qkv_expanded, &acts.beta_buf, &acts.decay_buf,
            state_f32, &acts.recurrent_out, head_dim, n_v_heads,
        )?;
    }

    // 9-11. Per-head RMSNorm + z gating + output projection (FUSED).
    rmsnorm_gate_silu_quantize_and_gemv(
        &infra.elementwise, &infra.stream, &infra.gemv_multi,
        &acts.quant_i8_buf, &acts.ascale_buf,
        &acts.recurrent_out, &acts.z_buf, linear_norm,
        n_v_heads, head_dim, eps, out_w, &acts.tmp,
    )?;
    Ok(())
}

/// Training-mode attention layer forward — same as [`forward_attention_layer`]
/// but writes `norm_x` to `acts.final_norm_x` during the input RMSNorm step.
#[allow(clippy::too_many_arguments)]
fn forward_attention_layer_training(
    infra: &ForwardInfraCudarc,
    acts: &mut ActivationsCudarc,
    layer_idx: usize,
    layer_w: &GpuLayerWeightsCudarc,
    eps: f32,
    probe: Option<(&AttentionProbeSpec, &mut AttentionProbeCapture)>,
) -> Result<(), CudarcKernelError> {
    let n_head = infra.config.n_head;
    let n_kv = infra.config.n_kv_head;
    let hd = infra.config.head_dim;
    let q_dim = n_head * hd;
    let kvd = n_kv * hd;
    let rotary_dim = if infra.config.rope_dimension_count > 0 {
        infra.config.rope_dimension_count
    } else {
        hd
    };
    let theta_base = infra.config.rope_theta;
    let pos = acts.pos;

    let wq = layer_w.attn_wq.as_ref().expect("attn_wq");
    let wk = layer_w.attn_wk.as_ref().expect("attn_wk");
    let wv = layer_w.attn_wv.as_ref().expect("attn_wv");
    let wo = layer_w.attn_wo.as_ref().expect("attn_wo");
    let q_norm = layer_w.attn_q_norm.as_ref().expect("attn_q_norm");
    let k_norm = layer_w.attn_k_norm.as_ref().expect("attn_k_norm");

    // 1. Q (gated), K, V projections with norm_x side-buffer write (training).
    rmsnorm_quantize_and_gemv_batch_with_norm_x(
        &infra.elementwise, &infra.stream, &infra.gemv_multi,
        &acts.quant_i8_buf, &acts.ascale_buf,
        &acts.x, &layer_w.input_norm, &acts.final_norm_x, eps, infra.config.n_embd,
        &[
            (wq, &acts.attn_qg),
            (wk, &acts.attn_k),
            (wv, &acts.attn_v),
        ],
    )?;

    // 2. Split QG into Q and gate
    infra.attention.launch_split_qg(
        &infra.stream, &acts.attn_qg, &acts.attn_q, &acts.attn_gate, hd, n_head,
    )?;

    // 3. Per-head RMSNorm on Q and K (in-place)
    infra.attention.launch_rmsnorm_batched(
        &infra.stream, &acts.attn_q, q_norm, &acts.attn_q, n_head, hd, eps,
    )?;
    infra.attention.launch_rmsnorm_batched(
        &infra.stream, &acts.attn_k, k_norm, &acts.attn_k, n_kv, hd, eps,
    )?;

    // 4. Partial RoPE on Q and K (in-place)
    infra.attention.launch_rope(
        &infra.stream, &acts.attn_q, &acts.attn_k, rotary_dim, hd, n_head, n_kv,
        pos, theta_base,
    )?;

    // 5. Append K, V to KV cache
    let key_cache = acts.layer_states[layer_idx]
        .key_cache.as_ref().expect("key_cache");
    let value_cache = acts.layer_states[layer_idx]
        .value_cache.as_ref().expect("value_cache");
    infra.attention.launch_kv_cache_append(
        &infra.stream, &acts.attn_k, &acts.attn_v, key_cache, value_cache, kvd, pos,
    )?;

    // 6. Flash attention decode
    let n_positions = pos + 1;
    infra.attention.launch_attention_decode(
        &infra.stream, &acts.attn_q, key_cache, value_cache, &acts.attn_out,
        hd, n_head, n_kv, n_positions,
    )?;

    // Issue 492 T1 GPU lane - capture tap. Whole k/v cache buffers are
    // downloaded (exact-length memcpy_dtoh, the residual-capture idiom) and
    // truncated host-side to the `0..=pos` rows the probe reads; q and
    // attn_out are `q_dim` downloads sliced to the probed head.
    if let Some((spec, cap)) = probe {
        debug_assert_eq!(spec.layer, layer_idx);
        let t_rows = pos + 1;
        let h0 = spec.head * hd;
        infra
            .stream
            .synchronize()
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;

        let mut q_all = vec![0.0f32; q_dim];
        infra
            .stream
            .memcpy_dtoh(&acts.attn_q, &mut q_all)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        cap.q_head.clear();
        cap.q_head.extend_from_slice(&q_all[h0..h0 + hd]);

        cap.k_cache.resize(key_cache.len(), 0.0f32);
        infra
            .stream
            .memcpy_dtoh(key_cache, &mut cap.k_cache)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        cap.k_cache.truncate(t_rows * kvd);

        cap.v_cache.resize(value_cache.len(), 0.0f32);
        infra
            .stream
            .memcpy_dtoh(value_cache, &mut cap.v_cache)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        cap.v_cache.truncate(t_rows * kvd);

        let mut out_all = vec![0.0f32; q_dim];
        infra
            .stream
            .memcpy_dtoh(&acts.attn_out, &mut out_all)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        cap.attn_out_head.clear();
        cap.attn_out_head.extend_from_slice(&out_all[h0..h0 + hd]);

        cap.position = pos;
        cap.head_dim = hd;
        cap.kvd = kvd;
        let heads_per_group = n_head / n_kv;
        cap.kv_off = (spec.head / heads_per_group) * hd;
    }

    // 7-8. Output gating + output projection (FUSED with quantize)
    gate_sigmoid_quantize_and_gemv(
        &infra.elementwise, &infra.stream, &infra.gemv_multi,
        &acts.quant_i8_buf, &acts.ascale_buf,
        &acts.attn_out, &acts.attn_gate, q_dim, wo, &acts.tmp,
    )?;
    Ok(())
}

/// Issue 618 — Device-pointer variant of `forward_attention_layer`.
///
/// Identical to `forward_attention_layer` except the 3 position-dependent
/// kernels (rope, kv_cache_append, attention_decode) read `pos` from
/// `acts.pos_dev_buf` (a 1-element device buffer) at kernel runtime instead
/// of from a scalar `int` arg baked into the kernel launch.
///
/// This makes the forward path replay-able via CUDA Graphs: the captured
/// graph references the device buffer (constant address), and per-token the
/// caller writes the new pos value to the buffer before `graph.launch()`.
///
/// Correctness invariant: `acts.pos_dev_buf` MUST contain `acts.pos as i32`
/// BEFORE this function is called. The caller is responsible for the memcpy.
#[cfg(feature = "cuda_graphs_forward")]
#[allow(clippy::too_many_arguments)]
fn forward_attention_layer_devpos(
    infra: &ForwardInfraCudarc,
    acts: &mut ActivationsCudarc,
    layer_idx: usize,
    layer_w: &GpuLayerWeightsCudarc,
    eps: f32,
    accumulate_out: bool,
) -> Result<(), CudarcKernelError> {
    let n_head = infra.config.n_head;
    let n_kv = infra.config.n_kv_head;
    let hd = infra.config.head_dim;
    let q_dim = n_head * hd;
    let kvd = n_kv * hd;
    let rotary_dim = if infra.config.rope_dimension_count > 0 {
        infra.config.rope_dimension_count
    } else {
        hd
    };
    let theta_base = infra.config.rope_theta;

    let wq = layer_w.attn_wq.as_ref().expect("attn_wq");
    let wk = layer_w.attn_wk.as_ref().expect("attn_wk");
    let wv = layer_w.attn_wv.as_ref().expect("attn_wv");
    let wo = layer_w.attn_wo.as_ref().expect("attn_wo");
    let q_norm = layer_w.attn_q_norm.as_ref().expect("attn_q_norm");
    let k_norm = layer_w.attn_k_norm.as_ref().expect("attn_k_norm");

    // 1. Q (gated), K, V projections — fused pre-attn RMSNorm + quantize.
    // Issue 624 — fuse the pre-attention RMSNorm + quantize into one kernel,
    // then dispatch all 3 GEMVs from the same int8 buffer. Saves 1 launch
    // per layer + eliminates the intermediate f32 norm_x memory traffic.
    //
    // Issue 980 T4.5 — the graph (devpos) twin of the eager rotation branch:
    // norm -> f32, rotate a copy, quantize + GEMV the folded q/k/v from it.
    // Every launch is graph-capturable (fixed device addresses; the RotationTables
    // buffers were allocated on this stream in with_shared before any capture).
    if let Some(rot) = &infra.rotation {
        let n = infra.config.n_embd;
        // Issue 980 T4 escalation — K1 (the eager twin's branch, graph-
        // capturable like every launch here).
        if rot.fused_geometry_ok(n) {
            rot.kernels.rmsnorm_rotate_quantize(
                &infra.stream,
                &acts.x,
                &layer_w.input_norm,
                rot.signs_for_width(n),
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                n,
                eps,
                rot.block_size,
            )?;
            gemv_prequantized_multi(
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &[
                    (wq, &acts.attn_qg),
                    (wk, &acts.attn_k),
                    (wv, &acts.attn_v),
                ],
                false,
            )?;
        } else {
            // Split fallback (odd Hadamard geometry — not Bonsai-2's 1024).
            infra.elementwise.launch_rmsnorm(
                &infra.stream,
                &acts.x,
                &layer_w.input_norm,
                &acts.norm_x,
                n,
                eps,
            )?;
            infra
                .stream
                .memcpy_dtod(&acts.norm_x, &mut acts.rot_scratch)
                .map_err(alloc_err)?;
            rot.kernels.fwht_rotate_forward(
                &infra.stream,
                &acts.rot_scratch,
                rot.signs_for_width(n),
                n,
                rot.block_size,
            )?;
            quantize_and_gemv_batch(
                &infra.elementwise,
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &acts.rot_scratch,
                &[
                    (wq, &acts.attn_qg),
                    (wk, &acts.attn_k),
                    (wv, &acts.attn_v),
                ],
            )?;
        }
    } else {
        rmsnorm_quantize_and_gemv_batch(
            &infra.elementwise, &infra.stream, &infra.gemv_multi,
            &acts.quant_i8_buf, &acts.ascale_buf,
            &acts.x, &layer_w.input_norm, eps, infra.config.n_embd,
            &[
                (wq, &acts.attn_qg),
                (wk, &acts.attn_k),
                (wv, &acts.attn_v),
            ],
        )?;
    }

    // 2. Split QG into Q and gate.
    infra.attention.launch_split_qg(
        &infra.stream,
        &acts.attn_qg,
        &acts.attn_q,
        &acts.attn_gate,
        hd,
        n_head,
    )?;

    // 3. Per-head RMSNorm on Q and K (in-place).
    infra.attention.launch_rmsnorm_batched(
        &infra.stream,
        &acts.attn_q,
        q_norm,
        &acts.attn_q,
        n_head,
        hd,
        eps,
    )?;
    infra.attention.launch_rmsnorm_batched(
        &infra.stream,
        &acts.attn_k,
        k_norm,
        &acts.attn_k,
        n_kv,
        hd,
        eps,
    )?;

    // 4. Partial RoPE on Q and K (in-place) — _devpos variant.
    infra.attention.launch_rope_devpos(
        &infra.stream,
        &acts.attn_q,
        &acts.attn_k,
        rotary_dim,
        hd,
        n_head,
        n_kv,
        &acts.pos_dev_buf,
        theta_base,
    )?;

    // 5. Append K, V to KV cache — _devpos variant.
    let key_cache = acts.layer_states[layer_idx]
        .key_cache
        .as_ref()
        .expect("key_cache");
    let value_cache = acts.layer_states[layer_idx]
        .value_cache
        .as_ref()
        .expect("value_cache");
    infra.attention.launch_kv_cache_append_devpos(
        &infra.stream,
        &acts.attn_k,
        &acts.attn_v,
        key_cache,
        value_cache,
        kvd,
        &acts.pos_dev_buf,
    )?;

    // 6. Flash attention decode — _devpos variant.
    //    (Kernel internally computes n_positions = *pos_dev + 1.)
    infra.attention.launch_attention_decode_devpos(
        &infra.stream,
        &acts.attn_q,
        key_cache,
        value_cache,
        &acts.attn_out,
        hd,
        n_head,
        n_kv,
        &acts.pos_dev_buf,
    )?;

    // 7–8. Output gating + output projection (FUSED with quantize).
    // Issue 626 — fuse `sigmoid(gate) * attn_out` + int8 quantize into one
    // kernel, then dispatch the dp4a wo GEMV. Saves 1 launch + the
    // intermediate f32 gated attn_out round-trip through HBM.
    // Issue 697 — accumulate variant folds the residual add into the GEMV.
    //
    // Issue 980 T4.5 — folded model (the eager twin's branch): the gate
    // splits out (pre-rotation pairing), the rotation runs in place on
    // `attn_out`, then quantize + GEMV as usual. Graph-capturable launches.
    //
    // Issue 980 T4 escalation — K4 fuses gate+sign+FWHT+quantize (3→1).
    if let Some(rot) = &infra.rotation {
        if rot.fused_geometry_ok(q_dim) {
            rot.kernels.gate_sigmoid_rotate_quantize(
                &infra.stream,
                &acts.attn_out,
                &acts.attn_gate,
                rot.signs_for_width(q_dim),
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                q_dim,
                rot.block_size,
            )?;
        } else {
            rot.kernels
                .gate_sigmoid(&infra.stream, &acts.attn_out, &acts.attn_gate, q_dim)?;
            rot.kernels.fwht_rotate_forward(
                &infra.stream,
                &acts.attn_out,
                rot.signs_for_width(q_dim),
                q_dim,
                rot.block_size,
            )?;
            infra.elementwise.launch_quantize(
                &infra.stream,
                &acts.attn_out,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                q_dim,
            )?;
        }
        gemv_prequantized_multi(
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &[(wo, if accumulate_out { &acts.x } else { &acts.tmp })],
            accumulate_out,
        )?;
    } else if accumulate_out {
        gate_sigmoid_quantize_and_gemv_accum(
            &infra.elementwise,
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &acts.attn_out,
            &acts.attn_gate,
            q_dim,
            wo,
            &acts.x,
        )?;
    } else {
        gate_sigmoid_quantize_and_gemv(
            &infra.elementwise,
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &acts.attn_out,
            &acts.attn_gate,
            q_dim,
            wo,
            &acts.tmp,
        )?;
    }
    Ok(())
}

/// Issue 618 — Device-pointer variant of `forward_from_x`.
///
/// Same as `forward_from_x` but:
///   1. Reads the input token from `acts.token_dev_buf` via the _devpos
///      embedding kernel (no `set_input_token` needed separately).
///   2. Uses `forward_attention_layer_devpos` for attention layers.
///   3. Does NOT call `set_input_token` — the caller writes the new token_id
///      to `acts.token_dev_buf` (and the new pos to `acts.pos_dev_buf`)
///      BEFORE calling this.
///
/// Designed for CUDA Graph capture: every dependency is a device buffer with
/// a fixed address, so the captured graph is fully replay-able.
///
/// Issue 696 — accepts a LoRA ctx (captured into the graph when present).
/// The LoRA kernels (`launch_qv_apply`, `launch_rmsnorm_quantize_with_norm_x`)
/// are pure kernel launches on stable device buffers, so they capture/replay
/// exactly like the backbone kernels.
#[cfg(feature = "cuda_graphs_forward")]
fn forward_from_x_devpos(
    infra: &ForwardInfraCudarc,
    acts: &mut ActivationsCudarc,
    mut lora_ctx: Option<&mut LoraFwdCtx<'_>>,
) -> Result<(), CudarcKernelError> {
    let n = infra.config.n_embd;
    let eps = infra.config.rms_norm_eps as f32;

    // 0. Embedding dequant from token_dev_buf — _devpos variant.
    infra.embedding.launch_dequant_row_devpos(
        &infra.stream,
        &infra.wte_pos_bits,
        &infra.wte_neg_bits,
        &infra.wte_scale,
        &acts.x,
        &acts.token_dev_buf,
        infra.wte_blocks64,
        infra.wte_groups_per_row,
        n,
    )?;
    // Issue 980 T4.5 — a Hadamard-latent embedding table stores rotated rows;
    // restore the primal basis right after the in-graph lookup (Hadamard
    // first, sign second — the `set_input_token` eager twin). Captured into
    // the graph like every other launch here.
    if let Some(rot) = &infra.rotation
        && rot.inverse_embedding
    {
        let signs = rot.signs_for_width(n);
        rot.kernels.fwht_rotate_inverse(
            &infra.stream,
            &acts.x,
            signs,
            n,
            rot.block_size,
        )?;
    }

    for layer_idx in 0..infra.config.n_layer {
        let layer_w = &infra.layers[layer_idx];
        let is_deltanet = infra.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

        // ── Pre-attention RMSNorm is fused into the layer functions' input
        // projection quantize (Issue 624). The layer functions call
        // rmsnorm_quantize_and_gemv_batch directly, saving 1 launch/layer.
        //
        // Issue 697 — the layer's out_proj GEMV accumulates directly into the
        // residual stream (out[row] += acc), so there is NO separate
        // residual_add after the layer.
        if is_deltanet {
            forward_deltanet_layer(infra, acts, layer_idx, layer_w, eps, lora_ctx.as_deref_mut(), true)?;
        } else {
            forward_attention_layer_devpos(infra, acts, layer_idx, layer_w, eps, true)?;
        }

        // ── Post-attention RMSNorm + FFN gate/up (fused) ──
        // Issue 623 — fuse RMSNorm + quantize into one kernel.
        // (The post-attn residual was folded into the out_proj GEMV above.)
        //
        // Issue 980 T4.5 — folded model (the eager twin's branch, verbatim):
        // norm -> f32 (`norm_x`), rotate a copy, quantize + GEMV gate/up from
        // it; SwiGLU into `ffn_hidden`, rotate it in place (folded down
        // input), quantize + accumulate-GEMV down. `x` stays primal. All
        // launches graph-capturable.
        //
        // Issue 980 T4 escalation — K1 + K5 (the eager twin's fused arms,
        // graph-capturable).
        if let Some(rot) = &infra.rotation {
            if rot.fused_geometry_ok(n) {
                rot.kernels.rmsnorm_rotate_quantize(
                    &infra.stream,
                    &acts.x,
                    &layer_w.post_attn_norm,
                    rot.signs_for_width(n),
                    &acts.quant_i8_buf,
                    &acts.ascale_buf,
                    n,
                    eps,
                    rot.block_size,
                )?;
                gemv_prequantized_multi(
                    &infra.stream,
                    &infra.gemv_multi,
                    &acts.quant_i8_buf,
                    &acts.ascale_buf,
                    &[
                        (&layer_w.gate_proj, &acts.ffn_gate),
                        (&layer_w.up_proj, &acts.ffn_up),
                    ],
                    false,
                )?;
                rot.kernels.swiglu_rotate_quantize(
                    &infra.stream,
                    &acts.ffn_gate,
                    &acts.ffn_up,
                    rot.signs_for_width(infra.config.mlp_hidden),
                    &acts.quant_i8_buf,
                    &acts.ascale_buf,
                    infra.config.mlp_hidden,
                    rot.block_size,
                )?;
                gemv_prequantized_multi(
                    &infra.stream,
                    &infra.gemv_multi,
                    &acts.quant_i8_buf,
                    &acts.ascale_buf,
                    &[(&layer_w.down_proj, &acts.x)],
                    true,
                )?;
                continue;
            }
            infra.elementwise.launch_rmsnorm(
                &infra.stream,
                &acts.x,
                &layer_w.post_attn_norm,
                &acts.norm_x,
                n,
                eps,
            )?;
            infra
                .stream
                .memcpy_dtod(&acts.norm_x, &mut acts.rot_scratch)
                .map_err(alloc_err)?;
            rot.kernels.fwht_rotate_forward(
                &infra.stream,
                &acts.rot_scratch,
                rot.signs_for_width(n),
                n,
                rot.block_size,
            )?;
            quantize_and_gemv_batch(
                &infra.elementwise,
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &acts.rot_scratch,
                &[
                    (&layer_w.gate_proj, &acts.ffn_gate),
                    (&layer_w.up_proj, &acts.ffn_up),
                ],
            )?;
            infra.elementwise.launch_swiglu(
                &infra.stream,
                &acts.ffn_gate,
                &acts.ffn_up,
                &acts.ffn_hidden,
                infra.config.mlp_hidden,
            )?;
            rot.kernels.fwht_rotate_forward(
                &infra.stream,
                &acts.ffn_hidden,
                rot.signs_for_width(infra.config.mlp_hidden),
                infra.config.mlp_hidden,
                rot.block_size,
            )?;
            infra.elementwise.launch_quantize(
                &infra.stream,
                &acts.ffn_hidden,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                infra.config.mlp_hidden,
            )?;
            gemv_prequantized_multi(
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &[(&layer_w.down_proj, &acts.x)],
                true,
            )?;
            continue;
        }
        rmsnorm_quantize_and_gemv_batch(
            &infra.elementwise,
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &acts.x,
            &layer_w.post_attn_norm,
            eps,
            n,
            &[
                (&layer_w.gate_proj, &acts.ffn_gate),
                (&layer_w.up_proj, &acts.ffn_up),
            ],
        )?;
        // Issue 625 — fuse SwiGLU + quantize into one kernel, then dispatch
        // down_proj GEMV. Saves 1 launch + intermediate ffn_hidden traffic.
        // Issue 697 — accumulate variant: down_proj adds directly into x, so
        // there is NO separate residual_add after the FFN.
        swiglu_quantize_and_gemv_accum(
            &infra.elementwise,
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &acts.ffn_gate,
            &acts.ffn_up,
            infra.config.mlp_hidden,
            &layer_w.down_proj,
            &acts.x,
        )?;
    }

    // ── Final RMSNorm + lm_head (fused) ──
    // Issue 623 — fuse the final RMSNorm + quantize, then dispatch lm_head GEMV.
    //
    // Issue 980 T4.5 — folded model (the eager twin's tail): norm -> f32 into
    // `norm_x`, rotate IN PLACE there (`norm_x` is the lm_head's only
    // consumer on this path), quantize + GEMV. Logits come out primal.
    //
    // Issue 980 T4 escalation — K1 (3→1).
    if let Some(rot) = &infra.rotation {
        if rot.fused_geometry_ok(n) {
            rot.kernels.rmsnorm_rotate_quantize(
                &infra.stream,
                &acts.x,
                &infra.final_norm,
                rot.signs_for_width(n),
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                n,
                eps,
                rot.block_size,
            )?;
            gemv_prequantized_multi(
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &[(&infra.lm_head, &acts.logits)],
                false,
            )?;
        } else {
            infra.elementwise.launch_rmsnorm(
                &infra.stream,
                &acts.x,
                &infra.final_norm,
                &acts.norm_x,
                n,
                eps,
            )?;
            rot.kernels.fwht_rotate_forward(
                &infra.stream,
                &acts.norm_x,
                rot.signs_for_width(n),
                n,
                rot.block_size,
            )?;
            quantize_and_gemv_batch(
                &infra.elementwise,
                &infra.stream,
                &infra.gemv_multi,
                &acts.quant_i8_buf,
                &acts.ascale_buf,
                &acts.norm_x,
                &[(&infra.lm_head, &acts.logits)],
            )?;
        }
    } else {
        rmsnorm_quantize_and_gemv_batch(
            &infra.elementwise,
            &infra.stream,
            &infra.gemv_multi,
            &acts.quant_i8_buf,
            &acts.ascale_buf,
            &acts.x,
            &infra.final_norm,
            eps,
            n,
            &[(&infra.lm_head, &acts.logits)],
        )?;
    }

    // ── Issue 697 — GPU-side argmax over the final logits ──
    // The decode loop downloads 8 bytes instead of the vocab-sized logits
    // vector (saves ~380 µs/token of dtoh + host argmax). The memset node is
    // CUDA-Graph-capturable, so the whole sequence replays per token.
    infra
        .stream
        .memset_zeros(&mut acts.argmax_buf)
        .map_err(alloc_err)?;
    infra.elementwise.launch_argmax_first(
        &infra.stream,
        &acts.logits,
        infra.config.vocab_size,
        &acts.argmax_buf,
    )?;

    // NOTE: do NOT increment acts.pos here. forward_from_x (the non-devpos
    // path) increments pos, but for the devpos path the caller
    // (forward_token_graph) owns pos increment because the graph launch only
    // replays GPU kernels (not this host-side += 1).
    Ok(())
}

fn reset_state(
    infra: &ForwardInfraCudarc,
    acts: &mut ActivationsCudarc,
) -> Result<(), CudarcKernelError> {
    // Issue 696 — zero the persistent state buffers IN PLACE (async memset)
    // instead of replacing them via clone_htod. Two reasons:
    //   1. Device addresses stay stable, so a captured CUDA Graph (Issue 618)
    //      remains valid across reset_state — the baked-in pointers still name
    //      the live (now zeroed) buffers. Previously the replaced buffers left
    //      the graph replaying against freed memory.
    //   2. No per-reset host zero-vec + GPU alloc + htod copy (~2 MB/layer for
    //      Bonsai's DeltaNet state). Semantics identical: all-zero f32 bits.
    for (i, lt) in infra.layer_types.iter().enumerate() {
        if *lt == DeltaNetLayerType::DeltaNet {
            if let Some(ref mut state) = acts.layer_states[i].deltanet_state {
                state.memset_zeros(&infra.stream).map_err(alloc_err)?;
            }
            if let Some(ref mut cstate) = acts.layer_states[i].conv_state {
                infra.stream.memset_zeros(cstate).map_err(alloc_err)?;
            }
        }
    }
    acts.pos = 0;
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────────
// Ternary GEMV helpers
// ───────────────────────────────────────────────────────────────────────────

/// GPU-resident ternary GEMV: `out = weights @ x`.
///
/// Quantizes `x` (f32 → int8) on the shared stream, then dispatches the dp4a
/// kernel. No CPU sync, no per-call allocation. **This is the production
/// path** — Issue 616 T4 proved the fused alternative is slower (see
/// `gemv_fused_into` for the negative-result documentation).
///
/// **Note (Issue 626):** after fusing the output-gate + quantize for both
/// DeltaNet (z-gating) and attention (output-gating), `gemv_into` is no
/// longer called from the production forward path. It is retained as a
/// utility — every fused helper (`rmsnorm_quantize_and_gemv_batch`,
/// `swiglu_quantize_and_gemv`, `gate_silu_quantize_and_gemv`,
/// `gate_sigmoid_quantize_and_gemv`) is built on `gemv_prequantized` instead.
#[allow(clippy::too_many_arguments, dead_code)]
fn gemv_into(
    elementwise: &ElementwiseKernels,
    stream: &Arc<CudaStream>,
    gemv: &CudaFunction,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    weights: &WeightBuffersCudarc,
    x: &CudaSlice<f32>,
    out: &CudaSlice<f32>,
) -> Result<(), CudarcKernelError> {
    if weights.m == 0 {
        return Ok(());
    }
    let n = weights.n;
    let ablocks = n.div_ceil(16);

    elementwise.launch_quantize(stream, x, quant_i8_buf, ascale_buf, n)?;

    let m_i32 = weights.m as i32;
    let int16_per_row = (n / 8) as i32;
    let groups_per_row = n.div_ceil(128) as i32;
    let ablock_i32 = 16i32;
    let ablocks_i32 = ablocks as i32;

    let grid_x = (weights.m as u32).div_ceil(WG_THREADS / 32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (WG_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(gemv)
            .arg(&weights.codes)
            .arg(&weights.wscale)
            .arg(quant_i8_buf)
            .arg(ascale_buf)
            .arg(out)
            .arg(&m_i32)
            .arg(&int16_per_row)
            .arg(&groups_per_row)
            .arg(&ablock_i32)
            .arg(&ablocks_i32)
            .launch(cfg)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Dispatch the dp4a GEMV **without** quantizing first. The caller MUST have
/// already populated `quant_i8_buf` + `ascale_buf` by calling
/// `elementwise.launch_quantize(stream, x, quant_i8_buf, ascale_buf, n)` with the
/// SAME `n` as `weights.n`.
///
/// Issue 620 — eliminates redundant quantize launches when multiple GEMVs read
/// the same input vector (e.g. 4 DeltaNet input projections all read `norm_x`).
/// The quantization is a deterministic function of the input, so the int8 +
/// ascale buffers are identical across calls — recomputing them is pure waste.
#[allow(clippy::too_many_arguments)]
fn gemv_prequantized(
    stream: &Arc<CudaStream>,
    gemv: &CudaFunction,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    weights: &WeightBuffersCudarc,
    out: &CudaSlice<f32>,
) -> Result<(), CudarcKernelError> {
    if weights.m == 0 {
        return Ok(());
    }
    let n = weights.n;
    let ablocks = n.div_ceil(16);

    let m_i32 = weights.m as i32;
    let int16_per_row = (n / 8) as i32;
    let groups_per_row = n.div_ceil(128) as i32;
    let ablock_i32 = 16i32;
    let ablocks_i32 = ablocks as i32;

    let grid_x = (weights.m as u32).div_ceil(WG_THREADS / 32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (WG_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(gemv)
            .arg(&weights.codes)
            .arg(&weights.wscale)
            .arg(quant_i8_buf)
            .arg(ascale_buf)
            .arg(out)
            .arg(&m_i32)
            .arg(&int16_per_row)
            .arg(&groups_per_row)
            .arg(&ablock_i32)
            .arg(&ablocks_i32)
            .launch(cfg)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Issue 697 / Issue 705 — dispatch up to 4 GEMVs sharing the SAME quantized
/// input as ONE kernel launch (`gemv_ternary_dp4a_multi_persistent`).
///
/// Motivation (nsys attribution): per-token decode issued 497 GEMV launches;
/// the m=48 a/b projections ran at the ~5.6 µs kernel latency floor (540 µs/
/// token for 6.6 MB of weights), and every launch paid its own
/// wave-quantization tail. Same-input groups (qkv+z+a+b; gate+up; q+k+v)
/// concatenate into one launch so the tiny rows ride along at ~zero marginal
/// cost. (Issue 705's persistent grid-stride variant is OPT-IN via
/// `RIIR_GEMV_PERSISTENT_GRID` — the measured DEFAULT is the UNCAPPED
/// one-warp-per-row grid: Bench 684 refuted the residency cap as a 1-2% loss
/// at every size; see the `GemvMultiPersistent` doc for the numbers. This
/// sentence previously claimed the cap as shipped behavior — Issue 722 H13.)
///
/// With `accumulate`, the kernel does `out[row] += acc` instead of a plain
/// store — used by the hot path to fold the residual add into the out_proj /
/// down_proj GEMVs (both have m == n_embd). The f32 add has the same operands
/// and evaluation order as the separate `residual_add_f32` launch did, so the
/// result is bit-identical.
///
/// **Caller invariant:** every `weights` entry must have the same `.n` (they
/// consume the same quantized activation), and `accumulate` requires every
/// segment's `m` to equal the `out` slice length.
fn gemv_prequantized_multi(
    stream: &Arc<CudaStream>,
    gemv_multi: &GemvMultiPersistent,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    gemvs: &[(&WeightBuffersCudarc, &CudaSlice<f32>)],
    accumulate: bool,
) -> Result<(), CudarcKernelError> {
    debug_assert!(gemvs.len() <= 4, "Issue 697: multi-GEMV supports up to 4 segments");
    if gemvs.is_empty() {
        return Ok(());
    }
    let n = gemvs[0].0.n;
    if n == 0 {
        return Ok(());
    }
    let total_m: usize = gemvs.iter().map(|(w, _)| w.m).sum();
    if total_m == 0 {
        return Ok(());
    }

    // Pad to exactly 4 segments: zero-length segments match no rows, so their
    // pointers are never dereferenced — reuse segment 0's pointers.
    let mut segs: Vec<(&WeightBuffersCudarc, &CudaSlice<f32>)> = Vec::with_capacity(4);
    segs.extend_from_slice(gemvs);
    while segs.len() < 4 {
        segs.push(segs[0]);
    }

    let m: [i32; 4] = segs.iter().map(|(w, _)| w.m as i32).collect::<Vec<_>>().try_into().unwrap();
    let int16_per_row = (n / 8) as i32;
    let groups_per_row = n.div_ceil(128) as i32;
    let ablock_i32 = 16i32;
    let ablocks_i32 = n.div_ceil(16) as i32;
    let acc_i32 = accumulate as i32;
    let total_rows_i32 = total_m as i32;

    // Issue 705 (filed as 702, renumbered) — grid = min(one-warp-per-row, cap). Default cap is u32::MAX
    // (uncapped — measured best, Bench 684); the loop degenerates to ≤ 1
    // iteration per warp, behaviorally identical to the pre-702 kernel. A
    // finite cap (RIIR_GEMV_PERSISTENT_GRID) activates grid-stride — measured a
    // 1-2% loss, retained only as the tuning/repro apparatus.
    // Plan 604 T1/T3 (Issue 987 G1/G2) — UNDERFILLED launches (total rows ≤
    // `r2_max_rows`, i.e. ≤ occupancy_grid blocks at the one-row shape) may
    // dispatch to an experimental variant: `function_r2` (two rows per warp,
    // MEASURED −4.4% — negative apparatus only) or `function_pf` (one row per
    // warp + one-iteration-ahead L2 prefetch — the T3 A/B arm). Argument
    // signatures are identical; only the fn handle changes (pf keeps the
    // one-row grid; r2 halves it). PF takes precedence when both are set.
    let underfilled = total_m <= gemv_multi.r2_max_rows;
    let use_pf = gemv_multi.pf_enabled && underfilled;
    let use_r2 = gemv_multi.r2_enabled && underfilled && !use_pf;
    let use_u4 = gemv_multi.u4_enabled && underfilled && !use_pf && !use_r2;
    let grid_x = if use_r2 {
        ((total_m as u32).div_ceil(2 * (WG_THREADS / 32)))
            .min(gemv_multi.grid)
            .max(1)
    } else {
        ((total_m as u32).div_ceil(WG_THREADS / 32))
            .min(gemv_multi.grid)
            .max(1)
    };
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (WG_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    let [s0, s1, s2, s3] = segs.as_slice() else { unreachable!() };
    let gemv_fn = if use_pf {
        &gemv_multi.function_pf
    } else if use_r2 {
        &gemv_multi.function_r2
    } else if use_u4 {
        &gemv_multi.function_u4
    } else {
        &gemv_multi.function
    };
    unsafe {
        stream
            .launch_builder(gemv_fn)
            .arg(&s0.0.codes)
            .arg(&s0.0.wscale)
            .arg(s0.1)
            .arg(&m[0])
            .arg(&s1.0.codes)
            .arg(&s1.0.wscale)
            .arg(s1.1)
            .arg(&m[1])
            .arg(&s2.0.codes)
            .arg(&s2.0.wscale)
            .arg(s2.1)
            .arg(&m[2])
            .arg(&s3.0.codes)
            .arg(&s3.0.wscale)
            .arg(s3.1)
            .arg(&m[3])
            .arg(quant_i8_buf)
            .arg(ascale_buf)
            .arg(&int16_per_row)
            .arg(&groups_per_row)
            .arg(&ablock_i32)
            .arg(&ablocks_i32)
            .arg(&acc_i32)
            .arg(&total_rows_i32)
            .launch(cfg)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
    }
    Ok(())
}

/// Quantize `x` once, then dispatch the dp4a GEMV for each `(weights, out)` pair
/// reading from the same quantized buffer. Issue 620 — eliminates redundant
/// quantize launches when multiple GEMVs share the same input vector.
///
/// **Note (Issue 624):** after fusing the pre-attention RMSNorm into
/// `rmsnorm_quantize_and_gemv_batch`, this function is no longer called from
/// the production forward path. It is retained as a utility for future
/// use-cases where the input does not need RMSNorm pre-processing.
///
/// **Caller invariant:** every `weights` entry MUST have the same `.n` (input
/// dimension), because they all consume the same quantized activation.
#[allow(clippy::too_many_arguments, dead_code)]
fn quantize_and_gemv_batch(
    elementwise: &ElementwiseKernels,
    stream: &Arc<CudaStream>,
    gemv: &GemvMultiPersistent,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    gemvs: &[(&WeightBuffersCudarc, &CudaSlice<f32>)],
) -> Result<(), CudarcKernelError> {
    if gemvs.is_empty() {
        return Ok(());
    }
    let n = gemvs[0].0.n;
    if n == 0 {
        return Ok(());
    }
    // Quantize once — all GEMVs consume the same int8 + ascale.
    elementwise.launch_quantize(stream, x, quant_i8_buf, ascale_buf, n)?;
    for (weights, _) in gemvs {
        debug_assert_eq!(
            weights.n, n,
            "Issue 620: all GEMVs in a batch must share the same input dimension"
        );
    }
    // Issue 697 — one multi-segment launch instead of N.
    gemv_prequantized_multi(stream, gemv, quant_i8_buf, ascale_buf, gemvs, false)?;
    Ok(())
}

/// Fused RMSNorm + quantize + multi-GEMV dispatch. Issue 623 — combines
/// `launch_rmsnorm` + `launch_quantize` + N× GEMV into 1 fused norm+quantize
/// kernel + N GEMV launches, saving 1 kernel launch + intermediate norm_x
/// memory traffic per call.
///
/// The fused kernel writes directly to `quant_i8_buf` + `ascale_buf` (no
/// intermediate f32 `norm_x` buffer needed).
///
/// **Caller invariant:** every `weights` entry MUST have `.n == dim` (the
/// RMSNorm dimension), because they all consume the same quantized activation.
#[allow(clippy::too_many_arguments)]
fn rmsnorm_quantize_and_gemv_batch(
    elementwise: &ElementwiseKernels,
    stream: &Arc<CudaStream>,
    gemv: &GemvMultiPersistent,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    gamma: &CudaSlice<f32>,
    eps: f32,
    dim: usize,
    gemvs: &[(&WeightBuffersCudarc, &CudaSlice<f32>)],
) -> Result<(), CudarcKernelError> {
    if gemvs.is_empty() {
        // Still need to write norm_x for non-GEMV consumers.
        // But if there are no GEMVs, skip entirely.
        return Ok(());
    }
    // Fused RMSNorm + quantize — writes int8 + ascale directly.
    elementwise.launch_rmsnorm_quantize(
        stream, x, gamma, quant_i8_buf, ascale_buf, dim, eps,
    )?;
    for (weights, _) in gemvs {
        debug_assert_eq!(
            weights.n, dim,
            "Issue 623: all GEMVs must share the RMSNorm dimension"
        );
    }
    // Issue 697 — one multi-segment launch instead of N.
    gemv_prequantized_multi(stream, gemv, quant_i8_buf, ascale_buf, gemvs, false)?;
    Ok(())
}

/// Issue 634 — sibling of [`rmsnorm_quantize_and_gemv_batch`] that ALSO
/// writes the f32 `norm_x` to `norm_x_out`. Used only by
/// `TernaryDeltanetGpuForwardCudarc::forward_token_with_final_hidden` so the
/// lm_head LoRA precompute can read the final normed hidden state.
///
/// Same launch cost as the un-sided variant; one extra elementwise write per
/// active thread. The hot-path forward keeps using the un-sided variant.
#[allow(clippy::too_many_arguments)]
fn rmsnorm_quantize_and_gemv_batch_with_norm_x(
    elementwise: &ElementwiseKernels,
    stream: &Arc<CudaStream>,
    gemv: &GemvMultiPersistent,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    gamma: &CudaSlice<f32>,
    norm_x_out: &CudaSlice<f32>,
    eps: f32,
    dim: usize,
    gemvs: &[(&WeightBuffersCudarc, &CudaSlice<f32>)],
) -> Result<(), CudarcKernelError> {
    if gemvs.is_empty() {
        // Even with no GEMV consumers, the norm_x side buffer is still
        // materialised (it's the whole point of this variant). So we MUST run
        // the sided kernel even if the GEMV is empty. Quantize-only path is
        // fine — the launcher doesn't know about GEMVs.
        elementwise.launch_rmsnorm_quantize_with_norm_x(
            stream, x, gamma, quant_i8_buf, ascale_buf, norm_x_out, dim, eps,
        )?;
        return Ok(());
    }
    // Fused RMSNorm + quantize + norm_x side write.
    elementwise.launch_rmsnorm_quantize_with_norm_x(
        stream, x, gamma, quant_i8_buf, ascale_buf, norm_x_out, dim, eps,
    )?;
    for (weights, _) in gemvs {
        debug_assert_eq!(
            weights.n, dim,
            "Issue 634: all GEMVs must share the RMSNorm dimension"
        );
    }
    // Issue 697 — one multi-segment launch instead of N.
    gemv_prequantized_multi(stream, gemv, quant_i8_buf, ascale_buf, gemvs, false)?;
    Ok(())
}

/// Issue 625 — fused SwiGLU + quantize + single dp4a GEMV.
///
/// Computes `silu(gate) * up`, quantizes to int8 in the same kernel, then
/// dispatches the dp4a GEMV for `weights` reading from the quantized buffer.
/// Saves 1 kernel launch + intermediate f32 `ffn_hidden` memory traffic per
/// call. Used for the FFN down_proj GEMV (single consumer of SwiGLU output).
#[allow(clippy::too_many_arguments)]
fn swiglu_quantize_and_gemv(
    elementwise: &ElementwiseKernels,
    stream: &Arc<CudaStream>,
    gemv: &CudaFunction,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    n: usize,
    weights: &WeightBuffersCudarc,
    out: &CudaSlice<f32>,
) -> Result<(), CudarcKernelError> {
    if weights.m == 0 {
        return Ok(());
    }
    debug_assert_eq!(
        weights.n, n,
        "Issue 625: GEMV input dim must match SwiGLU dim"
    );
    // Fused SwiGLU + quantize — writes int8 + ascale directly.
    elementwise.launch_swiglu_quantize(
        stream, gate, up, quant_i8_buf, ascale_buf, n,
    )?;
    gemv_prequantized(stream, gemv, quant_i8_buf, ascale_buf, weights, out)?;
    Ok(())
}

/// Issue 626 — fused silu-gate + quantize + single dp4a GEMV.
///
/// Computes `silu(gate) * x`, quantizes to int8 in the same kernel, then
/// dispatches the dp4a GEMV for `weights` reading from the quantized buffer.
/// Saves 1 kernel launch + intermediate f32 `recurrent_out` memory traffic per
/// call. Used for the DeltaNet out_proj GEMV (single consumer of z-gated
/// recurrent_out).
///
/// Issue 627 supersedes this in the production DeltaNet path (the forward path
/// now uses `rmsnorm_gate_silu_quantize_and_gemv` which fuses the per-head
/// RMSNorm too). Retained as a utility for callers that already have a
/// pre-normalized `x`.
#[allow(clippy::too_many_arguments, dead_code)]
fn gate_silu_quantize_and_gemv(
    elementwise: &ElementwiseKernels,
    stream: &Arc<CudaStream>,
    gemv: &CudaFunction,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    gate: &CudaSlice<f32>,
    n: usize,
    weights: &WeightBuffersCudarc,
    out: &CudaSlice<f32>,
) -> Result<(), CudarcKernelError> {
    if weights.m == 0 {
        return Ok(());
    }
    debug_assert_eq!(
        weights.n, n,
        "Issue 626: GEMV input dim must match gated-output dim"
    );
    // Fused silu-gate + quantize — writes int8 + ascale directly.
    elementwise.launch_gate_silu_quantize(
        stream, x, gate, quant_i8_buf, ascale_buf, n,
    )?;
    gemv_prequantized(stream, gemv, quant_i8_buf, ascale_buf, weights, out)?;
    Ok(())
}

/// Issue 626 — fused sigmoid-gate + quantize + single dp4a GEMV.
///
/// Computes `sigmoid(gate) * x`, quantizes to int8 in the same kernel, then
/// dispatches the dp4a GEMV for `weights` reading from the quantized buffer.
/// Saves 1 kernel launch + intermediate f32 `attn_out` memory traffic per
/// call. Used for the attention wo GEMV (single consumer of output-gated
/// attn_out).
///
/// Issue 705 — dispatches through the persistent multi launcher (1 segment).
/// This also fixes a latent Issue 697 bug: the non-accum else-branches passed
/// `&infra.gemv_multi` (the 22-arg multi kernel) into `gemv_prequantized`
/// (which pushes only the 10 single-kernel args) — undefined behavior on the
/// `forward_token_profiled` / devpos paths that never fired in the Issue 697
/// gates (they only exercised `accumulate_out = true`).
#[allow(clippy::too_many_arguments)]
fn gate_sigmoid_quantize_and_gemv(
    elementwise: &ElementwiseKernels,
    stream: &Arc<CudaStream>,
    gemv: &GemvMultiPersistent,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    gate: &CudaSlice<f32>,
    n: usize,
    weights: &WeightBuffersCudarc,
    out: &CudaSlice<f32>,
) -> Result<(), CudarcKernelError> {
    if weights.m == 0 {
        return Ok(());
    }
    debug_assert_eq!(
        weights.n, n,
        "Issue 626: GEMV input dim must match gated-output dim"
    );
    // Fused sigmoid-gate + quantize — writes int8 + ascale directly.
    elementwise.launch_gate_sigmoid_quantize(
        stream, x, gate, quant_i8_buf, ascale_buf, n,
    )?;
    gemv_prequantized_multi(stream, gemv, quant_i8_buf, ascale_buf, &[(weights, out)], false)?;
    Ok(())
}

/// Issue 627 — fused per-head RMSNorm + silu-gate + quantize + dp4a GEMV.
///
/// Computes per-head RMSNorm on `x` (with `gamma`), applies `silu(gate) *`,
/// quantizes to int8 in the same kernel, then dispatches the dp4a GEMV.
/// Saves 1 kernel launch per DeltaNet layer vs the Issue 626 path (which was
/// already RMSNorm → gate_silu_quantize → GEMV = 2 launches). This fuses
/// the RMSNorm into the gate+quantize kernel, making it 1 launch.
///
/// Used for the DeltaNet out_proj GEMV (the single consumer of the z-gated,
/// per-head-normalized recurrent_out). Issue 705 — dispatches through the
/// persistent multi launcher (1 segment); see the
/// `gate_sigmoid_quantize_and_gemv` doc for the latent Issue 697 arg-layout
/// bug this fixes on the non-accum paths.
#[allow(clippy::too_many_arguments)]
fn rmsnorm_gate_silu_quantize_and_gemv(
    elementwise: &ElementwiseKernels,
    stream: &Arc<CudaStream>,
    gemv: &GemvMultiPersistent,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    gate: &CudaSlice<f32>,
    gamma: &CudaSlice<f32>,
    n_v_heads: usize,
    head_dim: usize,
    eps: f32,
    weights: &WeightBuffersCudarc,
    out: &CudaSlice<f32>,
) -> Result<(), CudarcKernelError> {
    if weights.m == 0 {
        return Ok(());
    }
    let n = n_v_heads * head_dim;
    debug_assert_eq!(
        weights.n, n,
        "Issue 627: GEMV input dim must match recurrent_out dim"
    );
    // Fused per-head RMSNorm + silu-gate + quantize — writes int8 + ascale.
    elementwise.launch_rmsnorm_gate_silu_quantize(
        stream, x, gate, gamma, quant_i8_buf, ascale_buf, n_v_heads, head_dim, eps,
    )?;
    gemv_prequantized_multi(stream, gemv, quant_i8_buf, ascale_buf, &[(weights, out)], false)?;
    Ok(())
}

/// Issue 697 — accumulate variant of [`swiglu_quantize_and_gemv`]: the down_proj
/// GEMV writes `x[row] += acc` instead of `ffn_out[row] = acc`, folding the
/// residual add into the GEMV epilogue. `out_x` must be the residual stream
/// (`acts.x`, length == weights.m == n_embd). Bit-identical to the separate
/// `residual_add_f32` launch (same f32 operands, same order).
#[allow(clippy::too_many_arguments)]
fn swiglu_quantize_and_gemv_accum(
    elementwise: &ElementwiseKernels,
    stream: &Arc<CudaStream>,
    gemv_multi: &GemvMultiPersistent,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    gate: &CudaSlice<f32>,
    up: &CudaSlice<f32>,
    n: usize,
    weights: &WeightBuffersCudarc,
    out_x: &CudaSlice<f32>,
) -> Result<(), CudarcKernelError> {
    if weights.m == 0 {
        return Ok(());
    }
    debug_assert_eq!(
        weights.n, n,
        "Issue 697: GEMV input dim must match SwiGLU dim"
    );
    debug_assert_eq!(
        weights.m, out_x.len(),
        "Issue 697: accumulate GEMV requires m == residual length (n_embd)"
    );
    elementwise.launch_swiglu_quantize(stream, gate, up, quant_i8_buf, ascale_buf, n)?;
    gemv_prequantized_multi(
        stream,
        gemv_multi,
        quant_i8_buf,
        ascale_buf,
        &[(weights, out_x)],
        true,
    )?;
    Ok(())
}

/// Issue 697 — accumulate variant of [`rmsnorm_gate_silu_quantize_and_gemv`]
/// (DeltaNet out_proj → residual stream).
#[allow(clippy::too_many_arguments)]
fn rmsnorm_gate_silu_quantize_and_gemv_accum(
    elementwise: &ElementwiseKernels,
    stream: &Arc<CudaStream>,
    gemv_multi: &GemvMultiPersistent,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    gate: &CudaSlice<f32>,
    gamma: &CudaSlice<f32>,
    n_v_heads: usize,
    head_dim: usize,
    eps: f32,
    weights: &WeightBuffersCudarc,
    out_x: &CudaSlice<f32>,
) -> Result<(), CudarcKernelError> {
    if weights.m == 0 {
        return Ok(());
    }
    let n = n_v_heads * head_dim;
    debug_assert_eq!(
        weights.n, n,
        "Issue 697: GEMV input dim must match recurrent_out dim"
    );
    debug_assert_eq!(
        weights.m, out_x.len(),
        "Issue 697: accumulate GEMV requires m == residual length (n_embd)"
    );
    elementwise.launch_rmsnorm_gate_silu_quantize(
        stream, x, gate, gamma, quant_i8_buf, ascale_buf, n_v_heads, head_dim, eps,
    )?;
    gemv_prequantized_multi(
        stream,
        gemv_multi,
        quant_i8_buf,
        ascale_buf,
        &[(weights, out_x)],
        true,
    )?;
    Ok(())
}

/// Issue 697 — accumulate variant of [`gate_sigmoid_quantize_and_gemv`]
/// (attention wo → residual stream).
#[allow(clippy::too_many_arguments)]
fn gate_sigmoid_quantize_and_gemv_accum(
    elementwise: &ElementwiseKernels,
    stream: &Arc<CudaStream>,
    gemv_multi: &GemvMultiPersistent,
    quant_i8_buf: &CudaSlice<i8>,
    ascale_buf: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    gate: &CudaSlice<f32>,
    n: usize,
    weights: &WeightBuffersCudarc,
    out_x: &CudaSlice<f32>,
) -> Result<(), CudarcKernelError> {
    if weights.m == 0 {
        return Ok(());
    }
    debug_assert_eq!(
        weights.n, n,
        "Issue 697: GEMV input dim must match gated-output dim"
    );
    debug_assert_eq!(
        weights.m, out_x.len(),
        "Issue 697: accumulate GEMV requires m == residual length (n_embd)"
    );
    elementwise.launch_gate_sigmoid_quantize(stream, x, gate, quant_i8_buf, ascale_buf, n)?;
    gemv_prequantized_multi(
        stream,
        gemv_multi,
        quant_i8_buf,
        ascale_buf,
        &[(weights, out_x)],
        true,
    )?;
    Ok(())
}

/// Issue 616 T4 — fused quantize+dp4a GEMV (NEGATIVE RESULT, kept for
/// reproduction).
///
/// Takes f32 activations directly; all threads in the block cooperate to
/// quantize the activation into shared memory, then run the dp4a loop from
/// shared memory. Bit-identical to `gemv_into` but SLOWER in practice:
///
/// - **Per-lane on-the-fly variant:** 20.85 tok/s (register pressure from
///   holding 16 f32 + 16 int8 per lane causes local-memory spilling).
/// - **Cooperative shared-memory variant:** 36.94 tok/s (each of the M/8
///   blocks independently re-quantizes the same activation vector — M/8×
///   redundant quantization work vs the split path's single quantize).
/// - **Split path (production):** 48.4 tok/s (quantize runs once; the int8
///   buffer fits in L2 cache and is read by all GEMV blocks with no
///   redundancy).
///
/// Root cause: the split path is already optimal for this architecture
/// because (a) the 4090's 72MB L2 cache easily holds the ~5KB int8 buffer,
/// and (b) the quantize kernel is a small, fast launch (~5µs). The Issue 616
/// root-cause analysis over-estimated the quantize overhead — the actual
/// bottleneck after T3 (recurrence fix) is the dp4a kernel itself + the
/// DeltaNet elementwise ops, not the per-GEMV quantize launch.
#[allow(clippy::too_many_arguments, dead_code)]
fn gemv_fused_into(
    stream: &Arc<CudaStream>,
    gemv_fused: &CudaFunction,
    weights: &WeightBuffersCudarc,
    x: &CudaSlice<f32>,
    out: &CudaSlice<f32>,
) -> Result<(), CudarcKernelError> {
    if weights.m == 0 {
        return Ok(());
    }
    let n = weights.n;
    let ablocks = n.div_ceil(16);

    let m_i32 = weights.m as i32;
    let int16_per_row = (n / 8) as i32;
    let groups_per_row = n.div_ceil(128) as i32;
    let ablock_i32 = 16i32;
    let ablocks_i32 = ablocks as i32;
    let n_i32 = n as i32;

    // Shared memory: N int8 + ablocks f32 (ascale), both 4-byte aligned at
    // the f32 boundary since N is a multiple of 16 in practice.
    let shared_mem_bytes = (n + ablocks * 4) as u32;

    let grid_x = (weights.m as u32).div_ceil(WG_THREADS / 32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (WG_THREADS, 1, 1),
        shared_mem_bytes,
    };

    unsafe {
        stream
            .launch_builder(gemv_fused)
            .arg(&weights.codes)
            .arg(&weights.wscale)
            .arg(x)
            .arg(out)
            .arg(&m_i32)
            .arg(&int16_per_row)
            .arg(&groups_per_row)
            .arg(&ablock_i32)
            .arg(&ablocks_i32)
            .arg(&n_i32)
            .launch(cfg)
            .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────────
// Helpers
// ───────────────────────────────────────────────────────────────────────────

fn upload_f32_slice(stream: &Arc<CudaStream>, data: &[f32]) -> CudaSlice<f32> {
    if data.is_empty() {
        stream.alloc_zeros::<f32>(1).expect("alloc placeholder")
    } else {
        stream.clone_htod(data).expect("upload f32 slice")
    }
}

fn alloc_err(e: cudarc::driver::DriverError) -> CudarcKernelError {
    CudarcKernelError::Launch(e.to_string())
}

fn upload_layer_weights_cudarc(
    stream: &Arc<CudaStream>,
    l: &DeltaNetTernaryLayerWeights,
) -> GpuLayerWeightsCudarc {
    // Issue 980 T4 — the dense escape-set arms (Bonsai-2 `ssm_alpha`/`beta`).
    let dense_of = |g: &GateProjWeights| match g {
        GateProjWeights::Dense(data, _, _) => {
            if data.is_empty() {
                None
            } else {
                Some(upload_f32_slice(stream, data))
            }
        }
        GateProjWeights::Ternary(_) => None,
    };
    GpuLayerWeightsCudarc {
        dense_a: dense_of(&l.in_proj_a),
        dense_b: dense_of(&l.in_proj_b),
        in_proj_qkv: maybe_upload(stream, &l.in_proj_qkv),
        in_proj_z: maybe_upload(stream, &l.in_proj_z),
        // Issue 980: in_proj_a/b are the GateProjWeights escape-set enum —
        // ternary goes to the dp4a buffers, dense (Bonsai-2) to the fp32
        // escape-set slices consumed by `gemv_dense_f32`.
        in_proj_a: maybe_upload(stream, l.in_proj_a.as_ternary().unwrap_or(&katgpt_core::TernaryGroupWeights::new(0, 0))),
        in_proj_b: maybe_upload(stream, l.in_proj_b.as_ternary().unwrap_or(&katgpt_core::TernaryGroupWeights::new(0, 0))),
        out_proj: maybe_upload(stream, &l.out_proj),
        attn_wq: maybe_upload(stream, &l.attn_wq),
        attn_wk: maybe_upload(stream, &l.attn_wk),
        attn_wv: maybe_upload(stream, &l.attn_wv),
        attn_wo: maybe_upload(stream, &l.attn_wo),
        gate_proj: WeightBuffersCudarc::upload(stream, &l.gate_proj),
        up_proj: WeightBuffersCudarc::upload(stream, &l.up_proj),
        down_proj: WeightBuffersCudarc::upload(stream, &l.down_proj),
        input_norm: upload_f32_slice(stream, &l.input_norm),
        post_attn_norm: upload_f32_slice(stream, &l.post_attn_norm),
        conv1d_weight: if l.conv1d_weight.is_empty() { None } else { Some(upload_f32_slice(stream, &l.conv1d_weight)) },
        a_log: if l.a_log.is_empty() { None } else { Some(upload_f32_slice(stream, &l.a_log)) },
        dt_bias: if l.dt_bias.is_empty() { None } else { Some(upload_f32_slice(stream, &l.dt_bias)) },
        linear_norm: if l.linear_norm.is_empty() { None } else { Some(upload_f32_slice(stream, &l.linear_norm)) },
        attn_q_norm: if l.attn_q_norm.is_empty() { None } else { Some(upload_f32_slice(stream, &l.attn_q_norm)) },
        attn_k_norm: if l.attn_k_norm.is_empty() { None } else { Some(upload_f32_slice(stream, &l.attn_k_norm)) },
    }
}

fn maybe_upload(
    stream: &Arc<CudaStream>,
    w: &katgpt_core::TernaryGroupWeights,
) -> Option<WeightBuffersCudarc> {
    if w.rows == 0 { None } else { Some(WeightBuffersCudarc::upload(stream, w)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riir_infer_core::types::DeltaNetLayerType;

    fn cuda_or_skip() -> Option<()> {
        if CudaContext::new(0).is_err() {
            return None;
        }
        Some(())
    }

    // ── Plan 603 R2 — the host half-bit converters vs the `half` crate ──
    // (the reference implementation of IEEE f16/bf16 RN; test builds get
    // `half` via feature unification, production code deliberately does
    // not depend on it).

    #[test]
    fn f16_converter_roundtrip_matches_half_crate() {
        // Edge values + a deterministic LCG sweep over the f32 population.
        let edges = [
            0.0f32, -0.0,
            f32::MIN_POSITIVE * 2.0,          // deep subnormal region input
            5.9604645e-8,                      // 2^-24, smallest f16 subnormal
            2.9802322e-8,                      // 2^-25, the half-tip tie
            6.097555e-5,                       // largest subnormal
            6.1035156e-5,                      // smallest normal 2^-14
            1.0,
            1.0009766,                         // smallest normal step
            65504.0,                          // f16 max
            65519.0,                          // rounds DOWN to 65504
            65520.0,                          // the tie → inf (even rule)
            1e30,
            f32::INFINITY,
            f32::NAN,
        ];
        let mut state = 0x853c49e67d8a19b4u64; // SplitMix64
        let mut next = || {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        let mut xs: Vec<f32> = edges.to_vec();
        for _ in 0..200_000 {
            // Mix exponents across the whole f16-relevant range (and beyond).
            let bits = (next() as u32 & 0xff) << 23 | (next() as u32 >> 9) & 0x007f_ffff;
            xs.push(f32::from_bits(bits));
        }
        for x in xs {
            if x.is_nan() {
                let got = f32_to_f16_bits_rn(x);
                assert_ne!(got & 0x7c00, 0x7c00 - 1, "NaN must stay NaN: {got:04x}");
                continue;
            }
            let want = half::f16::from_f32(x).to_bits();
            let got = f32_to_f16_bits_rn(x);
            assert_eq!(got, want, "f16 bits mismatch at {x}: got {got:04x} want {want:04x}");
            // Widen must invert exactly what the reference widens.
            let back_got = f16_bits_to_f32(got);
            let back_want = half::f16::from_bits(want).to_f32();
            assert_eq!(
                back_got.to_bits(),
                back_want.to_bits(),
                "f16 widen mismatch at bits {got:04x}"
            );
        }
    }

    #[test]
    fn bf16_converter_matches_half_crate() {
        let mut state = 0x243f6a8885a308d3u64;
        let mut next = || {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z ^ (z >> 31)
        };
        let mut xs: Vec<f32> = vec![0.0, -0.0, 1.0, -1.0, 3.140625, f32::MAX, f32::MIN, f32::INFINITY];
        for _ in 0..200_000 {
            xs.push(f32::from_bits(next() as u32));
        }
        for x in xs {
            if x.is_nan() {
                let got = f32_to_bf16_bits_rn(x);
                assert_eq!(got & 0x7f80, 0x7f80, "NaN must stay NaN");
                assert_ne!(got & 0x007f, 0, "NaN must be quieted");
                continue;
            }
            let want = half::bf16::from_f32(x).to_bits();
            let got = f32_to_bf16_bits_rn(x);
            assert_eq!(got, want, "bf16 bits mismatch at {x}");
            let back_got = bf16_bits_to_f32(got);
            let back_want = half::bf16::from_bits(want).to_f32();
            assert_eq!(back_got.to_bits(), back_want.to_bits());
        }
    }

    /// Build a small QwenDeltaNet config suitable for a fast smoke test.
    ///
    /// All dims are multiples of 128 (the ternary group size). 2 layers:
    /// 1 Attention + 1 DeltaNet (exercises both code paths).
    fn small_test_config() -> Config {
        let mut c = Config::qwen_deltanet(
            2,
            vec![DeltaNetLayerType::Attention, DeltaNetLayerType::DeltaNet],
        );
        // Shrink the model to fit in a unit test. All dims must be multiples
        // of 128 (ternary group size).
        c.n_embd = 256;
        c.n_head = 2;
        c.n_kv_head = 2;
        c.head_dim = 128;
        c.mlp_hidden = 512;
        c.vocab_size = 256;
        c.block_size = 64;
        // The DeltaNet linear dims must also be multiples of 128.
        c.deltanet_linear_head_dim = 128;
        c.deltanet_linear_n_heads = 2;
        c.deltanet_linear_n_value_heads = 2;
        c
    }

    /// Smoke test: construct the forward struct, run one decode token, verify
    /// the dispatch chain works end-to-end without panicking and the output is
    /// finite (no NaN/Inf).
    ///
    /// Uses zero-initialized weights — the logits will be near-zero (zero ternary
    /// weights produce zero GEMV output). The purpose is to verify the kernel
    /// dispatch wiring, NOT numerical correctness (that's T8, the GOAT gate,
    /// which requires the real 27B model for a meaningful CPU-vs-GPU comparison).
    #[test]
    fn test_forward_smoke_zero_weights() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };

        let config = small_test_config();
        let weights = QwenDeltaNetTernaryWeights::zeros(&config);
        assert!(weights.invariants_hold(), "zero weights should pass invariant check");

        let mut fwd = TernaryDeltanetGpuForwardCudarc::new(&config, &weights)
            .expect("construct forward");

        fwd.set_input_token(0).expect("set_input_token");
        let logits = fwd.forward_token().expect("forward_token");

        assert_eq!(logits.len(), config.vocab_size);

        // All logits should be finite (no NaN/Inf). With zero weights, they
        // should all be exactly 0.0 (zero GEMV output + zero RMSNorm input).
        let mut nan_count = 0usize;
        let mut inf_count = 0usize;
        let mut nonzero_count = 0usize;
        for (i, &l) in logits.iter().enumerate() {
            if l.is_nan() {
                nan_count += 1;
            } else if l.is_infinite() {
                inf_count += 1;
            } else if l.abs() > 1e-10 {
                nonzero_count += 1;
                if nonzero_count <= 3 {
                    eprintln!("  nonzero logit[{i}] = {l:.6e}");
                }
            }
        }
        eprintln!(
            "[smoke] logits: {} NaN, {} Inf, {} nonzero (of {})",
            nan_count,
            inf_count,
            nonzero_count,
            logits.len()
        );
        assert_eq!(nan_count, 0, "NaN in logits");
        assert_eq!(inf_count, 0, "Inf in logits");
        // With zero weights, all logits should be exactly 0 (or very close).
        assert_eq!(nonzero_count, 0, "zero weights should produce zero logits");

        // Run a second token to verify the position counter advances and
        // the KV cache append works (pos=1 for the attention layer).
        fwd.set_input_token(1).expect("set_input_token(1)");
        let logits2 = fwd.forward_token().expect("forward_token(2)");
        assert_eq!(logits2.len(), config.vocab_size);
        let nan2 = logits2.iter().filter(|l| l.is_nan()).count();
        assert_eq!(nan2, 0, "NaN in second token logits");

        eprintln!("[smoke] 2-token forward PASS — dispatch chain wired correctly");
    }

    // ── T7.5 Phase 2/3 cross-validation helpers ──

    /// Set a small window of ternary entries to +1 in a checkerboard pattern.
    /// Uses 0.01 group scale (not the default 1.0) to keep activations in a
    /// reasonable magnitude range for the backward pass. Real ternary models use
    /// learned scales that are typically small, so this is more realistic than
    /// unit scale.
    fn seed_ternary(w: &mut katgpt_core::TernaryGroupWeights, rows: usize, cols: usize) {
        for r in 0..rows.min(w.rows) {
            for c in 0..cols.min(w.cols) {
                if (r + c).is_multiple_of(2) {
                    w.set(r, c, 1);
                }
            }
            let gpr = w.groups_per_row;
            for g in 0..gpr {
                w.group_scale[r * gpr + g] = half::f16::from_f32(0.01);
            }
        }
    }

    /// Issue 722 H2 — `set_lora` must reject non-DeltaNet / out-of-range
    /// target layers in EVERY build profile (was `debug_assert!`-only → the
    /// adapter uploaded + graph invalidated, then silently never fired in
    /// release because `apply_lora` only runs inside DeltaNet layers).
    #[test]
    fn test_set_lora_rejects_non_deltanet_layer() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let config = small_test_config();
        let weights = QwenDeltaNetTernaryWeights::zeros(&config);
        let mut fwd =
            TernaryDeltanetGpuForwardCudarc::new(&config, &weights).expect("construct forward");
        let mut rng = katgpt_core::Rng::new(42);
        let lora = riir_infer_core::deltanet::qv_lora::QvLora::new(
            8,
            config.n_embd,
            config.n_embd,
            config.n_embd,
            1.0,
            &mut rng,
        );
        // Layer 0 is Attention in small_test_config.
        let err = fwd
            .set_lora(0, &lora)
            .expect_err("attention layer must be rejected");
        assert!(
            err.to_string().contains("not a DeltaNet layer"),
            "got: {err}"
        );
        let err = fwd
            .set_lora(config.n_layer, &lora)
            .expect_err("out-of-range idx must be rejected");
        assert!(
            err.to_string().contains("n_layer"),
            "got: {err}"
        );
        // The one DeltaNet layer (idx 1) is accepted.
        fwd.set_lora(1, &lora).expect("DeltaNet layer accepted");
    }

    // ── Issue 504 T1 (Plan 370 confirm) — the closed-loop training forward ──

    /// Seed a LIVE backbone for the closed-loop gates: every ternary
    /// projection subset-seeded + the dense DeltaNet fields nonzero, so a
    /// qkv-side adapter delta can actually reach the logits. The original
    /// 504 fixtures seeded only wte/in_proj_qkv/norms/lm_head and left
    /// conv1d_weight/out_proj/linear_norm at zero — the zero conv1d
    /// annihilated the delta BEFORE it could reach anything (qkv → conv(0)
    /// → q/k/v = 0 → layer output 0), so adapted-vs-frozen max_diff was
    /// 0.0 REGARDLESS of whether the forward applied the adapter: T1a
    /// could never fire (false open-loop signature) and T1b/T1c passed
    /// vacuously. Found live on the 4090 validation run (Issue 504 T3
    /// validation step 2).
    fn seed_live_backbone(weights: &mut QwenDeltaNetTernaryWeights) {
        seed_ternary(&mut weights.wte, 16, 16);
        for layer in &mut weights.layers {
            let projections: [&mut katgpt_core::TernaryGroupWeights; 10] = [
                &mut layer.attn_wq,
                &mut layer.attn_wk,
                &mut layer.attn_wv,
                &mut layer.attn_wo,
                &mut layer.in_proj_qkv,
                &mut layer.in_proj_z,
                &mut layer.out_proj,
                &mut layer.gate_proj,
                &mut layer.up_proj,
                &mut layer.down_proj,
            ];
            for p in projections {
                // Full ROW extent, not a subset: in_proj_qkv is one 768-row
                // tensor covering [Q|K|V] — seeding only the first rows leaves
                // the K region zero, and with k=0 the DeltaNet state update
                // S += k·δ is identically zero, so the recurrence stays inert
                // (state 0, output 0) and NO qkv-side change — adapter
                // included — can ever reach the logits.
                seed_ternary(p, p.rows, 16);
            }
            // Issue 980: a/b are the GateProjWeights escape-set enum — the
            // fixtures build old-style (ternary) weights, so seed the Ternary
            // arms with the same checkerboard.
            match (&mut layer.in_proj_a, &mut layer.in_proj_b) {
                (GateProjWeights::Ternary(a), GateProjWeights::Ternary(b)) => {
                    seed_ternary(a, a.rows, 16);
                    seed_ternary(b, b.rows, 16);
                }
                _ => panic!("cudarc fixtures build old-style ternary a/b"),
            }
            // Dense DeltaNet fields: zero conv1d kills the qkv delta, zero
            // linear_norm zeroes the linear-attention output, zero out_proj
            // zeroes the layer contribution. a_log=-0.5 → decay -exp(-0.5),
            // dt_bias=0.05 → beta sigmoid(0.05) — both live.
            for v in layer.conv1d_weight.iter_mut() {
                *v = 0.1;
            }
            for v in layer.a_log.iter_mut() {
                *v = -0.5;
            }
            for v in layer.dt_bias.iter_mut() {
                *v = 0.05;
            }
            for v in layer.linear_norm.iter_mut() {
                *v = 1.0;
            }
            for g in layer.attn_q_norm.iter_mut() {
                *g = 1.0;
            }
            for g in layer.attn_k_norm.iter_mut() {
                *g = 1.0;
            }
            for g in layer.input_norm.iter_mut() {
                *g = 1.0;
            }
            for g in layer.post_attn_norm.iter_mut() {
                *g = 1.0;
            }
        }
        for g in weights.final_norm.iter_mut() {
            *g = 1.0;
        }
        seed_ternary(&mut weights.lm_head, 8, 8);
    }

    /// Deterministic non-zero Q+V adapter for the closed-loop gates: A drawn,
    /// B filled at a small constant so the adapter's forward effect is
    /// non-zero (B=0 would make attached ≡ detached — that IS step-1's
    /// identity contract and is pinned separately below).
    fn nonzero_b_adapter(config: &Config, seed: u64) -> riir_infer_core::deltanet::qv_lora::QvLora {
        let mut rng = katgpt_core::Rng::new(seed);
        let rank = 8;
        let q_dim = config.deltanet_linear_n_heads * config.deltanet_linear_head_dim;
        let v_dim = config.deltanet_linear_n_value_heads * config.deltanet_linear_head_dim;
        let mut lora = riir_infer_core::deltanet::qv_lora::QvLora::new(
            rank,
            config.n_embd,
            q_dim,
            v_dim,
            4.0,
            &mut rng,
        );
        // Small constant up-projections: the adapter output is non-zero but
        // bounded (norm_x is unit-ish after RMSNorm, so |B·(A·norm_x)| stays
        // small relative to the backbone logits).
        for v in lora.b_q.iter_mut() {
            *v = 0.05;
        }
        for v in lora.b_v.iter_mut() {
            *v = 0.05;
        }
        lora
    }

    /// Issue 504 T1 G1a — closed-loop non-vacuity: with a non-zero-B adapter
    /// attached AND enabled, the TRAINING forward's logits must differ from
    /// the frozen backbone's. (The open-loop defect was exactly that they
    /// could never differ.)
    #[test]
    fn test_training_forward_applies_attached_lora() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let config = small_test_config();
        let mut weights = QwenDeltaNetTernaryWeights::zeros(&config);
        seed_live_backbone(&mut weights);

        let adapter = nonzero_b_adapter(&config, 7);
        let frozen_logits = {
            let mut fwd =
                TernaryDeltanetGpuForwardCudarc::new(&config, &weights).expect("frozen fwd");
            fwd.set_input_token(3).expect("set token");
            let mut cache = MinimalActivationCache::new(
                1,
                config.deltanet_linear_n_value_heads,
                config.deltanet_linear_head_dim,
                config.n_embd,
            );
            let (logits, _) = fwd.forward_token_training(&mut cache).expect("frozen train fwd");
            logits
        };

        let adapted_logits = {
            let mut fwd =
                TernaryDeltanetGpuForwardCudarc::new(&config, &weights).expect("adapted fwd");
            fwd.attach_lora_layer(1, &adapter).expect("attach");
            assert!(fwd.lora_forward_enabled(), "default must be enabled");
            fwd.set_input_token(3).expect("set token");
            let mut cache = MinimalActivationCache::new(
                1,
                config.deltanet_linear_n_value_heads,
                config.deltanet_linear_head_dim,
                config.n_embd,
            );
            let (logits, _) = fwd
                .forward_token_training(&mut cache)
                .expect("adapted train fwd");
            logits
        };

        let max_diff = frozen_logits
            .iter()
            .zip(adapted_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[504-T1a] adapted-vs-frozen max_diff = {max_diff:.3e}");
        assert!(
            max_diff > 1e-4,
            "the training forward ignored the attached adapter (open-loop signature)"
        );
    }

    /// Issue 504 T1 G1b — the B=0 identity contract: a freshly-initialized
    /// adapter (standard LoRA zero-init) must leave the training forward
    /// bit-identical to the frozen backbone, so a closed-loop run's step-1
    /// loss matches the open-loop baseline exactly.
    #[test]
    fn test_training_forward_b_zero_is_identity() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let config = small_test_config();
        let mut weights = QwenDeltaNetTernaryWeights::zeros(&config);
        seed_live_backbone(&mut weights);

        let mut rng = katgpt_core::Rng::new(99);
        let q_dim = config.deltanet_linear_n_heads * config.deltanet_linear_head_dim;
        let v_dim = config.deltanet_linear_n_value_heads * config.deltanet_linear_head_dim;
        // B = 0 by construction (QvLora::new zero-inits the up-projections).
        let adapter =
            riir_infer_core::deltanet::qv_lora::QvLora::new(8, config.n_embd, q_dim, v_dim, 4.0, &mut rng);
        assert!(adapter.b_q.iter().all(|&v| v == 0.0), "B must be zero");

        let (frozen_logits, adapted_logits) = {
            let mut frozen =
                TernaryDeltanetGpuForwardCudarc::new(&config, &weights).expect("frozen fwd");
            frozen.set_input_token(5).expect("set token");
            let mut c1 = MinimalActivationCache::new(
                1,
                config.deltanet_linear_n_value_heads,
                config.deltanet_linear_head_dim,
                config.n_embd,
            );
            let out1 = frozen.forward_token_training(&mut c1).expect("frozen");

            let mut adapted =
                TernaryDeltanetGpuForwardCudarc::new(&config, &weights).expect("adapted fwd");
            adapted.attach_lora_layer(1, &adapter).expect("attach");
            adapted.set_input_token(5).expect("set token");
            let mut c2 = MinimalActivationCache::new(
                1,
                config.deltanet_linear_n_value_heads,
                config.deltanet_linear_head_dim,
                config.n_embd,
            );
            let out2 = adapted.forward_token_training(&mut c2).expect("adapted");
            (out1.0, out2.0)
        };

        let max_diff = frozen_logits
            .iter()
            .zip(adapted_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[504-T1b] B=0 identity max_diff = {max_diff:.3e}");
        assert_eq!(max_diff, 0.0, "B=0 adapter must be an exact identity");
    }

    /// Issue 504 T1 G1c — the master switch freezes EVERY forward (decode
    /// included): with a non-zero-B adapter attached and
    /// `set_lora_forward_enabled(false)`, `forward_token` must be bit-identical
    /// to the detached backbone. This is what keeps the training driver's
    /// prompt/eval windows frozen even with slots attached.
    #[test]
    fn test_forward_token_frozen_when_switch_off() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let config = small_test_config();
        let mut weights = QwenDeltaNetTernaryWeights::zeros(&config);
        seed_live_backbone(&mut weights);
        let adapter = nonzero_b_adapter(&config, 11);

        // The driver contract: `lora_forward_enabled` gates WHETHER a forward
        // applies attached slots — it is held OFF for the WHOLE frozen window,
        // never flipped mid-sequence. So each arm runs on its OWN instance at
        // pos 0 (an ON decode poisons conv/deltanet state for any later OFF
        // decode — that echo is by-design state carry, not a switch leak).
        //   frozen   : plain decode (pos 0)
        //   ON       : attached + enabled (default) — must DIFFER (non-vacuity:
        //              without it the freeze assert passes vacuously whenever
        //              the apply path is broken rather than when the switch
        //              genuinely freezes)
        //   OFF      : attached + disabled BEFORE any forward — must be
        //              bit-identical to frozen
        let (frozen_logits, on_logits, switched_logits) = {
            let mut frozen =
                TernaryDeltanetGpuForwardCudarc::new(&config, &weights).expect("frozen fwd");
            frozen.set_input_token(2).expect("set token");
            let f0 = frozen.forward_token().expect("frozen decode pos0");

            let mut adapted_on =
                TernaryDeltanetGpuForwardCudarc::new(&config, &weights).expect("adapted fwd");
            adapted_on.attach_lora_layer(1, &adapter).expect("attach");
            assert!(adapted_on.lora_forward_enabled(), "default must be enabled");
            adapted_on.set_input_token(2).expect("set token");
            let on = adapted_on.forward_token().expect("switch-on decode pos0");

            let mut adapted_off =
                TernaryDeltanetGpuForwardCudarc::new(&config, &weights).expect("off fwd");
            adapted_off.attach_lora_layer(1, &adapter).expect("attach");
            adapted_off.set_lora_forward_enabled(false);
            assert!(!adapted_off.lora_forward_enabled());
            adapted_off.set_input_token(2).expect("set token");
            let off = adapted_off.forward_token().expect("switched-off decode pos0");
            (f0, on, off)
        };

        let on_diff = frozen_logits
            .iter()
            .zip(on_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[504-T1c] switch-ON decode (non-vacuity, pos0) max_diff = {on_diff:.3e}");
        assert!(
            on_diff > 1e-4,
            "switch-ON decode must apply the attached adapter (fixture/path non-vacuity)"
        );

        let max_diff = frozen_logits
            .iter()
            .zip(switched_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[504-T1c] switch-off decode (pos0) max_diff = {max_diff:.3e}");
        assert_eq!(max_diff, 0.0, "switch-off must freeze the decode forward exactly");
    }

    /// Issue 504 T1 G1d — `update_lora_layer_weights` must refresh the slot
    /// WITHOUT realloc: after an update to different weights, the training
    /// forward must reflect the NEW adapter (this is the per-step training
    /// refresh path) and keep the shape contract (a shape-mismatched update
    /// is a loud `InvalidArg`, not silent aliasing).
    #[test]
    fn test_update_lora_layer_weights_refreshes_in_place() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let config = small_test_config();
        let weights = QwenDeltaNetTernaryWeights::zeros(&config);
        let mut fwd =
            TernaryDeltanetGpuForwardCudarc::new(&config, &weights).expect("fwd");
        let a1 = nonzero_b_adapter(&config, 21);
        let a2 = nonzero_b_adapter(&config, 22);
        assert_eq!(a1.rank, a2.rank, "test adapters must share the shape");
        fwd.update_lora_layer_weights(1, &a1).expect("first attach");
        assert_eq!(fwd.attached_lora_layers(), 1);
        fwd.update_lora_layer_weights(1, &a2).expect("in-place refresh");
        assert_eq!(fwd.attached_lora_layers(), 1, "update must NOT re-attach");
        // Shape mismatch is loud.
        let mut rng = katgpt_core::Rng::new(5);
        let wrong = riir_infer_core::deltanet::qv_lora::QvLora::new(
            16,
            config.n_embd,
            config.deltanet_linear_n_heads * config.deltanet_linear_head_dim,
            config.deltanet_linear_n_value_heads * config.deltanet_linear_head_dim,
            4.0,
            &mut rng,
        );
        let err = fwd
            .update_lora_layer_weights(1, &wrong)
            .expect_err("shape mismatch must be rejected");
        assert!(err.to_string().contains("shape mismatch"), "got: {err}");
    }

    /// Issue 722 H1/H11 — the speculative rotation must invalidate a
    /// captured graph: replaying a pre-rotation graph writes the pre-rotation
    /// logits allocation while `acts.logits` points at a pool buffer, so the
    /// post-spec graph call would silently download the WRONG position's
    /// logits (they look plausible — only a cross-check catches it). The
    /// rotation owns the invalidation (`spec_verify_dispatch` drops the
    /// graph); pre-fix this test failed at the max_diff assert.
    #[cfg(all(feature = "speculative_decode", feature = "cuda_graphs_forward"))]
    #[test]
    fn test_speculative_run_invalidates_captured_graph() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let config = small_test_config();
        let mut weights = QwenDeltaNetTernaryWeights::zeros(&config);
        // Seed non-zero weights so logits differ per position — zero weights
        // make every position's logits identical, which would make the
        // stale-graph read invisible. THE EMBEDDING IS LOAD-BEARING: without
        // seeding `wte`, every token decodes the same (zero) row and a
        // stale-graph read of the previous position's logits is
        // indistinguishable from the correct read (the first draft of this
        // test was vacuously green for exactly that reason).
        seed_ternary(&mut weights.wte, 16, 16);
        for layer in &mut weights.layers {
            seed_ternary(&mut layer.in_proj_qkv, 8, 8);
            seed_ternary(&mut layer.attn_wq, 8, 8);
            seed_ternary(&mut layer.attn_wk, 8, 8);
            seed_ternary(&mut layer.attn_wv, 8, 8);
            seed_ternary(&mut layer.attn_wo, 8, 8);
            for g in layer.input_norm.iter_mut() {
                *g = 1.0;
            }
            for g in layer.post_attn_norm.iter_mut() {
                *g = 1.0;
            }
            for g in layer.attn_q_norm.iter_mut() {
                *g = 1.0;
            }
            for g in layer.attn_k_norm.iter_mut() {
                *g = 1.0;
            }
        }
        for g in weights.final_norm.iter_mut() {
            *g = 1.0;
        }
        seed_ternary(&mut weights.lm_head, 8, 8);

        // Reference: eager decode of the same token stream.
        let tokens = [1usize, 2, 3];
        let mut eager =
            TernaryDeltanetGpuForwardCudarc::new(&config, &weights).expect("construct eager");
        let mut eager_logits = Vec::with_capacity(tokens.len());
        for &tok in &tokens {
            eager.set_input_token(tok).expect("set_input_token");
            let l = eager.forward_token().expect("eager forward");
            eager_logits.push(l);
        }

        // Graph + speculative interleaving: capture a graph, run one spec
        // verify (rotates the pool — must drop the graph), then call the
        // graph path again (must re-capture with the post-rotation addresses).
        // NOTE: graph capture requires the `new_graph_ready` construction
        // (event tracking off + dedicated non-blocking stream — capturing on
        // the default stream fails with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED).
        let mut mixed =
            TernaryDeltanetGpuForwardCudarc::new_graph_ready(&config, &weights)
                .expect("construct mixed");
        let g1 = mixed
            .forward_token_graph(tokens[0])
            .expect("graph forward (captures)");
        let out = mixed
            .forward_speculative_verify_argmax(&[tokens[1]])
            .expect("spec verify");
        assert_eq!(out.len(), 2, "K=1 verify returns K+1 argmaxes");
        let g3 = mixed
            .forward_token_graph(tokens[2])
            .expect("graph forward (re-captures)");

        // Position-2 logits must match the eager reference (graph replay is
        // bit-identical to eager dispatch per Bench 674; keep a small
        // tolerance so the gate stays about the RIGHT buffer, not fp
        // exactness). The stale-graph failure mode returns position-1's
        // logits — far outside this tolerance with seeded weights.
        let ref3 = &eager_logits[2];
        assert_eq!(g3.len(), ref3.len());
        let max_diff = g3
            .iter()
            .zip(ref3.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "post-spec graph logits diverge from eager reference (max_diff={max_diff:e}) \
             — stale graph replaying pre-rotation addresses?"
        );
        // Sanity: the pre-spec graph call matched the eager reference too.
        let d1 = g1
            .iter()
            .zip(eager_logits[0].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(d1 < 1e-4, "graph-vs-eager position 0 diverged: {d1:e}");
        eprintln!("[spec×graph] rotation invalidation PASS (pos2 max_diff={max_diff:e})");
    }
}
