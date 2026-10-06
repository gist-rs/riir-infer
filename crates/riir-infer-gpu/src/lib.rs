//! riir-infer-gpu — the GPU runtime + kernel layer of the LLM inference
//! substrate: device context, buffer upload/download helpers, the
//! pool-poison detector, the GPU transpose kernel, the persistent weight
//! buffer cache, and the CubeCL runtime (Metal/WGSL/SPIR-V via wgpu;
//! native CUDA behind `cuda_backend`).
//!
//! Upstream of every engine by design — zero `riir-*` dependencies
//! (enforced by `scripts/fence_gate.py` in CI; see BOUNDARY.md).
//!
//! Kernel tile arithmetic uses the same `a = a + b` idiom the engine
//! gpu crate ships (`#![allow(clippy::assign_op_pattern)]` there):
//! the smem-tile accumulation form is load-bearing for autovectorization
//! and `cargo clippy --fix` would red the build. Suppressed at crate
//! level because kernels live across these modules.

#![allow(clippy::assign_op_pattern)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]

pub mod buffer;
pub mod context;
pub mod pool_poison;

// GPU-side persistent weight buffer cache + the CubeCL runtime
// (JIT-compiled #[cube] kernels; the type aliases at the crate root keep
// backend-agnostic kernel dispatch readable). The wgpu-side transpose
// (gpu_transpose) was deleted 2026-09-28 — dead since Issue 572 landed the
// CubeCL-native `transpose_cubecl` beside it; see riir-infer Issue 017.
#[cfg(feature = "cubecl_runtime")]
pub mod cubecl_runtime;
#[cfg(feature = "cubecl_runtime")]
pub mod weight_buffer_cache;

// Root item re-exports (the same paths consumers use today through the
// engine gpu crate's re-export layer).
pub use buffer::{
    DownloadStaging, await_map_result, create_buffer, download_f32, download_f32_reuse,
    download_u32, download_u32_reuse, upload_f32,
};
pub use context::{GpuContext, GpuError};
#[cfg(feature = "cubecl_runtime")]
pub use cubecl_runtime::{
    ActiveComputeClient, ActiveDevice, ActiveRuntime, CubeCLContext, Handle, create_f32,
    create_u32, read_f32,
};

// ---------------------------------------------------------------------------
// P3 slice 2 (the elementwise/norm/matmul/attention family) — the CPU
// reference implementations, the GEMV/matmul/attention kernel families, the
// norm/activation/sampling kernels, and the fused epilogue set. Every
// quant-typed kernel consumes the block layouts from riir-infer-core's
// `quant` (Q4_K / Q8_KV) — the SEAM rewrite this slice carried out
// (`riir_engine::quant` -> `riir_infer_core::quant`).
// ---------------------------------------------------------------------------

/// CPU reference implementations for GPU shader validation.
pub mod cpu_reference;

#[cfg(feature = "cubecl_runtime")]
pub mod attention_causal_fused_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod attention_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod attention_q8kv_cubecl;
// The DQ phase-sensitivity fake-quant instrument (Plan 614 / Issue 026) —
// the pure spec/knobs are UNGATED (host-reference unit tests compile at
// default features); the CubeCL kernels are cubecl_runtime-gated like every
// kernel module.
pub mod dq_fakequant;
#[cfg(feature = "cubecl_runtime")]
pub mod dq_fakequant_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod elementwise_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod encoder_lane_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod epilogue;
#[cfg(feature = "gemma2_d2f")]
pub mod gemma2_d2f_sc;
#[cfg(feature = "cubecl_runtime")]
pub mod gemv_autotune;
#[cfg(feature = "cubecl_runtime")]
pub mod gemv_batched;
#[cfg(feature = "cubecl_runtime")]
pub mod gemv_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod gemv_f16_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod gemv_geglu_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod gemv_geglu_f16_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod gemv_geglu_q4k_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod gemv_q4k_batched_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod gemv_q4k_batched_rmsnorm_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod gemv_q4k_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod gemv_qkv_f16_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod gemv_qkv_q4k_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod matmul_cubecl;
#[cfg(feature = "swap_ab_gemm")]
pub mod matmul_swap_ab_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod norms_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod params_cache;
#[cfg(feature = "cubecl_runtime")]
pub mod sampling_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub mod transpose_cubecl;

// EXL3 (trellis-coded) GPU dequant — Issue 001 T7b. Consumes the core `exl3`
// reference tables (LUT/Hadamard) + zero-copy layer view as its oracle +
// upload source; bit-identical to the CPU scalar reference by construction.
#[cfg(feature = "exl3_gpu")]
pub mod exl3_dequant_cubecl;

#[cfg(feature = "cubecl_runtime")]
pub use attention_causal_fused_cubecl::CausalAttentionFusedCubeCL;
#[cfg(feature = "gemma2_d2f")]
pub use attention_cubecl::AttentionBlockCausalParams;
#[cfg(feature = "cubecl_runtime")]
pub use attention_cubecl::{AttentionCubeCL, AttentionParams};
#[cfg(feature = "cubecl_runtime")]
pub use attention_q8kv_cubecl::{AttentionQ8KVCubeCL, Q8KVBuffers};
#[cfg(feature = "cubecl_runtime")]
pub use elementwise_cubecl::{
    SigmoidCubeCL, SiluCubeCL, SituCubeCL, SoftmaxCubeCL, Split2CubeCL, Split4CubeCL, TopKCubeCL,
};
#[cfg(feature = "cubecl_runtime")]
pub use epilogue::{
    CodaReparam, DispatchBudget, GemvEpilogue, GemvResidualCubeCL, GemvResidualF16CubeCL,
    NormEpilogue, NormResidualCubeCL,
};
#[cfg(feature = "gemma2_d2f")]
pub use gemma2_d2f_sc::{
    D2fScConfig, D2fScState, ScTrainingForwardResult, compute_x0_estimate,
    init_w_sc_identity_padded, project_sc, project_sc_into,
};
#[cfg(feature = "cubecl_runtime")]
pub use gemv_autotune::GemvAutotune;
#[cfg(feature = "cubecl_runtime")]
pub use gemv_cubecl::GemvCubeCL;
#[cfg(feature = "cubecl_runtime")]
pub use gemv_f16_cubecl::{F16Handle, GemvF16CubeCL};
#[cfg(feature = "cubecl_runtime")]
pub use gemv_geglu_cubecl::GemvGegluCubeCL;
#[cfg(feature = "cubecl_runtime")]
pub use gemv_geglu_f16_cubecl::GemvGegluF16CubeCL;
#[cfg(feature = "cubecl_runtime")]
pub use gemv_geglu_q4k_cubecl::GemvGegluQ4KCubeCL;
#[cfg(feature = "cubecl_runtime")]
pub use gemv_q4k_batched_cubecl::GemvQ4KBatchedCubeCL;
#[cfg(feature = "cubecl_runtime")]
pub use gemv_q4k_batched_rmsnorm_cubecl::GemvQ4KBatchedRmsnormCubeCL;
#[cfg(feature = "cubecl_runtime")]
pub use gemv_q4k_cubecl::{GemvQ4KCubeCL, Q4KHandle};
#[cfg(feature = "cubecl_runtime")]
pub use gemv_qkv_f16_cubecl::GemvQkvF16CubeCL;
#[cfg(feature = "cubecl_runtime")]
pub use gemv_qkv_q4k_cubecl::GemvQkvQ4KCubeCL;
#[cfg(feature = "cubecl_runtime")]
pub use matmul_cubecl::MatmulCubeCL;
#[cfg(feature = "swap_ab_gemm")]
pub use matmul_swap_ab_cubecl::{MatmulSwapAb, SWAP_AB_M_THRESHOLD};
#[cfg(feature = "cubecl_runtime")]
pub use norms_cubecl::{
    LayerNormMeanBatchedCubeCL, ResidualAddCubeCL, RmsNormBatchedCubeCL, RmsNormCubeCL,
    RmsNormQkFusedCubeCL, RmsNormZgateFusedCubeCL,
};
#[cfg(all(feature = "cubecl_runtime", not(feature = "params_handle_cache")))]
pub use params_cache::params_handle;
#[cfg(feature = "cubecl_runtime")]
pub use transpose_cubecl::TransposeCubeCL;

// ---------------------------------------------------------------------------
// P3 slice 3 (the ternary gemv/gemm + metal + CUDA-raw families): the
// ternary decode GEMV family around the `gemv_ternary_cubecl` hub, the
// prefill GEMM family (cubecl cmma / simdgroup / tiled), the macOS
// metal-tensor family, the raw-CUDA prefill family, and the fused FFN /
// DeltaNet input-projection dispatchers. Weight formats are
// katgpt_core::TernaryGroupWeights bit-planes; the CUDA family compiles
// only off macOS (`not(target_os = "macos")`), the metal family only on
// macOS. `canonical_expand_forms` moved here from the engine gpu crate's
// `prefill_cuda_full` with its kernel family (the enum types it names
// live in `prefill_cuda_deltanet`).
// ---------------------------------------------------------------------------

#[cfg(feature = "ternary_gemv")]
pub mod deltanet_input_proj_fused;
#[cfg(feature = "ternary_gemm_batched")]
pub mod gemm_ternary_batched_cubecl;
#[cfg(feature = "ternary_gemm_simdgroup")]
pub mod gemm_ternary_block_contiguous_cubecl;
#[cfg(feature = "ternary_gemm_batched")]
pub mod gemm_ternary_cmma16_cubecl;
#[cfg(feature = "ternary_gemm_batched")]
pub mod gemm_ternary_cmma_i8_cubecl;
#[cfg(feature = "ternary_gemm_batched")]
pub mod gemm_ternary_cmma_i8_direct_cubecl;
#[cfg(feature = "ternary_gemm_batched")]
pub mod gemm_ternary_cmma_i8_psplit_cubecl;
#[cfg(feature = "ternary_gemm_batched")]
pub mod gemm_ternary_cmma_i8_t64_cubecl;
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub mod gemm_ternary_i8_mma_cuda_raw;
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "prefill_mmq_v2",
    not(target_os = "macos")
))]
pub mod gemm_ternary_i8_mma_v6_src;
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
pub mod gemm_ternary_metal_tensor;
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
pub mod gemm_ternary_metal_wgpu;
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
pub mod gemm_ternary_metal_zero_copy;
#[cfg(feature = "ternary_gemm_simdgroup")]
pub mod gemm_ternary_simdgroup_cubecl;
#[cfg(feature = "ternary_gemm_batched")]
pub mod gemm_ternary_tiled_cubecl;
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub mod gemv_ternary_block_contiguous_cubecl;
#[cfg(feature = "ternary_gemv")]
pub mod gemv_ternary_cubecl;
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub mod gemv_ternary_cuda_raw;
#[cfg(feature = "ternary_gemv")]
pub mod gemv_ternary_fma_cubecl;
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv_residual"))]
pub mod gemv_ternary_residual_cubecl;
#[cfg(feature = "ternary_gemv")]
pub mod gemv_ternary_scale_ab_cubecl;
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub mod prefill_cuda_attention;
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub mod prefill_cuda_attention_fa;
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub mod prefill_cuda_attention_gang;
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub mod prefill_cuda_attention_vec;
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub mod prefill_cuda_deltanet;
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub mod prefill_cuda_ffn;
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub mod prefill_cuda_gdn_chunked;
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub mod prefill_cuda_mma;
#[cfg(feature = "ternary_gemv")]
pub mod ternary_ffn_fused;

#[cfg(feature = "ternary_gemv")]
pub use deltanet_input_proj_fused::{
    GpuTernaryInputProj, InputProjOutputs, LayerInputProjWeightsRef, TernaryInputProjFused,
    TernaryInputProjHandles,
};
#[cfg(feature = "ternary_gemm_batched")]
pub use gemm_ternary_batched_cubecl::GemmTernaryBatchedCubeCL;
#[cfg(feature = "ternary_gemm_simdgroup")]
pub use gemm_ternary_block_contiguous_cubecl::GemmTernaryBlockContiguousCubeCL;
#[cfg(feature = "ternary_gemm_batched")]
pub use gemm_ternary_cmma_i8_cubecl::GemmTernaryCmmaI8CubeCL;
#[cfg(feature = "ternary_gemm_batched")]
pub use gemm_ternary_cmma16_cubecl::GemmTernaryCmma16CubeCL;
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "prefill_q8_act",
    not(target_os = "macos")
))]
pub use gemm_ternary_i8_mma_cuda_raw::q8_act_enabled;
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub use gemm_ternary_i8_mma_cuda_raw::{
    FoldMode, GemmI8MmaError, GemmI8MmaScratch, GemmTernaryI8MmaCuda, MmaGen, MmaTile, QuantDiv,
    mma_v2_enabled, mma_v5_enabled,
};
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "prefill_mmq_v2",
    not(target_os = "macos")
))]
pub use gemm_ternary_i8_mma_cuda_raw::{mmq_fmt_mode, mmq_v2_enabled};
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
pub use gemm_ternary_metal_tensor::MetalTensorGemm;
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
pub use gemm_ternary_metal_wgpu::MetalTensorWgpuGemm;
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
pub use gemm_ternary_metal_zero_copy::{MetalTensorZeroCopyGemm, ZeroCopyWeightCache};
#[cfg(feature = "ternary_gemm_simdgroup")]
pub use gemm_ternary_simdgroup_cubecl::GemmTernarySimdgroupCubeCL;
#[cfg(feature = "ternary_gemm_batched")]
pub use gemm_ternary_tiled_cubecl::{
    GemmTernaryTiled8x8CubeCL, GemmTernaryTiledCubeCL, GemmTernaryTiledXfixCubeCL,
};
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub use gemv_ternary_block_contiguous_cubecl::{
    GemvTernaryBlockContiguousCubeCL, TernaryHandleBlockContiguous,
};
#[cfg(feature = "ternary_gemv")]
pub use gemv_ternary_cubecl::{
    GemvTernaryCubeCL, GpuTernaryMatvec, InterleavedTernaryHandle, TernaryHandle,
    TernaryTritHandle, gemv_f16_scale_launch_count, prepare_block_contiguous_u32,
    set_gemv_use_f16_scale,
};
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub use gemv_ternary_cuda_raw::{
    GpuTernaryMatvecDp4a, TernaryGemmCudaRaw, TernaryGemmCudaRawError, WG_THREADS,
};
#[cfg(feature = "ternary_gemv")]
pub use gemv_ternary_fma_cubecl::{
    GemvTernaryFmaCubeCL, TernaryHandleFma, fma_enabled, fma_launch_count, pack_u32_digit_bytes,
    set_gemv_use_fma,
};
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv_residual"))]
pub use gemv_ternary_residual_cubecl::GemvTernaryResidualCubeCL;
#[cfg(feature = "ternary_gemv")]
pub use gemv_ternary_scale_ab_cubecl::GemvTernaryScaleAbCubeCL;
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub use prefill_cuda_attention::{
    AttnAcc, AttnDot, AttnInv, AttnPh3, AttnRes, CudaAttnKernels, RopeRot, SinCosForm,
};
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub use prefill_cuda_attention_fa::fa_scratch_rows;
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub use prefill_cuda_deltanet::{
    ConvFma, CudaDeltanetKernels, L2Accum, L2Inv, LogForm, PlaneSum, RecDot, RecScale, RecUpd,
    SigDiv, SigExp, canonical_expand_forms,
};
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub use prefill_cuda_ffn::{
    CudaFfnKernels, FfnAccum, FfnExp, FfnInvSqrt, FfnMean, FfnRecip, canonical_rmsnorm_forms,
    canonical_swiglu_forms, set_prefill_use_cuda_ffn,
};
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub use prefill_cuda_gdn_chunked::{CudaGdnChunkedKernels, GDN_CHUNK};
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub use prefill_cuda_mma::set_prefill_use_cuda_mma;
#[cfg(feature = "ternary_gemv")]
pub use ternary_ffn_fused::{
    GpuTernaryFfn, LayerFfnWeightsRef, TernaryFfnFused, TernaryFfnHandles,
};

// ---------------------------------------------------------------------------
// P3 slice 4a (riir-ai Plan 610): the Qwen full-attention family and the
// DeltaNet CLEAN kernels. The qwen family (decode/prefill gated kernels +
// the m16/m32/m64/m32-pipe/m32-kvf16/cmma/cmma-pv tiled-flash arms of
// Issues 771/844 + the q8-KV prefill arm of Plan 562) depends only on the
// CubeCL runtime and the params-handle cache; the deltanet kernels
// (chunked conv1d, the delta-rule chunked prefill of Issue 734 T4, the
// tree-masked batched verify of Issue 721) are self-contained around
// cubecl. Consumers reach them through the engine gpu crate's re-export
// layer (paths unchanged there).
// ---------------------------------------------------------------------------

/// Qwen3.5 full-attention layer GPU kernels (Issue 599 GPU-resident forward).
#[cfg(feature = "cubecl_runtime")]
pub mod qwen_attention_cubecl;
// Issue 771 / Bench 808: the M=16 two-rows-per-plane tiled flash arm.
#[cfg(feature = "cubecl_runtime")]
pub mod qwen_attention_prefill_m16_cubecl;
// Issue 844 T2: the M=32 four-rows-per-plane tiled flash arm.
#[cfg(feature = "cubecl_runtime")]
pub mod qwen_attention_prefill_m32_cubecl;
// Issue 844 T5: the M=64 eight-rows-per-plane arm (opt-in).
#[cfg(feature = "cubecl_runtime")]
pub mod qwen_attention_prefill_m64_cubecl;
// Issue 844 T5 residue: the K/V register-prefetch PIPELINED m32 arm.
#[cfg(feature = "cubecl_runtime")]
pub mod qwen_attention_prefill_m32_pipe_cubecl;
// Issue 844 T5-width: the f16-KV traffic-WIDTH probe arm.
#[cfg(feature = "cubecl_runtime")]
pub mod qwen_attention_prefill_m32_kvf16_cubecl;
// Issue 771 / Bench 809: the cmma score-matrix arm.
#[cfg(feature = "cubecl_runtime")]
pub mod qwen_attention_prefill_cmma_cubecl;
// Issue 771 / Bench 810: the PV-cmma successor arm.
#[cfg(feature = "cubecl_runtime")]
pub mod qwen_attention_prefill_cmma_pv_cubecl;
// Issue 771 T2c-a / Plan 562: the Q8-KV prefill flash arm.
#[cfg(feature = "cubecl_runtime")]
pub mod qwen_prefill_q8kv_cubecl;
// DeltaNet chunkwise parallel prefill (Issue 652 / Plan 533).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
pub mod deltanet_chunked_cubecl;
// Correct Delta-rule chunkwise parallel prefill (Issue 734 T4 — the Bench 662
// recorded algorithm).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill"
))]
pub mod deltanet_delta_rule_chunked;
// Tree-masked batched verify for DeltaNet layers (Issue 721 T2+T3).
#[cfg(all(feature = "cubecl_runtime", feature = "speculative_tree_verify"))]
pub mod deltanet_tree_verify_cubecl;

// ─────────────────────────────────────────────────────────────────────────────
// Plan 610 S4b (2026-09-23): the qwen38/cudarc SEAM cluster + the deltanet
// forward family + the ane_prefill rider. Moved from
// riir-ai/crates/riir-gpu; the SEAM rewrites are `riir_engine::` ->
// `riir_infer_core::` (all resolve through the carve's re-export surface).
// Held OUT: cudarc_kernels/backward.rs (gradient kernels = training surface,
// stays riir-gpu-side as `cudarc_backward_kernels`) and the
// gpu_training_resident-gated collector pair (stripped; the collector module
// stays riir-gpu-side as residue).
// ─────────────────────────────────────────────────────────────────────────────

// DeltaNet recurrence/state kernels shared by the CubeCL forward family.
// Gate WIDENED-in-truth from the home crate's bare `cubecl_runtime`:
// DeltaNetStateBuffers::new takes `&[DeltaNetLayerType]`, and that type
// exists only when the katgpt deltanet chain is on — which every real
// consumer reaches through ternary_gemv (the forward family) or the
// deltanet_inference mirror (hybrid_dispatch). Both legs flow through
// infer-core's own features, keeping variant+match-arm in sync there
// (a DIRECT katgpt-core leg here would compile the variant without
// infer-core's match arms — E0004, the S4b measured trap).
#[cfg(all(
    feature = "cubecl_runtime",
    any(feature = "ternary_gemv", feature = "deltanet_inference")
))]
pub mod deltanet_cubecl;
#[cfg(all(
    feature = "cubecl_runtime",
    any(feature = "ternary_gemv", feature = "deltanet_inference")
))]
pub use deltanet_cubecl::DeltaNetStateBuffers;
// riir-ai Issue 1004 R1: the multi-token recurrence with threadgroup-staged
// k/q/v (bit-identical to `DeltanetRecurrenceMultiTokenCubeCL`, opt-in).
#[cfg(feature = "deltanet_recurrence_smem_staged")]
pub mod deltanet_recurrence_staged_cubecl;
// riir-ai Issue 1004 R2: the fused GDN prework — conv1d + SiLU + q/k L2-norm
// + head expansion in ONE dispatch (bit-identical to the chunked-conv +
// batched-expand chain by construction, opt-in).
#[cfg(feature = "deltanet_prework_fused")]
pub mod deltanet_prework_fused_cubecl;

// Bonsai-2 ternary weight rotation tables (CubeCL arm).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub mod deltanet_rotation_cubecl;
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub use deltanet_rotation_cubecl::{RotationCubeCL, RotationTablesCubeCL};

// Bonsai-2 ternary weight rotation tables (cudarc arm, CUDA-only).
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub mod deltanet_rotation_cudarc;

// Fused pre-recurrence CubeCL kernel (GDN dispatch-fusion lane, opt-in).
// The any(test, ..) arm also admits `deltanet_inference`: the fused-parity
// tests consume crate::deltanet_cubecl, whose own gate carries that feature
// at the new home (the engine gpu crate gated it ternary_gemv-only — the
// latent coupling the S4b gate-tightening pass separated).
#[cfg(all(
    feature = "cubecl_runtime",
    any(test, feature = "ternary_gemv", feature = "deltanet_inference")
))]
pub mod deltanet_pre_rec_fused_cubecl;
#[cfg(all(
    feature = "cubecl_runtime",
    any(test, feature = "ternary_gemv", feature = "deltanet_inference")
))]
pub use deltanet_pre_rec_fused_cubecl::{
    DeltanetPreRecFusedCubeCL, deltanet_fused_pre_rec_launch_count, set_deltanet_fused_pre_rec,
};

// ---------------------------------------------------------------------------
// P3 slice 5 (the gemma kernel cluster — riir-ai Plan 610 S5). The gemma2
// CubeCL decode stack + its cluster dependents (llama, d2f, q4k weights,
// gemma4) + the RoPE/GeGLU + Wall kernel riders. The wall_config hinge:
// `WallConfig` re-homes to riir-infer-core (this crate consumes it as
// `riir_infer_core::wall_config::WallConfig` — the one engine-local import
// that forced gemma2_cubecl STAYS in the T2 audit). gemma2_forward +
// forward_prefill stay engine-side: their WGSL zoo (`kernels::GpuPipelines`)
// is shared with training/game consumers and cannot move to this
// inference-only repo (the audit's "share or split at P3" adjudication —
// its own slice).
// ---------------------------------------------------------------------------

// Gemma 2 CubeCL decode stack: GEMV/flash-attention dispatch, KV cache,
// weight buffers (Plan 106 T2.6 + 087 Phase 4.4-4.7).
#[cfg(feature = "cubecl_runtime")]
pub mod gemma2_cubecl;
// Gemma 2 Q4_K quantized weight upload (Plan 087 Phase 4.8) — wgpu-only,
// ungated like its engine home.
pub mod gemma2_q4k_weights;
// Llama CubeCL decode stack (consumes the gemma2 dispatch machinery).
#[cfg(feature = "cubecl_runtime")]
pub mod llama_cubecl;
// Gemma2 D2F block-causal decode (dllm lane; the feature also turns on
// fastrand + infer-core's dllm Config fields).
#[cfg(feature = "gemma2_d2f")]
pub mod gemma2_d2f;
// Gemma4 CubeCL stack (partial RoPE / QK-Norm / GeGLU / layer output scale).
#[cfg(feature = "gemma4_gpu")]
pub mod gemma4_cubecl;
// GPU-side RoPE + GeGLU kernels (Plan 106 T2.13) — the gemma decode
// cluster's positional/activation path.
#[cfg(feature = "cubecl_runtime")]
pub mod rope_geglu_cubecl;
// Wall Attention CubeCL kernels (Plan 173; gated like its engine home).
#[cfg(all(feature = "wall_attention", feature = "cubecl_runtime"))]
pub mod wall_cubecl;
// Shared GPU test helpers (Issue 712 heavy-model gate + page release).
// COPIED, not moved — the engine gpu crate keeps its own for the staying
// gemma2_forward tests (test-only infra, the cross-repo duplication class).
#[cfg(test)]
mod test_gpu_support;

// The CubeCL whole-forward: decode + prefill over the ternary Bonsai family.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub mod ternary_deltanet_gpu_forward;
// Root item surface (mirrors the engine gpu crate's root re-exports, so a
// retargeted consumer swaps `riir_gpu::` for `riir_infer_gpu::` with no
// other path changes).
#[cfg(all(
    feature = "ternary_gemm_batched",
    feature = "metal_tensor_gemm",
    target_os = "macos"
))]
pub use ternary_deltanet_gpu_forward::set_prefill_use_metal_tensor_zerocopy;
#[cfg(all(feature = "ternary_gemm_batched", feature = "ternary_gemm_simdgroup"))]
pub use ternary_deltanet_gpu_forward::set_prefill_use_simdgroup;
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub use ternary_deltanet_gpu_forward::{DequantWteRowCubeCL, TernaryDeltanetGpuForward};

// The speculative-decode drafting family (Issue 665) — modelless CPU-side
// draft models + the routing half, moved from the engine gpu crate (its
// gguf_loader edge resolves through riir-infer-core here). The verify/checkpoint
// dispatch methods remain on TernaryDeltanetGpuForward below.
#[cfg(feature = "speculative_decode")]
pub mod speculative_decode;

// The cudarc whole-forward (dp4a GEMV path, CUDA-only).
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub mod ternary_deltanet_gpu_forward_cudarc;

// Plan 618 item 5 (Issue 028 T4) — the cudarc Q4_K GEMV: the disaggregated
// container's prefill-copy format on the GPU. Same lane as the whole-forward
// (consumes its shared int8 activation buffers + WG_THREADS grid shape).
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub mod gemv_q4k_cuda_raw;
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub use gemv_q4k_cuda_raw::{Q4KGemvKernels, launch_gemv_q4k, launch_gemv_q4k_multi, q4k_blocks_as_bytes};

// Plan 618 S3 (Issue 028 T4) — the cudarc Q6_K GEMV: the matched-storage
// single-checkpoint control arm, the q4 module's twin (same int8 activation
// buffers, same grid shape, no min offset).
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub mod gemv_q6k_cuda_raw;
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub use gemv_q6k_cuda_raw::{Q6KGemvKernels, launch_gemv_q6k, launch_gemv_q6k_multi, q6k_blocks_as_bytes};

// The whole-prefill cudarc composition (mma/ffn/deltanet/attention kernels).
// Gate TIGHTENED from the home crate's `all(cubecl_runtime, not(macos))` to
// the cuda_raw posture its consumers' item re-exports already carry: the
// module launches cudarc kernels directly and reads
// `crate::deltanet_rotation_cudarc` ungated, so the loose decl only compiled
// because no real build ever separated cubecl_runtime from
// ternary_gemv_cuda_raw (Plan 610 S4b rider).
#[cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    not(target_os = "macos")
))]
pub mod prefill_cuda_full;

// Tree-verify driver over the S4a tree-mask kernels.
#[cfg(all(feature = "speculative_tree_verify", feature = "ternary_gemm_batched"))]
pub mod ternary_tree_verify_driver;

// Hybrid layer dispatcher (deltanet_inference-gated internals).
pub mod hybrid_dispatch;

// VRAM pre-flight budget checks (the forward's construction gate). Gate
// TIGHTENED from the home crate's bare `cubecl_runtime` to include
// ternary_gemv: the budget reads ternary_weights + the forward's SPEC_MAX_K
// (both ternary-gated) and its consumer is the ternary forward itself
// (Plan 610 S4b rider — the home posture always carried the ternary chain
// through default-feature unification).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub mod vram_budget;

// Weight readback + manifest (the forward's freeze/thaw diagnostics).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub mod weight_readback;

// Semantic-state readback (the Issue 994 probe family's manifest).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
pub mod state_readback;

// Dual ANE/GPU hybrid prefill (Issue 726; moved with the family — the
// forward's 125 ane_prefill cfg sites reach it as `crate::ane_prefill`).
#[cfg(feature = "ane_prefill")]
pub mod ane_prefill;

// Raw cudarc kernel wrappers (nvrtc-compiled CUDA sources). The backward
// (gradient) half stays engine-side: riir-gpu's `cudarc_backward_kernels`.
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub mod cudarc_kernels;
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub use cudarc_kernels::{
    AttentionKernels, AttentionScoreMmaKernels, CudarcKernelError, DeltanetKernels,
    ElementwiseKernels, EmbeddingDequantKernels, LoraDecodeKernels, QvLoraGpuCudarc,
    SPLITGQA_QG_ROWS_MAX_P,
};

// The qwen38 dense cudarc forward family (Issue 742; CUDA-only).
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub mod qwen38_dense_cudarc;
pub mod qwen38_dflash2;
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub mod qwen38_dflash2_gpu;
// Plan 614 Phase 2 — the keyed spec-decode lane: the pure decision layer is
// always-on (M3-testable); the cudarc loop composition rides `qwen38_spec`
// (implies ternary_gemv_cuda_raw; CUDA-only like its target).
pub mod qwen38_spec;
#[cfg(all(feature = "qwen38_spec", not(target_os = "macos")))]
pub mod qwen38_spec_cudarc;
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub mod qwen38_prefix_cache;
#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
pub mod qwen38_verify_mma;
