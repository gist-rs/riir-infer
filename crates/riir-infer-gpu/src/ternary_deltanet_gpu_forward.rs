//! GPU-resident forward for ternary Qwen3.5 DeltaNet (Issue 599 unblock).
//!
//! Keeps all activations on GPU between layers — eliminates per-dispatch
//! upload/download overhead (80% of per-token time per Bench 603).
//!
//! ## What this does
//!
//! The existing hook-based forward (`forward_qwen_deltanet_ternary_with_hook`)
//! uploads input + downloads output for EVERY ternary GEMV dispatch (~225
//! dispatches/token). Each dispatch pays ~1.9ms in wgpu sync overhead, totaling
//! ~430ms/token — 80% of the 1220ms forward time.
//!
//! This module keeps activations in persistent GPU buffers. The only CPU↔GPU
//! transfer per token is:
//! - Upload: token embedding (20 KB, once)
//! - Download: logits (993 KB, once)
//!
//! All 64 layers execute as a chain of GPU dispatches with zero sync points.
//!
//! ## Reused kernels
//!
//! - `GemvTernaryCubeCL` — ternary bit-plane GEMV (weight handles pre-uploaded)
//! - `RmsNormCubeCL` — RMS normalization
//! - `ResidualAddCubeCL` — residual connection
//! - `DeltanetConv1dCubeCL` — depthwise conv1d + SiLU
//! - `DeltanetRecurrenceCubeCL` — gated delta rule recurrence
//! - `DeltanetBetaDecayCubeCL` — beta/decay from raw projections
//! - `ExpandAndL2NormalizeHeadsCubeCL` — fused head-expansion + L2-norm + V-copy
//! - `DeltanetZGatingCubeCL` — z-gated output
//! - `DeltanetGatingCubeCL` — SwiGLU activation for FFN

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};
#[cfg(feature = "cubecl_runtime")]
use crate::deltanet_cubecl::{
    DeltanetBetaDecayCubeCL, DeltanetConv1dCubeCL, DeltanetGatingConcatCubeCL,
    DeltanetGatingCubeCL, DeltanetRecurrenceCubeCL, ExpandAndL2NormalizeHeadsCubeCL,
};
// Posture split (Plan 610 S4b): only the batched-prefill/chunked arms launch
// this kernel — gate the import with its uses so the plain decode posture
// stays unused-import-clean.
#[cfg(feature = "ternary_gemm_batched")]
use crate::deltanet_cubecl::DeltanetZGatingCubeCL;
/// Issue 637 T5: batched variants used only by `prefill`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
use crate::deltanet_cubecl::{
    DeltanetBetaDecayBatchedCubeCL, ExpandAndL2NormalizeHeadsBatchedCubeCL,
};
#[cfg(feature = "cubecl_runtime")]
use crate::elementwise_cubecl::Split4CubeCL;
#[cfg(all(feature = "cubecl_runtime", feature = "deltanet_recurrence_rowpar"))]
use crate::deltanet_cubecl::DeltanetRecurrenceRowParCubeCL;
#[cfg(feature = "cubecl_runtime")]
use crate::elementwise_cubecl::{CopyCubeCL, FillZerosCubeCL, Split2CubeCL};
use crate::gemv_cubecl::GemvCubeCL;
use crate::RotationCubeCL;
#[cfg(feature = "cubecl_runtime")]
use crate::gemv_ternary_cubecl::{GemvTernaryCubeCL, TernaryHandle};
// Issue 637 T3: batched prefill projections. Opt-in — the decode path never
// touches this and keeps using the GEMV above.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
use crate::gemm_ternary_batched_cubecl::GemmTernaryBatchedCubeCL;
// Issue 641 / Bench 645: simdgroup-matrix (cmma) ternary GEMM. The hardware
// cooperative-matrix path — 1.89× roll-up (8×32 variant) vs the plane-
// cooperative kernel's 1.08×. Opt-in; wired into `prefill_project` behind
// `PREFILL_USE_SIMDGROUP` when both features are compiled.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_gemm_simdgroup"
))]
use crate::gemm_ternary_simdgroup_cubecl::GemmTernarySimdgroupCubeCL;
// Plan 533 / Issue 652: chunked DeltaNet prefill (conv1d + recurrence).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill"
))]
use crate::deltanet_chunked_cubecl::DeltanetChunkedConv1dCubeCL;
// Issue 734 T4's multi-token recurrence launcher. Gated to match the type's own
// gate (deltanet_cubecl) + the dispatch site below: ungated, it breaks the
// public `cubecl_runtime + ternary_gemv` combo without
// `ternary_deltanet_chunked_prefill` (the exact riir-clippy `ternary_inference`
// dep shape — Issue 724's recorded pre-existing finding).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill"
))]
use crate::deltanet_cubecl::DeltanetRecurrenceMultiTokenCubeCL;
#[cfg(feature = "cubecl_runtime")]
use crate::norms_cubecl::{ResidualAddCubeCL, ResidualAddRmsNormCubeCL, RmsNormCubeCL, RmsNormQkFusedCubeCL, RmsNormZgateFusedCubeCL};
#[cfg(feature = "ternary_gemm_batched")]
use crate::norms_cubecl::RmsNormBatchedCubeCL;
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv_residual"))]
use crate::gemv_ternary_residual_cubecl::GemvTernaryResidualCubeCL;
#[cfg(feature = "cubecl_runtime")]
use crate::qwen_attention_cubecl::{
    QwenAttentionDecodeGatedCubeCL, QwenAttentionDecodeGatedSplitCubeCL,
    QwenAttentionDecodeGatedCombineCubeCL, split_decode_geometry,
    QwenKvCacheAppendCombinedCubeCL, QwenRopePartialCubeCL,
    QwenSplitQgCubeCL,
};
// Issue 936: the imports below feed ONLY the batched-prefill ladder
// (`prefill_attention_layer_batched` + the q8 scratch getter, both gated
// `all(cubecl_runtime, ternary_gemm_batched, ternary_attention_batched_prefill)`).
// Gated with their use sites — on `cubecl_runtime` alone they read as unused
// imports on every lane that compiles this module without the full ladder
// (the riir-clippy consumer set names exactly such a combination).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
use crate::qwen_attention_cubecl::{
    QwenAttentionPrefillGatedCubeCL,
    QwenAttentionPrefillTiledCubeCL,
    QwenKvCacheFillSplitBatchedCubeCL, QwenRopePartialBatchedCubeCL,
    QwenSplitKvBatchedCubeCL, QwenSplitQgBatchedCubeCL,
};
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
use crate::qwen_attention_prefill_m16_cubecl::QwenAttentionPrefillTiledM16CubeCL;
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
use crate::qwen_attention_prefill_m32_cubecl::QwenAttentionPrefillTiledM32CubeCL;
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
use crate::qwen_attention_prefill_m64_cubecl::QwenAttentionPrefillTiledM64CubeCL;
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
use crate::qwen_attention_prefill_m32_pipe_cubecl::QwenAttentionPrefillTiledM32PipeCubeCL;
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
use crate::qwen_attention_prefill_cmma_cubecl::QwenAttentionPrefillTiledCmmaCubeCL;
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
use crate::qwen_attention_prefill_cmma_pv_cubecl::QwenAttentionPrefillTiledCmmaPvCubeCL;
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
use crate::qwen_prefill_q8kv_cubecl::{launch_kv_quantize_q8, QwenAttentionPrefillTiledQ8CubeCL};

use riir_infer_core::deltanet::ternary_weights::{
    DeltaNetTernaryLayerWeights, QwenDeltaNetTernaryWeights,
};
// Issue 980 T4-ALT — the folded-model marker type on `TernaryDeltanetGpuForward`.
use riir_infer_core::deltanet::rotation::TernaryRotationConfig;
use riir_infer_core::types::{Config, DeltaNetLayerType};

#[cfg(feature = "cubecl_runtime")]
use riir_infer_core::deltanet::minimal_activation_cache::{
    MinimalActivationCache, MinimalLayerActivations,
};

// ───────────────────────────────────────────────────────────────────────────
// GPU-side embedding dequant kernel (Issue 604 T8 G4 — alloc-free set_input_token)
// ───────────────────────────────────────────────────────────────────────────
//
// The ternary wte table is stored as bit-planes (pos_bits_u32 + neg_bits_u32)
// plus per-group f16 scales. Dequantizing a single row for the current token
// is embarrassingly parallel — one thread per output column. This kernel
// eliminates the CPU-side `dequant_wte_row_into` call AND the per-token
// `create_from_slice` GPU allocation in `set_input_token`, making the
// embedding-lookup path truly alloc-free (G4).
//
// The bit-extraction layout mirrors `gemv_ternary_plane` in `gemv_ternary_cubecl.rs`:
//   col c → u64 block `c/64` → u32 word `(c%64)/32` → bit `c%32`
// Each u64 block occupies 2 consecutive u32 elements in the flattened handle.

/// Bits per u64 block (the ternary bit-plane unit).
const BITS_PER_BLOCK_DW: u32 = 64;
/// Ternary group size (matches `katgpt-types::GROUP_SIZE`).
const GROUP_SIZE_DW: u32 = 128;

/// CubeCL kernel: dequantize a single wte row into the output buffer.
///
/// One thread per output column. Reads the ternary bit-planes + group scale
/// for the given `row_idx`, reconstructs the ternary value, multiplies by
/// the group scale, and writes the result to `out[col]`.
///
/// # Parameters (scalar — zero Rust-heap allocation per token)
///
/// All shape/index arguments are passed as scalar `u32` (not an `Array<f32>` params
/// buffer) to avoid `client.create_from_slice` heap allocation — G4 alloc-free.
///
/// - `pos_bits_u32`: positive bit-plane, u64→2×u32 cast. Layout: `[rows * blocks64 * 2]`.
/// - `neg_bits_u32`: negative bit-plane, same layout.
/// - `group_scale_f32`: per-group f32 scales (f16 decoded at upload). Layout: `[rows * groups_per_row]`.
/// - `out`: output row `[n]` (f32). Must be pre-allocated.
/// - `row_idx`, `blocks64`, `groups_per_row`, `n`: scalar shape/index args.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn dequant_wte_row_f32(
    pos_bits_u32: &[u32],
    neg_bits_u32: &[u32],
    group_scale_f32: &[f32],
    out: &mut [f32],
    row_idx: u32,
    blocks64: u32,
    groups_per_row: u32,
    n: u32,
) {
    let col = ABSOLUTE_POS_X;
    if col >= n {
        terminate!();
    }

    // Locate the ternary value for (row_idx, col).
    let words_per_row = blocks64 * 2u32; // u64 → 2× u32
    let block_idx = col / BITS_PER_BLOCK_DW;
    let word_in_block = (col % BITS_PER_BLOCK_DW) / 32u32;
    let bit_pos = col % 32u32;

    let word_idx = (row_idx * words_per_row + block_idx * 2u32 + word_in_block) as usize;
    let pos_bit = (pos_bits_u32[word_idx] >> bit_pos) & 1u32;
    let neg_bit = (neg_bits_u32[word_idx] >> bit_pos) & 1u32;

    let sign_f = pos_bit as f32 - neg_bit as f32; // +1, 0, or -1

    // Per-group scale.
    let group = col / GROUP_SIZE_DW;
    let scale = group_scale_f32[(row_idx * groups_per_row + group) as usize];

    out[col as usize] = sign_f * scale;
}

/// Safe launcher wrapper for the wte dequant kernel.
/// `pub` — exported as the oracle for the Issue 734 Arm 8 CUDA twin's
/// bit-identity probe.
#[cfg(feature = "cubecl_runtime")]
pub struct DequantWteRowCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl DequantWteRowCubeCL {
    /// Launch the wte dequant kernel.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `pos_bits_u32` / `neg_bits_u32` have `rows * blocks64 * 2` u32 elements
    /// - `group_scale_f32` has `rows * groups_per_row` f32 elements
    /// - `out` has `n` f32 bytes allocated
    /// - `row_idx < rows`
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        pos_bits_u32: Handle,
        neg_bits_u32: Handle,
        group_scale_f32: Handle,
        out: Handle,
        row_idx: u32,
        blocks64: u32,
        groups_per_row: u32,
        n: u32,
    ) {
        let wg_size = 256u32;
        let num_wg = n.div_ceil(wg_size);

        // Bit-plane handles cover all rows; the kernel indexes the correct row
        // via `row_idx`. Lengths here are upper bounds for CubeCL metadata.
        let plane_len = (row_idx as usize + 1) * blocks64 as usize * 2;
        let scale_len = (row_idx as usize + 1) * groups_per_row as usize;

        unsafe {
            dequant_wte_row_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(pos_bits_u32, plane_len),
                BufferArg::from_raw_parts(neg_bits_u32, plane_len),
                BufferArg::from_raw_parts(group_scale_f32, scale_len),
                BufferArg::from_raw_parts(out, n as usize),
                row_idx,
                blocks64,
                groups_per_row,
                n,
            );
        }
    }
}

/// Per-layer pre-uploaded GPU weight handles.
#[cfg(feature = "cubecl_runtime")]
/// Per-layer GPU weight handles. `pub(crate)` — consumed by the Issue 721
/// tree-verify driver (sibling module).
pub(crate) struct GpuLayerWeights {
    // Issue 727 H2a: the four separate in_proj handles are dispatched ONLY by
    // `prefill` (which is gated on `ternary_gemm_batched`); the decode path
    // uses `in_proj_concat` below. Gating them on the same feature keeps the
    // default build free of the duplicated input-projection weight set.
    // pub(crate) — also consumed by the Issue 734 Arm 8 whole-prefill cudarc
    // driver (sibling module).
    #[cfg(feature = "ternary_gemm_batched")]
    pub(crate) in_proj_qkv: TernaryHandle,
    #[cfg(feature = "ternary_gemm_batched")]
    pub(crate) in_proj_z: TernaryHandle,
    #[cfg(feature = "ternary_gemm_batched")]
    pub(crate) in_proj_a: TernaryHandle,
    #[cfg(feature = "ternary_gemm_batched")]
    pub(crate) in_proj_b: TernaryHandle,
    // Issue 980 T4-ALT — the Bonsai-2 dense `ssm_alpha`/`ssm_beta` escape set
    // (BF16 in the file, fp32 here). `Some` ONLY on folded files whose a/b
    // are dense; the whole-prefill cudarc lane mirrors these into
    // `LayerF32::dense_a/b` for `gemm_dense_ab_batched` (the PRIMAL-input
    // GEMM — the escape set is neither rotated nor folded), and the CubeCL
    // decode lane (Plan 602 B2) dispatches them through `GemvCubeCL` on the
    // PRIMAL `norm_x`. When these are `Some`, the `in_proj_a/b` TernaryHandles
    // above carry an EMPTY dummy (rows 0) — nothing on a folded path reads
    // them.
    //
    // Plan 602 B2: UNGATED (was `ternary_gemm_batched`) — the decode lane's
    // feature set (the league manifest's `cubecl_runtime,ternary_gemv,...`)
    // does NOT carry `ternary_gemm_batched`, and the folded decode path
    // consumes these handles there. `None` on pre-rotation files — zero cost.
    pub(crate) in_proj_a_f32: Option<Handle>,
    pub(crate) in_proj_b_f32: Option<Handle>,
    // Issue 642 F3: concatenated qkv+z+a+b for single-GEMV input projection.
    pub(crate) in_proj_concat: TernaryHandle,
    pub(crate) out_proj: TernaryHandle,
    // Issue 727 H2 audit: `gate_proj`/`up_proj` vs `gate_up_proj` dual
    // residency is STRUCTURAL — decode dispatches the concat (single GEMV,
    // Issue 642 F2) while the training/capture paths dispatch the separate
    // handles (2 GEMVs into ffn_gate/ffn_up) and prefill dispatches batched
    // GEMMs into contiguous per-projection [P, mlp] buffers. A concat GEMM
    // interleaves [gate|up] per token row, so removing either set needs new
    // strided/batched-split kernels — deferred (Issue 727 H2).
    // Issue 727 H2 audit + Issue 734 Arm 8: `gate_proj`/`up_proj` are
    // pub(crate) for the whole-prefill cudarc driver (sibling module).
    pub(crate) gate_proj: TernaryHandle,
    pub(crate) up_proj: TernaryHandle,
    pub(crate) down_proj: TernaryHandle,
    // Issue 642 F2: concatenated gate+up weights for single-GEMV FFN input.
    // Pre-uploaded at load time; used by the decode path (forward_from_x).
    pub(crate) gate_up_proj: TernaryHandle,

    // Dense weight handles (f32 on GPU)
    pub(crate) input_norm: Handle,       // [n_embd]
    pub(crate) post_attn_norm: Handle,    // [n_embd]
    pub(crate) conv1d_weight: Handle,     // [conv_dim * kernel_size]
    pub(crate) a_log: Handle,             // [n_v_heads]
    pub(crate) dt_bias: Handle,           // [n_v_heads]
    pub(crate) linear_norm: Handle,       // [head_dim] (shared per-head gamma)

    // Attention layer handles (populated for Attention layers, empty for DeltaNet).
    // Issue 727 H1: the separate wk/wv handles were dead weight — never
    // dispatched anywhere (attn_wkv is built from the CPU weights via
    // `from_two_weights`, the Metal upload reads the CPU weights, and the
    // KV-append debug tap reads the `attn_kv` activation buffer, not these).
    pub(crate) attn_wq: Option<TernaryHandle>,
    // Issue 648 F9: concatenated WK+WV for single-GEMV K+V projection.
    // When present, replaces the two separate K and V GEMVs with one.
    pub(crate) attn_wkv: Option<TernaryHandle>,
    pub(crate) attn_wo: Option<TernaryHandle>,
    pub(crate) attn_q_norm: Option<Handle>,
    pub(crate) attn_k_norm: Option<Handle>,
}

/// Runtime switch for the Issue 619 row-parallel recurrence kernel.
///
/// Exists so an A/B can be measured **inside one process** with the model loaded
/// once. Comparing two separately-launched binaries proved untrustworthy: each
/// process reloads a 6.7 GB GGUF, and the resulting page-cache and thermal drift
/// made whichever variant ran *second* look 20–50% slower regardless of which
/// kernel it was (Issue 619 T5, runs 3–4).
///
/// Defaults to on when the feature is compiled in. Read once per DeltaNet layer
/// with a relaxed load — ~48 atomic loads per token against a ~60 ms token, i.e.
/// unmeasurable.
#[cfg(all(feature = "cubecl_runtime", feature = "deltanet_recurrence_rowpar"))]
static ROWPAR_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Enable/disable the row-parallel recurrence kernel at runtime (Issue 619 T5).
#[cfg(all(feature = "cubecl_runtime", feature = "deltanet_recurrence_rowpar"))]
pub fn set_recurrence_rowpar(on: bool) {
    ROWPAR_ENABLED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Snapshot of one attention layer's intermediate buffers (Bench 642 probe 1).
///
/// Bench 642's token-0 bisect narrowed the batched-prefill G1 failure to the
/// first attention layer (DeltaNet layers 0-2 matched sequential to ~1e-6;
/// layer 3 jumped to 1.708e0). This captures the layer's internals at three
/// points so the divergence can be attributed to a specific kernel rather than
/// to "the attention path".
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
#[derive(Default, Clone)]
pub struct AttnScratch {
    /// `norm_x` as the layer received it — the input to all three projections.
    /// Separates "my prefill fed the layer bad input" from "the projection
    /// kernel is wrong".
    pub norm_x_in: Vec<f32>,
    /// `attn_qg` straight out of the gated-Q projection (pre split/norm/RoPE).
    pub qg_proj: Vec<f32>,
    /// `attn_k` straight out of the K projection.
    pub k_proj: Vec<f32>,
    /// `attn_v` straight out of the V projection.
    pub v_proj: Vec<f32>,
    /// `attn_q` after split + per-head RMSNorm + partial RoPE.
    pub q_rope: Vec<f32>,
    /// `attn_k` after per-head RMSNorm + partial RoPE (as appended to the cache).
    pub k_rope: Vec<f32>,
    /// `attn_out` after the flash-attention decode + output gating.
    pub attn_out: Vec<f32>,
    /// `tmp` after the output projection — the layer's contribution to x.
    pub out_proj: Vec<f32>,
    /// `n_positions` the decode kernel was given (= pos + 1).
    pub n_positions: usize,
}

/// Issue 653: scratch buffers for batched attention prefill.
///
/// Allocated once per prefill call (not per layer). All buffers are sized for
/// P tokens and reused across all attention layers in the prompt.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
struct BatchedAttentionScratch {
    /// QG projection output: `[P, 2 * n_head * head_dim]` interleaved [q, gate]
    qg_b: Handle,
    /// Split Q: `[P, n_head * head_dim]`
    q_b: Handle,
    /// Split gate: `[P, n_head * head_dim]`
    gate_b: Handle,
    /// Combined KV projection: `[P, 2 * n_kv_head * head_dim]`
    kv_b: Handle,
    /// Split K: `[P, n_kv_head * head_dim]` (for RMSNorm + attention)
    k_b: Handle,
    /// Split V: `[P, n_kv_head * head_dim]` (for attention)
    v_b: Handle,
    /// Attention output: `[P, n_head * head_dim]`
    attn_out_b: Handle,
    /// Output projection result: `[P, n_embd]` (before residual add)
    out_proj_b: Handle,
}

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
thread_local! {
    /// Layer index to capture, or -1 for off.
    static ATTN_TAP_LAYER: std::cell::Cell<i64> = const { std::cell::Cell::new(-1) };
    /// Holds the FIRST capture at that layer — i.e. token 0 in both arms.
    static ATTN_TAP: std::cell::RefCell<Option<AttnScratch>> =
        const { std::cell::RefCell::new(None) };
}

/// Arm the attention tap for `layer` (or -1 to disable) and clear any prior
/// capture. Diagnostic only: capturing forces several GPU syncs mid-layer.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_attn_tap_layer(layer: i64) {
    ATTN_TAP_LAYER.with(|c| c.set(layer));
    ATTN_TAP.with(|c| *c.borrow_mut() = None);
}

/// Take the captured snapshot, leaving the tap empty.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn take_attn_tap() -> Option<AttnScratch> {
    ATTN_TAP.with(|c| c.borrow_mut().take())
}

/// Which positions the per-layer prefill capture tap retains.
///
/// The tap already reads back the **whole** `[P, n_embd]` activation buffer per
/// layer (it slices on the CPU rather than binding an offset view, so the
/// readback path cannot confound the comparison it adjudicates). Keeping one row
/// therefore throws away `P - 1` rows that are already on the host — which is
/// why capturing every position used to cost `P` full prefills (`O(P²)` work)
/// instead of one (`O(P)`).
#[cfg(feature = "ternary_gemm_batched")]
#[derive(Clone, Copy, Debug)]
enum CaptureRows {
    /// One position, chunk-relative. The historical behaviour.
    One(usize),
    /// Every position in the chunk, appended in position order.
    All,
}

/// Issue 545 / riir-train Plan 415 — passive attention-mass tap spec.
/// Selects the attention layers and query positions the FMID regen lane
/// reads back: `q_b` rows (post-QK-norm post-RoPE, the batched-attention
/// `[P, n_head, hd]` layout) plus each layer's K-cache valid prefix
/// (`[pos, kvd]` row-major, the same layout the CPU `HybridCache` uses) —
/// and, since Issue 452 T2's D1 port, the V-cache valid prefix (same layout).
///
/// Diagnostic/dataset shape, deliberately NOT a hot-path knob: arming a tap
/// forces the CubeCL prefill body (the whole-prefill cudarc lane refuses
/// while armed) and adds one `q_b` + one K-prefix + one V-prefix read per
/// tapped layer. Defined unconditionally so [`Self::prefill_tokens_chunk`]'s parameter list
/// is feature-stable; only the tap machinery is gated by `attn_mass_tap`.
#[cfg_attr(not(feature = "attn_mass_tap"), allow(dead_code))]
#[derive(Debug, Clone)]
pub struct AttnMassTapSpec {
    /// ABSOLUTE attention-layer indices to tap (strictly ascending,
    /// Attention-type layers only).
    pub layers: Vec<usize>,
    /// ABSOLUTE query positions whose q rows to keep (strictly ascending,
    /// every position < the full prompt length). Under a chunked prefill the
    /// tap routes each row to the chunk that owns it.
    pub row_positions: Vec<usize>,
}

/// Per-run readback for [`AttnMassTapSpec`]. One entry per spec layer, in
/// spec order; buffers are extended across prefill chunks in position order.
#[cfg_attr(not(feature = "attn_mass_tap"), allow(dead_code))]
#[derive(Debug, Default)]
pub struct AttnMassTapCapture {
    /// Per layer: `[n_kept_rows * q_dim]` f32 — the kept q rows concatenated
    /// in `row_positions` order (`q_dim = n_head * head_dim`).
    pub q_rows: Vec<Vec<f32>>,
    /// Per layer: `[n_valid * kvd]` f32 — the K-cache valid prefix,
    /// row-major `[pos, kvd]` (`kvd = n_kv_head * head_dim`).
    pub k_prefix: Vec<Vec<f32>>,
    /// Per layer: `[n_valid * kvd]` f32 — the V-cache valid prefix, row-major
    /// `[pos, kvd]` (same shape as `k_prefix`; Issue 452 T2's D1 lane — the
    /// sparse-vs-dense attention-output cos needs the value aggregation, and
    /// the same passive-readback contract covers it: one read per tapped
    /// layer, after the batched attention stage filled the caches).
    pub v_prefix: Vec<Vec<f32>>,
}

/// Diagnostic switch: make [`TernaryDeltanetGpuForward::prefill`] use P
/// sequential ternary **GEMV** calls on offset slices instead of the batched
/// GEMM, keeping the layer-major ordering, buffer offsets and batched norms
/// identical (Bench 642).
///
/// This is the control that separates *structure* from *numerics*. `prefill`
/// differs from the sequential decode path in two independent ways: it reorders
/// the loops from token-major to layer-major, and it swaps the GEMV for a GEMM
/// with a different accumulation order. Bench 642's G1 passes at P=1 and fails
/// from P=2 upward, which rules out a wiring bug but not either of these. With
/// this flag on, the GEMM difference is removed and only the reordering remains:
/// if G1 then passes, the reordering is sound and the divergence is the GEMM's
/// accumulation order amplified by the DeltaNet recurrence.
///
/// Same one-process A/B rationale as [`set_recurrence_rowpar`] — a 7 GB reload
/// per variant makes cross-process comparison untrustworthy.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_USE_GEMV: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// See [`PREFILL_USE_GEMV`].
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_use_gemv(on: bool) {
    PREFILL_USE_GEMV.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Plane-per-query tiled flash-attention prefill toggle (Issue 771 T2b, the
/// Bench 792 16K verdict's lever: the legacy per-(head, token) kernel measured
/// 63.2% of prefill @16K — 250–347 s of KV re-reads that a query-tiled kernel
/// cuts to the compute-bound floor).
///
/// **DEFAULT ON** (promoted 2026-08-29, Bench 801 — the delegated perf/sec
/// call, the Bench 771 promotion precedent): GOAT G1–G4 PASS (Bench 800,
/// commit `6667de456`) + the lossy-surface behavior gate — **0 greedy
/// next-token argmax flips across 7 lengths (128 → 16384)** through the real
/// batched prefill dispatch, legacy FNV anchors reproduced exactly in-run on
/// both sides. Sub-2048 is a measured wash (0.992–1.014×, noise); the win
/// grows with P (flash share ∝ P²): ~1.10× @2048 → ~1.52× @16K (quiet).
/// The tiled kernel changes the dot reduction order (`plane_sum` tree vs the
/// serial-256 chain) and the online-softmax tile granularity (per-position vs
/// 256-blocks), so results are FP-equivalent, not bit-identical — the pinned
/// prefill FNV anchors cover the LEGACY path (kill-switch below), the tiled
/// per-length anchors are recorded in Bench 800/805.
/// Kill-switch: `RIIR_PREFILL_TILED_FLASH=0` (or `off`/`false`) restores the
/// legacy kernel bit-identically; [`set_prefill_use_tiled_flash`] overrides
/// both. The legacy kernel remains the fallback for any `head_dim != 256`.
#[cfg(feature = "cubecl_runtime")]
static PREFILL_TILED_FLASH: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Launches dispatched through the tiled flash path (the vacuous-guard
/// counter — a green toggle that never reaches the kernel is the
/// "instrument that proves nothing" class).
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
pub(crate) fn note_tiled_flash_launch() {
    TILED_FLASH_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total launches dispatched through the tiled flash path so far.
#[cfg(feature = "cubecl_runtime")]
pub fn tiled_flash_launch_count() -> usize {
    TILED_FLASH_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the tiled flash-attention prefill path is active (DEFAULT ON since
/// Bench 805; env `RIIR_PREFILL_TILED_FLASH` is read exactly once — the first
/// caller wins the OnceLock — and the setter is authoritative after).
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_tiled_flash_enabled() -> bool {
    // Kill-switch semantics (the Bench 771 promotion pattern): unset or
    // "1"/"on"/"true" keeps the tiled kernel; "0"/"off"/"false" restores the
    // legacy kernel bit-identically.
    //
    // The env-derived value must be STORED into the live AtomicBool on the
    // first call — the OnceLock is bookkeeping only (never read), and
    // returning the static initial value directly left the env path
    // latent-broken until the Bench 805 promotion canaries (env=0 still
    // routed tiled; found by the fresh-process kill-switch run). After the
    // first call the setter is authoritative: the store is skipped so a
    // setter flip is never clobbered by a re-read.
    let env_enabled = !matches!(
        std::env::var("RIIR_PREFILL_TILED_FLASH")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "0" | "off" | "false",
    );
    if TILED_FLASH_INITIALIZED.set(env_enabled).is_ok() {
        PREFILL_TILED_FLASH.store(env_enabled, std::sync::atomic::Ordering::Relaxed);
    }
    PREFILL_TILED_FLASH.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the tiled flash path on/off (overrides the env var; the bench
/// harness's arm toggle — safe to flip between forwards, the kernel takes no
/// construction-time state).
#[cfg(feature = "cubecl_runtime")]
pub fn set_prefill_use_tiled_flash(on: bool) {
    let _ = TILED_FLASH_INITIALIZED.set(on);
    PREFILL_TILED_FLASH.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 771 / Bench 808: the M=16 two-rows-per-plane tiled flash arm —
// DEFAULT-ON for LONG prefills (see `TILED_FLASH_M16_MIN_P`); kill-switch env
// `RIIR_PREFILL_TILED_FLASH_M16=0`. BIT-IDENTICAL to the tiled kernel by
// construction (see qwen_attention_prefill_m16_cubecl.rs), so the promotion
// moves NO anchor — the default-state delta is wall-clock only.
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_M16: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Minimum query length for the m16 arm to route (the length gate, Bench 808
/// promotion). The m16 win is the Θ(P²) KV-traffic term: measured e2e ratio
/// **1.207× @16384** (m16 median 64.58 vs tiled 53.51 tok/s, FNV-identical)
/// and wash-to-−5% @2048 (its one clean pair 0.990×) — the interpolated
/// kernel-level progression 0.92 → ~1.0 → ~1.15 → 1.38 across 2048 → 4096 →
/// 8192 → 16384 puts the crossover between 4096 and 8192, so the gate banks
/// the proven long-prefill win and keeps the default tiled below it (the
/// `PREFILL_USE_TALL_GEMM` length-gate precedent: promote on bracketing
/// measurements + mechanism). The armed quiet-box 4096/8192 sweep lowers
/// this if 4096 measures a win.
#[cfg(feature = "cubecl_runtime")]
pub const TILED_FLASH_M16_MIN_P: usize = 8192;

/// Minimum query length for the m32 arm to route (the length gate, Issue 844
/// T2). Set equal to the m16 gate: the win is the same Θ(P²) KV-traffic term
/// (m32 halves unique KV bytes per row-pair vs m16), measured isolated-
/// kernel cooled at +11.0% @8192 / +12.6% @16384 / +27.4% @32768 (production
/// launcher +18.5% @16K), bit-identical outputs (0 bit-diffs vs m16). THE
/// ARM IS OPT-IN — the e2e A/B failed its gate at 0.929× @16K (sustained-
/// load class), so this constant only matters when the env/setter opts in.
#[cfg(feature = "cubecl_runtime")]
pub const TILED_FLASH_M32_MIN_P: usize = 8192;

/// The cmma score-matrix arm's length gate: the arm routes at/above this
/// length when enabled (macOS default-on since the 2026-08-31 re-promotion;
/// env kill-switch semantics (opt-in since the 2026-09-02 demotion, Bench
/// 841). Placed at 16384 by the Bench 809 quiet-box table (1.092x @4096,
/// **0.967x @8192** — m16 keeps that length, 1.363x @16384; @2048 a 1.001x
/// wash); briefly demoted to opt-in 2026-08-30 when the honest
/// prompt-geometry re-measure (post-`14cfd211f`, Issue 782 T4 attempt-2)
/// read **0.999x vs m16 at 16384**; re-promoted 2026-08-31 on clean AC cells
/// at ≥1.05 monotone in P (1.103x @16K / 1.317x @32K — Issue 782 T4/T5).
/// The m16 gate semantics mirror
/// [`Self::TILED_FLASH_M16_MIN_P`].
pub const TILED_FLASH_CMMA_MIN_P: usize = 16384;

#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_M16_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Launches dispatched through the tiled m16 flash path (the vacuous-guard
/// counter — the same instrument class as `TILED_FLASH_LAUNCHES`).
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_M16_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
pub(crate) fn note_tiled_flash_m16_launch() {
    TILED_FLASH_M16_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total launches dispatched through the tiled m16 flash path so far.
#[cfg(feature = "cubecl_runtime")]
pub fn tiled_flash_m16_launch_count() -> usize {
    TILED_FLASH_M16_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the M=16 tiled flash arm is active (DEFAULT ON for long prefills,
/// `p >= TILED_FLASH_M16_MIN_P`; env `RIIR_PREFILL_TILED_FLASH_M16=0` or
/// `off`/`false` restores the plain tiled arm everywhere). Env is read
/// exactly once — the first caller wins the OnceLock, and the env-derived
/// value is STORED into the live AtomicBool on that first call (the Bench 805
/// lesson: the OnceLock is bookkeeping only, never read; returning the static
/// initial value directly left the env path latent-broken there). The setter
/// is authoritative after. When enabled AND length-eligible, the arm takes
/// precedence over the plain tiled arm at the dispatch site (same head_dim-256
/// restriction; the tiled kill-switch does not gate it).
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_tiled_flash_m16_enabled() -> bool {
    // Kill-switch semantics (the Bench 805 promotion pattern): unset or
    // "1"/"on"/"true" keeps the m16 arm; "0"/"off"/"false" restores the
    // plain tiled arm everywhere.
    let env_enabled = !matches!(
        std::env::var("RIIR_PREFILL_TILED_FLASH_M16")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "0" | "off" | "false",
    );
    if TILED_FLASH_M16_INITIALIZED.set(env_enabled).is_ok() {
        TILED_FLASH_M16.store(env_enabled, std::sync::atomic::Ordering::Relaxed);
    }
    TILED_FLASH_M16.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the tiled m16 flash arm on/off (overrides the env var; the bench
/// harness's arm toggle — safe to flip between forwards).
#[cfg(feature = "cubecl_runtime")]
pub fn set_prefill_use_tiled_flash_m16(on: bool) {
    let _ = TILED_FLASH_M16_INITIALIZED.set(on);
    TILED_FLASH_M16.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 844 T2: the M=32 four-rows-per-plane tiled flash arm — the
// KV-traffic amortization lever the Bench 843 phase map ordered first (KV
// loads are 41→63% of the kernel, monotone in P). BIT-IDENTICAL to m16
// (probe twin AND production kernel measured 0 bit-diffs; e2e FNV equality
// incl. the pinned `c152813ee93aaa2a` @16K anchor). Isolated-kernel record:
// +11.0/+12.6/+27.4% cooled at 8K/16K/32K (production launcher +18.5% @16K).
// Lifecycle (the cmma lifecycle, one cycle faster): the back-to-back e2e A/B
// FAILED its G2 direction gate at 0.929× @16K (m32 passes degraded
// monotonically 85.5→70.3→64.3 tok/s under sustained load — the Bench-790
// class) and the arm landed OPT-IN; **the Bench-845 cooled long-pass re-open
// INVERTED it** — 3 rounds × (420 s cooldown + one full 16 K pass),
// alternating arms, medians m32 83.74 vs m16 80.76 = **1.037×**, 3/3 rounds
// ≥1.0×, FNV identical on every pass, 0 argmax flips (the loaded round —
// Xcode+Unity ambient spike — was m32's BIGGEST win at 1.279×, the opposite
// of the decay class). The 0.929× was a thermal-order artifact: back-to-back
// passes let m32 inherit sustained heat; cooled long-passes start every pass
// from the same thermal baseline. PROMOTED default-on macOS (the measured
// e2e +3.7% sits inside the kernel-arithmetic window: +12.6–18.5% kernel ×
// 31.2% attention share × ~90% stage ≈ +3.5–5.2%). Back-to-back sustained
// regimes (a burst of prefills with no cooldown) remain the recorded caveat
// — the smem-GEMM precedent (Bench 790 promoted default-ON with its own
// sustained-load decay note). Env `RIIR_PREFILL_TILED_FLASH_M32=0` kills the
// arm (restores m16) on any platform; `=1` opts in on non-macOS. Reopen
// trigger (a future demotion re-open): a scored-cell league re-measure.
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_M32: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(cfg!(target_os = "macos"));

#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_M32_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Launches dispatched through the m32 tiled flash path (the vacuous-guard
/// counter — the same instrument class as `TILED_FLASH_M16_LAUNCHES`).
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_M32_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
pub(crate) fn note_tiled_flash_m32_launch() {
    TILED_FLASH_M32_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total launches dispatched through the m32 tiled flash path so far.
#[cfg(feature = "cubecl_runtime")]
pub fn tiled_flash_m32_launch_count() -> usize {
    TILED_FLASH_M32_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the M=32 tiled flash arm is active (**OPT-IN since the 2026-09-02
/// same-day demotion — the e2e A/B failed its G2 gate at 0.929× @16K despite
/// the isolated +12.6% cooled kernel record; the sustained-load class, see
/// the static's doc**. Env `RIIR_PREFILL_TILED_FLASH_M32=1` or `on`/`true`
/// enables; unset or `0`/`off`/`false` keeps the m16 arm). Env is read
/// exactly once — the first caller wins the OnceLock, and the env-derived
/// value is STORED into the live AtomicBool on that first call (the Bench 805
/// lesson). The setter is authoritative after. Bit-identical to m16, so
/// routing through this arm moves no anchor.
/// Whether the M=32 tiled flash arm is active — **DEFAULT-ON macOS for long
/// prefills since the 2026-09-02 Bench-845 promotion** (`p >=
/// TILED_FLASH_M32_MIN_P`; the cooled long-pass re-measure inverted the
/// 0.929× back-to-back FAIL — m32 median 83.74 vs m16 80.76 = 1.037×, 3/3
/// rounds ≥1.0×, FNV/argmax identical throughout). Env
/// `RIIR_PREFILL_TILED_FLASH_M32`: `0`/`off`/`false` kills the arm on any
/// platform (restores m16); unset keeps the platform default (macOS ON,
/// elsewhere OFF); `1`/`on`/`true` opts in on non-macOS. Env is read
/// exactly once — the first caller wins the OnceLock and the env-derived
/// value is STORED into the live AtomicBool on that first call (the Bench
/// 805 lesson; the env path was latent-broken until 805's canaries — store
/// on first call). The setter is authoritative after. Routing: m32 takes
/// precedence over m16 when enabled AND length-eligible (same head_dim-256
/// restriction).
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_tiled_flash_m32_enabled() -> bool {
    // Platform-default + kill-switch semantics (the cmma promoted-lifecycle
    // pattern): unset keeps the platform default (macOS ON — the promotion
    // evidence is Metal-only; elsewhere OFF), "0"/"off"/"false" forces OFF,
    // "1"/"on"/"true" forces ON.
    let env = std::env::var("RIIR_PREFILL_TILED_FLASH_M32")
        .unwrap_or_default()
        .to_lowercase();
    let env_enabled = if env.is_empty() {
        cfg!(target_os = "macos")
    } else {
        matches!(env.as_str(), "1" | "on" | "true")
    };
    if TILED_FLASH_M32_INITIALIZED.set(env_enabled).is_ok() {
        TILED_FLASH_M32.store(env_enabled, std::sync::atomic::Ordering::Relaxed);
    }
    TILED_FLASH_M32.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the tiled m32 flash arm on/off (overrides the env var; the bench
/// harness's arm toggle — safe to flip between forwards).
#[cfg(feature = "cubecl_runtime")]
pub fn set_prefill_use_tiled_flash_m32(on: bool) {
    let _ = TILED_FLASH_M32_INITIALIZED.set(on);
    TILED_FLASH_M32.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The M=64 arm's length gate: the arm routes at/above this length when
/// opted in (KV amortization grows with P — the same threshold class as
/// [`TILED_FLASH_M32_MIN_P`]; below it the wider cube buys nothing and the
/// m32/m16 arms keep the work).
#[cfg(feature = "cubecl_runtime")]
pub const TILED_FLASH_M64_MIN_P: usize = 8192;

// Issue 844 T5: the M=64 eight-rows-per-plane tiled flash arm — the second
// step of the KV-traffic amortization lever (T2 built m32; its note named
// "m32/m64-class"). **OPT-IN ONLY — default OFF on every platform**: the m32
// sustained-load lesson (the 0.929× back-to-back e2e FAIL, the Bench-790
// thermal class) applies a fortiori at doubled per-cube arithmetic density,
// and the Bench-845 cooled long-pass that rescued m32 has NOT been run for
// m64. Env `RIIR_PREFILL_TILED_FLASH_M64`: `1`/`on`/`true` enables; unset or
// any other value keeps OFF (deliberately NOT the platform-default pattern —
// a probe-grade candidate never defaults). Env is read exactly once — the
// first caller wins the OnceLock and stores into the live AtomicBool (the
// Bench 805 lesson). The setter is authoritative after. Routing: m64 takes
// precedence over m32/m16 when enabled AND length-eligible (same
// head_dim-256 restriction). Bit-identity with m32/m16 is the expected
// class (per-row arithmetic verbatim + masked no-op iterations) — the probe
// twin measures it before any e2e claim.
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_M64: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_M64_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Launches dispatched through the m64 tiled flash path (the vacuous-guard
/// counter — the same instrument class as `TILED_FLASH_M32_LAUNCHES`).
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_M64_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
pub(crate) fn note_tiled_flash_m64_launch() {
    TILED_FLASH_M64_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total launches dispatched through the m64 tiled flash path so far.
#[cfg(feature = "cubecl_runtime")]
pub fn tiled_flash_m64_launch_count() -> usize {
    TILED_FLASH_M64_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the M=64 tiled flash arm is active (**OPT-IN ONLY — default OFF
/// everywhere**; see the static's doc for the lifecycle rationale).
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_tiled_flash_m64_enabled() -> bool {
    let env = std::env::var("RIIR_PREFILL_TILED_FLASH_M64")
        .unwrap_or_default()
        .to_lowercase();
    let env_enabled = matches!(env.as_str(), "1" | "on" | "true");
    if TILED_FLASH_M64_INITIALIZED.set(env_enabled).is_ok() {
        TILED_FLASH_M64.store(env_enabled, std::sync::atomic::Ordering::Relaxed);
    }
    TILED_FLASH_M64.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the tiled m64 flash arm on/off (overrides the env var; the probe
/// harness's arm toggle — safe to flip between forwards).
#[cfg(feature = "cubecl_runtime")]
pub fn set_prefill_use_tiled_flash_m64(on: bool) {
    let _ = TILED_FLASH_M64_INITIALIZED.set(on);
    TILED_FLASH_M64.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Minimum query length for the m32-pipe arm to route (the length gate —
/// same class as the m32/m64 gates: the latency the pipeline hides is the
/// long-P KV stream's, not the short-P one).
#[cfg(feature = "cubecl_runtime")]
pub const TILED_FLASH_M32_PIPE_MIN_P: usize = 8192;

// Issue 844 T5 residue: the K/V register-prefetch PIPELINED m32 arm — the
// latency-hiding branch of the KV-share lever, opened after Bench 899 closed
// Q-block-widening (m64 NO-GO) and the T1 map showed the kernel streaming at
// ~half DRAM peak (latency-exposed). **OPT-IN ONLY — default OFF on every
// platform** (the m32 sustained-load lesson applies to any candidate arm
// before its own cooled long-pass e2e; a probe-grade candidate never
// defaults). Env `RIIR_PREFILL_TILED_FLASH_M32_PIPE`: `1`/`on`/`true`
// enables; unset or any other value keeps OFF. Env is read exactly once —
// the first caller wins the OnceLock and stores into the live AtomicBool
// (the Bench 805 lesson). The setter is authoritative after. Routing: the
// pipe arm takes precedence over plain m32 only (same Q-block shape — m64
// stays the widest-shape arm above it). Bit-identity with m32/m16 is the
// expected class (compute body verbatim; the loads moved, not the math) —
// the probe twin measures it before any e2e claim.
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_M32_PIPE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_M32_PIPE_INITIALIZED: std::sync::OnceLock<bool> =
    std::sync::OnceLock::new();

/// Launches dispatched through the m32-pipe tiled flash path (the
/// vacuous-guard counter — the same instrument class as
/// `TILED_FLASH_M64_LAUNCHES`).
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_M32_PIPE_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
pub(crate) fn note_tiled_flash_m32_pipe_launch() {
    TILED_FLASH_M32_PIPE_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total launches dispatched through the m32-pipe tiled flash path so far.
#[cfg(feature = "cubecl_runtime")]
pub fn tiled_flash_m32_pipe_launch_count() -> usize {
    TILED_FLASH_M32_PIPE_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the M=32 K/V-pipelined flash arm is active (**OPT-IN ONLY —
/// default OFF everywhere**; see the static's doc for the lifecycle).
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_tiled_flash_m32_pipe_enabled() -> bool {
    let env = std::env::var("RIIR_PREFILL_TILED_FLASH_M32_PIPE")
        .unwrap_or_default()
        .to_lowercase();
    let env_enabled = matches!(env.as_str(), "1" | "on" | "true");
    if TILED_FLASH_M32_PIPE_INITIALIZED.set(env_enabled).is_ok() {
        TILED_FLASH_M32_PIPE.store(env_enabled, std::sync::atomic::Ordering::Relaxed);
    }
    TILED_FLASH_M32_PIPE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the m32-pipe flash arm on/off (overrides the env var; the probe
/// harness's arm toggle — safe to flip between forwards).
#[cfg(feature = "cubecl_runtime")]
pub fn set_prefill_use_tiled_flash_m32_pipe(on: bool) {
    let _ = TILED_FLASH_M32_PIPE_INITIALIZED.set(on);
    TILED_FLASH_M32_PIPE.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 771 / Bench 809: the cmma score-matrix tiled flash arm — the last
// Bench 800 §3 lever (smem K/V tile staging + tensor-core score dots +
// tile-parallel exps). TOLERANCE ARM (tile-order softmax + fragment-order
// dots — NOT bit-identical to tiled/m16). Lifecycle: promoted default-on at
// p>=16384 on 2026-08-29 (the 1.363x chunk-geometry quiet cell), DEMOTED back
// to opt-in on 2026-08-30 (the honest prompt-geometry path post-14cfd211f
// measured 0.999x, failing the promotion's own ≥1.05 direction gate — and the
// promoted default had NO target gate, exposing the Vulkan corruption), then
// RE-PROMOTED default-on at >=16384 **on macOS only** on 2026-08-31: the AC
// quiet cells landed ≥1.05 twice (1.103x @16K scored + 1.317x @32K,
// monotone in P — Issue 782 T4/T5), meeting the pre-committed reopen trigger.
// **DEMOTED to opt-in on 2026-09-02 (Issue 843 T4 / Bench 841): the cooled
// tri-cell (single process, Bench-831-(d) protocol, counter-proven routing)
// measured the m16 arm −8.2% on the attention stage @16K — twice across two
// processes (52.62s / 52.91s vs cmma's 59.41s / 57.63s) — and e2e confirms
// (m16 all-on 178.87s = 91.60 tok/s vs cmma's 183.33s / 192.91s the same
// day), refuting the 782 direction on this box. Legacy-family remains the
// default carrier (m16 ≥8192); cmma stays available via env/setter for its
// macOS-only record cells. Env `RIIR_PREFILL_TILED_FLASH_CMMA=1` opts in on
// any platform. The arm takes precedence over m16 → tiled → legacy at the
// dispatch site WHEN ENABLED (same head_dim-256 restriction).
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_CMMA: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_CMMA_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Launches dispatched through the cmma tiled flash path (the vacuous-guard
/// counter — the same instrument class as `TILED_FLASH_M16_LAUNCHES`).
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_CMMA_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
pub(crate) fn note_tiled_flash_cmma_launch() {
    TILED_FLASH_CMMA_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total launches dispatched through the cmma tiled flash path so far.
#[cfg(feature = "cubecl_runtime")]
pub fn tiled_flash_cmma_launch_count() -> usize {
    TILED_FLASH_CMMA_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the cmma score-matrix tiled flash arm is active — **OPT-IN
/// everywhere since the 2026-09-02 demotion** (Issue 843 T4 / Bench 841: the
/// cooled single-process tri-cell measured the m16 arm −8.2% on the attention
/// stage @16K, twice across two processes, with e2e confirming — refuting the
/// 782 direction on this box; the m16/legacy family carries the ≥8192
/// default). Env `RIIR_PREFILL_TILED_FLASH_CMMA`: `1`/`on`/`true` opts in on
/// any platform; unset or `0`/`off`/`false` keeps it OFF. (Lifecycle above:
/// 782 had promoted it default-on at ≥16384 on macOS — that promotion
/// measured e2e cells, and the tri-cell's stage-isolated cooled protocol
/// inverted it.) Env is read exactly once — the first caller wins the OnceLock
/// and the env-derived value is STORED into the live AtomicBool on that first
/// call (the Bench 805 lesson; the env path was latent-broken until 805's
/// canaries — store on first call). The setter is authoritative after.
/// Routing additionally requires the device to register the
/// `(f32, f32, f32, 8, 8, 8)` cmma MmaConfig (the Issue 828 shape guard —
/// without it the arms fall through the dispatch chain, so the env opt-in is
/// safe on any backend).
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_tiled_flash_cmma_enabled() -> bool {
    let env = std::env::var("RIIR_PREFILL_TILED_FLASH_CMMA")
        .unwrap_or_default()
        .to_lowercase();
    let env_enabled = matches!(env.as_str(), "1" | "on" | "true");
    if TILED_FLASH_CMMA_INITIALIZED.set(env_enabled).is_ok() {
        TILED_FLASH_CMMA.store(env_enabled, std::sync::atomic::Ordering::Relaxed);
    }
    TILED_FLASH_CMMA.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the cmma tiled flash arm on/off (overrides the env var; the bench
/// harness's arm toggle — safe to flip between forwards).
#[cfg(feature = "cubecl_runtime")]
pub fn set_prefill_use_tiled_flash_cmma(on: bool) {
    let _ = TILED_FLASH_CMMA_INITIALIZED.set(on);
    TILED_FLASH_CMMA.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 771 / Bench 810: the PV-cmma successor arm — O in 32 per-plane cmma
// accumulator fragments, PV on the tensor core, rare-event smem round-trip
// rescale (the Bench 809 record's "no fragment row-scale API" is answered by
// from_slice(Accumulator), verified in cubecl-core 0.11.0-pre.2). TOLERANCE
// ARM → DEFAULT-OFF (its own G2 verdict is unmeasured at production
// geometry). Opt in via env
// `RIIR_PREFILL_TILED_FLASH_CMMA_PV=1` or the setter; the arm takes
// precedence over cmma → m16 → tiled → legacy at the dispatch site. (The
// parent cmma arm was re-promoted macOS-default 2026-08-31; PV itself stays
// opt-in.)
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_CMMA_PV: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_CMMA_PV_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Launches dispatched through the PV-cmma tiled flash path (the vacuous-guard
/// counter — the same instrument class as `TILED_FLASH_CMMA_LAUNCHES`).
#[cfg(feature = "cubecl_runtime")]
static TILED_FLASH_CMMA_PV_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
pub(crate) fn note_tiled_flash_cmma_pv_launch() {
    TILED_FLASH_CMMA_PV_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total launches dispatched through the PV-cmma tiled flash path so far.
#[cfg(feature = "cubecl_runtime")]
pub fn tiled_flash_cmma_pv_launch_count() -> usize {
    TILED_FLASH_CMMA_PV_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the PV-cmma tiled flash arm is active (DEFAULT OFF — tolerance
/// arm; env `RIIR_PREFILL_TILED_FLASH_CMMA_PV=1`/`on`/`true` enables). Env is
/// read exactly once — the first caller wins the OnceLock and the env-derived
/// value is STORED into the live AtomicBool on that first call. Routing
/// additionally requires the device to register the `(f32, f32, f32, 8, 8,
/// 8)` cmma MmaConfig (the Issue 828 shape guard — shared with the parent
/// score arm; the arms fall through the dispatch chain without it). The
/// setter is authoritative after.
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_tiled_flash_cmma_pv_enabled() -> bool {
    let env_enabled = matches!(
        std::env::var("RIIR_PREFILL_TILED_FLASH_CMMA_PV")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "1" | "on" | "true",
    );
    if TILED_FLASH_CMMA_PV_INITIALIZED.set(env_enabled).is_ok() {
        TILED_FLASH_CMMA_PV.store(env_enabled, std::sync::atomic::Ordering::Relaxed);
    }
    TILED_FLASH_CMMA_PV.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the PV-cmma tiled flash arm on/off (overrides the env var; the bench
/// harness's arm toggle — safe to flip between forwards).
#[cfg(feature = "cubecl_runtime")]
pub fn set_prefill_use_tiled_flash_cmma_pv(on: bool) {
    let _ = TILED_FLASH_CMMA_PV_INITIALIZED.set(on);
    TILED_FLASH_CMMA_PV.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 771 T2c-a / Plan 562: the Q8-KV prefill flash arm — DEFAULT-OFF
// tolerance arm transplanted from the m4-prefill-engine distill (Research
// 360). Env `RIIR_Q8KV_PREFILL=1`/`on`/`true` enables; the setter is
// authoritative after. TOLERANCE-class: quantized K/V rides on top of the
// tiled kernel's FP-equivalent class, so the arm never claims an f32 anchor —
// the argmax-flip sweep + tolerance band are the behavior contract (the
// Issue 771 flash rule).
#[cfg(feature = "cubecl_runtime")]
static Q8KV_PREFILL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "cubecl_runtime")]
static Q8KV_PREFILL_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Launches dispatched through the Q8-KV prefill flash path (the
/// vacuous-guard counter — the same instrument class as
/// `TILED_FLASH_CMMA_PV_LAUNCHES`).
#[cfg(feature = "cubecl_runtime")]
static Q8KV_PREFILL_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
pub(crate) fn note_q8kv_prefill_launch() {
    Q8KV_PREFILL_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total attention launches dispatched through the Q8-KV prefill flash path
/// so far (the quantizer dispatches are not counted — the counter answers
/// "did the arm route?", not "how much work ran").
#[cfg(feature = "cubecl_runtime")]
pub fn q8kv_prefill_launch_count() -> usize {
    Q8KV_PREFILL_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Minimum PROMPT length for the Q8-KV arm to route (the Issue 782
/// prompt-geometry lesson: `active_prefill_len`, never chunk-local p).
/// Placed at the cmma slot (16384) — the ≥16K many-wave regime where the
/// 809 arm proved K/V-traffic reduction pays; the Bench 831(c) decode
/// plateau (occupancy-bound, bytes-free) does NOT transfer to prefill.
#[cfg(feature = "cubecl_runtime")]
pub const PREFILL_Q8KV_MIN_P: usize = 16384;

/// Whether the Q8-KV prefill flash arm is active (DEFAULT OFF). Env is read
/// exactly once — the first caller wins the OnceLock and the env-derived
/// value is STORED into the live AtomicBool on that first call (the Bench
/// 805 env-propagation lesson); the setter is authoritative after.
#[cfg(feature = "cubecl_runtime")]
pub fn prefill_q8kv_enabled() -> bool {
    let env_enabled = matches!(
        std::env::var("RIIR_Q8KV_PREFILL")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "1" | "on" | "true",
    );
    if Q8KV_PREFILL_INITIALIZED.set(env_enabled).is_ok() {
        Q8KV_PREFILL.store(env_enabled, std::sync::atomic::Ordering::Relaxed);
    }
    Q8KV_PREFILL.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the Q8-KV prefill flash arm on/off (overrides the env var; the
/// bench harness's arm toggle — safe to flip between forwards).
#[cfg(feature = "cubecl_runtime")]
pub fn set_prefill_q8kv(on: bool) {
    let _ = Q8KV_PREFILL_INITIALIZED.set(on);
    Q8KV_PREFILL.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Runtime switch (Issue 641 / Bench 645): when the `ternary_gemm_simdgroup`
/// feature is compiled, route prefill projections through the hardware
/// cooperative-matrix (cmma) kernel instead of the plane-cooperative one.
///
/// The simdgroup kernel's G2 roll-up is **1.89×** (8×32 variant, P=128) vs the
/// plane-cooperative kernel's 1.08× — a genuine 1.75× improvement on the
/// prefill path. Per-shape wins range from 1.20× (tiny-M `ssm_alpha/beta`) to
/// 8.91× (`attn_k/v`). The binding constraint preventing ≥3× is the ternary
/// dequant on scalar ALU (Bench 645 §"Why it doesn't reach 3×").
///
/// Defaults to `true` when the feature is compiled — opting into the feature
/// means wanting the hardware path. The GOAT gate for promoting
/// `ternary_gemm_simdgroup` to the default feature set still requires ≥3×
/// roll-up (currently FAILs at 1.89×); this flag is the runtime A/B control
/// for when both kernels are available.
///
/// Falls back to the plane-cooperative kernel automatically when:
/// - the device lacks cmma `(f32,f32,f32)` at 8×8×8
/// - `PREFILL_USE_GEMV` is also set (GEMV baseline takes priority)
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_gemm_simdgroup"
))]
static PREFILL_USE_SIMDGROUP: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// See [`PREFILL_USE_SIMDGROUP`].
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_gemm_simdgroup"
))]
pub fn set_prefill_use_simdgroup(on: bool) {
    PREFILL_USE_SIMDGROUP.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 768: the 32×32 input-reuse GEMM toggle. The Bench 774 phase
// decomposition measured the 8×32 kernel's #1 cost as INPUT activation
// global loads (42–63% marginal; every input element re-loaded M/8 = 2176×
// at the FFN shape = 91.2 GB/GEMM @2048). The 32×32 output tile reuses each
// input tile across 4 M-subtiles (traffic ÷4) and each weight dequant across
// 4 P-subtiles — kernel-level 1.44–1.70× at the production shapes, outputs
// BIT-IDENTICAL to the 8×32 kernel (same per-element k-tile accumulation
// order). Default ON; the shape heuristic keeps m < 64 || p < 32 on the
// older arms (the 32×32 tile pads small-m shapes — measured 0.773× at m=48).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_gemm_simdgroup"
))]
static PREFILL_USE_TALL_GEMM: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// See [`PREFILL_USE_TALL_GEMM`]. A/B hook for Bench 774's successor gates.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_gemm_simdgroup"
))]
pub fn set_prefill_use_tall_gemm(on: bool) {
    PREFILL_USE_TALL_GEMM.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 734 T3: workgroup-tiled (64×64) ternary GEMM toggle. On the 4090
// (Vulkan) every other fast path is unavailable — the Metal kernels are
// macOS-gated, `cmma_available` checks a Metal-only f32 signature, and
// cubecl-wgpu panics on CoopMma everywhere — so the plane-coop kernel ran
// at ~2.6 TFLOPS. The tiled kernel measured 14.6 TFLOPS at the same shapes
// (5.7× roll-up, Bench 734) with G1 rel-err 4.4e-7. Default ON; disable to
// A/B against the Issue 637 plane-coop kernel.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_USE_TILED_GEMM: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// See [`PREFILL_USE_TILED_GEMM`].
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_use_tiled_gemm(on: bool) {
    PREFILL_USE_TILED_GEMM.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 734 T6: NVIDIA cooperative-matrix (VK_KHR_cooperative_matrix) f16
// 16×16×16 tensor-core GEMM — the sg8 variant (256-thread workgroups, 128×64
// tiles) measured 1.64× the tiled kernel at the Bonsai shapes (32.8 TFLOPS,
// G1 max_rel 2.2e-4 — the f16 rounding class) and e2e 367.90 → 569.21 tok/s
// @2048 (argmax identical, logits max_rel 2.7e-3 — Bench 706). The
// `wgpu<spirv>` runtime's cubecl-spirv compiler implements CoopMma (the
// WGSL-backend panic the T2 probe found is dead code on this backend); the
// vendored runtime enables the extension + queries shapes. PROMOTED
// DEFAULT-ON 2026-08-20 (GOAT G1/G2/G3 PASS); the capability check gates it
// to NVIDIA — Metal reports no 16×16×16 f16 shape, so M3 falls through to
// the simdgroup path unchanged.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_USE_CMMA16: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// See [`PREFILL_USE_CMMA16`].
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_use_cmma16(on: bool) {
    PREFILL_USE_CMMA16.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 734 T7: NVIDIA int8 cooperative-matrix (i8×i8→i32 @ 16×16×32) tensor
// GEMM — exact ternary signs + hi/lo per-token int8 activations (per-group
// reduction for the weight scales). The T7 probe measured the load+mma
// ceiling at 764.8 TOPS (vs the f16 path's 327 TFLOPS-class) with EXACT
// integer products — the post-Bench-708 frontier (llama.cpp same-box prefill
// runs 5.59× ours at ~161 TFLOPS-effective). MEASURED VERDICT (Bench 709):
// the 4× raw mma rate does NOT move the big GEMM shapes (32-35 TFLOPS ≈
// cmma16 sg8 — the wall is the shared staging structure, not the tensor
// rate); the kernel WINS on small-m shapes (1.54× ssm_alpha/beta) and e2e at
// ≤4K (+0.8% @2048, +2.5% @4096, argmax identical, G1 max_rel 8e-5 — 3-27×
// tighter than f16). >4K is UNMEASURED (the 16K harness numbers were
// fast-but-wrong — the graceful-OOM unbound-handle path, Bench 709), so
// PROMOTED DEFAULT-ON with a conservative `p <= 4096` length gate (only the
// measured-win region crosses); longer contexts fall through to cmma16.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_USE_CMMA_I8: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// See [`PREFILL_USE_CMMA_I8`].
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_use_cmma_i8(on: bool) {
    PREFILL_USE_CMMA_I8.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 734 Lever 2: B-side DIRECT global→matrix loads for the i8 GEMM
// (cmma_tensor_addressing / OpCooperativeMatrixLoadTensorNV). When ON, the
// i8 prefill GEMM dispatches `launch_direct` (B via runtime-offset
// TensorView slices — no smem staging round-trip) instead of the staging
// `launch_sg8`; outputs are expected bit-identical (same B values, same mma
// order). Default follows env `RIIR_CMMA_I8_DIRECT` ("1"/"0"; default OFF
// until the GOAT gate measures it — the A/B escape hatch).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_CMMA_I8_DIRECT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
fn prefill_cmma_i8_direct() -> bool {
    static ENV_DEFAULT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env = *ENV_DEFAULT.get_or_init(|| {
        std::env::var("RIIR_CMMA_I8_DIRECT")
            .ok().is_some_and(|s| matches!(s.trim(), "1" | "true" | "on"))
    });
    PREFILL_CMMA_I8_DIRECT.load(std::sync::atomic::Ordering::Relaxed) || env
}

/// Force-enable/disable the B-direct i8 GEMM variant (overrides
/// `RIIR_CMMA_I8_DIRECT`). Issue 734 Lever 2 A/B knob.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_cmma_i8_direct(on: bool) {
    PREFILL_CMMA_I8_DIRECT.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 734 Lever 2 (tile arm): the 128×64-tile i8 GEMM variant — TOK_TILE
// 32→64, halving A-side global re-reads + staging per output (the A half of
// the Bench-709 staging-wall finding) at the same ~40 KB smem / 2 wgs/SM
// occupancy class. Expected bit-identical to the staging sg8 kernel (same
// q values via the shared pre-pass, same per-sub-tile mma accumulation,
// same reduction arithmetic + group order). Default follows env
// `RIIR_CMMA_I8_T64` ("1"/"0"; default OFF until the GOAT gate measures it —
// the A/B escape hatch). Takes precedence over the direct knob when both
// are set (they are alternative B/A-side rewrites of the same kernel).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_CMMA_I8_T64: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
fn prefill_cmma_i8_t64() -> bool {
    static ENV_DEFAULT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env = *ENV_DEFAULT.get_or_init(|| {
        std::env::var("RIIR_CMMA_I8_T64")
            .ok().is_some_and(|s| matches!(s.trim(), "1" | "true" | "on"))
    });
    PREFILL_CMMA_I8_T64.load(std::sync::atomic::Ordering::Relaxed) || env
}

/// Force-enable/disable the 128×64-tile i8 GEMM variant (overrides
/// `RIIR_CMMA_I8_T64`). Issue 734 Lever 2 tile-arm A/B knob.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_cmma_i8_t64(on: bool) {
    PREFILL_CMMA_I8_T64.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 734 Arm 5 (Bench 718): the two-round-partials i8 GEMM variant —
// staging + pipelined k-loop identical to sg8, per-group partials split
// into two store→reduce rounds over 16 buffers (44→28 KB workgroup smem →
// 3 wgs/SM on the 4090 — the occupancy attack on the latency-bound staging
// phase; four shrink axes closed neutral at 2 wgs/SM). **PROMOTED to the
// default for the p≤4096 NVIDIA i8 path (Bench 718 GOAT)**: G1 bit-identical
// (full-model Bench-710 FNV pins reproduced via RIIR_CMMA_I8_PSPLIT=1;
// synthetic 0/35.6M bit-diffs ×6), G2 kernel 1.026-1.107× (median ≈1.047×,
// 6 interleaved min-of-15 runs) + e2e 1.040-1.041× (2× 5-round interleaved,
// min AND median), G3 the sg8 default path bits unchanged (e2e sg8 arm
// reproduced the pin), G4 identical alloc pattern (same launcher scratch).
// Env `RIIR_CMMA_I8_PSPLIT=0|false|off` opts back to the sg8 kernel (the
// escape hatch); an explicit env value always wins over the setter+default.
// Checked after t64/direct (alternative rewrites of the same kernel).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_CMMA_I8_PSPLIT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
fn prefill_cmma_i8_psplit() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| {
        std::env::var("RIIR_CMMA_I8_PSPLIT")
            .ok()
            .map(|s| !matches!(s.trim(), "0" | "false" | "off"))
    });
    match env {
        Some(v) => v,
        None => PREFILL_CMMA_I8_PSPLIT.load(std::sync::atomic::Ordering::Relaxed),
    }
}

/// Force-enable/disable the two-round-partials i8 GEMM variant (overrides
/// the DEFAULT-ON state; an explicit `RIIR_CMMA_I8_PSPLIT` env value wins
/// over both). Issue 734 Arm 5 A/B knob.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_cmma_i8_psplit(on: bool) {
    PREFILL_CMMA_I8_PSPLIT.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 734 (Bench 709 follow-up): CHUNKED PREFILL — the 16K/32K working-set
// fix. On the 24 GB 4090, a single P=16384 prefill call exhausts VRAM through
// the sliced pool's per-size-class pages and the vendored graceful-OOM leaves
// handles UNBOUND — the prefill completes FAST-BUT-WRONG (measured 682 tok/s
// @16K, faster than the real 568 @2048; Bench 709). The fix: process the
// prompt in chunks of at most `prefill_chunk_max()` tokens (default 4096 —
// the measured-clean region), carrying GDN recurrent/conv state (persistent,
// updated in place) and attention KV (filled at absolute positions; later
// chunks attend over the CACHE) between chunks. Per-token arithmetic is
// unchanged — chunked output is bit-identical to an unchunked run that fit.
//
// Every chunk's p ≤ 4096 also rides the i8 cmma GEMM gate (Bench 709).
//
// `set_prefill_chunk_max` overrides the env (0 clears the override); env
// `RIIR_PREFILL_CHUNK_MAX` (parsed once, default 4096; "0" disables chunking
// entirely — escape hatch for A/B).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_CHUNK_MAX_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
fn prefill_chunk_max() -> usize {
    static ENV_DEFAULT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let env = *ENV_DEFAULT.get_or_init(|| {
        std::env::var("RIIR_PREFILL_CHUNK_MAX")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok()).map_or(4096, |v| if v == 0 { usize::MAX } else { v })
    });
    let override_ = PREFILL_CHUNK_MAX_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if override_ != 0 {
        override_
    } else {
        env
    }
}

/// See [`PREFILL_CHUNK_MAX_OVERRIDE`]. `n >= 1` caps the chunk width; `0`
/// clears the override (falls back to `RIIR_PREFILL_CHUNK_MAX` / 4096).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_chunk_max(n: usize) {
    PREFILL_CHUNK_MAX_OVERRIDE.store(n, std::sync::atomic::Ordering::Relaxed);
}

// Issue 655: F16 cmma toggle for the simdgroup GEMM. When true + device supports
// (f16, f16, f32) cmma, the prefill path dequants weights to f16 and uses
// mixed-precision cmma — Metal's simdgroup_matrix_8x8<half> has 2× instruction
// throughput. Default-off pending GOAT gate (G1 accuracy + G2 perf).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_gemm_simdgroup",
    feature = "ternary_gemm_simdgroup_f16"
))]
static PREFILL_USE_SIMDGROUP_F16: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_gemm_simdgroup",
    feature = "ternary_gemm_simdgroup_f16"
))]
#[allow(dead_code)]
pub fn set_prefill_use_simdgroup_f16(on: bool) {
    PREFILL_USE_SIMDGROUP_F16.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 767: scale-deferred simdgroup GEMM toggle. When true + device supports
// (f16, f16, f32) cmma, the prefill GEMM dequantizes weights to f16 SIGNS ONLY
// ({-1,0,1}, exact) and applies group scales to the f32 accumulator at group
// boundaries (every 128 K) — the per-element scale-mul (the measured dequant
// binder, Bench 645 follow-up #4) leaves the inner loop. Default-off pending
// the T2 G1 gate (P=128 argmax + max_rel vs the f32 path on real Bonsai).
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_gemm_simdgroup",
    feature = "ternary_gemm_simdgroup_f16"
))]
static PREFILL_USE_SIMDGROUP_DEFERRED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_gemm_simdgroup",
    feature = "ternary_gemm_simdgroup_f16"
))]
#[allow(dead_code)]
pub fn set_prefill_use_simdgroup_deferred(on: bool) {
    PREFILL_USE_SIMDGROUP_DEFERRED.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Plan 534 T4: metal::tensor matmul2d alternative prefill GEMM path. When on
// (+ the `metal_tensor_gemm` feature is compiled + macOS), `prefill_project`
// routes through the Metal cooperative-tensor matmul2d kernel instead of the
// CubeCL cmma path. Takes priority over simdgroup + GEMV when enabled.
//
// The kernel-level G2 (Issue 656) measured 3.72× roll-up vs CubeCL cmma on
// the 9 production GEMM shapes. This flag tests whether that gain translates
// end-to-end through the host round-trip (read CubeCL handle → Metal → write
// back).
//
// Default-off: the CubeCL cmma path stays the production default until the
// end-to-end GOAT (G1 logits + G2 tok/s) passes.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "metal_tensor_gemm",
    target_os = "macos"
))]
static PREFILL_USE_METAL_TENSOR: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// See [`PREFILL_USE_METAL_TENSOR`].
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "metal_tensor_gemm",
    target_os = "macos"
))]
pub fn set_prefill_use_metal_tensor(on: bool) {
    PREFILL_USE_METAL_TENSOR.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 657: zero-copy wgpu MSL passthrough variant of the metal::tensor path.
// When on (+ `metal_tensor_gemm` compiled + macOS + CubeCLContext exposed
// the shared wgpu device/queue), `prefill_project` dispatches the matmul2d
// kernel through wgpu's MSL passthrough API, sharing CubeCL's GPU buffers
// directly — no host round-trip, no cross-queue sync. Takes priority over
// the host-round-trip `PREFILL_USE_METAL_TENSOR` when both are on.
//
// Default-off: the CubeCL simdgroup cmma path stays the production default
// until the zero-copy GOAT (G1 logits + G2 tok/s ≥ 1.5×) passes.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "metal_tensor_gemm",
    target_os = "macos"
))]
static PREFILL_USE_METAL_TENSOR_WGPU: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// See [`PREFILL_USE_METAL_TENSOR_WGPU`].
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "metal_tensor_gemm",
    target_os = "macos"
))]
pub fn set_prefill_use_metal_tensor_wgpu(on: bool) {
    PREFILL_USE_METAL_TENSOR_WGPU.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 663 T5: zero-copy RAW-Metal variant. When on (+ `metal_tensor_gemm`
// compiled + macOS + the wgpu-hal fork that exposes `Buffer::raw_handle()`),
// `prefill_project` dispatches the matmul2d kernel via raw `MTLComputeCommandEncoder`
// on the shared CubeCL Metal device + queue — zero copy, zero host round-trip,
// zero staging buffer, zero bind group. Takes HIGHEST priority (over the wgpu
// MSL passthrough + the host round-trip paths) when enabled.
//
// This is the third attempt at capturing the 3.72× kernel-level matmul2d gain
// (Bench 645). Issues 656 (0.87×, separate device) + 657 (0.95×, staging
// copies) were both refuted by integration overhead. Issue 663 T4 validated the
// raw-Metal interop path works; T5 measures the end-to-end gain.
//
// Default-off: the CubeCL simdgroup cmma path stays the production default
// until the T5 GOAT (G1 logits + G2 tok/s ≥ 1.5×) passes.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "metal_tensor_gemm",
    target_os = "macos"
))]
static PREFILL_USE_METAL_TENSOR_ZEROCOPY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// See [`PREFILL_USE_METAL_TENSOR_ZEROCOPY`].
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "metal_tensor_gemm",
    target_os = "macos"
))]
pub fn set_prefill_use_metal_tensor_zerocopy(on: bool) {
    PREFILL_USE_METAL_TENSOR_ZEROCOPY.store(on, std::sync::atomic::Ordering::Relaxed);
}

// Issue 727 H3: one-time loud warning when a Metal-path flag is ON but its
// weight cache is unreachable — previously this fell through to CubeCL
// SILENTLY, so a mis-sequenced A/B (flag flipped after construction without
// the cache) would measure CubeCL twice with no trace. One eprintln per
// process keeps the dispatch hot path log-clean.
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
fn warn_metal_cache_missing_once(flag: &str, reason: &str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[Issue 727] {flag} is on but its dispatch is unavailable ({reason}); \
             falling through to CubeCL. This measurement is NOT exercising the \
             {flag} path."
        );
    }
}

/// Diagnostic switch: make [`TernaryDeltanetGpuForward::prefill`] apply the
/// layer input-norm and post-attention norm with P sequential
/// [`RmsNormCubeCL`] calls (the kernel the decode path uses) instead of one
/// [`RmsNormBatchedCubeCL`] call.
///
/// Tests whether the two RMSNorm kernels disagree. Prior evidence says they do
/// not — Bench 642 passes at P=1, which already exercises `RmsNormBatched` at
/// `dim = n_embd`, and the kernel's rows are independent so `seq_len > 1`
/// cannot change a row's result. Kept because it is the cheap way to *prove*
/// that rather than argue it.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_SEQ_RMSNORM: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// See [`PREFILL_SEQ_RMSNORM`].
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_seq_rmsnorm(on: bool) {
    PREFILL_SEQ_RMSNORM.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Diagnostic (Issue 640 T3): zero-fill every P-width scratch buffer at the top
/// of `prefill` before any kernel writes it.
///
/// `client.empty()` returns **recycled** pool memory with arbitrary stale
/// contents, so any region a prefill call reads without having written it first
/// observes whatever the previous call left behind — which varies with
/// allocation order and would explain Issue 640's run-to-run variance at P=128
/// (including the runs that land at exactly `0.000e0`, where the pool happened to
/// hold the right values).
///
/// This is a **probe, not a fix**. If zero-filling stabilizes G1 then something
/// reads unwritten memory, and the repair is to find and write that region — a
/// blanket zero-fill would only mask it while costing 15 extra dispatches per
/// prompt.
/// Bit `i` selects the `i`-th P-width scratch buffer, in [`PREFILL_SCRATCH_NAMES`]
/// order. `0` disables the fill; [`PREFILL_SCRATCH_ALL`] fills every buffer. A
/// mask rather than a flag so Issue 640 can be bisected to a single buffer
/// instead of stopping at "the blanket fill helps".
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_ZERO_SCRATCH: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Names of the 15 P-width scratch buffers, indexed by mask bit.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub const PREFILL_SCRATCH_NAMES: [&str; 15] = [
    "x_b", "normx_b", "qkv_b", "qkvx_b", "z_b", "a_b", "b_b", "beta_b", "decay_b", "rec_b",
    "tmp_b", "gate_b", "up_b", "hid_b", "ffnout_b",
];

/// Mask selecting every P-width scratch buffer.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub const PREFILL_SCRATCH_ALL: u32 = (1u32 << 15) - 1;

/// See [`PREFILL_ZERO_SCRATCH`]. Convenience: all buffers or none.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_zero_scratch(on: bool) {
    let mask = if on { PREFILL_SCRATCH_ALL } else { 0 };
    PREFILL_ZERO_SCRATCH.store(mask, std::sync::atomic::Ordering::Relaxed);
}

/// See [`PREFILL_ZERO_SCRATCH`]. Bit `i` => zero-fill `PREFILL_SCRATCH_NAMES[i]`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_zero_scratch_mask(mask: u32) {
    PREFILL_ZERO_SCRATCH.store(mask & PREFILL_SCRATCH_ALL, std::sync::atomic::Ordering::Relaxed);
}

/// Issue 640 aliasing test: route the residual adds through a second buffer and
/// ping-pong, instead of binding one buffer as both read operand and read-write
/// output.
///
/// `prefill` currently does `ResidualAdd(x_b, tmp_b -> x_b)` at 655,360 elements,
/// twice per layer. Per-element the arithmetic is safe — each thread touches only
/// its own index — but binding the same buffer to a read and a read-write slot is
/// aliasing UB in WGSL/Metal, and `launch_unchecked` does not validate it. Bench
/// 646 narrowed the race to something *inside* a dispatch (a readback after every
/// layer does not suppress it), which is exactly the shape aliasing UB would have.
///
/// The alternate buffers are allocated **unconditionally**, in both arms, so the
/// A/B differs only in whether the bindings alias — not in the allocation
/// pattern. That matters: Bench 646 showed allocation-pattern perturbations
/// (zero-filling) move the failure rate on their own, so leaving that variable
/// free would re-confound the experiment.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_PINGPONG_RESIDUAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// See [`PREFILL_PINGPONG_RESIDUAL`].
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_pingpong_residual(on: bool) {
    PREFILL_PINGPONG_RESIDUAL.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Issue 640 alloc-only control: allocate the full-size alternate buffers
/// without using them, so the aliasing A/B has an arm that differs from the
/// baseline in allocation pattern *only*.
///
/// Bench 646 measured allocation-pattern perturbations (zero-filling the
/// scratch) moving the failure rate by themselves, and Bench 647's first run was
/// voided partly because the alternates were allocated unconditionally — making
/// even the "baseline" arm differ from the shipping path. This knob restores a
/// true baseline and adds the control that separates the two variables.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_ALLOC_ALT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// See [`PREFILL_ALLOC_ALT`].
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_alloc_alt(on: bool) {
    PREFILL_ALLOC_ALT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Issue 640 aliasing test, second site: the per-head `RmsNormBatched` runs
/// in place (`rec_b -> rec_b`) over `p * n_v_heads` = 6144 rows. Same aliasing
/// question as [`PREFILL_PINGPONG_RESIDUAL`], different kernel and 100x narrower,
/// so it is switched separately to keep the attribution clean.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_DEALIAS_NORM: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// See [`PREFILL_DEALIAS_NORM`].
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_dealias_norm(on: bool) {
    PREFILL_DEALIAS_NORM.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Issue 640 strip-down bisect: which stage classes must run for the divergence
/// to appear?
///
/// Six rounds of building a synthetic reproducer *up* from hypothesised
/// properties all failed (repro/wgpu_metal_contention: shape, memory pressure,
/// bit-plane matmuls, recurrent state, residency shape, attention/KV, extreme
/// numerics — all 0 divergences under contention). Build-up carries no guarantee
/// of converging. Stripping the **real** path down does: it starts from a
/// configuration known to diverge 50% of the time under contention, so removing
/// stages until divergence stops must bracket the trigger.
///
/// Bit `i` enables the `i`-th stage class in [`PREFILL_STAGE_NAMES`]. Default is
/// all-on, i.e. the shipping path.
///
/// **Correctness is irrelevant here.** The bisect measures run-to-run
/// *reproducibility*, not agreement with sequential decode, so a stripped
/// pipeline producing meaningless logits is still a valid probe.
///
/// **Run with `set_prefill_zero_scratch(true)`.** Disabling a stage leaves its
/// output buffer unwritten, and `client.empty()` hands back recycled pool memory
/// whose contents vary — which would inject a second, unrelated source of
/// run-to-run variation and confound the very thing being measured.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_STAGE_MASK: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(PREFILL_STAGE_ALL);

/// Stage classes, indexed by bit position in [`PREFILL_STAGE_MASK`].
///
/// These three are the top-level split and are individually safe to disable:
/// each contributes to the residual stream `x_b` only through its own residual
/// add, so skipping one leaves `x_b` valid rather than uninitialised.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub const PREFILL_STAGE_NAMES: [&str; 3] = ["deltanet", "attention", "ffn"];

/// All stage classes enabled — the shipping path.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub const PREFILL_STAGE_ALL: u32 = (1u32 << 3) - 1;

/// See [`PREFILL_STAGE_MASK`]. Bit `i` enables `PREFILL_STAGE_NAMES[i]`.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_stage_mask(mask: u32) {
    PREFILL_STAGE_MASK.store(mask & PREFILL_STAGE_ALL, std::sync::atomic::Ordering::Relaxed);
}

/// Whether stage class `bit` is enabled.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
#[inline]
fn stage_on(bit: u32) -> bool {
    PREFILL_STAGE_MASK.load(std::sync::atomic::Ordering::Relaxed) & (1u32 << bit) != 0
}

// ── Decode stage mask (Issue 661 T1 — decode stage breakdown profiling) ──
//
// Mirrors the prefill stage mask but for `forward_from_x` (the decode path).
// Lets a profiling bench selectively disable stage classes to measure their
// wall-time contribution to the 105 ms/token decode budget.
//
// Bit layout:
//   0 = input norm        (RmsNorm / ResidualAddRmsNorm before each layer)
//   1 = deltanet layer    (forward_deltanet_layer_gpu — 8 dispatches)
//   2 = attention layer   (forward_attention_layer_gpu — 8 dispatches)
//   3 = mid norm          (ResidualAddRmsNorm after layer + before FFN)
//   4 = ffn               (gate_up GEMV + SwiGLU + down GEMV)
//
// Correctness is irrelevant when profiling (like the prefill mask): a stripped
// pipeline produces garbage logits but the wall time is still valid.
static DECODE_STAGE_MASK: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(DECODE_STAGE_ALL);

/// Decode stage class names, indexed by bit position in [`DECODE_STAGE_MASK`].
pub const DECODE_STAGE_NAMES: [&str; 5] =
    ["input_norm", "deltanet", "attention", "mid_norm", "ffn"];

/// All decode stage classes enabled — the shipping path.
pub const DECODE_STAGE_ALL: u32 = (1u32 << 5) - 1;

/// See [`DECODE_STAGE_MASK`]. Bit `i` enables [`DECODE_STAGE_NAMES`]`[i]`.
pub fn set_decode_stage_mask(mask: u32) {
    DECODE_STAGE_MASK.store(mask & DECODE_STAGE_ALL, std::sync::atomic::Ordering::Relaxed);
}

/// Whether decode stage class `bit` is enabled.
#[inline]
fn decode_stage_on(bit: u32) -> bool {
    DECODE_STAGE_MASK.load(std::sync::atomic::Ordering::Relaxed) & (1u32 << bit) != 0
}

// ── Issue 831 (a): split-K decode attention toggle ───────────────────────
//
// Default ON with a length gate (n_positions > 512 routes to the split
// pair); `RIIR_ATTN_SPLIT_DECODE=0` is the kill-switch. Env is resolved
// ONCE at first read (store-on-first-call — the Bench 805 env lesson: the
// env must reach the live atomic, not just OnceLock bookkeeping), and the
// setter is authoritative afterwards for A/B harnesses.

pub(crate) const ATTN_SPLIT_DECODE_MAX_SPLITS_DEFAULT: usize = 64;
const ATTN_SPLIT_DECODE_UNSET: usize = usize::MAX;

// Issue 864 T2: fire the oversized-KV-working-set warn ONCE per process —
// a benchmark loop constructing several forwards must not spam it.
static KV_WORKING_SET_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

use std::sync::atomic::Ordering;

static ATTN_SPLIT_DECODE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(ATTN_SPLIT_DECODE_UNSET);
static ATTN_SPLIT_DECODE_MIN_POS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(512);

/// Whether the length-gated split-K decode-attention path is enabled.
pub fn attn_split_decode_enabled() -> bool {
    let v = ATTN_SPLIT_DECODE.load(std::sync::atomic::Ordering::Relaxed);
    if v == ATTN_SPLIT_DECODE_UNSET {
        let env_on = std::env::var("RIIR_ATTN_SPLIT_DECODE").map_or(true, |v| v != "0");
        ATTN_SPLIT_DECODE
            .compare_exchange(
                ATTN_SPLIT_DECODE_UNSET,
                usize::from(env_on),
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .ok();
        env_on
    } else {
        v == 1
    }
}

/// Force the split-K decode path on/off (authoritative over the env).
pub fn set_attn_split_decode(on: bool) {
    ATTN_SPLIT_DECODE.store(usize::from(on), std::sync::atomic::Ordering::Relaxed);
}

/// The `n_positions` threshold above which decode attention splits.
pub fn attn_split_decode_min_pos() -> usize {
    ATTN_SPLIT_DECODE_MIN_POS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set the split threshold (A/B harness seam).
pub fn set_attn_split_decode_min_pos(n: usize) {
    ATTN_SPLIT_DECODE_MIN_POS.store(n, std::sync::atomic::Ordering::Relaxed);
}

// ── Issue 640 attention sub-stage bisect ────────────────────────────────
//
// Bench 652 localized the race to the attention path (REQUIRED + SUFFICIENT
// alone). These masks gate individual dispatch groups inside the per-token
// attention loop to narrow from "8 dispatches/token" to the specific
// kernel or buffer-reuse pattern that triggers the race.
//
// Sub-stages (indexed by bit in ATTN_SUBSTAGE_MASK):
//   0: Q + KV projections (ternary GEMVs — steps 1-2)
//   1: Split + QK-norm + RoPE (elementwise — steps 3-5)
//   2: KV cache append (position-indexed write — step 6)
//   3: Flash attention decode (growing read set — step 7)
//   4: Output projection GEMV (step 8)
//   5: Outer per-token RMSNorm + ResidualAdd (the loop body outside
//      forward_attention_layer_gpu)

/// Issue 640 attention sub-stage mask. Default: all on (shipping path).
///
/// Since Issue 771 T2 (Bench 792) the mask ALSO gates the Issue 653 batched
/// prefill attention path (`prefill_attention_layer_batched`) — the per-token
/// path this mask was built for was replaced wholesale by the batched path in
/// Issue 653, which shipped without the gating, silently killing the
/// instrument on the production path. The bit mapping on the batched path:
///   0: Q + KV batched GEMMs (steps 2-3) · 1: splits + QK-norm + RoPE
///   (steps 4-7) · 2: KV cache fill (step 8) · 3: causal flash attention
///   (step 9) · 4: output projection GEMM (step 10) · 5: input RMSNorm +
///   residual add (steps 1 + 11).
///
/// Available under `cubecl_runtime` alone (not requiring `ternary_gemm_batched`)
/// because `forward_attention_layer_gpu` — the decode-path attention forward —
/// calls `attn_sub_on` unconditionally. Without `ternary_gemm_batched` the
/// bisect setter [`set_attn_substage_mask`] is absent, so the mask stays at
/// its all-on default and every sub-stage runs (identical to no gate).
#[cfg(feature = "cubecl_runtime")]
static ATTN_SUBSTAGE_MASK: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(ATTN_SUBSTAGE_ALL);

/// Attention sub-stage names, indexed by bit in [`ATTN_SUBSTAGE_MASK`].
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub const ATTN_SUBSTAGE_NAMES: [&str; 6] = [
    "proj",    // Q + KV GEMVs
    "elementwise", // split + QK-norm + RoPE
    "kv_append",   // KV cache append
    "attn_decode", // flash attention decode
    "out_proj",    // output GEMV
    "outer_norm",  // per-token RMSNorm + ResidualAdd
];

/// All attention sub-stages enabled — the shipping path.
#[cfg(feature = "cubecl_runtime")]
pub const ATTN_SUBSTAGE_ALL: u32 = (1u32 << 6) - 1;

/// Set the attention sub-stage mask for Issue 640 bisect.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_attn_substage_mask(mask: u32) {
    ATTN_SUBSTAGE_MASK.store(mask & ATTN_SUBSTAGE_ALL, std::sync::atomic::Ordering::Relaxed);
}

/// Whether attention sub-stage `bit` is enabled.
#[cfg(feature = "cubecl_runtime")]
#[inline]
fn attn_sub_on(bit: u32) -> bool {
    ATTN_SUBSTAGE_MASK.load(std::sync::atomic::Ordering::Relaxed) & (1u32 << bit) != 0
}

/// Issue 640 partial-sync bisect: block on a full readback after every layer
/// `l <= k`, leaving layers `> k` to run asynchronously as usual.
///
/// Bench 645 established that the race is monotone in synchronization — 9/9
/// failures with nothing added, 3/9 with the scratch zero-filled (15 extra
/// dispatches), 0/5 with the capture tap's 64 blocking readbacks. The tap is
/// therefore useless as an observer: it suppresses what it is meant to watch.
/// This switch turns that suppression into an instrument by applying it to a
/// *prefix* of the layers.
///
/// If syncing through layer `k` suppresses the race but through `k-1` does not,
/// the race involves layer `k`'s dispatches — and the model's `layer_types` then
/// says whether that is a DeltaNet or an attention layer, which discriminates
/// the two remaining hypotheses (offset-view hazard tracking vs the attention
/// path's shared single-token scratch).
///
/// [`SYNC_THROUGH_DISABLED`] means no sync at all (the default, and the
/// configuration the 9/9 baseline was measured in).
///
/// Uses the same `read_one` full readback the capture tap uses, deliberately: the
/// bisect must reproduce the exact suppression that was measured, not a
/// different barrier that might have different semantics.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_SYNC_THROUGH_LAYER: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(SYNC_THROUGH_DISABLED);

/// Sentinel for [`PREFILL_SYNC_THROUGH_LAYER`]: synchronize after no layer.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub const SYNC_THROUGH_DISABLED: u32 = u32::MAX;

/// See [`PREFILL_SYNC_THROUGH_LAYER`]. Pass [`SYNC_THROUGH_DISABLED`] to disable.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_sync_through_layer(k: u32) {
    PREFILL_SYNC_THROUGH_LAYER.store(k, std::sync::atomic::Ordering::Relaxed);
}

/// Issue 640 Bench 657: synchronize (full `read_one` readback) after every
/// `k` tokens within the per-token attention loop. This tests whether a
/// per-token barrier suppresses the race — the structural-alignment test
/// that `tasks_max` (which flushes at arbitrary dispatch counts) cannot
/// answer cleanly. `SYNC_THROUGH_DISABLED` = no per-token sync (default).
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_SYNC_PER_TOKEN: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(SYNC_THROUGH_DISABLED);

/// See [`PREFILL_SYNC_PER_TOKEN`]. Pass [`SYNC_THROUGH_DISABLED`] to disable.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_sync_per_token(k: u32) {
    PREFILL_SYNC_PER_TOKEN.store(k, std::sync::atomic::Ordering::Relaxed);
}

/// Issue 640 Bench 658: lightweight flush (no readback) every `k` tokens.
/// Unlike [`PREFILL_SYNC_PER_TOKEN`] (which does a heavy `read_one`), this
/// calls `client.flush()` — submits the command buffer + recreates the compute
/// encoder WITHOUT a bus transfer. Tests whether the timing perturbation
/// from `read_one` (Bench 657) was the reason barriers made the race worse.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_FLUSH_PER_TOKEN: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(SYNC_THROUGH_DISABLED);

/// See [`PREFILL_FLUSH_PER_TOKEN`]. Pass [`SYNC_THROUGH_DISABLED`] to disable.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_flush_per_token(k: u32) {
    PREFILL_FLUSH_PER_TOKEN.store(k, std::sync::atomic::Ordering::Relaxed);
}

/// Issue 637 T5: batch the two per-token elementwise steps of the DeltaNet
/// layer — beta/decay and head-expansion+L2-norm — into one dispatch each
/// instead of `p` dispatches each.
///
/// Both are pure elementwise / row-independent functions of their own token's
/// data, so the batched kernels are bit-identical to the sequential ones (proved
/// at every P by `probe_637_t5_batched_elementwise`). conv1d and the recurrence
/// stay per-token: they carry `conv_state` / `state` across tokens and are
/// sequential by construction.
///
/// At P=128 over 48 DeltaNet layers this removes `2 x 128 x 48 = 12,288`
/// dispatches, leaving 96. Each dispatch costs ~25 CPU allocations plus launch
/// latency (Issue 638), and Bench 641 established that launch-overhead
/// amortization — not weight/ALU reuse — is where this path's wins come from.
///
/// Default **on**. The switch exists so the gate can measure both arms in one
/// process, which Issue 642 makes mandatory: wgpu on Metal degrades badly across
/// processes (14.06 -> 0.16 tok/s after the first), so cross-process A/B is not
/// trustworthy here.
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
static PREFILL_BATCH_ELEMENTWISE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// See [`PREFILL_BATCH_ELEMENTWISE`].
#[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemm_batched"))]
pub fn set_prefill_batch_elementwise(on: bool) {
    PREFILL_BATCH_ELEMENTWISE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the multi-token DeltaNet recurrence prefill path (Issue 734 T4) is
/// selected.
///
/// When ON, the per-token sequential recurrence loop (P rowpar dispatches per
/// layer) is replaced by ONE `DeltanetRecurrenceMultiTokenCubeCL` dispatch per
/// layer — the same per-token arithmetic in the same order (bit-identical
/// output; G1 e2e max_rel 0.000e0 at P=128/64/63/1, 2026-08-20), with the
/// ~34 µs GPU-side inter-kernel gap × P × n_layer dispatch tax eliminated
/// (Issue 734 T1a: the recurrence was ~3.3 s of an 8.7 s P=2048 block).
///
/// Default **on** since 2026-08-20 (Issue 734 T4). The toggle exists so the
/// GOAT gate can measure both arms in one process (mandatory per Issue 642).
/// Requires the `ternary_deltanet_chunked_prefill` feature. The historical
/// chunkwise-parallel solve (Bench 662's algorithm, Plan 533 + the Issue 734
/// `deltanet_delta_rule_chunked` kernels) stays OFF — its different summation
/// order diverges ~4e-3/layer on real data (the Issue 721 tree-verify class),
/// compounding to O(1) through 48 layers.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill"
))]
static PREFILL_CHUNKED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// See [`PREFILL_CHUNKED`].
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill"
))]
pub fn set_prefill_chunked(on: bool) {
    PREFILL_CHUNKED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The chunk size C for chunked prefill. 64 is the standard DeltaNet choice
/// (Yang & Wang 2024). Each chunk processes C tokens in parallel within the
/// chunk, with sequential state transition between chunks.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill"
))]
const PREFILL_CHUNK_SIZE: usize = 64;

/// Issue 658 Phase 2: when true, the conv1d step inside each DeltaNet layer's
/// prefill uses the chunked conv1d kernel (`DeltanetChunkedConv1dCubeCL`) instead
/// of the sequential per-token `DeltanetConv1dCubeCL` loop. This collapses P
/// conv1d dispatches per layer to ceil(P/C) dispatches — a dispatch-count
/// reduction.
///
/// **GOAT-validated (2026-08-14):** G1 PASS (argmax match + max_abs 0.22 < 0.5).
/// G2 PASS — rigorous interleaved measurement (5 warm pairs, 5 measure pairs,
/// seq→chunk interleaved to cancel thermal bias): **median per-pair speedup
/// 1.07×**, consistent across all 5 pairs (range 1.04×–1.08×). The initial
/// single-run measurement of 1.33× was inflated by thermal/frequency bias
/// (sequential-first runs cold; chunked-second benefits from the GPU frequency
/// boost). Promoted to DEFAULT-ON 2026-08-14.
///
/// **Decoupled from `PREFILL_CHUNKED`** (which gates the recurrence chunking).
/// Conv1d chunking is correct independently — the kernel G1-passes standalone.
/// Recurrence chunking (Plan 533 Phase 4) G1-fails on the wrong formula, so it
/// stays OFF.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill"
))]
static PREFILL_CHUNKED_CONV1D: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// See [`PREFILL_CHUNKED_CONV1D`].
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill"
))]
pub fn set_prefill_chunked_conv1d(on: bool) {
    PREFILL_CHUNKED_CONV1D.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Issue 653: when true, the attention layers in prefill use a batched
/// causal-masked flash attention kernel instead of P sequential decode
/// dispatches. Default false — opt-in until G2 ≥ 3× is measured.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
static PREFILL_ATTENTION_BATCHED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// See [`PREFILL_ATTENTION_BATCHED`].
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_attention_batched_prefill"
))]
pub fn set_prefill_attention_batched(on: bool) {
    PREFILL_ATTENTION_BATCHED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the Issue 734 Arm 8 whole-prefill cudarc arm may run: every
/// diagnostic / A-B knob at its shipping default (the bit-identity target
/// the Bench-710 pins were measured against). Any deviation keeps prefill
/// on the CubeCL path so the knob's own semantics stay intact.
#[cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemm_batched",
    feature = "ternary_gemv_cuda_raw",
    not(target_os = "macos")
))]
// Plan 610 S4b: the stacked `#[cfg(not(..))] { return fail(..) }` feature
// guards below produce unreachable-code whenever a later guard's feature is
// compiled in while an earlier one is off — postures the home crate never
// ran (its defaults carry the whole family). Allowed rather than restructured.
#[allow(unreachable_code)]
pub(crate) fn prefill_cuda_gate_ok() -> bool {
    use std::sync::atomic::Ordering;
    let trace = std::env::var("RIIR_PREFILL_CUDA_TRACE").is_ok_and(|s| matches!(s.trim(), "1" | "2" | "true" | "on"));
    let fail = |what: &str| {
        if trace {
            eprintln!("[734-arm8-gate] blocked by: {what}");
        }
        false
    };
    if PREFILL_PINGPONG_RESIDUAL.load(Ordering::Relaxed) {
        return fail("pingpong_residual");
    }
    if PREFILL_DEALIAS_NORM.load(Ordering::Relaxed) {
        return fail("dealias_norm");
    }
    if PREFILL_ALLOC_ALT.load(Ordering::Relaxed) {
        return fail("alloc_alt");
    }
    if PREFILL_ZERO_SCRATCH.load(Ordering::Relaxed) != 0 {
        return fail("zero_scratch");
    }
    if PREFILL_STAGE_MASK.load(Ordering::Relaxed) != PREFILL_STAGE_ALL {
        return fail("stage_mask");
    }
    // Issue 771 T2: the sub-stage mask now also gates the batched prefill
    // attention path — a partially-disabled sub-stage feeds garbage downstream,
    // so the numerics gate must refuse it exactly like stage_mask.
    #[cfg(all(
        feature = "cubecl_runtime",
        feature = "ternary_gemm_batched",
        feature = "ternary_attention_batched_prefill"
    ))]
    if ATTN_SUBSTAGE_MASK.load(Ordering::Relaxed) != ATTN_SUBSTAGE_ALL {
        return fail("attn_substage");
    }
    if PREFILL_SYNC_THROUGH_LAYER.load(Ordering::Relaxed) != SYNC_THROUGH_DISABLED {
        return fail("sync_through_layer");
    }
    if PREFILL_SYNC_PER_TOKEN.load(Ordering::Relaxed) != SYNC_THROUGH_DISABLED {
        return fail("sync_per_token");
    }
    if PREFILL_FLUSH_PER_TOKEN.load(Ordering::Relaxed) != SYNC_THROUGH_DISABLED {
        return fail("flush_per_token");
    }
    if !PREFILL_BATCH_ELEMENTWISE.load(Ordering::Relaxed) {
        return fail("batch_elementwise");
    }
    if PREFILL_USE_GEMV.load(Ordering::Relaxed) {
        return fail("use_gemv");
    }
    if PREFILL_SEQ_RMSNORM.load(Ordering::Relaxed) {
        return fail("seq_rmsnorm");
    }
    if prefill_cmma_i8_direct() {
        return fail("cmma_i8_direct");
    }
    if prefill_cmma_i8_t64() {
        return fail("cmma_i8_t64");
    }
    if !prefill_cmma_i8_psplit() {
        return fail("cmma_i8_psplit");
    }
    // The chunked-recurrence + batched-attention knobs only exist when
    // their features are compiled in; without the features the shipping
    // path runs the sequential variants and the arm is not eligible.
    #[cfg(all(
        feature = "cubecl_runtime",
        feature = "ternary_gemm_batched",
        feature = "ternary_deltanet_chunked_prefill"
    ))]
    if !PREFILL_CHUNKED.load(Ordering::Relaxed) {
        return fail("chunked");
    }
    #[cfg(not(all(
        feature = "cubecl_runtime",
        feature = "ternary_gemm_batched",
        feature = "ternary_deltanet_chunked_prefill"
    )))]
    {
        return fail("feature: ternary_deltanet_chunked_prefill");
    }
    #[cfg(all(
        feature = "cubecl_runtime",
        feature = "ternary_gemm_batched",
        feature = "ternary_attention_batched_prefill"
    ))]
    if !PREFILL_ATTENTION_BATCHED.load(Ordering::Relaxed) {
        return fail("attention_batched");
    }
    #[cfg(not(all(
        feature = "cubecl_runtime",
        feature = "ternary_gemm_batched",
        feature = "ternary_attention_batched_prefill"
    )))]
    {
        return fail("feature: ternary_attention_batched_prefill");
    }
    true
}

/// Whether the row-parallel recurrence kernel is currently selected.
#[cfg(all(feature = "cubecl_runtime", feature = "deltanet_recurrence_rowpar"))]
#[must_use]
pub fn recurrence_rowpar_enabled() -> bool {
    ROWPAR_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// GPU-resident forward pass for ternary Qwen3.5 DeltaNet.
///
/// All activations stay on GPU between layers. The forward processes
/// a single decode token and returns the logits.
///
/// # Architecture
///
/// - Pre-uploads all ternary + dense weights at construction time
/// - Allocates persistent activation buffers (reused across tokens)
/// - Chains CubeCL kernel dispatches without CPU sync points
/// - Only syncs once at the end (downloading logits)
#[cfg(feature = "cubecl_runtime")]
pub struct TernaryDeltanetGpuForward {
    // pub(crate) fields: consumed by the Issue 721 tree-verify driver
    // (`ternary_tree_verify_driver`) — sibling-module access.
    pub(crate) client: ComputeClient<ActiveRuntime>,
    pub(crate) config: Config,
    pub(crate) layer_types: Vec<DeltaNetLayerType>,

    // Pre-uploaded weights
    pub(crate) layers: Vec<GpuLayerWeights>,
    pub(crate) final_norm: Handle,
    pub(crate) lm_head: TernaryHandle,
    /// Embedding table as ternary bit-planes (uploaded once; GPU-side dequant
    /// per token eliminates the CPU dequant + per-token GPU alloc in
    /// `set_input_token` — Issue 604 T8 G4).
    pub(crate) wte_handle: TernaryHandle,

    // Persistent activation buffers
    // pub(crate) (x, logits) — consumed by the Issue 734 Arm 8 whole-prefill
    // cudarc driver's tail (sibling module).
    pub(crate) x: Handle,           // [n_embd] — hidden state
    /// Issue 980 T4-ALT — the Bonsai-2 Hadamard-folded marker + Plan 602 B3:
    /// `Some` whenever the loaded file declares `prism.hadamard` (the DEFAULT
    /// constructor accepts folded models since B2/B3 — the decode eager path
    /// carries the rotation). The whole-prefill cudarc lane builds its own
    /// `RotationTables` from this config lazily (the stack's OnceLock); every
    /// non-rotation-aware compute path (the CubeCL prefill body, training,
    /// the diagnostic capture) refuses loudly while it is `Some`.
    pub(crate) rotation: Option<TernaryRotationConfig>,
    /// Plan 602 B3 — the GPU-resident rotation tables (sign vectors uploaded
    /// as f32, width-keyed). `Some` iff `rotation` is; built ONCE here (the
    /// eager build also refuses a block size the FWHT kernel cannot serve).
    /// Every rotated launch clones a sign handle from here — alloc-free
    /// steady state.
    pub(crate) rot_tables:
        Option<crate::deltanet_rotation_cubecl::RotationTablesCubeCL>,
    /// Plan 602 B3 — the ROTATED copy of `norm_x`: staged by copy-rotate
    /// after each RMSNorm whose consumers are folded projections (the layer
    /// input norm, the post-attn norm, the final norm). `norm_x` itself
    /// stays PRIMAL — the dense a/b escape set consumes it, and
    /// `forward_token_with_final_hidden` keeps its primal-hidden contract.
    rot_scratch: Handle, // [n_embd]
    /// Plan 602 B3 — the `gdn_v_grouped` permute staging copy of
    /// `recurrent_out` (the permute kernel reads a staged source — no
    /// in-place race; the CPU twin stages into `rotation_buf` the same way).
    permute_tmp: Handle, // [v_dim]
    /// Issue 860 T3 — `x` holds an input this forward has not consumed yet.
    ///
    /// Set by the two deliberate writers of an *input* into `x`
    /// ([`set_input_token`](Self::set_input_token) and the final-chunk tail of
    /// `prefill_tokens_chunk`), cleared by the four `forward_from_x*` funnels
    /// that consume it. A forward with no fresh input re-runs the PREVIOUS
    /// token's embedding and returns a full, right-shaped, silently wrong
    /// logits vector — `debug_assert`ed in [`Self::consume_fresh_input`]
    /// rather than left to produce a plausible number.
    x_input_fresh: bool,
    pub(crate) norm_x: Handle,   // [n_embd] — RMSNorm output (tree-verify per-branch bridge reads it)
    pub(crate) qkv: Handle,            // [qkv_dim] — DeltaNet QKV (compact: Q/K are n_k_heads)
    pub(crate) qkv_expanded: Handle,   // [3 * n_v_heads * head_dim] — expanded + L2-normalized Q/K/V
    z_buf: Handle,          // [z_dim] — output gate
    a_raw: Handle,          // [n_v_heads] — decay gate raw
    b_raw: Handle,          // [n_v_heads] — beta gate raw
    // Issue 642 F3: concatenated input projection output [qkv_dim + z_dim + 2*n_v_heads].
    // Used by the fused input projection path. Split into qkv/z/a_raw/b_raw
    // by Split4CubeCL after the single GEMV.
    input_proj_out: Handle, // [qkv_dim + z_dim + 2*n_v_heads]
    pub(crate) beta_buf: Handle,       // [n_v_heads] — computed beta
    pub(crate) decay_buf: Handle,      // [n_v_heads] — computed decay
    pub(crate) recurrent_out: Handle,  // [n_v_heads * head_dim] — recurrence output
    pub(crate) tmp: Handle,      // [n_embd] — out_proj / ffn intermediate (tree-verify bridge scatters it)
    ffn_gate: Handle,       // [mlp_hidden]
    ffn_up: Handle,         // [mlp_hidden]
    ffn_hidden: Handle,     // [mlp_hidden] — SwiGLU output
    // Issue 642 F2: concatenated gate+up output from single GEMV [2*mlp_hidden].
    // Used by the fused FFN input path (forward_from_x). Allocated once,
    // reused every tick.
    ffn_gate_up: Handle,    // [2 * mlp_hidden]
    // Only used by the non-fused path; the `ternary_gemv_residual` feature
    // fuses the down-projection GEMV with the residual add and skips this
    // buffer entirely (Issue 616).
    #[cfg_attr(feature = "ternary_gemv_residual", allow(dead_code))]
    ffn_out: Handle,        // [n_embd]
    pub(crate) logits: Handle,         // [vocab_size]

    // Per-layer persistent state
    pub(crate) deltanet_states: Vec<Option<Handle>>,  // recurrent state per DeltaNet layer
    pub(crate) conv_states: Vec<Option<Handle>>,      // conv1d sliding window per DeltaNet layer

    // Issue 665 Phase 2: GPU-side backup buffers for speculative decode.
    // Allocated once at `new()`; reused across all checkpoint/rollback cycles.
    // Avoids per-checkpoint allocation (which fragments the memory pool — same
    // lesson as FillZerosCubeCL) + eliminates all CPU↔GPU syncs from the
    // checkpoint path (state→backup + backup→state are GPU-side CopyCubeCL).
    #[cfg(feature = "speculative_decode")]
    deltanet_state_backups: Vec<Option<Handle>>,
    #[cfg(feature = "speculative_decode")]
    conv_state_backups: Vec<Option<Handle>>,

    // Issue 727 H6: pre-allocated logits rotation pool for speculative verify
    // (the cudarc twin's proven pattern — SPEC_MAX_K vocab-sized buffers built
    // once at `new()`; verify swaps them in rotation instead of allocating K
    // fresh buffers per call, the exact pool-churn class reset_state's doc
    // warns caused SIGKILL under the Go arena).
    #[cfg(feature = "speculative_decode")]
    spec_logits_pool: Vec<Handle>,

    // Optional memory-pool cleanup counter (CUBECL_MEMORY_CLEANUP_INTERVAL)
    // When > 0, calls `client.memory_cleanup()` every N tokens to release
    // reclaimable pool slices, preventing dynamic-pool fragmentation from
    // accumulating over long generation runs (Issue 604 G2 crash workaround).
    cleanup_interval: usize,
    cleanup_counter: usize,

    // Attention layer buffers
    attn_qg: Handle,          // [2 * q_dim] — gated Q projection
    attn_q: Handle,           // [q_dim] — Q (split from qg)
    attn_gate: Handle,        // [q_dim] — gate (split from qg)
    // Issue 727 H10: the separate attn_k/attn_v buffers were dead — zero
    // reads since the Issue 648 F9 fusion (everything reads attn_kv).
    attn_out: Handle,         // [q_dim] — attention output
    // Issue 648 F9: concatenated K+V output buffer for single-GEMV projection.
    // K occupies [0..kvd], V occupies [kvd..2*kvd].
    attn_kv: Handle,          // [2 * kvd] — fused K+V projection
    // Issue 831 (a): split-K decode-attention partials — per-head per-split
    // online-softmax states (m, l, out[hd]) for the long-context decode
    // kernel. Sized at `new()` for the config's whole block_size with the
    // split-count cap applied (never re-allocated per call).
    attn_split_partials: Handle,
    // Issue 746 — KV cache contract (APPEND-ONLY + FULL-LENGTH): these
    // per-attention-layer caches are pre-allocated at `new()` for the full
    // `max_seq = config.block_size` positions and are only ever APPENDED at
    // `pos` — never evicted/trimmed/rotated/capped. `rollback_speculative[_gpu]`
    // relies on this write-before-read regime for phantom-token immunity:
    // attention decode at position p reads exactly `[0..=p]`
    // (`n_positions = pos + 1`), and every position a new sequence touches is
    // overwritten by its OWN KV append before any read consumes it — stale
    // data beyond `pos` is unreachable, so re-execution alone repairs the
    // cache after a rejected speculative verify (no trim needed).
    // Adding ANY evict/trim/rotate/cap API to these caches WITHOUT extending
    // the speculative rollback contract reintroduces the oMLX phantom-token
    // hazard (a rotated ring cannot trim — rejected verify tokens become
    // phantoms; oMLX `mlx_lm_mtp/cache_rollback.py`, DeepSeek-V4-Flash
    // sliding_window=128). See Issue 746.
    pub(crate) kv_key_caches: Vec<Option<Handle>>,   // per-attention-layer key cache
    pub(crate) kv_value_caches: Vec<Option<Handle>>, // per-attention-layer value cache

    // Issue 771 T2c-a / Plan 562: Q8-KV prefill scratch — K/V qs + scales
    // side buffers SHARED across layers within a forward (the Q8 attention
    // consumes the quantized rows within the same layer call, so one pair
    // sized for `block_size` rows suffices). Lazily allocated at the first
    // Q8-arm dispatch; a default build (arm off) never allocates it — the
    // ~151 MB at 32K geometry is paid only when the arm routes.
    #[cfg(all(
        feature = "cubecl_runtime",
        feature = "ternary_gemm_batched",
        feature = "ternary_attention_batched_prefill"
    ))]
    pub(crate) q8_prefill_scratch:
        std::sync::OnceLock<crate::qwen_prefill_q8kv_cubecl::Q8PrefillScratch>,

    // Issue 721: tree-verify T-row buffers (allocated lazily at first
    // `forward_tree_verify` call, sized for the largest requested T).
    #[cfg(feature = "speculative_tree_verify")]
    pub(crate) tree_buffers: Option<crate::ternary_tree_verify_driver::TreeVerifyGpuBuffers>,

    // Current position
    pub(crate) pos: usize,

    // Total length of the prefill prompt currently being driven. Set by the
    // prefill entry points (NOT the per-chunk length) — Issue 782: the flash
    // arm length gates (TILED_FLASH_M16_MIN_P / TILED_FLASH_CMMA_MIN_P) are
    // PROMPT-level semantics ("long prefills use the long-context arm"), but
    // the dispatch sits inside the chunk driver where every chunk is ≤
    // prefill_chunk_max() (4096) — gating on the chunk-local p made both arms
    // UNREACHABLE at exactly the lengths they were promoted for (every prompt
    // > 4096 chunks; no chunk ever reaches 8192/16384). Found by bench_809's
    // own ROUTING VACUITY guard on the first post-promotion @16K e2e (the
    // B817 step-4 cell).
    pub(crate) active_prefill_len: usize,

    // Plan 534 T2: Metal tensor GEMM context for the matmul2d prefill path.
    // `None` when the `metal_tensor_gemm` feature is off, non-macOS, or Metal
    // init failed (falls back to CubeCL silently).
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    metal_gemm: Option<crate::gemm_ternary_metal_tensor::MetalTensorGemm>,

    // Issue 657: zero-copy wgpu MSL passthrough variant. Shares CubeCL's
    // wgpu device/queue — dispatches the matmul2d kernel directly on
    // CubeCL-managed buffers. `None` when feature is off, non-macOS, or the
    // CubeCLContext didn't expose the wgpu device (e.g., CUDA backend).
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    metal_wgpu_gemm: Option<crate::gemm_ternary_metal_wgpu::MetalTensorWgpuGemm>,

    // Issue 663 T5: zero-copy RAW-Metal variant. Uses the wgpu-hal fork's
    // `Buffer::raw_handle()` to dispatch the matmul2d kernel via raw
    // `MTLComputeCommandEncoder` on the shared CubeCL Metal device + queue —
    // zero staging copy, zero bind group, zero host round-trip. `None` when
    // feature is off, non-macOS, or the CubeCLContext didn't expose the wgpu
    // device/queue.
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    metal_zerocopy_gemm: Option<crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm>,

    // Issue 726 T2: ANE hybrid prefill context + config. The ctx is
    // `NotWired` in T2 (see `crate::ane_prefill`) — the gate never fires
    // and prefill runs the GPU batched GEMM unchanged. T3 replaces the
    // ctx internals with the compiled Form C program bank.
    #[cfg(feature = "ane_prefill")]
    ane_prefill: crate::ane_prefill::AnePrefillCtx,
    #[cfg(feature = "ane_prefill")]
    ane_prefill_cfg: crate::ane_prefill::AnePrefillConfig,
    /// Issue 886 T5: the construction-time down-lane outcome (see
    /// [`crate::ane_prefill::AneDownReport`]) — the device-gate harness
    /// input for the ladder's landed-set + accounting pins.
    #[cfg(feature = "ane_prefill")]
    ane_down_report: crate::ane_prefill::AneDownReport,
}

#[cfg(feature = "cubecl_runtime")]
impl TernaryDeltanetGpuForward {
    /// Issue 994 — refuse to produce (or consume GPU work toward) results
    /// after a GPU memory-pool failure. A cubecl-wgpu reserve panic is
    /// survivable by construction (background threads absorb it), and every
    /// activation served from the pool after that point is deterministically
    /// corrupted: plausible shapes, bit-stable, wrong. The pool-poison
    /// detector records the failure; this check is the refusal half —
    /// called at construction, at every prefill-chunk and decode-funnel
    /// entry, and after every result-producing unit, so a poisoned pool can
    /// never return a successful result.
    fn refuse_if_pool_poisoned(phase: &str) {
        if let Some(detail) = crate::pool_poison::poisoned() {
            panic!(
                "[issue 994] GPU memory-pool failure detected {phase}: {detail}. \
                 Refusing instead of returning deterministically corrupted results. \
                 Remedy: shrink config.block_size / prompt length to the device budget \
                 (the caller-side clamp, Issue 510 T1/T2), free GPU memory, or move \
                 to the paged-KV lane (riir-train Issue 452 T5)."
            );
        }
    }

    /// Create a GPU-resident forward pass from loaded ternary weights.
    ///
    /// Pre-uploads ALL weights to GPU (~2s for 27B model). Buffers persist
    /// for the lifetime of this struct.
    ///
    /// The attention KV working set is sized from `config.block_size` — the
    /// model's declared context. Issue 864 measured that at a 262,144-token
    /// context this costs a ~50 s first-touch burst and multi-GiB residency,
    /// at STEADY-STATE PARITY (Bench 573 refuted the throughput half).
    /// Callers that do not drive the full context should clamp
    /// `config.block_size` before construction (the riir-train driver
    /// pattern, Issue 510 T1/T2); this constructor warns once when the
    /// resulting working set is large, naming the cost.
    pub fn new(
        ctx: &CubeCLContext,
        config: &Config,
        weights: &QwenDeltaNetTernaryWeights,
    ) -> Self {
        Self::new_with_rotation_policy(ctx, config, weights, /* allow_folded */ false)
    }

    /// Issue 980 T4-ALT / Plan 602 B2 — the folded-prefill constructor: the
    /// whole-prefill cudarc lane's entry. Since Plan 602 B2/B3 the DEFAULT
    /// constructor also accepts folded models (the CubeCL decode eager path
    /// carries the rotation), so this entry's remaining distinctions are the
    /// loud REQUIREMENT that the file be folded (a pre-rotation file here is
    /// a caller bug) — the per-path refusals (prefill body on Metal,
    /// training, capture) are keyed on the `rotation` marker either way.
    ///
    /// PANICS on a pre-rotation file (no `prism.hadamard` config) — use
    /// [`Self::new`] there.
    pub fn new_folded_prefill(
        ctx: &CubeCLContext,
        config: &Config,
        weights: &QwenDeltaNetTernaryWeights,
    ) -> Self {
        Self::new_with_rotation_policy(ctx, config, weights, /* require_folded */ true)
    }

    fn new_with_rotation_policy(
        ctx: &CubeCLContext,
        config: &Config,
        weights: &QwenDeltaNetTernaryWeights,
        require_folded: bool,
    ) -> Self {
        // Issue 994: construction itself uploads ~GB of weights — on a pool
        // that has already failed, refuse before the first upload.
        Self::refuse_if_pool_poisoned("at construction");
        // Issue 980 (Bonsai-2) / Plan 602 B2: folded models are ACCEPTED on
        // the default constructor — the decode eager path carries the
        // rotation at every folded site (B3), the dense escape-set a/b
        // dispatch as f32 GEMVs on the PRIMAL input, and the GPU sign tables
        // build here. The remaining loud refusals are per-PATH (the CubeCL
        // prefill body, training, the diagnostic capture — all refuse while
        // `rotation` is `Some`). Running folded weights UNROTATED remains the
        // silent-garbage class those guards prevent.
        if require_folded {
            assert!(
                weights.rotation.is_some(),
                "new_folded_prefill: not a Hadamard-folded model (no prism.hadamard config) \
                 — use TernaryDeltanetGpuForward::new for pre-rotation files"
            );
        }
        // Issue 994 refuse-to-run gate: estimate the constructor working set
        // (weights + activations + block_size-proportional KV) against the
        // adapter budget BEFORE any allocation. Silent self-consistent
        // corruption under pool exhaustion is the alternative (Bench 600:
        // bit-exact in-process, cos 0.18-0.42 vs an external reference).
        if let Err(reason) = crate::vram_budget::check_forward_budget(ctx, config, weights) {
            panic!(
                "TernaryDeltanetGpuForward::new: refusing to construct — {reason}"
            );
        }
        // Plan 602 B2 — build the GPU-resident rotation tables eagerly (the
        // sign vectors, uploaded once; every rotated launch reads these
        // handles — alloc-free steady state). A block size the split FWHT
        // kernel cannot serve refuses construction LOUD (the loader already
        // refused non-power-of-2 sizes; this bounds the shared-memory window).
        let rot_tables = weights.rotation.as_ref().map(|cfg| {
            crate::deltanet_rotation_cubecl::RotationTablesCubeCL::build(&ctx.client(), cfg)
                .expect("Bonsai-2 Hadamard rotation: the CubeCL kernel set cannot serve this file")
        });
        let client = ctx.client();
        let n = config.n_embd;
        let n_v_heads = config.deltanet_linear_n_value_heads;
        let head_dim = config.deltanet_linear_head_dim;
        let n_k_heads = config.deltanet_linear_n_heads;
        let q_dim = n_k_heads * head_dim;
        let k_dim = n_k_heads * head_dim;
        let v_dim = n_v_heads * head_dim;
        let qkv_dim = q_dim + k_dim + v_dim;
        let z_dim = v_dim;
        let conv_dim = qkv_dim;
        let kernel_size = config.deltanet_conv_kernel_size;
        let mlp = config.mlp_hidden;
        let vocab = config.vocab_size;

        let layer_types = if weights.layer_types.is_empty() {
            vec![DeltaNetLayerType::Attention; config.n_layer]
        } else {
            weights.layer_types.clone()
        };

        // Upload per-layer weights
        #[allow(unused_mut)]
        let mut layers: Vec<GpuLayerWeights> = weights
            .layers
            .iter()
            .map(|l| upload_layer_weights(&client, l))
            .collect();

        // Upload global weights
        let final_norm = upload_f32_slice(&client, &weights.final_norm);
        #[allow(unused_mut)]
        let mut lm_head = TernaryHandle::from_weights(&client, &weights.lm_head);
        #[allow(unused_mut)]
        let mut wte_handle = TernaryHandle::from_weights(&client, &weights.wte);

        // Allocate persistent activation buffers (zero-initialized)
        let zeros_n = vec![0.0f32; n];
        let x = client.create_from_slice(f32::as_bytes(&zeros_n));
        let norm_x = client.create_from_slice(f32::as_bytes(&zeros_n));
        let tmp = client.create_from_slice(f32::as_bytes(&zeros_n));
        let ffn_out = client.create_from_slice(f32::as_bytes(&zeros_n));

        let zeros_qkv = vec![0.0f32; qkv_dim];
        let qkv = client.create_from_slice(f32::as_bytes(&zeros_qkv));
        // Expanded QKV: 3 * n_v_heads * head_dim (Q/K broadcast from n_k → n_v during L2-norm)
        let zeros_qkv_exp = vec![0.0f32; 3 * n_v_heads * head_dim];
        let qkv_expanded = client.create_from_slice(f32::as_bytes(&zeros_qkv_exp));
        let zeros_z = vec![0.0f32; z_dim];
        let z_buf = client.create_from_slice(f32::as_bytes(&zeros_z));

        // Issue 642 F3: concatenated input projection output buffer.
        let input_proj_total = qkv_dim + z_dim + 2 * n_v_heads;
        let zeros_ip = vec![0.0f32; input_proj_total];
        let input_proj_out = client.create_from_slice(f32::as_bytes(&zeros_ip));

        let zeros_heads = vec![0.0f32; n_v_heads];
        let a_raw = client.create_from_slice(f32::as_bytes(&zeros_heads));
        let b_raw = client.create_from_slice(f32::as_bytes(&zeros_heads));
        let beta_buf = client.create_from_slice(f32::as_bytes(&zeros_heads));
        let decay_buf = client.create_from_slice(f32::as_bytes(&zeros_heads));

        let zeros_rec = vec![0.0f32; n_v_heads * head_dim];
        let recurrent_out = client.create_from_slice(f32::as_bytes(&zeros_rec));

        // Plan 602 B3 — the rotation-path scratch (folded models): the
        // rotated copy of `norm_x` + the gdn_v permute staging buffer.
        // Pre-allocated once (alloc-free steady state); dead buffers on
        // pre-rotation files (never dispatched — `rot_tables` gates).
        let rot_scratch = client.create_from_slice(f32::as_bytes(&zeros_n));
        let permute_tmp = client.create_from_slice(f32::as_bytes(&zeros_rec));

        let zeros_mlp = vec![0.0f32; mlp];
        let ffn_gate = client.create_from_slice(f32::as_bytes(&zeros_mlp));
        let ffn_up = client.create_from_slice(f32::as_bytes(&zeros_mlp));
        let ffn_hidden = client.create_from_slice(f32::as_bytes(&zeros_mlp));
        // Issue 642 F2: concatenated gate+up buffer for single-GEMV FFN input.
        let zeros_gate_up = vec![0.0f32; 2 * mlp];
        let ffn_gate_up = client.create_from_slice(f32::as_bytes(&zeros_gate_up));

        let zeros_vocab = vec![0.0f32; vocab];
        let logits = client.create_from_slice(f32::as_bytes(&zeros_vocab));

        // Allocate per-layer persistent state
        let state_dim = n_v_heads * head_dim * head_dim;
        let state_zeros = vec![0.0f32; state_dim];
        let conv_zeros = vec![0.0f32; conv_dim * kernel_size];

        let deltanet_states: Vec<Option<Handle>> = (0..config.n_layer)
            .map(|i| {
                if layer_types[i] == DeltaNetLayerType::DeltaNet {
                    Some(client.create_from_slice(f32::as_bytes(&state_zeros)))
                } else {
                    None
                }
            })
            .collect();

        let conv_states: Vec<Option<Handle>> = (0..config.n_layer)
            .map(|i| {
                if layer_types[i] == DeltaNetLayerType::DeltaNet {
                    Some(client.create_from_slice(f32::as_bytes(&conv_zeros)))
                } else {
                    None
                }
            })
            .collect();

        // Issue 665 Phase 2: GPU-side backup buffers for speculative decode.
        // Allocated once + reused — avoids per-checkpoint memory pool churn +
        // enables zero-CPU-sync checkpoint/rollback via CopyCubeCL.
        #[cfg(feature = "speculative_decode")]
        let deltanet_state_backups: Vec<Option<Handle>> = (0..config.n_layer)
            .map(|i| {
                if layer_types[i] == DeltaNetLayerType::DeltaNet {
                    Some(client.create_from_slice(f32::as_bytes(&state_zeros)))
                } else {
                    None
                }
            })
            .collect();
        #[cfg(feature = "speculative_decode")]
        let conv_state_backups: Vec<Option<Handle>> = (0..config.n_layer)
            .map(|i| {
                if layer_types[i] == DeltaNetLayerType::DeltaNet {
                    Some(client.create_from_slice(f32::as_bytes(&conv_zeros)))
                } else {
                    None
                }
            })
            .collect();

        // Issue 727 H6: logits rotation pool for speculative verify — mirrors
        // the cudarc twin. Pre-allocating kills the per-call `client.empty`
        // churn (mid-cycle allocations fragment the pool / can implicitly
        // sync — the reset_state SIGKILL lesson).
        #[cfg(feature = "speculative_decode")]
        let spec_logits_pool: Vec<Handle> = (0..Self::SPEC_MAX_K)
            .map(|_| client.empty(config.vocab_size * core::mem::size_of::<f32>()))
            .collect();

        // ── Attention layer buffers ──
        // Dimensions for full-attention layers
        let attn_n_head = config.n_head;
        let attn_n_kv = config.n_kv_head;
        let attn_hd = config.head_dim;
        let q_dim_attn = attn_n_head * attn_hd;
        let kvd_attn = attn_n_kv * attn_hd;

        let zeros_qg = vec![0.0f32; 2 * q_dim_attn];
        let attn_qg = client.create_from_slice(f32::as_bytes(&zeros_qg));
        let zeros_q = vec![0.0f32; q_dim_attn];
        let attn_q = client.create_from_slice(f32::as_bytes(&zeros_q));
        let attn_gate = client.create_from_slice(f32::as_bytes(&zeros_q));
        // Issue 648 F9: fused K+V buffer for single-GEMV projection.
        // (Issue 727 H10: the separate attn_k/attn_v buffers were dead — removed.)
        let zeros_kv2 = vec![0.0f32; 2 * kvd_attn];
        let attn_kv = client.create_from_slice(f32::as_bytes(&zeros_kv2));
        let attn_out = client.create_from_slice(f32::as_bytes(&zeros_q));
        // Issue 831 (a): split-K partials scratch — capped split count so the
        // footprint stays bounded for huge block_size (64 splits × 24 heads ×
        // 258 f32 = 1.6 MB worst case here; splits beyond the cap widen
        // split_len instead, per split_decode_geometry).
        let attn_split_max = config
            .block_size
            .div_ceil(attn_hd)
            .min(ATTN_SPLIT_DECODE_MAX_SPLITS_DEFAULT);
        let zeros_partials =
            vec![0.0f32; attn_n_head * attn_split_max * (attn_hd + 2)];
        let attn_split_partials = client.create_from_slice(f32::as_bytes(&zeros_partials));

        // Pre-allocate KV cache per attention layer (max_seq_len = block_size)
        let max_seq = config.block_size;
        let kv_cache_size = max_seq * kvd_attn;
        let kv_cache_zeros = vec![0.0f32; kv_cache_size];
        let n_attn_layers = layer_types
            .iter()
            .filter(|&&t| t == DeltaNetLayerType::Attention)
            .count();
        // Issue 864 T2: one warn per process when the KV working set is
        // block_size-proportional AND large — the measured signature of the
        // ~50 s first-touch burst (Bench 573). The approved remedy is a
        // caller-side clamp of `config.block_size` before construction (the
        // riir-train driver pattern, Issue 510 T1/T2) — the 93-site
        // constructor-parameter API (T1) was explicitly REJECTED by the
        // owner disposition of 2026-09-04.
        const KV_WARN_BYTES: usize = 256 * 1024 * 1024;
        let kv_bytes = kv_cache_size * 4 * 2 * n_attn_layers;
        if kv_bytes > KV_WARN_BYTES
            && !KV_WORKING_SET_WARNED.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "[Issue 864] attention KV working set = {} MiB (block_size {} × kvd {} × K+V × {} attention layers) — above the {} MiB warn line. If the workload never drives this many positions, clamp `config.block_size` to the real sequence bound BEFORE construction (the riir-train driver pattern); the unbounded variant measured a ~50 s first-touch burst at a 262,144-token context (riir-train Bench 573). Steady-state throughput is unaffected.",
                kv_bytes / (1024 * 1024),
                max_seq,
                kvd_attn,
                n_attn_layers,
                KV_WARN_BYTES / (1024 * 1024),
            );
        }
        let kv_key_caches: Vec<Option<Handle>> = (0..config.n_layer)
            .map(|i| {
                if layer_types[i] == DeltaNetLayerType::Attention {
                    Some(client.create_from_slice(f32::as_bytes(&kv_cache_zeros)))
                } else {
                    None
                }
            })
            .collect();
        let kv_value_caches: Vec<Option<Handle>> = (0..config.n_layer)
            .map(|i| {
                if layer_types[i] == DeltaNetLayerType::Attention {
                    Some(client.create_from_slice(f32::as_bytes(&kv_cache_zeros)))
                } else {
                    None
                }
            })
            .collect();

        // Plan 534 T2+T3: initialize the Metal tensor GEMM context + upload
        // all prefill-path weights to Metal buffers. macOS-only; falls back to
        // `None` (CubeCL path unchanged) on init failure or non-macOS.
        //
        // Issue 727 H3: the ctx + weight upload are gated on the dispatch flag
        // AT CONSTRUCTION — the metal-rs cache is a full weight copy onto a
        // separate device (Issue 656's separate-device path, measured 0.87×),
        // and uploading it when `PREFILL_USE_METAL_TENSOR` is off pins a dead
        // duplicate of the whole GEMM weight set for the forward's lifetime.
        // Flipping the flag after construction cannot retrofit the cache (it
        // needs the CPU weights) — the dispatch site warns once + falls through
        // to CubeCL. A/B harnesses must set the flag BEFORE `new()` (see
        // bench_656).
        #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
        let metal_gemm = {
            let _ = (&lm_head, &wte_handle); // borrow check: these exist
            #[cfg(feature = "ternary_gemm_batched")]
            let want_metal_rs =
                PREFILL_USE_METAL_TENSOR.load(std::sync::atomic::Ordering::Relaxed);
            // Without ternary_gemm_batched, prefill (the only metal-rs
            // consumer) is not compiled — never pay for the cache.
            #[cfg(not(feature = "ternary_gemm_batched"))]
            let want_metal_rs = false;
            if !want_metal_rs {
                eprintln!(
                    "[Issue 727] metal-rs GEMM cache skipped (PREFILL_USE_METAL_TENSOR \
                     off at construction; set it BEFORE new() to A/B this path)"
                );
                None
            } else {
                match crate::gemm_ternary_metal_tensor::MetalTensorGemm::new() {
                    Ok(g) => {
                        eprintln!(
                            "[Plan 534] Metal tensor GEMM initialized; uploading weights..."
                        );
                        Some(g)
                    }
                    Err(e) => {
                        eprintln!(
                            "[Plan 534] Metal tensor GEMM init failed ({e}); staying on CubeCL path"
                        );
                        None
                    }
                }
            }
        };

        // Issue 657: zero-copy wgpu MSL passthrough variant. Requires the
        // shared wgpu device/queue from CubeCLContext (available on the
        // non-CUDA wgpu path via `init_setup`).
        #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
        let metal_wgpu_gemm = if let (Some(device), Some(queue)) = (ctx.wgpu_device(), ctx.wgpu_queue()) {
                match crate::gemm_ternary_metal_wgpu::MetalTensorWgpuGemm::new(
                    device.clone(),
                    queue.clone(),
                ) {
                    Ok(g) => {
                        eprintln!("[Issue 657] wgpu MSL passthrough GEMM initialized (zero-copy)");
                        Some(g)
                    }
                    Err(e) => {
                        eprintln!(
                            "[Issue 657] wgpu MSL passthrough GEMM init failed ({e}); \
                             zero-copy path disabled"
                        );
                        None
                    }
                }
            } else {
                eprintln!("[Issue 657] wgpu device/queue not available (CUDA backend?); zero-copy path disabled");
                None
            };

        // Issue 663 T5: zero-copy RAW-Metal variant. Requires the wgpu-hal
        // fork (Buffer::raw_handle) + the shared wgpu device/queue.
        #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
        let metal_zerocopy_gemm = if let (Some(device), Some(queue)) = (ctx.wgpu_device(), ctx.wgpu_queue()) {
                match crate::gemm_ternary_metal_zero_copy::MetalTensorZeroCopyGemm::new(
                    device.clone(),
                    queue.clone(),
                ) {
                    Ok(g) => {
                        eprintln!(
                            "[Issue 663 T5] zero-copy raw-Metal GEMM initialized \
                             (device={:#x}, queue={:#x})",
                            g.device_ptr(),
                            g.queue_ptr()
                        );
                        Some(g)
                    }
                    Err(e) => {
                        eprintln!(
                            "[Issue 663 T5] zero-copy raw-Metal GEMM init failed ({e}); \
                             zero-copy-raw path disabled"
                        );
                        None
                    }
                }
            } else {
                eprintln!(
                    "[Issue 663 T5] wgpu device/queue not available (CUDA backend?); \
                     zero-copy-raw path disabled"
                );
                None
            };

        // Plan 550: initialize the ANE zero-copy IO context (raw device/queue
        // + the pack/unpack staging pipelines) — one OnceLock per process;
        // the split dispatch sites read the cached global. Failure is quiet:
        // the seam falls back to the measured host-IO split.
        #[cfg(all(
            feature = "ane_prefill",
            feature = "metal_tensor_gemm",
            all(target_os = "macos", target_arch = "aarch64")
        ))]
        {
            if let (Some(d), Some(q)) = (ctx.wgpu_device(), ctx.wgpu_queue()) {
                let _ = crate::ane_prefill::exec_zc::zc_context(d.clone(), q.clone());
            }
        }

        // Upload prefill-path weights to Metal. Each TernaryHandle that will
        // be dispatched through `prefill_project` gets a parallel Metal cache.
        // Skip weights not in the GEMM path (input_norm, conv1d, etc. are f32
        // elementwise ops, not GEMMs).
        #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
        if let Some(ref gemm) = metal_gemm {
            use crate::gemv_ternary_cubecl::{cast_u64_to_u32, prepare_group_scale_f32};
            use katgpt_core::TernaryGroupWeights;

            // Helper: upload one weight matrix from its source `TernaryGroupWeights`.
            let upload_one = |h: &mut TernaryHandle, w: &TernaryGroupWeights| {
                let pos = cast_u64_to_u32(&w.pos_bits);
                let neg = cast_u64_to_u32(&w.neg_bits);
                let scale = prepare_group_scale_f32(&w.group_scale);
                h.upload_to_metal(gemm, &pos, &neg, &scale);
            };

            // lm_head + wte (global weights).
            upload_one(&mut lm_head, &weights.lm_head);
            upload_one(&mut wte_handle, &weights.wte);

            // Per-layer weights. Reconstruct the raw weight refs from the
            // original `weights.layers` (the handles in `layers` lost the
            // original `TernaryGroupWeights` reference after upload).
            for (lh, wh) in layers.iter_mut().zip(weights.layers.iter()) {
                // DeltaNet path GEMMs.
                if wh.in_proj_qkv.rows > 0
                    && wh.in_proj_a.as_ternary().is_some()
                    && wh.in_proj_b.as_ternary().is_some()
                {
                    let a_t = wh.in_proj_a.as_ternary().expect("checked above");
                    let b_t = wh.in_proj_b.as_ternary().expect("checked above");
                    // The concat handle (in_proj_concat) is what prefill_project
                    // actually dispatches for the DeltaNet input projection.
                    let pos: Vec<u32> = [&wh.in_proj_qkv, &wh.in_proj_z, a_t, b_t]
                        .iter()
                        .flat_map(|w| cast_u64_to_u32(&w.pos_bits))
                        .collect();
                    let neg: Vec<u32> = [&wh.in_proj_qkv, &wh.in_proj_z, a_t, b_t]
                        .iter()
                        .flat_map(|w| cast_u64_to_u32(&w.neg_bits))
                        .collect();
                    let scale: Vec<f32> = [&wh.in_proj_qkv, &wh.in_proj_z, a_t, b_t]
                        .iter()
                        .flat_map(|w| prepare_group_scale_f32(&w.group_scale))
                        .collect();
                    lh.in_proj_concat.upload_to_metal(gemm, &pos, &neg, &scale);
                } else if wh.in_proj_qkv.rows > 0 {
                    // Plan 602 B2 — folded layer (dense a/b): the concat
                    // handle is qkv|z (mirrors `upload_layer_weights`); the
                    // metal-rs cache stays shape-consistent with the CubeCL
                    // handle even though the folded prefill never dispatches
                    // here (the CubeCL prefill body refuses folded models).
                    let pos: Vec<u32> = cast_u64_to_u32(&wh.in_proj_qkv.pos_bits)
                        .into_iter()
                        .chain(cast_u64_to_u32(&wh.in_proj_z.pos_bits))
                        .collect();
                    let neg: Vec<u32> = cast_u64_to_u32(&wh.in_proj_qkv.neg_bits)
                        .into_iter()
                        .chain(cast_u64_to_u32(&wh.in_proj_z.neg_bits))
                        .collect();
                    let scale: Vec<f32> = prepare_group_scale_f32(&wh.in_proj_qkv.group_scale)
                        .into_iter()
                        .chain(prepare_group_scale_f32(&wh.in_proj_z.group_scale))
                        .collect();
                    lh.in_proj_concat.upload_to_metal(gemm, &pos, &neg, &scale);
                }
                upload_one(&mut lh.out_proj, &wh.out_proj);
                // FFN: the concat gate+up handle is what prefill dispatches.
                {
                    let pos: Vec<u32> = cast_u64_to_u32(&wh.gate_proj.pos_bits)
                        .into_iter()
                        .chain(cast_u64_to_u32(&wh.up_proj.pos_bits))
                        .collect();
                    let neg: Vec<u32> = cast_u64_to_u32(&wh.gate_proj.neg_bits)
                        .into_iter()
                        .chain(cast_u64_to_u32(&wh.up_proj.neg_bits))
                        .collect();
                    let scale: Vec<f32> = prepare_group_scale_f32(&wh.gate_proj.group_scale)
                        .into_iter()
                        .chain(prepare_group_scale_f32(&wh.up_proj.group_scale))
                        .collect();
                    lh.gate_up_proj.upload_to_metal(gemm, &pos, &neg, &scale);
                }
                upload_one(&mut lh.down_proj, &wh.down_proj);
                // Attention path GEMMs (if present).
                if let Some(ref mut h) = lh.attn_wq { upload_one(h, &wh.attn_wq); }
                if wh.attn_wk.rows > 0 && wh.attn_wv.rows > 0 {
                    let pos: Vec<u32> = cast_u64_to_u32(&wh.attn_wk.pos_bits)
                        .into_iter()
                        .chain(cast_u64_to_u32(&wh.attn_wv.pos_bits))
                        .collect();
                    let neg: Vec<u32> = cast_u64_to_u32(&wh.attn_wk.neg_bits)
                        .into_iter()
                        .chain(cast_u64_to_u32(&wh.attn_wv.neg_bits))
                        .collect();
                    let scale: Vec<f32> = prepare_group_scale_f32(&wh.attn_wk.group_scale)
                        .into_iter()
                        .chain(prepare_group_scale_f32(&wh.attn_wv.group_scale))
                        .collect();
                    if let Some(ref mut h) = lh.attn_wkv { h.upload_to_metal(gemm, &pos, &neg, &scale); }
                }
                if let Some(ref mut h) = lh.attn_wo { upload_one(h, &wh.attn_wo); }
            }
            eprintln!("[Plan 534] Metal weight upload complete.");
        }

        // Issue 657 / Issue 727 H3: the wgpu MSL passthrough weight cache is
        // populated LAZILY at first dispatch (see `prefill_project`) — the
        // eager cache here was a full GPU→GPU copy of every GEMM weight even
        // when `PREFILL_USE_METAL_TENSOR_WGPU` never fires. The zerocopy cache
        // below stays eager: it only borrows raw `id<MTLBuffer>` pointers
        // (zero copy, effectively free).

        // Issue 663 T5: cache weights for the zero-copy RAW-Metal path.
        // Unlike the wgpu passthrough path (which copies CubeCL handles to
        // dedicated wgpu buffers), this path borrows the CubeCL handles'
        // raw `id<MTLBuffer>` pointers directly — zero copy, no staging
        // buffer, no GPU→GPU copy. The cache is valid for as long as the
        // TernaryHandle (and its CubeCL handles) stay alive.
        #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
        if let Some(ref gemm) = metal_zerocopy_gemm {
            let client_ref = &client;
            let cache_one = |h: &mut TernaryHandle| {
                h.zerocopy_cache = gemm.cache_weights(client_ref, h).ok();
            };
            cache_one(&mut lm_head);
            cache_one(&mut wte_handle);
            for lh in layers.iter_mut() {
                cache_one(&mut lh.in_proj_concat);
                cache_one(&mut lh.out_proj);
                cache_one(&mut lh.gate_up_proj);
                cache_one(&mut lh.down_proj);
                if let Some(ref mut h) = lh.attn_wq { cache_one(h); }
                if let Some(ref mut h) = lh.attn_wkv { cache_one(h); }
                if let Some(ref mut h) = lh.attn_wo { cache_one(h); }
            }
            eprintln!("[Issue 663 T5] zero-copy raw-Metal weight cache complete.");
        }

        // Issue 726 T3: ANE hybrid prefill bank. Registration is driven by
        // the runtime flag (set before construction — see `crate::ane_prefill`);
        // every refusal reports Unavailable and prefill fail-opens to the
        // GPU batched GEMM. The bank compiles Form C programs from the SPLIT
        // CPU weights (row-concat requant — no merged planes materialized).
        // Plan 549: the bank budget resolves programmatic override →
        // `RIIR_ANE_MAX_BYTES` env → default (the down_proj extension adds
        // ~8.5 GB over the 12.6 GB f=1.0 bank — raising it is always a
        // deliberate act).
        #[cfg(feature = "ane_prefill")]
        let ane_prefill_cfg = crate::ane_prefill::AnePrefillConfig {
            max_layers_hint: config.n_layer,
            ..crate::ane_prefill::AnePrefillConfig::default()
        };
        #[cfg(feature = "ane_prefill")]
        let (ane_prefill, ane_down_report) = {
            let mut cfg = ane_prefill_cfg;
            cfg.max_ane_bytes = crate::ane_prefill::prefill_ane_max_bytes_effective();
            // `mut` is consumed only by the macos+aarch64 eager-compile block
            // below (Issue 726 T6); every other platform keeps an idle ctx and
            // must not warn (removing `mut` would break the host build).
            #[allow(unused_mut)]
            let mut ctx = crate::ane_prefill::AnePrefillCtx::init(&cfg);
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                let gdn_layers: Vec<usize> = (0..config.n_layer)
                    .filter(|&i| layer_types[i] == DeltaNetLayerType::DeltaNet)
                    .collect();
                // Issue 726 T6: eager compile with progress — real-dims
                // compiles cost 0.69-1.2 s/program (P9), ~90 s for
                // Bonsai's 96-program bank; the load-time stall must be
                // observable, not silent.
                let compile_started = std::time::Instant::now();
                let mut gdn_done = 0usize;
                // Issue 886 T3: the measured down-registration descent —
                // replaces the static `down_cap` guess. An explicit
                // `RIIR_ANE_DOWN_LAYERS`/override stays a deliberate claim
                // (Plan 549 semantics, NO descent); unset + down toggle on
                // → the ladder starts ambitious (all GDN layers) and
                // halves its remaining target on every CONFIRMED refusal
                // (settle + re-measure + one retry, then halve). The
                // per-rung measured gate is the T1 machine-side headroom
                // term; the cumulative byte caps bound the lane.
                let down_explicit =
                    crate::ane_prefill::prefill_ane_down_layers_explicit();
                let mut ladder =
                    crate::ane_prefill::DownLadder::new(down_explicit, gdn_layers.len());
                let ladder_byte_cap =
                    crate::ane_prefill::prefill_ane_down_ladder_max_bytes_effective();
                let down_enabled = crate::ane_prefill::prefill_ane_down();
                let budget_ceiling = cfg.max_ane_bytes;
                let mut down_spent: u64 = 0;
                let mut down_landed = 0usize;
                let mut down_refusals = 0usize;
                // Issue 886 T5: per-layer outcome record for the device
                // gates (`ane_down_report`) — attempted/landed/refused in
                // registration order.
                let mut down_report = crate::ane_prefill::AneDownReport::default();
                // The measured gate, (re-)taken at rung boundaries ONLY
                // (each re-measure costs the 0.5 s T2 settle; per-layer
                // would add ~24 s at 48 layers).
                let mut measured_gate: Option<u64> = None;
                if down_enabled && ctx.is_ready() && !gdn_layers.is_empty() {
                    // T2: settle before EVERY headroom (re-)measure —
                    // including the first (oMLX measures the same way).
                    crate::ane_prefill::headroom::settle_before_remeasure();
                    measured_gate = crate::ane_prefill::headroom::machine_side_ceiling(
                        crate::ane_prefill::headroom_fraction_effective(),
                    );
                    eprintln!(
                        "[Issue 886] down ladder armed: target {} layers, measured gate {:.2} GB of {:.2} GB budget",
                        ladder.attempts_remaining(),
                        measured_gate.unwrap_or(0) as f64 / 1e9,
                        budget_ceiling as f64 / 1e9
                    );
                }
                if ctx.is_ready() && !gdn_layers.is_empty() {
                    eprintln!(
                        "[Issue 726] compiling ANE prefill bank: {} GDN layers x 2 programs (~90 s at real dims)...",
                        gdn_layers.len()
                    );
                }
                // Plan 602 B2 — the ANE prefill bank is primal-basis-shaped
                // (its compiled programs cannot carry the Hadamard rotation)
                // and the CubeCL prefill body refuses folded models anyway
                // (the whole-prefill cudarc lane is the folded prefill
                // surface). Skip registration entirely on a folded file: the
                // ANE lane reports Unavailable and prefill fail-opens — to
                // the loud folded refusal — instead of the old dense-a/b
                // `expect` panic poisoning construction for the DECODE lane.
                let ane_folded_skip = weights.rotation.is_some();
                if ane_folded_skip {
                    eprintln!(
                        "[Issue 980/Plan 602] Hadamard-folded model: ANE prefill bank registration SKIPPED (ANE programs are primal-shaped; prefill refuses folded on this lane)"
                    );
                }
                for i in 0..config.n_layer {
                    if ane_folded_skip {
                        break;
                    }
                    let l = &weights.layers[i];
                    let is_gdn = layer_types[i] == DeltaNetLayerType::DeltaNet;
                    // Issue 886 T3: the ladder's arithmetic gate (measured at
                    // rung boundaries) decides whether this layer's down op
                    // is attempted; `register_layer`'s per-op fail-open stays
                    // the safety net (a refused attempt leaves the slot
                    // empty, never poisons the bank).
                    let down_w = if down_enabled && is_gdn && ladder.wants_attempt() {
                        let bytes = (l.down_proj.rows * l.down_proj.cols) as u64;
                        let bank_bytes = ctx.bank_bytes_total().unwrap_or(0);
                        if crate::ane_prefill::down_admits(
                            budget_ceiling,
                            measured_gate,
                            ladder_byte_cap,
                            down_spent,
                            bank_bytes,
                            bytes,
                        ) {
                            Some(&l.down_proj)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let attempted = down_w.is_some();
                    if attempted {
                        down_report.attempted.push(i);
                    }
                    ctx.register_layer(
                        i,
                        is_gdn,
                        // Issue 980 / Plan 602 B2: a/b ternary-only here —
                        // folded files never reach this call (the
                        // `ane_folded_skip` break above); the 4-slot shape is
                        // the pre-rotation contract.
                        &[
                            &l.in_proj_qkv,
                            &l.in_proj_z,
                            l.in_proj_a.as_ternary().expect("Metal register_layer requires ternary in_proj_a"),
                            l.in_proj_b.as_ternary().expect("Metal register_layer requires ternary in_proj_b"),
                        ],
                        &[&l.gate_proj, &l.up_proj],
                        down_w,
                    );
                    // Early-out: once poisoned, further registers no-op.
                    if !ctx.is_ready() {
                        break;
                    }
                    if attempted {
                        ladder.on_attempt();
                        if ctx.down_registered(i) {
                            down_spent += (l.down_proj.rows * l.down_proj.cols) as u64;
                            down_landed += 1;
                            down_report.landed.push(i);
                        } else {
                            // Refusal rung: settle + re-measure (T2), retry
                            // ONCE if the refreshed gate admits, then either
                            // way confirm/halve per the ladder.
                            crate::ane_prefill::headroom::settle_before_remeasure();
                            measured_gate = crate::ane_prefill::headroom::machine_side_ceiling(
                                crate::ane_prefill::headroom_fraction_effective(),
                            );
                            let bytes = (l.down_proj.rows * l.down_proj.cols) as u64;
                            let bank_bytes = ctx.bank_bytes_total().unwrap_or(0);
                            let landed =
                                if crate::ane_prefill::down_admits(
                                    budget_ceiling,
                                    measured_gate,
                                    ladder_byte_cap,
                                    down_spent,
                                    bank_bytes,
                                    bytes,
                                ) {
                                    ctx.try_register_down(i, &l.down_proj)
                                } else {
                                    false
                                };
                            if landed {
                                down_spent += bytes;
                                down_landed += 1;
                                down_report.landed.push(i);
                            } else {
                                down_refusals += 1;
                                down_report.refused.push(i);
                                let keep_going = ladder.on_refusal_confirmed();
                                eprintln!(
                                    "[Issue 886] down ladder: layer {i} refused ({}/{} landed, {} refusals) — target now {}{}",
                                    down_landed,
                                    gdn_layers.len(),
                                    down_refusals,
                                    ladder.attempts_remaining(),
                                    if keep_going { "" } else { " — ladder done" }
                                );
                            }
                        }
                    }
                    if is_gdn {
                        gdn_done += 1;
                        if gdn_done.is_multiple_of(8) {
                            eprintln!(
                                "[Issue 726] ANE bank: {gdn_done}/{} GDN layers ({:.0}s elapsed)",
                                gdn_layers.len(),
                                compile_started.elapsed().as_secs_f32()
                            );
                        }
                    }
                }
                let sealed = ctx.finalize(&gdn_layers);
                down_report.bytes_spent = down_spent;
                match sealed.state() {
                    crate::ane_prefill::AnePrefillState::Ready(_) => {
                        if let Some((bytes, ceiling)) = sealed.bank_bytes() {
                            eprintln!(
                                "[Issue 726] ANE prefill bank ready: {} programs in {:.1}s, {:.2}/{:.2} GB budget{}",
                                gdn_layers.len() * 2 + down_landed,
                                compile_started.elapsed().as_secs_f32(),
                                bytes as f64 / 1e9,
                                ceiling as f64 / 1e9,
                                if down_landed > 0 {
                                    format!(
                                        " (+{down_landed} down via ladder, {down_refusals} refusals)"
                                    )
                                } else {
                                    String::new()
                                }
                            );
                        }
                    }
                    crate::ane_prefill::AnePrefillState::Unavailable(reason) => {
                        eprintln!(
                            "[Issue 726] ANE prefill unavailable after {:.1}s: {reason} — prefill fail-opens to GPU",
                            compile_started.elapsed().as_secs_f32()
                        );
                    }
                }
                (sealed, down_report)
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                (
                    ctx,
                    crate::ane_prefill::AneDownReport::default(),
                )
            }
        };

        // Issue 994 construction flush-gate: drain the per-stream error sink
        // and refuse (loud panic) when ANY construction-time allocation /
        // write / launch failed. Without this, the Issue 714 vendor patch's
        // non-fatal OOM routing leaves buffers unbound while the forward
        // still completes — the measured silent-corruption mechanism.
        if let Err(reason) = crate::vram_budget::drain_construction_errors(&client) {
            panic!(
                "TernaryDeltanetGpuForward::new: construction failed — {reason}"
            );
        }

        Self {
            // Issue 860 T3: nothing uploaded yet — the first forward must be
            // preceded by `set_input_token` or a `prefill`.
            x_input_fresh: false,
            rotation: weights.rotation.clone(),
            rot_tables,
            rot_scratch,
            permute_tmp,
            client,
            config: config.clone(),
            layer_types,
            layers,
            final_norm,
            lm_head,
            x,
            norm_x,
            qkv,
            qkv_expanded,
            z_buf,
            input_proj_out,
            a_raw,
            b_raw,
            beta_buf,
            decay_buf,
            recurrent_out,
            tmp,
            ffn_gate,
            ffn_up,
            ffn_hidden,
            ffn_gate_up,
            ffn_out,
            logits,
            deltanet_states,
            conv_states,
            #[cfg(feature = "speculative_decode")]
            deltanet_state_backups,
            #[cfg(feature = "speculative_decode")]
            conv_state_backups,
            #[cfg(feature = "speculative_decode")]
            spec_logits_pool,
            attn_qg,
            attn_q,
            attn_gate,
            attn_kv,
            attn_out,
            attn_split_partials,
            kv_key_caches,
            kv_value_caches,
            #[cfg(all(
                feature = "cubecl_runtime",
                feature = "ternary_gemm_batched",
                feature = "ternary_attention_batched_prefill"
            ))]
            q8_prefill_scratch: std::sync::OnceLock::new(),
            wte_handle,
            cleanup_interval: std::env::var("CUBECL_MEMORY_CLEANUP_INTERVAL")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&n| n > 0)
                .unwrap_or(0),
            cleanup_counter: 0,
            #[cfg(feature = "speculative_tree_verify")]
            tree_buffers: None,
            pos: 0,
            active_prefill_len: 0,
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            metal_gemm,
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            metal_wgpu_gemm,
            #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
            metal_zerocopy_gemm,
            #[cfg(feature = "ane_prefill")]
            ane_prefill,
            #[cfg(feature = "ane_prefill")]
            ane_prefill_cfg,
            #[cfg(feature = "ane_prefill")]
            ane_down_report,
        }
    }

    /// Forward pass for a single decode token (GPU-resident, zero sync between layers).
    ///
    /// **Takes no token — this is half of a two-call protocol.** The embedding
    /// upload is [`set_input_token`](Self::set_input_token); this method runs
    /// all layers on whatever `x` currently holds and downloads the logits:
    ///
    /// ```ignore
    /// fwd.set_input_token(&weights, tok); // uploads the embedding into `x`
    /// let logits = fwd.forward_token();   // consumes it
    /// ```
    ///
    /// Prefill is the other legitimate producer: `prefill`'s final chunk leaves
    /// the last position's hidden state in `x`, so one bare `forward_token`
    /// after a prefill is correct (`tests/prefill_tail_g1.rs`).
    ///
    /// Until Issue 860 this took a `_token: usize` it ignored, under a doc
    /// comment that claimed it uploaded the embedding. `forward_token(tok)`
    /// alone re-ran the PREVIOUS token and returned a full, right-shaped,
    /// plausible — and wrong — logits vector. The parameter is gone so the
    /// protocol is visible rather than implied (T1) and the missing upload is
    /// a `debug_assert` rather than a silent number (T3). The cudarc twin's
    /// `forward_token()` already had this signature; the two now agree.
    ///
    /// Returns the logits as a `Vec<f32>` (downloaded once at the end).
    pub fn forward_token(&mut self) -> Vec<f32> {
        self.forward_dispatch_only();

        // Optional periodic memory-pool cleanup (Issue 604 G2 crash workaround).
        // See `apply_optional_memory_config` docs in cubecl_runtime.rs.
        if self.cleanup_interval > 0 {
            self.cleanup_counter += 1;
            if self.cleanup_counter >= self.cleanup_interval {
                self.cleanup_counter = 0;
                self.client.memory_cleanup();
            }
        }

        self.read_logits()
    }

    /// Issue 860 T3 escape hatch — declare, on the caller's authority, that `x`
    /// already holds this forward's input.
    ///
    /// There is exactly one legitimate use: a probe whose *subject* is `x`
    /// itself, which must run a forward on whatever a previous forward left
    /// there. `tests/prefill_tail_g1.rs` is that probe — the prefill tail's
    /// entire observable surface is `x`, so it compares a bare follow-up
    /// dispatch after `prefill` against the same bare follow-up after a
    /// sequential decode. Both arms deliberately forward from a *hidden state*
    /// rather than an embedding; the gate is arm-vs-arm agreement, not
    /// next-token semantics.
    ///
    /// It is NOT a way to quiet the assert in a decode or eval loop. There, a
    /// missing `set_input_token` means the position is being predicted from
    /// the previous position's output and every later position is shifted —
    /// call `set_input_token` instead.
    pub fn mark_x_as_input(&mut self) {
        self.x_input_fresh = true;
    }

    /// Issue 860 T3 — assert `x` holds an input nothing has consumed yet, and
    /// consume it.
    ///
    /// Called at the top of every `forward_from_x*` funnel. `forward_from_x`
    /// mutates `x` in place as the residual stream, so after a forward `x`
    /// holds a *final hidden state*, not an embedding: a second forward
    /// without a fresh upload is unambiguously stale, never a caller being
    /// clever. Producers are `set_input_token` and the final-chunk tail of
    /// `prefill_tokens_chunk`; `reset_state` and the speculative rollbacks
    /// invalidate whatever was there.
    ///
    /// `debug_assert` and not `assert`: this is on the per-token decode path,
    /// and the failure it catches is a call-site protocol error, caught the
    /// first time the offending code runs in dev or in a test.
    fn consume_fresh_input(&mut self, entry: &str) {
        debug_assert!(
            self.x_input_fresh,
            "{entry}: no fresh input in `x` — call `set_input_token(&weights, tok)` \
             (or `prefill`) first. Forwarding here re-runs the PREVIOUS token and \
             returns a full, right-shaped, WRONG logits vector (Issue 860)."
        );
        self.x_input_fresh = false;
    }

    /// Dispatch all 832+ GPU operations for one decode token WITHOUT reading
    /// back the logits. Used for profiling the CPU-side dispatch encoding cost
    /// separately from the GPU sync cost (Issue 661 T1).
    ///
    /// The caller MUST call [`read_logits`] before reading `self.logits` —
    /// the GPU queue is async and the logits buffer is stale until drained.
    #[doc(alias = "forward_from_x")]
    pub fn forward_dispatch_only(&mut self) {
        // Issue 965 flush-before-consume: a graph-armed prefill elides its
        // per-chunk state writeback (the cudarc mirrors hold the truth until
        // the spec flush runs), and THIS pass is about to read the CubeCL
        // state handles (KV caches + GDN/conv states). Bring them current
        // first. O(1) atomic swap on every token that follows a reset or a
        // non-armed chunk; one bulk crossing per prompt at most.
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            feature = "ternary_gemm_batched",
            feature = "cubecl_runtime",
            not(target_os = "macos")
        ))]
        crate::prefill_cuda_full::prefill_spec_flush(self);
        let n = self.config.n_embd;
        let eps = self.config.rms_norm_eps as f32;
        let n_v_heads = self.config.deltanet_linear_n_value_heads;
        let head_dim = self.config.deltanet_linear_head_dim;
        let conv_dim = 2 * (self.config.deltanet_linear_n_heads * head_dim)
            + n_v_heads * head_dim;
        let kernel_size = self.config.deltanet_conv_kernel_size;
        let mlp = self.config.mlp_hidden;
        let vocab = self.config.vocab_size;
        let z_dim = n_v_heads * head_dim;

        self.forward_from_x(
            n, eps, n_v_heads, head_dim, z_dim, conv_dim, kernel_size, mlp, vocab,
        );
    }

    /// Download the logits buffer (forces GPU sync). Returns the logits as
    /// a `Vec<f32>`. Used after [`forward_dispatch_only`] to drain the queue.
    pub fn read_logits(&mut self) -> Vec<f32> {
        // Issue 994: the logits in a poisoned pool are corrupted garbage —
        // refuse the read instead of handing back plausible tokens.
        Self::refuse_if_pool_poisoned("at logits readback");
        let bytes = self
            .client
            .read_one(self.logits.clone())
            .expect("download logits");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Forward pass that returns BOTH logits AND the final normed hidden state.
    ///
    /// The hidden state (`norm_x` after the final RMSNorm, before `lm_head`) is
    /// needed for lm_head-only LoRA training (Plan 528 T1.3): the LoRA forward is
    /// `logits_lora = α · B @ (A @ norm_x)`, and the LoRA backward needs `norm_x`
    /// to compute `dL/dB = α · outer(dL/dlogits_lora, A @ norm_x)`.
    ///
    /// This adds one extra GPU→CPU download (5120 f32 = 20 KB) compared to
    /// [`forward_token`], which is negligible vs the logits download (993 KB).
    ///
    /// Returns `(logits, final_norm_x)` where `final_norm_x.len() == n_embd`.
    pub fn forward_token_with_final_hidden(&mut self) -> (Vec<f32>, Vec<f32>) {
        let n = self.config.n_embd;
        let eps = self.config.rms_norm_eps as f32;
        let n_v_heads = self.config.deltanet_linear_n_value_heads;
        let head_dim = self.config.deltanet_linear_head_dim;
        let z_dim = n_v_heads * head_dim;
        let conv_dim = 2 * (self.config.deltanet_linear_n_heads * head_dim) + z_dim;
        let kernel_size = self.config.deltanet_conv_kernel_size;
        let mlp = self.config.mlp_hidden;
        let vocab = self.config.vocab_size;

        self.forward_from_x(n, eps, n_v_heads, head_dim, z_dim, conv_dim, kernel_size, mlp, vocab);

        if self.cleanup_interval > 0 {
            self.cleanup_counter += 1;
            if self.cleanup_counter >= self.cleanup_interval {
                self.cleanup_counter = 0;
                self.client.memory_cleanup();
            }
        }

        // Download both norm_x (final hidden, before lm_head) and logits.
        let hidden_bytes = self
            .client
            .read_one(self.norm_x.clone())
            .expect("download final norm_x");
        let final_norm_x = f32::from_bytes(&hidden_bytes).to_vec();

        let logits_bytes = self
            .client
            .read_one(self.logits.clone())
            .expect("download logits");
        let logits = f32::from_bytes(&logits_bytes).to_vec();

        (logits, final_norm_x)
    }




    /// **Minimal-cache training forward** (Issue 641 T2) — saves `x_in` +
    /// `norm_x` for ALL layers, plus `qkv_expanded` + `beta` + `decay` for
    /// DeltaNet layers, into a [`MinimalActivationCache`].
    ///
    /// Unlike `forward_token_training` (which only saves 4 DeltaNet fields
    /// and skips attention layers), this saves the pre-RMSNorm hidden state
    /// (`x_in`) for every layer — the field the RMSNorm backward needs to
    /// compute `grad_x_in`.
    ///
    /// The backward (T3) recomputes the remaining 8 DeltaNet fields + all
    /// attention fields from these 5 (DeltaNet) / 2 (attention) saved values
    /// + frozen weights.
    ///
    /// # Cost
    ///
    /// One extra GPU sync per layer per token (download `x_in`). For the
    /// 64-layer model: 64 extra syncs/token. Acceptable for training
    /// precompute; never use on the inference path.
    ///
    /// # Backward compatibility
    ///
    /// The existing `forward_token_training` + TrainingActivationCollector
    /// path is unchanged. This is a NEW method for the recomputation backward
    /// (Issue 641 Path A).
    pub fn forward_token_training_minimal(
        &mut self,
        cache: &mut MinimalActivationCache,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = self.config.n_embd;
        let eps = self.config.rms_norm_eps as f32;
        let n_v_heads = self.config.deltanet_linear_n_value_heads;
        let head_dim = self.config.deltanet_linear_head_dim;
        let z_dim = n_v_heads * head_dim;
        let conv_dim = 2 * (self.config.deltanet_linear_n_heads * head_dim) + z_dim;
        let kernel_size = self.config.deltanet_conv_kernel_size;
        let mlp = self.config.mlp_hidden;
        let vocab = self.config.vocab_size;
        let n_layer = self.layer_types.len();

        self.forward_from_x_training_minimal(
            n, eps, n_v_heads, head_dim, z_dim, conv_dim, kernel_size, mlp, vocab, n_layer, cache,
        );

        if self.cleanup_interval > 0 {
            self.cleanup_counter += 1;
            if self.cleanup_counter >= self.cleanup_interval {
                self.cleanup_counter = 0;
                self.client.memory_cleanup();
            }
        }

        // Download both norm_x (final hidden, before lm_head) and logits.
        let hidden_bytes = self
            .client
            .read_one(self.norm_x.clone())
            .expect("download final norm_x");
        let final_norm_x = f32::from_bytes(&hidden_bytes).to_vec();

        let logits_bytes = self
            .client
            .read_one(self.logits.clone())
            .expect("download logits");
        let logits = f32::from_bytes(&logits_bytes).to_vec();

        (logits, final_norm_x)
    }

    /// Internal: layer loop with minimal-cache activation collection (x_in for
    /// ALL layers + DeltaNet extras). Mirrors `forward_from_x_training` but
    /// downloads `x_in` (the hidden state BEFORE the input RMSNorm) for every
    /// layer, and `norm_x` for attention layers too.
    #[allow(clippy::too_many_arguments)]
    fn forward_from_x_training_minimal(
        &mut self,
        n: usize,
        eps: f32,
        n_v_heads: usize,
        head_dim: usize,
        z_dim: usize,
        conv_dim: usize,
        kernel_size: usize,
        mlp: usize,
        vocab: usize,
        n_layer: usize,
        cache: &mut MinimalActivationCache,
    ) {
        // Issue 980 T4-ALT — training refuses folded models (the marker).
        assert!(
            self.rotation.is_none(),
            "Bonsai-2 folded training on the CubeCL forward: not wired (Issue 980 T4-ALT)"
        );
        Self::refuse_if_pool_poisoned("at training decode entry");
        self.consume_fresh_input("forward_from_x_training_minimal");
        cache.begin_token(n_layer);

        for (layer_idx, layer_w) in self.layers.iter().enumerate() {
            let is_deltanet = self.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

            // ── Save x_in: download the hidden state BEFORE the input RMSNorm ──
            // This is the field the RMSNorm backward needs.
            let x_in_bytes = self
                .client
                .read_one(self.x.clone())
                .expect("download x_in for minimal cache");
            let x_in = f32::from_bytes(&x_in_bytes).to_vec();

            // ── Attention/DeltaNet block: input RMSNorm ──
            unsafe {
                RmsNormCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.x.clone(),
                    layer_w.input_norm.clone(),
                    self.norm_x.clone(),
                    n,
                    eps,
                );
            }

            // ── Save norm_x: download the post-RMSNorm hidden state ──
            let norm_x_bytes = self
                .client
                .read_one(self.norm_x.clone())
                .expect("download norm_x for minimal cache");
            let norm_x = f32::from_bytes(&norm_x_bytes).to_vec();

            if is_deltanet {
                self.forward_deltanet_layer_gpu(
                    layer_idx,
                    layer_w,
                    n,
                    n_v_heads,
                    head_dim,
                    z_dim,
                    conv_dim,
                    kernel_size,
                    eps,
                );

                // Download DeltaNet extras after the expand+L2-norm step.
                let qkv_bytes = self
                    .client
                    .read_one(self.qkv_expanded.clone())
                    .expect("download qkv_expanded for minimal cache");
                let beta_bytes = self
                    .client
                    .read_one(self.beta_buf.clone())
                    .expect("download beta for minimal cache");
                let decay_bytes = self
                    .client
                    .read_one(self.decay_buf.clone())
                    .expect("download decay for minimal cache");

                cache.push_layer_activation(MinimalLayerActivations::deltanet(
                    x_in,
                    norm_x,
                    f32::from_bytes(&qkv_bytes).to_vec(),
                    f32::from_bytes(&beta_bytes).to_vec(),
                    f32::from_bytes(&decay_bytes).to_vec(),
                ));
            } else {
                // Attention layer: save x_in + norm_x only.
                self.forward_attention_layer_gpu(layer_idx, layer_w, eps);
                cache.push_layer_activation(MinimalLayerActivations::attention(x_in, norm_x));
            }

            // ── Residual add (DeltaNet/attention output → x) ──
            unsafe {
                ResidualAddCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.x.clone(),
                    self.tmp.clone(),
                    self.x.clone(),
                    n,
                );
            }

            // ── FFN block ──
            unsafe {
                RmsNormCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.x.clone(),
                    layer_w.post_attn_norm.clone(),
                    self.norm_x.clone(),
                    n,
                    eps,
                );
            }
            unsafe {
                GemvTernaryCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    &layer_w.gate_proj,
                    self.norm_x.clone(),
                    self.ffn_gate.clone(),
                );
            }
            unsafe {
                GemvTernaryCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    &layer_w.up_proj,
                    self.norm_x.clone(),
                    self.ffn_up.clone(),
                );
            }
            unsafe {
                DeltanetGatingCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.ffn_gate.clone(),
                    self.ffn_up.clone(),
                    self.ffn_hidden.clone(),
                    mlp,
                );
            }

            // Issue 616: fused FFN down-projection + ResidualAdd.
            #[cfg(feature = "ternary_gemv_residual")]
            unsafe {
                GemvTernaryResidualCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    &layer_w.down_proj,
                    self.ffn_hidden.clone(),
                    self.x.clone(),
                );
            }
            #[cfg(not(feature = "ternary_gemv_residual"))]
            {
                unsafe {
                    GemvTernaryCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        &layer_w.down_proj,
                        self.ffn_hidden.clone(),
                        self.ffn_out.clone(),
                    );
                }
                unsafe {
                    ResidualAddCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        self.x.clone(),
                        self.ffn_out.clone(),
                        self.x.clone(),
                        n,
                    );
                }
            }
        }

        // Save x_pre_finalnorm (the hidden state before the final RMSNorm).
        let x_final_bytes = self
            .client
            .read_one(self.x.clone())
            .expect("download x_pre_finalnorm for minimal cache");
        cache.x_pre_finalnorm = f32::from_bytes(&x_final_bytes).to_vec();

        // Final RMSNorm
        unsafe {
            RmsNormCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x.clone(),
                self.final_norm.clone(),
                self.norm_x.clone(),
                n,
                eps,
            );
        }

        // LM head: logits = W_lm_head @ norm_x
        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &self.lm_head,
                self.norm_x.clone(),
                self.logits.clone(),
            );
        }

        let _ = vocab;
        self.pos += 1;
        // Issue 994: minimal-training-funnel exit — same refusal rule.
        Self::refuse_if_pool_poisoned("after training decode");
    }
    /// state `self.x` after each layer's residual add (matching the CPU
    /// `layer_capture` tap in `forward_qwen_deltanet_ternary_with_capture`).
    ///
    /// Forces a GPU sync per layer (slow — for debugging only, never hot path).
    /// If `capture` is `Some(buf)`, each `buf[layer_idx]` receives a copy of
    /// `self.x[0..n_embd]` after that layer completes.
    pub fn forward_token_with_layer_capture(
        &mut self,
        capture: Option<&mut [Vec<f32>]>,
    ) -> Vec<f32> {
        let n = self.config.n_embd;
        let eps = self.config.rms_norm_eps as f32;
        let n_v_heads = self.config.deltanet_linear_n_value_heads;
        let head_dim = self.config.deltanet_linear_head_dim;
        let z_dim = n_v_heads * head_dim;
        let n_k_heads = self.config.deltanet_linear_n_heads;
        let q_dim = n_k_heads * head_dim;
        let qkv_dim = 2 * q_dim + z_dim;
        let conv_dim = qkv_dim;
        let kernel_size = self.config.deltanet_conv_kernel_size;
        let mlp = self.config.mlp_hidden;
        let vocab = self.config.vocab_size;

        self.forward_from_x_capture(
            n,
            eps,
            n_v_heads,
            head_dim,
            z_dim,
            conv_dim,
            kernel_size,
            mlp,
            vocab,
            capture,
        );

        let bytes = self
            .client
            .read_one(self.logits.clone())
            .expect("download logits");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Internal: process all layers starting from the hidden state in `self.x`.
    ///
    /// All dispatches are chained on GPU — no `read_one()` between layers.
    /// Plan 602 B3 — copy-rotate `norm_x` (PRIMAL) into `rot_scratch` (the
    /// folded basis): one `fwht_forward_copy_f32` dispatch (B1 — sign first,
    /// Hadamard second, bit-identical to the CPU `rotate_forward_inplace`
    /// twin). Called after every RMSNorm whose consumers are folded
    /// projections (the layer input norm, the post-attn norm feeding the
    /// FFN, the final norm feeding lm_head). No-op (by construction — never
    /// dispatched) on pre-rotation files.
    fn stage_rotated_norm_x(&self) {
        let n = self.config.n_embd;
        if let (Some(cfg), Some(tables)) = (&self.rotation, &self.rot_tables) {
            let signs = tables.signs_for_width(n).clone();
            unsafe {
                RotationCubeCL::launch_forward_copy::<ActiveRuntime>(
                    &self.client,
                    self.norm_x.clone(),
                    self.rot_scratch.clone(),
                    signs,
                    n,
                    n,
                    cfg.block_size,
                );
            }
        }
    }

    fn forward_from_x(
        &mut self,
        n: usize,
        eps: f32,
        n_v_heads: usize,
        head_dim: usize,
        z_dim: usize,
        conv_dim: usize,
        kernel_size: usize,
        mlp: usize,
        vocab: usize,
    ) {
        // Issue 980 T4-ALT / Plan 602 B3 — the decode funnels CARRY the
        // rotation on folded models now (this is the folded decode lane,
        // mirroring the cudarc eager wiring): every folded projection
        // consumes the rotated staging copy `rot_scratch`, the dense a/b
        // escape set keeps PRIMAL `norm_x`, and the output-side rotations
        // run in place on buffers whose only consumer is the next folded
        // GEMV. The still-refusing paths are prefill / training / the
        // diagnostic capture (see their own guards).
        Self::refuse_if_pool_poisoned("at decode entry");
        self.consume_fresh_input("forward_from_x");
        #[allow(unused_variables)]
        let n_layers = self.layers.len();

        // Arm 13 — decode mutates the CubeCL state handles; the cudarc
        // whole-prefill mirrors are stale until the next `sync_states`.
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            feature = "ternary_gemm_batched",
            feature = "cubecl_runtime",
            not(target_os = "macos")
        ))]
        crate::prefill_cuda_full::notify_cubcl_mutated();

        for (layer_idx, layer_w) in self.layers.iter().enumerate() {
            let is_deltanet = self.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

            // ── Input RMSNorm (or fused with previous layer's FFN residual) ──
            //
            // Issue 645 T3: for layers > 0, the previous layer's FFN residual add
            // is deferred and fused into this layer's input RMSNorm via
            // ResidualAddRmsNorm. This saves 1 dispatch per layer boundary × 63 =
            // 63 dispatches/token. The first layer has no preceding residual, so
            // it uses standalone RMSNorm.
            //
            // When `ternary_gemv_residual` is enabled, the previous layer's down
            // GEMV already applied the residual (fused into the GEMV writing
            // directly to `self.x`), so standalone RMSNorm suffices.
            if layer_idx == 0 || cfg!(feature = "ternary_gemv_residual") {
                // Standalone RMSNorm: first layer (no preceding residual), or
                // ternary_gemv_residual already applied the residual in the down GEMV.
                if decode_stage_on(0) {
                    unsafe {
                        RmsNormCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            self.x.clone(),
                            layer_w.input_norm.clone(),
                            self.norm_x.clone(),
                            n,
                            eps,
                        );
                    }
                }
            } else {
                // Issue 645 T3: fused previous FFN residual + this layer's input RMSNorm.
                // self.ffn_out still holds the previous layer's down GEMV output.
                if decode_stage_on(0) {
                    unsafe {
                        ResidualAddRmsNormCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            self.x.clone(),
                            self.ffn_out.clone(),
                            layer_w.input_norm.clone(),
                            self.norm_x.clone(),
                            n,
                            eps,
                        );
                    }
                }
            }

            // Plan 602 B3 — folded site 2 of 8: the layer's folded input
            // projections (GDN qkv|z, attention q/kv) consume the ROTATED
            // norm; the dense a/b escape set keeps PRIMAL `norm_x`. Staged
            // once here — both layer types read `rot_scratch`.
            if self.rot_tables.is_some() && decode_stage_on(0) {
                self.stage_rotated_norm_x();
            }

            if is_deltanet {
                if decode_stage_on(1) {
                    self.forward_deltanet_layer_gpu(layer_idx, layer_w, n, n_v_heads, head_dim, z_dim, conv_dim, kernel_size, eps);
                }
            } else if decode_stage_on(2) {
                    self.forward_attention_layer_gpu(layer_idx, layer_w, eps);
                }

            // Issue 645 T2: fused ResidualAdd + RMSNorm at the mid-layer boundary.
            // Replaces two separate dispatches (ResidualAdd + RmsNorm) with one.
            // Saves 1 dispatch per layer × 64 = 64 dispatches/token.
            if decode_stage_on(3) {
                unsafe {
                    ResidualAddRmsNormCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        self.x.clone(),
                        self.tmp.clone(),
                        layer_w.post_attn_norm.clone(),
                        self.norm_x.clone(),
                        n,
                        eps,
                    );
                }
                // Plan 602 B3 — folded site 6 of 8: the FFN's folded
                // gate|up concat consumes the ROTATED norm (staged fresh
                // after the post-attn norm; the previous staging is stale).
                if self.rot_tables.is_some() {
                    self.stage_rotated_norm_x();
                }
            }

            // Issue 642 F2: fused gate+up GEMV + concatenated SwiGLU.
            // Single GEMV into ffn_gate_up [2*mlp], then fused SwiGLU reads
            // gate from [0..mlp] and up from [mlp..2*mlp]. Saves 1 dispatch
            // per layer (2 GEMVs → 1 GEMV).
            if decode_stage_on(4) {
                // Plan 602 B3 — folded models read the rotated staging copy;
                // pre-rotation files read `norm_x` directly (unchanged path).
                let ffn_in = if self.rot_tables.is_some() {
                    self.rot_scratch.clone()
                } else {
                    self.norm_x.clone()
                };
                unsafe {
                    GemvTernaryCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        &layer_w.gate_up_proj,
                        ffn_in,
                        self.ffn_gate_up.clone(),
                    );
                }
                unsafe {
                    DeltanetGatingConcatCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        self.ffn_gate_up.clone(),
                        self.ffn_hidden.clone(),
                        mlp,
                    );
                }
                // Plan 602 B3 — folded site 7 of 8: the FFN down projection
                // is folded — rotate the SwiGLU output in place (its only
                // consumer is the down GEMV; the CPU twin rotates
                // `scratch.hidden` at the same site).
                if let (Some(cfg), Some(tables)) = (&self.rotation, &self.rot_tables) {
                    let signs = tables.signs_for_width(mlp).clone();
                    unsafe {
                        RotationCubeCL::launch_forward::<ActiveRuntime>(
                            &self.client,
                            self.ffn_hidden.clone(),
                            signs,
                            mlp,
                            mlp,
                            cfg.block_size,
                        );
                    }
                }
                // FFN down-projection.
                //
                // Issue 616: when `ternary_gemv_residual` is enabled, the down-projection
                // GEMV and the post-FFN ResidualAdd are fused into a single dispatch
                // (`GemvTernaryResidualCubeCL`). The residual lives in `self.x` (the
                // buffer is read for the residual, then overwritten in-place with
                // `x[row] += dot(down_row, ffn_hidden)`). Saves 64 dispatches/token.
                //
                // Without the feature, the down GEMV writes to `self.ffn_out`. The
                // post-FFN ResidualAdd is DEFERRED — it will be fused into the next
                // layer's input RMSNorm (Issue 645 T3) or applied standalone after
                // the loop (for the last layer).
                #[cfg(feature = "ternary_gemv_residual")]
                unsafe {
                    GemvTernaryResidualCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        &layer_w.down_proj,
                        self.ffn_hidden.clone(),
                        self.x.clone(),
                    );
                }
                #[cfg(not(feature = "ternary_gemv_residual"))]
                unsafe {
                    GemvTernaryCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        &layer_w.down_proj,
                        self.ffn_hidden.clone(),
                        self.ffn_out.clone(),
                    );
                }
            }
        }

        // Issue 645 T3: apply the last layer's deferred FFN residual add.
        // When `ternary_gemv_residual` is OFF, the last layer's down GEMV wrote
        // to `self.ffn_out` but the residual add hasn't happened yet (it was
        // deferred for fusion with a next layer that doesn't exist).
        #[cfg(not(feature = "ternary_gemv_residual"))]
        if n_layers > 0 {
            unsafe {
                ResidualAddCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.x.clone(),
                    self.ffn_out.clone(),
                    self.x.clone(),
                    n,
                );
            }
        }

        // Final RMSNorm
        unsafe {
            RmsNormCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x.clone(),
                self.final_norm.clone(),
                self.norm_x.clone(),
                n,
                eps,
            );
        }

        // Plan 602 B3 — folded site 8 of 8: the lm_head is folded; it
        // consumes the ROTATED final norm. Staged into `rot_scratch` so
        // `norm_x` stays PRIMAL — `forward_token_with_final_hidden`'s
        // hidden-state contract is preserved unchanged.
        if self.rot_tables.is_some() {
            self.stage_rotated_norm_x();
        }
        let lm_in = if self.rot_tables.is_some() {
            self.rot_scratch.clone()
        } else {
            self.norm_x.clone()
        };

        // LM head: logits = W_lm_head @ (rotated)? norm_x
        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &self.lm_head,
                lm_in,
                self.logits.clone(),
            );
        }

        let _ = vocab;
        self.pos += 1;
        // Issue 994: decode launches reserve uniform buffers from the same
        // pool — a poisoning mid-decode must not leave as a successful tick.
        Self::refuse_if_pool_poisoned("after decode");
    }

    /// Diagnostic variant of [`forward_from_x`] that optionally reads back
    /// `self.x` after each layer's residual add, forcing a per-layer sync.
    /// Used only for per-layer CPU-vs-GPU comparison (Issue 604 G1 debug).
    fn forward_from_x_capture(
        &mut self,
        n: usize,
        eps: f32,
        n_v_heads: usize,
        head_dim: usize,
        z_dim: usize,
        conv_dim: usize,
        kernel_size: usize,
        mlp: usize,
        vocab: usize,
        mut capture: Option<&mut [Vec<f32>]>,
    ) {
        Self::refuse_if_pool_poisoned("at diagnostic capture entry");
        // Issue 980 T4-ALT — layer-capture decode refuses folded models.
        assert!(
            self.rotation.is_none(),
            "Bonsai-2 folded layer-capture decode: not wired (Issue 980 T4-ALT)"
        );
        self.consume_fresh_input("forward_from_x_capture");
        for (layer_idx, layer_w) in self.layers.iter().enumerate() {
            let is_deltanet = self.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

            unsafe {
                RmsNormCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.x.clone(),
                    layer_w.input_norm.clone(),
                    self.norm_x.clone(),
                    n,
                    eps,
                );
            }

            if is_deltanet {
                self.forward_deltanet_layer_gpu(layer_idx, layer_w, n, n_v_heads, head_dim, z_dim, conv_dim, kernel_size, eps);
            } else {
                self.forward_attention_layer_gpu(layer_idx, layer_w, eps);
            }

            unsafe {
                ResidualAddCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.x.clone(),
                    self.tmp.clone(),
                    self.x.clone(),
                    n,
                );
            }

            unsafe {
                RmsNormCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.x.clone(),
                    layer_w.post_attn_norm.clone(),
                    self.norm_x.clone(),
                    n,
                    eps,
                );
            }

            unsafe {
                GemvTernaryCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    &layer_w.gate_proj,
                    self.norm_x.clone(),
                    self.ffn_gate.clone(),
                );
            }
            unsafe {
                GemvTernaryCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    &layer_w.up_proj,
                    self.norm_x.clone(),
                    self.ffn_up.clone(),
                );
            }
            unsafe {
                DeltanetGatingCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.ffn_gate.clone(),
                    self.ffn_up.clone(),
                    self.ffn_hidden.clone(),
                    mlp,
                );
            }
            // Issue 616: fused FFN down-projection + ResidualAdd (see forward_from_x).
            #[cfg(feature = "ternary_gemv_residual")]
            unsafe {
                GemvTernaryResidualCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    &layer_w.down_proj,
                    self.ffn_hidden.clone(),
                    self.x.clone(),
                );
            }
            #[cfg(not(feature = "ternary_gemv_residual"))]
            {
                unsafe {
                    GemvTernaryCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        &layer_w.down_proj,
                        self.ffn_hidden.clone(),
                        self.ffn_out.clone(),
                    );
                }

                unsafe {
                    ResidualAddCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        self.x.clone(),
                        self.ffn_out.clone(),
                        self.x.clone(),
                        n,
                    );
                }
            }

            // ── Per-layer capture tap (forces GPU sync — diagnostic only) ──
            // The sync here is what makes the diagnostic useful: it serializes
            // all GPU dispatches up to this point, ensuring the captured hidden
            // state reflects the true post-layer value. Without it, async
            // dispatch chaining could return stale data.
            if let Some(buf) = capture.as_deref_mut()
                && let Some(slot) = buf.get_mut(layer_idx)
            {
                let bytes = self
                    .client
                    .read_one(self.x.clone())
                    .expect("capture readback");
                let xs = f32::from_bytes(&bytes);
                slot.clear();
                slot.extend_from_slice(&xs[..n]);
            }
        }

        // Final RMSNorm
        unsafe {
            RmsNormCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.x.clone(),
                self.final_norm.clone(),
                self.norm_x.clone(),
                n,
                eps,
            );
        }

        // LM head: logits = W_lm_head @ norm_x
        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &self.lm_head,
                self.norm_x.clone(),
                self.logits.clone(),
            );
        }

        let _ = vocab;
        self.pos += 1;
    }
    ///
    /// All operations chain without CPU sync:
    /// input_proj → conv1d → beta/decay → expand+L2-norm → recurrence → RMSNorm → z_gating → out_proj
    #[allow(clippy::too_many_arguments)]
    fn forward_deltanet_layer_gpu(
        &self,
        layer_idx: usize,
        layer_w: &GpuLayerWeights,
        _n: usize,
        n_v_heads: usize,
        head_dim: usize,
        _z_dim: usize,
        conv_dim: usize,
        kernel_size: usize,
        eps: f32,
    ) {
        // 1-4. Fused input projections: qkv+z+a+b in a single GEMV, then split
        // (Issue 642 F3). All 4 share input = norm_x. Single concatenated GEMV
        // writes to input_proj_out, then Split4CubeCL copies regions into
        // qkv, z_buf, a_raw, b_raw. Saves 2 dispatches per DeltaNet layer
        // (4 GEMVs → 1 GEMV + 1 split).
        //
        // Plan 602 B2/B3 — folded site 3 of 8 (the GDN input split): on a
        // folded model the concat carries qkv|z ONLY (uploaded that way) and
        // consumes the ROTATED staging copy; the dense escape-set a/b are
        // neither rotated nor folded (whitepaper A.2) and dispatch as two f32
        // GEMVs on the PRIMAL `norm_x` — the `gemv_dense` cudarc twin.
        let z_dim = n_v_heads * head_dim;
        let n_k_heads = self.config.deltanet_linear_n_heads;
        let folded = self.rot_tables.is_some();
        if folded {
            unsafe {
                GemvTernaryCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    &layer_w.in_proj_concat,
                    self.rot_scratch.clone(),
                    self.input_proj_out.clone(),
                );
                let a_f32 = layer_w.in_proj_a_f32.as_ref().expect(
                    "folded GDN layer: dense ssm_alpha missing (loader contract guarantees the escape set)",
                );
                let b_f32 = layer_w.in_proj_b_f32.as_ref().expect(
                    "folded GDN layer: dense ssm_beta missing (loader contract guarantees the escape set)",
                );
                GemvCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    a_f32.clone(),
                    self.norm_x.clone(),
                    self.a_raw.clone(),
                    n_v_heads,
                    self.config.n_embd,
                );
                GemvCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    b_f32.clone(),
                    self.norm_x.clone(),
                    self.b_raw.clone(),
                    n_v_heads,
                    self.config.n_embd,
                );
                Split2CubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.input_proj_out.clone(),
                    self.qkv.clone(),
                    self.z_buf.clone(),
                    conv_dim,
                    z_dim,
                );
            }
        } else {
            unsafe {
                GemvTernaryCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    &layer_w.in_proj_concat,
                    self.norm_x.clone(),
                    self.input_proj_out.clone(),
                );
            }
        }

        let conv_state = self.conv_states[layer_idx].as_ref().expect("conv_state for DeltaNet layer");

        // Issue 764 T2 (GDN fusion lane): when the runtime toggle is on and
        // the geometry fits, ONE fused dispatch replaces the four below
        // (Split4 + Conv1d + BetaDecay + ExpandAndL2) — −3 launches per GDN
        // layer (−144/token on Bonsai), bit-identical outputs. Default OFF;
        // the shipping 4-dispatch chain below is the fallback (and the G1 oracle).
        // Plan 602 B3: folded models NEVER take the fusion — its input layout
        // expects the 4-region qkv|z|a|b concat, which folded models do not
        // produce (Split2 already staged qkv/z; a/b arrived via the dense
        // GEMVs).
        let fused_pre_rec = !folded
            && crate::deltanet_pre_rec_fused_cubecl::deltanet_fused_pre_rec_enabled()
            && crate::deltanet_pre_rec_fused_cubecl::DeltanetPreRecFusedCubeCL::supports(
                n_k_heads,
                n_v_heads,
                head_dim,
                kernel_size,
            );

        if fused_pre_rec {
            unsafe {
                crate::deltanet_pre_rec_fused_cubecl::DeltanetPreRecFusedCubeCL::launch::<
                    ActiveRuntime,
                >(
                    &self.client,
                    self.input_proj_out.clone(),
                    layer_w.conv1d_weight.clone(),
                    conv_state.clone(),
                    layer_w.a_log.clone(),
                    layer_w.dt_bias.clone(),
                    self.z_buf.clone(),
                    self.beta_buf.clone(),
                    self.decay_buf.clone(),
                    self.qkv_expanded.clone(),
                    n_k_heads,
                    n_v_heads,
                    head_dim,
                    kernel_size,
                );
            }
        } else {
        // Plan 602 B3 — the pre-folded path already staged qkv/z via
        // Split2CubeCL above; Split4 runs on the 4-region concat only.
        if !folded {
            unsafe {
                Split4CubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.input_proj_out.clone(),
                    self.qkv.clone(),       // [0..conv_dim]
                    self.z_buf.clone(),     // [conv_dim..conv_dim+z_dim]
                    self.a_raw.clone(),     // [..+n_v_heads]
                    self.b_raw.clone(),     // [..+n_v_heads]
                    conv_dim,
                    z_dim,
                    n_v_heads,
                    n_v_heads,
                );
            }
        }

        // 5. Conv1D (in-place on qkv, updates conv_state)
        unsafe {
            DeltanetConv1dCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.qkv.clone(),
                layer_w.conv1d_weight.clone(),
                conv_state.clone(),
                conv_dim,
                kernel_size,
            );
        }

        // 6. Beta/decay computation
        unsafe {
            DeltanetBetaDecayCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.a_raw.clone(),
                self.b_raw.clone(),
                layer_w.a_log.clone(),
                layer_w.dt_bias.clone(),
                self.beta_buf.clone(),
                self.decay_buf.clone(),
                n_v_heads,
            );
        }

        // 7. Expand Q/K from n_k_heads → n_v_heads + L2-normalize Q/K heads + copy V.
        //    Reads compact `qkv` ([Q(n_k×hd) | K(n_k×hd) | V(n_v×hd)]), writes
        //    expanded `qkv_expanded` ([Q(n_v×hd) | K(n_v×hd) | V(n_v×hd)]).
        //    This fused kernel replaces the (buggy) in-place L2NormalizeHeadsCubeCL
        //    that assumed Q/K were already n_v_heads wide — Issue 604 T5 + Issue 610.
        unsafe {
            ExpandAndL2NormalizeHeadsCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.qkv.clone(),
                self.qkv_expanded.clone(),
                n_k_heads,
                n_v_heads,
                head_dim,
            );
        }
        }

        // 8. Recurrence: update state, read output
        let state = self.deltanet_states[layer_idx].as_ref().expect("state for DeltaNet layer");
        // Beta/decay are already on GPU (beta_buf/decay_buf) — pass handles directly.
        // No CPU sync point.
        // Reads from qkv_expanded (post head-expansion + L2-norm).
        // Issue 619: prefer the row-parallel, register-blocked kernel when the
        // model's head_dim fits its fixed register unroll (128). Otherwise fall
        // back to the legacy serial-row kernel, which handles any head_dim.
        #[allow(unused_mut, reason = "only mutated when deltanet_recurrence_rowpar is enabled")]
        let mut dispatched = false;

        #[cfg(feature = "deltanet_recurrence_rowpar")]
        if recurrence_rowpar_enabled() && DeltanetRecurrenceRowParCubeCL::supports(head_dim) {
            unsafe {
                DeltanetRecurrenceRowParCubeCL::launch_with_gpu_handles::<ActiveRuntime>(
                    &self.client,
                    self.qkv_expanded.clone(),
                    self.beta_buf.clone(),
                    self.decay_buf.clone(),
                    state.clone(),
                    self.recurrent_out.clone(),
                    n_v_heads,
                    head_dim,
                );
            }
            dispatched = true;
        }

        if !dispatched {
            unsafe {
                DeltanetRecurrenceCubeCL::launch_with_gpu_handles::<ActiveRuntime>(
                    &self.client,
                    self.qkv_expanded.clone(),
                    self.beta_buf.clone(),
                    self.decay_buf.clone(),
                    state.clone(),
                    self.recurrent_out.clone(),
                    n_v_heads,
                    head_dim,
                );
            }
        }

        // 9+10. Fused per-head RMSNorm + z-gating (Issue 642 F6).
        //    Replaces two separate dispatches (RmsNormBatched + ZGating) with
        //    one fused kernel. Normalizes each head of recurrent_out with
        //    linear_norm gamma, then multiplies by silu(z_buf) — all in-place.
        //    Saves 1 dispatch per DeltaNet layer × 32 = 32 dispatches/token.
        unsafe {
            RmsNormZgateFusedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.recurrent_out.clone(),
                layer_w.linear_norm.clone(),
                self.z_buf.clone(),
                n_v_heads,
                head_dim,
                eps,
            );
        }

        // 11. Out projection: tmp = W_out @ recurrent_out
        //
        // Plan 602 B3 — folded site 4 of 8 (the ssm_out chain): on a folded
        // layer the recurrence output arrives in tiled [hd, nk, rep] head
        // order (`gdn_v_grouped`) and the fold expects the grouped
        // [hd, rep, nk] order — permute via the staged copy (the CPU twin
        // stages into `rotation_buf`; `permute_tmp` here), then the shared
        // sign+FWHT forward rotation runs IN PLACE on `recurrent_out` (its
        // only consumer is the folded out_proj GEMV). The per-head norm +
        // SiLU(z) gate above already ran — they are fold-invariant (they
        // commute with the head-order permute), the CPU step-10/11 order.
        if let (Some(cfg), Some(tables)) = (&self.rotation, &self.rot_tables) {
            let v_dim = n_v_heads * head_dim;
            if cfg.gdn_v_grouped {
                unsafe {
                    CopyCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        self.recurrent_out.clone(),
                        self.permute_tmp.clone(),
                        v_dim,
                    );
                    RotationCubeCL::launch_gdn_v_permute::<ActiveRuntime>(
                        &self.client,
                        self.permute_tmp.clone(),
                        self.recurrent_out.clone(),
                        v_dim,
                        v_dim,
                        head_dim,
                        cfg.gdn_k_groups,
                        n_v_heads / cfg.gdn_k_groups,
                    );
                }
            }
            let signs = tables.signs_for_width(v_dim).clone();
            unsafe {
                RotationCubeCL::launch_forward::<ActiveRuntime>(
                    &self.client,
                    self.recurrent_out.clone(),
                    signs,
                    v_dim,
                    v_dim,
                    cfg.block_size,
                );
            }
        }
        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &layer_w.out_proj,
                self.recurrent_out.clone(),
                self.tmp.clone(),
            );
        }
    }

    /// GPU-resident forward for a full-attention layer (Qwen3.5 gated attention).
    ///
    /// Implements:
    /// 1. Q (gated), K, V projections (ternary GEMV)
    /// 2. Split QG into Q and gate
    /// 3. Q/K per-head RMSNorm
    /// 4. Partial RoPE
    /// 5. KV cache append
    /// 6. Flash attention decode
    /// 7. Output gating (sigmoid)
    /// 8. Output projection (ternary GEMV) → writes to tmp
    ///
    /// `pub(crate)` for the in-crate prefill/decode variants. (The Issue 721
    /// per-branch tree-verify bridge — its former second consumer — was
    /// replaced by the ancestor-masked batched attention in
    /// `ternary_tree_verify_driver`.)
    pub(crate) fn forward_attention_layer_gpu(
        &self,
        layer_idx: usize,
        layer_w: &GpuLayerWeights,
        eps: f32,
    ) {
        let n_head = self.config.n_head;
        let n_kv = self.config.n_kv_head;
        let hd = self.config.head_dim;
        let q_dim = n_head * hd;
        let kvd = n_kv * hd;
        let rotary_dim = if self.config.rope_dimension_count > 0 {
            self.config.rope_dimension_count
        } else {
            hd // full rotation
        };
        let theta_base = self.config.rope_theta;
        let pos = self.pos;

        // ── Bench 642 probe 1: per-kernel tap (forces GPU syncs) ──
        #[cfg(feature = "ternary_gemm_batched")]
        let mut tap: Option<AttnScratch> = ATTN_TAP_LAYER.with(|c| c.get())
            .eq(&(layer_idx as i64))
            .then(|| ATTN_TAP.with(|t| t.borrow().is_none()))
            .and_then(|first| first.then(AttnScratch::default));

        // 1. Q (gated) projection: qg = W_q @ norm_x
        //    attn_wq produces [2*q_dim] interleaved [q(hd), gate(hd)] per head
        //
        // Plan 602 B3 — folded site 5 of 8: the attention projections are
        // folded — both GEMVs consume the ROTATED staging copy (attention
        // layers carry no a/b escape set, so the primal copy has no consumer
        // here; the cudarc twin stages the same rotated input).
        let folded = self.rot_tables.is_some();
        let attn_gemv_in = if folded {
            self.rot_scratch.clone()
        } else {
            self.norm_x.clone()
        };
        if attn_sub_on(0) {
        let wq = layer_w.attn_wq.as_ref().expect("attn_wq for Attention layer");
        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                &self.client,
                wq,
                attn_gemv_in.clone(),
                self.attn_qg.clone(),
            );
        }
        // Issue 648 F9: fused K+V projection via concatenated weights.
        // Single GEMV writes K to attn_kv[0..kvd] and V to attn_kv[kvd..2*kvd],
        // replacing two separate GEMV dispatches with one. Downstream kernels
        // (RMSNorm, RoPE) consume K from offset 0 unchanged; the KV cache
        // append uses the combined variant.
        let wkv = layer_w.attn_wkv.as_ref().expect("attn_wkv for Attention layer");
        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                &self.client,
                wkv,
                attn_gemv_in,
                self.attn_kv.clone(),
            );
        }
        } // sub-stage 0: projections

        #[cfg(feature = "ternary_gemm_batched")]
        if let Some(t) = tap.as_mut() {
            t.norm_x_in = self.tap_read(&self.norm_x, self.config.n_embd);
            t.qg_proj = self.tap_read(&self.attn_qg, 2 * q_dim);
            t.k_proj = self.tap_read(&self.attn_kv, kvd);
            t.v_proj = {
                // V is at offset kvd in the combined buffer
                let bytes = self.client.read_one(self.attn_kv.clone()).expect("tap readback");
                let all = f32::from_bytes(&bytes);
                all[kvd..kvd + kvd].to_vec()
            };
        }

        // 2. Split QG into Q and gate
        // 3. Q/K per-head RMSNorm (fused)
        // 4. Partial RoPE on Q and K
        if attn_sub_on(1) {
        unsafe {
            QwenSplitQgCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.attn_qg.clone(),
                self.attn_q.clone(),
                self.attn_gate.clone(),
                hd,
                n_head,
            );
        }

        //    Issue 648 F9: K is read from attn_kv[0..kvd] (first half of the
        //    combined buffer) — the kernel indexes k[head_idx*hd + tid] for
        //    head_idx in [0..n_kv), so offset 0 is correct.
        let q_norm = layer_w.attn_q_norm.as_ref().expect("attn_q_norm for Attention layer");
        let k_norm = layer_w.attn_k_norm.as_ref().expect("attn_k_norm for Attention layer");
        unsafe {
            RmsNormQkFusedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.attn_q.clone(),
                self.attn_kv.clone(),
                q_norm.clone(),
                k_norm.clone(),
                n_head,
                n_kv,
                hd,
                eps,
            );
        }

        //    Issue 648 F9: K is in attn_kv[0..kvd] — passed unchanged.
        unsafe {
            QwenRopePartialCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.attn_q.clone(),
                self.attn_kv.clone(),
                pos,
                rotary_dim,
                hd,
                n_head,
                n_kv,
                theta_base,
            );
        }
        } // sub-stage 1: elementwise (split + norm + rope)

        #[cfg(feature = "ternary_gemm_batched")]
        if let Some(t) = tap.as_mut() {
            t.q_rope = self.tap_read(&self.attn_q, q_dim);
            t.k_rope = self.tap_read(&self.attn_kv, kvd);
        }

        // 5. Append K, V to KV cache at position `pos`
        //    Issue 648 F9: use combined KV cache append — reads K from
        //    attn_kv[0..kvd] and V from attn_kv[kvd..2*kvd].
        let key_cache = self.kv_key_caches[layer_idx]
            .as_ref()
            .expect("key cache for Attention layer");
        let value_cache = self.kv_value_caches[layer_idx]
            .as_ref()
            .expect("value cache for Attention layer");
        if attn_sub_on(2) {
        unsafe {
            QwenKvCacheAppendCombinedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.attn_kv.clone(),
                key_cache.clone(),
                value_cache.clone(),
                kvd,
                pos,
            );
        }
        } // sub-stage 2: kv_append

        // 6+7. Issue 648 F10: fused flash attention decode + output gate.
        //      Replaces separate QwenAttentionDecodeCubeCL + QwenOutputGateCubeCL
        //      with a single QwenAttentionDecodeGatedCubeCL dispatch. The sigmoid
        //      gate is applied at the final write inside the decode kernel.
        let n_positions = pos + 1;
        if attn_sub_on(3) {
        // Issue 831 (a): length-gated split-K flash decode. Short contexts run
        // the single-workgroup kernel bit-identically (the tile loop degenerates
        // there); long contexts partition positions across n_splits workgroups
        // per head — the measured 15.8 ms/token @2K serial-scan cost is the
        // target (96% of the context penalty; see the kernel header).
        let (use_split, n_splits, split_len) = if attn_split_decode_enabled()
            && n_positions > attn_split_decode_min_pos()
        {
            let (s, l) = split_decode_geometry(
                n_positions,
                hd,
                ATTN_SPLIT_DECODE_MAX_SPLITS_DEFAULT,
            );
            (true, s, l)
        } else {
            (false, 1, 0)
        };
        if use_split {
            unsafe {
                QwenAttentionDecodeGatedSplitCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.attn_q.clone(),
                    key_cache.clone(),
                    value_cache.clone(),
                    self.attn_split_partials.clone(),
                    hd,
                    n_head,
                    n_kv,
                    n_positions,
                    n_splits,
                    split_len,
                );
            }
            unsafe {
                QwenAttentionDecodeGatedCombineCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.attn_split_partials.clone(),
                    self.attn_gate.clone(),
                    self.attn_out.clone(),
                    hd,
                    n_head,
                    n_splits,
                );
            }
        } else {
        unsafe {
            QwenAttentionDecodeGatedCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.attn_q.clone(),
                key_cache.clone(),
                value_cache.clone(),
                self.attn_gate.clone(),
                self.attn_out.clone(),
                hd,
                n_head,
                n_kv,
                n_positions,
            );
        }
        }
        } // sub-stage 3: attn_decode

        #[cfg(feature = "ternary_gemm_batched")]
        if let Some(t) = tap.as_mut() {
            t.attn_out = self.tap_read(&self.attn_out, q_dim);
            t.n_positions = n_positions;
        }

        // 8. Output projection: tmp = W_o @ attn_out
        //
        // Plan 602 B3 — folded site 5b: `wo` is folded; the GATED attention
        // output (the decode kernel applies the sigmoid gate at its final
        // write — the primal pairing, CPU step order) rotates IN PLACE before
        // the projection. `attn_out`'s only consumer is `wo`.
        if attn_sub_on(4) {
        if let (Some(cfg), Some(tables)) = (&self.rotation, &self.rot_tables) {
            let signs = tables.signs_for_width(q_dim).clone();
            unsafe {
                RotationCubeCL::launch_forward::<ActiveRuntime>(
                    &self.client,
                    self.attn_out.clone(),
                    signs,
                    q_dim,
                    q_dim,
                    cfg.block_size,
                );
            }
        }
        let wo = layer_w.attn_wo.as_ref().expect("attn_wo for Attention layer");
        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                &self.client,
                wo,
                self.attn_out.clone(),
                self.tmp.clone(),
            );
        }
        } // sub-stage 4: out_proj

        #[cfg(feature = "ternary_gemm_batched")]
        if let Some(mut t) = tap {
            t.out_proj = self.tap_read(&self.tmp, self.config.n_embd);
            ATTN_TAP.with(|c| *c.borrow_mut() = Some(t));
        }
    }

    /// Issue 771 T2c-a / Plan 562: lazily allocate (once) and return the
    /// shared Q8-KV prefill scratch. Sized for `block_size` rows at the
    /// instance's fixed KV geometry; reused across layers within a forward.
    #[cfg(all(
        feature = "cubecl_runtime",
        feature = "ternary_gemm_batched",
        feature = "ternary_attention_batched_prefill",
    ))]
    fn q8_prefill_scratch(&self) -> &crate::qwen_prefill_q8kv_cubecl::Q8PrefillScratch {
        self.q8_prefill_scratch.get_or_init(|| {
            let n_kv = self.config.n_kv_head;
            let hd = self.config.head_dim;
            let rows = self.config.block_size;
            let (qs_bytes, scales_bytes) =
                crate::qwen_prefill_q8kv_cubecl::Q8PrefillScratch::buffer_bytes(rows, n_kv, hd);
            crate::qwen_prefill_q8kv_cubecl::Q8PrefillScratch {
                key_qs: self.client.empty(qs_bytes),
                key_scales: self.client.empty(scales_bytes),
                value_qs: self.client.empty(qs_bytes),
                value_scales: self.client.empty(scales_bytes),
                rows,
            }
        })
    }

    /// Issue 653: batched attention layer forward for prefill.
    ///
    /// Processes all P tokens through one attention layer in a single set of
    /// batched dispatches, replacing the P sequential `forward_attention_layer_gpu`
    /// calls. The key new kernel is `QwenAttentionPrefillGatedCubeCL` — a
    /// causal-masked flash attention over P query tokens.
    ///
    /// Pipeline (all batched over P tokens):
    /// 1. Input RMSNorm (batched) → normx_b
    /// 2. Q projection (batched GEMM) → qg_b
    /// 3. KV projection (batched GEMM) → kv_b
    /// 4. Split QG → Q + gate (batched)
    /// 5. Split KV → K + V (batched)
    /// 6. Q/K RMSNorm (fused, P*n_head + P*n_kv workgroups)
    /// 7. RoPE (batched, per-token position)
    /// 8. KV cache fill (batched — writes positions 0..P)
    /// 9. Causal flash attention (batched, fused output gate)
    /// 10. Output projection (batched GEMM)
    /// 11. Residual add (batched)
    #[cfg(all(
        feature = "cubecl_runtime",
        feature = "ternary_gemm_batched",
        feature = "ternary_attention_batched_prefill",
    ))]
    #[allow(clippy::too_many_arguments, reason = "prefill wiring")]
    fn prefill_attention_layer_batched(
        &self,
        layer_idx: usize,
        eps: f32,
        p: usize,
        base_pos: usize,
        x_b: &Handle,
        normx_b: &Handle,
        residual_out: &Handle,
        scratch: &BatchedAttentionScratch,
    ) {
        let n_head = self.config.n_head;
        let n_kv = self.config.n_kv_head;
        let hd = self.config.head_dim;
        let n = self.config.n_embd;
        let kvd = n_kv * hd;
        let rotary_dim = if self.config.rope_dimension_count > 0 {
            self.config.rope_dimension_count
        } else {
            hd
        };
        let theta_base = self.config.rope_theta;
        let layer_w = &self.layers[layer_idx];

        // 1. Input RMSNorm (batched over P tokens). [sub-stage 5: outer_norm]
        if attn_sub_on(5) {
            Self::prefill_norm(
                &self.client,
                x_b,
                &layer_w.input_norm,
                normx_b,
                p,
                n,
                eps,
            );
        }

        // 2. Q projection (batched GEMM) — attn_wq writes [2*q_dim] per token.
        // 3. KV projection (batched GEMM) — attn_wkv writes [2*kvd] per token.
        // [sub-stage 0: proj]
        if attn_sub_on(0) {
            let wq = layer_w.attn_wq.as_ref().expect("attn_wq for Attention layer");
            self.prefill_project(wq, normx_b, &scratch.qg_b, p);
            let wkv = layer_w
                .attn_wkv
                .as_ref()
                .expect("attn_wkv for Attention layer");
            self.prefill_project(wkv, normx_b, &scratch.kv_b, p);
        }

        // 4. Split QG → Q + gate (batched).
        // 5. Split KV → K + V (batched).
        // 6. Q/K RMSNorm (fused). Pass P*n_head + P*n_kv as the head counts —
        //    q_b is [P, n_head, hd] contiguous, k_b is [P, n_kv, hd] contiguous.
        //    The kernel treats each (token, head) pair as a separate row.
        // 7. RoPE (batched) — each token t uses position `base_pos + t`
        //    (Issue 734 chunked prefill: absolute positions).
        // [sub-stage 1: elementwise]
        if attn_sub_on(1) {
            unsafe {
                QwenSplitQgBatchedCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    scratch.qg_b.clone(),
                    scratch.q_b.clone(),
                    scratch.gate_b.clone(),
                    hd,
                    n_head,
                    p,
                );
            }
            unsafe {
                QwenSplitKvBatchedCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    scratch.kv_b.clone(),
                    scratch.k_b.clone(),
                    scratch.v_b.clone(),
                    kvd,
                    p,
                );
            }
            let q_norm = layer_w.attn_q_norm.as_ref().expect("attn_q_norm");
            let k_norm = layer_w.attn_k_norm.as_ref().expect("attn_k_norm");
            unsafe {
                RmsNormQkFusedCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    scratch.q_b.clone(),
                    scratch.k_b.clone(),
                    q_norm.clone(),
                    k_norm.clone(),
                    p * n_head,
                    p * n_kv,
                    hd,
                    eps,
                );
            }
            unsafe {
                QwenRopePartialBatchedCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    scratch.q_b.clone(),
                    scratch.k_b.clone(),
                    rotary_dim,
                    hd,
                    n_head,
                    n_kv,
                    theta_base,
                    p,
                    base_pos,
                );
            }
        }

        // 8. KV cache fill — write post-RoPE K and raw V to cache positions
        //    `base_pos..base_pos+P` for decode continuity + LATER CHUNKS'
        //    attention (chunked prefill reads K/V from the cache). K comes
        //    from k_b (post-RMSNorm + RoPE), V comes from v_b (split,
        //    unmodified). This matches what the sequential decode path
        //    writes to the cache.
        let key_cache = self.kv_key_caches[layer_idx]
            .as_ref()
            .expect("key cache for Attention layer");
        let value_cache = self.kv_value_caches[layer_idx]
            .as_ref()
            .expect("value cache for Attention layer");
        // [sub-stage 2: kv_append]
        if attn_sub_on(2) {
            unsafe {
                QwenKvCacheFillSplitBatchedCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    scratch.k_b.clone(),
                    scratch.v_b.clone(),
                    key_cache.clone(),
                    value_cache.clone(),
                    kvd,
                    p,
                    base_pos,
                );
            }
        }

        // 9. Causal flash attention (batched) — processes all P query tokens.
        //    Issue 734 chunked prefill: for base_pos > 0 the K/V of previous
        //    chunks live ONLY in the cache, so the kernel reads K/V from the
        //    CACHE handles (they hold positions 0..base_pos+p after the fill
        //    above — same stream, in-order) with the absolute-position causal
        //    range 0..=base_pos+t. base_pos == 0 keeps the original scratch
        //    K/V read (single-chunk path, bit-compatible with the measured
        //    ≤4K numbers — the cache holds byte-identical copies, so the
        //    values are the same either way; the split exists only to keep
        //    the single-chunk path byte-for-byte on the pre-change dispatch
        //    pattern).
        // [sub-stage 3: attn_decode]
        if attn_sub_on(3) {
            let (kv_read, vv_read) = if base_pos > 0 {
                (key_cache.clone(), value_cache.clone())
            } else {
                (scratch.k_b.clone(), scratch.v_b.clone())
            };
            // Issue 771 T2b: the plane-per-query tiled kernel (tolerance arm,
            // DEFAULT ON since Bench 805 — kill-switch env
            // `RIIR_PREFILL_TILED_FLASH=0` restores the legacy kernel
            // bit-identically). Issue 771 / Bench 808: the M=16
            // two-rows-per-plane arm takes precedence for LONG prefills
            // (DEFAULT-ON, `p >= TILED_FLASH_M16_MIN_P`; kill-switch env
            // `RIIR_PREFILL_TILED_FLASH_M16=0`; bit-identical to the tiled
            // kernel by construction — the promotion moves no anchor).
            // Falls back to the legacy per-(head, token) kernel for any
            // head_dim != 256.
            let use_q8kv = prefill_q8kv_enabled()
                && hd == 256
                && self.active_prefill_len >= PREFILL_Q8KV_MIN_P;
            let use_cmma_pv = prefill_tiled_flash_cmma_pv_enabled()
                && hd == 256
                // Issue 828 shape guard: both cmma arms instantiate f32 8×8×8
                // fragments — without the device config the JIT lookup fails
                // at first dispatch, so route through the fallback chain
                // instead (defense in depth beyond the macOS target gate;
                // makes the env toggles safe on ANY backend).
                && QwenAttentionPrefillTiledCmmaCubeCL::cmma_available::<ActiveRuntime>(&self.client);
            let use_cmma = prefill_tiled_flash_cmma_enabled()
                && hd == 256
                && self.active_prefill_len >= TILED_FLASH_CMMA_MIN_P
                && QwenAttentionPrefillTiledCmmaCubeCL::cmma_available::<ActiveRuntime>(&self.client);
            let use_m16 = prefill_tiled_flash_m16_enabled()
                && hd == 256
                && self.active_prefill_len >= TILED_FLASH_M16_MIN_P;
            // Issue 844 T2: the m32 arm takes precedence over m16 —
            // DEFAULT-ON macOS (the 2026-09-02 Bench-845 cooled long-pass
            // promotion: 1.037× e2e median @16K, 3/3 rounds ≥1.0×, FNV
            // identical; the 0.929× back-to-back FAIL was a thermal-order
            // artifact). Non-macOS keeps m16 via the env/setter; kill-switch
            // `RIIR_PREFILL_TILED_FLASH_M32=0` restores m16 everywhere.
            let use_m32 = prefill_tiled_flash_m32_enabled()
                && hd == 256
                && self.active_prefill_len >= TILED_FLASH_M32_MIN_P;
            // Issue 844 T5: the m64 octet arm takes precedence over m32 —
            // OPT-IN ONLY (`RIIR_PREFILL_TILED_FLASH_M64=1`; default OFF on
            // every platform — the m32 sustained-load lesson applies a
            // fortiori; the promotion ladder in the module doc is
            // pre-registered, not yet run).
            let use_m64 = prefill_tiled_flash_m64_enabled()
                && hd == 256
                && self.active_prefill_len >= TILED_FLASH_M64_MIN_P;
            // Issue 844 T5 residue: the K/V-pipelined m32 arm — OPT-IN ONLY
            // (`RIIR_PREFILL_TILED_FLASH_M32_PIPE=1`; default OFF on every
            // platform). Same Q-block shape as m32, so it takes precedence
            // over plain m32 only; m64 stays the widest-shape arm above it.
            let use_m32_pipe = prefill_tiled_flash_m32_pipe_enabled()
                && hd == 256
                && self.active_prefill_len >= TILED_FLASH_M32_PIPE_MIN_P;
            let use_tiled = prefill_tiled_flash_enabled() && hd == 256;
            unsafe {
                if use_q8kv {
                    // Issue 771 T2c-a / Plan 562: the Q8-KV prefill flash arm
                    // (DEFAULT-OFF tolerance arm; env `RIIR_Q8KV_PREFILL=1`
                    // or the setter). Quantize the full [0, base_pos+p) cache
                    // range once per layer call, then run the tiled flash
                    // kernel with in-register Q8 dequant. Prompt-level gate
                    // (`active_prefill_len`) — the Issue 782 lesson.
                    note_q8kv_prefill_launch();
                    let q8 = self.q8_prefill_scratch();
                    let rows = base_pos + p;
                    debug_assert!(
                        rows <= q8.rows,
                        "q8 scratch sized {} rows but prefill needs {rows}",
                        q8.rows
                    );
                    // Same cache-vs-scratch source rule as the f32 chain
                    // below (base_pos == 0 keeps the scratch read — sub-stage
                    // diagnostics can leave the cache stale).
                    let (src_k, src_v) = if base_pos > 0 {
                        (key_cache.clone(), value_cache.clone())
                    } else {
                        (scratch.k_b.clone(), scratch.v_b.clone())
                    };
                    launch_kv_quantize_q8::<ActiveRuntime>(
                        &self.client,
                        src_k,
                        q8.key_qs.clone(),
                        q8.key_scales.clone(),
                        rows,
                        n_kv,
                        hd,
                    );
                    launch_kv_quantize_q8::<ActiveRuntime>(
                        &self.client,
                        src_v,
                        q8.value_qs.clone(),
                        q8.value_scales.clone(),
                        rows,
                        n_kv,
                        hd,
                    );
                    QwenAttentionPrefillTiledQ8CubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        scratch.q_b.clone(),
                        q8.key_qs.clone(),
                        q8.key_scales.clone(),
                        q8.value_qs.clone(),
                        q8.value_scales.clone(),
                        scratch.gate_b.clone(),
                        scratch.attn_out_b.clone(),
                        hd,
                        n_head,
                        n_kv,
                        p,
                        base_pos,
                    );
                } else if use_cmma_pv {
                    // Issue 771 / Bench 810: the PV-cmma successor arm
                    // (DEFAULT-OFF tolerance arm; env
                    // `RIIR_PREFILL_TILED_FLASH_CMMA_PV=1` or
                    // the setter).
                    note_tiled_flash_cmma_pv_launch();
                    QwenAttentionPrefillTiledCmmaPvCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        scratch.q_b.clone(),
                        kv_read,
                        vv_read,
                        scratch.gate_b.clone(),
                        scratch.attn_out_b.clone(),
                        hd,
                        n_head,
                        n_kv,
                        p,
                        base_pos,
                    );
                } else if use_cmma {
                    // Issue 771 / Bench 809: the cmma score-matrix arm
                    // (OPT-IN since the 2026-09-02 demotion — Bench 841's
                    // cooled tri-cell measured m16 −8.2% on the attention
                    // stage @16K twice, e2e confirming, refuting the 782
                    // direction; opt-in env `RIIR_PREFILL_TILED_FLASH_CMMA=1`;
                    // routes at `p >= TILED_FLASH_CMMA_MIN_P` when enabled).
                    note_tiled_flash_cmma_launch();
                    QwenAttentionPrefillTiledCmmaCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        scratch.q_b.clone(),
                        kv_read,
                        vv_read,
                        scratch.gate_b.clone(),
                        scratch.attn_out_b.clone(),
                        hd,
                        n_head,
                        n_kv,
                        p,
                        base_pos,
                    );
                } else if use_m64 {
                    note_tiled_flash_m64_launch();
                    QwenAttentionPrefillTiledM64CubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        scratch.q_b.clone(),
                        kv_read,
                        vv_read,
                        scratch.gate_b.clone(),
                        scratch.attn_out_b.clone(),
                        hd,
                        n_head,
                        n_kv,
                        p,
                        base_pos,
                    );
                } else if use_m32_pipe {
                    note_tiled_flash_m32_pipe_launch();
                    QwenAttentionPrefillTiledM32PipeCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        scratch.q_b.clone(),
                        kv_read,
                        vv_read,
                        scratch.gate_b.clone(),
                        scratch.attn_out_b.clone(),
                        hd,
                        n_head,
                        n_kv,
                        p,
                        base_pos,
                    );
                } else if use_m32 {
                    note_tiled_flash_m32_launch();
                    QwenAttentionPrefillTiledM32CubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        scratch.q_b.clone(),
                        kv_read,
                        vv_read,
                        scratch.gate_b.clone(),
                        scratch.attn_out_b.clone(),
                        hd,
                        n_head,
                        n_kv,
                        p,
                        base_pos,
                    );
                } else if use_m16 {
                    note_tiled_flash_m16_launch();
                    QwenAttentionPrefillTiledM16CubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        scratch.q_b.clone(),
                        kv_read,
                        vv_read,
                        scratch.gate_b.clone(),
                        scratch.attn_out_b.clone(),
                        hd,
                        n_head,
                        n_kv,
                        p,
                        base_pos,
                    );
                } else if use_tiled {
                    note_tiled_flash_launch();
                    QwenAttentionPrefillTiledCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        scratch.q_b.clone(),
                        kv_read,
                        vv_read,
                        scratch.gate_b.clone(),
                        scratch.attn_out_b.clone(),
                        hd,
                        n_head,
                        n_kv,
                        p,
                        base_pos,
                    );
                } else {
                    QwenAttentionPrefillGatedCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        scratch.q_b.clone(),
                        kv_read,
                        vv_read,
                        scratch.gate_b.clone(),
                        scratch.attn_out_b.clone(),
                        hd,
                        n_head,
                        n_kv,
                        p,
                        base_pos,
                    );
                }
            }
        }

        // 10. Output projection (batched GEMM) → scratch.out_proj_b (attn_out_b
        //     is n_head*hd per token but out_proj emits n_embd per token, so the
        //     projection writes its own scratch buffer — Issue 637 T3).
        // [sub-stage 4: out_proj]
        if attn_sub_on(4) {
            let wo = layer_w.attn_wo.as_ref().expect("attn_wo for Attention layer");
            self.prefill_project(wo, &scratch.attn_out_b, &scratch.out_proj_b, p);
        }

        // 11. Residual add: x_b += out_proj_b → residual_out (batched).
        // [sub-stage 5: outer_norm]
        if attn_sub_on(5) {
            unsafe {
                ResidualAddCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    x_b.clone(),
                    scratch.out_proj_b.clone(),
                    residual_out.clone(),
                    p * n,
                );
            }
        }
    }

    /// Blocking readback of `len` f32 from a handle — diagnostic tap only.
    #[cfg(feature = "ternary_gemm_batched")]
    fn tap_read(&self, h: &Handle, len: usize) -> Vec<f32> {
        let bytes = self.client.read_one(h.clone()).expect("tap readback");
        f32::from_bytes(&bytes)[..len].to_vec()
    }

    /// Set the input hidden state from a token ID via GPU-side dequant.
    ///
    /// Dispatches `dequant_wte_row_f32` into the existing `self.x` buffer —
    /// **zero CPU allocation, zero GPU allocation** per token (Issue 604 T8 G4).
    /// The ternary wte bit-planes are pre-uploaded in `new()` as a `TernaryHandle`;
    /// the dequant runs entirely on GPU, chained with the subsequent forward
    /// dispatches in the same command stream.
    /// Byte alignment wgpu enforces on a storage-buffer binding offset.
    ///
    /// **Measured, not assumed** — `tests/probe_637_offset_alignment.rs` reports
    /// wgpu/Metal's `min_storage_buffer_offset_alignment` as **32** on M3 Max
    /// (offsets of 4 and 16 are rejected by validation; 32 and up are accepted).
    /// wgpu's *default* limit is 256, which would have blocked slicing the
    /// per-head buffers below — `n_v_heads = 48` gives a 192-byte per-token
    /// stride, and those are exactly the `ssm_alpha/beta` projections carrying
    /// Bench 641's 12.92× win. Re-run the probe before assuming this holds on
    /// another backend.
    #[cfg(feature = "ternary_gemm_batched")]
    const OFFSET_ALIGN_F32: usize = 8;

    /// View token `t`'s slice of a `[P, stride]` buffer as a binding of
    /// **exactly** `stride` elements.
    ///
    /// Trimming the END is load-bearing, not cosmetic. Several kernels derive a
    /// dimension from the bound buffer length — `rmsnorm_f32` does
    /// `let dim = input.len()`, and every ternary GEMV does
    /// `let m = output.len()`. `BufferArg::from_raw_parts(handle, len)` does
    /// NOT constrain what the kernel sees when the handle is longer, so an
    /// `offset_start`-only view of row `t` leaks the remaining `P-1-t` rows into
    /// the kernel's notion of its own size.
    ///
    /// For `rmsnorm_f32` that means reducing x² over several tokens while
    /// `inv_dim` still says `1/dim` — a pure **scale** error on the normalized
    /// output. That was the Bench 642 G1 root cause: `norm_x` fed to each
    /// attention layer was mis-scaled, Q/K looked clean because their per-head
    /// RMSNorm cancels a scale, and V (unnormalized) carried the error onward.
    /// It escaped the earlier offset probe because that probe only tested the
    /// LAST row, where the remaining allocation happens to equal `stride`.
    #[cfg(feature = "ternary_gemm_batched")]
    fn tok_slice(h: &Handle, t: usize, stride_elems: usize, p: usize) -> Handle {
        let f = core::mem::size_of::<f32>();
        let sliced = h.clone().offset_start((t * stride_elems * f) as u64);
        let tail = (p - 1 - t) * stride_elems * f;
        if tail > 0 {
            sliced.offset_end(tail as u64)
        } else {
            sliced
        }
    }

    /// One projection over all P tokens — batched GEMM, simdgroup-matrix GEMM
    /// (when `ternary_gemm_simdgroup` is compiled), or P offset GEMVs when
    /// [`PREFILL_USE_GEMV`] is set (the structure-vs-numerics control).
    ///
    /// Plan 534 T5: when `PREFILL_USE_METAL_TENSOR` is on (+ `metal_tensor_gemm`
    /// compiled + macOS), routes through the Metal cooperative-tensor matmul2d
    /// kernel instead of CubeCL. Takes priority over simdgroup + GEMV. Uses a
    /// host round-trip: read CubeCL input handle → Metal buffer → matmul2d →
    /// write back into the CubeCL output handle via `client.write`.
    #[cfg(feature = "ternary_gemm_batched")]
    fn prefill_project(
        &self,
        w: &TernaryHandle,
        input: &Handle,
        output: &Handle,
        p: usize,
    ) {
        let client = &self.client;

        // Issue 663 T5: zero-copy RAW-Metal dispatch path. HIGHEST priority —
        // when the flag is on + the zero-copy GEMM context is initialized +
        // the weight has a zero-copy cache, dispatch the matmul2d kernel via
        // raw `MTLComputeCommandEncoder` on the shared CubeCL Metal device +
        // queue. Zero host round-trip, zero staging copy, zero bind group.
        // Falls through to the other paths on dispatch failure.
        #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
        if PREFILL_USE_METAL_TENSOR_ZEROCOPY.load(std::sync::atomic::Ordering::Relaxed)
            && let (Some(gemm), Some(cache)) = (&self.metal_zerocopy_gemm, &w.zerocopy_cache)
        {
            if let Err(e) = gemm.dispatch(client, cache, w, input, output, p) {
                eprintln!(
                    "[Issue 663 T5] zero-copy dispatch failed ({e}); falling through"
                );
            } else {
                return;
            }
        }

        // Issue 657: zero-copy wgpu MSL passthrough path. HIGHEST priority —
        // when the flag is on + the wgpu GEMM context is initialized, dispatch
        // the matmul2d kernel directly on CubeCL's GPU buffers via wgpu's MSL
        // passthrough. No host round-trip, no cross-queue sync. Falls through
        // to the host round-trip + CubeCL paths otherwise.
        //
        // Issue 727 H3: the per-weight cache is populated LAZILY on first
        // dispatch through this path (`get_or_init`) — a GPU→GPU copy from
        // the CubeCL handles. Builds that never enable the flag never pay
        // the copy; the A/B pattern (flag flipped post-construction) keeps
        // working because the cache materializes here, not at `new()`.
        #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
        if PREFILL_USE_METAL_TENSOR_WGPU.load(std::sync::atomic::Ordering::Relaxed)
            && let Some(gemm) = &self.metal_wgpu_gemm
        {
            let cache = w
                .wgpu_cache
                .get_or_init(|| gemm.cache_weights(client, w).ok());
            if let Some(cache) = cache {
                if let Err(e) = gemm.dispatch(client, cache, w, input, output, p) {
                    // Dispatch failed (e.g., buffer extraction error). Log + fall
                    // through to the next path rather than panicking.
                    eprintln!("[Issue 657] wgpu dispatch failed ({e}); falling through");
                } else {
                    return;
                }
            } else {
                warn_metal_cache_missing_once(
                    "PREFILL_USE_METAL_TENSOR_WGPU",
                    "wgpu weight cache build failed (device extraction)",
                );
            }
        }

        // Plan 534 T5: metal::tensor matmul2d path (host round-trip). When the
        // flag is on + the Metal context is initialized + the weight has a
        // Metal cache. Falls through to the CubeCL path otherwise.
        //
        // Issue 727 H3: the metal-rs cache is construction-gated on the flag
        // (it needs the CPU weights, so it cannot be retrofitted lazily).
        // Flag flipped after construction without the cache → one-time loud
        // warning + CubeCL fallthrough (previously this was SILENT — a
        // mis-configured A/B would measure CubeCL twice without a trace).
        #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
        if PREFILL_USE_METAL_TENSOR.load(std::sync::atomic::Ordering::Relaxed)
            && let (Some(gemm), Some(mw)) = (&self.metal_gemm, &w.metal)
        {
            self.prefill_project_metal(gemm, mw, w, input, output, p);
            return;
        }
        #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos", feature = "ternary_gemm_batched"))]
        if PREFILL_USE_METAL_TENSOR.load(std::sync::atomic::Ordering::Relaxed)
            && self.metal_gemm.is_none()
        {
            warn_metal_cache_missing_once(
                "PREFILL_USE_METAL_TENSOR",
                "the metal-rs weight cache was skipped at construction \
                 (flag was off) — set it BEFORE TernaryDeltanetGpuForward::new()",
            );
        }

        // Issue 734 Arm 6 (Bench 720): raw-CUDA mma.sync i8 GEMM e2e wiring —
        // Bench 719's 2.45-2.49× kernel dispatched through a host round-trip
        // bridge (read CubeCL input → CUDA quantize+GEMM → write CubeCL
        // output; the `prefill_project_metal` precedent). Bit-identity on real
        // activations is pinned by the Bench-710 full-model FNV gates; the
        // round-trip bus cost is part of the honest e2e measurement. DEFAULT
        // OFF (`RIIR_PREFILL_CUDA_MMA` / `set_prefill_use_cuda_mma`). Length-
        // gated to the Bench-719-validated region + the GROUP_COLS contract.
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            not(target_os = "macos")
        ))]
        if crate::prefill_cuda_mma::prefill_use_cuda_mma()
            && p <= 4096
            && w.n.is_multiple_of(128)
            && crate::prefill_cuda_mma::dispatch(&self.client, w, input, output, p)
        {
            return;
        }

        if PREFILL_USE_GEMV.load(std::sync::atomic::Ordering::Relaxed) {
            for t in 0..p {
                unsafe {
                    GemvTernaryCubeCL::launch::<ActiveRuntime>(
                        client,
                        w,
                        Self::tok_slice(input, t, w.n, p),
                        Self::tok_slice(output, t, w.m, p),
                    );
                }
            }
        } else {
            // Issue 734 T7: NVIDIA int8 cooperative-matrix tensor cores —
            // checked FIRST when enabled. The i8×i8→i32 @ 16×16×32 shape is
            // NVIDIA-only; the kernel quantizes activations per token
            // (hi/lo int8, exact weight signs) on-GPU. LENGTH-GATED to the
            // measured-win region (e2e +0.8% @2048, +2.5% @4096, small-m
            // shapes up to 1.54×, argmax identical, G1 max_rel 8e-5 vs f16's
            // 2.2e-4). Chunked prefill (Bench 710) routes EVERY chunk through
            // p ≤ 4096, so long contexts ride this gate per chunk; the
            // cmma16 fallback stays for p > 4096 single-chunk calls.
            #[cfg(feature = "ternary_gemm_batched")]
            if PREFILL_USE_CMMA_I8.load(std::sync::atomic::Ordering::Relaxed)
                && p <= 4096
                && crate::GemmTernaryCmmaI8CubeCL::i8_available::<ActiveRuntime>(client)
            {
                // Issue 734 Lever 2: the tile (128×64) and B-direct variants
                // — both expected bit-identical, env/override A/B
                // (`RIIR_CMMA_I8_T64` / `RIIR_CMMA_I8_DIRECT`). t64 takes
                // precedence (alternative rewrites of the same kernel).
                let use_t64 = prefill_cmma_i8_t64();
                let use_direct = !use_t64
                    && prefill_cmma_i8_direct()
                    && crate::GemmTernaryCmmaI8CubeCL::tensor_addressing_available::<ActiveRuntime>(
                        client,
                    );
                // Issue 734 Arm 5 (Bench 718): the two-round-partials variant —
                // same kernel family, 28 KB workgroup smem (3 wgs/SM on the
                // 4090); checked after t64/direct (alternative rewrite).
                let use_psplit = !use_t64
                    && !use_direct
                    && prefill_cmma_i8_psplit();
                unsafe {
                    if use_t64 {
                        crate::GemmTernaryCmmaI8CubeCL::launch_t64::<ActiveRuntime>(
                            client, w, input.clone(), output.clone(), p,
                        );
                    } else if use_direct {
                        crate::GemmTernaryCmmaI8CubeCL::launch_direct::<ActiveRuntime>(
                            client, w, input.clone(), output.clone(), p,
                        );
                    } else if use_psplit {
                        crate::GemmTernaryCmmaI8CubeCL::launch_psplit::<ActiveRuntime>(
                            client, w, input.clone(), output.clone(), p,
                        );
                    } else {
                        crate::GemmTernaryCmmaI8CubeCL::launch_sg8::<ActiveRuntime>(
                            client, w, input.clone(), output.clone(), p,
                        );
                    }
                }
                return;
            }

            // Issue 734 T6: NVIDIA cooperative-matrix tensor cores — checked
            // FIRST. The 16×16×16 f16 signature is NVIDIA-only (Metal reports
            // 8×8×8), so this never shadows the M3 simdgroup path below;
            // on the 4090 it measured 1.64× the tiled kernel (32.8 TFLOPS,
            // Bench 734 T6) at f16 input precision (G1 max_rel 2.2e-4).
            #[cfg(feature = "ternary_gemm_batched")]
            if PREFILL_USE_CMMA16.load(std::sync::atomic::Ordering::Relaxed)
                && crate::GemmTernaryCmma16CubeCL::f16_available::<ActiveRuntime>(client)
            {
                unsafe {
                    crate::GemmTernaryCmma16CubeCL::launch_sg8::<ActiveRuntime>(
                        client, w, input.clone(), output.clone(), p,
                    );
                }
                return;
            }

            // Issue 641 / Bench 645: when the simdgroup feature is compiled,
            // prefer the hardware cooperative-matrix kernel (1.89× roll-up)
            // over the plane-cooperative kernel (1.08×). The cmma check is a
            // cheap set lookup; cached device features make this effectively
            // branch-free after the first call. Checked FIRST so the M3's
            // proven cmma path keeps priority — the Issue 730 lesson (a
            // 4090-validated path is not automatically Metal-best; the first
            // cut of the tiled kernel below shadowed this branch on M3,
            // unmeasured there).
            #[cfg(feature = "ternary_gemm_simdgroup")]
            let use_simdgroup =
                PREFILL_USE_SIMDGROUP.load(std::sync::atomic::Ordering::Relaxed)
                    && GemmTernarySimdgroupCubeCL::cmma_available::<ActiveRuntime>(client);
            #[cfg(not(feature = "ternary_gemm_simdgroup"))]
            let use_simdgroup = false;

            // Issue 655: f16 cmma path — when the f16 feature is compiled + the
            // toggle is on + the device supports (f16, f16, f32) cmma, prefer
            // the f16 simdgroup kernel over the f32 one.
            #[cfg(all(feature = "ternary_gemm_simdgroup", feature = "ternary_gemm_simdgroup_f16"))]
            let use_simdgroup_f16 =
                PREFILL_USE_SIMDGROUP_F16.load(std::sync::atomic::Ordering::Relaxed)
                    && GemmTernarySimdgroupCubeCL::cmma_available_f16::<ActiveRuntime>(client);
            #[cfg(not(all(feature = "ternary_gemm_simdgroup", feature = "ternary_gemm_simdgroup_f16")))]
            let use_simdgroup_f16 = false;

            // Issue 767: scale-deferred f16-simdgroup path — same (f16,f16,f32)
            // capability gate; checked BEFORE the refuted in-kernel-scaled f16
            // variant so the deferred kernel wins when both toggles are set.
            #[cfg(all(feature = "ternary_gemm_simdgroup", feature = "ternary_gemm_simdgroup_f16"))]
            let use_simdgroup_deferred =
                PREFILL_USE_SIMDGROUP_DEFERRED.load(std::sync::atomic::Ordering::Relaxed)
                    && GemmTernarySimdgroupCubeCL::cmma_available_f16::<ActiveRuntime>(client)
                    // 8×32-shape heuristic (the kernel's tile): tiny-M / small-P
                    // stay on the f32 8×8 arm (Bench 645 dispatch note).
                    && w.m >= 64
                    && p >= 32;
            #[cfg(not(all(feature = "ternary_gemm_simdgroup", feature = "ternary_gemm_simdgroup_f16")))]
            let use_simdgroup_deferred = false;

            if use_simdgroup && use_simdgroup_deferred {
                // Issue 767 T1: scale-deferred cmma — f16 signs on the matrix
                // unit, group scales on the f32 accumulator at 128-K boundaries.
                #[cfg(all(feature = "ternary_gemm_simdgroup", feature = "ternary_gemm_simdgroup_f16"))]
                unsafe {
                    GemmTernarySimdgroupCubeCL::launch_scale_deferred::<ActiveRuntime>(
                        client,
                        w,
                        input.clone(),
                        output.clone(),
                        p,
                    );
                }
            } else if use_simdgroup && use_simdgroup_f16 {
                // Issue 655: f16 cmma — same dispatch heuristic as the f32 path
                // but via the f16 launcher.
                #[cfg(all(feature = "ternary_gemm_simdgroup", feature = "ternary_gemm_simdgroup_f16"))]
                unsafe {
                    GemmTernarySimdgroupCubeCL::launch_f16::<ActiveRuntime>(
                        client,
                        w,
                        input.clone(),
                        output.clone(),
                        p,
                    );
                }
            } else if use_simdgroup {
                // Dispatch heuristic (Bench 645 per-shape results + Issue 768):
                // - tiny-M (m < 64): 8×8 wins 1.41× vs 8×32's 1.20× (the
                //   8×32's P-amortization hurts when there are few M-tiles);
                //   also where the 32×32 tile pads (measured 0.773× at m=48)
                // - small-P (p < 32): 8×8 — the wider tiles waste sub-tiles
                // - m >= 64 && p >= 32: the 32×32 input-reuse tile (Issue 768,
                //   1.44–1.70× measured, bit-identical outputs) — else 8×32
                #[cfg(feature = "ternary_gemm_simdgroup")]
                {
                    // Issue 771 T1: the smem-staged 128×64 tier rides the
                    // tall-GEMM master switch (tall off = "old kernels only")
                    // plus its own A/B hook, and widens the shape heuristic
                    // (the tile pads m<128 / p<64 shapes).
                    let use_smem = w.m >= 128
                        && p >= 64
                        && PREFILL_USE_TALL_GEMM.load(std::sync::atomic::Ordering::Relaxed)
                        && GemmTernarySimdgroupCubeCL::prefill_smem_gemm_enabled();
                    let use_tall = w.m >= 64
                        && p >= 32
                        && PREFILL_USE_TALL_GEMM.load(std::sync::atomic::Ordering::Relaxed);
                    let use_8x32 = w.m >= 64 && p >= 32;
                    unsafe {
                        if use_smem {
                            GemmTernarySimdgroupCubeCL::launch_smem::<ActiveRuntime>(
                                client,
                                w,
                                input.clone(),
                                output.clone(),
                                p,
                            );
                        } else if use_tall {
                            GemmTernarySimdgroupCubeCL::launch_32x32::<ActiveRuntime>(
                                client,
                                w,
                                input.clone(),
                                output.clone(),
                                p,
                            );
                        } else if use_8x32 {
                            GemmTernarySimdgroupCubeCL::launch_8x32::<ActiveRuntime>(
                                client,
                                w,
                                input.clone(),
                                output.clone(),
                                p,
                            );
                        } else {
                            GemmTernarySimdgroupCubeCL::launch::<ActiveRuntime>(
                                client,
                                w,
                                input.clone(),
                                output.clone(),
                                p,
                            );
                        }
                    }
                }
            } else {
                // Issue 734 T3: the workgroup-tiled kernel — the 4090/Vulkan
                // path (7.79× the plane-coop kernel at the Bonsai shapes,
                // Bench 704; 19.9 TFLOPS vs 2.57). Reached only when cmma is
                // unavailable (NVIDIA Vulkan reports no Metal-signature
                // 8×8×8 f32) or the simdgroup flag is off — on M3 the cmma
                // branch above keeps priority. A/B with
                // `set_prefill_use_tiled_gemm(false)` to fall to plane-coop,
                // or disable simdgroup to force tiled on Metal.
                #[cfg(feature = "ternary_gemm_batched")]
                if PREFILL_USE_TILED_GEMM.load(std::sync::atomic::Ordering::Relaxed) {
                    unsafe {
                        crate::GemmTernaryTiledCubeCL::launch::<ActiveRuntime>(
                            client, w, input.clone(), output.clone(), p,
                        );
                    }
                } else {
                    unsafe {
                        GemmTernaryBatchedCubeCL::launch::<ActiveRuntime>(
                            client,
                            w,
                            input.clone(),
                            output.clone(),
                            p,
                        );
                    }
                }
            }
        }
    }

    /// Issue 726: configure the ANE hybrid prefill split (channel fraction,
    /// block tokens). Takes effect on the next prefill call. No-op when the
    /// `ane_prefill` feature is off (the field doesn't exist).
    #[cfg(feature = "ane_prefill")]
    pub fn set_ane_prefill_config(&mut self, cfg: crate::ane_prefill::AnePrefillConfig) {
        self.ane_prefill_cfg = cfg;
    }

    /// Issue 726: whether the ANE program bank is Ready on this forward
    /// (diagnostic for the GOAT harness — an Unavailable ctx means the
    /// hybrid arm is silently measuring the GPU path).
    #[cfg(feature = "ane_prefill")]
    pub fn ane_prefill_ready(&self) -> bool {
        self.ane_prefill.is_ready()
    }

    /// Issue 726: the ANE ctx state's unavailability reason, if any
    /// (GOAT harness logging).
    #[cfg(all(
        feature = "ane_prefill",
        all(target_os = "macos", target_arch = "aarch64")
    ))]
    pub fn ane_prefill_unavailable_reason(&self) -> Option<std::sync::Arc<str>> {
        match self.ane_prefill.state() {
            crate::ane_prefill::AnePrefillState::Unavailable(r) => Some(r.clone()),
            crate::ane_prefill::AnePrefillState::Ready(_) => None,
        }
    }

    /// Issue 886 T5: the construction-time down-lane outcome — the GDN
    /// layers the ladder attempted / landed / refused (registration order)
    /// plus cumulative down bytes. Empty when the down lane was off, an
    /// explicit claim registered nothing, or the ctx never went Ready. An
    /// `attempted` entry resolved by neither `landed` nor `refused` means
    /// the registration loop broke early (ctx poisoned) — a finding, never
    /// a silent gap.
    #[cfg(feature = "ane_prefill")]
    pub fn ane_down_report(&self) -> &crate::ane_prefill::AneDownReport {
        &self.ane_down_report
    }

    // ── Issue 726 T2: ANE hybrid prefill seam ─────────────────────────────
    //
    // The eligibility gate + dispatch seam for the dual ANE/GPU hybrid
    // prefill path (see `crate::ane_prefill` for the full contract). T2
    // ships the gate + this seam; the ANE ctx is `NotWired` until T3's
    // program bank lands, so both try-helpers always return false and the
    // existing GPU batched GEMM path runs byte-identically — fail-open by
    // construction.

    /// Evaluate the Issue 726 eligibility gate for one (layer, op, p).
    #[cfg(all(feature = "ane_prefill", feature = "ternary_gemm_batched"))]
    fn ane_prefill_plan(
        &self,
        p: usize,
        is_gdn: bool,
        op: crate::ane_prefill::AnePrefillOp,
    ) -> Option<crate::ane_prefill::AnePrefillPlan> {
        crate::ane_prefill::ane_prefill_eligible(
            p,
            is_gdn,
            op,
            self.ane_prefill.is_ready(),
            crate::ane_prefill::prefill_use_ane(),
            &self.ane_prefill_cfg,
        )
    }

    /// ANE dispatch seam for the fused `in_proj_concat` (qkv|z|a|b,
    /// 5120→16480). Returns true only when the ANE path produced ALL the
    /// projection outputs for this prompt (Phase B: full coverage — f = 1.0
    /// exact blocks only; partial coverage fail-opens to the GPU split
    /// GEMMs until the Phase C suffix/tail complement lands).
    ///
    /// Plan 549 Phase C: when the split-overlap toggle is on (exact blocks),
    /// the ANE computes the qkv segment while the GPU projects z/a/b
    /// concurrently — see [`Self::ane_prefill_inproj_split`].
    #[cfg(all(feature = "ane_prefill", feature = "ternary_gemm_batched"))]
    fn ane_prefill_try_inproj(
        &self,
        layer_idx: usize,
        normx_b: &Handle,
        qkv_b: &Handle,
        z_b: &Handle,
        a_b: &Handle,
        b_b: &Handle,
        p: usize,
    ) -> bool {
        let Some(plan) =
            self.ane_prefill_plan(p, true, crate::ane_prefill::AnePrefillOp::InProjConcat)
        else {
            return false;
        };
        // 887: f<1 has no executor (Phase C suffix complement) — the split
        // arm refuses it exactly like the full-coverage arm below.
        if crate::ane_prefill::prefill_ane_split()
            && plan.ane_blocks >= 1
            && plan.tail == 0
            && plan.channel_fraction >= 1.0
        {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                return self.ane_prefill_inproj_split(
                    layer_idx, normx_b, qkv_b, z_b, a_b, b_b, p, plan,
                );
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                let _ = (layer_idx, qkv_b, z_b, a_b, b_b);
                return false;
            }
        }
        if !plan.full_coverage() {
            return false; // Phase C: suffix-channel split + tail complement
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let Some(kernel) = self
                .ane_prefill
                .kernel(layer_idx, crate::ane_prefill::AnePrefillOp::InProjConcat)
            else {
                return false;
            };
            let n_v_heads = self.config.deltanet_linear_n_value_heads;
            // Segment widths in the fused row order [qkv | z | a | b]
            // (from_weights_concat order — the requant bank matches it).
            let n_k_heads = self.config.deltanet_linear_n_heads;
            let head_dim = self.config.deltanet_linear_head_dim;
            let q_dim = n_k_heads * head_dim;
            let v_dim = n_v_heads * head_dim;
            let qkv_dim = 2 * q_dim + v_dim;
            let z_dim = v_dim;
            let oc_total = qkv_dim + z_dim + 2 * n_v_heads;
            let segments = [
                (qkv_b.clone(), 0usize, qkv_dim),
                (z_b.clone(), qkv_dim, z_dim),
                (a_b.clone(), qkv_dim + z_dim, n_v_heads),
                (b_b.clone(), qkv_dim + z_dim + n_v_heads, n_v_heads),
            ];
            // 887: derive the output width from the BANK's compiled program,
            // never from config — a prefix-registered bank (split mode at
            // construction) must fail open here, not mis-unpack.
            let w = plan.ane_tokens / plan.ane_blocks;
            let kw = kernel.output_channels(w);
            if kw < oc_total {
                crate::ane_prefill::warn_width_mismatch_once("in_proj full-width", oc_total, kw);
                return false;
            }
            match crate::ane_prefill::exec::dispatch_full_width(
                &self.client,
                &kernel,
                normx_b,
                p,
                self.config.n_embd,
                kw,
                w,
                &segments,
            ) {
                Ok(()) => true,
                Err(e) => {
                    // Fail-open mid-prompt is NOT safe here (the block loop
                    // may have written partial segments) — surface loudly.
                    // The GPU path re-runs the FULL projection below only
                    // when we return false BEFORE any write; after writes,
                    // a partial dispatch is a hard error.
                    panic!("ANE in_proj dispatch failed mid-prompt: {e}");
                }
            }
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (layer_idx, normx_b, qkv_b, z_b, a_b, b_b);
            false
        }
    }

    /// Plan 549: the in_proj split-overlap dispatch — ANE computes the fused
    /// program's qkv prefix segment (rows `[0, qkv_dim)` of the fused row
    /// order) on a worker thread while the GPU projects the z/a/b complement
    /// concurrently. Clean fail-open: the job writes nothing before a
    /// successful join, so an eval failure returns false with zero
    /// partial-state risk (the fallback re-runs ALL splits — z/a/b twice is
    /// idempotent, correct, and rare).
    #[cfg(all(
        feature = "ane_prefill",
        feature = "ternary_gemm_batched",
        all(target_os = "macos", target_arch = "aarch64")
    ))]
    fn ane_prefill_inproj_split(
        &self,
        layer_idx: usize,
        normx_b: &Handle,
        qkv_b: &Handle,
        z_b: &Handle,
        a_b: &Handle,
        b_b: &Handle,
        p: usize,
        plan: crate::ane_prefill::AnePrefillPlan,
    ) -> bool {
        let Some(kernel) = self
            .ane_prefill
            .kernel(layer_idx, crate::ane_prefill::AnePrefillOp::InProjConcat)
        else {
            return false;
        };
        let layer_w = &self.layers[layer_idx];
        let n_v_heads = self.config.deltanet_linear_n_value_heads;
        let n_k_heads = self.config.deltanet_linear_n_heads;
        let head_dim = self.config.deltanet_linear_head_dim;
        let q_dim = n_k_heads * head_dim;
        let v_dim = n_v_heads * head_dim;
        let qkv_dim = 2 * q_dim + v_dim;
        // Program-relative segment: qkv IS the fused row prefix.
        let segments = [(qkv_b.clone(), 0usize, qkv_dim)];
        let w = plan.ane_tokens / plan.ane_blocks;
        // 887: the compiled width is the BANK's truth. A full-fused bank
        // (split off at construction) evaluates oc_total rows and discards
        // the non-segment ones (today's latency-hiding shape); a prefix bank
        // evaluates exactly qkv_dim. Both dispatch the same segment — the
        // guard only catches genuine mode mismatches (toggle flipped after
        // construction).
        let kw = kernel.output_channels(w);
        if kw < qkv_dim {
            crate::ane_prefill::warn_width_mismatch_once("in_proj split", qkv_dim, kw);
            return false;
        }
        // Plan 550 → Probe 779: the FULL zero-copy executor (GPU pack + GPU
        // unpack over the ANE's own io surfaces). Probe 779 proved the
        // executor deterministic in its production shape (10/10 varied-input
        // exact); host-IO split stays the fail-open fallback.
        #[cfg(all(
            feature = "metal_tensor_gemm",
            all(target_os = "macos", target_arch = "aarch64")
        ))]
        if crate::ane_prefill::prefill_ane_zero_copy()
            && let Some(zc) = crate::ane_prefill::exec_zc::zc_context_cached()
            && let Ok(job) = crate::ane_prefill::exec_zc::begin_split_overlapped_zc(
                &self.client,
                zc,
                kernel.clone(),
                normx_b,
                p,
                self.config.n_embd,
                kw,
                w,
                &segments,
            )
        {
            self.prefill_project(&layer_w.in_proj_z, normx_b, z_b, p);
            self.prefill_project(&layer_w.in_proj_a, normx_b, a_b, p);
            self.prefill_project(&layer_w.in_proj_b, normx_b, b_b, p);
            return match job.finish() {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("[ane] in_proj zc-out split failed clean → GPU fallback: {e}");
                    false
                }
            };
        }
        let Ok(job) = crate::ane_prefill::exec::begin_split_overlapped(
            &self.client,
            kernel,
            normx_b,
            p,
            self.config.n_embd,
            kw,
            w,
            &segments,
        ) else {
            return false;
        };
        // GPU complement — concurrent with the ANE worker.
        self.prefill_project(&layer_w.in_proj_z, normx_b, z_b, p);
        self.prefill_project(&layer_w.in_proj_a, normx_b, a_b, p);
        self.prefill_project(&layer_w.in_proj_b, normx_b, b_b, p);
        match job.finish(&self.client) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("[ane] in_proj split failed clean → GPU fallback: {e}");
                false
            }
        }
    }

    /// ANE dispatch seam for the fused `gate_up_proj` (gate|up,
    /// 5120→34816). Requires the layer to be GDN — attention-layer FFNs
    /// stay GPU. Phase B: full coverage only (same contract as the
    /// in_proj seam). Plan 549: split-overlap mode takes the gate segment
    /// ∥ GPU up.
    #[cfg(all(feature = "ane_prefill", feature = "ternary_gemm_batched"))]
    fn ane_prefill_try_gate_up(
        &self,
        layer_idx: usize,
        normx_b: &Handle,
        gate_b: &Handle,
        up_b: &Handle,
        p: usize,
    ) -> bool {
        let is_gdn = self.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;
        let Some(plan) =
            self.ane_prefill_plan(p, is_gdn, crate::ane_prefill::AnePrefillOp::GateUpProj)
        else {
            return false;
        };
        // 887: f<1 has no executor (Phase C suffix complement) — the split
        // arm refuses it exactly like the full-coverage arm below.
        if crate::ane_prefill::prefill_ane_split()
            && plan.ane_blocks >= 1
            && plan.tail == 0
            && plan.channel_fraction >= 1.0
        {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                return self.ane_prefill_gate_up_split(
                    layer_idx, normx_b, gate_b, up_b, p, plan,
                );
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                let _ = (layer_idx, gate_b, up_b);
                return false;
            }
        }
        if !plan.full_coverage() {
            return false; // Phase C
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let Some(kernel) = self
                .ane_prefill
                .kernel(layer_idx, crate::ane_prefill::AnePrefillOp::GateUpProj)
            else {
                return false;
            };
            let mlp = self.config.mlp_hidden;
            let segments = [
                (gate_b.clone(), 0usize, mlp),
                (up_b.clone(), mlp, mlp),
            ];
            // 887: bank-compiled width is the truth (see ane_prefill_try_inproj).
            let w = plan.ane_tokens / plan.ane_blocks;
            let kw = kernel.output_channels(w);
            if kw < 2 * mlp {
                crate::ane_prefill::warn_width_mismatch_once("gate_up full-width", 2 * mlp, kw);
                return false;
            }
            match crate::ane_prefill::exec::dispatch_full_width(
                &self.client,
                &kernel,
                normx_b,
                p,
                self.config.n_embd,
                kw,
                w,
                &segments,
            ) {
                Ok(()) => true,
                Err(e) => panic!("ANE gate_up dispatch failed mid-prompt: {e}"),
            }
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (layer_idx, normx_b, gate_b, up_b);
            false
        }
    }

    /// Plan 549: the gate_up split-overlap dispatch — ANE computes the fused
    /// program's gate prefix segment on a worker thread while the GPU
    /// projects the up complement concurrently (the ≈156 ms window that
    /// hides the fused eval ≈155 ms). Clean fail-open — see
    /// [`Self::ane_prefill_inproj_split`].
    #[cfg(all(
        feature = "ane_prefill",
        feature = "ternary_gemm_batched",
        all(target_os = "macos", target_arch = "aarch64")
    ))]
    fn ane_prefill_gate_up_split(
        &self,
        layer_idx: usize,
        normx_b: &Handle,
        gate_b: &Handle,
        up_b: &Handle,
        p: usize,
        plan: crate::ane_prefill::AnePrefillPlan,
    ) -> bool {
        let Some(kernel) = self
            .ane_prefill
            .kernel(layer_idx, crate::ane_prefill::AnePrefillOp::GateUpProj)
        else {
            return false;
        };
        let layer_w = &self.layers[layer_idx];
        let mlp = self.config.mlp_hidden;
        // gate IS the fused row prefix; the up complement runs on GPU.
        let segments = [(gate_b.clone(), 0usize, mlp)];
        let w = plan.ane_tokens / plan.ane_blocks;
        // 887: bank-compiled width is the truth (see ane_prefill_inproj_split).
        let kw = kernel.output_channels(w);
        if kw < mlp {
            crate::ane_prefill::warn_width_mismatch_once("gate_up split", mlp, kw);
            return false;
        }
        // Plan 550 → Probe 779: the FULL zero-copy executor; host-IO split
        // as fallback (same contract as the in_proj seam).
        #[cfg(all(
            feature = "metal_tensor_gemm",
            all(target_os = "macos", target_arch = "aarch64")
        ))]
        if crate::ane_prefill::prefill_ane_zero_copy()
            && let Some(zc) = crate::ane_prefill::exec_zc::zc_context_cached()
            && let Ok(job) = crate::ane_prefill::exec_zc::begin_split_overlapped_zc(
                &self.client,
                zc,
                kernel.clone(),
                normx_b,
                p,
                self.config.n_embd,
                kw,
                w,
                &segments,
            )
        {
            self.prefill_project(&layer_w.up_proj, normx_b, up_b, p);
            return match job.finish() {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("[ane] gate_up zc-out split failed clean → GPU fallback: {e}");
                    false
                }
            };
        }
        let Ok(job) = crate::ane_prefill::exec::begin_split_overlapped(
            &self.client,
            kernel,
            normx_b,
            p,
            self.config.n_embd,
            kw,
            w,
            &segments,
        ) else {
            return false;
        };
        self.prefill_project(&layer_w.up_proj, normx_b, up_b, p);
        match job.finish(&self.client) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("[ane] gate_up split failed clean → GPU fallback: {e}");
                false
            }
        }
    }

    /// Plan 549: the down_proj ANE seam — GDN layers only, runtime-gated
    /// (`set_prefill_ane_down`), exact blocks. Zero overlap window (the
    /// residual consumer needs ffnout) — this op rides the job API purely
    /// for its clean fail-open + one dependency-bound read; the win, if any,
    /// is ANE time vs GPU GEMM time net of the 286 MB input readback + pack.
    #[cfg(all(feature = "ane_prefill", feature = "ternary_gemm_batched"))]
    fn ane_prefill_try_down(
        &self,
        layer_idx: usize,
        hid_b: &Handle,
        ffnout_b: &Handle,
        p: usize,
    ) -> bool {
        if !crate::ane_prefill::prefill_ane_down() {
            return false;
        }
        let is_gdn = self.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;
        let Some(plan) =
            self.ane_prefill_plan(p, is_gdn, crate::ane_prefill::AnePrefillOp::DownProj)
        else {
            return false;
        };
        // 887: down is all-or-nothing per token axis — no channel
        // complement exists, so a fractional plan must never dispatch
        // (belt-and-braces: init already refuses f<1 ctx-wide).
        if plan.ane_blocks < 1 || plan.tail != 0 || plan.channel_fraction < 1.0 {
            return false;
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let Some(kernel) = self
                .ane_prefill
                .kernel(layer_idx, crate::ane_prefill::AnePrefillOp::DownProj)
            else {
                return false; // per-op fail-open (budget/compile skip)
            };
            let mlp = self.config.mlp_hidden;
            let n = self.config.n_embd;
            let segments = [(ffnout_b.clone(), 0usize, n)];
            let w = plan.ane_tokens / plan.ane_blocks;
            // 887: bank-compiled width is the truth (down never slices — a
            // narrower program here is a genuine mode mismatch).
            let kw = kernel.output_channels(w);
            if kw < n {
                crate::ane_prefill::warn_width_mismatch_once("down split", n, kw);
                return false;
            }
            // Plan 550 → Probe 779: the FULL zero-copy executor (kills the
            // 286 MB readback AND the output host path — Bench 776 measured
            // down as a LOSS under the host bridge partly because of it);
            // host-IO split as fallback.
            #[cfg(all(
                feature = "metal_tensor_gemm",
                all(target_os = "macos", target_arch = "aarch64")
            ))]
            if crate::ane_prefill::prefill_ane_zero_copy()
                && let Some(zc) = crate::ane_prefill::exec_zc::zc_context_cached()
                && let Ok(job) = crate::ane_prefill::exec_zc::begin_split_overlapped_zc(
                    &self.client,
                    zc,
                    kernel.clone(),
                    hid_b,
                    p,
                    mlp,
                    kw,
                    w,
                    &segments,
                )
            {
                return match job.finish() {
                    Ok(()) => true,
                    Err(e) => {
                        eprintln!("[ane] down zc-out split failed clean → GPU fallback: {e}");
                        false
                    }
                };
            }
            let Ok(job) = crate::ane_prefill::exec::begin_split_overlapped(
                &self.client,
                kernel,
                hid_b,
                p,
                mlp,
                kw,
                w,
                &segments,
            ) else {
                return false;
            };
            match job.finish(&self.client) {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("[ane] down split failed clean → GPU fallback: {e}");
                    false
                }
            }
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = (layer_idx, hid_b, ffnout_b);
            false
        }
    }

    /// Metal cooperative-tensor matmul2d projection (Plan 534 T5).
    ///
    /// Host round-trip: read CubeCL input handle → upload to Metal buffer →
    /// run matmul2d → read back → write into CubeCL output handle via
    /// `client.write`. The GEMM-level 3.72× speedup (Issue 656) must dominate
    /// the round-trip cost for this path to be a net win.
    #[cfg(all(
        feature = "ternary_gemm_batched",
        feature = "metal_tensor_gemm",
        target_os = "macos"
    ))]
    fn prefill_project_metal(
        &self,
        gemm: &crate::gemm_ternary_metal_tensor::MetalTensorGemm,
        mw: &crate::gemm_ternary_metal_tensor::MetalWeightCache,
        w: &TernaryHandle,
        input: &Handle,
        output: &Handle,
        p: usize,
    ) {
        use cubecl::prelude::*; // for Bytes

        // 1. Read the input handle back to host (DMA GPU→host).
        let input_bytes = self
            .client
            .read_one(input.clone())
            .expect("metal_tensor prefill: read input");
        let input_f32 = f32::from_bytes(&input_bytes);

        // 2. Upload activations to a reused Metal scratch buffer (T8).
        let in_buf = gemm.upload_f32_scratch(input_f32);

        // 3. Get reused output buffer + launch matmul2d.
        let out_len = p * w.m;
        let out_buf = gemm.create_output_scratch(out_len);
        gemm.launch(
            &mw.pos,
            &mw.neg,
            &mw.scale,
            &in_buf,
            &out_buf,
            w.blocks64 as u32,
            w.groups_per_row as u32,
            w.n as u32,
            w.m as u32,
            p as u32,
        );

        // 4. Read Metal output back to host.
        let out_f32 = gemm.read_f32(&out_buf, out_len);

        // 5. Write into the existing CubeCL output handle in-place.
        // `client.write` takes `cubecl_common::bytes::Bytes`; we construct it
        // via `from_bytes_vec`. The `Bytes` type is re-exported through the
        // `cubecl` crate's transitive re-export of `cubecl_common`.
        let out_bytes_vec = f32::as_bytes(&out_f32).to_vec();
        // Construct the `Bytes` wrapper that `client.write` expects. The type
        // comes from `cubecl_common::bytes::Bytes`, re-exported transitively
        // via the `cubecl` crate. We use the fully-qualified path to avoid
        // an explicit `use` that might conflict.
        //
        // NOTE: on macOS, `cubecl-runtime` enables `cubecl-common/serde`, which
        // gates the `bytes` module. So `cubecl::bytes::Bytes` is available.
        self.client.write(
            output,
            cubecl::bytes::Bytes::from_bytes_vec(out_bytes_vec),
        );
    }

    /// Row-wise RMSNorm over all P tokens — one batched call, or P single-token
    /// calls when [`PREFILL_SEQ_RMSNORM`] is set.
    #[cfg(feature = "ternary_gemm_batched")]
    fn prefill_norm(
        client: &ComputeClient<ActiveRuntime>,
        input: &Handle,
        gamma: &Handle,
        output: &Handle,
        p: usize,
        dim: usize,
        eps: f32,
    ) {
        if PREFILL_SEQ_RMSNORM.load(std::sync::atomic::Ordering::Relaxed) {
            for t in 0..p {
                unsafe {
                    RmsNormCubeCL::launch::<ActiveRuntime>(
                        client,
                        Self::tok_slice(input, t, dim, p),
                        gamma.clone(),
                        Self::tok_slice(output, t, dim, p),
                        dim,
                        eps,
                    );
                }
            }
        } else {
            unsafe {
                RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                    client,
                    input.clone(),
                    gamma.clone(),
                    output.clone(),
                    p,
                    dim,
                    eps,
                );
            }
        }
    }

    /// Batched prefill — process a whole prompt with each projection collapsed
    /// into one dispatch (Issue 637 T3).
    ///
    /// Returns the logits for the **final** prompt token, and leaves the
    /// forward positioned so `forward_token` can continue decoding from there:
    /// `self.x` holds the last token's hidden state and `self.pos == tokens.len()`.
    ///
    /// # What is batched and what is not
    ///
    /// Batched into one dispatch per layer: the input projections (qkv, z, a, b),
    /// the out-projection, and the FFN gate/up/down — plus every purely
    /// elementwise step, which batches for free by widening `n` (`ResidualAdd`,
    /// SwiGLU gating, z-gating) or by widening the row count (`RmsNormBatched`).
    ///
    /// Kept sequential, one dispatch per token, because they carry state across
    /// the token axis and cannot be reordered:
    /// - **conv1d** — slides `conv_state`
    /// - **the DeltaNet recurrence** — carries the recurrent `state`
    /// - **beta/decay and expand+L2-norm** — token-local; batched variants
    ///   exist (`DeltanetBetaDecayBatchedCubeCL` /
    ///   `ExpandAndL2NormalizeHeadsBatchedCubeCL`, Issue 637 T5) and dispatch
    ///   under [`PREFILL_BATCH_ELEMENTWISE`] — the per-token fallback below is
    ///   the structure-vs-numerics control
    /// - **whole attention layers** — RoPE + KV-cache append are position-indexed
    ///   (a batched variant exists behind [`PREFILL_ATTENTION_BATCHED`]; the
    ///   sequential fallback is the control)
    ///
    /// No copies are needed for the sequential steps: each reads its token's
    /// slice of the batched buffer via `Handle::offset_start`, which is why
    /// `OFFSET_ALIGN_F32` above is load-bearing.
    ///
    /// # Expect a small gain, not the 8.38× prefill gap
    ///
    /// Bench 641 measured the batched GEMM at **1.08×** over P sequential GEMVs
    /// on the projection roll-up (1.11× shape-adaptive). The wins concentrate in
    /// under-occupied shapes; `ffn_gate/up` and `ffn_down` are 63% of projection
    /// time and are already compute-saturated. This path inherits that ceiling —
    /// it does not close the gap to llama.cpp, and is opt-in for that reason.
    #[cfg(feature = "ternary_gemm_batched")]
    pub fn prefill(&mut self, tokens: &[usize]) -> Vec<f32> {
        self.prefill_with_layer_capture(tokens, 0, None)
    }

    /// Total wgpu MSL passthrough dispatch count (Issue 657 profiling).
    #[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
    pub fn metal_wgpu_dispatch_count(&self) -> u64 {
        self.metal_wgpu_gemm.as_ref().map_or(0, |g| g.dispatch_count())
    }

    /// Diagnostic variant of [`Self::prefill`] that captures the hidden state of
    /// ONE token after each layer's FFN residual add — the same tap point
    /// [`Self::forward_token_with_layer_capture`] uses, so the two are directly
    /// diffable layer by layer.
    ///
    /// Exists to bisect the Bench 642 G1 failure. Token 0 at P=2 is the useful
    /// probe: it is causally independent of token 1, so its per-layer states
    /// MUST match the sequential path exactly. The first layer where they
    /// diverge localizes the defect.
    ///
    /// Forces a GPU sync per layer — diagnostic only, never on a hot path.
    ///
    /// Issue 734 chunked prefill: when `tokens.len() > prefill_chunk_max()`, the
    /// prompt is processed in ≤chunk_max-token slices (see
    /// [`PREFILL_CHUNK_MAX_OVERRIDE`]) — the 24 GB 4090's 16K/32K working-set
    /// fix (Bench 709). `capture_token` is an ABSOLUTE token index; the capture
    /// routes to the chunk that owns it.
    #[cfg(feature = "ternary_gemm_batched")]
    pub fn prefill_with_layer_capture(
        &mut self,
        tokens: &[usize],
        capture_token: usize,
        mut capture: Option<&mut [Vec<f32>]>,
    ) -> Vec<f32> {
        let chunk_max = prefill_chunk_max();
        if tokens.len() <= chunk_max {
            self.active_prefill_len = tokens.len();
            return self.prefill_tokens_chunk(tokens, 0, true, CaptureRows::One(capture_token), capture, None);
        }

        // ── Multi-chunk driver (Issue 734, Bench 709): ≤chunk_max tokens per
        //    call bounds the P-width scratch to the measured-clean region
        //    while GDN recurrent/conv state (persistent handles, updated in
        //    place) and the attention KV cache (absolute-position fill) carry
        //    across chunks. Only the final chunk pays the tail (final norm +
        //    lm_head + logits read); non-final chunks return empty and rely on
        //    same-stream in-order execution (the Batch-21 implicit-ordering
        //    rule — no inter-chunk sync needed).
        self.active_prefill_len = tokens.len();
        let mut logits = Vec::new();
        let mut base = 0usize;
        let mut rest = tokens;
        while !rest.is_empty() {
            let p = rest.len().min(chunk_max);
            let chunk = &rest[..p];
            let is_final = p == rest.len();
            let cap = if capture.is_some() && capture_token >= base && capture_token < base + p {
                capture.as_deref_mut()
            } else {
                None
            };
            let local_capture = capture_token.saturating_sub(base);
            let out =
                self.prefill_tokens_chunk(chunk, base, is_final, CaptureRows::One(local_capture), cap, None);
            if is_final {
                logits = out;
            }
            base += p;
            rest = &rest[p..];
        }
        logits
    }

    /// Capture **every** position's per-layer hidden state in ONE prefill.
    ///
    /// `capture[layer_idx]` receives `P * n_embd` f32 in position-major order —
    /// row `t` is `[t * n_embd .. (t + 1) * n_embd]`, `P = tokens.len()`.
    /// Buffers are CLEARED first, so a caller may reuse them across samples.
    ///
    /// This is the `O(P)` replacement for calling
    /// [`Self::prefill_with_layer_capture`] once per position, which is `O(P²)`:
    /// that tap already reads the whole `[P, n]` activation back to the host for
    /// every layer and then keeps a single row, so `P - 1` rows per layer were
    /// being discarded after being paid for. Same tap, same values, same syncs —
    /// only the CPU-side slice changes.
    ///
    /// `capture` is a **PREFIX selector, not a subset selector.** The tap writes
    /// `buf[layer_idx]` at the *absolute* layer index, so `buf.len() == k`
    /// captures layers `0..k` and nothing else. Passing a compacted buffer for a
    /// strided layer map (say 16 entries for layers 0, 4, 8, …, 60) does NOT
    /// select those layers — it silently captures layers `0..16`. A consumer
    /// sampling strided layers must size the buffer to
    /// `max_wanted_layer + 1` and index it by absolute layer.
    ///
    /// Host cost is therefore `(max_wanted_layer + 1) * P * n_embd * 4` bytes,
    /// not `n_wanted * P * n_embd * 4`; at Bonsai's `n_embd = 5120` that is
    /// ~20 KB per layer per position, so a strided map reaching layer 63 costs
    /// the full ~1.31 MB per position regardless of how few layers it wants.
    #[cfg(feature = "ternary_gemm_batched")]
    pub fn prefill_with_all_positions_capture(
        &mut self,
        tokens: &[usize],
        mut capture: Option<&mut [Vec<f32>]>,
    ) -> Vec<f32> {
        if let Some(buf) = capture.as_deref_mut() {
            for b in buf.iter_mut() {
                b.clear();
            }
        }
        let chunk_max = prefill_chunk_max();
        if tokens.len() <= chunk_max {
            self.active_prefill_len = tokens.len();
            return self.prefill_tokens_chunk(tokens, 0, true, CaptureRows::All, capture, None);
        }
        self.active_prefill_len = tokens.len();
        let mut logits = Vec::new();
        let mut base = 0usize;
        let mut rest = tokens;
        while !rest.is_empty() {
            let p = rest.len().min(chunk_max);
            let chunk = &rest[..p];
            let is_final = p == rest.len();
            // Unlike the single-position driver, EVERY chunk carries the tap —
            // the rows accumulate in ascending `base_pos`.
            let out = self.prefill_tokens_chunk(
                chunk,
                base,
                is_final,
                CaptureRows::All,
                capture.as_deref_mut(),
                None,
            );
            if is_final {
                logits = out;
            }
            base += p;
            rest = &rest[p..];
        }
        logits
    }

    /// **Issue 545 / riir-train Plan 415 — prefill with the attention-mass tap.**
    ///
    /// One prefill of `tokens`; at every attention layer listed in `spec.layers`,
    /// after the batched attention stage, the tap reads back the layer's
    /// `q_b` rows at `spec.row_positions` (post-QK-norm post-RoPE queries,
    /// `[n_head, hd]`-per-token layout), the layer's K-cache valid prefix
    /// (`[pos, kvd]` row-major) and its V-cache valid prefix (same layout)
    /// into `AttnMassTapCapture`. Passive readback:
    /// no kernel sequence or math change (the tap-off/tap-on logits are
    /// bit-identical — the G3 pin), but it forces the CubeCL prefill body
    /// (the whole-prefill cudarc lane refuses while a tap is armed) and
    /// requires the batched attention path — an armed tap that produced no
    /// rows REFUSES here rather than returning a silently-empty readout.
    ///
    /// Chunked prefills (`tokens.len() > prefill_chunk_max()`, default 4096)
    /// are driven chunk-by-chunk: each row routes to the chunk that owns its
    /// absolute position, and K prefixes concatenate in position order — the
    /// 32K-ladder shape (Issue 452) is supported from day one.
    ///
    /// Dataset-generation tool (the FMID v2 regen lane), never a hot path.
    #[cfg(feature = "attn_mass_tap")]
    pub fn prefill_with_attn_mass_tap(
        &mut self,
        tokens: &[usize],
        spec: &AttnMassTapSpec,
    ) -> (Vec<f32>, AttnMassTapCapture) {
        assert!(!tokens.is_empty(), "prefill requires at least one token");
        for w in spec.layers.windows(2) {
            assert!(w[0] < w[1], "AttnMassTapSpec.layers must be strictly ascending");
        }
        for w in spec.row_positions.windows(2) {
            assert!(
                w[0] < w[1],
                "AttnMassTapSpec.row_positions must be strictly ascending"
            );
        }
        for &t in &spec.row_positions {
            assert!(
                t < tokens.len(),
                "AttnMassTapSpec row position {t} beyond prompt length {}",
                tokens.len()
            );
        }
        for &l in &spec.layers {
            assert!(
                l < self.config.n_layer
                    && self.layer_types[l] == DeltaNetLayerType::Attention,
                "AttnMassTapSpec layer {l} is not an attention layer"
            );
        }
        let mut cap = AttnMassTapCapture {
            q_rows: vec![Vec::new(); spec.layers.len()],
            k_prefix: vec![Vec::new(); spec.layers.len()],
            v_prefix: vec![Vec::new(); spec.layers.len()],
        };
        let chunk_max = prefill_chunk_max();
        let mut logits = Vec::new();
        let mut base = 0usize;
        let mut rest = tokens;
        while !rest.is_empty() {
            let p = rest.len().min(chunk_max);
            let chunk = &rest[..p];
            let is_final = p == rest.len();
            let out = self.prefill_tokens_chunk(
                chunk,
                base,
                is_final,
                CaptureRows::All,
                None,
                Some((spec, &mut cap)),
            );
            if is_final {
                logits = out;
            }
            base += p;
            rest = &rest[p..];
        }
        let q_dim = self.config.n_head * self.config.head_dim;
        let n_rows = spec.row_positions.len();
        for (ti, q) in cap.q_rows.iter().enumerate() {
            assert_eq!(
                q.len(),
                n_rows * q_dim,
                "attn_mass_tap layer {} produced {} q floats for {n_rows} rows \
                 — the batched attention prefill path was not taken \
                 (PREFILL_ATTENTION_BATCHED knob or feature state)",
                spec.layers[ti],
                q.len()
            );
        }
        (logits, cap)
    }

    /// Issue 994 / riir-train Issue 452 T4 — read back EVERY uploaded weight
    /// buffer and BLAKE3 it ([`crate::weight_readback::manifest`]). The
    /// ≥52K zero-failure corruption instrument: weight buffers are
    /// model-static, so a manifest taken at a known-good block size is the
    /// ground truth a suspect-class construction must reproduce bit-for-bit
    /// — and a manifest taken before vs after a prefill catches mid-forward
    /// weight overwrites. Diagnostic tooling, never a hot path (one full
    /// weight readback + hash per call).
    #[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
    pub fn weight_readback_manifest(&self) -> Vec<crate::weight_readback::WeightBufferEntry> {
        crate::weight_readback::manifest(self)
    }

    /// Issue 994 / riir-train Issue 452 T4 — read back the persistent
    /// semantic state and BLAKE3 it ([`crate::state_readback::manifest`]):
    /// per-DeltaNet-layer recurrent + conv state, per-attention-layer KV
    /// valid prefix, the hidden handoff `x` (pos > 0), and the position
    /// counter. The state after an N-token prefill cannot legitimately
    /// depend on `config.block_size`, so a known-good-class post-prefill
    /// manifest is the ground truth a suspect-class run must reproduce
    /// bit-for-bit — and a re-run after `reset_state` catches nondeterminism
    /// or pool aliasing. Diagnostic tooling, never a hot path (one state
    /// readback + hash per call).
    #[cfg(all(feature = "cubecl_runtime", feature = "ternary_gemv"))]
    pub fn state_readback_manifest(&self) -> Vec<crate::state_readback::StateBufferEntry> {
        crate::state_readback::manifest(self)
    }

    /// One chunk of a (possibly chunked) prefill — the original
    /// [`Self::prefill_with_layer_capture`] body. `base_pos` is the chunk's
    /// first ABSOLUTE position (0 for single-chunk calls); attention RoPE,
    /// KV fill, and the causal range are offset by it. The tail (final norm +
    /// lm_head + hidden-state handoff + logits readback) runs only when
    /// `is_final_chunk`.
    #[cfg(feature = "ternary_gemm_batched")]
    #[allow(clippy::too_many_arguments, reason = "chunk driver wiring")]
    fn prefill_tokens_chunk(
        &mut self,
        tokens: &[usize],
        base_pos: usize,
        is_final_chunk: bool,
        rows: CaptureRows,
        mut capture: Option<&mut [Vec<f32>]>,
        #[cfg_attr(not(feature = "attn_mass_tap"), allow(unused_mut, unused_variables))]
        mut mass_tap: Option<(&AttnMassTapSpec, &mut AttnMassTapCapture)>,
    ) -> Vec<f32> {
        assert!(!tokens.is_empty(), "prefill requires at least one token");
        // Issue 994: entry AND exit. The entry check is the per-chunk guard
        // in multi-chunk mode (every public prefill drives its chunks through
        // here), catching a mid-prefill poisoning at the earliest chunk
        // boundary instead of after 400 s of wasted compute; the exit check
        // keeps a poisoning that happened DURING this chunk from leaving as
        // a successful return.
        Self::refuse_if_pool_poisoned("at prefill chunk entry");
        if self.layer_types.contains(&DeltaNetLayerType::Attention) {
            assert!(
                base_pos + tokens.len() <= self.config.block_size,
                "prefill chunk [{base_pos}, {}) exceeds block_size {} — the KV \
                 caches are allocated at block_size rows (raise config.block_size \
                 to cover the full prompt)",
                base_pos + tokens.len(),
                self.config.block_size,
            );
        }

        // ── Issue 734 Arm 8: the whole-prefill cudarc migration ──
        // ZERO per-layer host crossings (tokens in → logits out). Falls
        // through to the CubeCL body below on any failure — bit-safe (both
        // paths compute the same values). Capture taps are diagnostic —
        // the arm does not support them.
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            feature = "ternary_gemm_batched",
            feature = "cubecl_runtime",
            not(target_os = "macos")
        ))]
        // Issue 545 / riir-train Plan 415: an armed attention-mass tap also
        // refuses the cudarc whole-prefill lane — the tap reads the CubeCL
        // body's batched-attention scratch (q_b) + KV handles, which the
        // cudarc lane does not materialize on this path. Same silent-bypass
        // class as a capture: refuse loudly (the lane stays available for
        // every capture-free call).
        //
        // The block wrapper carries the arm-8 cfg: inserting the tap lets
        // BETWEEN the attribute and the `if` detached the attribute onto a
        // let, ungating the `prefill_cuda_full` references on macOS and
        // without `ternary_gemv_cuda_raw` (E0433 on every such build).
        {
            #[cfg(feature = "attn_mass_tap")]
            let tap_blocks_cuda_lane = mass_tap.is_some();
            #[cfg(not(feature = "attn_mass_tap"))]
            let tap_blocks_cuda_lane = false;
            if capture.is_none() && !tap_blocks_cuda_lane {
                if let Some(logits) =
                    crate::prefill_cuda_full::try_whole_prefill_cuda(self, tokens, base_pos, is_final_chunk)
                {
                    self.pos = base_pos + tokens.len();
                    return logits;
                }
                // Issue 965 fall-through flush: a graph-armed chunk elides the
                // per-chunk state writeback (spec discipline — the cudarc mirrors
                // hold the truth), so ANY None return here — an early refusal,
                // an alloc failure, or the arm switched off after graphed chunks —
                // must bring the CubeCL handles current BEFORE the CubeCL body
                // below reads them. O(1) atomic swap when nothing is pending.
                crate::prefill_cuda_full::prefill_spec_flush(self);
                if std::env::var("RIIR_PREFILL_CUDA_TRACE").is_ok_and(|s| matches!(s.trim(), "1" | "true" | "on"))
                {
                    eprintln!(
                        "[734-arm8-wiring] arm returned None (knob={} gate={})",
                        crate::prefill_cuda_full::prefill_use_cuda_pub(),
                        prefill_cuda_gate_ok(),
                    );
                }
            }
        }

        // Issue 980 T4-ALT — fall-through ERROR semantics for folded models:
        // the CubeCL body below CANNOT compute a Hadamard-folded file (it
        // would run folded weights unrotated — silent garbage), so a folded
        // model whose cudarc whole-prefill lane refused (or is compiled out)
        // must ERROR here, never fall through. Reaching this line with the
        // marker set means the arm returned None or the cudarc features are
        // off — either way the prefill is not computable on this lane.
        if self.rotation.is_some() {
            panic!(
                "Bonsai-2 Hadamard-folded prefill: the cudarc whole-prefill lane \
                 refused or is unavailable (knob/gate/features — see \
                 RIIR_PREFILL_CUDA_TRACE=1) and the CubeCL fallback cannot \
                 compute folded weights unrotated (Issue 980 T4-ALT)"
            );
        }

        // Arm 13 — the CubeCL body below mutates the state handles; if the
        // cudarc arm is off / fell through, its mirrors are now stale.
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            feature = "ternary_gemm_batched",
            feature = "cubecl_runtime",
            not(target_os = "macos")
        ))]
        crate::prefill_cuda_full::notify_cubcl_mutated();

        let p = tokens.len();
        let n = self.config.n_embd;
        let eps = self.config.rms_norm_eps as f32;
        let n_v_heads = self.config.deltanet_linear_n_value_heads;
        let head_dim = self.config.deltanet_linear_head_dim;
        let n_k_heads = self.config.deltanet_linear_n_heads;
        let q_dim = n_k_heads * head_dim;
        let v_dim = n_v_heads * head_dim;
        let qkv_dim = 2 * q_dim + v_dim;
        let qkvx_dim = 3 * v_dim;
        let z_dim = v_dim;
        let conv_dim = qkv_dim;
        let kernel_size = self.config.deltanet_conv_kernel_size;
        let mlp = self.config.mlp_hidden;
        let f = core::mem::size_of::<f32>();

        // Every per-token stride must satisfy the binding-offset alignment, or
        // the slicing below trips wgpu validation at dispatch time.
        for (name, stride) in [
            ("n_embd", n),
            ("qkv_dim", qkv_dim),
            ("qkv_expanded", qkvx_dim),
            ("v_dim", v_dim),
            ("n_v_heads", n_v_heads),
            ("mlp_hidden", mlp),
        ] {
            assert!(
                stride.is_multiple_of(Self::OFFSET_ALIGN_F32),
                "prefill: {name} stride {stride} f32 is not a multiple of \
                 {} — per-token buffer slicing would violate the wgpu \
                 storage-buffer binding offset alignment",
                Self::OFFSET_ALIGN_F32
            );
        }

        // ── P-width scratch. Allocated per prefill call (once per prompt), so
        //    the decode path's alloc-free steady state is untouched. ──
        let mut x_b = self.client.empty(p * n * f);
        let pingpong = PREFILL_PINGPONG_RESIDUAL.load(std::sync::atomic::Ordering::Relaxed);
        let dealias_norm = PREFILL_DEALIAS_NORM.load(std::sync::atomic::Ordering::Relaxed);
        // Issue 640 aliasing A/B. The alternates are full-size whenever a
        // de-aliasing arm needs them, and also on demand via
        // [`PREFILL_ALLOC_ALT`] — that extra knob exists so the experiment can
        // separate "two more P-width allocations" from "the bindings no longer
        // alias". Bench 646 showed allocation-pattern changes move the failure
        // rate on their own, so without an alloc-only control the A/B would be
        // confounded. When nothing needs them they stay 4 bytes, which keeps the
        // true baseline byte-for-byte comparable to the pre-change path.
        let alloc_alt = pingpong
            || dealias_norm
            || PREFILL_ALLOC_ALT.load(std::sync::atomic::Ordering::Relaxed);
        let mut x_alt = if alloc_alt {
            self.client.empty(p * n * f)
        } else {
            self.client.empty(f)
        };
        let mut rec_alt = if alloc_alt {
            self.client.empty(p * v_dim * f)
        } else {
            self.client.empty(f)
        };
        let normx_b = self.client.empty(p * n * f);
        // Issue 658 Phase 2: when chunked conv1d is on, the conv1d step writes SiLU
        // outputs to a SEPARATE buffer (the chunked kernel cannot be in-place
        // because token t needs the RAW input of tokens t-1..t-3). After all
        // chunks, qkv_b is swapped with qkv_conv_b so the subsequent expand/L2-norm
        // step reads the SiLU outputs from qkv_b. Only allocated when chunked
        // conv1d is enabled; otherwise stays 4 bytes (the alloc-free baseline).
        #[cfg(all(
            feature = "cubecl_runtime",
            feature = "ternary_gemm_batched",
            feature = "ternary_deltanet_chunked_prefill"
        ))]
        let chunked_conv1d_on =
            PREFILL_CHUNKED_CONV1D.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(all(
            feature = "cubecl_runtime",
            feature = "ternary_gemm_batched",
            feature = "ternary_deltanet_chunked_prefill"
        )))]
        let chunked_conv1d_on = false;
        // `mut` + `qkv_conv_b` are only consumed when `chunked_conv1d_on` (runtime +
        // compile-time feature gate). `#[allow]` silences the default-config warnings.
        #[allow(unused_mut)]
        let mut qkv_b = self.client.empty(p * qkv_dim * f);
        #[allow(unused_mut, unused_variables)]
        let mut qkv_conv_b = if chunked_conv1d_on {
            self.client.empty(p * qkv_dim * f)
        } else {
            self.client.empty(f)
        };
        let qkvx_b = self.client.empty(p * qkvx_dim * f);
        let z_b = self.client.empty(p * z_dim * f);
        let a_b = self.client.empty(p * n_v_heads * f);
        let b_b = self.client.empty(p * n_v_heads * f);
        let beta_b = self.client.empty(p * n_v_heads * f);
        let decay_b = self.client.empty(p * n_v_heads * f);
        let mut rec_b = self.client.empty(p * v_dim * f);
        let tmp_b = self.client.empty(p * n * f);
        let gate_b = self.client.empty(p * mlp * f);
        let up_b = self.client.empty(p * mlp * f);
        let hid_b = self.client.empty(p * mlp * f);
        let ffnout_b = self.client.empty(p * n * f);

        // ── Plan 533: chunked prefill scratch (allocated once per prefill call).
        //    All buffers are sized for the full chunk size C, reused across all
        //    layers + chunks. The largest is delta_s at n_v_heads * d * d.
        #[cfg(all(
            feature = "cubecl_runtime",
            feature = "ternary_gemm_batched",
            feature = "ternary_deltanet_chunked_prefill"
        ))]
        // Issue 734 T4: the multi-token recurrence needs NO scratch (one
        // dispatch over the token-major buffers; state updated in place).
        let _ = &qkvx_dim;

        // ── Issue 653: batched attention scratch (allocated once per prefill call).
        //    All buffers are sized for P tokens, reused across all attention layers.
        #[cfg(all(
            feature = "cubecl_runtime",
            feature = "ternary_gemm_batched",
            feature = "ternary_attention_batched_prefill"
        ))]
        let attn_scratch = {
            #[cfg(all(
                feature = "cubecl_runtime",
                feature = "ternary_gemm_batched",
                feature = "ternary_attention_batched_prefill"
            ))]
            let batched_attn_wanted =
                PREFILL_ATTENTION_BATCHED.load(std::sync::atomic::Ordering::Relaxed);
            #[cfg(not(all(
                feature = "cubecl_runtime",
                feature = "ternary_gemm_batched",
                feature = "ternary_attention_batched_prefill"
            )))]
            let batched_attn_wanted = false;
            if batched_attn_wanted {
                let n_head = self.config.n_head;
                let n_kv = self.config.n_kv_head;
                let hd = self.config.head_dim;
                let q_dim = n_head * hd;
                let kvd = n_kv * hd;
                Some(BatchedAttentionScratch {
                    qg_b: self.client.empty(p * 2 * q_dim * f),
                    q_b: self.client.empty(p * q_dim * f),
                    gate_b: self.client.empty(p * q_dim * f),
                    kv_b: self.client.empty(p * 2 * kvd * f),
                    k_b: self.client.empty(p * kvd * f),
                    v_b: self.client.empty(p * kvd * f),
                    attn_out_b: self.client.empty(p * q_dim * f),
                    out_proj_b: self.client.empty(p * n * f),
                })
            } else {
                None
            }
        };

        // Issue 640 T3 probe: see [`PREFILL_ZERO_SCRATCH`]. `client.empty()`
        // hands back recycled pool memory, so this distinguishes "reads a region
        // it never wrote" from every other cause of the P=128 variance.
        let zero_mask = PREFILL_ZERO_SCRATCH.load(std::sync::atomic::Ordering::Relaxed);
        if zero_mask != 0 {
            // Order MUST match `PREFILL_SCRATCH_NAMES` — the bisect reports by
            // bit index and a mismatch would name the wrong buffer.
            for (bit, (h, len)) in [
                (&x_b, p * n),
                (&normx_b, p * n),
                (&qkv_b, p * qkv_dim),
                (&qkvx_b, p * qkvx_dim),
                (&z_b, p * z_dim),
                (&a_b, p * n_v_heads),
                (&b_b, p * n_v_heads),
                (&beta_b, p * n_v_heads),
                (&decay_b, p * n_v_heads),
                (&rec_b, p * v_dim),
                (&tmp_b, p * n),
                (&gate_b, p * mlp),
                (&up_b, p * mlp),
                (&hid_b, p * mlp),
                (&ffnout_b, p * n),
            ]
            .into_iter()
            .enumerate()
            {
                if zero_mask & (1u32 << bit) != 0 {
                    unsafe {
                        FillZerosCubeCL::launch::<ActiveRuntime>(&self.client, h.clone(), len);
                    }
                }
            }
        }

        // ── Embed every prompt token into its row of x_b ──
        let wte_blocks64 = self.wte_handle.blocks64 as u32;
        let wte_groups = self.wte_handle.groups_per_row as u32;
        for (t, &tok) in tokens.iter().enumerate() {
            unsafe {
                DequantWteRowCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    self.wte_handle.pos_bits_u32.clone(),
                    self.wte_handle.neg_bits_u32.clone(),
                    self.wte_handle.group_scale_f32.clone(),
                    Self::tok_slice(&x_b, t, n, p),
                    tok as u32,
                    wte_blocks64,
                    wte_groups,
                    n as u32,
                );
            }
        }

        for layer_idx in 0..self.layers.len() {
            let is_deltanet = self.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;

            // Issue 640 strip-down: a disabled stage class contributes nothing to
            // the residual stream, which stays valid either way.
            if is_deltanet && !stage_on(0) {
                // deltanet disabled
            } else if !is_deltanet && !stage_on(1) {
                // attention disabled
            } else if is_deltanet {
                let layer_w = &self.layers[layer_idx];

                Self::prefill_norm(
                    &self.client,
                    &x_b,
                    &layer_w.input_norm,
                    &normx_b,
                    p,
                    n,
                    eps,
                );

                // 1-4. Input projections — one dispatch each instead of P.
                //
                // Issue 726 T2: ANE hybrid seam. When the eligibility gate
                // fires (feature on + ctx ready + p ≥ 2048 — never in T2),
                // the fused in_proj_concat runs on the ANE and writes the
                // qkv/z/a/b output segments directly. Every other path
                // falls through to the split batched GEMMs, unchanged.
                #[cfg(feature = "ane_prefill")]
                let ane_inproj_done =
                    self.ane_prefill_try_inproj(layer_idx, &normx_b, &qkv_b, &z_b, &a_b, &b_b, p);
                #[cfg(not(feature = "ane_prefill"))]
                let ane_inproj_done = false;
                if !ane_inproj_done {
                    for (w, out) in [
                        (&layer_w.in_proj_qkv, &qkv_b),
                        (&layer_w.in_proj_z, &z_b),
                        (&layer_w.in_proj_a, &a_b),
                        (&layer_w.in_proj_b, &b_b),
                    ] {
                        self.prefill_project(w, &normx_b, out, p);
                    }
                }

                // 5-8. Per-token chain. conv1d and the recurrence carry state,
                //      so the token order here is load-bearing.
                let conv_state = self.conv_states[layer_idx]
                    .as_ref()
                    .expect("conv_state for DeltaNet layer")
                    .clone();
                let state = self.deltanet_states[layer_idx]
                    .as_ref()
                    .expect("state for DeltaNet layer")
                    .clone();

                // 5. conv1d. Sequential (per-token) or chunked (C tokens/dispatch).
                //    The chunked kernel is G1-verified (deltanet_chunked_cubecl.rs
                //    tests) and reduces P dispatches to ceil(P/C). It uses
                //    conv_state directly as the carry buffer (stride=ks, offset=1),
                //    so no carry-format translation is needed — the chunked kernel
                //    reads/writes conv_state positions 1..ks-1, leaving position 0
                //    untouched (stale, shifted out on the next decode before use).
                //
                //    The chunked kernel writes to a SEPARATE output buffer
                //    (`qkv_conv_b`) because token t needs the RAW input of tokens
                //    t-1..t-3 (in-place would read SiLU'd values). After all chunks,
                //    `qkv_b` is swapped with `qkv_conv_b` so the subsequent
                //    expand/L2-norm step reads SiLU outputs from qkv_b.
                //
                //    Constraint: P must be evenly divisible by C. For partial last
                //    chunks, the sequential fallback writes SiLU to qkv_b (not
                //    qkv_conv_b), and the swap would leave garbage in those tokens.
                //    A copy kernel would fix this but adds complexity for an edge
                //    case that doesn't arise in production (P=128, C=64). When P is
                //    not divisible by C, we fall back to the full sequential path.
                #[cfg(all(
                    feature = "cubecl_runtime",
                    feature = "ternary_gemm_batched",
                    feature = "ternary_deltanet_chunked_prefill"
                ))]
                let can_chunk_conv1d =
                    chunked_conv1d_on && p.is_multiple_of(PREFILL_CHUNK_SIZE);
                #[cfg(not(all(
                    feature = "cubecl_runtime",
                    feature = "ternary_gemm_batched",
                    feature = "ternary_deltanet_chunked_prefill"
                )))]
                let can_chunk_conv1d = false;

                if can_chunk_conv1d {
                    #[cfg(all(
                        feature = "cubecl_runtime",
                        feature = "ternary_gemm_batched",
                        feature = "ternary_deltanet_chunked_prefill"
                    ))]
                    {
                        let chunk_size = PREFILL_CHUNK_SIZE;
                        for chunk_start in (0..p).step_by(chunk_size) {
                            let chunk_end = chunk_start + chunk_size;
                            // P % C == 0 guaranteed by can_chunk_conv1d, so no
                            // partial chunk — every chunk is full.
                            let qkv_in_chunk = qkv_b
                                .clone()
                                .offset_start((chunk_start * qkv_dim * f) as u64)
                                .offset_end(((p - chunk_end) * qkv_dim * f) as u64);
                            let qkv_out_chunk = qkv_conv_b
                                .clone()
                                .offset_start((chunk_start * qkv_dim * f) as u64)
                                .offset_end(((p - chunk_end) * qkv_dim * f) as u64);
                            unsafe {
                                DeltanetChunkedConv1dCubeCL::launch::<ActiveRuntime>(
                                    &self.client,
                                    qkv_in_chunk,
                                    qkv_out_chunk,
                                    layer_w.conv1d_weight.clone(),
                                    conv_state.clone(),
                                    chunk_size,
                                    conv_dim,
                                    kernel_size,
                                    kernel_size, // carry_stride = ks (conv_state layout)
                                    1,           // carry_idx_offset = 1 (skip position 0)
                                );
                            }
                        }
                        // Swap so qkv_b holds the SiLU outputs for the expand step.
                        std::mem::swap(&mut qkv_b, &mut qkv_conv_b);
                    }
                } else {
                    // Sequential conv1d path (default + partial-chunk fallback).
                    for t in 0..p {
                        unsafe {
                            DeltanetConv1dCubeCL::launch::<ActiveRuntime>(
                                &self.client,
                                Self::tok_slice(&qkv_b, t, qkv_dim, p),
                                layer_w.conv1d_weight.clone(),
                                conv_state.clone(),
                                conv_dim,
                                kernel_size,
                            );
                        }
                    }
                }

                // 6-7. beta/decay and head-expansion. Both are elementwise over
                //      their own token's data — beta/decay reads only the `a_b` /
                //      `b_b` projections (independent of conv1d), expansion reads
                //      `qkv_b` after conv1d has written every token above. See
                //      [`PREFILL_BATCH_ELEMENTWISE`].
                if PREFILL_BATCH_ELEMENTWISE.load(std::sync::atomic::Ordering::Relaxed) {
                    unsafe {
                        DeltanetBetaDecayBatchedCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            a_b.clone(),
                            b_b.clone(),
                            layer_w.a_log.clone(),
                            layer_w.dt_bias.clone(),
                            beta_b.clone(),
                            decay_b.clone(),
                            n_v_heads,
                            p,
                        );
                    }
                    unsafe {
                        ExpandAndL2NormalizeHeadsBatchedCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            qkv_b.clone(),
                            qkvx_b.clone(),
                            n_k_heads,
                            n_v_heads,
                            head_dim,
                            p,
                        );
                    }
                } else {
                    for t in 0..p {
                        unsafe {
                            DeltanetBetaDecayCubeCL::launch::<ActiveRuntime>(
                                &self.client,
                                Self::tok_slice(&a_b, t, n_v_heads, p),
                                Self::tok_slice(&b_b, t, n_v_heads, p),
                                layer_w.a_log.clone(),
                                layer_w.dt_bias.clone(),
                                Self::tok_slice(&beta_b, t, n_v_heads, p),
                                Self::tok_slice(&decay_b, t, n_v_heads, p),
                                n_v_heads,
                            );
                        }
                        unsafe {
                            ExpandAndL2NormalizeHeadsCubeCL::launch::<ActiveRuntime>(
                                &self.client,
                                Self::tok_slice(&qkv_b, t, qkv_dim, p),
                                Self::tok_slice(&qkvx_b, t, qkvx_dim, p),
                                n_k_heads,
                                n_v_heads,
                                head_dim,
                            );
                        }
                    }
                }

                // 8. The recurrence, per token — carries `state`.
                #[cfg(all(
                    feature = "cubecl_runtime",
                    feature = "ternary_gemm_batched",
                    feature = "ternary_deltanet_chunked_prefill"
                ))]
                let chunked_enabled = PREFILL_CHUNKED.load(std::sync::atomic::Ordering::Relaxed)
                    && DeltanetRecurrenceMultiTokenCubeCL::supports(head_dim);
                #[cfg(not(all(
                    feature = "cubecl_runtime",
                    feature = "ternary_gemm_batched",
                    feature = "ternary_deltanet_chunked_prefill"
                )))]
                let chunked_enabled = false;

                if chunked_enabled {
                    // ── Multi-token recurrence (Issue 734 T4) ──
                    // ONE dispatch per layer replaces p sequential rowpar
                    // dispatches — same per-token arithmetic in the same order
                    // (bit-identical output), no extract/scratch plumbing, no
                    // GPU-side inter-kernel gaps. Reads the token-major
                    // qkvx_b/beta_b/decay_b directly, writes rec_b token-major,
                    // updates the persistent state handle in place.
                    #[cfg(all(
                        feature = "cubecl_runtime",
                        feature = "ternary_gemm_batched",
                        feature = "ternary_deltanet_chunked_prefill"
                    ))]
                    unsafe {
                        DeltanetRecurrenceMultiTokenCubeCL::launch_with_gpu_handles::<ActiveRuntime>(
                            &self.client,
                            qkvx_b.clone(),
                            beta_b.clone(),
                            decay_b.clone(),
                            state.clone(),
                            rec_b.clone(),
                            n_v_heads,
                            head_dim,
                            p,
                        );
                    }
                    // Without `ternary_deltanet_chunked_prefill` the dual-cfg
                    // `chunked_enabled` binding above compiles to `false`, so
                    // this arm is unreachable — the block must still compile
                    // without the (feature-gated) type.
                    #[cfg(not(all(
                        feature = "cubecl_runtime",
                        feature = "ternary_gemm_batched",
                        feature = "ternary_deltanet_chunked_prefill"
                    )))]
                    unreachable!();
                } else {
                    // ── Sequential recurrence (the existing path) ──
                    for t in 0..p {
                        #[allow(unused_mut, reason = "only mutated when deltanet_recurrence_rowpar is on")]
                        let mut dispatched = false;
                        #[cfg(feature = "deltanet_recurrence_rowpar")]
                        if recurrence_rowpar_enabled()
                            && DeltanetRecurrenceRowParCubeCL::supports(head_dim)
                        {
                            unsafe {
                                DeltanetRecurrenceRowParCubeCL::launch_with_gpu_handles::<ActiveRuntime>(
                                    &self.client,
                                    Self::tok_slice(&qkvx_b, t, qkvx_dim, p),
                                    Self::tok_slice(&beta_b, t, n_v_heads, p),
                                    Self::tok_slice(&decay_b, t, n_v_heads, p),
                                    state.clone(),
                                    Self::tok_slice(&rec_b, t, v_dim, p),
                                    n_v_heads,
                                    head_dim,
                                );
                            }
                            dispatched = true;
                        }
                        if !dispatched {
                            unsafe {
                                DeltanetRecurrenceCubeCL::launch_with_gpu_handles::<ActiveRuntime>(
                                    &self.client,
                                    Self::tok_slice(&qkvx_b, t, qkvx_dim, p),
                                    Self::tok_slice(&beta_b, t, n_v_heads, p),
                                    Self::tok_slice(&decay_b, t, n_v_heads, p),
                                    state.clone(),
                                    Self::tok_slice(&rec_b, t, v_dim, p),
                                    n_v_heads,
                                    head_dim,
                                );
                            }
                        }
                    }
                }

                // 9. Per-head RMSNorm over all tokens at once: the single-token
                //    call normalizes n_v_heads rows of head_dim, so P tokens is
                //    just p * n_v_heads rows of the same width.
                unsafe {
                    RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        rec_b.clone(),
                        layer_w.linear_norm.clone(),
                        if dealias_norm { rec_alt.clone() } else { rec_b.clone() },
                        p * n_v_heads,
                        head_dim,
                        eps,
                    );
                }
                if dealias_norm {
                    std::mem::swap(&mut rec_b, &mut rec_alt);
                }
                // 10. Z gating is pure elementwise — widening n batches it.
                unsafe {
                    DeltanetZGatingCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        rec_b.clone(),
                        z_b.clone(),
                        p * z_dim,
                    );
                }
                // 11. Out projection.
                self.prefill_project(&layer_w.out_proj, &rec_b, &tmp_b, p);

                unsafe {
                    ResidualAddCubeCL::launch::<ActiveRuntime>(
                        &self.client,
                        x_b.clone(),
                        tmp_b.clone(),
                        if pingpong { x_alt.clone() } else { x_b.clone() },
                        p * n,
                    );
                }
                if pingpong {
                    std::mem::swap(&mut x_b, &mut x_alt);
                }
            } else {
                // Attention layers.
                //
                // Issue 653: when PREFILL_ATTENTION_BATCHED is on, use a single
                // batched causal-masked flash attention kernel instead of P
                // sequential decode dispatches. The batched path also batches
                // the Q/KV projections (batched GEMM), split, Q/K RMSNorm,
                // RoPE, and KV cache fill — collapsing ~P*9 dispatches per
                // attention layer to ~9 dispatches total.
                #[cfg(all(
                    feature = "cubecl_runtime",
                    feature = "ternary_gemm_batched",
                    feature = "ternary_attention_batched_prefill"
                ))]
                let batched_attn =
                    PREFILL_ATTENTION_BATCHED.load(std::sync::atomic::Ordering::Relaxed)
                        && attn_scratch.is_some();
                #[cfg(not(all(
                    feature = "cubecl_runtime",
                    feature = "ternary_gemm_batched",
                    feature = "ternary_attention_batched_prefill"
                )))]
                let batched_attn = false;

                if batched_attn {
                    #[cfg(all(
                        feature = "cubecl_runtime",
                        feature = "ternary_gemm_batched",
                        feature = "ternary_attention_batched_prefill",
                    ))]
                    {
                        self.prefill_attention_layer_batched(
                            layer_idx,
                            eps,
                            p,
                            base_pos,
                            &x_b,
                            &normx_b,
                            if pingpong { &x_alt } else { &x_b },
                            attn_scratch.as_ref().expect("attn_scratch when batched_attn"),
                        );
                    }
                    // ── Issue 545 / riir-train Plan 415: attention-mass tap ──
                    // Passive readback after the batched attention stage wrote
                    // q_b (post-QK-norm post-RoPE) and filled the KV caches.
                    // Reads the WHOLE q_b and K-cache valid prefix and slices
                    // host-side — the capture tap's own convention ("the
                    // readback path itself cannot be a confound"). Costs one
                    // read per TAPPED layer only; never on the hot path.
                    #[cfg(feature = "attn_mass_tap")]
                    if let Some((spec, cap)) = mass_tap.as_mut()
                        && let Some(ti) = spec.layers.iter().position(|&l| l == layer_idx)
                    {
                        let scratch = attn_scratch.as_ref().expect(
                            "attn_mass_tap: batched attention scratch absent — \
                             PREFILL_ATTENTION_BATCHED must be on",
                        );
                        let tap_q_dim = self.config.n_head * self.config.head_dim;
                        let tap_kvd = self.config.n_kv_head * self.config.head_dim;
                        let qall = self
                            .client
                            .read_one(scratch.q_b.clone())
                            .expect("attn_mass_tap: q_b readback");
                        let qall = f32::from_bytes(&qall);
                        for &t in &spec.row_positions {
                            if (base_pos..base_pos + p).contains(&t) {
                                let local = t - base_pos;
                                cap.q_rows[ti]
                                    .extend_from_slice(&qall[local * tap_q_dim..(local + 1) * tap_q_dim]);
                            }
                        }
                        let k_handle = self.kv_key_caches[layer_idx]
                            .as_ref()
                            .expect("attn_mass_tap: key cache for Attention layer");
                        let kall = self
                            .client
                            .read_one(k_handle.clone())
                            .expect("attn_mass_tap: K-cache readback");
                        let kall = f32::from_bytes(&kall);
                        // Append ONLY this chunk's rows [base_pos, base_pos+p):
                        // earlier chunks already contributed their prefixes —
                        // appending the whole valid prefix again would grow the
                        // buffer quadratically across chunks.
                        cap.k_prefix[ti].extend_from_slice(
                            &kall[base_pos * tap_kvd..(base_pos + p) * tap_kvd],
                        );
                        // Issue 452 T2 (the D1 lane): the V-cache readback —
                        // the same passive read + chunk-row append as K above,
                        // on the value cache the same batched stage filled. No
                        // extra sync beyond the K read's (both are reads of
                        // buffers the launched kernels already wrote).
                        let v_handle = self.kv_value_caches[layer_idx]
                            .as_ref()
                            .expect("attn_mass_tap: value cache for Attention layer");
                        let vall = self
                            .client
                            .read_one(v_handle.clone())
                            .expect("attn_mass_tap: V-cache readback");
                        let vall = f32::from_bytes(&vall);
                        cap.v_prefix[ti].extend_from_slice(
                            &vall[base_pos * tap_kvd..(base_pos + p) * tap_kvd],
                        );
                    }
                    if pingpong {
                        std::mem::swap(&mut x_b, &mut x_alt);
                    }
                } else {
                // Sequential attention path (original): RoPE and the KV-cache
                // append are position-indexed, one decode dispatch per token.
                // Reading x_b's slice as the norm input and writing the residual
                // back through the same offset avoids any copy. Positions are
                // ABSOLUTE (base_pos + t) — correct under chunked prefill.
                for t in 0..p {
                    self.pos = base_pos + t;
                    let layer_w = &self.layers[layer_idx];
                    if attn_sub_on(5) {
                    unsafe {
                        RmsNormCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            Self::tok_slice(&x_b, t, n, p),
                            layer_w.input_norm.clone(),
                            self.norm_x.clone(),
                            n,
                            eps,
                        );
                    }
                    } // sub-stage 5: outer norm

                    // ── Issue 640 Bench 657/658: per-token sync diagnostic ──
                    {
                        let k_read = PREFILL_SYNC_PER_TOKEN.load(std::sync::atomic::Ordering::Relaxed);
                        if k_read != SYNC_THROUGH_DISABLED && (t as u32).is_multiple_of(k_read) {
                            let _ = self.client.read_one(self.norm_x.clone());
                        }
                        let k_flush = PREFILL_FLUSH_PER_TOKEN.load(std::sync::atomic::Ordering::Relaxed);
                        if k_flush != SYNC_THROUGH_DISABLED && (t as u32).is_multiple_of(k_flush) {
                            let _ = self.client.flush();
                        }
                    }

                    self.forward_attention_layer_gpu(layer_idx, layer_w, eps);
                    if attn_sub_on(5) {
                    unsafe {
                        ResidualAddCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            Self::tok_slice(&x_b, t, n, p),
                            self.tmp.clone(),
                            if pingpong {
                                Self::tok_slice(&x_alt, t, n, p)
                            } else {
                                Self::tok_slice(&x_b, t, n, p)
                            },
                            n,
                        );
                    }
                    } // sub-stage 5: outer residual
                }
                // Every token was written, so the alternate is complete.
                if pingpong {
                    std::mem::swap(&mut x_b, &mut x_alt);
                }
                } // end sequential attention fallback
            }

            // ── FFN block — batched for both layer kinds ──
            if stage_on(2) {
            let layer_w = &self.layers[layer_idx];
            // Issue 734 Arm 7 (Bench 721): the cudarc FFN-block migration —
            // the whole block (norm → gate/up GEMM → swiglu → down GEMM →
            // residual) on the cudarc stream with ONE read + ONE write per
            // layer, the Bench-720 G2 structural answer (the per-GEMM round
            // trip measured 0.095-0.234×; 83% of it CubeCL staging + pipeline
            // serialization). Bit-safety: falls through to the CubeCL body on
            // any failure — both paths compute the same values (Bench-721
            // FNV-gated bit-identity incl. the rmsnorm/swiglu forms).
            #[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
            let cuda_ffn_done = crate::prefill_cuda_ffn::prefill_use_cuda_ffn()
                && p <= 4096
                && n.is_multiple_of(128)
                && mlp.is_multiple_of(128)
                && crate::prefill_cuda_ffn::dispatch_ffn_block(
                    &self.client,
                    &layer_w.gate_proj,
                    &layer_w.up_proj,
                    &layer_w.down_proj,
                    &layer_w.post_attn_norm,
                    &x_b,
                    if pingpong { &x_alt } else { &x_b },
                    p,
                    n,
                    mlp,
                    eps,
                );
            #[cfg(not(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos"))))]
            let cuda_ffn_done = false;
            if !cuda_ffn_done {
            Self::prefill_norm(
                &self.client,
                &x_b,
                &layer_w.post_attn_norm,
                &normx_b,
                p,
                n,
                eps,
            );
            // Issue 726 T2: ANE hybrid seam for the FFN projections (fused
            // gate_up_proj). Same fail-open contract as the in_proj seam
            // above; the gate additionally requires the layer to be GDN
            // (attention-layer FFNs stay GPU — the contract scopes
            // acceleration to the GDN family).
            #[cfg(feature = "ane_prefill")]
            let ane_gate_up_done =
                self.ane_prefill_try_gate_up(layer_idx, &normx_b, &gate_b, &up_b, p);
            #[cfg(not(feature = "ane_prefill"))]
            let ane_gate_up_done = false;
            if !ane_gate_up_done {
                self.prefill_project(&layer_w.gate_proj, &normx_b, &gate_b, p);
                self.prefill_project(&layer_w.up_proj, &normx_b, &up_b, p);
            }
            // SwiGLU is elementwise — widening batches it.
            unsafe {
                DeltanetGatingCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    gate_b.clone(),
                    up_b.clone(),
                    hid_b.clone(),
                    p * mlp,
                );
            }
            // Plan 549: the down_proj ANE seam (runtime-gated, per-op
            // fail-open — skipped entirely when the toggle/bank is off).
            #[cfg(feature = "ane_prefill")]
            let ane_down_done = self.ane_prefill_try_down(layer_idx, &hid_b, &ffnout_b, p);
            #[cfg(not(feature = "ane_prefill"))]
            let ane_down_done = false;
            if !ane_down_done {
                self.prefill_project(&layer_w.down_proj, &hid_b, &ffnout_b, p);
            }
            unsafe {
                ResidualAddCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    x_b.clone(),
                    ffnout_b.clone(),
                    if pingpong { x_alt.clone() } else { x_b.clone() },
                    p * n,
                );
            }
            } // end !cuda_ffn_done (CubeCL FFN body)
            if pingpong {
                std::mem::swap(&mut x_b, &mut x_alt);
            }
            } // end FFN stage class

            // ── Issue 640 partial-sync bisect (diagnostic only) ──
            // Same full readback the capture tap performs, applied to a prefix of
            // the layers. See [`PREFILL_SYNC_THROUGH_LAYER`].
            {
                let k = PREFILL_SYNC_THROUGH_LAYER.load(std::sync::atomic::Ordering::Relaxed);
                if k != SYNC_THROUGH_DISABLED && (layer_idx as u32) <= k {
                    let _ = self
                        .client
                        .read_one(x_b.clone())
                        .expect("partial-sync readback");
                }
            }

            // ── Per-layer capture tap (forces a GPU sync — diagnostic only) ──
            // Reads the whole [P, n] buffer and slices on the CPU rather than
            // binding an offset view, so the readback path itself cannot be a
            // confound in the very comparison it is meant to adjudicate.
            if let Some(buf) = capture.as_deref_mut()
                && layer_idx < buf.len()
            {
                let all = self
                    .client
                    .read_one(x_b.clone())
                    .expect("capture readback");
                let all = f32::from_bytes(&all);
                match rows {
                    CaptureRows::One(t) => {
                        buf[layer_idx] = all[t * n..(t + 1) * n].to_vec();
                    }
                    // Slice to `p * n`: the scratch is allocated at the
                    // chunk-max P width, so the buffer can be WIDER than this
                    // chunk's token count and the tail rows are stale.
                    // Appending (not assigning) is what carries a multi-chunk
                    // prefill — chunks arrive in ascending `base_pos`, so
                    // position order is preserved by construction.
                    CaptureRows::All => {
                        buf[layer_idx].extend_from_slice(&all[..p * n]);
                    }
                }
            }
        }

        // ── Tail: only the final chunk needs logits ──
        self.pos = base_pos + p;
        if !is_final_chunk {
            // Non-final chunks skip the tail entirely — same-stream in-order
            // execution carries the pending dispatches into the next chunk's
            // stream (the Batch-21 implicit-ordering rule). Returns empty; the
            // driver keeps only the final chunk's logits.
            return Vec::new();
        }
        unsafe {
            RmsNormCubeCL::launch::<ActiveRuntime>(
                &self.client,
                Self::tok_slice(&x_b, p - 1, n, p),
                self.final_norm.clone(),
                self.norm_x.clone(),
                n,
                eps,
            );
        }
        unsafe {
            GemvTernaryCubeCL::launch::<ActiveRuntime>(
                &self.client,
                &self.lm_head,
                self.norm_x.clone(),
                self.logits.clone(),
            );
        }

        // Hand the decode loop the last token's hidden state — the second
        // legitimate producer of a fresh `x` (Issue 860 T3), and the reason
        // `tests/prefill_tail_g1.rs` may run one bare `forward_token` after a
        // prefill with no `set_input_token`.
        self.x_input_fresh = true;
        // One CopyCubeCL
        // device-to-device copy — the prior FillZeros + ResidualAdd pair
        // cost 2 dispatches + a needless zero-fill (Issue 727 H5).
        unsafe {
            CopyCubeCL::launch::<ActiveRuntime>(
                &self.client,
                Self::tok_slice(&x_b, p - 1, n, p),
                self.x.clone(),
                n,
            );
        }

        let bytes = self
            .client
            .read_one(self.logits.clone())
            .expect("download logits");
        // Issue 994: a reserve failure during this chunk must not leave as a
        // successful return — non-final chunks hit this every chunk (the
        // earliest boundary), the final chunk guards the actual logits.
        Self::refuse_if_pool_poisoned("after prefill chunk");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Upload `token`'s embedding into `x` — the *first* half of the decode
    /// protocol; [`forward_token`](Self::forward_token) is the second and takes
    /// no token of its own (Issue 860).
    pub fn set_input_token(&mut self, _weights: &QwenDeltaNetTernaryWeights, token: usize) {
        // Issue 994: the write itself is a pool reserve — refuse on a
        // poisoned pool before staging corrupt state for the decode tick.
        Self::refuse_if_pool_poisoned("at input-token staging");
        // Issue 860 T3: the fresh-input producer. Set before the launch, not
        // after — the launch is a queued dispatch, so "fresh" is about the
        // caller's protocol, not about GPU completion.
        self.x_input_fresh = true;
        let n = self.config.n_embd;
        let blocks64 = self.wte_handle.blocks64;
        let groups_per_row = self.wte_handle.groups_per_row;

        unsafe {
            DequantWteRowCubeCL::launch::<ActiveRuntime>(
                &self.client,
                self.wte_handle.pos_bits_u32.clone(),
                self.wte_handle.neg_bits_u32.clone(),
                self.wte_handle.group_scale_f32.clone(),
                self.x.clone(),
                token as u32,
                blocks64 as u32,
                groups_per_row as u32,
                n as u32,
            );
        }
        // Plan 602 B3 — folded site 1 of 8: the Hadamard-latent embedding
        // table stores ROTATED rows; restore the primal basis right after
        // the lookup (Hadamard first, sign second — the CPU twin
        // `rotate_inverse_inplace`; the fork's build_inp_embd `h = s·(H z)`).
        // One dispatch; skipped on pre-rotation files (`inverse_embedding`
        // is the loader's marker for a rotated wte).
        if let (Some(cfg), Some(tables)) = (&self.rotation, &self.rot_tables)
            && cfg.inverse_embedding
        {
            let signs = tables.signs_for_width(n).clone();
            unsafe {
                RotationCubeCL::launch_inverse::<ActiveRuntime>(
                    &self.client,
                    self.x.clone(),
                    signs,
                    n,
                    n,
                    cfg.block_size,
                );
            }
        }
    }

    /// Returns the current position counter (number of tokens decoded so far).
    ///
    /// Used by speculative decode callers to snapshot `pos` before speculation
    /// so it can be restored on rollback.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Plan 602 B2/B3 — is this a Hadamard-folded (Bonsai-2) model with the
    /// GPU rotation tables built? `false` on pre-rotation files (the
    /// untouched decode path). Test + league introspection surface.
    pub fn is_folded(&self) -> bool {
        self.rot_tables.is_some()
    }

    /// Reset all per-layer state (for starting a new sequence).
    ///
    /// Zeros the DeltaNet recurrent + conv state buffers. The attention KV
    /// caches are deliberately NOT zeroed: decode bounds its reads to
    /// `n_positions = pos + 1`, so after `pos = 0` only position 0 is read,
    /// and every position a new sequence touches is overwritten by its own
    /// KV append before any read consumes it — stale data beyond `pos` is
    /// unreachable (the same argument `rollback_speculative_gpu` relies on).
    /// Skipping the fill also keeps reset O(deltanet layers) instead of
    /// touching every attention layer's full block-size cache.
    ///
    /// Zero-fills the EXISTING GPU buffers in-place via `FillZerosCubeCL` —
    /// does NOT allocate new buffers. The previous implementation called
    /// `client.create_from_slice` for each of 128 buffers per call, which
    /// fragmented the CubeCL memory pool and caused SIGKILL under the Go
    /// arena (Plan 528 T0.2 blocker).
    pub fn reset_state(&mut self) {
        // Issue 994: zero-fill launches read the pool — refuse on poison.
        Self::refuse_if_pool_poisoned("at state reset");
        // Issue 860 T3: a new sequence must upload its own first token.
        self.x_input_fresh = false;
        // Issue 782: the flash-arm length gates read this — a stale value from
        // a previous long prompt would misroute a fresh short one.
        self.active_prefill_len = 0;
        let n_v_heads = self.config.deltanet_linear_n_value_heads;
        let head_dim = self.config.deltanet_linear_head_dim;
        let state_dim = n_v_heads * head_dim * head_dim;
        // Must match `new`'s allocation: conv_dim = qkv_dim = 2·(n_k·hd) +
        // n_v·hd (the PRE-expansion in_proj width — the conv runs before
        // expand_l2). The previous `3 * n_v * hd` form zero-filled
        // (3·n_v − 2·n_k − n_v)·hd·ks floats PAST the buffer for GQA
        // models (a silent GPU-side OOB into the CubeCL pool; 32 KB/layer
        // on Bonsai — found while wiring the Arm-13 reset hook).
        let conv_dim = 2 * (self.config.deltanet_linear_n_heads * head_dim) + n_v_heads * head_dim;
        let kernel_size = self.config.deltanet_conv_kernel_size;

        for i in 0..self.config.n_layer {
            if self.layer_types[i] == DeltaNetLayerType::DeltaNet {
                // Zero-fill existing deltanet state buffer in-place.
                if let Some(ref handle) = self.deltanet_states[i] {
                    unsafe {
                        FillZerosCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            handle.clone(),
                            state_dim,
                        );
                    }
                }
                // Zero-fill existing conv state buffer in-place.
                if let Some(ref handle) = self.conv_states[i] {
                    unsafe {
                        FillZerosCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            handle.clone(),
                            conv_dim * kernel_size,
                        );
                    }
                }
            } else {
                // Attention KV cache: stale data beyond pos is never read
                // (attention decode uses n_positions = pos + 1 to bound reads).
                // No zero-fill needed — just advance pos to 0.
            }
        }
        self.pos = 0;
        // Arm 13 — bring the cudarc whole-prefill mirrors to the same reset
        // values DEVICE-SIDE (memset, no PCIe) so the next prompt's chunk
        // skips the `sync_states` upload entirely.
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            feature = "ternary_gemm_batched",
            feature = "cubecl_runtime",
            not(target_os = "macos")
        ))]
        crate::prefill_cuda_full::notify_cubcl_reset();
    }

    /// Download all persistent state (DeltaNet recurrent + conv, Attention KV
    /// cache) from GPU buffers into a CPU [`HybridCache`]. Used by Plan 334
    /// hybrid training: GPU forward for prompt tokens (frozen, no LoRA) → state
    /// transfer → CPU forward for target tokens (with LoRA + activation saving
    /// for backward).
    ///
    /// The `cache` MUST be pre-initialized with
    /// [`HybridCache::with_layer_types`] using the same `config` + `layer_types`
    /// as this forward struct. Sizes are checked via `debug_assert_eq!`.
    ///
    /// Only the first `self.pos` positions of the KV caches are valid (positions
    /// `0..self.pos` were written by the prompt forward). The DeltaNet recurrent
    /// + conv states are overwritten wholesale.
    pub fn download_state_to_hybrid_cache(
        &self,
        cache: &mut riir_infer_core::deltanet::forward::HybridCache,
    ) {
        // Issue 994: the exported state of a poisoned pool is corrupt — refuse.
        Self::refuse_if_pool_poisoned("at state export");
        // Issue 965 flush-before-read: a graph-armed prefill leaves its state
        // in the cudarc mirrors until the spec flush; this export reads the
        // CubeCL handles, so bring them current first (O(1) when clean).
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            feature = "ternary_gemm_batched",
            feature = "cubecl_runtime",
            not(target_os = "macos")
        ))]
        crate::prefill_cuda_full::prefill_spec_flush(self);
        // 1. DeltaNet recurrent + conv states.
        for i in 0..self.config.n_layer {
            if self.layer_types[i] != DeltaNetLayerType::DeltaNet {
                continue;
            }
            if let Some(handle) = &self.deltanet_states[i] {
                let bytes = self
                    .client
                    .read_one(handle.clone())
                    .expect("download deltanet recurrent state");
                let state = f32::from_bytes(&bytes);
                let dst = &mut cache.deltanet_state.recurrent_states[i];
                debug_assert_eq!(
                    dst.len(),
                    state.len(),
                    "recurrent state size mismatch at layer {i}"
                );
                dst.copy_from_slice(state);
            }
            if let Some(handle) = &self.conv_states[i] {
                let bytes = self
                    .client
                    .read_one(handle.clone())
                    .expect("download conv state");
                let conv = f32::from_bytes(&bytes);
                let dst = &mut cache.deltanet_state.conv_states[i];
                debug_assert_eq!(
                    dst.len(),
                    conv.len(),
                    "conv state size mismatch at layer {i}"
                );
                dst.copy_from_slice(conv);
            }
        }

        // 2. Attention KV caches.
        //
        // GPU layout: `[block_size, kvd]` row-major (one row per position).
        // CPU layout: same. So a contiguous copy of the valid prefix suffices.
        let kvd = self.config.n_kv_head * self.config.head_dim;
        let n_positions = self.pos; // positions 0..pos were written by the prompt forward
        let n_floats = n_positions * kvd;

        for i in 0..self.config.n_layer {
            if self.layer_types[i] != DeltaNetLayerType::Attention {
                continue;
            }
            if let (Some(kh), Some(vh)) =
                (&self.kv_key_caches[i], &self.kv_value_caches[i])
            {
                let k_bytes = self
                    .client
                    .read_one(kh.clone())
                    .expect("download attention key cache");
                let k_data = f32::from_bytes(&k_bytes);
                let v_bytes = self
                    .client
                    .read_one(vh.clone())
                    .expect("download attention value cache");
                let v_data = f32::from_bytes(&v_bytes);

                let layer = &mut cache.kv_cache.layers[i];
                debug_assert_eq!(
                    layer.key.len(),
                    self.config.block_size * kvd,
                    "key cache size mismatch at layer {i}"
                );
                layer.key[..n_floats].copy_from_slice(&k_data[..n_floats]);
                layer.value[..n_floats].copy_from_slice(&v_data[..n_floats]);
            }
        }
    }

    // ───────────────────────────────────────────────────────────────
    // Issue 665: Speculative decoding — sequential speculative verify path
    // ───────────────────────────────────────────────────────────────
    //
    // Instead of batched verify (blocked by Issue 652 — the Delta rule chunkwise-
    // parallel algorithm G1 FAILED), this path queues K decode passes on the GPU
    // without intermediate syncs, then reads all K logits at once.
    //
    // The GPU processes dispatches in order, so each forward correctly reads the
    // previous forward's GPU-resident state (DeltaNet recurrent state, conv state,
    // KV cache). The ONLY CPU-GPU sync is the final read of K logits buffers.
    //
    // Speedup ceiling (Bench 669): sync is 5.60ms out of 42.86ms/token (13.1%).
    // At K=2 with α=0.7 acceptance, expected speedup ≈ 1.17×.

    /// Maximum draft length supported by [`Self::forward_speculative_verify`]
    /// (Issue 727 H6). The logits rotation pool is pre-allocated to this size
    /// at construction — the same cap + pool shape as the cudarc twin
    /// (`SPEC_MAX_K` there). Pool cost: 8 × vocab × 4 B (~4 MB at 131k vocab).
    #[cfg(feature = "speculative_decode")]
    // pub(crate): read by the Issue 994 VRAM budget estimator (vram_budget.rs).
    pub(crate) const SPEC_MAX_K: usize = 8;

    /// Run K forward passes for speculative verification WITHOUT intermediate
    /// syncs. Each forward processes one of the draft model's predicted tokens.
    ///
    /// Returns **K+1** logits vectors, position-aligned so every draft token
    /// verifies with a single indexed lookup — no separate `read_logits()`
    /// sync (Issue 669 follow-up):
    ///
    /// - `out[0]` — the pre-speculation logits (whatever `self.logits` held at
    ///   call time; the K forwards never overwrite that buffer). If the caller
    ///   ran a forward for the last committed token first, this predicts
    ///   `draft_tokens[0]`'s position.
    /// - `out[i]` for 1 ≤ i ≤ K — logits AFTER processing `draft_tokens[i-1]`,
    ///   predicting `draft_tokens[i]`'s position. `out[K]` predicts the
    ///   position AFTER the last draft (the resample seed / next-round start).
    ///
    /// Verify draft `d_i` with `argmax(out[i]) == d_i`; accept the matching
    /// prefix, reject + resample at the first divergence. `argmax(out[K])` is
    /// the token the model itself would emit after a fully-accepted draft —
    /// do NOT commit it without processing it (Issue 669 pos-misalignment).
    ///
    /// **IMPORTANT:** This method advances `self.pos` by K and updates all GPU-
    /// resident state (DeltaNet recurrence, conv, KV cache) for all K tokens.
    /// If the caller rejects any tokens, it MUST call `rollback_speculative` to
    /// restore the state to the pre-speculation checkpoint.
    ///
    /// # Arguments
    /// * `weights` — the model weights (for embedding lookup via `set_input_token`)
    /// * `draft_tokens` — K draft tokens predicted by the draft model (n-gram).
    ///   These are SPECULATIVE — the main model will verify them.
    ///
    /// # Returns
    /// K+1 logits vectors, each of length `vocab_size` (see the alignment
    /// contract above).
    #[cfg(feature = "speculative_decode")]
    pub fn forward_speculative_verify(
        &mut self,
        weights: &QwenDeltaNetTernaryWeights,
        draft_tokens: &[usize],
    ) -> Vec<Vec<f32>> {
        let k = draft_tokens.len();
        debug_assert!(k > 0, "draft_tokens must be non-empty");
        assert!(
            (1..=Self::SPEC_MAX_K).contains(&k),
            "speculative verify supports 1..={} drafts, got {k} — raise SPEC_MAX_K \
             (the logits pool is sized at construction)",
            Self::SPEC_MAX_K
        );
        // Issue 746 tripwire — the whole verify chunk must fit the full-length
        // KV cache. This twin has no per-forward pos guard, so this entry
        // check is the only floor: positions pos..pos+k-1 each append one KV
        // row into a buffer sized for `config.block_size` rows total.
        debug_assert!(
            self.pos + k <= self.config.block_size,
            "spec verify would overflow the full-length KV cache \
             (pos {} + k {k} > block_size {}) — Issue 746 tripwire on the \
             append-only KV contract",
            self.pos,
            self.config.block_size
        );
        // Issue 994: batched verify is its own result path (K+1 logits) —
        // entry + exit refusal like the funnels.
        Self::refuse_if_pool_poisoned("at speculative verify entry");

        // Strategy: run K forwards, each writing to a separate logits buffer.
        // We achieve this by swapping self.logits with a rotation-pool buffer
        // before each forward (Issue 727 H6 — the pool is pre-allocated at
        // construction; the prior per-call `client.empty` churn was the exact
        // pool-fragmentation class reset_state's doc warns about). The GPU
        // processes all dispatches in order, so each forward correctly reads
        // the previous forward's state.
        //
        // Handle rotation trace (K=2):
        //   Initial: self.logits = L0, pool = [P0, P1, ..]
        //   swap[0]: self.logits = P0, pool[0] = L0
        //   forward 0 → writes to P0
        //   swap[1]: self.logits = P1, pool[1] = P0
        //   forward 1 → writes to P1
        //   Result: pool = [L0(pre-spec), P0(fwd0), ..], self.logits = P1(fwd1)
        //
        // General: pool[i+1] = forward i's logits, self.logits = forward K-1's logits.
        // pool[0] is the original L0 — NOT stale: the K forwards write P0..P(K-1)
        // and never touch L0, so it still holds the pre-speculation logits.
        // Like the cudarc twin, there is NO un-rotation: the field keeps the
        // last pool buffer between cycles, so the next cycle's pool[0] receives
        // whatever self.logits held at call time (the live pre-speculation
        // prediction when the caller ran a forward for the last committed
        // token first).

        for (i, &token) in draft_tokens.iter().enumerate() {
            // Swap self.logits ↔ pool[i]. After the swap, self.logits points at
            // the pool buffer forward i writes into; pool[i] holds the previous
            // self.logits (the previous forward's output, or the original
            // buffer for i=0).
            std::mem::swap(&mut self.logits, &mut self.spec_logits_pool[i]);

            self.set_input_token(weights, token);
            self.forward_dispatch_only();
        }

        // After the loop:
        //   self.logits = P(k-1) → contains forward K-1's logits
        //   pool[0] = L0 → contains the pre-speculation logits
        //   pool[i] = P(i-1) for i in 1..K → contains forward (i-1)'s logits
        //
        // Read all K+1 logits buffers in ONE batched read — drains the GPU
        // pipeline once instead of K+1 times, and returns position-aligned
        // logits so the caller needs NO extra `read_logits()` sync for the
        // pre-speculation prediction (Issue 669 follow-up):
        //   out[0] = L0            → predicts draft_tokens[0]'s position
        //   out[i] = P(i-1)        → predicts draft_tokens[i]'s position (i ≥ 1)
        //   out[K] = P(K-1)        → predicts the position after the last draft
        let mut read_handles: Vec<Handle> = self.spec_logits_pool[..k].to_vec();
        read_handles.push(self.logits.clone());

        let all_bytes = self.client.read(read_handles);

        // Issue 994: verify's exit refusal — the K+1 readback of a poisoned
        // pool is exactly the structured-garbage class this lane exists for.
        Self::refuse_if_pool_poisoned("after speculative verify");
        all_bytes
            .into_iter()
            .map(|bytes| f32::from_bytes(&bytes).to_vec())
            .collect()
    }

    /// Checkpoint the current GPU-resident state for speculative rollback.
    ///
    /// Saves a snapshot of all mutable per-layer state (DeltaNet recurrent
    /// state, conv state) + the current position. The KV cache for attention
    /// layers doesn't need explicit checkpointing — stale entries beyond `pos`
    /// are never read (attention decode uses `n_positions = pos + 1`).
    ///
    /// Returns a [`SpeculativeCheckpoint`] that can be passed to
    /// [`rollback_speculative`] to undo the effects of rejected speculation.
    ///
    /// **Cost:** 1 GPU pipeline sync (batched read of all DeltaNet state handles).
    /// The prior implementation called `read_one` 96 times (48 layers × 2 states),
    /// each forcing a separate pipeline drain — 96 syncs. Batching into a single
    /// `client.read(vec![...])` call reduces this to 1 sync.
    #[cfg(feature = "speculative_decode")]
    pub fn checkpoint_speculative(&self) -> SpeculativeCheckpoint {
        // Issue 965 flush-before-checkpoint: a graph-armed prefill leaves its
        // state in the cudarc mirrors until the spec flush; the checkpoint
        // reads the CubeCL handles, so bring them current first.
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            feature = "ternary_gemm_batched",
            feature = "cubecl_runtime",
            not(target_os = "macos")
        ))]
        crate::prefill_cuda_full::prefill_spec_flush(self);
        let n_layers = self.config.n_layer;

        // Phase 1: collect all handles to read in one batch.
        // Each DeltaNet layer contributes 2 handles (recurrent state + conv state).
        // We track which (layer_idx, state_kind) each handle belongs to so we can
        // distribute the batched results correctly.
        #[derive(Clone, Copy)]
        enum StateKind {
            DeltaNet,
            Conv,
        }

        let mut handles: Vec<Handle> = Vec::new();
        let mut handle_map: Vec<(usize, StateKind)> = Vec::new();

        for i in 0..n_layers {
            if self.layer_types[i] == DeltaNetLayerType::DeltaNet {
                if let Some(ref handle) = self.deltanet_states[i] {
                    handles.push(handle.clone());
                    handle_map.push((i, StateKind::DeltaNet));
                }
                if let Some(ref handle) = self.conv_states[i] {
                    handles.push(handle.clone());
                    handle_map.push((i, StateKind::Conv));
                }
            }
        }

        // Phase 2: ONE batched read — drains the GPU pipeline once for all handles.
        let all_bytes = self.client.read(handles);

        // Phase 3: distribute results into per-layer Vecs.
        let mut states: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
        let mut convs: Vec<Vec<f32>> = vec![Vec::new(); n_layers];

        for (bytes, (layer_idx, kind)) in all_bytes.into_iter().zip(handle_map) {
            let data = f32::from_bytes(&bytes).to_vec();
            match kind {
                StateKind::DeltaNet => states[layer_idx] = data,
                StateKind::Conv => convs[layer_idx] = data,
            }
        }

        SpeculativeCheckpoint {
            pos: self.pos,
            deltanet_states: states,
            conv_states: convs,
        }
    }

    /// Rollback GPU-resident state to a previously saved checkpoint.
    ///
    /// This undoes the effects of K speculative forward passes that were
    /// rejected during verification. After rollback, `self.pos` and all
    /// per-layer state are restored to their pre-speculation values.
    ///
    /// **Cost:** 1 batched device submit (Issue 727 H7 — the write-side mirror
    /// of `checkpoint_speculative`'s batched read; the prior per-layer
    /// `client.write` calls were one submit each, 96 submits for a 48-layer
    /// model). Writes are non-blocking (no pipeline drain); only call it when
    /// speculation is rejected. When speculation succeeds (all K tokens
    /// accepted), no rollback is needed — the state is already correct.
    #[cfg(feature = "speculative_decode")]
    pub fn rollback_speculative(&mut self, checkpoint: &SpeculativeCheckpoint) {
        // Issue 860 T3: `x` still holds the last rejected draft's state.
        self.x_input_fresh = false;
        self.pos = checkpoint.pos;

        // Phase 1: collect every (handle, bytes) restore in order — layer 0's
        // recurrent state, layer 0's conv state, layer 1's, ... — so the batched
        // submission preserves the per-layer ordering the loop below produced.
        let mut writes: Vec<(Handle, cubecl::bytes::Bytes)> = Vec::new();
        for i in 0..self.config.n_layer {
            if self.layer_types[i] == DeltaNetLayerType::DeltaNet {
                if let Some(ref handle) = self.deltanet_states[i] {
                    let state = &checkpoint.deltanet_states[i];
                    if !state.is_empty() {
                        let bytes_vec = f32::as_bytes(state).to_vec();
                        writes.push((
                            handle.clone(),
                            cubecl::bytes::Bytes::from_bytes_vec(bytes_vec),
                        ));
                    }
                }
                if let Some(ref handle) = self.conv_states[i] {
                    let conv = &checkpoint.conv_states[i];
                    if !conv.is_empty() {
                        let bytes_vec = f32::as_bytes(conv).to_vec();
                        writes.push((
                            handle.clone(),
                            cubecl::bytes::Bytes::from_bytes_vec(bytes_vec),
                        ));
                    }
                }
            }
        }

        // Phase 2: ONE batched write submit for all destinations.
        self.client.write_many(writes);
        // Issue 965 — the handles were REWOUND: discard any pending spec flush
        // (applying it would resurrect the future mirror state over the
        // rewound handles) and mark the mirrors stale for the next sync.
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            feature = "ternary_gemm_batched",
            feature = "cubecl_runtime",
            not(target_os = "macos")
        ))]
        crate::prefill_cuda_full::prefill_spec_discard();
    }

    // -----------------------------------------------------------------------
    // Issue 665 Phase 2: GPU-side checkpoint/rollback (zero CPU sync)
    // -----------------------------------------------------------------------
    //
    // These methods keep the checkpoint entirely on the GPU side. Instead of
    // reading state to CPU (which drains the pipeline) and writing it back
    // (which allocates Bytes), they dispatch CopyCubeCL kernels that copy
    // state→backup and backup→state entirely on the GPU.
    //
    // The backup buffers are pre-allocated at `new()` time — no per-checkpoint
    // allocation, no memory-pool churn (same lesson as FillZerosCubeCL).
    //
    // **Cost:** 2 × n_deltanet_layers CopyCubeCL dispatches, all non-blocking.
    // Each dispatch is enqueued on the stream and executes when the GPU gets
    // to it. No CPU↔GPU fence, no pipeline drain.
    //
    // **Correctness:** The dispatches are ordered by the stream's FIFO queue.
    // checkpoint_speculative_gpu() dispatches run BEFORE any subsequent
    // forward_speculative_verify() dispatches. rollback_speculative_gpu()
    // dispatches run BEFORE any subsequent forward dispatches. The GPU sees a
    // consistent execution order.

    /// GPU-side checkpoint: copy all DeltaNet state to backup buffers.
    ///
    /// Dispatches `CopyCubeCL` for each DeltaNet layer's recurrent state +
    /// conv state (96 dispatches for 48 DeltaNet layers). ALL dispatches are
    /// non-blocking — enqueued on the stream, no CPU↔GPU sync.
    ///
    /// The backup buffers are pre-allocated at `new()` time, so there's zero
    /// memory allocation or pool churn.
    ///
    /// Returns `()` — the checkpoint is implicit in the backup buffers. Call
    /// [`rollback_speculative_gpu`] to restore.
    ///
    /// **Cost:** ~0.4ms GPU time for 160MB copy (400 GB/s unified memory),
    /// but this overlaps with subsequent GPU work — zero wall-clock cost if
    /// the next operation is a GPU dispatch.
    #[cfg(feature = "speculative_decode")]
    pub fn checkpoint_speculative_gpu(&self) {
        // Issue 994: checkpointing corrupt state bakes the corruption in.
        Self::refuse_if_pool_poisoned("at speculative checkpoint");
        // Issue 965 flush-before-checkpoint (same as checkpoint_speculative).
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            feature = "ternary_gemm_batched",
            feature = "cubecl_runtime",
            not(target_os = "macos")
        ))]
        crate::prefill_cuda_full::prefill_spec_flush(self);
        let n_layers = self.config.n_layer;
        let state_dim = self.config.deltanet_linear_n_value_heads
            * self.config.deltanet_linear_head_dim
            * self.config.deltanet_linear_head_dim;
        let conv_dim = 2 * (self.config.deltanet_linear_n_heads
            * self.config.deltanet_linear_head_dim)
            + (self.config.deltanet_linear_n_value_heads * self.config.deltanet_linear_head_dim);
        let conv_size = conv_dim * self.config.deltanet_conv_kernel_size;

        for i in 0..n_layers {
            if self.layer_types[i] == DeltaNetLayerType::DeltaNet {
                // Copy recurrent state → backup
                if let (Some(src), Some(dst)) =
                    (self.deltanet_states[i].as_ref(), self.deltanet_state_backups[i].as_ref())
                {
                    unsafe {
                        CopyCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            src.clone(),
                            dst.clone(),
                            state_dim,
                        );
                    }
                }
                // Copy conv state → backup
                if let (Some(src), Some(dst)) =
                    (self.conv_states[i].as_ref(), self.conv_state_backups[i].as_ref())
                {
                    unsafe {
                        CopyCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            src.clone(),
                            dst.clone(),
                            conv_size,
                        );
                    }
                }
            }
        }
    }

    /// GPU-side rollback: restore all DeltaNet state from backup buffers.
    ///
    /// Dispatches `CopyCubeCL` for each DeltaNet layer's backup→state pair.
    /// ALL dispatches are non-blocking — enqueued on the stream, no CPU↔GPU
    /// sync. Should be called when speculation is rejected.
    ///
    /// **IMPORTANT:** `self.pos` is also restored. The caller must set it
    /// before calling this (the checkpoint doesn't carry the position).
    #[cfg(feature = "speculative_decode")]
    pub fn rollback_speculative_gpu(&mut self, checkpoint_pos: usize) {
        // Issue 994: rollback copies read the pool — refuse on poison.
        Self::refuse_if_pool_poisoned("at speculative rollback");
        // Issue 860 T3: `x` still holds the last rejected draft's state.
        self.x_input_fresh = false;
        self.pos = checkpoint_pos;

        let n_layers = self.config.n_layer;
        let state_dim = self.config.deltanet_linear_n_value_heads
            * self.config.deltanet_linear_head_dim
            * self.config.deltanet_linear_head_dim;
        let conv_dim = 2 * (self.config.deltanet_linear_n_heads
            * self.config.deltanet_linear_head_dim)
            + (self.config.deltanet_linear_n_value_heads * self.config.deltanet_linear_head_dim);
        let conv_size = conv_dim * self.config.deltanet_conv_kernel_size;

        for i in 0..n_layers {
            if self.layer_types[i] == DeltaNetLayerType::DeltaNet {
                // Copy backup → recurrent state
                if let (Some(src), Some(dst)) =
                    (self.deltanet_state_backups[i].as_ref(), self.deltanet_states[i].as_ref())
                {
                    unsafe {
                        CopyCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            src.clone(),
                            dst.clone(),
                            state_dim,
                        );
                    }
                }
                // Copy backup → conv state
                if let (Some(src), Some(dst)) =
                    (self.conv_state_backups[i].as_ref(), self.conv_states[i].as_ref())
                {
                    unsafe {
                        CopyCubeCL::launch::<ActiveRuntime>(
                            &self.client,
                            src.clone(),
                            dst.clone(),
                            conv_size,
                        );
                    }
                }
            }
        }
        // Issue 965 — the handles were REWOUND: discard any pending spec flush
        // (applying it would resurrect the future mirror state over the
        // rewound handles) and mark the mirrors stale for the next sync.
        #[cfg(all(
            feature = "ternary_gemv_cuda_raw",
            feature = "ternary_gemm_batched",
            feature = "cubecl_runtime",
            not(target_os = "macos")
        ))]
        crate::prefill_cuda_full::prefill_spec_discard();
    }
}

/// Checkpoint of GPU-resident state for speculative decoding rollback (Issue 665).
///
/// Created by [`TernaryDeltanetGpuForward::checkpoint_speculative`] and consumed
/// by [`TernaryDeltanetGpuForward::rollback_speculative`]. Contains CPU copies
/// of all mutable per-layer state at the checkpoint point.
#[cfg(feature = "speculative_decode")]
pub struct SpeculativeCheckpoint {
    /// The position counter at checkpoint time.
    pos: usize,
    /// Per-layer DeltaNet recurrent state (empty for attention layers).
    deltanet_states: Vec<Vec<f32>>,
    /// Per-layer conv1d sliding window state (empty for attention layers).
    conv_states: Vec<Vec<f32>>,
}

/// Upload a single layer's weights to GPU.
#[cfg(feature = "cubecl_runtime")]
fn upload_layer_weights(
    client: &ComputeClient<ActiveRuntime>,
    l: &DeltaNetTernaryLayerWeights,
) -> GpuLayerWeights {
    GpuLayerWeights {
        // Issue 727 H2a: separate in_proj handles exist only for the batched
        // prefill path (same feature gate as `prefill`).
        // Issue 980 T4-ALT: dense a/b (Bonsai-2 escape set) upload as f32
        // Handles beside an EMPTY dummy TernaryHandle — no folded path reads
        // the ternary handles, and the dummy keeps mma-mirror construction
        // (lazy, keyed on use) untouched for the ternary a/b case.
        #[cfg(feature = "ternary_gemm_batched")]
        in_proj_qkv: TernaryHandle::from_weights(client, &l.in_proj_qkv),
        #[cfg(feature = "ternary_gemm_batched")]
        in_proj_z: TernaryHandle::from_weights(client, &l.in_proj_z),
        #[cfg(feature = "ternary_gemm_batched")]
        in_proj_a: match l.in_proj_a.as_ternary() {
            Some(t) => TernaryHandle::from_weights(client, t),
            None => TernaryHandle::from_weights(
                client,
                &katgpt_core::TernaryGroupWeights::new(0, 0),
            ),
        },
        #[cfg(feature = "ternary_gemm_batched")]
        in_proj_b: match l.in_proj_b.as_ternary() {
            Some(t) => TernaryHandle::from_weights(client, t),
            None => TernaryHandle::from_weights(
                client,
                &katgpt_core::TernaryGroupWeights::new(0, 0),
            ),
        },
        // Plan 602 B2: ungated (the decode lane consumes these without
        // `ternary_gemm_batched`).
        in_proj_a_f32: l.in_proj_a.as_dense().map(|(data, _, _)| upload_f32_slice(client, data)),
        in_proj_b_f32: l.in_proj_b.as_dense().map(|(data, _, _)| upload_f32_slice(client, data)),
        // Issue 642 F3: concatenated qkv+z+a+b for single-GEMV input projection.
        // Only meaningful for DeltaNet layers (Attention layers have empty in_proj_*).
        // Plan 602 B2: on folded files a/b are DENSE and dispatch as separate
        // f32 GEMVs on the PRIMAL input — the concat then carries the two
        // FOLDED projections only (qkv|z), and the decode path splits it with
        // `Split2CubeCL` (the `in_proj_a_f32`/`in_proj_b_f32` handles above
        // carry the escape set). Pre-rotation files keep the 4-way concat.
        in_proj_concat: if l.in_proj_qkv.rows > 0
            && l.in_proj_a.as_ternary().is_some()
            && l.in_proj_b.as_ternary().is_some()
        {
            TernaryHandle::from_weights_concat(client, &[
                &l.in_proj_qkv,
                &l.in_proj_z,
                l.in_proj_a.as_ternary().expect("checked above"),
                l.in_proj_b.as_ternary().expect("checked above"),
            ])
        } else if l.in_proj_qkv.rows > 0 {
            // Folded layer (dense a/b): qkv|z concat — the rotated-basis GEMV
            // input projection (Plan 602 B2).
            TernaryHandle::from_two_weights(client, &l.in_proj_qkv, &l.in_proj_z)
        } else {
            // Attention layer (empty in_proj_*): upload a dummy empty handle
            // to satisfy the struct field. No enabled path reads the concat.
            TernaryHandle::from_weights(client, &katgpt_core::TernaryGroupWeights::new(0, 0))
        },
        out_proj: TernaryHandle::from_weights(client, &l.out_proj),
        gate_proj: TernaryHandle::from_weights(client, &l.gate_proj),
        up_proj: TernaryHandle::from_weights(client, &l.up_proj),
        down_proj: TernaryHandle::from_weights(client, &l.down_proj),
        // Issue 642 F2: concatenated gate+up for single-GEMV FFN input path.
        gate_up_proj: TernaryHandle::from_two_weights(client, &l.gate_proj, &l.up_proj),

        input_norm: upload_f32_slice(client, &l.input_norm),
        post_attn_norm: upload_f32_slice(client, &l.post_attn_norm),
        conv1d_weight: upload_f32_slice(client, &l.conv1d_weight),
        a_log: upload_f32_slice(client, &l.a_log),
        dt_bias: upload_f32_slice(client, &l.dt_bias),
        linear_norm: upload_f32_slice(client, &l.linear_norm),

        attn_wq: if l.attn_wq.rows > 0 {
            Some(TernaryHandle::from_weights(client, &l.attn_wq))
        } else {
            None
        },
        // Issue 727 H1: no separate attn_wk/attn_wv handles — dead weight
        // (attn_wkv is built directly from the CPU weights below).
        // Issue 648 F9: concatenated WK+WV for single-GEMV K+V projection.
        attn_wkv: if l.attn_wk.rows > 0 && l.attn_wv.rows > 0 {
            Some(TernaryHandle::from_two_weights(client, &l.attn_wk, &l.attn_wv))
        } else {
            None
        },
        attn_wo: if l.attn_wo.rows > 0 {
            Some(TernaryHandle::from_weights(client, &l.attn_wo))
        } else {
            None
        },
        attn_q_norm: if !l.attn_q_norm.is_empty() {
            Some(upload_f32_slice(client, &l.attn_q_norm))
        } else {
            None
        },
        attn_k_norm: if !l.attn_k_norm.is_empty() {
            Some(upload_f32_slice(client, &l.attn_k_norm))
        } else {
            None
        },
    }
}

/// Upload a slice of f32 to a GPU buffer.
#[cfg(feature = "cubecl_runtime")]
fn upload_f32_slice(client: &ComputeClient<ActiveRuntime>, data: &[f32]) -> Handle {
    if data.is_empty() {
        return client.empty(4); // minimal allocation for empty
    }
    client.create_from_slice(f32::as_bytes(data))
}

#[cfg(all(test, feature = "cubecl_runtime", feature = "ternary_gemv"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};
    use katgpt_core::TernaryGroupWeights;

    /// Tolerance for GPU vs CPU comparison.
    const TOL: f32 = 1e-6;

    /// Build a small synthetic ternary wte with known values.
    ///
    /// 4 rows × 256 cols. Each row has a different scale + a mix of
    /// +1 / -1 / 0 values so all ternary branches are exercised.
    fn build_test_wte() -> TernaryGroupWeights {
        let rows = 4;
        let cols = 256;
        let mut w = TernaryGroupWeights::new(rows, cols);

        // Set known values + per-group scales.
        // Row 0: all +1 in even cols, -1 in odd cols (scale 0.5)
        // Row 1: all +1 (scale 1.0)
        // Row 2: all -1 (scale 0.25)
        // Row 3: all 0 (scale 1.0)
        for c in 0..cols {
            w.set(0, c, if c.is_multiple_of(2) { 1 } else { -1 });
            w.set(1, c, 1);
            w.set(2, c, -1);
            // row 3 left as 0
        }
        // Override per-group scales (2 groups per row for cols=256, group_size=128).
        // group_scale layout: [rows * groups_per_row], f16.
        use half::f16;
        for r in 0..rows {
            let scale = match r {
                0 => f16::from_f32(0.5),
                1 => f16::from_f32(1.0),
                2 => f16::from_f32(0.25),
                _ => f16::from_f32(1.0),
            };
            for g in 0..w.groups_per_row {
                w.group_scale[r * w.groups_per_row + g] = scale;
            }
        }
        w
    }

    /// CPU reference: dequantize row `row_idx` of the wte into a Vec<f32>.
    fn cpu_dequant_row(w: &TernaryGroupWeights, row_idx: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; w.cols];
        for c in 0..w.cols {
            let ternary = w.get(row_idx, c) as f32;
            let group = c / 128;
            let scale = w.group_scale[row_idx * w.groups_per_row + group].to_f32();
            out[c] = ternary * scale;
        }
        out
    }

    /// Test: GPU dequant kernel matches CPU reference for all rows.
    #[test]
    fn test_dequant_wte_row_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let w = build_test_wte();
        let handle = TernaryHandle::from_weights(&client, &w);

        let n = w.cols;
        let out = client.empty(n * std::mem::size_of::<f32>());

        for row_idx in 0..w.rows {
            unsafe {
                DequantWteRowCubeCL::launch::<ActiveRuntime>(
                    &client,
                    handle.pos_bits_u32.clone(),
                    handle.neg_bits_u32.clone(),
                    handle.group_scale_f32.clone(),
                    out.clone(),
                    row_idx as u32,
                    handle.blocks64 as u32,
                    handle.groups_per_row as u32,
                    n as u32,
                );
            }

            let bytes = client.read_one(out.clone()).expect("read output");
            let gpu_out = f32::from_bytes(&bytes);

            let cpu_out = cpu_dequant_row(&w, row_idx);

            let mut max_diff: f32 = 0.0;
            for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
                let diff = (g - c).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > TOL {
                    eprintln!(
                        "row {row_idx} col {i}: gpu={g:.6} cpu={c:.6} diff={diff:.6}"
                    );
                }
            }
            assert!(
                max_diff <= TOL,
                "row {row_idx}: GPU dequant max_diff={max_diff:.6} exceeds TOL={TOL}"
            );
            eprintln!("row {row_idx}: max_diff = {max_diff:.6} ✓");
        }
    }
}
