//! `DeltaNet` model types for hybrid DeltaNet/Attention inference (Plan 182).
//!
//! Provides weight structs, weight loading, CPU reference recurrence, and
//! CPU forward pass for hybrid models like Qwen 3.5 that mix `DeltaNet`
//! (linear recurrent) layers with standard attention layers.
//!
//! # Architecture
//!
//! `DeltaNet` layers replace attention with a Gated `DeltaNet` recurrence:
//! ```text
//! g_t = exp(-exp(A_log) * softplus(a + dt_bias))   // decay gate
//! β_t = sigmoid(b_t)                               // update rate
//! q̃ = l2norm(q), k̃ = l2norm(k)                   // normalized Q/K
//! S = g·S + k̃ ⊗ β(v − (g·S)^T k̃)               // delta rule update
//! output = q̃^T · S / √d                           // state read
//! ```
//!
//! Each head maintains a [`key_dim` × `val_dim`] state matrix.
//! For Qwen 3.5-0.8B: 16 heads × (128 × 128) = 2 MB recurrent state total.
//!
//! # Optimization Alignment (per .contexts/optimization.md)
//!
//! - Pre-allocated state buffers: state lives in fixed-size arrays, no per-token alloc
//! - Branch-free inner loop: state update is pure FMA, no conditional branching
//! - Cache-friendly: state is stored head-major for sequential access per head

pub mod forward;
pub mod reference;
#[cfg(feature = "gdn_tree_verify")]
pub mod tree_forward;
pub mod weights;

// Issue 741 T10 Phase C (2026-08-22): the training family moved to
// `riir-train-engine::deltanet` — backward / full_backward / layer_backward /
// attention_backward / model_backward / model_backward_recompute /
// lm_head_lora_train / qv_lora_train (6,291 LOC). The inference halves below
// stay. The one D4 widening the move needed: `forward::expand_heads_into`
// pub(super) -> pub (the recompute path re-derives the head expansion).

// Minimal activation cache (Issue 641 T1/T2). Owns the 5-field-per-DeltaNet-
// layer / 2-field-per-attention-layer activation contract shared between the
// GPU forward (`riir-gpu`) and the CPU recomputation backward (T3, now in
// `riir-train-engine::deltanet::model_backward_recompute`).
// Lives in the engine (not `riir-gpu`) to avoid a dependency cycle; `riir-gpu`
// re-exports these types for back-compat.
pub mod minimal_activation_cache;

// lm_head-only LoRA training lived here (Plan 528 T1.3) until Issue 741 T10
// Phase C moved it to `riir-train-engine::deltanet::lm_head_lora_train`.

// Q+V LoRA adapter for ternary DeltaNet layers (Plan 334 T2.3 / Issue 448 T7).
// Rank-r adapter on the compact Q and V projections of one DeltaNet layer.
// Consumes `LoraTargetGrad` from the model backward (now in
// `riir-train-engine::deltanet::model_backward`) and produces LoRA
// parameter gradients for AdamW. Sibling to `lm_head_lora_train` (also
// relocated) but for mid-layer / full-layer arms (B/C) instead of the
// lm_head-only arm A. INFERENCE-half adapter state stays here; the training
// step composition is `riir-train-engine::deltanet::qv_lora_train`.
#[cfg(feature = "deltanet_ternary_inference")]
pub mod qv_lora;

// Ternary-weight structs for the hybrid DeltaNet/Attention model (Issue 594).
// Mirrors `weights` but holds TernaryGroupWeights for the 12 projections +
// wte + lm_head. Requires the q2_0_ternary_bridge (for repack_q2_0_to_ternary_group)
// on top of deltanet_inference.
#[cfg(feature = "deltanet_ternary_inference")]
pub mod ternary_weights;

// Hadamard-folded (rotated) ternary support — Bonsai 2 (Issue 980).
// The MODULE compiles unconditionally (the forward dispatches on
// `weights.rotation: Option<_>`); the LOADER only parses `prism.hadamard.*`
// under the `bonsai2_hadamard` feature, so a folded file refuses to load —
// and rotation stays `None` — without it. Flag-off behavior on pre-rotation
// files is byte-identical (G3).
pub mod rotation;

// Ternary-weight forward pass (Issue 594). Mirrors `forward` but uses
// simd_ternary_group_matvec for the 12 projections/layer + wte + lm_head.
#[cfg(feature = "deltanet_ternary_inference")]
pub mod ternary_forward;

// Per-component profiling for the ternary forward (Issue 603). Instrumented
// copy of `ternary_forward` with `Instant::now()` around each section. Used
// to localize the ~370 ms/token gap not covered by GPU FFN + input_proj paths.
#[cfg(feature = "forward_profiling")]
pub mod profiling;

// FlashMemory periodic sparse attention for GQA layers (Issue 584 Phase 2).
// Wires katgpt-attn's GQA block cache + selector into Bonsai/Qwen3.5
// attention layers. Opt-in `flashmemory_gqa` feature.
#[cfg(feature = "flashmemory_gqa")]
pub mod flashmemory_gqa;

pub use forward::{
    DeltaNetState, HybridCache, HybridForwardScratch, PrefillContext, effective_rotary_dim,
    forward_qwen_deltanet, generate_greedy_qwen_deltanet, prefill_qwen_deltanet,
    prefill_qwen_deltanet_into,
};
pub use minimal_activation_cache::{
    DeltanetMinimalActs, MinimalActivationCache, MinimalLayerActivations,
};
pub use reference::{RecurrenceOutput, deltanet_recurrence_prefill, deltanet_recurrence_reference};
#[cfg(feature = "deltanet_ternary_inference")]
pub use ternary_forward::{
    forward_qwen_deltanet_ternary, forward_qwen_deltanet_ternary_hidden,
    forward_qwen_deltanet_ternary_with_capture, forward_qwen_deltanet_ternary_with_hook,
};
#[cfg(feature = "deltanet_ternary_inference")]
pub use ternary_weights::{DeltaNetTernaryLayerWeights, QwenDeltaNetTernaryWeights};
pub use weights::{DeltaNetLayerWeights, Proj, QwenDeltaNetWeights};
// The model_backward_recompute root re-export (qwen_deltanet_model_backward_with_recomputation
// + the 2 recompute_* fns) moved with the module to
// `riir_train_engine::deltanet` (Issue 741 T10 Phase C).
