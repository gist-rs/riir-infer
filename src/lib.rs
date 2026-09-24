//! # riir-infer-core — the model-based inference layer
//!
//! Proposal 041 / T1.1: the model layer extracted out of `riir-engine`
//! (transformer, deltanet, quantization, GGUF loading, CPU reference
//! paths). This crate is **modelless-inference shaped**: weight loading,
//! quantized forward passes, and architecture definitions — zero
//! cognition imports (measured, T1.4: the only non-move-set references
//! are the upstream `katgpt_quant::turboquant` re-export), zero training
//! code (Issue 741 evicted Tier C to riir-train-engine first).
//!
//! **Phase 1 layout:** the crate lives inside riir-ai and every module is
//! re-exported from `riir_engine` at the SAME paths (T1.2, the Issue 739 /
//! Bench 723 shape), so no consumer import changes. Phase 2 (repo
//! promotion) is gated on T1.5 build-cost evidence — see
//! `.proposals/041_riir_infer_model_based_inference_split.md`.
//!
//! Modules move in dependency order: leaves first (`types`, `simd`,
//! `rope`, `gemma_layer`, `llama_layer`, `quant`, `wall`,
//! `safetensors_loader`), then the interdependent blob atomically
//! (`transformer` + `deltanet` + `ternary_layer` + `spec_types` +
//! `dflash` + `gguf_loader` — a 3-cycle forces one chunk).

// ── Chunk A: leaves (moved 2026-08-27, Proposal 041 T1.1) ──────────────

/// Shared engine type aliases — pure leaf, zero `crate::` imports at move
/// time. 253 reference sites in riir-engine resolve through the re-export.
pub mod types;

/// SIMD helpers used by the model layer (leaf).
pub mod simd;

/// `RoPE` tables (leaf).
pub mod rope;

/// Gemma architecture layer helpers (leaf).
pub mod gemma_layer;

/// Llama architecture layer helpers (leaf).
pub mod llama_layer;

/// Quantization primitives: `Q2_0` / `Q4_K` / `Q5_K` / `Q6_K` / `Q8_KV`.
pub mod quant;

/// WALL attention (leaf; depends only on `simd`).
pub mod wall;

/// Wall Attention CONFIGURATION (Issue 019 Phase C.1 de-fork; re-homed
/// from riir-engine by Plan 610 S5 — the gemma-cluster unlock). A pure
/// re-export of the canonical `katgpt_types::WallConfig` behind the
/// `wall_attention` feature; the engine re-exports this module at its
/// historical `riir_engine::wall_config` path, so every consumer path
/// (riir-gpu's wall_decode/wall_mla, katgpt config plumbing) is unchanged.
pub mod wall_config;

/// safetensors weight loading (leaf; depends on `gemma_layer` + `types`).
pub mod safetensors_loader;

// ── Chunk B: the interdependent blob (moved atomically — 3-cycle
// transformer → ternary_layer → deltanet → transformer, 2026-08-27) ────

/// The transformer model layer (gemma2/gemma4/llama/mtp/raven/dllm/prefill
/// + tests). Largest module in the crate.
pub mod transformer;

/// The `DeltaNet` model layer (weights, forward, ternary weights/forward,
/// `flashmemory_gqa`, `tree_forward`, profiling). Gated exactly as in
/// riir-engine.
#[cfg(feature = "deltanet_inference")]
pub mod deltanet;

/// Ternary-weight transformer structs — only compiles with the ternary
/// substrate on (mirrors the engine-side gate + comment).
#[cfg(feature = "ternary_inference")]
pub mod ternary_layer;

/// Speculative-decode spec types.
pub mod spec_types;

/// `DFlash` drafter.
pub mod dflash;

/// GGUF weight loading.
pub mod gguf_loader;

/// SentencePiece/BPE tokenizers for GGUF-embedded vocabularies (native-only;
/// the sentencepiece-sys C++ backend cannot compile for wasm32).
#[cfg(not(target_arch = "wasm32"))]
pub mod tokenizer;

// Upstream re-export the model layer consumes (`crate::turboquant` —
// transformer/mod.rs + tests). Same gated form as riir-engine's copy;
// riir-engine keeps its own for its consumers.
#[cfg(feature = "turboquant")]
pub use katgpt_quant::turboquant;

// Issue 832 — NaN-safe comparator gates for the float sort sites fixed
// under this issue (triage + fix record: .issues/832_*).
#[cfg(test)]
mod issue832_float_order_tests;
