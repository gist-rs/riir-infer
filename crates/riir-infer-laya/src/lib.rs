//! The pinned-checkpoint encoder lane: tokenizer/config/weights substrate
//! over a flat-`Vec<f32>` forward with two compute backends (CPU `gemm` +
//! the macOS Metal MSL family).
//!
//! Layout:
//! - [`pyjson`] — the Python-JSON byte-format writer (UNGATED: one DRY
//!   home for every consumer; a lane's sequence rendering and a harness's
//!   modelless-lane state strings must be the SAME bytes);
//! - [`laya`] — the lane itself (feature `laya-riir`): checkpoint configs,
//!   the pinned BPE tokenizer, locate/verify/download weights, the answer
//!   envelopes, the temperature law, the script detector, and the `riir`
//!   forward (encoder / head / agent + the backend seam).
//!
//! Zero `riir-*` dependencies (fence-gated). Consumers path-dep this crate
//! behind the `laya-riir` / `laya-riir-metal` feature names; the macOS
//! metal arm is target-scoped so non-macOS hosts compile the feature to
//! nothing instead of failing at the dep tree.

/// The Python-JSON byte-format writer (the ONE writer for it — DRY home).
pub mod pyjson;

/// The pinned-checkpoint lane (substrate + the riir-owned forward).
#[cfg(feature = "laya-riir")]
pub mod laya;
