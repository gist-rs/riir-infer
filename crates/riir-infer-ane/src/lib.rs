//! The SHARED Core ML / Apple Neural Engine host — plan 619 T1.0, the
//! substrate-first extraction of the host mechanics `riir-infer-laya`'s
//! `ane.rs` owned into ONE home. Four mechanic groups, each lifted
//! verbatim (same algorithms, byte-identical error strings):
//!
//! 1. **artifact digest + verify + stage** — the blake3-dir-v1 bundle
//!    digest (sorted relative path, `rel \0 data \0` per file), the
//!    load-time verify (count / byte / digest walls), and the staging
//!    half of download-on-demand assembly (per-file `.part` + rename
//!    choreography is the CALLER's fetch closure — the wire primitive
//!    stays with each lane's weights substrate);
//! 2. **the content-addressed `.mlmodelc` compile cache** — cache key IS
//!    the digest, process-unique staging + atomic claim (the
//!    shared-fixed-path lesson, mechanized); the cache root is
//!    parameterized per lane (env name + default `~/Library/Caches/
//!    riir-infer/<lane>` subtree) so existing compiled bundles stay hit;
//! 3. **`MLModel` load with compute units + `functionName`** — the
//!    multi-function-package selector plan 619's clef lane needs, plus
//!    the load-time compute-plan walk ([`Placement`], the no-silent-
//!    fallback gate re-measured at every load);
//! 4. **the zero-copy `MLMultiArray` helpers** — fp16 input arrays over
//!    caller buffers (no-op deallocator), the documented
//!    `getBytesWithHandler` output reader, and the whole
//!    provider→prediction→output-lookup choreography in one
//!    predict call so no objc2 type ever crosses to a lane.
//!
//! Consumers: `riir-infer-laya` (feature `laya-riir-ane`) and
//! `riir-infer-clef` (plan 619). Everything Core ML touches runs inside
//! the caller's autoreleasepool discipline — unchanged from the lane
//! this was lifted from.
//!
//! macOS-only: off-macOS this crate is a documented stub and resolves
//! zero dependencies (the target-scoped dep posture). Never wasm32,
//! never a default.

// The one error type: a rendered message (every host failure is a
// string-rendered refusal — the lane maps it onto its own error enum
// verbatim, keeping the historical text byte-identical).
#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct Error(String);

#[cfg(target_os = "macos")]
impl Error {
    pub(crate) fn msg(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

#[cfg(target_os = "macos")]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(target_os = "macos")]
impl std::error::Error for Error {}

#[cfg(target_os = "macos")]
pub(crate) type Result<T> = std::result::Result<T, Error>;

// ── group 1: artifact digest + verify + stage ────────────────────────────

#[cfg(target_os = "macos")]
mod digest;

#[cfg(target_os = "macos")]
pub use digest::{blake3_dir_v1, collect_files, stage_files, verify_dir_digest};

// ── groups 2 + 3 + 4: compile cache, model load (compute units +
//    functionName) + plan walk, and the multi-array predict
//    choreography ─────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod cache;

#[cfg(target_os = "macos")]
mod model;

#[cfg(target_os = "macos")]
pub use model::{load_model, ArtifactPin, ComputeUnits, LoadedModel, Placement};

// Off-macOS: a documented stub that compiles to nothing and resolves zero
// dependencies — enabling a consumer's ANE feature on a non-macOS host is
// INERT (never a dep-tree failure), the laya-ane posture this crate
// inherits. The macOS-gated consumer modules stay cfg'd out there.
