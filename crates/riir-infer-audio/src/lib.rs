//! The audio model lane — `.issues/015`: loaders + streaming serving
//! state for PUBLISHED CoreML audio bundles, on substrate we already
//! own: the [`riir_infer_ane`] host (plan 619 T1.0's shared Core ML /
//! ANE home — this crate never grows a second copy of the machinery).
//! First lane: **silero-vad** (FluidInference `silero-vad-coreml`,
//! Apache-2.0; the pre-compiled `silero-vad-unified-256ms` bundle —
//! 256 ms chunks, recurrent hidden/cell state carried between chunks,
//! the streaming-state serving shape).
//!
//! Grounded in `.research/001_FluidInference_OnDevice_Audio_Stack.md`
//! (F1); recon + T1 cost-model prediction live in the issue file. The
//! lane loads EXTERNAL published `.mlmodelc` bundles through the host's
//! precompiled path — `compileModelAtURL` rejects a compiled bundle as
//! input, so the host verifies the digest walls and loads in place.
//!
//! Default build compiles EMPTY (the clef-ane posture). The lane module
//! gates on `all(feature = "vad-ane", target_os = "macos")` — the
//! artifact contract is macOS Core ML, and the default posture of every
//! consumer stays untouched. Never a default feature anywhere; every
//! promotion beyond opt-in is owner-gated (`.issues/015` T6's gate).
//!
//! Tasks: T1 the cost-model prediction (RECORDED, in the issue file) →
//! T2 the precompiled load path → T3 the three synthetic verdicts +
//! per-chunk latency → T6 the verdict doc + the GOAT gate before any
//! promotion. T4 (fluidaudio-rs FFI) and T5 (ONNX/ort) stay parked
//! unless T2 fails — it did not.

/// The silero-vad lane (feature `vad-ane`, macOS) — compiles to nothing
/// everywhere else.
#[cfg(all(feature = "vad-ane", target_os = "macos"))]
pub mod vad;
