//! The clef-text decision lane — plan 619: seat **clef-text-0.6b**
//! (Apache-2.0, Qwen3-0.6B + the Clef joint schema head, distilled from
//! clef-flash) as an in-process decision lane on the M3's Neural Engine,
//! on substrate we already own: the [`riir_infer_ane`] host pattern
//! generalized by exactly two new mechanics — the **multi-function
//! package** (`MLModelConfiguration.functionName = "L{bucket}"`, carried
//! by the shared host's `load_model`) and the **decoder→head two-graph
//! pipeline** with span-mean head packing.
//!
//! Grounded in `.research/012_Clef_Text_0_6b_CoreML_ANE_Lane.md`
//! (FluidUse `@ 0230049c`, Apache-2.0; HF revision sha-pinned there).
//! Plan: `.plans/619_clef_text_0_6b_ane_lane.md`.
//!
//! Default build compiles EMPTY (the laya-riir posture). The lane module
//! gates on `all(feature = "clef-ane", target_os = "macos")` — the
//! artifact contract is macOS Core ML, and the default posture of every
//! consumer stays untouched. Never a default feature anywhere; every
//! promotion beyond opt-in is owner-gated (plan 619's gate summary).
//!
//! Tasks: T1.2 artifact store → T1.3 loaders → T1.4 record encoder →
//! T1.5 host math → T1.6 the multi-function host → T1.7 `ClefText::
//! answer` → T2.1–T2.3 the parity gate + bench (the G-clef GOAT) →
//! T3 the reflex local posture (executed in riir-reflex).

/// The lane (feature `clef-ane`, macOS) — compiles to nothing everywhere
/// else.
#[cfg(all(feature = "clef-ane", target_os = "macos"))]
pub mod lane;
