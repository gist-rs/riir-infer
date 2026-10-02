//! `PairCosineAccum` — Plan 616 PROMOTION SHIM (2026-10-02).
//!
//! The streaming pair-similarity accumulator moved VERBATIM to
//! `katgpt_core::partition` (feature `minmax_partition`, forwarded by this
//! crate's `twt_profile`) and re-exports here at its historical path
//! (`twt::accum::PairCosineAccum` — the mod.rs root re-export and the
//! G4 alloc gate resolve unchanged). Its reduction discipline (single-pass
//! dot + squared norms in f64, no allocation, conservative 1.0 on
//! zero-norm/non-finite) and its tests live with the promoted type.

pub use katgpt_core::partition::PairCosineAccum;
