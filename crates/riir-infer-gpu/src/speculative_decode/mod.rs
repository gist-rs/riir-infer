//! Issue 665 — Speculative decoding for Metal decode.
//!
//! Modelless CPU-side draft model + sequential speculative verify path.
//!
//! ## Architecture (sequential speculative verify — NOT batched verify)
//!
//! The Delta rule recurrence (Issue 652) makes batched verify infeasible —
//! the chunkwise-parallel algorithm G1 FAILED. Instead, we use **sequential
//! speculative verify**: queue K decode passes on the GPU without intermediate
//! syncs, read all K logits at once, verify. The DeltaNet recurrent state
//! chains correctly across the K passes (GPU processes dispatches in order).
//!
//! ## Speedup ceiling (Issue 665 T1 honest assessment)
//!
//! The sync overhead is only 13.1% of decode (Bench 669). Eliminating it
//! entirely caps the speedup at 1.15×. At K=2 with realistic n-gram acceptance
//! (α=0.7 on repetitive text), expected speedup is ~1.17×. On creative text
//! (α=0.4), speculative decoding is net-negative.
//!
//! ## Modules
//!
//! - [`ngram_drafter`] — N-gram frequency-table draft model (modelless, CPU-only)
//!   + verbatim lookup fill (Issue 742 T2: K=16 prompt-lookup-decoding fill,
//!     in-vocab by construction, bounded fabricated fallbacks)

pub mod dspark_drafter;
pub mod ngram_drafter;
pub mod regime_router;

pub use dspark_drafter::{BlockOut, DrafterCache, DsparkDrafter};
pub use ngram_drafter::{
    LookupOutcome, NgramDrafter, DEFAULT_MAX_DRAFT, DEFAULT_MAX_LOOKUP_ORDER,
    DEFAULT_NGRAM_ORDER,
};
pub use regime_router::{
    Lane, MissMonitor, NeedleWatch, NoveltyVerdict, RegimeRouterConfig, RoutePlan, SwitchVerdict,
    TaskHint, WatchMode, route_plan,
};
