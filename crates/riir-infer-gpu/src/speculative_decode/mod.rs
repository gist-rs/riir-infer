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
//! ## Refinement-pass gate (Issue 1158)
//!
//! Every block drafter in this lane family (DSpark here, DFlash2 in
//! `qwen38_dflash2`) is **mask-input**: trained on fully-masked blocks and
//! correct only for single-pass J0 drafting. If a lane ever adds refinement
//! passes (re-feeding drafted tokens to the drafter), the gate is the τ
//! ladder: **a mean accepted length at J1 BELOW J0 is the mask-input collapse
//! signature** (JetSpec++, arXiv:2610.19792 Tables 4/12 — τ 5.86 → 2.39) and
//! the arm must be refused, not tuned. The fix is the token-input +
//! clean-prefix retrain (riir-train Plan 376 T3).
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
    DEFAULT_MAX_DRAFT, DEFAULT_MAX_LOOKUP_ORDER, DEFAULT_NGRAM_ORDER, LookupOutcome, NgramDrafter,
};
pub use regime_router::{
    Lane, MissMonitor, NeedleWatch, NoveltyVerdict, RegimeRouterConfig, RoutePlan, SwitchVerdict,
    TaskHint, WatchMode, route_plan,
};
