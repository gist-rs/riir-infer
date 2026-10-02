//! TWT phase-collapse lane (riir-infer Issue 022; katgpt-rs Research 594,
//! arXiv:2609.20100) — Phase 1 profiler + Phase 2 partition.
//!
//! The model→model collapse pipeline's first two stages, fully modelless:
//! profile a checkpoint's depthwise phase structure as the L×L expected
//! cosine-discrepancy matrix over a calibration corpus, then partition
//! depth into contiguous ε-redundant blocks by min-max dynamic
//! programming. Opt-in behind `twt_profile` — never a default without the
//! Phase-5 GOAT gate.
//!
//! Layout:
//! - [`accum`] — `PairCosineAccum` re-export shim (Plan 616: the type
//!   lives in katgpt-core `partition`).
//! - [`smatrix`] — the streaming S-matrix builder: layer-major capture
//!   arrival, position buckets, the cosine meter + the opt-in SVCCA meter
//!   arm (T1.3); `SMatrix` itself is the katgpt-core re-export (Plan 616).
//! - [`partition`] — the min-max DP re-export shim (Plan 616: the DP +
//!   pair-distance substrate live in katgpt-core `partition`) + the
//!   PRE-REGISTERED ε grid and kill rule (T1.6, lane-local — its
//!   pre-registration record is the file's git history) + the type-split
//!   forced-min floor (T2.1/T2.2, promoted with the DP).
//! - [`synth`] — deterministic planted corpora for the gates (T1.5).
//! - [`delta`] — the ΔS quant-damage map (T1.4, read-only diagnostic).
//! - [`audition`] — the Phase-3 zero-training surrogate pool math (the
//!   mean/RDSC merges) + the T3.3 per-channel branch-correction fit + the
//!   T3.2 selection pin (pure over `f32` slices; the laya-coupled apply
//!   half lives in the `twt_laya_audition` example, dev-dep direction).
//! - [`ternarize`] — the Phase-4 re-ternarization arms + the T4.2 κ
//!   budget (`twt_collapse`): the deterministic materializers that turn
//!   a merged operator into a DEPLOYABLE tensor (f16-dense / sign-majority
//!   / source-quant), pre-registered, gate-adjudicated.
//! - [`collapse_writer`] — the Phase-4 collapsed-GGUF writer
//!   (`twt_collapse`): reduced layer count + renumbered metadata + the
//!   `twt.*` provenance keys, member passthroughs as byte-copies.
//!
//! What the lane does NOT ship yet: the Bonsai audition (the apply path
//! needs GDN cache snapshot/restore — the Phase-5 prerequisite that
//! picks the real winners), the Phase-5 GOAT gate. The laya capture hook
//! lives in `riir-infer-laya` (`Encoder::forward_capture`, its own
//! `twt_profile` feature); the Bonsai/GDN capture sibling is an open
//! T1.2 half.

pub mod accum;
pub mod audition;
pub mod delta;
pub mod partition;
pub mod smatrix;
pub mod synth;

#[cfg(feature = "twt_collapse")]
pub mod collapse_writer;
#[cfg(feature = "twt_collapse")]
pub mod ternarize;

pub use accum::PairCosineAccum;
pub use audition::{
    mean_sq_err, merge_mean, merge_rdsc, selection_pin, CandRow, CorrectionFit, SelectionPin,
};
pub use delta::{delta, localize_by_block, DeltaMap};
#[cfg(feature = "twt_collapse")]
pub use collapse_writer::{
    emit_collapsed_gguf, twt_arm_codes_value, twt_block_table_value, twt_layer_types_value,
    CollapseSpec, CollapsedStats, LayerSource, LAYER_TYPES_LEGEND, TensorOut,
};
#[cfg(feature = "twt_collapse")]
pub use ternarize::{
    arm_dense_f16, arm_sign_majority, arm_source_quant, budget_ok, budget_ratio,
    materialization_rel_err, DenseF16Weights, TwtArm, KAPPA_BUDGET, ARM_B_TAU_CODE,
};
pub use partition::{
    brute_force_optimal, forced_min_blocks, kill_verdict, minmax_partition,
    partition_worst, Block, KillVerdict, PartitionError, KILL_FRACTION, KILL_MIN_MIDDLE_BLOCK,
    PRE_REGISTERED_EPS_GRID,
};
pub use smatrix::{Bucket, SMatrix, SMatrixBuilder, SMatrices, SvccaCfg};
pub use synth::{planted_corpus, planted_corpus_noisy, Lcg};

/// BLAKE3 of raw bytes — artifact provenance (the corpus digest the
/// capture driver records so a profile is reproducible bit-for-bit).
pub fn blake3_of(bytes: &[u8]) -> String {
    blake3::Hasher::new().update(bytes).finalize().to_hex().to_string()
}

/// Error surface of the TWT lane — every refusal is loud and names the
/// violated precondition.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TwtError {
    #[error("invalid ε {0}: must be finite and >= 0")]
    InvalidEps(f32),
    #[error("ε-grid arity mismatch: got {got} partitions, grid holds {want}")]
    GridMismatch { got: usize, want: usize },
    #[error("begin_forward(0): a forward must carry at least one row")]
    EmptyForward,
    #[error("begin_forward while a forward is still open (end_forward first)")]
    ForwardAlreadyOpen,
    #[error("push before begin_forward")]
    NoForwardOpen,
    #[error("layer {layer} out of range (n_layers = {layers})")]
    LayerOutOfRange { layer: usize, layers: usize },
    #[error("row {row} out of range (rows = {rows})")]
    RowOutOfRange { row: usize, rows: usize },
    #[error("state dim {got} != builder dim {want}")]
    DimMismatch { got: usize, want: usize },
    #[error("(layer {layer}, row {row}) written twice in one forward")]
    DoubleWrite { layer: usize, row: usize },
    #[error("non-finite state at (layer {layer}, row {row}) — refused, never pooled")]
    NonFiniteState { layer: usize, row: usize },
    #[error(
        "incomplete forward: {got} of {want} cells written (layer-major capture must cover every row)"
    )]
    IncompleteForward { got: usize, want: usize },
    #[error(
        "SVCCA probe floor: {got} retained rows <= dim (need >= {need}) — keep n_probe > d (Plan 349 T4.1 pitfall); grow the corpus or shrink the stride"
    )]
    ProbeFloor { got: usize, need: usize },
    #[error("merge member shape mismatch: expected {expected} elements, got {got}")]
    MergeShapeMismatch { expected: usize, got: usize },
    #[error("merge over zero members")]
    EmptyMerge,
    #[error("correction-fit slice mismatch: expected {expected} elements, got total {got}")]
    FitShapeMismatch { expected: usize, got: usize },
    #[error("row count × dim overflow")]
    RowOverflow,
    #[error(
        "selection cross-check: stated argmin index {stated} is not the minimum (actual {actual}) — the candidate table is tampered or the scan reordered it"
    )]
    ArgminMismatch { stated: usize, actual: usize },
    #[error("arm pool over zero members")]
    ArmEmptyPool,
    #[error("arm shape mismatch: expected {expected} elements, got {got}")]
    ArmShapeMismatch { expected: usize, got: usize },
    #[error("non-finite merged value — refused, never materialized")]
    NonFiniteMerged,
    #[error("f16 scale non-finite at (row {row}, group {group}) — merged magnitude overflows f16")]
    NonFiniteScale { row: usize, group: usize },
    #[error("degenerate baseline: ‖Bx‖ == 0 on every calibration row — the relative error is undefined")]
    DegenerateBaseline,
    #[error("block table does not tile [0, n_layer): {reason}")]
    BadBlockTable { reason: &'static str },
    #[error("collapsed plan for block [{start}, {end}) is incomplete: missing tensor suffixes {missing}")]
    IncompletePlan { start: usize, end: usize, missing: String },
    #[error("GGUF serialize failure: {0}")]
    GgufWrite(String),
}
