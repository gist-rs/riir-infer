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
//! - [`accum`] — `PairCosineAccum`, the streaming f64 dot+norm pair
//!   accumulator (T1.1; the cosine_distance reduction discipline).
//! - [`smatrix`] — the streaming S-matrix builder: layer-major capture
//!   arrival, position buckets, the cosine meter + the opt-in SVCCA meter
//!   arm (T1.3).
//! - [`partition`] — the two-pass min-max DP + the PRE-REGISTERED ε grid
//!   and kill rule (T1.6) + the type-split forced-min floor (T2.1/T2.2).
//! - [`synth`] — deterministic planted corpora for the gates (T1.5).
//! - [`delta`] — the ΔS quant-damage map (T1.4, read-only diagnostic).
//! - [`audition`] — the Phase-3 zero-training surrogate pool math (the
//!   mean/RDSC merges) + the T3.3 per-channel branch-correction fit + the
//!   T3.2 selection pin (pure over `f32` slices; the laya-coupled apply
//!   half lives in the `twt_laya_audition` example, dev-dep direction).
//!
//! What Phase 3 does NOT ship here: the collapsed-GGUF writer (Phase 4),
//! the GOAT gate (Phase 5). The laya capture hook lives in
//! `riir-infer-laya` (`Encoder::forward_capture`, its own `twt_profile`
//! feature); the Bonsai/GDN capture sibling is an open T1.2 half.

pub mod accum;
pub mod audition;
pub mod delta;
pub mod partition;
pub mod smatrix;
pub mod synth;

pub use accum::PairCosineAccum;
pub use audition::{
    mean_sq_err, merge_mean, merge_rdsc, selection_pin, CandRow, CorrectionFit, SelectionPin,
};
pub use delta::{delta, localize_by_block, DeltaMap};
pub use partition::{
    brute_force_optimal, forced_min_blocks, kill_verdict, minmax_partition,
    partition_worst, Block, KillVerdict, KILL_FRACTION, KILL_MIN_MIDDLE_BLOCK,
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
}
