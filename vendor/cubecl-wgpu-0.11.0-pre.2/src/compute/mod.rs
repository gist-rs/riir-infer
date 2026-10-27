pub(crate) mod controller;

mod storage;

pub(super) mod mem_manager;
pub(super) mod poll;
pub(super) mod schedule;
mod server;
pub(crate) mod stream;
pub(super) mod timings;

pub use server::*;
pub use storage::*;

/// Issue 660 diagnostics: bind group cache hit/miss counters.
pub use stream::{
    BG_CACHE_HITS, BG_CACHE_MISSES, BIND_GROUP_CACHE_ENABLED,
    PROF_BINDGROUP_NANOS, PROF_DISPATCH_COUNT, PROF_DISPATCH_NANOS, PROF_ENTRIES_NANOS,
    PROF_PASS_GET_NANOS, PROF_SET_BG_NANOS, PROF_SET_IMM_NANOS, PROF_SET_PIPELINE_NANOS,
    PROF_TRANSITIONS_NANOS, REGISTER_PIPELINE_PROFILE_ENABLED,
    reset_bind_group_cache_stats, reset_register_pipeline_profile,
    set_bind_group_cache_enabled, set_register_pipeline_profile_enabled,
};

/// Issue 761 diagnostics: whole-pipeline dispatch decomposition counters
/// (launch / enqueue / flush / submit / GPU-blocked regulate + write polls).
pub use stream::{
    ENQUEUE_EXEC_COUNT, ENQUEUE_EXEC_NANOS, ENQUEUE_WRITE_COUNT, ENQUEUE_WRITE_NANOS,
    FLUSH_COUNT, FLUSH_NANOS, LAUNCH_COUNT, LAUNCH_NANOS, PIPELINE_NANOS, PREPARE_NANOS,
    READ_COUNT, READ_NANOS, REGISTER_NANOS, REGULATE_COUNT, REGULATE_NANOS, SUBMIT_COUNT,
    WRITE_COUNT, WRITE_POLL_COUNT, WRITE_POLL_NANOS, reset_dispatch_profile,
};

/// Issue 761 follow-up 2: uniform-content cache toggle + diagnostics
/// (DEFAULT ON since Bench 771; `RIIR_UNIFORM_CACHE=0` is the kill-switch;
/// overrides for the A/B GOAT gate).
pub use mem_manager::{
    UNIFORM_CACHE_EVICTS, UNIFORM_CACHE_HITS, UNIFORM_CACHE_MISSES,
    reset_uniform_cache_stats, set_uniform_cache_override,
};
