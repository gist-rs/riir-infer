//! `DeltaNet` model types for hybrid DeltaNet/Attention inference (Plan 182).
//!
//! Provides weight structs, weight loading, CPU reference recurrence, and
//! CPU forward pass for hybrid models like Qwen 3.5 that mix `DeltaNet`
//! (linear recurrent) layers with standard attention layers.
//!
//! # Architecture
//!
//! `DeltaNet` layers replace attention with a Gated `DeltaNet` recurrence:
//! ```text
//! g_t = exp(-exp(A_log) * softplus(a + dt_bias))   // decay gate
//! β_t = sigmoid(b_t)                               // update rate
//! q̃ = l2norm(q), k̃ = l2norm(k)                   // normalized Q/K
//! S = g·S + k̃ ⊗ β(v − (g·S)^T k̃)               // delta rule update
//! output = q̃^T · S / √d                           // state read
//! ```
//!
//! Each head maintains a [`key_dim` × `val_dim`] state matrix.
//! For Qwen 3.5-0.8B: 16 heads × (128 × 128) = 2 MB recurrent state total.
//!
//! # Optimization Alignment (per .contexts/optimization.md)
//!
//! - Pre-allocated state buffers: state lives in fixed-size arrays, no per-token alloc
//! - Branch-free inner loop: state update is pure FMA, no conditional branching
//! - Cache-friendly: state is stored head-major for sequential access per head

pub mod forward;
// katgpt-rs Issue 929 / Research 614 (DiscoLoop) — the Bonsai-2
// loop-alignment probe machinery shared by the runner bin and the machinery
// gates (K caller-side weight-shared stack re-entries at the answer
// position + the per-loop cos/margin readout in the decode basis).
// Measurement-only; OPT-IN behind the root `issue929_bonsai2_probe` feature.
#[cfg(feature = "issue929_bonsai2_probe")]
pub mod loop_probe;
pub mod reference;
#[cfg(feature = "gdn_tree_verify")]
pub mod tree_forward;
pub mod weights;

// Differential KV eviction wired onto the hybrid attention KV path (Issue
// 012 — the model-bound quality gate for katgpt-rs Issue 882 P3 / Bench
// 894). Per-(layer, head) tables observe the post-softmax rows the forward
// already computes; eviction to a per-layer budget selects through the
// shipped sink-exempt rule and compacts the cache in place. The forward
// seam is `Option<&mut EvictorState>`: None is the existing path
// (bit-identical), armed with budget ≥ context is also bit-identical (T3).
#[cfg(feature = "kv_eviction")]
pub mod kv_evict;

// Issue 049 — the GDN delta-roll state journal (Research 629 E2's
// riir-infer half): post-commit state recovery by rank-1 unwind from the
// final state + per-token (u, k, β, g) contexts (~64× vs materialized
// per-token snapshots at K = V = 128), under the load-bearing per-head
// decay-product conditioning gate. Complements the rollback-free verify
// path (verify stays write-free); the named consumer seams are the
// spec-decode retry paths, kv_evict eviction recovery, and the
// disaggregated handoff lane. f32 unwind is exact algebra, NOT bit-exact —
// the G1 drift pin lives in `tests/gdn_delta_roll_gates.rs` (Bench 885).
#[cfg(feature = "gdn_delta_roll")]
pub mod delta_roll;

// Issue 741 T10 Phase C (2026-08-22): the training family moved to
// `riir-train-engine::deltanet` — backward / full_backward / layer_backward /
// attention_backward / model_backward / model_backward_recompute /
// lm_head_lora_train / qv_lora_train (6,291 LOC). The inference halves below
// stay. The one D4 widening the move needed: `forward::expand_heads_into`
// pub(super) -> pub (the recompute path re-derives the head expansion).

// Minimal activation cache (Issue 641 T1/T2). Owns the 5-field-per-DeltaNet-
// layer / 2-field-per-attention-layer activation contract shared between the
// GPU forward (`riir-gpu`) and the CPU recomputation backward (T3, now in
// `riir-train-engine::deltanet::model_backward_recompute`).
// Lives in the engine (not `riir-gpu`) to avoid a dependency cycle; `riir-gpu`
// re-exports these types for back-compat.
pub mod minimal_activation_cache;

// lm_head-only LoRA training lived here (Plan 528 T1.3) until Issue 741 T10
// Phase C moved it to `riir-train-engine::deltanet::lm_head_lora_train`.

// Q+V LoRA adapter for ternary DeltaNet layers (Plan 334 T2.3 / Issue 448 T7).
// Rank-r adapter on the compact Q and V projections of one DeltaNet layer.
// Consumes `LoraTargetGrad` from the model backward (now in
// `riir-train-engine::deltanet::model_backward`) and produces LoRA
// parameter gradients for AdamW. Sibling to `lm_head_lora_train` (also
// relocated) but for mid-layer / full-layer arms (B/C) instead of the
// lm_head-only arm A. INFERENCE-half adapter state stays here; the training
// step composition is `riir-train-engine::deltanet::qv_lora_train`.
#[cfg(feature = "deltanet_ternary_inference")]
pub mod qv_lora;

// Ternary-weight structs for the hybrid DeltaNet/Attention model (Issue 594).
// Mirrors `weights` but holds TernaryGroupWeights for the 12 projections +
// wte + lm_head. Requires the q2_0_ternary_bridge (for repack_q2_0_to_ternary_group)
// on top of deltanet_inference.
#[cfg(feature = "deltanet_ternary_inference")]
pub mod ternary_weights;

// Hadamard-folded (rotated) ternary support — Bonsai 2 (Issue 980).
// The MODULE compiles unconditionally (the forward dispatches on
// `weights.rotation: Option<_>`); the LOADER only parses `prism.hadamard.*`
// under the `bonsai2_hadamard` feature, so a folded file refuses to load —
// and rotation stays `None` — without it. Flag-off behavior on pre-rotation
// files is byte-identical (G3).
pub mod rotation;

// Ternary-weight forward pass (Issue 594). Mirrors `forward` but uses
// simd_ternary_group_matvec for the 12 projections/layer + wte + lm_head.
#[cfg(feature = "deltanet_ternary_inference")]
pub mod ternary_forward;

/// Run `f` on a worker of rayon's global pool and return its result; inline
/// when already on one. Called from a non-pool thread, `rayon::scope` injects
/// the closure into the pool and parks the caller on a latch, so a forward
/// whose projections fork onto the pool does not add a 17th runnable thread
/// beside 16 spin-yielding workers (see `forward_qwen_deltanet_ternary_with_hook`).
pub(crate) fn on_rayon_worker<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_some() {
        return f();
    }
    let mut out = None;
    rayon::scope(|_| out = Some(f()));
    out.expect("rayon::scope runs its body to completion")
}

/// Unit-test config for this lane (Issue 047): the real Qwen-DeltaNet preset
/// with only the multi-GiB surfaces shrunk.
///
/// The production preset carries `vocab_size` 151936 (embeddings allocated
/// TWICE by [`crate::deltanet::weights::QwenDeltaNetWeights::zeros`] — the
/// tied `lm_head` clones `wte`, ~2.3 GiB) and `block_size` 32768 (eager
/// `KVCache::new` = 512 MiB per attention layer). ~20 unit tests at that
/// preset run concurrently under the default test-thread count
/// (`available_parallelism`, 24 here) and exhaust the Windows commit limit;
/// the resulting `handle_alloc_error` abort surfaces as the misleading
/// `STATUS_STACK_BUFFER_OVERRUN` (0xc0000409) exit. The property under test
/// (prefill == sequential, zero-weights argmax, shape-vs-config, KV
/// population) holds at any dims, so shrink only `vocab_size` / `block_size`
/// / `mlp_hidden`; `n_embd` and the head geometry stay real. The ignored
/// `bench_batched_prefill_vs_decode` keeps the full preset as the real-dims
/// lane.
#[cfg(test)]
pub(crate) fn unit_test_config(
    n_layer: usize,
    layer_types: Vec<crate::types::DeltaNetLayerType>,
) -> crate::types::Config {
    let mut config = crate::types::Config::qwen_deltanet(n_layer, layer_types);
    // 512 keeps the hardcoded prompt ids in `test_generate_greedy_multi_token_prompt`
    // (`[bos, 100, 200, 300]`) valid; prompts embed `wte[token * n_embd]`.
    // (EOS is the `QWEN_DELTANET_EOS` const, not a config field.)
    config.vocab_size = 512;
    config.bos_token = 1;
    config.block_size = 256;
    config.mlp_hidden = 512;
    config
}

/// The TERNARY decode forwards' entry: with `decode_spin_pool` compiled in
/// and switched on, run `f` on THIS thread with the process-wide decode spin
/// pool installed (katgpt-rs Issue 936), so the Q2 / bit-plane q8 kernels and
/// the GDN / attention-head forks go to spinning workers instead of rayon.
/// Otherwise (switch off, 0-worker pool, or another thread holds the pool)
/// [`on_rayon_worker`]. Scoped to the ternary forwards: the f32 forward's
/// kernels are not pool-aware, and idle spinners would compete with rayon.
///
/// Gated with its consumers (every call site lives behind a set implying
/// `deltanet_ternary_inference` — ternary_forward/ternary_kvq/profiling);
/// compiling it at plain `deltanet_inference` is dead code there (Issue 046
/// surfaced the posture when the spectral spec test made it a readable build).
#[cfg(feature = "deltanet_ternary_inference")]
pub(crate) fn on_decode_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    #[cfg(feature = "decode_spin_pool")]
    let f = match decode_pool::try_run(f) {
        Ok(r) => return r,
        Err(f) => f,
    };
    on_rayon_worker(f)
}

/// The process-wide decode spin pool (katgpt-rs Issue 936 T3).
#[cfg(feature = "decode_spin_pool")]
pub mod decode_pool {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicBool, Ordering};

    static POOL: OnceLock<katgpt_core::SpinPool> = OnceLock::new();
    /// Held by the one thread whose forward currently has the pool installed.
    static BUSY: AtomicBool = AtomicBool::new(false);
    static ON: AtomicBool = AtomicBool::new(true);

    thread_local! {
        /// This thread is inside a pool-installed forward (nested calls run inline).
        static INSIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Turn the decode spin pool on or off; takes effect on the next forward.
    pub fn set_decode_spin_pool(on: bool) {
        ON.store(on, Ordering::Relaxed);
    }

    /// Run `f` on this thread with the pool installed, or hand `f` back
    /// (`Err`) when the switch is off or another thread holds the pool.
    pub(crate) fn try_run<R, F: FnOnce() -> R>(f: F) -> Result<R, F> {
        if INSIDE.with(|c| c.get()) {
            return Ok(f());
        }
        // A 0-worker pool (the off-macOS default) is never used: hand `f`
        // back so the forward keeps Bench 880's rayon-worker posture.
        let pool = POOL.get_or_init(katgpt_core::SpinPool::for_this_machine);
        if !ON.load(Ordering::Relaxed) || pool.workers() == 0 || BUSY.swap(true, Ordering::Acquire)
        {
            return Err(f);
        }
        struct Release;
        impl Drop for Release {
            fn drop(&mut self) {
                INSIDE.with(|c| c.set(false));
                BUSY.store(false, Ordering::Release);
            }
        }
        let _release = Release;
        INSIDE.with(|c| c.set(true));
        Ok(katgpt_core::with_spin_pool(pool, f))
    }
}
#[cfg(feature = "decode_spin_pool")]
pub use decode_pool::set_decode_spin_pool;

// Issue 919 T3 cell 2 — the ternary/GDN forward over a generic quantized KV
// cache (measurement lane; the G0 control pins bit-identity with the plain
// forward). Gated with `ternary_forward`, whose code it mirrors.
#[cfg(feature = "deltanet_ternary_inference")]
pub mod ternary_kvq;

// Issue 014 — the activation tap plan (linear-INPUT mapping + visit order)
// shared by the diagonal collector (T1) and the refit/retention-walk
// instrument (T2/T3). Both instruments must agree on the mapping or the
// diagonal mis-attributes, so it lives here; the hook asserts it per token.
#[cfg(feature = "act_diagonal_calibration")]
pub mod act_taps;

// Per-component profiling for the ternary forward (Issue 603). Instrumented
// copy of `ternary_forward` with `Instant::now()` around each section. Used
// to localize the ~370 ms/token gap not covered by GPU FFN + input_proj paths.
#[cfg(feature = "forward_profiling")]
pub mod profiling;

// FlashMemory periodic sparse attention for GQA layers (Issue 584 Phase 2).
// Wires katgpt-attn's GQA block cache + selector into Bonsai/Qwen3.5
// attention layers. Opt-in `flashmemory_gqa` feature.
#[cfg(feature = "flashmemory_gqa")]
pub mod flashmemory_gqa;

#[cfg(feature = "gdn_delta_roll")]
pub use delta_roll::{ConditioningRefusal, GdnStateJournal, MIN_PRODUCT_EPS_DEFAULT};
pub use forward::{
    DeltaNetState, HybridCache, HybridForwardScratch, PrefillContext, effective_rotary_dim,
    forward_qwen_deltanet, generate_greedy_qwen_deltanet, prefill_qwen_deltanet,
    prefill_qwen_deltanet_into,
};
#[cfg(feature = "kv_eviction")]
pub use forward::{
    forward_attention_layer_evictable, forward_qwen_deltanet_evictable,
    prefill_qwen_deltanet_chunk_into,
};
#[cfg(feature = "kv_eviction")]
pub use kv_evict::{EvictLayerConfig, EvictPolicy, EvictorState, LayerEvict};
pub use minimal_activation_cache::{
    DeltanetMinimalActs, MinimalActivationCache, MinimalLayerActivations,
};
pub use reference::{RecurrenceOutput, deltanet_recurrence_prefill, deltanet_recurrence_reference};
#[cfg(all(feature = "deltanet_ternary_inference", feature = "multi_host"))]
pub use ternary_forward::forward_qwen_deltanet_ternary_range;
#[cfg(feature = "deltanet_ternary_inference")]
pub use ternary_forward::{
    forward_qwen_deltanet_ternary, forward_qwen_deltanet_ternary_hidden,
    forward_qwen_deltanet_ternary_with_capture, forward_qwen_deltanet_ternary_with_hook,
};
#[cfg(feature = "deltanet_ternary_inference")]
pub use ternary_weights::{DeltaNetTernaryLayerWeights, QwenDeltaNetTernaryWeights, WeightScope};
pub use weights::{DeltaNetLayerWeights, Proj, QwenDeltaNetWeights};
// The model_backward_recompute root re-export (qwen_deltanet_model_backward_with_recomputation
// + the 2 recompute_* fns) moved with the module to
// `riir_train_engine::deltanet` (Issue 741 T10 Phase C).
