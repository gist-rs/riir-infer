//! Issue 726 T2 — dual ANE/GPU hybrid prefill: the **eligibility gate**.
//!
//! Accelerates the fused `in_proj_concat` (5120→16480) + `gate_up_proj`
//! (5120→34816) projections of GDN layers on the Apple Neural Engine during
//! ≥2048-token prefill blocks, splitting output channels across ANE (prefix)
//! and the GPU batched GEMM (suffix). Maps omlx PR #2756 (+ #2781: M3 Max
//! +57% prefill @32K) onto `ternary_deltanet_gpu_forward`'s prefill path.
//!
//! ## What T2 ships (this module)
//!
//! The gate contract (issue §"T2/T3 design contract"):
//!
//! ```text
//! p ≥ 2048 && layer_is_gdn && op ∈ {in_proj_concat, gate_up_proj}
//!        && ane_ctx_ready && feature_on
//!   → plan { ane_blocks = p/2048, tail = p%2048, f }
//! ```
//!
//! - [`ane_prefill_eligible`] — the pure gate (fully unit-tested).
//! - [`AnePrefillCtx`] — the runtime context state machine. The ctx only
//!   attempts the ≈ 90 s (real dims, P9) program-bank compile when the
//!   runtime flag was set
//!   BEFORE model construction; any refusal (flag off, compile failure,
//!   budget, partial bank, non-Apple-Silicon host) reports `Unavailable`
//!   and the prefill path runs the existing GPU batched GEMM unchanged —
//!   **fail-open by construction**.
//! - The dispatch seams live in `ternary_deltanet_gpu_forward`
//!   (`ane_prefill_try_inproj` / `ane_prefill_try_gate_up`): feature-off →
//!   compiled out entirely; feature-on + ctx not ready → the split GEMMs.
//!
//! ## What lands in T3
//!
//! The executor: Form C per-row int8 requant from the fused Q2_0 handles
//! (`ws[o] = s_row/127`, `q = round(127·sign·scale_g/s_row)`), the
//! 96-program single-procedure bank (eager compile ≈3 s, P8), serial
//! submission per block (P12: no dual-projection overlap at real dims),
//! segment writes into the split scratch handles, + the suffix/tail GPU
//! complement. The conv shape is fixed at `[1, 5120, 1, 2048]` — tokens
//! along W (the omlx convention; W ≥ 32 is a hard ANE constraint, P7).
//!
//! ## Numerics (P9/P10/P11, measured)
//!
//! fp16 direct is the exact form (5.7 TFLOPS, 2 B/weight); Form C int8 is
//! the perf form (10.9 TFLOPS, 1 B/weight, projection-level cosine 1.000011
//! with realistic scaled-ternary weights — the Q2_0 group scales are NEVER
//! 1.0, so exact int8 is unattainable and unnecessary). G1 arbiter: hidden-
//! state cosine ≥ 0.999 (stretch 0.9999) at T4.

use core::sync::atomic::{AtomicBool, Ordering};
// `ANE_DISPATCH_COUNT` (the only short-name AtomicU64 user) spells the
// qualified path and is macos+aarch64-gated — a plain import would orphan
// on other targets (the P22 sweep's off-platform leg).

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod bank;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod bridge;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod exec;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod headroom;
// Plan 550: the IOSurface zero-copy split executor — needs the metal crate
// (metal_tensor_gemm) for the raw pipeline dispatch on the shared queue.
#[cfg(all(
    feature = "metal_tensor_gemm",
    all(target_os = "macos", target_arch = "aarch64")
))]
pub mod exec_zc;
mod requant;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use bridge::BridgeKernel;
pub use requant::requant_per_row_int8;
pub use requant::TERNARY_GROUP_SIZE;

/// Fixed ANE block size in tokens — the compiled conv spatial width W=2048
/// (P7/P9 verified shape: tokens along W, `[1, 5120, 1, 2048]`).
pub const ANE_PREFILL_BLOCK_TOKENS: usize = 2048;

/// Minimum spatial width the ANE accepts (P7: W < 32 compiles but FAILS at
/// eval — encode the constraint so a misconfigured block fails open instead
/// of producing a runtime eval error deep in the executor).
pub const ANE_PREFILL_MIN_SPATIAL_W: usize = 32;

/// Issue 887 T2 — the ANE channel-slice ROW-GRID. **MEASURED on this
/// compiler 2026-09-21** (`channel_grid_bracket_pins_the_compiler_verdict`):
/// the bracket ran the grid's own premise down — aligned widths (64/128/192)
/// AND misaligned neighbors (48/96/100) all COMPILE and EVAL at both the
/// synthetic and the real in_proj geometry — so the 64-row grid is **NOT a
/// hardware constraint here**. Contrast the SPATIAL axis, where W<32
/// genuinely eval-breaks: [`ANE_PREFILL_MIN_SPATIAL_W`] is a measured floor,
/// this constant is not one. oMLX compiles every ANE output slice on a
/// 64-row grid anyway (128 on dual-ANE; source-verified at the pin —
/// `alignment = 128 if dual_ane else 64`, with `% 64` invariants on every
/// other backend's slice), and the grid stays adopted as that
/// oMLX-faithfulness CONVENTION, with the conservative policy it carries: a
/// prefix whose rows are off-grid keeps the full-width registration (see
/// [`registration_splits`]) — upstream-alignment discipline, not a measured
/// cliff.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub const ANE_PREFILL_CHANNEL_GRID: usize = 64;

/// Issue 887 T1(a) seam layer: which splits a (layer, op) registration hands
/// to the bank. In split-overlap mode the dispatch seams consume ONLY the
/// first segment of each fused program (in_proj: qkv; gate_up: gate) — the
/// remaining rows are the GPU complement's, so compiling them into the ANE
/// program is pure waste (evaluated, then discarded). Registering the
/// prefix split alone makes the channel split REAL at the bank level:
/// bytes, compile cost and eval width all track the ANE's true share, and
/// the per-row requant invariant keeps the dispatched rows bit-identical to
/// a full registration's prefix. `DownProj` never slices (single op, no GPU
/// complement exists — all-or-nothing per token axis). Off-grid prefixes
/// keep today's full-width registration (see [`ANE_PREFILL_CHANNEL_GRID`]).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn registration_splits<'a>(
    split_mode: bool,
    op: AnePrefillOp,
    splits: &'a [&'a katgpt_core::TernaryGroupWeights],
) -> Vec<&'a katgpt_core::TernaryGroupWeights> {
    let sliceable = matches!(op, AnePrefillOp::InProjConcat | AnePrefillOp::GateUpProj);
    if split_mode
        && sliceable
        && splits
            .first()
            .is_some_and(|s| s.rows.is_multiple_of(ANE_PREFILL_CHANNEL_GRID))
    {
        splits[..1].to_vec()
    } else {
        splits.to_vec()
    }
}

/// The accelerated projection ops. Set membership is enforced BY TYPE — the
/// ineligible ops (attention `wq`/`wkv`) have no variant and no seam call
/// site at all. `InProjConcat`/`GateUpProj` map 1:1 onto the omlx PR #2756
/// fused handles; `DownProj` is the Plan 549 stretch op (per-op fail-open —
/// its registration never poisons the bank).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnePrefillOp {
    /// Fused qkv|z|a|b input projection (5120→16480), GDN layers only.
    InProjConcat,
    /// Fused gate|up FFN projection (5120→34816), GDN layers only.
    GateUpProj,
    /// FFN down projection (34816→5120), GDN layers only. Plan 549: behind
    /// its own runtime toggle + non-poisoning registration — the 286 MB
    /// input readback makes its net win an open measurement question.
    DownProj,
}

/// Runtime configuration for the hybrid split. Defaults: half the output
/// channels to the ANE (omlx's ~55% slice), 2048-token blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnePrefillConfig {
    /// Fraction of output channels the ANE computes, in `(0.0, 1.0]`. The
    /// GPU batched GEMM computes the remaining suffix channels for all
    /// tokens plus every tail token's full width.
    pub channel_fraction: f32,
    /// Fixed block size in tokens (must be ≥ [`ANE_PREFILL_MIN_SPATIAL_W`];
    /// the default is the verified [`ANE_PREFILL_BLOCK_TOKENS`]).
    pub block_tokens: usize,
    /// Bank sizing hint (n_layer) — pass the model's layer count.
    pub max_layers_hint: usize,
    /// The T0 memory-budget ceiling for ANE-side int8 copies.
    pub max_ane_bytes: u64,
}

impl Default for AnePrefillConfig {
    fn default() -> Self {
        Self {
            channel_fraction: 1.0,
            block_tokens: ANE_PREFILL_BLOCK_TOKENS,
            max_layers_hint: 0,
            max_ane_bytes: bank_budget_default(),
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn bank_budget_default() -> u64 {
    // Issue 886 T1: the default is TWO-SIDED — min(model-side sizing bound,
    // machine-side headroom). An explicit override (programmatic or
    // RIIR_ANE_MAX_BYTES env) is a deliberate claim and bypasses this
    // resolution entirely (see `prefill_ane_max_bytes_effective`).
    let model_side = bank::DEFAULT_MAX_ANE_BYTES;
    let fraction = headroom_fraction_effective();
    let machine_side = headroom::machine_side_ceiling(fraction);
    let (ceiling, machine_binds) = headroom::two_sided_ceiling(model_side, machine_side);
    // Log the binding side ONCE per resolution call (this runs at most
    // twice per process — AnePrefillConfig::default + the effective fn).
    if machine_binds {
        eprintln!(
            "[ane] bank budget: MACHINE side binds ({:.2} GB <= model {:.2} GB, headroom fraction {fraction}) — the box is the constraint, not the model",
            ceiling as f64 / 1e9,
            model_side as f64 / 1e9
        );
    } else if machine_side.is_none() {
        eprintln!(
            "[ane] bank budget: machine-side measurement UNAVAILABLE — failing toward the model-side ceiling ({:.2} GB; Issue 886 T1's conservative divergence from oMLX's fail-open)",
            model_side as f64 / 1e9
        );
    }
    ceiling
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn bank_budget_default() -> u64 {
    u64::MAX
}

/// Issue 886 T1: the headroom fraction applied to measured available
/// memory when composing the machine-side budget term (oMLX's 0.70
/// phys-footprint gate constant, source-verified at the pin). Env surface
/// `RIIR_ANE_HEADROOM_FRACTION`; values outside `(0.0, 1.0]` fall back to
/// the 0.70 default with a one-time warning (an untrusted input never
/// silently zeroes the machine term). Read at most once per process.
pub fn headroom_fraction_effective() -> f64 {
    static FRACTION: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *FRACTION.get_or_init(|| {
        let default = 0.70;
        match std::env::var("RIIR_ANE_HEADROOM_FRACTION") {
            Ok(v) => match v.parse::<f64>() {
                Ok(f) if f > 0.0 && f <= 1.0 => f,
                _ => {
                    eprintln!(
                        "[ane] RIIR_ANE_HEADROOM_FRACTION='{v}' not in (0.0, 1.0] — using default {default}"
                    );
                    default
                }
            },
            Err(_) => default,
        }
    })
}

/// Issue 887: one-shot latch for the seam width-mismatch fail-open. A
/// registration/dispatch width divergence (e.g. `set_prefill_ane_split`
/// flipped after construction changed which splits were registered) must be
/// LOUD once per process, not once per prefill — every subsequent dispatch
/// cleanly fail-opens to the GPU.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
static WIDTH_MISMATCH_WARNED: AtomicBool = AtomicBool::new(false);

/// See [`WIDTH_MISMATCH_WARNED`].
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn warn_width_mismatch_once(what: &str, needed: usize, have: usize) {
    if !WIDTH_MISMATCH_WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[ane] {what}: program compiled at {have} channels but the dispatch needs {needed} — registration/dispatch mode mismatch (a toggle flipped after construction?). Failing open to the GPU for the rest of this process"
        );
    }
}

/// The dispatch plan produced by a firing gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnePrefillPlan {
    /// Number of full fixed-shape blocks the ANE computes.
    pub ane_blocks: usize,
    /// Remainder tokens after the last full block — ALWAYS the GPU's
    /// (exact fixed-shape blocks only; no synthetic padding, omlx #4).
    pub tail: usize,
    /// The channel split fraction from the config (carried for the
    /// executor + GOAT logging; the ANE computes `[0, f·oc)` channels).
    pub channel_fraction: f32,
    /// `ane_blocks * block_tokens` — the token prefix covered by the ANE.
    pub ane_tokens: usize,
}

impl AnePrefillPlan {
    /// True only when the ANE covers every token of every channel — i.e.
    /// the GPU projection dispatch can be skipped entirely. Partial
    /// coverage (tail > 0 or f < 1) requires the T3 GPU complement.
    pub fn full_coverage(&self) -> bool {
        self.tail == 0 && self.channel_fraction >= 1.0
    }
}

/// Runtime on/off switch (house pattern: `PREFILL_USE_SIMDGROUP` etc.).
/// Default-off — the feature flag opting the crate in does NOT opt the
/// path in; the GOAT harness flips this at T4/T5. Same stance as
/// `PREFILL_USE_METAL_TENSOR` (alternative GEMM path pending e2e GOAT).
static PREFILL_USE_ANE: AtomicBool = AtomicBool::new(false);

/// Plan 549: segment-split overlap mode (the omlx deployed shape). When on,
/// the seam dispatches the ANE on ONE segment of each fused projection while
/// the GPU computes the complement concurrently (in_proj: ANE=qkv ∥
/// GPU=z/a/b; gate_up: ANE=gate ∥ GPU=up) via the split-overlap job API.
/// Requires the base [`PREFILL_USE_ANE`] flag. Default off until the Bench
/// 776 GOAT; promoted per the delegated decision if G1+G2 pass.
static ANE_SPLIT_OVERLAP: AtomicBool = AtomicBool::new(false);

/// See [`ANE_SPLIT_OVERLAP`].
pub fn set_prefill_ane_split(on: bool) {
    ANE_SPLIT_OVERLAP.store(on, Ordering::Relaxed);
}

/// See [`ANE_SPLIT_OVERLAP`].
pub fn prefill_ane_split() -> bool {
    ANE_SPLIT_OVERLAP.load(Ordering::Relaxed)
}

/// Plan 550: the IOSurface **zero-copy** IO mode for the split executor —
/// GPU staging kernels over Metal texture views of the ANE's own io
/// surfaces, replacing the 231-749 ms/op host readback (Bench 776's
/// isolated 60-75% wall). Requires the split flag; when the ZC context
/// can't init (non-Metal backend, MSL compile failure) the seam falls back
/// to the measured host-IO split. Default off until the Bench 777 GOAT.
static ANE_ZERO_COPY: AtomicBool = AtomicBool::new(false);

/// See [`ANE_ZERO_COPY`].
pub fn set_prefill_ane_zero_copy(on: bool) {
    ANE_ZERO_COPY.store(on, Ordering::Relaxed);
}

/// See [`ANE_ZERO_COPY`].
pub fn prefill_ane_zero_copy() -> bool {
    ANE_ZERO_COPY.load(Ordering::Relaxed)
}

/// Issue 769 T10 (the second-queue variant, Bench 855 P1 verdict): stage the
/// ANE pack on a DEDICATED `MTLCommandQueue` instead of the shared CubeCL
/// one, fenced against the input producer with an `MTLEvent`.
///
/// P1 measured the pack's `wait_until_completed()` at 223.73 ms/op = 99.771%
/// backlog + 0.229% pack execution: the pack rides the shared queue, so
/// waiting on it drains everything already enqueued ahead of it. A dedicated
/// queue does NOT remove the producer dependency (the producer is itself the
/// tail of that backlog) - what it removes, by construction, is every unit
/// that landed on the shared queue AFTER the fence was committed, i.e. the
/// caller's GPU complement, which `begin_split_overlapped_zc` has no
/// handshake against and which therefore races the pack onto the FIFO today.
///
/// Requires the zero-copy flag. Default off until its own GOAT: promotion
/// needs the same-session two-arm re-baseline (Issue 769 ledger item (a)).
/// Programmatic setter -> `RIIR_ANE_ZC_SECOND_QUEUE` env -> off.
static ANE_ZC_SECOND_QUEUE: AtomicBool = AtomicBool::new(false);
/// The env half, read at most once per process (the
/// `prefill_ane_max_bytes_effective` pattern).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
static ANE_ZC_SECOND_QUEUE_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// See [`ANE_ZC_SECOND_QUEUE`].
pub fn set_prefill_ane_zc_second_queue(on: bool) {
    ANE_ZC_SECOND_QUEUE.store(on, Ordering::Relaxed);
}

/// See [`ANE_ZC_SECOND_QUEUE`]. Programmatic OR env - either arms it.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn prefill_ane_zc_second_queue() -> bool {
    if ANE_ZC_SECOND_QUEUE.load(Ordering::Relaxed) { true } else { *ANE_ZC_SECOND_QUEUE_ENV.get_or_init(|| {
            std::env::var("RIIR_ANE_ZC_SECOND_QUEUE").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        }) }
}

/// Issue 769 T10 P4 — the **attribution control** for the second-queue variant.
/// The `MTLEvent` producer fence needs `client.flush()` to be correct (a fence
/// signalled ahead of an unsubmitted producer orders nothing), so the
/// second-queue arm gets a flush the plain zero-copy arm does not, and their
/// delta conflates the queue with the flush. This toggle arms the flush
/// ALONE, which separates them in one process. Default off; implied by
/// [`prefill_ane_zc_second_queue`].
static ANE_ZC_PRODUCER_FLUSH: AtomicBool = AtomicBool::new(false);
/// The env half, read at most once per process.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
static ANE_ZC_PRODUCER_FLUSH_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// See [`ANE_ZC_PRODUCER_FLUSH`].
pub fn set_prefill_ane_zc_producer_flush(on: bool) {
    ANE_ZC_PRODUCER_FLUSH.store(on, Ordering::Relaxed);
}

/// See [`ANE_ZC_PRODUCER_FLUSH`]. Programmatic OR env.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn prefill_ane_zc_producer_flush() -> bool {
    if ANE_ZC_PRODUCER_FLUSH.load(Ordering::Relaxed) { true } else { *ANE_ZC_PRODUCER_FLUSH_ENV.get_or_init(|| {
            std::env::var("RIIR_ANE_ZC_PRODUCER_FLUSH").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        }) }
}

/// Plan 549: the `down_proj` op toggle. Read at REGISTRATION time (set
/// before `TernaryDeltanetGpuForward::new()` — down programs compile only
/// when this is on and the byte budget admits them) AND at dispatch time.
/// Default off.
static ANE_DOWN: AtomicBool = AtomicBool::new(false);

/// See [`ANE_DOWN`].
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn set_prefill_ane_down(on: bool) {
    ANE_DOWN.store(on, Ordering::Relaxed);
}

/// See [`ANE_DOWN`].
pub fn prefill_ane_down() -> bool {
    ANE_DOWN.load(Ordering::Relaxed)
}

/// Plan 549: programmatic override of the ANE bank byte ceiling (the
/// `RIIR_ANE_MAX_BYTES` env var is the deployment surface; this is the
/// harness surface). Read once at `TernaryDeltanetGpuForward::new()` — set
/// BEFORE construction. `None` = no override (env → default resolution).
static ANE_MAX_BYTES_OVERRIDE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// See [`ANE_MAX_BYTES_OVERRIDE`]. `0` clears the override.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn set_prefill_ane_max_bytes(bytes: u64) {
    ANE_MAX_BYTES_OVERRIDE.store(bytes, Ordering::Relaxed);
}

/// The effective bank ceiling: programmatic override → env → default.
pub fn prefill_ane_max_bytes_effective() -> u64 {
    let ov = ANE_MAX_BYTES_OVERRIDE.load(Ordering::Relaxed);
    if ov > 0 {
        return ov;
    }
    std::env::var("RIIR_ANE_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(bank_budget_default())
}

/// Plan 549 (measured, Bench 776): registering `DownProj` for ALL 48 GDN
/// layers (≈ +8.5 GB / +48 program instances on top of the 96-program base
/// bank) drives the ANE driver into `0x50004 Program load failure` — and the
/// cascade KILLS the base bank's subsequent loads too (layer 57's
/// InProjConcat failed after the down registrations exhausted the driver).
/// Down therefore registers for at most the first N GDN layers; the rest
/// fail-open per layer. Programmatic override → `RIIR_ANE_DOWN_LAYERS` env →
/// 0 (none).
static ANE_DOWN_LAYERS_OVERRIDE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX);

/// See [`ANE_DOWN_LAYERS_OVERRIDE`]. `u64::MAX` restores env resolution.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn set_prefill_ane_down_layers(n: u64) {
    ANE_DOWN_LAYERS_OVERRIDE.store(n, Ordering::Relaxed);
}

/// The effective down-registration cap (0 = register none).
pub fn prefill_ane_down_layers_effective() -> u64 {
    prefill_ane_down_layers_explicit().unwrap_or(0)
}

/// Issue 886 T3: `Some(cap)` ONLY when an explicit claim exists (the
/// programmatic override or the `RIIR_ANE_DOWN_LAYERS` env was set) — the
/// deliberate-claim path that keeps Plan 549's static semantics. `None`
/// when unset, in which case the down lane (when its runtime toggle is on)
/// runs the measured LADDER instead of a hand-tuned layer count.
pub fn prefill_ane_down_layers_explicit() -> Option<u64> {
    let ov = ANE_DOWN_LAYERS_OVERRIDE.load(Ordering::Relaxed);
    if ov != u64::MAX {
        return Some(ov);
    }
    std::env::var("RIIR_ANE_DOWN_LAYERS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
}

/// Issue 886 T3: the ladder's cumulative-byte cap for down registrations
/// (oMLX's env-forced initial cap; `_ANE_BANK_RETRY_MAX_BYTES` descent
/// floor analogue at 1 GiB). Resolution: programmatic override →
/// `RIIR_ANE_DOWN_LADDER_MAX_BYTES` env → 0 (no extra cap — the two-sided
/// budget already bounds total bank bytes; the cap is the deliberate
/// operator claim that narrows the down lane alone).
static ANE_DOWN_LADDER_MAX_BYTES_OVERRIDE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX);

/// Issue 886 T5 (the device-gate harness seam): programmatic override of
/// the down ladder's cumulative-byte cap — the setter half of
/// [`prefill_ane_down_ladder_max_bytes_effective`], mirroring
/// [`set_prefill_ane_max_bytes`]. Read at
/// `TernaryDeltanetGpuForward::new()` — set BEFORE construction. `u64::MAX`
/// restores env resolution (and is itself equivalent to no cap: no
/// cumulative down sum ever reaches it).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn set_prefill_ane_down_ladder_max_bytes(bytes: u64) {
    ANE_DOWN_LADDER_MAX_BYTES_OVERRIDE.store(bytes, Ordering::Relaxed);
}

/// See [`ANE_DOWN_LADDER_MAX_BYTES_OVERRIDE`].
pub fn prefill_ane_down_ladder_max_bytes_effective() -> u64 {
    let ov = ANE_DOWN_LADDER_MAX_BYTES_OVERRIDE.load(Ordering::Relaxed);
    if ov != u64::MAX {
        return ov;
    }
    std::env::var("RIIR_ANE_DOWN_LADDER_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

/// Issue 886 T3 — the measured down-registration descent, replacing the
/// hand-tuned `RIIR_ANE_DOWN_LAYERS` guess. A CALLER-SIDE policy over the
/// existing `try_register_splits` contract (the bank is untouched):
/// every rung registers WHOLE ops only — the ladder NEVER slices a weight
/// matrix, which is exactly what makes the T4 determinism guard hold by
/// construction (a rung's numerics are bit-identical to a non-ladder
/// registration of the same op: same requant, same compile — the ladder
/// only decides WHICH ops register, never HOW an op computes).
///
/// Descent semantics (oMLX's halving rung, adapted to whole-op rungs):
/// start ambitious (all GDN layers, or the explicit cap when set — an
/// explicit cap is a deliberate claim and keeps the Plan 549 semantics);
/// on each CONFIRMED refusal (the retry after a settle+re-measure also
/// failed) halve the remaining attempt target — a geometric backoff that
/// terminates in O(log n) refusals and stops grinding 0.7–1.2 s compile
/// attempts against a wall the box has already announced. Per-layer
/// fail-open is unchanged: a refused layer's slot stays empty and its
/// seam call site runs the GPU path.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub struct DownLadder {
    attempts_remaining: usize,
    /// An explicit `RIIR_ANE_DOWN_LAYERS`/override claim: static attempts,
    /// NO descent (Plan 549 semantics, exactly as today). The measured
    /// gate still applies either way — jetsam protection is not waivable
    /// by an env cap (that is this issue's whole point).
    explicit: bool,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl DownLadder {
    /// `explicit_cap` = the resolved `prefill_ane_down_layers_effective()`
    /// when an override/env was set (`Some`) — the deliberate-claim path
    /// with NO descent (Plan 549 back-compat). Otherwise the ladder starts
    /// ambitious: every GDN layer.
    pub fn new(explicit_cap: Option<u64>, total_gdn_layers: usize) -> Self {
        Self {
            attempts_remaining: match explicit_cap {
                Some(cap) => cap as usize,
                None => total_gdn_layers,
            },
            explicit: explicit_cap.is_some(),
        }
    }

    /// Whether the ladder still wants a down attempt for the next layer.
    pub fn wants_attempt(&self) -> bool {
        self.attempts_remaining > 0
    }

    /// Record a consumed attempt (the caller tries the registration).
    pub fn on_attempt(&mut self) {
        self.attempts_remaining = self.attempts_remaining.saturating_sub(1);
    }

    /// A CONFIRMED refusal (settle + re-measure retry also failed): under
    /// the LADDER, halve the remaining ambition (geometric backoff — stops
    /// grinding 0.7–1.2 s compile attempts against a wall the box announced);
    /// under an EXPLICIT cap, keep the operator's number (Plan 549's static
    /// semantics — the per-rung measured gate above still refuses before
    /// jetsam). Returns `false` when the ladder is done.
    pub fn on_refusal_confirmed(&mut self) -> bool {
        if !self.explicit {
            // Floor: one whole op minimum — oMLX's 1 GiB bank-floor
            // analogue is ONE op here (~178 MB per Bonsai down program);
            // never a fraction of an op.
            self.attempts_remaining /= 2;
        }
        self.attempts_remaining > 0
    }

    /// Remaining attempts (observable for the load-time report).
    pub fn attempts_remaining(&self) -> usize {
        self.attempts_remaining
    }
}

/// Issue 886 T5: the construction-time down-lane outcome, exposed for the
/// device gates (and any operator diagnostic). The ladder only decides
/// WHICH whole ops register, never HOW an op computes — so a rung's
/// numerics are bit-identical to a non-ladder registration of the same op
/// (the G1 pin), and this report is what makes that claim checkable: the
/// harness asserts the landed set AND per-layer accounting
/// (`attempted == landed + refused`) from it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AneDownReport {
    /// GDN layers the ladder gate ADMITTED an attempt for (in registration
    /// order). A layer here with no landed/refused resolution means the
    /// registration loop broke early (ctx poisoned) — a finding, never a
    /// silent gap.
    pub attempted: Vec<usize>,
    /// GDN layers whose down program registered.
    pub landed: Vec<usize>,
    /// GDN layers attempted and refused (slot left empty, fail-open).
    pub refused: Vec<usize>,
    /// Cumulative registered down bytes (`down_spent` at finalize).
    pub bytes_spent: u64,
}

/// The per-attempt byte admissibility arithmetic (pure — the ladder's
/// gate): a down op registers only when every cap admits it. `gate` is the
/// settle+measured machine-side headroom term from the last rung (see
/// `headroom::machine_side_ceiling`); `budget_ceiling` is the two-sided
/// effective ceiling the whole bank already enforces; `ladder_byte_cap`
/// (0 = uncapped) bounds CUMULATIVE down bytes (`down_spent`), the
/// operator's deliberate narrowing of the down lane alone.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn down_admits(
    budget_ceiling: u64,
    gate: Option<u64>,
    ladder_byte_cap: u64,
    down_spent: u64,
    bank_bytes: u64,
    down_bytes: u64,
) -> bool {
    // Overflow at any term REFUSES (the sum exceeds every representable
    // ceiling; saturation would collapse u64::MAX-boundary cases into a
    // false admit — pinned by the extremes row of the test table).
    let Some(after) = bank_bytes.checked_add(down_bytes) else {
        return false;
    };
    if after > budget_ceiling {
        return false;
    }
    if let Some(gate) = gate
        && after > gate
    {
        return false;
    }
    if ladder_byte_cap > 0
        && down_spent
            .checked_add(down_bytes)
            .is_none_or(|spent| spent > ladder_byte_cap)
    {
        return false;
    }
    true
}

/// See the `PREFILL_USE_ANE` static's docs above.
pub fn set_prefill_use_ane(on: bool) {
    PREFILL_USE_ANE.store(on, Ordering::Relaxed);
}

/// See the `PREFILL_USE_ANE` static's docs above.
pub fn prefill_use_ane() -> bool {
    PREFILL_USE_ANE.load(Ordering::Relaxed)
}

/// Successful full-width ANE dispatches (Issue 726 observability). The T4/T5
/// A/B harness asserts the hybrid arm's delta > 0 and the GPU arm's delta
/// == 0 — a silently fail-opened hybrid arm (e.g. the runtime flag left off
/// after construction) FAILS the gate instead of vacuously passing with
/// cosine 1.000 and ratio 1.0.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
static ANE_DISPATCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// See the `ANE_DISPATCH_COUNT` static's docs above.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn ane_dispatch_count() -> u64 {
    ANE_DISPATCH_COUNT.load(Ordering::Relaxed)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn note_ane_dispatch() {
    ANE_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// The ANE prefill context state machine (macOS + aarch64; other hosts get
/// a permanent `is_ready() == false` ctx). `Ready` is the ONLY state under
/// which the prefill seam may dispatch to the ANE — every other state
/// fail-opens to the GPU batched GEMM.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone)]
pub enum AnePrefillState {
    /// Init/register attempted + refused for the recorded reason (runtime
    /// flag off at load, program-compile failure, memory budget exceeded,
    /// incomplete bank). Fail-open.
    Unavailable(std::sync::Arc<str>),
    /// The Form C program bank compiled + complete for every GDN layer.
    Ready(Box<bank::AneProgramBank>),
}

/// The runtime context consulted by the eligibility gate. Owned by
/// `TernaryDeltanetGpuForward`.
#[derive(Debug, Clone)]
pub struct AnePrefillCtx {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    state: AnePrefillState,
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    _non_macos: (),
}

impl AnePrefillCtx {
    /// Begin bank construction at model load. Compiling the ~96-program
    /// bank costs ≈ 90 s at real dims (0.69-1.20 s/program, P9; P8's
    /// 0.028 s was toy dims) — the ctx only attempts it when the runtime
    /// flag was set BEFORE construction (`set_prefill_use_ane(true)`
    /// first); otherwise `Unavailable` immediately at zero cost. Register
    /// layers via [`Self::register_layer`], then [`Self::finalize`].
    pub fn init(config: &AnePrefillConfig) -> Self {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            if !prefill_use_ane() {
                return Self {
                    state: AnePrefillState::Unavailable(
                        "runtime flag off at model load — set_prefill_use_ane(true)                          before construction"
                            .into(),
                    ),
                };
            }
            // Issue 887 T1(a): a fractional channel_fraction is REFUSED at
            // zero compile cost, not silently ignored. The partial-coverage
            // executor (suffix-channel GPU complement) is Phase C and does
            // not exist; before this refusal the knob compiled a full bank
            // and dispatched full width — an A/B harness reading f=0.5 back
            // was measuring f=1.0 (the silent-wrong-attribution exposure
            // this issue was filed over). Fail-open loud, not silent-inert.
            if config.channel_fraction < 1.0 {
                return Self {
                    state: AnePrefillState::Unavailable(
                        format!(
                            "channel_fraction={} < 1.0 at model load — the partial-coverage \
                             executor (suffix-channel GPU complement) is Issue 887 Phase C \
                             and not wired; refusing at zero compile cost. Prefill stays on \
                             the GPU; set channel_fraction = 1.0 for the ANE path",
                            config.channel_fraction
                        )
                        .into(),
                    ),
                };
            }
            Self {
                state: AnePrefillState::Ready(Box::new(bank::AneProgramBank::new(
                    config.max_layers_hint,
                    config.block_tokens,
                    config.max_ane_bytes,
                ))),
            }
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = config;
            Self { _non_macos: () }
        }
    }

    /// Register one GDN layer's two fused programs from the SPLIT CPU
    /// weights (row-concat requant — see `bank`). Attention layers are
    /// skipped. Any error poisons the ctx (finalize reports Unavailable —
    /// fail-open; all-or-nothing keeps the gate honest). The Plan 549
    /// `down` weight is TRY-registered: compile/budget failure leaves the
    /// slot empty WITHOUT poisoning (per-op fail-open) — only the caller's
    /// own contract errors (double registration) poison.
    ///
    /// Issue 887 T1(a): in split-overlap mode the in_proj/gate_up programs
    /// register PREFIX-ONLY (the first split — the segment the dispatch
    /// seams actually consume; the remaining rows are the GPU complement's).
    /// The split toggle is read ONCE here, at registration (construction)
    /// time; flipping it later does not re-register — the seams' compiled-
    /// width guard fail-opens cleanly instead (see
    /// [`warn_width_mismatch_once`]).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn register_layer(
        &mut self,
        layer_idx: usize,
        is_gdn: bool,
        in_splits: &[&katgpt_core::TernaryGroupWeights],
        gate_up_splits: &[&katgpt_core::TernaryGroupWeights],
        down: Option<&katgpt_core::TernaryGroupWeights>,
    ) {
        let AnePrefillState::Ready(bank) = &mut self.state else {
            return; // already unavailable
        };
        if !is_gdn {
            return;
        }
        let split_mode = prefill_ane_split();
        let in_splits = registration_splits(split_mode, AnePrefillOp::InProjConcat, in_splits);
        let gate_up_splits =
            registration_splits(split_mode, AnePrefillOp::GateUpProj, gate_up_splits);
        let r1 = bank.register_splits(layer_idx, AnePrefillOp::InProjConcat, &in_splits);
        let r2 = bank.register_splits(layer_idx, AnePrefillOp::GateUpProj, &gate_up_splits);
        if let Err(e) = r1.and(r2) {
            self.state = AnePrefillState::Unavailable(e.into());
            return;
        }
        if let Some(w) = down
            && let Err(e) = bank.try_register_splits(layer_idx, AnePrefillOp::DownProj, &[w])
        {
            self.state = AnePrefillState::Unavailable(e.into());
        }
    }

    /// Issue 886 T3: whether the last `register_layer` call actually landed
    /// a `DownProj` program — `try_register_splits` fails OPEN (returns Ok
    /// with the slot empty), so the ladder's refusal detection observes the
    /// SLOT, not the return value.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn down_registered(&self, layer_idx: usize) -> bool {
        match &self.state {
            AnePrefillState::Ready(bank) => {
                bank.kernel(layer_idx, AnePrefillOp::DownProj).is_some()
            }
            _ => false,
        }
    }

    /// Issue 886 T3: whole-bank byte total (the ladder's arithmetic gate
    /// input). `None` once unavailable.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn bank_bytes_total(&self) -> Option<u64> {
        match &self.state {
            AnePrefillState::Ready(bank) => Some(bank.bytes_total()),
            _ => None,
        }
    }

    /// Issue 886 T3: the ladder's RETRY path — down-only registration for
    /// the layer whose attempt was refused (a full `register_layer` retry
    /// would double-register in/gate_up and poison the ctx). True iff the
    /// program landed.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn try_register_down(
        &mut self,
        layer_idx: usize,
        w: &katgpt_core::TernaryGroupWeights,
    ) -> bool {
        let AnePrefillState::Ready(bank) = &mut self.state else {
            return false;
        };
        if bank.try_register_splits(layer_idx, AnePrefillOp::DownProj, &[w]).is_err() {
            return false;
        }
        bank.kernel(layer_idx, AnePrefillOp::DownProj).is_some()
    }

    /// Seal the bank: stays `Ready` only if every GDN layer registered
    /// both ops (a partial bank must NOT fire — silent layer-by-layer path
    /// mixing is the Issue 066 class).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn finalize(mut self, gdn_layers: &[usize]) -> Self {
        if let AnePrefillState::Ready(bank) = &self.state
            && !bank.covers(gdn_layers)
        {
            self.state = AnePrefillState::Unavailable(
                "program bank incomplete — a GDN layer is missing an op".into(),
            );
        }
        self
    }

    /// The only state under which the seam may dispatch.
    pub fn is_ready(&self) -> bool {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            matches!(self.state, AnePrefillState::Ready(_))
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// The compiled kernel for (layer, op) — `None` unless Ready.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn kernel(
        &self,
        layer_idx: usize,
        op: AnePrefillOp,
    ) -> Option<std::sync::Arc<bridge::BridgeKernel>> {
        let AnePrefillState::Ready(bank) = &self.state else {
            return None;
        };
        bank.kernel(layer_idx, op).cloned()
    }

    /// T6 budget evidence: `(compiled_bytes, ceiling)` when the bank is
    /// Ready. Enforcement is at registration (`bank::register_splits`);
    /// this is the report surface for load-time logging.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn bank_bytes(&self) -> Option<(u64, u64)> {
        let AnePrefillState::Ready(bank) = &self.state else {
            return None;
        };
        Some((bank.bytes_total(), bank.max_bytes()))
    }

    /// Diagnostic accessor (T6 progress reporting; GOAT logging).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn state(&self) -> &AnePrefillState {
        &self.state
    }

    /// Non-macOS diagnostic reason.
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    pub fn unavailable_reason(&self) -> &'static str {
        "ane_prefill: non-Apple-Silicon host"
    }
}

/// The Issue 726 eligibility gate — pure, allocation-free, unit-tested.
///
/// Contract: `p ≥ block && layer_is_gdn && op ∈ {InProjConcat, GateUpProj}
/// && ctx_ready && runtime_on → plan`. Every miss returns `None` and the
/// caller fail-opens to the existing GPU dispatch, unchanged. The op-set
/// constraint is enforced by [`AnePrefillOp`]'s type (no ineligible variant
/// exists); the parameter is retained so call sites read the contract.
#[must_use]
pub fn ane_prefill_eligible(
    p: usize,
    is_gdn: bool,
    _op: AnePrefillOp,
    ctx_ready: bool,
    runtime_on: bool,
    config: &AnePrefillConfig,
) -> Option<AnePrefillPlan> {
    // Flag → ctx → layer kind → shape/config. Cheap fails first.
    if !runtime_on {
        return None;
    }
    if !ctx_ready {
        return None;
    }
    if !is_gdn {
        return None;
    }
    // P7: the ANE conv requires spatial width W ≥ 32 — narrower W compiles
    // but fails at eval. A misconfigured block is a fail-open, not a panic.
    if config.block_tokens < ANE_PREFILL_MIN_SPATIAL_W {
        return None;
    }
    if !(config.channel_fraction > 0.0 && config.channel_fraction <= 1.0) {
        return None;
    }
    // Exact fixed-shape blocks only: the ANE takes full blocks; the
    // remainder is the GPU tail (no synthetic padding — omlx #4).
    if p < config.block_tokens {
        return None;
    }
    let ane_blocks = p / config.block_tokens;
    Some(AnePrefillPlan {
        ane_blocks,
        tail: p % config.block_tokens,
        channel_fraction: config.channel_fraction,
        ane_tokens: ane_blocks * config.block_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: AnePrefillConfig = AnePrefillConfig {
        channel_fraction: 0.5,
        block_tokens: ANE_PREFILL_BLOCK_TOKENS,
        max_layers_hint: 0,
        max_ane_bytes: u64::MAX,
    };

    fn gate(p: usize, is_gdn: bool, op: AnePrefillOp) -> Option<AnePrefillPlan> {
        ane_prefill_eligible(p, is_gdn, op, true, true, &CFG)
    }

    #[test]
    fn gate_requires_at_least_one_full_block() {
        // 93-154-token clippy prompts + decode tails stay GPU BY DESIGN
        // (issue §"Honest scope").
        assert!(gate(1, true, AnePrefillOp::InProjConcat).is_none());
        assert!(gate(154, true, AnePrefillOp::GateUpProj).is_none());
        assert!(gate(2047, true, AnePrefillOp::InProjConcat).is_none());
        assert!(gate(2048, true, AnePrefillOp::InProjConcat).is_some());
    }

    #[test]
    fn gate_plan_arithmetic() {
        let plan = gate(4096, true, AnePrefillOp::InProjConcat).unwrap();
        assert_eq!(plan.ane_blocks, 2);
        assert_eq!(plan.tail, 0);
        assert_eq!(plan.ane_tokens, 4096);
        assert!(!plan.full_coverage()); // f = 0.5 → suffix remains

        let plan = gate(6143, true, AnePrefillOp::GateUpProj).unwrap();
        assert_eq!(plan.ane_blocks, 2);
        assert_eq!(plan.tail, 2047);
        assert_eq!(plan.ane_tokens, 4096);

        // 32K (the omlx #2781 top end): 16 full blocks, no tail.
        let plan = gate(32768, true, AnePrefillOp::InProjConcat).unwrap();
        assert_eq!(plan.ane_blocks, 16);
        assert_eq!(plan.tail, 0);
        assert_eq!(plan.ane_tokens, 32768);
    }

    #[test]
    fn gate_full_coverage_only_at_f1_tail0() {
        let full = AnePrefillConfig {
            channel_fraction: 1.0,
            block_tokens: 2048,
            max_layers_hint: 0,
            max_ane_bytes: u64::MAX,
        };
        let plan = ane_prefill_eligible(4096, true, AnePrefillOp::InProjConcat, true, true, &full)
            .unwrap();
        assert!(plan.full_coverage());
        // Partial fraction or tail → not full coverage.
        assert!(
            !gate(4096, true, AnePrefillOp::InProjConcat)
                .unwrap()
                .full_coverage()
        );
        assert!(
            !gate(6143, true, AnePrefillOp::InProjConcat)
                .unwrap()
                .full_coverage()
        );
    }

    #[test]
    fn gate_rejects_attention_layers() {
        // The contract scopes acceleration to the GDN family (48 of 64
        // Bonsai layers); attention-layer projections stay GPU.
        assert!(gate(4096, false, AnePrefillOp::InProjConcat).is_none());
        assert!(gate(4096, false, AnePrefillOp::GateUpProj).is_none());
    }

    #[test]
    fn gate_fail_open_when_ctx_not_ready() {
        let plan = ane_prefill_eligible(4096, true, AnePrefillOp::InProjConcat, false, true, &CFG);
        assert!(plan.is_none());
    }

    #[test]
    fn gate_fail_open_when_runtime_flag_off() {
        let plan = ane_prefill_eligible(4096, true, AnePrefillOp::InProjConcat, true, false, &CFG);
        assert!(plan.is_none());
    }

    #[test]
    fn gate_fail_open_on_bad_block_config() {
        // P7: W < 32 compiles but fails at eval — reject at the gate.
        let narrow = AnePrefillConfig {
            channel_fraction: 0.5,
            block_tokens: 31,
            max_layers_hint: 0,
            max_ane_bytes: u64::MAX,
        };
        assert!(
            ane_prefill_eligible(4096, true, AnePrefillOp::InProjConcat, true, true, &narrow)
                .is_none()
        );
    }

    #[test]
    fn gate_fail_open_on_bad_fraction() {
        for bad in [0.0f32, -0.5, 1.5] {
            let cfg = AnePrefillConfig {
                channel_fraction: bad,
                block_tokens: 2048,
                max_layers_hint: 0,
                max_ane_bytes: u64::MAX,
            };
            assert!(
                ane_prefill_eligible(4096, true, AnePrefillOp::InProjConcat, true, true, &cfg)
                    .is_none(),
                "fraction {bad} must fail open"
            );
        }
    }

    #[test]
    fn ctx_refuses_when_flag_off_at_load() {
        // Fail-open with ZERO load-time cost when the runtime flag was not
        // set before construction (the default state of the world).
        let before = prefill_use_ane();
        set_prefill_use_ane(false);
        let ctx = AnePrefillCtx::init(&CFG);
        assert!(!ctx.is_ready());
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert!(matches!(ctx.state(), AnePrefillState::Unavailable(_)));
        set_prefill_use_ane(before);
    }

    #[test]
    fn ctx_kernel_lookup_requires_ready() {
        // On a flag-off ctx (never Ready), the kernel accessor is None for
        // every (layer, op) — the seam cannot dispatch.
        let before = prefill_use_ane();
        set_prefill_use_ane(false);
        // Plan 610 S4b (the S4a-recorded warning class, fixed at the new
        // home): the ctx is only READ on macOS+aarch64 — gate the binding
        // with its use instead of warning on every other target.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let ctx = AnePrefillCtx::init(&CFG);
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert!(ctx.kernel(0, AnePrefillOp::InProjConcat).is_none());
        set_prefill_use_ane(before);
    }

    #[test]
    fn runtime_flag_defaults_off_and_round_trips() {
        // Read-modify-restore: the only test touching the global (the gate
        // tests pass the flag explicitly, so no cross-test coupling).
        let before = prefill_use_ane();
        set_prefill_use_ane(true);
        assert!(prefill_use_ane());
        set_prefill_use_ane(before);
    }

    #[test]
    fn ctx_bank_bytes_reports_only_when_ready() {
        // T6 budget-evidence accessor: Some((0, ceiling)) on a fresh Ready
        // bank (nothing compiled yet), None on every Unavailable ctx.
        // (Issue 887: uses an f=1.0 config — init REFUSES f<1 at zero cost
        // now, so an f=0.5 cfg can no longer reach Ready.)
        let full = AnePrefillConfig {
            channel_fraction: 1.0,
            block_tokens: ANE_PREFILL_BLOCK_TOKENS,
            max_layers_hint: 0,
            max_ane_bytes: u64::MAX,
        };
        let before = prefill_use_ane();
        set_prefill_use_ane(true);
        let _ctx = AnePrefillCtx::init(&full);
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(_ctx.bank_bytes(), Some((0, full.max_ane_bytes)));
        set_prefill_use_ane(false);
        let _off = AnePrefillCtx::init(&full);
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert!(_off.bank_bytes().is_none());
        set_prefill_use_ane(before);
    }

    /// Issue 887 T1(a): a fractional channel_fraction is refused at init —
    /// zero compile cost, loud reason, prefill stays GPU. Before this, f<1
    /// compiled a FULL bank and dispatched full width: the knob was inert
    /// and an A/B harness at f=0.5 silently measured f=1.0.
    #[test]
    fn ctx_refuses_fraction_below_one_at_load() {
        let before = prefill_use_ane();
        set_prefill_use_ane(true);
        let ctx = AnePrefillCtx::init(&CFG); // CFG carries f = 0.5
        assert!(!ctx.is_ready());
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert!(
            matches!(ctx.state(), AnePrefillState::Unavailable(reason) if reason.contains("887")),
            "the refusal must name the Phase C lane (Issue 887), got {:?}",
            ctx.state()
        );
        set_prefill_use_ane(before);
    }

    /// Issue 887 T1(a), registration policy: split-overlap mode registers
    /// the PREFIX split only (the segment the seams consume); off-grid
    /// prefixes, non-split mode and DownProj keep the full fused set.
    /// Headless twin of the bank-level `bank_byte_accounting_is_slice_faithful`
    /// pin — registration proper is compile-bound to the ANE bridge.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn registration_splits_policy_prefix_only_in_split_mode_on_grid() {
        use katgpt_core::TernaryGroupWeights;
        let qkv = TernaryGroupWeights::new(128, 32); // on-grid prefix
        let z = TernaryGroupWeights::new(128, 32);
        let a = TernaryGroupWeights::new(64, 32);
        let b = TernaryGroupWeights::new(64, 32);
        let in_splits = [&qkv, &z, &a, &b];
        // Split mode + on-grid → first split only (the true channel split:
        // bytes/compile/eval track the ANE's real share).
        let pol = registration_splits(true, AnePrefillOp::InProjConcat, &in_splits);
        assert_eq!(pol.len(), 1);
        assert_eq!(pol[0].rows, 128);
        // Non-split mode → full fused registration (the f=1 GOAT anchor —
        // bit-identical to the pre-887 bank).
        assert_eq!(
            registration_splits(false, AnePrefillOp::InProjConcat, &in_splits).len(),
            4
        );
        // DownProj never slices — no GPU complement exists (Phase C).
        assert_eq!(
            registration_splits(true, AnePrefillOp::DownProj, &in_splits).len(),
            4
        );
        // Off-grid prefix → full registration (the T2 grid is unverified on
        // our compiler; an unverified width must not reach eval).
        let odd = TernaryGroupWeights::new(100, 32);
        let odd_splits = [&odd, &z];
        assert_eq!(
            registration_splits(true, AnePrefillOp::GateUpProj, &odd_splits).len(),
            2
        );
        // On-grid gate_up prefix slices too (gate IS the fused row prefix).
        let gate = TernaryGroupWeights::new(192, 32);
        let up = TernaryGroupWeights::new(192, 32);
        let gu = [&gate, &up];
        assert_eq!(registration_splits(true, AnePrefillOp::GateUpProj, &gu).len(), 1);
        // Empty stays empty (caller-contract error upstream, never inflated).
        let empty: [&TernaryGroupWeights; 0] = [];
        assert!(registration_splits(true, AnePrefillOp::InProjConcat, &empty).is_empty());
    }

    // ---- Issue 886: two-sided budget + the measured down ladder ----

    /// Issue 886 T1: the default ceiling composition is min(model, machine)
    /// — asserted against INJECTED measurements (the live read has no
    /// box-independent golden; `headroom::two_sided_ceiling` tests hold the
    /// composition, this test pins the default fn's shape on this macOS
    /// target: never above the model term, and a positive number).
    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn bank_budget_default_is_two_sided_and_bounded_by_model() {
        let resolved = bank_budget_default();
        assert!(
            resolved <= bank::DEFAULT_MAX_ANE_BYTES,
            "default must never exceed the model-side term: {resolved}"
        );
        assert!(resolved > 0);
    }

    /// Issue 886 T3: the ladder state machine — ambitious start, halving on
    /// confirmed refusal, termination, and the deliberate-claim path.
    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn ladder_halves_on_confirmed_refusal_and_terminates() {
        let mut ladder = DownLadder::new(None, 48);
        assert!(ladder.wants_attempt());
        assert_eq!(ladder.attempts_remaining(), 48);
        ladder.on_attempt();
        assert_eq!(ladder.attempts_remaining(), 47);
        // First confirmed refusal: 47 -> 23.
        assert!(ladder.on_refusal_confirmed());
        assert_eq!(ladder.attempts_remaining(), 23);
        // Grind to zero: repeated halving of a small target lands at 0 and
        // reports done (the caller stops attempting — per-layer fail-open
        // continues to hold for everything already registered).
        let mut guard = 0;
        while ladder.on_refusal_confirmed() {
            guard += 1;
            assert!(guard < 20, "geometric descent must terminate");
        }
        assert_eq!(ladder.attempts_remaining(), 0);
        assert!(!ladder.wants_attempt());
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn ladder_explicit_cap_is_static_no_descent() {
        // Plan 549 back-compat: an explicit cap is a deliberate claim —
        // exactly that many attempts, and refusals do NOT halve (the
        // operator's number wins over the measured descent; the per-rung
        // measured gate still refuses before jetsam either way).
        let mut ladder = DownLadder::new(Some(4), 48);
        assert_eq!(ladder.attempts_remaining(), 4);
        ladder.on_attempt();
        ladder.on_attempt();
        assert_eq!(ladder.attempts_remaining(), 2);
        assert!(ladder.on_refusal_confirmed());
        assert_eq!(ladder.attempts_remaining(), 2, "explicit cap never halves");
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn ladder_zero_cap_means_no_attempts() {
        let ladder = DownLadder::new(Some(0), 48);
        assert!(!ladder.wants_attempt());
    }

    /// Issue 886 T3: the per-attempt admissibility arithmetic — every cap
    /// must admit: two-sided budget, measured gate, cumulative ladder cap.
    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn down_admits_requires_every_cap() {
        let ceiling = 1000u64;
        // Budget admits, gate admits, no ladder cap.
        assert!(down_admits(ceiling, Some(900), 0, 0, 500, 300));
        // Two-sided budget refuses: 500 + 300 + 300 > 1000.
        assert!(!down_admits(ceiling, Some(900), 0, 0, 500 + 300, 300));
        // Measured gate refuses though budget admits.
        assert!(!down_admits(ceiling, Some(700), 0, 0, 500, 300));
        // Gate measurement unavailable (None): the gate term abstains —
        // the two-sided ceiling already carries the conservative default.
        assert!(down_admits(ceiling, None, 0, 0, 500, 300));
        // Cumulative ladder cap refuses: spent + bytes > cap.
        assert!(!down_admits(ceiling, Some(900), 500, 400, 0, 300));
        assert!(down_admits(ceiling, Some(900), 500, 200, 0, 300));
        // Saturation safety: no panic at extremes.
        assert!(!down_admits(u64::MAX, None, 0, u64::MAX - 10, u64::MAX - 10, 30));
    }
}
