//! The riir forward's Metal backend (feature `laya-riir-metal`, macOS) —
//! the same op semantics as [`super::ops`], replayed as MSL compute
//! kernels (`.issues/005`, owner directive: the fair three-way compare is
//! all-Metal).
//!
//! Transcription provenance:
//! - **gelu erf is candle's own METAL kernel** (`candle-metal-kernels`
//!   `unary.metal`: the A&S 7.1.26 f32 erf + `x·(1+erf(x·√½))/2`) — copied
//!   verbatim, not re-derived: candle Metal is the chart column this lane
//!   is compared against, and it passes the G5 ≤ 1e-3 gate with exactly
//!   this kernel. The CPU lane keeps `libm::erff` (`.issues/003`) — each
//!   lane matches the candle posture it mirrors.
//! - **candle's flush shape, per PASS** — ops ENCODE into one pass-scoped
//!   command buffer and commit WITHOUT waiting; the wait happens only when
//!   the host genuinely reads a result ([`Backend::download_into`]: the
//!   scorer logits, the CLS row, the act logits — three syncs per forward).
//!   The v1 shape committed a fresh command buffer PER OP (~600–1400
//!   commits/forward — measured the lane's dominant overhead); the pass
//!   buffer plus a 1024-encode pipeline-flush cap keeps a long pass under
//!   Metal's per-buffer encoder ceiling without ever waiting mid-pass.
//!   A literal commit+wait per op measured 0.59 ms of round-trip per
//!   dispatch × ~1100 dispatches/forward — 2.5 s/forward, pure sync
//!   overhead. Device memory is the single writer between syncs — forward
//!   bodies never read op outputs host-side except through
//!   `download_into`.
//! - **generation-keyed chain cache** — activations flow device-side, so
//!   scratch slices are cached by `(ptr, len, generation)` with the
//!   generation bumped at every pass: host-authored buffers (rope tables,
//!   mask, `act_in`) rebuilt at recycled heap addresses can never hit a
//!   stale entry from an earlier epoch, and the write-first audit of the
//!   two forward bodies guarantees a hit's device copy is current. True
//!   weights (agent-owned `Vec`s: every `matmul_w` weight, LN scales,
//!   biases, the embedding table) live in a permanent cache keyed by
//!   `(ptr, len)` — stable for the agent's lifetime, uploaded once.
//! - **one batched simdgroup GEMM** covers all matmul shapes — including
//!   the all-heads attention batch — via explicit row/column strides plus
//!   a batch count with per-batch strides, in THREE tile geometries
//!   picked per call (32×64×64 narrow / 64×64×32 wide / 64×128×32 xwide;
//!   the geometry constants in this file document the pick),
//!   `simdgroup_multiply_accumulate` over 8×8 frags, threadgroup staging
//!   at padded odd strides (bank-conflict + overlap guards), and a
//!   guarded per-simdgroup store path for ragged edge tiles. The v1
//!   kernel was a naive 16×16 one-thread-per-element tile (the recorded
//!   honest baseline, ~3.5% of peak); this is the recorded optimization
//!   ladder climbed.
//!   The unsplit batch-1 dense calls (every encoder projection past the
//!   split rule, m ≥ 97) dispatch Apple's `MPSMatrixMultiplication`
//!   instead ([`mps`], reflex issue 020 T13, default ON,
//!   `LAYA_METAL_MPS=0` kill-switch): bit-identical to the narrow
//!   instance on every gated shape (`tests/metal_mps_gemm.rs`) and
//!   0.58–0.74× the whole forward at m 106–895.
//! - **attention is ONE fused dispatch per layer** — `flash_attn` consumes
//!   the packed qkv directly (split, rope, q-scale, scores, sliding window,
//!   softmax, value mix, head merge in-kernel) and materializes NO seq²
//!   scores parent; sliding-window layers walk only their windowed key
//!   slice, which is where the long-sequence win lives (window 64 vs seq
//!   317 ≈ 2.4× less attention FLOPs). The two-pass form normalizes
//!   without rescaling the accumulator. `LAYA_METAL_FLASH=0` falls back to
//!   the reference op sequence ([`Backend::attention_forward_default`]);
//!   the CPU lane keeps the identical per-head op order, so its numerics
//!   are bit-unchanged.
//!   Opt-in `LAYA_METAL_ROPE_HOIST=1` (reflex issue 020 T10 rung 2) adds
//!   the `attn_rope` pre-pass: the Q/K rope is derived ONCE per layer into
//!   a device scratch (the staging would otherwise re-derive it per
//!   query block × head × key tile), and `flash_attn`'s staging copies
//!   instead of rotating. Default-off: the in-kernel rope arm is the
//!   shipped behavior, bit-identical; promotion follows the probe.
//! - **softmax / LN are row-parallel** — one threadgroup (one simdgroup,
//!   32 lanes) per row with `simd_max`/`simd_sum` reductions; the v1
//!   kernels ran one thread per row (32× the GPU idle). Reduction order
//!   differs from the CPU lane's `candle_vec_sum` NEON order — exactly
//!   the drift budget the G5 gate holds (≤ 1e-3, the same room candle
//!   Metal passes within).
//!
//! Binding contract: every kernel declares `[[buffer(N)]]` /
//! `[[threadgroup(N)]]` attributes explicitly — device buffers at 0..,
//! then the 4-byte `constant` scalars, then the GEMM's three staging
//! buffers — and [`Metal::encode`] binds in exactly that order.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use metal::{
    Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoder, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};
use objc2::rc::autoreleasepool;

use super::super::{LayaError, Result};
use super::backend::{AttnScratch, Backend};

mod mps;

/// Shared storage with EXPLICIT tracked hazard tracking: the pipeline
/// flush commits a full command buffer mid-pass, and Metal only inserts
/// the cross-command-buffer memory barriers for TRACKED resources. On
/// macOS the DEFAULT is untracked (so merely dropping the flag changed
/// nothing — measured), and candle's `HazardTrackingModeUntracked` pairs
/// with their ONE long-lived command buffer, where intra-buffer encoder
/// ordering provides the visibility. With multiple committed buffers per
/// pass, untracked meant kernel N+1 read stale memory written by kernel N
/// — a timing-dependent race the G5 gate caught. The tracked cost is part
/// of the honest baseline.
const RESOURCE_OPTIONS: MTLResourceOptions =
    MTLResourceOptions::StorageModeShared.union(MTLResourceOptions::HazardTrackingModeTracked);

/// sgemm tile geometry — MUST mirror the MSL constants in the instances
/// below; the `metal_ops_smoke` ragged-shape arms exercise every edge path
/// this mirroring could get wrong. Shared staging law: a row stride must
/// EXCEED the tile's row width (33 over a 64-wide tile overlaps itself:
/// column 63 of row kk collides with column 30 of row kk+1) and stays odd
/// for banks — hence strides 65 (over 64-wide rows) and 33 (over 32-wide
/// rows) and 129 (over 128-wide rows).
///
/// Two instances picked per call (see `run_sgemm`):
/// - narrow `sgemm` (32×64×64) — the default: the whole many-wave regime
///   (grids past one wave), every short-k shape, and every badly
///   under-filled grid.
/// - xwide `sgemm_xwide` (64×128×32) — ONLY inside the measured single-wave
///   BAND: the grid must fill most of one wave but not spill past it
///   ([`WAVE_TG_FLOOR`] < ⌈m/64⌉·⌈n/128⌉·batch ≤ [`WAVE_TG_LIMIT`]) AND k
///   must run enough iterations to amortize the fat staging
///   (k ≥ [`XWAVE_K_MIN`]). One ~full wave of fat threadgroups (staging
///   intensity 42.7 MAC/staged element vs narrow's 21) beats several thin
///   waves of narrow tiles — but only while nothing queues behind it and
///   nothing under-fills it. BK stays 32: at BN 128 a BK-48 B tile is
///   [48][129] = 6192 floats and the pair outgrows 32 KB. Every n it
///   serves (1024, 3072, 5248) is an exact multiple of 128, so the bigger
///   BN pads nothing on the projection shapes.
/// - the middle `sgemm_wide` (64×64×32) instance is NO LONGER PICKED: the
///   2026-09-25 dispatch-sweep probe (60 hot-path cells, position-balanced,
///   3 rotated postures — riir-reflex Issue 020 T7) measured wide dominated
///   by xwide at identical m-tiling wherever xwide's band fits, and by
///   narrow everywhere else, in EVERY cell. The kernel stays compiled for
///   future retuning; the pick never routes to it. (Its own BK-48 history:
///   the largest k-chunk fitting 32 KB at 64×64 was tried and measured FLAT
///   — barriers are not the wide instance's binding constraint.)
///
/// Measured basis for the whole predicate (probe medians, pooled over
/// 3 rotated posture rounds × 4 shape populations — encoder projections at
/// m ∈ {231..512} and packed scale {1024..2048}, head MHA at batch = 16,
/// small-m m ∈ {1..188}):
/// - o/wo (n = 1024) at m 231–317: xwide's grid is 32–40 threadgroups =
///   one wave; −14…−18% vs the old wide pick. At m ≥ 370 the grid spills
///   past one wave and narrow wins (+9…+19% vs wide, +6…+19% vs xwide).
/// - qkv/wi (n ≥ 2048) at m ≤ 283 and across the packed scale: narrow
///   beats the old xwide pick by +2…+22% (xwide grids of 96–328 threadgroups
///   queue multiple waves; narrow's finer tiles keep every core fed).
/// - head MHA (batch = 16, k = 64 or n = 64): narrow beats the old wide
///   pick by +13…+29% at m ≥ 283 (wide grids of 240–512 threadgroups with
///   HALF the scheduling granularity); ties below. At 32 threadgroups the
///   k = 64 MHA shapes LOSE with xwide (heads@92: 28.3 vs 20.5 µs) — the
///   staging win does not amortize over two k-iterations, which is what
///   [`XWAVE_K_MIN`] encodes.
/// - m ≤ 188 projections: narrow is optimal or within noise of it. The
///   under-fill misses are the floor's basis: 24 threadgroups LOSE with
///   xwide (o@188 118.8 vs 104.8 µs, qkv@45 137.4 vs 108.0) while 32 WIN
///   (o@231 119.4 vs 143.1) — hence [`WAVE_TG_FLOOR`] exclusive at 24.
///
/// The instances are result-identical (each accumulates one k-ascending
/// chain per output — the shape-timing probe's divergence check reads
/// bit-identical against the CPU triple loop on every instance), so the
/// predicate is a pure dispatch change: G5 drift is untouched by it.
const WAVE_TG_LIMIT: u64 = 40;
/// The single-wave band's measured floor (exclusive): at 24 threadgroups
/// xwide under-fills the machine and LOSES to narrow (o@188, qkv@45); at
/// 32 it wins (o@231). See the [`WAVE_TG_LIMIT`] doc for the numbers.
const WAVE_TG_FLOOR: u64 = 24;
/// xwide's staging intensity only amortizes over enough k-iterations: the
/// k = 64 head-MHA shapes measured xwide LOSING at a grid size (32) where
/// the k ≥ 1024 projections WIN. 128 = four BK-32 iterations — the
/// shortest k measured winning (1024) sits far above it, the longest
/// losing (64) below.
const XWAVE_K_MIN: u32 = 128;
///
/// The ragged edge route reuses the staging front after the k-loop
/// (barrier-ordered); every instance's staging fits Metal's 32 KB
/// threadgroup limit (asserted below).
/// Narrow instance staging (A [32][65] + B [64][65] floats) and dispatch.
const NARROW_STAGING_BYTES: u64 = ((32 * 65 + 64 * 65) * std::mem::size_of::<f32>()) as u64;
const NARROW_THREADS: u64 = 512;
/// Wide instance staging (A [64][33] + B [32][65] floats) — the compiled
/// but unpicked kernel's geometry, kept beside its MSL source.
#[allow(dead_code)]
const WIDE_STAGING_BYTES: u64 = ((64 * 33 + 32 * 65) * std::mem::size_of::<f32>()) as u64;
/// xwide instance staging (A [64][33] + B [32][129] floats) and dispatch.
const XWIDE_STAGING_BYTES: u64 = ((64 * 33 + 32 * 129) * std::mem::size_of::<f32>()) as u64;
const XWIDE_THREADS: u64 = 1024;
/// Encoders per command buffer before a pipelined (no-wait) flush — keeps
/// a pass under Metal's per-buffer encoder ceiling without stalling; the
/// buffers join the committed list and are waited at the next sync.
const MAX_ENCODERS_PER_CB: u32 = 1024;

/// Widest row `ln_rows_wide` holds in registers (8 values × 256 threads).
const LN_WIDE_MAX_D: usize = 2048;

/// Split-K slice length (reflex issue 020 T11) — FIXED, a multiple of the
/// narrow BK (64 · 2 iterations per slice).
///
/// What this buys, and what it does not (measured, not assumed):
/// - A split result is a function of its OWN row: row i is Σ over the same
///   `SPLITK_KC`-slices, reduced in the same order, whatever m the call
///   carried. So two calls that BOTH split — the packed forward and the
///   per-question loop at the arena/parity shapes — are bit-identical per
///   row, exactly as two unsplit calls always were.
/// - A split call is NOT bit-identical to an unsplit one (one k-ascending
///   chain vs a sum of slice chains). That only happens when two calls on
///   the same row land on opposite sides of the [`SplitRule`]; the
///   difference is f32 summation order, bounded by G5 (measured prob drift
///   ≤ 7.5e-6 against the 1e-3 gate, top-1 unchanged on all 3 checkpoints).
/// - Making EVERY instance fold at this slice length (the global
///   bit-identity route) was built and measured: +18…+31% on the unsplit
///   m ≥ 188 shapes (live slice accumulators — register pressure, not the
///   fold arithmetic: KC 512 still cost +18%). Rejected.
const SPLITK_KC: u32 = 128;

/// The MPS arm's default row floor (reflex issue 020 T13). Every call it
/// can see is already past the split rule (≤ 3 row tiles always split),
/// so the floor only exists as the A/B's tuning seam.
const MPS_MIN_M_DEFAULT: u32 = 0;

/// When a batch-1 GEMM of `rows × n` over `k` is split (reflex issue 020
/// T11). Pinned by the per-shape sweep (`tests/metal_splitk_shape_sweep.rs`:
/// unsplit vs forced split per encoder projection, 15 rotated paired
/// rounds, M3 Max), which showed a narrow-TG ceiling alone is the WRONG
/// predictor:
///
/// | shape (n × k) | split wins at | first loss |
/// |---|---|---|
/// | attn out 1024 × 1024 | m ≤ 128 (0.56 → 0.95) | m 160 (1.09) |
/// | MLP down 1024 × 2624 | m ≤ 256 (0.35 → 0.95) | m 317 (1.17) |
/// | qkv 3072 × 1024 | m ≤ 80 (0.79 → 0.99) | m 106 (1.09) |
/// | MLP up 5248 × 1024 | m ≤ 80 (0.90 → 0.99) | m 106 (1.08) |
///
/// hence: split while the call has at most [`Self::max_row_tiles`] narrow
/// row tiles (m ≤ 96 — every shape wins there), OR its narrow grid is at
/// most [`Self::max_tgs`] threadgroups. Every sweep cell the rule splits
/// measured ≤ 1.0, and every losing cell is unsplit.
///
/// ⛔ The long-k extension the table ALSO supports — MLP down split to
/// m ≤ 256 via [`Self::long_k_max_tgs`] = 128 at `k ≥` [`Self::long_k`] —
/// did NOT survive the whole-forward A/B: paired against the first rule it
/// read seq 188 **1.020 (0/24 wins)**, seq 256 0.987, seq 140 0.994. An
/// isolated-GEMM win at the margin does not transfer to the forward (the
/// surrounding kernels change cache and occupancy), so the knob ships at
/// the base ceiling (64) and stays for the next re-measure. The row-tile
/// half: seq 24 / 46 / 54 / 80 **0.982 / 0.974 / 0.982 / 0.939** against
/// the first rule, 24/24 each. A tile choice was also
/// swept — the xwide 64×128 geometry reading each weight block once —
/// and LOST to narrow on every shape (weights sit in L2; re-reads were
/// never the binding cost — the f16-B finding again), so it is not built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitRule {
    /// Master switch.
    pub on: bool,
    /// Split while ⌈m/32⌉ ≤ this.
    pub max_row_tiles: u32,
    /// Split while ⌈n/64⌉·⌈m/32⌉ ≤ this …
    pub max_tgs: u64,
    /// … or ≤ this once `k ≥ long_k`.
    pub long_k_max_tgs: u64,
    /// The k at which [`Self::long_k_max_tgs`] applies.
    pub long_k: u32,
}

impl SplitRule {
    /// The shipped rule (the table above).
    pub const DEFAULT: Self = Self {
        on: true,
        max_row_tiles: 3,
        max_tgs: 64,
        long_k_max_tgs: 64,
        long_k: 2048,
    };
    /// The first rule (`512477e`): the narrow-TG ceiling alone — kept as
    /// the A/B arm for [`Self::DEFAULT`].
    pub const TG_CEILING_ONLY: Self = Self {
        on: true,
        max_row_tiles: 0,
        max_tgs: 64,
        long_k_max_tgs: 64,
        long_k: 2048,
    };

    /// Does a `rows × n` GEMM over `k` split?
    pub fn splits(&self, rows: u32, n: u32, k: u32) -> bool {
        if !self.on {
            return false;
        }
        let row_tiles = rows.div_ceil(32);
        let tgs = u64::from(n.div_ceil(64)) * u64::from(row_tiles);
        let ceiling = if k >= self.long_k {
            self.long_k_max_tgs
        } else {
            self.max_tgs
        };
        row_tiles <= self.max_row_tiles || tgs <= ceiling
    }
}

/// Widest operand list any `run` / `run_rows` kernel binds — the stack
/// array that replaced the per-dispatch `Vec` (riir-reflex Issue 020 T2).
/// Asserted at every call rather than assumed: a new kernel with a fifth
/// operand must RED here, never silently truncate its binding.
const MAX_RUN_BUFFERS: usize = 4;

/// flash_attn staging (Q tile [32][65] + Kᵀ tile [64][33] + V tile
/// [32][65] + scores/probs [32][33] + 64 row-stats floats + the [32][8]
/// diag(α) rescale frags of the one-pass softmax) and dispatch.
/// BQ = 32 query rows per threadgroup; the accumulator [32][64] lives as
/// one 8×8 frag per simdgroup (1024 threads = 32 simdgroups).
const FLASH_STAGING_BYTES: u64 =
    ((32 * 65 + 64 * 33 + 32 * 65 + 32 * 33 + 64 + 32 * 8) * std::mem::size_of::<f32>()) as u64;
const FLASH_THREADS: u64 = 1024;
/// Query-block rows of the fused attention kernel; hd is pinned to 64
/// (both shipped checkpoints — the kernel's rope pairing and tile mapping
/// assert it host-side).
const FLASH_HD: usize = 64;
/// Query rows per fused-attention threadgroup (mirrors the MSL `FBQ`).
const FLASH_BQ: u64 = 32;

/// The edge staging must fit inside the carved buffer's front half (the
/// k-loop's A/B tiles are dead by then; the barrier orders the reuse) and
/// the whole allocation must fit Metal's 32 KB threadgroup memory limit —
/// for BOTH instances.
/// (The asserts are const-foldable by construction — the guard exists to
/// fail the build the day a geometry edit outgrows its staging, so the
/// clippy always-true lint is allowed, not silent.)
#[allow(clippy::assertions_on_constants)]
const _: () = {
    assert!(16 * 128 <= 32 * 65 + 64 * 65);
    assert!(32 * 128 <= 64 * 33 + 32 * 65);
    assert!(32 * 128 <= 64 * 33 + 32 * 129);
    assert!(NARROW_STAGING_BYTES <= 32768);
    assert!(WIDE_STAGING_BYTES <= 32768);
    assert!(XWIDE_STAGING_BYTES <= 32768);
    assert!(FLASH_STAGING_BYTES <= 32768);
};

/// Debug trace flag (`LAYA_METAL_TRACE=1`): log every chain-cache miss.
fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LAYA_METAL_TRACE").as_deref() == Ok("1"))
}

/// Opt-in per-dispatch GPU profile (`LAYA_METAL_PROFILE=1`, reflex issue
/// 020 T11). MEASUREMENT ONLY: every dispatch gets its OWN command buffer,
/// committed and waited, and its `GPUStartTime..GPUEndTime` is recorded
/// under `(kernel, grid)`. That serializes the pass, so absolute wall is
/// NOT the shipped wall — read the per-kernel SHARES. Off, it costs one
/// cached bool per dispatch.
fn profile_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LAYA_METAL_PROFILE").as_deref() == Ok("1"))
}

/// One profiled dispatch: kernel name, dispatch grid (w, h, d), GPU seconds.
#[derive(Clone, Debug)]
pub struct ProfileRow {
    pub kernel: &'static str,
    pub grid: (u64, u64, u64),
    pub gpu_s: f64,
}

static PROFILE: Mutex<Vec<ProfileRow>> = Mutex::new(Vec::new());

/// Drain the rows recorded since the last call (empty unless
/// `LAYA_METAL_PROFILE=1`).
pub fn profile_take() -> Vec<ProfileRow> {
    std::mem::take(&mut *PROFILE.lock().expect("profile poison"))
}

/// Debug-trace instance id source (separates the per-checkpoint Metal
/// instances in the miss log).
fn next_trace_instance() -> usize {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

const KERNELS: &[&str] = &[
    "sgemm",
    "sgemm_wide",
    "sgemm_xwide",
    "sgemm_splitk",
    "splitk_reduce",
    "splitk_reduce_add",
    "splitk_reduce_glu",
    "attn_rope",
    "flash_attn",
    "add",
    "copy",
    "add_bias_row",
    "scale",
    "relu",
    "gelu_erf",
    "glu_gelu_gate",
    "ln_rows",
    "ln_rows_wide",
    "softmax_rows",
    "rope",
    "split_heads",
    "merge_heads",
    "gather_rows",
    "add_mask_bcast",
];

/// The MSL source. Sizes fit u32 (every pinned extent < 2³¹); `erf_as` is
/// candle's kernel verbatim (their constants, their op order). Every
/// pointer/scalar argument carries its explicit buffer-space index.
const MSL_HEAD: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── sgemm geometry, instances picked per-call by the single-wave rule
// (WAVE_TG_LIMIT, see run_sgemm): xwide `sgemm_xwide` (BM=64, BN=128,
// 1024 threads) ONLY when ⌈m/64⌉·⌈n/128⌉·batch fits ONE wave on this GPU —
// one full wave of fat threadgroups beats several thin ones, but only
// while nothing queues behind it; narrow `sgemm` (BM=32, 512 threads)
// otherwise — its finer tiles keep every core fed in the many-wave regime
// and its 32-row tiles pad little at short m. `sgemm_wide` (64×64, 1024
// threads) stays compiled but is NO LONGER PICKED: the 2026-09-25
// dispatch sweep measured it dominated in every hot-path cell. Shared
// laws: BK is per-instance (narrow 64 — A [32][65] + B [64][65]; xwide 32
// — a BK-48 B tile at BN 128 is [48][129] and the pair outgrows the
// limit); a staging stride must EXCEED the tile's row width or the tile
// overlaps itself (33 over a 64-wide tile corrupts every row from row 1's
// column 30 on); the ragged edge route reuses the staging front after the
// k-loop (barrier-ordered).

// candle-metal-kernels unary.metal — A&S 7.1.26 f32 erf, their constants.
inline float erf_as(float x) {
    const float a1 =  0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 =  1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 =  1.061405429f;
    const float p  =  0.3275911f;
    float sign = 1.0f;
    if (x < 0.0f) { sign = -1.0f; x = -x; }
    float t = 1.0f / (1.0f + p * x);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    return sign * y;
}

// candle unary.metal gelu_erf: x * (1 + erf(x * sqrt(1/2))) / 2.
inline float gelu_as(float x) {
    return x * (1.0f + erf_as(x * 0.70710678f)) / 2.0f;
}

// ── wide instance: 64×64 output per threadgroup, 32 simdgroups (1024
// threads), two row-twin accumulators per simdgroup (rows sgr·8 and
// (sgr+4)·8 of its column block).
"#;

/// The wide instance kernel (64×64 output per threadgroup, 1024 threads) —
/// kept COMPILED but no longer picked by `run_sgemm` (the 2026-09-25
/// dispatch sweep measured it dominated in every hot-path cell; the docs
/// at [`WAVE_TG_LIMIT`] carry the numbers). Any future re-pick must
/// re-run that sweep first.
const MSL_SGEMM_WIDE: &str = r#"
constant uint WBM = 64u;
constant uint WBN = 64u;
constant uint WBK = 32u;
constant uint WTAS = 33u;
constant uint WTBS = 65u;

kernel void sgemm_wide(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& a_rs [[buffer(6)]],
    constant uint& a_cs [[buffer(7)]],
    constant uint& b_rs [[buffer(8)]],
    constant uint& b_cs [[buffer(9)]],
    constant uint& a_bs [[buffer(10)]],
    constant uint& b_bs [[buffer(11)]],
    constant uint& c_bs [[buffer(12)]],
    threadgroup float* raw [[threadgroup(13)]],
    uint3 gtp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    // One carved staging allocation (24 960 B): A tile [64][33] = 2112
    // floats, B tile [32][65] = 2080, and the ragged-edge store path
    // reuses the front 4096 floats AFTER the k-loop (the barrier below
    // orders it past the last A/B loads). WBK 48 (the largest k-chunk
    // fitting 32 KB here) measured FLAT vs 32 — the k-loop's barriers are
    // not this instance's binding constraint.
    threadgroup float* ta = raw;
    threadgroup float* tb = raw + 64u * WTAS;
    threadgroup float* edge = raw;
    const uint m0 = gtp.y * WBM;
    const uint n0 = gtp.x * WBN;
    device const float* A = a + gtp.z * a_bs;
    device const float* B = b + gtp.z * b_bs;
    device float* C = out + gtp.z * c_bs;

    const uint sg = lid >> 5u;   // simdgroup id 0..31
    const uint lane = lid & 31u;
    const uint sgr = sg >> 3u;   // 8×8 block row 0..3 (the +4 twin below)
    const uint sgc = sg & 7u;    // 8×8 block col 0..7

    // Two accumulators: rows sgr·8 and (sgr+4)·8 for this simdgroup's
    // column block — 32 simdgroups × 2 blocks cover the 64×64 tile.
    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);

    for (uint t = 0u; t < k; t += WBK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Stage A [WBM][WBK] and B [WBK][WBN]: 2 of each 1024 elements per
        // thread. Element (kk, col) of B always lands at tb[kk·WTBS + col];
        // the two load mappings below both keep consecutive lanes on
        // consecutive global addresses for their layout.
        for (uint q = 0u; q < 2u; ++q) {
            // A tile [WBM][WBK] = [64][32]: two elements per thread.
            const uint idx = lid + q * 1024u;
            const uint r = idx >> 5u;
            const uint c = idx & 31u;
            const uint gr = m0 + r;
            const uint ac = t + c;
            ta[r * WTAS + c] = (gr < m && ac < k) ? A[gr * a_rs + ac * a_cs] : 0.0f;
        }
        for (uint q = 0u; q < 2u; ++q) {
            // B tile [WBK][WBN] = [32][64] at stride WTBS: k ∈ 0..31 from the
            // high bits, n ∈ 0..63 from the low (consecutive lanes →
            // consecutive n, coalesced along B's rows).
            const uint idx = lid + q * 1024u;
            const uint kk = idx >> 6u;
            const uint col = idx & 63u;
            const uint bc = t + kk;
            if (b_cs == 1u) {
                // Row-major B [k][n]: contiguous along n → n-fastest lanes.
                tb[kk * WTBS + col] =
                    (bc < k && n0 + col < n) ? B[bc * b_rs + n0 + col] : 0.0f;
            } else {
                tb[kk * WTBS + col] =
                    (bc < k && n0 + col < n) ? B[bc * b_rs + (n0 + col) * b_cs] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < WBK; kk += 8u) {
            simdgroup_float8x8 fa0, fa1, fb;
            simdgroup_load(fa0, ta + sgr * 8u * WTAS + kk, WTAS);
            simdgroup_load(fa1, ta + (sgr + 4u) * 8u * WTAS + kk, WTAS);
            simdgroup_load(fb, tb + kk * WTBS + sgc * 8u, WTBS);
            simdgroup_multiply_accumulate(acc0, fa0, fb, acc0);
            simdgroup_multiply_accumulate(acc1, fa1, fb, acc1);
        }
    }

    if ((m0 + WBM <= m) && (n0 + WBN <= n)) {
        simdgroup_store(acc0, C + (m0 + sgr * 8u) * n + (n0 + sgc * 8u), n);
        simdgroup_store(acc1, C + (m0 + sgr * 8u + 4u * 8u) * n + (n0 + sgc * 8u), n);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc0, edge + sg * 128u, 8u);
        simdgroup_store(acc1, edge + sg * 128u + 64u, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 2u; ++q) {
            const uint e = lane + q * 32u;
            const uint er = e >> 3u;
            const uint ec = e & 7u;
            const uint gr0 = m0 + sgr * 8u + er;
            const uint gr1 = gr0 + 4u * 8u;
            const uint gc = n0 + sgc * 8u + ec;
            if (gr0 < m && gc < n) { C[gr0 * n + gc] = edge[sg * 128u + e]; }
            if (gr1 < m && gc < n) { C[gr1 * n + gc] = edge[sg * 128u + 64u + e]; }
        }
    }
}
"#;

/// The xwide instance (the single-wave geometry: picked iff
/// ⌈m/64⌉·⌈n/128⌉·batch ≤ [`WAVE_TG_LIMIT`]): 64×128 output per
/// threadgroup, 32 simdgroups (1024 threads), four
/// accumulators per simdgroup — the row twins (sgr·8, (sgr+4)·8) × the
/// column twins (sgc·8, sgc·8+64) — covering the 8×16 grid of 8×8 blocks.
const MSL_SGEMM_XWIDE: &str = r#"
constant uint XBM = 64u;
constant uint XBN = 128u;
constant uint XBK = 32u;
constant uint XTAS = 33u;
constant uint XTBS = 129u;

kernel void sgemm_xwide(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& a_rs [[buffer(6)]],
    constant uint& a_cs [[buffer(7)]],
    constant uint& b_rs [[buffer(8)]],
    constant uint& b_cs [[buffer(9)]],
    constant uint& a_bs [[buffer(10)]],
    constant uint& b_bs [[buffer(11)]],
    constant uint& c_bs [[buffer(12)]],
    threadgroup float* raw [[threadgroup(13)]],
    uint3 gtp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    // Carved staging (24 960 B): A tile [64][33] = 2112 floats, B tile
    // [32][129] = 4128. The ragged-edge route reuses the front 4096
    // floats, in TWO phases (four accumulators need 8192 floats at once,
    // which the allocation does not hold — phase 1 drains the row-twin
    // pair, the barrier orders the reads past the reuse, phase 2 the
    // +32 row pair).
    threadgroup float* ta = raw;
    threadgroup float* tb = raw + 64u * XTAS;
    threadgroup float* edge = raw;
    const uint m0 = gtp.y * XBM;
    const uint n0 = gtp.x * XBN;
    device const float* A = a + gtp.z * a_bs;
    device const float* B = b + gtp.z * b_bs;
    device float* C = out + gtp.z * c_bs;

    const uint sg = lid >> 5u;   // simdgroup id 0..31
    const uint lane = lid & 31u;
    const uint sgr = sg >> 3u;   // row block 0..3 (the +4 twin below)
    const uint sgc = sg & 7u;    // column block 0..7 (the +8 twin below)

    // Four accumulators: rows sgr·8 / sgr·8+32 × columns sgc·8 /
    // sgc·8+64 — 32 simdgroups × 4 blocks cover the 64×128 tile.
    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);

    for (uint t = 0u; t < k; t += XBK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Stage A [64][32]: two elements per thread.
        for (uint q = 0u; q < 2u; ++q) {
            const uint idx = lid + q * 1024u;
            const uint r = idx >> 5u;
            const uint c = idx & 31u;
            const uint gr = m0 + r;
            const uint ac = t + c;
            ta[r * XTAS + c] = (gr < m && ac < k) ? A[gr * a_rs + ac * a_cs] : 0.0f;
        }
        // Stage B [32][128]: four elements per thread; k ∈ 0..31 from the
        // high bits, n ∈ 0..127 from the low (consecutive lanes →
        // consecutive n, coalesced along B's rows; the staged tile is the
        // SAME [K][N] layout for both B orientations).
        for (uint q = 0u; q < 4u; ++q) {
            const uint idx = lid + q * 1024u;
            const uint kk = idx >> 7u;
            const uint col = idx & 127u;
            const uint bc = t + kk;
            if (b_cs == 1u) {
                tb[kk * XTBS + col] =
                    (bc < k && n0 + col < n) ? B[bc * b_rs + n0 + col] : 0.0f;
            } else {
                tb[kk * XTBS + col] =
                    (bc < k && n0 + col < n) ? B[bc * b_rs + (n0 + col) * b_cs] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < XBK; kk += 8u) {
            simdgroup_float8x8 fa0, fa1, fb0, fb1;
            simdgroup_load(fa0, ta + sgr * 8u * XTAS + kk, XTAS);
            simdgroup_load(fa1, ta + (sgr + 4u) * 8u * XTAS + kk, XTAS);
            simdgroup_load(fb0, tb + kk * XTBS + sgc * 8u, XTBS);
            simdgroup_load(fb1, tb + kk * XTBS + sgc * 8u + 64u, XTBS);
            simdgroup_multiply_accumulate(acc00, fa0, fb0, acc00);
            simdgroup_multiply_accumulate(acc01, fa0, fb1, acc01);
            simdgroup_multiply_accumulate(acc10, fa1, fb0, acc10);
            simdgroup_multiply_accumulate(acc11, fa1, fb1, acc11);
        }
    }

    if ((m0 + XBM <= m) && (n0 + XBN <= n)) {
        simdgroup_store(acc00, C + (m0 + sgr * 8u) * n + (n0 + sgc * 8u), n);
        simdgroup_store(acc01, C + (m0 + sgr * 8u) * n + (n0 + sgc * 8u + 64u), n);
        simdgroup_store(acc10, C + (m0 + sgr * 8u + 32u) * n + (n0 + sgc * 8u), n);
        simdgroup_store(acc11, C + (m0 + sgr * 8u + 32u) * n + (n0 + sgc * 8u + 64u), n);
    } else {
        // Ragged tile: drain the four accumulators through the shared
        // front in two phases (a barrier after each store burst and after
        // each scalar drain orders the reuse).
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc00, edge + sg * 128u, 8u);
        simdgroup_store(acc01, edge + sg * 128u + 64u, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 2u; ++q) {
            const uint e = lane + q * 32u;
            const uint er = e >> 3u;
            const uint ec = e & 7u;
            const uint gr0 = m0 + sgr * 8u + er;
            const uint gc0 = n0 + sgc * 8u + ec;
            const uint gc1 = gc0 + 64u;
            if (gr0 < m && gc0 < n) { C[gr0 * n + gc0] = edge[sg * 128u + e]; }
            if (gr0 < m && gc1 < n) { C[gr0 * n + gc1] = edge[sg * 128u + 64u + e]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc10, edge + sg * 128u, 8u);
        simdgroup_store(acc11, edge + sg * 128u + 64u, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 2u; ++q) {
            const uint e = lane + q * 32u;
            const uint er = e >> 3u;
            const uint ec = e & 7u;
            const uint gr0 = m0 + sgr * 8u + 32u + er;
            const uint gc0 = n0 + sgc * 8u + ec;
            const uint gc1 = gc0 + 64u;
            if (gr0 < m && gc0 < n) { C[gr0 * n + gc0] = edge[sg * 128u + e]; }
            if (gr0 < m && gc1 < n) { C[gr0 * n + gc1] = edge[sg * 128u + 64u + e]; }
        }
    }
}
"#;

/// The narrow instance: 32×64 output per threadgroup, 16 simdgroups (512
/// threads), two column-twin accumulators per simdgroup (columns sgc·8 and
/// (sgc+4)·8 of its row block). Same staging laws as the wide instance.
const MSL_SGEMM_NARROW: &str = r#"
constant uint BM = 32u;
constant uint BN = 64u;
constant uint BK = 64u;
constant uint TAS = 65u;
constant uint TBS = 65u;

// out[b][m×n] = A[b][m×k] @ B[b][k×n]; element (i, j) of X at
// x[i·xrs + j·xcs]; batch b's operands start b·bxs elements in (the
// batch-1 call sites pass zero batch strides). Row-major B (b_cs == 1)
// stages coalesced along n; transposed B — the Wᵀ / Kᵀ shapes (b_rs == 1)
// — along k; the staged tile is the SAME [K][N] layout either way.
kernel void sgemm(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& a_rs [[buffer(6)]],
    constant uint& a_cs [[buffer(7)]],
    constant uint& b_rs [[buffer(8)]],
    constant uint& b_cs [[buffer(9)]],
    constant uint& a_bs [[buffer(10)]],
    constant uint& b_bs [[buffer(11)]],
    constant uint& c_bs [[buffer(12)]],
    threadgroup float* raw [[threadgroup(13)]],
    uint3 gtp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    // Carved staging (24 960 B): A tile [32][65] = 2080 floats, B tile
    // [64][65] = 4160; BK 64 halves the k-loop's two barriers per unit of
    // work against the BK 32 geometry. The ragged-edge route reuses the
    // front 2048 floats after the k-loop.
    threadgroup float* ta = raw;
    threadgroup float* tb = raw + 32u * TAS;
    threadgroup float* edge = raw;
    const uint m0 = gtp.y * BM;
    const uint n0 = gtp.x * BN;
    device const float* A = a + gtp.z * a_bs;
    device const float* B = b + gtp.z * b_bs;
    device float* C = out + gtp.z * c_bs;

    const uint sg = lid >> 5u;   // simdgroup id 0..15
    const uint lane = lid & 31u;
    const uint sgr = sg >> 2u;   // 8×8 block row 0..3
    const uint sgc = sg & 3u;    // 8×8 block col 0..3 (the +4 twin below)

    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);

    for (uint t = 0u; t < k; t += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 4u; ++q) {
            // A tile [BM][BK] = [32][64]: four elements per thread.
            const uint idx = lid + q * 512u;
            const uint r = idx >> 6u;
            const uint c = idx & 63u;
            const uint gr = m0 + r;
            const uint ac = t + c;
            ta[r * TAS + c] = (gr < m && ac < k) ? A[gr * a_rs + ac * a_cs] : 0.0f;
        }
        for (uint q = 0u; q < 8u; ++q) {
            // B tile [BK][BN] = [64][64]: k ∈ 0..63 from the high bits,
            // n ∈ 0..63 from the low (consecutive lanes → consecutive n,
            // coalesced along B's rows); the staged tile is the SAME [K][N]
            // layout for both B orientations.
            const uint idx = lid + q * 512u;
            const uint kk = idx >> 6u;
            const uint col = idx & 63u;
            const uint bc = t + kk;
            if (b_cs == 1u) {
                tb[kk * TBS + col] =
                    (bc < k && n0 + col < n) ? B[bc * b_rs + n0 + col] : 0.0f;
            } else {
                tb[kk * TBS + col] =
                    (bc < k && n0 + col < n) ? B[bc * b_rs + (n0 + col) * b_cs] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_float8x8 fa, fb0, fb1;
            simdgroup_load(fa, ta + sgr * 8u * TAS + kk, TAS);
            simdgroup_load(fb0, tb + kk * TBS + sgc * 8u, TBS);
            simdgroup_load(fb1, tb + kk * TBS + (sgc + 4u) * 8u, TBS);
            simdgroup_multiply_accumulate(acc0, fa, fb0, acc0);
            simdgroup_multiply_accumulate(acc1, fa, fb1, acc1);
        }
    }

    if ((m0 + BM <= m) && (n0 + BN <= n)) {
        simdgroup_store(acc0, C + (m0 + sgr * 8u) * n + (n0 + sgc * 8u), n);
        simdgroup_store(acc1, C + (m0 + sgr * 8u) * n + (n0 + sgc * 8u + 4u * 8u), n);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc0, edge + sg * 128u, 8u);
        simdgroup_store(acc1, edge + sg * 128u + 64u, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 2u; ++q) {
            const uint e = lane + q * 32u;
            const uint er = e >> 3u;
            const uint ec = e & 7u;
            const uint gr = m0 + sgr * 8u + er;
            const uint gc0 = n0 + sgc * 8u + ec;
            const uint gc1 = gc0 + 4u * 8u;
            if (gr < m && gc0 < n) { C[gr * n + gc0] = edge[sg * 128u + e]; }
            if (gr < m && gc1 < n) { C[gr * n + gc1] = edge[sg * 128u + 64u + e]; }
        }
    }
}
"#;

/// Split-K narrow instance (reflex issue 020 T11): the narrow body over ONE
/// k-slice `[z·kc, min(k, z·kc + kc))` per `grid.z`, writing its partial
/// product into `part + z·m·n` (row stride n); `splitk_reduce` then sums the
/// slices in ascending z. At small m the narrow grid is ⌈n/64⌉·⌈m/32⌉
/// threadgroups — 32 for the d×d projections at m ≤ 64, under one per core
/// on a 40-core part — and each walks the WHOLE k alone; slicing k
/// multiplies the grid without touching the tile math. Batch-1 only
/// (grid.z is the slice). The host always passes `kc =` [`SPLITK_KC`], so a
/// split result is a function of its own row alone (see that constant for
/// what is, and is not, bit-identical).
const MSL_SGEMM_SPLITK: &str = r#"
kernel void sgemm_splitk(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* part [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& a_rs [[buffer(6)]],
    constant uint& a_cs [[buffer(7)]],
    constant uint& b_rs [[buffer(8)]],
    constant uint& b_cs [[buffer(9)]],
    constant uint& kc [[buffer(10)]],
    threadgroup float* raw [[threadgroup(13)]],
    uint3 gtp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float* ta = raw;
    threadgroup float* tb = raw + 32u * TAS;
    threadgroup float* edge = raw;
    const uint m0 = gtp.y * BM;
    const uint n0 = gtp.x * BN;
    const uint k0 = gtp.z * kc;
    const uint k1 = min(k, k0 + kc);
    device const float* A = a;
    device const float* B = b;
    device float* C = part + gtp.z * (m * n);

    const uint sg = lid >> 5u;
    const uint lane = lid & 31u;
    const uint sgr = sg >> 2u;
    const uint sgc = sg & 3u;

    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);

    for (uint t = k0; t < k1; t += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 4u; ++q) {
            const uint idx = lid + q * 512u;
            const uint r = idx >> 6u;
            const uint c = idx & 63u;
            const uint gr = m0 + r;
            const uint ac = t + c;
            ta[r * TAS + c] = (gr < m && ac < k1) ? A[gr * a_rs + ac * a_cs] : 0.0f;
        }
        for (uint q = 0u; q < 8u; ++q) {
            const uint idx = lid + q * 512u;
            const uint kk = idx >> 6u;
            const uint col = idx & 63u;
            const uint bc = t + kk;
            if (b_cs == 1u) {
                tb[kk * TBS + col] =
                    (bc < k1 && n0 + col < n) ? B[bc * b_rs + n0 + col] : 0.0f;
            } else {
                tb[kk * TBS + col] =
                    (bc < k1 && n0 + col < n) ? B[bc * b_rs + (n0 + col) * b_cs] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_float8x8 fa, fb0, fb1;
            simdgroup_load(fa, ta + sgr * 8u * TAS + kk, TAS);
            simdgroup_load(fb0, tb + kk * TBS + sgc * 8u, TBS);
            simdgroup_load(fb1, tb + kk * TBS + (sgc + 4u) * 8u, TBS);
            simdgroup_multiply_accumulate(acc0, fa, fb0, acc0);
            simdgroup_multiply_accumulate(acc1, fa, fb1, acc1);
        }
    }

    if ((m0 + BM <= m) && (n0 + BN <= n)) {
        simdgroup_store(acc0, C + (m0 + sgr * 8u) * n + (n0 + sgc * 8u), n);
        simdgroup_store(acc1, C + (m0 + sgr * 8u) * n + (n0 + sgc * 8u + 4u * 8u), n);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(acc0, edge + sg * 128u, 8u);
        simdgroup_store(acc1, edge + sg * 128u + 64u, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 2u; ++q) {
            const uint e = lane + q * 32u;
            const uint er = e >> 3u;
            const uint ec = e & 7u;
            const uint gr = m0 + sgr * 8u + er;
            const uint gc0 = n0 + sgc * 8u + ec;
            const uint gc1 = gc0 + 4u * 8u;
            if (gr < m && gc0 < n) { C[gr * n + gc0] = edge[sg * 128u + e]; }
            if (gr < m && gc1 < n) { C[gr * n + gc1] = edge[sg * 128u + 64u + e]; }
        }
    }
}

// out[i] = Σ_z part[z·mn + i], z ascending (a fixed order — deterministic
// run to run).
kernel void splitk_reduce(device const float* part [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& mn [[buffer(2)]],
                          constant uint& slices [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= mn) { return; }
    float acc = part[gid];
    for (uint z = 1u; z < slices; ++z) { acc += part[z * mn + gid]; }
    out[gid] = acc;
}

// The residual-fold epilogue (reflex issue 020 T11, the last open rung):
// the split reduce and the encoder's residual add in ONE kernel —
// out[gid] = (Σ slices) + res[gid]. Bit-identical to the pair it replaces
// (the same k-ascending slice chain, then one add; IEEE addition
// commutes), so the fold removes a full m·n staging write+read and the
// add dispatch without touching a bit. out may alias res: each thread
// reads only res[gid] before writing out[gid], and no other thread
// touches that element.
kernel void splitk_reduce_add(device const float* part [[buffer(0)]],
                              device const float* res [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant uint& mn [[buffer(3)]],
                              constant uint& slices [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= mn) { return; }
    float acc = part[gid];
    for (uint z = 1u; z < slices; ++z) { acc += part[z * mn + gid]; }
    out[gid] = acc + res[gid];
}

// The GLU-fold epilogue: the MLP-up projection's split reduce and the
// `glu_gelu_gate` activation in ONE kernel. Output element (r, j) carries
// TWO chains — the activation half (fused[r, j]) and the gate half
// (fused[r, I + j]) — each summed over z ascending exactly as
// `splitk_reduce` sums it, then the glu kernel's own expression order
// (gelu_as(act) * gate). Bit-identical to reduce + glu_gelu_gate by
// construction; the fused [rows × 2i] staging never exists.
kernel void splitk_reduce_glu(device const float* part [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant uint& rows [[buffer(2)]],
                              constant uint& i_sz [[buffer(3)]],
                              constant uint& n [[buffer(4)]],
                              constant uint& slices [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= rows * i_sz) { return; }
    const uint r = gid / i_sz;
    const uint j = gid % i_sz;
    const uint mn = rows * n;
    const uint act_idx = r * n + j;
    const uint gate_idx = r * n + i_sz + j;
    float acc_act = part[act_idx];
    float acc_gate = part[gate_idx];
    for (uint z = 1u; z < slices; ++z) {
        acc_act += part[z * mn + act_idx];
        acc_gate += part[z * mn + gate_idx];
    }
    out[gid] = gelu_as(acc_act) * acc_gate;
}
"#;

/// The fused attention's rope pre-pass (opt-in, reflex issue 020 T10 rung
/// 2): derives the rope ONCE per layer for the Q and K thirds of qkv — Q
/// additionally takes the 1/√hd scale, the reference's rope-then-scale
/// order — into a packed `[2, seq, d]` scratch (Q front, K back). Without
/// it `flash_attn`'s staging re-derives every rope pair per (query block ×
/// head × key tile), ×heads redundant on Q and ×(window coverage) redundant
/// on K. One thread per rope pair; the per-element expressions are the
/// staging's own, in the same order, so both arms produce the same f32
/// bits and the G5 drift is carried by the softmax path alone.
const MSL_ATTN_ROPE: &str = r#"
kernel void attn_rope(
    device const float* qkv [[buffer(0)]],
    device const float* cos [[buffer(1)]],
    device const float* sin [[buffer(2)]],
    device float* rk [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const uint pairs_per_row = d / 2u;   // hd = 64: 32 pairs per head
    const uint row = gid / pairs_per_row;
    const uint p = gid % pairs_per_row;  // pair index within the row
    const uint j = p & 31u;              // rope pair index within hd
    const uint h = p >> 5u;              // head
    if (row >= seq) { return; }
    const float c = cos[row * FHD + j];
    const float s = sin[row * FHD + j];
    const uint hbase = h * FHD;
    // Q: rotate + scale (the flash staging's own expression order).
    {
        const uint qb = row * (3u * d) + hbase;
        const float qa = qkv[qb + j];
        const float qbp = qkv[qb + 32u + j];
        rk[row * d + hbase + j] = (qa * c - qbp * s) * scale;
        rk[row * d + hbase + 32u + j] = (qbp * c + qa * s) * scale;
    }
    // K: rotate, into the back half of the scratch.
    {
        const uint kbase = row * (3u * d) + d + hbase;
        const float ka = qkv[kbase + j];
        const float kbp = qkv[kbase + 32u + j];
        const uint ob = seq * d + row * d + hbase;
        rk[ob + j] = ka * c - kbp * s;
        rk[ob + 32u + j] = kbp * c + ka * s;
    }
}
"#;

/// The fused attention kernel (the Metal lane's flash form): ONE dispatch
/// per layer over the packed qkv — split, rope, q-scale, scores, sliding
/// window, softmax and value mix, and the head merge all in-kernel; the
/// seq² scores parent the reference sequence materializes (and its mask
/// add + multi-pass softmax + context re-read) never exist. The accumulator
/// is normalized with the ONE-PASS online form (riir-reflex Issue 020 T10
/// rung 3): each key tile is staged and scored ONCE, the row max / sum
/// ride in registers, and the accumulator is rescaled by exp(m_old − m_new)
/// through one diagonal 8×8 MMA per tile. It replaced a two-pass form whose
/// max-only pass re-staged every K tile (rope included) and re-ran every
/// score MMA. Sliding-window layers predicate on
/// `window` — each query block reads only its [q₀−w, q_end+w] key slice —
/// which is also where the FLOP win lives at seq ≫ window (the english
/// geometry: window 64, ~2/3 sliding layers); `window == seq` (the host
/// clamps) means full attention. The additive mask tensor the reference
/// sequence consumes describes the same allowed set and is never read
/// here.
const MSL_FLASH: &str = r#"
// Self-contained geometry (no sgemm instance constants referenced).
constant uint FBQ = 32u;   // query rows per threadgroup
constant uint FHD = 64u;   // head dim (asserted host-side)
constant uint FTAS = 65u;  // row stride over a 64-wide tile
constant uint FTKS = 33u;  // row stride over the Kᵀ tile's 32-wide kt rows

kernel void flash_attn(
    device const float* qkv [[buffer(0)]],
    device const float* rk [[buffer(1)]],
    device const float* cos [[buffer(2)]],
    device const float* sin [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& seq [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& hd [[buffer(7)]],
    constant uint& window [[buffer(8)]],
    constant uint& use_pre [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    threadgroup float* raw [[threadgroup(11)]],
    uint2 gtp [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    // Carved staging: ta Q tile [32][65] (rope+scale applied), tk Kᵀ tile
    // [64][33] ([kt][key], rope applied — both halves of each rope pair
    // live in the tile), tv V tile [32][65] ([key][hd] natural), ts
    // scores/probs [32][33], st row stats (l in st[32..64] at the drain;
    // m / l live in registers during the key loop), dg the [32][8]
    // diag(α) rescale rows. ta is reused for the accumulator after the key loop;
    // the barriers order every reuse.
    threadgroup float* ta = raw;
    threadgroup float* tk = raw + 32u * FTAS;
    threadgroup float* tv = tk + 64u * FTKS;
    threadgroup float* ts = tv + 32u * FTAS;
    threadgroup float* st = ts + 32u * FTKS;
    threadgroup float* dg = st + 64u;

    const uint q0 = gtp.x * FBQ;
    const uint h = gtp.y;
    const uint d = heads * FHD;
    device const float* Q = qkv + h * FHD;
    device const float* K = qkv + d + h * FHD;
    device const float* V = qkv + 2u * d + h * FHD;

    const uint sg = lid >> 5u;   // simdgroup id 0..31
    const uint sgr = sg >> 3u;   // accumulator row group 0..3
    const uint sgc = sg & 7u;    // accumulator col group 0..7

    // Key range [lo, hi): every (q, k) pair with |q − k| ≤ window for
    // every live row q of this block, clamped to the sequence. window
    // arrives clamped to seq (full attention ⇒ lo 0, hi seq).
    const uint last_row = min(q0 + FBQ - 1u, seq - 1u);
    const uint lo = (q0 > window) ? (q0 - window) : 0u;
    const uint hi = min(last_row + window + 1u, seq);

    // Stage the Q block once — one rope pair per thread, (r, j) and
    // (r, j + 32). `use_pre` copies the pre-roped, pre-scaled row out of
    // the attn_rope scratch ([row, d] natural layout); otherwise rotate-
    // half RoPE here, then the 1/√hd scale (the reference sequence's
    // rope-then-scale order). Rows past seq stage zeros (the ragged block
    // tail; their outputs are never stored).
    {
        const uint r = lid >> 5u;
        const uint j = lid & 31u;
        const uint row = q0 + r;
        float qa = 0.0f, qb = 0.0f;
        if (use_pre != 0u) {
            if (row < seq) {
                qa = rk[row * d + h * FHD + j];
                qb = rk[row * d + h * FHD + 32u + j];
            }
            ta[r * FTAS + j] = qa;
            ta[r * FTAS + 32u + j] = qb;
        } else {
            float c = 0.0f, s = 0.0f;
            if (row < seq) {
                qa = Q[row * (3u * d) + j];
                qb = Q[row * (3u * d) + 32u + j];
                c = cos[row * FHD + j];
                s = sin[row * FHD + j];
            }
            ta[r * FTAS + j] = (qa * c - qb * s) * scale;
            ta[r * FTAS + 32u + j] = (qb * c + qa * s) * scale;
        }
    }

    // ONE pass — online softmax (Issue 020 T10 rung 3). Each tile's
    // scores are computed ONCE: the row max m and sum l are carried in
    // registers (simdgroup `sg` owns row sg, so every lane of it holds the
    // same m / l), and the accumulator is rescaled by α = exp(m_old − m_new)
    // whenever the max moves. The rescale is ONE 8×8 MMA per tile against
    // a diagonal frag (dg) — cheaper than the retired max-only pass, which
    // re-staged every K tile (rope included) and re-ran every score MMA.
    // α = 1 exactly when the max does not move, and a tile with no live key
    // for a row leaves m, l and that accumulator row untouched.
    float m_reg = -3.402823466e+38f;
    float l_reg = 0.0f;
    simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
    for (uint t = lo; t < hi; t += FBQ) {
        const uint wk = min(FBQ, hi - t);   // live keys in this tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Stage Kᵀ: one rope pair per thread — (kt j, j+32) × key. `use_pre`
        // copies both pair rows out of the attn_rope scratch; otherwise
        // reads both pair elements (the partner is outside the staged kt
        // half but the same cache row), rotates, then writes both kt rows.
        if (use_pre != 0u) {
            const uint j = lid >> 5u;    // rope pair index 0..31
            const uint key = lid & 31u;  // tile key 0..31
            const uint krow = t + key;
            float ka = 0.0f, kb = 0.0f;
            if (krow < hi) {
                ka = rk[seq * d + krow * d + h * FHD + j];
                kb = rk[seq * d + krow * d + h * FHD + 32u + j];
            }
            tk[j * FTKS + key] = ka;
            tk[(j + 32u) * FTKS + key] = kb;
        } else {
            const uint j = lid >> 5u;    // rope pair index 0..31
            const uint key = lid & 31u;  // tile key 0..31
            const uint krow = t + key;
            float ka = 0.0f, kb = 0.0f;
            float c = 0.0f, s = 0.0f;
            if (krow < hi) {
                ka = K[krow * (3u * d) + j];
                kb = K[krow * (3u * d) + 32u + j];
                c = cos[krow * FHD + j];
                s = sin[krow * FHD + j];
            }
            tk[j * FTKS + key] = ka * c - kb * s;
            tk[(j + 32u) * FTKS + key] = kb * c + ka * s;
        }
        for (uint q = 0u; q < 2u; ++q) {
            // V tile [32 key][64 hd]: two elements per thread, natural
            // layout (no rope).
            const uint idx = lid + q * 1024u;
            const uint key = idx >> 6u;
            const uint col = idx & 63u;
            const uint krow = t + key;
            tv[key * FTAS + col] =
                (krow < hi) ? V[krow * (3u * d) + col] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // scores frag = Q tile × Kᵀ over the full hd contraction. The
        // scores tile is [32 rows][32 keys] — only the four key-col groups
        // (sgc < 4) hold live frags; sgc ≥ 4 must neither compute nor
        // store (their store would run past the 32-key row into the next
        // row of the [32][33] buffer).
        if (sgc < 4u) {
            simdgroup_float8x8 sf = simdgroup_float8x8(0.0f);
            for (uint kk = 0u; kk < FHD; kk += 8u) {
                simdgroup_float8x8 fa, fb;
                simdgroup_load(fa, ta + sgr * 8u * FTAS + kk, FTAS);
                simdgroup_load(fb, tk + kk * FTKS + sgc * 8u, FTKS);
                simdgroup_multiply_accumulate(sf, fa, fb, sf);
            }
            simdgroup_store(sf, ts + sgr * 8u * FTKS + sgc * 8u, FTKS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // One simdgroup per row (sg = row), one lane per key. Per-row
        // window predicate: the block key range is only a bounds
        // optimization — within a tile a key is live for a row only when
        // |q − k| ≤ window (the reference's mask row; out-of-window keys
        // are exp(f32::MIN − m) = 0 there, so skipping them is exact).
        // Padding keys (c ≥ wk) and out-of-window keys write p = 0 (their
        // V rows are staged zero, and l skips them).
        {
            const uint c = lid & 31u;
            const uint q = q0 + sg;
            const uint k = t + c;
            const uint dk = (k > q) ? (k - q) : (q - k);
            const bool live = c < wk && dk <= window;
            const float sc = live ? ts[sg * FTKS + c] : -3.402823466e+38f;
            const float m_new = max(m_reg, simd_max(sc));
            const float p = live ? precise::exp(sc - m_new) : 0.0f;
            const float alpha = precise::exp(m_reg - m_new);
            ts[sg * FTKS + c] = p;
            l_reg = l_reg * alpha + simd_sum(p);
            m_reg = m_new;
            // dg [32 rows][8]: row r carries α_r on its diagonal slot
            // (r & 7), so rows sgr·8.. form diag(α) for row group sgr.
            if (c < 8u) { dg[sg * 8u + c] = (c == (sg & 7u)) ? alpha : 0.0f; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // acc ← diag(α) × acc, then acc += P × V over this tile's 32 keys
        // (padding columns hold p = 0 against V rows staged zero).
        {
            simdgroup_float8x8 fd, sc_acc;
            simdgroup_load(fd, dg + sgr * 64u, 8u);
            simdgroup_multiply(sc_acc, fd, acc);
            acc = sc_acc;
        }
        for (uint kk = 0u; kk < FBQ; kk += 8u) {
            simdgroup_float8x8 fa, fb;
            simdgroup_load(fa, ts + sgr * 8u * FTKS + kk, FTKS);
            simdgroup_load(fb, tv + kk * FTAS + sgc * 8u, FTAS);
            simdgroup_multiply_accumulate(acc, fa, fb, acc);
        }
    }
    // Publish the per-row l (lane 0 of simdgroup `sg` owns row sg's).
    if ((lid & 31u) == 0u) { st[32u + sg] = l_reg; }

    // Drain: the accumulator frag → the ta front (Q is dead), then the
    // per-row 1/l normalize + the merged-heads store [seq, d].
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(acc, ta + (sgr * 8u) * 64u + sgc * 8u, 64u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const uint r = lid >> 5u;
        const uint j = lid & 31u;
        const uint row = q0 + r;
        if (row < seq) {
            const float inv = 1.0f / st[32u + r];
            const uint ob = row * d + h * FHD;
            out[ob + j] = ta[r * 64u + j] * inv;
            out[ob + 32u + j] = ta[r * 64u + 32u + j] * inv;
        }
    }
}
"#;

/// The kernels after the two sgemm instances.
const MSL_TAIL: &str = r#"

kernel void add(device float* x [[buffer(0)]],
                device const float* y [[buffer(1)]],
                constant uint& len [[buffer(2)]],
                uint gid [[thread_position_in_grid]]) {
    if (gid < len) { x[gid] += y[gid]; }
}

kernel void copy(device float* dst [[buffer(0)]],
                 device const float* src [[buffer(1)]],
                 constant uint& len [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid < len) { dst[gid] = src[gid]; }
}

kernel void add_bias_row(device float* x [[buffer(0)]],
                         device const float* bias [[buffer(1)]],
                         constant uint& len [[buffer(2)]],
                         constant uint& d [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < len) { x[gid] += bias[gid % d]; }
}

kernel void scale(device float* x [[buffer(0)]],
                  constant uint& len [[buffer(1)]],
                  constant float& s [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid < len) { x[gid] *= s; }
}

kernel void relu(device float* x [[buffer(0)]],
                 constant uint& len [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid < len && x[gid] < 0.0f) { x[gid] = 0.0f; }
}

kernel void gelu_erf(device float* x [[buffer(0)]],
                     constant uint& len [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < len) { x[gid] = gelu_as(x[gid]); }
}

// out[r, j] = gelu_erf(fused[r, j]) * fused[r, I + j] — the CPU lane's
// exact multiply order ((e+1)·0.5·v, then ·g).
kernel void glu_gelu_gate(device const float* fused [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint& rows [[buffer(2)]],
                          constant uint& i_sz [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= rows * i_sz) { return; }
    const uint r = gid / i_sz;
    const uint j = gid % i_sz;
    const uint base = r * 2u * i_sz;
    const float act = gelu_as(fused[base + j]);
    out[gid] = act * fused[base + i_sz + j];
}

// One threadgroup (one simdgroup) per row: strided lanes + simd
// reductions. Mean → centered² → 1/sqrt(var+eps) → (v−mean)·inv·w, the
// CPU op order; inv_d arrives as the CPU lane's f64-rounded constant.
kernel void ln_rows(device const float* x [[buffer(0)]],
                    device const float* w [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant uint& rows [[buffer(3)]],
                    constant uint& d [[buffer(4)]],
                    constant float& inv_d [[buffer(5)]],
                    constant float& eps [[buffer(6)]],
                    uint tpg [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_threadgroup]]) {
    if (tpg >= rows) { return; }
    device const float* row = x + tpg * d;
    device float* orow = out + tpg * d;
    float acc = 0.0f;
    for (uint i = lane; i < d; i += 32u) { acc += row[i]; }
    const float mean = simd_sum(acc) * inv_d;
    float vacc = 0.0f;
    for (uint i = lane; i < d; i += 32u) {
        const float c = row[i] - mean;
        vacc += c * c;
    }
    const float var = simd_sum(vacc) * inv_d;
    const float inv = 1.0f / sqrt(var + eps);
    for (uint i = lane; i < d; i += 32u) { orow[i] = (row[i] - mean) * inv * w[i]; }
}

// LayerNorm, 256 threads per row with the row held in REGISTERS (reflex
// issue 020 T11): ONE read of the row instead of `ln_rows`' three strided
// passes, 8× the lanes per row. Mean and variance reduce through simd_sum
// plus an 8-entry threadgroup array that EVERY thread sums in the same
// fixed order (deterministic, row-local — packed ≡ loop holds). d ≤ 2048
// (8 values per thread); the host keeps `ln_rows` above that.
kernel void ln_rows_wide(device const float* x [[buffer(0)]],
                         device const float* w [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         constant uint& rows [[buffer(3)]],
                         constant uint& d [[buffer(4)]],
                         constant float& inv_d [[buffer(5)]],
                         constant float& eps [[buffer(6)]],
                         uint tpg [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float part[8];
    if (tpg >= rows) { return; }
    device const float* row = x + tpg * d;
    device float* orow = out + tpg * d;
    const uint sg = lid >> 5u;
    const uint ln = lid & 31u;
    float v[8];
    float acc = 0.0f;
    for (uint j = 0u; j < 8u; ++j) {
        const uint i = lid + j * 256u;
        v[j] = i < d ? row[i] : 0.0f;
        acc += v[j];
    }
    acc = simd_sum(acc);
    if (ln == 0u) { part[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint g = 0u; g < 8u; ++g) { tot += part[g]; }
    const float mean = tot * inv_d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float vacc = 0.0f;
    for (uint j = 0u; j < 8u; ++j) {
        const uint i = lid + j * 256u;
        const float c = v[j] - mean;
        vacc += i < d ? c * c : 0.0f;
    }
    vacc = simd_sum(vacc);
    if (ln == 0u) { part[sg] = vacc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float vtot = 0.0f;
    for (uint g = 0u; g < 8u; ++g) { vtot += part[g]; }
    const float inv = 1.0f / sqrt(vtot * inv_d + eps);
    for (uint j = 0u; j < 8u; ++j) {
        const uint i = lid + j * 256u;
        if (i < d) { orow[i] = (v[j] - mean) * inv * w[i]; }
    }
}

// One threadgroup (one simdgroup) per row: strided lanes + simd
// reductions. The v1 kernel ran one thread per row — 32× the GPU idle.
kernel void softmax_rows(device float* x [[buffer(0)]],
                         constant uint& rows [[buffer(1)]],
                         constant uint& n [[buffer(2)]],
                         uint tpg [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]]) {
    if (tpg >= rows) { return; }
    device float* row = x + tpg * n;
    float mx = -3.402823466e+38f;
    for (uint i = lane; i < n; i += 32u) { mx = fmax(mx, row[i]); }
    mx = simd_max(mx);
    float sum = 0.0f;
    for (uint i = lane; i < n; i += 32u) {
        const float e = precise::exp(row[i] - mx);
        row[i] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    const float inv = 1.0f / sum;
    for (uint i = lane; i < n; i += 32u) { row[i] *= inv; }
}

// q[(h·seq+pos)·hd + j] paired with its +half twin; one thread per
// (h·seq+pos, j).
kernel void rope(device float* q [[buffer(0)]],
                 device const float* cos_t [[buffer(1)]],
                 device const float* sin_t [[buffer(2)]],
                 constant uint& seq [[buffer(3)]],
                 constant uint& heads [[buffer(4)]],
                 constant uint& hd [[buffer(5)]],
                 uint gid [[thread_position_in_grid]]) {
    const uint hf = hd / 2u;
    if (gid >= heads * seq * hf) { return; }
    const uint hps = gid / hf;
    const uint j = gid % hf;
    const uint base = hps * hd;
    const uint prow = (hps % seq) * hd;
    const float c = cos_t[prow + j];
    const float s = sin_t[prow + j];
    const float q1 = q[base + j];
    const float q2 = q[base + hf + j];
    q[base + j] = q1 * c - q2 * s;
    q[base + hf + j] = q2 * c + q1 * s;
}

kernel void split_heads(device const float* src [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& row_stride [[buffer(2)]],
                        constant uint& off [[buffer(3)]],
                        constant uint& seq [[buffer(4)]],
                        constant uint& heads [[buffer(5)]],
                        constant uint& hd [[buffer(6)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= heads * seq * hd) { return; }
    const uint h = gid / (seq * hd);
    const uint r = gid % (seq * hd);
    const uint s = r / hd;
    const uint i = r % hd;
    out[gid] = src[s * row_stride + off + h * hd + i];
}

kernel void merge_heads(device const float* src [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& seq [[buffer(2)]],
                        constant uint& heads [[buffer(3)]],
                        constant uint& hd [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
    const uint d = heads * hd;
    if (gid >= seq * d) { return; }
    const uint s = gid / d;
    const uint rem = gid % d;
    const uint h = rem / hd;
    const uint i = rem % hd;
    out[gid] = src[(h * seq + s) * hd + i];
}

kernel void gather_rows(device const float* x [[buffer(0)]],
                        device const uint* rows [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant uint& d [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    const uint r = gid / d;
    const uint i = gid % d;
    out[gid] = x[rows[r] * d + i];
}

// scores[r] += mask[r % mlen] — the sliding-window mask broadcast over
// every head's slab of the scores parent in ONE dispatch (was one add
// dispatch per head).
kernel void add_mask_bcast(device float* x [[buffer(0)]],
                           device const float* mask [[buffer(1)]],
                           constant uint& len [[buffer(2)]],
                           constant uint& mlen [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid < len) { x[gid] += mask[gid % mlen]; }
}
"#;

fn rt(detail: impl std::fmt::Display) -> LayaError {
    LayaError::Runtime(format!("riir metal backend: {detail}"))
}

/// A pass-scoped command buffer: created on the first encode after a sync,
/// appended-to by every op, committed at the sync (or the encoder cap).
struct PendingPass {
    cb: CommandBuffer,
    /// The ONE compute encoder this command buffer is filling (reflex Issue
    /// 018 T3). A default `MTLComputeCommandEncoder` is
    /// `MTLDispatchTypeSerial` — consecutive dispatches inside it already
    /// run in order with Metal inserting the memory barriers — so the
    /// previous one-encoder-per-op shape bought identical semantics at
    /// ~341 create/`endEncoding` pairs per forward. `end_encoding` is
    /// called exactly once, at commit.
    enc: ComputeCommandEncoder,
    encodes: u32,
}

/// A resolved kernel: the pipeline plus its `thread_execution_width`, read
/// ONCE at init. The width was previously fetched per dispatch — an ObjC
/// property message on the elementwise hot path (riir-reflex Issue 020 T2).
/// A chain-cache key: `(host ptr, len, epoch)`.
type ChainKey = (usize, usize, u64);

struct Kern {
    p: ComputePipelineState,
    width: u64,
}

/// The open pass buffer plus the committed-but-unwaited pipeline flushes.
/// `sync()` waits both — a flush must never let a `download_into` read
/// ahead of in-flight writes.
#[derive(Default)]
struct PendingState {
    open: Option<PendingPass>,
    committed: Vec<CommandBuffer>,
}

/// The Metal backend: one device + queue, the compiled-once kernel library,
/// the permanent weight cache, and the generation-keyed activation cache.
pub struct Metal {
    device: Device,
    queue: CommandQueue,
    pipelines: HashMap<&'static str, Kern>,
    /// `(ptr, len)` → device buffer for the agent-OWNED weight slices
    /// (stable addresses and contents for the agent's lifetime: every
    /// `matmul_w` weight, LN scales, biases, the embedding table).
    weights: Mutex<HashMap<(usize, usize), Buffer>>,
    /// `(ptr, len, gen)` → device buffer for activations and per-forward
    /// host-authored inputs. The generation (bumped at every pass) makes a
    /// recycled heap address miss instead of serving a stale epoch's
    /// bytes; within an epoch a hit's device copy is current because the
    /// forward bodies write every activation device-side before reading
    /// it (the write-first audit in the module doc).
    /// Each entry carries its last-TOUCH stamp ([`Metal::touch`]): the
    /// sub-slice download resolves to the most recently touched buffer at
    /// a base pointer, never the tightest container (see `download_into`).
    chain: Mutex<HashMap<ChainKey, (Buffer, u64)>>,
    /// Monotonic touch counter for the chain entries.
    touch_seq: AtomicU64,
    /// Device-resident TRANSPOSED projection weights — `Wᵀ` as row-major
    /// `[k, n]`, keyed by the ORIGINAL `W` slice's `(ptr, len)`. Permanent,
    /// first-miss build, never invalidated (same contract as `weights`).
    ///
    /// riir-reflex Issue 020 T4: `matmul_w` computes `dst = a @ Wᵀ`, and binding
    /// `W` row-major `[n, k]` means `b_cs = k` — consecutive staging lanes
    /// read `k` floats apart (4 KB at `d = 1024`), so every lane of a
    /// simdgroup touches a different cache line. Holding the transpose
    /// instead puts the SAME staged values on the kernel's `b_cs == 1`
    /// branch — the one the activation `matmul` already uses.
    weights_t: Mutex<HashMap<(usize, usize), Buffer>>,
    /// The pass-scoped command buffer + the committed drain list.
    pending: Mutex<PendingState>,
    /// The sync generation (how many host-read barriers have run).
    epoch: AtomicU64,
    /// Debug kill-switch (`LAYA_METAL_PER_OP_SYNC`): `1` = sync + write
    /// every result back to its host slice after each op (the slow flow);
    /// `2` = sync only (no writeback — bisects barrier vs host-freshness).
    per_op_sync: u8,
    /// Kill-switch (`LAYA_METAL_FLASH=0`): route `attention_forward` back
    /// through the reference op sequence instead of the fused kernel — the
    /// A/B and bisect posture, never a silent default.
    flash_disabled: bool,
    /// Opt-in (`LAYA_METAL_ROPE_HOIST=1`, reflex issue 020 T10 rung 2): run
    /// the `attn_rope` pre-pass before each fused attention and let
    /// `flash_attn`'s staging copy the pre-roped Q/K instead of re-deriving
    /// the rope per (query block × head × key tile). Default-off — the
    /// in-kernel rope arm is the shipped behavior; promotion follows the
    /// quiet-box probe.
    rope_hoist: bool,
    /// The rope-hoist scratch: one packed `[2, seq, d]` device buffer (Q
    /// front, K back), keyed `(d, capacity in floats)`, grow-only in seq so
    /// a suite's variable-length questions don't churn allocations.
    /// Persistent across passes — allocated outside the chain cache, same
    /// lifetime class as the weight buffers. Content staleness is harmless
    /// by construction: within a layer, `attn_rope` writes rows `< seq`
    /// before `flash_attn` reads them (write-first dispatch pair, serial
    /// GPU ordering), and no later op reads rows ≥ seq.
    rope_scratch: Mutex<Option<(usize, usize, Buffer)>>,
    /// The split-K decision (reflex issue 020 T11) — default
    /// [`SplitRule::DEFAULT`]; `LAYA_METAL_SPLITK=0` is the kill-switch,
    /// `LAYA_METAL_SPLITK_MAXTGS` overrides the base ceiling.
    split_rule: SplitRule,
    /// LayerNorm through `ln_rows_wide` (reflex issue 020 T11) — default ON;
    /// `LAYA_METAL_LN_WIDE=0` restores the one-simdgroup `ln_rows`.
    ln_wide: bool,
    /// The fold rungs (reflex issue 020 T11) — default ON since the
    /// quiet-box paired A/B promoted them (2026-09-26: 24/24 paired wins
    /// on every shape within the split rule's reach, medians −1.5…−5.9%,
    /// the no-op control flat at 1.002/1.001 above the crossover):
    /// - `fold_res` (`LAYA_METAL_FOLD_RES=0` is the kill-switch): the
    ///   encoder's two residual adds ride the split-K reduce
    ///   (`splitk_reduce_add`) when the whole call splits — one kernel
    ///   instead of reduce + staging + add.
    /// - `fold_glu` (`LAYA_METAL_FOLD_GLU=0` is the kill-switch): the
    ///   MLP-up projection's reduce applies the GLU gate directly
    ///   (`splitk_reduce_glu`) — the fused `[m × 2i]` staging round-trip
    ///   never happens.
    ///
    /// Both arms are bit-identical to the streams they replace by
    /// construction (same slice chains, the epilogue kernels' own
    /// expression order); the knob-off arm runs the unfused stream on a
    /// dedicated grow-only staging buffer so it stays HEAD's allocation
    /// posture, never a per-call alloc.
    fold_res: bool,
    fold_glu: bool,
    /// The fold fallback's GEMM staging (the encoder's retired
    /// `attn_out`/`fused` scratch role): capacity in f32 + buffer,
    /// grow-only with headroom, never shrunk; serial dispatch orders its
    /// reuse. Deliberately NOT shared with [`Self::splitk_scratch`]: a
    /// mixed plan's split runs use that buffer as their part slab in the
    /// same dispatch stream while the fallback stages into this one.
    fold_stage_scratch: Mutex<Option<(usize, Buffer)>>,
    /// The split-K partial scratch (capacity in f32, buffer) — grown with
    /// headroom, never shrunk; serial dispatch orders its reuse.
    splitk_scratch: Mutex<Option<(usize, Buffer)>>,
    /// Split-K GEMMs dispatched by this instance (the reach counter a test
    /// asserts, so a split arm can never pass on the plain kernel).
    splitk_count: AtomicU64,
    /// FOLD epilogues dispatched (`splitk_reduce_add` / `splitk_reduce_glu`)
    /// — the fold arm's reach counter, same law as [`Self::splitk_count`]:
    /// a fold arm must never pass on the unfused stream.
    fold_count: AtomicU64,
    /// The pass's row segmentation hint ([`Backend::set_row_segments`]).
    row_segments: Mutex<Vec<u32>>,
    /// The MPS GEMM arm (reflex issue 020 T13, [`mps`]): unsplit batch-1
    /// dense GEMMs with `m ≥ mps_min_m` dispatch Apple's
    /// `MPSMatrixMultiplication` instead of the narrow/xwide instance —
    /// bit-identical to narrow on every priced cell. Default ON since the
    /// paired whole-forward A/B promoted it (2026-09-26, 24/24 wins on every
    /// shape it reaches: loop 106 → 0.741, 188 → 0.657, 512 → 0.596, the
    /// typed 5×179 packed case → 0.575; the all-split controls flat at
    /// 1.000). `None` = off (`LAYA_METAL_MPS=0` is the kill-switch, or the
    /// framework did not resolve).
    mps: Option<mps::MpsGemm>,
    /// The arm's row floor (`LAYA_METAL_MPS_MIN_M`, default
    /// [`MPS_MIN_M_DEFAULT`]).
    mps_min_m: u32,
    /// MPS GEMMs dispatched — the arm's reach counter (same law as
    /// [`Self::splitk_count`]).
    mps_count: AtomicU64,
    /// Debug-trace instance id.
    trace_id: usize,
}

impl Metal {
    /// Build the backend — fails loud when no Metal device exists (the
    /// candle lane's `device_from_env` precedent; never a silent CPU
    /// fallback).
    pub fn new() -> Result<Self> {
        let Some(device) = Device::system_default() else {
            return Err(rt("LAYA_DEVICE=metal: no Metal device on this host"));
        };
        let queue = device.new_command_queue();
        let msl = format!(
            "{MSL_HEAD}{MSL_SGEMM_NARROW}{MSL_SGEMM_SPLITK}{MSL_SGEMM_WIDE}{MSL_SGEMM_XWIDE}{MSL_FLASH}{MSL_ATTN_ROPE}{MSL_TAIL}"
        );
        let lib = device
            .new_library_with_source(&msl, &metal::CompileOptions::new())
            .map_err(|e| rt(format!("MSL compile failed: {e}")))?;
        let mut pipelines = HashMap::with_capacity(KERNELS.len());
        for name in KERNELS {
            let f = lib
                .get_function(name, None)
                .map_err(|e| rt(format!("kernel {name}: {e}")))?;
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| rt(format!("pipeline {name}: {e}")))?;
            let width = p.thread_execution_width();
            pipelines.insert(*name, Kern { p, width });
        }
        Ok(Self {
            device,
            queue,
            pipelines,
            weights: Mutex::new(HashMap::new()),
            weights_t: Mutex::new(HashMap::new()),
            chain: Mutex::new(HashMap::new()),
            touch_seq: AtomicU64::new(0),
            pending: Mutex::new(PendingState::default()),
            epoch: AtomicU64::new(0),
            per_op_sync: std::env::var("LAYA_METAL_PER_OP_SYNC")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            flash_disabled: std::env::var("LAYA_METAL_FLASH").as_deref() == Ok("0"),
            rope_hoist: std::env::var("LAYA_METAL_ROPE_HOIST").as_deref() == Ok("1"),
            rope_scratch: Mutex::new(None),
            ln_wide: std::env::var("LAYA_METAL_LN_WIDE").as_deref() != Ok("0"),
            fold_res: std::env::var("LAYA_METAL_FOLD_RES").as_deref() != Ok("0"),
            fold_glu: std::env::var("LAYA_METAL_FOLD_GLU").as_deref() != Ok("0"),
            fold_stage_scratch: Mutex::new(None),
            split_rule: SplitRule {
                on: std::env::var("LAYA_METAL_SPLITK").as_deref() != Ok("0"),
                max_tgs: std::env::var("LAYA_METAL_SPLITK_MAXTGS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(SplitRule::DEFAULT.max_tgs),
                ..SplitRule::DEFAULT
            },
            splitk_scratch: Mutex::new(None),
            splitk_count: AtomicU64::new(0),
            fold_count: AtomicU64::new(0),
            row_segments: Mutex::new(Vec::new()),
            mps: (std::env::var("LAYA_METAL_MPS").as_deref() != Ok("0"))
                .then(mps::MpsGemm::new)
                .flatten(),
            mps_min_m: std::env::var("LAYA_METAL_MPS_MIN_M")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(MPS_MIN_M_DEFAULT),
            mps_count: AtomicU64::new(0),
            trace_id: next_trace_instance(),
        })
    }

    /// [`Self::new`] with the split-K knobs set explicitly instead of from
    /// `LAYA_METAL_SPLITK` / `LAYA_METAL_SPLITK_MAXTGS`
    /// — the A/B and test seam; env is process-global, this is per instance.
    pub fn with_splitk(on: bool, max_tgs: u64) -> Result<Self> {
        Self::with_split_rule(SplitRule {
            on,
            max_tgs,
            ..SplitRule::DEFAULT
        })
    }

    /// [`Self::new`] with an explicit [`SplitRule`] — the A/B seam for the
    /// rule itself (e.g. [`SplitRule::TG_CEILING_ONLY`], the first rule).
    pub fn with_split_rule(rule: SplitRule) -> Result<Self> {
        let mut m = Self::new()?;
        m.split_rule = rule;
        Ok(m)
    }

    /// Builder: route LayerNorm through `ln_rows_wide` (`true`, the
    /// default) or the one-simdgroup `ln_rows` — the A/B seam.
    pub fn with_ln_wide(mut self, on: bool) -> Self {
        self.ln_wide = on;
        self
    }

    /// Builder: the fold rungs' A/B seam — set both knobs explicitly
    /// instead of from `LAYA_METAL_FOLD_RES` / `LAYA_METAL_FOLD_GLU` (env
    /// is process-global, this is per instance).
    pub fn with_folds(mut self, res: bool, glu: bool) -> Self {
        self.fold_res = res;
        self.fold_glu = glu;
        self
    }

    /// Builder: the MPS GEMM arm's A/B seam (reflex issue 020 T13) — set it
    /// explicitly instead of from `LAYA_METAL_MPS` (env is process-global,
    /// this is per instance). `true` on a host whose MPS classes do not
    /// resolve stays off (see [`Self::mps_active`]).
    pub fn with_mps(mut self, on: bool) -> Self {
        self.mps = if on { mps::MpsGemm::new() } else { None };
        self
    }

    /// Whether the MPS GEMM arm is live on this instance.
    pub fn mps_active(&self) -> bool {
        self.mps.is_some()
    }

    /// MPS GEMMs this instance has dispatched — the arm's reach counter.
    pub fn mps_dispatches(&self) -> u64 {
        self.mps_count.load(Ordering::Relaxed)
    }

    /// Split-K GEMMs this instance has dispatched.
    pub fn splitk_dispatches(&self) -> u64 {
        self.splitk_count.load(Ordering::Relaxed)
    }

    /// Fold epilogues (`splitk_reduce_add` + `splitk_reduce_glu`) this
    /// instance has dispatched — the fold arm's reach counter.
    pub fn fold_dispatches(&self) -> u64 {
        self.fold_count.load(Ordering::Relaxed)
    }

    fn upload(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(data) as u64,
            RESOURCE_OPTIONS,
        )
    }

    fn upload_u32(&self, data: &[u32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            (data.len() * 4) as u64,
            RESOURCE_OPTIONS,
        )
    }

    /// Device-resident TRANSPOSE of an agent-owned projection weight:
    /// `W` is row-major `[n, k]`, the buffer holds `Wᵀ` row-major `[k, n]`.
    /// Permanent cache keyed by the ORIGINAL slice, first-miss build.
    ///
    /// The transpose runs on the host into a temporary that is dropped
    /// after the upload, so the steady-state host footprint is unchanged
    /// and the device holds ONE copy per weight (the untransposed form is
    /// never uploaded for a `matmul_w` operand).
    fn weight_t_buf(&self, w: &[f32], n: usize, k: usize) -> Buffer {
        assert_eq!(w.len(), n * k, "weight_t extent");
        let key = (w.as_ptr() as usize, w.len());
        let mut map = self.weights_t.lock().expect("weight_t cache poison");
        if let Some(b) = map.get(&key) {
            return b.clone();
        }
        let mut t = vec![0f32; n * k];
        // Row-blocked so the READ side is sequential per source row; the
        // values are copied, never combined, so the result is exact.
        for (row, src) in w.chunks_exact(k).enumerate() {
            for (kk, v) in src.iter().enumerate() {
                t[kk * n + row] = *v;
            }
        }
        let b = self.upload(&t);
        map.insert(key, b.clone());
        b
    }

    /// Device-resident copy of an agent-owned weight slice — permanent
    /// cache, first-miss copy, never invalidated.
    fn weight_buf(&self, data: &[f32]) -> Buffer {
        let key = (data.as_ptr() as usize, data.len());
        let mut map = self.weights.lock().expect("weight cache poison");
        if let Some(b) = map.get(&key) {
            return b.clone();
        }
        let b = self.upload(data);
        map.insert(key, b.clone());
        b
    }

    /// Activation / per-forward input: hit within the current epoch → the
    /// device copy is current (no copy); miss → create + copy.
    fn chain_buf(&self, data: &[f32]) -> Buffer {
        let epoch = self.epoch.load(Ordering::Relaxed);
        let key = (data.as_ptr() as usize, data.len(), epoch);
        let mut map = self.chain.lock().expect("chain cache poison");
        let stamp = self.touch_seq.fetch_add(1, Ordering::Relaxed);
        if let Some((b, t)) = map.get_mut(&key) {
            *t = stamp;
            return b.clone();
        }
        let b = self.upload(data);
        map.insert(key, (b.clone(), stamp));
        if trace_enabled() {
            eprintln!(
                "[trace] chain MISS inst {} ptr {:p} len {} epoch {epoch}",
                self.trace_id,
                data.as_ptr(),
                data.len()
            );
        }
        b
    }

    /// A device destination slot for this epoch's `(ptr, len)`. `dst` is
    /// the WHOLE parent slice the op writes into (the forward bodies pass
    /// whole parents + explicit offsets), so per-head loops share one slot
    /// and serial GPU ordering keeps it current. Dsts are write-first, so
    /// a hit reuses the existing buffer.
    fn chain_slot_for(&self, dst: &[f32]) -> Buffer {
        let epoch = self.epoch.load(Ordering::Relaxed);
        let key = (dst.as_ptr() as usize, dst.len(), epoch);
        let mut map = self.chain.lock().expect("chain cache poison");
        let stamp = self.touch_seq.fetch_add(1, Ordering::Relaxed);
        if let Some((b, t)) = map.get_mut(&key) {
            *t = stamp;
            return b.clone();
        }
        let b = self.scratch(dst.len());
        map.insert(key, (b.clone(), stamp));
        b
    }

    fn scratch(&self, len_f32: usize) -> Buffer {
        self.device.new_buffer(
            (std::mem::size_of::<f32>() * len_f32) as u64,
            RESOURCE_OPTIONS,
        )
    }

    /// The rope-hoist scratch for a `[2, seq, d]` pre-pass output — reuse
    /// while `(d, capacity)` covers the request, else allocate with headroom
    /// (a suite's per-question seq spread must not churn device
    /// allocations). Dropping a replaced buffer is safe even against
    /// unwaited commands: Metal command buffers retain their referenced
    /// objects, and every forward syncs before the next realloc point.
    fn rope_hoist_buf(&self, seq: usize, d: usize) -> Buffer {
        let need = 2 * seq * d;
        let mut g = self.rope_scratch.lock().expect("rope scratch poison");
        if let Some((cd, cap, b)) = g.as_ref()
            && *cd == d
            && *cap >= need
        {
            return b.clone();
        }
        let cap = need + need / 2;
        let b = self.scratch(cap);
        *g = Some((d, cap, b.clone()));
        b
    }

    /// Host-read barrier: commit the open pass buffer, then wait every
    /// committed-but-unwaited buffer (the pipeline flushes included).
    fn sync(&self) {
        let mut st = self.pending.lock().expect("pending cb poison");
        if let Some(pending) = st.open.take() {
            pending.enc.end_encoding();
            pending.cb.commit();
            st.committed.push(pending.cb);
        }
        for cb in st.committed.drain(..) {
            cb.wait_until_completed();
        }
    }

    /// One forward is beginning: drain any outstanding GPU work (a cleared
    /// chain entry releases its Buffer — that must never happen while
    /// commands still reference it), bump the pass epoch, and drop the
    /// previous pass's slots. Host-authored buffers (rope tables, mask,
    /// `act_in`, the id list) are rebuilt per forward — often at recycled
    /// heap addresses with fresh contents — so last pass's keys must never
    /// hit; within a pass the serial queue keeps every slot device-current.
    fn begin_pass_impl(&self) {
        self.sync();
        self.epoch.fetch_add(1, Ordering::Relaxed);
        self.chain.lock().expect("chain cache poison").clear();
    }

    fn copy_out(buf: &Buffer, out: &mut [f32]) {
        // SAFETY: the buffer was created with ≥ out.len() f32 elements in
        // storage mode shared, and the queue has been synced, so its
        // contents are host-readable.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.contents().cast::<f32>(),
                out.as_mut_ptr(),
                out.len(),
            );
        }
    }

    /// Per-op debug sync: barrier + (mode 1) write the slot's whole
    /// content back to its host slice (host bytes stay current — the
    /// pre-lazy semantics). Mode 2 barriers only.
    fn debug_writeback(&self, slot: &Buffer, dst: &mut [f32]) {
        if self.per_op_sync >= 1 {
            self.sync();
            if self.per_op_sync == 1 {
                Self::copy_out(slot, dst);
            }
        }
    }

    /// Encode one kernel dispatch into the PASS command buffer (no commit
    /// — the lazy shape; commit + wait happens in [`Self::sync`], or a
    /// pipelined no-wait flush at the encoder cap). `buffers` bind at
    /// `[[buffer(0..)]]`, then `uargs`/`fargs` as 4-byte `constant`
    /// scalars, then any staging buffers. `threadgroups` selects
    /// `dispatch_thread_groups` (the 2D/3D kernels) over
    /// `dispatch_threads` (the 1D elementwise kernels). Runs inside an
    /// autoreleasepool — the autoreleased encoders drain per op instead
    /// of accumulating on a thread with no Cocoa runloop.
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        p: &ComputePipelineState,
        buffers: &[(&Buffer, u64)],
        uargs: &[u32],
        fargs: &[f32],
        grid: MTLSize,
        tpg: MTLSize,
        tiles: Option<&[(u64, u64)]>,
        threadgroups: bool,
    ) -> Result<()> {
        autoreleasepool(|_| {
            let mut st = self.pending.lock().expect("pending cb poison");
            self.open_pass(&mut st);
            let pending = st.open.as_mut().expect("just inserted");
            let enc = &pending.enc;
            enc.set_compute_pipeline_state(p);
            for (i, (b, off)) in buffers.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), *off);
            }
            let mut idx = buffers.len() as u64;
            for v in uargs {
                enc.set_bytes(idx, 4, std::ptr::from_ref(v).cast::<c_void>());
                idx += 1;
            }
            for v in fargs {
                enc.set_bytes(idx, 4, std::ptr::from_ref(v).cast::<c_void>());
                idx += 1;
            }
            if let Some(tiles) = tiles {
                for (base, bytes) in tiles {
                    enc.set_threadgroup_memory_length(*base, *bytes);
                }
            }
            if threadgroups {
                enc.dispatch_thread_groups(grid, tpg);
            } else {
                enc.dispatch_threads(grid, tpg);
            }
            pending.encodes += 1;
            self.close_encode(
                &mut st,
                || {
                    self.pipelines
                        .iter()
                        .find(|(_, k)| std::ptr::eq(&*k.p, &**p))
                        .map_or("?", |(n, _)| *n)
                },
                (grid.width, grid.height, grid.depth),
            );
        });
        Ok(())
    }

    /// The shared post-encode step: under `LAYA_METAL_PROFILE=1` commit +
    /// wait + record this dispatch's GPU time; otherwise flush the pass at
    /// the [`MAX_ENCODERS_PER_CB`] cap.
    fn close_encode(
        &self,
        st: &mut PendingState,
        kernel: impl FnOnce() -> &'static str,
        grid: (u64, u64, u64),
    ) {
        if profile_enabled() {
            let done = st.open.take().expect("an encode just ran");
            done.enc.end_encoding();
            done.cb.commit();
            done.cb.wait_until_completed();
            let cb: &metal::CommandBufferRef = &done.cb;
            // SAFETY: GPUStartTime/GPUEndTime are CFTimeInterval (f64)
            // properties of a completed MTLCommandBuffer.
            // The legacy `objc` macro probes `feature = "cargo-clippy"`.
            #[allow(unexpected_cfgs)]
            let (t0, t1): (f64, f64) = {
                use metal::objc::{msg_send, sel, sel_impl};
                unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) }
            };
            PROFILE.lock().expect("profile poison").push(ProfileRow {
                kernel: kernel(),
                grid,
                gpu_s: t1 - t0,
            });
            return;
        }
        let full = st
            .open
            .as_ref()
            .is_some_and(|p| p.encodes >= MAX_ENCODERS_PER_CB);
        if full {
            let done = st.open.take().expect("checked above");
            done.enc.end_encoding();
            done.cb.commit();
            st.committed.push(done.cb);
        }
    }

    /// Open the pass buffer + its compute encoder if none is open.
    fn open_pass(&self, st: &mut PendingState) {
        if st.open.is_none() {
            let cb = self.queue.new_command_buffer().to_owned();
            let enc = cb.new_compute_command_encoder().to_owned();
            st.open = Some(PendingPass {
                cb,
                enc,
                encodes: 0,
            });
        }
    }

    /// One MPS GEMM into the open pass (reflex issue 020 T13): end the
    /// pass's compute encoder, let MPS encode into the command buffer,
    /// re-open a compute encoder for the ops that follow. Hazard-tracked
    /// buffers order the boundary exactly as the serial encoder did.
    fn run_sgemm_mps(
        &self,
        mps: &mps::MpsGemm,
        a: mps::Operand<'_>,
        b: mps::Operand<'_>,
        c: mps::Operand<'_>,
    ) -> Result<()> {
        self.mps_count.fetch_add(1, Ordering::Relaxed);
        autoreleasepool(|_| {
            let mut st = self.pending.lock().expect("pending cb poison");
            self.open_pass(&mut st);
            let pending = st.open.as_mut().expect("just opened");
            pending.enc.end_encoding();
            mps.encode(&self.device, &pending.cb, a, b, c);
            pending.enc = pending.cb.new_compute_command_encoder().to_owned();
            pending.encodes += 1;
            self.close_encode(&mut st, || "mps_sgemm", (u64::from(c.3), u64::from(c.2), 1));
        });
        Ok(())
    }

    /// The 1D elementwise dispatch.
    fn run(
        &self,
        kernel: &'static str,
        buffers: &[&Buffer],
        uargs: &[u32],
        fargs: &[f32],
        len: u64,
    ) -> Result<()> {
        let k = self
            .pipelines
            .get(kernel)
            .ok_or_else(|| rt(format!("kernel {kernel} missing")))?;
        let width = k.width;
        let groups = len.div_ceil(width).max(1);
        // Stack-resident operand list — `run` is ~250 of the ~341 dispatches
        // per forward and every one of them allocated a `Vec` here (reflex
        // riir-reflex Issue 020 T2). Every elementwise kernel binds ≤ 4 buffers.
        assert!(
            buffers.len() <= MAX_RUN_BUFFERS,
            "run({kernel}): {} operands exceeds MAX_RUN_BUFFERS",
            buffers.len()
        );
        let mut bufs = [(buffers[0], 0u64); MAX_RUN_BUFFERS];
        for (slot, b) in bufs.iter_mut().zip(buffers) {
            *slot = (*b, 0);
        }
        self.encode(
            &k.p,
            &bufs[..buffers.len()],
            uargs,
            fargs,
            MTLSize {
                width: groups * width,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
            None,
            false,
        )
    }

    /// The 1D dispatch at explicit operand byte offsets (whole-parent
    /// slots + offsets, the `add` mask-slab shape).
    #[allow(clippy::too_many_arguments)]
    fn run_at(
        &self,
        kernel: &'static str,
        a: (&Buffer, u64),
        b: (&Buffer, u64),
        uargs: &[u32],
        fargs: &[f32],
        len: u64,
    ) -> Result<()> {
        let k = self
            .pipelines
            .get(kernel)
            .ok_or_else(|| rt(format!("kernel {kernel} missing")))?;
        let width = k.width;
        let groups = len.div_ceil(width).max(1);
        self.encode(
            &k.p,
            &[a, b],
            uargs,
            fargs,
            MTLSize {
                width: groups * width,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
            None,
            false,
        )
    }

    /// The row-parallel reduction kernels (softmax / LN): ONE threadgroup —
    /// one simdgroup — per row, 32 threads, no staging memory.
    fn run_rows(
        &self,
        kernel: &'static str,
        buffers: &[&Buffer],
        uargs: &[u32],
        fargs: &[f32],
        rows: u64,
    ) -> Result<()> {
        let k = self
            .pipelines
            .get(kernel)
            .ok_or_else(|| rt(format!("kernel {kernel} missing")))?;
        assert!(
            buffers.len() <= MAX_RUN_BUFFERS,
            "run_rows({kernel}): {} operands exceeds MAX_RUN_BUFFERS",
            buffers.len()
        );
        let mut bufs = [(buffers[0], 0u64); MAX_RUN_BUFFERS];
        for (slot, b) in bufs.iter_mut().zip(buffers) {
            *slot = (*b, 0);
        }
        self.encode(
            &k.p,
            &bufs[..buffers.len()],
            uargs,
            fargs,
            MTLSize {
                width: rows.max(1),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
            None,
            true,
        )
    }

    /// Split-K slice count for a batch-1 `rows × n` GEMM over `k` (reflex
    /// issue 020 T11) — the [`SplitRule`] on this instance. `None` = run
    /// the unsliced dispatch. The slice length is FIXED ([`SPLITK_KC`]) —
    /// never a function of m.
    fn splitk_slices(&self, rows: u32, n: u32, k: u32) -> Option<u32> {
        let slices = k.div_ceil(SPLITK_KC);
        (self.split_rule.splits(rows, n, k) && slices >= 2).then_some(slices)
    }

    /// The split-K partial scratch, grown to hold `need` f32 (half again
    /// as headroom so a suite's per-question m spread does not churn it).
    fn splitk_buf(&self, need: usize) -> Buffer {
        let mut g = self.splitk_scratch.lock().expect("splitk scratch poison");
        if let Some((cap, b)) = g.as_ref()
            && *cap >= need
        {
            return b.clone();
        }
        let cap = need + need / 2;
        let b = self.scratch(cap);
        *g = Some((cap, b.clone()));
        b
    }

    /// The fold fallback's GEMM staging — the same grow-only shape as
    /// [`Self::splitk_buf`], but a separate buffer: within one dispatch
    /// stream a mixed plan's split runs take `splitk_buf` as their part
    /// slab while the fallback stages here, so sharing one would corrupt
    /// the other's rows.
    fn fold_stage_buf(&self, need: usize) -> Buffer {
        let mut g = self
            .fold_stage_scratch
            .lock()
            .expect("fold stage scratch poison");
        if let Some((cap, b)) = g.as_ref()
            && *cap >= need
        {
            return b.clone();
        }
        let cap = need + need / 2;
        let b = self.scratch(cap);
        *g = Some((cap, b.clone()));
        b
    }

    /// The unfused stream the fold replaces (and its mixed-plan fallback):
    /// GEMM into the instance's own staging buffer, then the add — today's
    /// op sequence with `fold_stage_buf` in the encoder's retired
    /// `attn_out` scratch role, so the control arm keeps HEAD's allocation
    /// posture (one grow-only buffer, never a per-call alloc).
    fn matmul_w_then_add(&self, a: &[f32], m: usize, k: usize, w: &[f32], n: usize, x: &mut [f32]) {
        let ab = self.chain_buf(a);
        let wb = self.weight_t_buf(w, n, k);
        let stage = self.fold_stage_buf(m * n);
        self.run_sgemm(
            (&ab, 0),
            (&wb, 0),
            (&stage, 0),
            &[
                m as u32, n as u32, k as u32, k as u32, // a_rs
                1,        // a_cs
                n as u32, // b_rs — Wᵀ row-major [k, n]
                1,        // b_cs
                0, 0, 0,
            ],
            m as u32,
            n as u32,
            1,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let xb = self.chain_buf(x);
        let len = m * n;
        self.run_at("add", (&xb, 0), (&stage, 0), &[len as u32], &[], len as u64)
            .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&xb, x);
    }

    /// [`Self::matmul_w_then_add`]'s GLU twin: GEMM into staging, then the
    /// gate kernel — the unfused stream, same allocation posture.
    fn matmul_w_then_glu(
        &self,
        a: &[f32],
        m: usize,
        k: usize,
        w: &[f32],
        i_sz: usize,
        act: &mut [f32],
    ) {
        let n = i_sz * 2;
        let ab = self.chain_buf(a);
        let wb = self.weight_t_buf(w, n, k);
        let stage = self.fold_stage_buf(m * n);
        self.run_sgemm(
            (&ab, 0),
            (&wb, 0),
            (&stage, 0),
            &[
                m as u32, n as u32, k as u32, k as u32, // a_rs
                1,        // a_cs
                n as u32, // b_rs — Wᵀ row-major [k, n]
                1,        // b_cs
                0, 0, 0,
            ],
            m as u32,
            n as u32,
            1,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let ob = self.chain_slot_for(act);
        self.run(
            "glu_gelu_gate",
            &[&stage, &ob],
            &[m as u32, i_sz as u32],
            &[],
            (m * i_sz) as u64,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&ob, act);
    }

    /// Split-K narrow GEMM: `slices` k-slices of `kc` into the partial
    /// scratch, then the fixed-order reduce into `out`. Same `uargs` layout
    /// as [`Self::run_sgemm`] (the batch strides are unused — batch 1).
    #[allow(clippy::too_many_arguments)]
    fn run_sgemm_splitk(
        &self,
        a: (&Buffer, u64),
        b: (&Buffer, u64),
        out: (&Buffer, u64),
        uargs: &[u32; 10],
        m: u32,
        n: u32,
        slices: u32,
        kc: u32,
    ) -> Result<()> {
        let (part, mn) = self.sgemm_splitk_parts(a, b, uargs, m, n, slices, kc)?;
        let red = self
            .pipelines
            .get("splitk_reduce")
            .ok_or_else(|| rt("kernel splitk_reduce missing"))?;
        let width = red.width;
        self.encode(
            &red.p,
            &[(&part, 0), out],
            &[mn as u32, slices],
            &[],
            MTLSize {
                width: (mn as u64).div_ceil(width) * width,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
            None,
            false,
        )
    }

    /// Split-K narrow GEMM with the RESIDUAL FOLD epilogue
    /// (`splitk_reduce_add`): out = (Σ slices) + res, one kernel instead
    /// of reduce + staging + add (reflex issue 020 T11). `out` and `res`
    /// are the same device slot (the encoder's residual stream — the
    /// buffer must be device-current; the encoder's LayerNorm always
    /// makes it so this epoch).
    #[allow(clippy::too_many_arguments)]
    fn run_sgemm_splitk_accum(
        &self,
        a: (&Buffer, u64),
        b: (&Buffer, u64),
        out_res: (&Buffer, u64),
        uargs: &[u32; 10],
        m: u32,
        n: u32,
        slices: u32,
        kc: u32,
    ) -> Result<()> {
        let (part, mn) = self.sgemm_splitk_parts(a, b, uargs, m, n, slices, kc)?;
        let red = self
            .pipelines
            .get("splitk_reduce_add")
            .ok_or_else(|| rt("kernel splitk_reduce_add missing"))?;
        let width = red.width;
        self.encode(
            &red.p,
            &[(&part, 0), out_res, out_res],
            &[mn as u32, slices],
            &[],
            MTLSize {
                width: (mn as u64).div_ceil(width) * width,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
            None,
            false,
        )
    }

    /// Split-K narrow GEMM with the GLU FOLD epilogue
    /// (`splitk_reduce_glu`): the activation-half and gate-half chains are
    /// both reduced in-kernel and the gate applied directly to `act` — the
    /// fused `[m × 2·i]` staging never exists (reflex issue 020 T11).
    #[allow(clippy::too_many_arguments)]
    fn run_sgemm_splitk_glu(
        &self,
        a: (&Buffer, u64),
        b: (&Buffer, u64),
        act: (&Buffer, u64),
        uargs: &[u32; 10],
        m: u32,
        i_sz: u32,
        slices: u32,
        kc: u32,
    ) -> Result<()> {
        let n = i_sz * 2;
        let (part, _mn) = self.sgemm_splitk_parts(a, b, uargs, m, n, slices, kc)?;
        let red = self
            .pipelines
            .get("splitk_reduce_glu")
            .ok_or_else(|| rt("kernel splitk_reduce_glu missing"))?;
        let width = red.width;
        let out_len = m as usize * i_sz as usize;
        self.encode(
            &red.p,
            &[(&part, 0), act],
            &[m, i_sz, n, slices],
            &[],
            MTLSize {
                width: (out_len as u64).div_ceil(width) * width,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
            None,
            false,
        )
    }

    /// The split-K PART dispatch shared by every epilogue: `slices`
    /// k-slices of `kc` into the partial scratch; returns the part buffer
    /// and the per-slice element count `m·n`.
    #[allow(clippy::too_many_arguments)]
    fn sgemm_splitk_parts(
        &self,
        a: (&Buffer, u64),
        b: (&Buffer, u64),
        uargs: &[u32; 10],
        m: u32,
        n: u32,
        slices: u32,
        kc: u32,
    ) -> Result<(Buffer, usize)> {
        self.splitk_count.fetch_add(1, Ordering::Relaxed);
        let mn = m as usize * n as usize;
        let part = self.splitk_buf(mn * slices as usize);
        let kern = self
            .pipelines
            .get("sgemm_splitk")
            .ok_or_else(|| rt("kernel sgemm_splitk missing"))?;
        self.encode(
            &kern.p,
            &[a, b, (&part, 0)],
            &[
                uargs[0], uargs[1], uargs[2], uargs[3], uargs[4], uargs[5], uargs[6], kc,
            ],
            &[],
            MTLSize {
                width: u64::from(n.div_ceil(64)),
                height: u64::from(m).div_ceil(32),
                depth: u64::from(slices),
            },
            MTLSize {
                width: NARROW_THREADS,
                height: 1,
                depth: 1,
            },
            Some(&[(13, NARROW_STAGING_BYTES)]),
            true,
        )?;
        Ok((part, mn))
    }

    /// The batched simdgroup GEMM dispatch: grid = (⌈n/BN⌉, ⌈m/BM⌉, batch),
    /// the instance picked by the single-wave BAND rule (see
    /// [`WAVE_TG_LIMIT`]): xwide only inside the band (grid mostly-fills but
    /// does not spill one wave) with enough k to amortize the staging;
    /// narrow otherwise. `uargs` =
    /// [m, n, k, `a_rs`, `a_cs`, `b_rs`, `b_cs`, `a_bs`, `b_bs`, `c_bs`] — element
    /// strides, then per-batch element strides (batch-1 callers pass
    /// zeros).
    #[allow(clippy::too_many_arguments)]
    fn run_sgemm(
        &self,
        a: (&Buffer, u64),
        b: (&Buffer, u64),
        out: (&Buffer, u64),
        uargs: &[u32; 10],
        m: u32,
        n: u32,
        batch: u32,
    ) -> Result<()> {
        if batch != 1 {
            return self.run_sgemm_one(a, b, out, uargs, m, n, batch, false);
        }
        let plan = self.split_plan(m, n, uargs[2]);
        if let [(_, _, split)] = plan.as_slice() {
            return self.run_sgemm_one(a, b, out, uargs, m, n, 1, *split);
        }
        // Mixed decisions across a packed pass's rows: one dispatch per run
        // of equal decisions, at row offsets (A row stride `a_rs`, C row
        // stride n). Every instance is row-local, so each row gets exactly
        // the bits the per-question loop gives it.
        for (row0, rows, split) in plan {
            let a_at = (a.0, a.1 + u64::from(row0) * u64::from(uargs[3]) * 4);
            let o_at = (out.0, out.1 + u64::from(row0) * u64::from(n) * 4);
            let mut u = *uargs;
            u[0] = rows;
            self.run_sgemm_one(a_at, b, o_at, &u, rows, n, 1, split)?;
        }
        Ok(())
    }

    /// The per-row split-K plan for a batch-1 GEMM of `m` rows: runs of
    /// `(row0, rows, split)`. Decided PER SEGMENT of the pass's row
    /// segmentation ([`Backend::set_row_segments`] — the packed forward's
    /// per-question seqs; one segment `[m]` otherwise, or when the hint does
    /// not describe this call), with the loop's own rule for a segment of
    /// that many rows — so a packed row takes the kernel the per-question
    /// loop takes for it, and the two paths stay bit-identical.
    fn split_plan(&self, m: u32, n: u32, k: u32) -> Vec<(u32, u32, bool)> {
        let segs = self.row_segments.lock().expect("row segments poison");
        let hinted =
            segs.len() > 1 && segs.iter().map(|&r| u64::from(r)).sum::<u64>() == u64::from(m);
        let decide = |rows: u32| self.splitk_slices(rows, n, k).is_some();
        if !hinted {
            return vec![(0, m, decide(m))];
        }
        let mut plan: Vec<(u32, u32, bool)> = Vec::with_capacity(segs.len());
        let mut row0 = 0u32;
        for &rows in segs.iter() {
            let split = decide(rows);
            match plan.last_mut() {
                Some(last) if last.2 == split => last.1 += rows,
                _ => plan.push((row0, rows, split)),
            }
            row0 += rows;
        }
        plan
    }

    /// One GEMM dispatch with the split decision already made: split-K when
    /// `split`, else the instance the single-wave band picks (every unsplit
    /// instance is result-identical, so that pick never changes bits).
    #[allow(clippy::too_many_arguments)]
    fn run_sgemm_one(
        &self,
        a: (&Buffer, u64),
        b: (&Buffer, u64),
        out: (&Buffer, u64),
        uargs: &[u32; 10],
        m: u32,
        n: u32,
        batch: u32,
        split: bool,
    ) -> Result<()> {
        if split {
            return self.run_sgemm_splitk(
                a,
                b,
                out,
                uargs,
                m,
                n,
                uargs[2].div_ceil(SPLITK_KC),
                SPLITK_KC,
            );
        }
        // The MPS arm (reflex issue 020 T13): unsplit, batch-1, both
        // operands dense-row (`a_cs == b_cs == 1`), C dense `[m, n]`.
        if let Some(mps) = &self.mps
            && batch == 1
            && uargs[4] == 1
            && uargs[6] == 1
            && m >= self.mps_min_m
        {
            let k = uargs[2];
            return self.run_sgemm_mps(
                mps,
                (a.0, a.1, m, k, uargs[3]),
                (b.0, b.1, k, n, uargs[5]),
                (out.0, out.1, m, n, n),
            );
        }
        let rows = u64::from(m).div_ceil(64);
        let cols = u64::from(n).div_ceil(128);
        let tgs = rows * cols * u64::from(batch);
        let k = uargs[2]; // uargs = [m, n, k, …] — see the doc above
        let (name, bm, bn, staging, threads) =
            if tgs > WAVE_TG_FLOOR && tgs <= WAVE_TG_LIMIT && k >= XWAVE_K_MIN {
                (
                    "sgemm_xwide",
                    64u64,
                    128u64,
                    XWIDE_STAGING_BYTES,
                    XWIDE_THREADS,
                )
            } else {
                ("sgemm", 32, 64, NARROW_STAGING_BYTES, NARROW_THREADS)
            };
        let k = self
            .pipelines
            .get(name)
            .ok_or_else(|| rt(format!("kernel {name} missing")))?;
        self.encode(
            &k.p,
            &[a, b, out],
            uargs,
            &[],
            MTLSize {
                width: u64::from(n.div_ceil(bn as u32)),
                height: u64::from(m).div_ceil(bm),
                depth: u64::from(batch),
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
            Some(&[(13, staging)]),
            true,
        )
    }
}

/// Classification of every backend arg (the lazy-sync correctness
/// contract, argued in the module doc):
/// - `weight_buf` — agent-owned, stable for the agent's lifetime
///   (`matmul_w` weights, LN scales, biases, the embedding table);
/// - `chain_buf` — activations + per-forward host-authored inputs, always
///   the WHOLE parent buffer (per-head access goes through explicit offset
///   args, so one slot per parent and serial GPU ordering makes every hit
///   device-current);
/// - `chain_slot_for` — destinations (one slot per parent extent per epoch,
///   results stay device-resident until `download_into`);
/// - everything else (`rows` in gather) — fresh upload per call.
#[allow(clippy::too_many_arguments)]
impl Backend for Metal {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn set_row_segments(&self, segs: &[usize]) {
        let mut g = self.row_segments.lock().expect("row segments poison");
        g.clear();
        g.extend(segs.iter().map(|&r| r as u32));
    }

    fn matmul(
        &self,
        a: &[f32],
        a_off: usize,
        m: usize,
        k: usize,
        b: &[f32],
        b_off: usize,
        n: usize,
        dst: &mut [f32],
        dst_off: usize,
    ) {
        assert!(a.len() >= m * k + a_off, "lhs extent");
        assert!(b.len() >= k * n + b_off, "rhs extent");
        assert!(dst.len() >= m * n + dst_off, "dst extent");
        let ab = self.chain_buf(a);
        // b is an ACTIVATION here (the probs@v rhs) — the chain class.
        let bb = self.chain_buf(b);
        let ob = self.chain_slot_for(dst);
        self.run_sgemm(
            (&ab, (a_off * 4) as u64),
            (&bb, (b_off * 4) as u64),
            (&ob, (dst_off * 4) as u64),
            &[
                m as u32, n as u32, k as u32, k as u32, // a_rs
                1,        // a_cs
                n as u32, // b_rs
                1,        // b_cs
                0,        // batch strides (batch = 1)
                0, 0,
            ],
            m as u32,
            n as u32,
            1,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&ob, dst);
    }

    fn matmul_w(&self, a: &[f32], m: usize, k: usize, w: &[f32], n: usize, dst: &mut [f32]) {
        assert_eq!(a.len(), m * k, "lhs extent");
        assert_eq!(w.len(), n * k, "weight extent");
        assert_eq!(dst.len(), m * n, "dst extent");
        let ab = self.chain_buf(a);
        // riir-reflex Issue 020 T4: bind the TRANSPOSE, row-major [k, n], so the
        // staging takes the kernel's coalesced `b_cs == 1` branch. The
        // staged tile is the same [K][N] block of the same values either
        // way — the k-accumulation order is untouched, so the result is
        // bit-identical to the `b_cs = k` binding.
        let wb = self.weight_t_buf(w, n, k);
        let ob = self.chain_slot_for(dst);
        self.run_sgemm(
            (&ab, 0),
            (&wb, 0),
            (&ob, 0),
            &[
                m as u32, n as u32, k as u32, k as u32, // a_rs
                1,        // a_cs
                n as u32, // b_rs — B = Wᵀ held row-major [k, n]
                1,        // b_cs
                0,        // batch strides (batch = 1)
                0, 0,
            ],
            m as u32,
            n as u32,
            1,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&ob, dst);
    }

    /// The residual-stream projection (reflex issue 020 T11, the last open
    /// rung): `x += a @ wᵀ` in ONE op. Fold arm (default ON since the A/B
    /// promotion; `LAYA_METAL_FOLD_RES=0` restores the unfused stream):
    /// when the whole call splits, the split-K reduce applies the residual
    /// directly
    /// (`splitk_reduce_add`) — the staging write+read and the add dispatch
    /// never happen. Bit-identical to [`Backend::matmul_w`] +
    /// [`Backend::add`] by construction: same slice chains, then one add
    /// (IEEE addition commutes), so G5 carries the fold at zero drift.
    /// Knob-off and mixed-plan (a packed call whose row segments disagree)
    /// run the unfused stream staged on [`Self::fold_stage_buf`].
    fn matmul_w_accum(&self, a: &[f32], m: usize, k: usize, w: &[f32], n: usize, x: &mut [f32]) {
        assert_eq!(a.len(), m * k, "lhs extent");
        assert_eq!(w.len(), n * k, "weight extent");
        assert_eq!(x.len(), m * n, "residual extent");
        if self.fold_res {
            let plan = self.split_plan(m as u32, n as u32, k as u32);
            if plan.iter().all(|(_, _, s)| *s) {
                let ab = self.chain_buf(a);
                let wb = self.weight_t_buf(w, n, k);
                // Read-modify-write target: the residual stream is
                // device-current this epoch (the LayerNorm before it wrote
                // the slot), so `chain_buf` — never the write-first slot.
                let xb = self.chain_buf(x);
                for (row0, rows, _) in &plan {
                    let u = [
                        *rows, n as u32, k as u32, k as u32, // a_rs
                        1,        // a_cs
                        n as u32, // b_rs — Wᵀ row-major [k, n]
                        1,        // b_cs
                        0,        // batch strides (batch = 1)
                        0, 0,
                    ];
                    self.run_sgemm_splitk_accum(
                        (&ab, ((*row0 as usize) * k * 4) as u64),
                        (&wb, 0),
                        (&xb, ((*row0 as usize) * n * 4) as u64),
                        &u,
                        *rows,
                        n as u32,
                        (k as u32).div_ceil(SPLITK_KC),
                        SPLITK_KC,
                    )
                    .unwrap_or_else(|e| panic!("{e}"));
                    self.fold_count.fetch_add(1, Ordering::Relaxed);
                }
                self.debug_writeback(&xb, x);
                return;
            }
        }
        self.matmul_w_then_add(a, m, k, w, n, x);
    }

    /// The MLP-up projection with its GLU epilogue folded (reflex issue
    /// 020 T11): `act = glu_gelu_gate(a @ wᵀ)` in ONE op. Fold arm
    /// (default ON since the A/B promotion; `LAYA_METAL_FOLD_GLU=0`
    /// restores the unfused stream): when the whole call splits,
    /// `splitk_reduce_glu` reduces BOTH halves in-kernel and applies the
    /// gate — the fused `[m × 2i]` staging
    /// round-trip never happens. Bit-identical to [`Backend::matmul_w`] +
    /// [`Backend::glu_gelu_gate`] by construction (same chains, the glu
    /// kernel's own expression order).
    fn matmul_w_glu(&self, a: &[f32], m: usize, k: usize, w: &[f32], i_sz: usize, act: &mut [f32]) {
        let n = i_sz * 2;
        assert_eq!(a.len(), m * k, "lhs extent");
        assert_eq!(w.len(), n * k, "weight extent");
        assert_eq!(act.len(), m * i_sz, "glu out extent");
        if self.fold_glu {
            let plan = self.split_plan(m as u32, n as u32, k as u32);
            if plan.iter().all(|(_, _, s)| *s) {
                let ab = self.chain_buf(a);
                let wb = self.weight_t_buf(w, n, k);
                let ob = self.chain_slot_for(act);
                for (row0, rows, _) in &plan {
                    let u = [
                        *rows, n as u32, k as u32, k as u32, // a_rs
                        1,        // a_cs
                        n as u32, // b_rs — Wᵀ row-major [k, n]
                        1,        // b_cs
                        0,        // batch strides (batch = 1)
                        0, 0,
                    ];
                    self.run_sgemm_splitk_glu(
                        (&ab, ((*row0 as usize) * k * 4) as u64),
                        (&wb, 0),
                        (&ob, ((*row0 as usize) * i_sz * 4) as u64),
                        &u,
                        *rows,
                        i_sz as u32,
                        (k as u32).div_ceil(SPLITK_KC),
                        SPLITK_KC,
                    )
                    .unwrap_or_else(|e| panic!("{e}"));
                    self.fold_count.fetch_add(1, Ordering::Relaxed);
                }
                self.debug_writeback(&ob, act);
                return;
            }
        }
        self.matmul_w_then_glu(a, m, k, w, i_sz, act);
    }

    fn matmul_kt(
        &self,
        q: &[f32],
        q_off: usize,
        m: usize,
        hd: usize,
        k: &[f32],
        k_off: usize,
        dst: &mut [f32],
        dst_off: usize,
    ) {
        assert!(q.len() >= m * hd + q_off, "q extent");
        assert!(k.len() >= m * hd + k_off, "k extent");
        assert!(dst.len() >= m * m + dst_off, "dst extent");
        let qb = self.chain_buf(q);
        // k is an ACTIVATION here (the per-head K matrix) — the chain class.
        let kb = self.chain_buf(k);
        let ob = self.chain_slot_for(dst);
        self.run_sgemm(
            (&qb, (q_off * 4) as u64),
            (&kb, (k_off * 4) as u64),
            (&ob, (dst_off * 4) as u64),
            &[
                m as u32, m as u32, hd as u32, hd as u32, // a_rs
                1,         // a_cs
                1,         // b_rs — B = Kᵀ, K row-major [m, hd]
                hd as u32, // b_cs
                0,         // batch strides (batch = 1)
                0, 0,
            ],
            m as u32,
            m as u32,
            1,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&ob, dst);
    }

    /// The fused attention block: ONE dispatch per layer over the packed
    /// qkv (see [`MSL_FLASH`]) — the split/rope/scale/scores/mask/softmax/
    /// value-mix/merge sequence and its seq² scores parent collapse into a
    /// single kernel that walks only each query block's windowed key slice.
    /// The offsets (reflex issue 020 T5's packed forward) bind at dispatch:
    /// `qkv_off`/`out_off` shift the qkv/out binds, `rope_row` shifts the
    /// cos/sin binds — the kernel sees an unpadded `[seq, …]` slab with
    /// local rows either way, so a packed dispatch is the unbatched
    /// kernel's exact math.
    /// `LAYA_METAL_FLASH=0` falls back to the reference sequence — which
    /// cannot bind offsets into a device buffer — so non-zero offsets
    /// there are a LOUD contract breach (the agent gates packed forwards
    /// on [`Backend::supports_packed_attention`]; this guard is the
    /// backstop).
    #[allow(clippy::too_many_arguments)]
    fn attention_forward(
        &self,
        qkv: &[f32],
        qkv_off: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
        rope_row: usize,
        scale: f32,
        seq: usize,
        heads: usize,
        hd: usize,
        window: usize,
        mask: Option<&[f32]>,
        scratch: &mut AttnScratch,
        out: &mut [f32],
        out_off: usize,
    ) {
        if self.flash_disabled || hd != FLASH_HD {
            if qkv_off != 0 || out_off != 0 || rope_row != 0 {
                panic!(
                    "packed attention offsets (qkv {qkv_off}, rope row {rope_row}, out \
                     {out_off}) reached the reference-sequence fallback — the fallback \
                     cannot bind offsets into a device buffer; the packed forward \
                     requires the fused kernel (supports_packed_attention gates it)"
                );
            }
            // The reference sequence consumes the mask tensor; the fused
            // kernel predicates on the window (the same allowed set). hd
            // outside the pinned geometry takes the reference path too.
            return self.attention_forward_default(
                qkv, qkv_off, rope_cos, rope_sin, rope_row, scale, seq, heads, hd, window, mask,
                scratch, out, out_off,
            );
        }
        let _ = mask; // the window describes the same allowed set
        let qb = self.chain_buf(qkv);
        // Rope tables: ONE slot per forward (whole packed tables), offset
        // per sequence at bind — never a per-sequence re-upload.
        let cb = self.chain_buf(rope_cos);
        let sb = self.chain_buf(rope_sin);
        let ob = self.chain_slot_for(out);
        // window clamped to seq: full attention ⇒ lo 0 / hi seq in-kernel
        // (and no u32 overflow in the key-range arithmetic).
        let w = window.min(seq) as u32;
        let d = heads * hd;
        // Opt-in rope hoist (issue 020 T10 rung 2): derive the Q/K rope ONCE
        // per layer into the scratch, then flash_attn's staging copies.
        // Off: the scratch bind is a harmless placeholder (the kernel's
        // `use_pre` arm is dead) and the staging rotates in place — the
        // shipped behavior, bit-identical.
        let use_pre = self.rope_hoist;
        let rk = if use_pre {
            self.rope_hoist_buf(seq, d)
        } else {
            qb.clone()
        };
        if use_pre {
            let rope_k = self
                .pipelines
                .get("attn_rope")
                .ok_or_else(|| rt("kernel attn_rope missing"))
                .unwrap_or_else(|e| panic!("{e}"));
            // One thread per rope pair: seq · d/2 threads, d even at the
            // pinned hd = 64. Same `run`-shaped grid convention.
            let pairs = ((seq * d) / 2) as u64;
            let twidth = rope_k.width.max(1);
            self.encode(
                &rope_k.p,
                &[
                    (&qb, (qkv_off * 4) as u64),
                    (&cb, (rope_row * hd * 4) as u64),
                    (&sb, (rope_row * hd * 4) as u64),
                    (&rk, 0),
                ],
                &[seq as u32, d as u32],
                &[scale],
                MTLSize {
                    width: pairs.div_ceil(twidth).max(1),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: twidth,
                    height: 1,
                    depth: 1,
                },
                None,
                true,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        }
        let k = self
            .pipelines
            .get("flash_attn")
            .ok_or_else(|| rt("kernel flash_attn missing"))
            .unwrap_or_else(|e| panic!("{e}"));
        self.encode(
            &k.p,
            &[
                (&qb, (qkv_off * 4) as u64),
                (&rk, 0),
                (&cb, (rope_row * hd * 4) as u64),
                (&sb, (rope_row * hd * 4) as u64),
                (&ob, (out_off * 4) as u64),
            ],
            &[seq as u32, heads as u32, hd as u32, w, u32::from(use_pre)],
            &[scale],
            MTLSize {
                width: u64::from(seq as u32).div_ceil(FLASH_BQ),
                height: u64::from(heads as u32),
                depth: 1,
            },
            MTLSize {
                width: FLASH_THREADS,
                height: 1,
                depth: 1,
            },
            Some(&[(11, FLASH_STAGING_BYTES)]),
            true,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&ob, out);
    }

    /// scores[h] ← q[h]·k[h]ᵀ for every head in ONE dispatch.
    fn matmul_kt_heads(
        &self,
        q: &[f32],
        k: &[f32],
        heads: usize,
        m: usize,
        hd: usize,
        dst: &mut [f32],
    ) {
        assert_eq!(q.len(), heads * m * hd, "q extent");
        assert_eq!(k.len(), heads * m * hd, "k extent");
        assert_eq!(dst.len(), heads * m * m, "dst extent");
        let qb = self.chain_buf(q);
        let kb = self.chain_buf(k);
        let ob = self.chain_slot_for(dst);
        self.run_sgemm(
            (&qb, 0),
            (&kb, 0),
            (&ob, 0),
            &[
                m as u32,
                m as u32,
                hd as u32,
                hd as u32,       // a_rs
                1,               // a_cs
                1,               // b_rs — B = Kᵀ, K row-major [m, hd]
                hd as u32,       // b_cs
                (m * hd) as u32, // a_bs
                (m * hd) as u32, // b_bs
                (m * m) as u32,  // c_bs
            ],
            m as u32,
            m as u32,
            heads as u32,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&ob, dst);
    }

    /// ctx[h] ← probs[h] @ v[h] for every head in ONE dispatch.
    fn matmul_heads(
        &self,
        a: &[f32],
        b: &[f32],
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
        dst: &mut [f32],
    ) {
        assert_eq!(a.len(), heads * m * k, "a extent");
        assert_eq!(b.len(), heads * k * n, "b extent");
        assert_eq!(dst.len(), heads * m * n, "dst extent");
        let ab = self.chain_buf(a);
        let bb = self.chain_buf(b);
        let ob = self.chain_slot_for(dst);
        self.run_sgemm(
            (&ab, 0),
            (&bb, 0),
            (&ob, 0),
            &[
                m as u32,
                n as u32,
                k as u32,
                k as u32,       // a_rs
                1,              // a_cs
                n as u32,       // b_rs
                1,              // b_cs
                (m * k) as u32, // a_bs
                (k * n) as u32, // b_bs
                (m * n) as u32, // c_bs
            ],
            m as u32,
            n as u32,
            heads as u32,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&ob, dst);
    }

    fn add(&self, x: &mut [f32], x_off: usize, y: &[f32], y_off: usize, len: usize) {
        assert!(x.len() >= len + x_off, "add x extent");
        assert!(y.len() >= len + y_off, "add y extent");
        let xb = self.chain_buf(x);
        // y: chain either way — attn_out (device-current) or the per-forward
        // mask (fresh this epoch, hit on later layers).
        let yb = self.chain_buf(y);
        self.run_at(
            "add",
            (&xb, (x_off * 4) as u64),
            (&yb, (y_off * 4) as u64),
            &[len as u32],
            &[],
            len as u64,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&xb, x);
    }

    /// scores[r] += mask[r % mlen] over the whole heads·seq² parent — the
    /// sliding-window mask for every head in ONE dispatch.
    fn add_mask_broadcast(&self, x: &mut [f32], mask: &[f32], heads: usize) {
        let mlen = mask.len();
        assert!(heads > 0, "heads");
        assert_eq!(x.len(), heads * mlen, "mask broadcast extent");
        let xb = self.chain_buf(x);
        let mb = self.chain_buf(mask);
        let len = x.len();
        self.run_at(
            "add_mask_bcast",
            (&xb, 0),
            (&mb, 0),
            &[len as u32, mlen as u32],
            &[],
            len as u64,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&xb, x);
    }

    fn add_bias_row(&self, x: &mut [f32], d: usize, bias: &[f32]) {
        assert_eq!(bias.len(), d, "bias extent");
        assert_eq!(x.len() % d, 0, "row extent");
        let xb = self.chain_buf(x);
        let bb = self.weight_buf(bias);
        self.run(
            "add_bias_row",
            &[&xb, &bb],
            &[x.len() as u32, d as u32],
            &[],
            x.len() as u64,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&xb, x);
    }

    fn scale(&self, x: &mut [f32], s: f32) {
        let xb = self.chain_buf(x);
        self.run("scale", &[&xb], &[x.len() as u32], &[s], x.len() as u64)
            .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&xb, x);
    }

    fn layer_norm_nobias_into(
        &self,
        x: &[f32],
        w: &[f32],
        eps: f32,
        d: usize,
        _sq: &mut Vec<f32>,
        out: &mut [f32],
    ) {
        assert_eq!(w.len(), d, "norm extent");
        assert_eq!(x.len(), out.len(), "ln extent");
        assert_eq!(x.len() % d, 0, "row extent");
        // The CPU lane's exact mean scale: f64 1/d rounded to f32.
        let inv_d = (1f64 / d as f64) as f32;
        let rows = x.len() / d;
        let xb = self.chain_buf(x);
        let wb = self.weight_buf(w);
        let ob = self.chain_slot_for(out);
        if self.ln_wide && d <= LN_WIDE_MAX_D {
            // reflex issue 020 T11: 256 threads per row, row in registers.
            let k = self
                .pipelines
                .get("ln_rows_wide")
                .unwrap_or_else(|| panic!("kernel ln_rows_wide missing"));
            self.encode(
                &k.p,
                &[(&xb, 0), (&wb, 0), (&ob, 0)],
                &[rows as u32, d as u32],
                &[inv_d, eps],
                MTLSize {
                    width: (rows as u64).max(1),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
                None,
                true,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        } else {
            self.run_rows(
                "ln_rows",
                &[&xb, &wb, &ob],
                &[rows as u32, d as u32],
                &[inv_d, eps],
                rows as u64,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        }
        self.debug_writeback(&ob, out);
    }

    fn softmax_rows(&self, x: &mut [f32], n: usize) {
        assert_eq!(x.len() % n, 0, "softmax row extent");
        let rows = x.len() / n;
        let xb = self.chain_buf(x);
        self.run_rows(
            "softmax_rows",
            &[&xb],
            &[rows as u32, n as u32],
            &[],
            rows as u64,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&xb, x);
    }

    fn relu(&self, x: &mut [f32]) {
        let xb = self.chain_buf(x);
        self.run("relu", &[&xb], &[x.len() as u32], &[], x.len() as u64)
            .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&xb, x);
    }

    fn gelu_erf(&self, x: &mut [f32]) {
        let xb = self.chain_buf(x);
        self.run("gelu_erf", &[&xb], &[x.len() as u32], &[], x.len() as u64)
            .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&xb, x);
    }

    fn glu_gelu_gate(&self, fused: &[f32], rows: usize, i_sz: usize, out: &mut [f32]) {
        assert_eq!(fused.len(), rows * 2 * i_sz, "fused extent");
        assert_eq!(out.len(), rows * i_sz, "glu out extent");
        let fb = self.chain_buf(fused);
        let ob = self.chain_slot_for(out);
        self.run(
            "glu_gelu_gate",
            &[&fb, &ob],
            &[rows as u32, i_sz as u32],
            &[],
            (rows * i_sz) as u64,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&ob, out);
    }

    fn apply_rope(
        &self,
        q: &mut [f32],
        seq: usize,
        heads: usize,
        hd: usize,
        cos: &[f32],
        sin: &[f32],
    ) {
        let half = hd / 2;
        assert_eq!(q.len(), heads * seq * hd, "q extent");
        assert_eq!(cos.len(), seq * hd, "cos extent");
        assert_eq!(sin.len(), seq * hd, "sin extent");
        let qb = self.chain_buf(q);
        // cos/sin: rebuilt fresh per forward → fresh this epoch; hit on the
        // layer's second call and on every later layer.
        let cb = self.chain_buf(cos);
        let sb = self.chain_buf(sin);
        self.run(
            "rope",
            &[&qb, &cb, &sb],
            &[seq as u32, heads as u32, hd as u32],
            &[],
            (heads * seq * half) as u64,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&qb, q);
    }

    fn split_heads(
        &self,
        src: &[f32],
        row_stride: usize,
        off: usize,
        seq: usize,
        heads: usize,
        hd: usize,
        out: &mut [f32],
    ) {
        assert_eq!(out.len(), heads * seq * hd, "split extent");
        let sb = self.chain_buf(src);
        let ob = self.chain_slot_for(out);
        self.run(
            "split_heads",
            &[&sb, &ob],
            &[
                row_stride as u32,
                off as u32,
                seq as u32,
                heads as u32,
                hd as u32,
            ],
            &[],
            (heads * seq * hd) as u64,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&ob, out);
    }

    fn merge_heads(&self, src: &[f32], seq: usize, heads: usize, hd: usize, out: &mut [f32]) {
        let d = heads * hd;
        assert_eq!(out.len(), seq * d, "merge extent");
        let sb = self.chain_buf(src);
        let ob = self.chain_slot_for(out);
        self.run(
            "merge_heads",
            &[&sb, &ob],
            &[seq as u32, heads as u32, hd as u32],
            &[],
            (seq * d) as u64,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&ob, out);
    }

    fn gather_rows(&self, x: &[f32], d: usize, rows: &[usize], out: &mut [f32]) {
        assert_eq!(out.len(), rows.len() * d, "gather extent");
        // x is an ACTIVATION (the head's hidden state) — the chain class
        // (the embedding gather is host-side in the encoder, so no
        // activation-sized weight ever lands in the permanent map).
        let xb = self.chain_buf(x);
        let u32s: Vec<u32> = rows.iter().map(|&r| r as u32).collect();
        let rb = self.upload_u32(&u32s);
        let ob = self.chain_slot_for(out);
        self.run(
            "gather_rows",
            &[&xb, &rb, &ob],
            &[d as u32],
            &[],
            out.len() as u64,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&ob, out);
    }

    fn copy_into(&self, src: &[f32], dst: &mut [f32]) {
        assert_eq!(src.len(), dst.len(), "copy extent");
        let sb = self.chain_buf(src);
        let db = self.chain_slot_for(dst);
        self.run(
            "copy",
            &[&db, &sb],
            &[dst.len() as u32],
            &[],
            dst.len() as u64,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    }

    fn copy_at(&self, src: &[f32], src_off: usize, dst: &mut [f32], dst_off: usize, len: usize) {
        assert!(src.len() >= len + src_off, "copy_at src extent");
        assert!(dst.len() >= len + dst_off, "copy_at dst extent");
        let sb = self.chain_buf(src);
        let db = self.chain_slot_for(dst);
        self.run_at(
            "copy",
            (&db, (dst_off * 4) as u64),
            (&sb, (src_off * 4) as u64),
            &[len as u32],
            &[],
            len as u64,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        self.debug_writeback(&db, dst);
    }

    /// The packed forward rides the fused kernel: offsets are dispatch
    /// binds, never kernel shapes. The reference-sequence fallback cannot
    /// make that claim (see [`Backend::attention_forward`]'s guard), so
    /// this answers for the fused path only — the agent falls back to the
    /// per-question loop when this is false.
    fn supports_packed_attention(&self, hd: usize) -> bool {
        !self.flash_disabled && hd == FLASH_HD
    }

    fn begin_pass(&self) {
        self.begin_pass_impl();
    }

    /// The fused kernel predicates on `window`; it consumes no mask tensor
    /// (`attention_forward` is literally `let _ = mask;`). The predicate
    /// MIRRORS that dispatch's own gate, so a posture that falls back to
    /// the reference op sequence — `LAYA_METAL_FLASH=0`, or an `hd` outside
    /// the pinned geometry — still gets its mask built.
    fn needs_window_mask(&self, hd: usize) -> bool {
        self.flash_disabled || hd != FLASH_HD
    }

    /// riir-reflex Issue 020 T1: take the host→device copy at load, not on the
    /// first forward. `weight_buf` is the permanent agent-lifetime cache
    /// keyed by `(ptr, len)`; warming it is a first-miss upload made early.
    fn warm_weight(&self, data: &[f32]) {
        if data.is_empty() {
            return;
        }
        let _ = self.weight_buf(data);
    }

    /// Build + upload `Wᵀ` at load. The untransposed form is deliberately
    /// NOT uploaded: `matmul_w` is the only consumer of these slices, so a
    /// second device copy would be dead bytes.
    fn warm_weight_2d(&self, data: &[f32], n: usize, k: usize) {
        if data.is_empty() {
            return;
        }
        let _ = self.weight_t_buf(data, n, k);
    }

    fn download_into(&self, src: &[f32], out: &mut [f32]) {
        assert!(src.len() <= out.len(), "download extent");
        self.sync();
        let ptr = src.as_ptr() as usize;
        let map = self.chain.lock().expect("chain cache poison");
        // The src may be a PREFIX of the written buffer (the CLS row is the
        // leading d of the whole hidden slot), so match by base pointer and
        // sufficient extent. Within one epoch several keys can share a base
        // pointer (a forward's scratch frees its host Vecs mid-case while
        // their keys stay in the map, and a later allocation reuses the
        // address), so the pick is newest epoch, then the most recently
        // TOUCHED entry — the buffer the program last produced or consumed
        // at that address. A dead key is by definition untouched since its
        // Vec died. The previous rule, the TIGHTEST container, picked a dead
        // key whenever one was tighter than the live buffer: reflex issue
        // 020 T11 measured it as the packed path's CLS row served from a
        // dead encoder-scratch key (act head [0.59, 0.43] vs the loop's
        // [1.0, 0.0], 7/30 processes — a heap-layout lottery the split-K
        // dispatch happened to win; its own earlier fix, from HashMap order
        // to tightest, was the same class one tie-break over).
        let cands: Vec<_> = map
            .iter()
            .filter(|(k, _)| k.0 == ptr && k.1 >= src.len())
            .collect();
        if trace_enabled() && cands.len() > 1 {
            let tight = cands
                .iter()
                .max_by(|a, b| a.0.2.cmp(&b.0.2).then_with(|| b.0.1.cmp(&a.0.1)))
                .map(|(k, _)| **k);
            let recent = cands
                .iter()
                .max_by(|a, b| a.0.2.cmp(&b.0.2).then_with(|| a.1.1.cmp(&b.1.1)))
                .map(|(k, _)| **k);
            if tight != recent {
                eprintln!(
                    "[trace] download_into {:p}: tightest {tight:?} ≠ most-recent {recent:?} ({} candidates)",
                    src.as_ptr(),
                    cands.len()
                );
            }
        }
        let slot = cands
            .iter()
            .max_by(|a, b| a.0.2.cmp(&b.0.2).then_with(|| a.1.1.cmp(&b.1.1)))
            .map(|(_, (b, _))| b.clone());
        let Some(b) = slot else {
            panic!(
                "download_into: no device buffer for this slice — host reads \
                 require a backend-produced buffer (the lazy-sync contract)"
            );
        };
        drop(map);
        Self::copy_out(&b, out);
    }
}
