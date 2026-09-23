//! Issue 734 Arm 8 — the **whole-prefill cudarc migration**: the entire
//! `prefill_tokens_chunk` pipeline on the cudarc stream with ZERO per-layer
//! host crossings (token bytes in, logits out).
//!
//! The structural context (Bench 720/721): the per-GEMM round trip measured
//! 0.095-0.234× and the per-layer FFN-block round trip 0.608-0.686× — every
//! host crossing is dead weight. This arm keeps EVERY activation on the
//! cudarc side from embedding to tail:
//!
//! ```text
//! tokens → dequant_wte_batch → x
//! └─ per layer (48 GDN + 16 attention):
//!     GDN:  norm → 4×in_proj mma → conv1d(+carry) → beta/decay → expand/L2
//!           → multi-token recurrence → per-head norm → z-gate → out_proj mma
//!           → residual
//!     ATT:  norm → wq/wkv mma → QG/KV splits → Q/K norms → RoPE → KV-fill
//!           → causal gated flash attention → wo mma → residual
//!     FFN:  norm → gate/up mma (one quantize) → swiglu → down mma → residual
//! └─ tail: ONE 20 KB crossing (the final x row) → the CubeCL final norm +
//!    GemvTernary lm_head (unproven lm_head bit-identity avoided entirely)
//! ```
//!
//! ## Persistent mirrors (cudarc side)
//!
//! - Projection weights: the arm-6 `TernaryHandle::cuda_mma_cache` lazy
//!   pattern (shared with the other arms — one mirror per weight). The
//!   Bench-720 arm-6 e2e already proved the mma GEMM bit-identical to the
//!   shipping psplit path for EVERY prefill projection on live 27B
//!   activations (FNV/argmax pins).
//! - f32 weights (norms, conv1d, a_log/dt_bias): mirrored once per process.
//! - wte: mirrored once via the same cache pattern (~220 MB).
//! - deltanet/conv states + KV caches: mirrored, synced from the CubeCL
//!   handles at every `base_pos == 0` chunk (or after any fall-through),
//!   carried on-device across chunks, and WRITTEN BACK after every chunk so
//!   each chunk is independently fall-through-safe (the CubeCL handles
//!   always hold a consistent post-chunk state).
//!
//! ## Knob
//!
//! `RIIR_PREFILL_CUDA` env ("1"/"true"/"on") or [`set_prefill_use_cuda`] —
//! explicit env wins over the setter; DEFAULT OFF. Requires the shipping
//! knob defaults (batched elementwise + chunked recurrence + batched
//! attention — checked by the caller's gate) and `p <= 4096`,
//! `head_dim == 128`, every GEMM input dim `% 128 == 0`. Any
//! init/read/alloc/launch failure falls through to the CubeCL prefill body
//! (bit-safe: both paths compute the same values).
//!
//! `RIIR_PREFILL_ATTN_MQ` (DEFAULT ON, Issue 734 Arm 9): selects the
//! multi-q attention kernel (`att_pf_mq8` — 8 q positions per block sharing
//! the K/V stream, L2 traffic /8) over the arm-8 single-q kernel. An A/B
//! knob ONLY — both kernels are probe-gated bit-identical to the SAME
//! CubeCL reference (scheduling, not numerics) — so it is deliberately not
//! part of `prefill_cuda_gate_ok`. Requires `head_dim == 256`.
//!
//! `RIIR_PREFILL_REC_MR` (DEFAULT ON — the r4w4 arm, Issue 904 / Bench 896):
//! routes the GDN rowpar recurrence through the multi-row ILP ladder
//! (`recmr_pf_r4w4`: 4 state-rows per warp × 4 warps per block — the ILP of
//! 4 independent t-serial chains hides the per-token shuffle-reduction
//! latency; kernel 2.19×/1.89× the legacy 32-thr kernel at the league
//! shapes, e2e +9.7%/+9.3% same-window, bit-identical by construction).
//! `0|off|false|legacy` restores the legacy kernel; `1..=9` selects a probe
//! arm; garbage fails OPEN to legacy. The geometry table lives in
//! `prefill_cuda_deltanet::REC_MR_GEOMETRIES` (that module's launcher owns
//! the dispatch).
//!
//! ## Graph mode (Issue 965) — DEFAULT ON
//!
//! CUDA-graph capture-once/replay-per-same-`p` chunking for the REGULAR
//! prefill path, reusing the Issue-742 T1.5 graph-verify machinery below.
//! **DEFAULT ON** (GOAT-promoted 2026-09-17, Bench 936: G1 pins bit-identical
//! across warm/capture/replay at both league shapes, the decode-continuity
//! probe green through the automatic flush hooks, pp2048 +10.9% / pp4096
//! +10.1% on the 4090). `RIIR_PREFILL_CUDA_GRAPHS=0|false|off` is the
//! kill-switch — it restores the raw-launch path verbatim. Rationale: the
//! 09-15 Windows update (KB5129195) raised WDDM per-launch submit cost
//! ~9% e2e on the league prefill rows — the only raw-launch surface left
//! (decode + the fork already replay graphs). Armed, the process adopts
//! SPEC-MODE WRITEBACK SEMANTICS (`spec = prefill_spec_mode() || graphs`):
//! per-chunk state writeback is elided (the mirrors carry on-device) and
//! `spec_dirty`/`spec_pos` mark the pending flush. The consumer wiring makes
//! the elision invisible: [`prefill_spec_flush`] runs automatically at the
//! decode funnel (`forward_dispatch_only`), the whole-prefill fall-through,
//! the hybrid-cache export, the speculative checkpoints and the tree-verify
//! entries; the state REWINDERS (the speculative rollbacks) call
//! [`prefill_spec_discard`] instead — a flush applied after a rewind would
//! resurrect the future mirror state. A prefill-only caller that resets
//! between prompts needs nothing. Capture width: `RIIR_PREFILL_GRAPH_P`
//! (lane-keyed default — the verify lane pins 16, the regular path 2048, the
//! league chunk; the pp4096 A/B sets 4096 explicitly). Chunks at any other
//! `p` run eager under the same spec semantics. Knob freeze: every knob the
//! captured layer loop consults is an env `OnceLock` (frozen at first read —
//! round 0) except [`set_prefill_gdn_chunked`], whose flip bumps the knob
//! generation and invalidates replay (stale graph → eager chunks; no
//! re-capture this process). Reset safety: `reset_state`'s memsets are
//! compute-stream ordered and the mirrored buffers are address-stable, so
//! replays after a reset re-advance the mirrors from zero — bit-identical by
//! construction (the G1 pins gate it on device).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cubecl::prelude::*;
use cubecl::server::Handle;
use cudarc::driver::safe::{CudaContext, CudaEvent, CudaGraph, CudaSlice, CudaStream, DevicePtr};

use crate::cubecl_runtime::ActiveRuntime;
use crate::gemm_ternary_i8_mma_cuda_raw::{GemmI8MmaScratch, GemmTernaryI8MmaCuda};
use crate::gemv_ternary_cubecl::TernaryHandle;
use crate::prefill_cuda_attention::{
    AttnAcc, AttnDot, AttnInv, AttnPh3, AttnRes, CudaAttnKernels, RopeRot, SinCosForm,
};
use crate::prefill_cuda_deltanet::{CudaDeltanetKernels, ConvFma, LogForm, SigDiv, SigExp};
use crate::prefill_cuda_ffn::CudaFfnKernels;
use crate::prefill_cuda_gdn_chunked::CudaGdnChunkedKernels;
use crate::prefill_cuda_mma::build_weight_cache;
use crate::ternary_deltanet_gpu_forward::TernaryDeltanetGpuForward;
use riir_infer_core::types::DeltaNetLayerType;

// ---------------------------------------------------------------------------
// Knob
// ---------------------------------------------------------------------------

static PREFILL_CUDA: AtomicBool = AtomicBool::new(false);
static GDN_CHUNKED: AtomicBool = AtomicBool::new(false);

/// Issue 965 — knob generation for the graph replay key: bumped by the one
/// setter-based knob the captured layer loop consults
/// ([`set_prefill_gdn_chunked`]); compared against the value stored at
/// capture so a mid-process flip invalidates replay (stale graph → eager
/// chunks). Env-based knobs need no generation — their `OnceLock`s freeze
/// at first read.
static GV_KNOB_GEN: AtomicUsize = AtomicUsize::new(0);

// Issue 742 T1.1 — SPEC MODE: skip the per-chunk writeback crossing. The
// T1.0 measurement (bench_742_t1_verify_cost) pinned the per-chunk fixed
// cost at ~105-115 ms (chunk-width curve intercept), dominated by the
// 414 MB mirror→CubeCL crossing (DtoH pinned slots + `client.write` at
// ~3.4 GB/s) that Arm 13 interleaves into the layer loop — invisible at
// p=2048 (hidden behind ~1.2 s of kernels) but THE wall at verify-sized
// chunks (C_verify = 10.9-14.5 vs the ≤2.0 bar). In a spec-decode verify
// loop the writeback's only consumers are absent: rollback restores from
// the DEVICE snapshot (not the writeback), and decode handoff happens once
// at loop exit. Contract: after spec chunks, the caller MUST either call
// [`prefill_spec_flush`] or run one non-spec chunk (auto-flush at the
// transition) before anything reads the CubeCL state handles (decode).
static PREFILL_SPEC: AtomicBool = AtomicBool::new(false);

fn prefill_spec_mode() -> bool {
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env = *ENV.get_or_init(|| {
        std::env::var("RIIR_PREFILL_SPEC")
            .ok()
            .map(|s| matches!(s.trim(), "1" | "true" | "on"))
    });
    env.unwrap_or_else(|| PREFILL_SPEC.load(Ordering::Relaxed))
}

/// Enable/disable spec mode (per-chunk writeback elision) — Issue 742
/// T1.1. Explicit `RIIR_PREFILL_SPEC` env wins over the setter. See the
/// module note above for the flush-before-decode contract.
pub fn set_prefill_spec_mode(on: bool) {
    PREFILL_SPEC.store(on, Ordering::Relaxed);
}

/// Replay the captured layer-loop graph `n` times and return the
/// per-replay MILLISECONDS (Issue 742 T1.2 timing probe — fixed base_pos,
/// numerically garbage by design; the timing is the datum). Returns None
/// when no graph was captured.
pub fn prefill_graph_probe_replay(n: usize) -> Option<f64> {
    let stack = full_stack()?;
    let g = stack.graph_probe.get()?.as_ref()?;
    let Ok(graph) = g.lock() else { return None };
    let graph = &graph.0;
    // Warm replay (first launch pays upload/instantiation lazily).
    graph.launch().ok()?;
    stack.stream.synchronize().ok()?;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        graph.launch().ok()?;
    }
    stack.stream.synchronize().ok()?;
    Some(t0.elapsed().as_secs_f64() * 1000.0 / n as f64)
}

/// Issue 742 T1.5 — total graph-verify replays served so far in this
/// process (0 when the stack/arm never ran). Diagnostic: proves the replay
/// path actually executed (vs silent eager fallback) in GPU runs.
pub fn prefill_gv_replay_count() -> usize {
    FULL_STACK
        .get()
        .and_then(|s| s.as_ref()).map_or(0, |s| s.gv_replays.load(Ordering::Relaxed))
}

/// Issue 967 — the capture-ladder pricing counters:
/// `(fallback chunks, fallback tokens, total tokens)` through the graph
/// lane. Fallback = an eager chunk at `p != capture_p()` with warm mirrors
/// (the exact set a second capture rung would serve); share =
/// `fallback_tokens / total_tokens`. All zeros when the graph lane never
/// armed (graphs off) — no lane, no ladder to price.
pub fn prefill_gv_fallback_counts() -> (usize, usize, usize) {
    FULL_STACK.get().and_then(|s| s.as_ref()).map_or((0, 0, 0), |s| {
        (
            s.gv_fallbacks.load(Ordering::Relaxed),
            s.gv_fallback_tokens.load(Ordering::Relaxed),
            s.gv_total_tokens.load(Ordering::Relaxed),
        )
    })
}

/// Issue 742 T1.5 — tight-loop replay timing of the graph-verify graph
/// (embed + layer loop, devpos kernels; FIXED pos/tokens — numerically
/// garbage, timing-only, the `prefill_graph_probe_replay` protocol). The
/// delta between this and the in-context per-chunk wall isolates the
/// per-chunk eager surroundings (uploads/submission batching) from the
/// graph execution itself.
pub fn prefill_gv_probe_replay(n: usize) -> Option<f64> {
    let stack = full_stack()?;
    let g = stack.graph_verify.get()?.as_ref()?;
    let Ok(graph) = g.lock() else { return None };
    graph.0.launch().ok()?;
    stack.stream.synchronize().ok()?;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        graph.0.launch().ok()?;
    }
    stack.stream.synchronize().ok()?;
    Some(t0.elapsed().as_secs_f64() * 1000.0 / n as f64)
}

/// Flush the spec-mode mirror state to the CubeCL handles (one full
/// writeback of everything the spec chunks advanced past). Returns true
/// when the CubeCL side is current (or nothing was pending). Call before
/// any consumer of the CubeCL state handles (decode) after spec chunks.
pub fn prefill_spec_flush(fwd: &TernaryDeltanetGpuForward) -> bool {
    let Some(stack) = full_stack() else {
        return true; // no arm state — nothing to flush
    };
    if !stack.spec_dirty.swap(false, Ordering::Relaxed) {
        return true;
    }
    let Some(states_lock) = stack.states.get().and_then(|s| s.as_ref()) else {
        return false;
    };
    let Ok(mut states) = states_lock.lock() else {
        return false;
    };
    let pos = stack.spec_pos.swap(0, Ordering::Relaxed);
    let kvd = fwd.config.n_kv_head * fwd.config.head_dim;
    // Quiesce the compute stream first — the writeback's DtoHs must see
    // the final mirror values.
    if stack.stream.synchronize().is_err() {
        return false;
    }
    let mut state_host = Vec::new();
    let mut kv_host = Vec::new();
    writeback_states(
        &fwd.client,
        &stack.stream,
        fwd,
        &mut states,
        0,
        pos,
        kvd,
        &mut state_host,
        &mut kv_host,
    )
}

fn prefill_use_cuda() -> bool {
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env = *ENV.get_or_init(|| {
        std::env::var("RIIR_PREFILL_CUDA")
            .ok()
            .map(|s| matches!(s.trim(), "1" | "true" | "on"))
    });
    env.unwrap_or_else(|| PREFILL_CUDA.load(Ordering::Relaxed))
}

/// Diagnostic accessor for the wiring trace (pub(crate)).
pub(crate) fn prefill_use_cuda_pub() -> bool {
    prefill_use_cuda()
}

/// Force-enable/disable the whole-prefill cudarc arm (overrides the
/// DEFAULT-OFF state; an explicit `RIIR_PREFILL_CUDA` env value wins over
/// both). Issue 734 Arm 8 A/B knob — public for the e2e benches.
pub fn set_prefill_use_cuda(on: bool) {
    PREFILL_CUDA.store(on, Ordering::Relaxed);
}

/// Per-phase timing trace (`RIIR_PREFILL_CUDA_TRACE=1`; `=2` additionally
/// syncs + accumulates per kernel class — distorts total, localizes
/// attribution).
fn trace_enabled() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| {
        std::env::var("RIIR_PREFILL_CUDA_TRACE").is_ok_and(|s| matches!(s.trim(), "1" | "2" | "true" | "on"))
    })
}

/// Multi-q attention kernel selection (Issue 734 Arm 9, DEFAULT ON): one
/// block processes 8 q positions sharing the K/V stream (L2 traffic /8).
/// `RIIR_PREFILL_ATTN_MQ=0|false|off` selects the arm-8 single-q kernel —
/// an A/B knob only: BOTH kernels are proven bit-identical to the SAME
/// CubeCL reference (the probe gates both), so this knob does NOT change
/// numerics and is deliberately NOT part of `prefill_cuda_gate_ok`'s
/// shipping-defaults set. Requires `head_dim == 256` (falls back to the
/// single-q kernel otherwise).
fn attention_mq_enabled() -> bool {
    static MQ: OnceLock<bool> = OnceLock::new();
    *MQ.get_or_init(|| {
        std::env::var("RIIR_PREFILL_ATTN_MQ").map_or(true, |s| !matches!(s.trim(), "0" | "false" | "off"))
    })
}

/// Issue 742 T1.6 — the mq8 attention arm selection (arms 0-2 are
/// bit-identical to the serial kernel — gate-pinned by
/// `bench_742_t1_attn_staged_g1`):
/// - default: `att_pf_mq8p` (L2 prefetch of tile t+1's K/V + unroll-4 —
///   the serial kernel is DRAM-latency-bound at long ctx, ~6 GB/s per
///   block measured)
/// - `QWEN38_PF_ATTN_STAGED=1`: `att_pf_mq8s` (the K-staging restructure;
///   MEASURED NEGATIVE 0.89x — kept as the A/B artifact)
/// - `QWEN38_PF_ATTN_PF=0` (without STAGED=1): the serial `att_pf_mq8`.
/// - Issue 742 T1.7 — `QWEN38_PF_ATTN_SPLITKV=1`: the split-KV arm
///   (`att_pf_mq8kv` partial + combine), engaged when
///   `n_head*ceil(p/8) <= 128` (the small-p verify regime); falls back to
///   the prefetch arm otherwise. TOLERANCE-CLASS (max_rel <= 1e-5 vs
///   serial — the cross-chunk merge reassociates the cascade's rounding),
///   hence OPT-IN: default-on needs the one-time re-pin of the path-level
///   bit-identity pins + the Issue-734 sibling's sign-off.
///   Issue 884 lever B extends the cap: `QWEN38_PF_ATTN_SPLIT_QTILES`
///   (default 128 — see [`attn_split_qtile_cap`]) lets long-p prefill
///   engage the split where the latency chain, not block count, is the
///   wall (Bench 889: attn 3.26x per 2x tokens). MEASURED NEGATIVE at
///   bp=0 prefill (Bench 890: the partial-out traffic term) — A/B artifact.
/// - Issue 898 — `QWEN38_PF_ATTN_GANG` (**DEFAULT ON** since Bench 892;
///   Issue 899 re-based the default rung + fixed the latent race (c)):
///   the head-ganged kv_group family — one block-set per (kv_group,
///   q-tile) serving all 6 heads; K/V read once per gang instead of 6x.
///   The Issue-899 occupancy ladder picks the rung ([`attn_gang_layout`]):
///   DEFAULT `g3` (24 rows/block, 2 blocks/SM — the Bench-893 G2 winner,
///   3.4%/4.2% faster than the 898 full gang at pp4096/pp2048); `GANG=6`
///   the 898 full gang; `GANG=2` the measured-negative third-gang (A/B
///   artifact). ALL rungs BIT-IDENTICAL to the vec kernel (gate-pinned by
///   `bench_898_attn_gang_g1`, 13 fixtures × 3 rungs incl. long-p prefill
///   and devpos twins). Issue 899 race (c): the 898 form's in-loop
///   `s_rmx[r] = new_max` was a cross-warp read-then-write hazard —
///   LATENT in the shipped arm since Bench 892 (fired tile-monotonically
///   under the split-gang's warp scheduling); fixed by the deferred
///   tid0-owned write in the same commit — the intended per-row chain is
///   unchanged (bit-identity re-pinned). Engagement gate: only when
///   `n_kv·ceil(p/8)·BPG >= 128` ([`attn_gang_engaged`]) — the
///   small-p/verify regime stays on the vec arm (measured 0.78-1.15x
///   there); non-6-head ratios fall back to the vec arm.
///   `QWEN38_PF_ATTN_GANG=0|false|off` restores the vec arm (the A/B
///   hatch; VEC=0 below it still selects the prefetch arm).
/// - Issue 896 — `QWEN38_PF_ATTN_VEC` (**DEFAULT ON** since Bench 891):
///   the vectorized-MLP arm (`att_pf_mq8v` — float4 K-row loads + deeper
///   unroll over the prefetch arm; BIT-IDENTICAL to the serial kernel,
///   gate-pinned by `bench_896_attn_vec_g1`). GOAT: G1 bit-identity (12
///   fixtures), G2 kernel-level 0.57-0.78x vs serial on every cell,
///   G3 e2e pp4096 2431 vs 2213 tok/s + pp2048 2860 vs 2805 (pins EXACT
///   both cells — bit-identical output needs no re-pin), G4 no new
///   allocations. `QWEN38_PF_ATTN_VEC=0|false|off` restores the prefetch
///   arm (the A/B escape hatch; PF=0 below it still selects the serial
///   kernel). Precedence: split > staged > gang > vec > pf > serial
///   (multi-knob sets are undefined behavior by convention; the first
///   match wins).
fn attn_arm() -> u8 {
    static ARM: OnceLock<u8> = OnceLock::new();
    *ARM.get_or_init(|| {
        let split = std::env::var("QWEN38_PF_ATTN_SPLITKV").is_ok_and(|s| matches!(s.trim(), "1" | "true" | "on"));
        if split {
            return 3;
        }
        let staged = std::env::var("QWEN38_PF_ATTN_STAGED").is_ok_and(|s| matches!(s.trim(), "1" | "true" | "on"));
        if staged {
            return 2;
        }
        // Issue 898 — the head-ganged kv_group arm. DEFAULT ON since Bench
        // 892 (all GOAT gates PASS + modelless — the bit-identical layout
        // change); `QWEN38_PF_ATTN_GANG=0|false|off` restores the vec arm
        // (the A/B hatch — the Bench-891 promotion shape). The engagement
        // gate (attn_gang_engaged, at the dispatch site) keeps the
        // small-p/verify regime on the vec arm.
        let gang = std::env::var("QWEN38_PF_ATTN_GANG").map_or(true, |s| !matches!(s.trim(), "0" | "false" | "off"));
        if gang {
            return 5;
        }
        let vec = std::env::var("QWEN38_PF_ATTN_VEC").map_or(true, |s| !matches!(s.trim(), "0" | "false" | "off"));
        if vec {
            return 4;
        }
        let pf = std::env::var("QWEN38_PF_ATTN_PF").map_or(true, |s| !matches!(s.trim(), "0" | "false" | "off"));
        if pf {
            1
        } else {
            0
        }
    })
}

/// Plan 605 T4 — the FA-class mma attention arm (`att_pf_fa[_dp]` + the
/// f32→f16 KV conversion pass). **DEFAULT-ON** (promoted 2026-09-20 under
/// the delegated perf/sec authority — the Bench-771/805 precedent — after
/// the full GOAT: G1 bench_946 15/15 + the Issue-750-T3 per-family
/// retention walk 0 flips × 6 families × 36 items, bench_948; G2 kernel
/// 5.4-5.7× at the league shapes, e2e pp4096 +11.5%; G3 FA never engages
/// below `FA_MIN_P` — the vec/tiny-p regime keeps its winner; G4 the f16
/// scratch is sized ONCE at block_size). KILL-SWITCH: `QWEN38_PF_ATTN_FA
/// =0|off|false` restores the incumbent ladder. TOLERANCE-CLASS (f16
/// operands + mma accumulation — max scaled err ≤ 9.6e-4 vs the serial
/// incumbent, bench_946; the opponent's production numerics class).
/// Precedes every other arm when engaged (the shape gates — ahd 256 +
/// n_head/n_kv 6 — are checked at the dispatch site; non-qualifying shapes
/// fall through to the incumbent ladder unchanged; the f16 scratch keeps
/// the attn_split discipline — stable addresses for the 965 graphs).
fn fa_engaged() -> bool {
    static FA: OnceLock<bool> = OnceLock::new();
    *FA.get_or_init(|| {
        // The Bench-805 env-once protocol: the env-derived value IS the
        // live value on the first call (no AtomicBool divergence — the
        // latent-bug class that promotion caught there).
        !std::env::var("QWEN38_PF_ATTN_FA")
            .is_ok_and(|s| matches!(s.trim(), "0" | "off" | "false"))
    })
}

/// Plan 605 T4 — the engagement predicate's chunk-length floor. Measured
/// (bench_946 `attn_fa_small_p_crossover`, 4090): FA beats BOTH ladder rungs
/// at every p ≥ 128 (fa/g3 0.569× @128 → 0.198× @2048; fa/vec 0.515× @128)
/// but LOSES to vec at p=32 (1.398× — the 3-launch conversion overhead) and
/// sits inside launch-overhead noise at 64 (0.976× vs vec). 128 is the
/// first rung with a clean ≥1.75× margin over the ladder's own pick; below
/// it the incumbent keeps the lane (decode chunks p=1 included).
const FA_MIN_P: usize = 128;

/// Issue 742 T1.7 — split-KV chunk length (DEFAULT 256 = one serial tile;
/// must be a nonzero multiple of 256 for tile alignment — invalid values
/// fall back to 256). The A/B knob for the split-arm sweep.
fn attn_split_chunk() -> usize {
    static CH: OnceLock<usize> = OnceLock::new();
    *CH.get_or_init(|| {
        let v = std::env::var("QWEN38_PF_ATTN_CHUNK")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(256);
        if v == 0 || !v.is_multiple_of(256) {
            256
        } else {
            v
        }
    })
}

/// Issue 742 T1.7 — the split-KV arm's parallelism cap: the split serves
/// the small-p regime where the serial kernel's grid (n_head*ceil(p/8))
/// under-fills the 128 SMs; above the cap the serial/prefetch arm already
/// has enough blocks and the split's extra combine + scratch would only
/// add overhead.
///
/// Issue 884 lever B (the long-p rung): at long-p PREFILL the wall flips —
/// the serial kernel's per-block KV-scan LATENCY CHAIN dominates while the
/// grid already over-fills the SMs (Bench 889: attn stage 3.26x per 2x
/// tokens, GEMM/GDN exactly linear; the T1.6 doc's ~6 GB/s per block).
/// `QWEN38_PF_ATTN_SPLIT_QTILES=<n>` overrides the 128-qtile routing cap
/// (any value 1..=65536; invalid/unset = 128 = the shipped small-p
/// routing, byte-identical behavior). OPT-IN and TOLERANCE-CLASS — the
/// cross-chunk merge reassociates rounding, so scored runs need their own
/// captured pin pair (see `bench_742_t1_attn_splitkv_g1` for the long-p
/// G1 re-pin). Scratch scales with the cap: `cap*8*n_chunks_max` m/l slots
/// and `cap*8*n_chunks_max*256` out (~1.6 GB at cap 12288, block_size
/// 4096, chunk 256); an allocation failure routes to the prefetch arm
/// (fail-safe), never a panic.
fn attn_split_qtile_cap(n_head: usize) -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    let cap = *CAP.get_or_init(|| {
        std::env::var("QWEN38_PF_ATTN_SPLIT_QTILES")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&v| (1..=65536).contains(&v))
            .unwrap_or(128)
    });
    n_head * usize::max(cap / n_head, 1)
}

/// Issue 884 lever B — the split-KV engagement predicate, shared by the
/// scratch preamble and the kernel dispatch (one definition; the two gate
/// sites must not drift).
fn attn_split_engaged(arm: u8, n_head: usize, p: usize) -> bool {
    arm == 3 && n_head * p.div_ceil(8) <= attn_split_qtile_cap(n_head)
}

/// Issue 899 — the gang-family layout selector (which ladder rung arm 5
/// launches): `3` = the GH=3 half-gang (DEFAULT — the Bench-893 G2
/// winner: 2 blocks/SM, 24 rows each, 3.4%/4.2% faster than the 898 full
/// gang at the pp4096/pp2048 decision cells, bit-identical by the same
/// construction), `6` = the Issue-898 full gang (98,880 B smem, 1
/// block/SM — the A/B artifact), `2` = the GH=2 third-gang (measured
/// NEGATIVE at the prefill cells — the ×3 KV L2 re-reads beat the 24
/// warps; A/B artifact only). `QWEN38_PF_ATTN_GANG=6|3|2` forces a
/// rung; `0|false|off` is consumed by `attn_arm` (the vec fallback) and
/// never reaches here.
fn attn_gang_layout() -> u8 {
    static LAYOUT: OnceLock<u8> = OnceLock::new();
    *LAYOUT.get_or_init(|| {
        std::env::var("QWEN38_PF_ATTN_GANG")
            .ok()
            .and_then(|s| match s.trim() {
                "6" => Some(6u8),
                "3" => Some(3u8),
                "2" => Some(2u8),
                _ => None,
            })
            .unwrap_or(3)
    })
}

/// Issue 899 — blocks per (kv_group, q-tile) for the active gang layout;
/// the engagement threshold scales by it (the split grid is BPG× wider).
fn attn_gang_bpg() -> usize {
    match attn_gang_layout() {
        6 => 1,
        2 => 3,
        _ => 2,
    }
}

/// Issue 898 — the head-gang engagement predicate: the gang grid is
/// `n_kv·ceil(p/8)·BPG` blocks (BPG = 1/2/3 by layout, Issue 899), each
/// block with 6/BPG-head rows — it only pays when the grid still fills
/// the 128 SMs (>= 128 blocks). Below that (the verify regime, small p)
/// the vec arm's wider grid wins; measured in `bench_898_attn_gang_g1`:
/// the ladder 0.48-0.51x serial at pp2048/pp4096 prefill, 0.78-1.15x at
/// p=16 (the vec arm keeps 0.56-0.84x there).
fn attn_gang_engaged(arm: u8, n_head: usize, n_kv: usize, p: usize) -> bool {
    arm == 5
        && n_kv != 0
        && n_head.is_multiple_of(n_kv)
        && n_head / n_kv == 6
        && n_kv * attn_gang_bpg() * p.div_ceil(8) >= 128
}

/// Issue 742 T1.4c — quantize-dedup A/B knob (DEFAULT ON): consecutive
/// GEMMs reading the same input buffer (qkv/z/a/b, gate/up, wq/wkv) share
/// one quantize. The skipped launches wrote byte-identical packed data to
/// the same scratch, so this is numerics-IDENTICAL; the knob exists purely
/// to A/B the wall-clock saving. `RIIR_PREFILL_QDEDUP=0|false|off` restores
/// the per-GEMM quantize.
fn quantize_dedup_enabled() -> bool {
    static DEDUP: OnceLock<bool> = OnceLock::new();
    *DEDUP.get_or_init(|| {
        std::env::var("RIIR_PREFILL_QDEDUP").map_or(true, |s| !matches!(s.trim(), "0" | "false" | "off"))
    })
}

/// Issue 742 T1.5 — graph-verify mode (the real verify path's CUDA-graph
/// replay): when armed, the spec chunk's layer loop runs through the devpos
/// kernels (base_pos from the device pos buffer) and, once a full chunk has
/// warmed every weight mirror, is CAPTURED into a graph that every
/// same-`p` chunk then replays at its own uploaded position. The captured
/// kernel sequence is the eager sequence with identical arithmetic — G1 is
/// bit-identity of the chunked-prefill logits pins (the standing gate).
/// `RIIR_PREFILL_GRAPH_VERIFY=0|false|off` restores the eager path (A/B).
fn gv_armed() -> bool {
    static GV: OnceLock<bool> = OnceLock::new();
    *GV.get_or_init(|| {
        std::env::var("RIIR_PREFILL_GRAPH_VERIFY").is_ok_and(|s| matches!(s.trim(), "1" | "true" | "on"))
    })
}

/// Issue 742 T1.8 — the verify tail's G1-strict fallback arm: p sequential
/// CubeCL GemvTernary lm_head rows (the decode lm_head kernel family,
/// bit-matched by the G1c decode-continuity pins) instead of the fast mma
/// GEMM tail. Each row pays a dtoh/upload/read crossing (~1-2 ms), so this
/// is the strict greedy-identity gate's arm, not the shipped operating
/// point. `RIIR_PREFILL_VERIFY_TAIL_GEMV=1` arms (opt-in); unset = the mma
/// tail (one weight read for all p rows, near-tie tolerance vs decode).
fn verify_tail_gemv() -> bool {
    static G: OnceLock<bool> = OnceLock::new();
    *G.get_or_init(|| {
        std::env::var("RIIR_PREFILL_VERIFY_TAIL_GEMV").is_ok_and(|s| matches!(s.trim(), "1" | "true" | "on"))
    })
}

/// Issue 965 — graph mode for the REGULAR prefill path. **DEFAULT ON** (the
/// GOAT posture: G1 pins bit-identical across warm/capture/replay at both
/// league shapes + the decode-continuity probe green through the automatic
/// flush hooks — Bench 936; the pp2048 A/B +10.9%, pp4096 +10.1% over the
/// raw-launch path on the 4090, 2026-09-17). `RIIR_PREFILL_CUDA_GRAPHS=0`
/// is the kill-switch — it restores the raw-launch path verbatim. When
/// armed the process adopts spec writeback semantics (see the module doc)
/// so the Issue-742 T1.5 capture machinery engages on plain chunks; the
/// flush-before-consume hooks make that invisible to every audited consumer
/// (decode funnel, fall-through, hybrid-cache export, speculative
/// checkpoints/rollbacks, tree verify).
fn graphs_armed() -> bool {
    static G: OnceLock<bool> = OnceLock::new();
    *G.get_or_init(|| {
        !std::env::var("RIIR_PREFILL_CUDA_GRAPHS")
            .is_ok_and(|s| matches!(s.trim(), "0" | "false" | "off"))
    })
}

/// Issue 742 T1.5 — the chunk width the graph capture targets. Chunks of
/// any OTHER `p` never capture/replay (the captured grids are `p`-shaped);
/// a wider warm-up prefill therefore cannot steal the capture.
/// `RIIR_PREFILL_GRAPH_P` overrides both defaults. Lane-keyed default
/// (Issue 965): the VERIFY lane (`RIIR_PREFILL_GRAPH_VERIFY=1`) pins **16**
/// — its cm=16 chunk loop, and the wide-warm-up-must-not-steal contract —
/// while the regular prefill path (graph mode default-on) pins **2048**,
/// the league chunk.
fn capture_p() -> usize {
    static P: OnceLock<usize> = OnceLock::new();
    *P.get_or_init(|| {
        std::env::var("RIIR_PREFILL_GRAPH_P")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .filter(|&p| (1..=4096).contains(&p))
            .unwrap_or(if gv_armed() { 16 } else { 2048 })
    })
}

/// Issue 734 Arm 12 — the GDN chunked recurrence on the cudarc stack (the
/// llama.cpp prefill shape). **Numerics-policy change (owner sign-off
/// 2026-08-23): this path is NOT bit-identical to the rowpar kernel** — the
/// chunkwise-parallel summation order diverges ~4e-3/layer (Bench 705
/// Candidate A: kernel-exact, e2e max_rel O(1), argmax stable) exactly like
/// llama.cpp's own chunked GDN prefill, which has no bit-exact serial
/// reference. The e2e gate for this arm is argmax-stability + distributional
/// agreement (KL / top-k / max_rel) + greedy-generation agreement — NOT the
/// Bench-710 FNV pin (the pin still guards the rowpar + psplit arms).
/// `RIIR_PREFILL_GDN_CHUNKED=1|true|on` (env wins over the setter).
fn gdn_chunked_enabled() -> bool {
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env = *ENV.get_or_init(|| {
        std::env::var("RIIR_PREFILL_GDN_CHUNKED")
            .ok()
            .map(|s| matches!(s.trim(), "1" | "true" | "on"))
    });
    env.unwrap_or_else(|| GDN_CHUNKED.load(Ordering::Relaxed))
}

/// A/B knob for the Arm-12 GDN chunked recurrence (see [`gdn_chunked_enabled`]).
/// Issue 965: a flip changes the captured layer loop's kernel sequence, so it
/// bumps [`GV_KNOB_GEN`] — the graph replay predicate folds the generation in
/// (stale graph → eager chunks; correct numerics, no re-capture this process).
/// Every OTHER captured-path knob is an env `OnceLock` (mid-process change
/// impossible by construction), so this is the one invalidation site.
pub fn set_prefill_gdn_chunked(on: bool) {
    GDN_CHUNKED.store(on, Ordering::Relaxed);
    GV_KNOB_GEN.fetch_add(1, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Canonical numerics forms
//
// The proven defaults (Bench 719/721) until the probe measures the exact
// lowerings; `bench_734_prefill_cuda_bitidentity` re-pins these.
// ---------------------------------------------------------------------------

/// Recurrence forms: natural fmad contraction + fma update + ASCENDING
/// xor-butterfly plane_sum (the probe-selected lowering) + merged rsqrt.
pub fn canonical_conv_forms() -> (ConvFma, SigExp, SigDiv) {
    (ConvFma::Fma, SigExp::Fast, SigDiv::Full)
}

pub fn canonical_beta_decay_forms() -> (LogForm, SigExp, SigDiv) {
    (LogForm::Fast, SigExp::Fast, SigDiv::Full)
}

pub fn canonical_zgate_forms() -> (SigExp, SigDiv) {
    (SigExp::Fast, SigDiv::Full)
}

pub fn canonical_rope_forms() -> (LogForm, SigExp, SinCosForm, RopeRot) {
    (LogForm::Fast, SigExp::Fast, SinCosForm::Fast, RopeRot::Fma)
}

pub fn canonical_attention_forms() -> (AttnDot, AttnAcc, SigExp, AttnPh3, AttnRes, AttnInv) {
    (
        AttnDot::Fma,
        AttnAcc::Fma,
        SigExp::Fast,
        AttnPh3::Mul,
        AttnRes::Fma,
        AttnInv::Full,
    )
}

// ---------------------------------------------------------------------------
// Mirrors
// ---------------------------------------------------------------------------

/// f32 weight mirrors for one layer (built once per process).
struct LayerF32 {
    input_norm: CudaSlice<f32>,     // [n]
    post_attn_norm: CudaSlice<f32>, // [n]
    conv1d_weight: CudaSlice<f32>,  // [conv_dim * ks]
    a_log: CudaSlice<f32>,          // [n_v]
    dt_bias: CudaSlice<f32>,        // [n_v]
    linear_norm: CudaSlice<f32>,    // [hd]
    attn_q_norm: Option<CudaSlice<f32>>, // [ahd]
    attn_k_norm: Option<CudaSlice<f32>>, // [ahd]
    /// Issue 980 T4-ALT — the Bonsai-2 dense `ssm_alpha`/`ssm_beta` escape
    /// set, mirrored fp32 `[n_v × n_embd]`. `Some` ONLY on folded files;
    /// consumed by `gemm_dense_ab_batched` on the PRIMAL normed input (the
    /// escape set is neither rotated nor folded).
    dense_a: Option<CudaSlice<f32>>, // [n_v * n]
    dense_b: Option<CudaSlice<f32>>, // [n_v * n]
}

/// Persistent per-layer state mirrors (allocated once, values synced per
/// prompt / after fall-throughs).
struct StateMirrors {
    dn_states: Vec<Option<CudaSlice<f32>>>,   // [n_v*hd*hd] per GDN layer
    conv_states: Vec<Option<CudaSlice<f32>>>, // [conv_dim*ks] per GDN layer
    kv_k: Vec<Option<CudaSlice<f32>>>,        // [block_size*kvd] per ATT layer
    kv_v: Vec<Option<CudaSlice<f32>>>,
}

// ---------------------------------------------------------------------------
// The stack
// ---------------------------------------------------------------------------

/// Grow-only activation staging, reused across every chunk.
struct FullBufs {
    x: Option<CudaSlice<f32>>,        // [p*n]
    normx: Option<CudaSlice<f32>>,    // [p*n]
    qkv_b: Option<CudaSlice<f32>>,    // [p*qkv_dim] raw in_proj out
    qkv_conv: Option<CudaSlice<f32>>, // [p*qkv_dim] SiLU conv out
    qkvx_b: Option<CudaSlice<f32>>,   // [p*3*v_dim] expanded
    z_b: Option<CudaSlice<f32>>,      // [p*v_dim]
    a_b: Option<CudaSlice<f32>>,      // [p*n_v]
    b_b: Option<CudaSlice<f32>>,
    beta_b: Option<CudaSlice<f32>>,
    decay_b: Option<CudaSlice<f32>>,
    rec_b: Option<CudaSlice<f32>>, // [p*v_dim]
    tmp_b: Option<CudaSlice<f32>>, // [p*n]
    gate_b: Option<CudaSlice<f32>>, // [p*mlp]
    up_b: Option<CudaSlice<f32>>,
    hid_b: Option<CudaSlice<f32>>,
    ffnout_b: Option<CudaSlice<f32>>, // [p*n]
    qg_b: Option<CudaSlice<f32>>,     // [p*2*qa]
    q_b: Option<CudaSlice<f32>>,      // [p*qa]
    agate_b: Option<CudaSlice<f32>>,  // [p*qa]
    kv_b: Option<CudaSlice<f32>>,     // [p*2*kvd]
    k_b: Option<CudaSlice<f32>>,      // [p*kvd]
    v_b: Option<CudaSlice<f32>>,
    attn_out_b: Option<CudaSlice<f32>>, // [p*qa]
    aproj_b: Option<CudaSlice<f32>>,    // [p*n]
    /// Arm-12 GDN chunked-recurrence scratch (grow-only, sized by p).
    gdn_lg: Option<CudaSlice<f32>>,   // [n_v * n_chunks * 64]
    gdn_ga: Option<CudaSlice<f32>>,
    gdn_dte: Option<CudaSlice<f32>>,
    gdn_td: Option<CudaSlice<f32>>,   // [n_v * n_chunks]
    gdn_x: Option<CudaSlice<f32>>,    // [n_v * n_chunks * 4096]
    gdn_t: Option<CudaSlice<f32>>,    // [n_v * n_chunks * 4096] (T-inverse)
    gdn_qkr: Option<CudaSlice<f32>>,
    /// `(max words, p, scratch)` — one scratch sized for the largest GEMM
    /// input dim serves every smaller dim (the kernel derives its extent
    /// from the `n` arg; indices stay inside `[0, p*n/4)`).
    scratch: Option<(usize, usize, GemmI8MmaScratch)>,
    /// Issue 742 T1.5 — bumped on ANY staging realloc. The captured
    /// graph-verify graph bakes buffer ADDRESSES; a realloc (a later wider
    /// chunk growing the staging) orphans it. The replay gate compares this
    /// against the generation recorded at capture (`gv_staging_gen`).
    staging_gen: usize,
    /// Issue 742 T1.5 — persistent token buffer for the graph-verify path
    /// (address-stable so the eager embed can be ordered before the replay;
    /// grow-only, sized to the largest chunk seen).
    tokens_dev: Option<CudaSlice<u32>>,
    /// Issue 742 T1.5 — the 1-element device pos buffer the devpos kernels
    /// read (address baked into the captured graph — allocated ONCE, never
    /// reallocated).
    pos_dev: Option<CudaSlice<i32>>,
    /// Issue 742 T1.7 — split-KV mq8 partial scratch (arm 3). Sized ONCE
    /// at first arm-3 use for the routing cap (`attn_split_qtile_cap` qtiles
    /// × n_chunks_max from `block_size/chunk_len`) so the addresses never
    /// move — a realloc would orphan the captured graph (the `staging_gen`
    /// guard catches any pathological re-size). One set serves all attn
    /// layers (single-stream ordering).
    attn_split_pm: Option<CudaSlice<f32>>,
    attn_split_pl: Option<CudaSlice<f32>>,
    attn_split_po: Option<CudaSlice<f32>>,
    /// The `n_chunks_max` the scratch was sized for (0 = unallocated).
    attn_split_chunks: usize,
    /// Plan 605 T2 — the FA-class f16 KV scratch (`kv_f32_to_f16` converts
    /// the live range per attn layer call; transient — one pair serves all
    /// attn layers, single-stream ordering). Sized ONCE at block_size
    /// (fa_scratch_rows(block_size) rows — padded to the nbatch_fa=32
    /// grid, the pad tail zeroed every conversion); stable addresses for
    /// the 965 graphs (the attn_split discipline).
    attn_fa_kh: Option<CudaSlice<u16>>,
    attn_fa_vh: Option<CudaSlice<u16>>,
    /// The padded-row count the FA scratch was sized for (0 = unallocated).
    attn_fa_rows: usize,
    /// Issue 980 T4-ALT — the folded-prefill rotation staging. `rot_scratch`
    /// `[p×n]` holds the ROTATED copy of each normed matmul input (the
    /// primal `normx` stays untouched for the dense escape-set a/b GEMM +
    /// the residual chain); `permute_tmp` `[p×v_dim]` is the
    /// `gdn_v_permute_batched` source copy. Grow-only like every staging
    /// buffer — stable addresses for the 965 prefill graphs.
    rot_scratch: Option<CudaSlice<f32>>, // [p*n]
    permute_tmp: Option<CudaSlice<f32>>, // [p*v_dim]
    x_row_host: Vec<f32>,
    /// Issue 742 T1.8 — verify-tail logits staging `[p * vocab]` (grow-only;
    /// grown in the chunk preamble BEFORE any capture region so a later
    /// verify chunk can never orphan the graph-verify capture).
    verify_logits: Option<CudaSlice<f32>>,
    /// Issue 742 T1.8 — verify-tail per-row argmax results `[p]` (packed u64,
    /// Issue-697 convention; memset before every launch).
    verify_argmax: Option<CudaSlice<u64>>,
    /// Issue 742 T1.8 — host landing buffer for the `[p]` packed argmax.
    verify_argmax_host: Vec<u64>,
    /// Issue 884 T2a — verify-tail NLL staging: `verify_nll` holds the
    /// device `[2p]` (lse, target-logit) pairs, `verify_nll_host` the host
    /// landing copy (grow-only, like the argmax staging).
    verify_nll: Option<CudaSlice<f32>>,
    verify_nll_host: Vec<f32>,
}

struct FullStack {
    stream: Arc<CudaStream>,
    /// Arm 13 — forked copy stream (waits on `stream` at fork; per-layer
    /// ordering re-established with a reusable event before each DtoH).
    copy_stream: Option<Arc<CudaStream>>,
    mma: GemmTernaryI8MmaCuda,
    ffn: CudaFfnKernels,
    dn: CudaDeltanetKernels,
    at: CudaAttnKernels,
    /// Arm 12 — None if NVRTC compile fails (falls back to rowpar).
    gdn_chunked: Option<CudaGdnChunkedKernels>,
    bufs: Mutex<FullBufs>,
    /// f32 weight mirrors — built once (None on any read/upload failure).
    f32_weights: OnceLock<Option<Arc<Vec<LayerF32>>>>,
    /// Issue 980 T4-ALT — folded-prefill rotation tables (kernels + sign
    /// vectors), built once from the first folded forward's config; the
    /// inner `None` covers pre-rotation files and build failures (a folded
    /// model then refuses at the fall-through gate in
    /// `prefill_tokens_chunk`).
    rotation_tables: OnceLock<Option<Arc<crate::deltanet_rotation_cudarc::RotationTables>>>,
    /// Persistent state mirrors — allocated once; Mutex for &mut slice views
    /// (the uploads need `try_slice_mut`).
    states: OnceLock<Option<Mutex<StateMirrors>>>,
    /// True right after a successful chunk (mirror == CubeCL handles); any
    /// fall-through clears it, forcing the next chunk to re-sync.
    mirror_clean: AtomicBool,
    /// Arm 13 — true when the mirrors were device-side memset to match a
    /// `reset_state` (skips the base_pos==0 re-upload); cleared by any
    /// CubeCL-side mutation and at every chunk entry.
    mirrors_at_reset: AtomicBool,
    /// Arm 13 — per-layer CACHED-pinned writeback slots (lazily allocated,
    /// grow-only). GDN layers: `[dn_state | conv_state]`; attention layers:
    /// `[k_rows | v_rows]` (p-sized, prefix-reused for smaller p).
    wb_slots: OnceLock<Option<Mutex<Vec<Option<WbSlot>>>>>,
    /// Arm 13 — device-side pre-chunk snapshot of every mirror (fall-through
    /// repair for the streamed writes; None if allocation failed → the
    /// chunk refuses the arm BEFORE any mutation).
    wb_snapshot: OnceLock<Option<Mutex<StateMirrors>>>,
    /// Issue 742 T1.1 — spec mode advanced the mirrors past the CubeCL
    /// handles (per-chunk writeback elided); the next non-spec chunk
    /// auto-flushes [0, spec_pos) before computing.
    spec_dirty: AtomicBool,
    /// Absolute position the spec chunks have advanced to (base_pos + p of
    /// the last spec chunk).
    spec_pos: AtomicUsize,
    /// Issue 742 T1.2 probe — the captured layer-loop graph (timing probe
    /// only; fixed base_pos — see the capture site).
    graph_probe: OnceLock<Option<Mutex<SendGraph>>>,
    /// Issue 742 T1.5 — the REAL graph-verify capture: the layer loop built
    /// from the devpos kernels (base_pos read from `bufs.pos_dev` at kernel
    /// runtime), so one capture serves EVERY chunk of the same `p` at any
    /// position. `None` once capture was ATTEMPTED and failed (permanent
    /// eager fallback for the process).
    graph_verify: OnceLock<Option<Mutex<SendGraph>>>,
    /// The `p` the `graph_verify` capture covers (0 = not captured yet).
    gv_p: AtomicUsize,
    /// The `bufs.staging_gen` at capture time — a mismatch (a wider chunk grew the
    /// staging) invalidates the graph (baked addresses dangle).
    gv_staging_gen: AtomicUsize,
    /// Issue 965 — the [`GV_KNOB_GEN`] at capture time; a mismatch (the
    /// gdn-chunked knob flipped post-capture) invalidates replay.
    gv_knob_gen: AtomicUsize,
    /// Issue 742 T1.5 — total graph replays served (diagnostic; read by the
    /// G1 test to prove the replay path actually ran).
    gv_replays: AtomicUsize,
    /// Issue 967 — eager chunks at p != capture_p() with WARM mirrors (the
    /// capture-ladder gap: exactly the set a second rung would recover;
    /// cold-mirror chunks can never capture, so they are excluded) and their
    /// token total. `gv_total_tokens` (all chunks through the graph lane) is
    /// the share denominator. Read by `prefill_gv_fallback_counts`.
    gv_fallbacks: AtomicUsize,
    gv_fallback_tokens: AtomicUsize,
    gv_total_tokens: AtomicUsize,
    /// True after any successful chunk completed in this process — every
    /// weight mirror a full 64-layer chunk touches is warm by then, so a
    /// capture cannot bake cold-mirror upload memcpys into the graph.
    gv_warm: AtomicBool,
    /// Issue 742 T1.8 — the final-norm gamma mirror (built once on the first
    /// verify-tail chunk; `[n]` f32).
    final_norm_gamma: OnceLock<Option<CudaSlice<f32>>>,
}

/// SAFETY: the probe graph is only launched from the chunk path on the
/// thread that owns the stack (single-threaded dispatch, the WbSlot
/// argument); the raw driver handles never cross threads while in use.
struct SendGraph(CudaGraph);
unsafe impl Send for SendGraph {}

/// A cached-pinned host slot for one layer's writeback payload. Allocated
/// via the low-level `malloc_host` with flags=0 (portable + CPU-cacheable —
/// NOT write-combined: the drain path READS this memory to feed the CubeCL
/// writes; WC reads would crawl). The `done` event is recorded on the copy
/// stream after the layer's async DtoHs and polled non-blocking by the
/// drain loop.
struct WbSlot {
    ptr: *mut f32,
    cap: usize,
    done: CudaEvent,
}

// SAFETY: the raw pointer is only dereferenced through the CUDA driver /
// the drain path on the thread that owns the chunk; `CudaEvent` is a driver
// handle. The slot lives in the process-global FULL_STACK and outlives all
// chunk calls (drop frees the host memory after synchronizing the event).
unsafe impl Send for WbSlot {}
unsafe impl Sync for WbSlot {}

impl Drop for WbSlot {
    fn drop(&mut self) {
        let _ = self.done.synchronize();
        unsafe {
            let _ = cudarc::driver::result::free_host(self.ptr as *mut _);
        }
    }
}

static FULL_STACK: OnceLock<Option<Arc<FullStack>>> = OnceLock::new();

fn full_stack() -> Option<Arc<FullStack>> {
    FULL_STACK
        .get_or_init(|| match build_full_stack() {
            Ok(s) => Some(Arc::new(s)),
            Err(e) => {
                eprintln!(
                    "[734-arm8] CUDA whole-prefill stack init failed ({e}) — prefill stays on CubeCL"
                );
                None
            }
        })
        .clone()
}

fn build_full_stack() -> Result<FullStack, String> {
    let ctx = CudaContext::new(0).map_err(|e| e.to_string())?;
    let stream = ctx.new_stream().map_err(|e| e.to_string())?;
    // Arm 13 — the copy stream (fork = waits on the compute stream now; the
    // per-layer ordering is re-established with `wb_order_ev` before each
    // DtoH so the copies never read a half-written mirror).
    let copy_stream = stream.fork().ok();
    let mma = GemmTernaryI8MmaCuda::new(ctx.clone()).map_err(|e| e.to_string())?;
    let ffn = CudaFfnKernels::new(ctx.clone()).map_err(|e| e.to_string())?;
    let dn = CudaDeltanetKernels::new(ctx.clone()).map_err(|e| e.to_string())?;
    // Arm 12: independent NVRTC module — a compile failure only disables the
    // chunked path (rowpar stays the default).
    let gdn_chunked = CudaGdnChunkedKernels::new(ctx.clone()).ok();
    let at = CudaAttnKernels::new(ctx).map_err(|e| e.to_string())?;
    Ok(FullStack {
        stream,
        copy_stream,
        mma,
        ffn,
        dn,
        at,
        gdn_chunked,
        bufs: Mutex::new(FullBufs {
            x: None,
            normx: None,
            qkv_b: None,
            qkv_conv: None,
            qkvx_b: None,
            z_b: None,
            a_b: None,
            b_b: None,
            beta_b: None,
            decay_b: None,
            rec_b: None,
            tmp_b: None,
            gate_b: None,
            up_b: None,
            hid_b: None,
            ffnout_b: None,
            qg_b: None,
            q_b: None,
            agate_b: None,
            kv_b: None,
            k_b: None,
            v_b: None,
            attn_out_b: None,
            aproj_b: None,
            gdn_lg: None,
            gdn_ga: None,
            gdn_dte: None,
            gdn_td: None,
            gdn_x: None,
            gdn_t: None,
            gdn_qkr: None,
            scratch: None,
            staging_gen: 0,
            tokens_dev: None,
            pos_dev: None,
            attn_split_pm: None,
            attn_split_pl: None,
            attn_split_po: None,
            attn_split_chunks: 0,
            attn_fa_kh: None,
            attn_fa_vh: None,
            attn_fa_rows: 0,
            rot_scratch: None,
            permute_tmp: None,
            x_row_host: Vec::new(),
            verify_logits: None,
            verify_argmax: None,
            verify_argmax_host: Vec::new(),
            verify_nll: None,
            verify_nll_host: Vec::new(),
        }),
        f32_weights: OnceLock::new(),
        // Issue 980 T4-ALT — the folded-prefill rotation tables (see the
        // field doc on the struct): first-fwd-wins, inner None = build
        // failure → the fall-through gate refuses folded loudly.
        rotation_tables: OnceLock::new(),
        states: OnceLock::new(),
        mirror_clean: AtomicBool::new(false),
        mirrors_at_reset: AtomicBool::new(false),
        wb_slots: OnceLock::new(),
        wb_snapshot: OnceLock::new(),
        spec_dirty: AtomicBool::new(false),
        spec_pos: AtomicUsize::new(0),
        graph_probe: OnceLock::new(),
        graph_verify: OnceLock::new(),
        gv_p: AtomicUsize::new(0),
        gv_staging_gen: AtomicUsize::new(0),
        gv_knob_gen: AtomicUsize::new(0),
        gv_replays: AtomicUsize::new(0),
        gv_fallbacks: AtomicUsize::new(0),
        gv_fallback_tokens: AtomicUsize::new(0),
        gv_total_tokens: AtomicUsize::new(0),
        gv_warm: AtomicBool::new(false),
        final_norm_gamma: OnceLock::new(),
    })
}

// ---------------------------------------------------------------------------
// Mirror builders
// ---------------------------------------------------------------------------

fn read_f32(client: &ComputeClient<ActiveRuntime>, h: &Handle) -> Option<Vec<f32>> {
    client
        .read_one(h.clone())
        .ok()
        .map(|b| f32::from_bytes(&b).to_vec())
}

fn build_f32_weights(
    client: &ComputeClient<ActiveRuntime>,
    stream: &Arc<CudaStream>,
    fwd: &TernaryDeltanetGpuForward,
) -> Option<Vec<LayerF32>> {
    let up = |v: Vec<f32>| stream.clone_htod(v.as_slice()).ok();
    let mut out = Vec::with_capacity(fwd.layers.len());
    for lw in &fwd.layers {
        let attn_q_norm = match &lw.attn_q_norm {
            Some(h) => Some(read_f32(client, h)?),
            None => None,
        };
        let attn_k_norm = match &lw.attn_k_norm {
            Some(h) => Some(read_f32(client, h)?),
            None => None,
        };
        // Issue 980 T4-ALT — mirror the dense escape-set a/b (folded files
        // only; `None` on pre-rotation files keeps the field shape). The
        // module gate already implies `ternary_gemm_batched`.
        let dense_a = lw
            .in_proj_a_f32
            .as_ref()
            .and_then(|h| read_f32(client, h))
            .and_then(|v| stream.clone_htod(v.as_slice()).ok());
        let dense_b = lw
            .in_proj_b_f32
            .as_ref()
            .and_then(|h| read_f32(client, h))
            .and_then(|v| stream.clone_htod(v.as_slice()).ok());
        out.push(LayerF32 {
            input_norm: up(read_f32(client, &lw.input_norm)?)?,
            post_attn_norm: up(read_f32(client, &lw.post_attn_norm)?)?,
            conv1d_weight: up(read_f32(client, &lw.conv1d_weight)?)?,
            a_log: up(read_f32(client, &lw.a_log)?)?,
            dt_bias: up(read_f32(client, &lw.dt_bias)?)?,
            linear_norm: up(read_f32(client, &lw.linear_norm)?)?,
            attn_q_norm: attn_q_norm.and_then(up),
            attn_k_norm: attn_k_norm.and_then(up),
            dense_a,
            dense_b,
        });
    }
    Some(out)
}

/// Allocate the persistent state mirrors (sizes from the config + layer
/// types; values synced separately).
fn build_states(
    stream: &Arc<CudaStream>,
    fwd: &TernaryDeltanetGpuForward,
    n_v: usize,
    hd: usize,
    conv_dim: usize,
    ks: usize,
    kvd: usize,
) -> Option<StateMirrors> {
    let mut dn_states = Vec::new();
    let mut conv_states = Vec::new();
    let mut kv_k = Vec::new();
    let mut kv_v = Vec::new();
    let dn_len = n_v * hd * hd;
    let conv_len = conv_dim * ks;
    let kv_len = fwd.config.block_size * kvd;
    for lt in &fwd.layer_types {
        if *lt == DeltaNetLayerType::DeltaNet {
            dn_states.push(Some(stream.alloc_zeros::<f32>(dn_len).ok()?));
            conv_states.push(Some(stream.alloc_zeros::<f32>(conv_len).ok()?));
            kv_k.push(None);
            kv_v.push(None);
        } else {
            dn_states.push(None);
            conv_states.push(None);
            kv_k.push(Some(stream.alloc_zeros::<f32>(kv_len).ok()?));
            kv_v.push(Some(stream.alloc_zeros::<f32>(kv_len).ok()?));
        }
    }
    Some(StateMirrors {
        dn_states,
        conv_states,
        kv_k,
        kv_v,
    })
}

/// The lazy per-weight mirror (the arm-6 `cuda_mma_cache` pattern — one
/// mirror per handle, shared across arms). GEMM consumers: the mirror may
/// carry packed codes ONLY on the fmt route (Plan 572).
fn mma_mirror(
    client: &ComputeClient<ActiveRuntime>,
    stream: &Arc<CudaStream>,
    w: &TernaryHandle,
) -> Option<Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>> {
    mma_mirror_impl(client, stream, w, true)
}

/// The bitplane-pair mirror (dequant consumers — wte): never packed; the
/// pair is always uploaded. Uses the SAME cache slot, so a handle must not
/// be mirrored under both policies (wte is never a GEMM weight).
fn mma_mirror_pair(
    client: &ComputeClient<ActiveRuntime>,
    stream: &Arc<CudaStream>,
    w: &TernaryHandle,
) -> Option<Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>> {
    mma_mirror_impl(client, stream, w, false)
}

fn mma_mirror_impl(
    client: &ComputeClient<ActiveRuntime>,
    stream: &Arc<CudaStream>,
    w: &TernaryHandle,
    allow_packed_route: bool,
) -> Option<Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>> {
    w.cuda_mma_cache
        .get_or_init(|| {
            build_weight_cache(client, stream, w, allow_packed_route)
                .map(Arc::new)
                .ok()
        })
        .clone()
}

// ---------------------------------------------------------------------------
// State sync / writeback
// ---------------------------------------------------------------------------

/// Upload the CubeCL state handles into the mirrors (prompt start, or after
/// a fall-through made them diverge). KV rows `[0, base_pos)` only when a
/// previous chunk's rows might be missing from the mirror.
fn sync_states(
    client: &ComputeClient<ActiveRuntime>,
    stream: &Arc<CudaStream>,
    fwd: &TernaryDeltanetGpuForward,
    states: &mut StateMirrors,
    base_pos: usize,
    kvd: usize,
) -> bool {
    let upload =
        |src: &Handle, dst: &mut CudaSlice<f32>| -> bool {
            let Ok(bytes) = client.read_one(src.clone()) else {
                return false;
            };
            let host = f32::from_bytes(&bytes);
            let Some(mut view) = dst.try_slice_mut(0..dst.len()) else {
                return false;
            };
            stream.memcpy_htod(host, &mut view).is_ok()
        };

    for (li, st) in fwd.deltanet_states.iter().enumerate() {
        if let (Some(src), Some(dst)) = (st, states.dn_states[li].as_mut())
            && !upload(src, dst)
        {
            return false;
        }
    }
    for (li, cs) in fwd.conv_states.iter().enumerate() {
        if let (Some(src), Some(dst)) = (cs, states.conv_states[li].as_mut())
            && !upload(src, dst)
        {
            return false;
        }
    }
    if base_pos > 0 {
        let f = core::mem::size_of::<f32>();
        let rows = base_pos * kvd;
        let total_rows = fwd.config.block_size * kvd;
        // Prefix view [0, rows): trim (total_rows - rows) elements off the end.
        let trim = ((total_rows - rows) * f) as u64;
        for li in 0..fwd.kv_key_caches.len() {
            let (Some(ksrc), Some(kdst), Some(vsrc), Some(vdst)) = (
                fwd.kv_key_caches[li].as_ref(),
                states.kv_k[li].as_mut(),
                fwd.kv_value_caches[li].as_ref(),
                states.kv_v[li].as_mut(),
            ) else {
                continue;
            };
            let k_view = ksrc.clone().offset_end(trim);
            let v_view = vsrc.clone().offset_end(trim);
            let upload_view = |view: &Handle, dst: &mut CudaSlice<f32>| -> bool {
                let Ok(bytes) = client.read_one(view.clone()) else {
                    return false;
                };
                let host = f32::from_bytes(&bytes);
                let Some(mut dv) = dst.try_slice_mut(0..rows) else {
                    return false;
                };
                stream.memcpy_htod(host, &mut dv).is_ok()
            };
            if !upload_view(&k_view, kdst) || !upload_view(&v_view, vdst) {
                return false;
            }
        }
    }
    true
}

/// Write the mutated mirrors back into the CubeCL handles (states + conv +
/// the KV rows this chunk filled) so every chunk is independently
/// fall-through-safe and decode can continue from the CubeCL side.
fn writeback_states(
    client: &ComputeClient<ActiveRuntime>,
    stream: &Arc<CudaStream>,
    fwd: &TernaryDeltanetGpuForward,
    states: &mut StateMirrors,
    base_pos: usize,
    p: usize,
    kvd: usize,
    state_host: &mut Vec<f32>,
    kv_host: &mut Vec<f32>,
) -> bool {
    let f = core::mem::size_of::<f32>();
    // deltanet + conv states (full).
    for (li, st) in fwd.deltanet_states.iter().enumerate() {
        if let (Some(dst_handle), Some(src)) = (st, states.dn_states[li].as_ref()) {
            let Some(view) = src.try_slice(0..src.len()) else {
                return false;
            };
            if state_host.len() < src.len() {
                state_host.resize(src.len(), 0.0);
            }
            if stream.memcpy_dtoh(&view, &mut state_host[..src.len()]).is_err() {
                return false;
            }
            client.write(
                dst_handle,
                cubecl::bytes::Bytes::from_bytes_vec(
                    f32::as_bytes(&state_host[..src.len()]).to_vec(),
                ),
            );
        }
    }
    for (li, cs) in fwd.conv_states.iter().enumerate() {
        if let (Some(dst_handle), Some(src)) = (cs, states.conv_states[li].as_ref()) {
            let Some(view) = src.try_slice(0..src.len()) else {
                return false;
            };
            if state_host.len() < src.len() {
                state_host.resize(src.len(), 0.0);
            }
            if stream.memcpy_dtoh(&view, &mut state_host[..src.len()]).is_err() {
                return false;
            }
            client.write(
                dst_handle,
                cubecl::bytes::Bytes::from_bytes_vec(
                    f32::as_bytes(&state_host[..src.len()]).to_vec(),
                ),
            );
        }
    }
    // KV rows [base_pos, base_pos + p).
    let rows = p * kvd;
    let total_rows = fwd.config.block_size * kvd;
    let trim_end = ((total_rows - (base_pos + p) * kvd) * f) as u64;
    for li in 0..fwd.kv_key_caches.len() {
        let (Some(kcache), Some(vcache), Some(ksrc), Some(vsrc)) = (
            fwd.kv_key_caches[li].as_ref(),
            fwd.kv_value_caches[li].as_ref(),
            states.kv_k[li].as_ref(),
            states.kv_v[li].as_ref(),
        ) else {
            continue;
        };
        if kv_host.len() < rows {
            kv_host.resize(rows, 0.0);
        }
        let write_one = |src: &CudaSlice<f32>,
                         cache: &Handle,
                         host: &mut Vec<f32>|
         -> bool {
            let Some(view) = src.try_slice(base_pos * kvd..(base_pos + p) * kvd) else {
                return false;
            };
            if stream.memcpy_dtoh(&view, &mut host[..rows]).is_err() {
                return false;
            }
            // Partial write at the absolute row offset (offset_start view).
            let dst = cache
                .clone()
                .offset_start(((base_pos * kvd) * f) as u64)
                .offset_end(trim_end);
            client.write(
                &dst,
                cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(&host[..rows]).to_vec()),
            );
            true
        };
        if !write_one(ksrc, kcache, kv_host) || !write_one(vsrc, vcache, kv_host) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Arm 13 — overlapped writeback (forked copy stream + cached-pinned slots)
// ---------------------------------------------------------------------------

/// What one layer's writeback payload is and where it lands.
#[derive(Clone, Copy)]
enum WbKind {
    /// GDN layer: `[dn_state | conv_state]` → 2 handles (full slices).
    Dn { li: usize },
    /// Attention layer: `[k_rows | v_rows]` → 2 handles (row window
    /// `[base_pos, base_pos+p)` of each KV cache).
    Kv { li: usize },
}

struct WbPending {
    kind: WbKind,
    /// Payload length in f32 elements copied this chunk (≤ slot cap; KV
    /// slots carry only the p-row window, Dn slots the full state).
    len: usize,
    /// Second-region split point (dn_len for Dn, p*kvd for Kv).
    split: usize,
}

/// Cached (NOT write-combined) pinned f32 allocation via the low-level
/// driver API — the drain path READS this memory to feed the CubeCL writes
/// (WC reads would crawl). Freed by `WbSlot::drop`.
///
/// # Safety
/// Caller must keep the allocation alive until any copy targeting it has
/// completed, and free via `result::free_host`.
unsafe fn malloc_cached_pinned_f32(ctx: &Arc<CudaContext>, len: usize) -> Option<*mut f32> {
    let _ = ctx.bind_to_thread();
    let ptr = unsafe {
        cudarc::driver::result::malloc_host(len * 4, 0)
            .ok()?
            .cast::<f32>()
    };
    (!ptr.is_null()).then_some(ptr)
}

/// Grow-only per-layer pinned slots (one slot per layer, payload-shaped).
fn ensure_wb_slots(
    stack: &FullStack,
    fwd: &TernaryDeltanetGpuForward,
    n_v: usize,
    hd: usize,
    conv_dim: usize,
    ks: usize,
    kvd: usize,
) -> bool {
    let slots_lock = stack.wb_slots.get_or_init(|| {
        let ctx = stack.stream.context();
        let mut slots: Vec<Option<WbSlot>> = Vec::with_capacity(fwd.layer_types.len());
        for lt in &fwd.layer_types {
            let cap = if *lt == DeltaNetLayerType::DeltaNet {
                n_v * hd * hd + conv_dim * ks
            } else {
                2 * 4096 * kvd // max p (the arm gate caps p ≤ 4096)
            };
            let done = ctx.new_event(None).ok()?;
            let ptr = unsafe { malloc_cached_pinned_f32(ctx, cap)? };
            slots.push(Some(WbSlot { ptr, cap, done }));
        }
        Some(Mutex::new(slots))
    });
    slots_lock.is_some()
}

/// Enqueue one layer's async DtoHs onto the copy stream. Ordering: the copy
/// stream first waits for the compute stream's progress UP TO NOW (the
/// caller enqueues this right after the layer's last mirror-mutating
/// kernels), then copies, then records `done`.
///
/// # Safety
/// The mirror slices must not be mutated again until `done` completes — by
/// construction each mirror is written only by its own layer's kernels
/// within a chunk, and the next chunk re-establishes ordering via
/// `sync_states`/memset on the compute stream before any mirror read.
#[allow(clippy::too_many_arguments, reason = "mirror + slot plumbing")]
unsafe fn wb_enqueue_layer(
    stack: &FullStack,
    copy: &Arc<CudaStream>,
    order_ev: &CudaEvent,
    slots: &[Option<WbSlot>],
    pend: &WbPending,
    states: &StateMirrors,
    base_pos: usize,
    kvd: usize,
) -> Result<(), String> {
    let li = match pend.kind {
        WbKind::Dn { li } | WbKind::Kv { li } => li,
    };
    let Some(slot) = slots[li].as_ref() else {
        return Err("wb slot".into());
    };
    if slot.cap < pend.len {
        return Err("wb slot too small".into());
    }
    order_ev
        .record(&stack.stream)
        .map_err(|e| format!("wb order record: {e}"))?;
    copy.wait(order_ev).map_err(|e| format!("wb order wait: {e}"))?;
    let do_copy = |src: &CudaSlice<f32>, dst_off: usize, n: usize| -> Result<(), String> {
        let view = src.as_view();
        let (src_ptr, _guard) = view.device_ptr(copy);
        // SAFETY: the slot region `[dst_off, dst_off+n)` is within `cap`
        // (checked above) and no other copy targets it until `done` fires.
        let dst = unsafe { std::slice::from_raw_parts_mut(slot.ptr.add(dst_off), n) };
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(dst, src_ptr, copy.cu_stream())
        }
        .map_err(|e| format!("wb dtoh: {e}"))
    };
    match pend.kind {
        WbKind::Dn { li } => {
            let (Some(dn), Some(cv)) = (
                states.dn_states[li].as_ref(),
                states.conv_states[li].as_ref(),
            ) else {
                return Err("wb dn mirrors".into());
            };
            do_copy(dn, 0, pend.split)?;
            do_copy(cv, pend.split, pend.len - pend.split)?;
        }
        WbKind::Kv { li } => {
            let (Some(kc), Some(vc)) = (states.kv_k[li].as_ref(), states.kv_v[li].as_ref()) else {
                return Err("wb kv mirrors".into());
            };
            let start = base_pos * kvd;
            let rows = pend.len / 2;
            let Some(k_view) = kc.try_slice(start..start + rows) else {
                return Err("wb k view".into());
            };
            let Some(v_view) = vc.try_slice(start..start + rows) else {
                return Err("wb v view".into());
            };
            let (k_ptr, _k_guard) = k_view.device_ptr(copy);
            // SAFETY: KV slot regions `[0, rows)` / `[rows, 2*rows)` within cap.
            let dst_k = unsafe { std::slice::from_raw_parts_mut(slot.ptr, rows) };
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async(dst_k, k_ptr, copy.cu_stream())
            }
            .map_err(|e| format!("wb dtoh k: {e}"))?;
            let (v_ptr, _v_guard) = v_view.device_ptr(copy);
            let dst_v = unsafe { std::slice::from_raw_parts_mut(slot.ptr.add(rows), rows) };
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async(dst_v, v_ptr, copy.cu_stream())
            }
            .map_err(|e| format!("wb dtoh v: {e}"))?;
        }
    }
    slot.done.record(copy).map_err(|e| format!("wb done: {e}"))?;
    Ok(())
}

/// Drain one completed slot into the CubeCL handles (non-blocking check;
/// `force` blocks on the done event — chunk end / failure flush).
#[allow(clippy::too_many_arguments, reason = "mirror + slot plumbing")]
fn wb_drain_one(
    client: &ComputeClient<ActiveRuntime>,
    slots: &[Option<WbSlot>],
    pend: &WbPending,
    fwd: &TernaryDeltanetGpuForward,
    base_pos: usize,
    kvd: usize,
    force: bool,
) -> Result<bool, String> {
    let li = match pend.kind {
        WbKind::Dn { li } | WbKind::Kv { li } => li,
    };
    let Some(slot) = slots[li].as_ref() else {
        return Err("drain slot".into());
    };
    if !force && !slot.done.is_complete() {
        return Ok(false);
    }
    if force {
        slot.done
            .synchronize()
            .map_err(|e| format!("drain sync: {e}"))?;
    }
    // SAFETY: the done event fired — the pinned payload is stable.
    let payload = unsafe { std::slice::from_raw_parts(slot.ptr, pend.len) };
    let bytes = |src: &[f32]| {
        cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(src).to_vec())
    };
    match pend.kind {
        WbKind::Dn { li } => {
            let (Some(dn_h), Some(cv_h)) = (
                fwd.deltanet_states[li].as_ref(),
                fwd.conv_states[li].as_ref(),
            ) else {
                return Err("drain dn handles".into());
            };
            client.write(dn_h, bytes(&payload[..pend.split]));
            client.write(cv_h, bytes(&payload[pend.split..pend.len]));
        }
        WbKind::Kv { li } => {
            let (Some(kc), Some(vc)) = (
                fwd.kv_key_caches[li].as_ref(),
                fwd.kv_value_caches[li].as_ref(),
            ) else {
                return Err("drain kv handles".into());
            };
            let rows = pend.len / 2;
            let f = core::mem::size_of::<f32>();
            let total_rows = fwd.config.block_size * kvd;
            let trim = ((total_rows - (base_pos * kvd) - rows) * f) as u64;
            let mk = |cache: &Handle, src: &[f32]| {
                let dst = cache
                    .clone()
                    .offset_start(((base_pos * kvd) * f) as u64)
                    .offset_end(trim);
                client.write(&dst, bytes(src));
            };
            mk(kc, &payload[..rows]);
            mk(vc, &payload[rows..pend.len]);
        }
    }
    Ok(true)
}

/// Arm 13 — allocate (once, grow-free — mirrors are fixed-size) the
/// device-side pre-chunk snapshot. Returns false if the VRAM is not
/// available (the arm refuses the chunk BEFORE any mutation → clean
/// fall-through).
fn ensure_wb_snapshot(
    stack: &FullStack,
    fwd: &TernaryDeltanetGpuForward,
    n_v: usize,
    hd: usize,
    conv_dim: usize,
    ks: usize,
    kvd: usize,
) -> bool {
    stack
        .wb_snapshot
        .get_or_init(|| {
            build_states(&stack.stream, fwd, n_v, hd, conv_dim, ks, kvd).map(Mutex::new)
        })
        .is_some()
}

/// Device-side snapshot of every mirror (dtod, on the COMPUTE stream —
/// ordered before this chunk's kernels by stream order).
fn wb_snapshot_take(
    stack: &FullStack,
    states: &StateMirrors,
    snap: &mut StateMirrors,
) -> Result<(), String> {
    let cp = |src: &Option<CudaSlice<f32>>,
              dst: &mut Option<CudaSlice<f32>>|
     -> Result<(), String> {
        if let (Some(s), Some(d)) = (src, dst) {
            stack
                .stream
                .memcpy_dtod(s, d)
                .map_err(|e| format!("snap dtod: {e}"))?;
        }
        Ok(())
    };
    for (src, dst) in states.dn_states.iter().zip(snap.dn_states.iter_mut()) {
        cp(src, dst)?;
    }
    for (src, dst) in states.conv_states.iter().zip(snap.conv_states.iter_mut()) {
        cp(src, dst)?;
    }
    for (src, dst) in states.kv_k.iter().zip(snap.kv_k.iter_mut()) {
        cp(src, dst)?;
    }
    for (src, dst) in states.kv_v.iter().zip(snap.kv_v.iter_mut()) {
        cp(src, dst)?;
    }
    Ok(())
}

/// Failure repair: restore every mirror from the snapshot (dtod), then
/// push the restored values to the CubeCL handles via the LEGACY sync
/// writeback — leaving the CubeCL side at the pre-chunk state so the
/// caller's fall-through recompute starts from the correct base (exactly
/// today's no-writeback-on-failure semantics). The copy stream is quiesced
/// first (its in-flight DtoHs read the mirrors).
fn wb_restore_on_failure(
    stack: &FullStack,
    client: &ComputeClient<ActiveRuntime>,
    states: &mut StateMirrors,
    fwd: &TernaryDeltanetGpuForward,
    p: usize,
    base_pos: usize,
    kvd: usize,
) {
    if let Some(copy) = stack.copy_stream.as_ref() {
        // Quiesce the copy stream (waits for in-flight DtoHs only).
        let _ = copy.synchronize();
    }
    let Some(snap_lock) = stack.wb_snapshot.get().and_then(|s| s.as_ref()) else {
        return;
    };
    let Ok(snap) = snap_lock.lock() else { return };
    let cp = |src: &Option<CudaSlice<f32>>,
              dst: &mut Option<CudaSlice<f32>>|
     -> Result<(), String> {
        if let (Some(s), Some(d)) = (src, dst) {
            stack
                .stream
                .memcpy_dtod(s, d)
                .map_err(|e| format!("restore dtod: {e}"))?;
        }
        Ok(())
    };
    for (src, dst) in snap.dn_states.iter().zip(states.dn_states.iter_mut()) {
        let _ = cp(src, dst);
    }
    for (src, dst) in snap.conv_states.iter().zip(states.conv_states.iter_mut()) {
        let _ = cp(src, dst);
    }
    for (src, dst) in snap.kv_k.iter().zip(states.kv_k.iter_mut()) {
        let _ = cp(src, dst);
    }
    for (src, dst) in snap.kv_v.iter().zip(states.kv_v.iter_mut()) {
        let _ = cp(src, dst);
    }
    // Wait for the restores, then sync-writeback the whole mirror set.
    let _ = stack.stream.synchronize();
    let mut state_host = Vec::new();
    let mut kv_host = Vec::new();
    // Issue 742 T1.1 — in spec mode the CubeCL handles are stale from ALL
    // spec chunks (each elided its writeback), so the repair must flush the
    // full [0, base_pos+p) KV range + states, not just this chunk's window
    // — otherwise the fall-through recomputes from a pre-spec state.
    let (wb_base, wb_p) = if prefill_spec_mode() {
        stack.spec_dirty.store(false, Ordering::Relaxed);
        (0, base_pos + p)
    } else {
        (base_pos, p)
    };
    let _ = writeback_states(
        client,
        &stack.stream,
        fwd,
        states,
        wb_base,
        wb_p,
        kvd,
        &mut state_host,
        &mut kv_host,
    );
}

// ---------------------------------------------------------------------------
// Arm 13 — sync elision hooks (reset memset / mutation dirty)
// ---------------------------------------------------------------------------

/// `TernaryDeltanetGpuForward::reset_state` calls this: the CubeCL state
/// handles were just re-initialized, so the mirrors are brought to the same
/// values DEVICE-SIDE (memset on the compute stream — no PCIe crossing)
/// and both flags are set to skip the next prompt's `sync_states` upload.
pub(crate) fn notify_cubcl_reset() {
    let Some(stack) = FULL_STACK.get().and_then(|s| s.as_ref()) else {
        return;
    };
    let Some(states_lock) = stack.states.get().and_then(|s| s.as_ref()) else {
        return;
    };
    let Ok(mut states) = states_lock.lock() else {
        return;
    };
    let zero = |v: &mut Option<CudaSlice<f32>>| {
        if let Some(s) = v.as_mut() {
            let _ = stack.stream.memset_zeros(s);
        }
    };
    for v in states.dn_states.iter_mut() {
        zero(v);
    }
    for v in states.conv_states.iter_mut() {
        zero(v);
    }
    for v in states.kv_k.iter_mut() {
        zero(v);
    }
    for v in states.kv_v.iter_mut() {
        zero(v);
    }
    stack.mirrors_at_reset.store(true, Ordering::Relaxed);
    stack.mirror_clean.store(true, Ordering::Relaxed);
    // A reset re-initializes both sides consistently — no spec-mode
    // pending writeback can exist past it.
    stack.spec_dirty.store(false, Ordering::Relaxed);
}

/// Any CubeCL-side mutation of the state handles (decode via
/// `forward_from_x`, the CubeCL chunk body when the arm is off) calls this:
/// the mirrors are stale until the next `sync_states` upload.
pub(crate) fn notify_cubcl_mutated() {
    let Some(stack) = FULL_STACK.get().and_then(|s| s.as_ref()) else {
        return;
    };
    stack.mirrors_at_reset.store(false, Ordering::Relaxed);
    stack.mirror_clean.store(false, Ordering::Relaxed);
}

/// Issue 965 — invalidate any pending spec flush WITHOUT applying it, and
/// mark the mirrors stale. This is the state-REWINDER hook (the speculative
/// rollbacks): they move the CubeCL handles backward in time, and a pending
/// flush applied AFTER a rewind would resurrect the future mirror state over
/// the rewound handles. Discard (not flush) is the correct verb here — the
/// next arm chunk re-syncs the mirrors from the rewound handles.
pub fn prefill_spec_discard() {
    let Some(stack) = FULL_STACK.get().and_then(|s| s.as_ref()) else {
        return;
    };
    stack.spec_dirty.store(false, Ordering::Relaxed);
    stack.spec_pos.store(0, Ordering::Relaxed);
    stack.mirrors_at_reset.store(false, Ordering::Relaxed);
    stack.mirror_clean.store(false, Ordering::Relaxed);
}

/// Run the whole prefill chunk on the cudarc stack. Returns `Some(logits)`
/// (empty for non-final chunks) on success, `None` to fall through to the
/// CubeCL body. See the module doc for the gate + mirror design.
///
/// # Panics (debug)
///
/// Debug-asserts mirror shapes; production failures return `None`
/// (fall-through — bit-safe).
#[allow(
    clippy::too_many_lines,
    reason = "the whole-prefill dispatch mirrors the CubeCL layer loop stage by stage"
)]
pub(crate) fn try_whole_prefill_cuda(
    fwd: &TernaryDeltanetGpuForward,
    tokens: &[usize],
    base_pos: usize,
    is_final: bool,
) -> Option<Vec<f32>> {
    match whole_prefill_inner(fwd, tokens, base_pos, is_final, false, None) {
        Some(WholeOut::Logits(v)) => Some(v),
        // Unreachable: `verify_tail` is false here.
        Some(WholeOut::Argmax(_)) | Some(WholeOut::ArgmaxNll(_, _)) => None,
        None => None,
    }
}

/// The per-position verify tail's output: the argmax token id at every one
/// of the chunk's `p` positions (row `i` predicts the token at
/// `base_pos + i + 1`).
type VerifyArgmax = Vec<u32>;

enum WholeOut {
    /// Final-row logits (`is_final` regular tail) — empty for non-final
    /// chunks.
    Logits(Vec<f32>),
    /// Issue 742 T1.8 — per-position argmax ids (the verify tail).
    Argmax(VerifyArgmax),
    /// Issue 884 T2a — per-position argmax ids AND negative log-likelihoods
    /// (the verify tail + the NLL gather; row `i` scores the target at
    /// `base_pos + i + 1`).
    ArgmaxNll(VerifyArgmax, Vec<f32>),
}

/// Issue 742 T1.8 — the verify chunk: the whole-prefill chunk (embed +
/// layers, graph-replay eligible at the capture `p`) PLUS the per-position
/// verify tail — a `p`-row final-norm + lm_head GEMM + device-side per-row
/// argmax, all on the cudarc side, downloading `p * 4` bytes of token ids.
/// The chunk state semantics are exactly a spec-mode chunk's (writeback
/// elided; `spec_pos = base_pos + p`), and the pre-chunk snapshot is taken
/// EVEN for graph replays (the verify loop's rollback consumer).
#[allow(
    clippy::too_many_lines,
    reason = "the whole-prefill dispatch mirrors the CubeCL layer loop stage by stage"
)]
fn whole_prefill_inner(
    fwd: &TernaryDeltanetGpuForward,
    tokens: &[usize],
    base_pos: usize,
    is_final: bool,
    verify_tail: bool,
    nll_targets: Option<&[usize]>,
) -> Option<WholeOut> {
    #[derive(Default)]
    struct StageTimes {
        gemm: u128,
        rms: u128,
        gdn: u128,
        attn: u128,
        other: u128,
    }
#[derive(Default)]
    struct GdnSubTimes {
        conv: u128,
        carry: u128,
        beta: u128,
        expand: u128,
        rec: u128,
        lnorm: u128,
        zgate: u128,
    }

if !prefill_use_cuda() {
        return None;
    }
    if !crate::ternary_deltanet_gpu_forward::prefill_cuda_gate_ok() {
        return None;
    }
    let cfg = &fwd.config;
    let p = tokens.len();
    let n = cfg.n_embd;
    let n_v = cfg.deltanet_linear_n_value_heads;
    let hd = cfg.deltanet_linear_head_dim;
    let n_k = cfg.deltanet_linear_n_heads;
    let q_dim = n_k * hd;
    let v_dim = n_v * hd;
    let qkv_dim = 2 * q_dim + v_dim;
    let conv_dim = qkv_dim;
    let ks = cfg.deltanet_conv_kernel_size;
    let mlp = cfg.mlp_hidden;
    let n_head = cfg.n_head;
    let n_kv = cfg.n_kv_head;
    let ahd = cfg.head_dim;
    let qa = n_head * ahd;
    let kvd = n_kv * ahd;
    let eps = cfg.rms_norm_eps as f32;
    if p == 0 || p > 4096 {
        return None;
    }
    if hd != 128 {
        return None;
    }
    if !n.is_multiple_of(128)
        || !mlp.is_multiple_of(128)
        || !v_dim.is_multiple_of(128)
        || !qa.is_multiple_of(128)
    {
        return None;
    }

    let trace = trace_enabled();
    let t0 = std::time::Instant::now();

    let Some(stack) = full_stack() else {
        if trace {
            eprintln!("[734-arm8] FALLTHROUGH: stack init");
        }
        return None;
    };
    let was_clean = stack.mirror_clean.load(Ordering::Relaxed);
    let was_at_reset = stack.mirrors_at_reset.load(Ordering::Relaxed);
    stack.mirror_clean.store(false, Ordering::Relaxed);
    stack.mirrors_at_reset.store(false, Ordering::Relaxed);

    let client = &fwd.client;
    let stream = &stack.stream;

    // ── Mirrors ──
    let f32w = stack
        .f32_weights
        .get_or_init(|| build_f32_weights(client, stream, fwd).map(Arc::new))
        .clone();
    let Some(f32w) = f32w else {
        if trace {
            eprintln!("[734-arm8] FALLTHROUGH: f32 weights");
        }
        return None;
    };
    // ── Issue 980 T4-ALT — the folded-prefill rotation tables ──
    // Built ONCE from the first folded forward's config (the f32_weights
    // first-fwd-wins pattern). A folded model whose build failed falls
    // through here → the `prefill_tokens_chunk` fall-through gate PANICS
    // (never silently unrotated).
    let rotation: Option<Arc<crate::deltanet_rotation_cudarc::RotationTables>> = stack
        .rotation_tables
        .get_or_init(|| {
            fwd.rotation.as_ref().and_then(|cfg| {
                crate::deltanet_rotation_cudarc::RotationTables::build(
                    stream.context(),
                    stream,
                    cfg,
                )
                .ok()
                .map(Arc::new)
            })
        })
        .clone();
    if fwd.rotation.is_some() && rotation.is_none() {
        if trace {
            eprintln!("[734-arm8] FALLTHROUGH: rotation tables build failed");
        }
        return None; // the fall-through gate errors loudly for folded
    }
    // Spec-verify refuses folded models (Issue 980 T4-ALT): the verify
    // drivers' draft/rollback discipline is not folded-wired — the whole
    // prefill chunk above is, but the per-position tail's consumers are not.
    // Loud refusal at the single entry every verify caller funnels through.
    if verify_tail && fwd.rotation.is_some() {
        panic!(
            "spec-verify tail on a Hadamard-folded model: not wired \
             (Issue 980 T4-ALT) — the verify drivers must refuse folded models"
        );
    }
    let states_arc = stack
        .states
        .get_or_init(|| build_states(stream, fwd, n_v, hd, conv_dim, ks, kvd).map(Mutex::new));
    let Some(states_lock) = states_arc else {
        if trace {
            eprintln!("[734-arm8] FALLTHROUGH: state alloc");
        }
        return None;
    };
    let Ok(mut states_guard) = states_lock.lock() else { return None };
    let states: &mut StateMirrors = &mut states_guard;
    let Some(wte) = mma_mirror_pair(client, stream, &fwd.wte_handle) else {
        if trace {
            eprintln!("[734-arm8] FALLTHROUGH: wte mirror");
        }
        return None;
    };

    // ── State sync (dirty mirror, or a fresh prompt that has NOT been
    //    device-side reset-memset — Arm 13 elides the unconditional
    //    base_pos==0 upload when `notify_cubcl_reset` already matched the
    //    mirrors to the reset CubeCL handles) ──
    if (!was_clean || (base_pos == 0 && !was_at_reset))
        && !sync_states(client, stream, fwd, states, base_pos, kvd)
    {
        if trace {
            eprintln!("[734-arm8] FALLTHROUGH: state sync");
        }
        return None;
    }
    let t_sync = t0.elapsed();

    // ── Arm 13 — overlapped writeback setup ──
    // Issue 742 T1.1 — spec mode elides the per-chunk writeback crossing
    // entirely (the T1.0 measurement: it is ~85% of the per-chunk fixed
    // cost at verify-sized chunks). The snapshot stays ON (rollback
    // substrate); the pinned slots are not even allocated.
    let spec = prefill_spec_mode() || graphs_armed();
    if !spec && stack.spec_dirty.swap(false, Ordering::Relaxed) {
        // Transition flush: spec chunks advanced the mirrors past the
        // CubeCL handles — bring CubeCL current BEFORE this chunk runs so
        // any later decode/fall-through sees a consistent state.
        let pos = stack.spec_pos.swap(0, Ordering::Relaxed).max(base_pos);
        let mut state_host = Vec::new();
        let mut kv_host = Vec::new();
        if !writeback_states(
            client,
            stream,
            fwd,
            states,
            0,
            pos,
            kvd,
            &mut state_host,
            &mut kv_host,
        ) {
            if trace {
                eprintln!("[734-arm8] FALLTHROUGH: spec transition flush");
            }
            return None;
        }
    }
    let Some(copy_stream) = stack.copy_stream.clone() else {
        if trace {
            eprintln!("[734-arm8] FALLTHROUGH: copy stream");
        }
        return None;
    };
    if !spec && !ensure_wb_slots(stack.as_ref(), fwd, n_v, hd, conv_dim, ks, kvd) {
        if trace {
            eprintln!("[734-arm8] FALLTHROUGH: wb slots");
        }
        return None;
    }
    // Issue 742 T1.5 — decide the replay path EARLY: a chunk that will be
    // served by a graph REPLAY needs neither the rollback snapshot nor the
    // wb order event (spec mode never enqueues wb; a replay launch error
    // leaves the mirrors untouched — the failure path flushes instead of
    // restoring). This is the per-chunk fixed-cost cut that lets the replay
    // actually bank the launch-overhead saving (the probe measured the
    // graph alone at 31.6 ms; the eager surroundings were eating it).
    let graphs = graphs_armed();
    let gv_active = (gv_armed() || graphs) && spec;
    let knob_ok = stack.gv_knob_gen.load(Ordering::Relaxed) == GV_KNOB_GEN.load(Ordering::Relaxed);
    let will_replay = gv_active
        && knob_ok
        && stack.graph_verify.get().is_some_and(|g| g.is_some())
        && stack.gv_p.load(Ordering::Relaxed) == p
        && stack.gv_staging_gen.load(Ordering::Relaxed)
            == stack
                .bufs
                .lock().map_or(usize::MAX, |b| b.staging_gen);
    // Issue 742 T1.8 — a verify chunk ALWAYS carries the rollback substrate:
    // the verify loop's accept/rollback consumer needs the pre-chunk state
    // even when the chunk itself is served by a graph REPLAY (the T1.5
    // replay-skip predates the verify loop's rollback consumer).
    let want_snapshot = !will_replay || verify_tail;
    if want_snapshot && !ensure_wb_snapshot(stack.as_ref(), fwd, n_v, hd, conv_dim, ks, kvd) {
        if trace {
            eprintln!("[734-arm8] FALLTHROUGH: wb snapshot VRAM");
        }
        return None;
    }
    let order_ev = if !will_replay {
        Some(stack.stream.context().new_event(None).ok()?)
    } else {
        None
    };
    if want_snapshot {
        let snap_lock = stack.wb_snapshot.get().and_then(|s| s.as_ref())?;
        let Ok(mut snap) = snap_lock.lock() else { return None };
        if wb_snapshot_take(stack.as_ref(), states, &mut snap).is_err() {
            if trace {
                eprintln!("[734-arm8] FALLTHROUGH: snapshot");
            }
            return None;
        }
    }

    // ── Tokens + embedding ──
    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    // Issue 742 T1.5 — when graph-verify is armed the token + position
    // payloads ride PERSISTENT device buffers (created in the grow block
    // below; `pos_dev`'s address is baked into the captured graph). The
    // device reference is materialized after the staging destructure.

    // ── Grow-only staging ──
    let max_gemm_n = n.max(v_dim).max(mlp);
    let words = p * (max_gemm_n / 4);
    // Arm-12 knob resolution (single source for grow + dispatch).
    let gdn_chunked_wanted = gdn_chunked_enabled() && stack.gdn_chunked.is_some();
    let Ok(mut bufs) = stack.bufs.lock() else {
        return None;
    };
    {
        macro_rules! grow {
            ($field:ident, $len:expr) => {
                if bufs.$field.as_ref().is_none_or(|s| s.len() < $len) {
                    bufs.$field = stream.alloc_zeros::<f32>($len).ok();
                    bufs.staging_gen += 1;
                }
                bufs.$field.as_ref()?;
            };
        }
        grow!(x, p * n);
        grow!(normx, p * n);
        grow!(qkv_b, p * qkv_dim);
        grow!(qkv_conv, p * qkv_dim);
        grow!(qkvx_b, p * 3 * v_dim);
        grow!(z_b, p * v_dim);
        grow!(a_b, p * n_v);
        grow!(b_b, p * n_v);
        grow!(beta_b, p * n_v);
        grow!(decay_b, p * n_v);
        grow!(rec_b, p * v_dim);
        grow!(tmp_b, p * n);
        grow!(gate_b, p * mlp);
        grow!(up_b, p * mlp);
        grow!(hid_b, p * mlp);
        grow!(ffnout_b, p * n);
        grow!(qg_b, p * 2 * qa);
        grow!(q_b, p * qa);
        grow!(agate_b, p * qa);
        grow!(kv_b, p * 2 * kvd);
        grow!(k_b, p * kvd);
        grow!(v_b, p * kvd);
        grow!(attn_out_b, p * qa);
        grow!(aproj_b, p * n);
        // Issue 980 T4-ALT — folded-prefill rotation staging (grow-only;
        // stable addresses for the 965 prefill graphs — growth bumps
        // staging_gen and re-captures exactly like every other buffer).
        // C0.5 (session 2 fix): rot_scratch must cover the WIDEST tabled
        // rotated width, not p×n — the unfused kill-switch lane rotates
        // inputs at 5120 AND 6144 (attn-wo/ssm_out) through this ONE
        // buffer, and the C0 `p*n` sizing asserted/crashed at the 6144 site
        // (16×5120=81920 vs 16×6144=98304, `g1_prefill_rotation_vs_cpu`
        // without prefill_q8_act — the only lane that runs that path; the
        // fused lane never touches rot_scratch at the narrow sites).
        if rotation.is_some() {
            let rot_w_max = rotation
                .as_ref()
                .and_then(|r| r.signs.iter().map(|(w, _)| *w).max())
                .unwrap_or(n)
                .max(n);
            grow!(rot_scratch, p * rot_w_max);
            grow!(permute_tmp, p * v_dim);
        }
        if gdn_chunked_wanted {
            let hc = n_v * CudaGdnChunkedKernels::n_chunks(p);
            grow!(gdn_lg, hc * 64);
            grow!(gdn_ga, hc * 64);
            grow!(gdn_dte, hc * 64);
            grow!(gdn_td, hc);
            grow!(gdn_x, hc * 4096);
            grow!(gdn_t, hc * 4096);
            grow!(gdn_qkr, hc * 4096);
        }
        if bufs
            .scratch
            .as_ref()
            .is_none_or(|(cw, cp, _)| *cw < words || *cp < p)
        {
            bufs.scratch = stack
                .mma
                .alloc_scratch(stream, max_gemm_n, p)
                .ok()
                .map(|s| (words, p, s));
            bufs.staging_gen += 1;
        }
        bufs.scratch.as_ref()?;
        // Issue 742 T1.5 — the graph-verify persistent payloads. `pos_dev`
        // is allocated ONCE (its address is baked into the captured graph);
        // `tokens_dev` may grow with `p` (the eager embed reads it fresh each
        // chunk — its address is never baked). Issue 965: graph mode (not
        // just spec+verify) needs the same persistent pair.
        if gv_active {
            if bufs.tokens_dev.as_ref().is_none_or(|s| s.len() < p) {
                bufs.tokens_dev = stream.alloc_zeros::<u32>(p).ok();
                bufs.staging_gen += 1;
            }
            if bufs.pos_dev.is_none() {
                bufs.pos_dev = stream.alloc_zeros::<i32>(1).ok();
                bufs.staging_gen += 1;
            }
        }
        if bufs.x_row_host.len() < n {
            bufs.x_row_host.resize(n, 0.0);
        }
        // Issue 742 T1.8 — verify-tail staging, grown in the chunk preamble
        // (BEFORE any capture region — a post-capture realloc would bump
        // `staging_gen` and orphan the graph-verify capture for good).
        if verify_tail {
            let vocab = fwd.config.vocab_size;
            if bufs.verify_logits.as_ref().is_none_or(|s| s.len() < p * vocab) {
                bufs.verify_logits = stream.alloc_zeros::<f32>(p * vocab).ok();
                bufs.staging_gen += 1;
            }
            if bufs.verify_argmax.as_ref().is_none_or(|s| s.len() < p) {
                bufs.verify_argmax = stream.alloc_zeros::<u64>(p).ok();
                bufs.staging_gen += 1;
            }
            if bufs.verify_argmax_host.len() < p {
                bufs.verify_argmax_host.resize(p, 0);
            }
            // Issue 884 T2a — NLL staging (only when the caller wants the
            // NLL tail): the [2p] device pairs + the host landing vec.
            if nll_targets.is_some() {
                if bufs.verify_nll.as_ref().is_none_or(|s| s.len() < 2 * p) {
                    bufs.verify_nll = stream.alloc_zeros::<f32>(2 * p).ok();
                    bufs.staging_gen += 1;
                }
                if bufs.verify_nll_host.len() < 2 * p {
                    bufs.verify_nll_host.resize(2 * p, 0.0);
                }
            }
        }
    }
    let t_alloc = t0.elapsed();

    // Destructure the staging (all Some by now).
    let FullBufs {
        x,
        normx,
        qkv_b,
        qkv_conv,
        qkvx_b,
        z_b,
        a_b,
        b_b,
        beta_b,
        decay_b,
        rec_b,
        tmp_b,
        gate_b,
        up_b,
        hid_b,
        ffnout_b,
        qg_b,
        q_b,
        agate_b,
        kv_b,
        k_b,
        v_b,
        attn_out_b,
        aproj_b,
        gdn_lg,
        gdn_ga,
        gdn_dte,
        gdn_td,
        gdn_x,
        gdn_t,
        gdn_qkr,
        scratch,
        staging_gen,
        tokens_dev,
        pos_dev,
        attn_split_pm,
        attn_split_pl,
        attn_split_po,
        attn_split_chunks,
        attn_fa_kh,
        attn_fa_vh,
        attn_fa_rows,
        rot_scratch,
        permute_tmp,
        x_row_host,
        verify_logits,
        verify_argmax,
        verify_argmax_host,
        verify_nll,
        verify_nll_host,
    } = &mut *bufs;
    let (Some(x), Some(normx), Some(qkv_b), Some(qkv_conv), Some(qkvx_b), Some(z_b)) =
        (x, normx, qkv_b, qkv_conv, qkvx_b, z_b)
    else {
        return None;
    };
    let (Some(a_b), Some(b_b), Some(beta_b), Some(decay_b), Some(rec_b), Some(tmp_b)) =
        (a_b, b_b, beta_b, decay_b, rec_b, tmp_b)
    else {
        return None;
    };
    let (Some(gate_b), Some(up_b), Some(hid_b), Some(ffnout_b), Some(qg_b), Some(q_b)) =
        (gate_b, up_b, hid_b, ffnout_b, qg_b, q_b)
    else {
        return None;
    };
    let (Some(agate_b), Some(kv_b), Some(k_b), Some(v_b), Some(attn_out_b), Some(aproj_b)) =
        (agate_b, kv_b, k_b, v_b, attn_out_b, aproj_b)
    else {
        return None;
    };
    // Issue 980 T4-ALT — rotation staging: `Some` only on folded runs (the
    // preamble grew them iff `rotation.is_some()`; a None here on a folded
    // run is an alloc failure the earlier `?`s already turned into a
    // fall-through).
    let (rot_scratch, permute_tmp): (Option<&CudaSlice<f32>>, Option<&CudaSlice<f32>>) =
        match (&rotation, rot_scratch.as_ref(), permute_tmp.as_ref()) {
            (Some(_), rs, pt) => (rs, pt),
            (None, _, _) => (None, None),
        };
    // Issue 742 T1.8 — verify-tail staging (Some iff `verify_tail` grew them
    // in the preamble; the argmax host buffer is a plain Vec).
    let verify_logits: Option<&mut CudaSlice<f32>> = if verify_tail {
        verify_logits.as_mut()
    } else {
        None
    };
    let verify_argmax: Option<&mut CudaSlice<u64>> = if verify_tail {
        verify_argmax.as_mut()
    } else {
        None
    };
    // Issue 884 T2a — Some iff the caller wants the NLL tail AND the
    // staging grew.
    let verify_nll: Option<&mut CudaSlice<f32>> = if verify_tail && nll_targets.is_some() {
        verify_nll.as_mut()
    } else {
        None
    };
    // Arm-12 scratch: Option<&CudaSlice> — Some iff wanted AND grown.
    let (gdn_lg, gdn_ga, gdn_dte, gdn_td, gdn_x, gdn_t, gdn_qkr) = (
        gdn_lg.as_ref(),
        gdn_ga.as_ref(),
        gdn_dte.as_ref(),
        gdn_td.as_ref(),
        gdn_x.as_ref(),
        gdn_t.as_ref(),
        gdn_qkr.as_ref(),
    );
    let gdn_chunked_ready = gdn_chunked_wanted
        && matches!(
            (gdn_lg, gdn_ga, gdn_dte, gdn_td, gdn_x, gdn_t, gdn_qkr),
            (Some(_), Some(_), Some(_), Some(_), Some(_), Some(_), Some(_))
        );
    let (_, _, scratch) = scratch.as_ref()?;

    // Issue 742 T1.5 — materialize the token device reference + (armed) the
    // pos upload. Both uploads are eager stream ops ordered before the
    // embed launch and any graph replay in this chunk.
    let gv = gv_active;
    let tokens_fresh: Option<CudaSlice<u32>> = if gv {
        None
    } else {
        Some(stream.clone_htod(tokens_u32.as_slice()).ok()?)
    };
    let pos_dev_ref: Option<&CudaSlice<i32>> = if gv {
        let td = tokens_dev.as_mut()?;
        {
            let mut view = td.try_slice_mut(0..tokens_u32.len())?;
            stream.memcpy_htod(tokens_u32.as_slice(), &mut view).ok()?;
        }
        let bp_i = [base_pos as i32];
        stream
            .memcpy_htod(bp_i.as_slice(), pos_dev.as_mut()?)
            .ok()?;
        pos_dev.as_ref()
    } else {
        None
    };

    // Issue 742 T1.7 — split-KV attention scratch (arm 3): allocated
    // EAGERLY in the chunk preamble — allocation during the graph-capture
    // region is illegal, and the CAPTURE chunk is typically the FIRST
    // qualifying chunk (the wide warm-up at p=2048 is over the split's
    // qtile cap and skips it), so the capture chunk's layer loop must find
    // the scratch already resident. Sized ONCE (qtile cap × n_chunks_max
    // from block_size/chunk_len — both process-constant) so the addresses
    // never move; a realloc would orphan the captured graph (the
    // staging_gen guard catches any re-size).
    if attn_split_engaged(attn_arm(), n_head, p) {
        let chunk_len = attn_split_chunk();
        let n_chunks_max = fwd.config.block_size.div_ceil(chunk_len);
        if *attn_split_chunks != n_chunks_max
            || attn_split_pm.is_none()
            || attn_split_pl.is_none()
            || attn_split_po.is_none()
        {
            let ml = attn_split_qtile_cap(n_head) * 8 * n_chunks_max;
            *attn_split_pm = stream.alloc_zeros::<f32>(ml).ok();
            *attn_split_pl = stream.alloc_zeros::<f32>(ml).ok();
            *attn_split_po = stream.alloc_zeros::<f32>(ml * ahd).ok();
            *attn_split_chunks = n_chunks_max;
            *staging_gen += 1;
        }
    }
    // Plan 605 T2 — the FA-class f16 KV scratch. Sized ONCE at block_size
    // (process-constant; the attn_split discipline — allocated in the
    // chunk preamble BEFORE any capture region so the capture chunk's
    // layer loop finds it resident; stable addresses forever). Engaged by
    // `fa_engaged()` alone (NOT the p-predicate — the scratch is sized at
    // block_size once; a small-p chunk in a large block keeps the mirrors
    // resident for the chunks that do qualify).
    if fa_engaged() && ahd == 256 && n_head == 6 * n_kv {
        let rows_max = crate::prefill_cuda_attention_fa::fa_scratch_rows(fwd.config.block_size);
        if *attn_fa_rows != rows_max || attn_fa_kh.is_none() || attn_fa_vh.is_none() {
            let elems = rows_max * kvd;
            *attn_fa_kh = stream.alloc_zeros::<u16>(elems).ok();
            *attn_fa_vh = stream.alloc_zeros::<u16>(elems).ok();
            *attn_fa_rows = rows_max;
            *staging_gen += 1;
        }
    }
    let tokens_dev_ref: &CudaSlice<u32> = if let Some(t) = tokens_fresh.as_ref() {
        t
    } else {
        tokens_dev.as_ref()?
    };

    // ── The canonical numerics forms ──
    let (ra, rm, rs) = crate::prefill_cuda_ffn::canonical_rmsnorm_forms();
    let (se, sr) = crate::prefill_cuda_ffn::canonical_swiglu_forms();
    let (rd, ru, rp, rsc) = crate::prefill_cuda_deltanet::canonical_recurrence_forms();
    let (cf, ce, cd) = canonical_conv_forms();
    let (bl, be, bd) = canonical_beta_decay_forms();
    let (ea, ei) = crate::prefill_cuda_deltanet::canonical_expand_forms();
    let (ze, zd) = canonical_zgate_forms();
    let (rl, re, rsc2, rr) = canonical_rope_forms();
    let (ad, aa, ae, aph, ars, ai) = canonical_attention_forms();
    let _ = (
        ra, rm, rs, se, sr, rd, ru, rp, rsc, cf, ce, cd, bl, be, bd, ea, ei, ze, zd, rl, re,
        rsc2, rr, ad, aa, ae, aph, ars, ai,
    );

    // ── Embedding ──
    // Issue 742 T1.5 — the embed is a plain kernel over address-stable
    // buffers (`x` staging + the persistent `tokens_dev`), so under
    // graph-verify it is CAPTURED INTO the graph (the whole chunk becomes ONE
    // graph launch). Eager paths (warm-up, capture-fallback, non-armed) run
    // it exactly as before.
    let embed = || -> Result<(), String> {
        unsafe {
            stack.dn.launch_dequant_wte_batch(
                stream,
                wte.pos.as_ref().expect("wte mirror carries the bitplane pair"),
                wte.neg.as_ref().expect("wte mirror carries the bitplane pair"),
                &wte.scale,
                x,
                tokens_dev_ref,
                fwd.wte_handle.blocks64,
                fwd.wte_handle.groups_per_row,
                n,
                p,
            )?;
        }
        // Issue 980 T4-ALT site (a) — a Hadamard-latent embedding table
        // stores ROTATED rows; restore the primal basis right after the
        // lookup (Hadamard first, sign second — the decode twin's
        // `fwht_rotate_inverse` guard, batched over the p rows, in place on
        // `x`). Rides INSIDE the captured region on the gv lanes (a plain
        // kernel node over address-stable buffers).
        if let Some(rot) = &rotation
            && rot.inverse_embedding
        {
            rot.kernels
                .fwht_rotate_inverse_batched(
                    stream,
                    x,
                    rot.signs_for_width(n),
                    p,
                    n,
                    rot.block_size,
                )
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    let t_embed;

    // ── Layer loop ──
    // TRACE=2: sync + accumulate per kernel class (distorts total; localizes
    // the GPU-time attribution). TRACE=3 additionally splits the GDN stage
    // per kernel (issue 772 T2 decomposition probe; same distortion class).
    let trace_level = std::env::var("RIIR_PREFILL_CUDA_TRACE")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let stage_sync = matches!(trace_level.as_str(), "2" | "3");
    let gdn_sub_sync = trace_level == "3";
    let stages = std::cell::RefCell::new(StageTimes::default());
    let gdn_sub = std::cell::RefCell::new(GdnSubTimes::default());
    // Arm 13 gate — env-forced mid-loop failure (tests the snapshot repair:
    // the fall-through must reproduce the psplit logits exactly).
    let wb_fail_at: Option<usize> = std::env::var("RIIR_PREFILL_WB_FAIL_AT")
        .ok()
        .and_then(|s| s.trim().parse().ok());
    // Arm 13 — streamed writeback bookkeeping.
    let wb_pending: std::cell::RefCell<Vec<WbPending>> = std::cell::RefCell::new(Vec::new());
    let wb_drain = |force: bool| -> Result<(), String> {
        let Ok(mut pend) = wb_pending.try_borrow_mut() else {
            return Ok(());
        };
        if pend.is_empty() {
            return Ok(());
        }
        let Some(slots_lock) = stack.wb_slots.get().and_then(|s| s.as_ref()) else {
            return Err("drain: no slots".into());
        };
        let Ok(slots) = slots_lock.lock() else {
            return Err("drain: slots lock".into());
        };
        let mut i = 0;
        let mut drained = 0usize;
        while i < pend.len() {
            match wb_drain_one(client, &slots, &pend[i], fwd, base_pos, kvd, force) {
                Ok(true) => {
                    pend.remove(i);
                    drained += 1;
                }
                Ok(false) => i += 1,
                Err(e) => return Err(e),
            }
        }
        if drained > 0 {
            // WDDM batches queue submissions — without a flush the writes
            // sit host-side until the tail's read forces them all out at
            // once (the 171 ms read). Flush here so the DMA overlaps the
            // GPU's remaining kernels.
            let _ = client.flush();
        }
        Ok(())
    };
    let wb_enqueue = |kind: WbKind, len: usize, split: usize, states: &StateMirrors| -> Result<(), String> {
        if spec {
            // Issue 742 T1.1 — spec mode: no writeback crossing. The mirrors
            // carry on-device; rollback restores from the snapshot.
            return Ok(());
        }
        let Some(slots_lock) = stack.wb_slots.get().and_then(|s| s.as_ref()) else {
            return Err("enqueue: no slots".into());
        };
        let Ok(slots) = slots_lock.lock() else {
            return Err("enqueue: slots lock".into());
        };
        let pend = WbPending { kind, len, split };
        // SAFETY: each mirror is written only by its own layer's kernels,
        // all of which precede this point on the compute stream.
        unsafe {
            wb_enqueue_layer(
                stack.as_ref(),
                &copy_stream,
                order_ev
                    .as_ref()
                    .expect("wb enqueue is non-spec; order_ev is Some"),
                &slots,
                &pend,
                states,
                base_pos,
                kvd,
            )
        }?;
        wb_pending.borrow_mut().push(pend);
        Ok(())
    };
    let run_layers = || -> Result<(), String> {
        type WCache = Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>;

let rms = |input: &CudaSlice<f32>,
                   gamma: &CudaSlice<f32>,
                   out: &CudaSlice<f32>,
                   rows: usize,
                   dim: usize|
         -> Result<(), String> {
            let t = std::time::Instant::now();
            let r = unsafe {
                stack
                    .ffn
                    .launch_rmsnorm(stream, ra, rm, rs, input, gamma, out, rows, dim, eps)
            };
            if stage_sync {
                let _ = stream.synchronize();
                stages.borrow_mut().rms += t.elapsed().as_micros();
            }
            r
        };
        // quantize + GEMM pair (one quantize per distinct input buffer use).
        // `quantize=false` (gemm_pre): the input was quantized by the
        // immediately-preceding gemm in the same group — qkv/z/a/b all read
        // `normx`, gate+up read `normx`, wq+wkv read `normx` — and only gemms
        // (which merely READ the scratch) run in between, so re-quantizing
        // writes identical bytes to the same scratch (Issue 742 T1.4c dedup:
        // ~224 redundant launches/chunk, bit-identical by construction).
        let gemm_pair =
            |w: &Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>,
             input: &CudaSlice<f32>,
             out: &CudaSlice<f32>,
             m: usize,
             n_in: usize,
             quantize: bool|
             -> Result<(), String> {
                let t = std::time::Instant::now();
                let r = (|| {
                    // dedup ON: only the group's first GEMM quantizes; dedup OFF
                    // (the A/B arm): every GEMM re-quantizes (the original path).
                    if quantize || !quantize_dedup_enabled() {
                        stack
                            .mma
                            .launch_prefill_quantize(stream, input, scratch, n_in, p)
                            .map_err(|e| e.to_string())?;
                    }
                    crate::prefill_cuda_mma::launch_prefill_gemm_cached(
                        &stack.mma,
                        stream,
                        w,
                        scratch,
                        out,
                        m,
                        n_in,
                        p,
                    )
                    .map_err(|e| e.to_string())
                })();
                if stage_sync {
                    let _ = stream.synchronize();
                    stages.borrow_mut().gemm += t.elapsed().as_micros();
                }
                r
            };
        let gemm = |w: &Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>,
                    input: &CudaSlice<f32>,
                    out: &CudaSlice<f32>,
                    m: usize,
                    n_in: usize|
         -> Result<(), String> { gemm_pair(w, input, out, m, n_in, true) };
        // GEMM on an input already packed in the scratch by the preceding
        // gemm of the same group (same input buffer + same n_in).
        let gemm_pre = |w: &Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>,
                        input: &CudaSlice<f32>,
                        out: &CudaSlice<f32>,
                        m: usize,
                        n_in: usize|
         -> Result<(), String> { gemm_pair(w, input, out, m, n_in, false) };
        // Issue 902 T1 — the fused gate+up pair: quantize once, then ONE
        // v11gu/v11gut launch (both [m, n] slabs from one B tile stage) when
        // `RIIR_PREFILL_MMQ_GU` arms it; the two-launch fallback otherwise
        // (bit-identical in both arms — unit-gated).
        let gemm_gu =
            |w0: &Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>,
             w1: &Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>,
             input: &CudaSlice<f32>,
             out0: &CudaSlice<f32>,
             out1: &CudaSlice<f32>,
             m: usize,
             n_in: usize|
             -> Result<(), String> {
                let t = std::time::Instant::now();
                let r = (|| {
                    stack
                        .mma
                        .launch_prefill_quantize(stream, input, scratch, n_in, p)
                        .map_err(|e| e.to_string())?;
                    crate::prefill_cuda_mma::launch_prefill_gemm_pair_cached(
                        &stack.mma,
                        stream,
                        w0,
                        w1,
                        scratch,
                        out0,
                        out1,
                        m,
                        n_in,
                        p,
                    )
                    .map_err(|e| e.to_string())
                })();
                if stage_sync {
                    let _ = stream.synchronize();
                    stages.borrow_mut().gemm += t.elapsed().as_micros();
                }
                r
            };
        // ── Issue 980 C0.5 — the fused-rotation GEMM helpers (folded lanes) ──
        // The rotated quantize rides the q8 pass when the route is on (ONE
        // memory pass — rotation is ~free); the q8 kill-switch falls back to
        // the C0 unfused pair (copy-rotate + the resolved-arm quantize).
        // BYTE-IDENTICAL by construction (the `fused_qrot_bitexact` unit
        // gate) — the folded pins hold across the fusion.
        #[cfg(feature = "prefill_q8_act")]
        let qrot_fused_on =
            crate::gemm_ternary_i8_mma_cuda_raw::q8_act_enabled() && rotation.is_some();
        #[cfg(not(feature = "prefill_q8_act"))]
        let qrot_fused_on = false;
        // Fused rotated quantize of `input` [p x n_in] into the q8 scratch.
        #[allow(clippy::too_many_arguments)]
        let quantize_rot = |input: &CudaSlice<f32>, n_in: usize| -> Result<(), String> {
            let Some(rot) = &rotation else { return Ok(()); };
            if qrot_fused_on {
                rot.kernels
                    .quantize_rotate_q8(
                        stream,
                        input,
                        rot.signs_for_width(n_in),
                        &scratch.q_hi_w,
                        &scratch.s_t,
                        n_in,
                        p,
                        rot.block_size,
                    )
                    .map_err(|e| e.to_string())
            } else if let Some(rs) = rot_scratch {
                rot.kernels
                    .fwht_rotate_copy_batched(
                        stream,
                        input,
                        rs,
                        rot.signs_for_width(n_in),
                        p,
                        n_in,
                        rot.block_size,
                    )
                    .map_err(|e| e.to_string())?;
                stack
                    .mma
                    .launch_prefill_quantize(stream, rs, scratch, n_in, p)
                    .map_err(|e| e.to_string())
            } else {
                Ok(())
            }
        };
        // GEMM on the PRIMAL input with the rotated quantize (first-in-group).
        let gemm_rot = |w: &Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>,
                        input: &CudaSlice<f32>,
                        out: &CudaSlice<f32>,
                        m: usize,
                        n_in: usize|
         -> Result<(), String> {
            let t = std::time::Instant::now();
            let r = (|| {
                quantize_rot(input, n_in)?;
                // Fused or unfused, the scratch now holds the ROTATED q8
                // bytes (quantize_rot dispatched the right arm).
                crate::prefill_cuda_mma::launch_prefill_gemm_cached(
                    &stack.mma, stream, w, scratch, out, m, n_in, p,
                )
                .map_err(|e| e.to_string())
            })();
            if stage_sync {
                let _ = stream.synchronize();
                stages.borrow_mut().gemm += t.elapsed().as_micros();
            }
            r
        };
        // GEMM reusing the group's rotated scratch (dedup twin of gemm_pre —
        // with the A/B dedup-OFF arm, re-runs the ROTATED quantize so the
        // bytes match the group head's).
        let gemm_pre_rot = |w: &Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>,
                            input: &CudaSlice<f32>,
                            out: &CudaSlice<f32>,
                            m: usize,
                            n_in: usize|
         -> Result<(), String> {
            let t = std::time::Instant::now();
            let r = (|| {
                if !quantize_dedup_enabled() {
                    quantize_rot(input, n_in)?;
                }
                crate::prefill_cuda_mma::launch_prefill_gemm_cached(
                    &stack.mma, stream, w, scratch, out, m, n_in, p,
                )
                .map_err(|e| e.to_string())
            })();
            if stage_sync {
                let _ = stream.synchronize();
                stages.borrow_mut().gemm += t.elapsed().as_micros();
            }
            r
        };
        // The gate+up pair on the rotated quantize.
        #[allow(clippy::too_many_arguments)]
        let gemm_gu_rot =
            |w0: &Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>,
             w1: &Arc<crate::prefill_cuda_mma::CudaMmaWeightCache>,
             input: &CudaSlice<f32>,
             out0: &CudaSlice<f32>,
             out1: &CudaSlice<f32>,
             m: usize,
             n_in: usize|
             -> Result<(), String> {
                let t = std::time::Instant::now();
                let r = (|| {
                    quantize_rot(input, n_in)?;
                    crate::prefill_cuda_mma::launch_prefill_gemm_pair_cached(
                        &stack.mma, stream, w0, w1, scratch, out0, out1, m, n_in, p,
                    )
                    .map_err(|e| e.to_string())
                })();
                if stage_sync {
                    let _ = stream.synchronize();
                    stages.borrow_mut().gemm += t.elapsed().as_micros();
                }
                r
            };
        let ffn_block =
            |x: &CudaSlice<f32>, lw3: &LayerF32, gate_c: &WCache, up_c: &WCache, down_c: &WCache| -> Result<(), String> {
                rms(x, &lw3.post_attn_norm, normx, p, n)?;
                // Issue 980 T4-ALT site (f) / C0.5 — on a folded model the
                // FFN is a folded consumer at BOTH ends: gate/up consume
                // the ROTATED q8 quantize of normx (fused, one pass) and
                // after SwiGLU the down-proj input (hid, width mlp) goes
                // through the same fused rotate+quantize — no standalone
                // rotation passes at either end. The residual chain stays
                // primal (x untouched).
                if rotation.is_some() {
                    gemm_gu_rot(gate_c, up_c, normx, gate_b, up_b, mlp, n)?;
                } else {
                    gemm_gu(gate_c, up_c, normx, gate_b, up_b, mlp, n)?;
                }
                unsafe {
                    stack
                        .ffn
                        .launch_swiglu(stream, se, sr, gate_b, up_b, hid_b, p * mlp)?;
                }
                // C0.5 hybrid: the WIDE hid row (17408 f32 = 68 KB smem → 1
                // block/SM, ~17% occupancy — measured −3.5% pp2048 as
                // single-pass fused) keeps the UNFUSED chain (in-place
                // rotate + the stock streaming quantize inside `gemm`) —
                // chunk-parallel, occupancy-healthy. The narrow sites ride
                // the fused kernel.
                if let Some(rot0) = &rotation {
                    rot0
                        .kernels
                        .fwht_rotate_forward_batched(
                            stream,
                            hid_b,
                            rot0.signs_for_width(mlp),
                            p,
                            mlp,
                            rot0.block_size,
                        )
                        .map_err(|e| e.to_string())?;
                }
                gemm(down_c, hid_b, ffnout_b, n, mlp)?;
                unsafe { stack.ffn.launch_residual(stream, x, ffnout_b, x, p * n)?; }
                Ok(())
            };

        for (li, lt) in fwd.layer_types.iter().enumerate() {
            let lw = &fwd.layers[li];
            let lw3 = &f32w[li];
            let t_stage = std::time::Instant::now();
            macro_rules! stage_done {
                ($which:ident) => {
                    if stage_sync {
                        let _ = stream.synchronize();
                        stages.borrow_mut().$which += t_stage.elapsed().as_micros();
                    }
                };
            }
            // Arm 13 — drain any completed writeback slots while dispatching
            // (host-slack overlap; never blocks).
            wb_drain(false)?;
            if Some(li) == wb_fail_at {
                return Err("forced wb failure (Arm-13 gate)".into());
            }
            if *lt == DeltaNetLayerType::DeltaNet {
                // GDN block.
                rms(x, &lw3.input_norm, normx, p, n)?;
                let qkv_c = mma_mirror(client, stream, &lw.in_proj_qkv).ok_or("qkv mirror")?;
                let z_c = mma_mirror(client, stream, &lw.in_proj_z).ok_or("z mirror")?;
                // Issue 980 T4-ALT site (b) / C0.5 — on a folded model qkv/z
                // consume the ROTATED q8 quantize of normx (the fused
                // rotate+quantize — one pass) while the dense escape-set a/b
                // (fp32, mirrored in LayerF32) run on the PRIMAL normx — the
                // escape set is neither rotated nor folded.
                if let (Some(rot0), (Some(da), Some(db))) =
                    (&rotation, (lw3.dense_a.as_ref(), lw3.dense_b.as_ref()))
                {
                    gemm_rot(&qkv_c, normx, qkv_b, qkv_dim, n)?;
                    gemm_pre_rot(&z_c, normx, z_b, v_dim, n)?;
                    rot0
                        .kernels
                        .gemm_dense_ab_batched(stream, normx, da, db, a_b, b_b, p, n_v, n)
                        .map_err(|e| e.to_string())?;
                } else {
                    let a_c = mma_mirror(client, stream, &lw.in_proj_a).ok_or("a mirror")?;
                    let b_c = mma_mirror(client, stream, &lw.in_proj_b).ok_or("b mirror")?;
                    gemm(&qkv_c, normx, qkv_b, qkv_dim, n)?;
                    gemm_pre(&z_c, normx, z_b, v_dim, n)?;
                    gemm_pre(&a_c, normx, a_b, n_v, n)?;
                    gemm_pre(&b_c, normx, b_b, n_v, n)?;
                }
                let (Some(conv_state), Some(dn_state)) = (
                    states.conv_states[li].as_ref(),
                    states.dn_states[li].as_ref(),
                ) else {
                    return Err("state mirrors".into());
                };
                unsafe {
                    // TRACE=3 per-kernel attribution: capture → launch → sync →
                    // accumulate (gated; the `Instant::now()` pairs stay
                    // unconditional — ns-scale, no control-flow change).
                    let mut t_sub = std::time::Instant::now();
                    stack.dn.launch_conv1d(
                        stream, cf, ce, cd, qkv_b, qkv_conv, &lw3.conv1d_weight, conv_state, p,
                        conv_dim, ks,
                    )?;
                    if gdn_sub_sync {
                        let _ = stream.synchronize();
                        gdn_sub.borrow_mut().conv += t_sub.elapsed().as_micros();
                    }
                    t_sub = std::time::Instant::now();
                    stack
                        .dn
                        .launch_carry_update(stream, qkv_b, conv_state, p, conv_dim, ks)?;
                    if gdn_sub_sync {
                        let _ = stream.synchronize();
                        gdn_sub.borrow_mut().carry += t_sub.elapsed().as_micros();
                    }
                    t_sub = std::time::Instant::now();
                    stack.dn.launch_beta_decay(
                        stream,
                        bl,
                        be,
                        bd,
                        a_b,
                        b_b,
                        &lw3.a_log,
                        &lw3.dt_bias,
                        beta_b,
                        decay_b,
                        n_v,
                        p * n_v,
                    )?;
                    if gdn_sub_sync {
                        let _ = stream.synchronize();
                        gdn_sub.borrow_mut().beta += t_sub.elapsed().as_micros();
                    }
                    t_sub = std::time::Instant::now();
                    stack.dn.launch_expand_l2(
                        stream,
                        ea,
                        ei,
                        qkv_conv,
                        qkvx_b,
                        n_k,
                        n_v,
                        hd,
                        p * 3 * v_dim,
                    )?;
                    if gdn_sub_sync {
                        let _ = stream.synchronize();
                        gdn_sub.borrow_mut().expand += t_sub.elapsed().as_micros();
                    }
                    t_sub = std::time::Instant::now();
                    if gdn_chunked_ready {
                        // Arm 12 — the chunked recurrence (NOT bit-identical to
                        // rowpar; the llama.cpp numerics class — see the knob doc).
                        stack
                            .gdn_chunked
                            .as_ref()
                            .expect("gdn_chunked_ready implies Some")
                            .launch_chunked(
                                stream, qkvx_b, beta_b, decay_b, dn_state, rec_b, gdn_lg.expect("grown"), gdn_ga.expect("grown"), gdn_dte.expect("grown"), gdn_td.expect("grown"), gdn_x.expect("grown"), gdn_t.expect("grown"), gdn_qkr.expect("grown"), n_v, p, v_dim,
                            )?;
                    } else {
                        stack.dn.launch_recurrence(
                            stream, rd, ru, rp, rsc, qkvx_b, beta_b, decay_b, dn_state, rec_b, hd,
                            n_v, p, v_dim,
                        )?;
                    }
                    if gdn_sub_sync {
                        let _ = stream.synchronize();
                        gdn_sub.borrow_mut().rec += t_sub.elapsed().as_micros();
                    }
                    t_sub = std::time::Instant::now();
                    // Per-head norm, in-place (same-index read/write — safe).
                    stack.ffn.launch_rmsnorm(
                        stream,
                        ra,
                        rm,
                        rs,
                        rec_b,
                        &lw3.linear_norm,
                        rec_b,
                        p * n_v,
                        hd,
                        eps,
                    )?;
                    if gdn_sub_sync {
                        let _ = stream.synchronize();
                        gdn_sub.borrow_mut().lnorm += t_sub.elapsed().as_micros();
                    }
                    t_sub = std::time::Instant::now();
                    stack
                        .dn
                        .launch_z_gating(stream, ze, zd, rec_b, z_b, p * v_dim)?;
                    if gdn_sub_sync {
                        let _ = stream.synchronize();
                        gdn_sub.borrow_mut().zgate += t_sub.elapsed().as_micros();
                    }
                }
                stage_done!(gdn);
                // Arm 13 — dn + conv states are final for this chunk:
                // stream their writeback payload (async DtoH on the copy
                // stream) while the rest of the layer + later layers run.
                wb_enqueue(
                    WbKind::Dn { li },
                    n_v * hd * hd + conv_dim * ks,
                    n_v * hd * hd,
                    states,
                )?;
                // Issue 980 T4-ALT site (c) / C0.5 — the GDN-out chain:
                // per-head rmsnorm + z_gating above are UNCHANGED (primal
                // math); the out_proj input then gets the tiled→grouped
                // V-head permute + rotation + q8 quantize in ONE fused pass
                // (grouped files; the unfused C0 chain is the kill-switch
                // fallback), or just the fused rotate+quantize when
                // ungrouped. rec_b stays PRIMAL either way.
                let out_c = mma_mirror(client, stream, &lw.out_proj).ok_or("out mirror")?;
                if let Some(rot0) = &rotation {
                    if rot0.gdn_v_grouped {
                        if qrot_fused_on {
                            rot0
                                .kernels
                                .quantize_permute_rotate_q8(
                                    stream,
                                    rec_b,
                                    rot0.signs_for_width(v_dim),
                                    &scratch.q_hi_w,
                                    &scratch.s_t,
                                    v_dim,
                                    p,
                                    rot0.block_size,
                                    hd,
                                    rot0.gdn_k_groups,
                                )
                                .map_err(|e| e.to_string())?;
                            crate::prefill_cuda_mma::launch_prefill_gemm_cached(
                                &stack.mma,
                                stream,
                                &out_c,
                                scratch,
                                tmp_b,
                                n,
                                v_dim,
                                p,
                            )
                            .map_err(|e| e.to_string())?;
                        } else if let Some(pt) = permute_tmp {
                            // Unfused kill-switch chain (C0).
                            rot0
                                .kernels
                                .gdn_v_permute_batched(
                                    stream, rec_b, pt, p, v_dim, hd, rot0.gdn_k_groups,
                                )
                                .map_err(|e| e.to_string())?;
                            rot0
                                .kernels
                                .fwht_rotate_forward_batched(
                                    stream,
                                    pt,
                                    rot0.signs_for_width(v_dim),
                                    p,
                                    v_dim,
                                    rot0.block_size,
                                )
                                .map_err(|e| e.to_string())?;
                            gemm(&out_c, pt, tmp_b, n, v_dim)?;
                        } else {
                            return Err("folded gdn-out: permute staging unavailable".into());
                        }
                    } else {
                        gemm_rot(&out_c, rec_b, tmp_b, n, v_dim)?;
                    }
                } else {
                    gemm(&out_c, rec_b, tmp_b, n, v_dim)?;
                }
                unsafe { stack.ffn.launch_residual(stream, x, tmp_b, x, p * n)?; }
            } else {
                // Attention block.
                rms(x, &lw3.input_norm, normx, p, n)?;
                let wq = lw.attn_wq.as_ref().ok_or("attn_wq")?;
                let wkv = lw.attn_wkv.as_ref().ok_or("attn_wkv")?;
                let wo = lw.attn_wo.as_ref().ok_or("attn_wo")?;
                let wq_c = mma_mirror(client, stream, wq).ok_or("wq mirror")?;
                let wkv_c = mma_mirror(client, stream, wkv).ok_or("wkv mirror")?;
                let wo_c = mma_mirror(client, stream, wo).ok_or("wo mirror")?;
                // Issue 980 T4-ALT site (d) / C0.5 — wq/wkv are folded
                // matmuls: they consume the ROTATED q8 quantize of normx
                // (the fused one-pass rotate+quantize; the unfused
                // copy-rotate chain is the kill-switch fallback). The
                // attention core (RoPE, qk-norm, the kernel's gate) stays
                // primal.
                if rotation.is_some() {
                    gemm_rot(&wq_c, normx, qg_b, 2 * qa, n)?;
                    gemm_pre_rot(&wkv_c, normx, kv_b, 2 * kvd, n)?;
                } else {
                    gemm(&wq_c, normx, qg_b, 2 * qa, n)?;
                    gemm_pre(&wkv_c, normx, kv_b, 2 * kvd, n)?;
                }
                let (Some(q_norm), Some(k_norm)) =
                    (lw3.attn_q_norm.as_ref(), lw3.attn_k_norm.as_ref())
                else {
                    return Err("qk norm mirrors".into());
                };
                let (Some(kcache), Some(vcache)) =
                    (states.kv_k[li].as_ref(), states.kv_v[li].as_ref())
                else {
                    return Err("kv mirrors".into());
                };
                let rotary_dim = if cfg.rope_dimension_count > 0 {
                    cfg.rope_dimension_count
                } else {
                    ahd
                };
                unsafe {
                    stack
                        .at
                        .launch_split_qg(stream, qg_b, q_b, agate_b, ahd, n_head, p)?;
                    stack.at.launch_split_kv(stream, kv_b, k_b, v_b, kvd, p)?;
                    // Q/K per-head norms, in-place (2 calls replicate the fused
                    // kernel's per-row math bit-exactly).
                    stack.ffn.launch_rmsnorm(
                        stream,
                        ra,
                        rm,
                        rs,
                        q_b,
                        q_norm,
                        q_b,
                        p * n_head,
                        ahd,
                        eps,
                    )?;
                    stack.ffn.launch_rmsnorm(
                        stream,
                        ra,
                        rm,
                        rs,
                        k_b,
                        k_norm,
                        k_b,
                        p * n_kv,
                        ahd,
                        eps,
                    )?;
                    if gv {
                        stack.at.launch_rope_devpos(
                            stream,
                            rl,
                            re,
                            rsc2,
                            rr,
                            q_b,
                            k_b,
                            rotary_dim / 2,
                            cfg.rope_theta,
                            ahd,
                            n_head,
                            n_kv,
                            p,
                            pos_dev_ref.expect("gv implies pos_dev"),
                        )?;
                    } else {
                        stack.at.launch_rope(
                            stream,
                            rl,
                            re,
                            rsc2,
                            rr,
                            q_b,
                            k_b,
                            rotary_dim / 2,
                            cfg.rope_theta,
                            ahd,
                            n_head,
                            n_kv,
                            p,
                            base_pos,
                        )?;
                    }
                    if gv {
                        stack.at.launch_kv_fill_devpos(
                            stream, k_b, v_b, kcache, vcache, kvd, p,
                            pos_dev_ref.expect("gv implies pos_dev"),
                        )?;
                    } else {
                        stack
                            .at
                            .launch_kv_fill(stream, k_b, v_b, kcache, vcache, kvd, p, base_pos)?;
                    }
                    // Arm 13 — this chunk's KV rows are final in the mirrors:
                    // stream them now (the attention kernel only READS the
                    // cache after this point).
                    wb_enqueue(WbKind::Kv { li }, 2 * p * kvd, p * kvd, states)?;
                    if attention_mq_enabled() && ahd == 256 {
                        let arm = attn_arm();
                        // Plan 605 T4 — the FA-class mma arm (`att_pf_fa`),
                        // DEFAULT-ON behind the engagement predicate
                        // (`p >= FA_MIN_P`; kill-switch
                        // `QWEN38_PF_ATTN_FA=0|off|false`; TOLERANCE-CLASS,
                        // bench_946 + the bench_948 retention walk). Precedes
                        // the incumbent ladder when engaged and the shape
                        // gates pass; the conversion pass + kernel are both
                        // devpos-capable (the gv lane captures them).
                        let mut fa_done = false;
                        if fa_engaged()
                            && p >= FA_MIN_P
                            && n_head == 6 * n_kv
                            && let (Some(kh), Some(vh), Some(kc), Some(vc)) = (
                                attn_fa_kh.as_ref(),
                                attn_fa_vh.as_ref(),
                                states.kv_k[li].as_ref(),
                                states.kv_v[li].as_ref(),
                            )
                        {
                            let rows = *attn_fa_rows;
                            let r = if gv {
                                let pos = pos_dev_ref.expect("gv implies pos_dev");
                                stack.at.launch_kv_f32_to_f16_devpos(
                                    stream, kc, vc, kh, vh, p, rows, n_kv, pos,
                                )
                                .and_then(|_| {
                                    stack.at.launch_attention_fa_devpos(
                                        stream, q_b, kh, vh, agate_b, attn_out_b, ahd,
                                        n_head, n_kv, p, pos,
                                    )
                                })
                            } else {
                                stack.at.launch_kv_f32_to_f16(
                                    stream, kc, vc, kh, vh, base_pos + p, rows, n_kv,
                                )
                                .and_then(|_| {
                                    stack.at.launch_attention_fa(
                                        stream, q_b, kh, vh, agate_b, attn_out_b, ahd,
                                        n_head, n_kv, p, base_pos,
                                    )
                                })
                            };
                            r?;
                            fa_done = true;
                        }
                        if !fa_done {
                        // Issue 742 T1.7 — the split-KV arm (opt-in,
                        // tolerance-class): engaged only in the small-p
                        // regime where the serial/prefetch grid
                        // (n_head*ceil(p/8)) under-fills the SMs; falls back
                        // to the prefetch arm otherwise. The gv arm uses the
                        // FIXED max grid (n_chunks_max from block_size) with
                        // neutral dead chunks — one capture serves every
                        // position; the eager arm uses the live grid.
                        let mut split_done = false;
                        if attn_split_engaged(arm, n_head, p) {
                            // Scratch was allocated in the chunk preamble
                            // (BEFORE any capture region — see there); the
                            // gv arm uses the FIXED max grid with neutral
                            // dead chunks, the eager arm the live grid.
                            let chunk_len = attn_split_chunk();
                            let n_chunks_max = *attn_split_chunks;
                            if let (Some(pm), Some(pl), Some(po)) = (
                                attn_split_pm.as_ref(),
                                attn_split_pl.as_ref(),
                                attn_split_po.as_ref(),
                            ) {
                                let r = if gv {
                                    stack.at.launch_attention_mq8_split_devpos(
                                        stream, q_b, kcache, vcache, agate_b,
                                        attn_out_b, pm, pl, po, ahd, n_head,
                                        n_kv, p,
                                        pos_dev_ref.expect("gv implies pos_dev"),
                                        chunk_len, n_chunks_max,
                                    )
                                } else {
                                    let n_live = (base_pos + p).div_ceil(chunk_len);
                                    stack.at.launch_attention_mq8_split(
                                        stream, q_b, kcache, vcache, agate_b,
                                        attn_out_b, pm, pl, po, ahd, n_head,
                                        n_kv, p, base_pos, chunk_len, n_live,
                                    )
                                };
                                split_done = r.is_ok();
                            }
                        }
                        if !split_done {
                        // Arm 3 falling through (over-cap p or scratch alloc
                        // failure) routes to the prefetch arm — the default.
                        // Arm 5 (Issue 898) routes to the vec arm when the
                        // gang grid under-fills the SMs (attn_gang_engaged).
                        let arm = if arm == 3 {
                            1
                        } else if arm == 5 && !attn_gang_engaged(arm, n_head, n_kv, p) {
                            4
                        } else {
                            arm
                        };
                        if gv {
                            let pos_dev = pos_dev_ref.expect("gv implies pos_dev");
                            match arm {
                                2 => {
                                    stack.at.launch_attention_mq8_staged_devpos(
                                        stream, q_b, kcache, vcache, agate_b, attn_out_b, ahd,
                                        n_head, n_kv, p, pos_dev,
                                    )?;
                                }
                                1 => {
                                    stack.at.launch_attention_mq8_prefetch_devpos(
                                        stream, q_b, kcache, vcache, agate_b, attn_out_b, ahd,
                                        n_head, n_kv, p, pos_dev,
                                    )?;
                                }
                                4 => {
                                    stack.at.launch_attention_mq8_vec_devpos(
                                        stream, q_b, kcache, vcache, agate_b, attn_out_b, ahd,
                                        n_head, n_kv, p, pos_dev,
                                    )?;
                                }
                                5 => {
                                    // Issue 898/899 — head-gang family;
                                    // the layout ladder picks the rung
                                    // (default g3, the Bench-893 winner).
                                    // Falls back to the vec arm on any
                                    // launch-contract mismatch (non-6-head
                                    // groups) — the split-arm fallthrough.
                                    let r = match attn_gang_layout() {
                                        6 => stack.at.launch_attention_mq8_gang_devpos(
                                            stream, q_b, kcache, vcache, agate_b, attn_out_b,
                                            ahd, n_head, n_kv, p, pos_dev,
                                        ),
                                        2 => stack.at.launch_attention_mq8_gang2_devpos(
                                            stream, q_b, kcache, vcache, agate_b, attn_out_b,
                                            ahd, n_head, n_kv, p, pos_dev,
                                        ),
                                        _ => stack.at.launch_attention_mq8_gang3_devpos(
                                            stream, q_b, kcache, vcache, agate_b, attn_out_b,
                                            ahd, n_head, n_kv, p, pos_dev,
                                        ),
                                    };
                                    if r.is_err() {
                                        stack.at.launch_attention_mq8_vec_devpos(
                                            stream, q_b, kcache, vcache, agate_b, attn_out_b,
                                            ahd, n_head, n_kv, p, pos_dev,
                                        )?;
                                    }
                                }
                                _ => {
                                    stack.at.launch_attention_mq8_devpos(
                                        stream, q_b, kcache, vcache, agate_b, attn_out_b, ahd,
                                        n_head, n_kv, p, pos_dev,
                                    )?;
                                }
                            }
                        } else {
                            match arm {
                                2 => {
                                    stack.at.launch_attention_mq8_staged(
                                        stream, q_b, kcache, vcache, agate_b, attn_out_b, ahd,
                                        n_head, n_kv, p, base_pos,
                                    )?;
                                }
                                1 => {
                                    stack.at.launch_attention_mq8_prefetch(
                                        stream, q_b, kcache, vcache, agate_b, attn_out_b, ahd,
                                        n_head, n_kv, p, base_pos,
                                    )?;
                                }
                                4 => {
                                    stack.at.launch_attention_mq8_vec(
                                        stream, q_b, kcache, vcache, agate_b, attn_out_b, ahd,
                                        n_head, n_kv, p, base_pos,
                                    )?;
                                }
                                5 => {
                                    // Issue 898/899 — head-gang family;
                                    // the layout ladder picks the rung
                                    // (default g3, the Bench-893 winner).
                                    // Falls back to the vec arm on any
                                    // launch-contract mismatch (non-6-head
                                    // groups) — the split-arm fallthrough.
                                    let r = match attn_gang_layout() {
                                        6 => stack.at.launch_attention_mq8_gang(
                                            stream, q_b, kcache, vcache, agate_b, attn_out_b,
                                            ahd, n_head, n_kv, p, base_pos,
                                        ),
                                        2 => stack.at.launch_attention_mq8_gang2(
                                            stream, q_b, kcache, vcache, agate_b, attn_out_b,
                                            ahd, n_head, n_kv, p, base_pos,
                                        ),
                                        _ => stack.at.launch_attention_mq8_gang3(
                                            stream, q_b, kcache, vcache, agate_b, attn_out_b,
                                            ahd, n_head, n_kv, p, base_pos,
                                        ),
                                    };
                                    if r.is_err() {
                                        stack.at.launch_attention_mq8_vec(
                                            stream, q_b, kcache, vcache, agate_b, attn_out_b,
                                            ahd, n_head, n_kv, p, base_pos,
                                        )?;
                                    }
                                }
                                _ => {
                                    stack.at.launch_attention_mq8(
                                        stream, q_b, kcache, vcache, agate_b, attn_out_b, ahd,
                                        n_head, n_kv, p, base_pos,
                                    )?;
                                }
                            }
                        }
                        }
                        } /* fa_done */
                    } else {
                        if gv {
                            // The generic attention kernel has no devpos twin —
                            // graph-verify requires the mq8 arm (ahd==256).
                            return Err("graph-verify requires mq8 (ahd==256)".into());
                        }
                        stack.at.launch_attention(
                            stream, ad, aa, ae, aph, ars, ai, q_b, kcache, vcache, agate_b,
                            attn_out_b, ahd, n_head, n_kv, p, base_pos,
                        )?;
                    }
                }
                stage_done!(attn);
                // Issue 980 T4-ALT site (e) / C0.5 — wo is a folded matmul:
                // the fused rotate+quantize consumes the gated attention
                // output (width qa) directly — no standalone rotation pass.
                // The residual add stays primal.
                if rotation.is_some() {
                    gemm_rot(&wo_c, attn_out_b, aproj_b, n, qa)?;
                } else {
                    gemm(&wo_c, attn_out_b, aproj_b, n, qa)?;
                }
                unsafe { stack.ffn.launch_residual(stream, x, aproj_b, x, p * n)?; }
            }

            // FFN block (both layer kinds).
            let gate_c = mma_mirror(client, stream, &lw.gate_proj).ok_or("gate mirror")?;
            let up_c = mma_mirror(client, stream, &lw.up_proj).ok_or("up mirror")?;
            let down_c = mma_mirror(client, stream, &lw.down_proj).ok_or("down mirror")?;
            ffn_block(x, lw3, &gate_c, &up_c, &down_c)?;
            stage_done!(other);
        }
        Ok(())
    };
    // Issue 742 T1.2 probe — capture the layer loop into a CUDA Graph
    // (fixed base_pos: TIMING-valid only — repeated replays double-advance
    // the recurrent state; the numeric output is garbage by design). The
    // probe isolates the launch/WDDM submit overhead from the GPU work:
    // if steady-state replay ≈ the memory floor (~11 ms at p=16), the
    // per-chunk fixed cost is launches and the real verify path needs
    // graph capture + device-side param indirection (the decode path's
    // pos_dev_buf pattern).
    let graph_probe = spec
        && std::env::var("RIIR_PREFILL_GRAPH_PROBE").is_ok_and(|s| matches!(s.trim(), "1" | "true" | "on"))
        && stack.graph_probe.get().is_none();
    let run_layers_result = if graph_probe {
        // cudarc's per-launch event tracking records an internal event per
        // launch; during capture those become cross-stream dependencies →
        // CUDA_ERROR_STREAM_CAPTURE_ISOLATION. The decode graph path's
        // SAFETY contract applies here too (single-stream dispatch within
        // the arm; explicit ordering events for the copy stream): disable
        // the implicit tracker for the probe.
        // SAFETY: the arm's chunk path is single-threaded, single-stream
        // (the wb copy stream's ordering uses OUR explicit CudaEvents).
        unsafe {
            stack.stream.context().disable_event_tracking();
        }
        embed().ok()?;
        t_embed = Some(t0.elapsed());
        match stream.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
        ) {
            Ok(()) => {
                let r = run_layers();
                let end = stream.end_capture(
                    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                );
                match (r, end) {
                    (Ok(()), Ok(Some(graph))) => {
                        let _ = graph.upload();
                        let _ = stream.synchronize();
                        let _ = stack
                            .graph_probe
                            .set(Some(Mutex::new(SendGraph(graph))));
                        eprintln!("[742-probe] layer loop captured (p={p})");
                        Ok(())
                    }
                    (r, end) => {
                        eprintln!(
                            "[742-probe] capture failed (layers err={:?}, end_capture err={:?}) — continuing uncaptured",
                            r.as_ref().err(),
                            end.as_ref().err()
                        );
                        r
                    }
                }
            }
            Err(e) => {
                eprintln!("[742-probe] begin_capture failed: {e}");
                run_layers()
            }
        }
    } else if gv && !stage_sync {
        // Issue 742 T1.5 — the graph-verify state machine (the REAL replay
        // path): capture ONCE once the weight mirrors are warm (a full chunk
        // has completed — a cold-mirror capture would bake upload memcpys
        // into the graph), then REPLAY every same-`p` chunk at its uploaded
        // position. The captured kernel sequence is exactly the eager one
        // (devpos twins — same arithmetic, base_pos from the device buffer),
        // so replay ≡ eager per chunk by construction; the standing G1
        // (chunked-prefill logits pins) gates the whole path.
        // Issue 965 — arm marker: graph mode reaches this branch on regular
        // (non-spec) chunks; one line keeps league run logs greppable.
        if graphs {
            static ARM_LOGGED: AtomicBool = AtomicBool::new(false);
            if !ARM_LOGGED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[965-graphs] graph capture armed (capture_p={}, knob_gen={})",
                    capture_p(),
                    GV_KNOB_GEN.load(Ordering::Relaxed)
                );
            }
        }
        // Issue 967 — every chunk through the graph lane counts its tokens
        // (the fallback-share denominator; replay/capture/fallback alike).
        stack.gv_total_tokens.fetch_add(p, Ordering::Relaxed);
        let captured_p = stack.gv_p.load(Ordering::Relaxed);
        let have_graph = stack.graph_verify.get().is_some_and(|g| g.is_some());
        let gen_ok = stack.gv_staging_gen.load(Ordering::Relaxed) == *staging_gen
            && stack.gv_knob_gen.load(Ordering::Relaxed) == GV_KNOB_GEN.load(Ordering::Relaxed);
        if have_graph && captured_p == p && gen_ok {
            // REPLAY — the embed is IN the graph (captured with the layer
            // loop); the whole chunk is one graph launch.
            t_embed = Some(t0.elapsed());
            let g = stack
                .graph_verify
                .get()
                .and_then(|g| g.as_ref())
                .expect("have_graph checked");
            match g.lock() {
                Ok(gr) => {
                    stack.gv_replays.fetch_add(1, Ordering::Relaxed);
                    let r = gr.0.launch().map_err(|e| e.to_string());
                    // A failed replay executed NOTHING — the mirrors still
                    // hold the pre-chunk state. Do NOT restore (no snapshot
                    // was taken); flush the current mirrors to CubeCL so the
                    // fall-through below recomputes from the correct state.
                    if r.is_err() {
                        let pos = stack.spec_pos.swap(0, Ordering::Relaxed).max(base_pos);
                        let mut state_host = Vec::new();
                        let mut kv_host = Vec::new();
                        let _ = writeback_states(
                            client,
                            stream,
                            fwd,
                            states,
                            0,
                            pos,
                            kvd,
                            &mut state_host,
                            &mut kv_host,
                        );
                    }
                    r
                }
                Err(_) => Err("graph_verify lock poisoned".into()),
            }
        } else if !have_graph
            && p == capture_p()
            && stack.gv_warm.load(Ordering::Relaxed)
            && stack.graph_verify.get().is_none()
        {
            // CAPTURE. cudarc's per-launch event tracking records an internal
            // event per launch; during capture those become cross-stream
            // dependencies → capture fails (the decode graph path's SAFETY
            // contract — disable it for the capture, as the probe does).
            // SAFETY: the armed chunk path is single-threaded, single-stream
            // (spec mode enqueues no wb copy-stream work).
            unsafe {
                stack.stream.context().disable_event_tracking();
            }
            // Issue 967 — price the per-rung memory ceiling: the free-memory
            // delta across (capture + instantiate + upload) is exactly the
            // increment a second ladder rung adds under a shared buffer pool
            // (the p-dependent persistent buffers are grow-only and shared).
            let mem0 = stack.stream.context().mem_get_info().ok();
            match stream.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            ) {
                Ok(()) => {
                    // The embed rides INSIDE the captured region (reads the
                    // persistent tokens_dev — address-stable), so every
                    // replay is the WHOLE chunk (embed + 64 layers) in one
                    // graph launch.
                    let r = embed().and_then(|()| run_layers());
                    t_embed = Some(t0.elapsed());
                    let end = stream.end_capture(
                        cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                    );
                    match (r, end) {
                        (Ok(()), Ok(Some(graph))) => {
                            let _ = graph.upload();
                            let _ = stream.synchronize();
                            let mem1 = stack.stream.context().mem_get_info().ok();
                            if let (Some((f0, _)), Some((f1, _))) = (mem0, mem1) {
                                // WDDM quirk measured 2026-09-17: cuMemGetInfo can
                                // report (0, 0) on this box — skip the line rather
                                // than print a confident zero (the nvidia-smi
                                // sampler in the pricing probe is the fallback).
                                if f0 > 0 && f1 > 0 {
                                    eprintln!(
                                        "[967-mem] capture p={p}: graph exec+upload = {} bytes (free {} -> {})",
                                        f0.saturating_sub(f1),
                                        f0,
                                        f1
                                    );
                                }
                            }
                            stack.gv_p.store(p, Ordering::Relaxed);
                            stack.gv_staging_gen.store(*staging_gen, Ordering::Relaxed);
                            stack
                                .gv_knob_gen
                                .store(GV_KNOB_GEN.load(Ordering::Relaxed), Ordering::Relaxed);
                            let _ = stack
                                .graph_verify
                                .set(Some(Mutex::new(SendGraph(graph))));
                            eprintln!(
                                "[742-gv] layer loop captured (p={p}, gen={staging_gen})"
                            );
                            // Capture records but does NOT execute — this
                            // launch IS the capture chunk's execution.
                            let launched = stack
                                .graph_verify
                                .get()
                                .and_then(|g| g.as_ref())
                                .and_then(|g| g.lock().ok())
                                .map(|gr| gr.0.launch());
                            match launched {
                                Some(Ok(())) => Ok(()),
                                Some(Err(e)) => Err(format!("gv first launch: {e}")),
                                None => Err("gv graph vanished".into()),
                            }
                        }
                        (r, end) => {
                            eprintln!(
                                "[742-gv] capture failed (layers err={:?}, end_capture err={:?}) — permanent eager fallback",
                                r.as_ref().err(),
                                end.as_ref().err()
                            );
                            // Mark attempted (Some(None)) so no later chunk
                            // retries; the mirrors may hold nothing from this
                            // chunk (record-only) — re-run eagerly for real.
                            let _ = stack.graph_verify.set(None);
                            match r {
                                Ok(()) => {
                                    embed().ok()?;
                                    let r2 = run_layers();
                                    if r2.is_ok() {
                                        stack.gv_warm.store(true, Ordering::Relaxed);
                                    }
                                    r2
                                }
                                err => err,
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[742-gv] begin_capture failed: {e} — permanent eager fallback");
                    let _ = stack.graph_verify.set(None);
                    embed().ok()?;
                    t_embed = Some(t0.elapsed());
                    let r = run_layers();
                    if r.is_ok() {
                        stack.gv_warm.store(true, Ordering::Relaxed);
                    }
                    r
                }
            }
        } else {
            // Warm-up chunk (mirrors cold) or post-capture-failure eager — run
            // the layer loop eagerly; a completed full chunk warms every
            // weight mirror, enabling the next chunk's capture.
            //
            // Issue 967 — price the capture-ladder gap: an eager chunk at
            // p != capture_p() with WARM mirrors is exactly the set a second
            // rung would recover (cold-mirror chunks can never capture —
            // the first chunk per process stays eager under any ladder).
            if p != capture_p() && stack.gv_warm.load(Ordering::Relaxed) {
                stack.gv_fallbacks.fetch_add(1, Ordering::Relaxed);
                stack.gv_fallback_tokens.fetch_add(p, Ordering::Relaxed);
            }
            embed().ok()?;
            t_embed = Some(t0.elapsed());
            let r = run_layers();
            if r.is_ok() {
                stack.gv_warm.store(true, Ordering::Relaxed);
            }
            r
        }
    } else {
        embed().ok()?;
        t_embed = Some(t0.elapsed());
        run_layers()
    };
    if run_layers_result.is_err() {
        if trace {
            eprintln!("[734-arm8] FALLTHROUGH: layer loop error");
        }
        if will_replay {
            // T1.5 — a replay-error chunk took NO snapshot (the failed
            // replay executed nothing) and already flushed the current
            // mirrors to CubeCL; restoring a stale snapshot would corrupt.
            return None;
        }
        // Arm 13 — streamed writes may already have landed for earlier
        // layers; restore the pre-chunk state (snapshot → mirrors → legacy
        // writeback) so the caller's CubeCL recompute starts correct.
        wb_restore_on_failure(stack.as_ref(), client, states, fwd, p, base_pos, kvd);
        return None;
    }
    let t_embed = t_embed.unwrap_or_else(|| t0.elapsed());
    let t_layers = t0.elapsed();
    if stage_sync {
        let s = stages.borrow();
        eprintln!(
            "[734-arm8-trace] layer stages (us): gemm={} rms={} gdn={} attn={} other={}",
            s.gemm, s.rms, s.gdn, s.attn, s.other
        );
    }
    if gdn_sub_sync {
        let g = gdn_sub.borrow();
        eprintln!(
            "[734-arm8-trace] gdn sub-stages (us): conv={} carry={} beta={} expand={} rec={} lnorm={} zgate={} (chunked={})",
            g.conv, g.carry, g.beta, g.expand, g.rec, g.lnorm, g.zgate, gdn_chunked_ready
        );
    }

    // ── Writebacks (every chunk — fall-through safety + decode continuity).
    //    Arm 13: the payloads were streamed during the layer loop (async
    //    DtoH on the copy stream); force-drain what's left — mostly the
    //    last layers' slots, already complete — then enqueue their CubeCL
    //    writes. The writes land BEFORE the tail's kernels in the queue; the
    //    logits read pays only any residual write DMA. ──
    if let Err(e) = wb_drain(true) {
        if trace {
            eprintln!("[734-arm8] FALLTHROUGH: wb drain ({e})");
        }
        wb_restore_on_failure(stack.as_ref(), client, states, fwd, p, base_pos, kvd);
        return None;
    }
    let t_wb = t0.elapsed();

    // ── Issue 742 T1.8 — the per-position verify tail ──
    //    All cudarc-side, zero logits crossings: a `p`-row final-norm over
    //    `x` into `normx` (the layer loop is complete — normx is scratch),
    //    the lm_head mma GEMM (ONE weight read for all p rows — the sp
    //    routing engages automatically at p ≤ 16), a device-side per-row
    //    argmax (Issue-697 first-index tie-break), and a `p * 8` byte dtoh.
    //    Numerics: the final-norm kernel + GEMM family are the arm's pinned
    //    layer-loop kernels; the lm_head through the mma GEMM is a DIFFERENT
    //    kernel than the decode path's rowtiled8 GemvTernary — greedy
    //    near-tie divergence vs sequential decode is possible and is exactly
    //    what the T1.8 harness G1 measures.
    if verify_tail {
        let t_tail = t0.elapsed();
        let fail = |states: &mut StateMirrors| {
            wb_restore_on_failure(stack.as_ref(), client, states, fwd, p, base_pos, kvd);
        };
        // G1-strict fallback arm — p sequential CubeCL GemvTernary rows (the
        // decode lm_head family, bit-matched by the G1c continuity pins);
        // the mma GEMM below stays the fast default. Each row: the raw x row
        // → fwd.x → the unmodified CubeCL final norm + GemvTernary lm_head
        // (the is_final tail's ONE-crossing pattern, per row) → CPU argmax
        // (Issue-697 first-index convention).
        if verify_tail_gemv() {
            let mut out: Vec<u32> = Vec::with_capacity(p);
            for r in 0..p {
                let row_start = r * n;
                let view = if let Some(v) = x.try_slice(row_start..row_start + n) { v } else {
                        fail(states);
                        return None;
                    };
                if stream.memcpy_dtoh(&view, x_row_host).is_err() {
                    fail(states);
                    return None;
                }
                client.write(
                    &fwd.x,
                    cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(x_row_host).to_vec()),
                );
                unsafe {
                    crate::norms_cubecl::RmsNormCubeCL::launch::<ActiveRuntime>(
                        client,
                        fwd.x.clone(),
                        fwd.final_norm.clone(),
                        fwd.norm_x.clone(),
                        n,
                        eps,
                    );
                    crate::gemv_ternary_cubecl::GemvTernaryCubeCL::launch::<ActiveRuntime>(
                        client,
                        &fwd.lm_head,
                        fwd.norm_x.clone(),
                        fwd.logits.clone(),
                    );
                }
                let Ok(bytes) = client.read_one(fwd.logits.clone()) else {
                    fail(states);
                    return None;
                };
                let logits = f32::from_bytes(&bytes);
                let mut best = 0usize;
                let mut best_v = logits.first().copied().unwrap_or(0.0);
                for (i, &l) in logits.iter().enumerate() {
                    if l > best_v {
                        best_v = l;
                        best = i;
                    }
                }
                out.push(best as u32);
            }
            stack.mirror_clean.store(true, Ordering::Relaxed);
            stack.spec_dirty.store(true, Ordering::Relaxed);
            stack.spec_pos.store(base_pos + p, Ordering::Relaxed);
            if trace {
                eprintln!(
                    "[734-arm8-trace] verify tail (gemv, {p} rows) {}us",
                    (t0.elapsed() - t_tail).as_micros()
                );
            }
            return Some(WholeOut::Argmax(out));
        }
        // Final-norm gamma mirror (once).
        let final_gamma = stack.final_norm_gamma.get_or_init(|| {
            read_f32(client, &fwd.final_norm).and_then(|v| stream.clone_htod(v.as_slice()).ok())
        });
        let Some(final_gamma) = final_gamma else {
            fail(states);
            return None;
        };
        let (Some(vl), Some(va)) = (verify_logits, verify_argmax) else {
            fail(states);
            return None;
        };
        // 1. p-row final norm (overwrites the last layer's normx — the same
        //    staging the layer loop uses; stream-ordered after it).
        if let Err(e) = unsafe {
            stack.ffn.launch_rmsnorm(
                stream, ra, rm, rs, x, final_gamma, normx, p, n, eps,
            )
        } {
            if trace {
                eprintln!("[734-arm8] verify tail: rmsnorm ({e})");
            }
            fail(states);
            return None;
        }
        // 2. lm_head mirror (lazy — a one-time ~1.6 GB i8 upload on the first
        //    verify chunk; cached in the TernaryHandle thereafter).
        let Some(lh) = mma_mirror(client, stream, &fwd.lm_head) else {
            fail(states);
            return None;
        };
        let vocab = fwd.config.vocab_size;
        // 3. quantize + GEMM → logits [p × vocab] (the activation-quant arm
        //    resolves inside the launcher — Issue 884 T2a).
        if let Err(e) = stack
            .mma
            .launch_prefill_quantize(stream, normx, scratch, n, p)
            .and_then(|()| {
                crate::prefill_cuda_mma::launch_prefill_gemm_cached(
                    &stack.mma,
                    stream,
                    &lh,
                    scratch,
                    vl,
                    vocab,
                    n,
                    p,
                )
            })
        {
            if trace {
                eprintln!("[734-arm8] verify tail: lm_head gemm ({e})");
            }
            fail(states);
            return None;
        }
        // 4. per-row argmax (memset → the packed-u64 convention needs a zero
        //    base — see the Issue-697 kernel doc).
        if stream.memset_zeros(va).is_err()
            || unsafe { stack.ffn.launch_argmax_rows(stream, vl, vocab, p, va) }.is_err()
        {
            fail(states);
            return None;
        }
        // 5. download p packed results → token ids.
        {
            let view = if let Some(v) = va.try_slice(0..p) { v } else {
                    fail(states);
                    return None;
                };
            if stream.memcpy_dtoh(&view, verify_argmax_host).is_err() {
                fail(states);
                return None;
            }
        }
        let out: Vec<u32> = verify_argmax_host[..p]
            .iter()
            .map(|packed| !(*packed as u32))
            .collect();
        // 6. Issue 884 T2a — the NLL gather (only when the caller supplied
        //    targets): lse + target logit per row from the SAME logits the
        //    argmax pass just read; NLL = lse − target, host-side. The
        //    targets upload is `p * 4` bytes per chunk (the tokens_fresh
        //    clone_htod precedent — no persistent staging needed).
        let nll_out: Option<Vec<f32>> = match (verify_nll, nll_targets) {
            (Some(nl), Some(targets)) => {
                debug_assert_eq!(targets.len(), p, "nll targets per row");
                let targets_i32: Vec<i32> = targets.iter().map(|&t| {
                    debug_assert!(t < vocab, "nll target {t} out of vocab");
                    t as i32
                }).collect();
                let targets_dev = stream.clone_htod(&targets_i32).ok()?;
                if unsafe {
                    stack
                        .ffn
                        .launch_nll_rows(stream, vl, &targets_dev, vocab, p, nl)
                }
                .is_err()
                {
                    fail(states);
                    return None;
                }
                {
                    let view = if let Some(v) = nl.try_slice(0..2 * p) { v } else {
                            fail(states);
                            return None;
                        };
                    if stream.memcpy_dtoh(&view, verify_nll_host).is_err() {
                        fail(states);
                        return None;
                    }
                }
                Some(
                    verify_nll_host[..2 * p]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|pair| pair[0] - pair[1])
                        .collect(),
                )
            }
            _ => None,
        };
        stack.mirror_clean.store(true, Ordering::Relaxed);
        stack.spec_dirty.store(true, Ordering::Relaxed);
        stack.spec_pos.store(base_pos + p, Ordering::Relaxed);
        if trace {
            eprintln!(
                "[734-arm8-trace] verify tail {}us (incl. first-use mirrors)",
                (t0.elapsed() - t_tail).as_micros()
            );
        }
        return Some(match nll_out {
            Some(nll) => WholeOut::ArgmaxNll(out, nll),
            None => WholeOut::Argmax(out),
        });
    }

    // ── Tail ──
    if !is_final {
        stack.mirror_clean.store(true, Ordering::Relaxed);
        if spec {
            stack.spec_dirty.store(true, Ordering::Relaxed);
            stack.spec_pos.store(base_pos + p, Ordering::Relaxed);
        }
        if trace {
            eprintln!(
                "[734-arm8-trace] p={p}: sync {}us alloc {}us embed {}us layers {}us(+{}) wb {}us(+{})",
                t_sync.as_micros(),
                t_alloc.as_micros(),
                t_embed.as_micros(),
                t_layers.as_micros(),
                (t_layers - t_embed).as_micros(),
                t_wb.as_micros(),
                (t_wb - t_layers).as_micros(),
            );
        }
        return Some(WholeOut::Logits(Vec::new()));
    }
    // ONE crossing: the final x row (20 KB) → self.x (decode continuity +
    // the CubeCL tail's final-norm input) — then the unmodified CubeCL
    // final norm + GemvTernary lm_head (bit-identical by construction).
    // Issue 980 T4-ALT site (g): on a folded model the final norm + rotation
    // run on the CUDARC side (the CubeCL tail has no rotation; the lm_head
    // weights are folded-stored and consume the ROTATED row as-is) — so the
    // p-row final-norm lands in `normx`, rotates in place, and the rotated
    // FINAL row crosses to `fwd.norm_x`, skipping the CubeCL RmsNorm. The
    // PRIMAL row still crosses to `fwd.x` (decode-continuity semantics
    // unchanged).
    let f = core::mem::size_of::<f32>();
    {
        let row_start = (p - 1) * n;
        let view = x.try_slice(row_start..p * n)?;
        stream.memcpy_dtoh(&view, x_row_host).ok()?;
    }
    let t_xrow = t0.elapsed();
    client.write(
        &fwd.x,
        cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(x_row_host).to_vec()),
    );
    let t_xwrite = t0.elapsed();
    let folded_tail = rotation.as_ref().map(|rot| {
        // The final-norm gamma mirror (the verify tail's OnceLock — shared).
        let gamma = stack.final_norm_gamma.get_or_init(|| {
            read_f32(client, &fwd.final_norm).and_then(|v| stream.clone_htod(v.as_slice()).ok())
        });
        let gamma = gamma.as_ref()?;
        // p-row final norm into normx (the layer loop is complete — normx is
        // scratch; the same launch_rmsnorm kernel the layer loop's `rms`
        // closure wraps — that closure is scoped inside `run_layers`) + the
        // forward rotation (all p rows; only the final row crosses — one
        // extra FWHT pass, ~0.02% of the chunk wall).
        unsafe { stack.ffn.launch_rmsnorm(stream, ra, rm, rs, x, gamma, normx, p, n, eps).ok()?; }
        rot.kernels
            .fwht_rotate_forward_batched(
                stream,
                normx,
                rot.signs_for_width(n),
                p,
                n,
                rot.block_size,
            )
            .ok()?;
        let row_start = (p - 1) * n;
        let view = normx.try_slice(row_start..p * n)?;
        stream.memcpy_dtoh(&view, x_row_host).ok()?;
        client.write(
            &fwd.norm_x,
            cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(x_row_host).to_vec()),
        );
        Some(())
    });
    if rotation.is_some() {
        // folded: norm + rotate already done on the cudarc side — skip the
        // CubeCL RmsNorm and go straight to the lm_head. A staging failure
        // here (gamma mirror / kernel / crossing) REFUSES the chunk — the
        // fall-through gate in `prefill_tokens_chunk` then errors loudly
        // (never silently unrotated).
        if folded_tail.is_none() {
            if trace {
                eprintln!("[734-arm8] FALLTHROUGH: folded final-tail staging failed");
            }
            return None;
        }
    } else {
        unsafe {
            crate::norms_cubecl::RmsNormCubeCL::launch::<ActiveRuntime>(
                client,
                fwd.x.clone(),
                fwd.final_norm.clone(),
                fwd.norm_x.clone(),
                n,
                eps,
            );
        }
    }
    let t_norm = t0.elapsed();
    unsafe {
        crate::gemv_ternary_cubecl::GemvTernaryCubeCL::launch::<ActiveRuntime>(
            client,
            &fwd.lm_head,
            fwd.norm_x.clone(),
            fwd.logits.clone(),
        );
    }
    let t_lm = t0.elapsed();
    let Ok(bytes) = client.read_one(fwd.logits.clone()) else {
        return None;
    };
    let t_read = t0.elapsed();
    stack.mirror_clean.store(true, Ordering::Relaxed);
    if spec {
        stack.spec_dirty.store(true, Ordering::Relaxed);
        stack.spec_pos.store(base_pos + p, Ordering::Relaxed);
    }
    if trace {
        eprintln!(
            "[734-arm8-trace] tail split: xrow {}us xwrite +{}us norm +{}us lmhead +{}us read +{}us",
            t_xrow.as_micros(),
            (t_xwrite - t_xrow).as_micros(),
            (t_norm - t_xwrite).as_micros(),
            (t_lm - t_norm).as_micros(),
            (t_read - t_lm).as_micros(),
        );
        eprintln!(
            "[734-arm8-trace] p={p}: sync {}us alloc {}us embed {}us layers {}us(+{}) wb {}us(+{}) tail {}us",
            t_sync.as_micros(),
            t_alloc.as_micros(),
            t_embed.as_micros(),
            t_layers.as_micros(),
            (t_layers - t_embed).as_micros(),
            t_wb.as_micros(),
            (t_wb - t_layers).as_micros(),
            t0.elapsed().as_micros(),
        );
    }
    let _ = f;
    Some(WholeOut::Logits(f32::from_bytes(&bytes).to_vec()))
}

// ---------------------------------------------------------------------------
// Issue 742 T1.8 — the verify loop's public surface
// ---------------------------------------------------------------------------

/// Issue 742 T1.8 — run ONE verify chunk and return the argmax token id at
/// every position (row `i` predicts the token at `base_pos + i + 1`).
///
/// Requires `set_prefill_spec_mode(true)` (the chunk runs with the writeback
/// elided — the mirrors carry the state; the pre-chunk snapshot is taken for
/// the rollback consumer). On `None` the arm fell through (the caller must
/// not treat the state as advanced).
///
/// After a SUCCESSFUL call the mirrors sit at `base_pos + p`; pair with
/// [`prefill_verify_rollback`] + [`prefill_verify_advance`] on a draft
/// mismatch.
pub fn prefill_verify_chunk_argmax(
    fwd: &TernaryDeltanetGpuForward,
    tokens: &[usize],
    base_pos: usize,
) -> Option<Vec<u32>> {
    match whole_prefill_inner(fwd, tokens, base_pos, /* is_final */ true, /* verify_tail */ true, None)
    {
        Some(WholeOut::Argmax(v)) => Some(v),
        _ => None,
    }
}

/// Issue 884 T2a — the per-position NLL/argmax verify chunk: the SAME
/// whole-prefill chunk as [`prefill_verify_chunk_argmax`], with the tail's
/// lm_head logits additionally scored against `targets` — one NLL per row
/// (`nll[i] = logsumexp(logits_i) - logits_i[targets[i]]`, row `i` being
/// the prediction for the token at `base_pos + i + 1`). The target for row
/// `i` is `targets[i]`; a teacher-forced walk passes the corpus token at
/// `base_pos + i + 1`.
///
/// Requires `set_prefill_spec_mode(true)`; the chunk state semantics are
/// identical (mirrors advance to `base_pos + p`). The NLL gather is a
/// second read of the device-side `[p × vocab]` logits — deterministic
/// (fixed-order online-softmax accumulation, no atomics), so the A/B arms
/// of the activation-quant instrument are paired row by row.
///
/// On `None` the arm fell through (the caller must not treat the state as
/// advanced).
#[allow(clippy::type_complexity, reason = "one instrument return shape")]
pub fn prefill_verify_chunk_nll(
    fwd: &TernaryDeltanetGpuForward,
    tokens: &[usize],
    base_pos: usize,
    targets: &[usize],
) -> Option<(Vec<u32>, Vec<f32>)> {
    if targets.len() != tokens.len() {
        return None;
    }
    match whole_prefill_inner(
        fwd,
        tokens,
        base_pos,
        /* is_final */ true,
        /* verify_tail */ true,
        Some(targets),
    ) {
        Some(WholeOut::ArgmaxNll(am, nll)) => Some((am, nll)),
        _ => None,
    }
}

/// Issue 742 T1.8 — advance the state over an ACCEPTED prefix (the verify
/// loop's re-commit after a rollback, or any non-final spec chunk). Runs the
/// same whole-prefill chunk with the tail elided; the tokens are committed
/// by construction (they came from a verified greedy stream). On `false` the
/// arm fell through — the caller's loop is invalid.
pub fn prefill_verify_advance(
    fwd: &TernaryDeltanetGpuForward,
    tokens: &[usize],
    base_pos: usize,
) -> bool {
    matches!(
        whole_prefill_inner(
            fwd,
            tokens,
            base_pos,
            /* is_final */ false,
            /* verify_tail */ false,
            None
        ),
        Some(WholeOut::Logits(_))
    )
}

/// Issue 742 T1.8 — rollback the mirrors to the pre-chunk snapshot after a
/// verify mismatch (restores dn/conv/KV mirrors, dtod on the compute stream —
/// ordered after the verify chunk's kernels).
///
/// The KV rows `[base_pos, ...)` still hold the failed chunk's writes; they
/// are overwritten-before-read by the re-run chunk (its KV fill covers
/// exactly `[base_pos, base_pos + accepted)` and the next chunk starts at
/// `base_pos + accepted`). `spec_pos` resets to `base_pos` so a later flush
/// covers the right range.
pub fn prefill_verify_rollback(base_pos: usize) -> bool {
    let Some(stack) = full_stack() else {
        return false;
    };
    let Some(states_lock) = stack.states.get().and_then(|s| s.as_ref()) else {
        return false;
    };
    let Some(snap_lock) = stack.wb_snapshot.get().and_then(|s| s.as_ref()) else {
        return false;
    };
    let (Ok(mut states), Ok(snap)) = (states_lock.lock(), snap_lock.lock()) else {
        return false;
    };
    let cp = |src: &Option<CudaSlice<f32>>,
              dst: &mut Option<CudaSlice<f32>>|
     -> Result<(), String> {
        if let (Some(s), Some(d)) = (src, dst) {
            stack
                .stream
                .memcpy_dtod(s, d)
                .map_err(|e| format!("verify rollback dtod: {e}"))?;
        }
        Ok(())
    };
    for (src, dst) in snap.dn_states.iter().zip(states.dn_states.iter_mut()) {
        if cp(src, dst).is_err() {
            return false;
        }
    }
    for (src, dst) in snap.conv_states.iter().zip(states.conv_states.iter_mut()) {
        if cp(src, dst).is_err() {
            return false;
        }
    }
    for (src, dst) in snap.kv_k.iter().zip(states.kv_k.iter_mut()) {
        if cp(src, dst).is_err() {
            return false;
        }
    }
    for (src, dst) in snap.kv_v.iter().zip(states.kv_v.iter_mut()) {
        if cp(src, dst).is_err() {
            return false;
        }
    }
    stack.spec_pos.store(base_pos, Ordering::Relaxed);
    true
}
