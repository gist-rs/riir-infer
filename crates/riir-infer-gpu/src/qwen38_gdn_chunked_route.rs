//! The dense-qwen38 Metal GDN prefill chunked-scan route (issue 1150
//! follow-up 2).
//!
//! Wires the Issue-734-T4 correct Delta-rule chunkwise family
//! ([`crate::deltanet_delta_rule_chunked`] — zero code consumers until now)
//! into the dense-qwen38 wgpu forward as an OPT-IN arm behind
//! `RIIR_QWEN38_GDN_CHUNKED=1` (promote-only env law: arms only on the exact
//! literal; unset/anything else = the incumbent
//! `DeltanetRecurrenceMultiTokenCubeCL` path byte-identical).
//!
//! # Why this lane exists
//!
//! The stage-timer pass measured the GDN scan (48 layers) at 31–32% of the
//! synchronized prefill pass (`.benchmarks/i1150_qwen38_wgpu_session.md`).
//! The incumbent multi-token kernel is the sequential per-token scan with the
//! dispatch gaps removed — one dispatch per layer-chunk, but the token chain
//! is still SERIAL inside every (head, row) cube. The chunkwise solve
//! re-expresses the same recursion as O(window) boundary transitions +
//! intra-window batched kernels (the algebra is exact — Bench 662's corrected
//! algorithm), trading the serial chain for parallel matmul-shaped work.
//!
//! # Numerics class (disclosed, never hidden)
//!
//! The chunkwise solve is algebraically exact but NOT bit-identical: its
//! summation order (log-space decay ratios + the `(I+X)` forward
//! substitution) diverges from the sequential path at the ~4e-3/layer class
//! measured on real model data (riir-infer Issue 734 T4 record: max_rel
//! 1.6–2.8 through 48 layers, argmax STABLE, e2e G1 FAIL at Bonsai's
//! bit-parity gate). That is why the Bonsai scored lane keeps the family
//! OFF. THIS lane is the dense-qwen38 comparison lane (NOT the scored league
//! lane): the arm is a LOSSY-CLASS opt-in per the Issue-1066 precedent —
//! throughput A/B + argmax-stability disclosure, no bit-parity claim, never
//! a default without the GOAT gate.
//!
//! # Dispatch shape
//!
//! Per GDN layer per ingest chunk: `chunk / 64` windows; per window the
//! repack kernel + the 8 chunkwise kernels, all stream-ordered (window N's
//! `state_update` feeds window N+1's `rhs`/`qs0` through the SAME state
//! handle — ordering is the stream's, no sync). Window C=64: the recorded,
//! kernel-tested shape (Plan 533 dispatch table; the dense forward's conv
//! already walks 64-token sub-chunks, so the windows align).
//!
//! # Known overheads (disclosed, measured in the A/B)
//!
//! - The chunkwise launchers build a small params buffer per call
//!   (`create_from_slice`, not the params cache) — ~3k tiny allocations per
//!   ingest pass at 48 layers × 8 windows. µs-class each; a promotion-time
//!   optimization only if the arm wins.
//! - 9 launches per window vs the incumbent's 1 per layer-chunk: ~3.4k
//!   extra dispatches per pass, ~34 µs-class gap each on the worst reading —
//!   ~0.3% of the GDN stage.

use std::sync::OnceLock;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::cubecl_runtime::ActiveRuntime;
use crate::deltanet_delta_rule_chunked::{
    DeltanetDeltaRuleChunkedCubeCL, DeltanetDrCumDecayCubeCL, DeltanetDrFwdSubCubeCL,
    DeltanetDrGramCubeCL, DeltanetDrOutputCubeCL, DeltanetDrQs0CubeCL, DeltanetDrRhsCubeCL,
    DeltanetDrScatterCubeCL, DeltanetDrStateUpdateCubeCL,
};

/// The chunkwise window (tokens per solve) — the recorded, kernel-tested
/// shape (Plan 533 dispatch table).
pub const GDN_CHUNK_C: usize = 64;

static ARMED: OnceLock<bool> = OnceLock::new();

/// `RIIR_QWEN38_GDN_CHUNKED=1` (the exact literal) arms the route; anything
/// else — including unset — is the incumbent path (promote-only env law).
#[must_use]
pub fn armed() -> bool {
    *ARMED.get_or_init(|| std::env::var("RIIR_QWEN38_GDN_CHUNKED").as_deref() == Ok("1"))
}

/// Per-window scratch, allocated once at session build (the hot path never
/// allocates; every buffer is reused across layers and windows).
pub struct GdnChunkedScratch {
    q_h: Handle,          // [n_head, C, d] head-major repack outputs
    k_h: Handle,          // [n_head, C, d]
    v_h: Handle,          // [n_head, C, d]
    beta_h: Handle,       // [n_head, C]
    alpha_h: Handle,      // [n_head, C]
    log_gamma: Handle,    // [n_head, C]   (cum_decay)
    gamma: Handle,        // [n_head, C]   (cum_decay)
    decay_to_end: Handle, // [n_head, C]   (cum_decay)
    total_decay: Handle,  // [n_head]      (cum_decay)
    x_mat: Handle,        // [n_head, C, C] (gram)
    qkr_mat: Handle,      // [n_head, C, C] (gram)
    rhs: Handle,          // [n_head, C, d] (rhs / fwd_sub)
    qs0: Handle,          // [n_head, C, d] (qs0 / output)
    u_sol: Handle,        // [n_head, C, d] (fwd_sub / output / state_update)
    o_chunk: Handle,      // [n_head, C, d] (output / scatter)
}

impl GdnChunkedScratch {
    /// Allocate zeroed scratch for `n_head` GDN v-heads of value dim `d`
    /// (both 128 on qwen3.8-27B) and the fixed [`GDN_CHUNK_C`] window.
    #[must_use]
    pub fn new(client: &ComputeClient<ActiveRuntime>, n_head: usize, d: usize) -> Self {
        let c = GDN_CHUNK_C;
        let z = |n: usize| client.create_from_slice(&vec![0u8; n * 4]);
        Self {
            q_h: z(n_head * c * d),
            k_h: z(n_head * c * d),
            v_h: z(n_head * c * d),
            beta_h: z(n_head * c),
            alpha_h: z(n_head * c),
            log_gamma: z(n_head * c),
            gamma: z(n_head * c),
            decay_to_end: z(n_head * c),
            total_decay: z(n_head),
            x_mat: z(n_head * c * c),
            qkr_mat: z(n_head * c * c),
            rhs: z(n_head * c * d),
            qs0: z(n_head * c * d),
            u_sol: z(n_head * c * d),
            o_chunk: z(n_head * c * d),
        }
    }
}

// ---------------------------------------------------------------------------
// Repack: token-major window → head-major window (the one new kernel this
// lane adds; everything downstream is the shared chunkwise family)
// ---------------------------------------------------------------------------

/// Token-major qkvx/beta/decay window → head-major q/k/v/beta/alpha.
///
/// The dense forward's buffers are token-major
/// (`qkvx[t, q|k|v blocks of n_head·d]`, `beta[t, h]`, `decay[t, h]`); the
/// chunkwise family reads head-major (`q[h, t, d]`, `beta[h, t]`). One thread
/// per (h, t, j) with j spanning `3d` vector slots + 2 scalar slots — one
/// dispatch per window, no data-movement kernel chain.
#[cube(launch_unchecked)]
fn qwen38_gdn_chunk_repack_f32(
    qkvx: &[f32],
    beta: &[f32],
    decay: &[f32],
    params: &[f32], // [c, v_dim, n_head, d]
    q_h: &mut [f32],
    k_h: &mut [f32],
    v_h: &mut [f32],
    beta_h: &mut [f32],
    alpha_h: &mut [f32],
) {
    let c = params[0usize] as usize;
    let v_dim = params[1usize] as usize;
    let n_head = params[2usize] as usize;
    let d = params[3usize] as usize;
    let idx = ABSOLUTE_POS;
    let j_span = 3 * d + 2;
    let per_head = c * j_span;
    let total = n_head * per_head;
    if idx >= total {
        terminate!();
    }
    let h = idx / per_head;
    let rem = idx % per_head;
    let t = rem / j_span;
    let j = rem % j_span;
    let head_off = h * d;
    if j < 2 * d {
        // q block slot (j < d) or k block slot (j < 2d) — the expanded-k
        // layout the multi-token kernel reads (head h at [h·d, (h+1)·d)).
        let dim = j % d;
        let src = if j < d {
            t * 3 * v_dim + head_off + dim
        } else {
            t * 3 * v_dim + v_dim + head_off + dim
        };
        let dst = h * c * d + t * d + dim;
        if j < d {
            q_h[dst] = qkvx[src];
        } else {
            k_h[dst] = qkvx[src];
        }
    } else if j < 3 * d {
        let dim = j - 2 * d;
        let src = t * 3 * v_dim + 2 * v_dim + head_off + dim;
        v_h[h * c * d + t * d + dim] = qkvx[src];
    } else if j == 3 * d {
        beta_h[h * c + t] = beta[t * n_head + h];
    } else {
        alpha_h[h * c + t] = decay[t * n_head + h];
    }
}

/// Launcher for the repack kernel (params through the cache — this lane runs
/// it per window per layer).
pub struct Qwen38GdnChunkRepackCubeCL;

impl Qwen38GdnChunkRepackCubeCL {
    /// # Safety
    /// - `qkvx`: `c * 3 * n_head * d`; `beta`/`decay`: `c * n_head` each
    ///   (window views — the caller offsets into the full-chunk buffers).
    /// - `q_h`/`k_h`/`v_h`: `n_head * c * d`; `beta_h`/`alpha_h`: `n_head * c`.
    #[allow(
        clippy::too_many_arguments,
        reason = "GPU kernel launch/dispatch: many buffer handles are inherent to the fused-kernel interface"
    )]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        qkvx_handle: Handle,
        beta_handle: Handle,
        decay_handle: Handle,
        q_h_handle: Handle,
        k_h_handle: Handle,
        v_h_handle: Handle,
        beta_h_handle: Handle,
        alpha_h_handle: Handle,
        c: usize,
        v_dim: usize,
        n_head: usize,
        d: usize,
    ) {
        let params: [f32; 4] = [c as f32, v_dim as f32, n_head as f32, d as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let total = n_head * c * (3 * d + 2);
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            qwen38_gdn_chunk_repack_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(qkvx_handle, c * 3 * v_dim),
                BufferArg::from_raw_parts(beta_handle, c * n_head),
                BufferArg::from_raw_parts(decay_handle, c * n_head),
                BufferArg::from_raw_parts(params_handle, 4),
                BufferArg::from_raw_parts(q_h_handle, n_head * c * d),
                BufferArg::from_raw_parts(k_h_handle, n_head * c * d),
                BufferArg::from_raw_parts(v_h_handle, n_head * c * d),
                BufferArg::from_raw_parts(beta_h_handle, n_head * c),
                BufferArg::from_raw_parts(alpha_h_handle, n_head * c),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The window pipeline
// ---------------------------------------------------------------------------

/// Run ONE [`GDN_CHUNK_C`]-token window of the chunkwise pipeline for one GDN
/// layer: repack → cum_decay → gram → rhs → qs0 → fwd_sub → output →
/// state_update → scatter, all stream-ordered (no sync; the NEXT window's
/// rhs/qs0 read the state this window's state_update wrote).
///
/// # Safety
///
/// Caller must ensure (shapes in f32 elements; windows are offset views into
/// the full-chunk buffers):
/// - `qkvx_window`: `c * 3 * n_head * d` readable (the L2-normalized expand
///   output — the same layout the multi-token kernel reads)
/// - `beta_window` / `decay_window`: `c * n_head` each (token-major)
/// - `state`: `n_head * d * d`, updated IN PLACE by the window
/// - `out_window`: `c * n_head * d` writable (token-major — the scatter
///   writes exactly the incumbent `rec_b` layout)
/// - `d` satisfies [`DeltanetDeltaRuleChunkedCubeCL::supports`] (128 here)
/// - the beta/decay values are the `DeltanetBetaDecayBatchedCubeCL` outputs
///   (β scalar and the per-token multiplicative decay α)
#[allow(
    clippy::too_many_arguments,
    reason = "GPU kernel pipeline seam: window views + scratch handles are the interface"
)]
pub unsafe fn ingest_window(
    client: &ComputeClient<ActiveRuntime>,
    s: &GdnChunkedScratch,
    qkvx_window: Handle,
    beta_window: Handle,
    decay_window: Handle,
    state: Handle,
    out_window: Handle,
    n_head: usize,
    d: usize,
) {
    let c = GDN_CHUNK_C;
    debug_assert!(
        DeltanetDeltaRuleChunkedCubeCL::supports(d),
        "chunked-gdn route requires d % 4 == 0 (got {d})"
    );
    let v_dim = n_head * d;
    unsafe {
        Qwen38GdnChunkRepackCubeCL::launch(
            client,
            qkvx_window,
            beta_window,
            decay_window,
            s.q_h.clone(),
            s.k_h.clone(),
            s.v_h.clone(),
            s.beta_h.clone(),
            s.alpha_h.clone(),
            c,
            v_dim,
            n_head,
            d,
        );
        DeltanetDrCumDecayCubeCL::launch(
            client,
            s.alpha_h.clone(),
            s.log_gamma.clone(),
            s.gamma.clone(),
            s.decay_to_end.clone(),
            s.total_decay.clone(),
            n_head,
            c,
        );
        DeltanetDrGramCubeCL::launch(
            client,
            s.q_h.clone(),
            s.k_h.clone(),
            s.beta_h.clone(),
            s.log_gamma.clone(),
            s.x_mat.clone(),
            s.qkr_mat.clone(),
            n_head,
            c,
            d,
        );
        // rhs + qs0 read the window's ENTRY state — they must precede
        // state_update (stream order is the ordering).
        DeltanetDrRhsCubeCL::launch(
            client,
            s.k_h.clone(),
            s.v_h.clone(),
            s.beta_h.clone(),
            s.gamma.clone(),
            state.clone(),
            s.rhs.clone(),
            n_head,
            c,
            d,
        );
        DeltanetDrQs0CubeCL::launch(
            client,
            s.q_h.clone(),
            s.gamma.clone(),
            state.clone(),
            s.qs0.clone(),
            n_head,
            c,
            d,
        );
        DeltanetDrFwdSubCubeCL::launch(
            client,
            s.x_mat.clone(),
            s.rhs.clone(),
            s.u_sol.clone(),
            n_head,
            c,
            d,
        );
        DeltanetDrOutputCubeCL::launch(
            client,
            s.qs0.clone(),
            s.qkr_mat.clone(),
            s.u_sol.clone(),
            s.o_chunk.clone(),
            n_head,
            c,
            d,
        );
        DeltanetDrStateUpdateCubeCL::launch(
            client,
            s.k_h.clone(),
            s.u_sol.clone(),
            s.decay_to_end.clone(),
            s.total_decay.clone(),
            state.clone(),
            n_head,
            c,
            d,
        );
        DeltanetDrScatterCubeCL::launch(client, s.o_chunk.clone(), out_window, n_head, c, d);
    }
}
