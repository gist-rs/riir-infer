//! Correct Delta-rule chunkwise parallel prefill recurrence (Issue 734 T4).
//!
//! Bench 662 (Plan 533) G1-FAILED the original chunked kernels because they
//! implemented **standard linear attention** (`S = α·S + β·v⊗kᵀ`) while the
//! production sequential kernel implements the **Delta rule**
//! (`S = α·S + β·(v − α·Sᵀk)⊗kᵀ` — the retrieval term makes the update depend
//! on the CURRENT state). This module implements the correct chunkwise
//! algorithm Bench 662 §"The correct chunkwise algorithm" recorded (the linear-
//! chain specialization of `katgpt-core::gdn_tree_verify`, which uses the same
//! math for speculative-decode tree verification):
//!
//! ```text
//! For a chunk of C tokens with boundary state S₀ (state layout [h, d_v, d_k]):
//!   γ_i = ∏_{m≤i} α_m                       (cumulative decay, log-space)
//!   X[i][j]   = β_i·(γ_i/γ_j)·(k_i·k_j)     j < i  (interaction matrix)
//!   RHS[i]    = β_i·v_i − β_i·γ_i·(S₀·k_i)
//!   (I + X)·U = RHS                          (forward substitution → U = δ)
//!   O_i       = (γ_i·(S₀·q_i) + Σ_{j≤i} (γ_i/γ_j)·(q_i·k_j)·U_j) / √d
//!   S_end     = γ_{C-1}·S₀ + Σ_i (γ_{C-1}/γ_i)·U_i ⊗ k_i
//! ```
//!
//! Derivation: unrolling `S_i = α_i·S_{i-1} + δ_i⊗k_i` with
//! `δ_i = β_i·v_i − β_i·γ_i·(S₀·k_i) − β_i·γ_i·Σ_{j<i}(1/γ_j)·(k_j·k_i)·δ_j`
//! gives exactly `(I+X)U = RHS`; `U_i = δ_i` is the per-token delta-rule
//! correction, so the solve IS the original recursion re-expressed, not an
//! approximation (no conditioning concern — unit lower-triangular, solved by
//! the same sequential substitution the sequential kernel performs).
//!
//! # Numerical safety (log-space decay with a floor)
//!
//! Cumulative decay is computed in log space (`log γ_i = Σ log α_m`) so decay
//! ratios are `exp(log γ_i − log γ_j)` — differences, never ratios of
//! underflowed products. `log γ` is floored at −60: α can legitimately hit
//! 0.0f32 (`exp(g)` underflow in `deltanet_beta_decay_f32`), and `log(0) = −∞`
//! would poison ratios of two underflowed values into `exp(−∞−(−∞)) = NaN`.
//! With the floor, a both-floored ratio reads 1; the true contributions at
//! γ < e⁻⁶⁰ ≈ 8.8e−27 sit far below the sequential path's own f32 flush
//! behavior, so the floor is behaviorally exact for every reachable input.
//!
//! # Dispatch shape (per chunk, C = `PREFILL_CHUNK_SIZE` = 64)
//!
//! | # | Kernel | Grid (threads) | Work/thread |
//! |---|---|---|---|
//! | 1 | `cum_decay` | n_head | C log + 2C exp (serial) |
//! | 2 | `gram` | n_head·C·C | 2 dots of length d + 2 exp |
//! | 3 | `rhs` | n_head·C·d | dot(S₀ row, k_i) + fold β,γ,v |
//! | 4 | `qs0` | n_head·C·d | dot(S₀ row, q_i) + fold γ |
//! | 5 | `fwd_sub` | n_head·d/4 | C²/2 × 4 FMAs (serial in i, RB=4 over d) |
//! | 6 | `output` | n_head·C·d | avg C/2 FMAs + 1/√d scale |
//! | 7 | `scatter` | C·n_head·d | pure transpose |
//! | 8 | `state_update` | n_head·d·d | C FMAs, in-place on state |
//!
//! Replaces the sequential path's C per-token dependent dispatches with 11
//! independent-schedule dispatches per chunk (3 layout extracts reuse the
//! Plan 533 kernels). On the 4090 the sequential chain costs ~34 µs of
//! GPU-side inter-kernel gap per dependent launch (Issue 734 T1a), so 2048
//! serialized dispatches/layer dominate the recurrence wall time.
//!
//! # FLOPs
//!
//! ~7% more arithmetic than sequential (Bench 662's analysis) but massively
//! parallel — the sequential dependency survives only inside `fwd_sub`'s
//! i-loop, which is thread-local: each (head, d/4-slice) thread owns the full
//! i-sweep for its 4 components, and its `j < i` reads are its OWN earlier
//! writes — zero cross-thread synchronization.

use cubecl::prelude::*;
use cubecl::server::Handle;

/// log-space floor for cumulative decay (see module docs — NaN guard for
/// underflowed α; contributions below e⁻⁶⁰ are beneath the sequential path's
/// own f32 flush behavior).
const LOG_GAMMA_FLOOR: f32 = -60.0;

/// Register-blocking width for the forward-substitution kernel (independent
/// accumulator chains per thread — hides the serial FMA-chain latency ~4×).
const FWD_SUB_RB: usize = 4;

/// Facade for the correct Delta-rule chunked pipeline.
pub struct DeltanetDeltaRuleChunkedCubeCL;

impl DeltanetDeltaRuleChunkedCubeCL {
    /// All kernels are shape-generic except `fwd_sub`, which register-blocks
    /// the value dimension in groups of 4 (named accumulators — cube locals
    /// are SSA, no runtime-indexed arrays).
    pub fn supports(d: usize) -> bool {
        d.is_multiple_of(FWD_SUB_RB) && d >= FWD_SUB_RB
    }
}

// ---------------------------------------------------------------------------
// 1. Cumulative decay (log-space) — one thread per head
// ---------------------------------------------------------------------------

/// Compute `log_gamma`, `gamma`, `decay_to_end`, `total_decay` per head.
///
/// - `alpha` [h, C] in; `log_gamma`/`gamma`/`decay_to_end` [h, C], `total_decay` [h] out.
/// - `decay_to_end[i] = γ_{C-1}/γ_i`, `total_decay = γ_{C-1}`.
#[cube(launch_unchecked)]
fn deltanet_dr_cum_decay_f32(
    alpha: &[f32],
    log_gamma: &mut [f32],
    gamma: &mut [f32],
    decay_to_end: &mut [f32],
    total_decay: &mut [f32],
    params: &[f32], // [n_head, C]
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let h = ABSOLUTE_POS;
    if h >= n_head {
        terminate!();
    }
    let base = h * c;
    let floor = f32::new(LOG_GAMMA_FLOOR);
    let mut acc = f32::new(0.0f32);
    let mut i = 0usize;
    while i < c {
        let a = alpha[base + i];
        acc = acc + a.ln();
        if acc < floor {
            acc = floor;
        }
        log_gamma[base + i] = acc;
        gamma[base + i] = acc.exp();
        i += 1;
    }
    let lg_end = log_gamma[base + (c - 1)];
    total_decay[h] = lg_end.exp();
    let mut j = 0usize;
    while j < c {
        decay_to_end[base + j] = (lg_end - log_gamma[base + j]).exp();
        j += 1;
    }
}

/// Launcher for the cumulative-decay kernel.
pub struct DeltanetDrCumDecayCubeCL;

impl DeltanetDrCumDecayCubeCL {
    /// # Safety
    /// - `alpha_handle`: `n_head * C` f32 elements (all in (0, 1]; 0 is
    ///   tolerated — the log-floor clamps it, see module docs).
    /// - `log_gamma`/`gamma`/`decay_to_end`: `n_head * C`; `total_decay`: `n_head`.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        alpha_handle: Handle,
        log_gamma_handle: Handle,
        gamma_handle: Handle,
        decay_to_end_handle: Handle,
        total_decay_handle: Handle,
        n_head: usize,
        c: usize,
    ) {
        let params: [f32; 2] = [n_head as f32, c as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        unsafe {
            deltanet_dr_cum_decay_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(n_head.max(1) as u32),
                BufferArg::from_raw_parts(alpha_handle, n_head * c),
                BufferArg::from_raw_parts(log_gamma_handle, n_head * c),
                BufferArg::from_raw_parts(gamma_handle, n_head * c),
                BufferArg::from_raw_parts(decay_to_end_handle, n_head * c),
                BufferArg::from_raw_parts(total_decay_handle, n_head),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Interaction (X) + decay-weighted QK Gram matrices — one thread per (h, i, j)
// ---------------------------------------------------------------------------

/// Compute `x_mat[h,i,j] = β_i·(γ_i/γ_j)·(k_i·k_j)` for `j < i` (0 elsewhere)
/// and `qkr_mat[h,i,j] = (γ_i/γ_j)·(q_i·k_j)` for `j ≤ i` (0 elsewhere).
///
/// The decay ratios are folded in HERE (one exp per matrix element) so the
/// downstream `fwd_sub`/`output` kernels never call exp in their hot loops.
#[cube(launch_unchecked)]
fn deltanet_dr_gram_f32(
    q: &[f32],
    k: &[f32],
    beta: &[f32],
    log_gamma: &[f32],
    x_mat: &mut [f32],
    qkr_mat: &mut [f32],
    params: &[f32], // [n_head, C, d]
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let d = params[2usize] as usize;
    let idx = ABSOLUTE_POS;
    let total = n_head * c * c;
    if idx >= total {
        terminate!();
    }
    let cc = c * c;
    let h = idx / cc;
    let rem = idx % cc;
    let i = rem / c;
    let j = rem % c;

    let vbase = h * c * d;
    let ki = vbase + i * d;
    let kj = vbase + j * d;
    let qi = vbase + i * d;

    let mut kk = f32::new(0.0f32);
    let mut qk = f32::new(0.0f32);
    let mut m = 0usize;
    while m < d {
        kk = kk + k[ki + m] * k[kj + m];
        qk = qk + q[qi + m] * k[kj + m];
        m += 1;
    }

    let lg_i = log_gamma[h * c + i];
    let lg_j = log_gamma[h * c + j];
    let off = h * cc + i * c + j;

    if j < i {
        let ratio = (lg_i - lg_j).exp();
        x_mat[off] = beta[h * c + i] * ratio * kk;
    } else {
        x_mat[off] = f32::new(0.0f32);
    }
    if j <= i {
        qkr_mat[off] = (lg_i - lg_j).exp() * qk;
    } else {
        qkr_mat[off] = f32::new(0.0f32);
    }
}

/// Launcher for the Gram kernel.
pub struct DeltanetDrGramCubeCL;

impl DeltanetDrGramCubeCL {
    /// # Safety
    /// - `q`/`k`: `n_head * C * d`; `beta`: `n_head * C`; `log_gamma`: `n_head * C`.
    /// - `x_mat`/`qkr_mat`: `n_head * C * C` each.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q_handle: Handle,
        k_handle: Handle,
        beta_handle: Handle,
        log_gamma_handle: Handle,
        x_mat_handle: Handle,
        qkr_mat_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        let params: [f32; 3] = [n_head as f32, c as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * c * c;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_dr_gram_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(q_handle, n_head * c * d),
                BufferArg::from_raw_parts(k_handle, n_head * c * d),
                BufferArg::from_raw_parts(beta_handle, n_head * c),
                BufferArg::from_raw_parts(log_gamma_handle, n_head * c),
                BufferArg::from_raw_parts(x_mat_handle, total),
                BufferArg::from_raw_parts(qkr_mat_handle, total),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. RHS: β_i·(v_i − γ_i·(S₀·k_i)) — one thread per (h, i, r)
// ---------------------------------------------------------------------------

/// `(S₀·k_i)[r] = Σ_c S₀[h,r,c]·k_i[c]` — contraction over the K columns of
/// the state row (the sequential kernel's `kv_mem` retrieval).
#[cube(launch_unchecked)]
fn deltanet_dr_rhs_f32(
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    gamma: &[f32],
    state: &[f32],
    rhs: &mut [f32],
    params: &[f32], // [n_head, C, d]
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let d = params[2usize] as usize;
    let idx = ABSOLUTE_POS;
    let total = n_head * c * d;
    if idx >= total {
        terminate!();
    }
    let cd = c * d;
    let h = idx / cd;
    let rem = idx % cd;
    let i = rem / d;
    let r = rem % d;

    let vec_off = h * cd + i * d;
    let s_row = h * d * d + r * d;

    let mut ks0 = f32::new(0.0f32);
    let mut m = 0usize;
    while m < d {
        ks0 = ks0 + state[s_row + m] * k[vec_off + m];
        m += 1;
    }
    let g = gamma[h * c + i];
    let b = beta[h * c + i];
    rhs[idx] = b * (v[vec_off + r] - g * ks0);
}

/// Launcher for the RHS kernel.
pub struct DeltanetDrRhsCubeCL;

impl DeltanetDrRhsCubeCL {
    /// # Safety
    /// - `k`/`v`: `n_head * C * d`; `beta`/`gamma`: `n_head * C`;
    ///   `state`: `n_head * d * d`; `rhs`: `n_head * C * d`.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        k_handle: Handle,
        v_handle: Handle,
        beta_handle: Handle,
        gamma_handle: Handle,
        state_handle: Handle,
        rhs_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        let params: [f32; 3] = [n_head as f32, c as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * c * d;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_dr_rhs_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(k_handle, total),
                BufferArg::from_raw_parts(v_handle, total),
                BufferArg::from_raw_parts(beta_handle, n_head * c),
                BufferArg::from_raw_parts(gamma_handle, n_head * c),
                BufferArg::from_raw_parts(state_handle, n_head * d * d),
                BufferArg::from_raw_parts(rhs_handle, total),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. QS0: γ_i·(S₀·q_i) — the pre-scaled boundary-state readout term
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn deltanet_dr_qs0_f32(
    q: &[f32],
    gamma: &[f32],
    state: &[f32],
    qs0: &mut [f32],
    params: &[f32], // [n_head, C, d]
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let d = params[2usize] as usize;
    let idx = ABSOLUTE_POS;
    let total = n_head * c * d;
    if idx >= total {
        terminate!();
    }
    let cd = c * d;
    let h = idx / cd;
    let rem = idx % cd;
    let i = rem / d;
    let r = rem % d;

    let vec_off = h * cd + i * d;
    let s_row = h * d * d + r * d;

    let mut acc = f32::new(0.0f32);
    let mut m = 0usize;
    while m < d {
        acc = acc + state[s_row + m] * q[vec_off + m];
        m += 1;
    }
    qs0[idx] = gamma[h * c + i] * acc;
}

/// Launcher for the QS0 kernel.
pub struct DeltanetDrQs0CubeCL;

impl DeltanetDrQs0CubeCL {
    /// # Safety
    /// - `q`: `n_head * C * d`; `gamma`: `n_head * C`;
    ///   `state`: `n_head * d * d`; `qs0`: `n_head * C * d`.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q_handle: Handle,
        gamma_handle: Handle,
        state_handle: Handle,
        qs0_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        let params: [f32; 3] = [n_head as f32, c as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * c * d;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_dr_qs0_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(q_handle, total),
                BufferArg::from_raw_parts(gamma_handle, n_head * c),
                BufferArg::from_raw_parts(state_handle, n_head * d * d),
                BufferArg::from_raw_parts(qs0_handle, total),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Forward substitution: (I + X) U = RHS
// ---------------------------------------------------------------------------

/// Solve `(I + X)U = RHS` in place (`u_sol` starts as RHS, is overwritten).
///
/// Thread (h, rb) owns value-components `[rb*4, rb*4+4)` of ALL C tokens: the
/// i-sweep is sequential within the thread (its `j < i` reads are its own
/// earlier writes — zero cross-thread synchronization), and the 4 named
/// accumulators give 4 independent FMA chains (cube locals are SSA — no
/// runtime-indexed arrays; the Issue 637 named-accumulator pattern).
#[cube(launch_unchecked)]
fn deltanet_dr_fwd_sub_f32(
    x_mat: &[f32],
    rhs: &[f32],
    u_sol: &mut [f32],
    params: &[f32], // [n_head, C, d]
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let d = params[2usize] as usize;
    let d4 = d / FWD_SUB_RB;
    let idx = ABSOLUTE_POS;
    let total = n_head * d4;
    if idx >= total {
        terminate!();
    }
    let h = idx / d4;
    let rb = idx % d4;
    let r0 = rb * FWD_SUB_RB;

    let xbase = h * c * c;
    let ubase = h * c * d;

    let mut i = 0usize;
    while i < c {
        let iu = ubase + i * d + r0;
        let mut a0 = rhs[iu];
        let mut a1 = rhs[iu + 1];
        let mut a2 = rhs[iu + 2];
        let mut a3 = rhs[iu + 3];

        let xr = xbase + i * c;
        let mut j = 0usize;
        while j < i {
            let xij = x_mat[xr + j];
            let ju = ubase + j * d + r0;
            let u0 = u_sol[ju];
            let u1 = u_sol[ju + 1];
            let u2 = u_sol[ju + 2];
            let u3 = u_sol[ju + 3];
            a0 = a0 - xij * u0;
            a1 = a1 - xij * u1;
            a2 = a2 - xij * u2;
            a3 = a3 - xij * u3;
            j += 1;
        }
        u_sol[iu] = a0;
        u_sol[iu + 1] = a1;
        u_sol[iu + 2] = a2;
        u_sol[iu + 3] = a3;
        i += 1;
    }
}

/// Launcher for the forward-substitution kernel.
pub struct DeltanetDrFwdSubCubeCL;

impl DeltanetDrFwdSubCubeCL {
    /// # Safety
    /// - `x_mat`: `n_head * C * C`; `rhs`/`u_sol`: `n_head * C * d`
    ///   (`u_sol` must NOT alias `rhs` — RHS rows are re-read for every i;
    ///   in production they are distinct scratch buffers).
    /// - `d` must satisfy [`DeltanetDeltaRuleChunkedCubeCL::supports`].
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x_mat_handle: Handle,
        rhs_handle: Handle,
        u_sol_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        debug_assert!(
            DeltanetDeltaRuleChunkedCubeCL::supports(d),
            "fwd_sub kernel requires d % 4 == 0"
        );
        let params: [f32; 3] = [n_head as f32, c as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * d / FWD_SUB_RB;
        let wg = 128usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_dr_fwd_sub_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(x_mat_handle, n_head * c * c),
                BufferArg::from_raw_parts(rhs_handle, n_head * c * d),
                BufferArg::from_raw_parts(u_sol_handle, n_head * c * d),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Output: O_i = scale · (qs0_i + Σ_{j≤i} qkr[i][j] · U_j)
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn deltanet_dr_output_f32(
    qs0: &[f32],
    qkr_mat: &[f32],
    u_sol: &[f32],
    o_chunk: &mut [f32],
    params: &[f32], // [n_head, C, d, scale]
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let d = params[2usize] as usize;
    let scale = params[3usize];
    let idx = ABSOLUTE_POS;
    let total = n_head * c * d;
    if idx >= total {
        terminate!();
    }
    let cd = c * d;
    let h = idx / cd;
    let rem = idx % cd;
    let i = rem / d;
    let r = rem % d;

    let mut acc = qs0[idx];
    let qbase = h * c * c + i * c;
    let ubase = h * cd;
    let mut j = 0usize;
    while j <= i {
        let qkr = qkr_mat[qbase + j];
        let u = u_sol[ubase + j * d + r];
        acc = acc + qkr * u;
        j += 1;
    }
    o_chunk[idx] = acc * scale;
}

/// Launcher for the output kernel.
pub struct DeltanetDrOutputCubeCL;

impl DeltanetDrOutputCubeCL {
    /// # Safety
    /// - `qs0`/`u_sol`/`o_chunk`: `n_head * C * d`; `qkr_mat`: `n_head * C * C`.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        qs0_handle: Handle,
        qkr_mat_handle: Handle,
        u_sol_handle: Handle,
        o_chunk_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        let scale = 1.0f32 / (d as f32).sqrt();
        let params: [f32; 4] = [n_head as f32, c as f32, d as f32, scale];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * c * d;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_dr_output_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(qs0_handle, total),
                BufferArg::from_raw_parts(qkr_mat_handle, n_head * c * c),
                BufferArg::from_raw_parts(u_sol_handle, total),
                BufferArg::from_raw_parts(o_chunk_handle, total),
                BufferArg::from_raw_parts(params_handle, 4),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Scatter: head-major [h, C, d] → token-major [C, n_head*d] (pure
//    transpose; the 1/√d scale is folded into the output kernel)
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn deltanet_dr_scatter_f32(
    o: &[f32],
    out: &mut [f32],
    params: &[f32], // [n_head, C, d]
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let d = params[2usize] as usize;
    let idx = ABSOLUTE_POS;
    let total = c * n_head * d;
    if idx >= total {
        terminate!();
    }
    // Token-major out: idx = t*(n_head*d) + h*d + dim.
    let hd = n_head * d;
    let t = idx / hd;
    let rem = idx % hd;
    let h = rem / d;
    let dim = rem % d;
    out[idx] = o[h * c * d + t * d + dim];
}

/// Launcher for the scatter kernel.
pub struct DeltanetDrScatterCubeCL;

impl DeltanetDrScatterCubeCL {
    /// # Safety
    /// - `o`: `n_head * C * d`; `out`: `C * n_head * d`.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        o_handle: Handle,
        out_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        let params: [f32; 3] = [n_head as f32, c as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = c * n_head * d;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_dr_scatter_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(o_handle, n_head * c * d),
                BufferArg::from_raw_parts(out_handle, total),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 8. State update (in-place): S = total_decay·S + Σ_i dte_i·U_i⊗k_i
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn deltanet_dr_state_update_f32(
    k: &[f32],
    u_sol: &[f32],
    decay_to_end: &[f32],
    total_decay: &[f32],
    state: &mut [f32],
    params: &[f32], // [n_head, C, d]
) {
    let n_head = params[0usize] as usize;
    let c = params[1usize] as usize;
    let d = params[2usize] as usize;
    let idx = ABSOLUTE_POS;
    let dd = d * d;
    let total = n_head * dd;
    if idx >= total {
        terminate!();
    }
    let h = idx / dd;
    let rc = idx % dd;
    let r = rc / d;
    let cc = rc % d;

    let mut acc = state[idx] * total_decay[h];
    let vbase = h * c * d;
    let mut i = 0usize;
    while i < c {
        let dte = decay_to_end[h * c + i];
        let ki = k[vbase + i * d + cc];
        let ui = u_sol[vbase + i * d + r];
        acc = acc + dte * ki * ui;
        i += 1;
    }
    state[idx] = acc;
}

/// Launcher for the state-update kernel.
pub struct DeltanetDrStateUpdateCubeCL;

impl DeltanetDrStateUpdateCubeCL {
    /// # Safety
    /// - `k`/`u_sol`: `n_head * C * d`; `decay_to_end`: `n_head * C`;
    ///   `total_decay`: `n_head`; `state`: `n_head * d * d` (in-place — each
    ///   thread reads then writes only its own element).
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        k_handle: Handle,
        u_sol_handle: Handle,
        decay_to_end_handle: Handle,
        total_decay_handle: Handle,
        state_handle: Handle,
        n_head: usize,
        c: usize,
        d: usize,
    ) {
        let params: [f32; 3] = [n_head as f32, c as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * d * d;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;
        unsafe {
            deltanet_dr_state_update_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(k_handle, n_head * c * d),
                BufferArg::from_raw_parts(u_sol_handle, n_head * c * d),
                BufferArg::from_raw_parts(decay_to_end_handle, n_head * c),
                BufferArg::from_raw_parts(total_decay_handle, n_head),
                BufferArg::from_raw_parts(state_handle, total),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — CPU reference implements the CORRECT Delta rule (the production
// sequential kernel's semantics), state layout [h, d_v, d_k].
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};

    /// Tolerance: the chunked path re-associates the same arithmetic
    /// (log-space decay, forward-substitution order) — expect ~1e-5 relative
    /// differences, not bit-identity.
    const TOL: f32 = 1e-4;

    /// Minimal deterministic PRNG (xorshift) for reproducible fixtures.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// CPU reference: the SEQUENTIAL Delta rule (matches
    /// `deltanet_recurrence_f32`), for C tokens from boundary state S.
    ///
    /// Returns (outputs [C, h*d] token-major, final state [h, d, d]).
    fn cpu_sequential_delta_rule(
        q: &[f32],     // [h, C, d]
        k: &[f32],     // [h, C, d]
        v: &[f32],     // [h, C, d]
        beta: &[f32],  // [h, C]
        alpha: &[f32], // [h, C]
        s0: &[f32],    // [h, d, d]
        h: usize,
        c: usize,
        d: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut state = s0.to_vec();
        let mut out = vec![0.0f32; c * h * d];
        let scale = 1.0f32 / (d as f32).sqrt();
        for t in 0..c {
            for hh in 0..h {
                let vbase = hh * c * d + t * d;
                let sbase = hh * d * d;
                // Step 1: decay.
                for r in 0..d {
                    for cc in 0..d {
                        state[sbase + r * d + cc] *= alpha[hh * c + t];
                    }
                }
                // Steps 2-4: retrieve + delta + update.
                for r in 0..d {
                    let mut mem = 0.0f32;
                    for cc in 0..d {
                        mem += state[sbase + r * d + cc] * k[vbase + cc];
                    }
                    let delta = beta[hh * c + t] * (v[vbase + r] - mem);
                    for cc in 0..d {
                        state[sbase + r * d + cc] += k[vbase + cc] * delta;
                    }
                }
                // Step 5: readout (after update).
                for r in 0..d {
                    let mut acc = 0.0f32;
                    for cc in 0..d {
                        acc += state[sbase + r * d + cc] * q[vbase + cc];
                    }
                    out[t * h * d + hh * d + r] = acc * scale;
                }
            }
        }
        (out, state)
    }

    fn download(client: &ComputeClient<ActiveRuntime>, h: &Handle) -> Vec<f32> {
        let bytes = client.read_one(h.clone()).expect("read back");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Launch the full chunked pipeline for one chunk (the same dispatch
    /// sequence `prefill_deltanet_recurrence_chunked` runs per chunk).
    struct ChunkBufs {
        log_gamma: Handle,
        gamma: Handle,
        dte: Handle,
        td: Handle,
        x: Handle,
        qkr: Handle,
        qs0: Handle,
        rhs: Handle,
        u: Handle,
        o: Handle,
        out: Handle,
    }

    #[allow(clippy::too_many_arguments, reason = "test harness mirrors the kernel buffer set")]
    fn run_pipeline(
        client: &ComputeClient<ActiveRuntime>,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        alpha: &[f32],
        state: &Handle,
        h: usize,
        c: usize,
        d: usize,
    ) -> ChunkBufs {
        let f4 = std::mem::size_of::<f32>();
        let q_h = client.create_from_slice(f32::as_bytes(q));
        let k_h = client.create_from_slice(f32::as_bytes(k));
        let v_h = client.create_from_slice(f32::as_bytes(v));
        let beta_h = client.create_from_slice(f32::as_bytes(beta));
        let alpha_h = client.create_from_slice(f32::as_bytes(alpha));

        let bufs = ChunkBufs {
            log_gamma: client.empty(h * c * f4),
            gamma: client.empty(h * c * f4),
            dte: client.empty(h * c * f4),
            td: client.empty(h * f4),
            x: client.empty(h * c * c * f4),
            qkr: client.empty(h * c * c * f4),
            qs0: client.empty(h * c * d * f4),
            rhs: client.empty(h * c * d * f4),
            u: client.empty(h * c * d * f4),
            o: client.empty(h * c * d * f4),
            out: client.empty(c * h * d * f4),
        };

        unsafe {
            DeltanetDrCumDecayCubeCL::launch::<ActiveRuntime>(
                client,
                alpha_h,
                bufs.log_gamma.clone(),
                bufs.gamma.clone(),
                bufs.dte.clone(),
                bufs.td.clone(),
                h,
                c,
            );
            DeltanetDrGramCubeCL::launch::<ActiveRuntime>(
                client,
                q_h.clone(),
                k_h.clone(),
                beta_h.clone(),
                bufs.log_gamma.clone(),
                bufs.x.clone(),
                bufs.qkr.clone(),
                h,
                c,
                d,
            );
            DeltanetDrRhsCubeCL::launch::<ActiveRuntime>(
                client,
                k_h.clone(),
                v_h,
                beta_h.clone(),
                bufs.gamma.clone(),
                state.clone(),
                bufs.rhs.clone(),
                h,
                c,
                d,
            );
            DeltanetDrQs0CubeCL::launch::<ActiveRuntime>(
                client,
                q_h.clone(),
                bufs.gamma.clone(),
                state.clone(),
                bufs.qs0.clone(),
                h,
                c,
                d,
            );
            DeltanetDrFwdSubCubeCL::launch::<ActiveRuntime>(
                client,
                bufs.x.clone(),
                bufs.rhs.clone(),
                bufs.u.clone(),
                h,
                c,
                d,
            );
            DeltanetDrOutputCubeCL::launch::<ActiveRuntime>(
                client,
                bufs.qs0.clone(),
                bufs.qkr.clone(),
                bufs.u.clone(),
                bufs.o.clone(),
                h,
                c,
                d,
            );
            DeltanetDrScatterCubeCL::launch::<ActiveRuntime>(
                client,
                bufs.o.clone(),
                bufs.out.clone(),
                h,
                c,
                d,
            );
            DeltanetDrStateUpdateCubeCL::launch::<ActiveRuntime>(
                client,
                k_h,
                bufs.u.clone(),
                bufs.dte.clone(),
                bufs.td.clone(),
                state.clone(),
                h,
                c,
                d,
            );
        }
        bufs
    }

    /// End-to-end chunked-pipeline vs sequential CPU reference at a small
    /// shape (h=2, C=8, d=8 — d%4==0 per the fwd_sub contract).
    #[test]
    fn test_delta_rule_chunked_matches_sequential() {
        let h = 2usize;
        let c = 8usize;
        let d = 8usize;
        assert!(DeltanetDeltaRuleChunkedCubeCL::supports(d));

        let mut rng = Rng(0x5eed_1234_5678_9abc);
        let rand_vec =
            |n: usize, rng: &mut Rng| (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect::<Vec<f32>>();

        let q = rand_vec(h * c * d, &mut rng);
        let mut k = rand_vec(h * c * d, &mut rng);
        // Normalize k rows (production L2-normalizes — keeps X well-behaved).
        for off in 0..h * c {
            let norm = (0..d).map(|i| k[off * d + i] * k[off * d + i]).sum::<f32>().sqrt();
            for i in 0..d {
                k[off * d + i] /= norm.max(1e-8);
            }
        }
        let v = rand_vec(h * c * d, &mut rng);
        // beta ∈ (0.2, 1), alpha ∈ (0.8, 1) — realistic GDN gates.
        let beta: Vec<f32> = (0..h * c).map(|_| 0.2 + 0.8 * rng.next_f32()).collect();
        let alpha: Vec<f32> = (0..h * c).map(|_| 0.8 + 0.2 * rng.next_f32()).collect();
        let s0 = rand_vec(h * d * d, &mut rng);

        // CPU reference (sequential Delta rule).
        let (ref_out, ref_state) =
            cpu_sequential_delta_rule(&q, &k, &v, &beta, &alpha, &s0, h, c, d);

        // GPU chunked pipeline.
        let ctx = CubeCLContext::new().expect("GPU init");
        let client = ctx.client();
        let state_h = client.create_from_slice(f32::as_bytes(&s0));

        let bufs = run_pipeline(&client, &q, &k, &v, &beta, &alpha, &state_h, h, c, d);

        let gpu_out = download(&client, &bufs.out);
        let gpu_state = download(&client, &state_h);

        let mut max_out = 0.0f32;
        for (g, r) in gpu_out.iter().zip(ref_out.iter()) {
            max_out = max_out.max((g - r).abs() / r.abs().max(1.0));
        }
        let mut max_state = 0.0f32;
        for (g, r) in gpu_state.iter().zip(ref_state.iter()) {
            max_state = max_state.max((g - r).abs() / r.abs().max(1.0));
        }
        eprintln!("delta-rule chunked vs sequential: out {max_out:.3e}, state {max_state:.3e}");
        assert!(max_out < TOL, "output rel err {max_out} >= {TOL}");
        assert!(max_state < TOL, "state rel err {max_state} >= {TOL}");
    }

    /// C=1 closed-form anchor: with α=1, β=1 the delta rule reduces to
    /// δ = v − (S₀·k) — a cheap independent check of the X/RHS math.
    #[test]
    fn test_delta_rule_chunked_c1_closed_form() {
        let h = 1usize;
        let c = 1usize;
        let d = 4usize;

        let mut rng = Rng(0xfeed_beef);
        let rand_vec =
            |n: usize, rng: &mut Rng| (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect::<Vec<f32>>();

        let q = rand_vec(d, &mut rng);
        let k = rand_vec(d, &mut rng);
        let v = rand_vec(d, &mut rng);
        let beta = vec![1.0f32];
        let alpha = vec![1.0f32];
        let s0 = rand_vec(d * d, &mut rng);

        let (ref_out, ref_state) =
            cpu_sequential_delta_rule(&q, &k, &v, &beta, &alpha, &s0, h, c, d);

        let ctx = CubeCLContext::new().expect("GPU init");
        let client = ctx.client();
        let state_h = client.create_from_slice(f32::as_bytes(&s0));

        let bufs = run_pipeline(&client, &q, &k, &v, &beta, &alpha, &state_h, h, c, d);

        let gpu_out = download(&client, &bufs.out);
        let gpu_state = download(&client, &state_h);
        for (g, r) in gpu_out.iter().zip(ref_out.iter()) {
            assert!((g - r).abs() < TOL, "out {g} vs {r}");
        }
        for (g, r) in gpu_state.iter().zip(ref_state.iter()) {
            assert!((g - r).abs() < TOL, "state {g} vs {r}");
        }
    }

    /// Production-shape stress: h=48, C=64, d=128, TWO chunks with state
    /// carry (the cross-chunk propagation through the in-place state update),
    /// across data regimes — isolates whether the e2e G1 failure reproduces
    /// synthetically (conditioning) or is production-data-specific.
    #[test]
    fn test_delta_rule_chunked_production_shape() {
        let h = 48usize;
        let c = 64usize;
        let d = 128usize;
        let tokens = 2 * c; // two chunks

        let regimes: &[(&str, f32, f32, f32, f32, bool)] = &[
            // (name, alpha_lo, alpha_hi, beta_lo, beta_hi, q_unit_norm)
            ("calm", 0.90, 1.00, 0.2, 1.0, true),
            ("weak-beta", 0.99, 1.00, 0.01, 0.3, true),
            ("mixed-decay", 0.50, 1.00, 0.2, 1.0, true),
            ("raw-q", 0.90, 1.00, 0.2, 1.0, false),
        ];

        let ctx = CubeCLContext::new().expect("GPU init");
        let client = ctx.client();

        for (name, a_lo, a_hi, b_lo, b_hi, q_unit) in regimes {
            let mut rng = Rng(0x1234_0000_beef_cafe);
            let rand_vec = |n: usize, rng: &mut Rng| {
                (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect::<Vec<f32>>()
            };

            let mut q = rand_vec(h * tokens * d, &mut rng);
            if *q_unit {
                for off in 0..h * tokens {
                    let norm = (0..d)
                        .map(|i| q[off * d + i] * q[off * d + i])
                        .sum::<f32>()
                        .sqrt();
                    for i in 0..d {
                        q[off * d + i] /= norm.max(1e-8);
                    }
                }
            }
            let mut k = rand_vec(h * tokens * d, &mut rng);
            for off in 0..h * tokens {
                let norm = (0..d)
                    .map(|i| k[off * d + i] * k[off * d + i])
                    .sum::<f32>()
                    .sqrt();
                for i in 0..d {
                    k[off * d + i] /= norm.max(1e-8);
                }
            }
            let v = rand_vec(h * tokens * d, &mut rng);
            let beta: Vec<f32> = (0..h * tokens)
                .map(|_| b_lo + (b_hi - b_lo) * rng.next_f32())
                .collect();
            let alpha: Vec<f32> = (0..h * tokens)
                .map(|_| a_lo + (a_hi - a_lo) * rng.next_f32())
                .collect();
            let s0 = rand_vec(h * d * d, &mut rng);

            // CPU reference over ALL tokens (sequential Delta rule).
            let (ref_out, ref_state) =
                cpu_sequential_delta_rule(&q, &k, &v, &beta, &alpha, &s0, h, tokens, d);

            // GPU: two chunks, state carried through the in-place update.
            let state_h = client.create_from_slice(f32::as_bytes(&s0));
            let mut gpu_out = vec![0.0f32; tokens * h * d];
            for chunk in 0..2 {
                let base = chunk * c;
                let mut q_c = Vec::with_capacity(h * c * d);
                let mut k_c = Vec::with_capacity(h * c * d);
                let mut v_c = Vec::with_capacity(h * c * d);
                let mut beta_c = Vec::with_capacity(h * c);
                let mut alpha_c = Vec::with_capacity(h * c);
                for hh in 0..h {
                    for t in base..base + c {
                        for i in 0..d {
                            q_c.push(q[hh * tokens * d + t * d + i]);
                            k_c.push(k[hh * tokens * d + t * d + i]);
                            v_c.push(v[hh * tokens * d + t * d + i]);
                        }
                        beta_c.push(beta[hh * tokens + t]);
                        alpha_c.push(alpha[hh * tokens + t]);
                    }
                }

                let bufs =
                    run_pipeline(&client, &q_c, &k_c, &v_c, &beta_c, &alpha_c, &state_h, h, c, d);
                let chunk_out = download(&client, &bufs.out);
                gpu_out[base * h * d..(base + c) * h * d].copy_from_slice(&chunk_out);
            }
            let gpu_state = download(&client, &state_h);

            let mut max_out = 0.0f32;
            for (g, r) in gpu_out.iter().zip(ref_out.iter()) {
                max_out = max_out.max((g - r).abs() / r.abs().max(1.0));
            }
            let mut max_state = 0.0f32;
            for (g, r) in gpu_state.iter().zip(ref_state.iter()) {
                max_state = max_state.max((g - r).abs() / r.abs().max(1.0));
            }
            eprintln!("regime {name}: out {max_out:.3e}, state {max_state:.3e}");
            assert!(
                max_out < TOL,
                "regime {name}: output rel err {max_out} >= {TOL}"
            );
            assert!(
                max_state < TOL,
                "regime {name}: state rel err {max_state} >= {TOL}"
            );
        }
    }
}
