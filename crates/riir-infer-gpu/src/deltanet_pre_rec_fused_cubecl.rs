//! CubeCL fused DeltaNet pre-recurrence chain (Issue 764 T2, M3 lane).
//!
//! Replaces the FOUR dispatches between the input-projection GEMV and the
//! recurrence in the decode path (`forward_deltanet_layer_gpu`):
//!
//! | # | Shipping kernel | Work |
//! |---|---|---|
//! | 2 | `Split4CubeCL` | split concat GEMV output → qkv / z / a / b |
//! | 3 | `DeltanetConv1dCubeCL` | depthwise conv1d + SiLU + state shift |
//! | 4 | `DeltanetBetaDecayCubeCL` | per-head beta/decay scalars |
//! | 5 | `ExpandAndL2NormalizeHeadsCubeCL` | GQA expand + L2-norm + V copy |
//!
//! with ONE dispatch — 4→1 per GDN layer, **−144 launches/token** on
//! Bonsai-27B (48 GDN layers × 3 saved). Bench 763 measured the GDN stage at
//! 20–22% of decode GPU time over 384 launches/token — the dispatch-count
//! class (NOT bandwidth-bound), which is exactly the lever this fusion pulls.
//!
//! # Grid design (race-free by ownership)
//!
//! `CubeCount(n_k + n_v, 1, 1)`, `CubeDim(128)` — one workgroup per source
//! k-head (Q/K sections) or per v-head (V/z/beta-decay):
//!
//! - **`w < n_k` (Q/K owner)** — thread `col` owns channel `w*hd + col` (Q)
//!   and `n_k*hd + w*hd + col` (K): conv-evals from the OLD state + the raw
//!   GEMV value (identical accumulation order to the shipping kernels),
//!   SiLUs, contributes to a shared-memory stage, then shifts+appends its
//!   channel's conv_state. The L2 norm is a thread-0 SERIAL ASCENDING sum —
//!   bit-identical to the shipping per-thread redundant norm (same order,
//!   same values) — broadcast via smem, then every destination head
//!   `h ∈ {w, w+n_k, …}` receives the same normalized values the shipping
//!   kernel recomputed per destination.
//! - **`w >= n_k` (v-head `h = w − n_k`)** — thread `col` owns V channel
//!   `2*n_k*hd + h*hd + col`: conv-eval + SiLU → expanded V (no norm), state
//!   shift+append, z-slice copy; thread 0 computes beta/decay[h] with the
//!   shipping expression sequence verbatim.
//!
//! Every `conv_state` channel is read AND written by exactly ONE workgroup
//! (Q/K channels by their owner `w == kh`; V channels by their `h`), with the
//! reads program-ordered before the writes inside the owning thread — there
//! are NO cross-workgroup races on any buffer. All barriers sit in
//! workgroup-uniform control flow (branch on `CUBE_POS_X`, never on `tid`).
//!
//! # Bit-identity (G1)
//!
//! Outputs are **bit-identical** to the 4-dispatch chain by construction:
//! - conv sum: k-ascending over `[old_state[1..ks), x]` — the exact shifted
//!   window the shipping kernel sums (it shifts first, then sums the new
//!   state; same values, same order);
//! - SiLU / sigmoid / softplus / exp: expression sequences copied verbatim
//!   from `deltanet_conv1d_f32` / `deltanet_beta_decay_f32`;
//! - L2 norm: serial c-ascending accumulation on one lane — the shipping
//!   kernel's own per-thread serial loop (its per-destination redundant
//!   copies compute the identical value from the identical inputs);
//! - the intermediate `qkv`/`a_raw`/`b_raw` materializations are SKIPPED
//!   (no consumer reads them once the expand is fused) — pure dead stores.
//!
//! # Wiring
//!
//! Runtime toggle (default OFF): env `RIIR_GDN_FUSED_PRE_REC` ("1"/"on"/"true")
//! or [`crate::set_deltanet_fused_pre_rec`]. When off — or when
//! [`DeltanetPreRecFusedCubeCL::supports`] rejects the geometry — the
//! shipping 4-dispatch chain runs unchanged. The launch counter
//! ([`crate::deltanet_fused_pre_rec_launch_count`]) is the vacuous guard:
//! outputs are bit-identical, so ONLY the counter proves the toggle reached
//! the kernel (the Bench 768 pattern).

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

// ── Runtime toggle (Bench 768 pattern) ──────────────────────────────────

/// Whether the fused pre-recurrence dispatch is active. Default: env
/// `RIIR_GDN_FUSED_PRE_REC` ("1"/"on"/"true" ⇒ on), else OFF.
static USE_FUSED_PRE_REC: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static FUSED_PRE_REC_INITIALIZED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Launches dispatched through the fused path (the vacuous-guard counter —
/// outputs are bit-identical to the 4-dispatch chain, so ONLY this counter
/// proves the toggle reached the kernel).
static FUSED_PRE_REC_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "cubecl_runtime")]
fn fused_pre_rec_enabled() -> bool {
    if FUSED_PRE_REC_INITIALIZED
        .set(matches!(
            std::env::var("RIIR_GDN_FUSED_PRE_REC")
                .unwrap_or_default()
                .to_lowercase()
                .as_str(),
            "1" | "on" | "true",
        ))
        .is_ok()
    {
        USE_FUSED_PRE_REC.store(
            FUSED_PRE_REC_INITIALIZED.get().copied().unwrap_or(false),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    USE_FUSED_PRE_REC.load(std::sync::atomic::Ordering::Relaxed)
}

/// Force the fused pre-recurrence GDN path on/off (overrides the env var;
/// the bench harness's arm toggle — dispatch-time, no construction coupling).
#[cfg(feature = "cubecl_runtime")]
pub fn set_deltanet_fused_pre_rec(on: bool) {
    let _ = FUSED_PRE_REC_INITIALIZED.set(on);
    USE_FUSED_PRE_REC.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the fused path is currently enabled (crate-internal dispatch check).
#[cfg(feature = "cubecl_runtime")]
pub(crate) fn deltanet_fused_pre_rec_enabled() -> bool {
    fused_pre_rec_enabled()
}

/// Total launches dispatched through the fused path so far.
pub fn deltanet_fused_pre_rec_launch_count() -> usize {
    FUSED_PRE_REC_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

// ── Kernel ──────────────────────────────────────────────────────────────

/// Workgroup size — also the smem stage half-width. `supports` requires
/// `head_dim <= 128`.
const WG: usize = 128;

/// Fused Split4 + Conv1d + BetaDecay + ExpandAndL2 kernel (Issue 764 T2).
///
/// See the module docs for the grid-ownership design + bit-identity argument.
///
/// ## Parameter Layout
///
/// - `input_proj`: concat GEMV output `[conv_dim | z_dim | n_v (a) | n_v (b)]`
/// - `conv_weight`: `[conv_dim, kernel_size]`
/// - `conv_state`: `[conv_dim, kernel_size]` (shifted + appended in place)
/// - `a_log`, `dt_bias`: `[n_v_heads]`
/// - `z_out`: `[z_dim]` (= `n_v_heads * head_dim`)
/// - `beta_out`, `decay_out`: `[n_v_heads]`
/// - `expanded_out`: `[Q(n_v×hd) | K(n_v×hd) | V(n_v×hd)]` — the recurrence input
/// - `params`: `[n_k_heads, n_v_heads, head_dim, kernel_size, conv_dim]` (f32)
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn deltanet_pre_rec_fused_f32(
    input_proj: &[f32],
    conv_weight: &[f32],
    conv_state: &mut [f32],
    a_log: &[f32],
    dt_bias: &[f32],
    z_out: &mut [f32],
    beta_out: &mut [f32],
    decay_out: &mut [f32],
    expanded_out: &mut [f32],
    params: &[f32],
) {
    let n_k = params[0usize] as u32;
    let n_v = params[1usize] as u32;
    let hd = params[2usize] as u32;
    let ks = params[3usize] as u32;
    let conv_dim = params[4usize] as u32;

    let w = CUBE_POS_X; // workgroup id: [0..n_k) Q/K owners, [n_k..n_k+n_v) V owners
    let tid = UNIT_POS;
    let col = tid as usize; // channel-within-head / element-within-head

    // smem stage: [0..hd) = Q post-conv values, [hd..2*hd) = K post-conv
    // values. ONE barrier separates staging from the per-lane redundant
    // norms (V2 — the V1 thread-0 serial reduction put ~128 dependent smem
    // round-trips on the critical path and measured −4.9% e2e; every lane
    // computing the serial-ascending norm itself is the shipping kernel's
    // own redundancy pattern, but from smem instead of global). All
    // barriers sit in workgroup-uniform control flow (never under a `tid`
    // guard) — the Issue 715 divergent-barrier class cannot arise.
    let mut smem = Shared::<[f32]>::new_slice(2usize * WG);

    if w < n_k {
        // ── Q/K owner workgroup for k-head `w` ──
        // Stage both sections' post-conv values in one phase (channels:
        // Q at w*hd, K at n_k*hd + w*hd).
        let q_ch = (w * hd + tid) as usize;
        let k_ch = ((n_k + w) * hd + tid) as usize;
        let ks_us = ks as usize;

        // Conv eval: sum over the NEW state window [old[1..ks), x] —
        // identical order to the shipping kernel. + SiLU (verbatim).
        let mut y_q = f32::new(0.0f32);
        let mut y_k = f32::new(0.0f32);
        if tid < hd {
            let q_base = q_ch * ks_us;
            let mut sum_q = f32::new(0.0f32);
            for k in 0..ks_us {
                let v = if k < ks_us - 1 {
                    conv_state[q_base + k + 1]
                } else {
                    input_proj[q_ch]
                };
                sum_q += v * conv_weight[q_base + k];
            }
            let neg_q = f32::new(0.0f32) - sum_q;
            let sig_q = f32::new(1.0f32) / (f32::new(1.0f32) + neg_q.exp());
            y_q = sum_q * sig_q;

            let k_base = k_ch * ks_us;
            let mut sum_k = f32::new(0.0f32);
            for k in 0..ks_us {
                let v = if k < ks_us - 1 {
                    conv_state[k_base + k + 1]
                } else {
                    input_proj[k_ch]
                };
                sum_k += v * conv_weight[k_base + k];
            }
            let neg_k = f32::new(0.0f32) - sum_k;
            let sig_k = f32::new(1.0f32) / (f32::new(1.0f32) + neg_k.exp());
            y_k = sum_k * sig_k;

            smem[col] = y_q;
            smem[WG + col] = y_k;
        }
        sync_cube();

        // Per-lane redundant serial-ascending L2 norms — bit-identical to
        // the shipping per-thread norms (same order, same values; the
        // shipping kernel pays the same redundancy reading global). Zero-
        // norm guard included (Issue 673 Bug D).
        if tid < hd {
            let hd_us = hd as usize;
            let mut sq_q = f32::new(0.0f32);
            let mut sq_k = f32::new(0.0f32);
            for c in 0..hd_us {
                let vq = smem[c];
                sq_q += vq * vq;
                let vk = smem[WG + c];
                sq_k += vk * vk;
            }
            let inv_q = if sq_q > f32::new(0.0f32) {
                f32::new(1.0f32) / sq_q.sqrt()
            } else {
                f32::new(0.0f32)
            };
            let inv_k = if sq_k > f32::new(0.0f32) {
                f32::new(1.0f32) / sq_k.sqrt()
            } else {
                f32::new(0.0f32)
            };

            // Write every destination head h ∈ {w, w+n_k, …} — the exact
            // values the shipping expand kernel recomputed per destination.
            let mut h = w;
            while h < n_v {
                let out = (h * hd + tid) as usize;
                expanded_out[out] = y_q * inv_q;
                expanded_out[(n_v * hd) as usize + out] = y_k * inv_k;
                h += n_k;
            }

            // Owner-only conv_state shift+append for the Q and K channels
            // (all reads complete before any write — program order).
            let q_base = q_ch * ks_us;
            for k in 0..ks_us - 1 {
                conv_state[q_base + k] = conv_state[q_base + k + 1];
            }
            conv_state[q_base + ks_us - 1] = input_proj[q_ch];
            let k_base = k_ch * ks_us;
            for k in 0..ks_us - 1 {
                conv_state[k_base + k] = conv_state[k_base + k + 1];
            }
            conv_state[k_base + ks_us - 1] = input_proj[k_ch];
        }
    } else {
        // ── V/z/beta-decay workgroup for v-head h = w − n_k ──
        if tid < hd {
            let h = w - n_k;
            let ch = (2u32 * n_k * hd + h * hd + tid) as usize;
            let ks_us = ks as usize;
            let state_base = ch * ks_us;

            // Conv eval (same order as shipping) + SiLU.
            let mut y = f32::new(0.0f32);
            for k in 0..ks_us {
                let v = if k < ks_us - 1 {
                    conv_state[state_base + k + 1]
                } else {
                    input_proj[ch]
                };
                y += v * conv_weight[state_base + k];
            }
            let neg_sum = f32::new(0.0f32) - y;
            let exp_neg = neg_sum.exp();
            let sig = f32::new(1.0f32) / (f32::new(1.0f32) + exp_neg);
            y = y * sig;

            // V section of expanded (no norm — verbatim copy semantics).
            expanded_out[(2u32 * n_v * hd + h * hd + tid) as usize] = y;

            // Owner conv_state shift+append for this V channel.
            for k in 0..ks_us - 1 {
                conv_state[state_base + k] = conv_state[state_base + k + 1];
            }
            conv_state[state_base + ks_us - 1] = input_proj[ch];

            // z slice copy for this head.
            z_out[(h * hd + tid) as usize] =
                input_proj[(conv_dim + h * hd + tid) as usize];
        }

        // beta/decay for head h (verbatim deltanet_beta_decay_f32).
        if tid == 0u32 {
            let h = w - n_k;
            let a_off = conv_dim + n_v * hd + h; // a_raw[h] in the concat
            let b_off = conv_dim + n_v * hd + n_v + h; // b_raw[h]

            let b_val = input_proj[b_off as usize];
            let neg_b = f32::new(0.0f32) - b_val;
            let beta = f32::new(1.0f32) / (f32::new(1.0f32) + neg_b.exp());
            beta_out[h as usize] = beta;

            let a_val = input_proj[a_off as usize] + dt_bias[h as usize];
            let sp = if a_val > f32::new(20.0f32) {
                a_val
            } else if a_val < f32::new(-20.0f32) {
                f32::new(0.0f32)
            } else {
                (f32::new(1.0f32) + a_val.exp()).ln()
            };
            let g = a_log[h as usize] * sp;
            decay_out[h as usize] = g.exp();
        }
    }
}

/// Launcher for the fused pre-recurrence chain (Issue 764 T2).
#[cfg(all(feature = "cubecl_runtime", any(test, feature = "ternary_gemv")))]
pub struct DeltanetPreRecFusedCubeCL;

#[cfg(all(feature = "cubecl_runtime", any(test, feature = "ternary_gemv")))]
impl DeltanetPreRecFusedCubeCL {
    /// Whether the fused kernel serves this geometry.
    ///
    /// - `head_dim <= 128` (the fixed smem stage / workgroup width);
    /// - `n_v_heads % n_k_heads == 0` (the tiled GQA broadcast);
    /// - `kernel_size >= 2` (the shipping models are ks=4; ks=1 would also
    ///   be correct but is unexercised — keep the conservative gate).
    #[inline]
    #[must_use]
    pub fn supports(n_k_heads: usize, n_v_heads: usize, head_dim: usize, kernel_size: usize) -> bool {
        (1..=WG).contains(&head_dim)
            && n_k_heads >= 1
            && n_v_heads.is_multiple_of(n_k_heads)
            && kernel_size >= 2
    }

    /// Launch the fused Split4+Conv1d+BetaDecay+ExpandAndL2 chain.
    ///
    /// Replaces the four shipping dispatches following the in_proj GEMV.
    ///
    /// # Safety
    /// - `input_proj_handle`: `conv_dim + z_dim + 2*n_v` f32 elements
    ///   (the `in_proj_concat` GEMV output layout).
    /// - `conv_weight_handle` / `conv_state_handle`: `conv_dim * kernel_size`
    ///   f32 elements each.
    /// - `a_log_handle` / `dt_bias_handle` / `beta_out` / `decay_out`:
    ///   `n_v_heads` f32 elements each.
    /// - `z_out_handle`: `z_dim` (`n_v_heads * head_dim`) f32 elements.
    /// - `expanded_handle`: `3 * n_v_heads * head_dim` f32 elements.
    /// - Geometry must pass [`Self::supports`].
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent to the fused-kernel interface")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input_proj_handle: Handle,
        conv_weight_handle: Handle,
        conv_state_handle: Handle,
        a_log_handle: Handle,
        dt_bias_handle: Handle,
        z_out_handle: Handle,
        beta_out_handle: Handle,
        decay_out_handle: Handle,
        expanded_handle: Handle,
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
        kernel_size: usize,
    ) {
        debug_assert!(
            Self::supports(n_k_heads, n_v_heads, head_dim, kernel_size),
            "geometry must pass supports()"
        );
        // Vacuous-guard counter (outputs are bit-identical — only this
        // proves the toggle reached the kernel).
        FUSED_PRE_REC_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let conv_dim = 2 * n_k_heads * head_dim + n_v_heads * head_dim;
        let z_dim = n_v_heads * head_dim;
        let params: [f32; 5] = [
            n_k_heads as f32,
            n_v_heads as f32,
            head_dim as f32,
            kernel_size as f32,
            conv_dim as f32,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));

        let num_wg = (n_k_heads + n_v_heads) as u32;

        unsafe {
            deltanet_pre_rec_fused_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(WG as u32),
                BufferArg::from_raw_parts(input_proj_handle, conv_dim + z_dim + 2 * n_v_heads),
                BufferArg::from_raw_parts(conv_weight_handle, conv_dim * kernel_size),
                BufferArg::from_raw_parts(conv_state_handle, conv_dim * kernel_size),
                BufferArg::from_raw_parts(a_log_handle, n_v_heads),
                BufferArg::from_raw_parts(dt_bias_handle, n_v_heads),
                BufferArg::from_raw_parts(z_out_handle, z_dim),
                BufferArg::from_raw_parts(beta_out_handle, n_v_heads),
                BufferArg::from_raw_parts(decay_out_handle, n_v_heads),
                BufferArg::from_raw_parts(expanded_handle, 3 * n_v_heads * head_dim),
                BufferArg::from_raw_parts(params_handle, 5),
            );
        }
    }
}

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};
    use crate::deltanet_cubecl::{
        DeltanetBetaDecayCubeCL, DeltanetConv1dCubeCL, ExpandAndL2NormalizeHeadsCubeCL,
    };
    use crate::elementwise_cubecl::Split4CubeCL;

    /// Deterministic LCG (no rand dep; reproducible fixtures).
    fn lcg(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (*state >> 33) as u32;
        (bits % 2001) as f32 / 1000.0 - 1.0 // [-1, 1)
    }

    struct Shape {
        n_k: usize,
        n_v: usize,
        hd: usize,
        ks: usize,
    }

    /// Run BOTH paths on identical inputs; return the fused path's outputs
    /// plus the shipping chain's outputs for bit-exact comparison
    /// (expanded, z, beta, decay, conv_state per arm).
    type PathOutputs = (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    );

    fn run_both(
        client: &ComputeClient<ActiveRuntime>,
        s: &Shape,
        seed: u64,
        tokens: usize,
    ) -> (PathOutputs, PathOutputs) {
        let conv_dim = 2 * s.n_k * s.hd + s.n_v * s.hd;
        let z_dim = s.n_v * s.hd;
        let total_in = conv_dim + z_dim + 2 * s.n_v;

        let mut rng = seed;
        let mut input_proj = vec![0.0f32; total_in];
        let mut conv_weight = vec![0.0f32; conv_dim * s.ks];
        let mut a_log = vec![0.0f32; s.n_v];
        let mut dt_bias = vec![0.0f32; s.n_v];
        for v in input_proj.iter_mut() {
            *v = lcg(&mut rng);
        }
        for v in conv_weight.iter_mut() {
            *v = lcg(&mut rng);
        }
        for v in a_log.iter_mut() {
            *v = lcg(&mut rng) * 0.5 + 0.5; // positive-ish
        }
        for v in dt_bias.iter_mut() {
            *v = lcg(&mut rng);
        }
        // Conv state starts zeroed (the forward's reset_state contract).
        let conv_state0 = vec![0.0f32; conv_dim * s.ks];

        // ── Shipping chain (Split4 → Conv1d → BetaDecay → ExpandAndL2), with
        // fresh copies of the mutable buffers per path. Per-token inputs are
        // re-uploaded (state carries across tokens).
        let cw_h = client.create_from_slice(f32::as_bytes(&conv_weight));
        let al_h = client.create_from_slice(f32::as_bytes(&a_log));
        let db_h = client.create_from_slice(f32::as_bytes(&dt_bias));

        let ship_state = client.create_from_slice(f32::as_bytes(&conv_state0));
        let fused_state = client.create_from_slice(f32::as_bytes(&conv_state0));

        let qkv_len = conv_dim;
        let qkv_ship = client.empty(qkv_len * 4);
        let z_ship = client.empty(z_dim * 4);
        let z_fused = client.empty(z_dim * 4);
        let a_ship = client.empty(s.n_v * 4);
        let b_ship = client.empty(s.n_v * 4);
        let beta_ship = client.empty(s.n_v * 4);
        let decay_ship = client.empty(s.n_v * 4);
        let beta_fused = client.empty(s.n_v * 4);
        let decay_fused = client.empty(s.n_v * 4);
        let exp_ship = client.empty(3 * s.n_v * s.hd * 4);
        let exp_fused = client.empty(3 * s.n_v * s.hd * 4);

        for t in 0..tokens {
            // Shipping: write the (deterministic) input into the buffer each
            // token — values change per token to exercise state evolution.
            let mut tok_input = input_proj.clone();
            for (i, v) in tok_input.iter_mut().enumerate() {
                *v += (t as f32) * 0.001 * ((i % 7) as f32 - 3.0);
            }
            let tok_ip_h = client.create_from_slice(f32::as_bytes(&tok_input));

            unsafe {
                Split4CubeCL::launch::<ActiveRuntime>(
                    client,
                    tok_ip_h.clone(),
                    qkv_ship.clone(),
                    z_ship.clone(),
                    a_ship.clone(),
                    b_ship.clone(),
                    conv_dim,
                    z_dim,
                    s.n_v,
                    s.n_v,
                );
                DeltanetConv1dCubeCL::launch::<ActiveRuntime>(
                    client,
                    qkv_ship.clone(),
                    cw_h.clone(),
                    ship_state.clone(),
                    conv_dim,
                    s.ks,
                );
                DeltanetBetaDecayCubeCL::launch::<ActiveRuntime>(
                    client,
                    a_ship.clone(),
                    b_ship.clone(),
                    al_h.clone(),
                    db_h.clone(),
                    beta_ship.clone(),
                    decay_ship.clone(),
                    s.n_v,
                );
                ExpandAndL2NormalizeHeadsCubeCL::launch::<ActiveRuntime>(
                    client,
                    qkv_ship.clone(),
                    exp_ship.clone(),
                    s.n_k,
                    s.n_v,
                    s.hd,
                );

                // Fused (single dispatch).
                DeltanetPreRecFusedCubeCL::launch::<ActiveRuntime>(
                    client,
                    tok_ip_h.clone(),
                    cw_h.clone(),
                    fused_state.clone(),
                    al_h.clone(),
                    db_h.clone(),
                    z_fused.clone(),
                    beta_fused.clone(),
                    decay_fused.clone(),
                    exp_fused.clone(),
                    s.n_k,
                    s.n_v,
                    s.hd,
                    s.ks,
                );
            }
        }

        let read = |h: Handle| -> Vec<f32> {
            f32::from_bytes(&client.read_one(h).expect("readback")).to_vec()
        };
        (
            (
                read(exp_ship),
                read(z_ship),
                read(beta_ship),
                read(decay_ship),
                read(ship_state),
            ),
            (
                read(exp_fused),
                read(z_fused),
                read(beta_fused),
                read(decay_fused),
                read(fused_state),
            ),
        )
    }

    fn assert_bit_exact(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                x.to_bits() == y.to_bits(),
                "{what}[{i}]: {x} (shipping) != {y} (fused) — not bit-identical"
            );
        }
    }

    #[test]
    fn fused_matches_shipping_gqa_small() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        let s = Shape { n_k: 2, n_v: 6, hd: 16, ks: 4 };
        assert!(DeltanetPreRecFusedCubeCL::supports(s.n_k, s.n_v, s.hd, s.ks));
        for seed in [1u64, 42, 0xDEAD] {
            let (ship, fused) = run_both(&client, &s, seed, 3);
            assert_bit_exact(&ship.0, &fused.0, "expanded");
            assert_bit_exact(&ship.1, &fused.1, "z");
            assert_bit_exact(&ship.2, &fused.2, "beta");
            assert_bit_exact(&ship.3, &fused.3, "decay");
            assert_bit_exact(&ship.4, &fused.4, "conv_state");
        }
    }

    #[test]
    fn fused_matches_shipping_non_gqa_hd128() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        // Production-like single token: n_k == n_v (no GQA), full head_dim.
        let s = Shape { n_k: 4, n_v: 4, hd: 128, ks: 4 };
        assert!(DeltanetPreRecFusedCubeCL::supports(s.n_k, s.n_v, s.hd, s.ks));
        let (ship, fused) = run_both(&client, &s, 7, 2);
        assert_bit_exact(&ship.0, &fused.0, "expanded");
        assert_bit_exact(&ship.1, &fused.1, "z");
        assert_bit_exact(&ship.2, &fused.2, "beta");
        assert_bit_exact(&ship.3, &fused.3, "decay");
        assert_bit_exact(&ship.4, &fused.4, "conv_state");
    }

    #[test]
    fn fused_matches_shipping_production_geometry() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();
        // Bonsai-class GQA ratio (1:4) at full head_dim.
        let s = Shape { n_k: 8, n_v: 32, hd: 128, ks: 4 };
        assert!(DeltanetPreRecFusedCubeCL::supports(s.n_k, s.n_v, s.hd, s.ks));
        let (ship, fused) = run_both(&client, &s, 99, 2);
        assert_bit_exact(&ship.0, &fused.0, "expanded");
        assert_bit_exact(&ship.1, &fused.1, "z");
        assert_bit_exact(&ship.2, &fused.2, "beta");
        assert_bit_exact(&ship.3, &fused.3, "decay");
        assert_bit_exact(&ship.4, &fused.4, "conv_state");
    }

    #[test]
    fn supports_gate() {
        assert!(DeltanetPreRecFusedCubeCL::supports(8, 32, 128, 4));
        assert!(!DeltanetPreRecFusedCubeCL::supports(8, 30, 128, 4)); // n_v % n_k != 0
        assert!(!DeltanetPreRecFusedCubeCL::supports(8, 32, 256, 4)); // hd > WG
        assert!(!DeltanetPreRecFusedCubeCL::supports(8, 32, 128, 1)); // ks < 2
        assert!(!DeltanetPreRecFusedCubeCL::supports(0, 32, 128, 4));
    }
}
