//! Tree-masked batched verify kernels for DeltaNet layers (Issue 721 T2+T3).
//!
//! Ports the CPU oracle (`riir_infer_core::deltanet::tree_forward`, backed by
//! katgpt-core's `verify_gdn_tree` — Plan 424, arXiv:2607.06763 §3.4) onto
//! CubeCL so a whole speculative draft **tree** is verified against a DeltaNet
//! layer in a constant number of dispatches, reading the layer's ternary
//! weights once — the only wall-clock-viable verify shape at the measured
//! tree acceptance (Issue 717 G3a FAIL 0.326× chain / Bench 694).
//!
//! # The masked-solve factorization (why these kernels are tiny)
//!
//! The sequential recurrence `Sₜ = αₜ(I − βₜkₜkₜᵀ)Sₜ₋₁ + βₜkₜvₜᵀ` applied
//! along a TREE collapses to a masked triangular solve over the ancestor
//! partial order — the state never has to be rolled back or forked:
//!
//! ```text
//! X[i][j] = 𝟙[j ≺ i] · (aᵢ/aⱼ) · βᵢ · (kᵢᵀkⱼ)     (T×T, ancestor-masked)
//! RHS[i]  = βᵢvᵢ − βᵢaᵢ(kᵢᵀS₀)                    (WS₀-folding)
//! Solve   (I + X)U = RHS    (unit-lower-triangular, forward substitution)
//! O[i]    = (1/√dₖ)(aᵢqᵢᵀS₀ + Σ_{j⪯i} Y[i][j]·U[j]),  Y = scale·ratio·(qᵢᵀkⱼ)
//! ```
//!
//! At T=64, d=128 the solve is ~0.5 MFLOP/head — noise next to the per-node
//! projections, which ride the existing batched ternary GEMM
//! ([`crate::GemmTernaryBatchedCubeCL`], Issue 637).
//!
//! # Layouts (all node-axis buffers token-major)
//!
//! | Buffer | Layout |
//! |---|---|
//! | `qkv_expanded` | `[T × 3·H·d]` — Q at 0, K at H·d, V at 2·H·d (the existing `ExpandAndL2NormalizeHeadsBatchedCubeCL` output) |
//! | `beta` / `decay` | `[T × H]` (the existing `DeltanetBetaDecayBatchedCubeCL` output) |
//! | `cld` | `[T × H]` per-head cumulative log-decay (kernel 2's output) |
//! | `state` | `[H × d_v × d_k]` per-head GPU layout — row = value dim `d`, col = key dim `m` (`d·d_k + m`) — read-only, never written by verify |
//! | `x`, `y` | `[H × T × T]` |
//! | `rhs`, `u` | `[H × T × d_v]` (head-major) |
//! | `out` | `[T × H · d_v]` token-major (feeds the batched RMSNorm+z-gate) |
//!
//! The GPU state layout is the transpose of the CPU oracle's `S₀[m·d_v + d]`
//! (`riir_infer_core::deltanet::tree_forward::transpose_state` exists for exactly
//! this reason) — these kernels index the GPU layout directly, no transpose
//! pass.
//!
//! # Constraint: T ≤ 64
//!
//! Ancestor masks upload as two u32 words per node. The Issue 717/721 harness
//! runs budget 16–64; raising T needs wider words (katgpt-core's
//! `TreeTopology` already supports arbitrary T — extend the upload then).

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 — tree conv1d gather (T2)
// ─────────────────────────────────────────────────────────────────────────────

/// Depthwise conv1d over tree-structured windows, with SiLU.
///
/// Per node `k` (topo index), channel `ch`:
///
/// ```text
/// out[k, ch] = silu( Σ_ki window(k, ch, ki) · conv_weight[ch·K + ki] )
/// ```
///
/// where `window(k, ch, ki)` is either a raw ancestor projection row
/// (`raw_qkv[anc·conv_dim + ch]`, LUT code `< T`) or a committed conv-state
/// slot (`conv_state[ch·K + state_col]`, LUT code `≥ T`, `state_col = code − T`).
/// This mirrors `forward_tree_deltanet_layer` step 2 exactly — the LUT is
/// precomputed on the CPU (channel-independent per `(k, ki)`).
///
/// Verify never writes the conv state (read-only, no shift) — the accepted
/// path is committed later by sequential replay.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn tree_conv1d_gather_f32(
    raw_qkv: &[f32],
    conv_weight: &[f32],
    conv_state: &[f32],
    lut: &[u32],
    output: &mut [f32],
    params: &[f32],
) {
    let t = params[0usize] as usize;
    let conv_dim = params[1usize] as usize;
    let kernel_size = params[2usize] as usize;

    let idx = ABSOLUTE_POS;
    if idx >= t * conv_dim {
        terminate!();
    }

    let k = idx / conv_dim;
    let ch = idx % conv_dim;
    let cw = ch * kernel_size;

    let mut sum = f32::new(0.0f32);
    for ki in 0..kernel_size {
        let code = lut[k * kernel_size + ki];
        if code < (t as u32) {
            let anc = code as usize;
            sum += raw_qkv[anc * conv_dim + ch] * conv_weight[cw + ki];
        } else {
            let state_col = (code as usize) - t;
            sum += conv_state[cw + state_col] * conv_weight[cw + ki];
        }
    }

    // SiLU (matches `deltanet_conv1d_f32`)
    let neg = f32::new(0.0f32) - sum;
    let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg.exp());
    output[idx] = sum * sig;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 — per-head cumulative log-decay
// ─────────────────────────────────────────────────────────────────────────────

/// Per-head cumulative log-decay: `cld[k, h] = Σ_{j ⪯ k} ln(decay[k, h])`
/// walking topo order (parents precede children).
///
/// One thread per head; the thread walks all T nodes sequentially. The CPU
/// oracle recomputes this per head (`recompute_cumulative_log_decay`) from
/// GPU-computed decays — doing it on-device avoids a 48-layer readback.
///
/// The pair kernels (`build_xy`, `build_rhs`, `compute_out`) exp the
/// DIFFERENCE of cld entries — never an absolute cld — so f32 log space is
/// well-conditioned for path products (T ≤ 64, decays in (0, 1]).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn tree_cumulative_log_decay_f32(
    decay: &[f32],
    parent: &[u32],
    cld_out: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let t = params[1usize] as usize;

    let h = ABSOLUTE_POS;
    if h >= n_head {
        terminate!();
    }

    for k in 0..t {
        let la = decay[k * n_head + h].ln();
        let p = parent[k];
        if p == u32::MAX {
            cld_out[k * n_head + h] = la;
        } else {
            cld_out[k * n_head + h] = cld_out[p as usize * n_head + h] + la;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3 — build X and Y (fused; one ancestor-pair pass)
// ─────────────────────────────────────────────────────────────────────────────

/// Interaction matrices over ancestor pairs, per head:
///
/// ```text
/// X[h,i,j] = 𝟙[j ≺ i] · ratio(i,j) · βᵢ · (kᵢᵀkⱼ)      — solve operand
/// Y[h,i,j] = 𝟙[j ⪯ i] · scale · ratio(i,j) · (qᵢᵀkⱼ)   — output operand
/// ratio(i,j) = exp(cld[i] − cld[j])
/// ```
///
/// Non-pair entries are written to ZERO (not skipped) — `forward_sub` and
/// `compute_out` sum over full rows relying on the zeros.
///
/// `anc_lo[i]`/`anc_hi[i]` are the PROPER-ancestor bitmasks of topo node `i`
/// (bits 0..31 / 32..63). Ancestor-or-self adds the self bit in-kernel.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn tree_build_xy_f32(
    qkv_expanded: &[f32],
    beta: &[f32],
    cld: &[f32],
    anc_lo: &[u32],
    anc_hi: &[u32],
    x_out: &mut [f32],
    y_out: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let t = params[1usize] as usize;
    let d = params[2usize] as usize; // head_dim (d_k == d_v == d)

    let idx = ABSOLUTE_POS;
    if idx >= n_head * t * t {
        terminate!();
    }

    let h = idx / (t * t);
    let rem = idx % (t * t);
    let i = rem / t;
    let j = rem % t;

    let hd = h * d; // head offset within one node's [3·H·d] row
    let row_i = i * (3 * n_head * d);
    let row_j = j * (3 * n_head * d);
    let k_sec = n_head * d; // K section offset
    let q_sec = 0usize;

    let mut x_val = f32::new(0.0f32);
    let mut y_val = f32::new(0.0f32);

    if j <= i {
        // proper-ancestor bit of j in i's mask
        let anc_bit = if j < 32 {
            (anc_lo[i] >> ((j % 32) as u32)) & 1u32
        } else {
            (anc_hi[i] >> (((j - 32) % 32) as u32)) & 1u32
        };
        let is_proper = (j < i) && (anc_bit == 1);
        let is_anc_or_self = (anc_bit == 1) || (j == i);

        if is_proper || is_anc_or_self {
            let ratio = (cld[i * n_head + h] - cld[j * n_head + h]).exp();

            if is_proper {
                // kᵢᵀkⱼ
                let mut kk = f32::new(0.0f32);
                for m in 0..d {
                    kk += qkv_expanded[row_i + k_sec + hd + m]
                        * qkv_expanded[row_j + k_sec + hd + m];
                }
                let beta_i = beta[i * n_head + h];
                x_val = ratio * beta_i * kk;
            }

            if is_anc_or_self {
                // qᵢᵀkⱼ
                let mut qk = f32::new(0.0f32);
                for m in 0..d {
                    qk += qkv_expanded[row_i + q_sec + hd + m]
                        * qkv_expanded[row_j + k_sec + hd + m];
                }
                let scale = f32::new(1.0f32) / (params[2usize]).sqrt();
                y_val = scale * ratio * qk;
            }
        }
    }

    x_out[idx] = x_val;
    y_out[idx] = y_val;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4 — build folded RHS
// ─────────────────────────────────────────────────────────────────────────────

/// `RHS[h,i,d] = βᵢ·vᵢ[d] − βᵢ·aᵢ·(kᵢᵀS₀)[d]` with `aᵢ = exp(cld[i])` (the
/// total root→i path decay) and `(kᵢᵀS₀)[d] = Σ_m k[m]·S₀[d·d_k + m]`
/// (GPU state layout — value-major rows).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn tree_build_rhs_f32(
    qkv_expanded: &[f32],
    beta: &[f32],
    cld: &[f32],
    state: &[f32],
    rhs_out: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let t = params[1usize] as usize;
    let d = params[2usize] as usize; // head_dim

    let idx = ABSOLUTE_POS;
    if idx >= n_head * t * d {
        terminate!();
    }

    let h = idx / (t * d);
    let rem = idx % (t * d);
    let i = rem / d;
    let dv = rem % d;

    let hd = h * d;
    let row_i = i * (3 * n_head * d);
    let k_sec = n_head * d;
    let v_sec = 2 * n_head * d;
    let state_head = h * d * d;

    // (kᵢᵀS₀)[dv] = Σ_m k[m] · S₀[dv·d + m]
    let mut ks0 = f32::new(0.0f32);
    for m in 0..d {
        ks0 += qkv_expanded[row_i + k_sec + hd + m] * state[state_head + dv * d + m];
    }

    let beta_i = beta[i * n_head + h];
    let a_i = cld[i * n_head + h].exp();
    let v_i = qkv_expanded[row_i + v_sec + hd + dv];

    rhs_out[h * t * d + i * d + dv] = beta_i * v_i - beta_i * a_i * ks0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5 — forward substitution
// ─────────────────────────────────────────────────────────────────────────────

/// Solve `(I + X)U = RHS` per head by forward substitution over topo order.
///
/// One thread owns one (head, value-column) pair and walks `i = 0..T`
/// sequentially; `j < i` entries of `U` are already final in that thread's
/// own column (X[i][j] is zero for non-ancestors, so the full `j < i` sum is
/// the masked sum). X rows are shared across a head's threads (broadcast
/// reads).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn tree_forward_sub_f32(
    x: &[f32],
    rhs: &[f32],
    u_out: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let t = params[1usize] as usize;
    let d = params[2usize] as usize; // head_dim

    let idx = ABSOLUTE_POS;
    if idx >= n_head * d {
        terminate!();
    }

    let h = idx / d;
    let dv = idx % d;

    for i in 0..t {
        let mut acc = rhs[h * t * d + i * d + dv];
        for j in 0..i {
            acc -= x[h * t * t + i * t + j] * u_out[h * t * d + j * d + dv];
        }
        u_out[h * t * d + i * d + dv] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 6 — compute outputs
// ─────────────────────────────────────────────────────────────────────────────

/// `O[i, h·d + dv] = scale·aᵢ·(qᵢᵀS₀)[dv] + Σ_{j ≤ i} Y[h,i,j]·U[h,j,dv]`
/// written token-major (feeds the batched RMSNorm + z-gate).
///
/// Y is zero outside ancestor-or-self pairs, so the full `j ≤ i` sum is the
/// masked sum.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn tree_compute_out_f32(
    qkv_expanded: &[f32],
    y: &[f32],
    u: &[f32],
    cld: &[f32],
    state: &[f32],
    out: &mut [f32],
    params: &[f32],
) {
    let n_head = params[0usize] as usize;
    let t = params[1usize] as usize;
    let d = params[2usize] as usize; // head_dim

    let idx = ABSOLUTE_POS;
    if idx >= n_head * t * d {
        terminate!();
    }

    let h = idx / (t * d);
    let rem = idx % (t * d);
    let i = rem / d;
    let dv = rem % d;

    let hd = h * d;
    let row_i = i * (3 * n_head * d);
    let q_sec = 0usize;
    let state_head = h * d * d;

    // (qᵢᵀS₀)[dv] = Σ_m q[m] · S₀[dv·d + m]
    let mut qs0 = f32::new(0.0f32);
    for m in 0..d {
        qs0 += qkv_expanded[row_i + q_sec + hd + m] * state[state_head + dv * d + m];
    }

    let scale = f32::new(1.0f32) / (params[2usize]).sqrt();
    let a_i = cld[i * n_head + h].exp();
    let mut acc = scale * a_i * qs0;

    for j in 0..=i {
        acc += y[h * t * t + i * t + j] * u[h * t * d + j * d + dv];
    }

    out[i * n_head * d + hd + dv] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU-side plan (LUT + ancestor masks, topo-ordered)
// ─────────────────────────────────────────────────────────────────────────────

/// CPU-precomputed topology artifacts uploaded once per verify cycle
/// (shared across all layers — the topology is layer-independent).
///
/// Input contract: `parent[k]` is the TOPO index of node k's parent
/// (parents precede children), `u32::MAX` for the root. Callers holding an
/// original-index tree should emit nodes in topo order and remap parents
/// first (BFS order is a valid topo order).
///
/// G4: buffers are built once per cycle and uploaded per launch; steady-state
/// verify itself performs no allocation (all GPU buffers preallocated at
/// `T_max` by the driver).
#[derive(Clone, Debug)]
pub struct TreeVerifyPlan {
    /// Number of nodes.
    pub t: usize,
    /// Topo-indexed parent (`u32::MAX` = root).
    pub parent: Vec<u32>,
    /// Topo depth per node.
    pub depth: Vec<usize>,
    /// Conv-window codes `[T × K]`: `< t` = ancestor topo index,
    /// `≥ t` = committed conv-state column (`code − t`).
    pub conv_lut: Vec<u32>,
    /// Proper-ancestor bitmask of node i, bits 0..31.
    pub anc_lo: Vec<u32>,
    /// Proper-ancestor bitmask of node i, bits 32..63.
    pub anc_hi: Vec<u32>,
}

/// Max tree size supported by the two-word ancestor mask upload.
pub const TREE_VERIFY_MAX_T: usize = 64;

impl TreeVerifyPlan {
    /// Build the plan from topo-ordered parent pointers.
    ///
    /// # Panics
    /// Panics if `t == 0`, `t > TREE_VERIFY_MAX_T`, or a parent index is not
    /// strictly less than its child (not a topo-ordered single-root tree).
    pub fn from_parents_topo(parent: &[u32], kernel_size: usize) -> Self {
        let t = parent.len();
        assert!(t > 0, "tree must have at least one node");
        assert!(
            t <= TREE_VERIFY_MAX_T,
            "T={t} exceeds the two-word ancestor mask ({TREE_VERIFY_MAX_T}); \
             widen anc_lo/anc_hi first"
        );

        // Depth via parent walk (parents precede children).
        let mut depth = vec![0usize; t];
        for k in 0..t {
            let p = parent[k];
            if p != u32::MAX {
                assert!(
                    (p as usize) < k,
                    "parent[{k}]={p} does not precede its child — not topo-ordered"
                );
                depth[k] = depth[p as usize] + 1;
            }
        }

        // Proper-ancestor bitmasks: ancestors(k) = ancestors(parent(k)) | bit(parent(k)).
        let mut anc_lo = vec![0u32; t];
        let mut anc_hi = vec![0u32; t];
        for k in 0..t {
            let p = parent[k];
            if p != u32::MAX {
                let p = p as usize;
                anc_lo[k] = anc_lo[p];
                anc_hi[k] = anc_hi[p];
                if p < 32 {
                    anc_lo[k] |= 1u32 << p;
                } else {
                    anc_hi[k] |= 1u32 << (p - 32);
                }
            }
        }

        // Conv LUT (mirrors `forward_tree_deltanet_layer` step 2 exactly).
        let mut conv_lut = vec![0u32; t * kernel_size];
        for k in 0..t {
            let depth_k = depth[k];
            let committed_count = kernel_size.saturating_sub(1 + depth_k);
            for ki in 0..kernel_size {
                if ki < committed_count {
                    // committed tail: conv_state[ch·K + depth_k + ki]
                    conv_lut[k * kernel_size + ki] =
                        (t + depth_k + ki) as u32;
                } else {
                    // tree slot: ancestor at (depth_k − tree_pos) steps up,
                    // tree_pos = ki − (kernel_size − 1 − depth_k)
                    let tree_pos = ki as isize + depth_k as isize
                        - (kernel_size as isize - 1);
                    let steps_up = depth_k as isize - tree_pos;
                    let mut ancestor = k;
                    for _ in 0..steps_up {
                        ancestor = parent[ancestor] as usize;
                    }
                    conv_lut[k * kernel_size + ki] = ancestor as u32;
                }
            }
        }

        Self {
            t,
            parent: parent.to_vec(),
            depth,
            conv_lut,
            anc_lo,
            anc_hi,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Launchers
// ─────────────────────────────────────────────────────────────────────────────

/// Tree conv1d gather launcher (kernel 1).
#[cfg(feature = "cubecl_runtime")]
pub struct TreeConv1dGatherCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl TreeConv1dGatherCubeCL {
    /// # Safety
    ///
    /// - `raw_qkv`: `t * conv_dim` f32 elements (raw pre-conv projections,
    ///   topo-ordered).
    /// - `conv_weight` / `conv_state`: `conv_dim * kernel_size` f32 each.
    /// - `lut`: `t * kernel_size` u32 codes from [`TreeVerifyPlan`].
    /// - `output`: `t * conv_dim` f32 elements.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        raw_qkv: Handle,
        conv_weight: Handle,
        conv_state: Handle,
        lut: Handle,
        output: Handle,
        t: usize,
        conv_dim: usize,
        kernel_size: usize,
    ) {
        let params: [f32; 3] = [t as f32, conv_dim as f32, kernel_size as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = t * conv_dim;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;

        unsafe {
            tree_conv1d_gather_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(raw_qkv, total),
                BufferArg::from_raw_parts(conv_weight, conv_dim * kernel_size),
                BufferArg::from_raw_parts(conv_state, conv_dim * kernel_size),
                BufferArg::from_raw_parts(lut, t * kernel_size),
                BufferArg::from_raw_parts(output, total),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

/// Per-head cumulative log-decay launcher (kernel 2).
#[cfg(feature = "cubecl_runtime")]
pub struct TreeCumulativeLogDecayCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl TreeCumulativeLogDecayCubeCL {
    /// # Safety
    ///
    /// - `decay`: `t * n_head` f32 (token-major).
    /// - `parent`: `t` u32 (`u32::MAX` = root, topo-ordered).
    /// - `cld_out`: `t * n_head` f32.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        decay: Handle,
        parent: Handle,
        cld_out: Handle,
        n_head: usize,
        t: usize,
    ) {
        let params: [f32; 2] = [n_head as f32, t as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let wg = n_head.clamp(1, 256);
        let n_wg = n_head.div_ceil(wg).max(1) as u32;

        unsafe {
            tree_cumulative_log_decay_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(decay, t * n_head),
                BufferArg::from_raw_parts(parent, t),
                BufferArg::from_raw_parts(cld_out, t * n_head),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

/// X/Y interaction-matrix launcher (kernel 3).
#[cfg(feature = "cubecl_runtime")]
pub struct TreeBuildXYCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl TreeBuildXYCubeCL {
    /// # Safety
    ///
    /// - `qkv_expanded`: `t * 3 * n_head * d` f32 (Q | K | V per node).
    /// - `beta`, `cld`: `t * n_head` f32 each.
    /// - `anc_lo`, `anc_hi`: `t` u32 each.
    /// - `x_out`, `y_out`: `n_head * t * t` f32 each.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        qkv_expanded: Handle,
        beta: Handle,
        cld: Handle,
        anc_lo: Handle,
        anc_hi: Handle,
        x_out: Handle,
        y_out: Handle,
        n_head: usize,
        t: usize,
        d: usize,
    ) {
        let params: [f32; 3] = [n_head as f32, t as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * t * t;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;

        unsafe {
            tree_build_xy_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(qkv_expanded, t * 3 * n_head * d),
                BufferArg::from_raw_parts(beta, t * n_head),
                BufferArg::from_raw_parts(cld, t * n_head),
                BufferArg::from_raw_parts(anc_lo, t),
                BufferArg::from_raw_parts(anc_hi, t),
                BufferArg::from_raw_parts(x_out, total),
                BufferArg::from_raw_parts(y_out, total),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

/// Folded-RHS launcher (kernel 4).
#[cfg(feature = "cubecl_runtime")]
pub struct TreeBuildRhsCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl TreeBuildRhsCubeCL {
    /// # Safety
    ///
    /// - `state`: `n_head * d * d` f32 (GPU layout, value-major rows).
    /// - `rhs_out`: `n_head * t * d` f32.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        qkv_expanded: Handle,
        beta: Handle,
        cld: Handle,
        state: Handle,
        rhs_out: Handle,
        n_head: usize,
        t: usize,
        d: usize,
    ) {
        let params: [f32; 3] = [n_head as f32, t as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * t * d;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;

        unsafe {
            tree_build_rhs_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(qkv_expanded, t * 3 * n_head * d),
                BufferArg::from_raw_parts(beta, t * n_head),
                BufferArg::from_raw_parts(cld, t * n_head),
                BufferArg::from_raw_parts(state, n_head * d * d),
                BufferArg::from_raw_parts(rhs_out, total),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

/// Forward-substitution launcher (kernel 5).
#[cfg(feature = "cubecl_runtime")]
pub struct TreeForwardSubCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl TreeForwardSubCubeCL {
    /// # Safety
    ///
    /// - `x`: `n_head * t * t` f32.
    /// - `rhs`, `u_out`: `n_head * t * d` f32 each.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x: Handle,
        rhs: Handle,
        u_out: Handle,
        n_head: usize,
        t: usize,
        d: usize,
    ) {
        let params: [f32; 3] = [n_head as f32, t as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * d;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;

        unsafe {
            tree_forward_sub_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(x, n_head * t * t),
                BufferArg::from_raw_parts(rhs, n_head * t * d),
                BufferArg::from_raw_parts(u_out, n_head * t * d),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

/// Output launcher (kernel 6).
#[cfg(feature = "cubecl_runtime")]
pub struct TreeComputeOutCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl TreeComputeOutCubeCL {
    /// # Safety
    ///
    /// - `y`: `n_head * t * t` f32; `u`, `cld`: as above.
    /// - `out`: `t * n_head * d` f32 (token-major).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        qkv_expanded: Handle,
        y: Handle,
        u: Handle,
        cld: Handle,
        state: Handle,
        out: Handle,
        n_head: usize,
        t: usize,
        d: usize,
    ) {
        let params: [f32; 3] = [n_head as f32, t as f32, d as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = n_head * t * d;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;

        unsafe {
            tree_compute_out_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(qkv_expanded, t * 3 * n_head * d),
                BufferArg::from_raw_parts(y, n_head * t * t),
                BufferArg::from_raw_parts(u, n_head * t * d),
                BufferArg::from_raw_parts(cld, t * n_head),
                BufferArg::from_raw_parts(state, n_head * d * d),
                BufferArg::from_raw_parts(out, t * n_head * d),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Driver-support kernels (Issue 721 T4) — batched elementwise + row moves
// ─────────────────────────────────────────────────────────────────────────────

/// Per-row Split4: splits `[p × (l1+l2+l3+l4)]` rows into four `[p × li]`
/// buffers. The single-token [`Split4CubeCL`](crate::elementwise_cubecl::
/// Split4CubeCL) splits one flat row; the tree projections produce T rows of
/// concatenated `[qkv | z | a | b]`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn split4_batched_f32(
    input: &[f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    out4: &mut [f32],
    params: &[f32],
) {
    let l1 = params[0usize] as usize;
    let l2 = params[1usize] as usize;
    let l3 = params[2usize] as usize;
    let l4 = params[3usize] as usize;
    let p = params[4usize] as usize;
    let row_len = l1 + l2 + l3 + l4;
    let total = p * row_len;

    let idx = ABSOLUTE_POS;
    if idx >= total {
        terminate!();
    }

    let row = idx / row_len;
    let col = idx % row_len;
    let val = input[idx];

    if col < l1 {
        out1[row * l1 + col] = val;
    } else if col < l1 + l2 {
        out2[row * l2 + (col - l1)] = val;
    } else if col < l1 + l2 + l3 {
        out3[row * l3 + (col - l1 - l2)] = val;
    } else {
        out4[row * l4 + (col - l1 - l2 - l3)] = val;
    }
}

/// Launcher for the per-row Split4.
#[cfg(feature = "cubecl_runtime")]
pub struct Split4BatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl Split4BatchedCubeCL {
    /// # Safety
    ///
    /// - `input`: `p * (l1+l2+l3+l4)` f32; each `outN`: `p * lN` f32.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch: many buffer handles are inherent")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        input: Handle,
        out1: Handle,
        out2: Handle,
        out3: Handle,
        out4: Handle,
        l1: usize,
        l2: usize,
        l3: usize,
        l4: usize,
        p: usize,
    ) {
        let row_len = l1 + l2 + l3 + l4;
        let params: [f32; 5] = [l1 as f32, l2 as f32, l3 as f32, l4 as f32, p as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = p * row_len;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;

        unsafe {
            split4_batched_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(input, total),
                BufferArg::from_raw_parts(out1, p * l1),
                BufferArg::from_raw_parts(out2, p * l2),
                BufferArg::from_raw_parts(out3, p * l3),
                BufferArg::from_raw_parts(out4, p * l4),
                BufferArg::from_raw_parts(params_handle, 5),
            );
        }
    }
}

/// Per-row SwiGLU over concatenated `[p × 2n]` gate|up rows → `[p × n]`.
/// The single-token [`DeltanetGatingConcatCubeCL`] splits one flat
/// `[gate(n) | up(n)]` buffer globally; the T-row FFN produces per-row splits.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gating_concat_batched_f32(
    gate_up: &[f32],
    output: &mut [f32],
    params: &[f32],
) {
    let n = params[0usize] as usize;
    let p = params[1usize] as usize;
    let total = p * n;

    let idx = ABSOLUTE_POS;
    if idx >= total {
        terminate!();
    }

    let row = idx / n;
    let col = idx % n;
    let g = gate_up[row * 2 * n + col];
    let u = gate_up[row * 2 * n + n + col];
    let neg_g = f32::new(0.0f32) - g;
    let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_g.exp());
    output[idx] = g * sig * u;
}

/// Launcher for the per-row SwiGLU.
#[cfg(feature = "cubecl_runtime")]
pub struct GatingConcatBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GatingConcatBatchedCubeCL {
    /// # Safety
    ///
    /// - `gate_up`: `p * 2 * n` f32; `output`: `p * n` f32.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        gate_up: Handle,
        output: Handle,
        n: usize,
        p: usize,
    ) {
        let params: [f32; 2] = [n as f32, p as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let total = p * n;
        let wg = 256usize;
        let n_wg = total.div_ceil(wg).max(1) as u32;

        unsafe {
            gating_concat_batched_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(gate_up, p * 2 * n),
                BufferArg::from_raw_parts(output, total),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

/// Copy row `row` of a `[rows × n]` buffer into a flat `[n]` buffer.
/// (`dst[i] = src[row*n + i]` — the per-branch attention bridge.)
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn row_gather_f32(src: &[f32], dst: &mut [f32], params: &[f32]) {
    let n = params[0usize] as usize;
    let row = params[1usize] as usize;
    let i = ABSOLUTE_POS;
    if i >= n {
        terminate!();
    }
    dst[i] = src[row * n + i];
}

/// Copy a flat `[n]` buffer into row `row` of a `[rows × n]` buffer.
/// (`dst[row*n + i] = src[i]`)
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn row_scatter_f32(src: &[f32], dst: &mut [f32], params: &[f32]) {
    let n = params[0usize] as usize;
    let row = params[1usize] as usize;
    let i = ABSOLUTE_POS;
    if i >= n {
        terminate!();
    }
    dst[row * n + i] = src[i];
}

/// Launcher for row gather/scatter (Issue 721 T4 per-branch attention bridge).
#[cfg(feature = "cubecl_runtime")]
pub struct RowGatherScatterCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl RowGatherScatterCubeCL {
    /// # Safety
    ///
    /// - gather: `src` >= `(row+1)*n` f32, `dst` >= `n` f32.
    pub unsafe fn gather<R: Runtime>(
        client: &ComputeClient<R>,
        src: Handle,
        dst: Handle,
        n: usize,
        row: usize,
    ) {
        let params: [f32; 2] = [n as f32, row as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let wg = 256usize;
        let n_wg = n.div_ceil(wg).max(1) as u32;
        unsafe {
            row_gather_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(src, (row + 1) * n),
                BufferArg::from_raw_parts(dst, n),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }

    /// # Safety
    ///
    /// - scatter: `src` >= `n` f32, `dst` >= `(row+1)*n` f32.
    pub unsafe fn scatter<R: Runtime>(
        client: &ComputeClient<R>,
        src: Handle,
        dst: Handle,
        n: usize,
        row: usize,
    ) {
        let params: [f32; 2] = [n as f32, row as f32];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));
        let wg = 256usize;
        let n_wg = n.div_ceil(wg).max(1) as u32;
        unsafe {
            row_scatter_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg as u32),
                BufferArg::from_raw_parts(src, n),
                BufferArg::from_raw_parts(dst, (row + 1) * n),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — GPU kernels vs the katgpt-core CPU oracle
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};

    /// f32 tolerance: the GPU path recomputes decay ratios in f32 (the CPU
    /// oracle accumulates cld in f64) and reduces dots in a different order.
    /// 1e-3 matches the module-wide tolerance of the neighboring kernels.
    const TOL: f32 = 1e-3;

    /// Deterministic LCG — house pattern for kernel fixtures.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (self.0 >> 33) as u32;
            lo + (bits as f32 / u32::MAX as f32) * (hi - lo)
        }
    }

    /// A 9-node tree, already topo-ordered (BFS): root; 3 children; each
    /// child has 1–2 children. Covers chain + branching + mixed depths.
    fn tree9() -> Vec<u32> {
        // node: 0=root, 1..3 = children of 0, 4..5 = children of 1,
        // 6 = child of 2, 7..8 = children of 3
        vec![
            u32::MAX,
            0,
            0,
            0,
            1,
            1,
            2,
            3,
            3,
        ]
    }

    /// CPU reference for the tree conv1d gather — a direct transcription of
    /// `forward_tree_deltanet_layer` step 2 (without the projection inputs).
    fn cpu_tree_conv(
        raw_qkv: &[f32],
        conv_weight: &[f32],
        conv_state: &[f32],
        plan: &TreeVerifyPlan,
        conv_dim: usize,
        kernel_size: usize,
    ) -> Vec<f32> {
        let t = plan.t;
        let mut out = vec![0.0f32; t * conv_dim];
        for k in 0..t {
            let depth_k = plan.depth[k];
            let committed_count = kernel_size.saturating_sub(1 + depth_k);
            for ch in 0..conv_dim {
                let cw = ch * kernel_size;
                let mut sum = 0.0f32;
                for ki in 0..committed_count {
                    sum += conv_state[cw + depth_k + ki] * conv_weight[cw + ki];
                }
                for ki in committed_count..kernel_size {
                    let ancestor = plan.conv_lut[k * kernel_size + ki] as usize;
                    sum += raw_qkv[ancestor * conv_dim + ch] * conv_weight[cw + ki];
                }
                let sig = 1.0 / (1.0 + (-sum).exp());
                out[k * conv_dim + ch] = sum * sig;
            }
        }
        out
    }

    #[test]
    fn test_tree_conv1d_gather_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let t = 9usize;
        let conv_dim = 12usize;
        let kernel_size = 4usize;
        let plan = TreeVerifyPlan::from_parents_topo(&tree9(), kernel_size);

        let mut rng = Lcg(0x5EED_7211);
        let raw_qkv: Vec<f32> = (0..t * conv_dim).map(|_| rng.next_f32(-0.5, 0.5)).collect();
        let conv_weight: Vec<f32> = (0..conv_dim * kernel_size)
            .map(|_| rng.next_f32(-0.3, 0.3))
            .collect();
        let conv_state: Vec<f32> = (0..conv_dim * kernel_size)
            .map(|_| rng.next_f32(-0.5, 0.5))
            .collect();

        let expected = cpu_tree_conv(&raw_qkv, &conv_weight, &conv_state, &plan, conv_dim, kernel_size);

        let raw_h = client.create_from_slice(f32::as_bytes(&raw_qkv));
        let w_h = client.create_from_slice(f32::as_bytes(&conv_weight));
        let st_h = client.create_from_slice(f32::as_bytes(&conv_state));
        let lut_h = client.create_from_slice(bytemuck::cast_slice(&plan.conv_lut));
        let out_h = client.empty(t * conv_dim * std::mem::size_of::<f32>());

        unsafe {
            TreeConv1dGatherCubeCL::launch::<ActiveRuntime>(
                &client, raw_h, w_h, st_h, lut_h, out_h.clone(), t, conv_dim, kernel_size,
            );
        }

        let bytes = client.read_one(out_h).expect("read output");
        let gpu = f32::from_bytes(&bytes);
        assert_eq!(gpu.len(), expected.len());
        let mut max_diff = 0.0f32;
        for (i, (&g, &c)) in gpu.iter().zip(expected.iter()).enumerate() {
            let diff = (g - c).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < TOL,
                "conv[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6} > {TOL}"
            );
        }
        eprintln!("tree conv1d gather GOAT ✓ max_diff={max_diff:.6}");
    }

    /// Full masked-solve pipeline (kernels 2–6) vs per-head
    /// `verify_gdn_tree` — the SAME primitive the CPU engine oracle uses.
    #[test]
    fn test_tree_masked_solve_matches_cpu_oracle() {
        use katgpt_core::gdn_tree_verify::{
            build_topology, verify_gdn_tree, GdnLayerParams, GdnTreeVerifier,
        };

        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let t = 9usize;
        let n_head = 4usize;
        let d = 32usize; // smaller than production 128 for test speed
        let parent = tree9();
        let plan = TreeVerifyPlan::from_parents_topo(&parent, 4);

        let mut rng = Lcg(0xC0_FFEE_7211);

        // Random q/k per head per node, L2-normalized (the pipeline's
        // expand+L2 kernel output). v random. beta in (0.1, 0.9), decay in
        // (0.8, 0.99) — realistic gate ranges.
        let mut qkv_expanded = vec![0.0f32; t * 3 * n_head * d];
        for i in 0..t {
            for h in 0..n_head {
                for sec in 0..3 {
                    let base = i * 3 * n_head * d + sec * n_head * d + h * d;
                    for m in 0..d {
                        qkv_expanded[base + m] = rng.next_f32(-0.5, 0.5);
                    }
                    if sec < 2 {
                        // L2-normalize Q and K sections
                        let base = i * 3 * n_head * d + sec * n_head * d + h * d;
                        let norm: f32 =
                            qkv_expanded[base..base + d].iter().map(|v| v * v).sum::<f32>().sqrt();
                        for m in 0..d {
                            qkv_expanded[base + m] /= norm.max(1e-8);
                        }
                    }
                }
            }
        }
        let beta: Vec<f32> = (0..t * n_head).map(|_| rng.next_f32(0.1, 0.9)).collect();
        let decay: Vec<f32> = (0..t * n_head).map(|_| rng.next_f32(0.8, 0.99)).collect();

        // GPU-layout state per head: [d_v × d_k], value-major rows.
        let state: Vec<f32> = (0..n_head * d * d).map(|_| rng.next_f32(-0.05, 0.05)).collect();

        // ── CPU oracle: per-head verify_gdn_tree (topology is identity-mapped
        // because our fixture is already topo-ordered; cld per head). ──
        let mut expected = vec![0.0f32; t * n_head * d]; // token-major [i, h*d + dv]
        {
            // katgpt-core alphas are original-indexed; our fixture IS the
            // topo order, so original index == topo index.
            for h in 0..n_head {
                let alphas_h: Vec<f32> = (0..t).map(|k| decay[k * n_head + h]).collect();
                let betas_h: Vec<f32> = (0..t).map(|k| beta[k * n_head + h]).collect();
                let topo = build_topology(
                    &parent
                        .iter()
                        .map(|&p| if p == u32::MAX { usize::MAX } else { p as usize })
                        .collect::<Vec<_>>(),
                    &alphas_h,
                );
                // topo_order of an already-topo-ordered parent list is the
                // identity — assert that rather than assume it.
                assert_eq!(
                    topo.topo_order,
                    (0..t).collect::<Vec<_>>(),
                    "fixture must be identity-topo-ordered"
                );

                // q/k/v per head, node-indexed (topo == original here).
                let mut q_h = vec![0.0f32; t * d];
                let mut k_h = vec![0.0f32; t * d];
                let mut v_h = vec![0.0f32; t * d];
                // q/k/v are interleaved per position at stride `n_head * d`:
                // slot 0 = q, slot 1 = k, slot 2 = v. The q term's `0 *` factor
                // was spelled out for symmetry; clippy `erasing_op` (deny) rejects
                // it, so the slot index lives in this comment and the stride is
                // hoisted instead of repeated three times.
                let qkv_stride = 3 * n_head * d;
                for i in 0..t {
                    let q_base = i * qkv_stride + h * d;
                    let k_base = i * qkv_stride + n_head * d + h * d;
                    let v_base = i * qkv_stride + 2 * n_head * d + h * d;
                    q_h[i * d..(i + 1) * d].copy_from_slice(&qkv_expanded[q_base..q_base + d]);
                    k_h[i * d..(i + 1) * d].copy_from_slice(&qkv_expanded[k_base..k_base + d]);
                    v_h[i * d..(i + 1) * d].copy_from_slice(&qkv_expanded[v_base..v_base + d]);
                }

                // Transpose the GPU-layout state (value-major) into the
                // oracle's S₀ (key-major): s0[m*d_v + d] = state[d*d_k + m].
                let mut s0 = vec![0.0f32; d * d];
                for dv in 0..d {
                    for m in 0..d {
                        s0[m * d + dv] = state[h * d * d + dv * d + m];
                    }
                }

                let params = GdnLayerParams {
                    keys: &k_h,
                    values: &v_h,
                    queries: &q_h,
                    alphas: &alphas_h,
                    betas: &betas_h,
                };
                let mut verifier = GdnTreeVerifier::new(t, d, d);
                let out_h = verify_gdn_tree(&mut verifier, &topo, &params, &s0, d, d);

                // out_h is topo-indexed [i * d_v] per head; scatter to token-major.
                for i in 0..t {
                    for dv in 0..d {
                        expected[i * n_head * d + h * d + dv] = out_h[i * d + dv];
                    }
                }
            }
        }

        // ── GPU pipeline ──
        let qkv_h = client.create_from_slice(f32::as_bytes(&qkv_expanded));
        let beta_h = client.create_from_slice(f32::as_bytes(&beta));
        let decay_h = client.create_from_slice(f32::as_bytes(&decay));
        let parent_h = client.create_from_slice(bytemuck::cast_slice(&parent));
        let anc_lo_h = client.create_from_slice(bytemuck::cast_slice(&plan.anc_lo));
        let anc_hi_h = client.create_from_slice(bytemuck::cast_slice(&plan.anc_hi));
        let state_h = client.create_from_slice(f32::as_bytes(&state));

        let cld_h = client.empty(t * n_head * std::mem::size_of::<f32>());
        let x_h = client.empty(n_head * t * t * std::mem::size_of::<f32>());
        let y_h = client.empty(n_head * t * t * std::mem::size_of::<f32>());
        let rhs_h = client.empty(n_head * t * d * std::mem::size_of::<f32>());
        let u_h = client.empty(n_head * t * d * std::mem::size_of::<f32>());
        let out_h = client.empty(t * n_head * d * std::mem::size_of::<f32>());

        unsafe {
            TreeCumulativeLogDecayCubeCL::launch::<ActiveRuntime>(
                &client, decay_h.clone(), parent_h.clone(), cld_h.clone(), n_head, t,
            );
            TreeBuildXYCubeCL::launch::<ActiveRuntime>(
                &client,
                qkv_h.clone(),
                beta_h.clone(),
                cld_h.clone(),
                anc_lo_h.clone(),
                anc_hi_h.clone(),
                x_h.clone(),
                y_h.clone(),
                n_head,
                t,
                d,
            );
            TreeBuildRhsCubeCL::launch::<ActiveRuntime>(
                &client,
                qkv_h.clone(),
                beta_h.clone(),
                cld_h.clone(),
                state_h.clone(),
                rhs_h.clone(),
                n_head,
                t,
                d,
            );
            TreeForwardSubCubeCL::launch::<ActiveRuntime>(
                &client, x_h.clone(), rhs_h.clone(), u_h.clone(), n_head, t, d,
            );
            TreeComputeOutCubeCL::launch::<ActiveRuntime>(
                &client,
                qkv_h.clone(),
                y_h.clone(),
                u_h.clone(),
                cld_h.clone(),
                state_h.clone(),
                out_h.clone(),
                n_head,
                t,
                d,
            );
        }

        let bytes = client.read_one(out_h).expect("read output");
        let gpu = f32::from_bytes(&bytes);
        assert_eq!(gpu.len(), expected.len());
        let mut max_diff = 0.0f32;
        for (i, (&g, &c)) in gpu.iter().zip(expected.iter()).enumerate() {
            let diff = (g - c).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < TOL,
                "solve[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6} > {TOL}"
            );
        }
        eprintln!("tree masked solve GOAT ✓ max_diff={max_diff:.6}");
    }

    /// T=1 chain: the masked solve must degenerate to the single-token
    /// recurrence read (`O = scale·(qᵀS + β·k·(v − kᵀS)·…)`) — pinned by the
    /// oracle itself, so this test only guards the launcher plumbing at the
    /// degenerate shape.
    #[test]
    fn test_tree_solve_t1_chain() {
        use katgpt_core::gdn_tree_verify::{
            build_topology, verify_gdn_tree, GdnLayerParams, GdnTreeVerifier,
        };

        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let t = 1usize;
        let n_head = 2usize;
        let d = 16usize;
        let parent = vec![u32::MAX];
        let plan = TreeVerifyPlan::from_parents_topo(&parent, 4);

        let mut rng = Lcg(7);
        let mut qkv_expanded = vec![0.0f32; t * 3 * n_head * d];
        for i in 0..t {
            for h in 0..n_head {
                for sec in 0..3 {
                    let base = i * 3 * n_head * d + sec * n_head * d + h * d;
                    for m in 0..d {
                        qkv_expanded[base + m] = rng.next_f32(-0.5, 0.5);
                    }
                    if sec < 2 {
                        let norm: f32 =
                            qkv_expanded[base..base + d].iter().map(|v| v * v).sum::<f32>().sqrt();
                        for m in 0..d {
                            qkv_expanded[base + m] /= norm.max(1e-8);
                        }
                    }
                }
            }
        }
        let beta: Vec<f32> = (0..t * n_head).map(|_| rng.next_f32(0.1, 0.9)).collect();
        let decay: Vec<f32> = (0..t * n_head).map(|_| rng.next_f32(0.8, 0.99)).collect();
        let state: Vec<f32> = (0..n_head * d * d).map(|_| rng.next_f32(-0.05, 0.05)).collect();

        // CPU oracle
        let mut expected = vec![0.0f32; t * n_head * d];
        for h in 0..n_head {
            let alphas_h: Vec<f32> = (0..t).map(|k| decay[k * n_head + h]).collect();
            let betas_h: Vec<f32> = (0..t).map(|k| beta[k * n_head + h]).collect();
            let topo = build_topology(&[usize::MAX], &alphas_h);
            let mut q_h = vec![0.0f32; t * d];
            let mut k_h = vec![0.0f32; t * d];
            let mut v_h = vec![0.0f32; t * d];
            {
                let mut sections: [&mut [f32]; 3] = [&mut q_h, &mut k_h, &mut v_h];
                for (sec, dst) in sections.iter_mut().enumerate() {
                    let base = sec * n_head * d + h * d;
                    dst[..d].copy_from_slice(&qkv_expanded[base..base + d]);
                }
            }
            let mut s0 = vec![0.0f32; d * d];
            for dv in 0..d {
                for m in 0..d {
                    s0[m * d + dv] = state[h * d * d + dv * d + m];
                }
            }
            let params = GdnLayerParams {
                keys: &k_h,
                values: &v_h,
                queries: &q_h,
                alphas: &alphas_h,
                betas: &betas_h,
            };
            let mut verifier = GdnTreeVerifier::new(t, d, d);
            let out_h = verify_gdn_tree(&mut verifier, &topo, &params, &s0, d, d);
            for i in 0..t {
                for dv in 0..d {
                    expected[i * n_head * d + h * d + dv] = out_h[i * d + dv];
                }
            }
        }

        // GPU
        let qkv_h = client.create_from_slice(f32::as_bytes(&qkv_expanded));
        let beta_h = client.create_from_slice(f32::as_bytes(&beta));
        let decay_h = client.create_from_slice(f32::as_bytes(&decay));
        let parent_h = client.create_from_slice(bytemuck::cast_slice(&parent));
        let anc_lo_h = client.create_from_slice(bytemuck::cast_slice(&plan.anc_lo));
        let anc_hi_h = client.create_from_slice(bytemuck::cast_slice(&plan.anc_hi));
        let state_h = client.create_from_slice(f32::as_bytes(&state));

        let cld_h = client.empty(t * n_head * std::mem::size_of::<f32>());
        let x_h = client.empty(n_head * t * t * std::mem::size_of::<f32>());
        let y_h = client.empty(n_head * t * t * std::mem::size_of::<f32>());
        let rhs_h = client.empty(n_head * t * d * std::mem::size_of::<f32>());
        let u_h = client.empty(n_head * t * d * std::mem::size_of::<f32>());
        let out_h = client.empty(t * n_head * d * std::mem::size_of::<f32>());

        unsafe {
            TreeCumulativeLogDecayCubeCL::launch::<ActiveRuntime>(
                &client, decay_h.clone(), parent_h.clone(), cld_h.clone(), n_head, t,
            );
            TreeBuildXYCubeCL::launch::<ActiveRuntime>(
                &client, qkv_h.clone(), beta_h.clone(), cld_h.clone(), anc_lo_h.clone(),
                anc_hi_h.clone(), x_h.clone(), y_h.clone(), n_head, t, d,
            );
            TreeBuildRhsCubeCL::launch::<ActiveRuntime>(
                &client, qkv_h.clone(), beta_h.clone(), cld_h.clone(), state_h.clone(),
                rhs_h.clone(), n_head, t, d,
            );
            TreeForwardSubCubeCL::launch::<ActiveRuntime>(
                &client, x_h.clone(), rhs_h.clone(), u_h.clone(), n_head, t, d,
            );
            TreeComputeOutCubeCL::launch::<ActiveRuntime>(
                &client, qkv_h.clone(), y_h.clone(), u_h.clone(), cld_h.clone(),
                state_h.clone(), out_h.clone(), n_head, t, d,
            );
        }

        let bytes = client.read_one(out_h).expect("read output");
        let gpu = f32::from_bytes(&bytes);
        let mut max_diff = 0.0f32;
        for (i, (&g, &c)) in gpu.iter().zip(expected.iter()).enumerate() {
            let diff = (g - c).abs();
            max_diff = max_diff.max(diff);
            assert!(diff < TOL, "t1[{i}] GPU={g:.6} CPU={c:.6} diff={diff:.6}");
        }
        eprintln!("tree solve T=1 GOAT ✓ max_diff={max_diff:.6}");
    }
}
