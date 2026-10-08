//! Issue 1005 T8 — the GPU eDLM forward (`edlm_gpu` feature): f16-resident
//! weights, multi-query GQA flash attention with per-row visibility bounds,
//! parity vs the CPU T7 cached path as the gate (tolerance class, laya-G5
//! style: argmax exact + disclosed drift — never bit-identity).
//!
//! # The decomposition (T7's, unchanged)
//!
//! eDLM never needs a general bool mask on the GPU. The T7 state-prefix
//! decomposition maps every phase to two key segments, visited in order:
//!
//! - **state prefill**: no seed; own window `0..own_total` (state_bidir) or
//!   `0..i + 1` (causal) — `branch_mask` over an all-state segment;
//! - **branch continuations**: the seed segment `0..sl` (the state KV,
//!   device-resident) then the pass's own window — all state keys + causal
//!   within the branch (`branch_continuation_allow`; state keys precede
//!   every branch query), per-query own windows `[seg, seg + j + 1)` so a
//!   BATCHED pass over several branch rows isolates each row's keys.
//!
//! One kernel shape serves all three ([`EdlmAttnMultiCubeCL`]).
//!
//! # Weights (the upload-path decision)
//!
//! Q8_0 → f32 EXACT on the host (the same `dequant_f16_to_f32` the CPU
//! streaming path reads) → round-to-nearest f16 → f16-resident buffers
//! (~15.7 GB on the 8B release: fits the 24 GB 4090 and unified-memory M3).
//! Every GEMM accumulates f32 (the f16 GEMV family's law), so the only
//! numerics faces vs the CPU f32 path are (1) the f16 weight rounding
//! (rel ≤ 2^-11 per weight) and (2) the attention kernel's online-softmax
//! accumulation order. Quant-resident GPU dequant kernels are the documented
//! fallback if the tolerance gate ever reds — a bigger kernel with no
//! measurement-lane need (this lane is CC BY-NC, local bench only).
//!
//! # Sync posture (the device fold, `EDLM_GPU_FOLD`)
//!
//! DEFAULT (fold ON): everything between the pass-final readback runs ON
//! DEVICE — the GEMMs ([`MatmulF16bCubeCL`] / the CMMA postures), the
//! multi-row RMSNorm, the fused QKV fold ([`EdlmQkvFoldCubeCL`]: split +
//! per-head qk-norm + RoPE), the attention, the SiLU gate, and the residual
//! adds (in place on the device). ZERO mid-pass readbacks; the only reads
//! are the pass-final hidden (every pass) and the per-layer KV capture
//! (state prefill only). KILL-SWITCH `EDLM_GPU_FOLD=0` restores the v1
//! hybrid shape — the GEMMs/norms/attention on GPU and the small-ops on the
//! host between FOUR per-layer readbacks (the exact core helpers, never
//! copies) — the parity anchor arm the fold-vs-host arms test exercises.
//! Both arms consume ONE [`PassGeometry`].
//!
//! # State cache posture (the GPU-resident KV carry)
//!
//! `state_prefill` retains each layer's combined `[keys | values]` buffer ON
//! THE DEVICE ([`EdlmGpuStateKv::device_kv`]) beside the T7 host cache (the
//! host copy stays for parity + the pointer head). Branch passes attend the
//! device seed directly — the per-row-per-layer host copy + re-upload of the
//! combined `[state | branch]` cache is gone (at the 8B release shape:
//! ~3.1 MB × n_layer per branch row of traffic eliminated).
//!
//! # Out of scope (disclosed)
//!
//! - The packed-mask forward (option isolation) — rows/cached semantics only,
//!   the CPU cached path's own caveat.
//! - CUDA graphs capture (the fold landed the device-resident forward it
//!   requires; the graphs unit is next — replay tolerates no host
//!   readbacks, and the fold left exactly two per pass).
//!
//! The CMMA tensor-core GEMM is IN: the four projections dispatch through
//! [`crate::matmul_f16b_cmma_cubecl`] unless `EDLM_GPU_CMMA=0` (the scalar
//! kernel stays the kill-switch posture; the GOAT evidence:
//! `tests/edlm_matmul_cmma_goat.rs` + the issue row). The GPU-resident KV
//! carry + the batched multi-branch pass are IN (the seed-segment attention
//! kernel + [`EdlmGpuModel::forward_branches`] over all rows in one pass;
//! GOAT: `tests/edlm_branch_batch_goat.rs` + the issue row). The GPU-resident
//! elementwise fold is IN (the fused QKV fold + the SiLU gate + device
//! residual adds; kill-switch `EDLM_GPU_FOLD=0`; GOAT:
//! `tests/edlm_fold_goat.rs` + the issue row).

#[cfg(feature = "edlm_gpu")]
use crate::cubecl_runtime::{
    ActiveRuntime, CubeCLContext, assert_binding_derives_units, create_f32, create_u32, read_f32,
};
use crate::matmul_f16b_cubecl::MatmulF16bCubeCL;
#[cfg(feature = "edlm_gpu")]
use cubecl::prelude::*;
#[cfg(feature = "edlm_gpu")]
use cubecl::server::Handle;
#[cfg(feature = "edlm_gpu")]
use half::f16 as half_f16;
#[cfg(feature = "edlm_gpu")]
use riir_infer_core::rope::{RopeFreqTable, apply_rope_with_freq};
#[cfg(feature = "edlm_gpu")]
use riir_infer_core::transformer::edlm::{
    BranchRow, EdlmGgufModel, EdlmLayerKv, EdlmPointerHead, EdlmStateKv, EdlmWeights,
    PackedEncoding, qk_norm_inplace,
};
#[cfg(feature = "edlm_gpu")]
use riir_infer_core::types::{Config, kv_dim, rmsnorm_with_gamma_eps, swiglu};

// ── kernels ─────────────────────────────────────────────────────────

/// Multi-row RMSNorm: `out[r, i] = x[r, i] / sqrt(mean(x[r,:]^2) + eps) ·
/// gamma[i]` — the layer-norm shape (one normed stream feeding the QKV and
/// gate/up GEMMs for the whole sequence).
///
/// Grid `(rows, 1, 1)`, 256-thread cubes striding the row (n is unbounded —
/// the while-loop stride covers any n), the eps riding `params[0]` (the
/// crate's params-buffer discipline: no u32→f32 cast issues, no 5th array).
/// The sum-of-squares order differs from the CPU sequential sum —
/// tolerance-class, never bit-identity.
#[cfg(feature = "edlm_gpu")]
#[cube(launch_unchecked)]
fn edlm_rmsnorm_rows_f32(x: &[f32], gamma: &[f32], params: &[f32], out: &mut [f32]) {
    let n = gamma.len() as u32;
    let row = CUBE_POS_X;
    let base = (row * n) as usize;
    let tid = UNIT_POS;

    let mut local = f32::new(0.0f32);
    let mut i = tid;
    while i < n {
        let v = x[base + i as usize];
        local += v * v;
        i += 256u32;
    }

    let mut red = Shared::<[f32]>::new_slice(256usize);
    red[tid as usize] = local;
    sync_cube();
    if tid < 128u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 128u32) as usize];
    }
    sync_cube();
    if tid < 64u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 64u32) as usize];
    }
    sync_cube();
    if tid < 32u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 32u32) as usize];
    }
    sync_cube();
    if tid < 16u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 16u32) as usize];
    }
    sync_cube();
    if tid < 8u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 8u32) as usize];
    }
    sync_cube();
    if tid < 4u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 4u32) as usize];
    }
    sync_cube();
    if tid < 2u32 {
        red[tid as usize] = red[tid as usize] + red[(tid + 2u32) as usize];
    }
    sync_cube();
    if tid < 1u32 {
        red[0usize] = red[0usize] + red[1usize];
    }
    sync_cube();

    let sum_sq = red[0usize];
    let inv_rms = f32::new(1.0f32) / (sum_sq / (n as f32) + params[0usize]).sqrt();
    let mut i = tid;
    while i < n {
        out[base + i as usize] = x[base + i as usize] * inv_rms * gamma[i as usize];
        i += 256u32;
    }
}

/// The fused QKV fold — the device-resident half of the sync-posture fold:
/// per `(head-unit, query)` cube, apply the host small-ops ON DEVICE and
/// scatter straight into the layouts the downstream kernels read, so the
/// per-layer readback of the raw QKV (and the re-upload of the folded
/// q/keys/values) never happens.
///
/// Grid `(n_head + 2·n_kv_head, seq_q, 1)`, 128-thread cubes == head_dim
/// (the 8B release shape, asserted at the launcher). Unit `u` selects the
/// work: `u < n_head` → q head (qk-norm + RoPE → `q_rope`);
/// `n_head ≤ u < n_head + n_kv` → k head (qk-norm + RoPE → keys row
/// `seg[qi]`); else → v copy-through (values row `n_positions + seg[qi]`).
/// `params = [n_head, n_kv_head, head_dim, eps, n_positions]`.
///
/// Numerics: the per-head sum-of-squares rides an unrolled cube reduction
/// (tree order vs the CPU's sequential sum) and the rotation angles ride
/// the GPU sin/cos — tolerance-class faces, the same class the rmsnorm and
/// attention kernels already carry. `pos == 0` needs no fast path: the
/// general rotation is exact identity there (cos 0 = 1, sin 0 = 0).
#[cfg(feature = "edlm_gpu")]
#[cube(launch_unchecked)]
fn edlm_qkv_fold_f32(
    qkv: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    pos_u32: &[u32],
    seg_u32: &[u32],
    freq: &[f32],
    params: &[f32],
    q_rope: &mut [f32],
    kv: &mut [f32],
) {
    let n_head = params[0usize] as u32;
    let n_kv = params[1usize] as u32;
    let hd = params[2usize] as u32;
    let eps = params[3usize];
    let n_pos = params[4usize] as u32;
    let kvd = n_kv * hd;
    let q_dim = n_head * hd;
    let lq = q_dim + 2u32 * kvd;
    let half = hd / 2u32;
    let unit = CUBE_POS_X;
    let qi = CUBE_POS_Y;
    let tid = UNIT_POS;
    let row_base = (qi * lq) as usize;

    if unit < n_head {
        // ── q head: qk-norm + RoPE → q_rope ──
        let h = unit;
        let src = row_base + (h * hd) as usize;
        let v = qkv[src + tid as usize];
        let mut red = Shared::<[f32]>::new_slice(128usize);
        red[tid as usize] = v * v;
        sync_cube();
        if tid < 64u32 {
            red[tid as usize] = red[tid as usize] + red[(tid + 64u32) as usize];
        }
        sync_cube();
        if tid < 32u32 {
            red[tid as usize] = red[tid as usize] + red[(tid + 32u32) as usize];
        }
        sync_cube();
        if tid < 16u32 {
            red[tid as usize] = red[tid as usize] + red[(tid + 16u32) as usize];
        }
        sync_cube();
        if tid < 8u32 {
            red[tid as usize] = red[tid as usize] + red[(tid + 8u32) as usize];
        }
        sync_cube();
        if tid < 4u32 {
            red[tid as usize] = red[tid as usize] + red[(tid + 4u32) as usize];
        }
        sync_cube();
        if tid < 2u32 {
            red[tid as usize] = red[tid as usize] + red[(tid + 2u32) as usize];
        }
        sync_cube();
        if tid < 1u32 {
            red[0usize] = red[0usize] + red[1usize];
        }
        sync_cube();
        let inv = f32::new(1.0f32) / (red[0usize] / (hd as f32) + eps).sqrt();
        // The half-split rotation pair: (i, i + half) — dim `tid < half` is
        // the "x" element rotating against `tid + half`, dim `tid ≥ half` is
        // the "y" element of pair `tid - half` (the host rope's
        // `apply_rope_heads_precomputed` pairing; the table index is the
        // pair's x index in both cases).
        let is_x = tid < half;
        let pidx = if is_x { tid + half } else { tid - half };
        let fidx = if is_x { tid } else { tid - half };
        let angle = (pos_u32[qi as usize] as f32) * freq[fidx as usize];
        let c = angle.cos();
        let s = angle.sin();
        let a = v * (inv * q_norm[tid as usize]);
        let partner = qkv[src + pidx as usize] * (inv * q_norm[pidx as usize]);
        let x = if is_x { a } else { partner };
        let y = if is_x { partner } else { a };
        let out = if is_x { x * c - y * s } else { x * s + y * c };
        q_rope[(qi * q_dim + h * hd + tid) as usize] = out;
    } else if unit < n_head + n_kv {
        // ── k head: qk-norm + RoPE → keys row seg[qi] ──
        let h = unit - n_head;
        let src = row_base + (q_dim + h * hd) as usize;
        let v = qkv[src + tid as usize];
        let mut red = Shared::<[f32]>::new_slice(128usize);
        red[tid as usize] = v * v;
        sync_cube();
        if tid < 64u32 {
            red[tid as usize] = red[tid as usize] + red[(tid + 64u32) as usize];
        }
        sync_cube();
        if tid < 32u32 {
            red[tid as usize] = red[tid as usize] + red[(tid + 32u32) as usize];
        }
        sync_cube();
        if tid < 16u32 {
            red[tid as usize] = red[tid as usize] + red[(tid + 16u32) as usize];
        }
        sync_cube();
        if tid < 8u32 {
            red[tid as usize] = red[tid as usize] + red[(tid + 8u32) as usize];
        }
        sync_cube();
        if tid < 4u32 {
            red[tid as usize] = red[tid as usize] + red[(tid + 4u32) as usize];
        }
        sync_cube();
        if tid < 2u32 {
            red[tid as usize] = red[tid as usize] + red[(tid + 2u32) as usize];
        }
        sync_cube();
        if tid < 1u32 {
            red[0usize] = red[0usize] + red[1usize];
        }
        sync_cube();
        let inv = f32::new(1.0f32) / (red[0usize] / (hd as f32) + eps).sqrt();
        let is_x = tid < half;
        let pidx = if is_x { tid + half } else { tid - half };
        let fidx = if is_x { tid } else { tid - half };
        let angle = (pos_u32[qi as usize] as f32) * freq[fidx as usize];
        let c = angle.cos();
        let s = angle.sin();
        let a = v * (inv * k_norm[tid as usize]);
        let partner = qkv[src + pidx as usize] * (inv * k_norm[pidx as usize]);
        let x = if is_x { a } else { partner };
        let y = if is_x { partner } else { a };
        let out = if is_x { x * c - y * s } else { x * s + y * c };
        let krow = seg_u32[qi as usize];
        kv[(krow * kvd + h * hd + tid) as usize] = out;
    } else {
        // ── v copy-through → values row n_pos + seg[qi] ──
        let h = unit - n_head - n_kv;
        let src = row_base + (q_dim + kvd + h * hd) as usize;
        let vrow = n_pos + seg_u32[qi as usize];
        kv[((vrow * kvd + h * hd) + tid) as usize] = qkv[src + tid as usize];
    }
}

/// Multi-query GQA flash attention with a device-resident seed segment and
/// per-query own-window visibility bounds — the eDLM eligibility law (see
/// the module doc). One cube per (head, query): grid `(n_head, seq_q, 1)`,
/// 128-thread cubes == head_dim (the 8B release shape, asserted at the
/// launcher). `params = [n_head, n_kv_head, head_dim, scale, n_seed]`.
///
/// Two source segments, visited IN ORDER (the T7 law — state keys precede
/// every branch query; the online softmax carries across the seam):
/// - **seed** `[0, n_seed)` from `seed_kv` — the device-resident state KV
///   (same `[keys | values]` layout as `kv`; a 1-element dummy when
///   `n_seed == 0`, the loop body never reads it);
/// - **own window** `[t_start[qi], t_end[qi])` from `kv` — per-query bounds
///   express the causal/bidir state pass AND the batched multi-branch row
///   isolation (`[seg, seg + j + 1)`).
///
/// Online softmax over key tiles of 128 (the `attention_decode` family's
/// shape): each thread owns one key per tile and computes the full smem-Q dot
/// against its global K row; the tile max/sum ride unrolled cube reductions;
/// the output dimension `tid` accumulates its own weighted V sum, rescaled by
/// the running-max correction each tile, normalized by the final running sum.
/// Accumulation order differs from the CPU plain softmax — tolerance-class.
/// The tile body is spelled twice (seed / own) rather than selected at
/// runtime: a slice variable switch is codegen-risk for zero win (the same
/// discipline as the hand-unrolled reductions).
#[cfg(feature = "edlm_gpu")]
#[cube(launch_unchecked)]
fn edlm_attn_multi_f32(
    q: &[f32],
    kv: &[f32],
    seed_kv: &[f32],
    t_start: &[u32],
    t_end: &[u32],
    params: &[f32],
    out: &mut [f32],
) {
    let n_head = params[0usize] as u32;
    let n_kv = params[1usize] as u32;
    let hd = params[2usize] as u32;
    let scale = params[3usize];
    let n_seed = params[4usize] as u32;
    let kvd = n_kv * hd;
    // The combined cache is [keys(n_pos·kvd) | values(n_pos·kvd)] — the V of
    // key j lives at kv_half + j·kvd (the attention_decode family's layout);
    // the seed cache has the SAME layout over n_seed keys.
    let n_pos = (kv.len() as u32) / (2u32 * kvd);
    let kv_half = (kv.len() as u32) / 2u32;
    let seed_half = n_seed * kvd;

    let h = CUBE_POS_X;
    let qi = CUBE_POS_Y;
    let kvh = h * n_kv / n_head;
    let tid = UNIT_POS;

    let mut q_smem = Shared::<[f32]>::new_slice(128usize);
    let mut w = Shared::<[f32]>::new_slice(128usize);
    let mut red = Shared::<[f32]>::new_slice(128usize);
    q_smem[tid as usize] = q[((qi * n_head + h) * hd + tid) as usize];
    sync_cube();

    let lo = t_start[qi as usize];
    let hi = t_end[qi as usize];
    let big_neg = f32::new(-1.0e30f32);
    let mut m = big_neg;
    let mut ssum = f32::new(0.0f32);
    let mut acc = f32::new(0.0f32);

    // ── phase 0: the seed keys [0, n_seed) — the device-resident state KV ──
    let mut j0 = 0u32;
    while j0 < n_seed {
        // ── score: thread tid owns key j0 + tid (full smem-Q dot) ──
        let j = j0 + tid;
        let mut s = big_neg;
        if j < n_seed {
            let kro = (j * kvd + kvh * hd) as usize;
            let mut dot = f32::new(0.0f32);
            let mut t = 0u32;
            while t < hd {
                dot += q_smem[t as usize] * seed_kv[kro + t as usize];
                t += 1u32;
            }
            s = dot * scale;
        }

        // ── tile max ──
        red[tid as usize] = s;
        sync_cube();
        let mut off = 64u32;
        while off >= 1u32 {
            if tid < off {
                let other = red[(tid + off) as usize];
                if other > red[tid as usize] {
                    red[tid as usize] = other;
                }
            }
            sync_cube();
            off /= 2u32;
        }
        let tile_max = red[0usize];

        // ── tile-local weights + sum (0 for masked keys — the guard, not
        //    the exp, keeps them out) ──
        let mut my_exp = f32::new(0.0f32);
        if j < n_seed {
            my_exp = f32::exp(s - tile_max);
        }
        w[tid as usize] = my_exp;
        red[tid as usize] = my_exp;
        sync_cube();
        let mut off = 64u32;
        while off >= 1u32 {
            if tid < off {
                red[tid as usize] = red[tid as usize] + red[(tid + off) as usize];
            }
            sync_cube();
            off /= 2u32;
        }
        let tile_sum = red[0usize];

        // ── weighted V accumulation for MY output dimension (tid) ──
        let mut tile_val = f32::new(0.0f32);
        let lane_count = n_seed - j0;
        let mut bound = hd;
        if lane_count < hd {
            bound = lane_count;
        }
        let v_base = (kvh * hd) as usize;
        let mut jj = 0u32;
        while jj < bound {
            tile_val += w[jj as usize]
                * seed_kv[(seed_half + (j0 + jj) * kvd) as usize + v_base + tid as usize];
            jj += 1u32;
        }

        // ── online softmax update (the family's correction form) ──
        let mut new_max = m;
        if tile_max > new_max {
            new_max = tile_max;
        }
        let prev_corr = f32::exp(m - new_max);
        let curr_corr = f32::exp(tile_max - new_max);
        ssum = ssum * prev_corr + tile_sum * curr_corr;
        acc = acc * prev_corr + tile_val * curr_corr;
        m = new_max;

        sync_cube();
        j0 += 128u32;
    }

    // ── phase 1: the pass's own window [lo, hi) ──
    j0 = lo;
    while j0 < hi {
        // ── score: thread tid owns key j0 + tid (full smem-Q dot) ──
        let j = j0 + tid;
        let mut s = big_neg;
        if j < hi && j < n_pos {
            let kro = (j * kvd + kvh * hd) as usize;
            let mut dot = f32::new(0.0f32);
            let mut t = 0u32;
            while t < hd {
                dot += q_smem[t as usize] * kv[kro + t as usize];
                t += 1u32;
            }
            s = dot * scale;
        }

        // ── tile max ──
        red[tid as usize] = s;
        sync_cube();
        let mut off = 64u32;
        while off >= 1u32 {
            if tid < off {
                let other = red[(tid + off) as usize];
                if other > red[tid as usize] {
                    red[tid as usize] = other;
                }
            }
            sync_cube();
            off /= 2u32;
        }
        let tile_max = red[0usize];

        // ── tile-local weights + sum (0 for masked keys — the guard, not
        //    the exp, keeps them out) ──
        let mut my_exp = f32::new(0.0f32);
        if j < hi && j < n_pos {
            my_exp = f32::exp(s - tile_max);
        }
        w[tid as usize] = my_exp;
        red[tid as usize] = my_exp;
        sync_cube();
        let mut off = 64u32;
        while off >= 1u32 {
            if tid < off {
                red[tid as usize] = red[tid as usize] + red[(tid + off) as usize];
            }
            sync_cube();
            off /= 2u32;
        }
        let tile_sum = red[0usize];

        // ── weighted V accumulation for MY output dimension (tid) ──
        // Keys j0..j0+bound, bound = min(hi − j0, hd); hi ≤ n_pos (the
        // launcher's window law) keeps the V reads in range.
        let mut tile_val = f32::new(0.0f32);
        let lane_count = hi - j0;
        let mut bound = hd;
        if lane_count < hd {
            bound = lane_count;
        }
        let v_base = (kvh * hd) as usize;
        let mut jj = 0u32;
        while jj < bound {
            tile_val +=
                w[jj as usize] * kv[(kv_half + (j0 + jj) * kvd) as usize + v_base + tid as usize];
            jj += 1u32;
        }

        // ── online softmax update (the family's correction form) ──
        let mut new_max = m;
        if tile_max > new_max {
            new_max = tile_max;
        }
        let prev_corr = f32::exp(m - new_max);
        let curr_corr = f32::exp(tile_max - new_max);
        ssum = ssum * prev_corr + tile_sum * curr_corr;
        acc = acc * prev_corr + tile_val * curr_corr;
        m = new_max;

        sync_cube();
        j0 += 128u32;
    }

    out[((qi * n_head + h) * hd + tid) as usize] = acc / ssum;
}

// ── launchers ───────────────────────────────────────────────────────

/// Parameters for [`EdlmAttnMultiCubeCL::launch`] — the attention geometry of
/// one dispatch.
#[cfg(feature = "edlm_gpu")]
#[derive(Clone, Copy, Debug)]
pub struct EdlmAttnMultiParams {
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// Key positions in the pass's own combined `[keys | values]` buffer.
    pub n_positions: usize,
    /// Seed (state-prefix) key count in `seed_kv` — the device-resident KV
    /// carry. `0` = no seed (the state pass); the caller then passes a
    /// 1-element dummy the kernel never reads.
    pub n_seed: usize,
    /// Query rows.
    pub seq_q: usize,
    /// `1 / sqrt(head_dim)` — the caller computes it exactly like the CPU
    /// path (`1.0 / (hd as f32).sqrt()`).
    pub scale: f32,
}

/// Launcher for [`edlm_attn_multi_f32`].
#[cfg(feature = "edlm_gpu")]
pub struct EdlmAttnMultiCubeCL;

#[cfg(feature = "edlm_gpu")]
impl EdlmAttnMultiCubeCL {
    /// Launch the multi-query attention: `out[sq, n_head·hd]` from
    /// `q[sq, n_head·hd]`, the pass's own `kv[2·n_positions·kvd]`
    /// (`[keys | values]`), the device-resident seed KV
    /// `seed_kv[2·n_seed·kvd]` (1-element dummy when `n_seed == 0`), and
    /// per-query own windows `t_start/t_end[sq]` (query `i` attends the seed
    /// segment `[0, n_seed)` then own keys `[t_start(i), t_end(i))`).
    pub fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q: Handle,
        kv: Handle,
        seed_kv: Handle,
        t_start: Handle,
        t_end: Handle,
        out: Handle,
        p: &EdlmAttnMultiParams,
    ) {
        assert_eq!(
            p.head_dim, 128,
            "the attention cube is head_dim 128 (the 8B release shape); \
             a different head_dim needs a kernel variant"
        );
        assert!(
            p.n_head > 0 && p.n_kv_head > 0,
            "head counts must be positive"
        );
        assert_eq!(
            p.n_head % p.n_kv_head,
            0,
            "GQA mapping needs n_head % n_kv_head == 0"
        );
        assert!(p.seq_q > 0 && p.n_positions > 0, "empty attention dispatch");
        assert!(
            p.scale.is_finite() && p.scale > 0.0,
            "scale must be positive"
        );
        let kvd = p.n_kv_head * p.head_dim;
        // The kernel derives n_pos from the bound kv buffer (the `.issues/515`
        // class guard); the other shapes from the params. The windows' own
        // bounds (`t_end ≤ n_positions`, `t_start < t_end`) are the caller's
        // law — the model pass constructs them from the visibility table, the
        // tests pin the boundary shape.
        assert_binding_derives_units(&kv, 2 * kvd, p.n_positions, "EdlmAttnMulti kv");
        assert_binding_derives_units(&q, p.n_head * p.head_dim, p.seq_q, "EdlmAttnMulti q");
        if p.n_seed > 0 {
            assert_binding_derives_units(
                &seed_kv,
                2 * kvd,
                p.n_seed,
                "EdlmAttnMulti seed_kv",
            );
        } else {
            assert_binding_derives_units(&seed_kv, 1, 1, "EdlmAttnMulti seed dummy");
        }
        assert_binding_derives_units(&t_start, 1, p.seq_q, "EdlmAttnMulti t_start");
        assert_binding_derives_units(&t_end, 1, p.seq_q, "EdlmAttnMulti t_end");
        assert_binding_derives_units(&out, p.n_head * p.head_dim, p.seq_q, "EdlmAttnMulti out");

        let params_h = create_f32(
            client,
            &[
                p.n_head as f32,
                p.n_kv_head as f32,
                p.head_dim as f32,
                p.scale,
                p.n_seed as f32,
            ],
        );

        // SAFETY: buffer sizes asserted above. The dummy seed binds ONE
        // element (a zero-count raw-part binding is not a valid slice; the
        // kernel never reads it at n_seed == 0).
        let seed_count = if p.n_seed > 0 { 2 * p.n_seed * kvd } else { 1 };
        unsafe {
            edlm_attn_multi_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(p.n_head as u32, p.seq_q as u32, 1),
                CubeDim::new_1d(128),
                BufferArg::from_raw_parts(q, p.seq_q * p.n_head * p.head_dim),
                BufferArg::from_raw_parts(kv, 2 * p.n_positions * kvd),
                BufferArg::from_raw_parts(seed_kv, seed_count),
                BufferArg::from_raw_parts(t_start, p.seq_q),
                BufferArg::from_raw_parts(t_end, p.seq_q),
                BufferArg::from_raw_parts(params_h, 5),
                BufferArg::from_raw_parts(out, p.seq_q * p.n_head * p.head_dim),
            );
        }
    }
}

/// Launcher for [`edlm_rmsnorm_rows_f32`].
#[cfg(feature = "edlm_gpu")]
pub struct EdlmRmsNormRowsCubeCL;

#[cfg(feature = "edlm_gpu")]
impl EdlmRmsNormRowsCubeCL {
    /// Launch the multi-row RMSNorm: `out[rows, n]` from `x[rows, n]` and
    /// `gamma[n]` at `eps`.
    pub fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x: Handle,
        gamma: Handle,
        out: Handle,
        rows: usize,
        n: usize,
        eps: f32,
    ) {
        assert!(rows > 0 && n > 0, "rmsnorm dims must be positive");
        assert_binding_derives_units(&x, n, rows, "EdlmRmsNormRows x");
        assert_binding_derives_units(&gamma, n, 1, "EdlmRmsNormRows gamma");
        assert_binding_derives_units(&out, n, rows, "EdlmRmsNormRows out");
        let params_h = create_f32(client, &[eps]);

        // SAFETY: buffer sizes asserted above.
        unsafe {
            edlm_rmsnorm_rows_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x, rows * n),
                BufferArg::from_raw_parts(gamma, n),
                BufferArg::from_raw_parts(params_h, 1),
                BufferArg::from_raw_parts(out, rows * n),
            );
        }
    }
}

/// The fused QKV fold's shape bundle (the launcher's assertions + params).
#[cfg(feature = "edlm_gpu")]
pub struct EdlmQkvFoldParams {
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// The rmsnorm eps (f32 — the host path casts the f64 config value the
    /// same way).
    pub eps: f32,
    /// Key positions in the pass's own combined `[keys | values]` buffer.
    pub n_positions: usize,
    /// Query rows.
    pub seq_q: usize,
}

/// Launcher for [`edlm_qkv_fold_f32`].
#[cfg(feature = "edlm_gpu")]
pub struct EdlmQkvFoldCubeCL;

#[cfg(feature = "edlm_gpu")]
impl EdlmQkvFoldCubeCL {
    /// Launch the fused fold: from the raw QKV GEMM output `qkv[seq_q, lq]`
    /// (lq = `n_head·hd + 2·n_kv_head·hd`), write `q_rope[seq_q, n_head·hd]`
    /// (per-head qk-norm + RoPE) and the combined `kv[2·n_positions, kvd]`
    /// in `[keys | values]` layout — query `g`'s k/v land at row `seg[g]`
    /// (its row's own-buffer segment base + its in-row offset), values at
    /// `n_positions + seg[g]`. `pos[g]` drives the RoPE angle; `freq` is the
    /// model's precomputed table (`hd/2` entries).
    pub fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        qkv: Handle,
        q_norm: Handle,
        k_norm: Handle,
        pos: Handle,
        seg: Handle,
        freq: Handle,
        q_rope: Handle,
        kv: Handle,
        p: &EdlmQkvFoldParams,
    ) {
        assert_eq!(
            p.head_dim, 128,
            "the fold cube is head_dim 128 (the 8B release shape); \
             a different head_dim needs a kernel variant"
        );
        assert!(p.n_head > 0 && p.n_kv_head > 0, "head counts must be positive");
        let kvd = p.n_kv_head * p.head_dim;
        let q_dim = p.n_head * p.head_dim;
        let lq = q_dim + 2 * kvd;
        let units = p.n_head + 2 * p.n_kv_head;
        assert!(p.seq_q > 0 && p.n_positions > 0, "empty fold dispatch");
        assert_binding_derives_units(&qkv, lq, p.seq_q, "EdlmQkvFold qkv");
        assert_binding_derives_units(&q_norm, p.head_dim, 1, "EdlmQkvFold q_norm");
        assert_binding_derives_units(&k_norm, p.head_dim, 1, "EdlmQkvFold k_norm");
        assert_binding_derives_units(&pos, 1, p.seq_q, "EdlmQkvFold pos");
        assert_binding_derives_units(&seg, 1, p.seq_q, "EdlmQkvFold seg");
        assert_binding_derives_units(&freq, p.head_dim / 2, 1, "EdlmQkvFold freq");
        assert_binding_derives_units(&q_rope, q_dim, p.seq_q, "EdlmQkvFold q_rope");
        assert_binding_derives_units(&kv, 2 * kvd, p.n_positions, "EdlmQkvFold kv");

        let params_h = create_f32(
            client,
            &[
                p.n_head as f32,
                p.n_kv_head as f32,
                p.head_dim as f32,
                p.eps,
                p.n_positions as f32,
            ],
        );

        // SAFETY: buffer sizes asserted above.
        unsafe {
            edlm_qkv_fold_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(units as u32, p.seq_q as u32, 1),
                CubeDim::new_1d(128),
                BufferArg::from_raw_parts(qkv, p.seq_q * lq),
                BufferArg::from_raw_parts(q_norm, p.head_dim),
                BufferArg::from_raw_parts(k_norm, p.head_dim),
                BufferArg::from_raw_parts(pos, p.seq_q),
                BufferArg::from_raw_parts(seg, p.seq_q),
                BufferArg::from_raw_parts(freq, p.head_dim / 2),
                BufferArg::from_raw_parts(params_h, 5),
                BufferArg::from_raw_parts(q_rope, p.seq_q * q_dim),
                BufferArg::from_raw_parts(kv, 2 * p.n_positions * kvd),
            );
        }
    }
}

// ── the model ───────────────────────────────────────────────────────

/// One layer's GPU-resident weights (f16) + the host-side qk-norm gammas the
/// hybrid pipeline applies between readbacks.
#[cfg(feature = "edlm_gpu")]
struct EdlmGpuLayerWeights {
    /// `[(n_head + 2·n_kv_head)·hd, n_embd]` f16 — the q|k|v row-concat of
    /// attn_q/attn_k/attn_v (one GEMM, one readback).
    qkv: Handle,
    /// `[n_embd, n_head·hd]` f16 (attn_output).
    wo: Handle,
    /// `[2·mlp_hidden, n_embd]` f16 — the gate|up row-concat (one GEMM, one
    /// readback).
    gateup: Handle,
    /// `[n_embd, mlp_hidden]` f16 (ffn_down).
    down: Handle,
    /// `[n_embd]` f32 (attn_norm — the GPU rmsnorm's gamma).
    attn_norm: Handle,
    /// `[n_embd]` f32 (ffn_norm — the GPU rmsnorm's gamma).
    post_attn_norm: Handle,
    /// `[head_dim]` f32 host copy (attn_q_norm — the host-path qk-norm
    /// between readbacks; the device fold reads the handle below).
    q_norm: Vec<f32>,
    /// `[head_dim]` f32 host copy (attn_k_norm).
    k_norm: Vec<f32>,
    /// `[head_dim]` f32 device (attn_q_norm — the fold kernel's gamma).
    q_norm_d: Handle,
    /// `[head_dim]` f32 device (attn_k_norm).
    k_norm_d: Handle,
}

/// The GPU-resident state cache: the T7 host cache (parity inspection + the
/// pointer head's state-side reads) plus each layer's combined
/// `[keys | values]` device buffer, `state_len` keys each — the branch pass
/// attends these directly (the KV carry). At the 8B release shape this is
/// ~3.1 MB × n_layer of VRAM per 384-token state.
#[cfg(feature = "edlm_gpu")]
struct EdlmGpuStateKv {
    host: EdlmStateKv,
    /// Per-layer combined `[keys | values]` device buffers (f32), in layer
    /// order — the seed segment [`EdlmAttnMultiCubeCL`] reads.
    device_kv: Vec<Handle>,
}

/// One pass segment: an independent `(ids, pos)` token run. The batched
/// multi-branch pass concatenates the rows; each row's keys occupy its own
/// own-buffer segment and its queries attend only the seed + that segment.
#[cfg(feature = "edlm_gpu")]
struct PassRow<'a> {
    ids: &'a [usize],
    pos: &'a [usize],
}

/// The shared pass prologue's output — the geometry BOTH sync postures
/// consume ([`EdlmGpuModel::pass_geometry`]; one copy — divergent geometry
/// between the fold and host arms would void the A/B).
#[cfg(feature = "edlm_gpu")]
struct PassGeometry {
    sq_total: usize,
    own_total: usize,
    seg: Vec<usize>,
    t_start: Vec<u32>,
    t_end: Vec<u32>,
    attn_params: EdlmAttnMultiParams,
}

/// The batched pass's outputs: (concatenated raw hiddens `[Σsq · n_embd]`,
/// the state pass's captured per-layer K/V — empty on branch passes, the
/// per-layer own combined `[keys | values]` device handles — the state pass
/// retains these as the KV carry).
#[cfg(feature = "edlm_gpu")]
type PassOutputs = (Vec<f32>, Vec<EdlmLayerKv>, Vec<Handle>);

/// The GPU eDLM model: f16-resident weights on the CubeCL runtime + the
/// GPU-resident state cache (`EdlmGpuStateKv`: the T7 host K/V + the device
/// buffers), serving the SAME API shape the CPU streaming lane does
/// (`state_prefill` → `forward_branches`).
///
/// Construct via [`EdlmGpuModel::open`] (GGUF, one layer dequanted at a
/// time) or [`EdlmGpuModel::from_weights`] (in-memory — the parity-test
/// lane). Both share the f16 upload conversion.
#[cfg(feature = "edlm_gpu")]
pub struct EdlmGpuModel {
    client: ComputeClient<ActiveRuntime>,
    pub config: Config,
    /// Host embedding table (rows are gathered CPU-side and uploaded — the
    /// full table never goes VRAM; only the sequence's rows are needed).
    wte: Vec<f32>,
    final_norm: Vec<f32>,
    pub pointer: Option<EdlmPointerHead>,
    layers: Vec<EdlmGpuLayerWeights>,
    freq: RopeFreqTable,
    /// The precomputed rope table ON DEVICE (`head_dim/2` f32) — the fold
    /// kernel reads it per query (uploaded once at construction).
    freq_d: Handle,
    scale: f32,
    /// The cached state prefix — the T7 host K/V + the device-resident
    /// per-layer buffers the branch pass attends directly (the KV carry:
    /// no host copy, no per-row re-upload). Filled by
    /// [`EdlmGpuModel::state_prefill`], consumed by
    /// [`EdlmGpuModel::forward_branches`].
    state: Option<EdlmGpuStateKv>,
    /// The GEMM posture: [`GemmPosture::Cmma`] routes the four per-layer
    /// projections through the cooperative-matrix (tensor-core) kernels,
    /// [`GemmPosture::Scalar`] keeps the scalar tiled kernel. Resolved once
    /// at construction from `EDLM_GPU_CMMA` (`"0"` = scalar; unset/anything
    /// else = the measured default), overridable per-instance by the test
    /// seam.
    posture: GemmPosture,
    /// The sg8 arm (the Bench 706 ladder rung): armed by
    /// `EDLM_GPU_CMMA_SG8=1`, consulted ONLY at the [`GemmPosture::Cmma`]
    /// posture above the [`SG8_MIN_M`] crossover.
    sg8_armed: bool,
    /// The sync posture: device fold (qk-norm/RoPE/SwiGLU/residuals on the
    /// GPU, zero mid-pass readbacks) vs the v1 host small-ops path (four
    /// readbacks per layer — the parity anchor). Resolved once at
    /// construction from `EDLM_GPU_FOLD` (`"0"` = host path; unset = fold,
    /// the measured default), overridable per-instance by [`with_fold`].
    fold: bool,
}

/// The GEMM posture for the lane's four per-layer projections.
#[cfg(feature = "edlm_gpu")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GemmPosture {
    /// The scalar tiled kernel — the kill-switch arm.
    Scalar,
    /// Tensor-core: the v1 (32-thread) kernel below the sg8 crossover, the
    /// sg8 (8-subgroup, 128×64) kernel above it when armed.
    Cmma,
    /// Test/GOAT seam: FORCE the sg8 kernel for every cmma-eligible m.
    /// Never a construction default — the env resolves to [`GemmPosture::Cmma`].
    CmmaSg8,
}

/// Convert a host f32 slice to f16 bytes (round-to-nearest — the documented
/// numerics face of the f16 residency decision).
#[cfg(feature = "edlm_gpu")]
fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let f16v: Vec<half_f16> = if data.len() < 4096 {
        data.iter().map(|&v| half_f16::from_f32(v)).collect()
    } else {
        use rayon::prelude::*;
        data.par_iter().map(|&v| half_f16::from_f32(v)).collect()
    };
    bytemuck::cast_slice::<half_f16, u8>(&f16v).to_vec()
}

/// The env kill-switch for the tensor-core GEMM posture, read ONCE per
/// process. `EDLM_GPU_CMMA=0` restores the scalar tiled kernel; unset or
/// any other value takes the measured default (CMMA — the GOAT lane's
/// verdict lives in `tests/edlm_matmul_cmma_goat.rs` + the issue row).
#[cfg(feature = "edlm_gpu")]
fn cmma_env_default() -> bool {
    match std::env::var("EDLM_GPU_CMMA") {
        Ok(v) => v != "0",
        Err(_) => true,
    }
}

/// The sg8 arm of the tensor-core posture, read ONCE per process.
/// DEFAULT ON (PROMOTED 2026-10-08 — the GOAT interleaved table, 4090:
/// sg8 ≥ 1.19× v1 at EVERY measured m from 8 to 2048, ≥ 1.40× at 12 of 13
/// cells — no crossover exists, so `SG8_MIN_M` sits at the cmma-eligibility
/// floor; the dead-row waste that made v1 LOSE at m=8 does not bind sg8,
/// whose 8 subgroups stage dead rows in parallel and amortize the B panel
/// per 64 cols). `EDLM_GPU_CMMA_SG8=0` restores the v1 kernel at every m.
#[cfg(feature = "edlm_gpu")]
fn cmma_sg8_env_default() -> bool {
    match std::env::var("EDLM_GPU_CMMA_SG8") {
        Ok(v) => v != "0",
        Err(_) => true,
    }
}

/// The device-fold sync posture, read ONCE per process. DEFAULT ON — the
/// qk-norm/RoPE/SwiGLU small-ops run ON DEVICE (the fused QKV fold + the
/// silu gate + device residual adds), so a pass has ZERO mid-pass readbacks
/// (v1 paid four per layer plus the host round-trips between them; the
/// GOAT evidence: `tests/edlm_fold_goat.rs` + the issue row).
/// `EDLM_GPU_FOLD=0` restores the v1 host small-ops path — the parity
/// anchor arm, still exercised by the fold-vs-host arms test.
#[cfg(feature = "edlm_gpu")]
fn fold_env_default() -> bool {
    match std::env::var("EDLM_GPU_FOLD") {
        Ok(v) => v != "0",
        Err(_) => true,
    }
}

#[cfg(feature = "edlm_gpu")]
impl EdlmGpuModel {
    /// Open an `edlm`-arch GGUF and upload every layer as f16 (one layer
    /// dequanted at a time — the streaming loader's host-footprint law; the
    /// Q8_0 dequant is exact, the f16 conversion is the rounding face).
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let core = EdlmGgufModel::open(path).map_err(|e| e.to_string())?;
        Self::from_core(core)
    }

    /// In-memory constructor — the parity-test lane: the SAME `EdlmWeights`
    /// the CPU oracle runs, uploaded f16.
    pub fn from_weights(weights: &EdlmWeights, config: &Config) -> Result<Self, String> {
        if config.head_dim != 128 {
            return Err(format!(
                "EdlmGpuModel needs head_dim 128 (the attention cube's shape), got {}",
                config.head_dim
            ));
        }
        let ctx = CubeCLContext::new().map_err(|e| format!("CubeCL context: {e:?}"))?;
        let client = ctx.client();

        let layers = weights
            .layers
            .iter()
            .map(|l| {
                let mut qkv = l.base.attn_wq.clone();
                qkv.extend_from_slice(&l.base.attn_wk);
                qkv.extend_from_slice(&l.base.attn_wv);
                let mut gateup = l.base.gate_proj.clone();
                gateup.extend_from_slice(&l.base.up_proj);
                Ok(EdlmGpuLayerWeights {
                    qkv: upload_f16(&client, &qkv)?,
                    wo: upload_f16(&client, &l.base.attn_wo)?,
                    gateup: upload_f16(&client, &gateup)?,
                    down: upload_f16(&client, &l.base.down_proj)?,
                    attn_norm: create_f32(&client, &l.base.input_norm),
                    post_attn_norm: create_f32(&client, &l.base.post_attn_norm),
                    q_norm: l.q_norm.clone(),
                    k_norm: l.k_norm.clone(),
                    q_norm_d: create_f32(&client, &l.q_norm),
                    k_norm_d: create_f32(&client, &l.k_norm),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let freq_d = create_f32(&client, freq.as_slice());
        Ok(Self {
            client,
            config: config.clone(),
            wte: weights.wte.clone(),
            final_norm: weights.final_norm.clone(),
            pointer: weights.pointer.clone(),
            layers,
            freq_d,
            freq,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            state: None,
            posture: if cmma_env_default() {
                GemmPosture::Cmma
            } else {
                GemmPosture::Scalar
            },
            sg8_armed: cmma_sg8_env_default(),
            fold: fold_env_default(),
        })
    }

    fn from_core(core: EdlmGgufModel) -> Result<Self, String> {
        let config = core.config.clone();
        if config.head_dim != 128 {
            return Err(format!(
                "EdlmGpuModel needs head_dim 128 (the attention cube's shape), got {}",
                config.head_dim
            ));
        }
        let ctx = CubeCLContext::new().map_err(|e| format!("CubeCL context: {e:?}"))?;
        let client = ctx.client();

        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let deq = |name: String| -> Result<Vec<f32>, String> {
                core.gguf
                    .dequant_f16_to_f32(&name)
                    .map_err(|e| format!("tensor {name}: {e}"))
            };
            let wq = deq(format!("blk.{i}.attn_q.weight"))?;
            let wk = deq(format!("blk.{i}.attn_k.weight"))?;
            let wv = deq(format!("blk.{i}.attn_v.weight"))?;
            let wo = deq(format!("blk.{i}.attn_output.weight"))?;
            let gate = deq(format!("blk.{i}.ffn_gate.weight"))?;
            let up = deq(format!("blk.{i}.ffn_up.weight"))?;
            let down = deq(format!("blk.{i}.ffn_down.weight"))?;
            let attn_norm = deq(format!("blk.{i}.attn_norm.weight"))?;
            let post_attn_norm = deq(format!("blk.{i}.ffn_norm.weight"))?;
            let q_norm = deq(format!("blk.{i}.attn_q_norm.weight"))?;
            let k_norm = deq(format!("blk.{i}.attn_k_norm.weight"))?;
            if q_norm.len() != config.head_dim || k_norm.len() != config.head_dim {
                return Err(format!(
                    "layer {i}: QK-norm gamma length {}/{} != head_dim {}",
                    q_norm.len(),
                    k_norm.len(),
                    config.head_dim
                ));
            }
            let mut qkv = wq;
            qkv.extend_from_slice(&wk);
            qkv.extend_from_slice(&wv);
            let mut gateup = gate;
            gateup.extend_from_slice(&up);
            layers.push(EdlmGpuLayerWeights {
                qkv: upload_f16(&client, &qkv)?,
                wo: upload_f16(&client, &wo)?,
                gateup: upload_f16(&client, &gateup)?,
                down: upload_f16(&client, &down)?,
                attn_norm: create_f32(&client, &attn_norm),
                post_attn_norm: create_f32(&client, &post_attn_norm),
                q_norm_d: create_f32(&client, &q_norm),
                k_norm_d: create_f32(&client, &k_norm),
                q_norm,
                k_norm,
            });
        }

        let freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        let freq_d = create_f32(&client, freq.as_slice());
        Ok(Self {
            client,
            config,
            wte: core.wte,
            final_norm: core.final_norm,
            pointer: core.pointer,
            layers,
            freq_d,
            freq,
            scale,
            state: None,
            posture: if cmma_env_default() {
                GemmPosture::Cmma
            } else {
                GemmPosture::Scalar
            },
            sg8_armed: cmma_sg8_env_default(),
            fold: fold_env_default(),
        })
    }

    /// The runtime's name — provenance for every latency figure quoted
    /// beside this lane (box state is part of the claim).
    pub fn runtime_name(&self) -> &'static str {
        ActiveRuntime::name(&self.client)
    }

    /// The GEMM posture this instance dispatches (provenance for parity
    /// rows: the postures are tolerance-equivalent, never identical).
    pub fn matmul_posture(&self) -> &'static str {
        match self.posture {
            GemmPosture::Scalar => "scalar",
            GemmPosture::Cmma => "cmma",
            GemmPosture::CmmaSg8 => "cmma-sg8",
        }
    }

    /// The sync posture this instance dispatches (provenance for latency
    /// rows: `true` = device fold, `false` = the v1 host small-ops path).
    pub fn fold_enabled(&self) -> bool {
        self.fold
    }

    /// A/B seam: pin the sync posture explicitly (the env default is
    /// process-global; the GOAT lane needs BOTH arms in one process).
    pub fn with_fold(mut self, fold: bool) -> Self {
        self.fold = fold;
        self
    }

    /// Test seam: pin the GEMM posture explicitly (the env default is
    /// process-global; parity tests need EVERY posture in one process).
    #[cfg(test)]
    fn with_posture(mut self, posture: GemmPosture) -> Self {
        self.posture = posture;
        self
    }

    /// The sg8 floor: sg8 measured ≥ 1.19× v1 at EVERY m ≥ 8 (the GOAT
    /// interleaved table, 4090, 2026-10-08 — no crossover below the
    /// cmma-eligibility floor of 16), so the pin is the eligibility floor
    /// itself. Recorded in the issue row.
    #[cfg(feature = "edlm_gpu")]
    const SG8_MIN_M: usize = 16;

    /// The one dispatch site for the lane's four per-layer projections.
    /// Shape law (the GOAT lane's interleaved medians, wgpu-spirv, 4090):
    /// cmma loses at M=8 (0.60× — staging ALU over dead tile rows) and wins
    /// from M≈16 (1.09×) upward (1.51× at 18, 3.4× at 87, 8× at 512) — so
    /// short branch rows stay scalar and everything prefill-class goes
    /// tensor-core. `EDLM_GPU_CMMA=0` forces scalar for ALL m.
    fn matmul_f16b(
        &self,
        a: Handle,
        b: Handle,
        out: Handle,
        m: usize,
        n: usize,
        p: usize,
    ) {
        if self.posture != GemmPosture::Scalar && m >= 16 {
            let want_sg8 = match self.posture {
                GemmPosture::CmmaSg8 => true,
                GemmPosture::Cmma => self.sg8_armed && m >= Self::SG8_MIN_M,
                GemmPosture::Scalar => false,
            };
            if want_sg8 {
                crate::MatmulF16bCmmaCubeCL::launch_sg8::<ActiveRuntime>(
                    &self.client, a, b, out, m, n, p,
                );
            } else {
                crate::MatmulF16bCmmaCubeCL::launch::<ActiveRuntime>(&self.client, a, b, out, m, n, p);
            }
        } else {
            MatmulF16bCubeCL::launch::<ActiveRuntime>(&self.client, a, b, out, m, n, p);
        }
    }

    /// Release fully-free GPU pool pages back to the driver (the Issue-712
    /// law: freed slices stay committed inside their pages otherwise).
    pub fn memory_cleanup(&self) {
        self.client.memory_cleanup();
    }

    /// Run the state tokens only and capture their per-layer K/V + the
    /// final-normed state hiddens — the T7 prefill, on the GPU, with the
    /// per-layer combined `[keys | values]` buffers RETAINED on the device
    /// (the KV carry). The state's own eligibility is `branch_mask` over an
    /// all-state segment: causal, widened to state↔state bidirectional when
    /// `state_bidir`.
    pub fn state_prefill(
        &mut self,
        state_ids: &[usize],
        state_pos: &[usize],
        state_bidir: bool,
    ) -> Result<(), String> {
        let sl = state_ids.len();
        assert_eq!(state_pos.len(), sl, "state_pos must match state_ids");
        assert!(
            sl > 0 && sl <= self.config.block_size,
            "state len {sl} out of range"
        );
        let pass = [PassRow {
            ids: state_ids,
            pos: state_pos,
        }];
        let (h, layers, device_kv) = self.forward_rows_batched(&pass, None, state_bidir)?;
        let mut hidden = h;
        let n = self.config.n_embd;
        for chunk in hidden.chunks_exact_mut(n) {
            rmsnorm_with_gamma_eps(chunk, &self.final_norm, self.config.rms_norm_eps);
        }
        self.state = Some(EdlmGpuStateKv {
            host: EdlmStateKv {
                state_len: sl,
                kvd: kv_dim(&self.config),
                state_ids: state_ids.to_vec(),
                layers,
                hidden,
            },
            device_kv,
        });
        Ok(())
    }

    /// The cached state prefix — parity inspection + the pointer head's
    /// state-side read (same shape the CPU lane returns).
    pub fn state_kv(&self) -> Result<&EdlmStateKv, String> {
        self.state
            .as_ref()
            .map(|s| &s.host)
            .ok_or_else(|| "no state cache: call state_prefill first".to_string())
    }

    /// Branch rows continuing the cached state prefix — the T7 continuation
    /// on the GPU, ALL ROWS IN ONE BATCHED PASS (the multi-branch follow-up:
    /// the four per-layer projections run at `m = Σ branch_len` once, not
    /// once per row, and the seed KV is attended device-resident). Output =
    /// the CPU `forward_edlm_branches_cached` shape: per-row
    /// `[(state hiddens | branch hiddens)]` post-final-norm, the pointer head
    /// reads markers from it directly.
    pub fn forward_branches(
        &self,
        enc: &PackedEncoding,
        rows: &[BranchRow],
    ) -> Result<Vec<Vec<f32>>, String> {
        let cache = self
            .state
            .as_ref()
            .ok_or_else(|| "no state cache: call state_prefill first".to_string())?;
        cache
            .host
            .check_matches(&self.config, &enc.ids[..enc.state_len])
            .map_err(|e| e.to_string())?;
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let n = self.config.n_embd;
        let sl = enc.state_len;
        let pass: Vec<PassRow<'_>> = rows
            .iter()
            .map(|row| PassRow {
                ids: &enc.ids[row.start..row.end],
                pos: &enc.pos[row.start..row.end],
            })
            .collect();
        let (h, _, _) = self.forward_rows_batched(&pass, Some((&cache.device_kv, sl)), false)?;
        let mut out = Vec::with_capacity(rows.len());
        let mut off = 0usize;
        for row in rows {
            let bl = row.branch_len();
            let mut hidden = h[off * n..(off + bl) * n].to_vec();
            off += bl;
            for chunk in hidden.chunks_exact_mut(n) {
                rmsnorm_with_gamma_eps(chunk, &self.final_norm, self.config.rms_norm_eps);
            }
            let mut full = Vec::with_capacity((sl + bl) * n);
            full.extend_from_slice(&cache.host.hidden);
            full.extend_from_slice(&hidden);
            out.push(full);
        }
        Ok(out)
    }

    /// One batched pass over independent row segments with an optional
    /// device-resident seed KV. Returns the concatenated raw (pre-final-norm)
    /// hiddens `[Σsq · n_embd]`, the per-layer own K/V when this is the state
    /// pass (seed `None` — the capture), and the per-layer own combined
    /// `[keys | values]` device handles (the state pass retains them as the
    /// KV carry; branch passes drop them).
    ///
    /// Eligibility (the module doc's segment table):
    /// - seed present (branch continuation): query `j` of row `r` attends
    ///   ALL seed keys + own `[seg[r], seg[r] + j + 1)` — state keys precede
    ///   every branch query, causal within the branch, rows isolated;
    /// - seed absent (the state pass): every query attends all own keys when
    ///   `state_bidir`, else own `[0, g + 1)` (causal).
    ///
    /// The sync posture dispatches on [`Self::fold`]: the device fold (the
    /// fused QKV fold + silu gate + device residual adds — ZERO mid-pass
    /// readbacks) or the v1 host small-ops path (four readbacks per layer —
    /// the parity anchor; both arms consume ONE [`PassGeometry`]).
    fn forward_rows_batched(
        &self,
        rows: &[PassRow<'_>],
        seed: Option<(&[Handle], usize)>,
        state_bidir: bool,
    ) -> Result<PassOutputs, String> {
        let geo = self.pass_geometry(rows, seed, state_bidir)?;
        if self.fold {
            self.forward_rows_batched_device(rows, seed, &geo)
        } else {
            self.forward_rows_batched_host(rows, seed, &geo)
        }
    }

    /// The shared pass prologue — validation, own-segment offsets, the
    /// visibility windows, and the attention params. ONE copy consumed by
    /// BOTH sync postures: if the two arms ever constructed different
    /// geometry, the fold-vs-host A/B would be meaningless.
    fn pass_geometry(
        &self,
        rows: &[PassRow<'_>],
        seed: Option<(&[Handle], usize)>,
        state_bidir: bool,
    ) -> Result<PassGeometry, String> {
        let config = &self.config;
        let n_kv = config.n_kv_head;
        let hd = config.head_dim;
        let n_layer = config.n_layer;
        let sq_total: usize = rows.iter().map(|r| r.ids.len()).sum();
        assert!(sq_total > 0, "empty pass");
        for (ri, r) in rows.iter().enumerate() {
            assert_eq!(r.ids.len(), r.pos.len(), "row {ri}: pos must match ids");
            assert!(
                !r.ids.is_empty() && r.ids.len() <= config.block_size,
                "row {ri} len {} out of range",
                r.ids.len()
            );
        }

        let (seed_layers, sl) = match seed {
            Some((handles, len)) => (Some(handles), len),
            None => (None, 0usize),
        };
        if let Some(handles) = seed_layers {
            assert_eq!(handles.len(), n_layer, "seed layer count drift");
        }
        let has_seed = seed_layers.is_some();

        // Own-segment offsets: row r's tokens are own keys [seg[r], seg[r+1]).
        let mut seg = Vec::with_capacity(rows.len() + 1);
        let mut acc = 0usize;
        for r in rows {
            seg.push(acc);
            acc += r.ids.len();
        }
        seg.push(acc);
        let own_total = acc;

        // The visibility windows (own-buffer coordinates) + the seed length
        // (the segment table in this fn's doc).
        let mut t_start = vec![0u32; sq_total];
        let mut t_end = vec![0u32; sq_total];
        {
            let mut g = 0usize;
            for (ri, r) in rows.iter().enumerate() {
                for j in 0..r.ids.len() {
                    match (has_seed, state_bidir) {
                        (true, _) => {
                            t_start[g] = seg[ri] as u32;
                            t_end[g] = (seg[ri] + j + 1) as u32;
                        }
                        (false, true) => {
                            t_start[g] = 0;
                            t_end[g] = own_total as u32;
                        }
                        (false, false) => {
                            t_start[g] = 0;
                            t_end[g] = (seg[ri] + j + 1) as u32;
                        }
                    }
                    g += 1;
                }
            }
        }

        let attn_params = EdlmAttnMultiParams {
            n_head: config.n_head,
            n_kv_head: n_kv,
            head_dim: hd,
            n_positions: own_total,
            n_seed: sl,
            seq_q: sq_total,
            scale: self.scale,
        };
        Ok(PassGeometry {
            sq_total,
            own_total,
            seg,
            t_start,
            t_end,
            attn_params,
        })
    }

    /// The v1 sync posture (the parity anchor): GEMMs/norms/attention on the
    /// GPU, the small-ops (split, qk-norm, RoPE, SwiGLU, residuals) on the
    /// host between FOUR per-layer readbacks. `EDLM_GPU_FOLD=0` selects this
    /// arm; the fold-vs-host arms test keeps it exercised.
    fn forward_rows_batched_host(
        &self,
        rows: &[PassRow<'_>],
        seed: Option<(&[Handle], usize)>,
        geo: &PassGeometry,
    ) -> Result<PassOutputs, String> {
        let config = &self.config;
        let n = config.n_embd;
        let hd = config.head_dim;
        let q_dim = config.n_head * hd;
        let kvd = kv_dim(config);
        let n_kv = config.n_kv_head;
        let mlp = config.mlp_hidden;
        let n_layer = config.n_layer;
        let eps = config.rms_norm_eps as f32;
        let sq_total = geo.sq_total;
        let own_total = geo.own_total;
        let seg = &geo.seg;
        let lq = q_dim + 2 * kvd;
        let t_start = &geo.t_start;
        let t_end = &geo.t_end;
        let attn_params = &geo.attn_params;
        let (seed_layers, _sl) = match seed {
            Some((handles, len)) => (Some(handles), len),
            None => (None, 0usize),
        };

        // Embedding rows on the host (no wte upload — only these rows exist
        // on the device).
        let mut h = vec![0.0f32; sq_total * n];
        let mut g = 0usize;
        for r in rows {
            for &id in r.ids {
                let off = id * n;
                h[g * n..(g + 1) * n].copy_from_slice(&self.wte[off..off + n]);
                g += 1;
            }
        }

        // Reused staging (allocated once per pass, sized to the pass).
        let mut q_rope = vec![0.0f32; sq_total * q_dim];
        let mut keys = vec![0.0f32; own_total * kvd];
        let mut values = vec![0.0f32; own_total * kvd];
        let mut kv_capture: Vec<EdlmLayerKv> = Vec::new();
        let mut own_handles: Vec<Handle> = Vec::with_capacity(n_layer);
        let capture = seed_layers.is_none();

        for li in 0..n_layer {
            let lw = &self.layers[li];

            // ── attention block ──
            let xr = h.clone();
            let h_h = create_f32(&self.client, &h);
            let hn1 = self.client.empty(sq_total * n * core::mem::size_of::<f32>());
            EdlmRmsNormRowsCubeCL::launch::<ActiveRuntime>(
                &self.client,
                h_h,
                lw.attn_norm.clone(),
                hn1.clone(),
                sq_total,
                n,
                eps,
            );
            let qkv_out = self.client.empty(sq_total * lq * core::mem::size_of::<f32>());
            self.matmul_f16b(hn1, lw.qkv.clone(), qkv_out.clone(), sq_total, n, lq);
            let qkv_h = read_f32(&self.client, qkv_out).map_err(|e| e.to_string())?;

            // Host small-ops: split, per-head qk-norm, RoPE — the EXACT core
            // helpers the CPU path runs (parity carries; no re-derivation).
            // Row r's K/V land at its own-buffer segment [seg[r], ..).
            let mut g = 0usize;
            for (ri, r) in rows.iter().enumerate() {
                for (j, &p) in r.pos.iter().enumerate() {
                    let row = &qkv_h[g * lq..(g + 1) * lq];
                    let mut q_row = row[..q_dim].to_vec();
                    qk_norm_inplace(
                        &mut q_row,
                        &lw.q_norm,
                        config.n_head,
                        hd,
                        config.rms_norm_eps,
                    );
                    apply_rope_with_freq(&mut q_row, &mut [], p, hd, self.freq.as_slice());
                    q_rope[g * q_dim..(g + 1) * q_dim].copy_from_slice(&q_row);

                    let mut k_row = row[q_dim..q_dim + kvd].to_vec();
                    qk_norm_inplace(&mut k_row, &lw.k_norm, n_kv, hd, config.rms_norm_eps);
                    apply_rope_with_freq(&mut k_row, &mut [], p, hd, self.freq.as_slice());
                    let ko = (seg[ri] + j) * kvd;
                    keys[ko..ko + kvd].copy_from_slice(&k_row);

                    // V: untouched by qk-norm and RoPE.
                    values[ko..ko + kvd].copy_from_slice(&row[q_dim + kvd..lq]);
                    g += 1;
                }
            }

            let mut kv_combined = Vec::with_capacity(2 * own_total * kvd);
            kv_combined.extend_from_slice(&keys);
            kv_combined.extend_from_slice(&values);
            let kv_h = create_f32(&self.client, &kv_combined);
            let seed_h = match seed_layers {
                Some(handles) => handles[li].clone(),
                None => self.client.empty(core::mem::size_of::<f32>()),
            };
            let q_h = create_f32(&self.client, &q_rope);
            let ts_h = create_u32(&self.client, t_start);
            let te_h = create_u32(&self.client, t_end);
            let attn_h = self.client.empty(sq_total * q_dim * core::mem::size_of::<f32>());
            EdlmAttnMultiCubeCL::launch::<ActiveRuntime>(
                &self.client,
                q_h,
                kv_h.clone(),
                seed_h,
                ts_h,
                te_h,
                attn_h.clone(),
                attn_params,
            );

            let wo_out = self.client.empty(sq_total * n * core::mem::size_of::<f32>());
            self.matmul_f16b(attn_h, lw.wo.clone(), wo_out.clone(), sq_total, q_dim, n);
            let wo_h = read_f32(&self.client, wo_out).map_err(|e| e.to_string())?;
            for i in 0..sq_total * n {
                h[i] = xr[i] + wo_h[i];
            }

            // ── MLP block ──
            let h2_h = create_f32(&self.client, &h);
            let hn2 = self.client.empty(sq_total * n * core::mem::size_of::<f32>());
            EdlmRmsNormRowsCubeCL::launch::<ActiveRuntime>(
                &self.client,
                h2_h,
                lw.post_attn_norm.clone(),
                hn2.clone(),
                sq_total,
                n,
                eps,
            );
            let gu_out = self.client.empty(sq_total * 2 * mlp * core::mem::size_of::<f32>());
            self.matmul_f16b(hn2, lw.gateup.clone(), gu_out.clone(), sq_total, n, 2 * mlp);
            let gu = read_f32(&self.client, gu_out).map_err(|e| e.to_string())?;
            let mut mlp_in = vec![0.0f32; sq_total * mlp];
            for r in 0..sq_total {
                let row = &gu[r * 2 * mlp..(r + 1) * 2 * mlp];
                swiglu(
                    &mut mlp_in[r * mlp..(r + 1) * mlp],
                    &row[..mlp],
                    &row[mlp..2 * mlp],
                );
            }
            let mi_h = create_f32(&self.client, &mlp_in);
            let down_out = self.client.empty(sq_total * n * core::mem::size_of::<f32>());
            self.matmul_f16b(mi_h, lw.down.clone(), down_out.clone(), sq_total, mlp, n);
            let d_h = read_f32(&self.client, down_out).map_err(|e| e.to_string())?;
            for i in 0..sq_total * n {
                h[i] += d_h[i];
            }

            if capture {
                kv_capture.push(EdlmLayerKv {
                    k: keys[..own_total * kvd].to_vec(),
                    v: values[..own_total * kvd].to_vec(),
                });
            }
            own_handles.push(kv_h);
        }

        Ok((h, kv_capture, own_handles))
    }

    /// The device-fold sync posture (the measured default): the fused QKV
    /// fold ([`EdlmQkvFoldCubeCL`]) runs split + qk-norm + RoPE ON DEVICE,
    /// the SwiGLU gate and the residual adds ride the elementwise kernels,
    /// and the hiddens update IN PLACE on the device (`h += wo; h += down`)
    /// — a pass has ZERO mid-pass readbacks (v1 paid four per layer plus
    /// the host round-trips between them). The only readbacks are the
    /// pass-final hidden (once) and, on the state pass, one per-layer KV
    /// capture (once per prefill — branch passes attend the device seed
    /// directly).
    fn forward_rows_batched_device(
        &self,
        rows: &[PassRow<'_>],
        seed: Option<(&[Handle], usize)>,
        geo: &PassGeometry,
    ) -> Result<PassOutputs, String> {
        let config = &self.config;
        let n = config.n_embd;
        let hd = config.head_dim;
        let q_dim = config.n_head * hd;
        let kvd = kv_dim(config);
        let mlp = config.mlp_hidden;
        let n_layer = config.n_layer;
        let eps = config.rms_norm_eps as f32;
        let lq = q_dim + 2 * kvd;
        let f32b = core::mem::size_of::<f32>();
        let sq_total = geo.sq_total;
        let own_total = geo.own_total;
        let (seed_layers, _sl) = match seed {
            Some((handles, len)) => (Some(handles), len),
            None => (None, 0usize),
        };
        let capture = seed_layers.is_none();

        // Embedding rows on the host (no wte upload), uploaded ONCE per
        // pass — the device hiddens update in place from here on.
        let mut h = vec![0.0f32; sq_total * n];
        let mut g = 0usize;
        for r in rows {
            for &id in r.ids {
                let off = id * n;
                h[g * n..(g + 1) * n].copy_from_slice(&self.wte[off..off + n]);
                g += 1;
            }
        }
        let h_dev = create_f32(&self.client, &h);

        // Per-query fold inputs: the rope position + the query's own-buffer
        // KEY-ROW write base — seg[ri] + j (the row base PLUS the in-row
        // offset, exactly the host arm's `(seg[ri] + j) · kvd` scatter;
        // the row base alone would stamp every query of a row onto the
        // SAME key row). Tiny u32 uploads, once per pass.
        let mut pos_flat = Vec::with_capacity(sq_total);
        let mut seg_of_q = vec![0u32; sq_total];
        {
            let mut g = 0usize;
            for (ri, r) in rows.iter().enumerate() {
                for (j, &p) in r.pos.iter().enumerate() {
                    pos_flat.push(p as u32);
                    seg_of_q[g] = (geo.seg[ri] + j) as u32;
                    g += 1;
                }
            }
        }
        let pos_d = create_u32(&self.client, &pos_flat);
        let seg_d = create_u32(&self.client, &seg_of_q);
        let ts_d = create_u32(&self.client, &geo.t_start);
        let te_d = create_u32(&self.client, &geo.t_end);
        let dummy_seed = self.client.empty(f32b);

        // Layer-invariant staging, allocated once per pass (the client queue
        // orders every launch, so reuse is race-free).
        let hn1_d = self.client.empty(sq_total * n * f32b);
        let hn2_d = self.client.empty(sq_total * n * f32b);
        let qkv_d = self.client.empty(sq_total * lq * f32b);
        let q_rope_d = self.client.empty(sq_total * q_dim * f32b);
        let attn_d = self.client.empty(sq_total * q_dim * f32b);
        let wo_d = self.client.empty(sq_total * n * f32b);
        let gu_d = self.client.empty(sq_total * 2 * mlp * f32b);
        let mi_d = self.client.empty(sq_total * mlp * f32b);
        let down_d = self.client.empty(sq_total * n * f32b);
        let fold_params = EdlmQkvFoldParams {
            n_head: config.n_head,
            n_kv_head: config.n_kv_head,
            head_dim: hd,
            eps,
            n_positions: own_total,
            seq_q: sq_total,
        };

        let mut kv_capture: Vec<EdlmLayerKv> = Vec::new();
        let mut own_handles: Vec<Handle> = Vec::with_capacity(n_layer);
        for li in 0..n_layer {
            let lw = &self.layers[li];
            // The state pass RETAINS one combined KV buffer per layer (the
            // KV carry) — a fresh allocation per layer, never a reused one.
            let kv_d = self.client.empty(2 * own_total * kvd * f32b);
            let seed_h = match seed_layers {
                Some(handles) => handles[li].clone(),
                None => dummy_seed.clone(),
            };

            // ── attention block (device-resident) ──
            EdlmRmsNormRowsCubeCL::launch::<ActiveRuntime>(
                &self.client,
                h_dev.clone(),
                lw.attn_norm.clone(),
                hn1_d.clone(),
                sq_total,
                n,
                eps,
            );
            self.matmul_f16b(hn1_d.clone(), lw.qkv.clone(), qkv_d.clone(), sq_total, n, lq);
            EdlmQkvFoldCubeCL::launch::<ActiveRuntime>(
                &self.client,
                qkv_d.clone(),
                lw.q_norm_d.clone(),
                lw.k_norm_d.clone(),
                pos_d.clone(),
                seg_d.clone(),
                self.freq_d.clone(),
                q_rope_d.clone(),
                kv_d.clone(),
                &fold_params,
            );
            EdlmAttnMultiCubeCL::launch::<ActiveRuntime>(
                &self.client,
                q_rope_d.clone(),
                kv_d.clone(),
                seed_h,
                ts_d.clone(),
                te_d.clone(),
                attn_d.clone(),
                &geo.attn_params,
            );
            self.matmul_f16b(attn_d.clone(), lw.wo.clone(), wo_d.clone(), sq_total, q_dim, n);
            // Residual add, in place on the device (the host arm's
            // `h[i] = xr[i] + wo_h[i]`).
            unsafe {
                crate::elementwise_cubecl::AddCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    h_dev.clone(),
                    sq_total * n,
                    wo_d.clone(),
                    sq_total * n,
                    0,
                    0,
                    sq_total * n,
                )
            };

            // ── MLP block (device-resident) ──
            EdlmRmsNormRowsCubeCL::launch::<ActiveRuntime>(
                &self.client,
                h_dev.clone(),
                lw.post_attn_norm.clone(),
                hn2_d.clone(),
                sq_total,
                n,
                eps,
            );
            self.matmul_f16b(hn2_d.clone(), lw.gateup.clone(), gu_d.clone(), sq_total, n, 2 * mlp);
            // SwiGLU: silu(first half) · second half — the host arm's
            // `swiglu` law, on device.
            unsafe {
                crate::elementwise_cubecl::GluSiluGateCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    gu_d.clone(),
                    mi_d.clone(),
                    sq_total,
                    mlp,
                )
            };
            self.matmul_f16b(mi_d.clone(), lw.down.clone(), down_d.clone(), sq_total, mlp, n);
            unsafe {
                crate::elementwise_cubecl::AddCubeCL::launch::<ActiveRuntime>(
                    &self.client,
                    h_dev.clone(),
                    sq_total * n,
                    down_d.clone(),
                    sq_total * n,
                    0,
                    0,
                    sq_total * n,
                )
            };

            if capture {
                // Once per prefill: the host cache for parity inspection +
                // the pointer head (the branch passes never read this).
                let kv_host = read_f32(&self.client, kv_d.clone()).map_err(|e| e.to_string())?;
                kv_capture.push(EdlmLayerKv {
                    k: kv_host[..own_total * kvd].to_vec(),
                    v: kv_host[own_total * kvd..2 * own_total * kvd].to_vec(),
                });
            }
            own_handles.push(kv_d);
        }

        let h_final = read_f32(&self.client, h_dev).map_err(|e| e.to_string())?;
        Ok((h_final, kv_capture, own_handles))
    }
}

/// Upload a host f32 slice as an f16 device buffer (round-to-nearest).
#[cfg(feature = "edlm_gpu")]
fn upload_f16(client: &ComputeClient<ActiveRuntime>, data: &[f32]) -> Result<Handle, String> {
    let bytes = f32_to_f16_bytes(data);
    Ok(client.create_from_slice(&bytes))
}

#[cfg(all(test, feature = "edlm_gpu"))]
mod tests;
