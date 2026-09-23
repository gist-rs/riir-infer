//! CubeCL fused causal attention kernel for Gemma-2 training forward (Issue 430 Option B).
//!
//! Eliminates the 52 GPU↔CPU sync points per forward (from Option C) by computing
//! the full causal attention — QK^T + softcap + softmax + weighted-sum — in a single
//! GPU dispatch per (head, query_pos) pair, entirely GPU-resident.
//!
//! # Design
//!
//! Single-dispatch fused kernel (simplified flash-attention, no tiling):
//! - Grid: `(n_head * seq_len, 1, 1)` workgroups — one per (head, query_pos)
//! - Block: 256 threads (= head_dim for Gemma-2-2B)
//!
//! Each workgroup computes `attn_out[i, h, 0..head_dim]` for one query position `i`
//! and one head `h`. The kernel operates in three phases within each workgroup:
//!
//! 1. **Score computation**: thread `j` (key position) computes
//!    `score[j] = dot(Q[i,h,:], K[j,kv_group,:]) * scale`, applies softcap, and
//!    sets `-inf` for masked (future) positions. Q[i,h,:] is loaded into shared
//!    memory once; each thread reads K[j,kv_group,:] sequentially.
//!
//! 2. **Softmax**: parallel max-reduce → exp → sum-reduce → normalize.
//!    Numerically-stable softmax with a while-loop tree reduction over
//!    shared memory (handles any power-of-2 seq_len ≤ 256 — replaced the
//!    Issue 430 G1-FAIL 8-step unrolled form, which was incomplete for
//!    seq_len=256 and double-counted for small seq_len).
//!
//! 3. **Weighted sum**: thread `d` (output dimension) computes
//!    `attn_out[i,h,d] = sum_j softmax[j] * V[j,kv_group,d]`. The softmax weights
//!    are in shared memory from phase 2.
//!
//! # Layout
//!
//! Gemma-2-2B uses GQA: `n_head=8`, `n_kv_head=4`, `head_dim=256`.
//! Query head `h` maps to KV group `h * n_kv_head / n_head = h / 2`.
//! So heads 0,1 → kv_group 0; heads 2,3 → kv_group 1; etc.
//!
//! - Q: `[seq_len, q_dim]` row-major, `q_dim = n_head * head_dim = 2048`
//! - K: `[seq_len, kv_dim]` row-major, `kv_dim = n_kv_head * head_dim = 1024`
//! - V: `[seq_len, kv_dim]` row-major (same as K)
//! - attn_out: `[seq_len, q_dim]` row-major (same layout as Q)
//!
//! # GQA handling
//!
//! The kernel computes `kv_group = h * n_kv_head / n_head` internally and uses
//! `kv_group * head_dim` as the offset into K/V rows. This is identical to the
//! CPU reference (`compute_attention_batched_cpu` in `resident.rs`).
//!
//! # Constraints
//!
//! - `seq_len` must be ≤ 256 and a power of 2 (the reduction is hardcoded for
//!   256 threads). Gemma-2 training uses seq_len=256, which fits exactly.
//! - `head_dim` must be ≤ 256 (Q is loaded into shared memory with 256 threads).
//!   Gemma-2-2B uses head_dim=256, which fits exactly.
//!
//! # Safety
//!
//! Feature-gated behind `gpu_training_resident`. The G1 correctness test validates
//! equivalence against `compute_attention_batched_cpu` at seq_len=256. G2 perf
//! measurement deferred to the GPU-free validation window (post run2).

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

/// Shared-memory slot count + workgroup size for the fused attention kernel
/// (Issue 688 H3 — single source of truth).
///
/// This constant is load-bearing in THREE places that must agree:
/// the kernel's `cube_size`, the three `Shared::<[f32]>::new_slice`
/// allocations, and the launcher's `CubeDim::new_1d`. The kernel cannot see
/// the actual CubeDim at runtime, and the launcher cannot see the kernel
/// constant — changing either independently silently breaks phase-0's
/// stride loop and phase-1's `j = tid` coverage.
#[cfg(feature = "cubecl_runtime")]
const ATTENTION_BLOCK: usize = 256;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
// ---------------------------------------------------------------------------
// Fused causal attention kernel
// ---------------------------------------------------------------------------

/// CubeCL fused causal attention kernel for Gemma-2 training forward.
///
/// See module docs for the full algorithm description.
///
/// ## Parameters
///
/// - `q`: `[f32; seq_len * q_dim]` — Q post-RoPE, row-major.
/// - `k`: `[f32; seq_len * kv_dim]` — K post-RoPE, row-major.
/// - `v`: `[f32; seq_len * kv_dim]` — V pre-RoPE, row-major.
/// - `params`: `[f32; 9]` — `[scale, softcap, neg_inf, head_dim, seq_len, q_dim, kv_dim, n_kv_head, n_head]`
///   precomputed on CPU (f32 encoding avoids u32→f32 casting issues in CubeCL v0.10).
/// - `output`: `[f32; seq_len * q_dim]` — attention output.
///
/// ## Dispatch
///
/// `CubeCount::Static(n_head * seq_len, 1, 1)`, `CubeDim::new_1d(256)`.
///
/// Each workgroup handles one (head, query_pos) pair:
/// - `CUBE_POS_X = h * seq_len + i`
/// - `UNIT_POS = tid` (0..255, maps to key position `j` in phases 1–2, output dim `d` in phase 3)
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn causal_attention_fused_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    params: &[f32],
    output: &mut [f32],
) {
    // ── Unpack params (all precomputed on CPU as f32) ──
    let scale = params[0usize];
    let softcap = params[1usize];
    let neg_inf = params[2usize];
    let head_dim = params[3usize] as u32;
    let seq_len = params[4usize] as u32;
    let q_dim = params[5usize] as u32;
    let kv_dim = params[6usize] as u32;
    let n_kv_head = params[7usize] as u32;
    let n_head = params[8usize] as u32;

    // Must equal ATTENTION_BLOCK (the launcher's CubeDim + smem sizing). The
    // CubeCL cube macro cannot reference the module const — keep them tied by
    // the launcher's asserts + this comment (Issue 688 H3).
    let cube_size = 256u32;
    let tid = UNIT_POS;
    let cube_id = CUBE_POS_X;
    let h = cube_id / seq_len; // head index
    let i = cube_id % seq_len; // query position

    // GQA: kv_group = h * n_kv_head / n_head
    let kv_group = h * n_kv_head / n_head;
    let q_head_off = h * head_dim; // offset within a q_dim-wide row
    let kv_head_off = kv_group * head_dim; // offset within a kv_dim-wide row

    // ── Phase 0: Load Q[i, h, :] into shared memory ──
    // SharedMemory must be statically sized in CubeCL v0.10. head_dim=256 for
    // Gemma-2-2B; the launcher asserts head_dim ≤ ATTENTION_BLOCK.
    //
    // Read-range invariant (Issue 688 H5): lanes with `d >= head_dim` are
    // NEVER written and MUST NOT be read — phase 1 below bounds at `d2 <
    // head_dim`. A "simplification" to unguarded full-block copies/reads
    // (exactly the pre-Issue-430 edit class) would read uninitialized memory.
    let mut smem_q = Shared::<[f32]>::new_slice(256usize);
    let mut d = tid;
    while d < head_dim {
        smem_q[d as usize] = q[(i * q_dim + q_head_off + d) as usize];
        d += cube_size;
    }
    sync_cube();

    // ── Phase 1: Compute raw attention scores ──
    // Thread tid (= j, key position) computes dot(Q[i,h,:], K[j,kv_group,:]) * scale.
    // For j > i (future position), set score to -inf (causal mask).
    // SharedMemory statically sized at ATTENTION_BLOCK (launcher asserts
    // seq_len ≤ ATTENTION_BLOCK). Same read-range invariant as phase 0:
    // lanes with `tid > i` are written neg_inf below and lanes `tid ≥ seq_len`
    // are read ONLY through the reduction, whose strides derive from seq_len —
    // do not "simplify" to full-block reads.
    let mut smem_scores = Shared::<[f32]>::new_slice(256usize);
    let j = tid;
    if j <= i {
        let mut dot = f32::new(0.0f32);
        let mut d2 = 0u32;
        while d2 < head_dim {
            dot += smem_q[d2 as usize] * k[(j * kv_dim + kv_head_off + d2) as usize];
            d2 += 1u32;
        }
        let mut score = dot * scale;
        // Gemma-2 attention logit softcapping.
        if softcap > f32::new(0.0f32) {
            score = softcap * (score / softcap).tanh();
        }
        smem_scores[j as usize] = score;
    } else {
        smem_scores[j as usize] = neg_inf;
    }
    sync_cube();

    // ── Phase 2: Numerically stable softmax over j = 0..=i ──
    //
    // BUG FIXES (Issue 430 G1 FAIL, validated on 4090 2026-08-09):
    //   (1) The original code reduced IN-PLACE on smem_scores, destroying the
    //       original score values before the exp step. Fix: reduce into a
    //       SEPARATE scratch buffer (smem_reduce), preserving smem_scores.
    //   (2) The original 7-step unrolled reduction was INCOMPLETE for
    //       seq_len=256 (needs 8 halvings: 256→1) — only reduced the first
    //       half. For seq_len=4, the hardcoded `if tid < 1u32` final step
    //       executed redundantly and DOUBLE-COUNTED in the sum reduction
    //       (producing a systematic 2/3 scaling of the output).
    // Fix: replace both unrolled reductions with a single `while`-loop-based
    // tree reduction that handles any power-of-2 seq_len ≤ 256 correctly.
    let mut smem_reduce = Shared::<[f32]>::new_slice(256usize);

    // Phase 2a: parallel max reduction (loop-based, log2(seq_len) steps).
    if tid < seq_len {
        smem_reduce[tid as usize] = smem_scores[tid as usize];
    }
    sync_cube();
    let mut stride = seq_len / 2u32;
    while stride > 0u32 {
        if tid < stride {
            let a = smem_reduce[tid as usize];
            let b = smem_reduce[(tid + stride) as usize];
            if a > b {
                smem_reduce[tid as usize] = a;
            } else {
                smem_reduce[tid as usize] = b;
            }
        }
        sync_cube();
        stride /= 2u32;
    }

    let max_score = smem_reduce[0usize];

    // Phase 2b: exp(score - max), store back to smem_scores.
    // smem_scores still holds the ORIGINAL scores (reduction went to smem_reduce).
    if j <= i {
        let diff = smem_scores[j as usize] - max_score;
        smem_scores[j as usize] = diff.exp();
    } else {
        smem_scores[j as usize] = f32::new(0.0f32);
    }
    sync_cube();

    // Phase 2c: parallel sum reduction (loop-based) into smem_reduce.
    // Copy exp values from smem_scores first, then reduce in smem_reduce.
    if tid < seq_len {
        smem_reduce[tid as usize] = smem_scores[tid as usize];
    }
    sync_cube();
    let mut stride_sum = seq_len / 2u32;
    while stride_sum > 0u32 {
        if tid < stride_sum {
            smem_reduce[tid as usize] =
                smem_reduce[tid as usize] + smem_reduce[(tid + stride_sum) as usize];
        }
        sync_cube();
        stride_sum /= 2u32;
    }

    let sum_exp = smem_reduce[0usize];
    let inv_sum = f32::new(1.0f32) / sum_exp;

    // Normalize smem_scores in-place to softmax probabilities.
    if j <= i {
        smem_scores[j as usize] = smem_scores[j as usize] * inv_sum;
    } else {
        smem_scores[j as usize] = f32::new(0.0f32);
    }
    sync_cube();

    // ── Phase 3: Weighted sum — attn_out[i,h,d] = sum_j softmax[j] * V[j,kv_group,d] ──
    // Thread tid now maps to output dimension d (= tid).
    let d3 = tid;
    if d3 < head_dim {
        let mut acc = f32::new(0.0f32);
        let mut j3 = 0u32;
        while j3 <= i {
            let weight = smem_scores[j3 as usize];
            let v_val = v[(j3 * kv_dim + kv_head_off + d3) as usize];
            acc += weight * v_val;
            j3 += 1u32;
        }
        output[(i * q_dim + q_head_off + d3) as usize] = acc;
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

/// CubeCL fused causal attention launcher (Issue 430 Option B).
///
/// Wraps `causal_attention_fused_f32` with precomputed parameters.
/// Computes full causal attention in a single GPU dispatch — no CPU sync,
/// no intermediate score matrix readback.
///
/// # GQA
///
/// Supports grouped-query attention via `n_kv_head < n_head`. The kernel
/// computes `kv_group = h * n_kv_head / n_head` internally, matching the CPU
/// reference (`compute_attention_batched_cpu`).
///
/// # Safety
///
/// Buffer handles must have correct sizes:
/// - `q_handle`: `seq_len * q_dim` f32 elements
/// - `k_handle`: `seq_len * kv_dim` f32 elements
/// - `v_handle`: `seq_len * kv_dim` f32 elements
/// - `output_handle`: `seq_len * q_dim` f32 elements
///
/// `seq_len` must be ≤ 256 and a power of 2. `head_dim` must be ≤ 256.
#[cfg(feature = "cubecl_runtime")]
pub struct CausalAttentionFusedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl CausalAttentionFusedCubeCL {
    /// Launch fused causal attention kernel.
    ///
    /// See struct docs for the full contract.
    ///
    /// # Safety
    ///
    /// Caller must guarantee:
    /// - `q_handle` has `seq_len * q_dim` f32 elements
    /// - `k_handle` has `seq_len * kv_dim` f32 elements
    /// - `v_handle` has `seq_len * kv_dim` f32 elements
    /// - `output_handle` has `seq_len * q_dim` f32 elements
    /// - `seq_len` is a power of 2 and ≤ 256
    /// - `head_dim` is ≤ 256
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q_handle: Handle,
        k_handle: Handle,
        v_handle: Handle,
        output_handle: Handle,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        seq_len: usize,
        q_dim: usize,
        kv_dim: usize,
        softcap: f32,
    ) {
        assert!(
            seq_len <= ATTENTION_BLOCK,
            "CausalAttentionFusedCubeCL requires seq_len <= {ATTENTION_BLOCK} (got {seq_len}); \
             the shared-memory reduction is hardcoded for {ATTENTION_BLOCK} threads"
        );
        assert!(
            head_dim <= ATTENTION_BLOCK,
            "CausalAttentionFusedCubeCL requires head_dim <= {ATTENTION_BLOCK} (got {head_dim}); \
             the Q shared-memory load uses {ATTENTION_BLOCK} threads"
        );
        assert!(
            seq_len.is_power_of_two(),
            "CausalAttentionFusedCubeCL requires seq_len to be a power of 2 (got {seq_len})"
        );
        // Dim-derivation invariants (Issue 688 H4): the kernel mixes the
        // passed q_dim/kv_dim with the derived `q_head_off = h * head_dim` —
        // a mismatch gives silent OOB reads under launch_unchecked.
        assert_eq!(
            q_dim,
            n_head * head_dim,
            "q_dim {q_dim} != n_head {n_head} × head_dim {head_dim}"
        );
        assert_eq!(
            kv_dim,
            n_kv_head * head_dim,
            "kv_dim {kv_dim} != n_kv_head {n_kv_head} × head_dim {head_dim}"
        );
        // GQA divisibility (Issue 688 H6): floor-division grouping is only
        // the standard even mapping when n_head is a multiple of n_kv_head.
        assert!(
            n_head.is_multiple_of(n_kv_head),
            "non-standard GQA grouping: n_head {n_head} not divisible by n_kv_head {n_kv_head}"
        );

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let neg_inf = f32::NEG_INFINITY;
        // NOTE (Issue 688 H2): the params buffer is re-created per launch (26
        // device allocations per training step for 9 never-changing floats
        // once seq_len is fixed). Caching requires keying on seq_len in the
        // caller's handle struct — deferred; per-call create_from_slice is
        // the current cost.
        let params: &[f32] = &[
            scale,
            softcap,
            neg_inf,
            head_dim as f32,
            seq_len as f32,
            q_dim as f32,
            kv_dim as f32,
            n_kv_head as f32,
            n_head as f32,
        ];
        let params_handle = client.create_from_slice(f32::as_bytes(params));

        let n_workgroups = n_head * seq_len;

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            causal_attention_fused_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_workgroups as u32, 1, 1),
                CubeDim::new_1d(ATTENTION_BLOCK as u32),
                BufferArg::from_raw_parts(q_handle, seq_len * q_dim),
                BufferArg::from_raw_parts(k_handle, seq_len * kv_dim),
                BufferArg::from_raw_parts(v_handle, seq_len * kv_dim),
                BufferArg::from_raw_parts(params_handle, 9),
                BufferArg::from_raw_parts(output_handle, seq_len * q_dim),
            );
        }
    }
}
