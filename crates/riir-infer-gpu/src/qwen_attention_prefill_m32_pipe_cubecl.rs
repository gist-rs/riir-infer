//! Issue 844 T5 residue: the **K/V register-prefetch pipeline** variant of the
//! m32 tiled flash arm — the latency-hiding branch of the KV-share lever,
//! opened after Bench 899 closed the Q-block-widening branch (m64 = −76% —
//! register spills) and the T1 phase map (Bench 843) fixed the target: KV
//! loads are 41→63% of the kernel's wall time, monotone in P, with the kernel
//! streaming at ~half DRAM peak — latency-exposed, not bandwidth-saturated.
//!
//! ## The mechanism (one line)
//!
//! **VERDICT (Bench 900, 2026-09-10): MEASURED NO-GO** — kept opt-in-only,
//! default OFF everywhere. −20.7% / −19.6% / −45.4% vs the production m32
//! arm at 8K/16K/32K, ×3 interleaved cooled rounds, every round consistent.
//! The latency-hiding branch of the KV-share lever is CLOSED by measurement
//! (Q-block-widening closed in Bench 899; register prefetch here;
//! smem-tile-burst demoted by analysis + this evidence). Full record:
//! `.benchmarks/900_issue844_t5residue_kv_pipe_no_go.md`.
//!
//! In the m32 kernel the K row load sits at the top of the iteration and the
//! V row load sits *after* the softmax block, so the load pipeline drains
//! twice per position: the V loads stall behind the whole K-consume chain
//! (partials → `plane_sum` shuffles → exp). This variant issues the **next**
//! position's K row AND V row (16 f32 loads per lane, all independent) at the
//! top of the current iteration, before the dependent compute — one full
//! iteration of arithmetic (~16 plane-uniform exp-class ops + 64 FMAs) then
//! overlaps both loads' latency. Depth-1 covers L2-class latency (the KV
//! stream is shared by neighboring cubes of the same head, so L2 hits are the
//! common case); depth-2 died with it — depth-1 measured NEGATIVE (Bench
//! 900), so more of the same mechanism is not a candidate.
//!
//! ## Why registers, not smem tiles
//!
//! The m64 NO-GO (Bench 899) is the a-fortiori warning against per-cube
//! arithmetic-density plays: each lane already needs ONLY its own 8 dims of
//! each K/V row (the cross-lane communication is the `plane_sum` reduction,
//! not the loads), so smem staging buys no reuse the L1 doesn't already give
//! — while adding bank-conflict and occupancy hazards (Apple's 32 KB/core
//! threadgroup budget caps cubes-per-core at 1–2 with any real tile).
//! Register prefetch keeps the cube shape, the grid, the smem budget (zero),
//! and the coalescing pattern byte-identical to m32; the cost is +16 live
//! registers (~half the margin that killed m64) and 16 register moves.
//!
//! ## Numerics: a TOLERANCE arm, measured — NOT the m32/m16 0-diff class
//!
//! The compute body is the m32 text verbatim and the FLOP count is
//! identical, but the probe twin MEASURED worst_rel 3.0e-6 (~25 ulps of the
//! output scale, vs the 1e-4 Bench-774 bar — 30× headroom) and ~19.7M
//! bit-diffs over the 11-fixture G1 set: moving the loads changed the Air
//! fast-math reassociation context (the Bench-843 instrument-note class —
//! m32/m64 kept 0-diffs because their load placement was verbatim too;
//! hoisting the loads crosses it). Semantically identical, same math per
//! row, different rounding order — so this arm is gated (and would promote)
//! as a TOLERANCE arm (the q8kv class): the 1e-4 gate is its bar, and any
//! promotion claim carries the tolerance-arm duties (argmax-flip analysis;
//! no FNV-equality claims).
//!
//! ## Toggle (OPT-IN ONLY — default OFF on every platform)
//!
//! Env `RIIR_PREFILL_TILED_FLASH_M32_PIPE=1` (or the setter). The m32
//! sustained-load lesson (the 0.929× back-to-back e2e FAIL, Bench-790
//! thermal class) applies to any candidate arm before its own cooled
//! long-pass e2e has run — a probe-grade candidate never defaults. Env is
//! read exactly once (the Bench-805 OnceLock pattern); the setter is
//! authoritative after.
//!
//! Pre-registered GO/NO-GO (kernel level, this harness's protocol):
//! interleaved median-of-3, cooled, vs the PRODUCTION m32 arm at
//! 8192/16384/32768 — ≥3% at ≥2 lengths, reproduced ×2, is the GO bar into
//! the cooled-long-pass e2e ladder (the Bench-845 class); anything less is a
//! NO-GO that closes the latency-hiding branch by measurement, and the KV
//! share narrows to traffic-WIDTH (f16-KV class, numerics-gated per Bench
//! 773) and the B~102 distill lead.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

/// Queries per cube (= 4 per plane at `CubeDim::new_1d(256)`) — the m32
/// shape, unchanged.
#[cfg(feature = "cubecl_runtime")]
const TILED_M32_PIPE_Q_PER_CUBE: u32 = 32;

/// The M=32 four-rows-per-plane tiled causal flash-attention prefill, K/V
/// register-prefetch pipelined (Issue 844 T5 residue). Same buffer contract
/// and `params` layout as
/// [`crate::qwen_attention_prefill_tiled_m32_cubecl::qwen_attention_prefill_tiled_m32_f32`]:
/// `[head_dim, n_head, n_kv_head, p, scale, q_offset, q_tiles]`.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_prefill_tiled_m32_pipe_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    gate: &[f32],
    attn_out: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as u32;
    let n_head = params[1usize] as u32;
    let n_kv_head = params[2usize] as u32;
    let p = params[3usize] as u32;
    let scale = params[4usize];
    let q_offset = params[5usize] as u32;
    let q_tiles = params[6usize] as u32;

    let cube_id = CUBE_POS_X;
    let head_idx = cube_id / q_tiles;
    let q_tile = cube_id % q_tiles;

    // Plane/lane decomposition (simdgroup i = threads [32i, 32i+32)).
    let pl = UNIT_POS / 32u32;
    let lane = UNIT_POS_PLANE;

    // Each plane owns FOUR ADJACENT query rows — identical to m32.
    let q_base = q_tile * TILED_M32_PIPE_Q_PER_CUBE + pl * 4u32;
    let q_pos_a = q_base;
    let q_pos_b = q_base + 1u32;
    let q_pos_c = q_base + 2u32;
    let q_pos_d = q_base + 3u32;
    let active_a = q_pos_a < p;
    let active_b = q_pos_b < p;
    let active_c = q_pos_c < p;
    let active_d = q_pos_d < p;

    // GQA: map query head to key/value head group.
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_head_off = kv_group * head_dim;
    let q_stride = n_head * head_dim;
    let kv_stride = n_kv_head * head_dim;
    let q_abs_a = q_pos_a + q_offset;
    let q_abs_b = q_pos_b + q_offset;
    let q_abs_c = q_pos_c + q_offset;
    let q_abs_d = q_pos_d + q_offset;
    let q_off_a = (q_pos_a * q_stride + head_idx * head_dim) as usize;
    let q_off_b = (q_pos_b * q_stride + head_idx * head_dim) as usize;
    let q_off_c = (q_pos_c * q_stride + head_idx * head_dim) as usize;
    let q_off_d = (q_pos_d * q_stride + head_idx * head_dim) as usize;
    let dims_base = (lane * 8u32) as usize;

    // Lane dims, rows A-D — the m32 prologue verbatim (hand-unrolled
    // statically-indexed scalars; the Batch-49 local-memory-demotion class).
    let qa0 = if active_a { query[q_off_a + dims_base] } else { f32::new(0.0f32) };
    let qa1 = if active_a { query[q_off_a + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qa2 = if active_a { query[q_off_a + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qa3 = if active_a { query[q_off_a + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qa4 = if active_a { query[q_off_a + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qa5 = if active_a { query[q_off_a + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qa6 = if active_a { query[q_off_a + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qa7 = if active_a { query[q_off_a + dims_base + 7usize] } else { f32::new(0.0f32) };
    let qb0 = if active_b { query[q_off_b + dims_base] } else { f32::new(0.0f32) };
    let qb1 = if active_b { query[q_off_b + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qb2 = if active_b { query[q_off_b + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qb3 = if active_b { query[q_off_b + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qb4 = if active_b { query[q_off_b + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qb5 = if active_b { query[q_off_b + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qb6 = if active_b { query[q_off_b + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qb7 = if active_b { query[q_off_b + dims_base + 7usize] } else { f32::new(0.0f32) };
    let qc0 = if active_c { query[q_off_c + dims_base] } else { f32::new(0.0f32) };
    let qc1 = if active_c { query[q_off_c + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qc2 = if active_c { query[q_off_c + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qc3 = if active_c { query[q_off_c + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qc4 = if active_c { query[q_off_c + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qc5 = if active_c { query[q_off_c + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qc6 = if active_c { query[q_off_c + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qc7 = if active_c { query[q_off_c + dims_base + 7usize] } else { f32::new(0.0f32) };
    let qd0 = if active_d { query[q_off_d + dims_base] } else { f32::new(0.0f32) };
    let qd1 = if active_d { query[q_off_d + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qd2 = if active_d { query[q_off_d + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qd3 = if active_d { query[q_off_d + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qd4 = if active_d { query[q_off_d + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qd5 = if active_d { query[q_off_d + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qd6 = if active_d { query[q_off_d + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qd7 = if active_d { query[q_off_d + dims_base + 7usize] } else { f32::new(0.0f32) };

    // Online softmax state per row, per lane — the m32 shape verbatim.
    let mut run_max_a = f32::new(-1e30f32);
    let mut run_sum_a = f32::new(0.0f32);
    let mut run_max_b = f32::new(-1e30f32);
    let mut run_sum_b = f32::new(0.0f32);
    let mut run_max_c = f32::new(-1e30f32);
    let mut run_sum_c = f32::new(0.0f32);
    let mut run_max_d = f32::new(-1e30f32);
    let mut run_sum_d = f32::new(0.0f32);
    let mut oa0 = f32::new(0.0f32);
    let mut oa1 = f32::new(0.0f32);
    let mut oa2 = f32::new(0.0f32);
    let mut oa3 = f32::new(0.0f32);
    let mut oa4 = f32::new(0.0f32);
    let mut oa5 = f32::new(0.0f32);
    let mut oa6 = f32::new(0.0f32);
    let mut oa7 = f32::new(0.0f32);
    let mut ob0 = f32::new(0.0f32);
    let mut ob1 = f32::new(0.0f32);
    let mut ob2 = f32::new(0.0f32);
    let mut ob3 = f32::new(0.0f32);
    let mut ob4 = f32::new(0.0f32);
    let mut ob5 = f32::new(0.0f32);
    let mut ob6 = f32::new(0.0f32);
    let mut ob7 = f32::new(0.0f32);
    let mut oc0 = f32::new(0.0f32);
    let mut oc1 = f32::new(0.0f32);
    let mut oc2 = f32::new(0.0f32);
    let mut oc3 = f32::new(0.0f32);
    let mut oc4 = f32::new(0.0f32);
    let mut oc5 = f32::new(0.0f32);
    let mut oc6 = f32::new(0.0f32);
    let mut oc7 = f32::new(0.0f32);
    let mut od0 = f32::new(0.0f32);
    let mut od1 = f32::new(0.0f32);
    let mut od2 = f32::new(0.0f32);
    let mut od3 = f32::new(0.0f32);
    let mut od4 = f32::new(0.0f32);
    let mut od5 = f32::new(0.0f32);
    let mut od6 = f32::new(0.0f32);
    let mut od7 = f32::new(0.0f32);

    // Uniform loop bound — the m32 formula verbatim.
    let q_last = q_tile * TILED_M32_PIPE_Q_PER_CUBE + (TILED_M32_PIPE_Q_PER_CUBE - 1u32);
    let n_loop = if q_last < p { q_last } else { p - 1u32 } + q_offset + 1u32;

    // ── The pipeline change ─────────────────────────────────────────────
    // K/V rows live in prefetch registers: the prologue loads position 0,
    // each iteration PROMOTES the prefetched row (plain register moves) and
    // then issues position pos+1's 16 independent loads BEFORE the dependent
    // compute — the m32 body's K load (top-of-loop) and V load
    // (post-softmax) both collapse into that one overlapped issue point.
    // `pos + 1 < n_loop` bounds the loads: n_loop ≤ p + q_offset = the KV
    // buffer's row count, so pos+1 ≤ n_loop−1 is always in-bounds.
    let kb0 = kv_head_off as usize;
    let mut kp0 = if 0u32 < n_loop { key[kb0 + dims_base] } else { f32::new(0.0f32) };
    let mut kp1 = if 0u32 < n_loop { key[kb0 + dims_base + 1usize] } else { f32::new(0.0f32) };
    let mut kp2 = if 0u32 < n_loop { key[kb0 + dims_base + 2usize] } else { f32::new(0.0f32) };
    let mut kp3 = if 0u32 < n_loop { key[kb0 + dims_base + 3usize] } else { f32::new(0.0f32) };
    let mut kp4 = if 0u32 < n_loop { key[kb0 + dims_base + 4usize] } else { f32::new(0.0f32) };
    let mut kp5 = if 0u32 < n_loop { key[kb0 + dims_base + 5usize] } else { f32::new(0.0f32) };
    let mut kp6 = if 0u32 < n_loop { key[kb0 + dims_base + 6usize] } else { f32::new(0.0f32) };
    let mut kp7 = if 0u32 < n_loop { key[kb0 + dims_base + 7usize] } else { f32::new(0.0f32) };
    let mut vp0 = if 0u32 < n_loop { value[kb0 + dims_base] } else { f32::new(0.0f32) };
    let mut vp1 = if 0u32 < n_loop { value[kb0 + dims_base + 1usize] } else { f32::new(0.0f32) };
    let mut vp2 = if 0u32 < n_loop { value[kb0 + dims_base + 2usize] } else { f32::new(0.0f32) };
    let mut vp3 = if 0u32 < n_loop { value[kb0 + dims_base + 3usize] } else { f32::new(0.0f32) };
    let mut vp4 = if 0u32 < n_loop { value[kb0 + dims_base + 4usize] } else { f32::new(0.0f32) };
    let mut vp5 = if 0u32 < n_loop { value[kb0 + dims_base + 5usize] } else { f32::new(0.0f32) };
    let mut vp6 = if 0u32 < n_loop { value[kb0 + dims_base + 6usize] } else { f32::new(0.0f32) };
    let mut vp7 = if 0u32 < n_loop { value[kb0 + dims_base + 7usize] } else { f32::new(0.0f32) };

    let mut pos = 0u32;
    while pos < n_loop {
        // Promote: this iteration's K/V values (register moves).
        let k0 = kp0;
        let k1 = kp1;
        let k2 = kp2;
        let k3 = kp3;
        let k4 = kp4;
        let k5 = kp5;
        let k6 = kp6;
        let k7 = kp7;
        let v0 = vp0;
        let v1 = vp1;
        let v2 = vp2;
        let v3 = vp3;
        let v4 = vp4;
        let v5 = vp5;
        let v6 = vp6;
        let v7 = vp7;

        // Prefetch pos+1 — issued before any dependent compute of THIS
        // iteration, landing during the ~full iteration of arithmetic below.
        let has_next = pos + 1u32 < n_loop;
        let n_base = ((pos + 1u32) * kv_stride + kv_head_off) as usize;
        kp0 = if has_next { key[n_base + dims_base] } else { f32::new(0.0f32) };
        kp1 = if has_next { key[n_base + dims_base + 1usize] } else { f32::new(0.0f32) };
        kp2 = if has_next { key[n_base + dims_base + 2usize] } else { f32::new(0.0f32) };
        kp3 = if has_next { key[n_base + dims_base + 3usize] } else { f32::new(0.0f32) };
        kp4 = if has_next { key[n_base + dims_base + 4usize] } else { f32::new(0.0f32) };
        kp5 = if has_next { key[n_base + dims_base + 5usize] } else { f32::new(0.0f32) };
        kp6 = if has_next { key[n_base + dims_base + 6usize] } else { f32::new(0.0f32) };
        kp7 = if has_next { key[n_base + dims_base + 7usize] } else { f32::new(0.0f32) };
        vp0 = if has_next { value[n_base + dims_base] } else { f32::new(0.0f32) };
        vp1 = if has_next { value[n_base + dims_base + 1usize] } else { f32::new(0.0f32) };
        vp2 = if has_next { value[n_base + dims_base + 2usize] } else { f32::new(0.0f32) };
        vp3 = if has_next { value[n_base + dims_base + 3usize] } else { f32::new(0.0f32) };
        vp4 = if has_next { value[n_base + dims_base + 4usize] } else { f32::new(0.0f32) };
        vp5 = if has_next { value[n_base + dims_base + 5usize] } else { f32::new(0.0f32) };
        vp6 = if has_next { value[n_base + dims_base + 6usize] } else { f32::new(0.0f32) };
        vp7 = if has_next { value[n_base + dims_base + 7usize] } else { f32::new(0.0f32) };

        // ── The m32 compute body, VERBATIM from here to the loop end ──
        let partial_a = qa0 * k0 + qa1 * k1 + qa2 * k2 + qa3 * k3
            + qa4 * k4 + qa5 * k5 + qa6 * k6 + qa7 * k7;
        let partial_b = qb0 * k0 + qb1 * k1 + qb2 * k2 + qb3 * k3
            + qb4 * k4 + qb5 * k5 + qb6 * k6 + qb7 * k7;
        let partial_c = qc0 * k0 + qc1 * k1 + qc2 * k2 + qc3 * k3
            + qc4 * k4 + qc5 * k5 + qc6 * k6 + qc7 * k7;
        let partial_d = qd0 * k0 + qd1 * k1 + qd2 * k2 + qd3 * k3
            + qd4 * k4 + qd5 * k5 + qd6 * k6 + qd7 * k7;
        // Reduce + broadcast within this plane's 32 lanes (tree order — the
        // same FP-equivalent class as the m32 kernel, unchanged per row).
        let score_a = plane_sum(partial_a) * scale;
        let score_b = plane_sum(partial_b) * scale;
        let score_c = plane_sum(partial_c) * scale;
        let score_d = plane_sum(partial_d) * scale;

        let in_causal_a = active_a && pos <= q_abs_a;
        let in_causal_b = active_b && pos <= q_abs_b;
        let in_causal_c = active_c && pos <= q_abs_c;
        let in_causal_d = active_d && pos <= q_abs_d;
        let masked_a = if in_causal_a { score_a } else { f32::new(-1e30f32) };
        let masked_b = if in_causal_b { score_b } else { f32::new(-1e30f32) };
        let masked_c = if in_causal_c { score_c } else { f32::new(-1e30f32) };
        let masked_d = if in_causal_d { score_d } else { f32::new(-1e30f32) };

        let new_max_a = if masked_a > run_max_a { masked_a } else { run_max_a };
        let new_max_b = if masked_b > run_max_b { masked_b } else { run_max_b };
        let new_max_c = if masked_c > run_max_c { masked_c } else { run_max_c };
        let new_max_d = if masked_d > run_max_d { masked_d } else { run_max_d };

        // Exp hoist per row: each plane-uniform exp runs on lane 0 ONLY and
        // is broadcast with one shuffle (the Bench 786 lesson).
        let w_a0 = if lane == 0u32 { (masked_a - new_max_a).exp() } else { f32::new(0.0f32) };
        let corr_a0 = if lane == 0u32 { (run_max_a - new_max_a).exp() } else { f32::new(0.0f32) };
        let w_b0 = if lane == 0u32 { (masked_b - new_max_b).exp() } else { f32::new(0.0f32) };
        let corr_b0 = if lane == 0u32 { (run_max_b - new_max_b).exp() } else { f32::new(0.0f32) };
        let w_c0 = if lane == 0u32 { (masked_c - new_max_c).exp() } else { f32::new(0.0f32) };
        let corr_c0 = if lane == 0u32 { (run_max_c - new_max_c).exp() } else { f32::new(0.0f32) };
        let w_d0 = if lane == 0u32 { (masked_d - new_max_d).exp() } else { f32::new(0.0f32) };
        let corr_d0 = if lane == 0u32 { (run_max_d - new_max_d).exp() } else { f32::new(0.0f32) };
        let w_a = plane_broadcast(w_a0, 0u32);
        let corr_a = plane_broadcast(corr_a0, 0u32);
        let w_b = plane_broadcast(w_b0, 0u32);
        let corr_b = plane_broadcast(corr_b0, 0u32);
        let w_c = plane_broadcast(w_c0, 0u32);
        let corr_c = plane_broadcast(corr_c0, 0u32);
        let w_d = plane_broadcast(w_d0, 0u32);
        let corr_d = plane_broadcast(corr_d0, 0u32);

        run_sum_a = run_sum_a * corr_a + w_a;
        run_sum_b = run_sum_b * corr_b + w_b;
        run_sum_c = run_sum_c * corr_c + w_c;
        run_sum_d = run_sum_d * corr_d + w_d;

        oa0 = oa0 * corr_a + w_a * v0;
        oa1 = oa1 * corr_a + w_a * v1;
        oa2 = oa2 * corr_a + w_a * v2;
        oa3 = oa3 * corr_a + w_a * v3;
        oa4 = oa4 * corr_a + w_a * v4;
        oa5 = oa5 * corr_a + w_a * v5;
        oa6 = oa6 * corr_a + w_a * v6;
        oa7 = oa7 * corr_a + w_a * v7;
        ob0 = ob0 * corr_b + w_b * v0;
        ob1 = ob1 * corr_b + w_b * v1;
        ob2 = ob2 * corr_b + w_b * v2;
        ob3 = ob3 * corr_b + w_b * v3;
        ob4 = ob4 * corr_b + w_b * v4;
        ob5 = ob5 * corr_b + w_b * v5;
        ob6 = ob6 * corr_b + w_b * v6;
        ob7 = ob7 * corr_b + w_b * v7;
        oc0 = oc0 * corr_c + w_c * v0;
        oc1 = oc1 * corr_c + w_c * v1;
        oc2 = oc2 * corr_c + w_c * v2;
        oc3 = oc3 * corr_c + w_c * v3;
        oc4 = oc4 * corr_c + w_c * v4;
        oc5 = oc5 * corr_c + w_c * v5;
        oc6 = oc6 * corr_c + w_c * v6;
        oc7 = oc7 * corr_c + w_c * v7;
        od0 = od0 * corr_d + w_d * v0;
        od1 = od1 * corr_d + w_d * v1;
        od2 = od2 * corr_d + w_d * v2;
        od3 = od3 * corr_d + w_d * v3;
        od4 = od4 * corr_d + w_d * v4;
        od5 = od5 * corr_d + w_d * v5;
        od6 = od6 * corr_d + w_d * v6;
        od7 = od7 * corr_d + w_d * v7;
        run_max_a = new_max_a;
        run_max_b = new_max_b;
        run_max_c = new_max_c;
        run_max_d = new_max_d;

        pos += 1u32;
    }

    // Epilogue, rows A-D — the m32 gated-attention shape per row (sigmoid
    // gate, one exp per output element), verbatim.
    if active_a {
        let inv_sum = f32::new(1.0f32) / run_sum_a;
        let g_off = q_off_a + dims_base;
        let g0 = gate[g_off];
        let g1 = gate[g_off + 1usize];
        let g2 = gate[g_off + 2usize];
        let g3 = gate[g_off + 3usize];
        let g4 = gate[g_off + 4usize];
        let g5 = gate[g_off + 5usize];
        let g6 = gate[g_off + 6usize];
        let g7 = gate[g_off + 7usize];
        let neg0 = f32::new(0.0f32) - g0;
        let neg1 = f32::new(0.0f32) - g1;
        let neg2 = f32::new(0.0f32) - g2;
        let neg3 = f32::new(0.0f32) - g3;
        let neg4 = f32::new(0.0f32) - g4;
        let neg5 = f32::new(0.0f32) - g5;
        let neg6 = f32::new(0.0f32) - g6;
        let neg7 = f32::new(0.0f32) - g7;
        attn_out[g_off] = oa0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = oa1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = oa2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = oa3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = oa4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = oa5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = oa6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = oa7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
    if active_b {
        let inv_sum = f32::new(1.0f32) / run_sum_b;
        let g_off = q_off_b + dims_base;
        let g0 = gate[g_off];
        let g1 = gate[g_off + 1usize];
        let g2 = gate[g_off + 2usize];
        let g3 = gate[g_off + 3usize];
        let g4 = gate[g_off + 4usize];
        let g5 = gate[g_off + 5usize];
        let g6 = gate[g_off + 6usize];
        let g7 = gate[g_off + 7usize];
        let neg0 = f32::new(0.0f32) - g0;
        let neg1 = f32::new(0.0f32) - g1;
        let neg2 = f32::new(0.0f32) - g2;
        let neg3 = f32::new(0.0f32) - g3;
        let neg4 = f32::new(0.0f32) - g4;
        let neg5 = f32::new(0.0f32) - g5;
        let neg6 = f32::new(0.0f32) - g6;
        let neg7 = f32::new(0.0f32) - g7;
        attn_out[g_off] = ob0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = ob1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = ob2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = ob3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = ob4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = ob5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = ob6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = ob7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
    if active_c {
        let inv_sum = f32::new(1.0f32) / run_sum_c;
        let g_off = q_off_c + dims_base;
        let g0 = gate[g_off];
        let g1 = gate[g_off + 1usize];
        let g2 = gate[g_off + 2usize];
        let g3 = gate[g_off + 3usize];
        let g4 = gate[g_off + 4usize];
        let g5 = gate[g_off + 5usize];
        let g6 = gate[g_off + 6usize];
        let g7 = gate[g_off + 7usize];
        let neg0 = f32::new(0.0f32) - g0;
        let neg1 = f32::new(0.0f32) - g1;
        let neg2 = f32::new(0.0f32) - g2;
        let neg3 = f32::new(0.0f32) - g3;
        let neg4 = f32::new(0.0f32) - g4;
        let neg5 = f32::new(0.0f32) - g5;
        let neg6 = f32::new(0.0f32) - g6;
        let neg7 = f32::new(0.0f32) - g7;
        attn_out[g_off] = oc0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = oc1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = oc2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = oc3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = oc4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = oc5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = oc6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = oc7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
    if active_d {
        let inv_sum = f32::new(1.0f32) / run_sum_d;
        let g_off = q_off_d + dims_base;
        let g0 = gate[g_off];
        let g1 = gate[g_off + 1usize];
        let g2 = gate[g_off + 2usize];
        let g3 = gate[g_off + 3usize];
        let g4 = gate[g_off + 4usize];
        let g5 = gate[g_off + 5usize];
        let g6 = gate[g_off + 6usize];
        let g7 = gate[g_off + 7usize];
        let neg0 = f32::new(0.0f32) - g0;
        let neg1 = f32::new(0.0f32) - g1;
        let neg2 = f32::new(0.0f32) - g2;
        let neg3 = f32::new(0.0f32) - g3;
        let neg4 = f32::new(0.0f32) - g4;
        let neg5 = f32::new(0.0f32) - g5;
        let neg6 = f32::new(0.0f32) - g6;
        let neg7 = f32::new(0.0f32) - g7;
        attn_out[g_off] = od0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = od1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = od2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = od3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = od4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = od5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = od6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = od7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
}

/// Launch the M=32 K/V register-prefetch pipelined tiled flash attention
/// prefill (Issue 844 T5 residue). Same buffer contract as
/// [`crate::QwenAttentionPrefillTiledM32CubeCL::launch`], including the
/// chunked `base_pos` cache semantics — callers route through
/// [`crate::ternary_deltanet_gpu_forward::prefill_tiled_flash_m32_pipe_enabled`]
/// (OPT-IN ONLY — default OFF everywhere; head_dim must be 256).
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionPrefillTiledM32PipeCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionPrefillTiledM32PipeCubeCL {
    /// # Safety
    /// Same contract as [`crate::QwenAttentionPrefillTiledM32CubeCL::launch`] —
    /// identical handle shapes and chunked `base_pos` semantics; `head_dim`
    /// must be 256.
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        key_handle: Handle,
        value_handle: Handle,
        gate_handle: Handle,
        attn_out_handle: Handle,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        p: usize,
        base_pos: usize,
    ) {
        const MAX_WG_X: u32 = 65535;

        debug_assert_eq!(
            head_dim, 256,
            "tiled m32-pipe flash kernel is head_dim-256 specialized"
        );
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Grid = n_head × q_tiles (32 queries per cube) — identical to the
        // m32 launcher (65535 / 24 / 32 = 85 tiles per chunk at Bonsai).
        let q_tiles_total = p.div_ceil(TILED_M32_PIPE_Q_PER_CUBE as usize).max(1);
        let tiles_per_chunk = (MAX_WG_X as usize / n_head.max(1) / 32).max(1);
        let mut t0 = 0usize;
        while t0 < p {
            let tiles_left = q_tiles_total - t0.div_ceil(TILED_M32_PIPE_Q_PER_CUBE as usize);
            let tiles = tiles_per_chunk.min(tiles_left);
            let tc = (tiles * TILED_M32_PIPE_Q_PER_CUBE as usize).min(p - t0);
            let params: [f32; 7] = [
                head_dim as f32,
                n_head as f32,
                n_kv_head as f32,
                tc as f32,
                scale,
                (base_pos + t0) as f32,
                tiles as f32,
            ];
            let params_handle =
                crate::params_cache::params_handle(client, f32::as_bytes(&params));
            let q_len = tc * n_head * head_dim;
            let kv_len = (base_pos + p) * n_kv_head * head_dim;
            let q_slice = query_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64);
            let g_slice = gate_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64);
            let o_slice =
                attn_out_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64);
            let n_cubes = n_head * tiles;
            unsafe {
                qwen_attention_prefill_tiled_m32_pipe_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(n_cubes as u32, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(q_slice, q_len),
                    BufferArg::from_raw_parts(key_handle.clone(), kv_len),
                    BufferArg::from_raw_parts(value_handle.clone(), kv_len),
                    BufferArg::from_raw_parts(g_slice, q_len),
                    BufferArg::from_raw_parts(o_slice, q_len),
                    BufferArg::from_raw_parts(params_handle, 7),
                );
            }
            t0 += tc;
        }
    }
}
