//! Issue 844 T5-width: the f16-KV traffic-width variant of the production
//! tiled flash-attention m32 prefill — the last OPEN branch of the
//! KV-amortization lever after Bench 899 (Q-block widening) and Bench 900
//! (K/V register prefetch) both closed NO-GO.
//!
//! The T1 phase map (Bench 843) measured KV loads at 41→63% of the kernel's
//! wall time, monotone in P — the kernel is memory-streaming bound. This arm
//! halves the K/V **traffic width**: K/V rows are stored as `f16` (1 KB per
//! 256-dim row instead of 2 KB), upconverted to f32 in registers at load,
//! and every subsequent operation (dot chains, online softmax, PV, gate) is
//! the m32 arithmetic text **verbatim in f32** — the f16-storage/f32-
//! accumulation numerics class that is the production standard elsewhere
//! (llama.cpp KV defaults to F16 with FP32 attention accumulation; its Metal
//! FA path even dequantizes quantized KV → F16 before flash attention;
//! vLLM's FP8-KV collapse blamed accumulation, not narrow storage).
//!
//! ## Numerics class (pre-registered BEFORE measurement)
//!
//! This is a **tolerance arm, NOT the bit-identity class**: K/V values round
//! to f16 on the write side (round-to-nearest-even, ≤ 2^-11 relative per
//! element), so outputs differ from the f32-KV kernel by the propagated
//! rounding — the q8kv/m32-pipe tolerance family, and the exact shape the
//! Bench-773 metric lesson prescribes gating on:
//!
//! - **G1 primary — argmax identity**: 0 output-argmax flips vs the
//!   production m32 kernel on the full G1 fixture family (exact-tile,
//!   ragged, peaked ×2, sub-cube, cube-boundary, chunked cache, multi-chunk).
//!   ANY flip is a band failure for f16 storage at that class → the
//!   bf16-storage/f32-compute arm is the DESIGNED fallback (the Qwen-3.5-
//!   class bf16-KV caveat), not a silent tolerance loosening.
//! - **G1 primary — normalized max_abs band**: worst |f16−f32| / max|out| ≤
//!   5e-3 (≈10× the single-element f16 quantum at unit scale; expectation is
//!   the ~1e-3..2.4e-3 class Bench 773 recorded for f16-activation). Raw
//!   max_rel is RECORDED, never gated — near-zero outputs make it the wrong
//!   metric (the Bench-773 lesson verbatim).
//! - **G1 determinism**: two consecutive launches on the same fixture are
//!   bit-identical (no atomics; the launch-determinism class).
//! - **G2 perf gate**: kvf16 vs the PRODUCTION m32 kernel ≥3% kernel-level
//!   at ≥2 lengths, reproduced ×2, cooled-window protocol — the family bar.
//!   The f32→f16 cache-conversion pass is NOT in the timed path (both arms
//!   read pre-resident caches); its e2e charge is analyzed in the bench doc:
//!   one O(1×) KV read + 0.5× write per chunk vs attention's ~P/2 average
//!   causal re-reads — amortized noise at the measured lengths.
//!
//! ## Measured verdict (Bench 901, 2026-09-10 — G1 FAILED, branch NO-GO)
//!
//! The pre-registered G1 identity gate FIRED on first measurement (the M3
//! Metal production shapes, 11-fixture family): **23 output-argmax flips vs
//! production m32 across 78,867 rows** (0.03–0.25% per fixture — 0 on six
//! fixtures, 1 @p128/1.0, 2 @p33, 4 @p128/6.0, 16 @p3000/chunked), at a
//! normalized max_abs of **2.0e-4..4.8e-3** (the band ≤5e-3 HELD — peaked
//! fixtures at the top) and launch determinism PASS. The failure is
//! **precision-class (f16 mantissa), not range-class**: that refutes the
//! designed bf16 fallback on the same axis — bf16's mantissa is 8× coarser
//! (2⁻⁹ vs 2⁻¹² relative quantum) while its wide-exponent benefit addresses
//! an overflow failure mode that did not occur (Bonsai K/V at |·| ≤ ~6 sits
//! deep inside f16's normal range). The traffic-width branch closes NO-GO at
//! both storage precisions; with Bench 899 (tiling) and Bench 900 (latency-
//! hiding) the KV-share lever is fully closed, and Bench 837's Q8-KV wash
//! independently says an even-narrower width did not pay either. Per the
//! pre-registered gate order the G2 timing never ran (G1 gates first).
//!
//! ## Keep-arm record (the m64/m32-pipe lifecycle pattern)
//!
//! Opt-in probe arm only — **no production dispatch route exists yet** (the
//! production KV cache is f32; wiring the f16 write side into
//! `ternary_deltanet_gpu_forward` is the follow-up unit, gated on this
//! arm's G1+G2 verdicts). No env gate, no launch counter until a dispatch
//! route exists (the m64 pattern: counters observe the PRODUCTION route).
//! The launcher is exported for the bench_844 T1 harness, which owns the
//! G1/G2 protocol.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;
#[cfg(feature = "cubecl_runtime")]
use half::f16 as half_f16;

/// Queries per cube (= 4 per plane at `CubeDim::new_1d(256)`) — the m32
/// tiling, unchanged.
#[cfg(feature = "cubecl_runtime")]
const TILED_M32_KVF16_Q_PER_CUBE: u32 = 32;

/// The M=32 f16-KV traffic-width tiled causal flash-attention prefill
/// (Issue 844 T5-width). Buffer contract = the m32 kernel's EXCEPT
/// `key`/`value` are f16-element buffers (`(base_pos + p) * n_kv_head *
/// head_dim` f16 elements — half the bytes); `query`/`gate`/`attn_out`/
/// `params` are unchanged f32 with the identical
/// `[head_dim, n_head, n_kv_head, p, scale, q_offset, q_tiles]` layout.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_prefill_tiled_m32_kvf16_f32(
    query: &[f32],
    key: &[half_f16],
    value: &[half_f16],
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

    // Each plane owns FOUR ADJACENT query rows — the m32 quartet structure.
    let q_base = q_tile * TILED_M32_KVF16_Q_PER_CUBE + pl * 4u32;
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
    // Chunk-local bases of the four query rows (q/gate/attn_out are
    // chunk-sliced).
    let q_off_a = (q_pos_a * q_stride + head_idx * head_dim) as usize;
    let q_off_b = (q_pos_b * q_stride + head_idx * head_dim) as usize;
    let q_off_c = (q_pos_c * q_stride + head_idx * head_dim) as usize;
    let q_off_d = (q_pos_d * q_stride + head_idx * head_dim) as usize;
    let dims_base = (lane * 8u32) as usize;

    // Lane dims, row A (hand-unrolled: statically-indexed scalars stay in
    // registers — the Batch-49 local-memory-demotion class).
    let qa0 = if active_a { query[q_off_a + dims_base] } else { f32::new(0.0f32) };
    let qa1 = if active_a { query[q_off_a + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qa2 = if active_a { query[q_off_a + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qa3 = if active_a { query[q_off_a + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qa4 = if active_a { query[q_off_a + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qa5 = if active_a { query[q_off_a + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qa6 = if active_a { query[q_off_a + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qa7 = if active_a { query[q_off_a + dims_base + 7usize] } else { f32::new(0.0f32) };
    // Row B.
    let qb0 = if active_b { query[q_off_b + dims_base] } else { f32::new(0.0f32) };
    let qb1 = if active_b { query[q_off_b + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qb2 = if active_b { query[q_off_b + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qb3 = if active_b { query[q_off_b + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qb4 = if active_b { query[q_off_b + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qb5 = if active_b { query[q_off_b + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qb6 = if active_b { query[q_off_b + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qb7 = if active_b { query[q_off_b + dims_base + 7usize] } else { f32::new(0.0f32) };
    // Row C.
    let qc0 = if active_c { query[q_off_c + dims_base] } else { f32::new(0.0f32) };
    let qc1 = if active_c { query[q_off_c + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qc2 = if active_c { query[q_off_c + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qc3 = if active_c { query[q_off_c + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qc4 = if active_c { query[q_off_c + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qc5 = if active_c { query[q_off_c + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qc6 = if active_c { query[q_off_c + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qc7 = if active_c { query[q_off_c + dims_base + 7usize] } else { f32::new(0.0f32) };
    // Row D.
    let qd0 = if active_d { query[q_off_d + dims_base] } else { f32::new(0.0f32) };
    let qd1 = if active_d { query[q_off_d + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qd2 = if active_d { query[q_off_d + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qd3 = if active_d { query[q_off_d + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qd4 = if active_d { query[q_off_d + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qd5 = if active_d { query[q_off_d + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qd6 = if active_d { query[q_off_d + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qd7 = if active_d { query[q_off_d + dims_base + 7usize] } else { f32::new(0.0f32) };

    // Online softmax state per row, per lane (uniform across the plane after
    // every broadcast).
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

    // Uniform loop bound: the max causal position across the cube's 32
    // queries (+1). Out-of-causal positions mask to a bit-exact no-op.
    let q_last = q_tile * TILED_M32_KVF16_Q_PER_CUBE + (TILED_M32_KVF16_Q_PER_CUBE - 1u32);
    let n_loop = if q_last < p { q_last } else { p - 1u32 } + q_offset + 1u32;

    let mut pos = 0u32;
    while pos < n_loop {
        // ONE K row load shared by all four rows — the traffic-halving win,
        // now at HALF the byte width (f16 storage, f32 upconvert on load).
        let k_base = (pos * kv_stride + kv_head_off) as usize;
        let k0 = f32::cast_from(key[k_base + dims_base]);
        let k1 = f32::cast_from(key[k_base + dims_base + 1usize]);
        let k2 = f32::cast_from(key[k_base + dims_base + 2usize]);
        let k3 = f32::cast_from(key[k_base + dims_base + 3usize]);
        let k4 = f32::cast_from(key[k_base + dims_base + 4usize]);
        let k5 = f32::cast_from(key[k_base + dims_base + 5usize]);
        let k6 = f32::cast_from(key[k_base + dims_base + 6usize]);
        let k7 = f32::cast_from(key[k_base + dims_base + 7usize]);

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
        // is broadcast with one shuffle (the Bench 786 reduce-on-all-lanes
        // lesson; no shuffle inside a divergent branch).
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

        // ONE V row load shared by all four rows — f16 storage, f32 on load.
        let v_base = k_base;
        let v0 = f32::cast_from(value[v_base + dims_base]);
        let v1 = f32::cast_from(value[v_base + dims_base + 1usize]);
        let v2 = f32::cast_from(value[v_base + dims_base + 2usize]);
        let v3 = f32::cast_from(value[v_base + dims_base + 3usize]);
        let v4 = f32::cast_from(value[v_base + dims_base + 4usize]);
        let v5 = f32::cast_from(value[v_base + dims_base + 5usize]);
        let v6 = f32::cast_from(value[v_base + dims_base + 6usize]);
        let v7 = f32::cast_from(value[v_base + dims_base + 7usize]);

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
    // gate, one exp per output element), f32 verbatim.
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

/// Launch the M=32 f16-KV traffic-width tiled flash attention prefill
/// (Issue 844 T5-width). Launch contract = the m32 kernel's, with the K/V
/// handles carrying **f16 elements** (`(base_pos + p) * n_kv_head *
/// head_dim` elements — half the bytes); chunking, the 65535-workgroup
/// guard, and the params layout are identical. Requires `head_dim == 256`.
/// No production dispatch route yet — see the module doc (keep-arm record).
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionPrefillTiledM32KvF16CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionPrefillTiledM32KvF16CubeCL {
    /// # Safety
    /// Same contract as [`crate::QwenAttentionPrefillTiledM32CubeCL::launch`]
    /// except `key_handle`/`value_handle` must be f16-element buffers of
    /// `(base_pos + p) * n_kv_head * head_dim` elements. `head_dim` must be
    /// 256.
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
            "tiled m32-kvf16 flash kernel is head_dim-256 specialized"
        );
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Grid = n_head × q_tiles (32 queries per cube) — chunk on query-TILE
        // boundaries with the same 65535 guard as the m32 launcher.
        let q_tiles_total = p.div_ceil(TILED_M32_KVF16_Q_PER_CUBE as usize).max(1);
        let tiles_per_chunk = (MAX_WG_X as usize / n_head.max(1) / 32).max(1);
        let mut t0 = 0usize; // query-token base of the chunk
        while t0 < p {
            let tiles_left = q_tiles_total - t0.div_ceil(TILED_M32_KVF16_Q_PER_CUBE as usize);
            let tiles = tiles_per_chunk.min(tiles_left);
            let tc = (tiles * TILED_M32_KVF16_Q_PER_CUBE as usize).min(p - t0);
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
            // f16 ELEMENTS (the bytes halve; the element count matches m32's
            // f32-element count).
            let kv_len = (base_pos + p) * n_kv_head * head_dim;
            let q_slice = query_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64);
            let g_slice = gate_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64);
            let o_slice =
                attn_out_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64);
            let n_cubes = n_head * tiles;
            unsafe {
                qwen_attention_prefill_tiled_m32_kvf16_f32::launch_unchecked::<R>(
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
