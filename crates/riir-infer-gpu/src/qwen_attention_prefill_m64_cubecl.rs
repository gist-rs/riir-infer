//! Issue 844 T5: the M=64 eight-rows-per-plane variant of the tiled
//! flash-attention prefill — the second step of the KV-traffic amortization
//! lever the Bench 843 phase map ordered (T2 built m32; its own note named
//! "m32/m64-class" as the lever class).
//!
//! ## ⛔ MEASURED NO-GO (2026-09-10, keep-arm record — do not promote)
//!
//! The pre-registered kernel gate (≥3% cooled vs the production m32 arm)
//! **FAILED decisively, reproduced ×2** (interleaved median-of-3, 20 s
//! cooldowns, M3 Max, this exact production kernel):
//! m64p vs m32p = **−75.7% @8192, −75.9% @16384, −65.3% @32768** — m64
//! sits at ~1.0 TF effective vs m32's 1.6–1.7 TF. The signature is the
//! register-spill class named below: ~160 live f32 values per lane
//! exceeds the per-thread register budget, the per-row accumulators demote
//! to local memory, and all 8 serial chains pay the round-trip every
//! iteration — swamping the KV-bytes-per-row-pair halving. It also matches
//! T1 F4's latency wall: m32's win came largely from MORE independent
//! chains (16→32); m64 doubles per-chain arithmetic instead. **The "widen
//! the Q-block further" branch of the KV-amortization lever is closed by
//! measurement.** The arm stays compiled (opt-in only, env
//! `RIIR_PREFILL_TILED_FLASH_M64=1`) as the artifact + the bit-identity
//! reference; promotion is OFF the table without a NEW mechanism (e.g.
//! smem-staged accumulators), not a re-measure.
//!
//! This kernel gives each of its 8 planes **EIGHT ADJACENT query rows**
//! (`64` per cube) — each cooperative K/V row load now serves 64 query rows
//! instead of m32's 32, halving unique KV bytes per row-pair again (the
//! T2 projection: the full amortization ≈ −9%/−12% e2e @16K/32K, of which
//! m32 captured the first slice). The octet's causal bounds diverge across
//! at most seven positions, so the shared-loop waste is ≤7 bit-exact no-op
//! iterations per plane.
//!
//! ## Numerics: BIT-IDENTICAL to the m32/m16 kernels (the measured class)
//!
//! Per-row arithmetic (dot chain, `plane_sum` tree, online softmax, PV
//! chain, gated epilogue) is the m32 shape **verbatim** per row; the widened
//! `n_loop` adds only bit-exact no-op iterations (masked = −1e30 →
//! w = exp(−1e30 − finite) = 0, correction = 1, and `x·1 + 0` / `o·1 + 0·v`
//! are exact in IEEE) — the same kernel-to-kernel bit-identity class the m16
//! → m32 pair MEASURED at 0 bit-diffs (the m32 module doc; the probe twin
//! re-measures it for this pair).
//!
//! ## Register pressure IS the experiment
//!
//! ~160 live f32 values per lane (64 q + 64 o + 16 softmax state + K/V)
//! vs ~90 at m32 — if the compiler demotes static-indexed scalars to local
//! memory (the Batch-49 class the hand-unrolling exists to prevent), the
//! spill shows up directly as kernel time and the probe records the honest
//! negative. No production change ships without the probe's cooled-window
//! verdict.
//!
//! ## Toggle: OPT-IN ONLY, default OFF on every platform — and now measured NO-GO
//!
//! Env `RIIR_PREFILL_TILED_FLASH_M64=1`/`on`/`true` opts in (unset or any
//! other value keeps the m32/m16 chain). The promotion ladder below was
//! run to its first gate and failed it (the NO-GO record at the top of
//! this doc): the arm remains available for replication and as the
//! bit-identity reference, never as a default.
//!
//! ## Dispatch contract
//!
//! Identical to [`crate::QwenAttentionPrefillTiledM32CubeCL`] except
//! `q_tiles = ceil(p / 64)` and the Metal 65535-workgroup chunk guard
//! divides by 64 (42 tiles per chunk at the Bonsai shape). Requires
//! `head_dim == 256` (the launcher enforces; other head dims fall through
//! the dispatch chain to the m32/m16/tiled/legacy kernels).

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

/// Queries per cube (= 8 per plane at `CubeDim::new_1d(256)`).
#[cfg(feature = "cubecl_runtime")]
const TILED_M64_Q_PER_CUBE: u32 = 64;

/// The M=64 eight-rows-per-plane tiled causal flash-attention prefill
/// (Issue 844 T5). Same buffer contract and `params` layout as
/// [`crate::qwen_attention_prefill_m32_cubecl::qwen_attention_prefill_tiled_m32_f32`]:
/// `[head_dim, n_head, n_kv_head, p, scale, q_offset, q_tiles]`.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_prefill_tiled_m64_f32(
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

    // Each plane owns EIGHT ADJACENT query rows — the octet's causal bounds
    // diverge across at most seven positions, so the shared-loop waste is ≤7
    // bit-exact no-op iterations per plane.
    let q_base = q_tile * TILED_M64_Q_PER_CUBE + pl * 8u32;
    let q_pos_a = q_base;
    let q_pos_b = q_base + 1u32;
    let q_pos_c = q_base + 2u32;
    let q_pos_d = q_base + 3u32;
    let q_pos_e = q_base + 4u32;
    let q_pos_f = q_base + 5u32;
    let q_pos_g = q_base + 6u32;
    let q_pos_h = q_base + 7u32;
    let active_a = q_pos_a < p;
    let active_b = q_pos_b < p;
    let active_c = q_pos_c < p;
    let active_d = q_pos_d < p;
    let active_e = q_pos_e < p;
    let active_f = q_pos_f < p;
    let active_g = q_pos_g < p;
    let active_h = q_pos_h < p;

    // GQA: map query head to key/value head group.
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_head_off = kv_group * head_dim;
    let q_stride = n_head * head_dim;
    let kv_stride = n_kv_head * head_dim;
    let q_abs_a = q_pos_a + q_offset;
    let q_abs_b = q_pos_b + q_offset;
    let q_abs_c = q_pos_c + q_offset;
    let q_abs_d = q_pos_d + q_offset;
    let q_abs_e = q_pos_e + q_offset;
    let q_abs_f = q_pos_f + q_offset;
    let q_abs_g = q_pos_g + q_offset;
    let q_abs_h = q_pos_h + q_offset;
    // Chunk-local bases of the eight query rows (q/gate/attn_out are
    // chunk-sliced).
    let q_off_a = (q_pos_a * q_stride + head_idx * head_dim) as usize;
    let q_off_b = (q_pos_b * q_stride + head_idx * head_dim) as usize;
    let q_off_c = (q_pos_c * q_stride + head_idx * head_dim) as usize;
    let q_off_d = (q_pos_d * q_stride + head_idx * head_dim) as usize;
    let q_off_e = (q_pos_e * q_stride + head_idx * head_dim) as usize;
    let q_off_f = (q_pos_f * q_stride + head_idx * head_dim) as usize;
    let q_off_g = (q_pos_g * q_stride + head_idx * head_dim) as usize;
    let q_off_h = (q_pos_h * q_stride + head_idx * head_dim) as usize;
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
    // Row E.
    let qe0 = if active_e { query[q_off_e + dims_base] } else { f32::new(0.0f32) };
    let qe1 = if active_e { query[q_off_e + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qe2 = if active_e { query[q_off_e + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qe3 = if active_e { query[q_off_e + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qe4 = if active_e { query[q_off_e + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qe5 = if active_e { query[q_off_e + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qe6 = if active_e { query[q_off_e + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qe7 = if active_e { query[q_off_e + dims_base + 7usize] } else { f32::new(0.0f32) };
    // Row F.
    let qf0 = if active_f { query[q_off_f + dims_base] } else { f32::new(0.0f32) };
    let qf1 = if active_f { query[q_off_f + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qf2 = if active_f { query[q_off_f + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qf3 = if active_f { query[q_off_f + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qf4 = if active_f { query[q_off_f + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qf5 = if active_f { query[q_off_f + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qf6 = if active_f { query[q_off_f + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qf7 = if active_f { query[q_off_f + dims_base + 7usize] } else { f32::new(0.0f32) };
    // Row G.
    let qg0 = if active_g { query[q_off_g + dims_base] } else { f32::new(0.0f32) };
    let qg1 = if active_g { query[q_off_g + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qg2 = if active_g { query[q_off_g + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qg3 = if active_g { query[q_off_g + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qg4 = if active_g { query[q_off_g + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qg5 = if active_g { query[q_off_g + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qg6 = if active_g { query[q_off_g + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qg7 = if active_g { query[q_off_g + dims_base + 7usize] } else { f32::new(0.0f32) };
    // Row H.
    let qh0 = if active_h { query[q_off_h + dims_base] } else { f32::new(0.0f32) };
    let qh1 = if active_h { query[q_off_h + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qh2 = if active_h { query[q_off_h + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qh3 = if active_h { query[q_off_h + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qh4 = if active_h { query[q_off_h + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qh5 = if active_h { query[q_off_h + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qh6 = if active_h { query[q_off_h + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qh7 = if active_h { query[q_off_h + dims_base + 7usize] } else { f32::new(0.0f32) };

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
    let mut run_max_e = f32::new(-1e30f32);
    let mut run_sum_e = f32::new(0.0f32);
    let mut run_max_f = f32::new(-1e30f32);
    let mut run_sum_f = f32::new(0.0f32);
    let mut run_max_g = f32::new(-1e30f32);
    let mut run_sum_g = f32::new(0.0f32);
    let mut run_max_h = f32::new(-1e30f32);
    let mut run_sum_h = f32::new(0.0f32);
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
    let mut oe0 = f32::new(0.0f32);
    let mut oe1 = f32::new(0.0f32);
    let mut oe2 = f32::new(0.0f32);
    let mut oe3 = f32::new(0.0f32);
    let mut oe4 = f32::new(0.0f32);
    let mut oe5 = f32::new(0.0f32);
    let mut oe6 = f32::new(0.0f32);
    let mut oe7 = f32::new(0.0f32);
    let mut of0 = f32::new(0.0f32);
    let mut of1 = f32::new(0.0f32);
    let mut of2 = f32::new(0.0f32);
    let mut of3 = f32::new(0.0f32);
    let mut of4 = f32::new(0.0f32);
    let mut of5 = f32::new(0.0f32);
    let mut of6 = f32::new(0.0f32);
    let mut of7 = f32::new(0.0f32);
    let mut og0 = f32::new(0.0f32);
    let mut og1 = f32::new(0.0f32);
    let mut og2 = f32::new(0.0f32);
    let mut og3 = f32::new(0.0f32);
    let mut og4 = f32::new(0.0f32);
    let mut og5 = f32::new(0.0f32);
    let mut og6 = f32::new(0.0f32);
    let mut og7 = f32::new(0.0f32);
    let mut oh0 = f32::new(0.0f32);
    let mut oh1 = f32::new(0.0f32);
    let mut oh2 = f32::new(0.0f32);
    let mut oh3 = f32::new(0.0f32);
    let mut oh4 = f32::new(0.0f32);
    let mut oh5 = f32::new(0.0f32);
    let mut oh6 = f32::new(0.0f32);
    let mut oh7 = f32::new(0.0f32);

    // Uniform loop bound: the max causal position across the cube's 64
    // queries (+1). Out-of-causal positions mask to a bit-exact no-op.
    let q_last = q_tile * TILED_M64_Q_PER_CUBE + (TILED_M64_Q_PER_CUBE - 1u32);
    let n_loop = if q_last < p { q_last } else { p - 1u32 } + q_offset + 1u32;

    let mut pos = 0u32;
    while pos < n_loop {
        // ONE K row load shared by all eight rows — the traffic win.
        let k_base = (pos * kv_stride + kv_head_off) as usize;
        let k0 = key[k_base + dims_base];
        let k1 = key[k_base + dims_base + 1usize];
        let k2 = key[k_base + dims_base + 2usize];
        let k3 = key[k_base + dims_base + 3usize];
        let k4 = key[k_base + dims_base + 4usize];
        let k5 = key[k_base + dims_base + 5usize];
        let k6 = key[k_base + dims_base + 6usize];
        let k7 = key[k_base + dims_base + 7usize];

        let partial_a = qa0 * k0 + qa1 * k1 + qa2 * k2 + qa3 * k3
            + qa4 * k4 + qa5 * k5 + qa6 * k6 + qa7 * k7;
        let partial_b = qb0 * k0 + qb1 * k1 + qb2 * k2 + qb3 * k3
            + qb4 * k4 + qb5 * k5 + qb6 * k6 + qb7 * k7;
        let partial_c = qc0 * k0 + qc1 * k1 + qc2 * k2 + qc3 * k3
            + qc4 * k4 + qc5 * k5 + qc6 * k6 + qc7 * k7;
        let partial_d = qd0 * k0 + qd1 * k1 + qd2 * k2 + qd3 * k3
            + qd4 * k4 + qd5 * k5 + qd6 * k6 + qd7 * k7;
        let partial_e = qe0 * k0 + qe1 * k1 + qe2 * k2 + qe3 * k3
            + qe4 * k4 + qe5 * k5 + qe6 * k6 + qe7 * k7;
        let partial_f = qf0 * k0 + qf1 * k1 + qf2 * k2 + qf3 * k3
            + qf4 * k4 + qf5 * k5 + qf6 * k6 + qf7 * k7;
        let partial_g = qg0 * k0 + qg1 * k1 + qg2 * k2 + qg3 * k3
            + qg4 * k4 + qg5 * k5 + qg6 * k6 + qg7 * k7;
        let partial_h = qh0 * k0 + qh1 * k1 + qh2 * k2 + qh3 * k3
            + qh4 * k4 + qh5 * k5 + qh6 * k6 + qh7 * k7;
        // Reduce + broadcast within this plane's 32 lanes (tree order — the
        // same FP-equivalent class as the m32 kernel, unchanged per row).
        let score_a = plane_sum(partial_a) * scale;
        let score_b = plane_sum(partial_b) * scale;
        let score_c = plane_sum(partial_c) * scale;
        let score_d = plane_sum(partial_d) * scale;
        let score_e = plane_sum(partial_e) * scale;
        let score_f = plane_sum(partial_f) * scale;
        let score_g = plane_sum(partial_g) * scale;
        let score_h = plane_sum(partial_h) * scale;

        let in_causal_a = active_a && pos <= q_abs_a;
        let in_causal_b = active_b && pos <= q_abs_b;
        let in_causal_c = active_c && pos <= q_abs_c;
        let in_causal_d = active_d && pos <= q_abs_d;
        let in_causal_e = active_e && pos <= q_abs_e;
        let in_causal_f = active_f && pos <= q_abs_f;
        let in_causal_g = active_g && pos <= q_abs_g;
        let in_causal_h = active_h && pos <= q_abs_h;
        let masked_a = if in_causal_a { score_a } else { f32::new(-1e30f32) };
        let masked_b = if in_causal_b { score_b } else { f32::new(-1e30f32) };
        let masked_c = if in_causal_c { score_c } else { f32::new(-1e30f32) };
        let masked_d = if in_causal_d { score_d } else { f32::new(-1e30f32) };
        let masked_e = if in_causal_e { score_e } else { f32::new(-1e30f32) };
        let masked_f = if in_causal_f { score_f } else { f32::new(-1e30f32) };
        let masked_g = if in_causal_g { score_g } else { f32::new(-1e30f32) };
        let masked_h = if in_causal_h { score_h } else { f32::new(-1e30f32) };

        let new_max_a = if masked_a > run_max_a { masked_a } else { run_max_a };
        let new_max_b = if masked_b > run_max_b { masked_b } else { run_max_b };
        let new_max_c = if masked_c > run_max_c { masked_c } else { run_max_c };
        let new_max_d = if masked_d > run_max_d { masked_d } else { run_max_d };
        let new_max_e = if masked_e > run_max_e { masked_e } else { run_max_e };
        let new_max_f = if masked_f > run_max_f { masked_f } else { run_max_f };
        let new_max_g = if masked_g > run_max_g { masked_g } else { run_max_g };
        let new_max_h = if masked_h > run_max_h { masked_h } else { run_max_h };

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
        let w_e0 = if lane == 0u32 { (masked_e - new_max_e).exp() } else { f32::new(0.0f32) };
        let corr_e0 = if lane == 0u32 { (run_max_e - new_max_e).exp() } else { f32::new(0.0f32) };
        let w_f0 = if lane == 0u32 { (masked_f - new_max_f).exp() } else { f32::new(0.0f32) };
        let corr_f0 = if lane == 0u32 { (run_max_f - new_max_f).exp() } else { f32::new(0.0f32) };
        let w_g0 = if lane == 0u32 { (masked_g - new_max_g).exp() } else { f32::new(0.0f32) };
        let corr_g0 = if lane == 0u32 { (run_max_g - new_max_g).exp() } else { f32::new(0.0f32) };
        let w_h0 = if lane == 0u32 { (masked_h - new_max_h).exp() } else { f32::new(0.0f32) };
        let corr_h0 = if lane == 0u32 { (run_max_h - new_max_h).exp() } else { f32::new(0.0f32) };
        let w_a = plane_broadcast(w_a0, 0u32);
        let corr_a = plane_broadcast(corr_a0, 0u32);
        let w_b = plane_broadcast(w_b0, 0u32);
        let corr_b = plane_broadcast(corr_b0, 0u32);
        let w_c = plane_broadcast(w_c0, 0u32);
        let corr_c = plane_broadcast(corr_c0, 0u32);
        let w_d = plane_broadcast(w_d0, 0u32);
        let corr_d = plane_broadcast(corr_d0, 0u32);
        let w_e = plane_broadcast(w_e0, 0u32);
        let corr_e = plane_broadcast(corr_e0, 0u32);
        let w_f = plane_broadcast(w_f0, 0u32);
        let corr_f = plane_broadcast(corr_f0, 0u32);
        let w_g = plane_broadcast(w_g0, 0u32);
        let corr_g = plane_broadcast(corr_g0, 0u32);
        let w_h = plane_broadcast(w_h0, 0u32);
        let corr_h = plane_broadcast(corr_h0, 0u32);

        run_sum_a = run_sum_a * corr_a + w_a;
        run_sum_b = run_sum_b * corr_b + w_b;
        run_sum_c = run_sum_c * corr_c + w_c;
        run_sum_d = run_sum_d * corr_d + w_d;
        run_sum_e = run_sum_e * corr_e + w_e;
        run_sum_f = run_sum_f * corr_f + w_f;
        run_sum_g = run_sum_g * corr_g + w_g;
        run_sum_h = run_sum_h * corr_h + w_h;

        // ONE V row load shared by all eight rows.
        let v_base = k_base;
        let v0 = value[v_base + dims_base];
        let v1 = value[v_base + dims_base + 1usize];
        let v2 = value[v_base + dims_base + 2usize];
        let v3 = value[v_base + dims_base + 3usize];
        let v4 = value[v_base + dims_base + 4usize];
        let v5 = value[v_base + dims_base + 5usize];
        let v6 = value[v_base + dims_base + 6usize];
        let v7 = value[v_base + dims_base + 7usize];

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
        oe0 = oe0 * corr_e + w_e * v0;
        oe1 = oe1 * corr_e + w_e * v1;
        oe2 = oe2 * corr_e + w_e * v2;
        oe3 = oe3 * corr_e + w_e * v3;
        oe4 = oe4 * corr_e + w_e * v4;
        oe5 = oe5 * corr_e + w_e * v5;
        oe6 = oe6 * corr_e + w_e * v6;
        oe7 = oe7 * corr_e + w_e * v7;
        of0 = of0 * corr_f + w_f * v0;
        of1 = of1 * corr_f + w_f * v1;
        of2 = of2 * corr_f + w_f * v2;
        of3 = of3 * corr_f + w_f * v3;
        of4 = of4 * corr_f + w_f * v4;
        of5 = of5 * corr_f + w_f * v5;
        of6 = of6 * corr_f + w_f * v6;
        of7 = of7 * corr_f + w_f * v7;
        og0 = og0 * corr_g + w_g * v0;
        og1 = og1 * corr_g + w_g * v1;
        og2 = og2 * corr_g + w_g * v2;
        og3 = og3 * corr_g + w_g * v3;
        og4 = og4 * corr_g + w_g * v4;
        og5 = og5 * corr_g + w_g * v5;
        og6 = og6 * corr_g + w_g * v6;
        og7 = og7 * corr_g + w_g * v7;
        oh0 = oh0 * corr_h + w_h * v0;
        oh1 = oh1 * corr_h + w_h * v1;
        oh2 = oh2 * corr_h + w_h * v2;
        oh3 = oh3 * corr_h + w_h * v3;
        oh4 = oh4 * corr_h + w_h * v4;
        oh5 = oh5 * corr_h + w_h * v5;
        oh6 = oh6 * corr_h + w_h * v6;
        oh7 = oh7 * corr_h + w_h * v7;
        run_max_a = new_max_a;
        run_max_b = new_max_b;
        run_max_c = new_max_c;
        run_max_d = new_max_d;
        run_max_e = new_max_e;
        run_max_f = new_max_f;
        run_max_g = new_max_g;
        run_max_h = new_max_h;

        pos += 1u32;
    }

    // Epilogue, rows A-H — the m32 gated-attention shape per row (sigmoid
    // gate, one exp per output element).
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
    if active_e {
        let inv_sum = f32::new(1.0f32) / run_sum_e;
        let g_off = q_off_e + dims_base;
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
        attn_out[g_off] = oe0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = oe1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = oe2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = oe3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = oe4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = oe5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = oe6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = oe7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
    if active_f {
        let inv_sum = f32::new(1.0f32) / run_sum_f;
        let g_off = q_off_f + dims_base;
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
        attn_out[g_off] = of0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = of1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = of2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = of3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = of4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = of5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = of6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = of7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
    if active_g {
        let inv_sum = f32::new(1.0f32) / run_sum_g;
        let g_off = q_off_g + dims_base;
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
        attn_out[g_off] = og0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = og1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = og2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = og3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = og4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = og5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = og6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = og7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
    if active_h {
        let inv_sum = f32::new(1.0f32) / run_sum_h;
        let g_off = q_off_h + dims_base;
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
        attn_out[g_off] = oh0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = oh1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = oh2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = oh3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = oh4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = oh5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = oh6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = oh7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
}

/// Launch the M=64 eight-rows-per-plane tiled flash attention prefill
/// (Issue 844 T5). Same buffer contract as
/// [`crate::QwenAttentionPrefillTiledM32CubeCL::launch`], including the
/// chunked `base_pos` cache semantics — callers route through
/// [`crate::ternary_deltanet_gpu_forward::prefill_tiled_flash_m64_enabled`]
/// (OPT-IN ONLY — default OFF everywhere; head_dim must be 256).
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionPrefillTiledM64CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionPrefillTiledM64CubeCL {
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
            "tiled m64 flash kernel is head_dim-256 specialized"
        );
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Grid = n_head × q_tiles (64 queries per cube) — chunk on query-TILE
        // boundaries with the same 65535 guard as the m32 launcher
        // (65535 / 24 / 64 = 42 tiles per chunk at the Bonsai shape).
        let q_tiles_total = p.div_ceil(TILED_M64_Q_PER_CUBE as usize).max(1);
        let tiles_per_chunk = (MAX_WG_X as usize / n_head.max(1) / 64).max(1);
        let mut t0 = 0usize; // query-token base of the chunk
        while t0 < p {
            let tiles_left = q_tiles_total - t0.div_ceil(TILED_M64_Q_PER_CUBE as usize);
            let tiles = tiles_per_chunk.min(tiles_left);
            let tc = (tiles * TILED_M64_Q_PER_CUBE as usize).min(p - t0);
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
                qwen_attention_prefill_tiled_m64_f32::launch_unchecked::<R>(
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
