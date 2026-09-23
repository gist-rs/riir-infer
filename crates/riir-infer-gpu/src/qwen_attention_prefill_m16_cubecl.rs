//! Issue 771 / Bench 808: the M=16 two-rows-per-plane variant of the tiled
//! flash-attention prefill (`qwen_attention_prefill_tiled_f32`, DEFAULT ON
//! since Bench 805).
//!
//! The tiled kernel gives each of its 8 planes ONE query row. This variant
//! gives each plane TWO ADJACENT query rows (`TILED_M16_Q_PER_CUBE = 16` per
//! cube) plus a lane-0 exp hoist, closing two of the three Bench 800 §3
//! levers in one bit-identical step:
//!
//! - **KV traffic ÷2 per row** — each cooperative K/V row load now serves 16
//!   query rows instead of 8 (the kernel's cost is dominated by per-position
//!   work; Bench 792 measured the stage at 63.2% of prefill @16K, and Bench
//!   800 measured the tiled kernel ~2.7× above its compute floor there).
//! - **Plane-uniform exp redundancy collapsed** — `w`/`correction` are
//!   plane-uniform values; the tiled kernel computes both exps on ALL 32
//!   lanes. Here lane 0 alone computes each exp and
//!   [`plane_broadcast`] ships it with ONE shuffle (Metal `simd_broadcast`;
//!   the Bench 786 lesson is honored: the broadcast runs on all lanes, only
//!   the expensive computation is lane-guarded).
//!
//! ## Numerics: BIT-IDENTICAL to the tiled kernel (the design claim)
//!
//! - Per-row dot: the same 8-FMA chain + `plane_sum` tree, unchanged.
//! - Exp hoist: lane 0 computes the SAME f32 value the all-lane exp produced
//!   (the operands are plane-uniform, `exp` is deterministic); the broadcast
//!   delivers exactly that value to every lane. No zero-add trickery — the
//!   broadcast IS the value.
//! - Loop-bound widening: `n_loop` covers the pair-set's last row (16 rows
//!   per cube instead of 8). Each row's extra out-of-causal iterations are
//!   bit-exact no-ops, exactly as in the tiled kernel (whose `n_loop` already
//!   spans the whole cube): masked = −1e30 → w = exp(−1e30 − finite) = 0,
//!   correction = exp(0) = 1, and `x·1 + 0` / `o·1 + 0·v` are exact in IEEE
//!   (o never holds −0: it starts +0 and every update adds a ≥0 term).
//!
//! Therefore the pinned tiled per-length FNV anchors (`99a0733c45a0e663`
//! @2048, `c152813ee93aaa2a` @16K) hold with this arm enabled — the @16K
//! anchor is the DEFAULT-state pin again since the 2026-08-30 cmma
//! demotion (the 08-29 cmma promotion had moved default routing at 16K to
//! the cmma arm, FNV `c303e9ff390aae0d`) — asserted at
//! kernel level by `tests/bench_808_issue771_flash_m16.rs` (bit-identity vs
//! the tiled kernel, flat + peaked data, multi-chunk query slicing) and at
//! model level by the e2e A/B's FNV-equality gate (transitivity through the
//! single consumer path, `prefill_attention_layer_batched` sub-stage 3).
//!
//! ## Toggle (the Bench 808 promotion)
//!
//! DEFAULT-ON for LONG prefills — the dispatch routes this arm when enabled
//! AND `p >= crate::TILED_FLASH_M16_MIN_P` (8192): the win is the Θ(P²)
//! KV-traffic term (measured 1.207× e2e @16384; wash @2048), so the gate
//! banks the proven win and keeps the default tiled below it (the
//! `PREFILL_USE_TALL_GEMM` length-gate precedent). Kill-switch env
//! `RIIR_PREFILL_TILED_FLASH_M16=0` (or `off`/`false`) restores the plain
//! tiled arm everywhere; the tiled kill-switch (`RIIR_PREFILL_TILED_FLASH=0`)
//! does not gate this arm. Because the arm is bit-identical to the tiled
//! kernel, the promotion moves NO anchor — the default-state delta is
//! wall-clock only.
//!
//! ## Dispatch contract
//!
//! Identical to [`crate::QwenAttentionPrefillTiledCubeCL`] except
//! `q_tiles = ceil(p / 16)` and the Metal 65535-workgroup chunk guard divides
//! by 16. Requires `head_dim == 256` (the launcher enforces; other head dims
//! fall back to the legacy kernel).

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

/// Queries per cube (= 2 per plane at `CubeDim::new_1d(256)`).
#[cfg(feature = "cubecl_runtime")]
const TILED_M16_Q_PER_CUBE: u32 = 16;

/// The M=16 two-rows-per-plane tiled causal flash-attention prefill
/// (Issue 771 / Bench 808). Same buffer contract and `params` layout as
/// [`crate::qwen_attention_cubecl::qwen_attention_prefill_tiled_f32`]:
/// `[head_dim, n_head, n_kv_head, p, scale, q_offset, q_tiles]`.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_prefill_tiled_m16_f32(
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

    // Each plane owns two ADJACENT query rows — the pair's causal bounds
    // diverge at exactly one position, so the shared-loop waste is ≤1
    // iteration per plane (a bit-exact no-op, see the module docs).
    let q_pos_a = q_tile * TILED_M16_Q_PER_CUBE + pl * 2u32;
    let q_pos_b = q_pos_a + 1u32;
    let active_a = q_pos_a < p;
    let active_b = q_pos_b < p;

    // GQA: map query head to key/value head group.
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_head_off = kv_group * head_dim;
    let q_stride = n_head * head_dim;
    let kv_stride = n_kv_head * head_dim;
    let q_abs_a = q_pos_a + q_offset;
    let q_abs_b = q_pos_b + q_offset;
    // Chunk-local bases of the two query rows (q/gate/attn_out are
    // chunk-sliced).
    let q_off_a = (q_pos_a * q_stride + head_idx * head_dim) as usize;
    let q_off_b = (q_pos_b * q_stride + head_idx * head_dim) as usize;
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
    // Lane dims, row B.
    let qb0 = if active_b { query[q_off_b + dims_base] } else { f32::new(0.0f32) };
    let qb1 = if active_b { query[q_off_b + dims_base + 1usize] } else { f32::new(0.0f32) };
    let qb2 = if active_b { query[q_off_b + dims_base + 2usize] } else { f32::new(0.0f32) };
    let qb3 = if active_b { query[q_off_b + dims_base + 3usize] } else { f32::new(0.0f32) };
    let qb4 = if active_b { query[q_off_b + dims_base + 4usize] } else { f32::new(0.0f32) };
    let qb5 = if active_b { query[q_off_b + dims_base + 5usize] } else { f32::new(0.0f32) };
    let qb6 = if active_b { query[q_off_b + dims_base + 6usize] } else { f32::new(0.0f32) };
    let qb7 = if active_b { query[q_off_b + dims_base + 7usize] } else { f32::new(0.0f32) };

    // Online softmax state per row, per lane (uniform across the plane after
    // every broadcast).
    let mut run_max_a = f32::new(-1e30f32);
    let mut run_sum_a = f32::new(0.0f32);
    let mut run_max_b = f32::new(-1e30f32);
    let mut run_sum_b = f32::new(0.0f32);
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

    // Uniform loop bound: the max causal position across the cube's 16
    // queries (+1). Out-of-causal positions mask to a bit-exact no-op.
    let q_last = q_tile * TILED_M16_Q_PER_CUBE + (TILED_M16_Q_PER_CUBE - 1u32);
    let n_loop = if q_last < p { q_last } else { p - 1u32 } + q_offset + 1u32;

    let mut pos = 0u32;
    while pos < n_loop {
        // ONE K row load shared by both rows — the traffic-halving win.
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
        // Reduce + broadcast within this plane's 32 lanes (tree order — the
        // same FP-equivalent class as the tiled kernel, unchanged per row).
        let score_a = plane_sum(partial_a) * scale;
        let score_b = plane_sum(partial_b) * scale;

        let in_causal_a = active_a && pos <= q_abs_a;
        let in_causal_b = active_b && pos <= q_abs_b;
        let masked_a = if in_causal_a { score_a } else { f32::new(-1e30f32) };
        let masked_b = if in_causal_b { score_b } else { f32::new(-1e30f32) };

        let new_max_a = if masked_a > run_max_a { masked_a } else { run_max_a };
        let new_max_b = if masked_b > run_max_b { masked_b } else { run_max_b };

        // Exp hoist: each plane-uniform exp runs on lane 0 ONLY and is
        // broadcast with one shuffle (`plane_broadcast` lowers to a single
        // `simd_shuffle` on Metal). The broadcast itself executes on ALL
        // lanes — only the expensive computation is lane-guarded (the
        // Bench 786 reduce-on-all-lanes lesson; no shuffle inside a
        // divergent branch).
        let w_a0 = if lane == 0u32 { (masked_a - new_max_a).exp() } else { f32::new(0.0f32) };
        let corr_a0 = if lane == 0u32 { (run_max_a - new_max_a).exp() } else { f32::new(0.0f32) };
        let w_b0 = if lane == 0u32 { (masked_b - new_max_b).exp() } else { f32::new(0.0f32) };
        let corr_b0 = if lane == 0u32 { (run_max_b - new_max_b).exp() } else { f32::new(0.0f32) };
        let w_a = plane_broadcast(w_a0, 0u32);
        let corr_a = plane_broadcast(corr_a0, 0u32);
        let w_b = plane_broadcast(w_b0, 0u32);
        let corr_b = plane_broadcast(corr_b0, 0u32);

        run_sum_a = run_sum_a * corr_a + w_a;
        run_sum_b = run_sum_b * corr_b + w_b;

        // ONE V row load shared by both rows.
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
        run_max_a = new_max_a;
        run_max_b = new_max_b;

        pos += 1u32;
    }

    // Epilogue, row A then row B — the same gated-attention shape as the
    // tiled kernel (sigmoid gate, one exp per output element).
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
}

/// Launch the M=16 two-rows-per-plane tiled flash attention prefill
/// (Issue 771 / Bench 808). Same buffer contract as
/// [`crate::QwenAttentionPrefillTiledCubeCL::launch`], including the chunked
/// `base_pos` cache semantics — callers route through
/// [`crate::ternary_deltanet_gpu_forward::prefill_tiled_flash_m16_enabled`]
/// (DEFAULT-ON for `p >= crate::TILED_FLASH_M16_MIN_P`; head_dim must be
/// 256).
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionPrefillTiledM16CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionPrefillTiledM16CubeCL {
    /// # Safety
    /// Same contract as [`crate::QwenAttentionPrefillTiledCubeCL::launch`] —
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
            "tiled m16 flash kernel is head_dim-256 specialized"
        );
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Grid = n_head × q_tiles (16 queries per cube) — chunk on query-TILE
        // boundaries with the same 65535 guard as the tiled launcher
        // (65535 / 24 / 16 = 170 tiles per chunk at the Bonsai shape).
        let q_tiles_total = p.div_ceil(TILED_M16_Q_PER_CUBE as usize).max(1);
        let tiles_per_chunk = (MAX_WG_X as usize / n_head.max(1) / 16).max(1);
        let mut t0 = 0usize; // query-token base of the chunk
        while t0 < p {
            let tiles_left = q_tiles_total - t0.div_ceil(TILED_M16_Q_PER_CUBE as usize);
            let tiles = tiles_per_chunk.min(tiles_left);
            let tc = (tiles * TILED_M16_Q_PER_CUBE as usize).min(p - t0);
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
                qwen_attention_prefill_tiled_m16_f32::launch_unchecked::<R>(
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
