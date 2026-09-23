//! Issue 771 / Bench 809: the cmma score-matrix tiled flash-attention prefill
//! (`qwen_attention_prefill_tiled_cmma_f32`) — the last of the Bench 800 §3
//! levers, and the first smem user in the tiled-flash family.
//!
//! The m16 kernel (Bench 808) keeps the per-position SIMT loop: per K/V
//! position every lane issues 8+8 loads (shared across planes via L1 — an
//! 8× L1 re-read of the same K/V row), runs a `plane_sum` shuffle tree per
//! row, and serializes both rows' exps on lane 0. This kernel restructures
//! the loop into FA2 position TILES so that:
//!
//! - **K/V tiles are staged ONCE per cube** into threadgroup memory — the
//!   8× L1 re-read collapses to one coalesced global read per element per
//!   tile (the Bench 792 memory-bound verdict: flash is a KV re-read kernel
//!   with effective bandwidth far below peak; the m16 win was exactly a
//!   traffic-÷2, so traffic converts ~1:1 into e2e speedup).
//! - **Score dots run on the tensor core** — `cmma::execute` [8q × 8k] ×
//!   [8k × 8pos] per plane, replacing the 8-FMA + `plane_sum` chains (the
//!   Bench 786/800 lever; on the 4090 the score phase was 39% of the
//!   attention kernel and its T-a24 mma arm won 1.19× at kernel level).
//! - **The exps go tile-parallel** — 16 exps per tile spread over 16 lanes
//!   instead of 2-per-position serialized on lane 0 (the m16 hoist's
//!   residual serialization).
//!
//! ## Numerics: TOLERANCE ARM (NOT bit-identical — by design)
//!
//! The cmma dot accumulates in fragment-tree order and the softmax is
//! restructured from position-order online to tile-order online (tile max,
//! per-tile correction) — both are FP-equivalent reorderings of the same
//! math, but not bit-identical. Gated as the Bench 800 tolerance class:
//! max_abs band + f64 CPU oracle + peaked/massive-channel fixtures (the
//! Bench 786 vacuous-gate lesson). **DEFAULT OFF** — promotion moves every
//! prefill FNV anchor and is the owner's call after the e2e A/B (the Bench
//! 800 → 805 precedent).
//!
//! ## Layout (the 32 KB Apple threadgroup budget binds every choice)
//!
//! - Cube = `CubeDim::new_1d(256)` = 8 planes × 32 lanes; **M = 16 query
//!   rows per cube** (2 real rows per plane — the m16 q-mapping, so the G1
//!   fixtures are directly comparable).
//! - `q_tile` [16 × 256] f32 (16 KB) staged once per cube; each plane's
//!   cmma A-window is its aligned 8-row half (rows 0..8 or 8..16 — the
//!   window is padded, the plane reads only its 2 real rows back).
//! - `kv_tile` [8 × 256] f32 (8 KB): staged with the tile's K rows for the
//!   score phase, then OVERWRITTEN with the tile's V rows (overlapped with
//!   the softmax phase — disjoint buffers, no extra barrier).
//! - `s_tile`/`p_tile`: plane-private [8 × 8] f32 each (2 × 2 KB) — the
//!   cmma score store and the softmax P write (separate buffers so the
//!   softmax needs no barrier before its writes).
//! - Total 7,168 f32 = **28 KB** of the 32 KB budget; **3 `sync_cube` per
//!   8-position tile** (K staged → scores+softmax/V staged → PV). Separate
//!   K and V buffers would need 36 KB (over budget); N=16 tiles via
//!   global-slice cmma operands are the recorded v2 levers.
//! - O stays in per-lane scalar registers (8 dims × 2 rows) exactly like
//!   m16 — the per-row online-softmax correction multiplies scalars, not
//!   cmma fragments (fragment rescaling has no API on this CubeCL version).
//!
//! ## Dispatch contract
//!
//! Identical to [`crate::QwenAttentionPrefillTiledM16CubeCL`] (same `params`
//! layout, `q_tiles = ceil(p / 16)`, 65535-workgroup chunk guard ÷16,
//! head_dim-256 specialization, chunked `base_pos` cache semantics). At the
//! dispatch site the arm takes precedence over m16 → tiled → legacy when
//! enabled. Requires `head_dim == 256`.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

/// Queries per cube (= 2 real rows per plane at `CubeDim::new_1d(256)`).
#[cfg(feature = "cubecl_runtime")]
const CMMA_Q_PER_CUBE: u32 = 16;

/// KV positions per tile. 8 is the largest N that fits the 32 KB budget
/// alongside the staged Q tile (see the module docs).
#[cfg(feature = "cubecl_runtime")]
const CMMA_N_TILE: u32 = 8;

/// K-dim steps per score accumulation (head_dim 256 / 8-wide cmma k-step).
#[cfg(feature = "cubecl_runtime")]
const CMMA_K_STEPS: u32 = 32;

/// The cmma score-matrix tiled causal flash-attention prefill (Issue 771 /
/// Bench 809). Same buffer contract and `params` layout as
/// [`crate::qwen_attention_prefill_m16_cubecl::qwen_attention_prefill_tiled_m16_f32`]:
/// `[head_dim, n_head, n_kv_head, p, scale, q_offset, q_tiles]`.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_prefill_tiled_cmma_f32(
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
    let q_tile_base = (cube_id % q_tiles) * CMMA_Q_PER_CUBE;

    // Plane/lane decomposition (simdgroup i = threads [32i, 32i+32)).
    let pl = UNIT_POS / 32u32;
    let lane = UNIT_POS_PLANE;
    let tid = UNIT_POS;

    // The plane's two REAL query rows (cube-local), the m16 q-mapping.
    let q_pos_a = q_tile_base + pl * 2u32;
    let q_pos_b = q_pos_a + 1u32;
    let active_a = q_pos_a < p;
    let active_b = q_pos_b < p;
    let q_abs_a = q_pos_a + q_offset;
    let q_abs_b = q_pos_b + q_offset;

    // GQA: map query head to key/value head group.
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_head_off = kv_group * head_dim;
    let q_stride = n_head * head_dim;
    let kv_stride = n_kv_head * head_dim;

    // Chunk-local bases of the two query rows (q/gate/attn_out are
    // chunk-sliced by the launcher).
    let q_off_a = (q_pos_a * q_stride + head_idx * head_dim) as usize;
    let q_off_b = (q_pos_b * q_stride + head_idx * head_dim) as usize;
    let dims_base = (lane * 8u32) as usize;

    // ── Shared memory (28 KB of the 32 KB Apple budget) ──
    // q_tile [16 × 256]: staged once. kv_tile [8 × 256]: K rows for the
    // score phase, then V rows (overlapped with softmax). s_tile / p_tile:
    // plane-private [8 × 8] score / probability regions.
    let mut q_tile = Shared::<[f32]>::new_slice(4096usize);
    let mut kv_tile = Shared::<[f32]>::new_slice(2048usize);
    let mut s_tile = Shared::<[f32]>::new_slice(512usize);
    let mut p_tile = Shared::<[f32]>::new_slice(512usize);
    let s_base = (pl * 64u32) as usize;
    // The plane's cmma A-window: its aligned 8-row half of q_tile. The real
    // rows sit at window-local (pl*2) % 8 and +1.
    let win_base = if pl >= 4u32 { 8u32 } else { 0u32 };
    let lr0 = ((pl * 2u32) & 7u32) as usize;
    let lr1 = lr0 + 1usize;

    // ── Stage the cube's 16 query rows once ──
    for j in 0..16u32 {
        let idx = tid + j * 256u32;
        let r = idx / 256u32;
        let d = idx % 256u32;
        let grow = q_tile_base + r;
        // Q is staged PRE-SCALED (the m16 kernel multiplies each plane_sum
        // result by `scale` instead — same math, one multiply per staged
        // element here; the fp-order delta is inside this arm's tolerance
        // class).
        q_tile[idx as usize] = if grow < p {
            query[((grow * q_stride + head_idx * head_dim) + d) as usize] * scale
        } else {
            f32::new(0.0f32)
        };
    }
    sync_cube();

    // Online softmax state per real row, per lane (plane-uniform after every
    // tile — all lanes compute the same reductions redundantly, no shuffles).
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
    // queries (+1) — the m16 formula.
    let q_last = q_tile_base + (CMMA_Q_PER_CUBE - 1u32);
    let n_loop = if q_last < p { q_last } else { p - 1u32 } + q_offset + 1u32;

    let mut pos0 = 0u32;
    while pos0 < n_loop {
        // ── Stage the tile's K rows ──
        for j in 0..8u32 {
            let idx = tid + j * 256u32;
            let jj = idx / 256u32;
            let d = idx % 256u32;
            let pos_abs = pos0 + jj;
            kv_tile[idx as usize] = if pos_abs < n_loop {
                key[((pos_abs * kv_stride + kv_head_off) + d) as usize]
            } else {
                f32::new(0.0f32)
            };
        }
        sync_cube(); // K staged for all planes

        // ── Score tile: acc_s = Q_win · K^T via the tensor core ──
        // Per plane: [8 rows × 8 k-dims] × [8 k-dims × 8 positions],
        // accumulated over 32 k-steps. The window's non-real rows carry
        // other planes' query data — never read back (plane-private s_tile).
        let acc_s = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::Undefined, 0.0f32,
        );
        let mut ki = 0u32;
        while ki < CMMA_K_STEPS {
            let mat_q = cmma::Matrix::<f32>::from_slice(
                cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
                cmma::MatrixLayout::RowMajor,
                q_tile.slice((win_base * 256u32 + ki * 8u32) as usize, 4096usize),
                256u32,
            );
            // kv_tile is [pos][dim] row-major; ColMajor B = K^T (the GEMM
            // kernels' staged-X idiom).
            let mat_k = cmma::Matrix::<f32>::from_slice(
                cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
                cmma::MatrixLayout::ColMajor,
                kv_tile.slice((ki * 8u32) as usize, 2048usize),
                256u32,
            );
            cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_q, &mat_k, &acc_s, &acc_s);
            ki += 1u32;
        }
        cmma::store(
            s_tile.slice_mut(s_base, s_base + 64usize),
            &acc_s, 8u32, cmma::MatrixLayout::RowMajor,
        );
        sync_cube(); // S stored; kv_tile free for V

        // ── Softmax (scalar, tile-order online) ∥ stage V ──
        // Disjoint buffers: softmax reads s_tile / writes p_tile while the
        // staging loop overwrites kv_tile with the tile's V rows — no extra
        // barrier between them (the one piece of overlap this layout buys).
        // Row max over the 8 masked scores (lane-redundant, no shuffles):
        let mut m0 = run_max_a;
        let mut m1 = run_max_b;
        for j in 0..8u32 {
            let pos_abs = pos0 + j;
            let s_a = s_tile[s_base + lr0 * 8 + j as usize];
            let s_b = s_tile[s_base + lr1 * 8 + j as usize];
            let masked_a = if active_a && pos_abs <= q_abs_a { s_a } else { f32::new(-1e30f32) };
            let masked_b = if active_b && pos_abs <= q_abs_b { s_b } else { f32::new(-1e30f32) };
            if masked_a > m0 { m0 = masked_a; }
            if masked_b > m1 { m1 = masked_b; }
        }
        let new_max_a = m0;
        let new_max_b = m1;
        // exp(0) = 1.0 exactly (MSL spec — the m16 kernel's masked no-op
        // invariant), so the branch is bit-equivalent to the unconditional
        // exp and removes the common-path correction exp entirely.
        let corr_a = if new_max_a > run_max_a {
            (run_max_a - new_max_a).exp()
        } else {
            f32::new(1.0f32)
        };
        let corr_b = if new_max_b > run_max_b {
            (run_max_b - new_max_b).exp()
        } else {
            f32::new(1.0f32)
        };

        // 16 exps spread over 16 lanes (the m16 kernel ran these 2-per-
        // position serialized on lane 0). Each exp lane reads its own score,
        // applies the causal mask, and writes P for the PV phase.
        if lane < 16u32 {
            let wrow = lane / 8u32;
            let wj = (lane % 8u32) as usize;
            let lr_r = if wrow == 0u32 { lr0 } else { lr1 };
            let (active_r, q_abs_r, new_max_r) = if wrow == 0u32 {
                (active_a, q_abs_a, new_max_a)
            } else {
                (active_b, q_abs_b, new_max_b)
            };
            let pos_abs = pos0 + wj as u32;
            let s = s_tile[s_base + lr_r * 8 + wj];
            let masked = if active_r && pos_abs <= q_abs_r { s } else { f32::new(-1e30f32) };
            p_tile[s_base + lr_r * 8 + wj] = (masked - new_max_r).exp();
        }
        for j in 0..8u32 {
            let idx = tid + j * 256u32;
            let jj = idx / 256u32;
            let d = idx % 256u32;
            let pos_abs = pos0 + jj;
            kv_tile[idx as usize] = if pos_abs < n_loop {
                value[((pos_abs * kv_stride + kv_head_off) + d) as usize]
            } else {
                f32::new(0.0f32)
            };
        }
        sync_cube(); // P written + V staged

        // ── PV: o = o·corr + Σ_j P[r,j]·V[j] (scalar, per-lane dims) ──
        // The online-softmax correction multiplies SCALAR registers — the
        // fragment-rescale problem never arises (no row-scale op on this
        // CubeCL cmma API; recorded as the PV-cmma successor lever).
        oa0 = oa0 * corr_a;
        oa1 = oa1 * corr_a;
        oa2 = oa2 * corr_a;
        oa3 = oa3 * corr_a;
        oa4 = oa4 * corr_a;
        oa5 = oa5 * corr_a;
        oa6 = oa6 * corr_a;
        oa7 = oa7 * corr_a;
        ob0 = ob0 * corr_b;
        ob1 = ob1 * corr_b;
        ob2 = ob2 * corr_b;
        ob3 = ob3 * corr_b;
        ob4 = ob4 * corr_b;
        ob5 = ob5 * corr_b;
        ob6 = ob6 * corr_b;
        ob7 = ob7 * corr_b;
        let mut ts_a = f32::new(0.0f32);
        let mut ts_b = f32::new(0.0f32);
        let mut jj = 0u32;
        while jj < CMMA_N_TILE {
            let wa = p_tile[s_base + lr0 * 8 + jj as usize];
            let wb = p_tile[s_base + lr1 * 8 + jj as usize];
            ts_a = ts_a + wa;
            ts_b = ts_b + wb;
            let v_base = (jj * 256u32) as usize + dims_base;
            let v0 = kv_tile[v_base];
            let v1 = kv_tile[v_base + 1usize];
            let v2 = kv_tile[v_base + 2usize];
            let v3 = kv_tile[v_base + 3usize];
            let v4 = kv_tile[v_base + 4usize];
            let v5 = kv_tile[v_base + 5usize];
            let v6 = kv_tile[v_base + 6usize];
            let v7 = kv_tile[v_base + 7usize];
            oa0 = oa0 + wa * v0;
            oa1 = oa1 + wa * v1;
            oa2 = oa2 + wa * v2;
            oa3 = oa3 + wa * v3;
            oa4 = oa4 + wa * v4;
            oa5 = oa5 + wa * v5;
            oa6 = oa6 + wa * v6;
            oa7 = oa7 + wa * v7;
            ob0 = ob0 + wb * v0;
            ob1 = ob1 + wb * v1;
            ob2 = ob2 + wb * v2;
            ob3 = ob3 + wb * v3;
            ob4 = ob4 + wb * v4;
            ob5 = ob5 + wb * v5;
            ob6 = ob6 + wb * v6;
            ob7 = ob7 + wb * v7;
            jj += 1u32;
        }
        run_sum_a = run_sum_a * corr_a + ts_a;
        run_sum_b = run_sum_b * corr_b + ts_b;
        run_max_a = new_max_a;
        run_max_b = new_max_b;

        pos0 += CMMA_N_TILE;
        // Loop-back barrier (the B57 smem-role-reversal class): the next
        // tile's K stage overwrites kv_tile, whose V rows THIS tile's PV
        // reads. Per-thread program order does not bound CROSS-WARP
        // progress — a fast warp can reach the next staging loop while a
        // slow warp is still in PV. The gate caught exactly this race
        // (p=300/base=256, max_abs 2.1e-2) before it could ship.
        sync_cube();
    }

    // Epilogue, row A then row B — the same gated-attention shape as the
    // tiled/m16 kernels (sigmoid gate, one exp per output element).
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

/// Launch the cmma score-matrix tiled flash attention prefill (Issue 771 /
/// Bench 809). Same buffer contract as
/// [`crate::QwenAttentionPrefillTiledM16CubeCL::launch`], including the
/// chunked `base_pos` cache semantics — callers route through
/// [`crate::ternary_deltanet_gpu_forward::prefill_tiled_flash_cmma_enabled`]
/// (macOS default-on at ≥16384 since the 2026-08-31 re-promotion —
/// tolerance arm; head_dim must be 256).
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionPrefillTiledCmmaCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionPrefillTiledCmmaCubeCL {
    /// Issue 828 shape-aware routing guard: whether the device registers the
    /// `(f32, f32, f32, 8, 8, 8)` cmma `MmaConfig` — the exact fragment shape
    /// BOTH cmma flash arms instantiate (`cmma::execute::<f32, f32, f32, f32>`
    /// over 8×8×8 `Matrix::<f32>` fragments: this score kernel AND the Bench
    /// 810 PV kernel). Metal's native simdgroup shape; NVIDIA's
    /// VK_KHR_cooperative_matrix exposure is f16/i8 at 16×16 (the Bench 706
    /// T6 record lists no f32 config), so without the guard an env opt-in on
    /// such a device fails the JIT config lookup at first dispatch.
    /// Same checker convention as
    /// [`crate::GemmTernarySimdgroupCubeCL::cmma_available`] /
    /// [`crate::MatmulSwapAb::cmma_available`]; call before selecting either
    /// arm so the dispatch falls through the cmma → m16 → tiled → legacy
    /// chain instead (the `MatmulSwapAb::launch_auto` pattern).
    pub fn cmma_available<R: Runtime>(client: &ComputeClient<R>) -> bool {
        use cubecl::ir::features::MmaConfig;
        use cubecl::ir::{ElemType, FloatKind};

        client.features().matmul.cmma.contains(&MmaConfig {
            a_type: ElemType::Float(FloatKind::F32).into(),
            b_type: ElemType::Float(FloatKind::F32).into(),
            cd_type: ElemType::Float(FloatKind::F32).into(),
            m: 8,
            n: 8,
            k: 8,
        })
    }

    /// # Safety
    /// Same contract as [`crate::QwenAttentionPrefillTiledM16CubeCL::launch`]
    /// — identical handle shapes and chunked `base_pos` semantics; `head_dim`
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
            "tiled cmma flash kernel is head_dim-256 specialized"
        );
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Grid = n_head × q_tiles (16 queries per cube) — chunk on query-TILE
        // boundaries with the same 65535 guard as the m16 launcher.
        let q_tiles_total = p.div_ceil(CMMA_Q_PER_CUBE as usize).max(1);
        let tiles_per_chunk = (MAX_WG_X as usize / n_head.max(1) / 16).max(1);
        let mut t0 = 0usize; // query-token base of the chunk
        while t0 < p {
            let tiles_left = q_tiles_total - t0.div_ceil(CMMA_Q_PER_CUBE as usize);
            let tiles = tiles_per_chunk.min(tiles_left);
            let tc = (tiles * CMMA_Q_PER_CUBE as usize).min(p - t0);
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
                qwen_attention_prefill_tiled_cmma_f32::launch_unchecked::<R>(
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
