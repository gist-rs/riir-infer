//! Issue 771 / Bench 810: the PV-cmma successor arm of the Bench 809 cmma
//! score-matrix tiled flash kernel (`qwen_attention_prefill_tiled_cmma_pv_f32`).
//!
//! Bench 809 moved the SCORE dots onto the tensor core but kept the PV phase
//! (`O += P·V`) as per-lane scalar FMAs: each lane owns 8 output dims per real
//! row and walks the 8 V rows of the tile from smem — ~224 instructions and 64
//! V smem loads per lane per tile, with an 8× smem amplification across the
//! cube's planes. This arm moves PV onto the tensor core too:
//!
//! - **O lives in 32 per-plane `[8×8]` cmma accumulator fragments** (one per
//!   8-dim column block of head_dim 256), created ONCE before the tile loop
//!   and accumulated across tiles by `cmma::execute` — the PV dot product
//!   never touches scalar registers and the V tile is read from smem by the
//!   tensor pipeline once per block (no per-lane re-read amplification).
//! - **The online-softmax correction (`O ·= corr`) is a RARE-EVENT smem
//!   round-trip.** `corr` differs from 1.0 only when the running max grows —
//!   for random-ish scores the max stabilizes after the first few tiles
//!   (expected updates ≈ ln(n_tiles)), so the steady state pays nothing. On
//!   the rare event, each accumulator is stored to the plane's (now-dead)
//!   `s_tile` region, scaled by 16 lanes, and loaded back with
//!   `Matrix::from_slice(MatrixIdent::Accumulator, …)` — the fragment reload
//!   the Bench 809 record called the missing API (verified present in
//!   cubecl-core 0.11.0-pre.2 `frontend/cmma.rs`; the MSL lowering is proven
//!   by this arm's G1 gates, the same standard as `cmma::fill` in Bench 773).
//! - The first tile skips the round-trip entirely: `corr` is 0 there (max
//!   moves off the −1e30 sentinel) but O is still all-zero, so scaling is a
//!   no-op.
//!
//! ## Numerics: TOLERANCE ARM (NOT bit-identical — by design)
//!
//! Same class as Bench 809: fragment-order dots (now in BOTH the score and
//! PV phases) plus tile-order online softmax. Gated as the Bench 800 family
//! tolerance class: max_abs band vs the cmma twin + f64 CPU oracle + peaked /
//! massive-channel / **late-max-update** fixtures (the late-max fixture
//! forces the round-trip path deep in the tile loop — without it the rescale
//! branch would be vacuous in every short-p gate). **DEFAULT OFF** — this arm
//! rides the Bench 809 promotion decision (it is a successor of a
//! default-off arm; its own promotion can only follow the parent's).
//!
//! ## Layout (unchanged 28 KB budget)
//!
//! Identical smem plan to Bench 809: `q_tile` [16×256] staged once (Q
//! pre-scaled), `kv_tile` [8×256] staged K then OVERWRITTEN with V
//! (overlapped with the softmax phase), plane-private `s_tile`/`p_tile`
//! [8×8] each. `p_tile` is zero-initialized once at kernel start — the PV
//! A-operand loads ALL 8 window rows from it, and the 6 padded rows must read
//! 0.0 (never written by the 16-lane exp loop), both for determinism and so
//! padded O rows stay finite. The `s_tile` region doubles as the o_stage
//! scratch for the rescale round-trip and the epilogue drain (it is dead
//! after the exps read it; `p_tile` stays live for the PV operand + row
//! sums). 4 `sync_cube` per 8-position tile — the same count as Bench 809 —
//! plus ~3 syncs × 32 blocks per rare rescale event and 2 × 32 once at the
//! epilogue drain.
//!
//! ## Dispatch contract
//!
//! Identical to [`crate::QwenAttentionPrefillTiledCmmaCubeCL`] (same `params`
//! layout, launcher, chunked `base_pos` semantics). At the dispatch site the
//! arm takes precedence over cmma → m16 → tiled → legacy when enabled (env
//! `RIIR_PREFILL_TILED_FLASH_CMMA_PV=1`). Requires `head_dim == 256`.
//!
//! ## MEASURED NEGATIVE — stays DEFAULT-OFF permanently (2026-09-02, Bench 839)
//!
//! First quiet-ish e2e (warm pair, same process, back-to-back, 16K, load
//! 4.9–7.6 sibling-CPU-only, GPU-exclusive): default cmma 203.19 s → 80.63
//! tok/s vs PV 309.86 s → 52.88 tok/s = **0.656× (a 1.53× LOSS)**, FNV
//! `c303e9ff390aae0d` + argmax 96205 IDENTICAL both arms (0 greedy flips —
//! numerics fine, perf is not). The cost is structural: the ~3-sync ×
//! 32-block smem round-trip rescale (`cmma::store` → 16-lane scale →
//! `load_with_layout(Accumulator)`) plus the per-tile fragment
//! store/reload is NOT amortized — the "rare event" fires often enough at
//! production P that PV-on-tensor-core loses to the cmma parent's
//! scalar-ALU PV on registers. A fragment-native row-scale API does not
//! exist in cubecl-0.11.0-pre.2 → this family is successor-material ONLY
//! if that API appears. Keep as the reproducible negative-result artifact
//! (the Bench-769 precedent).

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

/// Queries per cube (= 2 real rows per plane at `CubeDim::new_1d(256)`).
#[cfg(feature = "cubecl_runtime")]
const PV_Q_PER_CUBE: u32 = 16;

/// KV positions per tile (identical shape to the Bench 809 arm).
#[cfg(feature = "cubecl_runtime")]
const PV_N_TILE: u32 = 8;

/// K-dim steps per score accumulation (head_dim 256 / 8-wide cmma k-step).
#[cfg(feature = "cubecl_runtime")]
const PV_K_STEPS: u32 = 32;

/// The PV-cmma tiled causal flash-attention prefill (Issue 771 / Bench 810).
/// Same buffer contract and `params` layout as
/// [`crate::qwen_attention_prefill_cmma_cubecl::qwen_attention_prefill_tiled_cmma_f32`]:
/// `[head_dim, n_head, n_kv_head, p, scale, q_offset, q_tiles]`.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_prefill_tiled_cmma_pv_f32(
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
    let q_tile_base = (cube_id % q_tiles) * PV_Q_PER_CUBE;

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

    // ── Shared memory (28 KB of the 32 KB Apple budget — identical to 809) ──
    let mut q_tile = Shared::<[f32]>::new_slice(4096usize);
    let mut kv_tile = Shared::<[f32]>::new_slice(2048usize);
    let mut s_tile = Shared::<[f32]>::new_slice(512usize);
    let mut p_tile = Shared::<[f32]>::new_slice(512usize);
    // Per-plane rescale flags (8 f32 = 32 B; total smem 28,704 B of 32,768).
    // Makes the round-trip branch CUBE-UNIFORM — a sync_cube inside
    // plane-divergent control flow is UB (the p=17 corruption).
    let mut grew_flags = Shared::<[f32]>::new_slice(8usize);
    let s_base = (pl * 64u32) as usize;
    // The plane's cmma A-window: its aligned 8-row half of q_tile. The real
    // rows sit at window-local (pl*2) % 8 and +1.
    let win_base = if pl >= 4u32 { 8u32 } else { 0u32 };
    let lr0 = ((pl * 2u32) & 7u32) as usize;
    let lr1 = lr0 + 1usize;

    // The PV A-operand reads ALL 8 window rows of p_tile; the 6 padded rows
    // are never written by the exp loop, so zero the whole buffer once.
    // (Padded rows then contribute exact 0.0 to their padded O rows — no NaN
    // path, deterministic across runs.)
    p_tile[tid as usize] = f32::new(0.0f32);
    p_tile[tid as usize + 256usize] = f32::new(0.0f32);

    // ── Stage the cube's 16 query rows once (pre-scaled, as 809) ──
    for j in 0..16u32 {
        let idx = tid + j * 256u32;
        let r = idx / 256u32;
        let d = idx % 256u32;
        let grow = q_tile_base + r;
        q_tile[idx as usize] = if grow < p {
            query[((grow * q_stride + head_idx * head_dim) + d) as usize] * scale
        } else {
            f32::new(0.0f32)
        };
    }
    sync_cube();

    // Online softmax state per real row (plane-uniform after every tile —
    // all lanes compute the same reductions redundantly, no shuffles). O no
    // longer lives here — it lives in the 32 accumulator fragments below.
    let mut run_max_a = f32::new(-1e30f32);
    let mut run_sum_a = f32::new(0.0f32);
    let mut run_max_b = f32::new(-1e30f32);
    let mut run_sum_b = f32::new(0.0f32);

    // The 32 O accumulator fragments: [8 window-rows × 8 dims] each, column
    // block c covers global dims [c*8, c*8+8). Created ONCE; accumulated
    // across tiles by cmma::execute; rescaled in place via the rare-event
    // round-trip; drained at the epilogue. Explicit variables (no dynamic
    // register indexing — the corpus B49 local-memory-demotion class).
    let mut acc00 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc01 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc02 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc03 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc04 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc05 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc06 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc07 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc08 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc09 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc10 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc11 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc12 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc13 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc14 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc15 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc16 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc17 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc18 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc19 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc20 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc21 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc22 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc23 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc24 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc25 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc26 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc27 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc28 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc29 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc30 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let mut acc31 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
        cmma::MatrixLayout::Undefined, 0.0f32,
    );

    // Uniform loop bound: the max causal position across the cube's 16
    // queries (+1) — the m16 formula.
    let q_last = q_tile_base + (PV_Q_PER_CUBE - 1u32);
    let n_loop = if q_last < p { q_last } else { p - 1u32 } + q_offset + 1u32;

    // Rescale-lane constants (loop-invariant): the exp-lane mapping — lane
    // < 16 scales window row lr0 with corr_a (wrow 0) or lr1 with corr_b.
    let sc_wrow = lane / 8u32;
    let sc_wj = (lane % 8u32) as usize;
    let sc_lr = if sc_wrow == 0u32 { lr0 } else { lr1 };

    let mut pos0 = 0u32;
    let mut loop_i = 1u32;
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

        // ── Score tile: acc_s = Q_win · K^T via the tensor core (as 809) ──
        let acc_s = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::Undefined, 0.0f32,
        );
        let mut ki = 0u32;
        while ki < PV_K_STEPS {
            let mat_q = cmma::Matrix::<f32>::from_slice(
                cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
                cmma::MatrixLayout::RowMajor,
                q_tile.slice((win_base * 256u32 + ki * 8u32) as usize, 4096usize),
                256u32,
            );
            // kv_tile is [pos][dim] row-major; ColMajor B = K^T.
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
        // exp(0) = 1.0 exactly (MSL spec — the masked no-op invariant), so the
        // branch is bit-equivalent to the unconditional exp. grew_* doubles as
        // the rescale trigger (corr != 1.0 ⟺ the max grew).
        let grew_a = new_max_a > run_max_a;
        let grew_b = new_max_b > run_max_b;
        let corr_a = if grew_a {
            (run_max_a - new_max_a).exp()
        } else {
            f32::new(1.0f32)
        };
        let corr_b = if grew_b {
            (run_max_b - new_max_b).exp()
        } else {
            f32::new(1.0f32)
        };
        // Publish this plane's grew flag; the cube-wide OR is read after the
        // NEXT sync (visibility rides it — zero extra barriers), so the
        // barrier-bearing rescale branch is uniform across all 8 planes.
        grew_flags[pl as usize] = if grew_a || grew_b {
            f32::new(1.0f32)
        } else {
            f32::new(0.0f32)
        };

        // 16 exps spread over 16 lanes; each writes P for the PV phase.
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
        sync_cube(); // P written + V staged; s_tile dead; flags visible

        // Cube-uniform rescale gate (the OR of all 8 planes' flags).
        let mut any_grew = false;
        for pi in 0..8u32 {
            if grew_flags[pi as usize] > 0.5f32 {
                any_grew = true;
            }
        }
        let do_rescale = any_grew && loop_i > 1u32;
        if do_rescale {
            // ── Rare-event rescale: O ·= corr via the smem round-trip ──
            // s_tile is dead (exps done); p_tile stays live (PV operand).
            // Store each accumulator to the plane's 64-f32 s_tile region,
            // scale the two real rows, load back as an Accumulator fragment.
            // 3 syncs per block: store→scale, scale→load, load→next store.
            let sc_cr = if sc_wrow == 0u32 { corr_a } else { corr_b };
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc00, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc00,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc01, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc01,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc02, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc02,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc03, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc03,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc04, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc04,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc05, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc05,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc06, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc06,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc07, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc07,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc08, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc08,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc09, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc09,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc10, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc10,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc11, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc11,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc12, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc12,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc13, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc13,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc14, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc14,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc15, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc15,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc16, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc16,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc17, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc17,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc18, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc18,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc19, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc19,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc20, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc20,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc21, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc21,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc22, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc22,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc23, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc23,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc24, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc24,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc25, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc25,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc26, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc26,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc27, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc27,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc28, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc28,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc29, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc29,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc30, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc30,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            cmma::store(
                s_tile.slice_mut(s_base, s_base + 64usize),
                &acc31, 8u32, cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
            if lane < 16u32 {
                let o_idx = s_base + sc_lr * 8 + sc_wj;
                s_tile[o_idx] = s_tile[o_idx] * sc_cr;
            }
            sync_cube();
            cmma::load_with_layout::<f32, f32, cmma::Plane>(
                &mut acc31,
                s_tile.slice(s_base, s_base + 64usize),
                8u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_cube();
        }

        // ── PV: 32 tensor-core column blocks, accumulating in place ──
        // O_c += P_win · V_block. A = P window [8 rows × 8 pos] (the SAME
        // descriptor for every block — hoisted); B = kv_tile read RowMajor at
        // column-block base c*8 with stride 256 = V[pos k][dim c*8+n] — V
        // AS-IS (NOT transposed: the score phase contracts Q·K^T → ColMajor,
        // but PV contracts P·V → RowMajor. The first build shipped ColMajor
        // here and the G1 gate caught it: the output index selected a V
        // POSITION and the contraction ran over dims — case-C probe: dim 0
        // correct by coincidence, the rest exactly 0).
        let mat_p = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::A, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            p_tile.slice(s_base, s_base + 64usize),
            8u32,
        );
        let mat_v00 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(0usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v00, &acc00, &acc00);
        let mat_v01 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(8usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v01, &acc01, &acc01);
        let mat_v02 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(16usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v02, &acc02, &acc02);
        let mat_v03 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(24usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v03, &acc03, &acc03);
        let mat_v04 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(32usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v04, &acc04, &acc04);
        let mat_v05 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(40usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v05, &acc05, &acc05);
        let mat_v06 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(48usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v06, &acc06, &acc06);
        let mat_v07 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(56usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v07, &acc07, &acc07);
        let mat_v08 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(64usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v08, &acc08, &acc08);
        let mat_v09 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(72usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v09, &acc09, &acc09);
        let mat_v10 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(80usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v10, &acc10, &acc10);
        let mat_v11 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(88usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v11, &acc11, &acc11);
        let mat_v12 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(96usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v12, &acc12, &acc12);
        let mat_v13 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(104usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v13, &acc13, &acc13);
        let mat_v14 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(112usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v14, &acc14, &acc14);
        let mat_v15 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(120usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v15, &acc15, &acc15);
        let mat_v16 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(128usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v16, &acc16, &acc16);
        let mat_v17 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(136usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v17, &acc17, &acc17);
        let mat_v18 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(144usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v18, &acc18, &acc18);
        let mat_v19 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(152usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v19, &acc19, &acc19);
        let mat_v20 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(160usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v20, &acc20, &acc20);
        let mat_v21 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(168usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v21, &acc21, &acc21);
        let mat_v22 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(176usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v22, &acc22, &acc22);
        let mat_v23 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(184usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v23, &acc23, &acc23);
        let mat_v24 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(192usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v24, &acc24, &acc24);
        let mat_v25 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(200usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v25, &acc25, &acc25);
        let mat_v26 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(208usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v26, &acc26, &acc26);
        let mat_v27 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(216usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v27, &acc27, &acc27);
        let mat_v28 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(224usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v28, &acc28, &acc28);
        let mat_v29 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(232usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v29, &acc29, &acc29);
        let mat_v30 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(240usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v30, &acc30, &acc30);
        let mat_v31 = cmma::Matrix::<f32>::from_slice(
            cmma::MatrixIdent::B, 8usize, 8usize, 8usize,
            cmma::MatrixLayout::RowMajor,
            kv_tile.slice(248usize, 2048usize),
            256u32,
        );
        cmma::execute::<f32, f32, f32, f32, cmma::Plane>(&mat_p, &mat_v31, &acc31, &acc31);

        // Row sums: scalar, unchanged from 809 (lanes read their own rows of
        // p_tile — after sync3, so the 16-lane writes are visible).
        let mut ts_a = f32::new(0.0f32);
        let mut ts_b = f32::new(0.0f32);
        let mut jj = 0u32;
        while jj < PV_N_TILE {
            ts_a = ts_a + p_tile[s_base + lr0 * 8 + jj as usize];
            ts_b = ts_b + p_tile[s_base + lr1 * 8 + jj as usize];
            jj += 1u32;
        }
        run_sum_a = run_sum_a * corr_a + ts_a;
        run_sum_b = run_sum_b * corr_b + ts_b;
        run_max_a = new_max_a;
        run_max_b = new_max_b;

        pos0 += PV_N_TILE;
        loop_i += 1u32;
        // Loop-back barrier (the B57 smem-role-reversal class, the same 4th
        // sync the Bench 809 gate caught): the next tile's K stage overwrites
        // kv_tile, whose V rows THIS tile's PV reads; and the next score store
        // overwrites s_tile, whose P this tile's PV reads. Per-thread program
        // order does not bound CROSS-WARP progress.
        sync_cube();
    }

    // ── Epilogue: fragment drain + gated normalization ──
    // Block c is drained by lane c (lane c owns dims [c*8, c*8+8) — exactly
    // `dims_base`), so the gate/out offsets are loop-invariant per lane.
    let inv_sum_a = f32::new(1.0f32) / run_sum_a;
    let inv_sum_b = f32::new(1.0f32) / run_sum_b;
    let go_a = q_off_a + dims_base;
    let go_b = q_off_b + dims_base;
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc00, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 0u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc01, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 1u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc02, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 2u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc03, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 3u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc04, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 4u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc05, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 5u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc06, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 6u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc07, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 7u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc08, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 8u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc09, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 9u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc10, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 10u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc11, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 11u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc12, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 12u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc13, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 13u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc14, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 14u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc15, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 15u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc16, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 16u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc17, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 17u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc18, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 18u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc19, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 19u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc20, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 20u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc21, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 21u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc22, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 22u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc23, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 23u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc24, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 24u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc25, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 25u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc26, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 26u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc27, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 27u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc28, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 28u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc29, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 29u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc30, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 30u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
    cmma::store(
        s_tile.slice_mut(s_base, s_base + 64usize),
        &acc31, 8u32, cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    if lane == 31u32 {
        for j in 0..8usize {
            if active_a {
                let g = gate[go_a + j];
                attn_out[go_a + j] = s_tile[s_base + lr0 * 8 + j] * inv_sum_a
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
            if active_b {
                let g = gate[go_b + j];
                attn_out[go_b + j] = s_tile[s_base + lr1 * 8 + j] * inv_sum_b
                    / (f32::new(1.0f32) + (f32::new(0.0f32) - g).exp());
            }
        }
    }
    sync_cube();
}

/// Launch the PV-cmma tiled flash attention prefill (Issue 771 / Bench 810).
/// Same buffer contract as [`crate::QwenAttentionPrefillTiledCmmaCubeCL::launch`],
/// including the chunked `base_pos` cache semantics — callers route through
/// [`crate::ternary_deltanet_gpu_forward::prefill_tiled_flash_cmma_pv_enabled`]
/// (DEFAULT-OFF — tolerance arm; head_dim must be 256).
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionPrefillTiledCmmaPvCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionPrefillTiledCmmaPvCubeCL {
    /// # Safety
    /// Same contract as [`crate::QwenAttentionPrefillTiledCmmaCubeCL::launch`]
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
            "tiled cmma-pv flash kernel is head_dim-256 specialized"
        );
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Grid = n_head × q_tiles (16 queries per cube) — chunk on query-TILE
        // boundaries with the same 65535 guard as the 809 launcher.
        let q_tiles_total = p.div_ceil(PV_Q_PER_CUBE as usize).max(1);
        let tiles_per_chunk = (MAX_WG_X as usize / n_head.max(1) / 16).max(1);
        let mut t0 = 0usize; // query-token base of the chunk
        while t0 < p {
            let tiles_left = q_tiles_total - t0.div_ceil(PV_Q_PER_CUBE as usize);
            let tiles = tiles_per_chunk.min(tiles_left);
            let tc = (tiles * PV_Q_PER_CUBE as usize).min(p - t0);
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
                qwen_attention_prefill_tiled_cmma_pv_f32::launch_unchecked::<R>(
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
