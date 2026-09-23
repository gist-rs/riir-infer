//! CubeCL flash attention kernel for Gemma 2 decode (Plan 106 T2.5).
//!
//! Implements online softmax with logit softcapping and Grouped Query Attention (GQA).
//! Each workgroup (256 threads) handles one attention head, cooperatively scanning
//! all KV positions in tiles with parallel reduction.
//!
//! # Algorithm
//!
//! Matches `attention_softcap_parallel.wgsl` exactly:
//!
//! 1. Each workgroup handles one query head (head_idx = workgroup_id).
//! 2. GQA mapping: `kv_group = head_idx * n_kv_head / n_head`.
//! 3. For each tile of 256 KV positions:
//!    a. Thread tid computes softcapped Q·K score for position (tile_base + tid).
//!    b. Unrolled parallel max reduction → tile_max.
//!    c. Compute `exp(score - tile_max)`.
//!    d. All threads accumulate weighted values for their output dimension.
//!    e. Unrolled parallel sum reduction → tile_sum.
//!    f. Online softmax update: running_max, running_sum, running_out.
//! 4. Normalize and write output.
//!
//! # Softcapping
//!
//! Gemma 2 applies logit softcapping before softmax:
//! ```text
//! raw = Q · K * scale
//! score = softcap * tanh(raw / softcap)
//! ```
//!
//! # CubeCL v0.10 Workarounds
//!
//! Uses exactly 3 `Array<f32>` parameters (matching proven `matmul_tiled_f32` pattern)
//! to avoid `u32: From<NativeExpand<u32>>` macro expansion bug. The KV buffer stores
//! keys followed by values in a single combined array:
//! ```text
//! kv = [keys(n_pos × kv_stride) | values(n_pos × kv_stride)]
//! n_positions = kv.len() / (2 × n_kv_head × head_dim)
//! ```
//!
//! All dimensions are hardcoded for Gemma 2 2B:
//! - n_head = 8, n_kv_head = 4, head_dim = 256, softcap = 50.0, scale = 0.0625
//!
//! Parallel reductions are unrolled (8 hardcoded steps) instead of while-loop
//! step variables to avoid the NativeExpand macro bug.
//!
//! # Dispatch
//!
//! | CubeDim       | CubeCount         | Responsibility    |
//! |---------------|-------------------|--------------------|
//! | `new_1d(256)` | `(n_head, 1, 1)`  | 1 workgroup/head   |
//!
//! # Shared Memory
//!
//! 1 × `SharedMemory<f32, 256>` = 1 KB (reused for scores, exp, reductions).

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

// ---------------------------------------------------------------------------
// Flash attention decode kernel — matches attention_softcap_parallel.wgsl
// ---------------------------------------------------------------------------

/// CubeCL flash attention decode with online softmax and logit softcapping.
///
/// Implements Gemma 2 decode-time attention for all heads in a single dispatch.
/// Each workgroup (256 threads) handles one query head, scanning the full KV cache
/// in tiles with parallel max/sum reduction and online softmax accumulation.
///
/// ## 3-Array Parameter Layout
///
/// Uses exactly 3 `Array<f32>` parameters (matching proven `matmul_tiled_f32`
/// pattern). The `kv` buffer stores keys followed by values (combined layout):
/// ```text
/// kv = [keys(n_pos × kv_stride) | values(n_pos × kv_stride)]
/// kv_half = kv.len() / 2
/// ```
///
/// ## GQA Mapping (Gemma 2 2B)
///
/// n_head=8, n_kv_head=4 → 2 query heads per KV head.
/// ```text
/// heads 0,1 → kv_head 0   |   heads 2,3 → kv_head 1
/// heads 4,5 → kv_head 2   |   heads 6,7 → kv_head 3
/// ```
///
/// ## Dimensions (derived from array lengths)
///
/// ```text
/// query.len()     = n_head × head_dim      = 2048
/// kv.len()        = 2 × n_positions × kv_stride
/// attn_out.len()  = n_head × head_dim      = 2048
/// n_positions     = kv.len() / (2 × n_kv_head × head_dim)
/// ```
///
/// ## CubeCL v0.10 Body Constraints
///
/// The `#[cube(launch_unchecked)]` macro triggers `u32: From<NativeExpand<u32>>`
/// with certain kernel body patterns. Key constraints discovered:
/// - While-loop reduction step variables (`let mut step = ...`) cause NativeExpand confusion
/// - Inline `if/else` expressions for assignment trigger the macro bug
/// - Workaround: unrolled reductions (8 hardcoded steps), guard inside accumulation loop
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion generates this pattern; not user-writable")]
#[cube(launch_unchecked)]
fn attention_decode_f32(query: &[f32], kv: &[f32], attn_out: &mut [f32]) {
    // ── Gemma 2 2B constants ──
    let head_dim = 256u32;
    let n_head = 8u32;
    let n_kv_head = 4u32;
    let kv_stride = n_kv_head * head_dim; // 1024
    let softcap = f32::new(50.0f32);
    let scale = f32::new(0.0625f32); // 1/√256
    let cube_size = 256u32;
    // ── Derive dimensions from array lengths ──
    let kv_half = kv.len() as u32 / 2u32;
    let n_positions = kv_half / kv_stride;

    // ── Workgroup/thread assignment ──
    let head_idx = ABSOLUTE_POS as u32 / cube_size;
    let tid = UNIT_POS;

    // Guard: skip if overdispatched beyond n_head
    if head_idx >= n_head {
        terminate!();
    }

    let head_off = head_idx * head_dim;

    // Guard: no positions — write zeros and exit
    if n_positions == 0u32 {
        if tid < head_dim {
            attn_out[(head_off + tid) as usize] = f32::new(0.0f32);
        }
        terminate!();
    }

    // GQA: compute which KV head this query head attends to
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_off = kv_group * head_dim;

    // Whether this thread owns a valid output dimension
    let valid_dim = tid < head_dim;

    // ── Shared memory for reductions (1 KB) — declared once ──
    let mut smem = Shared::<[f32]>::new_slice(256usize);

    // ── Online softmax running state (per-thread) ──
    let mut running_max = f32::new(-1e30f32);
    let mut running_sum = f32::new(0.0f32);
    let mut running_out = f32::new(0.0f32);

    // ── Process KV positions in tiles of 256 ──
    let n_tiles = n_positions.div_ceil(cube_size);

    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_base = tile * cube_size;
        let pos = tile_base + tid;
        let valid_pos = pos < n_positions;

        // ── Phase 1: Each thread computes softcapped Q·K score ──
        let mut my_score = f32::new(-1e30f32);
        if valid_pos {
            let k_base = pos * kv_stride + kv_off;
            let mut dot = f32::new(0.0f32);
            let mut d = 0u32;
            while d < head_dim {
                dot += query[(head_off + d) as usize] * kv[(k_base + d) as usize];
                d += 1u32;
            }
            let raw = dot * scale;
            my_score = softcap * f32::tanh(raw / softcap);
        }
        smem[tid as usize] = my_score;
        sync_cube();

        // ── Phase 2: Unrolled parallel max reduction → tile_max ──
        if tid < 128u32 && smem[(tid + 128u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 128u32) as usize];
        }
        sync_cube();
        if tid < 64u32 && smem[(tid + 64u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 64u32) as usize];
        }
        sync_cube();
        if tid < 32u32 && smem[(tid + 32u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 32u32) as usize];
        }
        sync_cube();
        if tid < 16u32 && smem[(tid + 16u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 16u32) as usize];
        }
        sync_cube();
        if tid < 8u32 && smem[(tid + 8u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 8u32) as usize];
        }
        sync_cube();
        if tid < 4u32 && smem[(tid + 4u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 4u32) as usize];
        }
        sync_cube();
        if tid < 2u32 && smem[(tid + 2u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 2u32) as usize];
        }
        sync_cube();
        if tid < 1u32 && smem[1usize] > smem[0usize] {
            smem[0usize] = smem[1usize];
        }
        sync_cube();
        let tile_max = smem[0usize];
        sync_cube();

        // ── Phase 3: Compute exp(score - tile_max) ──
        let mut my_exp = f32::new(0.0f32);
        if valid_pos {
            my_exp = f32::exp(my_score - tile_max);
        }
        smem[tid as usize] = my_exp;
        sync_cube();

        // ── Phase 4: Weighted value accumulation for output dimension tid ──
        let mut tile_val = f32::new(0.0f32);
        if valid_dim {
            // Iterate full cube_size; out-of-bounds smem entries are 0 from Phase 3
            // so only valid positions contribute to the weighted sum.
            let mut i = 0u32;
            while i < cube_size {
                let pos_i = tile_base + i;
                if pos_i < n_positions {
                    tile_val = tile_val
                        + smem[i as usize]
                            * kv[(kv_half + pos_i * kv_stride + kv_off + tid) as usize];
                }
                i += 1u32;
            }
        }

        // ── Phase 5: Unrolled parallel sum reduction → tile_sum ──
        sync_cube();
        if tid < 128u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 128u32) as usize];
        }
        sync_cube();
        if tid < 64u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 64u32) as usize];
        }
        sync_cube();
        if tid < 32u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 32u32) as usize];
        }
        sync_cube();
        if tid < 16u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 16u32) as usize];
        }
        sync_cube();
        if tid < 8u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 8u32) as usize];
        }
        sync_cube();
        if tid < 4u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 4u32) as usize];
        }
        sync_cube();
        if tid < 2u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 2u32) as usize];
        }
        sync_cube();
        if tid < 1u32 {
            smem[0usize] = smem[0usize] + smem[1usize];
        }
        sync_cube();
        let tile_sum = smem[0usize];

        // ── Phase 6: Online softmax update ──
        let mut new_max = running_max;
        if tile_max > new_max {
            new_max = tile_max;
        }
        let prev_corr = f32::exp(running_max - new_max);
        let curr_corr = f32::exp(tile_max - new_max);

        running_sum *= prev_corr;

        running_sum += tile_sum * curr_corr;
        running_out *= prev_corr;
        running_out += tile_val * curr_corr;
        running_max = new_max;

        sync_cube();
        tile += 1u32;
    }

    // ── Final: Normalize and write output ──
    if valid_dim {
        attn_out[(head_off + tid) as usize] = running_out / running_sum;
    }
}

// ---------------------------------------------------------------------------
// Folded dispatch kernel (Plan 179 D1)
// ---------------------------------------------------------------------------

/// CubeCL flash attention decode with folded query tokens for higher GPU occupancy.
///
/// Same algorithm as [`attention_decode_f32`] but each workgroup processes
/// `fold_factor` query tokens for one head. This increases GPU occupancy when
/// `n_head < SM_count` (common for speculative decode with small models).
///
/// ## 5-Array Parameter Layout
///
/// - `query`: `[f32; seq_len_q × n_head × head_dim]` — query vectors (batched).
/// - `kv`: `[f32; 2 × n_positions × kv_stride]` — combined keys ‖ values.
/// - `params`: `[f32; 4]` — `[n_head_f32, fold_factor_f32, seq_len_q_f32, 0.0]`.
/// - `attn_out`: `[f32; seq_len_q × n_head × head_dim]` — output (batched).
///
/// ## Dispatch
///
/// `CubeCount::Static(n_head, 1, 1)` workgroups of 256 threads each.
/// Within each workgroup, threads process `fold_factor` query tokens sequentially.
///
/// ## Differences from `attention_decode_f32`
///
/// | Aspect | `attention_decode_f32` | `attention_decode_folded_f32` |
/// |--------|----------------------|-------------------------------|
/// | Query tokens | 1 per workgroup | `fold_factor` per workgroup |
/// | Query layout | `[n_head × head_dim]` | `[seq_len_q × n_head × head_dim]` |
/// | Output layout | `[n_head × head_dim]` | `[seq_len_q × n_head × head_dim]` |
/// | Params | none (hardcoded) | 4-element buffer |
#[cfg(feature = "fold_dispatch")]
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion generates this pattern; not user-writable")]
#[cube(launch_unchecked)]
fn attention_decode_folded_f32(
    query: &[f32],
    kv: &[f32],
    params: &[f32],
    attn_out: &mut [f32],
) {
    // ── Gemma 2 2B constants ──
    let head_dim = 256u32;
    let n_kv_head = 4u32;
    let kv_stride = n_kv_head * head_dim; // 1024
    let softcap = f32::new(50.0f32);
    let scale = f32::new(0.0625f32); // 1/√256
    let cube_size = 256u32;

    // ── Read fold parameters ──
    let n_head = params[0usize] as u32;
    let fold = params[1usize] as u32;
    let _seq_len_q = params[2usize] as u32;

    // ── Derive dimensions from array lengths ──
    let kv_half = kv.len() as u32 / 2u32;
    let n_positions = kv_half / kv_stride;

    // ── Workgroup/thread assignment ──
    let head_idx = ABSOLUTE_POS as u32 / cube_size;
    let tid = UNIT_POS;

    // Guard: skip if overdispatched beyond n_head
    if head_idx >= n_head {
        terminate!();
    }

    // GQA: compute which KV head this query head attends to
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_off = kv_group * head_dim;

    // Whether this thread owns a valid output dimension
    let valid_dim = tid < head_dim;

    // ── Shared memory for reductions (1 KB) ──
    let mut smem = Shared::<[f32]>::new_slice(256usize);

    // ── Process each folded query token sequentially ──
    // Each workgroup handles `fold` query tokens for its assigned head.
    let mut q_tok = 0u32;
    while q_tok < fold {
        let q_head_off = (q_tok * n_head + head_idx) * head_dim;
        let out_head_off = (q_tok * n_head + head_idx) * head_dim;

        // Guard: no positions — write zeros
        if n_positions == 0u32
            && valid_dim {
                attn_out[(out_head_off + tid) as usize] = f32::new(0.0f32);
            }

        // ── Online softmax running state (per-token) ──
        // Only compute attention when there are KV positions.
        if n_positions > 0u32 {
            let mut running_max = f32::new(-1e30f32);
            let mut running_sum = f32::new(0.0f32);
            let mut running_out = f32::new(0.0f32);

            // ── Process KV positions in tiles of 256 ──
            let n_tiles = n_positions.div_ceil(cube_size);

            let mut tile = 0u32;
            while tile < n_tiles {
                let tile_base = tile * cube_size;
                let pos = tile_base + tid;
                let valid_pos = pos < n_positions;

                // ── Phase 1: Each thread computes softcapped Q·K score ──
                let mut my_score = f32::new(-1e30f32);
                if valid_pos {
                    let k_base = pos * kv_stride + kv_off;
                    let mut dot = f32::new(0.0f32);
                    let mut d = 0u32;
                    while d < head_dim {
                        dot += query[(q_head_off + d) as usize] * kv[(k_base + d) as usize];
                        d += 1u32;
                    }
                    let raw = dot * scale;
                    my_score = softcap * f32::tanh(raw / softcap);
                }
                smem[tid as usize] = my_score;
                sync_cube();

                // ── Phase 2: Unrolled parallel max reduction → tile_max ──
                if tid < 128u32 && smem[(tid + 128u32) as usize] > smem[tid as usize] {
                    smem[tid as usize] = smem[(tid + 128u32) as usize];
                }
                sync_cube();
                if tid < 64u32 && smem[(tid + 64u32) as usize] > smem[tid as usize] {
                    smem[tid as usize] = smem[(tid + 64u32) as usize];
                }
                sync_cube();
                if tid < 32u32 && smem[(tid + 32u32) as usize] > smem[tid as usize] {
                    smem[tid as usize] = smem[(tid + 32u32) as usize];
                }
                sync_cube();
                if tid < 16u32 && smem[(tid + 16u32) as usize] > smem[tid as usize] {
                    smem[tid as usize] = smem[(tid + 16u32) as usize];
                }
                sync_cube();
                if tid < 8u32 && smem[(tid + 8u32) as usize] > smem[tid as usize] {
                    smem[tid as usize] = smem[(tid + 8u32) as usize];
                }
                sync_cube();
                if tid < 4u32 && smem[(tid + 4u32) as usize] > smem[tid as usize] {
                    smem[tid as usize] = smem[(tid + 4u32) as usize];
                }
                sync_cube();
                if tid < 2u32 && smem[(tid + 2u32) as usize] > smem[tid as usize] {
                    smem[tid as usize] = smem[(tid + 2u32) as usize];
                }
                sync_cube();
                if tid < 1u32 && smem[1usize] > smem[0usize] {
                    smem[0usize] = smem[1usize];
                }
                sync_cube();
                let tile_max = smem[0usize];
                sync_cube();

                // ── Phase 3: Compute exp(score - tile_max) ──
                let mut my_exp = f32::new(0.0f32);
                if valid_pos {
                    my_exp = f32::exp(my_score - tile_max);
                }
                smem[tid as usize] = my_exp;
                sync_cube();

                // ── Phase 4: Weighted value accumulation for output dimension tid ──
                let mut tile_val = f32::new(0.0f32);
                if valid_dim {
                    let mut i = 0u32;
                    while i < cube_size {
                        let pos_i = tile_base + i;
                        if pos_i < n_positions {
                            tile_val = tile_val
                                + smem[i as usize]
                                    * kv[(kv_half + pos_i * kv_stride + kv_off + tid) as usize];
                        }
                        i += 1u32;
                    }
                }

                // ── Phase 5: Unrolled parallel sum reduction → tile_sum ──
                sync_cube();
                if tid < 128u32 {
                    smem[tid as usize] = smem[tid as usize] + smem[(tid + 128u32) as usize];
                }
                sync_cube();
                if tid < 64u32 {
                    smem[tid as usize] = smem[tid as usize] + smem[(tid + 64u32) as usize];
                }
                sync_cube();
                if tid < 32u32 {
                    smem[tid as usize] = smem[tid as usize] + smem[(tid + 32u32) as usize];
                }
                sync_cube();
                if tid < 16u32 {
                    smem[tid as usize] = smem[tid as usize] + smem[(tid + 16u32) as usize];
                }
                sync_cube();
                if tid < 8u32 {
                    smem[tid as usize] = smem[tid as usize] + smem[(tid + 8u32) as usize];
                }
                sync_cube();
                if tid < 4u32 {
                    smem[tid as usize] = smem[tid as usize] + smem[(tid + 4u32) as usize];
                }
                sync_cube();
                if tid < 2u32 {
                    smem[tid as usize] = smem[tid as usize] + smem[(tid + 2u32) as usize];
                }
                sync_cube();
                if tid < 1u32 {
                    smem[0usize] = smem[0usize] + smem[1usize];
                }
                sync_cube();
                let tile_sum = smem[0usize];

                // ── Phase 6: Online softmax update ──
                let mut new_max = running_max;
                if tile_max > new_max {
                    new_max = tile_max;
                }
                let prev_corr = f32::exp(running_max - new_max);
                let curr_corr = f32::exp(tile_max - new_max);

                running_sum *= prev_corr;

                running_sum += tile_sum * curr_corr;
                running_out *= prev_corr;
                running_out += tile_val * curr_corr;
                running_max = new_max;

                sync_cube();
                tile += 1u32;
            }

            // ── Final: Normalize and write output for this query token ──
            if valid_dim {
                attn_out[(out_head_off + tid) as usize] = running_out / running_sum;
            }
        } // end if n_positions > 0u32

        sync_cube();
        q_tok += 1u32;
    }
}

// ---------------------------------------------------------------------------
// Generic LLaMA-family flash attention decode kernel
// ---------------------------------------------------------------------------

/// Generic CubeCL flash attention decode kernel for LLaMA-family models.
///
/// Same algorithm as [`attention_decode_f32`] but reads dimensions from a params buffer,
/// supporting any LLaMA architecture configuration (MiniCPM5, Gemma 2, etc.).
///
/// ## 4-Array Parameter Layout
///
/// - `query`: `[f32; n_head × head_dim]` — query vectors (all heads).
/// - `kv`: `[f32; 2 × n_positions × kv_stride]` — combined keys ‖ values.
/// - `params`: `[f32; 5]` — `[n_head, n_kv_head, head_dim, softcap, scale]`.
/// - `attn_out`: `[f32; n_head × head_dim]` — output (all heads).
///
/// ## Softcap handling
///
/// When `softcap == 0.0`, softcapping is bypassed (raw score = scaled dot product).
/// This is the case for LLaMA/MiniCPM models which don't use logit softcapping.
///
/// ## Constraints
///
/// - `head_dim` must be ≤ 256 (cube size)
/// - `n_head` must be ≤ 256 (reasonable for all current models)
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion generates this pattern; not user-writable")]
#[cube(launch_unchecked)]
fn attention_decode_llama_f32(
    query: &[f32],
    kv: &[f32],
    params: &[f32],
    attn_out: &mut [f32],
) {
    // Read dimensions from params buffer (f32 → u32 cast, exact for small integers)
    let n_head = params[0usize] as u32;
    let n_kv_head = params[1usize] as u32;
    let head_dim = params[2usize] as u32;
    let softcap = params[3usize];
    let scale = params[4usize];
    let cube_size = 256u32;

    let kv_stride = n_kv_head * head_dim;

    // ── Derive dimensions from array lengths ──
    let kv_half = kv.len() as u32 / 2u32;
    let n_positions = kv_half / kv_stride;

    // ── Workgroup/thread assignment ──
    let head_idx = ABSOLUTE_POS as u32 / cube_size;
    let tid = UNIT_POS;

    // Guard: skip if overdispatched beyond n_head
    if head_idx >= n_head {
        terminate!();
    }

    let head_off = head_idx * head_dim;

    // Guard: no positions — write zeros and exit
    if n_positions == 0u32 {
        if tid < head_dim {
            attn_out[(head_off + tid) as usize] = f32::new(0.0f32);
        }
        terminate!();
    }

    // GQA: compute which KV head this query head attends to
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_off = kv_group * head_dim;

    // Whether this thread owns a valid output dimension
    let valid_dim = tid < head_dim;

    // ── Shared memory for reductions (1 KB) — declared once ──
    let mut smem = Shared::<[f32]>::new_slice(256usize);

    // ── Online softmax running state (per-thread) ──
    let mut running_max = f32::new(-1e30f32);
    let mut running_sum = f32::new(0.0f32);
    let mut running_out = f32::new(0.0f32);

    // ── Process KV positions in tiles of 256 ──
    let n_tiles = n_positions.div_ceil(cube_size);

    // Precompute whether softcap is enabled (softcap > 0)
    let has_softcap = softcap > f32::new(0.0f32);

    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_base = tile * cube_size;
        let pos = tile_base + tid;
        let valid_pos = pos < n_positions;

        // ── Phase 1: Each thread computes (optionally softcapped) Q·K score ──
        let mut my_score = f32::new(-1e30f32);
        if valid_pos {
            let k_base = pos * kv_stride + kv_off;
            let mut dot = f32::new(0.0f32);
            let mut d = 0u32;
            while d < head_dim {
                dot += query[(head_off + d) as usize] * kv[(k_base + d) as usize];
                d += 1u32;
            }
            let raw = dot * scale;
            // Conditional softcap: bypass when softcap == 0.0
            if has_softcap {
                my_score = softcap * f32::tanh(raw / softcap);
            }
            if !has_softcap {
                my_score = raw;
            }
        }
        smem[tid as usize] = my_score;
        sync_cube();

        // ── Phase 2: Unrolled parallel max reduction → tile_max ──
        if tid < 128u32 && smem[(tid + 128u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 128u32) as usize];
        }
        sync_cube();
        if tid < 64u32 && smem[(tid + 64u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 64u32) as usize];
        }
        sync_cube();
        if tid < 32u32 && smem[(tid + 32u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 32u32) as usize];
        }
        sync_cube();
        if tid < 16u32 && smem[(tid + 16u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 16u32) as usize];
        }
        sync_cube();
        if tid < 8u32 && smem[(tid + 8u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 8u32) as usize];
        }
        sync_cube();
        if tid < 4u32 && smem[(tid + 4u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 4u32) as usize];
        }
        sync_cube();
        if tid < 2u32 && smem[(tid + 2u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 2u32) as usize];
        }
        sync_cube();
        if tid < 1u32 && smem[1usize] > smem[0usize] {
            smem[0usize] = smem[1usize];
        }
        sync_cube();
        let tile_max = smem[0usize];
        sync_cube();

        // ── Phase 3: Compute exp(score - tile_max) ──
        let mut my_exp = f32::new(0.0f32);
        if valid_pos {
            my_exp = f32::exp(my_score - tile_max);
        }
        smem[tid as usize] = my_exp;
        sync_cube();

        // ── Phase 4: Weighted value accumulation for output dimension tid ──
        let mut tile_val = f32::new(0.0f32);
        if valid_dim {
            let mut i = 0u32;
            while i < cube_size {
                let pos_i = tile_base + i;
                if pos_i < n_positions {
                    tile_val = tile_val
                        + smem[i as usize]
                            * kv[(kv_half + pos_i * kv_stride + kv_off + tid) as usize];
                }
                i += 1u32;
            }
        }

        // ── Phase 5: Unrolled parallel sum reduction → tile_sum ──
        sync_cube();
        if tid < 128u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 128u32) as usize];
        }
        sync_cube();
        if tid < 64u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 64u32) as usize];
        }
        sync_cube();
        if tid < 32u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 32u32) as usize];
        }
        sync_cube();
        if tid < 16u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 16u32) as usize];
        }
        sync_cube();
        if tid < 8u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 8u32) as usize];
        }
        sync_cube();
        if tid < 4u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 4u32) as usize];
        }
        sync_cube();
        if tid < 2u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 2u32) as usize];
        }
        sync_cube();
        if tid < 1u32 {
            smem[0usize] = smem[0usize] + smem[1usize];
        }
        sync_cube();
        let tile_sum = smem[0usize];

        // ── Phase 6: Online softmax update ──
        let mut new_max = running_max;
        if tile_max > new_max {
            new_max = tile_max;
        }
        let prev_corr = f32::exp(running_max - new_max);
        let curr_corr = f32::exp(tile_max - new_max);

        running_sum *= prev_corr;

        running_sum += tile_sum * curr_corr;
        running_out *= prev_corr;
        running_out += tile_val * curr_corr;
        running_max = new_max;

        sync_cube();
        tile += 1u32;
    }

    // ── Final: Normalize and write output ──
    if valid_dim {
        attn_out[(head_off + tid) as usize] = running_out / running_sum;
    }
}

// ---------------------------------------------------------------------------
// Block-causal flash attention kernel (Plan 108 T2)
// ---------------------------------------------------------------------------

/// CubeCL flash attention decode with block-causal masking (Plan 108 T2).
///
/// Same algorithm as [`attention_decode_f32`] but with block-causal attention
/// boundary. Only attends to positions `0..t_n` where `t_n` is computed from
/// `block_causal_t_n(query_pos, prompt_len, block_size, n_positions)`.
///
/// # Block-Causal Masking Rules
///
/// Mirrors `riir_engine::transformer::block_causal_t_n`:
/// - If `query_pos < prompt_len`: `t_n = prompt_len` (prompt sees all prompt)
/// - Else: `t_n = min(prompt_len + (block_idx + 1) * block_size, n_positions)`
///   where `block_idx = (query_pos - prompt_len) / block_size`
///
/// # 4-Array Parameter Layout
///
/// Uses 4 `Array<f32>` parameters to pass block-causal metadata via a params
/// buffer (matching the `rmsnorm_f32` pattern for CubeCL v0.10 compatibility):
/// - `query`: `[f32; n_head × head_dim]` — query vectors (all heads).
/// - `kv`: `[f32; 2 × n_positions × kv_stride]` — combined keys ‖ values.
/// - `params`: `[f32; 3]` — `[query_pos, prompt_len, block_size]` as f32 integers.
/// - `attn_out`: `[f32; n_head × head_dim]` — output (all heads).
///
/// # Differences from `attention_decode_f32`
///
/// | Aspect | `attention_decode_f32` | `attention_block_causal_f32` |
/// |--------|----------------------|------------------------------|
/// | Attention boundary | `n_positions` (all) | `t_n` (block-causal) |
/// | Tile count | `ceil(n_pos / 256)` | `ceil(t_n / 256)` |
/// | Score mask | `pos < n_positions` | `pos < t_n` |
/// | Value accumulation | `pos_i < n_positions` | `pos_i < t_n` |
/// | Parameters | 3 arrays | 4 arrays (extra params) |
#[cfg(feature = "gemma2_d2f")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion generates this pattern; not user-writable")]
#[cube(launch_unchecked)]
fn attention_block_causal_f32(
    query: &[f32],
    kv: &[f32],
    params: &[f32],
    attn_out: &mut [f32],
) {
    // ── Gemma 2 2B constants ──
    let head_dim = 256u32;
    let n_head = 8u32;
    let n_kv_head = 4u32;
    let kv_stride = n_kv_head * head_dim; // 1024
    let softcap = f32::new(50.0f32);
    let scale = f32::new(0.0625f32); // 1/√256
    let cube_size = 256u32;

    // ── Read block-causal params (f32 → u32 cast, exact for small integers) ──
    let query_pos = params[0usize] as u32;
    let prompt_len = params[1usize] as u32;
    let block_size_param = params[2usize] as u32;

    // ── Derive dimensions from array lengths ──
    let kv_half = kv.len() as u32 / 2u32;
    let n_positions = kv_half / kv_stride;

    // ── Workgroup/thread assignment ──
    let head_idx = ABSOLUTE_POS as u32 / cube_size;
    let tid = UNIT_POS;

    // Guard: skip if overdispatched beyond n_head
    if head_idx >= n_head {
        terminate!();
    }

    let head_off = head_idx * head_dim;

    // Guard: empty KV cache — write zeros and exit
    if n_positions == 0u32 {
        if tid < head_dim {
            attn_out[(head_off + tid) as usize] = f32::new(0.0f32);
        }
        terminate!();
    }

    // ── Compute block-causal attention boundary t_n ──
    // Same logic as riir_engine::transformer::block_causal_t_n:
    // - Prompt positions: attend to all prompt positions (bidirectional)
    // - Generation positions: bidirectional within block, causal across blocks
    let mut t_n = prompt_len;
    if query_pos >= prompt_len {
        let gen_offset = query_pos - prompt_len;
        let block_idx = gen_offset / block_size_param;
        let block_end = prompt_len + (block_idx + 1u32) * block_size_param;
        t_n = block_end;
        if block_end > n_positions {
            t_n = n_positions;
        }
    }

    // Guard: block-causal mask excludes all positions — write zeros and exit
    if t_n == 0u32 {
        if tid < head_dim {
            attn_out[(head_off + tid) as usize] = f32::new(0.0f32);
        }
        terminate!();
    }

    // GQA: compute which KV head this query head attends to
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_off = kv_group * head_dim;

    // Whether this thread owns a valid output dimension
    let valid_dim = tid < head_dim;

    // ── Shared memory for reductions (1 KB) — declared once ──
    let mut smem = Shared::<[f32]>::new_slice(256usize);

    // ── Online softmax running state (per-thread) ──
    let mut running_max = f32::new(-1e30f32);
    let mut running_sum = f32::new(0.0f32);
    let mut running_out = f32::new(0.0f32);

    // ── Process KV positions in tiles of 256, up to t_n ──
    let n_tiles = t_n.div_ceil(cube_size);

    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_base = tile * cube_size;
        let pos = tile_base + tid;
        let valid_pos = pos < t_n;

        // ── Phase 1: Each thread computes softcapped Q·K score ──
        let mut my_score = f32::new(-1e30f32);
        if valid_pos {
            let k_base = pos * kv_stride + kv_off;
            let mut dot = f32::new(0.0f32);
            let mut d = 0u32;
            while d < head_dim {
                dot += query[(head_off + d) as usize] * kv[(k_base + d) as usize];
                d += 1u32;
            }
            let raw = dot * scale;
            my_score = softcap * f32::tanh(raw / softcap);
        }
        smem[tid as usize] = my_score;
        sync_cube();

        // ── Phase 2: Unrolled parallel max reduction → tile_max ──
        if tid < 128u32 && smem[(tid + 128u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 128u32) as usize];
        }
        sync_cube();
        if tid < 64u32 && smem[(tid + 64u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 64u32) as usize];
        }
        sync_cube();
        if tid < 32u32 && smem[(tid + 32u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 32u32) as usize];
        }
        sync_cube();
        if tid < 16u32 && smem[(tid + 16u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 16u32) as usize];
        }
        sync_cube();
        if tid < 8u32 && smem[(tid + 8u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 8u32) as usize];
        }
        sync_cube();
        if tid < 4u32 && smem[(tid + 4u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 4u32) as usize];
        }
        sync_cube();
        if tid < 2u32 && smem[(tid + 2u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 2u32) as usize];
        }
        sync_cube();
        if tid < 1u32 && smem[1usize] > smem[0usize] {
            smem[0usize] = smem[1usize];
        }
        sync_cube();
        let tile_max = smem[0usize];
        sync_cube();

        // ── Phase 3: Compute exp(score - tile_max) ──
        let mut my_exp = f32::new(0.0f32);
        if valid_pos {
            my_exp = f32::exp(my_score - tile_max);
        }
        smem[tid as usize] = my_exp;
        sync_cube();

        // ── Phase 4: Weighted value accumulation for output dimension tid ──
        let mut tile_val = f32::new(0.0f32);
        if valid_dim {
            // Iterate full cube_size; only positions < t_n contribute
            // (out-of-bounds smem entries are 0 from Phase 3).
            let mut i = 0u32;
            while i < cube_size {
                let pos_i = tile_base + i;
                if pos_i < t_n {
                    tile_val = tile_val
                        + smem[i as usize]
                            * kv[(kv_half + pos_i * kv_stride + kv_off + tid) as usize];
                }
                i += 1u32;
            }
        }

        // ── Phase 5: Unrolled parallel sum reduction → tile_sum ──
        sync_cube();
        if tid < 128u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 128u32) as usize];
        }
        sync_cube();
        if tid < 64u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 64u32) as usize];
        }
        sync_cube();
        if tid < 32u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 32u32) as usize];
        }
        sync_cube();
        if tid < 16u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 16u32) as usize];
        }
        sync_cube();
        if tid < 8u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 8u32) as usize];
        }
        sync_cube();
        if tid < 4u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 4u32) as usize];
        }
        sync_cube();
        if tid < 2u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 2u32) as usize];
        }
        sync_cube();
        if tid < 1u32 {
            smem[0usize] = smem[0usize] + smem[1usize];
        }
        sync_cube();
        let tile_sum = smem[0usize];

        // ── Phase 6: Online softmax update ──
        let mut new_max = running_max;
        if tile_max > new_max {
            new_max = tile_max;
        }
        let prev_corr = f32::exp(running_max - new_max);
        let curr_corr = f32::exp(tile_max - new_max);

        running_sum *= prev_corr;

        running_sum += tile_sum * curr_corr;
        running_out *= prev_corr;
        running_out += tile_val * curr_corr;
        running_max = new_max;

        sync_cube();
        tile += 1u32;
    }

    // ── Final: Normalize and write output ──
    if valid_dim {
        attn_out[(head_off + tid) as usize] = running_out / running_sum;
    }
}

// ---------------------------------------------------------------------------
// Public parameter struct
// ---------------------------------------------------------------------------

/// External parameters for the CubeCL flash attention decode kernel.
///
/// Used for validation and array size computation in the launcher.
/// The parametric kernel (`attention_decode_llama_f32`) reads these from
/// a params buffer, supporting any LLaMA-family model configuration.
///
/// # Combined KV Buffer Layout
///
/// The kernel expects a combined KV buffer: `keys || values`.
/// Use [`AttentionParams::combine_kv`] to create this buffer from separate
/// key/value arrays, or provide your own pre-combined buffer.
///
/// # Example
///
/// ```rust,ignore
/// let params = AttentionParams::default();
/// let params = AttentionParams { n_positions: 512, ..Default::default() };
/// let combined_kv = params.combine_kv(&keys, &values);
/// ```
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone, Copy, Debug)]
pub struct AttentionParams {
    /// Number of query heads (e.g., 8 for Gemma 2 2B, 16 for MiniCPM5-1B).
    pub n_head: usize,
    /// Number of KV heads (e.g., 4 for Gemma 2 2B, 2 for MiniCPM5-1B).
    pub n_kv_head: usize,
    /// Head dimension (e.g., 256 for Gemma 2 2B, 128 for MiniCPM5-1B).
    /// Must be ≤ 256 (cube size).
    pub head_dim: usize,
    /// Number of valid KV positions (derived from combined KV buffer length).
    pub n_positions: usize,
    /// Logit softcap value (50.0 for Gemma 2, 0.0 for LLaMA/MiniCPM).
    /// When 0.0, softcapping is bypassed.
    pub softcap: f32,
    /// Attention scale: 1/√head_dim.
    pub scale: f32,
}

#[cfg(feature = "cubecl_runtime")]
impl AttentionParams {
    /// KV stride: `n_kv_head * head_dim` bytes per position.
    #[inline]
    pub fn kv_stride(&self) -> usize {
        self.n_kv_head * self.head_dim
    }

    /// Combine separate key and value arrays into a single buffer.
    ///
    /// Layout: `[keys(n_pos × kv_stride) | values(n_pos × kv_stride)]`.
    pub fn combine_kv(&self, keys: &[f32], values: &[f32]) -> Vec<f32> {
        let kv_len = self.n_positions * self.kv_stride();
        assert_eq!(keys.len(), kv_len, "keys length mismatch");
        assert_eq!(values.len(), kv_len, "values length mismatch");
        let mut combined = Vec::with_capacity(kv_len * 2);
        combined.extend_from_slice(keys);
        combined.extend_from_slice(values);
        combined
    }

    /// Split a combined KV buffer back into separate key/value slices.
    pub fn split_kv<'a>(&self, combined: &'a [f32]) -> (&'a [f32], &'a [f32]) {
        let kv_len = self.n_positions * self.kv_stride();
        assert_eq!(combined.len(), kv_len * 2, "combined KV length mismatch");
        combined.split_at(kv_len)
    }
}

#[cfg(feature = "cubecl_runtime")]
impl Default for AttentionParams {
    fn default() -> Self {
        Self {
            n_head: 8,
            n_kv_head: 4,
            head_dim: 256,
            n_positions: 0,
            softcap: 50.0,
            scale: 0.0625, // 1/√256
        }
    }
}

/// Parameters for block-causal attention kernel (Plan 108 T2).
///
/// Extends [`AttentionParams`] with block-causal masking parameters:
/// `pos`, `prompt_len`, and `block_size`. These control the per-position
/// attention boundary via `block_causal_t_n` logic.
///
/// # Combined KV Buffer Layout
///
/// Same as [`AttentionParams`]: combined KV buffer `keys ‖ values`.
/// Use [`AttentionBlockCausalParams::combine_kv`] to create this buffer.
///
/// # Kernel Params Buffer
///
/// The launcher creates a 3-element `f32` params buffer:
/// ```text
/// [query_pos, prompt_len, block_size]
/// ```
/// These are cast to `u32` inside the kernel (exact for integers < 2²⁴).
///
/// # Block-Causal Masking Rules
///
/// Mirrors `riir_engine::transformer::block_causal_t_n`:
/// - Prompt positions (`pos < prompt_len`): attend to all prompt positions
/// - Generation positions: bidirectional within block, causal across blocks
#[cfg(feature = "gemma2_d2f")]
#[derive(Clone, Copy, Debug)]
pub struct AttentionBlockCausalParams {
    /// Number of query heads (must be 8 for Gemma 2 2B).
    pub n_head: usize,
    /// Number of KV heads (must be 4 for Gemma 2 2B).
    pub n_kv_head: usize,
    /// Head dimension (must be 256 for Gemma 2 2B).
    pub head_dim: usize,
    /// Number of valid KV positions in the combined cache.
    pub n_positions: usize,
    /// Logit softcap value (50.0 for Gemma 2).
    pub softcap: f32,
    /// Attention scale: 1/√head_dim (0.0625 for head_dim=256).
    pub scale: f32,
    /// Query position index — which token's query we're computing attention for.
    pub pos: usize,
    /// Number of prompt tokens (prefix with bidirectional attention).
    pub prompt_len: usize,
    /// Block size for generation tokens (bidirectional within block).
    pub block_size: usize,
}

#[cfg(feature = "gemma2_d2f")]
impl AttentionBlockCausalParams {
    /// KV stride: `n_kv_head * head_dim` bytes per position.
    #[inline]
    pub fn kv_stride(&self) -> usize {
        self.n_kv_head * self.head_dim
    }

    /// Combine separate key and value arrays into a single buffer.
    ///
    /// Layout: `[keys(n_pos × kv_stride) | values(n_pos × kv_stride)]`.
    pub fn combine_kv(&self, keys: &[f32], values: &[f32]) -> Vec<f32> {
        let kv_len = self.n_positions * self.kv_stride();
        assert_eq!(keys.len(), kv_len, "keys length mismatch");
        assert_eq!(values.len(), kv_len, "values length mismatch");
        let mut combined = Vec::with_capacity(kv_len * 2);
        combined.extend_from_slice(keys);
        combined.extend_from_slice(values);
        combined
    }

    /// Split a combined KV buffer back into separate key/value slices.
    pub fn split_kv<'a>(&self, combined: &'a [f32]) -> (&'a [f32], &'a [f32]) {
        let kv_len = self.n_positions * self.kv_stride();
        assert_eq!(combined.len(), kv_len * 2, "combined KV length mismatch");
        combined.split_at(kv_len)
    }

    /// Compute the block-causal attention boundary `t_n`.
    ///
    /// Returns the number of KV positions this query position attends to.
    /// Same logic as `riir_engine::transformer::block_causal_t_n`.
    pub fn t_n(&self) -> usize {
        if self.pos < self.prompt_len {
            self.prompt_len
        } else {
            let block_idx = (self.pos - self.prompt_len) / self.block_size;
            let block_end = self.prompt_len + (block_idx + 1) * self.block_size;
            block_end.min(self.n_positions)
        }
    }
}

#[cfg(feature = "gemma2_d2f")]
impl Default for AttentionBlockCausalParams {
    fn default() -> Self {
        Self {
            n_head: 8,
            n_kv_head: 4,
            head_dim: 256,
            n_positions: 0,
            softcap: 50.0,
            scale: 0.0625,
            pos: 0,
            prompt_len: 0,
            block_size: 16,
        }
    }
}

// ---------------------------------------------------------------------------
// Public launcher
// ---------------------------------------------------------------------------

/// CubeCL flash attention decode launcher.
///
/// Provides the public API for launching the CubeCL flash attention kernel
/// for LLaMA-family decode. Single dispatch handles all query heads.
///
/// Uses a combined KV buffer (keys || values) plus a params buffer for
/// dynamic model dimensions.
///
/// # Example
///
/// ```rust,ignore
/// let ctx = CubeCLContext::new()?;
/// let client = ctx.client();
///
/// let params = AttentionParams { n_positions: 512, ..Default::default() };
/// let combined_kv = params.combine_kv(&keys, &values);
/// let kv_handle = client.create_from_slice(f32::as_bytes(&combined_kv));
///
/// AttentionCubeCL::launch::<ActiveRuntime>(
///     &client, q_handle, kv_handle, out_handle, &params,
/// );
/// ```
#[cfg(feature = "cubecl_runtime")]
pub struct AttentionCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)] // Used in forward pass wiring
impl AttentionCubeCL {
    /// Launch flash attention decode kernel for LLaMA-family models.
    ///
    /// Single dispatch for all query heads. Each workgroup (256 threads)
    /// handles one head with tiled KV scan and online softmax.
    ///
    /// Uses combined KV buffer layout: `[keys | values]`.
    /// Use [`AttentionParams::combine_kv`] to create the combined buffer.
    ///
    /// Dispatch: `(n_head, 1, 1)` workgroups of 256 threads each.
    ///
    /// **Fold dispatch (Plan 179 D1):** When `fold_dispatch` feature is enabled and
    /// `seq_len_q > 1`, uses [`fold_factor`] to coalesce query tokens for higher
    /// GPU occupancy during speculative decode. Requires GOAT proof validation.
    ///
    /// # Constraints
    ///
    /// - `head_dim` must be ≤ 256 (cube size)
    /// - `softcap` == 0.0 disables softcapping (for LLaMA/MiniCPM models)
    pub fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        kv_handle: Handle,
        attn_out_handle: Handle,
        params: &AttentionParams,
    ) {
        assert!(
            params.head_dim <= 256,
            "head_dim must be ≤ 256 (cube size), got {}",
            params.head_dim
        );
        assert!(params.n_head > 0, "n_head must be > 0");
        assert!(params.n_kv_head > 0, "n_kv_head must be > 0");

        let n_head = params.n_head as u32;
        let q_len = params.n_head * params.head_dim;
        let combined_kv_len = 2 * params.n_positions * params.n_kv_head * params.head_dim;

        // Both kernels below compute `n_positions = kv.len() / 2 / kv_stride`
        // from the BOUND BUFFER, so an oversized binding silently changes the
        // kernel's idea of its own shape and it reads the value half from the
        // wrong offset — an identically-zero output, no panic (riir-train
        // `.issues/511`, class `.issues/515`).
        crate::cubecl_runtime::assert_binding_derives_units(
            &kv_handle,
            2 * params.n_kv_head * params.head_dim,
            params.n_positions,
            "AttentionCubeCL::launch kv",
        );

        // Fast path: use hardcoded kernel for Gemma 2 2B constants.
        // The generic kernel reads params from a buffer per dispatch, preventing
        // CubeCL JIT from constant-folding dimensions into optimized Metal shaders.
        // This fast path avoids the params buffer allocation and per-tile branching
        // overhead, recovering ~24% decode throughput (31.4 → 41.4 tok/s on M3 Max).
        if params.n_head == 8
            && params.n_kv_head == 4
            && params.head_dim == 256
            && params.softcap == 50.0
        {
            // SAFETY: Caller guarantees correct buffer sizes.
            unsafe {
                attention_decode_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(8, 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(query_handle, q_len),
                    BufferArg::from_raw_parts(kv_handle, combined_kv_len),
                    BufferArg::from_raw_parts(attn_out_handle, q_len),
                );
            }
            return;
        }

        // Generic path: parametric kernel for other architectures
        // Create kernel params buffer: [n_head, n_kv_head, head_dim, softcap, scale]
        let kernel_params: &[f32] = &[
            params.n_head as f32,
            params.n_kv_head as f32,
            params.head_dim as f32,
            params.softcap,
            params.scale,
        ];
        let params_handle = client.create_from_slice(f32::as_bytes(kernel_params));

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            attention_decode_llama_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(query_handle, q_len),
                BufferArg::from_raw_parts(kv_handle, combined_kv_len),
                BufferArg::from_raw_parts(params_handle, 5),
                BufferArg::from_raw_parts(attn_out_handle, q_len),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Folded dispatch launcher (Plan 179 D1)
// ---------------------------------------------------------------------------

/// CubeCL flash attention folded dispatch launcher (Plan 179 D1).
///
/// Launches the `attention_decode_folded_f32` kernel for speculative decode
/// with multiple query tokens. Each workgroup (256 threads) handles one head
/// but processes `fold_factor` query tokens sequentially, increasing GPU
/// occupancy when `n_head < SM_count`.
///
/// Feature-gated behind `fold_dispatch` — opt-in until GOAT proof validates
/// ≥5% gain on Apple Silicon.
#[cfg(feature = "fold_dispatch")]
#[cfg(feature = "cubecl_runtime")]
impl AttentionCubeCL {
    /// Launch flash attention decode with folded query tokens for speculative decode.
    ///
    /// Each workgroup (256 threads) handles one head but processes multiple query
    /// tokens. The `fold_factor` controls how many tokens are coalesced per workgroup.
    ///
    /// Dispatch: `(n_head, 1, 1)` workgroups of 256 threads each.
    ///
    /// # Arguments
    ///
    /// - `client`: CubeCL compute client
    /// - `query_handle`: `[seq_len_q × n_head × head_dim]` f32 — batched queries
    /// - `kv_handle`: `[2 × n_positions × n_kv_head × head_dim]` f32 — combined KV
    /// - `attn_out_handle`: `[seq_len_q × n_head × head_dim]` f32 — batched output
    /// - `params`: attention parameters (n_head, head_dim, etc.)
    /// - `seq_len_q`: number of query tokens (must be > 1 for folding benefit)
    ///
    /// # Panics
    ///
    /// Panics if params don't match Gemma 2 2B constants (n_head=8, n_kv_head=4,
    /// head_dim=256, softcap=50.0) or if seq_len_q == 0.
    pub fn launch_folded<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        kv_handle: Handle,
        attn_out_handle: Handle,
        params: &AttentionParams,
        seq_len_q: usize,
    ) {
        assert_eq!(
            params.n_head, 8,
            "fold_dispatch only supports Gemma 2 2B n_head=8"
        );
        assert_eq!(
            params.n_kv_head, 4,
            "fold_dispatch only supports Gemma 2 2B n_kv_head=4"
        );
        assert_eq!(
            params.head_dim, 256,
            "fold_dispatch only supports head_dim=256"
        );
        assert_eq!(
            params.softcap, 50.0,
            "fold_dispatch only supports softcap=50.0"
        );
        assert!(seq_len_q > 0, "seq_len_q must be > 0");

        // The folded kernel derives `n_positions = kv.len() / 2 / kv_stride`
        // from the BOUND BUFFER exactly like the decode kernels — same
        // `.issues/515` class, so same guard (515 T2 sweep: this launcher was
        // the unguarded sibling of `launch`'s T4 guard).
        crate::cubecl_runtime::assert_binding_derives_units(
            &kv_handle,
            2 * params.n_kv_head * params.head_dim,
            params.n_positions,
            "AttentionCubeCL::launch_folded kv",
        );

        let n_head = params.n_head as u32;
        let workgroup_size = 256u32;
        let fold = fold_factor(n_head, seq_len_q as u32, workgroup_size);

        let q_len = seq_len_q * params.n_head * params.head_dim;
        let combined_kv_len = 2 * params.n_positions * params.n_kv_head * params.head_dim;

        // Params buffer: [n_head, fold, seq_len_q, 0.0]
        let kernel_params: &[f32] = &[params.n_head as f32, fold as f32, seq_len_q as f32, 0.0];
        let params_handle = client.create_from_slice(f32::as_bytes(kernel_params));

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            attention_decode_folded_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(query_handle, q_len),
                BufferArg::from_raw_parts(kv_handle, combined_kv_len),
                BufferArg::from_raw_parts(params_handle, 4),
                BufferArg::from_raw_parts(attn_out_handle, q_len),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Block-causal attention launcher (Plan 108 T2)
// ---------------------------------------------------------------------------

/// CubeCL flash attention block-causal decode launcher (Plan 108 T2).
///
/// Launches the `attention_block_causal_f32` kernel for Gemma 2 decode
/// with block-causal masking. Single dispatch handles all 8 query heads.
///
/// The kernel receives a 3-element `f32` params buffer:
/// ```text
/// [query_pos, prompt_len, block_size]
/// ```
#[cfg(feature = "gemma2_d2f")]
impl AttentionCubeCL {
    /// Launch flash attention block-causal decode kernel for Gemma 2.
    ///
    /// Single dispatch for all 8 query heads. Each workgroup (256 threads)
    /// handles one head with tiled KV scan and online softmax, attending
    /// only to positions `0..t_n` where `t_n = block_causal_t_n(...)`.
    ///
    /// Uses combined KV buffer layout: `[keys | values]`.
    /// Use [`AttentionBlockCausalParams::combine_kv`] to create the combined buffer.
    ///
    /// Dispatch: `(n_head, 1, 1)` workgroups of 256 threads each.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `query_handle`: n_head × head_dim = 2048 f32 elements
    /// - `kv_handle`: 2 × n_positions × n_kv_head × head_dim f32 elements
    ///   (keys followed by values)
    /// - `attn_out_handle`: n_head × head_dim = 2048 f32 elements
    ///
    /// # Panics
    ///
    /// Panics if params don't match Gemma 2 2B constants (n_head=8, n_kv_head=4,
    /// head_dim=256) or if block_size is 0.
    pub fn launch_block_causal<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        kv_handle: Handle,
        attn_out_handle: Handle,
        params: &AttentionBlockCausalParams,
    ) {
        assert_eq!(params.n_head, 8, "Only Gemma 2 2B n_head=8 supported");
        assert_eq!(params.n_kv_head, 4, "Only Gemma 2 2B n_kv_head=4 supported");
        assert_eq!(
            params.head_dim, 256,
            "Only Gemma 2 2B head_dim=256 supported"
        );
        assert!(params.block_size > 0, "block_size must be > 0");

        let n_head = params.n_head as u32;
        let q_len = params.n_head * params.head_dim;
        let combined_kv_len = 2 * params.n_positions * params.n_kv_head * params.head_dim;

        // The block-causal kernel derives `n_positions` from the BOUND kv
        // buffer exactly like the decode kernels — same `.issues/515` class,
        // same guard (515 T2 sweep: this launcher was the unguarded sibling
        // of `launch`'s T4 guard).
        crate::cubecl_runtime::assert_binding_derives_units(
            &kv_handle,
            2 * params.n_kv_head * params.head_dim,
            params.n_positions,
            "AttentionCubeCL::launch_block_causal kv",
        );

        // Create kernel params buffer: [query_pos, prompt_len, block_size] as f32
        let kernel_params: &[f32] = &[
            params.pos as f32,
            params.prompt_len as f32,
            params.block_size as f32,
        ];
        let params_handle = client.create_from_slice(f32::as_bytes(kernel_params));

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            attention_block_causal_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(query_handle, q_len),
                BufferArg::from_raw_parts(kv_handle, combined_kv_len),
                BufferArg::from_raw_parts(params_handle, 3),
                BufferArg::from_raw_parts(attn_out_handle, q_len),
            );
        }
    }
}

/// Compute the fold factor for coalescing query tokens into fewer CubeCL workgroups.
///
/// TokenSpeed insight: when `seq_len_q > 1` (speculative decode), each workgroup
/// processes one head. If n_head < GPU SM count, many SMs sit idle. Folding
/// multiple query tokens into the same workgroup increases occupancy.
///
/// Returns the largest divisor of `seq_len_q` that, when combined with `n_head`,
/// fits within `workgroup_size` threads. Returns 1 if no folding is beneficial.
///
/// # Arguments
/// - `n_head`: number of query heads
/// - `seq_len_q`: number of query tokens (1 for autoregressive, >1 for speculative)
/// - `workgroup_size`: threads per workgroup (typically 256)
///
/// # Examples
/// - `(8, 1, 256)` → 1 (single-token decode, nothing to fold)
/// - `(8, 4, 256)` → 4 (4 tokens × 8 heads = 32, well within 256)
/// - `(32, 8, 256)` → 8 (8 tokens, 32 heads → max_fold=8, largest divisor of 8 ≤ 8 is 8)
#[cfg(feature = "fold_dispatch")]
pub fn fold_factor(n_head: u32, seq_len_q: u32, workgroup_size: u32) -> u32 {
    if seq_len_q <= 1 || n_head == 0 {
        return 1;
    }

    let max_fold = seq_len_q.min(workgroup_size / n_head);

    if max_fold <= 1 {
        return 1;
    }

    // Find the largest divisor of seq_len_q that is <= max_fold.
    // Search downward from max_fold for efficiency.
    (1..=max_fold)
        .rev()
        .find(|&d| seq_len_q.is_multiple_of(d))
        .unwrap_or(1)
}


#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests;
