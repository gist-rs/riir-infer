//! CubeCL attention decode kernel for Qwen3.5 full-attention layers (Issue 599).
//!
//! Implements single-token decode attention with:
//! - Partial RoPE (first `rotary_dim` of `head_dim`)
//! - Q/K RMSNorm per head
//! - GQA (n_kv_head ≤ n_head)
//! - Online softmax (flash attention pattern)
//! - Output gating (sigmoid)
//!
//! # Dispatch
//!
//! | Kernel | CubeDim | CubeCount | Responsibility |
//! |--------|---------|-----------|----------------|
//! | `qwen_rope_partial_f32` | `new_1d(128)` | `(ceil(n_elem/128), 1, 1)` | elementwise RoPE |
//! | `qwen_split_qg_f32` | `new_1d(128)` | `(ceil(q_dim/128), 1, 1)` | split qg → q + gate |
//! | `qwen_kv_cache_append_f32` | `new_1d(128)` | `(ceil(kvd/128), 1, 1)` | copy K/V to cache |
//! | `qwen_attention_decode_f32` | `new_1d(head_dim)` | `(n_head, 1, 1)` | flash attention |
//! | `qwen_output_gate_f32` | `new_1d(128)` | `(ceil(q_dim/128), 1, 1)` | sigmoid gate |

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]

// ---------------------------------------------------------------------------
// Partial RoPE kernel
// ---------------------------------------------------------------------------

/// Apply partial RoPE to Q and K in-place.
///
/// Rotates the first `rotary_dim` dimensions of each head using:
/// ```text
/// out[2i]   = x[2i] * cos(theta) - x[2i+1] * sin(theta)
/// out[2i+1] = x[2i] * sin(theta) + out[2i+1] * cos(theta)
/// ```
/// where `theta = pos * inv_freq[i]`, `inv_freq[i] = 1 / theta_base^(2i/rotary_dim)`.
///
/// Dimensions beyond `rotary_dim` pass through unchanged.
///
/// Each thread handles one pair (2 elements). Threads beyond `n_head * rotary_dim / 2`
/// terminate.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn qwen_rope_partial_f32(
    q: &mut [f32],
    k: &mut [f32],
    params: &[f32],
) {
    // params layout: [rotary_dim_pairs, pos_f32, theta_base, head_dim, n_head, n_kv_head]
    let rotary_pairs = params[0usize] as usize;
    let pos = params[1usize];
    let theta_base = params[2usize];
    let head_dim = params[3usize] as usize;
    let n_head = params[4usize] as usize;
    let n_kv_head = params[5usize] as usize;

    let idx = ABSOLUTE_POS;

    // Each thread handles one (head, pair) for both Q and K.
    // Total pairs per buffer: n_head * rotary_pairs (for Q) or n_kv_head * rotary_pairs (for K).
    // We dispatch enough threads to cover the larger (Q) buffer.
    let total_q_pairs = n_head * rotary_pairs;

    if idx >= total_q_pairs {
        terminate!();
    }

    // Decode (head, pair_idx) from idx
    let head = idx / rotary_pairs;
    let pair = idx % rotary_pairs;

    // Compute frequency: inv_freq = 1 / theta_base^(2*pair / rotary_dim)
    let rotary_dim_f = (2usize * rotary_pairs) as f32;
    let exponent = (2usize * pair) as f32 / rotary_dim_f;
    // powf not available in CubeCL — use exp(log(theta_base) * exponent)
    let log_base = theta_base.ln();
    let inv_freq = (log_base * (f32::new(0.0f32) - exponent)).exp();

    let theta = pos * inv_freq;
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    // Q rotation (rotate-half / GPT-NeoX convention — matches CPU
    // `apply_rope_heads_precomputed`: pairs are (vec[i], vec[i + half])
    // where half = rotary_dim / 2 = rotary_pairs, NOT interleaved
    // (vec[2*pair], vec[2*pair+1]).
    //
    // Issue 604 G1 (2026-08-11): the original code used the interleaved
    // (GPT-J) convention, which is identity at pos=0 (so G1 pos 0 passed
    // bit-exactly) but diverges at pos>=1 — every attention layer fed
    // wrongly-rotated Q/K, causing max_rel to compound from 0.5% at layer 3
    // to 112% in the final logits. Fixed by switching to the rotate-half
    // pairing that the CPU reference uses.
    let q_head_off = head * head_dim;
    let q_i0 = q_head_off + pair;
    let q_i1 = q_i0 + rotary_pairs;
    let q0 = q[q_i0];
    let q1 = q[q_i1];
    q[q_i0] = q0 * cos_t - q1 * sin_t;
    q[q_i1] = q0 * sin_t + q1 * cos_t;

    // K rotation (only if this head index < n_kv_head)
    if head < n_kv_head {
        let k_head_off = head * head_dim;
        let k_i0 = k_head_off + pair;
        let k_i1 = k_i0 + rotary_pairs;
        let k0 = k[k_i0];
        let k1 = k[k_i1];
        k[k_i0] = k0 * cos_t - k1 * sin_t;
        k[k_i1] = k0 * sin_t + k1 * cos_t;
    }
}

/// Launch partial RoPE kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenRopePartialCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenRopePartialCubeCL {
    /// Apply partial RoPE to Q and K in-place.
    ///
    /// # Safety
    /// - `q_handle`: `n_head * head_dim` f32 elements
    /// - `k_handle`: `n_kv_head * head_dim` f32 elements
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q_handle: Handle,
        k_handle: Handle,
        pos: usize,
        rotary_dim: usize,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        theta_base: f32,
    ) {
        let rotary_pairs = rotary_dim / 2;
        let total_pairs = n_head * rotary_pairs;
        let params: [f32; 6] = [
            rotary_pairs as f32,
            pos as f32,
            theta_base,
            head_dim as f32,
            n_head as f32,
            n_kv_head as f32,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));

        let wg_size = 128u32;
        let num_wg = (total_pairs as u32).div_ceil(wg_size).max(1);

        unsafe {
            qwen_rope_partial_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(q_handle, n_head * head_dim),
                BufferArg::from_raw_parts(k_handle, n_kv_head * head_dim),
                BufferArg::from_raw_parts(params_handle, 6),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Split QG (gated Q projection output) into Q and gate
// ---------------------------------------------------------------------------

/// Split the interleaved `[q(hd), gate(hd)]` per-head layout into separate
/// `q` and `gate` buffers.
///
/// Input layout: `qg[h] = [q[h*hd .. h*hd+hd], gate[h*hd .. h*hd+hd]]`
/// Output: `q[h*hd .. (h+1)*hd]` and `gate[h*hd .. (h+1)*hd]`
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn qwen_split_qg_f32(
    qg: &[f32],
    q: &mut [f32],
    gate: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as usize;
    let n_head = params[1usize] as usize;
    let idx = ABSOLUTE_POS;

    if idx >= n_head * head_dim {
        terminate!();
    }

    let head = idx / head_dim;
    let dim = idx % head_dim;

    // Source in qg: head * 2*hd + dim (for q), head * 2*hd + hd + dim (for gate)
    let q_src = head * 2usize * head_dim + dim;
    let gate_src = q_src + head_dim;

    q[idx] = qg[q_src];
    gate[idx] = qg[gate_src];
}

/// Launch QG split kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenSplitQgCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenSplitQgCubeCL {
    /// Split interleaved QG buffer into separate Q and gate buffers.
    ///
    /// # Safety
    /// - `qg_handle`: `2 * n_head * head_dim` f32 elements
    /// - `q_handle`, `gate_handle`: `n_head * head_dim` f32 elements each
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        qg_handle: Handle,
        q_handle: Handle,
        gate_handle: Handle,
        head_dim: usize,
        n_head: usize,
    ) {
        let total = n_head * head_dim;
        let params: [f32; 2] = [head_dim as f32, n_head as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));

        let wg_size = 128u32;
        let num_wg = (total as u32).div_ceil(wg_size).max(1);

        unsafe {
            qwen_split_qg_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(qg_handle, 2 * total),
                BufferArg::from_raw_parts(q_handle, total),
                BufferArg::from_raw_parts(gate_handle, total),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// KV cache append kernel
// ---------------------------------------------------------------------------

/// Append current K and V vectors to the KV cache at position `pos`.
///
/// Copies `k_vec[kvd]` to `kv_cache_key[pos * kvd .. (pos+1) * kvd]`
/// and `v_vec[kvd]` to `kv_cache_value[pos * kvd .. (pos+1) * kvd]`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn qwen_kv_cache_append_f32(
    k_vec: &[f32],
    v_vec: &[f32],
    kv_cache_key: &mut [f32],
    kv_cache_value: &mut [f32],
    params: &[f32],
) {
    let kvd = params[0usize] as usize;
    let pos = params[1usize] as usize;
    let idx = ABSOLUTE_POS;

    if idx >= kvd {
        terminate!();
    }

    let cache_off = pos * kvd + idx;
    kv_cache_key[cache_off] = k_vec[idx];
    kv_cache_value[cache_off] = v_vec[idx];
}

/// Issue 648 F9: KV cache append from a combined KV buffer.
///
/// Same as `qwen_kv_cache_append_f32` but reads K from `kv_vec[0..kvd]`
/// and V from `kv_vec[kvd..2*kvd]` in a single buffer. Eliminates the
/// need for a separate V GEMV dispatch — the combined K+V GEMV writes
/// both in one shot.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn qwen_kv_cache_append_combined_f32(
    kv_vec: &[f32],
    kv_cache_key: &mut [f32],
    kv_cache_value: &mut [f32],
    params: &[f32],
) {
    let kvd = params[0usize] as usize;
    let pos = params[1usize] as usize;
    let idx = ABSOLUTE_POS;

    if idx >= kvd {
        terminate!();
    }

    let cache_off = pos * kvd + idx;
    kv_cache_key[cache_off] = kv_vec[idx];
    kv_cache_value[cache_off] = kv_vec[kvd + idx];
}

/// Launch KV cache append kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenKvCacheAppendCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenKvCacheAppendCubeCL {
    /// Append K and V to the KV cache at position `pos`.
    ///
    /// # Safety
    /// - `k_handle`, `v_handle`: `kvd` f32 elements each
    /// - `key_cache_handle`, `value_cache_handle`: `max_seq_len * kvd` f32 each
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        k_handle: Handle,
        v_handle: Handle,
        key_cache_handle: Handle,
        value_cache_handle: Handle,
        kvd: usize,
        pos: usize,
    ) {
        let params: [f32; 2] = [kvd as f32, pos as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));

        let wg_size = 128u32;
        let num_wg = (kvd as u32).div_ceil(wg_size).max(1);

        unsafe {
            qwen_kv_cache_append_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(k_handle, kvd),
                BufferArg::from_raw_parts(v_handle, kvd),
                BufferArg::from_raw_parts(key_cache_handle, (pos + 1) * kvd),
                BufferArg::from_raw_parts(value_cache_handle, (pos + 1) * kvd),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

/// Launch KV cache append kernel from a combined KV buffer (Issue 648 F9).
#[cfg(feature = "cubecl_runtime")]
pub struct QwenKvCacheAppendCombinedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenKvCacheAppendCombinedCubeCL {
    /// Append K and V to the KV cache at position `pos` from a combined buffer.
    ///
    /// K occupies `kv_handle[0..kvd]`, V occupies `kv_handle[kvd..2*kvd]`.
    ///
    /// # Safety
    /// - `kv_handle`: `2 * kvd` f32 elements (K at [0..kvd], V at [kvd..2*kvd])
    /// - `key_cache_handle`, `value_cache_handle`: `max_seq_len * kvd` f32 each
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        kv_handle: Handle,
        key_cache_handle: Handle,
        value_cache_handle: Handle,
        kvd: usize,
        pos: usize,
    ) {
        let params: [f32; 2] = [kvd as f32, pos as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));

        let wg_size = 128u32;
        let num_wg = (kvd as u32).div_ceil(wg_size).max(1);

        unsafe {
            qwen_kv_cache_append_combined_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(kv_handle, 2 * kvd),
                BufferArg::from_raw_parts(key_cache_handle, (pos + 1) * kvd),
                BufferArg::from_raw_parts(value_cache_handle, (pos + 1) * kvd),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Flash attention decode kernel (parameterized)
// ---------------------------------------------------------------------------

/// Parameterized flash attention decode for Qwen3.5.
///
/// Each workgroup (head_dim threads) handles one query head, scanning the
/// full KV cache with online softmax accumulation.
///
/// ## Parameter Layout
///
/// - `query`: `[n_head * head_dim]`
/// - `key_cache`: `[n_pos × kv_stride]` where `kv_stride = n_kv_head * head_dim`
/// - `value_cache`: `[n_pos × kv_stride]`
/// - `attn_out`: `[n_head * head_dim]` output
/// - `params`: `[head_dim, n_head, n_kv_head, n_positions, scale]`
///
/// ## GQA Mapping
///
/// `kv_group(h) = h * n_kv_head / n_head`
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_decode_f32(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as u32;
    let n_head = params[1usize] as u32;
    let n_kv_head = params[2usize] as u32;
    let n_positions = params[3usize] as u32;
    let scale = params[4usize];

    let cube_size = head_dim; // CubeDim::new_1d(head_dim)
    let head_idx = ABSOLUTE_POS as u32 / cube_size;
    let tid = UNIT_POS;

    if head_idx >= n_head {
        terminate!();
    }

    let head_off = head_idx * head_dim;

    // Guard: no positions — write zeros
    if n_positions == 0u32 {
        if tid < head_dim {
            attn_out[(head_off + tid) as usize] = f32::new(0.0f32);
        }
        terminate!();
    }

    // GQA mapping
    let kv_stride = n_kv_head * head_dim;
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_off = kv_group * head_dim;

    let valid_dim = tid < head_dim;

    // Shared memory for max reduction + weight storage.
    //
    // Issue 612 (2026-08-11): the original code allocated 256 elements but only
    // dispatched cube_size (= head_dim = 128) threads. The first reduction step
    // read smem[tid + 128] which was uninitialized garbage, corrupting the max
    // and sum for any tile with >1 valid position (i.e. every token after the
    // first). Additionally, the parallel sum reduction was destructive — it
    // overwrote the weights that Phase 4 subsequently read, compounding the
    // error.
    //
    // Fix: (a) size smem to head_dim (= cube_size = 128), (b) start the max
    // reduction at stride 64 (= head_dim/2), (c) replace the destructive
    // parallel sum reduction with a serial sum computed inline during Phase 4's
    // position loop (zero extra cost — that loop already iterates over every
    // position).
    //
    // Issue 654 (2026-08-13): head_dim=256 for Bonsai-27B. smem must cover the
    // full cube (256 threads), and the reduction tree must start at stride 128.
    // For head_dim=128, the stride-128 step is guarded by `cube_size > 128`
    // (uniform runtime branch, safe with sync_cube) so existing head_dim=128
    // callers are unaffected.
    let mut smem = Shared::<[f32]>::new_slice(256usize);

    // Online softmax state
    let mut running_max = f32::new(-1e30f32);
    let mut running_sum = f32::new(0.0f32);
    let mut running_out = f32::new(0.0f32);

    // Process KV in tiles of head_dim (cube_size)
    let n_tiles = n_positions.div_ceil(cube_size);
    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_base = tile * cube_size;
        let pos = tile_base + tid;
        let valid_pos = pos < n_positions;

        // Phase 1: Q·K score
        let mut my_score = f32::new(-1e30f32);
        if valid_pos {
            let k_base = pos * kv_stride + kv_off;
            let mut dot = f32::new(0.0f32);
            let mut d = 0u32;
            while d < head_dim {
                dot += query[(head_off + d) as usize] * key_cache[(k_base + d) as usize];
                d += 1u32;
            }
            my_score = dot * scale;
        }
        smem[tid as usize] = my_score;
        sync_cube();

        // Phase 2: parallel max reduction (cube_size threads → 1).
        // For head_dim=256 (Bonsai-27B): stride 128 → 64 → … → 1.
        // For head_dim=128: stride 64 → … → 1 (stride-128 guarded out).
        // The guard is a uniform runtime branch — safe with sync_cube.
        //
        // Issue 715 (2026-08-17): this tree had three defects, all fixed by the
        // loop below. (1) The stride-64 step compared `smem[tid + 64]` but
        // assigned `smem[tid + 32]` — a copy-paste slip that mixed an unrelated
        // lane into the max. (2) The stride-32 step was missing entirely (64 →
        // 16). (3) It stopped at stride 8, leaving `smem[0]` a max over 32 of the
        // 256 lanes. As in the gated twin, `tile_max` only sets the softmax shift,
        // so results were correct but needlessly close to the f32 overflow edge.
        if cube_size > 128u32 {
            if tid < 128u32 && smem[(tid + 128u32) as usize] > smem[tid as usize] {
                smem[tid as usize] = smem[(tid + 128u32) as usize];
            }
            sync_cube();
        }
        let mut stride = 64u32;
        while stride > 0u32 {
            if tid < stride && smem[(tid + stride) as usize] > smem[tid as usize] {
                smem[tid as usize] = smem[(tid + stride) as usize];
            }
            sync_cube();
            stride /= 2u32;
        }

        let tile_max = smem[0usize];
        // Issue 715 race (a): see the gated twin — thread 0 overwrites `smem[0]`
        // in Phase 3 while slower lanes may still be reading it here.
        sync_cube();

        // Online softmax update: rescale running state
        let new_max = if tile_max > running_max {
            tile_max
        } else {
            running_max
        };

        let exp_prev = (running_max - new_max).exp();
        let exp_tile = (tile_max - new_max).exp();

        running_sum = running_sum * exp_prev;
        running_out = running_out * exp_prev;
        running_max = new_max;

        // Phase 3: compute exp(score - max) and store in smem.
        // These weights are NOT reduced — Phase 4 reads them directly.
        if valid_pos {
            smem[tid as usize] = exp_tile * (my_score - tile_max).exp();
        } else {
            smem[tid as usize] = f32::new(0.0f32);
        }
        sync_cube();

        // Phase 4: weighted value accumulation + serial sum.
        // Each thread iterates over all positions in the tile, accumulating
        // both the output (weight * V) and the tile sum. The serial sum
        // replaces the destructive parallel reduction that previously
        // corrupted the weights (Issue 612).
        let mut tile_sum = f32::new(0.0f32);
        let mut acc = f32::new(0.0f32);
        let mut p = 0u32;
        while p < cube_size {
            let kv_pos = tile_base + p;
            if kv_pos < n_positions {
                let weight = smem[p as usize];
                tile_sum = tile_sum + weight;
                if valid_dim {
                    let v_idx = kv_pos * kv_stride + kv_off + tid;
                    acc += weight * value_cache[v_idx as usize];
                }
            }
            p += 1u32;
        }
        if valid_dim {
            running_out += acc;
        }
        running_sum = running_sum + tile_sum;

        // Issue 715 race (b): barrier on the loop back-edge — the next tile's
        // Phase 1 overwrites `smem[tid]` that this tile's Phase 4 loop is still
        // reading. Only reachable when `n_tiles >= 2`. See the gated twin.
        sync_cube();

        tile += 1u32;
    }

    // Final normalization
    if valid_dim {
        let inv_sum = f32::new(1.0f32) / running_sum;
        attn_out[(head_off + tid) as usize] = running_out * inv_sum;
    }
}

/// Launch Qwen3.5 attention decode kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionDecodeCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionDecodeCubeCL {
    /// Launch flash attention decode for Qwen3.5.
    ///
    /// # Safety
    /// - `query_handle`: `n_head * head_dim` f32 elements
    /// - `key_cache_handle`, `value_cache_handle`: `n_positions * n_kv_head * head_dim` f32 each
    /// - `attn_out_handle`: `n_head * head_dim` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        key_cache_handle: Handle,
        value_cache_handle: Handle,
        attn_out_handle: Handle,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        n_positions: usize,
    ) {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let params: [f32; 5] = [
            head_dim as f32,
            n_head as f32,
            n_kv_head as f32,
            n_positions as f32,
            scale,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let q_len = n_head * head_dim;
        let kv_len = n_positions * n_kv_head * head_dim;

        unsafe {
            qwen_attention_decode_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head as u32, 1, 1),
                CubeDim::new_1d(head_dim as u32),
                BufferArg::from_raw_parts(query_handle, q_len),
                BufferArg::from_raw_parts(key_cache_handle, kv_len),
                BufferArg::from_raw_parts(value_cache_handle, kv_len),
                BufferArg::from_raw_parts(attn_out_handle, q_len),
                BufferArg::from_raw_parts(params_handle, 5),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Issue 648 F10: Gated flash attention decode (fused output gate)
// ---------------------------------------------------------------------------

/// Gated flash attention decode for Qwen3.5.
///
/// Identical to `qwen_attention_decode_f32` but applies `sigmoid(gate)` to
/// the output in the final write, eliminating the separate
/// `qwen_output_gate_f32` dispatch.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_decode_gated_f32(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    gate: &[f32],
    attn_out: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as u32;
    let n_head = params[1usize] as u32;
    let n_kv_head = params[2usize] as u32;
    let n_positions = params[3usize] as u32;
    let scale = params[4usize];

    let cube_size = head_dim;
    let head_idx = ABSOLUTE_POS as u32 / cube_size;
    let tid = UNIT_POS;

    if head_idx >= n_head {
        terminate!();
    }

    let head_off = head_idx * head_dim;

    if n_positions == 0u32 {
        if tid < head_dim {
            attn_out[(head_off + tid) as usize] = f32::new(0.0f32);
        }
        terminate!();
    }

    let kv_stride = n_kv_head * head_dim;
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_off = kv_group * head_dim;

    let valid_dim = tid < head_dim;

    // Shared memory for max reduction + weight storage.
    // Issue 654 (2026-08-13): head_dim=256 for Bonsai-27B. smem must cover the
    // full cube (256 threads). For head_dim=128, the stride-128 reduction step
    // is guarded by `cube_size > 128` (uniform runtime branch, safe with
    // sync_cube) so existing head_dim=128 callers are unaffected.
    let mut smem = Shared::<[f32]>::new_slice(256usize);

    let mut running_max = f32::new(-1e30f32);
    let mut running_sum = f32::new(0.0f32);
    let mut running_out = f32::new(0.0f32);

    let n_tiles = n_positions.div_ceil(cube_size);
    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_base = tile * cube_size;
        let pos = tile_base + tid;
        let valid_pos = pos < n_positions;

        let mut my_score = f32::new(-1e30f32);
        if valid_pos {
            let k_base = pos * kv_stride + kv_off;
            let mut dot = f32::new(0.0f32);
            let mut d = 0u32;
            while d < head_dim {
                dot += query[(head_off + d) as usize] * key_cache[(k_base + d) as usize];
                d += 1u32;
            }
            my_score = dot * scale;
        }
        smem[tid as usize] = my_score;
        sync_cube();

        // Max reduction. For head_dim=256: stride 128 → 64 → … → 1.
        // For head_dim=128: stride 64 → … → 1 (stride-128 guarded out).
        //
        // Issue 715 (2026-08-17): the tree previously stopped at stride 8, so
        // `smem[0]` was the max of only 32 of the 256 lanes, not the tile max.
        // That is numerically (not logically) wrong — the weights below work out
        // to `exp(my_score - new_max)` for ANY `tile_max`, so results stayed
        // correct, but an underestimated shift pushes `exp(my_score - tile_max)`
        // toward the f32 overflow edge for no reason. Run the tree to stride 1.
        if cube_size > 128u32 {
            if tid < 128u32 && smem[(tid + 128u32) as usize] > smem[tid as usize] {
                smem[tid as usize] = smem[(tid + 128u32) as usize];
            }
            sync_cube();
        }
        let mut stride = 64u32;
        while stride > 0u32 {
            if tid < stride && smem[(tid + stride) as usize] > smem[tid as usize] {
                smem[tid as usize] = smem[(tid + stride) as usize];
            }
            sync_cube();
            stride /= 2u32;
        }

        let tile_max = smem[0usize];
        // Issue 715 race (a): every thread reads `smem[0]` here, and Phase 3
        // below has thread 0 overwrite `smem[0]` with its weight. Without a
        // barrier, thread 0 can clobber the slot before a slower lane in another
        // SIMD group has read it.
        sync_cube();

        let new_max = if tile_max > running_max {
            tile_max
        } else {
            running_max
        };

        let exp_prev = (running_max - new_max).exp();
        let exp_tile = (tile_max - new_max).exp();

        running_sum = running_sum * exp_prev;
        running_out = running_out * exp_prev;
        running_max = new_max;

        if valid_pos {
            smem[tid as usize] = exp_tile * (my_score - tile_max).exp();
        } else {
            smem[tid as usize] = f32::new(0.0f32);
        }
        sync_cube();

        let mut tile_sum = f32::new(0.0f32);
        let mut acc = f32::new(0.0f32);
        let mut p = 0u32;
        while p < cube_size {
            let kv_pos = tile_base + p;
            if kv_pos < n_positions {
                let weight = smem[p as usize];
                tile_sum = tile_sum + weight;
                if valid_dim {
                    let v_idx = kv_pos * kv_stride + kv_off + tid;
                    acc += weight * value_cache[v_idx as usize];
                }
            }
            p += 1u32;
        }
        if valid_dim {
            running_out += acc;
        }
        running_sum = running_sum + tile_sum;

        // Issue 715 race (b) — THE nondeterminism bug. The loop above reads
        // every `smem[p]` for this tile; the next iteration's Phase 1 writes
        // `smem[tid] = my_score`. With no barrier on the back-edge, a thread
        // that finishes its read loop early races ahead and clobbers a weight
        // another thread has not consumed yet, so the attention output depends
        // on scheduling.
        //
        // This can only fire when `n_tiles >= 2`, i.e. `n_positions > cube_size`
        // (= head_dim = 256 for Bonsai-27B) — which is exactly the observed
        // signature: 1- and 16-token probes were bit-identical across runs while
        // 306-token probes were not. Same class as the Issue 612 smem corruption;
        // that fix removed the destructive reduction but left the back-edge open.
        sync_cube();

        tile += 1u32;
    }

    // Final normalization + fused output gate (Issue 648 F10).
    // Replaces the separate qwen_output_gate_f32 dispatch.
    if valid_dim {
        let inv_sum = f32::new(1.0f32) / running_sum;
        let raw = running_out * inv_sum;
        // sigmoid(gate[head_off + tid])
        let g = gate[(head_off + tid) as usize];
        let neg_g = f32::new(0.0f32) - g;
        let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_g.exp());
        attn_out[(head_off + tid) as usize] = raw * sig;
    }
}

/// Launch gated flash attention decode for Qwen3.5 (Issue 648 F10).
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionDecodeGatedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionDecodeGatedCubeCL {
    /// Launch gated flash attention decode — fuses the output gate (sigmoid)
    /// into the decode kernel's final write.
    ///
    /// # Safety
    /// - `query_handle`: `n_head * head_dim` f32 elements
    /// - `key_cache_handle`, `value_cache_handle`: `n_positions * n_kv_head * head_dim` f32 each
    /// - `gate_handle`: `n_head * head_dim` f32 elements
    /// - `attn_out_handle`: `n_head * head_dim` f32 elements
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        key_cache_handle: Handle,
        value_cache_handle: Handle,
        gate_handle: Handle,
        attn_out_handle: Handle,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        n_positions: usize,
    ) {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let params: [f32; 5] = [
            head_dim as f32,
            n_head as f32,
            n_kv_head as f32,
            n_positions as f32,
            scale,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let q_len = n_head * head_dim;
        let kv_len = n_positions * n_kv_head * head_dim;

        unsafe {
            qwen_attention_decode_gated_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head as u32, 1, 1),
                CubeDim::new_1d(head_dim as u32),
                BufferArg::from_raw_parts(query_handle, q_len),
                BufferArg::from_raw_parts(key_cache_handle, kv_len),
                BufferArg::from_raw_parts(value_cache_handle, kv_len),
                BufferArg::from_raw_parts(gate_handle, q_len),
                BufferArg::from_raw_parts(attn_out_handle, q_len),
                BufferArg::from_raw_parts(params_handle, 5),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Issue 831 (a): split-K gated flash attention decode (Metal long-context fix)
// ---------------------------------------------------------------------------
//
// The single-workgroup-per-head decode kernel above scans ALL n_positions
// inside one workgroup — grid (n_head, 1, 1) = 24 workgroups on a 40-core
// M3 Max, each walking ceil(n_pos/256) tiles of (256-serial-FMA K-dot +
// 256-serial V-accumulate + ~11 sync_cube barriers). Measured (Bench 831
// follow-up, 2026-09-01): at 2K ctx the attention decode read costs
// 15.8 ms/token = 96% of the entire context penalty — ~17 GB/s effective
// against ~400 GB/s of Metal bandwidth — while both league opponents hold
// decode flat with context (llama.cpp 27.14, Rapid-MLX 27.30 at 2K vs ours
// 20.08). Same structural class as the weaver dot-per-row cliff (Issue 698)
// and the B44 serial-row-dot corpus rule.
//
// Fix = the split-K flash decode (the cudarc/qwen38 twin shipped it as
// "split-KV + splitgqa", riir-ai Issue 742): partition positions across
// n_splits workgroups per head, each computing a PARTIAL online-softmax
// state (m, l, unnormalized out), then a combine kernel merges the partials
// by log-sum-exp and applies the normalization + sigmoid gate.
//
// Numerics: the merge changes the floating-point summation order (tree vs
// chain online updates), so outputs are NOT bit-identical to the single
// kernel — gated by max_abs tolerance at unit level and GREEDY-TOKEN
// identity end-to-end (the Bench 742 loop-stream gate; the dotma lesson:
// a numerics knob that flips near-tie argmax is a wrong answer, not a
// speedup).
//
// Partial layout (head-major, so the combine's per-head sweep reads
// contiguous memory):
//
// ```text
// partials[(head * n_splits + split) * partial_stride + 0] = running_max
// partials[(head * n_splits + split) * partial_stride + 1] = running_sum
// partials[(head * n_splits + split) * partial_stride + 2 + d] = out[d]
// ```
//
// with `partial_stride = 2 + head_dim` and `out[d]` UNNORMALIZED
// (Σ w·v at scale exp(score - m)) and UNGATED — the combine applies both.

/// Compute the split geometry for a decode attention call.
///
/// `max_splits` bounds the persistent partials scratch (host-side); splits
/// beyond it widen `split_len` instead (split_len stays a multiple of the
/// `head_dim`-position tile so every split owns whole tiles).
pub fn split_decode_geometry(n_positions: usize, head_dim: usize, max_splits: usize) -> (usize, usize) {
    let tile = head_dim; // positions-per-tile == head_dim in the decode kernels
    let n_splits = n_positions.div_ceil(tile).min(max_splits.max(1));
    let split_len = n_positions.div_ceil(n_splits).div_ceil(tile) * tile;
    (n_splits, split_len)
}

/// Split-K partial pass of the gated flash attention decode.
///
/// Workgroup `(head_idx, split_idx)` scans ONLY its position slice
/// `[split_idx * split_len, min(split_idx * split_len + split_len, n_positions))`
/// with the same tile loop as `qwen_attention_decode_gated_f32`, then writes
/// the partial online-softmax state `(m, l, out[head_dim])` — no
/// normalization, no gate (the combine owns both).
///
/// K-dot shape (Issue 831 residual, 2026-09-01): the thread-per-position
/// serial dot gained 4 independent accumulator chains — 4 K loads in flight
/// per thread, the one lever that moved the cold kernel at multi-wave
/// lengths (~1.8x at 4096 ctx; flat at 2048 where one wave saturates). An
/// alternative one-plane-per-position shape (lane-split chunks +
/// `plane_sum`) was built and measured at a TIE with this shape at
/// production occupancy — probe_831_attn_pair_time holds the full A/B
/// record; the simpler single-path kernel shipped instead.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_decode_gated_split_f32(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    partials: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as u32;
    let n_head = params[1usize] as u32;
    let n_kv_head = params[2usize] as u32;
    let n_positions = params[3usize] as u32;
    let scale = params[4usize];
    let split_len = params[5usize] as u32;
    let n_splits = params[6usize] as u32;

    let head_idx = CUBE_POS_X;
    let split_idx = CUBE_POS_Y;
    let tid = UNIT_POS;

    if head_idx >= n_head {
        terminate!();
    }

    let split_base = split_idx * split_len;
    let pstride = head_dim + 2u32;
    let pbase = ((head_idx * n_splits + split_idx) * pstride) as usize;

    if split_base >= n_positions {
        // Over-dispatched split (unreachable with host-side geometry — guarded
        // anyway): write an EMPTY partial (m = -inf, l = 0, out = 0) so the
        // combine's log-sum-exp merge contributes exactly nothing and no stale
        // buffer byte can leak in (0 × stale would still poison via NaN).
        if tid == 0u32 {
            partials[pbase] = f32::new(-1e30f32);
            partials[pbase + 1usize] = f32::new(0.0f32);
        }
        if tid < head_dim {
            partials[pbase + 2usize + tid as usize] = f32::new(0.0f32);
        }
        terminate!();
    }

    let head_off = head_idx * head_dim;
    let kv_stride = n_kv_head * head_dim;
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_off = kv_group * head_dim;
    let valid_dim = tid < head_dim;
    let split_end = (split_base + split_len).min(n_positions);

    let mut smem = Shared::<[f32]>::new_slice(256usize);

    let mut running_max = f32::new(-1e30f32);
    let mut running_sum = f32::new(0.0f32);
    let mut running_out = f32::new(0.0f32);

    let n_tiles = split_end.div_ceil(head_dim);
    let first_tile = split_base / head_dim;

    let mut tile = first_tile;
    while tile < n_tiles {
        let tile_base = tile * head_dim;
        let pos = tile_base + tid;
        let valid_pos = pos < split_end;

        // Thread-per-position K-dot, 4 independent accumulator chains — 4 K
        // loads in flight per thread. Measured (probe_831, 2026-09-01): the
        // lane-split (one-plane-per-position) shape TIES this shape cold at
        // production occupancy (~5 workgroups/core caps the machine), and
        // the unroll is the one lever that moved the multi-wave lengths
        // (~1.8x at 4096 ctx); it is flat at 2048 where the single wave
        // already saturates the resident in-flight budget.
        let mut my_score = f32::new(-1e30f32);
        if valid_pos {
            let k_base = pos * kv_stride + kv_off;
            let mut d0 = f32::new(0.0f32);
            let mut d1 = f32::new(0.0f32);
            let mut d2 = f32::new(0.0f32);
            let mut d3 = f32::new(0.0f32);
            let mut d = 0u32;
            while d + 4u32 <= head_dim {
                let k0 = key_cache[(k_base + d) as usize];
                let k1 = key_cache[(k_base + d + 1u32) as usize];
                let k2 = key_cache[(k_base + d + 2u32) as usize];
                let k3 = key_cache[(k_base + d + 3u32) as usize];
                d0 += query[(head_off + d) as usize] * k0;
                d1 += query[(head_off + d + 1u32) as usize] * k1;
                d2 += query[(head_off + d + 2u32) as usize] * k2;
                d3 += query[(head_off + d + 3u32) as usize] * k3;
                d += 4u32;
            }
            while d < head_dim {
                d0 += query[(head_off + d) as usize] * key_cache[(k_base + d) as usize];
                d += 1u32;
            }
            my_score = ((d0 + d1) + (d2 + d3)) * scale;
        }
        smem[tid as usize] = my_score;
        sync_cube();

        // Same max-reduction tree as the single kernel (Issue 715: run to
        // stride 1; race guards (a)/(b) preserved — same tile loop shape).
        if head_dim > 128u32 {
            if tid < 128u32 && smem[(tid + 128u32) as usize] > smem[tid as usize] {
                smem[tid as usize] = smem[(tid + 128u32) as usize];
            }
            sync_cube();
        }
        let mut stride = 64u32;
        while stride > 0u32 {
            if tid < stride && smem[(tid + stride) as usize] > smem[tid as usize] {
                smem[tid as usize] = smem[(tid + stride) as usize];
            }
            sync_cube();
            stride /= 2u32;
        }

        let tile_max = smem[0usize];
        sync_cube();

        let new_max = if tile_max > running_max {
            tile_max
        } else {
            running_max
        };

        let exp_prev = (running_max - new_max).exp();
        let exp_tile = (tile_max - new_max).exp();

        running_sum = running_sum * exp_prev;
        running_out = running_out * exp_prev;
        running_max = new_max;

        // Scores survive in the register copy (`my_score`) — the single
        // kernel's scheme; the weight phase re-uses it directly.
        if valid_pos {
            smem[tid as usize] = exp_tile * (my_score - tile_max).exp();
        } else {
            smem[tid as usize] = f32::new(0.0f32);
        }
        sync_cube();

        // V-accumulate + weight sum, 4 independent chains (same MLP logic
        // as the K-dot unroll). valid_n = positions of this tile below
        // split_end — replaces the per-position guard with a loop bound.
        let valid_n = head_dim.min(split_end.saturating_sub(tile_base));
        let mut ts0 = f32::new(0.0f32);
        let mut ts1 = f32::new(0.0f32);
        let mut ts2 = f32::new(0.0f32);
        let mut ts3 = f32::new(0.0f32);
        let mut acc0 = f32::new(0.0f32);
        let mut acc1 = f32::new(0.0f32);
        let mut acc2 = f32::new(0.0f32);
        let mut acc3 = f32::new(0.0f32);
        let mut p = 0u32;
        let block_end = valid_n & !3u32;
        while p < block_end {
            let w0 = smem[p as usize];
            let w1 = smem[(p + 1u32) as usize];
            let w2 = smem[(p + 2u32) as usize];
            let w3 = smem[(p + 3u32) as usize];
            ts0 += w0;
            ts1 += w1;
            ts2 += w2;
            ts3 += w3;
            if valid_dim {
                let v_row = (tile_base + p) * kv_stride + kv_off + tid;
                acc0 += w0 * value_cache[v_row as usize];
                acc1 += w1 * value_cache[(v_row + kv_stride) as usize];
                acc2 += w2 * value_cache[(v_row + 2u32 * kv_stride) as usize];
                acc3 += w3 * value_cache[(v_row + 3u32 * kv_stride) as usize];
            }
            p += 4u32;
        }
        while p < valid_n {
            let weight = smem[p as usize];
            ts0 += weight;
            if valid_dim {
                let v_idx = (tile_base + p) * kv_stride + kv_off + tid;
                acc0 += weight * value_cache[v_idx as usize];
            }
            p += 1u32;
        }
        if valid_dim {
            running_out += (acc0 + acc1) + (acc2 + acc3);
        }
        running_sum = running_sum + (ts0 + ts1) + (ts2 + ts3);

        // Issue 715 race (b): back-edge barrier before the next tile's
        // Phase-1 smem writes.
        sync_cube();

        tile += 1u32;
    }

    // Write the partial state — head-major layout; thread 0 writes (m, l),
    // every valid thread writes its own out dim.
    if tid == 0u32 {
        partials[pbase] = running_max;
        partials[pbase + 1usize] = running_sum;
    }
    if valid_dim {
        partials[pbase + 2usize + tid as usize] = running_out;
    }
}

/// Combine pass: merge the per-split partial online-softmax states by
/// log-sum-exp, normalize, and apply the fused sigmoid output gate.
///
/// One workgroup per head; each thread owns output dimension `tid`.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_decode_gated_combine_f32(
    partials: &[f32],
    gate: &[f32],
    attn_out: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as u32;
    let n_head = params[1usize] as u32;
    let n_splits = params[2usize] as u32;

    let head_idx = CUBE_POS_X;
    let tid = UNIT_POS;

    if head_idx >= n_head {
        terminate!();
    }

    let head_off = head_idx * head_dim;
    let valid_dim = tid < head_dim;
    let pstride = head_dim + 2u32;
    let hbase = head_idx * n_splits;

    // Pass 1: global max across splits (redundant per thread — 1 broadcast
    // load per split, at most a few dozen iterations).
    let mut global_max = f32::new(-1e30f32);
    let mut s = 0u32;
    while s < n_splits {
        let m = partials[((hbase + s) * pstride) as usize];
        if m > global_max {
            global_max = m;
        }
        s += 1u32;
    }

    // Pass 2: merge — out = Σ w_s·out_s, l = Σ w_s·l_s, w_s = exp(m_s - M).
    let mut sum_l = f32::new(0.0f32);
    let mut acc = f32::new(0.0f32);
    s = 0u32;
    while s < n_splits {
        let pbase = ((hbase + s) * pstride) as usize;
        let m = partials[pbase];
        let l = partials[pbase + 1usize];
        let w = (m - global_max).exp();
        sum_l = sum_l + w * l;
        if valid_dim {
            acc += w * partials[pbase + 2usize + tid as usize];
        }
        s += 1u32;
    }

    if valid_dim {
        let inv_sum = f32::new(1.0f32) / sum_l;
        let raw = acc * inv_sum;
        let g = gate[(head_off + tid) as usize];
        let neg_g = f32::new(0.0f32) - g;
        let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_g.exp());
        attn_out[(head_off + tid) as usize] = raw * sig;
    }
}

/// Launch the split partial pass. `partials_handle` must be at least
/// `n_head * n_splits * (2 + head_dim)` f32.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionDecodeGatedSplitCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionDecodeGatedSplitCubeCL {
    /// # Safety
    /// - `query_handle`: `n_head * head_dim` f32
    /// - `key_cache_handle`/`value_cache_handle`: `n_positions * n_kv_head * head_dim` f32 each
    /// - `partials_handle`: `n_head * n_splits * (2 + head_dim)` f32
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        key_cache_handle: Handle,
        value_cache_handle: Handle,
        partials_handle: Handle,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        n_positions: usize,
        n_splits: usize,
        split_len: usize,
    ) {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let params: [f32; 7] = [
            head_dim as f32,
            n_head as f32,
            n_kv_head as f32,
            n_positions as f32,
            scale,
            split_len as f32,
            n_splits as f32,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let q_len = n_head * head_dim;
        let kv_len = n_positions * n_kv_head * head_dim;
        let partial_len = n_head * n_splits * (head_dim + 2);

        unsafe {
            qwen_attention_decode_gated_split_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head as u32, n_splits as u32, 1),
                CubeDim::new_1d(head_dim as u32),
                BufferArg::from_raw_parts(query_handle, q_len),
                BufferArg::from_raw_parts(key_cache_handle, kv_len),
                BufferArg::from_raw_parts(value_cache_handle, kv_len),
                BufferArg::from_raw_parts(partials_handle, partial_len),
                BufferArg::from_raw_parts(params_handle, 7),
            );
        }
    }
}

/// Launch the combine pass.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionDecodeGatedCombineCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionDecodeGatedCombineCubeCL {
    /// # Safety
    /// - `partials_handle`: `n_head * n_splits * (2 + head_dim)` f32
    /// - `gate_handle`, `attn_out_handle`: `n_head * head_dim` f32 each
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        partials_handle: Handle,
        gate_handle: Handle,
        attn_out_handle: Handle,
        head_dim: usize,
        n_head: usize,
        n_splits: usize,
    ) {
        let params: [f32; 3] = [head_dim as f32, n_head as f32, n_splits as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let q_len = n_head * head_dim;
        let partial_len = n_head * n_splits * (head_dim + 2);

        unsafe {
            qwen_attention_decode_gated_combine_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head as u32, 1, 1),
                CubeDim::new_1d(head_dim as u32),
                BufferArg::from_raw_parts(partials_handle, partial_len),
                BufferArg::from_raw_parts(gate_handle, q_len),
                BufferArg::from_raw_parts(attn_out_handle, q_len),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Issue 771 residual rung 1: f16-KV packed decode (2026-09-01)
// ---------------------------------------------------------------------------
//
// The 831 residual measured the split decode pair at ~80-90 GB/s cold @2048
// under the ~5-workgroups/core occupancy cap — per-thread outstanding loads
// bound the plateau, so halving BOTH the KV bytes AND the load count is the
// named rung (the Bench-756 4090 f16-KV precedent; there it cost decode −3.4%
// on cudarc where CVT competed, but this path is occupancy-capped on Metal).
//
// Design: the KV cache rides as **packed f16 pairs** — one u32 word covers two
// adjacent dims (lo = even dim, hi = odd dim), so every K-dot iteration loads
// 8 dims in 4 words and every V-accumulate thread owns 2 dims per word load.
// Decode to f32 in-kernel via exact bit-math (`dec_f16` — ±0, subnormal,
// normal, inf, NaN payload-widening all exact; pinned exhaustively by the G0
// gate `tests/kv_f16_gates.rs::kv_f16_conversion_is_exhaustively_exact`),
// using `f32::reinterpret` (cubecl `Operator::Reinterpret` → WGSL Bitcast,
// implemented by the vendored cubecl-wgpu compiler).
//
// The partials layout, barrier structure (Issue 715 guards verbatim), and the
// combine kernel are UNCHANGED — the combine still consumes f32 partials. The
// per-dim V summation order matches the f32 kernel's 4-chain interleaving
// exactly; the only numerics delta vs the f32 kernel is f16 storage rounding
// of K/V (band-gated, G1).
//
// Toggle: DEFAULT OFF (probe/gate harnesses drive it via the setter). This is
// the kernel-level rung instrument — production wiring (append + kvfill +
// prefill attention + layer-state allocation, the "trio") lands only if the
// A/B clears the promote-if gate.

static KV_F16_DECODE_UNSET: usize = usize::MAX;
static KV_F16_DECODE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(KV_F16_DECODE_UNSET);
static KV_F16_DECODE_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Policy toggle for the f16-KV packed decode path (env
/// `RIIR_KV_F16_DECODE`, default OFF; store-on-first-call — the Bench 805
/// env lesson — and the setter is authoritative afterwards).
///
/// VERDICT CONTEXT (Bench 831 follow-up, 2026-09-01): the rung this was
/// built for is **REFUTED** — the packed-f16 split pair measured 1.44-1.60×
/// SLOWER than f32 at every length (interleaved median ratios 0.62-0.72);
/// the 2K plateau is occupancy/issue-bound (114 GB/s ≪ ~400 peak), so
/// halving KV bytes buys nothing and the in-kernel decode adds ALU issue
/// pressure the capped occupancy cannot hide. The kernels stay as the
/// reproducible negative-result artifact (the Bench-769 precedent); no
/// production wiring exists or is planned. Re-arm only on a mechanism
/// change (e.g. a bandwidth-bound KV path or hardware-half loads).
pub fn kv_f16_decode_enabled() -> bool {
    let v = KV_F16_DECODE.load(std::sync::atomic::Ordering::Relaxed);
    if v == KV_F16_DECODE_UNSET {
        let env_on = std::env::var("RIIR_KV_F16_DECODE").is_ok_and(|v| {
                !matches!(v.to_lowercase().as_str(), "0" | "off" | "false")
            });
        KV_F16_DECODE
            .compare_exchange(
                KV_F16_DECODE_UNSET,
                usize::from(env_on),
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            )
            .ok();
        return env_on;
    }
    v != 0
}

/// Force the f16-KV packed decode path on/off (overrides the env var; the
/// probe/gate arm toggle).
pub fn set_kv_f16_decode(on: bool) {
    KV_F16_DECODE.store(usize::from(on), std::sync::atomic::Ordering::Relaxed);
}

/// Total f16-split launches dispatched so far (the vacuous guard — a
/// production wiring that never dispatches must fail loudly).
pub fn kv_f16_decode_launch_count() -> usize {
    KV_F16_DECODE_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Pack f32 values into u32 f16 pairs: word `w` = bits(elem 2w) | bits(elem
/// 2w+1) << 16 (lo = even dim). Panics on an odd-length slice — the packed
/// layout requires even alignment (production kvd/head_dim always are).
pub fn pack_kv_f16_pairs(src: &[f32]) -> Vec<u32> {
    assert!(
        src.len().is_multiple_of(2),
        "pack_kv_f16_pairs requires even len"
    );
    src.as_chunks::<2>()
        .0
        .iter()
        .map(|c| {
            let lo = half::f16::from_f32(c[0]).to_bits() as u32;
            let hi = half::f16::from_f32(c[1]).to_bits() as u32;
            lo | (hi << 16)
        })
        .collect()
}

/// Decode one f16 value (16-bit pattern in a u32) to f32 — EXACT for every
/// input (±0, subnormal, normal, inf, NaN with payload widened in place).
/// Pinned exhaustively (all 65536 patterns) by the G0 gate.
#[cfg(feature = "cubecl_runtime")]
#[cube]
fn dec_f16(h: u32) -> f32 {
    let sign_bit = (h & 0x8000u32) << 16u32;
    let exp16 = (h >> 10u32) & 0x1Fu32;
    let man16 = h & 0x3FFu32;
    // Zero / negative-zero: `reinterpret(sign_bit)` is exactly ±0.0.
    let mut val = f32::reinterpret(sign_bit);
    if (h & 0x7FFFu32) != 0u32 {
        if exp16 == 0u32 {
            // Subnormal: man16 * 2^-24, exact via the 2^-14 anchor — both
            // operands carry the sign so the difference is exact.
            let anchor = 113u32 << 23u32; // 2^-14
            val = f32::reinterpret(sign_bit | (anchor | (man16 << 13u32)))
                - f32::reinterpret(sign_bit | anchor);
        } else if exp16 == 31u32 {
                // inf / NaN — payload widened into the f32 mantissa MSBs.
                val = f32::reinterpret(sign_bit | 0x7F800000u32 | (man16 << 13u32));
            } else {
                // Normal: exponent bias 15 → 127 (+112), mantissa << 13.
                val = f32::reinterpret(
                    sign_bit | ((exp16 + 112u32) << 23u32) | (man16 << 13u32),
                );
            }
    }
    val
}

/// Split-K partial pass over **packed f16 KV** — the f16 twin of
/// `qwen_attention_decode_gated_split_f32`. Same geometry, same barrier
/// structure (Issue 715 guards verbatim), same f32 partials layout; the
/// combine kernel is shared unchanged.
///
/// K-dot: thread-per-position over u32 words — 4 word loads (8 dims) per
/// unrolled iteration in 4 independent chains, HALF the load count and HALF
/// the bytes of the f32 kernel at the same 4-chain ILP.
/// V-accumulate: each of the first `head_dim/2` threads owns the dim PAIR
/// (2t, 2t+1) — one word load covers both dims per position (8 accumulators
/// over the 4-chain unroll); threads `>= head_dim/2` skip V work but still
/// join every barrier. Per-dim summation order matches the f32 kernel.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_decode_gated_split_f16(
    query: &[f32],
    key_cache: &[u32],
    value_cache: &[u32],
    partials: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as u32;
    let n_head = params[1usize] as u32;
    let n_kv_head = params[2usize] as u32;
    let n_positions = params[3usize] as u32;
    let scale = params[4usize];
    let split_len = params[5usize] as u32;
    let n_splits = params[6usize] as u32;

    let head_idx = CUBE_POS_X;
    let split_idx = CUBE_POS_Y;
    let tid = UNIT_POS;

    if head_idx >= n_head {
        terminate!();
    }

    let split_base = split_idx * split_len;
    let pstride = head_dim + 2u32;
    let pbase = ((head_idx * n_splits + split_idx) * pstride) as usize;

    if split_base >= n_positions {
        // Over-dispatched split: write an EMPTY partial (same contract as the
        // f32 kernel — the combine's log-sum-exp merge contributes nothing).
        if tid == 0u32 {
            partials[pbase] = f32::new(-1e30f32);
            partials[pbase + 1usize] = f32::new(0.0f32);
        }
        if tid < head_dim {
            partials[pbase + 2usize + tid as usize] = f32::new(0.0f32);
        }
        terminate!();
    }

    let head_off = head_idx * head_dim;
    let kv_stride = n_kv_head * head_dim;
    let kv_stride_words = kv_stride >> 1u32;
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_off_words = (kv_group * head_dim) >> 1u32;
    let hd_half = head_dim >> 1u32;
    let v_active = tid < hd_half;
    let split_end = (split_base + split_len).min(n_positions);

    let mut smem = Shared::<[f32]>::new_slice(256usize);

    let mut running_max = f32::new(-1e30f32);
    let mut running_sum = f32::new(0.0f32);
    let mut running_out_lo = f32::new(0.0f32);
    let mut running_out_hi = f32::new(0.0f32);

    let n_tiles = split_end.div_ceil(head_dim);
    let first_tile = split_base / head_dim;

    let mut tile = first_tile;
    while tile < n_tiles {
        let tile_base = tile * head_dim;
        let pos = tile_base + tid;
        let valid_pos = pos < split_end;

        // Thread-per-position K-dot over packed f16 words — 4 word loads in
        // flight (8 dims/iteration), 4 independent accumulator chains.
        let mut my_score = f32::new(-1e30f32);
        if valid_pos {
            let kw = pos * kv_stride_words + kv_off_words;
            let mut d0 = f32::new(0.0f32);
            let mut d1 = f32::new(0.0f32);
            let mut d2 = f32::new(0.0f32);
            let mut d3 = f32::new(0.0f32);
            let mut wd = 0u32;
            let mut dim = 0u32;
            while wd + 4u32 <= hd_half {
                let p0 = key_cache[(kw + wd) as usize];
                let p1 = key_cache[(kw + wd + 1u32) as usize];
                let p2 = key_cache[(kw + wd + 2u32) as usize];
                let p3 = key_cache[(kw + wd + 3u32) as usize];
                let q0 = query[(head_off + dim) as usize];
                let q1 = query[(head_off + dim + 1u32) as usize];
                let q2 = query[(head_off + dim + 2u32) as usize];
                let q3 = query[(head_off + dim + 3u32) as usize];
                let q4 = query[(head_off + dim + 4u32) as usize];
                let q5 = query[(head_off + dim + 5u32) as usize];
                let q6 = query[(head_off + dim + 6u32) as usize];
                let q7 = query[(head_off + dim + 7u32) as usize];
                d0 += q0 * dec_f16(p0 & 0xFFFFu32) + q1 * dec_f16(p0 >> 16u32);
                d1 += q2 * dec_f16(p1 & 0xFFFFu32) + q3 * dec_f16(p1 >> 16u32);
                d2 += q4 * dec_f16(p2 & 0xFFFFu32) + q5 * dec_f16(p2 >> 16u32);
                d3 += q6 * dec_f16(p3 & 0xFFFFu32) + q7 * dec_f16(p3 >> 16u32);
                wd += 4u32;
                dim += 8u32;
            }
            while wd < hd_half {
                let pw = key_cache[(kw + wd) as usize];
                let q0 = query[(head_off + dim) as usize];
                let q1 = query[(head_off + dim + 1u32) as usize];
                d0 += q0 * dec_f16(pw & 0xFFFFu32) + q1 * dec_f16(pw >> 16u32);
                wd += 1u32;
                dim += 2u32;
            }
            my_score = ((d0 + d1) + (d2 + d3)) * scale;
        }
        smem[tid as usize] = my_score;
        sync_cube();

        // Same max-reduction tree as the f32 kernel (Issue 715: run to stride
        // 1; race guards (a)/(b) preserved).
        if head_dim > 128u32 {
            if tid < 128u32 && smem[(tid + 128u32) as usize] > smem[tid as usize] {
                smem[tid as usize] = smem[(tid + 128u32) as usize];
            }
            sync_cube();
        }
        let mut stride = 64u32;
        while stride > 0u32 {
            if tid < stride && smem[(tid + stride) as usize] > smem[tid as usize] {
                smem[tid as usize] = smem[(tid + stride) as usize];
            }
            sync_cube();
            stride /= 2u32;
        }

        let tile_max = smem[0usize];
        sync_cube();

        let new_max = if tile_max > running_max {
            tile_max
        } else {
            running_max
        };

        let exp_prev = (running_max - new_max).exp();
        let exp_tile = (tile_max - new_max).exp();

        running_sum = running_sum * exp_prev;
        running_out_lo = running_out_lo * exp_prev;
        running_out_hi = running_out_hi * exp_prev;
        running_max = new_max;

        if valid_pos {
            smem[tid as usize] = exp_tile * (my_score - tile_max).exp();
        } else {
            smem[tid as usize] = f32::new(0.0f32);
        }
        sync_cube();

        // V-accumulate over dim PAIRS (thread t owns dims 2t, 2t+1 — one word
        // load per position). ts accumulation stays ALL-threads (running_sum
        // is computed redundantly per-thread, exactly like the f32 kernel);
        // only the V loads sit behind `v_active`.
        let valid_n = head_dim.min(split_end.saturating_sub(tile_base));
        let mut ts0 = f32::new(0.0f32);
        let mut ts1 = f32::new(0.0f32);
        let mut ts2 = f32::new(0.0f32);
        let mut ts3 = f32::new(0.0f32);
        let mut a0a = f32::new(0.0f32);
        let mut a0b = f32::new(0.0f32);
        let mut a1a = f32::new(0.0f32);
        let mut a1b = f32::new(0.0f32);
        let mut a2a = f32::new(0.0f32);
        let mut a2b = f32::new(0.0f32);
        let mut a3a = f32::new(0.0f32);
        let mut a3b = f32::new(0.0f32);
        let mut p = 0u32;
        let block_end = valid_n & !3u32;
        while p < block_end {
            let w0 = smem[p as usize];
            let w1 = smem[(p + 1u32) as usize];
            let w2 = smem[(p + 2u32) as usize];
            let w3 = smem[(p + 3u32) as usize];
            ts0 += w0;
            ts1 += w1;
            ts2 += w2;
            ts3 += w3;
            if v_active {
                let vrow = (tile_base + p) * kv_stride_words + kv_off_words + tid;
                let v0 = value_cache[vrow as usize];
                let v1 = value_cache[(vrow + kv_stride_words) as usize];
                let v2 = value_cache[(vrow + 2u32 * kv_stride_words) as usize];
                let v3 = value_cache[(vrow + 3u32 * kv_stride_words) as usize];
                a0a += w0 * dec_f16(v0 & 0xFFFFu32);
                a0b += w0 * dec_f16(v0 >> 16u32);
                a1a += w1 * dec_f16(v1 & 0xFFFFu32);
                a1b += w1 * dec_f16(v1 >> 16u32);
                a2a += w2 * dec_f16(v2 & 0xFFFFu32);
                a2b += w2 * dec_f16(v2 >> 16u32);
                a3a += w3 * dec_f16(v3 & 0xFFFFu32);
                a3b += w3 * dec_f16(v3 >> 16u32);
            }
            p += 4u32;
        }
        while p < valid_n {
            let weight = smem[p as usize];
            ts0 += weight;
            if v_active {
                let vidx = (tile_base + p) * kv_stride_words + kv_off_words + tid;
                let vv = value_cache[vidx as usize];
                a0a += weight * dec_f16(vv & 0xFFFFu32);
                a0b += weight * dec_f16(vv >> 16u32);
            }
            p += 1u32;
        }
        if v_active {
            running_out_lo += (a0a + a1a) + (a2a + a3a);
            running_out_hi += (a0b + a1b) + (a2b + a3b);
        }
        running_sum = running_sum + (ts0 + ts1) + (ts2 + ts3);

        // Issue 715 race (b): back-edge barrier before the next tile's
        // Phase-1 smem writes.
        sync_cube();

        tile += 1u32;
    }

    // Write the partial state — same f32 layout; thread 0 writes (m, l), each
    // V-active thread writes its dim PAIR.
    if tid == 0u32 {
        partials[pbase] = running_max;
        partials[pbase + 1usize] = running_sum;
    }
    if v_active {
        partials[pbase + 2usize + (2u32 * tid) as usize] = running_out_lo;
        partials[pbase + 2usize + (2u32 * tid + 1u32) as usize] = running_out_hi;
    }
}

/// Launch the f16 packed split partial pass. `key_cache_handle` /
/// `value_cache_handle` carry **u32 f16 pairs** — byte size
/// `n_positions * n_kv_head * head_dim * 2` (word count `... / 4`).
/// `head_dim` must be even. `partials_handle` is the shared f32 layout.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionDecodeGatedSplitF16CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionDecodeGatedSplitF16CubeCL {
    /// # Safety
    /// - `query_handle`: `n_head * head_dim` f32
    /// - `key_cache_handle`/`value_cache_handle`: `n_positions * kvd / 2` u32
    ///   words each (`kvd = n_kv_head * head_dim` elements packed 2-per-word)
    /// - `partials_handle`: `n_head * n_splits * (2 + head_dim)` f32
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        key_cache_handle: Handle,
        value_cache_handle: Handle,
        partials_handle: Handle,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        n_positions: usize,
        n_splits: usize,
        split_len: usize,
    ) {
        debug_assert!(
            head_dim.is_multiple_of(2) && (n_kv_head * head_dim).is_multiple_of(2),
            "f16 packed KV decode requires even head_dim and kv_stride"
        );
        KV_F16_DECODE_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let params: [f32; 7] = [
            head_dim as f32,
            n_head as f32,
            n_kv_head as f32,
            n_positions as f32,
            scale,
            split_len as f32,
            n_splits as f32,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let q_len = n_head * head_dim;
        let kv_words = n_positions * (n_kv_head * head_dim) / 2;
        let partial_len = n_head * n_splits * (head_dim + 2);

        unsafe {
            qwen_attention_decode_gated_split_f16::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head as u32, n_splits as u32, 1),
                CubeDim::new_1d(head_dim as u32),
                BufferArg::from_raw_parts(query_handle, q_len),
                BufferArg::from_raw_parts(key_cache_handle, kv_words),
                BufferArg::from_raw_parts(value_cache_handle, kv_words),
                BufferArg::from_raw_parts(partials_handle, partial_len),
                BufferArg::from_raw_parts(params_handle, 7),
            );
        }
    }
}

/// G0 probe kernel: decode `packed` u32 f16 pairs into f32 (one word per
/// thread, lo → out[2i], hi → out[2i+1]). Dispatch exactly `packed.len()`
/// threads — no bounds check (launch_unchecked caller-owned size contract).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn kv_f16_decode_probe(packed: &[u32], out: &mut [f32]) {
    let i = ABSOLUTE_POS;
    let w = packed[i];
    out[2 * i] = dec_f16(w & 0xFFFFu32);
    out[2 * i + 1] = dec_f16(w >> 16u32);
}

/// Launch the G0 decode probe. `out` must hold `2 * words` f32.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenKvF16DecodeProbeCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenKvF16DecodeProbeCubeCL {
    /// # Safety
    /// - `packed_handle`: `words` u32
    /// - `out_handle`: `2 * words` f32
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        packed_handle: Handle,
        out_handle: Handle,
        words: usize,
    ) {
        KV_F16_DECODE_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let wg_size = 128u32;
        let num_wg = (words as u32).div_ceil(wg_size).max(1);
        unsafe {
            kv_f16_decode_probe::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(packed_handle, words),
                BufferArg::from_raw_parts(out_handle, 2 * words),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Output gating kernel (sigmoid gate)
// ---------------------------------------------------------------------------

/// Apply output gating: `attn_out[i] *= sigmoid(gate[i])`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn qwen_output_gate_f32(
    attn_out: &mut [f32],
    gate: &[f32],
    params: &[f32],
) {
    let n = params[0usize] as usize;
    let idx = ABSOLUTE_POS;

    if idx >= n {
        terminate!();
    }

    let g = gate[idx];
    let neg_g = f32::new(0.0f32) - g;
    let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_g.exp());
    attn_out[idx] = attn_out[idx] * sig;
}

/// Launch output gating kernel.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenOutputGateCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenOutputGateCubeCL {
    /// Apply sigmoid output gating in-place.
    ///
    /// # Safety
    /// - `attn_out_handle`, `gate_handle`: `n` f32 elements each
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        attn_out_handle: Handle,
        gate_handle: Handle,
        n: usize,
    ) {
        let params: [f32; 1] = [n as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let wg_size = 128u32;
        let num_wg = (n as u32).div_ceil(wg_size).max(1);

        unsafe {
            qwen_output_gate_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(attn_out_handle, n),
                BufferArg::from_raw_parts(gate_handle, n),
                BufferArg::from_raw_parts(params_handle, 1),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Issue 653: Batched attention prefill kernels (causal-masked flash attention)
// ---------------------------------------------------------------------------
//
// Replaces the P sequential decode dispatches per attention layer with a
// single batched causal-masked flash attention. Each cube handles one
// (head, query_pos) pair, using online softmax over key positions 0..=query_pos
// (causal mask). The pattern extends `qwen_attention_decode_gated_f32` to P
// queries — same running max/sum/output accumulation, just with the query
// position derived from CUBE_POS_X instead of a fixed single-token query.

/// Batched QG split for P tokens (Issue 653).
///
/// Same semantics as `qwen_split_qg_f32` but operates on all P tokens at once.
/// Each thread handles one element across the full P × n_head × head_dim output.
///
/// Input layout:  `qg[P, n_head, 2*head_dim]` — per-token interleaved [q, gate]
/// Output layout: `q[P, n_head, head_dim]`, `gate[P, n_head, head_dim]`
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn qwen_split_qg_batched_f32(
    qg: &[f32],
    q: &mut [f32],
    gate: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as usize;
    let n_head = params[1usize] as usize;
    let p = params[2usize] as usize;
    let idx = ABSOLUTE_POS;

    let total_per_token = n_head * head_dim;
    let total = p * total_per_token;
    if idx >= total {
        terminate!();
    }

    let token = idx / total_per_token;
    let within = idx % total_per_token;
    let head = within / head_dim;
    let dim = within % head_dim;

    let qg_stride = n_head * 2usize * head_dim;
    let src = token * qg_stride + head * 2usize * head_dim + dim;
    let gate_src = src + head_dim;

    q[idx] = qg[src];
    gate[idx] = qg[gate_src];
}

/// Launch batched QG split.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenSplitQgBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenSplitQgBatchedCubeCL {
    /// Split interleaved QG buffer into Q and gate for all P tokens.
    ///
    /// # Safety
    /// - `qg_handle`: `P * 2 * n_head * head_dim` f32 elements
    /// - `q_handle`, `gate_handle`: `P * n_head * head_dim` f32 each
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        qg_handle: Handle,
        q_handle: Handle,
        gate_handle: Handle,
        head_dim: usize,
        n_head: usize,
        p: usize,
    ) {
        // Metal grid guard (Issue 726, 2026-08-19): the x-dimension is capped
        // at 65535 workgroups on Metal (wgpu validation panics above it; CUDA
        // allows 2^31-1 which is why this never failed on the 4090). At
        // Bonsai-27B dims (24 heads x 256) the flat grid is 48*p workgroups —
        // illegal from p >= 1366. Chunk on TOKEN boundaries: every element's
        // arithmetic is independent, the kernel derives token/head/dim from
        // its LOCAL flat index against per-chunk params, and the sliced
        // handles make those local indexes land at the chunk's absolute
        // offset. Bit-identical to the single launch.
        const MAX_WG_X: u32 = 65535;
        let wg_size = 128u32;
        let per_token = n_head * head_dim;
        let wg_per_token = (per_token as u32).div_ceil(wg_size).max(1);
        let tokens_per_chunk = (MAX_WG_X / wg_per_token).max(1) as usize;
        let mut t0 = 0usize;
        while t0 < p {
            let tc = tokens_per_chunk.min(p - t0);
            let elems = tc * per_token;
            let params: [f32; 3] = [head_dim as f32, n_head as f32, tc as f32];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
            let num_wg = (elems as u32).div_ceil(wg_size).max(1);
            unsafe {
                qwen_split_qg_batched_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(num_wg, 1, 1),
                    CubeDim::new_1d(wg_size),
                    BufferArg::from_raw_parts(
                        qg_handle.clone().offset_start(((t0 * 2 * per_token) * 4) as u64),
                        tc * 2 * per_token,
                    ),
                    BufferArg::from_raw_parts(
                        q_handle.clone().offset_start((t0 * per_token * 4) as u64),
                        elems,
                    ),
                    BufferArg::from_raw_parts(
                        gate_handle.clone().offset_start((t0 * per_token * 4) as u64),
                        elems,
                    ),
                    BufferArg::from_raw_parts(params_handle, 3),
                );
            }
            t0 += tc;
        }
    }
}

/// Batched partial RoPE for P tokens (Issue 653).
///
/// Applies partial RoPE to Q and K for all P tokens, where each token's
/// rotation uses its own position (= `base_pos + token index` — Issue 734
/// chunked prefill: chunk-local rows carry ABSOLUTE positions). Uses the same
/// rotate-half (GPT-NeoX) convention as `qwen_rope_partial_f32`.
///
/// Each thread handles one (token, head, pair) triple.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_rope_partial_batched_f32(
    q: &mut [f32],
    k: &mut [f32],
    params: &[f32],
) {
    // params: [rotary_dim_pairs, theta_base, head_dim, n_head, n_kv_head, p, base_pos]
    let rotary_pairs = params[0usize] as usize;
    let theta_base = params[1usize];
    let head_dim = params[2usize] as usize;
    let n_head = params[3usize] as usize;
    let n_kv_head = params[4usize] as usize;
    let p = params[5usize] as usize;
    let base_pos = params[6usize] as usize;

    let idx = ABSOLUTE_POS;
    let total_q_pairs = p * n_head * rotary_pairs;

    if idx >= total_q_pairs {
        terminate!();
    }

    // Decode (token, head, pair) from idx
    let q_stride = n_head * rotary_pairs;
    let token = idx / q_stride;
    let within = idx % q_stride;
    let head = within / rotary_pairs;
    let pair = within % rotary_pairs;

    let pos = (base_pos + token) as f32;

    // Compute frequency: inv_freq = 1 / theta_base^(2*pair / rotary_dim)
    let rotary_dim_f = (2usize * rotary_pairs) as f32;
    let exponent = (2usize * pair) as f32 / rotary_dim_f;
    let log_base = theta_base.ln();
    let inv_freq = (log_base * (f32::new(0.0f32) - exponent)).exp();

    let theta = pos * inv_freq;
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    // Q rotation (rotate-half / GPT-NeoX convention)
    let q_row_stride = n_head * head_dim;
    let q_head_off = token * q_row_stride + head * head_dim;
    let q_i0 = q_head_off + pair;
    let q_i1 = q_i0 + rotary_pairs;
    let q0 = q[q_i0];
    let q1 = q[q_i1];
    q[q_i0] = q0 * cos_t - q1 * sin_t;
    q[q_i1] = q0 * sin_t + q1 * cos_t;

    // K rotation — only if this thread is within n_kv_head range
    let total_k_pairs = p * n_kv_head * rotary_pairs;
    if idx < total_k_pairs {
        let k_stride = n_kv_head * rotary_pairs;
        let k_token = idx / k_stride;
        let k_within = idx % k_stride;
        let k_head = k_within / rotary_pairs;
        let k_pair = k_within % rotary_pairs;

        let k_pos = (base_pos + k_token) as f32;
        let k_exponent = (2usize * k_pair) as f32 / rotary_dim_f;
        let k_inv_freq = (log_base * (f32::new(0.0f32) - k_exponent)).exp();
        let k_theta = k_pos * k_inv_freq;
        let k_cos = k_theta.cos();
        let k_sin = k_theta.sin();

        let k_row_stride = n_kv_head * head_dim;
        let k_head_off = k_token * k_row_stride + k_head * head_dim;
        let k_i0 = k_head_off + k_pair;
        let k_i1 = k_i0 + rotary_pairs;
        let k0 = k[k_i0];
        let k1 = k[k_i1];
        k[k_i0] = k0 * k_cos - k1 * k_sin;
        k[k_i1] = k0 * k_sin + k1 * k_cos;
    }
}

/// Launch batched partial RoPE.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenRopePartialBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenRopePartialBatchedCubeCL {
    /// Apply partial RoPE to Q and K for all P tokens in-place.
    ///
    /// Each token `t` uses position `base_pos + t` for its rotation
    /// (`base_pos = 0` recovers the original single-launch semantics).
    ///
    /// # Safety
    /// - `q_handle`: `P * n_head * head_dim` f32 elements (modified in-place)
    /// - `k_handle`: `P * n_kv_head * head_dim` f32 elements (modified in-place)
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q_handle: Handle,
        k_handle: Handle,
        rotary_dim: usize,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        theta_base: f32,
        p: usize,
        base_pos: usize,
    ) {
        // Metal grid guard (Issue 730): (p*n_head*rotary_pairs)/128 workgroups
        // exceeds the 65535 x-cap at long prompts (196608 wg at p=32768 with
        // Bonsai's 24 heads x 32 pairs). Both the Q and K rotations derive
        // token/head/pair from the LOCAL flat index against per-chunk params,
        // so token-boundary chunking with sliced handles + p=chunk is
        // bit-identical.
        const MAX_WG_X: u32 = 65535;
        let rotary_pairs = rotary_dim / 2;
        let wg_size = 128u32;
        let per_token = n_head * rotary_pairs;
        let wg_per_token = (per_token as u32).div_ceil(wg_size).max(1);
        let tokens_per_chunk = (MAX_WG_X / wg_per_token).max(1) as usize;
        let mut t0 = 0usize;
        while t0 < p {
            let tc = tokens_per_chunk.min(p - t0);
            let total_q_pairs = tc * per_token;
            let params: [f32; 7] = [
                rotary_pairs as f32,
                theta_base,
                head_dim as f32,
                n_head as f32,
                n_kv_head as f32,
                tc as f32,
                (base_pos + t0) as f32,
            ];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
            let num_wg = (total_q_pairs as u32).div_ceil(wg_size).max(1);
            unsafe {
                qwen_rope_partial_batched_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(num_wg, 1, 1),
                    CubeDim::new_1d(wg_size),
                    BufferArg::from_raw_parts(
                        q_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64),
                        tc * n_head * head_dim,
                    ),
                    BufferArg::from_raw_parts(
                        k_handle.clone().offset_start((t0 * n_kv_head * head_dim * 4) as u64),
                        tc * n_kv_head * head_dim,
                    ),
                    BufferArg::from_raw_parts(params_handle, 7),
                );
            }
            t0 += tc;
        }
    }
}

/// Batched KV cache fill for P tokens (Issue 653).
///
/// Copies all P tokens' K and V from a combined projection buffer to the KV
/// cache at positions 0..P. Replaces P sequential `QwenKvCacheAppendCombined`
/// dispatches with a single batched fill.
///
/// Input layout: `kv[P, 2*kvd]` — per-token combined [K(t), V(t)]
/// Output: `key_cache[P * kvd]`, `value_cache[P * kvd]`
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn qwen_kv_cache_fill_batched_f32(
    kv: &[f32],
    key_cache: &mut [f32],
    value_cache: &mut [f32],
    params: &[f32],
) {
    let kvd = params[0usize] as usize;
    let p = params[1usize] as usize;
    let idx = ABSOLUTE_POS;

    let total = p * kvd;
    if idx >= total {
        terminate!();
    }

    let token = idx / kvd;
    let dim = idx % kvd;

    // Combined KV: token row is [token * 2*kvd + dim] for K, [+kvd] for V
    let kv_off = token * 2usize * kvd + dim;
    key_cache[idx] = kv[kv_off];
    value_cache[idx] = kv[kv_off + kvd];
}

/// Launch batched KV cache fill.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenKvCacheFillBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenKvCacheFillBatchedCubeCL {
    /// Fill KV cache positions 0..P from a combined KV projection buffer.
    ///
    /// # Safety
    /// - `kv_handle`: `P * 2 * kvd` f32 elements
    /// - `key_cache_handle`, `value_cache_handle`: `P * kvd` f32 each
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        kv_handle: Handle,
        key_cache_handle: Handle,
        value_cache_handle: Handle,
        kvd: usize,
        p: usize,
    ) {
        // Metal grid guard (Issue 730): (p*kvd)/128 exceeds the 65535 x-cap at
        // long prompts. Token-boundary chunking; cache writes are
        // token-local (the sliced cache handles land at the chunk's
        // positions). Bit-identical.
        const MAX_WG_X: u32 = 65535;
        let wg_size = 128u32;
        let wg_per_token = ((kvd as u32).div_ceil(wg_size)).max(1);
        let tokens_per_chunk = (MAX_WG_X / wg_per_token).max(1) as usize;
        let mut t0 = 0usize;
        while t0 < p {
            let tc = tokens_per_chunk.min(p - t0);
            let total = tc * kvd;
            let params: [f32; 2] = [kvd as f32, tc as f32];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
            let num_wg = (total as u32).div_ceil(wg_size).max(1);
            unsafe {
                qwen_kv_cache_fill_batched_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(num_wg, 1, 1),
                    CubeDim::new_1d(wg_size),
                    BufferArg::from_raw_parts(
                        kv_handle.clone().offset_start((t0 * 2 * kvd * 4) as u64),
                        tc * 2 * kvd,
                    ),
                    BufferArg::from_raw_parts(
                        key_cache_handle.clone().offset_start((t0 * kvd * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(
                        value_cache_handle.clone().offset_start((t0 * kvd * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(params_handle, 2),
                );
            }
            t0 += tc;
        }
    }
}

/// Batched KV cache fill from separate K and V buffers (Issue 653).
///
/// Copies post-RoPE K and raw V from separate contiguous `[P, kvd]` buffers
/// to the KV cache at positions 0..P. This is the correct fill path for the
/// batched attention prefill — the K has been RMSNormed + RoPE'd, matching
/// what the sequential decode path writes to the cache.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn qwen_kv_cache_fill_split_batched_f32(
    k: &[f32],
    v: &[f32],
    key_cache: &mut [f32],
    value_cache: &mut [f32],
    params: &[f32],
) {
    let kvd = params[0usize] as usize;
    let p = params[1usize] as usize;
    let idx = ABSOLUTE_POS;
    let total = p * kvd;
    if idx >= total {
        terminate!();
    }
    // K and V are already contiguous [P, kvd], same layout as the cache.
    key_cache[idx] = k[idx];
    value_cache[idx] = v[idx];
}

/// Launch batched KV cache fill from separate K/V buffers.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenKvCacheFillSplitBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenKvCacheFillSplitBatchedCubeCL {
    /// Fill KV cache positions `base_pos..base_pos+P` from separate K and V
    /// buffers (`base_pos = 0` recovers the original fill-at-0 semantics —
    /// Issue 734 chunked prefill).
    ///
    /// # Safety
    /// - `k_handle`, `v_handle`: `P * kvd` f32 each (chunk-local rows)
    /// - `key_cache_handle`, `value_cache_handle`: `(base_pos + P) * kvd`
    ///   f32 each — writes land at cache rows `[base_pos, base_pos+P)`
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        k_handle: Handle,
        v_handle: Handle,
        key_cache_handle: Handle,
        value_cache_handle: Handle,
        kvd: usize,
        p: usize,
        base_pos: usize,
    ) {
        // Metal grid guard (Issue 730): (p*kvd)/128 exceeds the 65535 x-cap at
        // long prompts (262144 wg at p=32768, kvd=1024). Token-boundary
        // chunking — the K/V inputs slice at the chunk-local offset while the
        // CACHE handles slice at the ABSOLUTE offset (base_pos + t0).
        const MAX_WG_X: u32 = 65535;
        let wg_size = 128u32;
        let wg_per_token = ((kvd as u32).div_ceil(wg_size)).max(1);
        let tokens_per_chunk = (MAX_WG_X / wg_per_token).max(1) as usize;
        let mut t0 = 0usize;
        while t0 < p {
            let tc = tokens_per_chunk.min(p - t0);
            let total = tc * kvd;
            let params: [f32; 2] = [kvd as f32, tc as f32];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
            let num_wg = (total as u32).div_ceil(wg_size).max(1);
            let kv_off = (t0 * kvd * 4) as u64;
            let cache_off = ((base_pos + t0) * kvd * 4) as u64;
            unsafe {
                qwen_kv_cache_fill_split_batched_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(num_wg, 1, 1),
                    CubeDim::new_1d(wg_size),
                    BufferArg::from_raw_parts(k_handle.clone().offset_start(kv_off), total),
                    BufferArg::from_raw_parts(v_handle.clone().offset_start(kv_off), total),
                    BufferArg::from_raw_parts(
                        key_cache_handle.clone().offset_start(cache_off),
                        total,
                    ),
                    BufferArg::from_raw_parts(
                        value_cache_handle.clone().offset_start(cache_off),
                        total,
                    ),
                    BufferArg::from_raw_parts(params_handle, 2),
                );
            }
            t0 += tc;
        }
    }
}

/// Batched KV split for P tokens (Issue 653).
///
/// Splits the combined `[P, 2*kvd]` KV projection buffer into separate
/// K `[P, kvd]` and V `[P, kvd]` contiguous buffers. Needed because the
/// RMSNorm kernel requires contiguous rows of head_dim elements, and the
/// attention kernel expects K and V in separate contiguous buffers.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn qwen_split_kv_batched_f32(
    kv: &[f32],
    k: &mut [f32],
    v: &mut [f32],
    params: &[f32],
) {
    let kvd = params[0usize] as usize;
    let p = params[1usize] as usize;
    let idx = ABSOLUTE_POS;
    let total = p * kvd;
    if idx >= total {
        terminate!();
    }
    let token = idx / kvd;
    let dim = idx % kvd;
    let kv_off = token * 2usize * kvd + dim;
    k[idx] = kv[kv_off];
    v[idx] = kv[kv_off + kvd];
}

/// Launch batched KV split.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenSplitKvBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenSplitKvBatchedCubeCL {
    /// Split combined KV buffer into separate K and V for all P tokens.
    ///
    /// # Safety
    /// - `kv_handle`: `P * 2 * kvd` f32 elements
    /// - `k_handle`, `v_handle`: `P * kvd` f32 each
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        kv_handle: Handle,
        k_handle: Handle,
        v_handle: Handle,
        kvd: usize,
        p: usize,
    ) {
                // Metal grid guard (Issue 730): same class as the KV fill — token-
        // boundary chunked, sliced handles, per-chunk params. Bit-identical.
        const MAX_WG_X: u32 = 65535;
        let wg_size = 128u32;
        let wg_per_token = ((kvd as u32).div_ceil(wg_size)).max(1);
        let tokens_per_chunk = (MAX_WG_X / wg_per_token).max(1) as usize;
        let mut t0 = 0usize;
        while t0 < p {
            let tc = tokens_per_chunk.min(p - t0);
            let total = tc * kvd;
            let params: [f32; 2] = [kvd as f32, tc as f32];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
            let num_wg = (total as u32).div_ceil(wg_size).max(1);
            unsafe {
                qwen_split_kv_batched_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(num_wg, 1, 1),
                    CubeDim::new_1d(wg_size),
                    BufferArg::from_raw_parts(
                        kv_handle.clone().offset_start((t0 * 2 * kvd * 4) as u64),
                        tc * 2 * kvd,
                    ),
                    BufferArg::from_raw_parts(
                        k_handle.clone().offset_start((t0 * kvd * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(
                        v_handle.clone().offset_start((t0 * kvd * 4) as u64),
                        total,
                    ),
                    BufferArg::from_raw_parts(params_handle, 2),
                );
            }
            t0 += tc;
        }
    }
}

/// Causal-masked flash attention prefill for Qwen3.5 (Issue 653).
///
/// Extends `qwen_attention_decode_gated_f32` to handle P query tokens in a
/// single dispatch set. Each cube handles one (head, query_pos) pair, scanning
/// key positions 0..=query_pos (causal mask) with online softmax.
///
/// ## Parameter Layout
///
/// - `query`: `[P, n_head, head_dim]` — all query tokens
/// - `key`: `[P, n_kv_head, head_dim]` — all key tokens (no cache semantics)
/// - `value`: `[P, n_kv_head, head_dim]` — all value tokens
/// - `gate`: `[P, n_head, head_dim]` — per-token output gates
/// - `attn_out`: `[P, n_head, head_dim]` output
/// - `params`: `[head_dim, n_head, n_kv_head, P, scale]`
///
/// ## Dispatch
///
/// `CubeCount::Static(n_head * P, 1, 1)`, `CubeDim::new_1d(head_dim)`.
/// Each cube = one (head, query_pos). Thread `tid` = output dimension.
///
/// ## Causal Mask
///
/// Query position `q_pos` attends only to key positions `0..=q_pos`.
/// This is the standard causal mask for autoregressive prefill.
///
/// ## GQA Mapping
///
/// `kv_group(h) = h * n_kv_head / n_head`
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_prefill_gated_f32(
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
    // Metal grid guard (Issue 726, 2026-08-19): when the launch is chunked on
    // token boundaries, `p` is the CHUNK's token count and `q_offset` is the
    // chunk's first absolute position. q_pos/gate/attn_out offsets are
    // chunk-local (the handles are sliced); only the CAUSAL RANGE needs the
    // absolute position. q_offset = 0 recovers the single-launch semantics.
    let q_offset = params[5usize] as u32;

    let cube_size = head_dim;
    let cube_id = ABSOLUTE_POS as u32 / cube_size;
    // Decode (head_idx, q_pos) from cube_id.
    // Layout: cube_id = head_idx * P + q_pos
    let head_idx = cube_id / p;
    let q_pos = cube_id % p;
    let q_pos_abs = q_pos + q_offset;
    let tid = UNIT_POS;

    if head_idx >= n_head {
        terminate!();
    }

    // GQA: map query head to key/value head group
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_head_off = kv_group * head_dim;

    // Strides: each token row in the batched buffer
    let q_stride = n_head * head_dim;
    let kv_stride = n_kv_head * head_dim;

    // Query base offset: query[q_pos, head_idx, :]
    let q_token_off = q_pos * q_stride + head_idx * head_dim;

    let valid_dim = tid < head_dim;

    // Shared memory for max reduction + weight storage.
    // head_dim = 256 for Bonsai-27B; smem must cover the full cube.
    // The reduction tree starts at stride 128 (= head_dim/2).
    let mut smem = Shared::<[f32]>::new_slice(256usize);

    // Online softmax state
    let mut running_max = f32::new(-1e30f32);
    let mut running_sum = f32::new(0.0f32);
    let mut running_out = f32::new(0.0f32);

    // Causal: query position q_pos attends to key positions 0..=q_pos
    // (absolute — the K/V handles are never sliced across chunks).
    let n_positions = q_pos_abs + 1u32;
    let n_tiles = n_positions.div_ceil(cube_size);
    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_base = tile * cube_size;
        let pos = tile_base + tid;
        let valid_pos = pos < n_positions;

        // Phase 1: Q·K score — dot(query[q_pos, head, :], key[pos, kv_group, :])
        let mut my_score = f32::new(-1e30f32);
        if valid_pos {
            let k_base = pos * kv_stride + kv_head_off;
            let mut dot = f32::new(0.0f32);
            let mut d = 0u32;
            while d < head_dim {
                dot += query[(q_token_off + d) as usize] * key[(k_base + d) as usize];
                d += 1u32;
            }
            my_score = dot * scale;
        }
        smem[tid as usize] = my_score;
        sync_cube();

        // Phase 2: parallel max reduction (cube_size threads → 1).
        // Full reduction tree for head_dim=256: stride 128 → 64 → … → 1.
        //
        // Issue 715 (2026-08-17): the tree stopped at stride 8, so `smem[0]` held
        // the max of 32 of the 256 lanes rather than the tile max — a softmax
        // shift that is correct but needlessly near the f32 overflow edge (the
        // weights below reduce to `exp(my_score - new_max)` for any `tile_max`).
        if tid < 128u32 && smem[(tid + 128u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 128u32) as usize];
        }
        sync_cube();
        let mut stride = 64u32;
        while stride > 0u32 {
            if tid < stride && smem[(tid + stride) as usize] > smem[tid as usize] {
                smem[tid as usize] = smem[(tid + stride) as usize];
            }
            sync_cube();
            stride /= 2u32;
        }

        let tile_max = smem[0usize];
        // Issue 715 race (a): thread 0 overwrites `smem[0]` in Phase 3 while
        // slower lanes may still be reading it here.
        sync_cube();

        // Online softmax update: rescale running state
        let new_max = if tile_max > running_max {
            tile_max
        } else {
            running_max
        };

        let exp_prev = (running_max - new_max).exp();
        let exp_tile = (tile_max - new_max).exp();

        running_sum = running_sum * exp_prev;
        running_out = running_out * exp_prev;
        running_max = new_max;

        // Phase 3: compute exp(score - max) and store in smem.
        if valid_pos {
            smem[tid as usize] = exp_tile * (my_score - tile_max).exp();
        } else {
            smem[tid as usize] = f32::new(0.0f32);
        }
        sync_cube();

        // Phase 4: weighted value accumulation + serial sum (Issue 612 pattern).
        let mut tile_sum = f32::new(0.0f32);
        let mut acc = f32::new(0.0f32);
        let mut pp = 0u32;
        while pp < cube_size {
            let kv_pos = tile_base + pp;
            if kv_pos < n_positions {
                let weight = smem[pp as usize];
                tile_sum = tile_sum + weight;
                if valid_dim {
                    let v_idx = kv_pos * kv_stride + kv_head_off + tid;
                    acc += weight * value[v_idx as usize];
                }
            }
            pp += 1u32;
        }
        if valid_dim {
            running_out += acc;
        }
        running_sum = running_sum + tile_sum;

        // Issue 715 race (b): barrier on the loop back-edge — the next tile's
        // Phase 1 overwrites `smem[tid]` that Phase 4 above is still reading.
        // Only reachable when `n_tiles >= 2`. See the decode kernels.
        sync_cube();

        tile += 1u32;
    }

    // Final normalization + fused output gate (same as decode kernel).
    if valid_dim {
        let inv_sum = f32::new(1.0f32) / running_sum;
        let raw = running_out * inv_sum;
        // sigmoid(gate[q_pos, head_idx, tid])
        let g_off = (q_pos * q_stride + head_idx * head_dim + tid) as usize;
        let g = gate[g_off];
        let neg_g = f32::new(0.0f32) - g;
        let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_g.exp());
        attn_out[(q_token_off + tid) as usize] = raw * sig;
    }
}

/// Launch causal-masked flash attention prefill.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionPrefillGatedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionPrefillGatedCubeCL {
    /// Launch causal flash attention prefill for P query tokens.
    ///
    /// `base_pos` is the chunk's first ABSOLUTE position (Issue 734 chunked
    /// prefill): the causal range becomes `0..=base_pos+t` and the K/V
    /// binding length becomes `(base_pos + p) * kv_stride` — pass the KV
    /// **cache** handles for `base_pos > 0` (they hold positions `0..base_pos+p`
    /// after the fill) and the scratch K/V for `base_pos == 0` (unchanged
    /// single-launch semantics).
    ///
    /// # Safety
    /// - `query_handle`: `P * n_head * head_dim` f32 elements
    /// - `key_handle`, `value_handle`: `(base_pos + P) * n_kv_head * head_dim`
    ///   f32 each when reading the cache; `P * n_kv_head * head_dim` when
    ///   `base_pos == 0`
    /// - `gate_handle`: `P * n_head * head_dim` f32 elements
    /// - `attn_out_handle`: `P * n_head * head_dim` f32 elements
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
        // Metal grid guard (Issue 726, 2026-08-19): one cube per (head, token)
        // gives n_head*p cubes — illegal above 65535 on Metal from
        // p >= 2731 at Bonsai dims (24 heads). Chunk on token boundaries:
        // query/gate/attn_out are sliced to the chunk (their offsets are
        // chunk-local), key/value are bound at FULL absolute length (causal
        // reads are absolute), and the kernel gets the chunk's absolute base
        // via q_offset.
        const MAX_WG_X: u32 = 65535;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let tokens_per_chunk = (MAX_WG_X as usize / n_head.max(1)).max(1);
        let mut t0 = 0usize;
        while t0 < p {
            let tc = tokens_per_chunk.min(p - t0);
            let params: [f32; 6] = [
                head_dim as f32,
                n_head as f32,
                n_kv_head as f32,
                tc as f32,
                scale,
                (base_pos + t0) as f32,
            ];
            let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
            let q_len = tc * n_head * head_dim;
            // Absolute K/V length: chunked callers bind the CACHE (holds
            // 0..base_pos+p); single-launch callers bind scratch rows 0..p.
            let kv_len = (base_pos + p) * n_kv_head * head_dim;
            let n_cubes = n_head * tc;
            let q_slice = query_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64);
            let g_slice = gate_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64);
            let o_slice = attn_out_handle.clone().offset_start((t0 * n_head * head_dim * 4) as u64);
            unsafe {
                qwen_attention_prefill_gated_f32::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(n_cubes as u32, 1, 1),
                    CubeDim::new_1d(head_dim as u32),
                    BufferArg::from_raw_parts(q_slice, q_len),
                    BufferArg::from_raw_parts(key_handle.clone(), kv_len),
                    BufferArg::from_raw_parts(value_handle.clone(), kv_len),
                    BufferArg::from_raw_parts(g_slice, q_len),
                    BufferArg::from_raw_parts(o_slice, q_len),
                    BufferArg::from_raw_parts(params_handle, 6),
                );
            }
            t0 += tc;
        }
    }
}

// ---------------------------------------------------------------------------
// Issue 771 T2b: plane-per-query TILED flash-attention prefill (tolerance arm)
// ---------------------------------------------------------------------------

/// Queries per cube (= plane count at `CubeDim::new_1d(256)`).
const TILED_Q_PER_CUBE: u32 = 8;

/// Plane-per-query tiled causal flash attention prefill (Issue 771 T2b, the
/// Bench 792 16K verdict's lever).
///
/// The legacy [`qwen_attention_prefill_gated_f32`] dispatches one cube per
/// (head, query token) with `head_dim` threads where **each thread computes one
/// KV-position score via a 256-iteration serial dot chain**, then a 9-barrier
/// smem max reduction, then a serial 256-iteration weighted-V loop — ~600
/// serial-chained ops + 9 barriers per (query, position). Bench 792 measured
/// that shape at **63.2% of prefill @16K** (548.95 s → 201.98 s with the
/// sub-stage masked): 15–20× off both the bandwidth and compute bounds, and
/// ~10× behind llama.cpp's flash kernel on the same silicon (their pp16384 =
/// 101.33 tok/s vs our 29.85–35.51 with flash on).
///
/// This kernel is the classic FA2 restructuring, in the plane idiom the
/// `deltanet_recurrence_f32_rowpar` substrate already proved (register
/// blocking + `plane_sum`, zero threadgroup memory, zero barriers):
///
/// - Cube = `CubeDim::new_1d(256)` = **8 planes × 32 lanes; each plane owns ONE
///   query row** (M=8 queries per cube). All softmax state lives in per-lane
///   registers — the smem max-reduction tree and every `sync_cube` disappear.
/// - Each lane owns 8 contiguous head dims (`head_dim/32 = 8` for Bonsai), so
///   a K or V row is loaded cooperatively in ONE coalesced pass (32 lanes ×
///   32 B) and is shared across the 8 planes via L1 — KV traffic ÷8, and the
///   Θ(P²) serial-dot chains collapse to 8 FMAs + a 5-step shuffle per
///   (query, position).
/// - Causal: the loop runs to the cube's max query position (uniform control
///   flow) with a per-plane register mask — planes diverge harmlessly since
///   there are no barriers to skew.
///
/// ## Numerics (tolerance arm — NOT bit-identical)
///
/// The dot reduces through `plane_sum`'s tree order and the online softmax
/// updates per-position instead of per-256-block, so results are
/// FP-equivalent, not bit-identical (the same class as the rowpar kernel's
/// documented ~9e-5). DEFAULT ON since Bench 801 (kill-switch env
/// `RIIR_PREFILL_TILED_FLASH=0` restores the legacy kernel bit-identically);
/// the pinned FNV anchors cover the legacy kill-switch path, the tiled
/// per-length anchors are recorded in Bench 800/801.
///
/// ## Dispatch contract
///
/// - `CubeDim::new_1d(256)`, `CubeCount::Static(n_head * q_tiles)` where
///   `q_tiles = ceil(p / 8)`; `cube_id = head_idx * q_tiles + q_tile`.
/// - Requires `head_dim == 256` (the launcher enforces; other head dims fall
///   back to the legacy kernel).
/// - Same buffer contract as the gated kernel, including the chunked-launch
///   `q_offset` semantics (Issue 726 guard).
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_prefill_tiled_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    gate: &[f32],
    attn_out: &mut [f32],
    params: &[f32],
) {
    // params: [head_dim, n_head, n_kv_head, p, scale, q_offset, q_tiles]
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

    // Plane/lane decomposition (simdgroup i = threads [32i, 32i+32) — the
    // mapping the smem GEMM and the rowpar kernel both rely on).
    let pl = UNIT_POS / 32u32;
    let lane = UNIT_POS_PLANE;

    let q_pos = q_tile * TILED_Q_PER_CUBE + pl;
    let active = q_pos < p;

    // GQA: map query head to key/value head group.
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_head_off = kv_group * head_dim;
    let q_stride = n_head * head_dim;
    let kv_stride = n_kv_head * head_dim;
    let q_pos_abs = q_pos + q_offset;
    // Chunk-local base of this query row (q/gate/attn_out are chunk-sliced).
    let q_off = (q_pos * q_stride + head_idx * head_dim) as usize;
    let dims_base = (lane * 8u32) as usize;

    // Lane dims (hand-unrolled: statically-indexed scalars stay in registers —
    // the Batch-49 local-memory-demotion class).
    let q0 = if active { query[q_off + dims_base] } else { f32::new(0.0f32) };
    let q1 = if active { query[q_off + dims_base + 1usize] } else { f32::new(0.0f32) };
    let q2 = if active { query[q_off + dims_base + 2usize] } else { f32::new(0.0f32) };
    let q3 = if active { query[q_off + dims_base + 3usize] } else { f32::new(0.0f32) };
    let q4 = if active { query[q_off + dims_base + 4usize] } else { f32::new(0.0f32) };
    let q5 = if active { query[q_off + dims_base + 5usize] } else { f32::new(0.0f32) };
    let q6 = if active { query[q_off + dims_base + 6usize] } else { f32::new(0.0f32) };
    let q7 = if active { query[q_off + dims_base + 7usize] } else { f32::new(0.0f32) };

    // Online softmax state, per lane (uniform across the plane after every
    // plane_sum broadcast).
    let mut run_max = f32::new(-1e30f32);
    let mut run_sum = f32::new(0.0f32);
    let mut o0 = f32::new(0.0f32);
    let mut o1 = f32::new(0.0f32);
    let mut o2 = f32::new(0.0f32);
    let mut o3 = f32::new(0.0f32);
    let mut o4 = f32::new(0.0f32);
    let mut o5 = f32::new(0.0f32);
    let mut o6 = f32::new(0.0f32);
    let mut o7 = f32::new(0.0f32);

    // Uniform loop bound: the max causal position across the cube's 8 queries
    // (+1). Inactive planes and out-of-causal-range positions mask to w = 0.
    let q_last = q_tile * TILED_Q_PER_CUBE + (TILED_Q_PER_CUBE - 1u32);
    let n_loop = if q_last < p { q_last } else { p - 1u32 } + q_offset + 1u32;

    let mut pos = 0u32;
    while pos < n_loop {
        let k_base = (pos * kv_stride + kv_head_off) as usize;
        let k0 = key[k_base + dims_base];
        let k1 = key[k_base + dims_base + 1usize];
        let k2 = key[k_base + dims_base + 2usize];
        let k3 = key[k_base + dims_base + 3usize];
        let k4 = key[k_base + dims_base + 4usize];
        let k5 = key[k_base + dims_base + 5usize];
        let k6 = key[k_base + dims_base + 6usize];
        let k7 = key[k_base + dims_base + 7usize];

        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3
            + q4 * k4 + q5 * k5 + q6 * k6 + q7 * k7;
        // Reduce + broadcast within this plane's 32 lanes (tree order — the
        // documented FP-equivalent-not-bit-identical class).
        let score = plane_sum(partial) * scale;

        let in_causal = active && pos <= q_pos_abs;
        let masked = if in_causal { score } else { f32::new(-1e30f32) };

        let new_max = if masked > run_max { masked } else { run_max };
        let w = (masked - new_max).exp();
        let correction = (run_max - new_max).exp();
        run_sum = run_sum * correction + w;

        let v_base = k_base;
        let v0 = value[v_base + dims_base];
        let v1 = value[v_base + dims_base + 1usize];
        let v2 = value[v_base + dims_base + 2usize];
        let v3 = value[v_base + dims_base + 3usize];
        let v4 = value[v_base + dims_base + 4usize];
        let v5 = value[v_base + dims_base + 5usize];
        let v6 = value[v_base + dims_base + 6usize];
        let v7 = value[v_base + dims_base + 7usize];

        o0 = o0 * correction + w * v0;
        o1 = o1 * correction + w * v1;
        o2 = o2 * correction + w * v2;
        o3 = o3 * correction + w * v3;
        o4 = o4 * correction + w * v4;
        o5 = o5 * correction + w * v5;
        o6 = o6 * correction + w * v6;
        o7 = o7 * correction + w * v7;
        run_max = new_max;

        pos += 1u32;
    }

    if active {
        let inv_sum = f32::new(1.0f32) / run_sum;
        let g_off = q_off + dims_base;
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
        attn_out[g_off] = o0 * inv_sum / (f32::new(1.0f32) + neg0.exp());
        attn_out[g_off + 1usize] = o1 * inv_sum / (f32::new(1.0f32) + neg1.exp());
        attn_out[g_off + 2usize] = o2 * inv_sum / (f32::new(1.0f32) + neg2.exp());
        attn_out[g_off + 3usize] = o3 * inv_sum / (f32::new(1.0f32) + neg3.exp());
        attn_out[g_off + 4usize] = o4 * inv_sum / (f32::new(1.0f32) + neg4.exp());
        attn_out[g_off + 5usize] = o5 * inv_sum / (f32::new(1.0f32) + neg5.exp());
        attn_out[g_off + 6usize] = o6 * inv_sum / (f32::new(1.0f32) + neg6.exp());
        attn_out[g_off + 7usize] = o7 * inv_sum / (f32::new(1.0f32) + neg7.exp());
    }
}

/// Launch the plane-per-query tiled causal flash attention prefill
/// (Issue 771 T2b). Same buffer contract as
/// [`QwenAttentionPrefillGatedCubeCL::launch`], including the chunked
/// `base_pos` cache semantics — callers route through
/// [`crate::ternary_deltanet_gpu_forward::prefill_tiled_flash_enabled`]
/// (DEFAULT ON since Bench 801; head_dim must be 256).
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionPrefillTiledCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionPrefillTiledCubeCL {
    /// # Safety
    /// Same contract as [`QwenAttentionPrefillGatedCubeCL::launch`] — identical
    /// handle shapes and chunked `base_pos` semantics; `head_dim` must be 256.
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

debug_assert_eq!(head_dim, 256, "tiled flash kernel is head_dim-256 specialized");
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Grid = n_head × q_tiles (8 queries per cube) — chunk on query-TILE
        // boundaries with the same 65535 guard as the legacy launcher.
        let q_tiles_total = p.div_ceil(TILED_Q_PER_CUBE as usize).max(1);
        let tiles_per_chunk = (MAX_WG_X as usize / n_head.max(1) / 8).max(1);
        let mut t0 = 0usize; // query-token base of the chunk
        while t0 < p {
            let tiles_left = q_tiles_total - t0.div_ceil(TILED_Q_PER_CUBE as usize);
            let tiles = tiles_per_chunk.min(tiles_left);
            let tc = (tiles * TILED_Q_PER_CUBE as usize).min(p - t0);
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
                qwen_attention_prefill_tiled_f32::launch_unchecked::<R>(
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

// ---------------------------------------------------------------------------
// Issue 721 T4a: tree-verify attention kernels (ancestor-masked)
// ---------------------------------------------------------------------------

/// Batched partial RoPE with explicit per-node positions (Issue 721 T4a).
///
/// Same rotation math as `qwen_rope_partial_batched_f32`, but each node's
/// rotation angle comes from a `positions[node]` upload (= `base_pos +
/// depth[node]`) instead of the row index — tree rows are TOPO-indexed while
/// RoPE positions follow tree depth, so the identity `pos == row` that holds
/// for prefill does NOT hold for tree verify.
///
/// Each thread handles one (node, head, pair) triple; K is rotated by the
/// first `t * n_kv_head * rotary_pairs` threads (same fused shape as the
/// prefill variant).
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_rope_partial_tree_f32(
    q: &mut [f32],
    k: &mut [f32],
    positions: &[u32],
    params: &[f32],
) {
    // params: [rotary_dim_pairs, theta_base, head_dim, n_head, n_kv_head, t]
    let rotary_pairs = params[0usize] as usize;
    let theta_base = params[1usize];
    let head_dim = params[2usize] as usize;
    let n_head = params[3usize] as usize;
    let n_kv_head = params[4usize] as usize;
    let t = params[5usize] as usize;

    let idx = ABSOLUTE_POS;
    let total_q_pairs = t * n_head * rotary_pairs;

    if idx >= total_q_pairs {
        terminate!();
    }

    let q_stride = n_head * rotary_pairs;
    let token = idx / q_stride;
    let within = idx % q_stride;
    let head = within / rotary_pairs;
    let pair = within % rotary_pairs;

    let pos = positions[token] as f32;

    let rotary_dim_f = (2usize * rotary_pairs) as f32;
    let exponent = (2usize * pair) as f32 / rotary_dim_f;
    let log_base = theta_base.ln();
    let inv_freq = (log_base * (f32::new(0.0f32) - exponent)).exp();

    let theta = pos * inv_freq;
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    // Q rotation (rotate-half / GPT-NeoX convention)
    let q_row_stride = n_head * head_dim;
    let q_head_off = token * q_row_stride + head * head_dim;
    let q_i0 = q_head_off + pair;
    let q_i1 = q_i0 + rotary_pairs;
    let q0 = q[q_i0];
    let q1 = q[q_i1];
    q[q_i0] = q0 * cos_t - q1 * sin_t;
    q[q_i1] = q0 * sin_t + q1 * cos_t;

    // K rotation — fused into the same launch (first n_kv threads of the grid)
    let total_k_pairs = t * n_kv_head * rotary_pairs;
    if idx < total_k_pairs {
        let k_stride = n_kv_head * rotary_pairs;
        let k_token = idx / k_stride;
        let k_within = idx % k_stride;
        let k_head = k_within / rotary_pairs;
        let k_pair = k_within % rotary_pairs;

        let k_pos = positions[k_token] as f32;
        let k_exponent = (2usize * k_pair) as f32 / rotary_dim_f;
        let k_inv_freq = (log_base * (f32::new(0.0f32) - k_exponent)).exp();
        let k_theta = k_pos * k_inv_freq;
        let k_cos = k_theta.cos();
        let k_sin = k_theta.sin();

        let k_row_stride = n_kv_head * head_dim;
        let k_head_off = k_token * k_row_stride + k_head * head_dim;
        let k_i0 = k_head_off + k_pair;
        let k_i1 = k_i0 + rotary_pairs;
        let k0 = k[k_i0];
        let k1 = k[k_i1];
        k[k_i0] = k0 * k_cos - k1 * k_sin;
        k[k_i1] = k0 * k_sin + k1 * k_cos;
    }
}

/// Launch tree-verify partial RoPE (per-node positions).
#[cfg(feature = "cubecl_runtime")]
pub struct QwenRopePartialTreeCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenRopePartialTreeCubeCL {
    /// Apply partial RoPE to Q and K for all `t` tree nodes in-place, each at
    /// its uploaded position.
    ///
    /// # Safety
    /// - `q_handle`: `t * n_head * head_dim` f32 elements (modified in-place)
    /// - `k_handle`: `t * n_kv_head * head_dim` f32 elements (modified in-place)
    /// - `positions_handle`: `t` u32 elements
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q_handle: Handle,
        k_handle: Handle,
        positions_handle: Handle,
        rotary_dim: usize,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        theta_base: f32,
        t: usize,
    ) {
        let rotary_pairs = rotary_dim / 2;
        let total_q_pairs = t * n_head * rotary_pairs;
        let params: [f32; 6] = [
            rotary_pairs as f32,
            theta_base,
            head_dim as f32,
            n_head as f32,
            n_kv_head as f32,
            t as f32,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));

        let wg_size = 128u32;
        let num_wg = (total_q_pairs as u32).div_ceil(wg_size).max(1);

        unsafe {
            qwen_rope_partial_tree_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(q_handle, t * n_head * head_dim),
                BufferArg::from_raw_parts(k_handle, t * n_kv_head * head_dim),
                BufferArg::from_raw_parts(positions_handle, t),
                BufferArg::from_raw_parts(params_handle, 6),
            );
        }
    }
}

/// Ancestor-masked gated flash attention for tree verify (Issue 721 T4a).
///
/// Extends `qwen_attention_prefill_gated_f32` from a causal mask to a TREE
/// mask: each cube handles one (head, node) query whose key/value set is the
/// union of
///
/// - the **committed prefix** — `key_cache`/`value_cache` positions
///   `0..n_committed` (visible to every node), and
/// - the **tree nodes** — `tree_k`/`tree_v` rows, where node `j` is visible
///   to query node `k` iff `j` is an ancestor-or-self of `k` (bit `j` in
///   `anc_lo[k] | anc_hi[k] | self-bit(k)`).
///
/// Online softmax runs over "virtual positions" `0..(n_committed + t)`:
/// `vp < n_committed` → cache slot `vp`; `vp ≥ n_committed` → tree node
/// `vp - n_committed`. Visibility is enforced by scoring masked lanes at
/// `-1e30` and giving them zero weight — the same mechanism the existing
/// kernels use for out-of-range lanes, so all `sync_cube()` calls stay in
/// uniform control flow.
///
/// ## Layouts
///
/// - `query`: `[t, n_head, head_dim]`; `gate`/`attn_out`: same shape
/// - `tree_k` (post-RoPE), `tree_v`: `[t, n_kv_head, head_dim]`
/// - `key_cache`, `value_cache`: `[n_committed, n_kv_head, head_dim]` prefix
///   of the per-layer decode caches (verify never writes them)
/// - `anc_lo`/`anc_hi`: `[t]` u32 proper-ancestor bitmasks
/// - `params`: `[head_dim, n_head, n_kv_head, t, n_committed, scale]`
///
/// ## Dispatch
///
/// `CubeCount::Static(n_head * t)`, `CubeDim::new_1d(head_dim)`;
/// `cube_id = head_idx * t + node`. GQA: `kv_group(h) = h * n_kv_head / n_head`.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::assign_op_pattern, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn qwen_attention_tree_gated_f32(
    query: &[f32],
    tree_k: &[f32],
    tree_v: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    gate: &[f32],
    anc_lo: &[u32],
    anc_hi: &[u32],
    attn_out: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as u32;
    let n_head = params[1usize] as u32;
    let n_kv_head = params[2usize] as u32;
    let t = params[3usize] as u32;
    let n_committed = params[4usize] as u32;
    let scale = params[5usize];

    let cube_size = head_dim;
    let cube_id = ABSOLUTE_POS as u32 / cube_size;
    // Layout: cube_id = head_idx * t + node
    let head_idx = cube_id / t;
    let node = cube_id % t;
    let tid = UNIT_POS;

    if head_idx >= n_head {
        terminate!();
    }

    // Ancestor-or-self visibility mask for the query node.
    let node_us = node as usize;
    let self_lo = if node < 32u32 { 1u32 << node } else { 0u32 };
    let self_hi = if node >= 32u32 { 1u32 << (node - 32u32) } else { 0u32 };
    let vis_lo = anc_lo[node_us] | self_lo;
    let vis_hi = anc_hi[node_us] | self_hi;

    let kv_stride = n_kv_head * head_dim;
    let q_stride = n_head * head_dim;
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_head_off = kv_group * head_dim;
    let q_token_off = node * q_stride + head_idx * head_dim;

    let valid_dim = tid < head_dim;

    // Shared memory for max reduction + weight storage (head_dim ≤ 256;
    // stride-128 reduction step guarded by `cube_size > 128`).
    let mut smem = Shared::<[f32]>::new_slice(256usize);

    let mut running_max = f32::new(-1e30f32);
    let mut running_sum = f32::new(0.0f32);
    let mut running_out = f32::new(0.0f32);

    let n_virtual = n_committed + t;
    let n_tiles = n_virtual.div_ceil(cube_size);
    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_base = tile * cube_size;
        let vp = tile_base + tid;
        let in_range = vp < n_virtual;
        let is_cache = vp < n_committed;
        // Wraps when vp < n_committed — only read when !is_cache.
        let tree_j = vp - n_committed;
        let visible = if !in_range {
            false
        } else if is_cache {
            true
        } else if tree_j < 32u32 {
            ((vis_lo >> tree_j) & 1u32) == 1u32
        } else {
            ((vis_hi >> (tree_j - 32u32)) & 1u32) == 1u32
        };

        // Phase 1: Q·K score over the lane's virtual position.
        let mut my_score = f32::new(-1e30f32);
        if visible {
            let mut dot = f32::new(0.0f32);
            let mut d = 0u32;
            if is_cache {
                let k_base = vp * kv_stride + kv_head_off;
                while d < head_dim {
                    dot += query[(q_token_off + d) as usize] * key_cache[(k_base + d) as usize];
                    d += 1u32;
                }
            } else {
                let k_base = tree_j * kv_stride + kv_head_off;
                while d < head_dim {
                    dot += query[(q_token_off + d) as usize] * tree_k[(k_base + d) as usize];
                    d += 1u32;
                }
            }
            my_score = dot * scale;
        }
        smem[tid as usize] = my_score;
        sync_cube();

        // Phase 2: parallel max reduction (full tree to stride 1 — Issue 715).
        if cube_size > 128u32 {
            if tid < 128u32 && smem[(tid + 128u32) as usize] > smem[tid as usize] {
                smem[tid as usize] = smem[(tid + 128u32) as usize];
            }
            sync_cube();
        }
        let mut stride = 64u32;
        while stride > 0u32 {
            if tid < stride && smem[(tid + stride) as usize] > smem[tid as usize] {
                smem[tid as usize] = smem[(tid + stride) as usize];
            }
            sync_cube();
            stride /= 2u32;
        }

        let tile_max = smem[0usize];
        // Issue 715 race (a): Phase 3 overwrites smem[0] — barrier first.
        sync_cube();

        // Online softmax rescale.
        let new_max = if tile_max > running_max {
            tile_max
        } else {
            running_max
        };
        let exp_prev = (running_max - new_max).exp();
        let exp_tile = (tile_max - new_max).exp();
        running_sum = running_sum * exp_prev;
        running_out = running_out * exp_prev;
        running_max = new_max;

        // Phase 3: per-lane weights (masked / out-of-range lanes get 0).
        if visible {
            smem[tid as usize] = exp_tile * (my_score - tile_max).exp();
        } else {
            smem[tid as usize] = f32::new(0.0f32);
        }
        sync_cube();

        // Phase 4: weighted value accumulation. `w == 0` lanes (masked or
        // underflowed) contribute nothing to sum or accumulator.
        let zero = f32::new(0.0f32);
        let mut tile_sum = zero;
        let mut acc = zero;
        let mut p = 0u32;
        while p < cube_size {
            let vp2 = tile_base + p;
            let w = smem[p as usize];
            if w != zero {
                tile_sum = tile_sum + w;
                if valid_dim {
                    if vp2 < n_committed {
                        let v_idx = vp2 * kv_stride + kv_head_off + tid;
                        acc += w * value_cache[v_idx as usize];
                    } else {
                        let v_idx = (vp2 - n_committed) * kv_stride + kv_head_off + tid;
                        acc += w * tree_v[v_idx as usize];
                    }
                }
            }
            p += 1u32;
        }
        if valid_dim {
            running_out += acc;
        }
        running_sum = running_sum + tile_sum;

        // Issue 715 race (b): back-edge barrier before the next tile's Phase 1
        // overwrites smem[tid].
        sync_cube();

        tile += 1u32;
    }

    // Final normalization + fused output gate (sigmoid).
    if valid_dim {
        let inv_sum = f32::new(1.0f32) / running_sum;
        let raw = running_out * inv_sum;
        let g = gate[(q_token_off + tid) as usize];
        let neg_g = f32::new(0.0f32) - g;
        let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_g.exp());
        attn_out[(q_token_off + tid) as usize] = raw * sig;
    }
}

/// Launch ancestor-masked gated flash attention for tree verify.
#[cfg(feature = "cubecl_runtime")]
pub struct QwenAttentionTreeGatedCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl QwenAttentionTreeGatedCubeCL {
    /// Verify a whole draft tree through one attention layer in one dispatch.
    ///
    /// # Safety
    /// - `query_handle`: `t * n_head * head_dim` f32 elements
    /// - `tree_k_handle`, `tree_v_handle`: `t * n_kv_head * head_dim` f32 each
    /// - `key_cache_handle`, `value_cache_handle`: per-layer decode caches
    ///   (only the first `n_committed * n_kv_head * head_dim` elements are read)
    /// - `gate_handle`, `attn_out_handle`: `t * n_head * head_dim` f32 each
    /// - `anc_lo_handle`, `anc_hi_handle`: `t` u32 each (proper ancestors)
    #[allow(clippy::too_many_arguments, reason = "GPU kernel launch")]
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        tree_k_handle: Handle,
        tree_v_handle: Handle,
        key_cache_handle: Handle,
        value_cache_handle: Handle,
        gate_handle: Handle,
        anc_lo_handle: Handle,
        anc_hi_handle: Handle,
        attn_out_handle: Handle,
        head_dim: usize,
        n_head: usize,
        n_kv_head: usize,
        t: usize,
        n_committed: usize,
    ) {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let params: [f32; 6] = [
            head_dim as f32,
            n_head as f32,
            n_kv_head as f32,
            t as f32,
            n_committed as f32,
            scale,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let q_len = t * n_head * head_dim;
        let tree_kv_len = t * n_kv_head * head_dim;
        let cache_len = n_committed * n_kv_head * head_dim;
        let n_cubes = n_head * t;

        unsafe {
            qwen_attention_tree_gated_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_cubes as u32, 1, 1),
                CubeDim::new_1d(head_dim as u32),
                BufferArg::from_raw_parts(query_handle, q_len),
                BufferArg::from_raw_parts(tree_k_handle, tree_kv_len),
                BufferArg::from_raw_parts(tree_v_handle, tree_kv_len),
                BufferArg::from_raw_parts(key_cache_handle, cache_len),
                BufferArg::from_raw_parts(value_cache_handle, cache_len),
                BufferArg::from_raw_parts(gate_handle, q_len),
                BufferArg::from_raw_parts(anc_lo_handle, t),
                BufferArg::from_raw_parts(anc_hi_handle, t),
                BufferArg::from_raw_parts(attn_out_handle, q_len),
                BufferArg::from_raw_parts(params_handle, 6),
            );
        }
    }
}

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::{ActiveRuntime, CubeCLContext};

    /// Tolerance for GPU vs CPU comparison.
    const TOL: f32 = 1e-4;

    /// CPU reference for Qwen3.5-style GQA attention decode.
    ///
    /// Computes: for each query head h,
    ///   scores[j] = dot(Q[h], K_cache[j, kv_group(h)]) * scale
    ///   weights = softmax(scores)
    ///   out[h] = Σ_j weights[j] * V_cache[j, kv_group(h)]
    fn cpu_attention_decode(
        query: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        n_positions: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_stride = n_kv_head * head_dim;
        let mut out = vec![0.0f32; n_head * head_dim];

        for h in 0..n_head {
            let head_off = h * head_dim;
            let kv_group = h * n_kv_head / n_head;
            let kv_off = kv_group * head_dim;

            // Compute scores
            let mut scores = vec![0.0f32; n_positions];
            let mut max_score = f32::NEG_INFINITY;
            for (j, score) in scores.iter_mut().enumerate() {
                let k_base = j * kv_stride + kv_off;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += query[head_off + d] * key_cache[k_base + d];
                }
                *score = dot * scale;
                if *score > max_score {
                    max_score = *score;
                }
            }

            // Softmax
            let mut sum = 0.0f32;
            for score in scores.iter_mut().take(n_positions) {
                *score = (*score - max_score).exp();
                sum += *score;
            }

            // Weighted sum of values
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, &score) in scores.iter().enumerate().take(n_positions) {
                    let v_idx = j * kv_stride + kv_off + d;
                    acc += score * value_cache[v_idx];
                }
                out[head_off + d] = acc / sum;
            }
        }
        out
    }

    /// Issue 612: multi-position attention decode must match CPU reference.
    ///
    /// The original kernel had three bugs that only manifested for
    /// n_positions > 1 (i.e. token 1+):
    ///   1. smem[128..255] uninitialized (256-element smem, 128 threads)
    ///   2. max reduction read garbage from smem[tid+128]
    ///   3. destructive sum reduction corrupted weights for Phase 4
    ///
    /// This test uses n_positions=3 to exercise all three code paths.
    #[test]
    fn test_attention_decode_multi_position_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_head: usize = 4;
        let n_kv_head: usize = 2; // GQA: 2:1 ratio
        let head_dim: usize = 128;
        let n_positions: usize = 3; // >1 to exercise the multi-position path

        let q_len = n_head * head_dim;
        let kv_len = n_positions * n_kv_head * head_dim;

        // Build deterministic test data with distinct values per head/position
        // to expose indexing bugs.
        let mut query = vec![0.0f32; q_len];
        let mut key_cache = vec![0.0f32; kv_len];
        let mut value_cache = vec![0.0f32; kv_len];

        for h in 0..n_head {
            for d in 0..head_dim {
                query[h * head_dim + d] = 0.01 * ((h + 1) as f32) + 0.001 * (d as f32);
            }
        }
        for j in 0..n_positions {
            for kvh in 0..n_kv_head {
                for d in 0..head_dim {
                    let base = j * n_kv_head * head_dim + kvh * head_dim;
                    key_cache[base + d] = 0.01 * ((j + 1) as f32) + 0.001 * (d as f32);
                    value_cache[base + d] = 0.02 * ((j + 1) as f32) + 0.001 * (d as f32);
                }
            }
        }

        // Upload to GPU
        let q_handle = client.create_from_slice(f32::as_bytes(&query));
        let k_handle = client.create_from_slice(f32::as_bytes(&key_cache));
        let v_handle = client.create_from_slice(f32::as_bytes(&value_cache));
        let out_handle = client.empty(std::mem::size_of::<f32>() * q_len);

        // Run GPU kernel
        unsafe {
            QwenAttentionDecodeCubeCL::launch::<ActiveRuntime>(
                &client,
                q_handle,
                k_handle,
                v_handle,
                out_handle.clone(),
                head_dim,
                n_head,
                n_kv_head,
                n_positions,
            );
        }

        let gpu_bytes = client.read_one(out_handle).expect("read output");
        let gpu_out = f32::from_bytes(&gpu_bytes);

        // CPU reference
        let cpu_out = cpu_attention_decode(
            &query,
            &key_cache,
            &value_cache,
            n_head,
            n_kv_head,
            head_dim,
            n_positions,
        );

        // Compare
        let mut max_diff: f32 = 0.0;
        let mut worst_idx: usize = 0;
        for i in 0..q_len {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                worst_idx = i;
            }
        }

        assert!(
            max_diff <= TOL,
            "Multi-position attention mismatch: max_diff={max_diff:.6} at idx={worst_idx} \
             (gpu={:.6}, cpu={:.6}). Issue 612: smem size + reduction bug.",
            gpu_out[worst_idx],
            cpu_out[worst_idx]
        );
    }

    /// Issue 612: single-position attention should still match (regression guard).
    #[test]
    fn test_attention_decode_single_position_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_head: usize = 4;
        let n_kv_head: usize = 2;
        let head_dim: usize = 128;
        let n_positions: usize = 1;

        let q_len = n_head * head_dim;
        let kv_len = n_positions * n_kv_head * head_dim;

        let mut query = vec![0.0f32; q_len];
        let mut key_cache = vec![0.0f32; kv_len];
        let mut value_cache = vec![0.0f32; kv_len];

        for h in 0..n_head {
            for d in 0..head_dim {
                query[h * head_dim + d] = 0.01 * ((h + 1) as f32);
            }
        }
        for j in 0..n_positions {
            for kvh in 0..n_kv_head {
                for d in 0..head_dim {
                    let base = j * n_kv_head * head_dim + kvh * head_dim;
                    key_cache[base + d] = 0.01;
                    value_cache[base + d] = 0.02;
                }
            }
        }

        let q_handle = client.create_from_slice(f32::as_bytes(&query));
        let k_handle = client.create_from_slice(f32::as_bytes(&key_cache));
        let v_handle = client.create_from_slice(f32::as_bytes(&value_cache));
        let out_handle = client.empty(std::mem::size_of::<f32>() * q_len);

        unsafe {
            QwenAttentionDecodeCubeCL::launch::<ActiveRuntime>(
                &client,
                q_handle,
                k_handle,
                v_handle,
                out_handle.clone(),
                head_dim,
                n_head,
                n_kv_head,
                n_positions,
            );
        }

        let gpu_bytes = client.read_one(out_handle).expect("read output");
        let gpu_out = f32::from_bytes(&gpu_bytes);

        let cpu_out = cpu_attention_decode(
            &query,
            &key_cache,
            &value_cache,
            n_head,
            n_kv_head,
            head_dim,
            n_positions,
        );

        let mut max_diff: f32 = 0.0;
        for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }

        assert!(
            max_diff <= TOL,
            "Single-position attention mismatch: max_diff={max_diff:.6}"
        );
    }

    // ── Issue 648 G1 tests ───────────────────────────────────────────────────

    /// CPU reference: apply sigmoid output gate.
    fn cpu_output_gate(attn_out: &[f32], gate: &[f32]) -> Vec<f32> {
        attn_out
            .iter()
            .zip(gate.iter())
            .map(|(&o, &g)| {
                let sig = 1.0 / (1.0 + (-g).exp());
                o * sig
            })
            .collect()
    }

    /// Issue 648 F10 G1: gated attention decode must produce bit-identical
    /// results to separate decode + output gate.
    #[test]
    fn test_gated_decode_matches_separate() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_head = 4usize;
        let n_kv_head = 2usize;
        let head_dim = 128usize;
        let n_positions = 3usize;
        let q_dim = n_head * head_dim;
        let kv_stride = n_kv_head * head_dim;

        // Random query, key/value cache, and gate
        let mut rng = simple_seed_rng(64810);
        let query: Vec<f32> = (0..q_dim).map(|_| rng.next()).collect();
        let key_cache: Vec<f32> = (0..n_positions * kv_stride)
            .map(|_| rng.next())
            .collect();
        let value_cache: Vec<f32> = (0..n_positions * kv_stride)
            .map(|_| rng.next())
            .collect();
        let gate: Vec<f32> = (0..q_dim).map(|_| rng.next() * 2.0 - 1.0).collect();

        let q_handle = client.create_from_slice(f32::as_bytes(&query));
        let key_handle = client.create_from_slice(f32::as_bytes(&key_cache));
        let value_handle = client.create_from_slice(f32::as_bytes(&value_cache));
        let gate_handle = client.create_from_slice(f32::as_bytes(&gate));

        // Path A: separate decode + output gate
        let out_sep = client.empty(q_dim * 4);
        unsafe {
            QwenAttentionDecodeCubeCL::launch::<ActiveRuntime>(
                &client,
                q_handle.clone(),
                key_handle.clone(),
                value_handle.clone(),
                out_sep.clone(),
                head_dim,
                n_head,
                n_kv_head,
                n_positions,
            );
            QwenOutputGateCubeCL::launch::<ActiveRuntime>(
                &client,
                out_sep.clone(),
                gate_handle.clone(),
                q_dim,
            );
        }
        let sep_bytes = client.read_one(out_sep).expect("read sep output");
        let sep_out = f32::from_bytes(&sep_bytes);

        // Path B: fused gated decode
        let out_fused = client.empty(q_dim * 4);
        unsafe {
            QwenAttentionDecodeGatedCubeCL::launch::<ActiveRuntime>(
                &client,
                q_handle,
                key_handle,
                value_handle,
                gate_handle,
                out_fused.clone(),
                head_dim,
                n_head,
                n_kv_head,
                n_positions,
            );
        }
        let fused_bytes = client.read_one(out_fused).expect("read fused output");
        let fused_out = f32::from_bytes(&fused_bytes);

        // G1: bit-identical
        let mut max_diff: f32 = 0.0;
        for (s, f) in sep_out.iter().zip(fused_out.iter()) {
            max_diff = max_diff.max((s - f).abs());
        }
        assert!(
            max_diff <= TOL,
            "Gated decode vs separate mismatch: max_diff={max_diff:.6}"
        );

        // Also verify against CPU reference (decode + gate)
        let cpu_decoded = cpu_attention_decode(
            &query,
            &key_cache,
            &value_cache,
            n_head,
            n_kv_head,
            head_dim,
            n_positions,
        );
        let cpu_gated = cpu_output_gate(&cpu_decoded, &gate);
        let mut cpu_diff: f32 = 0.0;
        for (g, c) in fused_out.iter().zip(cpu_gated.iter()) {
            cpu_diff = cpu_diff.max((g - c).abs());
        }
        assert!(
            cpu_diff <= TOL,
            "Gated decode vs CPU mismatch: cpu_diff={cpu_diff:.6}"
        );
    }

    /// Issue 648 F9 G1: combined KV cache append must produce bit-identical
    /// cache state to separate K + V append.
    #[test]
    fn test_kv_cache_append_combined_matches_separate() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_kv_head = 2usize;
        let head_dim = 128usize;
        let kvd = n_kv_head * head_dim;
        let pos = 5usize;

        let mut rng = simple_seed_rng(64809);
        let k_vec: Vec<f32> = (0..kvd).map(|_| rng.next()).collect();
        let v_vec: Vec<f32> = (0..kvd).map(|_| rng.next()).collect();

        // Build combined buffer: [K | V]
        let kv_vec: Vec<f32> = k_vec.iter().chain(v_vec.iter()).copied().collect();

        // Path A: separate K + V append
        let key_cache_sep = client.empty((pos + 1) * kvd * 4);
        let value_cache_sep = client.empty((pos + 1) * kvd * 4);
        let k_h = client.create_from_slice(f32::as_bytes(&k_vec));
        let v_h = client.create_from_slice(f32::as_bytes(&v_vec));
        unsafe {
            QwenKvCacheAppendCubeCL::launch::<ActiveRuntime>(
                &client,
                k_h,
                v_h,
                key_cache_sep.clone(),
                value_cache_sep.clone(),
                kvd,
                pos,
            );
        }
        let sep_key_bytes = client.read_one(key_cache_sep).expect("read key cache");
        let sep_key = f32::from_bytes(&sep_key_bytes);
        let sep_val_bytes = client
            .read_one(value_cache_sep)
            .expect("read value cache");
        let sep_val = f32::from_bytes(&sep_val_bytes);

        // Path B: combined KV append
        let key_cache_comb = client.empty((pos + 1) * kvd * 4);
        let value_cache_comb = client.empty((pos + 1) * kvd * 4);
        let kv_h = client.create_from_slice(f32::as_bytes(&kv_vec));
        unsafe {
            QwenKvCacheAppendCombinedCubeCL::launch::<ActiveRuntime>(
                &client,
                kv_h,
                key_cache_comb.clone(),
                value_cache_comb.clone(),
                kvd,
                pos,
            );
        }
        let comb_key_bytes = client.read_one(key_cache_comb).expect("read key cache");
        let comb_key = f32::from_bytes(&comb_key_bytes);
        let comb_val_bytes = client
            .read_one(value_cache_comb)
            .expect("read value cache");
        let comb_val = f32::from_bytes(&comb_val_bytes);

        // G1: bit-identical
        let mut key_diff: f32 = 0.0;
        for (s, c) in sep_key.iter().zip(comb_key.iter()) {
            key_diff = key_diff.max((s - c).abs());
        }
        assert!(key_diff == 0.0, "Key cache mismatch: key_diff={key_diff:.6}");

        let mut val_diff: f32 = 0.0;
        for (s, c) in sep_val.iter().zip(comb_val.iter()) {
            val_diff = val_diff.max((s - c).abs());
        }
        assert!(
            val_diff == 0.0,
            "Value cache mismatch: val_diff={val_diff:.6}"
        );
    }

    /// Issue 654: head_dim=256 decode kernel smem bug. Before the fix,
    /// `smem = new_slice(128)` with `CubeDim::new_1d(256)` caused threads
    /// 128-255 to write OOB, the max reduction to drop positions 128-255,
    /// and V accumulation to read garbage. This test uses head_dim=256 +
    /// n_positions=200 to exercise the full cube + multi-tile path.
    #[test]
    fn test_attention_decode_head_dim_256_n_positions_200() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // Bonsai-27B shape: GQA 6:1, head_dim=256
        let n_head: usize = 6;
        let n_kv_head: usize = 1;
        let head_dim: usize = 256;
        let n_positions: usize = 200; // >128 → exercises stride-128 reduction

        let q_len = n_head * head_dim;
        let kv_len = n_positions * n_kv_head * head_dim;

        // Deterministic test data with distinct values per head/position/dim
        // to expose indexing bugs.
        let mut rng = simple_seed_rng(65401);
        let query: Vec<f32> = (0..q_len).map(|_| rng.next() * 0.1).collect();
        let key_cache: Vec<f32> = (0..kv_len).map(|_| rng.next() * 0.1).collect();
        let value_cache: Vec<f32> = (0..kv_len).map(|_| rng.next() * 0.1).collect();
        let gate: Vec<f32> = (0..q_len).map(|_| rng.next() * 2.0 - 1.0).collect();

        // ── Non-gated decode ──
        let q_handle = client.create_from_slice(f32::as_bytes(&query));
        let k_handle = client.create_from_slice(f32::as_bytes(&key_cache));
        let v_handle = client.create_from_slice(f32::as_bytes(&value_cache));
        let out_handle = client.empty(q_len * 4);
        unsafe {
            QwenAttentionDecodeCubeCL::launch::<ActiveRuntime>(
                &client,
                q_handle.clone(),
                k_handle.clone(),
                v_handle.clone(),
                out_handle.clone(),
                head_dim,
                n_head,
                n_kv_head,
                n_positions,
            );
        }
        let gpu_bytes = client.read_one(out_handle).expect("read decode output");
        let gpu_out = f32::from_bytes(&gpu_bytes);

        let cpu_out = cpu_attention_decode(
            &query,
            &key_cache,
            &value_cache,
            n_head,
            n_kv_head,
            head_dim,
            n_positions,
        );

        let mut max_diff: f32 = 0.0;
        let mut worst_idx: usize = 0;
        for i in 0..q_len {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                worst_idx = i;
            }
        }
        // head_dim=256 accumulates more FP error than the head_dim=128 path.
        // Use a relaxed tolerance that still catches the smem bug (which
        // produces O(1) divergence, not O(1e-3) FP noise).
        let tol_256: f32 = 1e-3;
        assert!(
            max_diff <= tol_256,
            "head_dim=256 decode vs CPU: max_diff={max_diff:.6} at idx={worst_idx} (tol={tol_256})"
        );

        // ── Gated decode ──
        let gate_handle = client.create_from_slice(f32::as_bytes(&gate));
        let out_gated = client.empty(q_len * 4);
        unsafe {
            QwenAttentionDecodeGatedCubeCL::launch::<ActiveRuntime>(
                &client,
                q_handle,
                k_handle,
                v_handle,
                gate_handle,
                out_gated.clone(),
                head_dim,
                n_head,
                n_kv_head,
                n_positions,
            );
        }
        let gated_bytes = client.read_one(out_gated).expect("read gated output");
        let gated_out = f32::from_bytes(&gated_bytes);

        let cpu_gated = cpu_output_gate(&cpu_out, &gate);
        let mut gated_diff: f32 = 0.0;
        for (g, c) in gated_out.iter().zip(cpu_gated.iter()) {
            gated_diff = gated_diff.max((g - c).abs());
        }
        assert!(
            gated_diff <= tol_256,
            "head_dim=256 gated decode vs CPU: gated_diff={gated_diff:.6} (tol={tol_256})"
        );
    }

    /// Issue 715 regression gate — the multi-tile (`n_tiles >= 2`) decode path.
    ///
    /// Every pre-existing decode test ran `n_positions <= 200`, i.e. strictly
    /// inside one KV tile (`cube_size = head_dim = 256`), so the tile loop body
    /// executed exactly once and its back-edge was never taken. The two smem
    /// races fixed in Issue 715 are reachable only on the second and later
    /// iterations, which is why a 2.94% run-to-run spread in arm_c perplexity
    /// coexisted with a green test suite.
    ///
    /// This gate pins both halves of the fix:
    ///   G1 — correctness: multi-tile online softmax still matches the CPU
    ///        reference (catches a barrier placed so aggressively it breaks the
    ///        running-max rescale, as well as the original corruption).
    ///   G2 — determinism: 5 back-to-back launches on identical inputs must be
    ///        BIT-identical, not merely within tolerance. A tolerance-only
    ///        assert cannot see a race that perturbs the low mantissa bits, and
    ///        that is exactly the failure that escaped.
    #[test]
    fn test_attention_decode_multi_tile_deterministic_issue_715() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        // Bonsai-27B shape: GQA 6:1, head_dim=256.
        let n_head: usize = 6;
        let n_kv_head: usize = 1;
        let head_dim: usize = 256;
        // 300 > cube_size (256) → n_tiles = 2. This is the whole point of the
        // test; keep it strictly above head_dim if the shape ever changes.
        let n_positions: usize = 300;
        assert!(
            n_positions > head_dim,
            "test must straddle the tile boundary to exercise the back-edge"
        );

        let q_len = n_head * head_dim;
        let kv_len = n_positions * n_kv_head * head_dim;

        let mut rng = simple_seed_rng(715_715);
        let query: Vec<f32> = (0..q_len).map(|_| rng.next() * 0.1).collect();
        let key_cache: Vec<f32> = (0..kv_len).map(|_| rng.next() * 0.1).collect();
        let value_cache: Vec<f32> = (0..kv_len).map(|_| rng.next() * 0.1).collect();
        let gate: Vec<f32> = (0..q_len).map(|_| rng.next() * 2.0 - 1.0).collect();

        let q_handle = client.create_from_slice(f32::as_bytes(&query));
        let k_handle = client.create_from_slice(f32::as_bytes(&key_cache));
        let v_handle = client.create_from_slice(f32::as_bytes(&value_cache));
        let gate_handle = client.create_from_slice(f32::as_bytes(&gate));

        let cpu_out = cpu_attention_decode(
            &query,
            &key_cache,
            &value_cache,
            n_head,
            n_kv_head,
            head_dim,
            n_positions,
        );
        let cpu_gated = cpu_output_gate(&cpu_out, &gate);

        // ── G1 + G2 over N repeats, non-gated and gated ──
        //
        // A smem race is probabilistic, so a handful of launches is a weak
        // detector: the whole reason Issue 715 escaped is that each individual
        // launch usually comes out right. Repeat enough to make a surviving race
        // very likely to show, while keeping the test in the low seconds (the
        // launches are small — 6 heads × 300 positions).
        const REPEATS: usize = 64;
        let mut prev_plain: Option<Vec<f32>> = None;
        let mut prev_gated: Option<Vec<f32>> = None;
        for run in 0..REPEATS {
            let out_handle = client.empty(q_len * 4);
            unsafe {
                QwenAttentionDecodeCubeCL::launch::<ActiveRuntime>(
                    &client,
                    q_handle.clone(),
                    k_handle.clone(),
                    v_handle.clone(),
                    out_handle.clone(),
                    head_dim,
                    n_head,
                    n_kv_head,
                    n_positions,
                );
            }
            let plain: Vec<f32> = f32::from_bytes(
                &client.read_one(out_handle).expect("read decode output"),
            )
            .to_vec();

            let out_gated = client.empty(q_len * 4);
            unsafe {
                QwenAttentionDecodeGatedCubeCL::launch::<ActiveRuntime>(
                    &client,
                    q_handle.clone(),
                    k_handle.clone(),
                    v_handle.clone(),
                    gate_handle.clone(),
                    out_gated.clone(),
                    head_dim,
                    n_head,
                    n_kv_head,
                    n_positions,
                );
            }
            let gated: Vec<f32> =
                f32::from_bytes(&client.read_one(out_gated).expect("read gated output")).to_vec();

            // G1: correctness vs CPU reference.
            let tol: f32 = 1e-3;
            let plain_diff = plain
                .iter()
                .zip(cpu_out.iter())
                .fold(0.0f32, |m, (g, c)| m.max((g - c).abs()));
            assert!(
                plain_diff <= tol,
                "run {run}: multi-tile decode vs CPU max_diff={plain_diff:.6} (tol={tol})"
            );
            let gated_diff = gated
                .iter()
                .zip(cpu_gated.iter())
                .fold(0.0f32, |m, (g, c)| m.max((g - c).abs()));
            assert!(
                gated_diff <= tol,
                "run {run}: multi-tile gated decode vs CPU max_diff={gated_diff:.6} (tol={tol})"
            );

            // G2: bit-exact reproducibility across launches.
            match prev_plain {
                Some(ref p) => {
                    let first = plain.iter().zip(p.iter()).position(|(a, b)| a != b);
                    assert!(
                        first.is_none(),
                        "run {run}: multi-tile decode NOT bit-identical to run {} \
                         (first differing element {:?}) — smem race regression",
                        run - 1,
                        first
                    );
                }
                None => prev_plain = Some(plain),
            }
            match prev_gated {
                Some(ref p) => {
                    let first = gated.iter().zip(p.iter()).position(|(a, b)| a != b);
                    assert!(
                        first.is_none(),
                        "run {run}: multi-tile gated decode NOT bit-identical to run {} \
                         (first differing element {:?}) — smem race regression",
                        run - 1,
                        first
                    );
                }
                None => prev_gated = Some(gated),
            }
        }
    }

    /// Simple deterministic PRNG for tests (no external dep).
    struct SimpleRng {
        state: u64,
    }

    impl SimpleRng {
        fn next(&mut self) -> f32 {
            // xorshift64
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            // Map to [-1, 1)
            ((self.state as i64 as f64) / (i64::MAX as f64)) as f32
        }
    }

    fn simple_seed_rng(seed: u64) -> SimpleRng {
        SimpleRng {
            state: if seed == 0 { 0xdead_beef_cafe_babe } else { seed },
        }
    }

    // ── Issue 721 T4a: tree attention + tree RoPE kernel tests ────────────

    /// CPU reference for the tree partial RoPE (rotate-half, first
    /// `rotary_dim` of each head row), matching
    /// `qwen_rope_partial_tree_f32`'s math at explicit per-node positions.
    fn cpu_rope_partial_tree(
        x: &[f32],
        positions: &[u32],
        rotary_dim: usize,
        head_dim: usize,
        n_heads: usize,
        theta_base: f32,
    ) -> Vec<f32> {
        let pairs = rotary_dim / 2;
        let mut out = x.to_vec();
        for (ti, &p) in positions.iter().enumerate() {
            for h in 0..n_heads {
                let off = ti * n_heads * head_dim + h * head_dim;
                for i in 0..pairs {
                    let exponent = (2 * i) as f32 / rotary_dim as f32;
                    let inv_freq = (theta_base.ln() * -exponent).exp();
                    let theta = p as f32 * inv_freq;
                    let (c, s) = (theta.cos(), theta.sin());
                    let (a, b) = (out[off + i], out[off + i + pairs]);
                    out[off + i] = a * c - b * s;
                    out[off + i + pairs] = a * s + b * c;
                }
            }
        }
        out
    }

    /// Issue 721 T4a: tree RoPE must apply each node's UPLOADED position
    /// (base_pos + depth), not its row index.
    #[test]
    fn test_rope_partial_tree_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let t = 9usize;
        let n_head = 4usize;
        let n_kv = 2usize;
        let head_dim = 128usize;
        let rotary_dim = 64usize; // partial
        let theta_base = 1_000_000.0f32;

        // Tree of depths 0,1,1,1,2,2,2,3,3 at base_pos = 5 — positions are
        // NOT the row index and several are equal (shared depths).
        let positions: Vec<u32> = vec![5, 6, 6, 6, 7, 7, 7, 8, 8];

        let mut rng = simple_seed_rng(0x721_721);
        let q_orig: Vec<f32> = (0..t * n_head * head_dim).map(|_| rng.next() * 0.5).collect();
        let k_orig: Vec<f32> = (0..t * n_kv * head_dim).map(|_| rng.next() * 0.5).collect();

        let q_h = client.create_from_slice(f32::as_bytes(&q_orig));
        let k_h = client.create_from_slice(f32::as_bytes(&k_orig));
        let pos_h = client.create_from_slice(bytemuck::cast_slice(&positions));

        unsafe {
            QwenRopePartialTreeCubeCL::launch::<ActiveRuntime>(
                &client,
                q_h.clone(),
                k_h.clone(),
                pos_h,
                rotary_dim,
                head_dim,
                n_head,
                n_kv,
                theta_base,
                t,
            );
        }

        let q_bytes = client.read_one(q_h).expect("read q");
        let q_gpu = f32::from_bytes(&q_bytes);
        let k_bytes = client.read_one(k_h).expect("read k");
        let k_gpu = f32::from_bytes(&k_bytes);
        let q_cpu = cpu_rope_partial_tree(&q_orig, &positions, rotary_dim, head_dim, n_head, theta_base);
        let k_cpu = cpu_rope_partial_tree(&k_orig, &positions, rotary_dim, head_dim, n_kv, theta_base);

        let mut worst = 0.0f32;
        for i in 0..q_gpu.len() {
            worst = worst.max((q_gpu[i] - q_cpu[i]).abs());
        }
        for i in 0..k_gpu.len() {
            worst = worst.max((k_gpu[i] - k_cpu[i]).abs());
        }
        assert!(
            worst <= TOL,
            "tree RoPE mismatch: worst={worst:.6} (positions must override row index)"
        );
    }

    /// CPU reference for ancestor-masked gated tree attention: query node k
    /// attends over [committed cache 0..n_committed] ∪ [ancestor-or-self tree
    /// nodes], softmax, then sigmoid output gate. Visibility here walks the
    /// parent chain directly — independent of the bitmask upload the kernel
    /// consumes, so an anc-mask bug cannot hide.
    #[allow(clippy::too_many_arguments, reason = "test reference")]
    fn cpu_attention_tree(
        query: &[f32],
        tree_k: &[f32],
        tree_v: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        gate: &[f32],
        parent: &[u32],
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        t: usize,
        n_committed: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_stride = n_kv_head * head_dim;
        let q_stride = n_head * head_dim;
        let mut out = vec![0.0f32; t * q_stride];

        let chain_of = |k: usize| -> Vec<usize> {
            let mut chain = vec![k];
            let mut cur = k;
            while parent[cur] != u32::MAX {
                cur = parent[cur] as usize;
                chain.push(cur);
            }
            chain
        };

        for k in 0..t {
            let vis = chain_of(k);
            for h in 0..n_head {
                let kv_off = (h * n_kv_head / n_head) * head_dim;
                let q_off = k * q_stride + h * head_dim;

                // (score, source index, is_cache)
                let mut entries: Vec<(f32, usize, bool)> = Vec::new();
                let mut max_s = f32::NEG_INFINITY;
                for p in 0..n_committed {
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += query[q_off + d] * key_cache[p * kv_stride + kv_off + d];
                    }
                    let s = dot * scale;
                    max_s = max_s.max(s);
                    entries.push((s, p, true));
                }
                for &j in &vis {
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += query[q_off + d] * tree_k[j * kv_stride + kv_off + d];
                    }
                    let s = dot * scale;
                    max_s = max_s.max(s);
                    entries.push((s, j, false));
                }

                let mut sum = 0.0;
                for e in entries.iter_mut() {
                    e.0 = (e.0 - max_s).exp();
                    sum += e.0;
                }

                for d in 0..head_dim {
                    let mut acc = 0.0;
                    for &(w, idx, is_cache) in entries.iter() {
                        let v = if is_cache {
                            value_cache[idx * kv_stride + kv_off + d]
                        } else {
                            tree_v[idx * kv_stride + kv_off + d]
                        };
                        acc += w * v;
                    }
                    let raw = acc / sum;
                    let g = gate[q_off + d];
                    out[q_off + d] = raw * (1.0 / (1.0 + (-g).exp()));
                }
            }
        }
        out
    }

    /// Issue 721 T4a: the ancestor-masked gated attention kernel must match
    /// the CPU tree reference — committed-prefix cache read + ancestor-or-self
    /// tree visibility, GQA, fused sigmoid gate.
    ///
    /// n_committed = 130 with head_dim = 128 makes n_virtual = 139 > cube_size
    /// → 2 tiles, exercising the Issue 715 back-edge barrier under the tree
    /// mask (a masked lane in tile 0 must not perturb tile 1's softmax).
    #[test]
    fn test_attention_tree_gated_matches_cpu() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let t = 9usize;
        let n_head = 4usize;
        let n_kv = 2usize; // GQA 2:1
        let head_dim = 128usize;
        let n_committed = 130usize;

        // Branching tree: root 0 → {1, 2, 3}; 1 → {4, 5}; 2 → {6}; 4 → {7, 8}
        let parent: Vec<u32> = vec![u32::MAX, 0, 0, 0, 1, 1, 2, 4, 4];

        // Proper-ancestor bitmasks (what the driver uploads from the plan).
        let mut anc_lo = vec![0u32; t];
        let mut anc_hi = vec![0u32; t];
        for k in 0..t {
            let mut cur = k;
            while parent[cur] != u32::MAX {
                cur = parent[cur] as usize;
                if cur < 32 {
                    anc_lo[k] |= 1 << cur;
                } else {
                    anc_hi[k] |= 1 << (cur - 32);
                }
            }
        }

        let q_len = t * n_head * head_dim;
        let tree_kv_len = t * n_kv * head_dim;
        let cache_len = n_committed * n_kv * head_dim;

        let mut rng = simple_seed_rng(0x721_a77e);
        let query: Vec<f32> = (0..q_len).map(|_| rng.next() * 0.5).collect();
        let tree_k: Vec<f32> = (0..tree_kv_len).map(|_| rng.next() * 0.5).collect();
        let tree_v: Vec<f32> = (0..tree_kv_len).map(|_| rng.next() * 0.5).collect();
        let key_cache: Vec<f32> = (0..cache_len).map(|_| rng.next() * 0.5).collect();
        let value_cache: Vec<f32> = (0..cache_len).map(|_| rng.next() * 0.5).collect();
        let gate: Vec<f32> = (0..q_len).map(|_| rng.next()).collect();

        let q_h = client.create_from_slice(f32::as_bytes(&query));
        let tk_h = client.create_from_slice(f32::as_bytes(&tree_k));
        let tv_h = client.create_from_slice(f32::as_bytes(&tree_v));
        let kc_h = client.create_from_slice(f32::as_bytes(&key_cache));
        let vc_h = client.create_from_slice(f32::as_bytes(&value_cache));
        let g_h = client.create_from_slice(f32::as_bytes(&gate));
        let alo_h = client.create_from_slice(bytemuck::cast_slice(&anc_lo));
        let ahi_h = client.create_from_slice(bytemuck::cast_slice(&anc_hi));
        let out_h = client.empty(std::mem::size_of::<f32>() * q_len);

        unsafe {
            QwenAttentionTreeGatedCubeCL::launch::<ActiveRuntime>(
                &client,
                q_h,
                tk_h,
                tv_h,
                kc_h,
                vc_h,
                g_h,
                alo_h,
                ahi_h,
                out_h.clone(),
                head_dim,
                n_head,
                n_kv,
                t,
                n_committed,
            );
        }

        let out_bytes = client.read_one(out_h).expect("read output");
        let gpu_out = f32::from_bytes(&out_bytes);
        let cpu_out = cpu_attention_tree(
            &query, &tree_k, &tree_v, &key_cache, &value_cache, &gate, &parent, n_head, n_kv,
            head_dim, t, n_committed,
        );

        let mut max_diff = 0.0f32;
        let mut worst_idx = 0usize;
        for i in 0..q_len {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                worst_idx = i;
            }
        }
        assert!(
            max_diff <= TOL,
            "tree attention mismatch: max_diff={max_diff:.6} at idx={worst_idx} \
             (gpu={:.6}, cpu={:.6})",
            gpu_out[worst_idx],
            cpu_out[worst_idx]
        );
    }
}
