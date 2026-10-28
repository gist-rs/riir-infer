//! Q4_K × f32 tiled GEMM + row-gather kernels (Issue 1150 — the dense-qwen3.8
//! wgpu lane's prefill engine).
//!
//! `gemm_q4k_tile_f32`: `out[p, m] = x[p, n] · W[m, n]ᵀ` with W in GGUF Q4_K
//! (dequantized in-kernel), x f32. Tile: 32 rows × 32 tokens per cube, KTILE=64
//! staged in shared memory (12 KB total shared — under wgpu's 16 352-byte
//! default `max_workgroup_storage_size`; the Q4KHandle carries the pre-decoded
//! d/dmin pairs so the hot loop is 4 dequants + 16 FMAs per k-step per thread).
//!
//! Deliberately simple v1 (Issue 1150 G2 posture): correctness first, weights
//! re-read P/32 times from cache/DRAM. Optimization path (register accums →
//! bigger tiles, f16 activations) recorded in the issue, gated on measurement.
//!
//! `gather_rows_f32`: `out[i, :] = src[indices[i], :]` — the embedding lookup
//! for f32-precomputed `token_embd` and the last-row slice ahead of lm_head.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_q4k_cubecl::{Q4KHandle, get_min_k4, get_scale_k4};

// Q4_K geometry — LOCAL consts (comptime resolves module-level consts, not
// `use` imports; the gemv_geglu_q4k_cubecl precedent).
const Q4K_BLOCK: u32 = 256;
const Q4K_WORDS: u32 = 36;

const TM: usize = 32;
const TP: usize = 32;
const KTILE: usize = 64;
const X_TILE_LEN: usize = TP * KTILE; // 2048
const ACC_TILE_LEN: usize = TM * TP; // 1024
const THREADS: usize = 256;

#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemm_q4k_tile_f32(
    weight_q4k: &[u32],
    d_dmin: &[f32],
    x: &[f32],
    out: &mut [f32],
    params: &[f32],
) {
    // params: [n_cols, p_tokens, m_rows]
    let n = params[0usize] as usize;
    let p = params[1usize] as usize;
    let m = params[2usize] as usize;

    let base_m = CUBE_POS_X as usize * TM;
    let base_p = CUBE_POS_Y as usize * TP;
    let tid = UNIT_POS as usize;

    let tok = tid / 8; // 0..32 — this thread's token within the tile
    let row0 = (tid % 8) * 4; // 4 rows per thread: row0..row0+4

    let mut x_tile = Shared::<[f32]>::new_slice(X_TILE_LEN);
    let mut acc_tile = Shared::<[f32]>::new_slice(ACC_TILE_LEN);

    // Zero this thread's accumulator slice (rows row0..row0+4 at `tok`).
    let acc_base = tok * TM;
    acc_tile[acc_base + row0] = f32::new(0.0f32);
    acc_tile[acc_base + row0 + 1usize] = f32::new(0.0f32);
    acc_tile[acc_base + row0 + 2usize] = f32::new(0.0f32);
    acc_tile[acc_base + row0 + 3usize] = f32::new(0.0f32);
    sync_cube();

    let blocks_per_row = n / Q4K_BLOCK as usize;
    let q4k_stride = blocks_per_row * Q4K_WORDS as usize;
    let dd_stride = blocks_per_row * 2usize;

    let mut k0 = 0usize;
    while k0 < n {
        // Stage x tile: [TP tokens × KTILE cols].
        let mut idx = tid;
        while idx < X_TILE_LEN {
            let t = idx / KTILE;
            let c = idx % KTILE;
            let global_t = base_p + t;
            if global_t < p {
                x_tile[idx] = x[global_t * n + k0 + c];
            } else {
                x_tile[idx] = f32::new(0.0f32);
            }
            idx += THREADS;
        }
        sync_cube();

        // 4 owned rows × KTILE k-steps: dequant once per (row, col), FMA into
        // the thread's 4 accumulators. KTILE=64 spans ONE Q4_K sub-block pair
        // (cols k0..k0+64 hit sub-blocks 2j and 2j+1 of block k0/256 — the
        // nibble addressing below is fully general).
        let mut kk = 0usize;
        while kk < KTILE {
            let col = k0 + kk;
            let block_idx = col / Q4K_BLOCK as usize;
            let k_in_block = col % Q4K_BLOCK as usize;
            let sub_block = k_in_block / 32usize;
            let p_in_block = k_in_block % 32usize;
            let pair = sub_block / 2usize;
            let is_high = sub_block % 2usize;
            let qs_byte_idx = pair * 32usize + p_in_block;
            let byte_in_word = qs_byte_idx % 4usize;
            let shift_bits = (byte_in_word * 8usize) as u32;
            let word_off = 4usize + qs_byte_idx / 4usize;

            let xv = x_tile[tok * KTILE + kk];
            let mut r = 0usize;
            while r < 4usize {
                let row = base_m + row0 + r;
                let mut w = f32::new(0.0f32);
                if row < m {
                    let rbo_q4k = row * q4k_stride + block_idx * Q4K_WORDS as usize;
                    let rbo_dd = row * dd_stride + block_idx * 2usize;
                    let rd = d_dmin[rbo_dd];
                    let rdmin = d_dmin[rbo_dd + 1usize];
                    let rs0 = weight_q4k[rbo_q4k + 1usize];
                    let rs1 = weight_q4k[rbo_q4k + 2usize];
                    let rs2 = weight_q4k[rbo_q4k + 3usize];
                    let sc = get_scale_k4(sub_block as u32, rs0, rs1, rs2);
                    let min_val = get_min_k4(sub_block as u32, rs0, rs1, rs2);
                    let qs_byte = (weight_q4k[rbo_q4k + word_off] >> shift_bits) & 0xFFu32;
                    let mut nibble = qs_byte & 0x0Fu32;
                    if is_high == 1usize {
                        nibble = (qs_byte >> 4u32) & 0x0Fu32;
                    }
                    w = rd * sc * (nibble as f32) - rdmin * min_val;
                }
                acc_tile[acc_base + row0 + r] += w * xv;
                r += 1usize;
            }
            kk += 1usize;
        }
        sync_cube();
        k0 += KTILE;
    }

    // Drain the accumulator tile to global out[p, m].
    let mut idx = tid;
    while idx < ACC_TILE_LEN {
        let t = idx / TM;
        let row = idx % TM;
        let global_t = base_p + t;
        let global_row = base_m + row;
        if global_t < p && global_row < m {
            out[global_t * m + global_row] = acc_tile[idx];
        }
        idx += THREADS;
    }
}

/// Launcher for the tiled Q4_K × f32 GEMM (full-n reduction, output overwritten).
#[cfg(feature = "cubecl_runtime")]
pub struct GemmQ4KTileCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemmQ4KTileCubeCL {
    /// # Safety
    /// - `x_handle`: `p * handle.n` f32 (row-major)
    /// - `out_handle`: `p * handle.m` f32 (row-major, fully overwritten)
    /// - `p` must be a multiple of [`TP`] (32) — ragged tails go through the
    ///   decode path (Issue 1150).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &Q4KHandle,
        x_handle: Handle,
        out_handle: Handle,
        p: usize,
    ) {
        assert!(p.is_multiple_of(TP), "p ({p}) must be a multiple of {TP}");
        assert!(
            handle.n.is_multiple_of(KTILE),
            "n must be a multiple of {KTILE}"
        );
        let n_wg_x = (handle.m as u32).div_ceil(TM as u32).max(1);
        let n_wg_y = (p as u32).div_ceil(TP as u32).max(1);
        let params: [f32; 3] = [handle.n as f32, p as f32, handle.m as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let weight_len = handle.m * handle.blocks_per_row() * Q4K_WORDS as usize;
        let dd_len = handle.m * handle.blocks_per_row() * 2;

        unsafe {
            gemm_q4k_tile_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg_x, n_wg_y, 1),
                CubeDim::new_1d(THREADS as u32),
                BufferArg::from_raw_parts(handle.weight_q4k.clone(), weight_len),
                BufferArg::from_raw_parts(handle.d_dmin.clone(), dd_len),
                BufferArg::from_raw_parts(x_handle, p * handle.n),
                BufferArg::from_raw_parts(out_handle, p * handle.m),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// gather_rows_f32 — embedding lookup / last-row slice
// ---------------------------------------------------------------------------

#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gather_rows_f32(src: &[f32], indices: &[u32], out: &mut [f32], params: &[f32]) {
    let n = params[0usize] as usize;
    let total = params[1usize] as usize;
    let idx = ABSOLUTE_POS;
    if idx >= total {
        terminate!();
    }
    let row = idx / n;
    let col = idx % n;
    let src_row = indices[row] as usize;
    out[idx] = src[src_row * n + col];
}

/// Launcher: `out[i, :] = src[indices[i], :]` for `rows` rows of width `n`.
#[cfg(feature = "cubecl_runtime")]
pub struct GatherRowsF32CubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GatherRowsF32CubeCL {
    /// # Safety
    /// - `src_handle`: at least `max(indices) + 1` rows of `n` f32
    /// - `indices_handle`: `rows` u32
    /// - `out_handle`: `rows * n` f32
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        src_handle: Handle,
        indices_handle: Handle,
        out_handle: Handle,
        src_rows: usize,
        rows: usize,
        n: usize,
    ) {
        let total = rows * n;
        let params: [f32; 2] = [n as f32, total as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let wg = 256u32;
        let n_wg = (total as u32).div_ceil(wg).max(1);
        unsafe {
            gather_rows_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_wg, 1, 1),
                CubeDim::new_1d(wg),
                BufferArg::from_raw_parts(src_handle, src_rows * n),
                BufferArg::from_raw_parts(indices_handle, rows),
                BufferArg::from_raw_parts(out_handle, total),
                BufferArg::from_raw_parts(params_handle, 2),
            );
        }
    }
}
