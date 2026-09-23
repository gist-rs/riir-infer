//! Fused RMSNorm + Q4_K dequant + GEMM kernel (Issue 438 Phase 2).
//!
//! Fuses the RMSNorm pre-normalization INTO the Q4_K batched GEMV, eliminating
//! the CPU RMSNorm loop + the separate normed buffer upload. The two-kernel
//! pattern:
//!
//! 1. **Pre-pass** `compute_inv_rms_batched`: `input[batch × n]` → `inv_rms[batch]`
//!    (tiny output — 1 scalar per position). This is the only pass that needs
//!    to reduce over the full hidden dimension.
//!
//! 2. **Fused GEMV** `gemv_q4k_batched_rmsnorm`:
//!    `output[pos, row] = Σ_col dequant(W[row, col]) * input[pos, col] * inv_rms[pos] * gamma[col]`
//!
//!    Each lane applies `* inv_rms[pos] * gamma[col]` inline during the dot
//!    product — same memory access pattern as the base kernel, just with an
//!    extra 2 multiplies per element (negligible vs the Q4_K dequant cost).
//!
//! # Why this helps
//!
//! The previous experiments (element-wise GPU port + deferred readback) both
//! FAILED because they INCREASED Metal memory pressure. This fusion is
//! different:
//!
//! - **Fewer GPU buffers**: eliminates the separate `normed_attn` upload buffer
//!   (the raw hidden is already on GPU from the previous layer's output).
//! - **Same dispatch count**: the fused GEMV is still 1 dispatch per weight
//!   site — the RMSNorm is folded INTO the existing kernel, not added as a
//!   separate dispatch.
//! - **Eliminates CPU RMSNorm**: saves ~30s/step at 12B (128 positions × 48
//!   layers × 4 RMSNorm sites).
//!
//! # Safety
//!
//! Same as [`crate::GemvQ4KBatchedCubeCL`] — the caller must ensure correct
//! buffer sizes and device plane support.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use crate::gemv_q4k_cubecl::{Q4KHandle, Q4K_BLOCK_SIZE, Q4K_WORDS_PER_BLOCK};

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_q4k_cubecl::{get_min_k4, get_scale_k4};

// ---------------------------------------------------------------------------
// Pre-pass: compute inv_rms for each position
// ---------------------------------------------------------------------------

/// Compute `inv_rms[pos] = 1 / sqrt(mean(input[pos, :]^2) + eps)` for each
/// position in the batch.
///
/// Dispatch: `CubeCount::Static(batch, 1, 1)`, `CubeDim::new_1d(32)`.
/// Each cube handles one position (selected by `CUBE_POS_X`); the 32 lanes
/// cooperatively sum the squared elements via `plane_sum` (one plane per
/// cube — Issue 611: was 256 threads = 8 redundant planes).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn compute_inv_rms(
    input: &[f32],
    inv_rms_out: &mut [f32],
    params: &[f32],
) {
    let n = params[0usize] as u32;
    let batch = params[1usize] as u32;
    let pos = CUBE_POS_X;

    if pos >= batch {
        terminate!();
    }

    let base = pos * n;
    let lane = UNIT_POS_PLANE;

    // Cooperative sum of squares across n elements.
    let mut partial = f32::new(0.0f32);
    let mut col = lane;
    while col < n {
        let v = input[(base + col) as usize];
        partial += v * v;
        col += PLANE_DIM;
    }

    let sum_sq = plane_sum(partial);

    if lane == 0u32 {
        let mean_sq = sum_sq / n as f32;
        let inv_rms_val = f32::new(1.0f32) / (mean_sq + EPSILON_F32).sqrt();
        inv_rms_out[pos as usize] = inv_rms_val;
    }
}

/// Constant eps used by the kernel. GGUF models use 1e-6 for Gemma-4.
/// We bake it in to avoid passing it as a parameter (reduces arg count).
const EPSILON_F32: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Fused GEMV: RMSNorm + Q4_K dequant + dot product
// ---------------------------------------------------------------------------

/// Fused batched Q4_K GEMV with inline RMSNorm.
///
/// Computes:
/// ```text
/// output[pos, row] = Σ_col dequant(W[row, col]) * input[pos, col] * inv_rms[pos] * gamma[col]
/// ```
///
/// Dispatch: 2D — `CubeCount::Static(ceil(m/8), batch, 1)`, `CubeDim::new_1d(256)`.
/// Each plane handles one `(position, row)` output element.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_q4k_batched_rmsnorm(
    weight_q4k: &[u32],
    d_dmin: &[f32],
    input_batch: &[f32],
    gamma: &[f32],
    inv_rms_batch: &[f32],
    output_batch: &mut [f32],
    params: &[f32],
) {
    let batch = params[0usize] as u32;
    let n_per_pos = input_batch.len() as u32 / batch;
    let m = output_batch.len() as u32 / batch;
    let blocks_per_row = n_per_pos / Q4K_BLOCK_SIZE;
    let q4k_stride = blocks_per_row * Q4K_WORDS_PER_BLOCK;
    let dd_stride = blocks_per_row * 2u32;

    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let pos = ABSOLUTE_POS_Y;

    if row >= m || pos >= batch {
        terminate!();
    }

    let input_offset = pos * n_per_pos;
    let output_offset = pos * m;
    let inv_rms_val = inv_rms_batch[pos as usize];

    let lane = UNIT_POS_PLANE;

    let mut partial = f32::new(0.0f32);

    let mut block_idx = 0u32;
    while block_idx < blocks_per_row {
        let col_base = block_idx * Q4K_BLOCK_SIZE;
        let bo_q4k = row * q4k_stride + block_idx * Q4K_WORDS_PER_BLOCK;
        let bo_dd = row * dd_stride + block_idx * 2u32;

        let d = d_dmin[bo_dd as usize];
        let dmin = d_dmin[(bo_dd + 1u32) as usize];

        let s0 = weight_q4k[(bo_q4k + 1u32) as usize];
        let s1 = weight_q4k[(bo_q4k + 2u32) as usize];
        let s2 = weight_q4k[(bo_q4k + 3u32) as usize];

        let mut k = lane;
        while k < Q4K_BLOCK_SIZE {
            let col = col_base + k;

            let sub_block = k / 32u32;
            let p_in_block = k % 32u32;

            let sc = get_scale_k4(sub_block, s0, s1, s2);
            let min_val = get_min_k4(sub_block, s0, s1, s2);

            let pair = sub_block / 2u32;
            let is_high = sub_block % 2u32;

            let qs_byte_idx = pair * 32u32 + p_in_block;
            let qs_word_idx = bo_q4k + 4u32 + qs_byte_idx / 4u32;
            let byte_in_word = qs_byte_idx % 4u32;
            let shift_bits = byte_in_word * 8u32;
            let qs_byte = (weight_q4k[qs_word_idx as usize] >> shift_bits) & 0xFFu32;

            let mut nibble = qs_byte & 0x0Fu32;
            if is_high == 1u32 {
                nibble = (qs_byte >> 4u32) & 0x0Fu32;
            }

            let dequant = d * sc * (nibble as f32) - dmin * min_val;

            // Inline RMSNorm: normalize the input before accumulating.
            let normed_input = input_batch[(input_offset + col) as usize] * inv_rms_val * gamma[col as usize];

            partial += dequant * normed_input;

            k += PLANE_DIM;
        }

        block_idx += 1u32;
    }

    let result = plane_sum(partial);

    if lane == 0u32 {
        output_batch[(output_offset + row) as usize] = result;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fused RMSNorm + Q4_K batched GEMV launcher.
///
/// Performs the equivalent of:
/// 1. CPU RMSNorm: `normed[pos] = input[pos] / rms(input[pos]) * gamma`
/// 2. GPU GEMV: `output = W @ normed`
///
/// In TWO GPU dispatches (pre-pass inv_rms + fused GEMV), eliminating the
/// CPU RMSNorm loop + the normed buffer upload.
///
/// # Layout
///
/// - `input_batch`: raw (un-normalized) `[batch × n]` row-major
/// - `gamma`: norm scale `[n]`
/// - `output_batch`: `[batch × m]` row-major (caller pre-allocates)
/// - `weight`: the Q4_K handle `[m × n]`
///
/// # Safety
///
/// Same as [`crate::GemvQ4KBatchedCubeCL`]. Device must support plane ops.
#[cfg(feature = "cubecl_runtime")]
pub struct GemvQ4KBatchedRmsnormCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl GemvQ4KBatchedRmsnormCubeCL {
    /// Launch fused RMSNorm + Q4_K GEMV.
    ///
    /// # Safety
    ///
    /// See struct docs. Device must support plane operations.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &Q4KHandle,
        input_handle: Handle,
        gamma_handle: Handle,
        output_handle: Handle,
        batch: usize,
    ) {
        let n = handle.n;

        // ── Pre-pass: compute inv_rms for each position ──
        let inv_rms_handle = client.empty(batch * core::mem::size_of::<f32>());
        let rms_params = [n as f32, batch as f32, 0.0, 0.0];
        let rms_params_handle = client.create_from_slice(f32::as_bytes(&rms_params));

        // 1 cube per position, PLANE_DIM threads (32 on Metal/CUDA).
        // Issue 611: was 256 (8 planes × 32 lanes) — all 8 planes redundantly
        // computed the identical full sum + 8 threads wrote the same value
        // (benign race, correct result, 8× wasted work). One plane suffices:
        // the 32 lanes stride by PLANE_DIM to cover all n columns, then
        // plane_sum reduces exactly the lanes that contributed.
        // If batch > 65535, we'd need a 2D dispatch — but batch ≤ seq_len ≤ 1024.
        unsafe {
            compute_inv_rms::launch_unchecked::<R>(
                client,
                CubeCount::Static(batch as u32, 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(input_handle.clone(), batch * n),
                BufferArg::from_raw_parts(inv_rms_handle.clone(), batch),
                BufferArg::from_raw_parts(rms_params_handle, 1),
            );
        }

        // ── Fused GEMV: RMSNorm + dequant + dot product ──
        let wg_size = 256u32;
        let plane_size = 32u32;
        let rows_per_wg = wg_size / plane_size; // 8
        let num_wg_x = (handle.m as u32).div_ceil(rows_per_wg).max(1);
        let num_wg_y = batch as u32;

        let m = handle.m;
        let blocks_per_row = handle.blocks_per_row();
        let weight_len = m * blocks_per_row * (Q4K_WORDS_PER_BLOCK as usize);
        let dd_len = m * blocks_per_row * 2;

        let gemv_params = [batch as f32, 0.0, 0.0, 0.0];
        let gemv_params_handle = client.create_from_slice(f32::as_bytes(&gemv_params));

        unsafe {
            gemv_q4k_batched_rmsnorm::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.weight_q4k.clone(), weight_len),
                BufferArg::from_raw_parts(handle.d_dmin.clone(), dd_len),
                BufferArg::from_raw_parts(input_handle, batch * n),
                BufferArg::from_raw_parts(gamma_handle, n),
                BufferArg::from_raw_parts(inv_rms_handle, batch),
                BufferArg::from_raw_parts(output_handle, batch * m),
                BufferArg::from_raw_parts(gemv_params_handle, 1),
            );
        }
    }
}
