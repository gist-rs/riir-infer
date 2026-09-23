//! Batched Q4_K fused dequant+GEMM kernel (Plan 320 C3 GPU backward).
//!
//! Extends [`crate::gemv_q4k_cubecl`] from single-vector GEMV to a batched
//! matrix-matrix multiply: processes all `batch` positions in ONE GPU dispatch
//! per weight site, instead of `batch` separate GEMV dispatches.
//!
//! # Why this exists
//!
//! The per-position GPU backward (`gemma4_backward_q4k_gpu`) issues ~43K
//! separate GEMV dispatches (128 positions × 7 sites × 48 layers). Each
//! dispatch forces a GPU sync (`read_one`), and the per-call overhead (~1ms)
//! dominates over the actual compute (~0.1ms). This batched kernel collapses
//! 128 dispatches → 1 per weight site (336 total for the full backward),
//! eliminating the sync bottleneck.
//!
//! # Operation
//!
//! `output_batch[batch × m] = dequant_q4k(weight[m × n]) @ input_batch[batch × n]^T`
//!
//! Both input and output are row-major: `input_batch[pos * n + col]`,
//! `output_batch[pos * m + row]`. The weight is shared across the batch.
//!
//! # Dispatch
//!
//! 2D plane (subgroup) kernel:
//! - CubeDim: `new(256, 1, 1)` — 256 threads = 8 planes (Metal subgroup=32)
//! - CubeCount: `Static(ceil(m/8), batch, 1)`
//! - Each plane handles one (position, row) pair

#[cfg(feature = "cubecl_runtime")]
use cubecl::features::Plane;

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use crate::gemv_q4k_cubecl::{Q4KHandle, Q4K_BLOCK_SIZE, Q4K_WORDS_PER_BLOCK};

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_q4k_cubecl::{get_min_k4, get_scale_k4};

// ---------------------------------------------------------------------------
// Batched plane kernel (subgroup cooperative dot product)
// ---------------------------------------------------------------------------

/// Batched Q4_K dequant+GEMM kernel using plane (subgroup) cooperative dot product.
///
/// Each plane handles one `(position, row)` output element. Lanes within the
/// plane cooperatively compute the dot product with `plane_sum()` reduction.
///
/// Batched Q4_K dequant+GEMM kernel using plane (subgroup) cooperative dot product.
///
/// Each plane handles one `(position, row)` output element. Lanes within the
/// plane cooperatively compute the dot product with `plane_sum()` reduction.
///
/// Dispatch: 1D (flattened `m × batch` rows). `CubeCount::Static(ceil(m*batch/8), 1, 1)`,
/// `CubeDim::new_1d(256)`. We flatten batch into the X dimension because CubeCL 0.10
/// on wgpu/Metal doesn't reliably dispatch the Y dimension with `new_1d` cubes.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_q4k_batched_plane(
    weight_q4k: &[u32],
    d_dmin: &[f32],
    input_batch: &[f32],
    output_batch: &mut [f32],
    params: &[f32],
) {
    let batch = params[0usize] as u32;
    let n_per_pos = input_batch.len() as u32 / batch;
    let m = output_batch.len() as u32 / batch;
    let blocks_per_row = n_per_pos / Q4K_BLOCK_SIZE;
    let q4k_stride = blocks_per_row * Q4K_WORDS_PER_BLOCK;
    let dd_stride = blocks_per_row * 2u32;

    // 2D dispatch: X = rows (8 per workgroup via planes), Y = batch position.
    // This avoids the wgpu 65535 workgroup limit that 1D flattened dispatch hits
    // for large m × batch (e.g. down_t: m=15360 × batch=128 = 245K workgroups).
    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let pos = ABSOLUTE_POS_Y;

    if row >= m || pos >= batch {
        terminate!();
    }

    let input_offset = pos * n_per_pos;
    let output_offset = pos * m;

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

        // Each lane processes elements at stride PLANE_DIM within this 256-element block.
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

            partial += dequant * input_batch[(input_offset + col) as usize];

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

/// Batched Q4_K dequant+GEMM launcher.
///
/// Computes `output_batch[batch × m] = dequant_q4k(weight) @ input_batch[batch × n]^T`
/// in a single GPU dispatch, processing all `batch` positions simultaneously.
///
/// # Layout
///
/// - `input_batch`: row-major `[batch, n]` — `input_batch[pos * n + col]`
/// - `output_batch`: row-major `[batch, m]` — `output_batch[pos * m + row]`
/// - `weight`: the Q4_K handle (same as single GEMV, `[m × n]`)
///
/// # Safety
///
/// - `input_handle` must have `batch × n` f32 elements
/// - `output_handle` must have `batch × m` f32 elements
/// - `batch > 0`
#[cfg(feature = "cubecl_runtime")]
pub struct GemvQ4KBatchedCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl GemvQ4KBatchedCubeCL {
    /// Launch batched Q4_K dequant+GEMM using the plane (subgroup) kernel.
    ///
    /// # Safety
    ///
    /// See struct docs. Additionally, the device must support plane operations
    /// (Metal on Apple Silicon does). Call `has_plane()` to check.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &Q4KHandle,
        input_handle: Handle,
        output_handle: Handle,
        batch: usize,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32; // Metal subgroup size on Apple Silicon
        let rows_per_wg = wg_size / plane_size; // 8
        // 2D dispatch: X = rows (ceil(m/8) workgroups), Y = batch positions.
        // Keeps each dimension under the wgpu 65535 limit.
        let num_wg_x = (handle.m as u32).div_ceil(rows_per_wg).max(1);
        let num_wg_y = batch as u32;

        let m = handle.m;
        let n = handle.n;
        let blocks_per_row = handle.blocks_per_row();
        let weight_len = m * blocks_per_row * (Q4K_WORDS_PER_BLOCK as usize);
        let dd_len = m * blocks_per_row * 2;

        // Pass batch as a 4-element params array (CubeCL pattern — avoid scalar args).
        // Padded to 4 elements to avoid any alignment issues with 1-element arrays.
        // Stack array — zero allocation.
        let params = [batch as f32, 0.0, 0.0, 0.0];
        let params_handle = client.create_from_slice(f32::as_bytes(&params));

        unsafe {
            gemv_q4k_batched_plane::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg_x, num_wg_y, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.weight_q4k.clone(), weight_len),
                BufferArg::from_raw_parts(handle.d_dmin.clone(), dd_len),
                BufferArg::from_raw_parts(input_handle, batch * n),
                BufferArg::from_raw_parts(output_handle, batch * m),
                BufferArg::from_raw_parts(params_handle, 1),
            );
        }
    }

    /// Check if the device supports plane (subgroup) operations.
    pub fn has_plane<R: Runtime>(client: &ComputeClient<R>) -> bool {
        client.features().plane.contains(Plane::Ops)
    }
}

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::context::GpuContext;
    use crate::cubecl_runtime::ActiveRuntime;
    use riir_infer_core::quant::q4k::{quantize_row_q4_k, BlockQ4K, QK_K};
    use bytemuck::Zeroable;

    fn quantize_matrix(data: &[f32], m: usize, n: usize) -> Vec<BlockQ4K> {
        assert!(n.is_multiple_of(QK_K));
        let blocks_per_row = n / QK_K;
        let mut blocks = vec![BlockQ4K::zeroed(); m * blocks_per_row];
        for row in 0..m {
            let src = &data[row * n..(row + 1) * n];
            let dst = &mut blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
            quantize_row_q4_k(src, dst);
        }
        blocks
    }

    /// CPU reference: batched GEMV. `output[pos, row] = sum_col W[row, col] * input[pos, col]`.
    fn cpu_batched_gemv(weight: &[f32], input_batch: &[f32], m: usize, n: usize, batch: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * m];
        for pos in 0..batch {
            for row in 0..m {
                let mut sum = 0.0f32;
                for col in 0..n {
                    sum += weight[row * n + col] * input_batch[pos * n + col];
                }
                output[pos * m + row] = sum;
            }
        }
        output
    }

    fn run_batched_gemv(
        weight_f32: &[f32],
        input_batch: &[f32],
        m: usize,
        n: usize,
        batch: usize,
    ) -> Vec<f32> {
        let ctx = GpuContext::new().expect("GPU init");
        let client = ctx.cubecl_client();
        let blocks = quantize_matrix(weight_f32, m, n);
        let handle = Q4KHandle::from_blocks(&client, &blocks, m, n);

        let input_handle = client.create_from_slice(f32::as_bytes(input_batch));
        let output_handle = client.empty(batch * m * core::mem::size_of::<f32>());

        unsafe {
            GemvQ4KBatchedCubeCL::launch::<ActiveRuntime>(
                &client, &handle, input_handle, output_handle.clone(), batch,
            );
        }

        let bytes = client.read_one(output_handle).unwrap();
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
    }

    #[test]
    fn test_batched_zeros() {
        let m = 256;
        let n = 256;
        let batch = 4;
        let weight = vec![0.5f32; m * n];
        let input = vec![0.0f32; batch * n];
        let output = run_batched_gemv(&weight, &input, m, n, batch);
        for v in &output {
            assert!(*v < 1e-4, "expected ~0, got {v}");
        }
    }

    #[test]
    fn test_batched_matches_cpu_small() {
        // Smallest meaningful batched GEMV: m=256, n=256, batch=2 (blocks_per_row=1).
        let m = 256;
        let n = 256;
        let batch = 2;
        // Use constant weights (0.01) so Q4_K quantizes cleanly.
        let weight = vec![0.5f32; m * n];
        let input: Vec<f32> = (0..batch * n).map(|i| (i as f32) * 0.001).collect();

        let cpu_out = cpu_batched_gemv(&weight, &input, m, n, batch);
        let gpu_out = run_batched_gemv(&weight, &input, m, n, batch);

        // Debug: print first few values per batch position.
        eprintln!("CPU pos0 [0..4]: {:?}", &cpu_out[0..4]);
        eprintln!("GPU pos0 [0..4]: {:?}", &gpu_out[0..4]);
        eprintln!("CPU pos1 [256..260]: {:?}", &cpu_out[256..260]);
        eprintln!("GPU pos1 [256..260]: {:?}", &gpu_out[256..260]);
        eprintln!("GPU len: {}", gpu_out.len());

        // Constant weights → deterministic output. CPU: 0.01 * sum(input[pos]).
        for (i, (cpu, gpu)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
            let diff = (cpu - gpu).abs();
            assert!(diff < 0.5, "batched GEMV mismatch at {i}: cpu={cpu:.6} gpu={gpu:.6} diff={diff:.6}");
        }
    }

    #[test]
    fn test_batched_matches_single_gemv_blocks1() {
        // Test with blocks_per_row=1 (n=256), the dimension that test_batched_matches_cpu used.
        let m = 512;
        let n = 256;
        let batch = 4;

        let weight: Vec<f32> = (0..m * n).map(|i| ((i as u32).wrapping_mul(1103515245) as f32 % 1.0) - 0.5).collect();
        let input_batch: Vec<f32> = (0..batch * n).map(|i| ((i as u32).wrapping_mul(12345) as f32 % 1.0) - 0.5).collect();

        let batched_out = run_batched_gemv(&weight, &input_batch, m, n, batch);

        // Single GEMV per position.
        let ctx = GpuContext::new().expect("GPU init");
        let client = ctx.cubecl_client();
        let blocks = quantize_matrix(&weight, m, n);
        let handle = Q4KHandle::from_blocks(&client, &blocks, m, n);

        for pos in 0..batch {
            let single_input = &input_batch[pos * n..(pos + 1) * n];
            let in_handle = client.create_from_slice(f32::as_bytes(single_input));
            let out_handle = client.empty(m * core::mem::size_of::<f32>());
            unsafe {
                crate::gemv_q4k_cubecl::GemvQ4KCubeCL::launch::<ActiveRuntime>(
                    &client, &handle, in_handle, out_handle.clone(),
                );
            }
            let single_out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(
                &client.read_one(out_handle).unwrap()
            ).to_vec();

            for row in 0..m {
                let b = batched_out[pos * m + row];
                let s = single_out[row];
                let diff = (b - s).abs();
                assert!(diff < 1e-3, "batched vs single (blocks1) mismatch pos={pos} row={row}: batched={b:.5} single={s:.5} diff={diff:.6}");
            }
        }
    }

    #[test]
    fn test_batched_matches_single_gemv() {
        // Verify batched kernel gives same results as calling single GEMV per position.
        let m = 256;
        let n = 512;
        let batch = 3;

        let weight: Vec<f32> = (0..m * n).map(|i| ((i as u32).wrapping_mul(1103515245) as f32 % 1.0) - 0.5).collect();
        let input_batch: Vec<f32> = (0..batch * n).map(|i| ((i as u32).wrapping_mul(12345) as f32 % 1.0) - 0.5).collect();

        let batched_out = run_batched_gemv(&weight, &input_batch, m, n, batch);

        // Single GEMV per position.
        let ctx = GpuContext::new().expect("GPU init");
        let client = ctx.cubecl_client();
        let blocks = quantize_matrix(&weight, m, n);
        let handle = Q4KHandle::from_blocks(&client, &blocks, m, n);

        for pos in 0..batch {
            let single_input = &input_batch[pos * n..(pos + 1) * n];
            let in_handle = client.create_from_slice(f32::as_bytes(single_input));
            let out_handle = client.empty(m * core::mem::size_of::<f32>());
            unsafe {
                crate::gemv_q4k_cubecl::GemvQ4KCubeCL::launch::<ActiveRuntime>(
                    &client, &handle, in_handle, out_handle.clone(),
                );
            }
            let single_out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(
                &client.read_one(out_handle).unwrap()
            ).to_vec();

            for row in 0..m {
                let b = batched_out[pos * m + row];
                let s = single_out[row];
                let diff = (b - s).abs();
                assert!(diff < 1e-3, "batched vs single mismatch pos={pos} row={row}: batched={b:.5} single={s:.5} diff={diff:.6}");
            }
        }
    }
}
