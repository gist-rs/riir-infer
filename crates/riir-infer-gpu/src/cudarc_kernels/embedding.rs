//! Issue 615 T6 — Embedding dequant CUDA kernel (cudarc + nvrtc).
//!
//! Ports the CubeCL `dequant_wte_row_f32` kernel to raw CUDA. Dequantizes a
//! single row of the ternary embedding table into an f32 output buffer.
//!
//! ## Ternary bit-plane layout
//!
//! The ternary embedding table stores weights as bit-planes:
//! - `pos_bits_u32`: positive bit-plane (u64 → 2×u32 cast). Layout: `[rows * blocks64 * 2]`.
//! - `neg_bits_u32`: negative bit-plane, same layout.
//! - `group_scale_f32`: per-group f32 scales. Layout: `[rows * groups_per_row]`.
//!
//! For column `c`:
//! - u64 block index = `c / 64`
//! - u32 word within block = `(c % 64) / 32`
//! - bit position within u32 = `c % 32`
//! - sign = `+1` if pos_bit set, `-1` if neg_bit set, `0` if neither
//! - group = `c / GROUP_SIZE`
//! - output[c] = sign * scale[group]
//!
//! ## Dispatch
//!
//! One thread per output column. `ceil(n / 256)` blocks × 256 threads.

#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;

use super::CudarcKernelError;

/// Ternary group size (matches `katgpt-types::GROUP_SIZE`).
const GROUP_SIZE: u32 = 128;

const EMBEDDING_CUDA_SRC: &str = r#"
extern "C" __global__ void dequant_wte_row_f32(
    const unsigned int* __restrict__ pos_bits_u32,  // [rows * blocks64 * 2]
    const unsigned int* __restrict__ neg_bits_u32,  // [rows * blocks64 * 2]
    const float* __restrict__ group_scale_f32,       // [rows * groups_per_row]
    float* __restrict__ out,                         // [n] output row
    const int row_idx,
    const int blocks64,
    const int groups_per_row,
    const int n,
    const int group_size)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    // Locate the ternary value for (row_idx, col).
    const int words_per_row = blocks64 * 2;  // u64 → 2× u32
    const int block_idx = col / 64;          // BITS_PER_BLOCK
    const int word_in_block = (col % 64) / 32;
    const int bit_pos = col % 32;

    const int word_idx = row_idx * words_per_row + block_idx * 2 + word_in_block;
    const unsigned int pos_bit = (pos_bits_u32[word_idx] >> bit_pos) & 1u;
    const unsigned int neg_bit = (neg_bits_u32[word_idx] >> bit_pos) & 1u;

    const float sign_f = (float)pos_bit - (float)neg_bit;  // +1, 0, or -1

    // Per-group scale.
    const int group = col / group_size;
    const float scale = group_scale_f32[row_idx * groups_per_row + group];

    out[col] = sign_f * scale;
}

// Issue 618 — device-pointer variant: reads row_idx from a device pointer
// instead of a scalar arg. Required for CUDA Graph capture: the scalar would
// be baked into the captured graph, but a device pointer is dereferenced at
// kernel runtime, so updating the device buffer before graph.launch() lets
// the graph process a different token each call.
extern "C" __global__ void dequant_wte_row_f32_devpos(
    const unsigned int* __restrict__ pos_bits_u32,
    const unsigned int* __restrict__ neg_bits_u32,
    const float* __restrict__ group_scale_f32,
    float* __restrict__ out,
    const int* __restrict__ row_idx_dev,   // device pointer to row_idx
    const int blocks64,
    const int groups_per_row,
    const int n,
    const int group_size)
{
    const int row_idx = *row_idx_dev;  // dereference at kernel runtime
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    const int words_per_row = blocks64 * 2;
    const int block_idx = col / 64;
    const int word_in_block = (col % 64) / 32;
    const int bit_pos = col % 32;

    const int word_idx = row_idx * words_per_row + block_idx * 2 + word_in_block;
    const unsigned int pos_bit = (pos_bits_u32[word_idx] >> bit_pos) & 1u;
    const unsigned int neg_bit = (neg_bits_u32[word_idx] >> bit_pos) & 1u;

    const float sign_f = (float)pos_bit - (float)neg_bit;
    const int group = col / group_size;
    const float scale = group_scale_f32[row_idx * groups_per_row + group];

    out[col] = sign_f * scale;
}
"#;

/// Holds the compiled embedding dequant kernel.
pub struct EmbeddingDequantKernels {
    dequant: CudaFunction,
    /// Issue 618 — device-pointer variant for CUDA Graph capture.
    dequant_devpos: CudaFunction,
    _module: Arc<CudaModule>,
}

impl EmbeddingDequantKernels {
    /// Compile the embedding dequant kernel via nvrtc.
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, CudarcKernelError> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            EMBEDDING_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;

        let module = ctx
            .load_module(ptx)
            .map_err(|e| CudarcKernelError::Compile(e.to_string()))?;

        let dequant = module
            .load_function("dequant_wte_row_f32")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;
        let dequant_devpos = module
            .load_function("dequant_wte_row_f32_devpos")
            .map_err(|e| CudarcKernelError::Compile(format!("{e}")))?;

        Ok(Self {
            dequant,
            dequant_devpos,
            _module: module,
        })
    }

    /// Dequantize a single row of the ternary embedding table.
    ///
    /// - `pos_bits_u32`, `neg_bits_u32`: bit-planes, `[rows * blocks64 * 2]` u32 each
    /// - `group_scale_f32`: per-group scales, `[rows * groups_per_row]` f32
    /// - `out`: output row, `[n]` f32 (pre-allocated)
    /// - `row_idx`: which row to dequantize
    /// - `blocks64`: number of u64 blocks per row (= `ceil(n / 64)`)
    /// - `groups_per_row`: number of scale groups per row (= `ceil(n / GROUP_SIZE)`)
    /// - `n`: output row length
    pub fn launch_dequant_row(
        &self,
        stream: &CudaStream,
        pos_bits_u32: &cudarc::driver::safe::CudaSlice<u32>,
        neg_bits_u32: &cudarc::driver::safe::CudaSlice<u32>,
        group_scale_f32: &cudarc::driver::safe::CudaSlice<f32>,
        out: &cudarc::driver::safe::CudaSlice<f32>,
        row_idx: usize,
        blocks64: usize,
        groups_per_row: usize,
        n: usize,
    ) -> Result<(), CudarcKernelError> {
        let row_idx_i32 = row_idx as i32;
        let blocks64_i32 = blocks64 as i32;
        let groups_per_row_i32 = groups_per_row as i32;
        let n_i32 = n as i32;
        let group_size_i32 = GROUP_SIZE as i32;
        let grid_x = (n as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.dequant)
                .arg(pos_bits_u32)
                .arg(neg_bits_u32)
                .arg(group_scale_f32)
                .arg(out)
                .arg(&row_idx_i32)
                .arg(&blocks64_i32)
                .arg(&groups_per_row_i32)
                .arg(&n_i32)
                .arg(&group_size_i32)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }

    /// Issue 618 — Device-pointer variant of `launch_dequant_row`.
    ///
    /// Same as `launch_dequant_row` but `row_idx` is read from `row_idx_dev`
    /// (a 1-element device buffer) at kernel runtime. Required for CUDA Graph
    /// capture: the scalar arg would be baked into the graph, but a device
    /// pointer is dereferenced per-launch.
    ///
    /// The caller must write the desired `row_idx` value to `row_idx_dev`
    /// BEFORE the kernel runs (typically via `memcpy_htod` on the same stream,
    /// ordered before `graph.launch()`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_dequant_row_devpos(
        &self,
        stream: &CudaStream,
        pos_bits_u32: &cudarc::driver::safe::CudaSlice<u32>,
        neg_bits_u32: &cudarc::driver::safe::CudaSlice<u32>,
        group_scale_f32: &cudarc::driver::safe::CudaSlice<f32>,
        out: &cudarc::driver::safe::CudaSlice<f32>,
        row_idx_dev: &cudarc::driver::safe::CudaSlice<i32>,
        blocks64: usize,
        groups_per_row: usize,
        n: usize,
    ) -> Result<(), CudarcKernelError> {
        let blocks64_i32 = blocks64 as i32;
        let groups_per_row_i32 = groups_per_row as i32;
        let n_i32 = n as i32;
        let group_size_i32 = GROUP_SIZE as i32;
        let grid_x = (n as u32).div_ceil(256).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.dequant_devpos)
                .arg(pos_bits_u32)
                .arg(neg_bits_u32)
                .arg(group_scale_f32)
                .arg(out)
                .arg(row_idx_dev)
                .arg(&blocks64_i32)
                .arg(&groups_per_row_i32)
                .arg(&n_i32)
                .arg(&group_size_i32)
                .launch(cfg)
                .map_err(|e| CudarcKernelError::Launch(e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::safe::CudaContext;

    /// Bits per u64 block (the ternary bit-plane unit).
    const BITS_PER_BLOCK: u32 = 64;

    fn cuda_or_skip() -> Option<()> {
        if CudaContext::new(0).is_err() {
            return None;
        }
        Some(())
    }

    /// CPU reference: dequantize a single ternary embedding row.
    fn cpu_dequant_row(
        pos_bits_u32: &[u32],
        neg_bits_u32: &[u32],
        group_scale_f32: &[f32],
        row_idx: usize,
        blocks64: usize,
        groups_per_row: usize,
        n: usize,
    ) -> Vec<f32> {
        let words_per_row = blocks64 * 2;
        let mut out = vec![0.0f32; n];
        for (col, out_val) in out.iter_mut().enumerate() {
            let block_idx = col / BITS_PER_BLOCK as usize;
            let word_in_block = (col % BITS_PER_BLOCK as usize) / 32;
            let bit_pos = col % 32;
            let word_idx = row_idx * words_per_row + block_idx * 2 + word_in_block;
            let pos_bit = (pos_bits_u32[word_idx] >> bit_pos) & 1;
            let neg_bit = (neg_bits_u32[word_idx] >> bit_pos) & 1;
            let sign = pos_bit as f32 - neg_bit as f32;
            let group = col / GROUP_SIZE as usize;
            let scale = group_scale_f32[row_idx * groups_per_row + group];
            *out_val = sign * scale;
        }
        out
    }

    #[test]
    fn test_dequant_wte_row_matches_cpu() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = EmbeddingDequantKernels::new(ctx).expect("compile");

        // Simulate 4 rows × 5120 columns (Bonsai-27B n_embd).
        let rows = 4usize;
        let n = 5120usize;
        let blocks64 = n.div_ceil(BITS_PER_BLOCK as usize); // 80
        let words_per_row = blocks64 * 2; // 160
        let groups_per_row = n.div_ceil(GROUP_SIZE as usize); // 40

        // Build deterministic bit-planes with mixed signs.
        let mut pos_bits = vec![0u32; rows * words_per_row];
        let mut neg_bits = vec![0u32; rows * words_per_row];
        let mut scales = vec![0.0f32; rows * groups_per_row];

        // Row 2 is the one we'll dequantize.
        let target_row = 2usize;
        for col in 0..n {
            let block_idx = col / 64;
            let word_in_block = (col % 64) / 32;
            let bit_pos = col % 32;
            let word_idx = target_row * words_per_row + block_idx * 2 + word_in_block;
            // Pattern: every 3rd col is +1, every 5th is -1, rest 0.
            if col.is_multiple_of(3) {
                pos_bits[word_idx] |= 1u32 << bit_pos;
            }
            if col.is_multiple_of(5) && !col.is_multiple_of(3) {
                neg_bits[word_idx] |= 1u32 << bit_pos;
            }
        }
        for g in 0..groups_per_row {
            scales[target_row * groups_per_row + g] = 0.5 + (g as f32) * 0.01;
        }

        let cpu_out = cpu_dequant_row(
            &pos_bits,
            &neg_bits,
            &scales,
            target_row,
            blocks64,
            groups_per_row,
            n,
        );

        let pos_dev = stream.clone_htod(&pos_bits).unwrap();
        let neg_dev = stream.clone_htod(&neg_bits).unwrap();
        let scale_dev = stream.clone_htod(&scales).unwrap();
        let out_dev = stream.alloc_zeros::<f32>(n).unwrap();

        kernels
            .launch_dequant_row(
                &stream,
                &pos_dev,
                &neg_dev,
                &scale_dev,
                &out_dev,
                target_row,
                blocks64,
                groups_per_row,
                n,
            )
            .expect("launch");
        stream.synchronize().expect("sync");

        let mut gpu_out = vec![0f32; n];
        stream.memcpy_dtoh(&out_dev, &mut gpu_out).unwrap();

        // Compare — should be bit-exact (only sign × scale, no float math).
        let mut max_diff = 0f32;
        for i in 0..n {
            let diff = (gpu_out[i] - cpu_out[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!(
            "[dequant_wte_row] n={n}, row={target_row}: max_diff={max_diff:.4e}"
        );
        assert!(max_diff < 1e-6, "dequant max_diff {max_diff:.4e}");
    }

    /// Issue 618 — verify the device-pointer variant produces the same output
    /// as the scalar variant.
    #[test]
    fn test_dequant_wte_row_devpos_matches_scalar() {
        let Some(_) = cuda_or_skip() else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let kernels = EmbeddingDequantKernels::new(ctx).expect("compile");

        let rows = 4usize;
        let n = 5120usize;
        let blocks64 = n.div_ceil(BITS_PER_BLOCK as usize);
        let words_per_row = blocks64 * 2;
        let groups_per_row = n.div_ceil(GROUP_SIZE as usize);

        let mut pos_bits = vec![0u32; rows * words_per_row];
        let mut neg_bits = vec![0u32; rows * words_per_row];
        let mut scales = vec![0.0f32; rows * groups_per_row];

        let target_row = 2usize;
        for col in 0..n {
            let block_idx = col / 64;
            let word_in_block = (col % 64) / 32;
            let bit_pos = col % 32;
            let word_idx = target_row * words_per_row + block_idx * 2 + word_in_block;
            if col.is_multiple_of(3) {
                pos_bits[word_idx] |= 1u32 << bit_pos;
            }
            if col.is_multiple_of(5) && !col.is_multiple_of(3) {
                neg_bits[word_idx] |= 1u32 << bit_pos;
            }
        }
        for g in 0..groups_per_row {
            scales[target_row * groups_per_row + g] = 0.5 + (g as f32) * 0.01;
        }

        let pos_dev = stream.clone_htod(&pos_bits).unwrap();
        let neg_dev = stream.clone_htod(&neg_bits).unwrap();
        let scale_dev = stream.clone_htod(&scales).unwrap();

        // Scalar variant.
        let out_scalar = stream.alloc_zeros::<f32>(n).unwrap();
        kernels
            .launch_dequant_row(&stream, &pos_dev, &neg_dev, &scale_dev, &out_scalar,
                target_row, blocks64, groups_per_row, n)
            .expect("scalar launch");

        // Devpos variant — write target_row to a 1-element device buffer.
        let row_idx_buf = stream.clone_htod(&[target_row as i32]).unwrap();
        let out_devpos = stream.alloc_zeros::<f32>(n).unwrap();
        kernels
            .launch_dequant_row_devpos(&stream, &pos_dev, &neg_dev, &scale_dev, &out_devpos,
                &row_idx_buf, blocks64, groups_per_row, n)
            .expect("devpos launch");
        stream.synchronize().expect("sync");

        let mut scalar_out = vec![0f32; n];
        let mut devpos_out = vec![0f32; n];
        stream.memcpy_dtoh(&out_scalar, &mut scalar_out).unwrap();
        stream.memcpy_dtoh(&out_devpos, &mut devpos_out).unwrap();

        let mut max_diff = 0f32;
        for i in 0..n {
            let diff = (scalar_out[i] - devpos_out[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!(
            "[dequant_devpos_vs_scalar] n={n}, row={target_row}: max_diff={max_diff:.4e}"
        );
        assert!(max_diff < 1e-6, "devpos dequant max_diff {max_diff:.4e}");
    }
}
