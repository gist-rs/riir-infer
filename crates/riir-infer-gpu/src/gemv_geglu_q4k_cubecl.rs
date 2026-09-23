//! Fused dual GEMV + GeGLU CubeCL kernel with Q4_K quantized weights (Plan 171 T29).
//!
//! Single-dispatch kernel that computes:
//! 1. `gate[row] = dot(dequant_q4k(W_gate), input)` — GEMV with Q4_K weights
//! 2. `up[row]   = dot(dequant_q4k(W_up), input)`   — GEMV with Q4_K weights
//! 3. `output[row] = GELU(gate[row]) * up[row]`     — GeGLU activation
//!
//! This replaces 3 separate dispatches (gate Q4_K GEMV + up Q4_K GEMV + GeGLU) with 1,
//! saving 2 dispatches per layer × 26 layers = **52 dispatches** per decode token.
//!
//! # Dispatch Savings
//!
//! | Location | Before | After | Savings |
//! |----------|--------|-------|---------|
//! | MLP gate+up+GeGLU | 3 dispatches | 1 | 2 |
//! | **Per layer** | | | **2** |
//! | **26 layers** | | | **52** |
//!
//! # Weight Layout
//!
//! ```text
//! weight_q4k_combined = [W_gate_blocks(m rows) | W_up_blocks(m rows)]
//! d_dmin_combined     = [m, n, 0, 0, W_gate_dd, W_up_dd]  (4 f32 header + data)
//! output              = [GeGLU result(m)]
//! ```
//!
//! # CubeCL 4-Parameter Constraint
//!
//! CubeCL kernels are limited to exactly 4 `Array` parameters. We need:
//! 1. `weight_q4k_combined` — concatenated [W_gate|W_up] packed blocks
//! 2. `d_dmin_combined` — concatenated d/dmin data + params header
//! 3. `input` — shared input vector
//! 4. `output` — GeGLU result
//!
//! Parameters (m, n) are encoded in the first 4 f32 elements of `d_dmin_combined`
//! as a header: `[m, n, 0, 0]`. All d_dmin data offsets are adjusted by `DD_HEADER = 4`.
//!
//! # Algorithm
//!
//! Plane (subgroup) cooperative:
//! - Each plane handles one output row.
//! - Two sequential passes: first computes gate dot product, then up dot product.
//! - After both dot products, lane 0 applies GeGLU and writes the result.
//!
//! # GELU Tanh Approximation
//!
//! ```text
//! GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//! ```

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(all(test, feature = "cubecl_runtime"))]
use riir_infer_core::quant::q4k::BlockQ4K;

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_q4k_cubecl::Q4KHandle;

/// Q4_K super-block size: 256 elements per block.
const Q4K_BLOCK_SIZE: u32 = 256;

/// Q4_K block size in u32 words: 144 bytes / 4 = 36 words.
const Q4K_WORDS_PER_BLOCK: u32 = 36;

/// Number of f32 header values prepended to d_dmin_combined.
///
/// Header layout: `[m, n, 0, 0]` — 4 f32 values.
/// All d_dmin data offsets must be adjusted by this constant.
const DD_HEADER: u32 = 4;

// ---------------------------------------------------------------------------
// CubeCL helper functions (must be in-file — #[cube] functions are not importable)
// ---------------------------------------------------------------------------

/// Extract byte at index (0–3) from a u32 word (little-endian).
#[cfg(feature = "cubecl_runtime")]
#[cube]
fn get_byte(word: u32, idx: u32) -> u32 {
    (word >> (idx * 8u32)) & 0xFFu32
}

/// Decode 6-bit scale for sub-block j (0..7) from packed scale words.
///
/// Groups 0–3: `sc = byte(s0, j) & 63`
/// Groups 4–7: `sc = (byte(s2, j-4) & 0x0F) | ((byte(s0, j-4) >> 6) << 4)`
///
/// Returns f32 in range [0, 63].
#[cfg(feature = "cubecl_runtime")]
#[cube]
fn get_scale_k4(j: u32, s0: u32, _s1: u32, s2: u32) -> f32 {
    let sc_low = get_byte(s0, j) & 63u32;

    let j2 = j - 4u32;
    let sj4 = get_byte(s2, j2);
    let sjm4 = get_byte(s0, j2);
    let sc_high = (sj4 & 0x0Fu32) | ((sjm4 >> 6u32) << 4u32);

    let mut sc_u = sc_low;
    if j >= 4u32 {
        sc_u = sc_high;
    }

    sc_u as f32
}

/// Decode 6-bit min for sub-block j (0..7) from packed scale words.
///
/// Groups 0–3: `m = byte(s1, j) & 63`
/// Groups 4–7: `m = (byte(s2, j-4) >> 4) | ((byte(s1, j-4) >> 6) << 4)`
///
/// Returns f32 in range [0, 63].
#[cfg(feature = "cubecl_runtime")]
#[cube]
fn get_min_k4(j: u32, _s0: u32, s1: u32, s2: u32) -> f32 {
    let m_low = get_byte(s1, j) & 63u32;

    let j2 = j - 4u32;
    let sj4 = get_byte(s2, j2);
    let sj = get_byte(s1, j2);
    let m_high = (sj4 >> 4u32) | ((sj >> 6u32) << 4u32);

    let mut m_u = m_low;
    if j >= 4u32 {
        m_u = m_high;
    }

    m_u as f32
}

// ---------------------------------------------------------------------------
// Fused dual GEMV + GeGLU: Q4_K weights, plane cooperative
// ---------------------------------------------------------------------------

/// CubeCL fused dual GEMV + GeGLU kernel (Q4_K weights, plane cooperative).
///
/// Computes for each output row:
/// 1. Read params from d_dmin header: m, n.
/// 2. Gate pass: cooperative Q4_K dequant + dot product for gate row.
/// 3. Up pass: cooperative Q4_K dequant + dot product for up row (offset by m rows).
/// 4. `plane_sum()` reduction for both passes.
/// 5. GeGLU: `output = GELU(gate) * up`, lane 0 writes result.
///
/// ## Parameter Layout
///
/// - `weight_q4k_combined`: concatenated [W_gate|W_up] packed Q4_K blocks as `Array<u32>`
/// - `d_dmin_combined`: `[m, n, 0, 0, W_gate_dd, W_up_dd]` as `Array<f32>`
/// - `input`: `[f32; n]` — shared input vector
/// - `output`: `[f32; m]` — GeGLU results
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(m/8), 1, 1)`, `CubeDim::new_1d(256)`.
/// Each 256-thread workgroup has 8 planes (Metal subgroup=32), each plane handles one row.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_geglu_q4k_plane(
    weight_q4k_combined: &[u32],
    d_dmin_combined: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    // Read params from d_dmin header
    let m = d_dmin_combined[0usize] as u32;
    let n = d_dmin_combined[1usize] as u32;

    let blocks_per_row = n / Q4K_BLOCK_SIZE;
    let q4k_stride_per_row = blocks_per_row * Q4K_WORDS_PER_BLOCK; // u32 words per row
    let dd_stride_per_row = blocks_per_row * 2u32; // f32 elements per row

    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    if row >= m {
        terminate!();
    }

    // Gate section offsets
    let gate_q4k = row * q4k_stride_per_row;
    let gate_dd = DD_HEADER + row * dd_stride_per_row;

    // Up section offsets: up starts after gate's m rows
    let up_q4k = m * q4k_stride_per_row + row * q4k_stride_per_row;
    let up_dd = DD_HEADER + m * dd_stride_per_row + row * dd_stride_per_row;

    // ── Gate dot product ──
    let mut gate_partial = f32::new(0.0f32);

    let mut block_idx = 0u32;
    while block_idx < blocks_per_row {
        let col_base = block_idx * Q4K_BLOCK_SIZE;
        let blk_q4k = gate_q4k + block_idx * Q4K_WORDS_PER_BLOCK;
        let blk_dd = gate_dd + block_idx * 2u32;

        let d = d_dmin_combined[blk_dd as usize];
        let dmin = d_dmin_combined[(blk_dd + 1u32) as usize];

        let s0 = weight_q4k_combined[(blk_q4k + 1u32) as usize];
        let s1 = weight_q4k_combined[(blk_q4k + 2u32) as usize];
        let s2 = weight_q4k_combined[(blk_q4k + 3u32) as usize];

        let mut k = lane;
        while k < Q4K_BLOCK_SIZE {
            let col = col_base + k;

            let sub_block = k / 32u32;
            let pos = k % 32u32;

            let sc = get_scale_k4(sub_block, s0, s1, s2);
            let min_val = get_min_k4(sub_block, s0, s1, s2);

            let pair = sub_block / 2u32;
            let is_high = sub_block % 2u32;

            let qs_byte_idx = pair * 32u32 + pos;
            let qs_word_idx = blk_q4k + 4u32 + qs_byte_idx / 4u32;
            let byte_in_word = qs_byte_idx % 4u32;
            let shift_bits = byte_in_word * 8u32;
            let qs_byte = (weight_q4k_combined[qs_word_idx as usize] >> shift_bits) & 0xFFu32;

            let mut nibble = qs_byte & 0x0Fu32;
            if is_high == 1u32 {
                nibble = (qs_byte >> 4u32) & 0x0Fu32;
            }

            let dequant = d * sc * (nibble as f32) - dmin * min_val;
            gate_partial += dequant * input[col as usize];

            k += PLANE_DIM;
        }

        block_idx += 1u32;
    }

    let gate = plane_sum(gate_partial);

    // ── Up dot product ──
    let mut up_partial = f32::new(0.0f32);

    let mut block_idx2 = 0u32;
    while block_idx2 < blocks_per_row {
        let col_base = block_idx2 * Q4K_BLOCK_SIZE;
        let blk_q4k = up_q4k + block_idx2 * Q4K_WORDS_PER_BLOCK;
        let blk_dd = up_dd + block_idx2 * 2u32;

        let d = d_dmin_combined[blk_dd as usize];
        let dmin = d_dmin_combined[(blk_dd + 1u32) as usize];

        let s0 = weight_q4k_combined[(blk_q4k + 1u32) as usize];
        let s1 = weight_q4k_combined[(blk_q4k + 2u32) as usize];
        let s2 = weight_q4k_combined[(blk_q4k + 3u32) as usize];

        let mut k = lane;
        while k < Q4K_BLOCK_SIZE {
            let col = col_base + k;

            let sub_block = k / 32u32;
            let pos = k % 32u32;

            let sc = get_scale_k4(sub_block, s0, s1, s2);
            let min_val = get_min_k4(sub_block, s0, s1, s2);

            let pair = sub_block / 2u32;
            let is_high = sub_block % 2u32;

            let qs_byte_idx = pair * 32u32 + pos;
            let qs_word_idx = blk_q4k + 4u32 + qs_byte_idx / 4u32;
            let byte_in_word = qs_byte_idx % 4u32;
            let shift_bits = byte_in_word * 8u32;
            let qs_byte = (weight_q4k_combined[qs_word_idx as usize] >> shift_bits) & 0xFFu32;

            let mut nibble = qs_byte & 0x0Fu32;
            if is_high == 1u32 {
                nibble = (qs_byte >> 4u32) & 0x0Fu32;
            }

            let dequant = d * sc * (nibble as f32) - dmin * min_val;
            up_partial += dequant * input[col as usize];

            k += PLANE_DIM;
        }

        block_idx2 += 1u32;
    }

    let up = plane_sum(up_partial);

    // ── GeGLU: output = GELU(gate) * up ──
    // Only lane 0 has the final reduced values.
    if lane == 0u32 {
        // GELU tanh approximation:
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        let sqrt_2_over_pi = f32::new(0.797_884_6f32);
        let coeff = f32::new(0.044715f32);
        let half_val = f32::new(0.5f32);
        let one = f32::new(1.0f32);

        let x_cubed = gate * gate * gate;
        let inner = sqrt_2_over_pi * (gate + coeff * x_cubed);
        let tanh_val = f32::tanh(inner);
        let gelu = half_val * gate * (one + tanh_val);

        output[row as usize] = gelu * up;
    }
}

// ---------------------------------------------------------------------------
// Combined Q4_K GeGLU handle
// ---------------------------------------------------------------------------

/// Combined GPU handles for fused Q4_K gate+up GeGLU projection.
///
/// Stores concatenated gate and up weight and d/dmin buffers for single-dispatch
/// fused dual GEMV + GeGLU with Q4_K quantized weights.
///
/// # Buffer Layout
///
/// ```text
/// weight_q4k_combined = [W_gate_blocks(m rows) | W_up_blocks(m rows)]
/// d_dmin_combined     = [m, n, 0, 0, W_gate_dd, W_up_dd]
/// ```
#[cfg(feature = "cubecl_runtime")]
pub struct Q4KGegluHandle {
    /// Concatenated Q4_K packed blocks for gate|up as `Array<u32>`.
    pub weight_q4k_combined: Handle,
    /// Concatenated d/dmin with params header as `Array<f32>`.
    /// Layout: `[m, n, 0, 0, W_gate_dd, W_up_dd]`
    pub d_dmin_combined: Handle,
    /// Output dimension (number of rows for each of gate and up).
    pub m: usize,
    /// Input dimension (must be multiple of 256).
    pub n: usize,
}

#[cfg(feature = "cubecl_runtime")]
impl Q4KGegluHandle {
    /// Build combined Q4_K GeGLU handles from separate gate and up `Q4KHandle`s.
    ///
    /// Concatenates gate and up weight blocks and d/dmin buffers into single
    /// contiguous buffers, prepending a 4-element f32 header to d_dmin:
    /// `[m, n, 0, 0]`.
    ///
    /// Both handles must share the same input dimension `n` and output dimension `m`.
    pub fn from_separate(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        h_gate: &Q4KHandle,
        h_up: &Q4KHandle,
    ) -> Self {
        assert_eq!(
            h_gate.n, h_up.n,
            "Gate and up handles must have the same input dimension n"
        );
        assert_eq!(
            h_gate.m, h_up.m,
            "Gate and up handles must have the same output dimension m"
        );

        let m = h_gate.m;
        let n = h_gate.n;

        // Concatenate weight_q4k: read each handle's raw bytes and combine
        let weight_combined_bytes = {
            let gate_bytes = client.read_one(h_gate.weight_q4k.clone()).unwrap();
            let up_bytes = client.read_one(h_up.weight_q4k.clone()).unwrap();

            let mut combined =
                Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            combined.extend_from_slice(&gate_bytes);
            combined.extend_from_slice(&up_bytes);
            combined
        };

        // Concatenate d_dmin: header + gate + up
        let dd_combined_data = {
            let gate_bytes = client.read_one(h_gate.d_dmin.clone()).unwrap();
            let up_bytes = client.read_one(h_up.d_dmin.clone()).unwrap();

            let header: Vec<f32> = vec![m as f32, n as f32, 0.0f32, 0.0f32];

            let mut combined = Vec::with_capacity(
                DD_HEADER as usize * 4 + gate_bytes.len() + up_bytes.len(),
            );
            combined.extend_from_slice(f32::as_bytes(&header));
            combined.extend_from_slice(&gate_bytes);
            combined.extend_from_slice(&up_bytes);
            combined
        };

        let weight_q4k_combined = client.create_from_slice(&weight_combined_bytes);
        let d_dmin_combined = client.create_from_slice(&dd_combined_data);

        Self {
            weight_q4k_combined,
            d_dmin_combined,
            m,
            n,
        }
    }

    /// Block count per row.
    #[inline]
    pub fn blocks_per_row(&self) -> usize {
        self.n / Q4K_BLOCK_SIZE as usize
    }
}

// ---------------------------------------------------------------------------
// Launcher struct
// ---------------------------------------------------------------------------

/// CubeCL fused dual GEMV + GeGLU launcher (Q4_K weights).
///
/// Replaces three separate dispatches:
/// 1. `GemvQ4KCubeCL::launch(weight_gate, input, gate_output, m, n)`
/// 2. `GemvQ4KCubeCL::launch(weight_up, input, up_output, m, n)`
/// 3. `GegluCubeCL::launch(gate_output, up_output, output, m)`
///
/// With one fused dispatch:
/// `GemvGegluQ4KCubeCL::launch(combined, input, output)`
///
/// The caller must prepare a `Q4KGegluHandle` with concatenated gate and up weight
/// and d/dmin buffers.
#[cfg(feature = "cubecl_runtime")]
pub struct GemvGegluQ4KCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl GemvGegluQ4KCubeCL {
    /// Launch fused dual GEMV + GeGLU kernel (Q4_K weights, plane cooperative).
    ///
    /// Computes `output[row] = GELU(dot(dequant_q4k(W_gate_row), input))
    ///                        * dot(dequant_q4k(W_up_row), input)`
    /// for each output row in a single GPU dispatch.
    ///
    /// Dispatch: `ceil(m / 8)` workgroups of 256 threads.
    /// Each 256-thread workgroup has 8 planes (Metal subgroup=32),
    /// each plane handles one row with two sequential dot products + GeGLU.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `handle.weight_q4k_combined`: 2 × m × blocks_per_row × 36 u32 elements
    /// - `handle.d_dmin_combined`: 4 (header) + 2 × m × blocks_per_row × 2 f32 elements
    /// - `input_handle`: `n` f32 elements (n must be multiple of 256)
    /// - `output_handle`: `m` f32 elements
    ///
    /// The client must support `Plane::Ops` (subgroup operations).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &Q4KGegluHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32; // Metal subgroup size on Apple Silicon
        let rows_per_wg = wg_size / plane_size; // 8
        let num_wg = (handle.m as u32).div_ceil(rows_per_wg).max(1);

        let blocks_per_row = handle.blocks_per_row();
        let q4k_stride = (Q4K_WORDS_PER_BLOCK as usize) * blocks_per_row;
        let dd_stride = 2 * blocks_per_row;

        // Total weight: gate section + up section (2 × m rows)
        let total_weight_len = 2 * handle.m * q4k_stride;
        // Total dd: header + gate section + up section
        let total_dd_len = DD_HEADER as usize + 2 * handle.m * dd_stride;

        // SAFETY: Caller guarantees correct buffer sizes and plane support.
        unsafe {
            gemv_geglu_q4k_plane::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.weight_q4k_combined.clone(), total_weight_len),
                BufferArg::from_raw_parts(handle.d_dmin_combined.clone(), total_dd_len),
                BufferArg::from_raw_parts(input_handle, handle.n),
                BufferArg::from_raw_parts(output_handle, handle.m),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::CubeCLContext;
    use bytemuck::Zeroable;
    use crate::cubecl_runtime::ActiveRuntime;
    use riir_infer_core::quant::q4k::{QK_K, dequantize_row_q4_k, quantize_row_q4_k};

    /// CPU reference: dual GEMV + GeGLU with Q4_K quantization.
    fn cpu_q4k_geglu(
        weight_gate: &[f32],
        weight_up: &[f32],
        input: &[f32],
        m: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; m];

        let blocks_per_row = n.div_ceil(QK_K);
        let padded_n = blocks_per_row * QK_K;

        for row in 0..m {
            // Quantize + dequantize gate row
            let mut padded_row = vec![0.0f32; padded_n];
            padded_row[..n].copy_from_slice(&weight_gate[row * n..(row + 1) * n]);

            let mut blocks = vec![BlockQ4K::zeroed(); blocks_per_row];
            quantize_row_q4_k(&padded_row, &mut blocks);

            let mut dequant = vec![0.0f32; padded_n];
            dequantize_row_q4_k(&blocks, &mut dequant);

            let mut gate = 0.0f32;
            for j in 0..n {
                gate += dequant[j] * input[j];
            }

            // Quantize + dequantize up row
            padded_row[..n].copy_from_slice(&weight_up[row * n..(row + 1) * n]);

            let mut blocks_up = vec![BlockQ4K::zeroed(); blocks_per_row];
            quantize_row_q4_k(&padded_row, &mut blocks_up);

            let mut dequant_up = vec![0.0f32; padded_n];
            dequantize_row_q4_k(&blocks_up, &mut dequant_up);

            let mut up = 0.0f32;
            for j in 0..n {
                up += dequant_up[j] * input[j];
            }

            // GELU tanh approximation
            let x_cubed = gate * gate * gate;
            let inner = 0.797_884_6 * (gate + 0.044715 * x_cubed);
            let tanh_val = inner.tanh();
            let gelu = 0.5 * gate * (1.0 + tanh_val);

            output[row] = gelu * up;
        }

        output
    }

    /// Helper: quantize weight rows to Q4_K blocks.
    fn quantize_weight_rows(weight: &[f32], rows: usize, n: usize) -> Vec<BlockQ4K> {
        let blocks_per_row = n.div_ceil(QK_K);
        let padded_n = blocks_per_row * QK_K;
        let mut all_blocks = Vec::new();

        for row in 0..rows {
            let mut padded_row = vec![0.0f32; padded_n];
            let src = &weight[row * n..(row + 1) * n];
            padded_row[..n].copy_from_slice(src);

            let start = all_blocks.len();
            all_blocks.resize(start + blocks_per_row, BlockQ4K::zeroed());
            quantize_row_q4_k(&padded_row, &mut all_blocks[start..]);
        }

        all_blocks
    }

    /// Launch fused Q4_K GeGLU GEMV and compare against CPU reference.
    fn launch_and_verify(
        weight_gate: &[f32],
        weight_up: &[f32],
        input: &[f32],
        m: usize,
        n: usize,
        tolerance: f32,
        test_name: &str,
    ) {
        let ctx = CubeCLContext::new().expect("CubeCL init");
        let client = ctx.client();

        // Quantize weights
        let blocks_gate = quantize_weight_rows(weight_gate, m, n);
        let blocks_up = quantize_weight_rows(weight_up, m, n);

        // Pad n to multiple of QK_K (256) if needed
        let blocks_per_row = n.div_ceil(QK_K);
        let effective_n = blocks_per_row * QK_K;

        // Build combined handle via separate Q4KHandles
        let h_gate = Q4KHandle::from_blocks(&client, &blocks_gate, m, effective_n);
        let h_up = Q4KHandle::from_blocks(&client, &blocks_up, m, effective_n);
        let combined = Q4KGegluHandle::from_separate(&client, &h_gate, &h_up);

        // Pad input if needed
        let mut padded_input = input.to_vec();
        if padded_input.len() < effective_n {
            padded_input.resize(effective_n, 0.0);
        }

        let input_handle = client.create_from_slice(f32::as_bytes(&padded_input));
        let output_handle = client.empty(m * core::mem::size_of::<f32>());

        unsafe {
            GemvGegluQ4KCubeCL::launch::<ActiveRuntime>(
                &client,
                &combined,
                input_handle,
                output_handle.clone(),
            );
        }

        let result_bytes = client.read_one(output_handle).unwrap();
        let result = f32::from_bytes(&result_bytes);

        let expected = cpu_q4k_geglu(weight_gate, weight_up, input, m, n);

        let mut max_err = 0.0f32;
        for i in 0..m {
            let err = (result[i] - expected[i]).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < tolerance,
                "{test_name}: Element {i}: expected {}, got {} (diff {})",
                expected[i],
                result[i],
                err
            );
        }
        eprintln!("{test_name}: max_err = {max_err:.6} (tolerance = {tolerance})");
    }

    #[test]
    fn test_gemv_geglu_q4k_identity() {
        // Zero weights → zero gate + zero up → GeGLU(0)*0 = 0
        let m = 4;
        let n = 256;

        let weight_gate = vec![0.0f32; m * n];
        let weight_up = vec![0.0f32; m * n];
        let input = vec![1.0f32; n];

        launch_and_verify(
            &weight_gate,
            &weight_up,
            &input,
            m,
            n,
            1.0,
            "identity_zeros",
        );
    }

    #[test]
    fn test_gemv_geglu_q4k_general() {
        // 2×256 matrices for gate and up with sine-wave values.
        let m = 2;
        let n = 256;

        let weight_gate: Vec<f32> =
            (0..m * n).map(|i| (i as f32 * 0.1).sin() * 2.0).collect();
        let weight_up: Vec<f32> =
            (0..m * n).map(|i| (i as f32 * 0.05).cos() * 1.5).collect();
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();

        launch_and_verify(
            &weight_gate,
            &weight_up,
            &input,
            m,
            n,
            10.0,
            "general",
        );
    }

    #[test]
    fn test_gemv_geglu_q4k_gemma2_dims() {
        // Realistic Gemma 2 MLP dimensions scaled down: m=8, n=256.
        // In real Gemma 2: m=9216, n=2304.
        let m = 8usize;
        let n = 256usize;

        let weight_gate: Vec<f32> = (0..m * n)
            .map(|i| ((i % 97) as f32 - 48.0) / 100.0)
            .collect();
        let weight_up: Vec<f32> = (0..m * n)
            .map(|i| ((i % 53) as f32 - 26.0) / 100.0)
            .collect();
        let input: Vec<f32> =
            (0..n).map(|i| ((i % 7) as f32 - 3.0) / 10.0).collect();

        launch_and_verify(
            &weight_gate,
            &weight_up,
            &input,
            m,
            n,
            15.0,
            "gemma2_dims",
        );
    }
}
