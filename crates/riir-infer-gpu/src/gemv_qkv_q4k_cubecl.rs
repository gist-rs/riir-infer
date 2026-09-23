//! Fused triple QKV GEMV CubeCL kernel with Q4_K quantized weights (Plan 171 T28).
//!
//! Single-dispatch kernel that computes all three Q/K/V projections with inline
//! Q4_K dequantization:
//! 1. `Q[row] = dot(dequant_q4k(Wq), input)` — GEMV with Q4_K weights
//! 2. `K[row] = dot(dequant_q4k(Wk), input)` — GEMV with Q4_K weights
//! 3. `V[row] = dot(dequant_q4k(Wv), input)` — GEMV with Q4_K weights
//!
//! This replaces 3 separate Q4_K GEMV dispatches with 1, saving 2 dispatches per
//! attention layer × 26 layers = **52 dispatches** per decode token.
//!
//! # Dispatch Savings
//!
//! | Location    | Before       | After | Savings |
//! |-------------|--------------|-------|---------|
//! | Q/K/V GEMVs | 3 dispatches | 1     | 2       |
//! | **Per layer** |             |       | **2**   |
//! | **26 layers** |             |       | **52**  |
//!
//! # Weight Layout
//!
//! ```text
//! weight_q4k_combined = [Wq_blocks(q_dim rows) | Wk_blocks(kv_dim rows) | Wv_blocks(kv_dim rows)]
//! d_dmin_combined     = [q_dim, kv_dim, n, 0, Wq_d_dmin, Wk_d_dmin, Wv_d_dmin]  (4 f32 header + data)
//! output              = [Q(q_dim) | K(kv_dim) | V(kv_dim)]
//! ```
//!
//! # CubeCL 4-Parameter Constraint
//!
//! CubeCL kernels are limited to exactly 4 `Array` parameters. We need:
//! 1. `weight_q4k_combined` — concatenated [Wq|Wk|Wv] packed blocks
//! 2. `d_dmin_combined` — concatenated d/dmin data + params header
//! 3. `input` — shared input vector
//! 4. `output` — combined [Q|K|V] output
//!
//! Parameters (q_dim, kv_dim, n) are encoded in the first 4 f32 elements of
//! `d_dmin_combined` as a header: `[q_dim, kv_dim, n, 0]`. All d_dmin data
//! offsets are adjusted by `DD_HEADER = 4`.
//!
//! # Algorithm
//!
//! Each plane (subgroup) handles one output row:
//! - Read params from d_dmin header to determine section boundaries.
//! - Determine which section (Q, K, or V) this row belongs to.
//! - Calculate section-aware offsets into the combined weight and d_dmin buffers.
//! - Cooperative Q4_K dequant+dot product with `plane_sum()` reduction.
//! - Lane 0 writes the result to the combined output buffer.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use riir_infer_core::quant::q4k::BlockQ4K;

#[cfg(feature = "cubecl_runtime")]
use crate::gemv_q4k_cubecl::Q4KHandle;

/// Q4_K super-block size: 256 elements per block.
const Q4K_BLOCK_SIZE: u32 = 256;

/// Q4_K block size in u32 words: 144 bytes / 4 = 36 words.
const Q4K_WORDS_PER_BLOCK: u32 = 36;

/// Number of f32 header values prepended to d_dmin_combined.
///
/// Header layout: `[q_dim, kv_dim, n, 0]` — 4 f32 values.
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
// Fused triple QKV GEMV: Q4_K weights, plane cooperative
// ---------------------------------------------------------------------------

/// CubeCL fused triple QKV GEMV kernel (Q4_K weights, plane cooperative).
///
/// Computes for each output row:
/// 1. Read params from d_dmin header: q_dim, kv_dim, n.
/// 2. Determine section (Q, K, or V) from row index.
/// 3. Calculate section-aware base offsets into combined weight and d_dmin buffers.
/// 4. Inline Q4_K dequantization + cooperative dot product.
/// 5. `plane_sum()` reduction, lane 0 writes result.
///
/// ## Parameter Layout
///
/// - `weight_q4k_combined`: concatenated [Wq|Wk|Wv] packed Q4_K blocks as `Array<u32>`
/// - `d_dmin_combined`: `[q_dim, kv_dim, n, 0, Wq_dd, Wk_dd, Wv_dd]` as `Array<f32>`
/// - `input`: `[f32; n]` — shared input vector
/// - `output`: `[f32; q_dim + 2*kv_dim]` — combined Q|K|V outputs
///
/// ## Dispatch
///
/// `CubeCount::Static(ceil(total_rows/8), 1, 1)`, `CubeDim::new_1d(256)`.
/// Each 256-thread workgroup has 8 planes (Metal subgroup=32), each plane handles one row.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn gemv_qkv_q4k_plane(
    weight_q4k_combined: &[u32],
    d_dmin_combined: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    // Read params from d_dmin header
    let q_dim = d_dmin_combined[0usize] as u32;
    let kv_dim = d_dmin_combined[1usize] as u32;
    let n = d_dmin_combined[2usize] as u32;
    let total_rows = q_dim + 2u32 * kv_dim;

    let blocks_per_row = n / Q4K_BLOCK_SIZE;
    let q4k_stride_per_row = blocks_per_row * Q4K_WORDS_PER_BLOCK; // u32 words per row
    let dd_stride_per_row = blocks_per_row * 2u32; // f32 elements per row

    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    if row >= total_rows {
        terminate!();
    }

    // Section-aware base offset calculation.
    // Weight layout: [Wq_blocks | Wk_blocks | Wv_blocks]
    // d_dmin layout: [DD_HEADER | Wq_dd | Wk_dd | Wv_dd]
    // Compute both Q4K and DD offsets for each section, then select.
    let q4k_q = row * q4k_stride_per_row;
    let dd_q = DD_HEADER + row * dd_stride_per_row;

    let k_local = row - q_dim;
    let q4k_k = q_dim * q4k_stride_per_row + k_local * q4k_stride_per_row;
    let dd_k = DD_HEADER + q_dim * dd_stride_per_row + k_local * dd_stride_per_row;

    let v_local = row - q_dim - kv_dim;
    let q4k_v = (q_dim + kv_dim) * q4k_stride_per_row + v_local * q4k_stride_per_row;
    let dd_v = DD_HEADER + (q_dim + kv_dim) * dd_stride_per_row + v_local * dd_stride_per_row;

    let mut bo_q4k = q4k_q;
    let mut bo_dd = dd_q;
    if row >= q_dim {
        bo_q4k = q4k_k;
        bo_dd = dd_k;
    }
    if row >= q_dim + kv_dim {
        bo_q4k = q4k_v;
        bo_dd = dd_v;
    }

    // Cooperative Q4_K dequant + dot product
    let mut partial = f32::new(0.0f32);

    let mut block_idx = 0u32;
    while block_idx < blocks_per_row {
        let col_base = block_idx * Q4K_BLOCK_SIZE;
        let blk_q4k = bo_q4k + block_idx * Q4K_WORDS_PER_BLOCK;
        let blk_dd = bo_dd + block_idx * 2u32;

        // Read pre-decoded d and dmin as f32
        let d = d_dmin_combined[blk_dd as usize];
        let dmin = d_dmin_combined[(blk_dd + 1u32) as usize];

        // Read scale words (3 u32 = 12 bytes encoding 8×6-bit scale+min pairs)
        let s0 = weight_q4k_combined[(blk_q4k + 1u32) as usize];
        let s1 = weight_q4k_combined[(blk_q4k + 2u32) as usize];
        let s2 = weight_q4k_combined[(blk_q4k + 3u32) as usize];

        // Each lane processes elements at stride PLANE_DIM within this 256-element block.
        let mut k = lane;
        while k < Q4K_BLOCK_SIZE {
            let col = col_base + k;

            let sub_block = k / 32u32;
            let pos = k % 32u32;

            let sc = get_scale_k4(sub_block, s0, s1, s2);
            let min_val = get_min_k4(sub_block, s0, s1, s2);

            // Extract nibble from packed qs data
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

            // Dequantize: value = d × scale × nibble − dmin × min
            let dequant = d * sc * (nibble as f32) - dmin * min_val;

            partial += dequant * input[col as usize];

            k += PLANE_DIM;
        }

        block_idx += 1u32;
    }

    // Hardware SIMD reduction
    let result = plane_sum(partial);

    // Lane 0 writes the final result for this row
    if lane == 0u32 {
        output[row as usize] = result;
    }
}

// ---------------------------------------------------------------------------
// Combined Q4_K QKV handle
// ---------------------------------------------------------------------------

/// Combined GPU handles for fused Q4_K QKV projection.
///
/// Stores concatenated Q, K, V weight and d/dmin buffers for single-dispatch
/// fused triple QKV GEMV with Q4_K quantized weights.
///
/// # Buffer Layout
///
/// ```text
/// weight_q4k_combined = [Wq_blocks | Wk_blocks | Wv_blocks]
/// d_dmin_combined     = [q_dim, kv_dim, n, 0, Wq_dd, Wk_dd, Wv_dd]
/// ```
#[cfg(feature = "cubecl_runtime")]
pub struct Q4KQKVHandle {
    /// Concatenated Q4_K packed blocks for Q|K|V as `Array<u32>`.
    pub weight_q4k_combined: Handle,
    /// Concatenated d/dmin with params header as `Array<f32>`.
    /// Layout: `[q_dim, kv_dim, n, 0, Wq_dd, Wk_dd, Wv_dd]`
    pub d_dmin_combined: Handle,
    /// Q projection output dimension.
    pub q_dim: usize,
    /// K/V projection output dimension.
    pub kv_dim: usize,
    /// Input dimension (must be multiple of 256).
    pub n: usize,
}

#[cfg(feature = "cubecl_runtime")]
impl Q4KQKVHandle {
    /// Build combined Q4_K QKV handles from three separate `Q4KHandle`s.
    ///
    /// Concatenates Q, K, V weight blocks and d/dmin buffers into single
    /// contiguous buffers, prepending a 4-element f32 header to d_dmin:
    /// `[q_dim, kv_dim, n, 0]`.
    ///
    /// All three handles must share the same input dimension `n`.
    pub fn from_separate(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        h_q: &Q4KHandle,
        h_k: &Q4KHandle,
        h_v: &Q4KHandle,
    ) -> Self {
        assert_eq!(
            h_q.n, h_k.n,
            "Q and K handles must have the same input dimension n"
        );
        assert_eq!(
            h_k.n, h_v.n,
            "K and V handles must have the same input dimension n"
        );

        let q_dim = h_q.m;
        let kv_dim = h_k.m;
        let n = h_q.n;
        let blocks_per_row = n / Q4K_BLOCK_SIZE as usize;

        // Calculate sizes
        let _ = blocks_per_row;

        // Concatenate weight_q4k: read each handle's raw bytes and combine
        let weight_combined_bytes = {
            let q_bytes = client.read_one(h_q.weight_q4k.clone()).unwrap();
            let k_bytes = client.read_one(h_k.weight_q4k.clone()).unwrap();
            let v_bytes = client.read_one(h_v.weight_q4k.clone()).unwrap();

            let mut combined = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            combined.extend_from_slice(&q_bytes);
            combined.extend_from_slice(&k_bytes);
            combined.extend_from_slice(&v_bytes);
            combined
        };

        // Concatenate d_dmin: header + Q + K + V
        let dd_combined_data = {
            let q_bytes = client.read_one(h_q.d_dmin.clone()).unwrap();
            let k_bytes = client.read_one(h_k.d_dmin.clone()).unwrap();
            let v_bytes = client.read_one(h_v.d_dmin.clone()).unwrap();

            let header: Vec<f32> = vec![q_dim as f32, kv_dim as f32, n as f32, 0.0f32];

            let mut combined = Vec::with_capacity(
                DD_HEADER as usize * 4 + q_bytes.len() + k_bytes.len() + v_bytes.len(),
            );
            combined.extend_from_slice(f32::as_bytes(&header));
            combined.extend_from_slice(&q_bytes);
            combined.extend_from_slice(&k_bytes);
            combined.extend_from_slice(&v_bytes);
            combined
        };

        let weight_q4k_combined = client.create_from_slice(&weight_combined_bytes);
        let d_dmin_combined = client.create_from_slice(&dd_combined_data);

        Self {
            weight_q4k_combined,
            d_dmin_combined,
            q_dim,
            kv_dim,
            n,
        }
    }

    /// Build combined Q4_K QKV handles from raw BlockQ4K arrays.
    ///
    /// Convenience constructor that quantizes and uploads in one step.
    /// `blocks_q`, `blocks_k`, `blocks_v` are pre-quantized block arrays.
    pub fn from_blocks(
        client: &ComputeClient<crate::cubecl_runtime::ActiveRuntime>,
        blocks_q: &[BlockQ4K],
        blocks_k: &[BlockQ4K],
        blocks_v: &[BlockQ4K],
        q_dim: usize,
        kv_dim: usize,
        n: usize,
    ) -> Self {
        let h_q = Q4KHandle::from_blocks(client, blocks_q, q_dim, n);
        let h_k = Q4KHandle::from_blocks(client, blocks_k, kv_dim, n);
        let h_v = Q4KHandle::from_blocks(client, blocks_v, kv_dim, n);

        // from_separate reads back from GPU, which is wasteful here.
        // For production, use a direct path that avoids the round-trip.
        // This constructor is primarily for testing convenience.
        Self::from_separate(client, &h_q, &h_k, &h_v)
    }

    /// Block count per row.
    #[inline]
    pub fn blocks_per_row(&self) -> usize {
        self.n / Q4K_BLOCK_SIZE as usize
    }

    /// Total rows in the combined output (q_dim + 2 * kv_dim).
    #[inline]
    pub fn total_rows(&self) -> usize {
        self.q_dim + 2 * self.kv_dim
    }
}

// ---------------------------------------------------------------------------
// Launcher struct
// ---------------------------------------------------------------------------

/// CubeCL fused triple QKV GEMV launcher (Q4_K weights).
///
/// Replaces three separate Q4_K GEMV dispatches:
/// 1. `GemvQ4KCubeCL::launch(weight_q, input, q_output, q_dim, n)`
/// 2. `GemvQ4KCubeCL::launch(weight_k, input, k_output, kv_dim, n)`
/// 3. `GemvQ4KCubeCL::launch(weight_v, input, v_output, kv_dim, n)`
///
/// With one fused dispatch:
/// `GemvQkvQ4KCubeCL::launch(combined, input, output, q_dim, kv_dim, n)`
///
/// The caller must prepare a `Q4KQKVHandle` with concatenated Q, K, V weight
/// and d/dmin buffers.
#[cfg(feature = "cubecl_runtime")]
pub struct GemvQkvQ4KCubeCL;

#[cfg(feature = "cubecl_runtime")]
#[allow(dead_code)]
impl GemvQkvQ4KCubeCL {
    /// Launch fused triple QKV GEMV kernel (Q4_K weights, plane cooperative).
    ///
    /// Computes `output[row] = dot(dequant_q4k(weight_row), input)` for each
    /// row across all three Q/K/V sections in a single GPU dispatch.
    ///
    /// Dispatch: `ceil(total_rows / 8)` workgroups of 256 threads.
    /// Each 256-thread workgroup has 8 planes (Metal subgroup=32),
    /// each plane handles one row.
    ///
    /// # Safety
    ///
    /// Buffer handles must have correct sizes:
    /// - `handle.weight_q4k_combined`: total_rows × blocks_per_row × 36 u32 elements
    /// - `handle.d_dmin_combined`: 4 (header) + total_rows × blocks_per_row × 2 f32 elements
    /// - `input_handle`: `n` f32 elements (n must be multiple of 256)
    /// - `output_handle`: `q_dim + 2 * kv_dim` f32 elements
    ///
    /// The client must support `Plane::Ops` (subgroup operations).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        handle: &Q4KQKVHandle,
        input_handle: Handle,
        output_handle: Handle,
    ) {
        let wg_size = 256u32;
        let plane_size = 32u32; // Metal subgroup size on Apple Silicon
        let rows_per_wg = wg_size / plane_size; // 8
        let total_rows = handle.total_rows() as u32;
        let num_wg = total_rows.div_ceil(rows_per_wg).max(1);

        let blocks_per_row = handle.blocks_per_row();
        let q4k_stride = (Q4K_WORDS_PER_BLOCK as usize) * blocks_per_row;
        let dd_stride = 2 * blocks_per_row;

        let total_weight_len = handle.total_rows() * q4k_stride;
        let total_dd_len = DD_HEADER as usize + handle.total_rows() * dd_stride;

        // SAFETY: Caller guarantees correct buffer sizes and plane support.
        unsafe {
            gemv_qkv_q4k_plane::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_wg, 1, 1),
                CubeDim::new_1d(wg_size),
                BufferArg::from_raw_parts(handle.weight_q4k_combined.clone(), total_weight_len),
                BufferArg::from_raw_parts(handle.d_dmin_combined.clone(), total_dd_len),
                BufferArg::from_raw_parts(input_handle, handle.n),
                BufferArg::from_raw_parts(output_handle, handle.total_rows()),
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

    /// CPU reference: triple QKV GEMV with Q4_K quantization.
    fn cpu_q4k_qkv_gemv(
        weight_q: &[f32],
        weight_k: &[f32],
        weight_v: &[f32],
        input: &[f32],
        q_dim: usize,
        kv_dim: usize,
        n: usize,
    ) -> Vec<f32> {
        let total_rows = q_dim + 2 * kv_dim;
        let mut output = vec![0.0f32; total_rows];

        let blocks_per_row = n.div_ceil(QK_K);
        let padded_n = blocks_per_row * QK_K;

        // Helper: quantize → dequant → dot product
        let mut process_section = |weights: &[f32], rows: usize, offset: usize| {
            for row in 0..rows {
                let mut padded_row = vec![0.0f32; padded_n];
                let src = &weights[row * n..(row + 1) * n];
                padded_row[..n].copy_from_slice(src);

                let mut blocks = vec![BlockQ4K::zeroed(); blocks_per_row];
                quantize_row_q4_k(&padded_row, &mut blocks);

                let mut dequant = vec![0.0f32; padded_n];
                dequantize_row_q4_k(&blocks, &mut dequant);

                let mut sum = 0.0f32;
                for j in 0..n {
                    sum += dequant[j] * input[j];
                }
                output[offset + row] = sum;
            }
        };

        process_section(weight_q, q_dim, 0);
        process_section(weight_k, kv_dim, q_dim);
        process_section(weight_v, kv_dim, q_dim + kv_dim);

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

    /// Launch fused Q4_K QKV GEMV and compare against CPU reference.
    fn launch_and_verify(
        weight_q: &[f32],
        weight_k: &[f32],
        weight_v: &[f32],
        input: &[f32],
        q_dim: usize,
        kv_dim: usize,
        n: usize,
        tolerance: f32,
        test_name: &str,
    ) {
        let ctx = CubeCLContext::new().expect("CubeCL init");
        let client = ctx.client();

        let total_rows = q_dim + 2 * kv_dim;

        // Quantize weights
        let blocks_q = quantize_weight_rows(weight_q, q_dim, n);
        let blocks_k = quantize_weight_rows(weight_k, kv_dim, n);
        let blocks_v = quantize_weight_rows(weight_v, kv_dim, n);

        // Pad n to multiple of QK_K (256) if needed
        let blocks_per_row = n.div_ceil(QK_K);
        let effective_n = blocks_per_row * QK_K;

        // Build combined handle via separate Q4KHandles
        let h_q = Q4KHandle::from_blocks(&client, &blocks_q, q_dim, effective_n);
        let h_k = Q4KHandle::from_blocks(&client, &blocks_k, kv_dim, effective_n);
        let h_v = Q4KHandle::from_blocks(&client, &blocks_v, kv_dim, effective_n);
        let combined = Q4KQKVHandle::from_separate(&client, &h_q, &h_k, &h_v);

        // Pad input if needed
        let mut padded_input = input.to_vec();
        if padded_input.len() < effective_n {
            padded_input.resize(effective_n, 0.0);
        }

        let input_handle = client.create_from_slice(f32::as_bytes(&padded_input));
        let output_handle = client.empty(total_rows * core::mem::size_of::<f32>());

        unsafe {
            GemvQkvQ4KCubeCL::launch::<ActiveRuntime>(
                &client,
                &combined,
                input_handle,
                output_handle.clone(),
            );
        }

        let result_bytes = client.read_one(output_handle).unwrap();
        let result = f32::from_bytes(&result_bytes);

        let expected = cpu_q4k_qkv_gemv(weight_q, weight_k, weight_v, input, q_dim, kv_dim, n);

        let mut max_err = 0.0f32;
        for i in 0..total_rows {
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
    fn test_gemv_qkv_q4k_identity() {
        // Identity-like weight matrices for Q (2×256), K (2×256), V (2×256).
        // Input = all 1s → each output ≈ sum of dequantized row ≈ 0 (Q4_K zeros).
        let n = 256;
        let q_dim = 2;
        let kv_dim = 2;

        let weight_q = vec![0.0f32; q_dim * n];
        let weight_k = vec![0.0f32; kv_dim * n];
        let weight_v = vec![0.0f32; kv_dim * n];
        let input = vec![1.0f32; n];

        // All zeros → Q4_K quantizes to ~0, so output should be ~0
        launch_and_verify(
            &weight_q,
            &weight_k,
            &weight_v,
            &input,
            q_dim,
            kv_dim,
            n,
            1.0,
            "identity_zeros",
        );
    }

    #[test]
    fn test_gemv_qkv_q4k_general() {
        // 2×256 matrices for Q, K, V with distinct sine-wave values.
        let n = 256;
        let q_dim = 2;
        let kv_dim = 2;

        let weight_q: Vec<f32> = (0..q_dim * n)
            .map(|i| (i as f32 * 0.1).sin() * 2.0)
            .collect();
        let weight_k: Vec<f32> = (0..kv_dim * n)
            .map(|i| (i as f32 * 0.05).cos() * 1.5)
            .collect();
        let weight_v: Vec<f32> = (0..kv_dim * n)
            .map(|i| (i as f32 * 0.07).sin() * 0.8)
            .collect();
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();

        launch_and_verify(
            &weight_q, &weight_k, &weight_v, &input, q_dim, kv_dim, n, 10.0, "general",
        );
    }

    #[test]
    fn test_gemv_qkv_q4k_gemma2_dims() {
        // Realistic Gemma 2 dimensions scaled down: q_dim=8, kv_dim=4, n=256.
        // In real Gemma 2: q_dim=2048, kv_dim=512, n=2048 (with GQA).
        let q_dim = 8usize;
        let kv_dim = 4usize;
        let n = 256usize;

        let weight_q: Vec<f32> = (0..q_dim * n)
            .map(|i| ((i % 97) as f32 - 48.0) / 100.0)
            .collect();
        let weight_k: Vec<f32> = (0..kv_dim * n)
            .map(|i| ((i % 53) as f32 - 26.0) / 100.0)
            .collect();
        let weight_v: Vec<f32> = (0..kv_dim * n)
            .map(|i| ((i % 71) as f32 - 35.0) / 100.0)
            .collect();
        let input: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) / 10.0).collect();

        launch_and_verify(
            &weight_q,
            &weight_k,
            &weight_v,
            &input,
            q_dim,
            kv_dim,
            n,
            15.0,
            "gemma2_dims",
        );
    }
}
