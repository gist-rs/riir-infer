//! `Q8_0` (8-bit symmetric) quantization format for KV cache compression.
//!
//! Ported from llama.cpp's `ggml-quants.c` and `ggml-common.h`.
//! Reference: <https://github.com/ggml-org/llama.cpp>
//!
//! # Block Layout (34 bytes for 32 elements = 8.5 bpw)
//!
//! ```text
//! Offset  Size   Field
//! 0       2      d      — f16 block scale (symmetric, no offset)
//! 2       32     qs     — 32 × 8-bit signed quantized values
//! ```
//!
//! Dequant formula: `value = d * qs[i]`

use half::f16;

/// Block size: 32 elements per block.
pub const Q8_BLOCK_SIZE: usize = 32;

/// Size of one `Q8_0` block in bytes (2 + 32 = 34).
pub const BLOCK_BYTES: usize = 34;

/// A `Q8_0` quantization block.
///
/// Represents 32 weight values in 34 bytes (8.5 bits per weight).
/// Layout matches llama.cpp's `block_q8_0` for GPU upload compatibility.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ8_0 {
    /// Block scale as f16 bits (symmetric quantization, no offset).
    pub d: u16,
    /// 32 × 8-bit signed quantized values (clamped to [-127, 127]).
    pub qs: [i8; Q8_BLOCK_SIZE],
}

// Static assert: BlockQ8_0 must be exactly 34 bytes.
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == BLOCK_BYTES);

// ── Quantize ────────────────────────────────────────────────────

/// Quantize a row of f32 values to `Q8_0` blocks.
///
/// `src` length must be a multiple of `Q8_BLOCK_SIZE` (32).
/// `dst` length must be `src.len() / Q8_BLOCK_SIZE`.
pub fn quantize_row_q8_0(src: &[f32], dst: &mut [BlockQ8_0]) {
    assert!(
        src.len().is_multiple_of(Q8_BLOCK_SIZE),
        "src length must be multiple of {Q8_BLOCK_SIZE}, got {}",
        src.len()
    );
    let nb = src.len() / Q8_BLOCK_SIZE;
    assert!(
        dst.len() >= nb,
        "dst too short: need {nb} blocks, got {}",
        dst.len()
    );

    for i in 0..nb {
        quantize_block(
            &src[i * Q8_BLOCK_SIZE..(i + 1) * Q8_BLOCK_SIZE],
            &mut dst[i],
        );
    }
}

/// Quantize one block of 32 f32 values using symmetric quantization.
///
/// Scale `d = max_abs / 127.0`, each value quantized as `round(v / d)`
/// clamped to [-127, 127] (avoids -128 for symmetry).
fn quantize_block(src: &[f32], block: &mut BlockQ8_0) {
    debug_assert_eq!(src.len(), Q8_BLOCK_SIZE);

    // Find max absolute value — 4 independent accumulators break the
    // loop-carried f32::max dependency (non-associative → not auto-vectorizable).
    // Out-of-order execution overlaps the 4 independent reduction chains.
    let mut m0 = 0.0f32;
    let mut m1 = 0.0f32;
    let mut m2 = 0.0f32;
    let mut m3 = 0.0f32;
    let chunks = Q8_BLOCK_SIZE / 4;
    for c in 0..chunks {
        let i = c * 4;
        m0 = m0.max(src[i].abs());
        m1 = m1.max(src[i + 1].abs());
        m2 = m2.max(src[i + 2].abs());
        m3 = m3.max(src[i + 3].abs());
    }
    let max_abs = m0.max(m1).max(m2).max(m3);

    // Compute scale: d = max_abs / 127.0
    let d = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
    block.d = f16::from_f32(d).to_bits();

    if d == 0.0 {
        // All zeros — skip quantization loop
        block.qs.fill(0);
        return;
    }

    // Pre-compute inverse to replace division with multiplication in hot loop
    let inv_d = 1.0 / d;
    // 4-wide chunked quantize: independent writes per lane help auto-vectorize
    // the f32→i32 cast + clamp + i8 store pipeline.
    for c in 0..chunks {
        let i = c * 4;
        let q0 = (src[i] * inv_d).round() as i32;
        let q1 = (src[i + 1] * inv_d).round() as i32;
        let q2 = (src[i + 2] * inv_d).round() as i32;
        let q3 = (src[i + 3] * inv_d).round() as i32;
        // Clamp to [-127, 127] to avoid -128 (symmetric range)
        block.qs[i] = q0.clamp(-127, 127) as i8;
        block.qs[i + 1] = q1.clamp(-127, 127) as i8;
        block.qs[i + 2] = q2.clamp(-127, 127) as i8;
        block.qs[i + 3] = q3.clamp(-127, 127) as i8;
    }
}

// ── Dequantize ──────────────────────────────────────────────────

/// Dequantize `Q8_0` blocks to f32 values.
///
/// `src` contains the quantized blocks, `dst` must have length `n`
/// where `n` is a multiple of `Q8_BLOCK_SIZE` and `src.len() >= n / Q8_BLOCK_SIZE`.
///
/// Ported from llama.cpp's `dequantize_row_q8_0`.
pub fn dequantize_row_q8_0(src: &[BlockQ8_0], dst: &mut [f32]) {
    let n = dst.len();
    assert!(
        n.is_multiple_of(Q8_BLOCK_SIZE),
        "dst length must be multiple of {Q8_BLOCK_SIZE}"
    );
    let nb = n / Q8_BLOCK_SIZE;
    assert!(
        src.len() >= nb,
        "src too short: need {nb} blocks, got {}",
        src.len()
    );

    for (i, block) in src.iter().take(nb).enumerate() {
        let d = f32::from(f16::from_bits(block.d));
        let out_base = i * Q8_BLOCK_SIZE;

        // Process 4 elements at a time to help LLVM auto-vectorize the broadcast multiply
        let qs = &block.qs;
        let chunks = Q8_BLOCK_SIZE / 4;
        for c in 0..chunks {
            let j = c * 4;
            dst[out_base + j] = d * qs[j] as f32;
            dst[out_base + j + 1] = d * qs[j + 1] as f32;
            dst[out_base + j + 2] = d * qs[j + 2] as f32;
            dst[out_base + j + 3] = d * qs[j + 3] as f32;
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn test_block_q8_0_size() {
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), BLOCK_BYTES);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
        assert_eq!(std::mem::align_of::<BlockQ8_0>(), 2); // u16 alignment
    }

    #[test]
    fn test_quantize_dequantize_zeros() {
        let src = vec![0.0f32; Q8_BLOCK_SIZE];
        let mut blocks = [BlockQ8_0::zeroed(); 1];
        quantize_row_q8_0(&src, &mut blocks);

        let mut dst = vec![0.0f32; Q8_BLOCK_SIZE];
        dequantize_row_q8_0(&blocks, &mut dst);

        for (i, &v) in dst.iter().enumerate() {
            assert!(
                v.abs() < f32::EPSILON,
                "Zero roundtrip failed at index {i}: {v}"
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_uniform() {
        let src = vec![1.5f32; Q8_BLOCK_SIZE];
        let mut blocks = [BlockQ8_0::zeroed(); 1];
        quantize_row_q8_0(&src, &mut blocks);

        let mut dst = vec![0.0f32; Q8_BLOCK_SIZE];
        dequantize_row_q8_0(&blocks, &mut dst);

        for (i, &v) in dst.iter().enumerate() {
            let err = (v - 1.5).abs();
            assert!(
                err < 1.0 / 127.0,
                "Uniform roundtrip error too large at index {i}: {v} (error {err})"
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_random() {
        // Deterministic pseudo-random values in [-10, 10]
        let src: Vec<f32> = (0..Q8_BLOCK_SIZE * 4)
            .map(|i| {
                let x = ((i as f32 * 12.989_8 + 78.233).sin() * 43758.545).fract();
                (x - 0.5) * 20.0
            })
            .collect();

        let nb = src.len() / Q8_BLOCK_SIZE;
        let mut blocks = vec![BlockQ8_0::zeroed(); nb];
        quantize_row_q8_0(&src, &mut blocks);

        let mut dst = vec![0.0f32; src.len()];
        dequantize_row_q8_0(&blocks, &mut dst);

        let mut max_err = 0.0f32;
        let mut mse = 0.0f32;
        for (&orig, &deq) in src.iter().zip(dst.iter()) {
            let err = (orig - deq).abs();
            if err > max_err {
                max_err = err;
            }
            mse += (orig - deq) * (orig - deq);
        }
        let rmse = (mse / src.len() as f32).sqrt();

        // Q8_0 max quantization error should be < 1/127 of the value range
        let tolerance = 1.0 / 127.0 * 20.0; // scale factor × 1/127
        assert!(
            max_err < tolerance,
            "Max error too large: {max_err} (tolerance {tolerance})"
        );
        assert!(
            rmse < tolerance * 0.5,
            "RMSE too large: {rmse} (tolerance {})",
            tolerance * 0.5
        );
    }

    #[test]
    fn test_quantize_clamps_to_symmetric_range() {
        // Values that would exceed i8 range without clamping
        let src: Vec<f32> = (0..Q8_BLOCK_SIZE)
            .map(|i| if i < 16 { 500.0 } else { -500.0 })
            .collect();

        let mut blocks = [BlockQ8_0::zeroed(); 1];
        quantize_row_q8_0(&src, &mut blocks);

        // All quantized values should be >= -127 (upper bound is implicit: i8 max = 127)
        for (i, &q) in blocks[0].qs.iter().enumerate() {
            assert!(
                q >= -127,
                "Quantized value at index {i} is {q}, expected >= -127"
            );
        }
    }

    #[test]
    fn test_dequantize_matches_formula() {
        // Verify dequantize output matches d * qs[i] formula
        let src: Vec<f32> = (0..Q8_BLOCK_SIZE)
            .map(|i| (i as f32 * 0.7 - 10.0).sin() * 3.0)
            .collect();

        let mut blocks = [BlockQ8_0::zeroed(); 1];
        quantize_row_q8_0(&src, &mut blocks);
        let block = &blocks[0];

        // Manually dequantize using the formula
        let d = f32::from(f16::from_bits(block.d));
        let mut expected = [0.0f32; Q8_BLOCK_SIZE];
        for (i, &q) in block.qs.iter().enumerate() {
            expected[i] = d * q as f32;
        }

        // Verify dequantize_row_q8_0 matches manual result
        let mut dst = vec![0.0f32; Q8_BLOCK_SIZE];
        dequantize_row_q8_0(&blocks, &mut dst);

        for (i, (&a, &b)) in expected.iter().zip(dst.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "Manual vs function mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_compression_ratio() {
        // 32 f32 values = 128 bytes → 1 block = 34 bytes
        let src = vec![1.0f32; Q8_BLOCK_SIZE];
        let mut blocks = [BlockQ8_0::zeroed(); 1];
        quantize_row_q8_0(&src, &mut blocks);

        let original_bytes = src.len() * 4; // 128
        let compressed_bytes = std::mem::size_of::<BlockQ8_0>(); // 34
        let ratio = original_bytes as f64 / compressed_bytes as f64;

        assert!(
            ratio > 3.7,
            "Compression ratio too low: {ratio:.1}x (expected ~3.76x)"
        );
    }

    #[test]
    fn test_multi_block_roundtrip() {
        let total_elements = Q8_BLOCK_SIZE * 8;
        // Mixed pattern: sine wave with varying amplitude per block
        let src: Vec<f32> = (0..total_elements)
            .map(|i| {
                let block_idx = i / Q8_BLOCK_SIZE;
                let amplitude = (block_idx + 1) as f32;
                (i as f32 / 8.0).sin() * amplitude
            })
            .collect();

        let nb = total_elements / Q8_BLOCK_SIZE;
        let mut blocks = vec![BlockQ8_0::zeroed(); nb];
        quantize_row_q8_0(&src, &mut blocks);

        let mut dst = vec![0.0f32; total_elements];
        dequantize_row_q8_0(&blocks, &mut dst);

        let mut max_err = 0.0f32;
        for (&orig, &deq) in src.iter().zip(dst.iter()) {
            let err = (orig - deq).abs();
            if err > max_err {
                max_err = err;
            }
        }

        // Q8_0 at 8.5 bpw should have very low error
        assert!(
            max_err < 1.0 / 127.0 * 8.0 + f32::EPSILON,
            "Multi-block max error too large: {max_err}"
        );
    }
}
