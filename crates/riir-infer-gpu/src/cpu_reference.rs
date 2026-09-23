//! CPU reference implementations for GPU shader validation.
//!
//! Provides numerically-correct CPU implementations of algorithms that run on GPU
//! via WGSL compute shaders. Used for:
//! - GOAT proof testing (GPU must match CPU within tolerance)
//! - Kernel correctness tests
//! - Debugging GPU shader bugs
//!
//! Each function mirrors the corresponding WGSL shader's algorithm exactly,
//! including edge-case handling (out-of-bounds → zero contribution).
//! Accumulation precision (riir-ai Issue 709 H2): all reference fns accumulate
//! in **f32**, matching the kernels — over long rows, sequential f32
//! accumulation rounds at ~1e-4 relative, so parity tolerances tighter than
//! that are not supported by this reference. (The only f64 accumulation is
//! `compare_f32`'s mean.)
//!
//! # Shader mapping
//!
//! | Function | WGSL shader |
//! |----------|------------|
//! | `gemv` | `kernels/gemv.wgsl` |
//! | `softmax` | `kernels/softmax.wgsl` |
//! | `rmsnorm` | `kernels/rmsnorm_gamma_batch.wgsl` |
//! | `lora_forward` | `kernels/lora_a.wgsl` + `kernels/lora_b.wgsl` |
//!
//! # Comparison utility
//!
//! `compare_f32` provides element-wise comparison returning max absolute error,
//! mean absolute error, and a pass/fail boolean. Use it in tests:
//!
//! ```ignore
//! let (max_err, mean_err, pass) = cpu_reference::compare_f32(&cpu, &gpu, 1e-4);
//! assert!(pass, "max_err={max_err}, mean_err={mean_err}");
//! ```

/// CPU reference for `gemv.wgsl`: `output[M] = weight[M,N] @ input[N]`.
///
/// Weight is row-major: `weight[i * n + j]` is element (i, j).
/// Each output element is the dot product of one weight row with the input vector.
pub fn gemv(weight: &[f32], input: &[f32], m: usize, n: usize) -> Vec<f32> {
    assert_eq!(weight.len(), m * n, "weight must be [M×N]");
    assert_eq!(input.len(), n, "input must be [N]");

    let mut output = vec![0.0f32; m];
    (0..m).for_each(|i| {
        let row_off = i * n;
        let mut sum: f32 = 0.0;
        for j in 0..n {
            sum += weight[row_off + j] * input[j];
        }
        output[i] = sum;
    });
    output
}

/// CPU reference for `softmax.wgsl`: stable two-pass softmax (in-place on row).
///
/// Applies per-row:
/// 1. Find max value in the row
/// 2. Subtract max, exponentiate, accumulate sum
/// 3. Normalize by sum
///
/// `data` is a row-major `[rows × cols]` array.
pub fn softmax(data: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(data.len(), rows * cols, "data must be [rows × cols]");

    for r in 0..rows {
        let offset = r * cols;

        // Pass 1: find max
        let mut max_val = data[offset];
        for c in 1..cols {
            if data[offset + c] > max_val {
                max_val = data[offset + c];
            }
        }

        // Pass 2: exp, sum
        let mut sum: f32 = 0.0;
        for c in 0..cols {
            let exp_val = (data[offset + c] - max_val).exp();
            data[offset + c] = exp_val;
            sum += exp_val;
        }

        // Pass 3: normalize
        let inv_sum = 1.0 / sum;
        for c in 0..cols {
            data[offset + c] *= inv_sum;
        }
    }
}

/// CPU reference for `rmsnorm_gamma_batch.wgsl`: RMS normalization.
///
/// Applies per vector of length `dim`:
/// ```text
/// output[i] = input[i] * rsqrt(mean(input²) + eps) * gamma[i]
/// ```
///
/// `input` is `[batch * dim]` row-major, `gamma` is `[dim]`.
/// Returns a new vector (does not modify input in-place).
pub fn rmsnorm(input: &[f32], gamma: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let batch = input.len() / dim;
    assert_eq!(input.len(), batch * dim, "input must be [batch × dim]");
    assert_eq!(gamma.len(), dim, "gamma must be [dim]");

    let mut output = vec![0.0f32; input.len()];
    for b in 0..batch {
        let offset = b * dim;

        // Pass 1: sum of squares
        let mut sum_sq: f32 = 0.0;
        for d in 0..dim {
            let v = input[offset + d];
            sum_sq += v * v;
        }

        // Pass 2: normalize and apply gamma
        let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        for d in 0..dim {
            output[offset + d] = input[offset + d] * inv_rms * gamma[d];
        }
    }
    output
}

/// CPU reference for `lora_a.wgsl` + `lora_b.wgsl`: LoRA forward pass.
///
/// Computes the LoRA contribution:
/// ```text
/// output = (alpha / rank) * B @ (A @ input)
/// ```
///
/// - `input`: `[in_dim]`
/// - `a`: `[rank × in_dim]` row-major LoRA A matrix
/// - `b`: `[out_dim × rank]` row-major LoRA B matrix
///
/// Returns `[out_dim]` — the LoRA delta only (without base weight contribution).
/// This matches the two-dispatch GPU pattern:
/// 1. Dispatch 1 (`lora_a.wgsl`): `intermediate[r] = A[r,:] @ input`
/// 2. Dispatch 2 (`lora_b.wgsl`): `output[i] = base_sum + alpha * B[i,:] @ intermediate`
///
/// **Parity trap (riir-ai Issue 709 H3):** `lora_b.wgsl` INCLUDES the base
/// weight GEMV (it binds `base_weight` at binding 0 and adds `base_sum`) — its
/// output is `W₀·input + delta`, NOT the delta alone. For a GOAT parity test
/// against the GPU path, compare against `gemv(input, base_weight) +
/// lora_forward(...)` (or use a zero base weight); this fn alone differs from
/// the kernel output by the entire base term.
pub fn lora_forward(
    input: &[f32],
    a: &[f32],
    b: &[f32],
    alpha: f32,
    rank: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), in_dim, "input must be [in_dim]");
    assert_eq!(a.len(), rank * in_dim, "a must be [rank × in_dim]");
    assert_eq!(b.len(), out_dim * rank, "b must be [out_dim × rank]");

    // Dispatch 1: intermediate = A @ input  →  [rank]
    let mut intermediate = vec![0.0f32; rank];
    for r in 0..rank {
        let mut sum: f32 = 0.0;
        for j in 0..in_dim {
            sum += a[r * in_dim + j] * input[j];
        }
        intermediate[r] = sum;
    }

    // Dispatch 2: output = alpha * B @ intermediate  →  [out_dim]
    let mut output = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut sum: f32 = 0.0;
        for r in 0..rank {
            sum += b[i * rank + r] * intermediate[r];
        }
        output[i] = alpha * sum;
    }

    output
}

/// Compare two f32 arrays within tolerance.
///
/// Returns `(max_abs_error, mean_abs_error, pass)` where:
/// - `max_abs_error`: largest element-wise absolute difference
/// - `mean_abs_error`: average element-wise absolute difference
/// - `pass`: `true` if `max_abs_error <= tolerance`
///
/// **Contract note (riir-ai Issue 709 H10): this PANICS on the first element
/// exceeding tolerance** — the returned `pass` can therefore never be `false`
/// on return, and only the first mismatch is reported. Callers that want a
/// non-panicking verdict must pre-check or catch; a full error-profile variant
/// does not exist yet.
///
/// Panics if the arrays have different lengths.
pub fn compare_f32(cpu: &[f32], gpu: &[f32], tolerance: f32) -> (f32, f64, bool) {
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "length mismatch: cpu={}, gpu={}",
        cpu.len(),
        gpu.len()
    );

    let mut max_abs: f32 = 0.0;
    let mut sum_abs: f64 = 0.0;

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        sum_abs += diff as f64;

        if diff > tolerance {
            // Report first mismatch with full detail
            panic!(
                "mismatch at [{i}]: cpu={c:+.8}, gpu={g:+.8}, diff={diff:.8} > tolerance {tolerance}"
            );
        }
    }

    let mean_abs = if cpu.is_empty() {
        0.0
    } else {
        sum_abs / cpu.len() as f64
    };
    let pass = max_abs <= tolerance;

    (max_abs, mean_abs, pass)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemv_identity() {
        let n = 4;
        let m = 4;
        // 4×4 identity matrix
        let weight: Vec<f32> = (0..m)
            .flat_map(|i| (0..n).map(move |j| if i == j { 1.0f32 } else { 0.0f32 }))
            .collect();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = gemv(&weight, &input, m, n);
        assert_eq!(output, input);
    }

    #[test]
    fn test_gemv_rectangular() {
        // 2×3 weight @ 3×1 input
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let input = vec![1.0, 1.0, 1.0];
        let output = gemv(&weight, &input, 2, 3);
        assert_eq!(output.len(), 2);
        assert!(
            (output[0] - 6.0).abs() < 1e-6,
            "row 0: expected 6, got {}",
            output[0]
        );
        assert!(
            (output[1] - 15.0).abs() < 1e-6,
            "row 1: expected 15, got {}",
            output[1]
        );
    }

    #[test]
    fn test_softmax_uniform() {
        let mut data = vec![1.0f32; 8];
        softmax(&mut data, 1, 8);
        for v in &data {
            assert!(
                (v - (1.0 / 8.0)).abs() < 1e-6,
                "uniform input should give uniform output"
            );
        }
    }

    #[test]
    fn test_softmax_peaked() {
        let mut data = vec![1.0f32, 2.0, 3.0, 100.0];
        softmax(&mut data, 1, 4);
        // 100 dominates
        assert!(
            data[3] > 0.99,
            "peaked value should dominate: got {}",
            data[3]
        );
        assert!(data[0] < 0.01);
        // Should sum to 1
        let sum: f32 = data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {sum}"
        );
    }

    #[test]
    fn test_softmax_multirow() {
        let mut data = vec![1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        softmax(&mut data, 2, 3);
        // Row 0: [1,2,3] → ~[0.0900, 0.2447, 0.6652]
        let sum0: f32 = data[0..3].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5, "row 0 should sum to 1");
        // Row 1: [10,20,30] → same distribution as [1,2,3]
        let sum1: f32 = data[3..6].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-5, "row 1 should sum to 1");
    }

    #[test]
    fn test_rmsnorm_ones() {
        let input = vec![1.0f32; 4];
        let gamma = vec![1.0f32; 4];
        let output = rmsnorm(&input, &gamma, 4, 1e-6);
        // rms = sqrt(4/4) = 1.0, so output = input / rms * gamma = [1,1,1,1]
        for v in &output {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_rmsnorm_known() {
        let input = vec![3.0f32, 4.0];
        let gamma = vec![1.0f32, 2.0];
        // mean(x²) = (9+16)/2 = 12.5, rms = sqrt(12.5) ≈ 3.5355
        // output[0] = 3/3.5355 * 1 ≈ 0.8485
        // output[1] = 4/3.5355 * 2 ≈ 2.2627
        let output = rmsnorm(&input, &gamma, 2, 1e-6);
        let expected_0 = 3.0 / (12.5f32 + 1e-6).sqrt() * 1.0;
        let expected_1 = 4.0 / (12.5f32 + 1e-6).sqrt() * 2.0;
        assert!((output[0] - expected_0).abs() < 1e-5);
        assert!((output[1] - expected_1).abs() < 1e-5);
    }

    #[test]
    fn test_lora_forward_identity() {
        let in_dim = 4;
        let rank = 2;
        let out_dim = 4;
        let input = vec![1.0f32, 2.0, 3.0, 4.0];

        // A: identity-like [rank × in_dim] — first two rows pick elements 0,1
        let a = vec![
            1.0, 0.0, 0.0, 0.0, // row 0: picks input[0]
            0.0, 1.0, 0.0, 0.0, // row 1: picks input[1]
        ];

        // B: identity-like [out_dim × rank] — maps back
        let b = vec![
            1.0, 0.0, // row 0: picks intermediate[0]
            0.0, 1.0, // row 1: picks intermediate[1]
            0.0, 0.0, // row 2: zero
            0.0, 0.0, // row 3: zero
        ];

        let alpha = 0.5;
        let output = lora_forward(&input, &a, &b, alpha, rank, in_dim, out_dim);

        // intermediate = [1.0, 2.0], output = 0.5 * [1.0, 2.0, 0.0, 0.0]
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
        assert!((output[2]).abs() < 1e-6);
        assert!((output[3]).abs() < 1e-6);
    }

    #[test]
    fn test_compare_f32_exact() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0, 3.0];
        let (max_err, mean_err, pass) = compare_f32(&a, &b, 1e-4);
        assert!(pass);
        assert_eq!(max_err, 0.0);
        assert_eq!(mean_err, 0.0);
    }

    #[test]
    fn test_compare_f32_within_tolerance() {
        let a = vec![1.0f32];
        let b = vec![1.0f32 + 5e-5];
        let (max_err, _, pass) = compare_f32(&a, &b, 1e-4);
        assert!(pass);
        assert!(max_err < 1e-4);
    }

    #[test]
    #[should_panic(expected = "mismatch at [0]")]
    fn test_compare_f32_exceeds_tolerance() {
        let a = vec![1.0f32];
        let b = vec![2.0f32];
        compare_f32(&a, &b, 0.1);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_compare_f32_length_mismatch() {
        let a = vec![1.0f32];
        let b = vec![1.0f32, 2.0];
        compare_f32(&a, &b, 1e-4);
    }
}
