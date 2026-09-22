//! CPU reference implementation of `DeltaNet` linear recurrence.
//!
//! Provides a simple, correct CPU implementation of the Gated `DeltaNet` recurrence
//! for GOAT proof testing. The GPU kernel must match this reference within float tolerance.
//!
//! # Algorithm (Gated `DeltaNet` / Mamba-style)
//!
//! For each head h (`0..n_head`), the recurrence at each token position t is:
//!
//! ```text
//! β_t = exp(-softplus(A_log[h]))           // decay per head
//! S_t[h] = β_t * S_{t-1}[h] + v_t[h] ⊗ k_t[h]^T   // state update (outer product)
//! y_t[h] = S_t[h] * q_t[h]                 // state read (matvec)
//! ```
//!
//! The conv1d preprocessing and gating (`SwiGLU`) are applied before/after the recurrence.
//! This reference implements only the core recurrence for targeted verification.

/// Result of a single `DeltaNet` recurrence step (one token position).
pub struct RecurrenceOutput {
    /// Output after state read: [`n_head` * `head_dim`]
    pub output: Vec<f32>,
    /// Updated recurrent state: [`n_head` * `head_dim` * `head_dim`]
    pub state: Vec<f32>,
}

/// Run one step of the `DeltaNet` linear recurrence on CPU.
///
/// This is the **reference implementation** for GOAT proof T9.
/// The GPU kernel `deltanet_recurrence_f32` must produce output matching this
/// within float tolerance (|cpu - gpu| < 1e-3 per element).
///
/// # Arguments
///
/// * `q` - Query vectors: [`n_head` * `head_dim`]
/// * `k` - Key vectors: [`n_head` * `head_dim`]
/// * `v` - Value vectors: [`n_head` * `head_dim`]
/// * `state` - Current recurrent state: [`n_head` * `head_dim` * `head_dim`] (modified in-place)
/// * `n_head` - Number of recurrence heads
/// * `head_dim` - Dimension per head (128 for Qwen 3.5)
/// * `beta` - Decay factor (typically exp(-softplus(A_log)), ~0.99)
///
/// # Returns
///
/// `RecurrenceOutput` with updated state and output vector.
pub fn deltanet_recurrence_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    state: &mut [f32],
    n_head: usize,
    head_dim: usize,
    beta: f32,
) -> RecurrenceOutput {
    let state_dim_per_head = head_dim * head_dim;
    let mut output = vec![0.0f32; n_head * head_dim];

    for h in 0..n_head {
        let state_offset = h * state_dim_per_head;
        let head_offset = h * head_dim;

        // State update: S[h] = beta * S[h] + v[h] ⊗ k[h]^T
        // S[row * head_dim + col] = beta * S[row * head_dim + col] + v[row] * k[col]
        for row in 0..head_dim {
            let v_r = v[head_offset + row];
            for col in 0..head_dim {
                let idx = state_offset + row * head_dim + col;
                state[idx] = beta * state[idx] + v_r * k[head_offset + col];
            }
        }

        // State read: output[h, r] = Σ_c S[h, r, c] * q[h, c]
        for row in 0..head_dim {
            let mut dot = 0.0f32;
            for col in 0..head_dim {
                dot += state[state_offset + row * head_dim + col] * q[head_offset + col];
            }
            output[head_offset + row] = dot;
        }
    }

    RecurrenceOutput {
        output,
        state: state.to_vec(),
    }
}

/// Run multi-step `DeltaNet` recurrence (for prefill / sequence processing).
///
/// Processes a sequence of tokens, updating state at each position.
/// Returns outputs for all positions: [`seq_len` * `n_head` * `head_dim`].
#[allow(clippy::too_many_arguments)]
pub fn deltanet_recurrence_prefill(
    queries: &[f32],   // [seq_len * n_head * head_dim]
    keys: &[f32],      // [seq_len * n_head * head_dim]
    values: &[f32],    // [seq_len * n_head * head_dim]
    state: &mut [f32], // [n_head * head_dim * head_dim]
    n_head: usize,
    head_dim: usize,
    beta: f32,
    seq_len: usize,
) -> Vec<f32> {
    let head_dim_total = n_head * head_dim;
    let mut all_outputs = Vec::with_capacity(seq_len * head_dim_total);

    for t in 0..seq_len {
        let offset = t * head_dim_total;
        let result = deltanet_recurrence_reference(
            &queries[offset..offset + head_dim_total],
            &keys[offset..offset + head_dim_total],
            &values[offset..offset + head_dim_total],
            state,
            n_head,
            head_dim,
            beta,
        );
        all_outputs.extend_from_slice(&result.output);
        // state is already updated in-place by the reference
        state.copy_from_slice(&result.state);
    }

    all_outputs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GOAT proof T9: single-step recurrence produces correct output.
    ///
    /// With zero initial state, q=k=v=1.0, beta=1.0:
    ///   S[row,col] = 0 + 1.0 * 1.0 = 1.0 for all (row,col)
    ///   output[row] = `Σ_col` S[row,col] * q[col] = `head_dim` * 1.0 = 128.0
    #[test]
    fn test_recurrence_zero_state_identity() {
        let n_head = 4;
        let head_dim = 8;

        let q = vec![1.0f32; n_head * head_dim];
        let k = vec![1.0f32; n_head * head_dim];
        let v = vec![1.0f32; n_head * head_dim];
        let mut state = vec![0.0f32; n_head * head_dim * head_dim];

        let result = deltanet_recurrence_reference(&q, &k, &v, &mut state, n_head, head_dim, 1.0);

        // After first step with beta=1.0, q=k=v=1.0:
        // S = I-like outer product: S[row,col] = 1.0
        // output = Σ_col 1.0 * 1.0 = head_dim = 8.0
        for h in 0..n_head {
            for r in 0..head_dim {
                assert!(
                    (result.output[h * head_dim + r] - head_dim as f32).abs() < 1e-4,
                    "output[head={h}, row={r}] = {}, expected {}",
                    result.output[h * head_dim + r],
                    head_dim,
                );
            }
        }

        // State should be all 1.0 (v[row] * k[col] = 1.0 * 1.0)
        for h in 0..n_head {
            for r in 0..head_dim {
                for c in 0..head_dim {
                    assert!(
                        (result.state[h * head_dim * head_dim + r * head_dim + c] - 1.0).abs()
                            < 1e-4,
                        "state[head={h}, row={r}, col={c}] should be 1.0"
                    );
                }
            }
        }
    }

    /// GOAT proof T9: decay factor correctly attenuates previous state.
    ///
    /// After 2 steps with beta=0.5:
    ///   Step 1: S = 0 + v⊗k^T = v*k^T
    ///   Step 2: S = 0.5 * v⊗k^T + v⊗k^T = 1.5 * v⊗k^T
    #[test]
    fn test_recurrence_decay() {
        let n_head = 2;
        let head_dim = 4;
        let beta = 0.5f32;

        let q = vec![1.0f32; n_head * head_dim];
        let k = vec![1.0f32; n_head * head_dim];
        let v = vec![1.0f32; n_head * head_dim];
        let mut state = vec![0.0f32; n_head * head_dim * head_dim];

        // Step 1
        let _step1 = deltanet_recurrence_reference(&q, &k, &v, &mut state, n_head, head_dim, beta);
        // State: 0 + 1.0*1.0 = 1.0 everywhere

        // Step 2: state = 0.5 * 1.0 + 1.0*1.0 = 1.5
        let step2 = deltanet_recurrence_reference(&q, &k, &v, &mut state, n_head, head_dim, beta);

        for h in 0..n_head {
            for r in 0..head_dim {
                // output = Σ_col 1.5 * 1.0 = 1.5 * head_dim
                let expected = 1.5 * head_dim as f32;
                assert!(
                    (step2.output[h * head_dim + r] - expected).abs() < 1e-3,
                    "step2 output[head={h}, row={r}] = {}, expected {expected}",
                    step2.output[h * head_dim + r],
                );
            }
        }
    }

    /// GOAT proof T9: multi-step prefill accumulates state correctly.
    #[test]
    fn test_recurrence_prefill_accumulates() {
        let n_head = 2;
        let head_dim = 4;
        let seq_len = 3;
        let beta = 0.9f32;

        let queries = vec![1.0f32; seq_len * n_head * head_dim];
        let keys = vec![1.0f32; seq_len * n_head * head_dim];
        let values = vec![1.0f32; seq_len * n_head * head_dim];
        let mut state = vec![0.0f32; n_head * head_dim * head_dim];

        let outputs = deltanet_recurrence_prefill(
            &queries, &keys, &values, &mut state, n_head, head_dim, beta, seq_len,
        );

        // After step 1: state = 1.0 everywhere, output = head_dim
        // After step 2: state = 0.9*1.0 + 1.0 = 1.9, output = 1.9 * head_dim
        // After step 3: state = 0.9*1.9 + 1.0 = 2.71, output = 2.71 * head_dim
        let expected_step1 = 1.0 * head_dim as f32;
        let expected_step2 = 1.9 * head_dim as f32;
        let expected_step3 = 2.71 * head_dim as f32;

        for h in 0..n_head {
            for r in 0..head_dim {
                let step1_out = outputs[h * head_dim + r];
                let step2_out = outputs[n_head * head_dim + h * head_dim + r];
                let step3_out = outputs[2 * n_head * head_dim + h * head_dim + r];

                assert!(
                    (step1_out - expected_step1).abs() < 1e-2,
                    "step1 output = {step1_out}, expected {expected_step1}"
                );
                assert!(
                    (step2_out - expected_step2).abs() < 1e-2,
                    "step2 output = {step2_out}, expected {expected_step2}"
                );
                assert!(
                    (step3_out - expected_step3).abs() < 1e-2,
                    "step3 output = {step3_out}, expected {expected_step3}"
                );
            }
        }
    }

    /// GOAT proof T9: non-trivial q/k/v produce correct matvec result.
    #[test]
    fn test_recurrence_nontrivial_values() {
        let n_head = 1;
        let head_dim = 3;

        let q: Vec<f32> = vec![1.0, 2.0, 3.0];
        let k: Vec<f32> = vec![0.5, 1.0, 1.5];
        let v: Vec<f32> = vec![2.0, 0.0, -1.0];
        let mut state = vec![0.0f32; head_dim * head_dim];

        let result = deltanet_recurrence_reference(&q, &k, &v, &mut state, n_head, head_dim, 1.0);

        // S[row,col] = v[row] * k[col]
        // S[0,:] = 2.0 * [0.5, 1.0, 1.5] = [1.0, 2.0, 3.0]
        // S[1,:] = 0.0 * [0.5, 1.0, 1.5] = [0.0, 0.0, 0.0]
        // S[2,:] = -1.0 * [0.5, 1.0, 1.5] = [-0.5, -1.0, -1.5]

        // output[0] = S[0,:] · q = 1.0*1.0 + 2.0*2.0 + 3.0*3.0 = 14.0
        // output[1] = S[1,:] · q = 0.0
        // output[2] = S[2,:] · q = -0.5*1.0 + -1.0*2.0 + -1.5*3.0 = -0.5 - 2.0 - 4.5 = -7.0

        assert!(
            (result.output[0] - 14.0).abs() < 1e-4,
            "output[0] = {}",
            result.output[0]
        );
        assert!(
            (result.output[1] - 0.0).abs() < 1e-4,
            "output[1] = {}",
            result.output[1]
        );
        assert!(
            (result.output[2] - (-7.0)).abs() < 1e-4,
            "output[2] = {}",
            result.output[2]
        );
    }
}
