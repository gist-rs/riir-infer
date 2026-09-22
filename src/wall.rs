//! Wall Attention — CPU reference implementation (Plan 192).
//!
//! Replaces `RoPE` with diagonal gate multiplications on Q and K for stronger
//! length generalisation at lower compute cost.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sigmoid followed by soft-clamp: `min(sigmoid(x), max_val)`.
///
/// The sigmoid already maps to (0, 1), so `min` is sufficient.
/// Delegates to `katgpt_core::simd::fast_sigmoid` (Cephes polynomial).
#[inline]
pub fn soft_clamp_sigmoid(x: f32, max_val: f32) -> f32 {
    katgpt_core::simd::fast_sigmoid(x).min(max_val)
}

// ---------------------------------------------------------------------------
// Gate projection
// ---------------------------------------------------------------------------

/// Zero-alloc gate projection: writes soft-clamped gate values into `out`.
///
/// * `hidden`        — `[d_model]`
/// * `w_g`           — `[gate_proj_dim, d_model]` row-major
/// * `out`           — `[gate_proj_dim]` pre-allocated output buffer
///
/// Uses `simd_dot_f32` for the per-row dot product.
pub fn wall_gate_project_into(
    hidden: &[f32],
    w_g: &[f32],
    bias: f32,
    gate_max: f32,
    gate_proj_dim: usize,
    d_model: usize,
    out: &mut [f32],
) {
    assert_eq!(hidden.len(), d_model, "hidden length mismatch");
    assert_eq!(w_g.len(), gate_proj_dim * d_model, "w_g length mismatch");
    assert!(out.len() >= gate_proj_dim, "out buffer too small");

    for i in 0..gate_proj_dim {
        let row = &w_g[i * d_model..(i + 1) * d_model];
        let logit = crate::simd::simd_dot_f32(row, hidden, d_model) + bias;
        out[i] = soft_clamp_sigmoid(logit, gate_max);
    }
}

// NOTE: prefer _into/_inplace variants in hot paths
pub fn wall_gate_project(
    hidden: &[f32],
    w_g: &[f32],
    bias: f32,
    gate_max: f32,
    gate_proj_dim: usize,
    d_model: usize,
) -> Vec<f32> {
    let mut gates = vec![0.0f32; gate_proj_dim];
    wall_gate_project_into(
        hidden,
        w_g,
        bias,
        gate_max,
        gate_proj_dim,
        d_model,
        &mut gates,
    );
    gates
}

// ---------------------------------------------------------------------------
// Prefix accumulation
// ---------------------------------------------------------------------------

/// Zero-alloc prefix decode: accumulates log-gate into prefix in-place.
///
/// * `log_gate` — `[head_dim]` input
/// * `prefix`   — `[head_dim]` mutable prefix, updated in-place
///
/// After return: `prefix[d] += log_gate[d]` for all `d`.
pub fn wall_prefix_decode_into(log_gate: &[f32], prefix: &mut [f32]) {
    assert_eq!(log_gate.len(), prefix.len(), "dimension mismatch");
    let len = log_gate.len();
    let chunks = len / 4;
    for c in 0..chunks {
        let d = c * 4;
        prefix[d] += log_gate[d];
        prefix[d + 1] += log_gate[d + 1];
        prefix[d + 2] += log_gate[d + 2];
        prefix[d + 3] += log_gate[d + 3];
    }
    for d in (chunks * 4)..len {
        prefix[d] += log_gate[d];
    }
}

// NOTE: prefer _into/_inplace variants in hot paths
pub fn wall_prefix_decode(log_gate: &[f32], prefix_prev: &[f32]) -> Vec<f32> {
    assert_eq!(log_gate.len(), prefix_prev.len(), "dimension mismatch");
    let mut out = prefix_prev.to_vec();
    wall_prefix_decode_into(log_gate, &mut out);
    out
}

/// Prefill: sequential prefix-sum over the sequence dimension.
///
/// * `log_gates` — `[seq_len * head_dim]`, flat layout (outer = seq, inner = `head_dim`)
/// * Returns     — `[seq_len * head_dim]` where `out[t * head_dim + d] = sum_{k=0..=t} log_gates[k * head_dim + d]`
pub fn wall_prefix_prefill(log_gates: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    assert_eq!(
        log_gates.len(),
        seq_len * head_dim,
        "log_gates length mismatch"
    );

    let mut out = vec![0.0f32; seq_len * head_dim];

    // First position — copy directly.
    if seq_len > 0 {
        out[..head_dim].copy_from_slice(&log_gates[..head_dim]);
    }

    // Sequential scan.
    for t in 1..seq_len {
        let (left, right) = out.split_at_mut(t * head_dim);
        let prev = &left[(t - 1) * head_dim..t * head_dim];
        let dst = &mut right[..head_dim];
        let cur = &log_gates[t * head_dim..(t + 1) * head_dim];

        // Chunked 4-wide accumulation for auto-vectorization.
        let chunks = head_dim / 4;
        let remainder = head_dim % 4;
        for c in 0..chunks {
            let d = c * 4;
            unsafe {
                dst[d] = *prev.get_unchecked(d) + *cur.get_unchecked(d);
                dst[d + 1] = *prev.get_unchecked(d + 1) + *cur.get_unchecked(d + 1);
                dst[d + 2] = *prev.get_unchecked(d + 2) + *cur.get_unchecked(d + 2);
                dst[d + 3] = *prev.get_unchecked(d + 3) + *cur.get_unchecked(d + 3);
            }
        }
        for d in (chunks * 4)..(chunks * 4 + remainder) {
            dst[d] = prev[d] + cur[d];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Q / K rescaling
// ---------------------------------------------------------------------------

/// Zero-alloc Q/K rescaling: modifies `q` and `k` in-place.
///
/// * `q`         — `[n_heads * head_dim]` mutated in-place
/// * `k`         — `[n_kv_heads * head_dim]` mutated in-place
/// * `prefix_q`  — `[head_dim]`
/// * `prefix_k`  — `[head_dim]`
///
/// Uses stack-allocated exp tables (`head_dim` ≤ 256) to avoid heap allocation.
/// Q is scaled *up* (`exp(prefix_q)`) and K is scaled *down* (`exp(-prefix_k)`),
/// implementing the exponential decay that Wall Attention uses in place of `RoPE`.
pub fn wall_rescale_qk_inplace(
    q: &mut [f32],
    k: &mut [f32],
    prefix_q: &[f32],
    prefix_k: &[f32],
    head_dim: usize,
) {
    assert!(
        q.len().is_multiple_of(head_dim),
        "q length must be a multiple of head_dim"
    );
    assert!(
        k.len().is_multiple_of(head_dim),
        "k length must be a multiple of head_dim"
    );
    assert_eq!(prefix_q.len(), head_dim, "prefix_q length mismatch");
    assert_eq!(prefix_k.len(), head_dim, "prefix_k length mismatch");

    // Pre-compute exponentials into stack arrays (head_dim ≤ 256 for all known models).
    assert!(
        head_dim <= 256,
        "head_dim > 256 not supported in stack path"
    );
    let mut eq = [0.0f32; 256];
    let mut ek = [0.0f32; 256];
    for d in 0..head_dim {
        eq[d] = prefix_q[d].exp();
        ek[d] = (-prefix_k[d]).exp();
    }
    let (eq, ek) = (&eq[..head_dim], &ek[..head_dim]);

    // Rescale Q in-place. Iterate per-head blocks so the eq index resets each
    // head — eliminates the `% head_dim` modulo (integer division) from the
    // inner loop and lets LLVM auto-vectorize the element-wise multiply.
    // Bounds checked once per head on the slice; inner zip loop is check-free.
    for head_start in (0..q.len()).step_by(head_dim) {
        let q_head = &mut q[head_start..head_start + head_dim];
        for (qv, &e) in q_head.iter_mut().zip(eq) {
            *qv *= e;
        }
    }

    // Rescale K in-place (same pattern).
    for head_start in (0..k.len()).step_by(head_dim) {
        let k_head = &mut k[head_start..head_start + head_dim];
        for (kv, &e) in k_head.iter_mut().zip(ek) {
            *kv *= e;
        }
    }
}

// NOTE: prefer _into/_inplace variants in hot paths
pub fn wall_rescale_qk(
    q: &[f32],
    k: &[f32],
    prefix_q: &[f32],
    prefix_k: &[f32],
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut q_out = q.to_vec();
    let mut k_out = k.to_vec();
    wall_rescale_qk_inplace(&mut q_out, &mut k_out, prefix_q, prefix_k, head_dim);
    (q_out, k_out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- soft_clamp_sigmoid -------------------------------------------------

    #[test]
    fn test_soft_clamp_boundary() {
        // Sigmoid at extreme negative → ~0.
        let v = soft_clamp_sigmoid(-100.0, 0.87);
        assert!(v < 1e-10, "expected ~0, got {v}");

        // Sigmoid at extreme positive → ~1, clamped to gate_max.
        let v = soft_clamp_sigmoid(100.0, 0.87);
        assert!((v - 0.87).abs() < 1e-6, "expected 0.87, got {v}");

        // Sigmoid(0) = 0.5, gate_max = 0.87 → 0.5.
        let v = soft_clamp_sigmoid(0.0, 0.87);
        assert!((v - 0.5).abs() < 1e-6, "expected 0.5, got {v}");

        // gate_max > 1 should not clamp sigmoid(0).
        let v = soft_clamp_sigmoid(0.0, 2.0);
        assert!((v - 0.5).abs() < 1e-6, "expected 0.5, got {v}");
    }

    // -- wall_gate_project --------------------------------------------------

    #[test]
    fn test_gate_project_known() {
        let d_model = 2;
        let gate_proj_dim = 3;
        // Identity-ish weight matrix (3 rows × 2 cols).
        let w_g: Vec<f32> = vec![
            1.0, 0.0, // row 0
            0.0, 1.0, // row 1
            1.0, 1.0, // row 2
        ];
        let hidden: Vec<f32> = vec![1.0, 2.0];
        let bias = 0.0;
        let gate_max = 0.87;

        let gates = wall_gate_project(&hidden, &w_g, bias, gate_max, gate_proj_dim, d_model);
        assert_eq!(gates.len(), 3);

        // sigmoid(1) ≈ 0.7311 < 0.87 → unclamped
        let expected_0 = 1.0_f32 / (1.0 + (-1.0_f32).exp());
        assert!(
            (gates[0] - expected_0).abs() < 1e-5,
            "gate[0] expected {expected_0}, got {}",
            gates[0]
        );

        // sigmoid(2) ≈ 0.8808 > 0.87 → clamped to 0.87
        let expected_1 = 1.0_f32 / (1.0 + (-2.0_f32).exp());
        assert!(
            (gates[1] - gate_max).abs() < 1e-5,
            "gate[1] expected clamped to {gate_max}, got {}",
            gates[1]
        );
        assert!(expected_1 > gate_max, "sanity: sigmoid(2) > 0.87");

        // sigmoid(3) > 0.87 → clamped
        assert!(
            (gates[2] - gate_max).abs() < 1e-5,
            "gate[2] expected clamped to {gate_max}, got {}",
            gates[2]
        );
    }

    // -- wall_prefix_decode -------------------------------------------------

    #[test]
    fn test_prefix_decode_accumulation() {
        let head_dim = 4;
        let mut prefix = vec![0.0f32; head_dim];

        // Step 1
        let g1: Vec<f32> = vec![0.1, -0.2, 0.3, 0.0];
        prefix = wall_prefix_decode(&g1, &prefix);
        let expected = [0.1f32, -0.2, 0.3, 0.0];
        for (i, (&a, &b)) in prefix.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "step 1 [{i}]: {a} != {b}");
        }

        // Step 2
        let g2: Vec<f32> = vec![0.5, 0.1, -0.1, 0.2];
        prefix = wall_prefix_decode(&g2, &prefix);
        let expected = [0.6f32, -0.1, 0.2, 0.2];
        for (i, (&a, &b)) in prefix.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "step 2 [{i}]: {a} != {b}");
        }

        // Step 3
        let g3: Vec<f32> = vec![-0.3, 0.4, 0.0, -0.5];
        prefix = wall_prefix_decode(&g3, &prefix);
        let expected = [0.3f32, 0.3, 0.2, -0.3];
        for (i, (&a, &b)) in prefix.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "step 3 [{i}]: {a} != {b}");
        }
    }

    // -- wall_prefix_prefill ------------------------------------------------

    #[test]
    fn test_prefix_prefill() {
        let head_dim = 2;
        let seq_len = 3;
        // [t0d0, t0d1, t1d0, t1d1, t2d0, t2d1]
        let log_gates: Vec<f32> = vec![
            1.0, 2.0, // t=0
            0.5, -1.0, // t=1
            -0.5, 0.5, // t=2
        ];

        let out = wall_prefix_prefill(&log_gates, seq_len, head_dim);
        assert_eq!(out.len(), 6);

        // t=0: just copy
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        // t=1: cumulative
        assert!((out[2] - 1.5).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
        // t=2: cumulative
        assert!((out[4] - 1.0).abs() < 1e-6);
        assert!((out[5] - 1.5).abs() < 1e-6);
    }

    // -- wall_rescale_qk ----------------------------------------------------

    #[test]
    fn test_rescale_identity() {
        let head_dim = 3;
        let q: Vec<f32> = vec![1.0, 2.0, 3.0];
        let k: Vec<f32> = vec![4.0, 5.0, 6.0];
        let prefix_q: Vec<f32> = vec![0.0; head_dim];
        let prefix_k: Vec<f32> = vec![0.0; head_dim];

        let (q_out, k_out) = wall_rescale_qk(&q, &k, &prefix_q, &prefix_k, head_dim);

        // exp(0) = 1 → no change.
        for i in 0..3 {
            assert!((q_out[i] - q[i]).abs() < 1e-6, "q_out[{i}] mismatch");
            assert!((k_out[i] - k[i]).abs() < 1e-6, "k_out[{i}] mismatch");
        }
    }

    #[test]
    fn test_rescale_decay() {
        let head_dim = 2;
        let q: Vec<f32> = vec![1.0, 1.0];
        let k: Vec<f32> = vec![1.0, 1.0];
        let prefix_q: Vec<f32> = vec![1.0, 2.0]; // positive → exp() > 1 → Q scaled up
        let prefix_k: Vec<f32> = vec![1.0, 2.0]; // positive → exp(-x) < 1 → K scaled down

        let (q_out, k_out) = wall_rescale_qk(&q, &k, &prefix_q, &prefix_k, head_dim);

        // Q: exp(1) ≈ 2.718, exp(2) ≈ 7.389
        assert!((q_out[0] - 1.0_f32.exp()).abs() < 1e-4);
        assert!((q_out[1] - 2.0_f32.exp()).abs() < 1e-4);
        // Q should be scaled up.
        assert!(q_out[0] > 1.0);
        assert!(q_out[1] > 1.0);

        // K: exp(-1) ≈ 0.368, exp(-2) ≈ 0.135
        assert!((k_out[0] - (-1.0_f32).exp()).abs() < 1e-4);
        assert!((k_out[1] - (-2.0_f32).exp()).abs() < 1e-4);
        // K should be scaled down.
        assert!(k_out[0] < 1.0);
        assert!(k_out[1] < 1.0);
    }

    // -- Numerical stability ------------------------------------------------

    #[test]
    fn test_numerical_stability() {
        let head_dim = 8;
        let seq_len = 8192;

        // All gates = 0.01 (small positive) → prefix at t=8191 is ~81.92
        let log_gates = vec![0.01f32; seq_len * head_dim];
        let prefix = wall_prefix_prefill(&log_gates, seq_len, head_dim);

        // No NaN / Inf.
        for (i, &v) in prefix.iter().enumerate() {
            assert!(v.is_finite(), "prefix[{i}] is not finite: {v}");
        }

        // Last position should be ≈ 0.01 * 8192 = 81.92
        for d in 0..head_dim {
            let v = prefix[(seq_len - 1) * head_dim + d];
            let expected = 0.01 * seq_len as f32;
            assert!(
                (v - expected).abs() < 0.1,
                "prefix end mismatch at d={d}: {v} vs {expected}"
            );
        }

        // Now exercise wall_rescale_qk with these large prefix values.
        let prefix_q = &prefix[(seq_len - 1) * head_dim..seq_len * head_dim];
        let prefix_k = prefix_q; // same

        let q = vec![1.0f32; head_dim];
        let k = vec![1.0f32; head_dim];
        let (q_out, k_out) = wall_rescale_qk(&q, &k, prefix_q, prefix_k, head_dim);

        for (i, &v) in q_out.iter().enumerate() {
            assert!(v.is_finite(), "q_out[{i}] not finite: {v}");
        }
        for (i, &v) in k_out.iter().enumerate() {
            assert!(v.is_finite(), "k_out[{i}] not finite: {v}");
        }
    }

    // -- _into / _inplace variants match allocating versions ----------------

    #[test]
    fn test_gate_project_into_matches_allocating() {
        let d_model = 4;
        let gate_proj_dim = 3;
        let w_g: Vec<f32> = vec![1.0, 0.0, -0.5, 0.3, 0.0, 1.0, 0.2, -0.1, 0.5, 0.5, 0.5, 0.5];
        let hidden: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bias = 0.1;
        let gate_max = 0.87;

        let alloc = wall_gate_project(&hidden, &w_g, bias, gate_max, gate_proj_dim, d_model);
        let mut into_buf = vec![0.0f32; gate_proj_dim];
        wall_gate_project_into(
            &hidden,
            &w_g,
            bias,
            gate_max,
            gate_proj_dim,
            d_model,
            &mut into_buf,
        );

        for i in 0..gate_proj_dim {
            assert!(
                (alloc[i] - into_buf[i]).abs() < 1e-6,
                "gate_project mismatch at {i}: alloc={}, into={}",
                alloc[i],
                into_buf[i]
            );
        }
    }

    #[test]
    fn test_prefix_decode_into_matches_allocating() {
        let head_dim = 4;
        let log_gate: Vec<f32> = vec![0.1, -0.2, 0.3, 0.0];
        let prefix_prev: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];

        let alloc = wall_prefix_decode(&log_gate, &prefix_prev);
        let mut into_buf = prefix_prev.clone();
        wall_prefix_decode_into(&log_gate, &mut into_buf);

        for i in 0..head_dim {
            assert!(
                (alloc[i] - into_buf[i]).abs() < 1e-6,
                "prefix_decode mismatch at {i}: alloc={}, into={}",
                alloc[i],
                into_buf[i]
            );
        }
    }

    #[test]
    fn test_rescale_qk_inplace_matches_allocating() {
        let head_dim = 4;
        let q: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let k: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let prefix_q: Vec<f32> = vec![0.5, -0.5, 1.0, -1.0];
        let prefix_k: Vec<f32> = vec![0.2, 0.3, -0.1, 0.0];

        let (q_alloc, k_alloc) = wall_rescale_qk(&q, &k, &prefix_q, &prefix_k, head_dim);
        let mut q_into = q.clone();
        let mut k_into = k.clone();
        wall_rescale_qk_inplace(&mut q_into, &mut k_into, &prefix_q, &prefix_k, head_dim);

        for i in 0..head_dim {
            assert!(
                (q_alloc[i] - q_into[i]).abs() < 1e-6,
                "q mismatch at {i}: alloc={}, into={}",
                q_alloc[i],
                q_into[i]
            );
            assert!(
                (k_alloc[i] - k_into[i]).abs() < 1e-6,
                "k mismatch at {i}: alloc={}, into={}",
                k_alloc[i],
                k_into[i]
            );
        }
    }

    #[test]
    fn test_rescale_qk_inplace_multi_head() {
        // 2 heads × head_dim=3 = 6 elements — tests the modulo indexing.
        let head_dim = 3;
        let q: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let k: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let prefix_q: Vec<f32> = vec![0.0; head_dim]; // exp(0)=1 → identity
        let prefix_k: Vec<f32> = vec![0.0; head_dim];

        let (q_alloc, k_alloc) = wall_rescale_qk(&q, &k, &prefix_q, &prefix_k, head_dim);
        let mut q_into = q.clone();
        let mut k_into = k.clone();
        wall_rescale_qk_inplace(&mut q_into, &mut k_into, &prefix_q, &prefix_k, head_dim);

        for i in 0..6 {
            assert!((q_alloc[i] - q_into[i]).abs() < 1e-6, "q multi_head[{i}]");
            assert!((k_alloc[i] - k_into[i]).abs() < 1e-6, "k multi_head[{i}]");
        }
    }
}
