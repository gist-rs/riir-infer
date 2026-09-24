//! Rotary Position Embeddings (`RoPE`) for Gemma 2 (Plan 087).
//!
//! Applies complex rotation to Q/K vectors per head:
//!   For each pair (`q[2i]`, `q[2i+1]`):
//! ```text
//! angle = pos / (theta ^ (2i / head_dim))
//! q[2i]     = q[2i] * cos(angle) - q[2i+1] * sin(angle)
//! q[2i+1]   = q[2i] * sin(angle) + q[2i+1] * cos(angle)
//! ```

/// Pre-computed `RoPE` frequency table.
///
/// Frequencies depend only on `theta` and `head_dim`, which are model constants.
/// Pre-computing eliminates `powf` calls from the hot path (Issue 024).
pub struct RopeFreqTable {
    /// `freq[i] = 1.0 / theta.powf(2.0 * i / head_dim)` for i in `0..head_dim/2`
    freq: Vec<f32>,
}

impl RopeFreqTable {
    /// Build a frequency table for the given theta and `head_dim`.
    pub fn new(theta: f32, head_dim: usize) -> Self {
        let half = head_dim / 2;
        let freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        Self { freq }
    }

    /// Get the frequency table slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.freq
    }
}

/// Apply `RoPE` using a pre-computed frequency table (zero `powf` calls in hot path).
///
/// Pre-computes sin/cos per position into a stack buffer, then applies to all
/// Q/K heads — eliminates redundant `sin_cos` calls across heads.
pub fn apply_rope_with_freq(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    freq_table: &[f32],
) {
    let half = freq_table.len();

    // Fast path: pos=0 is identity (all angles = 0, sin=0, cos=1)
    if pos == 0 {
        return;
    }

    let n_heads_q = q.len() / head_dim;
    let n_heads_k = k.len() / head_dim;

    // Pre-compute sin/cos once per position — reused across all heads
    let mut cos_sin = [0.0f32; 512]; // Stack buffer: supports half_dim up to 256 (head_dim 512)
    let use_stack = half <= 256;
    let mut heap_buf;
    let buf: &mut [f32] = if use_stack {
        &mut cos_sin[..half * 2]
    } else {
        heap_buf = vec![0.0f32; half * 2];
        &mut heap_buf
    };

    // Phase 1: pre-compute cos and sin for this position
    let pos_f = pos as f32;
    for i in 0..half {
        let angle = pos_f * freq_table[i];
        let (sin_a, cos_a) = angle.sin_cos();
        buf[i] = cos_a; // cos at [0..half)
        buf[half + i] = sin_a; // sin at [half..2*half)
    }

    let cos_table = &buf[..half];
    let sin_table = &buf[half..];

    // Phase 2: apply rotation to each Q head
    for h in 0..n_heads_q {
        let offset = h * head_dim;
        apply_rope_heads_precomputed(&mut q[offset..offset + head_dim], cos_table, sin_table);
    }

    // Phase 3: apply rotation to each K head
    for h in 0..n_heads_k {
        let offset = h * head_dim;
        apply_rope_heads_precomputed(&mut k[offset..offset + head_dim], cos_table, sin_table);
    }
}

/// Apply **partial** rotary `RoPE` to Q and K buffers in-place (Issue 594).
///
/// Only the first `rotary_dim` dimensions of each head are rotated (paired as
/// `(i, i + rotary_dim/2)` for `i in 0..rotary_dim/2`); the remaining
/// `head_dim - rotary_dim` dims pass through unchanged. This matches the
/// `partial_rotary_factor` / `rope.dimension_count` convention used by Qwen3.5
/// full-attention layers and Gemma-4 full-attention layers.
///
/// For `rotary_dim == head_dim` this degenerates to a full rotation (identical
/// to [`apply_rope_with_freq`]). For `rotary_dim == 0` it is a no-op.
///
/// `freq_table` MUST have `rotary_dim / 2` entries (built via
/// `RopeFreqTable::new(theta, rotary_dim)` — the `dim` in the `inv_freq`
/// denominator is `rotary_dim`, not `head_dim`, matching `HuggingFace`'s
/// `compute_default_rope_parameters`).
///
/// **mrope note (Issue 594):** Qwen3.5 uses multimodal `RoPE` with
/// `rope.dimension_sections = [11, 11, 10, 0]`. For **text-only** inference
/// all sections share the same position ID (T=H=W=pos, 4th=0), so mrope
/// collapses to standard partial `RoPE` on `rotary_dim` dims. No section
/// handling is needed for text-only decode.
pub fn apply_partial_rope_with_freq(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    rotary_dim: usize,
    freq_table: &[f32],
) {
    debug_assert!(
        rotary_dim <= head_dim,
        "rotary_dim must not exceed head_dim"
    );
    debug_assert!(
        rotary_dim.is_multiple_of(2),
        "rotary_dim must be even (rotate-half convention)"
    );
    debug_assert!(
        freq_table.len() >= rotary_dim / 2,
        "freq_table too short for rotary_dim"
    );

    // Fast paths: pos=0 is identity; rotary_dim=0 is a no-op.
    if pos == 0 || rotary_dim == 0 {
        return;
    }

    let half_rot = rotary_dim / 2;
    let n_heads_q = q.len() / head_dim;
    let n_heads_k = k.len() / head_dim;

    // Pre-compute cos/sin once per position — reused across all heads.
    // Stack buffer supports half_rot up to 256 (rotary_dim up to 512).
    let mut cos_sin = [0.0f32; 512];
    let use_stack = half_rot <= 256;
    let mut heap_buf;
    let buf: &mut [f32] = if use_stack {
        &mut cos_sin[..half_rot * 2]
    } else {
        heap_buf = vec![0.0f32; half_rot * 2];
        &mut heap_buf
    };

    let pos_f = pos as f32;
    for i in 0..half_rot {
        let angle = pos_f * freq_table[i];
        let (sin_a, cos_a) = angle.sin_cos();
        buf[i] = cos_a;
        buf[half_rot + i] = sin_a;
    }
    let cos_table = &buf[..half_rot];
    let sin_table = &buf[half_rot..];

    // Apply partial rotation to each Q head: rotate the first rotary_dim dims,
    // leave dims [rotary_dim..head_dim) untouched.
    for h in 0..n_heads_q {
        let off = h * head_dim;
        apply_rope_heads_precomputed(&mut q[off..off + rotary_dim], cos_table, sin_table);
    }
    for h in 0..n_heads_k {
        let off = h * head_dim;
        apply_rope_heads_precomputed(&mut k[off..off + rotary_dim], cos_table, sin_table);
    }
}

/// Apply `RoPE` to a single buffer in-place (for `RoVE` value rotation).
///
/// This is the V-only counterpart of [`apply_rope_with_freq`]. It rotates
/// `buf` — laid out as `[n_heads * head_dim]` — using the **same rotate-half
/// convention** as Q/K. Called once per token per layer when the
/// `rotary_value_embedding` feature is enabled (Plan 557 T3.B).
///
/// **Zero allocation** for `head_dim ≤ 512` (stack buffer); falls back to a
/// single heap allocation for pathological dims.
///
/// # `RoVE` convention note
//
// riir-engine's RoPE uses the **rotate-half** convention (pairs `vec[i]` with
// `vec[i + half]`), matching HuggingFace's `rotate_half`. The katgpt-core RoVE
// substrate (`RopeAction`) uses **adjacent pairs** `(2i, 2i+1)` — a DIFFERENT
// rotation subgroup. For the RoVE paper's equivalence claim to hold (rotating
// V by the same RoPE as Q/K), the V rotation MUST use the same convention as
// Q/K. Therefore this function reuses riir-engine's own RoPE rather than
// katgpt-core's `RopeAction`. The katgpt-core substrate remains valuable for
// attention-matching compaction tests (Phase 4 G9/G10) and benchmarking, but
// is NOT the forward-path V rotation (convention mismatch).
#[cfg(feature = "rotary_value_embedding")]
pub fn apply_rope_values(buf: &mut [f32], pos: usize, head_dim: usize, freq_table: &[f32]) {
    let half = freq_table.len();

    // Fast path: pos=0 is identity (all angles = 0, sin=0, cos=1)
    if pos == 0 {
        return;
    }

    let n_heads = buf.len() / head_dim;

    // Pre-compute sin/cos once per position — reused across all heads
    let mut cos_sin = [0.0f32; 512];
    let use_stack = half <= 256;
    let mut heap_buf;
    let buf_local: &mut [f32] = if use_stack {
        &mut cos_sin[..half * 2]
    } else {
        heap_buf = vec![0.0f32; half * 2];
        &mut heap_buf
    };

    let pos_f = pos as f32;
    for i in 0..half {
        let angle = pos_f * freq_table[i];
        let (sin_a, cos_a) = angle.sin_cos();
        buf_local[i] = cos_a;
        buf_local[half + i] = sin_a;
    }

    let cos_table = &buf_local[..half];
    let sin_table = &buf_local[half..];

    for h in 0..n_heads {
        let offset = h * head_dim;
        apply_rope_heads_precomputed(&mut buf[offset..offset + head_dim], cos_table, sin_table);
    }
}

/// Apply the **inverse** `RoPE` rotation to a single buffer in-place.
///
/// Computes `R_{-pos}` — the group inverse of the forward `RoPE` at `pos`.
/// For rotate-half `RoPE` this means: cos is identical, sin is negated.
///
/// Used after attention aggregation to de-rotate the output back into the
/// query's local frame (`RoVE` Plan 557 T3.B):
/// `final_i = R_{-i} · Σ_j A_ij · R_j · V_j = Σ_j A_ij · R_{j-i} · V_j`
///
/// **Zero allocation** for `head_dim ≤ 512`.
#[cfg(feature = "rotary_value_embedding")]
pub fn apply_inverse_rope_output(buf: &mut [f32], pos: usize, head_dim: usize, freq_table: &[f32]) {
    let half = freq_table.len();

    // Fast path: pos=0 is identity
    if pos == 0 {
        return;
    }

    let n_heads = buf.len() / head_dim;

    // Pre-compute sin/cos once per position — reused across all heads
    let mut cos_sin = [0.0f32; 512];
    let use_stack = half <= 256;
    let mut heap_buf;
    let buf_local: &mut [f32] = if use_stack {
        &mut cos_sin[..half * 2]
    } else {
        heap_buf = vec![0.0f32; half * 2];
        &mut heap_buf
    };

    let pos_f = pos as f32;
    for i in 0..half {
        let angle = pos_f * freq_table[i];
        let (sin_a, cos_a) = angle.sin_cos();
        buf_local[i] = cos_a;
        buf_local[half + i] = sin_a; // stored as +sin; apply_inverse_rope_heads negates it
    }

    let cos_table = &buf_local[..half];
    let sin_table = &buf_local[half..];

    for h in 0..n_heads {
        let offset = h * head_dim;
        apply_inverse_rope_heads_precomputed(
            &mut buf[offset..offset + head_dim],
            cos_table,
            sin_table,
        );
    }
}

/// Inverse of [`apply_rope_heads_precomputed`].
///
/// Forward: `vec'[i] = vec[i]·cos - vec[i+half]·sin; vec'[i+half] = vec[i]·sin + vec[i+half]·cos`
/// Inverse: `vec'[i] = vec[i]·cos + vec[i+half]·sin; vec'[i+half] = -vec[i]·sin + vec[i+half]·cos`
///
/// Equivalently: the forward rotation matrix transposed (rotation matrices are
/// orthogonal, so transpose = inverse).
#[cfg(feature = "rotary_value_embedding")]
#[inline]
fn apply_inverse_rope_heads_precomputed(vec: &mut [f32], cos_table: &[f32], sin_table: &[f32]) {
    let half = cos_table.len();
    for i in 0..half {
        let cos_a = cos_table[i];
        let sin_a = sin_table[i];
        let x0 = vec[i];
        let x1 = vec[i + half];
        // Inverse: swap the sign of sin in the cross terms.
        vec[i] = x0 * cos_a + x1 * sin_a;
        vec[i + half] = -x0 * sin_a + x1 * cos_a;
    }
}

/// Apply `RoPE` rotation using pre-computed cos/sin tables (zero transcendentals).
///
/// This is the inner loop after sin/cos have been pre-computed per position.
/// `cos_table[i]` and `sin_table[i]` must contain the pre-computed values.
#[inline]
fn apply_rope_heads_precomputed(vec: &mut [f32], cos_table: &[f32], sin_table: &[f32]) {
    let half = cos_table.len();
    // Process 4 pairs at a time to help LLVM auto-vectorize.
    // Each pair: (vec[i], vec[i+half]) rotated by (cos[i], sin[i]).
    let chunks = half / 4;

    for c in 0..chunks {
        let i = c * 4;
        // Load cos/sin for 4 elements
        let c0 = cos_table[i];
        let s0 = sin_table[i];
        let c1 = cos_table[i + 1];
        let s1 = sin_table[i + 1];
        let c2 = cos_table[i + 2];
        let s2 = sin_table[i + 2];
        let c3 = cos_table[i + 3];
        let s3 = sin_table[i + 3];
        // Load vec pairs
        let x0 = vec[i];
        let y0 = vec[i + half];
        let x1 = vec[i + 1];
        let y1 = vec[i + 1 + half];
        let x2 = vec[i + 2];
        let y2 = vec[i + 2 + half];
        let x3 = vec[i + 3];
        let y3 = vec[i + 3 + half];
        // Rotate: (x*cos - y*sin, x*sin + y*cos)
        vec[i] = x0 * c0 - y0 * s0;
        vec[i + half] = x0 * s0 + y0 * c0;
        vec[i + 1] = x1 * c1 - y1 * s1;
        vec[i + 1 + half] = x1 * s1 + y1 * c1;
        vec[i + 2] = x2 * c2 - y2 * s2;
        vec[i + 2 + half] = x2 * s2 + y2 * c2;
        vec[i + 3] = x3 * c3 - y3 * s3;
        vec[i + 3 + half] = x3 * s3 + y3 * c3;
    }

    // Handle remaining elements
    for i in (chunks * 4)..half {
        let cos_a = cos_table[i];
        let sin_a = sin_table[i];
        let x0 = vec[i];
        let x1 = vec[i + half];
        vec[i] = x0 * cos_a - x1 * sin_a;
        vec[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

#[allow(dead_code)]
fn apply_rope_heads_with_freq(vec: &mut [f32], pos: usize, freq: &[f32]) {
    let half = freq.len();
    for i in 0..half {
        let angle = pos as f32 * freq[i];
        let (sin_a, cos_a) = angle.sin_cos();
        let x0 = vec[i];
        let x1 = vec[i + half];
        vec[i] = x0 * cos_a - x1 * sin_a;
        vec[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

/// Apply `RoPE` to Q and K vectors in-place.
/// `q` and `k` are [`n_head` * `head_dim`] buffers (all heads concatenated).
/// `pos` is the current position in the sequence.
/// `head_dim` is the dimension per head.
/// `theta` is the base frequency (typically 10000.0).
///
/// **Note**: This allocates a frequency table on every call.
/// Prefer creating a `RopeFreqTable` once and using `apply_rope_with_freq` in hot paths.
pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    let table = RopeFreqTable::new(theta, head_dim);
    apply_rope_with_freq(q, k, pos, head_dim, table.as_slice());
}

/// Apply `RoPE` using the "rotate half" convention (LLaMA/Gemma style).
/// Pairs vec[i] with vec[i + half] instead of consecutive (2i, 2i+1).
/// This matches `HuggingFace`'s `rotate_half`: [-x2, x1] rotation pattern.
///
/// Note: Prefer `apply_rope_with_freq` which uses a pre-computed frequency table.
#[allow(dead_code)]
fn apply_rope_heads(vec: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let (sin_a, cos_a) = angle.sin_cos();
        let x0 = vec[i]; // first half element
        let x1 = vec[i + half]; // second half element
        vec[i] = x0 * cos_a - x1 * sin_a;
        vec[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_zero_position_is_identity() {
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let q_orig = q.clone();
        let k_orig = k.clone();

        apply_rope(&mut q, &mut k, 0, 4, 10000.0);

        // pos=0 → angle=0 for all pairs → cos=1, sin=0 → identity
        for i in 0..q.len() {
            assert!(
                (q[i] - q_orig[i]).abs() < 1e-6,
                "q[{i}] changed at pos=0: expected {}, got {}",
                q_orig[i],
                q[i]
            );
        }
        for i in 0..k.len() {
            assert!(
                (k[i] - k_orig[i]).abs() < 1e-6,
                "k[{i}] changed at pos=0: expected {}, got {}",
                k_orig[i],
                k[i]
            );
        }
    }

    #[test]
    fn test_rope_rotation_pair() {
        // head_dim=2, one head: pair (1.0, 0.0) rotated by angle at pos=1
        let mut q = vec![1.0, 0.0];
        let mut k = vec![0.0, 0.0];
        apply_rope(&mut q, &mut k, 1, 2, 10000.0);

        // angle = 1.0 / 10000^(0/2) = 1.0
        let angle = 1.0_f32;
        let expected_x0 = 1.0 * angle.cos() - 0.0 * angle.sin();
        let expected_x1 = 1.0 * angle.sin() + 0.0 * angle.cos();

        assert!(
            (q[0] - expected_x0).abs() < 1e-5,
            "q[0] expected {expected_x0}, got {}",
            q[0]
        );
        assert!(
            (q[1] - expected_x1).abs() < 1e-5,
            "q[1] expected {expected_x1}, got {}",
            q[1]
        );
    }

    #[test]
    fn test_rope_multi_head() {
        // 2 heads, head_dim=2: q = [h0_0, h0_1, h1_0, h1_1]
        let mut q = vec![1.0, 0.0, 1.0, 0.0];
        let mut k = vec![0.0, 0.0, 0.0, 0.0];
        apply_rope(&mut q, &mut k, 1, 2, 10000.0);

        // Both heads should rotate identically (same freq for i=0, head_dim=2)
        assert!(
            (q[0] - q[2]).abs() < 1e-6,
            "heads should rotate identically: q[0]={}, q[2]={}",
            q[0],
            q[2]
        );
        assert!(
            (q[1] - q[3]).abs() < 1e-6,
            "heads should rotate identically: q[1]={}, q[3]={}",
            q[1],
            q[3]
        );
    }

    #[test]
    fn test_rope_preserves_norm() {
        let mut q = vec![3.0, 4.0]; // norm = 5
        let mut k = vec![0.0; 2];
        let norm_sq: f32 = q[0] * q[0] + q[1] * q[1];
        let norm_before = norm_sq.sqrt();

        apply_rope(&mut q, &mut k, 42, 2, 10000.0);

        let norm_after = (q[0] * q[0] + q[1] * q[1]).sqrt();
        assert!(
            (norm_after - norm_before).abs() < 1e-5,
            "RoPE should preserve vector norm: before={norm_before}, after={norm_after}"
        );
    }

    #[test]
    fn test_rope_gqa_different_kv_heads() {
        // n_head=2, n_kv_head=1 → q has 2 heads, k has 1 head
        let head_dim = 2;
        let mut q = vec![1.0, 0.0, 1.0, 0.0]; // 2 heads × 2 dim
        let mut k = vec![1.0, 0.0]; // 1 head × 2 dim

        apply_rope(&mut q, &mut k, 5, head_dim, 10000.0);

        // All heads use same rotation for same position → first pair of q should match k
        assert!((q[0] - k[0]).abs() < 1e-6, "q[0]={}, k[0]={}", q[0], k[0]);
        assert!((q[1] - k[1]).abs() < 1e-6, "q[1]={}, k[1]={}", q[1], k[1]);
    }

    /// Verify "rotate half" convention matches `HuggingFace` for `head_dim=4`.
    /// `HuggingFace`: `rotate_half([a,b,c,d`]) = [-c,-d,a,b]
    /// `q_rotated` = q * cos + `rotate_half(q`) * sin
    /// For [a,b,c,d] with freq[0]=f0, freq[1]=f1, pos=p:
    ///   result[0] = a*cos(f0*p) - c*sin(f0*p)   (pair 0 with 2)
    ///   result[1] = b*cos(f1*p) - d*sin(f1*p)   (pair 1 with 3)
    ///   result[2] = c*cos(f0*p) + a*sin(f0*p)
    ///   result[3] = d*cos(f1*p) + b*sin(f1*p)
    #[test]
    fn test_rope_rotate_half_convention() {
        let head_dim = 4;
        let pos = 3;
        let theta: f32 = 10000.0;

        // freq[i] = 1 / theta^(2i/head_dim)
        let f0 = 1.0 / theta.powf(0.0 / head_dim as f32); // = 1.0
        let f1 = 1.0 / theta.powf(2.0 / head_dim as f32); // = 1/100

        let angle0 = pos as f32 * f0; // = 3.0
        let angle1 = pos as f32 * f1; // = 0.03

        let a = 1.0_f32;
        let b = 2.0_f32;
        let c = 3.0_f32;
        let d = 4.0_f32;

        // "rotate half" convention: pair (0,2) and (1,3)
        let expected = [
            a * angle0.cos() - c * angle0.sin(),
            b * angle1.cos() - d * angle1.sin(),
            c * angle0.cos() + a * angle0.sin(),
            d * angle1.cos() + b * angle1.sin(),
        ];

        let mut q = vec![a, b, c, d];
        let mut k = vec![0.0; 4];
        apply_rope(&mut q, &mut k, pos, head_dim, theta);

        for i in 0..4 {
            assert!(
                (q[i] - expected[i]).abs() < 1e-5,
                "q[{i}] expected {}, got {}",
                expected[i],
                q[i]
            );
        }
    }

    // ── RoVE (Rotary Value Embeddings) tests — Plan 557 T3.B ────────────
    //
    // These tests verify the V-only RoPE functions that wire RoVE into the
    // forward path. The convention is rotate-half (matching Q/K), NOT the
    // adjacent-pair convention used by katgpt-core's `RopeAction`.

    #[cfg(feature = "rotary_value_embedding")]
    fn make_freq_table(theta: f32, head_dim: usize) -> Vec<f32> {
        RopeFreqTable::new(theta, head_dim).as_slice().to_vec()
    }

    #[cfg(feature = "rotary_value_embedding")]
    #[test]
    fn test_rove_values_identity_at_pos_zero() {
        let head_dim = 4;
        let freq = make_freq_table(10000.0, head_dim);
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let v_orig = v.clone();
        apply_rope_values(&mut v, 0, head_dim, &freq);
        for i in 0..4 {
            assert!((v[i] - v_orig[i]).abs() < 1e-6, "v[{i}] changed at pos=0");
        }
    }

    #[cfg(feature = "rotary_value_embedding")]
    #[test]
    fn test_rove_values_matches_qk_rotation() {
        // RoVE V rotation must be identical to the Q/K RoPE rotation for the
        // same position — this is the convention-consistency invariant.
        let head_dim = 4;
        let pos = 3;
        let theta = 10000.0;
        let freq = make_freq_table(theta, head_dim);

        // Rotate a vector as Q (via apply_rope_with_freq) and as V (via apply_rope_values)
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k_dummy = vec![0.0; 4];
        apply_rope_with_freq(&mut q, &mut k_dummy, pos, head_dim, &freq);

        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope_values(&mut v, pos, head_dim, &freq);

        for i in 0..4 {
            assert!(
                (q[i] - v[i]).abs() < 1e-6,
                "V rotation must match Q rotation: q[{i}]={}, v[{i}]={}",
                q[i],
                v[i]
            );
        }
    }

    #[cfg(feature = "rotary_value_embedding")]
    #[test]
    fn test_rove_inverse_identity_at_pos_zero() {
        let head_dim = 4;
        let freq = make_freq_table(10000.0, head_dim);
        let mut out = vec![1.0, 2.0, 3.0, 4.0];
        let out_orig = out.clone();
        apply_inverse_rope_output(&mut out, 0, head_dim, &freq);
        for i in 0..4 {
            assert!(
                (out[i] - out_orig[i]).abs() < 1e-6,
                "out[{i}] changed at pos=0"
            );
        }
    }

    #[cfg(feature = "rotary_value_embedding")]
    #[test]
    fn test_rove_round_trip_forward_then_inverse() {
        // R_pos · R_{-pos} = I — the fundamental group property.
        let head_dim = 8;
        let pos = 17;
        let freq = make_freq_table(10000.0, head_dim);

        let original = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut buf = original.clone();

        // Forward then inverse = identity
        apply_rope_values(&mut buf, pos, head_dim, &freq);
        apply_inverse_rope_output(&mut buf, pos, head_dim, &freq);

        for i in 0..head_dim {
            assert!(
                (buf[i] - original[i]).abs() < 1e-5,
                "round-trip failed at [{i}]: expected {}, got {}",
                original[i],
                buf[i]
            );
        }
    }

    #[cfg(feature = "rotary_value_embedding")]
    #[test]
    fn test_rove_round_trip_inverse_then_forward() {
        // R_{-pos} · R_pos = I — also identity (commutative within a 1-param group).
        let head_dim = 8;
        let pos = 23;
        let freq = make_freq_table(10000.0, head_dim);

        let original = vec![0.5, -0.3, 0.8, 0.1, -0.7, 0.4, 0.2, -0.9];
        let mut buf = original.clone();

        apply_inverse_rope_output(&mut buf, pos, head_dim, &freq);
        apply_rope_values(&mut buf, pos, head_dim, &freq);

        for i in 0..head_dim {
            assert!(
                (buf[i] - original[i]).abs() < 1e-5,
                "inverse-then-forward failed at [{i}]: expected {}, got {}",
                original[i],
                buf[i]
            );
        }
    }

    #[cfg(feature = "rotary_value_embedding")]
    #[test]
    fn test_rove_values_multi_head() {
        // 2 heads × head_dim=4: v = [h0_0, h0_1, h0_2, h0_3, h1_0, h1_1, h1_2, h1_3]
        let head_dim = 4;
        let pos = 5;
        let freq = make_freq_table(10000.0, head_dim);

        // Both heads should rotate identically (same position → same rotation)
        let mut v = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        apply_rope_values(&mut v, pos, head_dim, &freq);

        // head 0 and head 1 should match
        for i in 0..head_dim {
            assert!(
                (v[i] - v[head_dim + i]).abs() < 1e-6,
                "heads should rotate identically: v[{i}]={}, v[{}]={}",
                v[i],
                head_dim + i,
                v[head_dim + i]
            );
        }
    }

    #[cfg(feature = "rotary_value_embedding")]
    #[test]
    fn test_rove_preserves_norm() {
        let head_dim = 4;
        let pos = 42;
        let freq = make_freq_table(10000.0, head_dim);

        let mut v = vec![3.0f32, 4.0, 1.0, 2.0];
        let norm_before: f32 = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2) + v[3].powi(2)).sqrt();

        apply_rope_values(&mut v, pos, head_dim, &freq);

        let norm_after: f32 = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2) + v[3].powi(2)).sqrt();
        assert!(
            (norm_after - norm_before).abs() < 1e-5,
            "RoVE should preserve norm: before={norm_before}, after={norm_after}"
        );
    }

    #[cfg(feature = "rotary_value_embedding")]
    #[test]
    fn test_rove_gqa_different_head_counts() {
        // GQA: n_kv_head < n_head. V has fewer heads than Q.
        // V rotation applies to n_kv heads; output inverse applies to n_head heads.
        let head_dim = 4;
        let pos = 7;
        let freq = make_freq_table(10000.0, head_dim);

        // V: 2 kv heads × 4 dim = 8 elements
        let mut v = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
        apply_rope_values(&mut v, pos, head_dim, &freq);

        // Each kv head rotated independently — head 0 and head 1 start different,
        // but the rotation is the same. Verify head 0 rotated correctly.
        let angle0 = pos as f32 * freq[0]; // freq[0] = 1/theta^0 = 1.0
        let expected_h0_0 = 1.0 * angle0.cos() - 0.0 * angle0.sin();
        assert!(
            (v[0] - expected_h0_0).abs() < 1e-5,
            "v[0]={}, expected={}",
            v[0],
            expected_h0_0
        );

        // head 1 starts with 2.0 — verify it's rotated by the same angle
        let expected_h1_0 = 2.0 * angle0.cos() - 0.0 * angle0.sin();
        assert!(
            (v[4] - expected_h1_0).abs() < 1e-5,
            "v[4]={}, expected={}",
            v[4],
            expected_h1_0
        );
    }
}

/// Undo llama.cpp's Q/K row permutation for `llama`-architecture GGUFs.
///
/// `convert_hf_to_gguf.py` (`LlamaModel.permute`) stores Q/K for the
/// INTERLEAVED RoPE convention: within each head, GGUF row `2j + s` is HF row
/// `s·(hd/2) + j`. This crate's RoPE is rotate-half (pairs `i` with
/// `i + hd/2`, HF's `rotate_half`), so the rows go back to HF order here.
/// Without it every position `> 0` rotates the wrong dimension pairs — finite,
/// plausible, and wrong.
///
/// `w` is row-major `[n_heads · head_dim, n_in]`.
pub fn unpermute_interleaved_rows(w: Vec<f32>, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let rows = n_heads * head_dim;
    assert!(
        head_dim.is_multiple_of(2) && rows > 0 && w.len().is_multiple_of(rows),
        "unpermute_interleaved_rows: shape"
    );
    let n_in = w.len() / rows;
    let half = head_dim / 2;
    let mut out = vec![0.0f32; w.len()];
    for h in 0..n_heads {
        for j in 0..half {
            for s in 0..2 {
                let src = (h * head_dim + 2 * j + s) * n_in;
                let dst = (h * head_dim + s * half + j) * n_in;
                out[dst..dst + n_in].copy_from_slice(&w[src..src + n_in]);
            }
        }
    }
    out
}

#[cfg(test)]
mod unpermute_tests {
    use super::unpermute_interleaved_rows;

    /// llama.cpp `LlamaModel.permute`, transcribed: `reshape(n_head, 2,
    /// hd/2, n_in).swapaxes(1, 2)` — HF row `s·half + j` lands at GGUF row
    /// `2j + s`.
    fn permute(w: &[f32], n_heads: usize, hd: usize) -> Vec<f32> {
        let n_in = w.len() / (n_heads * hd);
        let half = hd / 2;
        let mut out = vec![0.0; w.len()];
        for h in 0..n_heads {
            for s in 0..2 {
                for j in 0..half {
                    let src = (h * hd + s * half + j) * n_in;
                    let dst = (h * hd + 2 * j + s) * n_in;
                    out[dst..dst + n_in].copy_from_slice(&w[src..src + n_in]);
                }
            }
        }
        out
    }

    /// The direction, pinned on a head where the map is NOT an involution
    /// (`hd = 6`; at `hd = 4` both directions agree): GGUF rows
    /// `[0, 3, 1, 4, 2, 5]` go back to HF `[0..6)`.
    #[test]
    fn unpermute_restores_hf_row_order() {
        let gguf: Vec<f32> = [0.0, 3.0, 1.0, 4.0, 2.0, 5.0].to_vec();
        assert_eq!(
            unpermute_interleaved_rows(gguf, 1, 6),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }

    /// Inverse of the converter's permutation for multi-head, multi-column
    /// (GQA-shaped) weights.
    #[test]
    fn unpermute_inverts_the_converter() {
        let (n_heads, hd, n_in) = (3, 8, 5);
        let w: Vec<f32> = (0..n_heads * hd * n_in).map(|i| i as f32).collect();
        assert_eq!(
            unpermute_interleaved_rows(permute(&w, n_heads, hd), n_heads, hd),
            w
        );
    }
}
