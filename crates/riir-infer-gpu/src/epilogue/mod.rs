//! CODA-inspired epilogue fusion infrastructure for CubeCL GEMV decode (Plan 106 Track 3).
//!
//! Provides composable post-GEMV operation types that eliminate separate GPU dispatches
//! by fusing elementwise operations into the GEMV kernel's output write phase.
//!
//! # Background
//!
//! CODA (Research 67) reparameterizes Transformer computation as GEMM-plus-epilogue
//! programs. The key insight: memory-bound operations (normalization, residual adds,
//! activations) can execute *while the GEMV output is still in registers*, avoiding
//! global memory round-trips.
//!
//! For our CubeCL decode path (batch_size=1), each GEMV produces one output vector.
//! The plane cooperative kernel has lane 0 hold the final dot product for each row.
//! We can apply additional elementwise operations at this point without a separate
//! dispatch — lane 0 reads residual[row], adds to dot product, writes result.
//!
//! # Architecture
//!
//! ```text
//! GEMV Plane Kernel (per row):
//!   1. Cooperative dot product: plane_sum(partial)
//!   2. Epilogue (lane 0 only):
//!      a. Read residual[row] (optional)
//!      b. result = dot + residual[row]
//!      c. Write result
//!
//! RMSNorm + ResidualAdd Fusion:
//!   Single dispatch: rmsnorm(input) → output[i] = normed[i] + residual[i]
//!
//! CODA Delayed RMSNorm (future):
//!   GEMV with row-scale: result[row] = r[row] * dot(weight_row, input)
//! ```
//!
//! # Fusion Savings
//!
//! | Fusion | Before | After | Savings/layer | Savings/26 layers |
//! |--------|--------|-------|---------------|-------------------|
//! | GEMV + ResidualAdd | 2 dispatches | 1 | 1 | 26 |
//! | RMSNorm + ResidualAdd | 2 dispatches | 1 | 2 | 52 |
//! | GEMV + RowScale (CODA) | 2 dispatches | 1 | 1 | 26 |
//!
//! Total potential: ~5 dispatches saved per layer (19 → 14), ~26.3% reduction.
//!
//! # CubeCL v0.10 Constraints
//!
//! - No generics in `#[cube]` kernels — each fused variant is a separate kernel function
//! - Limited Array parameters (~4-5 per kernel) — pack params into combined buffers
//! - No cross-workgroup sync — epilogue must be workgroup-local
//! - `launch_unchecked` for performance — caller must ensure buffer sizes
//!
//! # References
//!
//! - CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs (Research 67)
//! - Plan 106 Track 3: CODA Epilogue Fusion

// ---------------------------------------------------------------------------
// Submodules: fused kernel implementations
// ---------------------------------------------------------------------------

// Fused RMSNorm + ResidualAdd kernel (Track 3 T2.10/T2.11).
// Single dispatch: output[i] = rmsnorm(input)[i] + residual[i].
#[cfg(feature = "cubecl_runtime")]
mod norm_residual_cubecl;
#[cfg(feature = "cubecl_runtime")]
#[allow(unused_imports)] // Consumer is GPU scaffolding, not yet wired (Issue 429 clippy).
pub use norm_residual_cubecl::NormResidualBatchedCubeCL;
#[cfg(feature = "cubecl_runtime")]
pub use norm_residual_cubecl::NormResidualCubeCL;

// Fused GEMV + ResidualAdd kernels (Track 3 T2.10/T2.12).
// Single dispatch: output[row] = dot(weight_row, input) + residual[row].
// F32 and F16 weight variants.
#[cfg(feature = "cubecl_runtime")]
mod gemv_residual_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub use gemv_residual_cubecl::{GemvResidualCubeCL, GemvResidualF16CubeCL};

// CODA epilogue primitives (Track 3 T2.11).
// Building blocks for delayed RMSNorm reparameterization.
// - PartialRmsCubeCL: compute inv_rms scale factor
// - NormWeightScaleCubeCL: apply gamma scaling
// - RowScaleCubeCL: apply delayed RMS scale
// - SwigluCubeCL: SwiGLU activation
#[cfg(feature = "cubecl_runtime")]
mod coda_primitives_cubecl;
#[cfg(feature = "cubecl_runtime")]
pub use coda_primitives_cubecl::SwigluCubeCL;

// ---------------------------------------------------------------------------
// Epilogue type definitions
// ---------------------------------------------------------------------------

/// Epilogue operations that can be fused into a GEMV kernel's output write.
///
/// Each variant describes what additional work happens after the dot product
/// is computed but before the result is written to global memory.
///
/// # CubeCL Implementation Note
///
/// Since CubeCL v0.10 doesn't support generic kernel parameterization,
/// each variant maps to a separate compiled kernel function. The enum is used
/// at the Rust level to organize dispatch logic and kernel selection.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemvEpilogue {
    /// No epilogue — standard GEMV: `output[row] = dot(weight_row, input)`.
    None,

    /// Residual add: `output[row] = dot(weight_row, input) + residual[row]`.
    ///
    /// Requires 1 additional input buffer: `residual: Array<f32>`.
    ///
    /// Used after Wo (post-attention) and down (post-MLP) projections
    /// where the GEMV output is immediately added to a residual connection.
    ResidualAdd,

    /// Row-scale: `output[row] = scale[row] * dot(weight_row, input)`.
    ///
    /// Requires 1 additional input buffer: `scale: Array<f32>`.
    ///
    /// Used for CODA delayed RMSNorm: the RMS scale factor `r` is applied
    /// *after* the GEMV instead of *before* (as a separate RMSNorm kernel).
    /// The algebraic identity: `RMSNorm(x) @ W = r * (x * gamma) @ W`.
    RowScale,

    /// Row-scale + residual add: `output[row] = scale[row] * dot + residual[row]`.
    ///
    /// Requires 2 additional input buffers: `scale: Array<f32>`, `residual: Array<f32>`.
    ///
    /// Combines CODA delayed scale with residual addition in a single fused op.
    /// Used for the full CODA post-attention path:
    /// `hidden = r * (Wo @ attn_out) + residual`.
    RowScaleResidual,
}

impl GemvEpilogue {
    /// Returns the number of additional input buffers required by this epilogue.
    pub fn extra_input_count(self) -> usize {
        match self {
            Self::None => 0,
            Self::ResidualAdd => 1,
            Self::RowScale => 1,
            Self::RowScaleResidual => 2,
        }
    }

    /// Returns the total Array parameter count for a GEMV kernel with this epilogue.
    ///
    /// Base GEMV uses 3 Arrays (weight, input, output).
    /// Each epilogue adds extra input buffers.
    pub fn total_array_count(self) -> usize {
        3 + self.extra_input_count()
    }

    /// Returns true if this epilogue needs a residual input buffer.
    pub fn needs_residual(self) -> bool {
        matches!(self, Self::ResidualAdd | Self::RowScaleResidual)
    }

    /// Returns true if this epilogue needs a row-scale input buffer.
    pub fn needs_scale(self) -> bool {
        matches!(self, Self::RowScale | Self::RowScaleResidual)
    }
}

impl std::fmt::Display for GemvEpilogue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::ResidualAdd => write!(f, "+residual"),
            Self::RowScale => write!(f, "×scale"),
            Self::RowScaleResidual => write!(f, "×scale+residual"),
        }
    }
}

/// Normalization epilogue operations that can be fused together.
///
/// In Gemma 2's post-norm architecture, each sub-layer output goes through:
/// `rmsnorm(sublayer_output) + residual` — two separate dispatches that can be fused.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormEpilogue {
    /// Standard RMSNorm only: `output[i] = input[i] * inv_rms * gamma[i]`.
    None,

    /// RMSNorm + ResidualAdd: `output[i] = rmsnorm(input)[i] + residual[i]`.
    ///
    /// This is Gemma 2's post-attention and post-MLP pattern:
    /// ```text
    /// wo_out = GEMV(Wo, attn_out)
    /// normed = rmsnorm(wo_out, post_attn_norm)  // dispatch 1
    /// hidden = normed + residual                 // dispatch 2
    /// ```
    /// Fused into: `hidden[i] = rmsnorm(wo_out)[i] + residual[i]` (1 dispatch).
    ResidualAdd,
}

impl NormEpilogue {
    /// Returns the number of additional input buffers required.
    pub fn extra_input_count(self) -> usize {
        match self {
            Self::None => 0,
            Self::ResidualAdd => 1,
        }
    }

    /// Returns true if this epilogue needs a residual input buffer.
    pub fn needs_residual(self) -> bool {
        matches!(self, Self::ResidualAdd)
    }
}

impl std::fmt::Display for NormEpilogue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "norm_only"),
            Self::ResidualAdd => write!(f, "norm+residual"),
        }
    }
}

/// CODA algebraic reparameterization descriptor.
///
/// Describes how RMSNorm can be algebraically delayed past a GEMV operation.
/// The key identity (CODA §3.2.1):
///
/// ```text
/// RMSNorm(x @ W + z) * γ @ W' = r * ((x @ W + z) * γ) @ W'
/// ```
///
/// Where `r` is a per-row scalar that commutes with the matrix multiply.
/// This lets us compute `r` once and apply it *after* the next GEMV,
/// eliminating a separate RMSNorm dispatch.
///
/// # Decode Path Application
///
/// For Gemma 2 post-attention → MLP transition:
///
/// ```text
/// Standard (2 dispatches):
///   normed = rmsnorm(hidden, pre_mlp_norm)
///   gate = gemv(W_gate, normed)
///
/// CODA (2 dispatches, but r is computed en passant):
///   D, r = rmsnorm_with_r(hidden, pre_mlp_norm)  // compute D=hidden*gamma, r=1/sqrt(mean(hidden²)+eps)
///   gate = gemv_scaled(W_gate, D, r)              // r * (D @ W_gate)
/// ```
///
/// The savings come when multiple GEMVs share the same `r`:
/// ```text
/// Standard: rmsnorm(1) + gate_gemv(1) + up_gemv(1) = 3 dispatches
/// CODA:     norm_with_r(1) + scaled_gate(1) + scaled_up(1) = 3 dispatches
/// ```
///
/// Wait — same count? The savings come from **fusing r into the last GEMV of
/// the group** and **eliminating the pre-GEMV rmsnorm write**. On GPU, the win
/// is avoiding the global memory write of the rmsnormed vector (which the next
/// GEMV would immediately read back). With CubeCL handle-to-handle passing,
/// this write is already cheap. The main benefit is reducing CubeCL dispatch
/// overhead (command buffer submissions) and kernel launch latency.
#[derive(Debug, Clone, Copy)]
pub struct CodaReparam {
    /// The delayed RMS scale factor `r = 1 / sqrt(mean(x²) + eps)`.
    /// Applied as a row-scale after the target GEMV.
    pub rms_scale: bool,

    /// The norm-weight gamma scale applied before the target GEMV.
    /// When `rms_scale` is true, this replaces the pre-GEMV rmsnorm.
    pub gamma_scale: bool,
}

impl CodaReparam {
    /// No CODA reparameterization — standard execution order.
    pub const NONE: Self = Self {
        rms_scale: false,
        gamma_scale: false,
    };

    /// Full CODA: delay RMS scale past GEMV, apply gamma scale before.
    pub const FULL: Self = Self {
        rms_scale: true,
        gamma_scale: true,
    };

    /// Only apply gamma scale (partial CODA, for pre-fusion testing).
    pub const GAMMA_ONLY: Self = Self {
        rms_scale: false,
        gamma_scale: true,
    };
}

/// Dispatch reduction estimate for a forward layer with and without epilogue fusion.
///
/// Tracks the number of GPU dispatches (kernel launches) per transformer layer
/// to quantify the impact of epilogue fusion optimizations.
#[derive(Debug, Clone, Copy)]
pub struct DispatchBudget {
    /// Attention section: rmsnorm + QKV GEMV + RoPE + KV store + attention.
    pub attention: u32,

    /// Post-attention: Wo GEMV + rmsnorm + residual add.
    pub post_attention: u32,

    /// MLP section: rmsnorm + gate/up GEMV + GeGLU + down GEMV + rmsnorm + residual add.
    pub mlp: u32,

    /// Total per layer.
    pub total: u32,
}

impl DispatchBudget {
    /// Current (pre-fusion) dispatch count per layer.
    ///
    /// ```text
    /// Attention:
    ///   1. rmsnorm(input_norm)
    ///   2. gemv Q
    ///   3. gemv K
    ///   4. gemv V
    ///   5. rope Q
    ///   6. rope K
    ///   7. kv_store
    ///   8. kv_compact
    ///   9. attention
    ///
    /// Post-attention:
    ///   10. gemv Wo
    ///   11. rmsnorm(post_attn_norm)
    ///   12. residual_add
    ///
    /// MLP:
    ///   13. rmsnorm(pre_mlp_norm)
    ///   14. gemv gate
    ///   15. gemv up
    ///   16. geglu
    ///   17. gemv down
    ///   18. rmsnorm(post_mlp_norm)
    ///   19. residual_add
    /// ```
    pub const BASELINE: Self = Self {
        attention: 9,
        post_attention: 3,
        mlp: 7,
        total: 19,
    };

    /// After fusing RMSNorm + ResidualAdd (post-attention and post-MLP).
    ///
    /// Saves 2 dispatches per layer (steps 11+12 → 1, steps 18+19 → 1).
    pub const FUSED_NORM_RESIDUAL: Self = Self {
        attention: 9,
        post_attention: 2, // gemv Wo, fused(norm + residual)
        mlp: 6,            // rmsnorm, gate, up, geglu, down, fused(norm + residual)
        total: 17,
    };

    /// After also fusing GEMV + ResidualAdd for Wo and down projections.
    ///
    /// Saves 2 more dispatches per layer (Wo+residual fused, down+residual fused).
    pub const FUSED_GEMV_RESIDUAL: Self = Self {
        attention: 9,
        post_attention: 1, // fused(gemv Wo + rmsnorm + residual)
        mlp: 5,            // rmsnorm, gate, up, geglu, fused(gemv down + rmsnorm + residual)
        total: 15,
    };

    /// Full CODA reparameterization target.
    ///
    /// ```text
    /// Attention:     9 dispatches (unchanged)
    /// Post-attn:     1 dispatch  (fused gemv Wo + rmsnorm + residual)
    /// MLP:           4 dispatches (norm_with_r, fused(gate+up), geglu, fused(down+r_scale+rmsnorm+residual))
    /// ```
    pub const CODA_TARGET: Self = Self {
        attention: 9,
        post_attention: 1,
        mlp: 4,
        total: 14,
    };

    /// Total dispatches for 26 layers.
    pub fn total_26_layers(self) -> u32 {
        self.total * 26
    }

    /// Reduction percentage vs baseline.
    pub fn reduction_vs_baseline(self) -> f32 {
        let baseline = Self::BASELINE.total as f32;
        (baseline - self.total as f32) / baseline * 100.0
    }
}

impl std::fmt::Display for DispatchBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "attn={} post_attn={} mlp={} total={} (26 layers: {})",
            self.attention,
            self.post_attention,
            self.mlp,
            self.total,
            self.total_26_layers()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epilogue_extra_inputs() {
        assert_eq!(GemvEpilogue::None.extra_input_count(), 0);
        assert_eq!(GemvEpilogue::ResidualAdd.extra_input_count(), 1);
        assert_eq!(GemvEpilogue::RowScale.extra_input_count(), 1);
        assert_eq!(GemvEpilogue::RowScaleResidual.extra_input_count(), 2);
    }

    #[test]
    fn test_epilogue_needs() {
        assert!(!GemvEpilogue::None.needs_residual());
        assert!(GemvEpilogue::ResidualAdd.needs_residual());
        assert!(!GemvEpilogue::RowScale.needs_residual());
        assert!(GemvEpilogue::RowScaleResidual.needs_residual());

        assert!(!GemvEpilogue::None.needs_scale());
        assert!(!GemvEpilogue::ResidualAdd.needs_scale());
        assert!(GemvEpilogue::RowScale.needs_scale());
        assert!(GemvEpilogue::RowScaleResidual.needs_scale());
    }

    #[test]
    fn test_epilogue_total_arrays() {
        // Base GEMV: weight, input, output = 3
        assert_eq!(GemvEpilogue::None.total_array_count(), 3);
        assert_eq!(GemvEpilogue::ResidualAdd.total_array_count(), 4);
        assert_eq!(GemvEpilogue::RowScale.total_array_count(), 4);
        assert_eq!(GemvEpilogue::RowScaleResidual.total_array_count(), 5);
    }

    #[test]
    fn test_norm_epilogue() {
        assert_eq!(NormEpilogue::None.extra_input_count(), 0);
        assert_eq!(NormEpilogue::ResidualAdd.extra_input_count(), 1);
        assert!(!NormEpilogue::None.needs_residual());
        assert!(NormEpilogue::ResidualAdd.needs_residual());
    }

    #[test]
    fn test_dispatch_budget_baseline() {
        let baseline = DispatchBudget::BASELINE;
        assert_eq!(baseline.total, 19);
        assert_eq!(baseline.total_26_layers(), 494);
    }

    #[test]
    fn test_dispatch_budget_fused_norm_residual() {
        let fused = DispatchBudget::FUSED_NORM_RESIDUAL;
        assert_eq!(fused.total, 17);
        assert_eq!(fused.total_26_layers(), 442);
        // 2 dispatches saved per layer = 52 total
        assert_eq!(
            DispatchBudget::BASELINE.total_26_layers() - fused.total_26_layers(),
            52
        );
    }

    #[test]
    fn test_dispatch_budget_fused_gemv_residual() {
        let fused = DispatchBudget::FUSED_GEMV_RESIDUAL;
        assert_eq!(fused.total, 15);
        assert_eq!(fused.total_26_layers(), 390);
        // 4 dispatches saved per layer = 104 total
        assert_eq!(
            DispatchBudget::BASELINE.total_26_layers() - fused.total_26_layers(),
            104
        );
    }

    #[test]
    fn test_dispatch_budget_coda_target() {
        let coda = DispatchBudget::CODA_TARGET;
        assert_eq!(coda.total, 14);
        assert_eq!(coda.total_26_layers(), 364);
        // 5 dispatches saved per layer = 130 total
        assert_eq!(
            DispatchBudget::BASELINE.total_26_layers() - coda.total_26_layers(),
            130
        );
    }

    #[test]
    fn test_reduction_percentages() {
        assert_eq!(DispatchBudget::BASELINE.reduction_vs_baseline(), 0.0);
        assert!((DispatchBudget::FUSED_NORM_RESIDUAL.reduction_vs_baseline() - 10.53).abs() < 0.1);
        assert!((DispatchBudget::FUSED_GEMV_RESIDUAL.reduction_vs_baseline() - 21.05).abs() < 0.1);
        assert!((DispatchBudget::CODA_TARGET.reduction_vs_baseline() - 26.32).abs() < 0.1);
    }

    #[test]
    fn test_coda_reparam() {
        const { assert!(!CodaReparam::NONE.rms_scale) };
        const { assert!(!CodaReparam::NONE.gamma_scale) };
        const { assert!(CodaReparam::FULL.rms_scale) };
        const { assert!(CodaReparam::FULL.gamma_scale) };
        const { assert!(!CodaReparam::GAMMA_ONLY.rms_scale) };
        const { assert!(CodaReparam::GAMMA_ONLY.gamma_scale) };
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", GemvEpilogue::None), "none");
        assert_eq!(format!("{}", GemvEpilogue::ResidualAdd), "+residual");
        assert_eq!(format!("{}", GemvEpilogue::RowScale), "×scale");
        assert_eq!(
            format!("{}", GemvEpilogue::RowScaleResidual),
            "×scale+residual"
        );
        assert_eq!(format!("{}", NormEpilogue::None), "norm_only");
        assert_eq!(format!("{}", NormEpilogue::ResidualAdd), "norm+residual");
    }
}
