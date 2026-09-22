//! SIMD dispatch for inference kernels.
//!
//! All SIMD kernels (NEON, AVX2, WASM SIMD128, scalar) now live in
//! `katgpt_core::simd`. This module is a thin re-export so that
//! `crate::simd::*` paths resolve unchanged after Plan 008 Step 7.
//!
//! # History
//!
//! Previously this module hosted a local `wasm32.rs` with WASM SIMD128
//! reimplementations of dot/sum/scale/exp/outer-product/matvec/ternary
//! kernels, gated behind `#[cfg(target_arch = "wasm32", target_feature =
//! "simd128")]`. The non-WASM path already did `pub use katgpt_core::simd::*;`.
//!
//! Plan 008 Step 7 unified this: the WASM SIMD128 paths were ported into
//! `katgpt_core::simd` (`sum_sq`, `outer_product_acc`, `project_ternary_simd`), the
//! thin wrappers (`dot_f32_simd`, `matmul_f32_simd`) were dropped, and
//! `simd/wasm32.rs` was deleted. Call sites that used the dropped wrappers
//! were updated to use the core API directly.

pub use katgpt_core::simd::*;
