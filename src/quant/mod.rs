//! Quantization formats for inference weight compression.
//!
//! GGML-ecosystem k-quants (GGUF side) for reduced memory bandwidth during
//! GPU decode inference — plus, behind the `exl3` feature, the first
//! safetensors-side format: EXL3 trellis-coded weights (Issue 001), kept
//! loader-decoupled from `GgmlType` per the T2 seam decision there.

#[cfg(feature = "exl3")]
pub mod exl3;
pub mod ptq1_0;
pub mod q2_0;
pub mod q2k;
pub mod q3k;
pub mod q4k;
pub mod q5k;
pub mod q6k;
pub mod q8kv;
