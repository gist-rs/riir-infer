//! Quantization formats for inference weight compression.
//!
//! Implements k-quant formats from the GGML ecosystem for reduced memory
//! bandwidth during GPU decode inference.

pub mod ptq1_0;
pub mod q2_0;
pub mod q2k;
pub mod q3k;
pub mod q4k;
pub mod q5k;
pub mod q6k;
pub mod q8kv;
