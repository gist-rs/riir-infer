//! Quantization formats for inference weight compression.
//!
//! GGML-ecosystem k-quants (GGUF side) for reduced memory bandwidth during
//! GPU decode inference — plus, behind the `exl3` feature, the first
//! safetensors-side format: EXL3 trellis-coded weights (Issue 001), kept
//! loader-decoupled from `GgmlType` per the T2 seam decision there.

#[cfg(feature = "exl3")]
pub mod exl3;
#[cfg(feature = "exl3")]
pub mod exl3_pack;
// Issue 036 T1/T2 — the LittleBit-derived init-only sub-1-bit PTQ transform
// (seeded Halko SVD + Dual-SVID init + residual restack). Opt-in (`svd_lbit`);
// measurement-only per the lossy-surface law, never a serving path.
#[cfg(feature = "svd_lbit")]
pub mod svd_lbit;
pub mod ptq1_0;
pub mod q2_0;
pub mod lut_grid;
// Issue 040 — the description-length (MDL floor) audit lane: per-tensor
// histogram entropy of the stored symbols vs stored bits, two-part honest
// (side-info split). Report-only; opt-in per the measurement-only law.
#[cfg(feature = "desc_len")]
pub mod desc_len;
pub mod q2k;
pub mod q3k;
pub mod q4k;
pub mod q5k;
pub mod q6k;
pub mod q8kv;
pub mod kvq_ab;
pub mod kvq_harness;
