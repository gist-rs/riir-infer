//! Q4_K quantized weight buffers for Gemma 2 GPU inference.
//!
//! Quantizes f32 weights to Q4_K format (4.5 bpw) on upload, reducing
//! GPU memory bandwidth by ~7× during decode. The WGSL GEMV kernel
//! dequantizes on-the-fly, avoiding full f32 weight materialization.

use wgpu::Buffer;

use bytemuck::Zeroable;

use crate::buffer::upload_bytes;
use crate::context::GpuError;
use riir_infer_core::gemma_layer::GemmaTransformerWeights;
use riir_infer_core::quant::q4k::{BlockQ4K, QK_K, quantize_row_q4_k};
use riir_infer_core::types::Config;

/// Per-layer Q4_K quantized weight buffers.
///
/// Each projection is stored as packed `BlockQ4K` bytes uploaded as a raw
/// `array<u32>` storage buffer. The WGSL `gemv_q4k` kernel unpacks and
/// dequantizes on-the-fly during the dot product.
pub struct GpuGemmaLayerWeightsQ4K {
    pub attn_wq: Buffer,
    pub attn_wk: Buffer,
    pub attn_wv: Buffer,
    pub attn_wo: Buffer,
    pub gate_proj: Buffer,
    pub up_proj: Buffer,
    pub down_proj: Buffer,
}

/// All Gemma 2 Q4_K quantized weight buffers on GPU.
pub struct GpuGemmaWeightBuffersQ4K {
    /// Token embedding quantized as Q4_K — used for tied lm_head matmul.
    pub wte: Buffer,
    /// Per-layer Q4_K quantized projection weights.
    pub layers: Vec<GpuGemmaLayerWeightsQ4K>,
}

/// GGUF tensor names for a single layer.
struct GgufLayerNames {
    attn_q: String,
    attn_k: String,
    attn_v: String,
    attn_output: String,
    gate_proj: String,
    up_proj: String,
    down_proj: String,
}

impl GpuGemmaWeightBuffersQ4K {
    /// Quantize and upload Gemma 2 weights to GPU as Q4_K blocks.
    ///
    /// All projection weights are quantized to Q4_K (4.5 bpw) before upload.
    /// The host-side embedding lookup uses the f32 set's `wte_cpu`
    /// (`GpuGemmaWeightBuffers::norms_only`, Issue 720) — this struct no
    /// longer keeps its own f32 embedding clone (it had zero readers).
    pub fn from_weights(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weights: &GemmaTransformerWeights,
        config: &Config,
    ) -> Self {
        // Quantize and upload tied embedding (used as lm_head)
        let wte_q4k = quantize_projection(&weights.wte, config.vocab_size, config.n_embd);
        let wte = upload_bytes(
            device,
            queue,
            bytemuck::cast_slice(&wte_q4k),
            "gemma2_q4k_wte",
        );

        // Quantize and upload per-layer weights
        let layers = weights
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let q_dim = config.n_head * config.head_dim;
                let kv_dim = config.n_kv_head * config.head_dim;
                let n = config.n_embd;
                let mlp = config.mlp_hidden;

                GpuGemmaLayerWeightsQ4K {
                    attn_wq: upload_projection(
                        device,
                        queue,
                        &l.attn_wq,
                        q_dim,
                        n,
                        &format!("gemma2_q4k_l{i}_wq"),
                    ),
                    attn_wk: upload_projection(
                        device,
                        queue,
                        &l.attn_wk,
                        kv_dim,
                        n,
                        &format!("gemma2_q4k_l{i}_wk"),
                    ),
                    attn_wv: upload_projection(
                        device,
                        queue,
                        &l.attn_wv,
                        kv_dim,
                        n,
                        &format!("gemma2_q4k_l{i}_wv"),
                    ),
                    attn_wo: upload_projection(
                        device,
                        queue,
                        &l.attn_wo,
                        n,
                        q_dim,
                        &format!("gemma2_q4k_l{i}_wo"),
                    ),
                    gate_proj: upload_projection(
                        device,
                        queue,
                        &l.gate_proj,
                        mlp,
                        n,
                        &format!("gemma2_q4k_l{i}_gate"),
                    ),
                    up_proj: upload_projection(
                        device,
                        queue,
                        &l.up_proj,
                        mlp,
                        n,
                        &format!("gemma2_q4k_l{i}_up"),
                    ),
                    down_proj: upload_projection(
                        device,
                        queue,
                        &l.down_proj,
                        n,
                        mlp,
                        &format!("gemma2_q4k_l{i}_down"),
                    ),
                }
            })
            .collect();

        Self {
            wte,
            layers,
        }
    }

    /// Load Q4_K weights directly from a GGUF file via zero-copy mmap upload.
    ///
    /// For Q4_K tensors, the mmap'd bytes are uploaded directly to GPU —
    /// no CPU-side dequantization or re-quantization needed.
    pub fn from_gguf(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gguf: &riir_infer_core::gguf_loader::GgufFile,
        config: &Config,
    ) -> Result<Self, GpuError> {
        use riir_infer_core::gguf_loader::GgmlType;

        // Upload embedding — Q4_K zero-copy from mmap
        let wte_slice = gguf
            .tensor_slice("token_embd.weight")
            .ok_or_else(|| GpuError::ShaderError("token_embd.weight not found in GGUF".into()))?;
        let wte = upload_bytes(device, queue, wte_slice, "gemma2_q4k_wte_gguf");

        // Upload per-layer weights
        let mut layers = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            // GGUF tensor names for Gemma 2
            let ln = GgufLayerNames {
                attn_q: format!("blk.{i}.attn_q.weight"),
                attn_k: format!("blk.{i}.attn_k.weight"),
                attn_v: format!("blk.{i}.attn_v.weight"),
                attn_output: format!("blk.{i}.attn_output.weight"),
                gate_proj: format!("blk.{i}.ffn_gate.weight"),
                up_proj: format!("blk.{i}.ffn_up.weight"),
                down_proj: format!("blk.{i}.ffn_down.weight"),
            };

            // Validate each tensor is Q4_K
            for name in [
                &ln.attn_q,
                &ln.attn_k,
                &ln.attn_v,
                &ln.attn_output,
                &ln.gate_proj,
                &ln.up_proj,
                &ln.down_proj,
            ] {
                if let Some(info) = gguf.tensor_info(name) {
                    if info.ggml_type != GgmlType::Q4_K {
                        return Err(GpuError::ShaderError(format!(
                            "tensor '{name}' is {:?}, expected Q4_K",
                            info.ggml_type
                        )));
                    }
                } else {
                    return Err(GpuError::ShaderError(format!(
                        "tensor '{name}' not found in GGUF file"
                    )));
                }
            }

            // Zero-copy upload: mmap slice → GPU buffer
            let upload_slice = |name: &str, label: &str| -> Result<Buffer, GpuError> {
                let slice = gguf
                    .tensor_slice(name)
                    .ok_or_else(|| GpuError::ShaderError(format!("tensor '{name}' not found")))?;
                Ok(upload_bytes(device, queue, slice, label))
            };

            layers.push(GpuGemmaLayerWeightsQ4K {
                attn_wq: upload_slice(&ln.attn_q, &format!("gguf_q4k_l{i}_wq"))?,
                attn_wk: upload_slice(&ln.attn_k, &format!("gguf_q4k_l{i}_wk"))?,
                attn_wv: upload_slice(&ln.attn_v, &format!("gguf_q4k_l{i}_wv"))?,
                attn_wo: upload_slice(&ln.attn_output, &format!("gguf_q4k_l{i}_wo"))?,
                gate_proj: upload_slice(&ln.gate_proj, &format!("gguf_q4k_l{i}_gate"))?,
                up_proj: upload_slice(&ln.up_proj, &format!("gguf_q4k_l{i}_up"))?,
                down_proj: upload_slice(&ln.down_proj, &format!("gguf_q4k_l{i}_down"))?,
            });
        }

        Ok(Self {
            wte,
            layers,
        })
    }

    /// Calculate total GPU memory used by Q4_K weights (bytes).
    pub fn gpu_bytes(&self, config: &Config) -> u64 {
        let q_dim = config.n_head * config.head_dim;
        let kv_dim = config.n_kv_head * config.head_dim;
        let n = config.n_embd;
        let mlp = config.mlp_hidden;

        // bytes per row = (n_cols / 256) * 144
        let row_bytes = |cols: usize| -> u64 { cols.div_ceil(QK_K) as u64 * 144 };

        let per_layer = row_bytes(n) * (q_dim + 2 * kv_dim) as u64  // wq + wk + wv
            + row_bytes(q_dim) * n as u64                           // wo
            + row_bytes(n) * (2 * mlp) as u64                       // gate + up
            + row_bytes(mlp) * n as u64; // down

        let wte_bytes = row_bytes(n) * config.vocab_size as u64;

        wte_bytes + per_layer * config.n_layer as u64
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Quantize a projection matrix [rows, cols] row-by-row to Q4_K blocks.
///
/// Returns packed `BlockQ4K` array ready for GPU upload.
/// Each row of `cols` elements produces `ceil(cols / 256)` blocks.
fn quantize_projection(data: &[f32], rows: usize, cols: usize) -> Vec<BlockQ4K> {
    assert_eq!(data.len(), rows * cols, "Projection size mismatch");
    let blocks_per_row = cols.div_ceil(QK_K);
    let total_blocks = rows * blocks_per_row;
    let mut blocks = Vec::with_capacity(total_blocks);

    // Pad row to multiple of QK_K if needed
    let padded_cols = blocks_per_row * QK_K;
    let mut padded_row = vec![0.0f32; padded_cols];

    for row in 0..rows {
        let src = &data[row * cols..(row + 1) * cols];
        padded_row[..cols].copy_from_slice(src);
        // Zero-pad tail (already zeroed from allocation, just need to clear
        // if reused across iterations with different data)
        if padded_cols > cols {
            padded_row[cols..].fill(0.0);
        }

        let start = blocks.len();
        blocks.resize(start + blocks_per_row, BlockQ4K::zeroed());

        quantize_row_q4_k(&padded_row, &mut blocks[start..]);
    }

    blocks
}

/// Quantize and upload a projection matrix to GPU as Q4_K packed bytes.
fn upload_projection(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[f32],
    rows: usize,
    cols: usize,
    label: &str,
) -> Buffer {
    let blocks = quantize_projection(data, rows, cols);
    let bytes = bytemuck::cast_slice(&blocks);
    upload_bytes(device, queue, bytes, label)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_quantize_projection_dims() {
        // Simulate a small projection: 8 rows × 256 cols = 1 block per row
        let data = vec![0.5f32; 8 * 256];
        let blocks = quantize_projection(&data, 8, 256);
        assert_eq!(blocks.len(), 8, "Should produce 8 blocks (1 per row)");
    }

    #[test]
    fn test_quantize_projection_padded() {
        // 4 rows × 300 cols → 2 blocks per row (ceil(300/256) = 2)
        let data = vec![0.3f32; 4 * 300];
        let blocks = quantize_projection(&data, 4, 300);
        assert_eq!(
            blocks.len(),
            8,
            "Should produce 8 blocks (2 per row × 4 rows)"
        );
    }

    #[test]
    fn test_quantize_projection_roundtrip() {
        let rows = 4;
        let cols = 512;
        let original: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.1).sin()) * 2.0)
            .collect();

        let blocks = quantize_projection(&original, rows, cols);

        // Dequantize and verify
        let mut reconstructed = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let row_blocks = &blocks[row * 2..(row + 1) * 2];
            let mut row_data = vec![0.0f32; 512];
            riir_infer_core::quant::q4k::dequantize_row_q4_k(row_blocks, &mut row_data);
            reconstructed.extend_from_slice(&row_data[..cols]);
        }

        let mut max_err = 0.0f32;
        for (&orig, &deq) in original.iter().zip(reconstructed.iter()) {
            let err = (orig - deq).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 0.5, "Roundtrip max error too large: {max_err}");
    }

    #[test]
    fn test_gpu_memory_estimate() {
        // Gemma 2 2B: 26 layers, projections sum to known size
        let config = Config::gemma2_2b();
        let n = config.n_embd; // 2304
        let q_dim = config.n_head * config.head_dim; // 2048
        let kv_dim = config.n_kv_head * config.head_dim; // 1024
        let mlp = config.mlp_hidden; // 9216
        let vocab = config.vocab_size; // 256000

        // Calculate expected Q4_K size
        let row_bytes = |cols: usize| -> u64 { cols.div_ceil(256) as u64 * 144 };

        // Per layer: 7 projections
        let per_layer = row_bytes(n) * (q_dim + 2 * kv_dim) as u64  // wq + wk + wv
            + row_bytes(q_dim) * n as u64                           // wo
            + row_bytes(n) * (2 * mlp) as u64                       // gate + up
            + row_bytes(mlp) * n as u64; // down

        // WTE (tied lm_head)
        let wte_bytes = row_bytes(n) * vocab as u64;

        let total = wte_bytes + per_layer * config.n_layer as u64;
        let total_mb = total as f64 / (1024.0 * 1024.0);

        // Should be ~1.5 GB (vs ~10.4 GB for f32)
        assert!(
            total_mb < 2000.0,
            "Q4_K memory estimate too high: {total_mb:.0} MB"
        );
        assert!(
            total_mb > 500.0,
            "Q4_K memory estimate too low: {total_mb:.0} MB"
        );

        // Verify f32 would be ~7× larger
        let per_layer_f32 = q_dim * n       // wq
            + kv_dim * n                     // wk
            + kv_dim * n                     // wv
            + n * q_dim                      // wo
            + mlp * n                        // gate
            + mlp * n                        // up
            + n * mlp; // down
        let f32_bytes = (vocab * n + config.n_layer * per_layer_f32) as u64 * 4;
        let ratio = f32_bytes as f64 / total as f64;
        assert!(ratio > 6.0, "Compression ratio too low: {ratio:.1}×");
    }

    #[test]
    fn test_block_alignment() {
        // Verify all Gemma 2 projection dimensions are divisible by QK_K (256)
        let config = Config::gemma2_2b();
        let n = config.n_embd;
        let q_dim = config.n_head * config.head_dim;
        let kv_dim = config.n_kv_head * config.head_dim;
        let mlp = config.mlp_hidden;
        let vocab = config.vocab_size;

        assert_eq!(n % QK_K, 0, "n_embd must be divisible by {QK_K}");
        assert_eq!(q_dim % QK_K, 0, "q_dim must be divisible by {QK_K}");
        assert_eq!(kv_dim % QK_K, 0, "kv_dim must be divisible by {QK_K}");
        assert_eq!(mlp % QK_K, 0, "mlp_hidden must be divisible by {QK_K}");
        assert_eq!(vocab % QK_K, 0, "vocab_size must be divisible by {QK_K}");
    }
}
