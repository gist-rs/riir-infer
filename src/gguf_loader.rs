//! GGUF v2/v3 parser with mmap zero-copy tensor access.
//!
//! Adapted from `RuVector`'s `ruvector-decompiler/src/model_gguf.rs` (ADR-138).
//! Adds mmap-based zero-copy tensor access, Config extraction for Gemma 2,
//! and F16 dequantization for weight loading.
//!
//! # File Format
//!
//! ```text
//! Header: magic(4) + version(4) + tensor_count(8) + metadata_count(8)
//! Metadata KV pairs: key(string) + value_type(u32) + value(varies)
//! Tensor infos: name(string) + n_dims(u32) + dimensions([u64]) + type(u32) + offset(u64)
//! Padding to ALIGNMENT (default 32)
//! Tensor data: raw bytes (aligned, contiguous)
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::gemma_layer::{
    GemmaLayerWeights, GemmaLayerWeightsF16, GemmaTransformerWeights, GemmaTransformerWeightsF16,
};
use crate::quant::ptq1_0::{BlockPtq1_0, dequantize_row_ptq1_0};
use crate::quant::q2_0::{BlockQ2_0, dequantize_row_q2_0};
use crate::quant::q2k::{BlockQ2K, dequantize_row_q2_k};
use crate::quant::q3k::{BlockQ3K, dequantize_row_q3_k};
use crate::quant::q4k::{BlockQ4K, dequantize_row_q4_k};
use crate::quant::q5k::{BlockQ5K, dequantize_row_q5_k};
use crate::quant::q6k::{BlockQ6K, dequantize_row_q6_k};
use crate::quant::q8kv::{BlockQ8_0, dequantize_row_q8_0};
use crate::safetensors_loader::bf16_to_f32;
#[cfg(feature = "deltanet_inference")]
use crate::types::DeltaNetLayerType;
use crate::types::{Config, ModelArchitecture};

/// GGUF magic number: "GGUF" in little-endian.
const GGUF_MAGIC: u32 = 0x46554747;
/// Default alignment for GGUF files.
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

// ── GGML type IDs ──────────────────────────────────────────────

/// GGML tensor type IDs (from gguf spec).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    /// Ternary-Bonsai 2-bit ternary format (`PrismML` fork). `Q2_0_g128`, NOT
    /// `BitNet`'s `i2_s` — the two are different formats.
    /// Block: f16 scale + 32B packed 2-bit codes = 34B per 128 weights (2.125 bpw).
    /// Plan 333 / katgpt-rs Issue 578.
    Q2_0 = 42,
    /// Prism ternary at group 128, base-3 dense-trit packing (fork commit
    /// `01fd9521c`). 28 B per 128 weights (1.75 bpw) — the Bonsai-2 decode
    /// pack's tensor type. Same trits + f16 group scales as `Q2_0` for a
    /// ternary-g128 checkpoint, denser wire encoding. Issue 980 T6.
    PTQ1_0 = 143,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    /// Upstream ggml assigns `GGML_TYPE_BF16 = 30` (29 is `IQ1_M`). This was
    /// previously mapped to 29, which meant no real BF16 GGUF could be opened
    /// at all — `from_id(30)` returned `None` and `GgufFile::open` failed with
    /// "unknown tensor type". Found via `Ternary-Bonsai-27B-dspark-Q4_1.gguf`,
    /// whose `dspark.markov_head_a.weight` / `log_snr_fc*.weight` are type 30;
    /// confirmed empirically at exactly 2.0 bytes/element (riir-ai Issue 717).
    BF16 = 30,
}

impl GgmlType {
    fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            // 29 is upstream IQ1_M (unsupported); BF16 is 30.
            30 => Some(Self::BF16),
            // Ternary-Bonsai (PrismML fork GGML_TYPE_Q2_0). Legacy files label
            // the group-128 ternary payload as id 42; the fork's PQ2_0 rename
            // (id 142, formerly Q2_0_G128) is byte-identical — same f16 scale +
            // 32B packed 2-bit codes per 128 weights. We relabeled the file to
            // 142 (fork-tip builds refuse id-42 Q2_0), so both ids map here.
            42 | 142 => Some(Self::Q2_0),
            // Prism ternary g128, base-3 dense trits (Bonsai-2 decode pack,
            // Issue 980 T6). Fork commit 01fd9521c.
            143 => Some(Self::PTQ1_0),
            _ => None,
        }
    }

    /// Block size in bytes for this quantization type.
    /// Returns `(block_size_bytes, weights_per_block)`.
    pub fn block_info(&self) -> Option<(usize, usize)> {
        match self {
            Self::F32 => None, // no blocking
            Self::F16 => None,
            Self::Q4_K => Some((144, 256)), // BlockQ4K = 144 bytes per 256 weights
            Self::Q5_K => Some((176, 256)), // BlockQ5K = 176 bytes per 256 weights
            Self::Q6_K => Some((210, 256)), // BlockQ6K = 210 bytes per 256 weights
            Self::Q2_K => Some((84, 256)),  // BlockQ2K = 84 bytes per 256 weights
            Self::Q3_K => Some((110, 256)), // BlockQ3K = 110 bytes per 256 weights
            Self::Q4_0 => Some((18, 32)),
            Self::Q4_1 => Some((20, 32)),
            Self::Q5_0 => Some((22, 32)),
            Self::Q5_1 => Some((24, 32)),
            Self::Q8_0 => Some((34, 32)),
            Self::Q8_1 => Some((36, 32)),
            // Ternary-Bonsai: 34 bytes per 128 weights (2.125 bpw).
            Self::Q2_0 => Some((34, 128)),
            // Prism ternary g128: 24B qs + 2B qh + 2B f16 scale per 128 (1.75 bpw).
            Self::PTQ1_0 => Some((28, 128)),
            _ => None,
        }
    }

    /// Bytes needed for a tensor with `n_elements` values in this format.
    pub fn tensor_bytes(&self, n_elements: usize) -> usize {
        match self {
            Self::F32 => n_elements * 4,
            Self::F16 | Self::BF16 => n_elements * 2,
            Self::Q4_K => {
                let (block_bytes, weights_per_block) = self.block_info().unwrap();
                (n_elements / weights_per_block) * block_bytes
            }
            _ => {
                if let Some((block_bytes, weights_per_block)) = self.block_info() {
                    (n_elements / weights_per_block) * block_bytes
                } else {
                    n_elements * 4 // fallback to f32 size
                }
            }
        }
    }
}

// ── Metadata value type IDs ────────────────────────────────────

const META_TYPE_UINT8: u32 = 0;
const META_TYPE_INT8: u32 = 1;
const META_TYPE_UINT16: u32 = 2;
const META_TYPE_INT16: u32 = 3;
const META_TYPE_UINT32: u32 = 4;
const META_TYPE_INT32: u32 = 5;
const META_TYPE_FLOAT32: u32 = 6;
const META_TYPE_BOOL: u32 = 7;
const META_TYPE_STRING: u32 = 8;
const META_TYPE_ARRAY: u32 = 9;
const META_TYPE_UINT64: u32 = 10;
const META_TYPE_INT64: u32 = 11;
const META_TYPE_FLOAT64: u32 = 12;

// ── Metadata value ─────────────────────────────────────────────

/// GGUF metadata value (simplified for model loading).
#[derive(Clone, Debug)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            Self::I8(v) if *v >= 0 => Some(*v as u64),
            Self::I16(v) if *v >= 0 => Some(*v as u64),
            Self::I32(v) if *v >= 0 => Some(*v as u64),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            // Issue 980: the HF converter writes prism.hadamard.gdn_v_grouped
            // as a GGUF BOOL — coercing it as 0 silently disabled the
            // ssm_out permute on the real file.
            Self::Bool(v) => Some(*v as u64),
            Self::F32(v) => Some(*v as u64),
            Self::F64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F32(v) => Some(*v as f64),
            Self::F64(v) => Some(*v),
            Self::U32(v) => Some(*v as f64),
            Self::U64(v) => Some(*v as f64),
            Self::I32(v) => Some(*v as f64),
            Self::I64(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Bool accessor (GGUF bool KV values — e.g.
    /// `tokenizer.ggml.add_space_prefix`).
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            Self::U8(v) => Some(*v != 0),
            Self::I8(v) => Some(*v != 0),
            Self::U32(v) => Some(*v != 0),
            Self::I32(v) => Some(*v != 0),
            _ => None,
        }
    }
}

// ── Tensor info ────────────────────────────────────────────────

/// Parsed tensor metadata from GGUF file.
#[derive(Clone, Debug)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: GgmlType,
    /// Offset relative to tensor data start (after header + padding).
    pub offset: u64,
    /// Byte offset in the mmap where tensor data begins (absolute).
    pub data_start: u64,
    /// Total bytes of tensor data.
    pub byte_len: u64,
}

impl GgufTensorInfo {
    /// Total number of elements in the tensor.
    #[inline]
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

// ── Parsed GGUF file ──────────────────────────────────────────

/// Parsed GGUF file with mmap-backed zero-copy tensor access.
pub struct GgufFile {
    /// Memory-mapped file data.
    mmap: memmap2::Mmap,
    /// GGUF version (2 or 3).
    pub version: u32,
    /// Parsed metadata key-value pairs.
    pub metadata: HashMap<String, GgufValue>,
    /// Parsed tensor info (indexed by name for O(1) lookup).
    tensor_map: HashMap<String, GgufTensorInfo>,
    /// Tensor info list (in file order).
    pub tensor_infos: Vec<GgufTensorInfo>,
    /// Byte offset in mmap where tensor data section begins.
    #[allow(dead_code)]
    tensor_data_offset: u64,
    /// Alignment (from metadata or default 32).
    #[allow(dead_code)]
    alignment: u64,
}

impl GgufFile {
    /// Open and parse a GGUF file using mmap.
    ///
    /// Only reads the header, metadata, and tensor info sections.
    /// Tensor data is accessed lazily via `tensor_slice()`.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open GGUF file: {}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("failed to mmap GGUF file: {}", path.display()))?;

        let mut cursor: usize = 0;

        // Parse header
        let magic = read_u32(&mmap, &mut cursor)?;
        if magic != GGUF_MAGIC {
            bail!("not a GGUF file (magic: 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x})");
        }

        let version = read_u32(&mmap, &mut cursor)?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version: {version} (expected 2 or 3)");
        }

        let tensor_count = read_u64(&mmap, &mut cursor)?;
        let metadata_count = read_u64(&mmap, &mut cursor)?;

        // Parse metadata KV pairs
        let mut metadata = HashMap::with_capacity(metadata_count as usize);
        for _ in 0..metadata_count {
            let key = read_gguf_string(&mmap, &mut cursor)?;
            let value = read_gguf_value(&mmap, &mut cursor)?;
            metadata.insert(key, value);
        }

        // Parse tensor infos
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        let mut tensor_infos = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mmap, &mut cursor)?;
            let n_dims = read_u32(&mmap, &mut cursor)? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(read_u64(&mmap, &mut cursor)? as usize);
            }
            let type_id = read_u32(&mmap, &mut cursor)?;
            let offset = read_u64(&mmap, &mut cursor)?;

            let ggml_type = GgmlType::from_id(type_id)
                .with_context(|| format!("unknown GGML type {type_id} for tensor '{name}'"))?;

            let n_elements: usize = shape.iter().product();
            let byte_len = ggml_type.tensor_bytes(n_elements) as u64;

            tensor_infos.push(GgufTensorInfo {
                name,
                shape,
                ggml_type,
                offset,
                data_start: 0, // filled after we know tensor_data_offset
                byte_len,
            });
        }

        // Align to tensor data section
        let padding = (alignment - (cursor as u64 % alignment)) % alignment;
        let tensor_data_offset = cursor as u64 + padding;

        // Fill absolute data_start for each tensor
        let mut tensor_map = HashMap::with_capacity(tensor_infos.len());
        for info in &mut tensor_infos {
            info.data_start = tensor_data_offset + info.offset;
            tensor_map.insert(info.name.clone(), info.clone());
        }

        Ok(Self {
            mmap,
            version,
            metadata,
            tensor_map,
            tensor_infos,
            tensor_data_offset,
            alignment,
        })
    }

    /// Get a zero-copy byte slice for a tensor by name.
    ///
    /// Returns `None` if the tensor is not found.
    /// The slice points directly into the mmap'd file data.
    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensor_map.get(name)?;
        let start = info.data_start as usize;
        let end = start + info.byte_len as usize;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[start..end])
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_map.get(name)
    }

    /// Get raw `Q4_K` blocks for a tensor (zero-copy from the mmap'd file).
    ///
    /// Returns the packed `&[BlockQ4K]` slice pointing directly into the
    /// mmap'd GGUF data — no dequantization, no allocation. This is the
    /// quant-resident load primitive: callers can upload these blocks
    /// straight to the GPU without ever materializing f32 host RAM.
    ///
    /// Returns an error if the tensor is not found or is not `Q4_K` type.
    /// The caller is responsible for knowing the matrix shape `(m, n)`;
    /// it is NOT stored in the block array (`Q4_K` is row-major: `m` rows of
    /// `n / QK_K` blocks each).
    pub fn q4k_tensor_blocks(&self, name: &str) -> Result<&[BlockQ4K]> {
        let info = self
            .tensor_map
            .get(name)
            .with_context(|| format!("tensor '{name}' not found in GGUF file"))?;
        if info.ggml_type != GgmlType::Q4_K {
            bail!(
                "tensor '{name}' is {:?}, expected Q4_K for quant-resident load",
                info.ggml_type
            );
        }
        let slice = self
            .tensor_slice(name)
            .with_context(|| format!("tensor '{name}' data out of bounds"))?;
        // SAFETY: BlockQ4K is #[repr(C)] with no padding (all fields are u8/u16
        // arrays), so bytemuck::cast_slice is sound for any aligned byte slice.
        let blocks: &[BlockQ4K] = bytemuck::cast_slice(slice);
        Ok(blocks)
    }

    /// Get raw `Q2_0` blocks for a tensor (zero-copy from the mmap'd file).
    ///
    /// Returns the packed `&[BlockQ2_0]` slice pointing directly into the
    /// mmap'd GGUF data — no dequantization, no allocation. The ternary-DeltaNet
    /// loader (Issue 594) feeds these to `repack_q2_0_to_ternary_group`.
    ///
    /// Returns an error if the tensor is not found or is not `Q2_0` type.
    /// The caller is responsible for knowing the matrix shape `(rows, cols)`;
    /// it is NOT stored in the block array (`Q2_0` is row-major: `rows` rows of
    /// `cols / 128` blocks each).
    pub fn q2_0_tensor_blocks(&self, name: &str) -> Result<&[BlockQ2_0]> {
        let info = self
            .tensor_map
            .get(name)
            .with_context(|| format!("tensor '{name}' not found in GGUF file"))?;
        if info.ggml_type != GgmlType::Q2_0 {
            bail!(
                "tensor '{name}' is {:?}, expected Q2_0 for ternary repack",
                info.ggml_type
            );
        }
        let slice = self
            .tensor_slice(name)
            .with_context(|| format!("tensor '{name}' data out of bounds"))?;
        // SAFETY: BlockQ2_0 is #[repr(C)] Pod (all fields are u8/u16 arrays),
        // so bytemuck::cast_slice is sound for any aligned byte slice.
        let blocks: &[BlockQ2_0] = bytemuck::cast_slice(slice);
        Ok(blocks)
    }

    /// Get raw `PTQ1_0` blocks for a tensor (zero-copy from the mmap'd file).
    ///
    /// The Bonsai-2 decode pack's ternary format (type 143; Issue 980 T6).
    /// Same contract as [`Self::q2_0_tensor_blocks`]: row-major, the caller
    /// supplies the matrix shape.
    pub fn ptq1_0_tensor_blocks(&self, name: &str) -> Result<&[BlockPtq1_0]> {
        let info = self
            .tensor_map
            .get(name)
            .with_context(|| format!("tensor '{name}' not found in GGUF file"))?;
        if info.ggml_type != GgmlType::PTQ1_0 {
            bail!(
                "tensor '{name}' is {:?}, expected PTQ1_0 for ternary repack",
                info.ggml_type
            );
        }
        let slice = self
            .tensor_slice(name)
            .with_context(|| format!("tensor '{name}' data out of bounds"))?;
        // SAFETY: BlockPtq1_0 is #[repr(C)] Pod (u8 arrays + u16), so
        // bytemuck::cast_slice is sound for any aligned byte slice.
        let blocks: &[BlockPtq1_0] = bytemuck::cast_slice(slice);
        Ok(blocks)
    }

    /// Get a metadata string value.
    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    /// Get a metadata u64 value.
    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| v.as_u64())
    }

    /// Get a metadata f64 value.
    pub fn metadata_f64(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).and_then(|v| v.as_f64())
    }

    /// Get a metadata array value (Issue 980: `prism.hadamard.*` arrays).
    pub fn metadata_array(&self, key: &str) -> Option<&[GgufValue]> {
        self.metadata.get(key).and_then(|v| v.as_array())
    }

    /// Get architecture name from metadata.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata_string("general.architecture")
    }

    /// Dequantize a single row of a 2D tensor to f32 (zero host allocation
    /// beyond the output row).
    ///
    /// This is the lazy embedding-lookup primitive: instead of materializing
    /// the full `[vocab × n_embd]` token embedding as f32 (~4 GB for Gemma-4),
    /// callers can look up individual rows on demand directly from the mmap.
    ///
    /// # Arguments
    /// * `name` - tensor name (e.g. `"token_embd.weight"`)
    /// * `row_idx` - 0-based row index
    /// * `row_len` - number of elements per row (the tensor's inner dim)
    ///
    /// Returns the dequantized row as `Vec<f32>` of length `row_len`.
    pub fn dequant_tensor_row(
        &self,
        name: &str,
        row_idx: usize,
        row_len: usize,
    ) -> Result<Vec<f32>> {
        let info = self
            .tensor_map
            .get(name)
            .with_context(|| format!("tensor '{name}' not found in GGUF file"))?;
        let slice = self
            .tensor_slice(name)
            .with_context(|| format!("tensor '{name}' data out of bounds"))?;

        let out = vec![0.0f32; row_len];
        // Dispatch by quant type — for each, slice the blocks for this row only.
        match info.ggml_type {
            GgmlType::F16 => {
                let mut out = out;
                let start = row_idx * row_len;
                for (i, j) in (start..start + row_len).enumerate() {
                    let off = j * 2;
                    let bits = u16::from_le_bytes([slice[off], slice[off + 1]]);
                    out[i] = half::f16::from_bits(bits).to_f32();
                }
                Ok(out)
            }
            GgmlType::F32 => {
                let start = row_idx * row_len * 4;
                let mut out = out;
                for (i, dst) in out.iter_mut().enumerate().take(row_len) {
                    let off = start + i * 4;
                    *dst = f32::from_le_bytes([
                        slice[off],
                        slice[off + 1],
                        slice[off + 2],
                        slice[off + 3],
                    ]);
                }
                Ok(out)
            }
            GgmlType::Q4_K => {
                let blocks_per_row = row_len / crate::quant::q4k::QK_K;
                let start_block = row_idx * blocks_per_row;
                let row_blocks: &[BlockQ4K] = bytemuck::cast_slice(
                    &slice[start_block * std::mem::size_of::<BlockQ4K>()
                        ..(start_block + blocks_per_row) * std::mem::size_of::<BlockQ4K>()],
                );
                let mut out = out;
                dequantize_row_q4_k(row_blocks, &mut out);
                Ok(out)
            }
            GgmlType::Q2_K => {
                // K-quant Q2_K: 256 weights per 84-byte super-block (Issue 780
                // U3 wall-1 — the DFlash2 drafter's quant-resident path).
                let blocks_per_row = row_len / crate::quant::q2k::QK_K;
                let start_byte = row_idx * blocks_per_row * std::mem::size_of::<BlockQ2K>();
                let row_bytes = &slice
                    [start_byte..start_byte + blocks_per_row * std::mem::size_of::<BlockQ2K>()];
                let row_blocks: &[BlockQ2K] = bytemuck::cast_slice(row_bytes);
                let mut out = out;
                dequantize_row_q2_k(row_blocks, &mut out);
                Ok(out)
            }
            GgmlType::Q3_K => {
                // K-quant Q3_K: 256 weights per 110-byte super-block.
                let blocks_per_row = row_len / crate::quant::q3k::QK_K;
                let start_byte = row_idx * blocks_per_row * std::mem::size_of::<BlockQ3K>();
                let row_bytes = &slice
                    [start_byte..start_byte + blocks_per_row * std::mem::size_of::<BlockQ3K>()];
                let row_blocks: &[BlockQ3K] = bytemuck::cast_slice(row_bytes);
                let mut out = out;
                dequantize_row_q3_k(row_blocks, &mut out);
                Ok(out)
            }
            GgmlType::Q6_K => {
                // Q6_K: 256 weights per 210-byte super-block.
                let qk_k = 256usize;
                let blocks_per_row = row_len / qk_k;
                let block_bytes = std::mem::size_of::<crate::quant::q6k::BlockQ6K>();
                let start_byte = row_idx * blocks_per_row * block_bytes;
                let row_bytes = &slice[start_byte..start_byte + blocks_per_row * block_bytes];
                let row_blocks: &[crate::quant::q6k::BlockQ6K] = bytemuck::cast_slice(row_bytes);
                let mut out = out;
                crate::quant::q6k::dequantize_row_q6_k(row_blocks, &mut out);
                Ok(out)
            }
            GgmlType::Q2_0 => {
                // Ternary-Bonsai: 128 weights per 34-byte block (2.125 bpw).
                let qk = crate::quant::q2_0::Q2_0_BLOCK_SIZE;
                let blocks_per_row = row_len / qk;
                let block_bytes = std::mem::size_of::<BlockQ2_0>();
                let start_byte = row_idx * blocks_per_row * block_bytes;
                let row_bytes = &slice[start_byte..start_byte + blocks_per_row * block_bytes];
                let row_blocks: &[BlockQ2_0] = bytemuck::cast_slice(row_bytes);
                let mut out = out;
                dequantize_row_q2_0(row_blocks, &mut out);
                Ok(out)
            }
            GgmlType::PTQ1_0 => {
                // Prism ternary g128: 128 weights per 28-byte block (1.75 bpw).
                // Base-3 dense trits — decode is exact to the ternary alphabet.
                let qk = crate::quant::ptq1_0::PTQ1_0_BLOCK_SIZE;
                let blocks_per_row = row_len / qk;
                let block_bytes = std::mem::size_of::<BlockPtq1_0>();
                let start_byte = row_idx * blocks_per_row * block_bytes;
                let row_bytes = &slice[start_byte..start_byte + blocks_per_row * block_bytes];
                let row_blocks: &[BlockPtq1_0] = bytemuck::cast_slice(row_bytes);
                let mut out = out;
                dequantize_row_ptq1_0(row_blocks, &mut out);
                Ok(out)
            }
            other => bail!(
                "single-row dequant for tensor '{name}' ({other:?}) not implemented; \
                 use dequant_f16_to_f32 for the full tensor"
            ),
        }
    }

    /// Dequantize an F16 tensor to f32 Vec.
    ///
    /// GGUF stores F16 as IEEE 754 half-precision (same as `half::f16`).
    pub fn dequant_f16_to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensor_map
            .get(name)
            .with_context(|| format!("tensor '{name}' not found in GGUF file"))?;

        let slice = self
            .tensor_slice(name)
            .with_context(|| format!("tensor '{name}' data out of bounds"))?;

        match info.ggml_type {
            GgmlType::F16 => {
                let n = info.n_elements();
                let mut out = Vec::with_capacity(n);
                // Fast path: aligned little-endian input → direct u16 cast.
                #[cfg(target_endian = "little")]
                if let Ok(bits) = bytemuck::try_cast_slice::<_, u16>(slice) {
                    bits.iter()
                        .map(|&b| half::f16::from_bits(b).to_f32())
                        .for_each(|f| out.push(f));
                    return Ok(out);
                }
                for chunk in slice.as_chunks::<2>().0 {
                    let bits = u16::from_le_bytes(*chunk);
                    out.push(half::f16::from_bits(bits).to_f32());
                }
                Ok(out)
            }
            GgmlType::F32 => {
                let n = info.n_elements();
                // Zero-copy cast on aligned little-endian data: GGUF F32 tensors are
                // little-endian f32, matching Rust's f32 layout on LE platforms.
                // Falls back to per-element loop on alignment issues or big-endian.
                #[cfg(target_endian = "little")]
                if let Ok(out) = bytemuck::try_cast_slice::<_, f32>(slice) {
                    let out = out.to_vec();
                    debug_assert_eq!(out.len(), n);
                    return Ok(out);
                }
                let mut out = Vec::with_capacity(n);
                for chunk in slice.as_chunks::<4>().0 {
                    out.push(f32::from_le_bytes(*chunk));
                }
                Ok(out)
            }
            GgmlType::BF16 => {
                let n = info.n_elements();
                let mut out = Vec::with_capacity(n);
                #[cfg(target_endian = "little")]
                if let Ok(bits) = bytemuck::try_cast_slice::<_, u16>(slice) {
                    bits.iter()
                        .map(|&b| bf16_to_f32(b))
                        .for_each(|f| out.push(f));
                    return Ok(out);
                }
                for chunk in slice.as_chunks::<2>().0 {
                    let bits = u16::from_le_bytes(*chunk);
                    out.push(bf16_to_f32(bits));
                }
                Ok(out)
            }
            GgmlType::Q8_0 => {
                let n = info.n_elements();
                let mut out = vec![0.0f32; n];
                // Cast raw bytes to BlockQ8_0 blocks (repr(C), Pod).
                let blocks: &[BlockQ8_0] = bytemuck::cast_slice(slice);
                dequantize_row_q8_0(blocks, &mut out);
                Ok(out)
            }
            GgmlType::Q4_K => {
                let n = info.n_elements();
                // Q4_K packs 256 weights per 144-byte super-block. The dequant
                // helper asserts `n` is a multiple of QK_K; this is guaranteed by
                // the GGUF writer (tensor dims are padded up to QK_K multiples).
                let mut out = vec![0.0f32; n];
                let blocks: &[BlockQ4K] = bytemuck::cast_slice(slice);
                dequantize_row_q4_k(blocks, &mut out);
                Ok(out)
            }
            GgmlType::Q2_K => {
                // K-quant Q2_K: 256 weights per 84-byte super-block (2.625
                // bpw). Same QK_K divisibility guarantee as Q4_K.
                let n = info.n_elements();
                let mut out = vec![0.0f32; n];
                let blocks: &[BlockQ2K] = bytemuck::cast_slice(slice);
                dequantize_row_q2_k(blocks, &mut out);
                Ok(out)
            }
            GgmlType::Q3_K => {
                // K-quant Q3_K: 256 weights per 110-byte super-block (3.4375
                // bpw).
                let n = info.n_elements();
                let mut out = vec![0.0f32; n];
                let blocks: &[BlockQ3K] = bytemuck::cast_slice(slice);
                dequantize_row_q3_k(blocks, &mut out);
                Ok(out)
            }
            GgmlType::Q5_K => {
                // Q5_K packs 256 weights per 176-byte super-block. Same QK_K
                // divisibility guarantee as Q4_K.
                let n = info.n_elements();
                let mut out = vec![0.0f32; n];
                let blocks: &[BlockQ5K] = bytemuck::cast_slice(slice);
                dequantize_row_q5_k(blocks, &mut out);
                Ok(out)
            }
            GgmlType::Q6_K => {
                // Q6_K packs 256 weights per 210-byte super-block. Same QK_K
                // divisibility guarantee as Q4_K.
                let n = info.n_elements();
                let mut out = vec![0.0f32; n];
                let blocks: &[BlockQ6K] = bytemuck::cast_slice(slice);
                dequantize_row_q6_k(blocks, &mut out);
                Ok(out)
            }
            GgmlType::Q2_0 => {
                // Ternary-Bonsai: 128 weights per 34-byte block (2.125 bpw).
                // Decodes all four 2-bit codes faithfully, including the +2d fourth
                // state (the TernaryGroupWeights bridge rejects code 3 separately).
                let n = info.n_elements();
                let mut out = vec![0.0f32; n];
                let blocks: &[BlockQ2_0] = bytemuck::cast_slice(slice);
                dequantize_row_q2_0(blocks, &mut out);
                Ok(out)
            }
            GgmlType::PTQ1_0 => {
                // Prism ternary g128 (Issue 980 T6): 128 weights per 28-byte
                // block; decodes exactly to the ternary alphabet.
                let n = info.n_elements();
                let mut out = vec![0.0f32; n];
                let blocks: &[BlockPtq1_0] = bytemuck::cast_slice(slice);
                dequantize_row_ptq1_0(blocks, &mut out);
                Ok(out)
            }
            GgmlType::Q4_1 => {
                // 32 weights per 20-byte block: f16 d, f16 m, 16 packed nibbles.
                let n = info.n_elements();
                let mut out = vec![0.0f32; n];
                dequantize_row_q4_1(slice, &mut out)?;
                Ok(out)
            }
            GgmlType::Q4_0 => {
                // 32 weights per 18-byte block: f16 d, 16 packed nibbles —
                // Q4_1 without the offset. The qwen35-hybrid small checkpoints
                // (Qwen3.5-0.8B-Q4_0) ship in this quant.
                let n = info.n_elements();
                let mut out = vec![0.0f32; n];
                dequantize_row_q4_0(slice, &mut out)?;
                Ok(out)
            }
            other => bail!(
                "tensor '{name}' is {other:?}, expected F16, BF16, F32, Q8_0, Q4_1, Q4_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, or Q2_0 for dequant"
            ),
        }
    }

    /// Try to dequantize a tensor, returning `None` if not found.
    ///
    /// Useful for optional tensors (e.g., `ssm_a` may not be present in all GGUF files).
    pub fn try_dequant_f16_to_f32(&self, name: &str) -> Result<Option<Vec<f32>>> {
        if !self.tensor_map.contains_key(name) {
            return Ok(None);
        }
        Ok(Some(self.dequant_f16_to_f32(name)?))
    }
}

// ── Binary readers (cursor-based on mmap slice) ───────────────

fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32> {
    if *cursor + 4 > data.len() {
        bail!("GGUF parse: unexpected end of file reading u32 at offset {cursor}");
    }
    let val = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(val)
}

fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64> {
    if *cursor + 8 > data.len() {
        bail!("GGUF parse: unexpected end of file reading u64 at offset {cursor}");
    }
    let val = u64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(val)
}

fn read_f32(data: &[u8], cursor: &mut usize) -> Result<f32> {
    if *cursor + 4 > data.len() {
        bail!("GGUF parse: unexpected end of file reading f32 at offset {cursor}");
    }
    let val = f32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(val)
}

fn read_f64(data: &[u8], cursor: &mut usize) -> Result<f64> {
    if *cursor + 8 > data.len() {
        bail!("GGUF parse: unexpected end of file reading f64 at offset {cursor}");
    }
    let val = f64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(val)
}

fn read_gguf_string(data: &[u8], cursor: &mut usize) -> Result<String> {
    let len = read_u64(data, cursor)? as usize;
    if len > 65536 {
        bail!("GGUF string too long: {len} bytes");
    }
    if *cursor + len > data.len() {
        bail!("GGUF parse: unexpected end of file reading string at offset {cursor}");
    }
    let s = std::str::from_utf8(&data[*cursor..*cursor + len])
        .with_context(|| format!("invalid UTF-8 in GGUF string at offset {cursor}"))?
        .to_string();
    *cursor += len;
    Ok(s)
}

fn read_gguf_value(data: &[u8], cursor: &mut usize) -> Result<GgufValue> {
    let type_id = read_u32(data, cursor)?;
    match type_id {
        META_TYPE_UINT8 => {
            if *cursor + 1 > data.len() {
                bail!("truncated metadata u8");
            }
            let v = data[*cursor];
            *cursor += 1;
            Ok(GgufValue::U8(v))
        }
        META_TYPE_INT8 => {
            if *cursor + 1 > data.len() {
                bail!("truncated metadata i8");
            }
            let v = data[*cursor] as i8;
            *cursor += 1;
            Ok(GgufValue::I8(v))
        }
        META_TYPE_UINT16 => {
            if *cursor + 2 > data.len() {
                bail!("truncated metadata u16");
            }
            let v = u16::from_le_bytes([data[*cursor], data[*cursor + 1]]);
            *cursor += 2;
            Ok(GgufValue::U16(v))
        }
        META_TYPE_INT16 => {
            if *cursor + 2 > data.len() {
                bail!("truncated metadata i16");
            }
            let v = i16::from_le_bytes([data[*cursor], data[*cursor + 1]]);
            *cursor += 2;
            Ok(GgufValue::I16(v))
        }
        META_TYPE_UINT32 => Ok(GgufValue::U32(read_u32(data, cursor)?)),
        META_TYPE_INT32 => {
            let v = read_u32(data, cursor)?;
            Ok(GgufValue::I32(v as i32))
        }
        META_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(data, cursor)?)),
        META_TYPE_BOOL => {
            if *cursor + 1 > data.len() {
                bail!("truncated metadata bool");
            }
            let v = data[*cursor] != 0;
            *cursor += 1;
            Ok(GgufValue::Bool(v))
        }
        META_TYPE_STRING => Ok(GgufValue::String(read_gguf_string(data, cursor)?)),
        META_TYPE_ARRAY => read_gguf_array(data, cursor),
        META_TYPE_UINT64 => Ok(GgufValue::U64(read_u64(data, cursor)?)),
        META_TYPE_INT64 => {
            let v = read_u64(data, cursor)?;
            Ok(GgufValue::I64(v as i64))
        }
        META_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(data, cursor)?)),
        _ => bail!("unknown GGUF metadata value type: {type_id}"),
    }
}

fn read_gguf_array(data: &[u8], cursor: &mut usize) -> Result<GgufValue> {
    let elem_type = read_u32(data, cursor)?;
    let count = read_u64(data, cursor)? as usize;
    if count > 10_000_000 {
        bail!("GGUF array too large: {count} elements");
    }

    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let v = match elem_type {
            META_TYPE_UINT8 => {
                if *cursor + 1 > data.len() {
                    bail!("truncated array u8");
                }
                let v = data[*cursor];
                *cursor += 1;
                GgufValue::U8(v)
            }
            META_TYPE_INT8 => {
                if *cursor + 1 > data.len() {
                    bail!("truncated array i8");
                }
                let v = data[*cursor] as i8;
                *cursor += 1;
                GgufValue::I8(v)
            }
            META_TYPE_UINT16 => {
                if *cursor + 2 > data.len() {
                    bail!("truncated array u16");
                }
                let v = u16::from_le_bytes([data[*cursor], data[*cursor + 1]]);
                *cursor += 2;
                GgufValue::U16(v)
            }
            META_TYPE_INT16 => {
                if *cursor + 2 > data.len() {
                    bail!("truncated array i16");
                }
                let v = i16::from_le_bytes([data[*cursor], data[*cursor + 1]]);
                *cursor += 2;
                GgufValue::I16(v)
            }
            META_TYPE_UINT32 => GgufValue::U32(read_u32(data, cursor)?),
            META_TYPE_INT32 => {
                let v = read_u32(data, cursor)?;
                GgufValue::I32(v as i32)
            }
            META_TYPE_FLOAT32 => GgufValue::F32(read_f32(data, cursor)?),
            META_TYPE_BOOL => {
                if *cursor + 1 > data.len() {
                    bail!("truncated array bool");
                }
                let v = data[*cursor] != 0;
                *cursor += 1;
                GgufValue::Bool(v)
            }
            META_TYPE_STRING => GgufValue::String(read_gguf_string(data, cursor)?),
            META_TYPE_ARRAY => read_gguf_array(data, cursor)?,
            META_TYPE_UINT64 => GgufValue::U64(read_u64(data, cursor)?),
            META_TYPE_INT64 => {
                let v = read_u64(data, cursor)?;
                GgufValue::I64(v as i64)
            }
            META_TYPE_FLOAT64 => GgufValue::F64(read_f64(data, cursor)?),
            _ => bail!("unknown GGUF array element type: {elem_type}"),
        };
        values.push(v);
    }
    Ok(GgufValue::Array(values))
}

// ── GGUF → Gemma 2 weight loading ─────────────────────────────

/// Load Gemma 2 weights from a GGUF file into f32 weight struct.
///
/// Supports F16 and F32 tensor types. Dequantizes F16 → f32.
/// `RMSNorm` offset (+1.0) is already applied by the GGUF converter.
///
/// Dequantize a `Q4_1` tensor into `out` (riir-ai Issue 717).
///
/// `Q4_1` block: `{ f16 d; f16 m; u8 qs[16] }` = 20 bytes per 32 weights.
/// Nibble layout follows llama.cpp: `qs[j]` low nibble is weight `j`, high
/// nibble is weight `j + 16`. Value = `d * q + m`.
///
/// `GgmlType::Q4_1` already had block info (20/32) but no decoder, so any `Q4_1`
/// tensor failed at dequant time. Needed for `dspark.markov_head_b.weight`.
fn dequantize_row_q4_1(slice: &[u8], out: &mut [f32]) -> Result<()> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 20;
    if !out.len().is_multiple_of(QK) {
        bail!(
            "Q4_1 dequant: element count {} is not a multiple of {QK}",
            out.len()
        );
    }
    let n_blocks = out.len() / QK;
    let need = n_blocks * BLOCK_BYTES;
    if slice.len() < need {
        bail!(
            "Q4_1 dequant: need {need} bytes for {n_blocks} blocks, got {}",
            slice.len()
        );
    }
    for (b, chunk) in slice[..need]
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        let m = half::f16::from_bits(u16::from_le_bytes([chunk[2], chunk[3]])).to_f32();
        let qs = &chunk[4..20];
        let base = b * QK;
        for (j, &q) in qs.iter().enumerate() {
            out[base + j] = d * f32::from(q & 0x0F) + m;
            out[base + j + QK / 2] = d * f32::from(q >> 4) + m;
        }
    }
    Ok(())
}

/// Dequantize a `Q4_0` tensor into `out`.
///
/// `Q4_0` block: `{ f16 d; u8 qs[16] }` = 18 bytes per 32 weights — `Q4_1`
/// without the `m` offset. Same nibble order: low nibble is weight `j`,
/// high nibble is weight `j + 16`. Value = `d * (q - 8)` (ggml reference
/// `dequantize_row_q4_0`). Needed by the qwen35-hybrid small checkpoints
/// (Qwen3.5-0.8B ships `Q4_0`).
fn dequantize_row_q4_0(slice: &[u8], out: &mut [f32]) -> Result<()> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 18;
    if !out.len().is_multiple_of(QK) {
        bail!(
            "Q4_0 dequant: element count {} is not a multiple of {QK}",
            out.len()
        );
    }
    let n_blocks = out.len() / QK;
    let need = n_blocks * BLOCK_BYTES;
    if slice.len() < need {
        bail!(
            "Q4_0 dequant: need {need} bytes for {n_blocks} blocks, got {}",
            slice.len()
        );
    }
    for (b, chunk) in slice[..need]
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
    {
        let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        let qs = &chunk[2..18];
        let base = b * QK;
        for (j, &q) in qs.iter().enumerate() {
            out[base + j] = d * (f32::from(q & 0x0F) - 8.0);
            out[base + j + QK / 2] = d * (f32::from(q >> 4) - 8.0);
        }
    }
    Ok(())
}

/// GGUF tensor name mapping for Gemma 2:
/// - `token_embd.weight` → `wte`
/// - `output_norm.weight` → `final_norm`
/// - `blk.N.attn_norm.weight` → `layers[N].input_norm`
/// - `blk.N.attn_q.weight` → `layers[N].attn_wq`
/// - `blk.N.attn_k.weight` → `layers[N].attn_wk`
/// - `blk.N.attn_v.weight` → `layers[N].attn_wv`
/// - `blk.N.attn_output.weight` → `layers[N].attn_wo`
/// - `blk.N.post_attention_norm.weight` → `layers[N].post_attn_norm`
/// - `blk.N.ffn_norm.weight` → `layers[N].pre_mlp_norm`
/// - `blk.N.ffn_gate.weight` → `layers[N].gate_proj`
/// - `blk.N.ffn_up.weight` → `layers[N].up_proj`
/// - `blk.N.ffn_down.weight` → `layers[N].down_proj`
/// - `blk.N.post_ffw_norm.weight` → `layers[N].post_mlp_norm`
pub fn load_gemma2_weights_gguf(path: &Path) -> Result<(Config, GemmaTransformerWeights)> {
    let gguf = GgufFile::open(path)?;

    // Validate architecture
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "gemma2" {
        bail!("expected gemma2 architecture, got '{arch}'");
    }

    // Build Config from metadata
    let config = config_from_gguf_metadata(&gguf)?;

    let n_layer = config.n_layer;

    // Load global weights
    let wte = gguf.dequant_f16_to_f32("token_embd.weight")?;
    let final_norm = gguf.dequant_f16_to_f32("output_norm.weight")?;
    // RMSNorm offset is already applied by GGUF converter (+1)

    let mut layers = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let ln = gguf_layer_names(i);
        let input_norm = gguf.dequant_f16_to_f32(&ln.attn_norm)?;
        let post_attn_norm = gguf.dequant_f16_to_f32(&ln.post_attn_norm)?;
        let pre_mlp_norm = gguf.dequant_f16_to_f32(&ln.ffn_norm)?;
        let post_mlp_norm = gguf.dequant_f16_to_f32(&ln.post_ffw_norm)?;

        let layer = GemmaLayerWeights {
            attn_wq: gguf.dequant_f16_to_f32(&ln.attn_q)?,
            attn_wk: gguf.dequant_f16_to_f32(&ln.attn_k)?,
            attn_wv: gguf.dequant_f16_to_f32(&ln.attn_v)?,
            attn_wo: gguf.dequant_f16_to_f32(&ln.attn_output)?,
            gate_proj: gguf.dequant_f16_to_f32(&ln.ffn_gate)?,
            up_proj: gguf.dequant_f16_to_f32(&ln.ffn_up)?,
            down_proj: gguf.dequant_f16_to_f32(&ln.ffn_down)?,
            input_norm,
            post_attn_norm,
            pre_mlp_norm,
            post_mlp_norm,
        };
        layers.push(layer);
    }

    let weights = GemmaTransformerWeights {
        wte,
        final_norm,
        layers,
        #[cfg(feature = "delta_routing")]
        delta_routing_query: (0..n_layer).map(|_| vec![0.0; config.n_embd]).collect(),
        #[cfg(feature = "delta_routing")]
        delta_routing_norm: (0..n_layer).map(|_| vec![1.0; config.n_embd]).collect(),
    };

    Ok((config, weights))
}

/// Load Gemma-2 f16 weights DIRECTLY from the GGUF's F16 tensors (no f32
/// intermediate — 5.2 GiB peak instead of 10.4; the standard
/// [`load_gemma2_weights_gguf`] f32 path doubles the peak). Used by the
/// `vk_calibration` and `row_logit_floor_ppl` bins.
///
/// F16-type tensors only: a quantized gemma-2 GGUF must go through the
/// f32 dequant path (not this loader).
pub fn load_gemma2_f16_direct(gguf: &GgufFile, config: &Config) -> Result<GemmaTransformerWeightsF16> {
    let read_f16 = |name: &str| -> Result<Vec<half::f16>> {
        let slice = gguf
            .tensor_slice(name)
            .context(format!("tensor '{name}' data out of bounds"))?;
        #[cfg(target_endian = "little")]
        {
            let bits = bytemuck::cast_slice::<_, u16>(slice);
            Ok(bits.iter().map(|&b| half::f16::from_bits(b)).collect())
        }
        #[cfg(target_endian = "big")]
        {
            let _ = slice;
            bail!("big-endian host: use the f32 dequant path");
        }
    };
    let read_norm = |name: &str| -> Result<Vec<f32>> { gguf.dequant_f16_to_f32(name) };

    let wte = read_f16("token_embd.weight")?;
    let final_norm = read_norm("output_norm.weight")?;
    let mut layers = Vec::with_capacity(config.n_layer);
    for i in 0..config.n_layer {
        let f = |suffix: &str| format!("blk.{i}.{suffix}");
        layers.push(GemmaLayerWeightsF16 {
            attn_wq: read_f16(&f("attn_q.weight"))?,
            attn_wk: read_f16(&f("attn_k.weight"))?,
            attn_wv: read_f16(&f("attn_v.weight"))?,
            attn_wo: read_f16(&f("attn_output.weight"))?,
            gate_proj: read_f16(&f("ffn_gate.weight"))?,
            up_proj: read_f16(&f("ffn_up.weight"))?,
            down_proj: read_f16(&f("ffn_down.weight"))?,
            input_norm: read_norm(&f("attn_norm.weight"))?,
            post_attn_norm: read_norm(&f("post_attention_norm.weight"))?,
            pre_mlp_norm: read_norm(&f("ffn_norm.weight"))?,
            post_mlp_norm: read_norm(&f("post_ffw_norm.weight"))?,
        });
    }
    Ok(GemmaTransformerWeightsF16 {
        wte,
        final_norm,
        layers,
        #[cfg(feature = "delta_routing")]
        delta_routing_query: (0..config.n_layer).map(|_| vec![0.0; config.n_embd]).collect(),
        #[cfg(feature = "delta_routing")]
        delta_routing_norm: (0..config.n_layer).map(|_| vec![1.0; config.n_embd]).collect(),
    })
}

/// Build `Config` from GGUF metadata keys for Gemma 2.
///
/// `pub` since the vk_calibration bin (Issue 883 P0) builds its config
/// from the same open [`GgufFile`] it streams f16 tensors from — the
/// f32 loader path would double-peak past the box's free RAM.
pub fn config_from_gguf_metadata(gguf: &GgufFile) -> Result<Config> {
    let prefix = "gemma2.";

    let context_length = gguf
        .metadata_u64(&format!("{prefix}context_length"))
        .unwrap_or(8192) as usize;
    let n_embd = gguf
        .metadata_u64(&format!("{prefix}embedding_length"))
        .unwrap_or(2304) as usize;
    let n_layer = gguf
        .metadata_u64(&format!("{prefix}block_count"))
        .unwrap_or(26) as usize;
    let mlp_hidden = gguf
        .metadata_u64(&format!("{prefix}feed_forward_length"))
        .unwrap_or(9216) as usize;
    let n_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count"))
        .unwrap_or(8) as usize;
    let n_kv_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count_kv"))
        .unwrap_or(4) as usize;
    let rms_norm_eps = gguf
        .metadata_f64(&format!("{prefix}attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-6);
    let head_dim = gguf
        .metadata_u64(&format!("{prefix}attention.key_length"))
        .unwrap_or((n_embd / n_head) as u64) as usize;

    // Get vocab_size from token_embd tensor shape.
    // GGUF stores token_embd as [n_embd, vocab_size] for Gemma 2 (inner dim first),
    // so use last() not first() to get vocab_size.
    let vocab_size = gguf
        .tensor_info("token_embd.weight")
        .and_then(|info| info.shape.last().copied())
        .unwrap_or(256_000);

    // Start from gemma2_2b defaults, override with metadata
    let mut config = Config::gemma2_2b();
    config.vocab_size = vocab_size;
    config.block_size = context_length;
    config.n_embd = n_embd;
    config.n_head = n_head;
    config.head_dim = head_dim;
    config.mlp_hidden = mlp_hidden;
    config.n_layer = n_layer;
    config.n_kv_head = n_kv_head;
    config.rms_norm_eps = rms_norm_eps;

    Ok(config)
}

/// GGUF tensor names for a single Gemma 2 layer.
#[derive(Debug, Clone)]
struct GgufLayerNames {
    attn_norm: String,
    attn_q: String,
    attn_k: String,
    attn_v: String,
    attn_output: String,
    post_attn_norm: String,
    ffn_norm: String,
    ffn_gate: String,
    ffn_up: String,
    ffn_down: String,
    post_ffw_norm: String,
}

fn gguf_layer_names(layer_idx: usize) -> GgufLayerNames {
    let i = layer_idx;
    GgufLayerNames {
        attn_norm: format!("blk.{i}.attn_norm.weight"),
        attn_q: format!("blk.{i}.attn_q.weight"),
        attn_k: format!("blk.{i}.attn_k.weight"),
        attn_v: format!("blk.{i}.attn_v.weight"),
        attn_output: format!("blk.{i}.attn_output.weight"),
        post_attn_norm: format!("blk.{i}.post_attention_norm.weight"),
        ffn_norm: format!("blk.{i}.ffn_norm.weight"),
        ffn_gate: format!("blk.{i}.ffn_gate.weight"),
        ffn_up: format!("blk.{i}.ffn_up.weight"),
        ffn_down: format!("blk.{i}.ffn_down.weight"),
        post_ffw_norm: format!("blk.{i}.post_ffw_norm.weight"),
    }
}

// ── GGUF → LLaMA weight loading ──────────────────────────────

/// `LLaMA` layer tensor names from GGUF.
struct LlamaGgufLayerNames {
    attn_norm: String,
    attn_q: String,
    attn_k: String,
    attn_v: String,
    attn_output: String,
    ffn_norm: String,
    ffn_gate: String,
    ffn_up: String,
    ffn_down: String,
}

fn llama_gguf_layer_names(layer_idx: usize) -> LlamaGgufLayerNames {
    let i = layer_idx;
    LlamaGgufLayerNames {
        attn_norm: format!("blk.{i}.attn_norm.weight"),
        attn_q: format!("blk.{i}.attn_q.weight"),
        attn_k: format!("blk.{i}.attn_k.weight"),
        attn_v: format!("blk.{i}.attn_v.weight"),
        attn_output: format!("blk.{i}.attn_output.weight"),
        ffn_norm: format!("blk.{i}.ffn_norm.weight"),
        ffn_gate: format!("blk.{i}.ffn_gate.weight"),
        ffn_up: format!("blk.{i}.ffn_up.weight"),
        ffn_down: format!("blk.{i}.ffn_down.weight"),
    }
}

/// Load LLaMA-family weights from a GGUF file into f32 weight struct.
///
/// Supports F16 and F32 tensor types. Dequantizes F16 → f32.
/// Works with any LLaMA-architecture GGUF model (`LLaMA`, Mistral, `MiniCPM`, etc.).
///
/// GGUF tensor name mapping for `LLaMA`:
/// - `token_embd.weight` → `wte`
/// - `output_norm.weight` → `final_norm`
/// - `output.weight` → `lm_head` (separate, not tied)
/// - `blk.N.attn_norm.weight` → `layers[N].input_norm`
/// - `blk.N.attn_q.weight` → `layers[N].attn_wq`
/// - `blk.N.attn_k.weight` → `layers[N].attn_wk`
/// - `blk.N.attn_v.weight` → `layers[N].attn_wv`
/// - `blk.N.attn_output.weight` → `layers[N].attn_wo`
/// - `blk.N.ffn_norm.weight` → `layers[N].post_attn_norm`
/// - `blk.N.ffn_gate.weight` → `layers[N].gate_proj`
/// - `blk.N.ffn_up.weight` → `layers[N].up_proj`
/// - `blk.N.ffn_down.weight` → `layers[N].down_proj`
pub fn load_llama_weights_gguf(
    path: &Path,
) -> Result<(Config, crate::llama_layer::LlamaTransformerWeights)> {
    let gguf = GgufFile::open(path)?;

    // Validate architecture
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "llama" {
        bail!("expected llama architecture, got '{arch}'");
    }

    // Build Config from metadata
    let config = llama_config_from_gguf_metadata(&gguf)?;

    let n_layer = config.n_layer;

    // Load global weights
    let wte = gguf.dequant_f16_to_f32("token_embd.weight")?;
    let final_norm = gguf.dequant_f16_to_f32("output_norm.weight")?;

    // Load lm_head. LLaMA 3.2 3B (and other small LLaMA models) use tied
    // embeddings — there is no separate `output.weight` tensor. Fall back to
    // the token embedding weights in that case.
    let lm_head = if let Ok(w) = gguf.dequant_f16_to_f32("output.weight") {
        w
    } else {
        log::info!(
            "output.weight not found — assuming tied embeddings, using token_embd.weight as lm_head"
        );
        wte.clone()
    };

    let mut layers = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let ln = llama_gguf_layer_names(i);
        let layer = crate::llama_layer::LlamaLayerWeights {
            attn_wq: gguf.dequant_f16_to_f32(&ln.attn_q)?,
            attn_wk: gguf.dequant_f16_to_f32(&ln.attn_k)?,
            attn_wv: gguf.dequant_f16_to_f32(&ln.attn_v)?,
            attn_wo: gguf.dequant_f16_to_f32(&ln.attn_output)?,
            gate_proj: gguf.dequant_f16_to_f32(&ln.ffn_gate)?,
            up_proj: gguf.dequant_f16_to_f32(&ln.ffn_up)?,
            down_proj: gguf.dequant_f16_to_f32(&ln.ffn_down)?,
            input_norm: gguf.dequant_f16_to_f32(&ln.attn_norm)?,
            post_attn_norm: gguf.dequant_f16_to_f32(&ln.ffn_norm)?,
        };
        layers.push(layer);
    }

    let weights = crate::llama_layer::LlamaTransformerWeights {
        wte,
        lm_head,
        final_norm,
        layers,
    };

    Ok((config, weights))
}

/// Build `Config` from GGUF metadata keys for LLaMA-family models.
fn llama_config_from_gguf_metadata(gguf: &GgufFile) -> Result<Config> {
    let prefix = "llama.";

    let context_length = gguf
        .metadata_u64(&format!("{prefix}context_length"))
        .unwrap_or(4096) as usize;
    let n_embd = gguf
        .metadata_u64(&format!("{prefix}embedding_length"))
        .unwrap_or(4096) as usize;
    let n_layer = gguf
        .metadata_u64(&format!("{prefix}block_count"))
        .unwrap_or(32) as usize;
    let mlp_hidden = gguf
        .metadata_u64(&format!("{prefix}feed_forward_length"))
        .unwrap_or(11008) as usize;
    let n_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count"))
        .unwrap_or(32) as usize;
    let n_kv_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count_kv"))
        .unwrap_or(n_head as u64) as usize;
    let rms_norm_eps = gguf
        .metadata_f64(&format!("{prefix}attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5);
    let head_dim = gguf
        .metadata_u64(&format!("{prefix}attention.key_length"))
        .unwrap_or((n_embd / n_head) as u64) as usize;
    let rope_theta = gguf
        .metadata_f64(&format!("{prefix}rope.freq_base"))
        .unwrap_or(10000.0) as f32;

    // Get vocab_size from token_embd tensor shape.
    // GGUF stores token_embd as [n_embd, vocab_size] for LLaMA (inner dim first).
    let vocab_size = gguf
        .tensor_info("token_embd.weight")
        .and_then(|info| info.shape.last().copied())
        .unwrap_or(32000);

    let mut config = Config::micro();
    config.vocab_size = vocab_size;
    config.block_size = context_length;
    config.n_embd = n_embd;
    config.n_head = n_head;
    config.head_dim = head_dim;
    config.mlp_hidden = mlp_hidden;
    config.n_layer = n_layer;
    config.n_kv_head = n_kv_head;
    config.rms_norm_eps = rms_norm_eps;
    config.rope_theta = rope_theta;
    config.model_arch = ModelArchitecture::Llama;
    config.use_rope = true;
    config.tied_embeddings = false;
    config.rms_norm_offset = false;
    config.post_norm = false;
    config.attn_logit_softcapping = 0.0;
    config.final_logit_softcapping = 0.0;

    Ok(config)
}

/// Build `Config` from GGUF metadata keys for Qwen 2 family models.
///
/// Qwen 2/2.5 uses the same LLaMA-architecture weight layout but with a
/// different metadata prefix (`qwen2.` instead of `llama.`) and different
/// default values (higher `rope_theta`, smaller `rms_norm_eps`).
fn qwen2_config_from_gguf_metadata(gguf: &GgufFile) -> Result<Config> {
    let prefix = "qwen2.";

    let context_length = gguf
        .metadata_u64(&format!("{prefix}context_length"))
        .unwrap_or(32768) as usize;
    let n_embd = gguf
        .metadata_u64(&format!("{prefix}embedding_length"))
        .unwrap_or(2048) as usize;
    let n_layer = gguf
        .metadata_u64(&format!("{prefix}block_count"))
        .unwrap_or(36) as usize;
    let mlp_hidden = gguf
        .metadata_u64(&format!("{prefix}feed_forward_length"))
        .unwrap_or(11008) as usize;
    let n_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count"))
        .unwrap_or(16) as usize;
    let n_kv_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count_kv"))
        .unwrap_or(n_head as u64) as usize;
    let rms_norm_eps = gguf
        .metadata_f64(&format!("{prefix}attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-6);
    let head_dim = gguf
        .metadata_u64(&format!("{prefix}attention.key_length"))
        .unwrap_or((n_embd / n_head) as u64) as usize;
    let rope_theta = gguf
        .metadata_f64(&format!("{prefix}rope.freq_base"))
        .unwrap_or(1_000_000.0) as f32;

    // Get vocab_size from token_embd tensor shape.
    let vocab_size = gguf
        .tensor_info("token_embd.weight")
        .and_then(|info| info.shape.last().copied())
        .unwrap_or(151_936);

    let mut config = Config::micro();
    config.vocab_size = vocab_size;
    config.block_size = context_length;
    config.n_embd = n_embd;
    config.n_head = n_head;
    config.head_dim = head_dim;
    config.mlp_hidden = mlp_hidden;
    config.n_layer = n_layer;
    config.n_kv_head = n_kv_head;
    config.rms_norm_eps = rms_norm_eps;
    config.rope_theta = rope_theta;
    config.model_arch = ModelArchitecture::Llama; // Qwen 2 uses LLaMA forward pass
    config.use_rope = true;
    config.tied_embeddings = false;
    config.rms_norm_offset = false;
    config.post_norm = false;
    config.attn_logit_softcapping = 0.0;
    config.final_logit_softcapping = 0.0;

    Ok(config)
}

/// Load Qwen 2/2.5 weights from a GGUF file into f32 weight struct.
///
/// Qwen 2 uses the same LLaMA-architecture weight layout (same tensor
/// names, same weight structure) — only the metadata prefix differs
/// (`qwen2.` vs `llama.`). This function is a thin wrapper that validates
/// the `qwen2` architecture string and uses `qwen2.` metadata prefix.
///
/// Supports `Q8_0`, `Q6_K`, `Q5_K`, `Q5_0`, `Q4_K`, `Q4_0`, `Q3_K`, `Q2_K`, `Q2_0` (ternary), F16, BF16, F32.
/// All quantized types are dequantized to f32 during loading.
///
/// # Arguments
///
/// * `path` - Path to the GGUF file.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, the architecture is not
/// `"qwen2"`, or required tensors are missing.
pub fn load_qwen2_weights_gguf(
    path: &Path,
) -> Result<(Config, crate::llama_layer::LlamaTransformerWeights)> {
    let gguf = GgufFile::open(path)?;

    // Validate architecture
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "qwen2" {
        bail!("expected qwen2 architecture, got '{arch}'");
    }

    // Build Config from metadata
    let config = qwen2_config_from_gguf_metadata(&gguf)?;

    let n_layer = config.n_layer;

    // Load global weights (same tensor names as LLaMA)
    let wte = gguf.dequant_f16_to_f32("token_embd.weight")?;
    let final_norm = gguf.dequant_f16_to_f32("output_norm.weight")?;
    let lm_head = gguf.dequant_f16_to_f32("output.weight")?;

    let mut layers = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let ln = llama_gguf_layer_names(i);
        let layer = crate::llama_layer::LlamaLayerWeights {
            attn_wq: gguf.dequant_f16_to_f32(&ln.attn_q)?,
            attn_wk: gguf.dequant_f16_to_f32(&ln.attn_k)?,
            attn_wv: gguf.dequant_f16_to_f32(&ln.attn_v)?,
            attn_wo: gguf.dequant_f16_to_f32(&ln.attn_output)?,
            gate_proj: gguf.dequant_f16_to_f32(&ln.ffn_gate)?,
            up_proj: gguf.dequant_f16_to_f32(&ln.ffn_up)?,
            down_proj: gguf.dequant_f16_to_f32(&ln.ffn_down)?,
            input_norm: gguf.dequant_f16_to_f32(&ln.attn_norm)?,
            post_attn_norm: gguf.dequant_f16_to_f32(&ln.ffn_norm)?,
        };
        layers.push(layer);
    }

    let weights = crate::llama_layer::LlamaTransformerWeights {
        wte,
        lm_head,
        final_norm,
        layers,
    };

    Ok((config, weights))
}

// ── GGUF → Qwen3.5 DeltaNet weight loading ─────────────────────

/// GGUF tensor names for a Qwen3.5 `DeltaNet` (recurrent) layer.
#[allow(dead_code)]
struct Qwen35DeltanetGgufNames {
    attn_qkv: String,
    attn_gate: String,
    ssm_conv1d: String,
    ssm_dt: String,
    ssm_a: String,
    ssm_beta: String,
    ssm_alpha: String,
    ssm_norm: String,
    ssm_out: String,
    attn_norm: String,
    post_attn_norm: String,
    ffn_gate: String,
    ffn_up: String,
    ffn_down: String,
}

#[allow(dead_code)]
fn qwen35_deltanet_gguf_names(i: usize) -> Qwen35DeltanetGgufNames {
    Qwen35DeltanetGgufNames {
        attn_qkv: format!("blk.{i}.attn_qkv.weight"),
        attn_gate: format!("blk.{i}.attn_gate.weight"),
        ssm_conv1d: format!("blk.{i}.ssm_conv1d.weight"),
        ssm_dt: format!("blk.{i}.ssm_dt.bias"),
        ssm_a: format!("blk.{i}.ssm_a"),
        ssm_beta: format!("blk.{i}.ssm_beta.weight"),
        ssm_alpha: format!("blk.{i}.ssm_alpha.weight"),
        ssm_norm: format!("blk.{i}.ssm_norm.weight"),
        ssm_out: format!("blk.{i}.ssm_out.weight"),
        attn_norm: format!("blk.{i}.attn_norm.weight"),
        post_attn_norm: format!("blk.{i}.post_attention_norm.weight"),
        ffn_gate: format!("blk.{i}.ffn_gate.weight"),
        ffn_up: format!("blk.{i}.ffn_up.weight"),
        ffn_down: format!("blk.{i}.ffn_down.weight"),
    }
}

/// GGUF tensor names for a Qwen3.5 full-attention layer.
#[allow(dead_code)]
struct Qwen35AttentionGgufNames {
    attn_q: String,
    attn_k: String,
    attn_v: String,
    attn_output: String,
    attn_q_norm: String,
    attn_k_norm: String,
    attn_norm: String,
    post_attn_norm: String,
    ffn_gate: String,
    ffn_up: String,
    ffn_down: String,
}

#[allow(dead_code)]
fn qwen35_attention_gguf_names(i: usize) -> Qwen35AttentionGgufNames {
    Qwen35AttentionGgufNames {
        attn_q: format!("blk.{i}.attn_q.weight"),
        attn_k: format!("blk.{i}.attn_k.weight"),
        attn_v: format!("blk.{i}.attn_v.weight"),
        attn_output: format!("blk.{i}.attn_output.weight"),
        attn_q_norm: format!("blk.{i}.attn_q_norm.weight"),
        attn_k_norm: format!("blk.{i}.attn_k_norm.weight"),
        attn_norm: format!("blk.{i}.attn_norm.weight"),
        post_attn_norm: format!("blk.{i}.post_attention_norm.weight"),
        ffn_gate: format!("blk.{i}.ffn_gate.weight"),
        ffn_up: format!("blk.{i}.ffn_up.weight"),
        ffn_down: format!("blk.{i}.ffn_down.weight"),
    }
}

/// Load Qwen3.5 `DeltaNet` hybrid model weights from a GGUF file into f32 weight struct.
///
/// Supports F16, BF16, and F32 tensor types. Dequantizes to f32.
///
/// The GGUF file must have `general.architecture = "qwen35"`.
///
/// # GGUF Tensor Name Mapping
///
/// **Global:**
/// - `token_embd.weight` → `wte`
/// - `output_norm.weight` → `final_norm`
/// - `output.weight` → `lm_head` (if not tied)
///
/// **`DeltaNet` (recurrent) layers:**
/// - `blk.N.attn_qkv.weight` → `in_proj_qkv`
/// - `blk.N.attn_gate.weight` → `in_proj_z`
/// - `blk.N.ssm_conv1d.weight` → `conv1d_weight`
/// - `blk.N.ssm_dt.bias` → `dt_bias`
/// - `blk.N.ssm_a` → `a_log`
/// - `blk.N.ssm_beta.weight` → `in_proj_b`
/// - `blk.N.ssm_alpha.weight` → `in_proj_a`
/// - `blk.N.ssm_norm.weight` → `linear_norm`
/// - `blk.N.ssm_out.weight` → `out_proj`
///
/// **Full attention layers:**
/// - `blk.N.attn_q.weight` → `attn_wq`
/// - `blk.N.attn_k.weight` → `attn_wk`
/// - `blk.N.attn_v.weight` → `attn_wv`
/// - `blk.N.attn_output.weight` → `attn_wo`
///
/// **Both:**
/// - `blk.N.attn_norm.weight` → `input_norm`
/// - `blk.N.post_attention_norm.weight` → `post_attn_norm`
/// - `blk.N.ffn_gate.weight` → `gate_proj`
/// - `blk.N.ffn_up.weight` → `up_proj`
/// - `blk.N.ffn_down.weight` → `down_proj`
///
/// # Layer Type Computation
///
/// llama.cpp uses `qwen35.full_attention_interval` metadata to determine layer types:
/// - Layer `i` is **full attention** if `(i + 1) % full_attention_interval == 0`
/// - Otherwise it's **`DeltaNet`** (linear recurrent)
#[cfg(feature = "deltanet_inference")]
pub fn load_qwen_deltanet_weights_gguf(
    path: &Path,
) -> Result<(Config, crate::deltanet::weights::QwenDeltaNetWeights)> {
    use crate::deltanet::weights::Proj;

    let gguf = GgufFile::open(path)?;

    // Validate architecture
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "qwen35" {
        bail!("expected qwen35 architecture, got '{arch}'");
    }

    // Build Config from metadata
    let (config, layer_types) = qwen35_deltanet_config_from_gguf_metadata(&gguf)?;

    let n_layer = config.n_layer;

    // Load global weights
    let wte = gguf.dequant_f16_to_f32("token_embd.weight")?;
    let final_norm = gguf.dequant_f16_to_f32("output_norm.weight")?;

    // Check if lm_head is separate or tied
    let lm_head = if let Some(lm) = gguf.try_dequant_f16_to_f32("output.weight")? {
        crate::deltanet::weights::Proj::dense(lm, config.vocab_size, config.n_embd)
    } else {
        // tied embeddings — clone wte for the lm_head data
        crate::deltanet::weights::Proj::dense(wte.clone(), config.vocab_size, config.n_embd)
    };

    // Load per-layer weights
    // Shape dims for `Proj::dense` construction (mirror `zeros`).
    let n = config.n_embd;
    let q_dim = config.n_head * config.head_dim;
    let kvd = config.n_kv_head * config.head_dim;
    let mlp = config.mlp_hidden;
    let lhd = config.deltanet_linear_head_dim;
    let lkh = config.deltanet_linear_n_heads;
    let lvh = config.deltanet_linear_n_value_heads;
    let l_qkv_out = lkh * lhd + 2 * lvh * lhd;
    let l_a_out = lkh;
    let l_z_out = lvh * lhd;
    let l_out_in = lvh * lhd;

    let mut layers = Vec::with_capacity(n_layer);
    for (i, lt) in layer_types.iter().enumerate() {
        let is_linear = *lt == DeltaNetLayerType::DeltaNet;

        let layer = if is_linear {
            let ln = qwen35_deltanet_gguf_names(i);

            // Load DeltaNet-specific tensors
            // GGUF stores QKV fused, but our struct expects in_proj_qkv directly
            let in_proj_qkv = gguf.dequant_f16_to_f32(&ln.attn_qkv)?;
            let in_proj_z = gguf.dequant_f16_to_f32(&ln.attn_gate)?;
            let conv1d_weight = gguf.dequant_f16_to_f32(&ln.ssm_conv1d)?;
            let dt_bias = gguf.dequant_f16_to_f32(&ln.ssm_dt)?;
            let a_log = gguf.dequant_f16_to_f32(&ln.ssm_a)?;
            let in_proj_b = gguf.dequant_f16_to_f32(&ln.ssm_beta)?;
            let in_proj_a = gguf.dequant_f16_to_f32(&ln.ssm_alpha)?;
            let linear_norm = gguf.dequant_f16_to_f32(&ln.ssm_norm)?;
            let out_proj = gguf.dequant_f16_to_f32(&ln.ssm_out)?;

            // MLP + norms
            let input_norm = gguf.dequant_f16_to_f32(&ln.attn_norm)?;
            let post_attn_norm = gguf.dequant_f16_to_f32(&ln.post_attn_norm)?;
            let gate_proj = gguf.dequant_f16_to_f32(&ln.ffn_gate)?;
            let up_proj = gguf.dequant_f16_to_f32(&ln.ffn_up)?;
            let down_proj = gguf.dequant_f16_to_f32(&ln.ffn_down)?;

            crate::deltanet::weights::DeltaNetLayerWeights {
                // Full attention: empty for DeltaNet layers
                attn_wq: Proj::empty(),
                attn_wk: Proj::empty(),
                attn_wv: Proj::empty(),
                attn_wo: Proj::empty(),
                attn_q_norm: Vec::new(),
                attn_k_norm: Vec::new(),
                // Linear attention
                in_proj_qkv: Proj::dense(in_proj_qkv, l_qkv_out, n),
                in_proj_a: Proj::dense(in_proj_a, l_a_out, n),
                in_proj_b: Proj::dense(in_proj_b, l_a_out, n),
                in_proj_z: Proj::dense(in_proj_z, l_z_out, n),
                out_proj: Proj::dense(out_proj, n, l_out_in),
                // DeltaNet-specific
                conv1d_weight,
                a_log,
                dt_bias,
                linear_norm,
                // MLP + norms
                gate_proj: Proj::dense(gate_proj, mlp, n),
                up_proj: Proj::dense(up_proj, mlp, n),
                down_proj: Proj::dense(down_proj, n, mlp),
                input_norm,
                post_attn_norm,
            }
        } else {
            let ln = qwen35_attention_gguf_names(i);

            crate::deltanet::weights::DeltaNetLayerWeights {
                // Full attention
                attn_wq: Proj::dense(gguf.dequant_f16_to_f32(&ln.attn_q)?, 2 * q_dim, n),
                attn_wk: Proj::dense(gguf.dequant_f16_to_f32(&ln.attn_k)?, kvd, n),
                attn_wv: Proj::dense(gguf.dequant_f16_to_f32(&ln.attn_v)?, kvd, n),
                attn_wo: Proj::dense(gguf.dequant_f16_to_f32(&ln.attn_output)?, n, q_dim),
                attn_q_norm: gguf.dequant_f16_to_f32(&ln.attn_q_norm)?,
                attn_k_norm: gguf.dequant_f16_to_f32(&ln.attn_k_norm)?,
                // Linear attention: empty for attention layers
                in_proj_qkv: Proj::empty(),
                in_proj_a: Proj::empty(),
                in_proj_b: Proj::empty(),
                in_proj_z: Proj::empty(),
                out_proj: Proj::empty(),
                conv1d_weight: Vec::new(),
                a_log: Vec::new(),
                dt_bias: Vec::new(),
                linear_norm: Vec::new(),
                // MLP + norms
                gate_proj: Proj::dense(gguf.dequant_f16_to_f32(&ln.ffn_gate)?, mlp, n),
                up_proj: Proj::dense(gguf.dequant_f16_to_f32(&ln.ffn_up)?, mlp, n),
                down_proj: Proj::dense(gguf.dequant_f16_to_f32(&ln.ffn_down)?, n, mlp),
                input_norm: gguf.dequant_f16_to_f32(&ln.attn_norm)?,
                post_attn_norm: gguf.dequant_f16_to_f32(&ln.post_attn_norm)?,
            }
        };
        layers.push(layer);
    }

    let weights = crate::deltanet::weights::QwenDeltaNetWeights {
        wte,
        final_norm,
        lm_head,
        layers,
        layer_types,
    };

    Ok((config, weights))
}

// ── GGUF → Qwen3.5 DeltaNet ternary weight loading (Issue 594) ──

/// Load a `Q2_0` tensor from the GGUF file and repack it to
/// [`TernaryGroupWeights`] via [`repack_q2_0_to_ternary_group`].
///
/// GGUF 2D tensor shape is `[ne0=cols, ne1=rows]` (ne0 is the inner/fastest
/// dimension). The repack is row-major: `rows` rows of `cols/128` blocks.
#[cfg(feature = "deltanet_ternary_inference")]
fn load_ternary_proj(gguf: &GgufFile, name: &str) -> Result<katgpt_core::TernaryGroupWeights> {
    use crate::quant::ptq1_0::repack_ptq1_0_to_ternary_group;
    use crate::quant::q2_0::repack_q2_0_to_ternary_group;

    let info = gguf
        .tensor_info(name)
        .with_context(|| format!("tensor '{name}' not found in GGUF file"))?;
    let shape = &info.shape;
    anyhow::ensure!(
        shape.len() == 2,
        "tensor '{name}' has {} dims, expected 2D for a projection",
        shape.len()
    );
    let cols = shape[0]; // ne0 = in_features
    let rows = shape[1]; // ne1 = out_features

    // Issue 980 T6: both Bonsai ternary wire formats repack into the same
    // TernaryGroupWeights container (same trits + f16 group scales for a
    // ternary-g128 checkpoint — the PTQ1_0 pack is just the denser encoding).
    let tg = match info.ggml_type {
        GgmlType::Q2_0 => {
            let blocks = gguf.q2_0_tensor_blocks(name)?;
            repack_q2_0_to_ternary_group(blocks, rows, cols)
                .with_context(|| format!("repack failed for tensor '{name}' [{rows}×{cols}]"))?
        }
        GgmlType::PTQ1_0 => {
            let blocks = gguf.ptq1_0_tensor_blocks(name)?;
            repack_ptq1_0_to_ternary_group(blocks, rows, cols)
                .with_context(|| format!("repack failed for tensor '{name}' [{rows}×{cols}]"))?
        }
        other => bail!(
            "tensor '{name}' is {other:?} — expected Q2_0 (id 42/142) or PTQ1_0 (id 143) for ternary repack"
        ),
    };
    Ok(tg)
}

/// Load one `DeltaNet` gate projection (`ssm_alpha`/`ssm_beta` → `in_proj_a`/`b`)
/// dispatching on the tensor's storage type (Issue 980 T2).
///
/// - `Q2_0` (type 42/142) → ternary repack — the PRE-ROTATION Bonsai lane.
/// - BF16 (type 30) → dense dequant — the Bonsai 2 escape set (`5120 × 48`,
///   full-precision, neither rotated nor quantized).
///
/// Both-file regression: the old file must keep loading byte-identically
/// through the ternary arm.
#[cfg(feature = "deltanet_ternary_inference")]
fn load_gate_proj(
    gguf: &GgufFile,
    name: &str,
) -> Result<crate::deltanet::ternary_weights::GateProjWeights> {
    use crate::deltanet::ternary_weights::GateProjWeights;
    let info = gguf
        .tensor_info(name)
        .with_context(|| format!("tensor '{name}' not found in GGUF file"))?;
    match info.ggml_type {
        GgmlType::Q2_0 | GgmlType::PTQ1_0 => {
            Ok(GateProjWeights::Ternary(load_ternary_proj(gguf, name)?))
        }
        GgmlType::BF16 | GgmlType::F16 | GgmlType::F32 => {
            anyhow::ensure!(
                info.shape.len() == 2,
                "gate projection '{name}' has {} dims, expected 2D",
                info.shape.len()
            );
            let cols = info.shape[0]; // ne0 = in_features (n_embd)
            let rows = info.shape[1]; // ne1 = out_features (n_v_heads)
            let data = gguf.dequant_f16_to_f32(name)?;
            Ok(GateProjWeights::Dense(data, rows, cols))
        }
        other => bail!(
            "gate projection '{name}' is {other:?} — expected Q2_0 (ternary, pre-rotation) or BF16/F16/F32 (dense, Bonsai 2)"
        ),
    }
}

/// Load ternary weights for a hybrid DeltaNet/Attention model from a `Q2_0` GGUF
/// file (Issue 594).
///
/// Mirrors [`load_qwen_deltanet_weights_gguf`] but keeps the `Q2_0` tensors as
/// [`TernaryGroupWeights`] instead of dequantizing to f32. This is the only
/// way to load Ternary-Bonsai-27B within its 6.67 GiB memory budget — the
/// embedding table and LM head are also `Q2_0` (~5.1 GB each as f32).
///
/// **G1 gate:** [`QwenDeltaNetTernaryWeights::invariants_hold`] is checked
/// after load. A mis-parsed `Q2_0_g128` block typically violates the
/// `pos_bits & neg_bits == 0` invariant.
///
/// Returns `Err` if the architecture is not `qwen35`.
#[cfg(feature = "deltanet_ternary_inference")]
pub fn load_qwen_deltanet_ternary_weights_gguf(
    path: &Path,
) -> Result<(
    Config,
    crate::deltanet::ternary_weights::QwenDeltaNetTernaryWeights,
)> {
    use crate::deltanet::ternary_weights::{
        DeltaNetTernaryLayerWeights, GateProjWeights, QwenDeltaNetTernaryWeights,
    };
    use katgpt_core::TernaryGroupWeights;

    let gguf = GgufFile::open(path)?;

    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "qwen35" {
        bail!("expected qwen35 architecture, got '{arch}'");
    }

    let (config, layer_types) = qwen35_deltanet_config_from_gguf_metadata(&gguf)?;
    let n_layer = config.n_layer;

    // Issue 980 (T1): a file declaring prism.hadamard must be honored or
    // refused LOUDLY. Without the bonsai2_hadamard feature we cannot apply
    // the rotation, so a folded file refuses here instead of producing
    // stock-llama.cpp-class garbage (the dev repo's Q2_0-prism-fork-required
    // fixture exists to make that failure loud).
    #[cfg(feature = "bonsai2_hadamard")]
    let rotation = crate::deltanet::rotation::parse_prism_hadamard(&gguf)?;
    #[cfg(not(feature = "bonsai2_hadamard"))]
    {
        if gguf.metadata_u64("prism.hadamard.version").is_some() {
            bail!(
                "GGUF declares prism.hadamard (Hadamard-folded ternary) but the \"bonsai2_hadamard\" \
                 feature is OFF — refusing to load a rotated file we cannot honor. \
                 Rebuild with --features bonsai2_hadamard."
            );
        }
    }
    #[cfg(not(feature = "bonsai2_hadamard"))]
    let rotation: Option<crate::deltanet::rotation::TernaryRotationConfig> = None;

    // Global weights — token_embd.weight and output.weight are Q2_0 (type 42).
    // Staying ternary saves ~10 GB vs dequantizing to f32.
    let wte = load_ternary_proj(&gguf, "token_embd.weight")?;
    let final_norm = gguf.dequant_f16_to_f32("output_norm.weight")?;
    // output.weight may be absent if tied; check before loading.
    let lm_head = if gguf.tensor_info("output.weight").is_some() {
        load_ternary_proj(&gguf, "output.weight")?
    } else {
        // Tied embeddings — clone the bit-plane layout. TernaryGroupWeights is
        // all-Vec, so this is a deep copy of the bit-planes + scales.
        wte.clone()
    };

    let mut layers = Vec::with_capacity(n_layer);
    for (i, lt) in layer_types.iter().enumerate() {
        let is_linear = *lt == DeltaNetLayerType::DeltaNet;

        let layer = if is_linear {
            let ln = qwen35_deltanet_gguf_names(i);

            // Ternary projections (Q2_0 → repack)
            let in_proj_qkv = load_ternary_proj(&gguf, &ln.attn_qkv)?;
            let in_proj_z = load_ternary_proj(&gguf, &ln.attn_gate)?;
            // Issue 980 T2: a/b dispatch on storage type (ternary old file /
            // dense BF16 Bonsai 2).
            let in_proj_a = load_gate_proj(&gguf, &ln.ssm_alpha)?;
            let in_proj_b = load_gate_proj(&gguf, &ln.ssm_beta)?;
            let out_proj = load_ternary_proj(&gguf, &ln.ssm_out)?;
            let gate_proj = load_ternary_proj(&gguf, &ln.ffn_gate)?;
            let up_proj = load_ternary_proj(&gguf, &ln.ffn_up)?;
            let down_proj = load_ternary_proj(&gguf, &ln.ffn_down)?;

            // Dense fields (F32 in GGUF)
            let conv1d_weight = gguf.dequant_f16_to_f32(&ln.ssm_conv1d)?;
            let a_log = gguf.dequant_f16_to_f32(&ln.ssm_a)?;
            let dt_bias = gguf.dequant_f16_to_f32(&ln.ssm_dt)?;
            let linear_norm = gguf.dequant_f16_to_f32(&ln.ssm_norm)?;
            let input_norm = gguf.dequant_f16_to_f32(&ln.attn_norm)?;
            let post_attn_norm = gguf.dequant_f16_to_f32(&ln.post_attn_norm)?;

            DeltaNetTernaryLayerWeights {
                // Full attention: empty for DeltaNet layers
                attn_wq: TernaryGroupWeights::new(0, 0),
                attn_wk: TernaryGroupWeights::new(0, 0),
                attn_wv: TernaryGroupWeights::new(0, 0),
                attn_wo: TernaryGroupWeights::new(0, 0),
                // Linear attention
                in_proj_qkv,
                in_proj_a,
                in_proj_b,
                in_proj_z,
                out_proj,
                // SwiGLU MLP
                gate_proj,
                up_proj,
                down_proj,
                // Dense fields
                attn_q_norm: Vec::new(),
                attn_k_norm: Vec::new(),
                conv1d_weight,
                a_log,
                dt_bias,
                linear_norm,
                input_norm,
                post_attn_norm,
            }
        } else {
            let ln = qwen35_attention_gguf_names(i);

            // Ternary projections (Q2_0 → repack)
            // NOTE (Issue 594): blk.N.attn_q is [5120 × 12288] = q concatenated
            // with a gate. The loader loads it as-is; the forward pass splits.
            let attn_wq = load_ternary_proj(&gguf, &ln.attn_q)?;
            let attn_wk = load_ternary_proj(&gguf, &ln.attn_k)?;
            let attn_wv = load_ternary_proj(&gguf, &ln.attn_v)?;
            let attn_wo = load_ternary_proj(&gguf, &ln.attn_output)?;
            let gate_proj = load_ternary_proj(&gguf, &ln.ffn_gate)?;
            let up_proj = load_ternary_proj(&gguf, &ln.ffn_up)?;
            let down_proj = load_ternary_proj(&gguf, &ln.ffn_down)?;

            // Dense fields (F32 in GGUF)
            let attn_q_norm = gguf.dequant_f16_to_f32(&ln.attn_q_norm)?;
            let attn_k_norm = gguf.dequant_f16_to_f32(&ln.attn_k_norm)?;
            let input_norm = gguf.dequant_f16_to_f32(&ln.attn_norm)?;
            let post_attn_norm = gguf.dequant_f16_to_f32(&ln.post_attn_norm)?;

            DeltaNetTernaryLayerWeights {
                // Full attention
                attn_wq,
                attn_wk,
                attn_wv,
                attn_wo,
                // Linear attention: empty for attention layers
                in_proj_qkv: TernaryGroupWeights::new(0, 0),
                in_proj_a: GateProjWeights::empty(),
                in_proj_b: GateProjWeights::empty(),
                in_proj_z: TernaryGroupWeights::new(0, 0),
                out_proj: TernaryGroupWeights::new(0, 0),
                // SwiGLU MLP
                gate_proj,
                up_proj,
                down_proj,
                // Dense fields
                attn_q_norm,
                attn_k_norm,
                conv1d_weight: Vec::new(),
                a_log: Vec::new(),
                dt_bias: Vec::new(),
                linear_norm: Vec::new(),
                input_norm,
                post_attn_norm,
            }
        };
        layers.push(layer);
    }

    // Cross-check the parsed folded set against OUR structural map (the
    // fold is honored by construction in the forward: qkv/z/ssm_out/ffn/
    // attn/lm_head rotate their inputs; a/b never do). An unknown folded
    // weight would silently skip its rotation — refuse instead (the fork's
    // is_foldable_weight posture).
    if rotation.is_some() {
        use crate::deltanet::rotation as r;
        for name_val in gguf
            .metadata_array("prism.hadamard.weight_names")
            .unwrap_or(&[])
        {
            let Some(name) = name_val.as_str() else {
                bail!("prism.hadamard weight name not a string");
            };
            if !r::is_known_folded_name(name, n_layer, &layer_types) {
                bail!(
                    "prism.hadamard weight '{name}' is not on this engine's verified Hadamard-aware matmul path — refusing"
                );
            }
        }
    }

    let weights = QwenDeltaNetTernaryWeights {
        wte,
        final_norm,
        lm_head,
        layers,
        layer_types,
        rotation,
    };

    // G1 post-load sanity gate: every non-empty projection satisfies the
    // bit-plane invariant (pos_bits & neg_bits == 0). A mis-parsed Q2_0 block
    // typically fails this.
    anyhow::ensure!(
        weights.invariants_hold(),
        "G1 gate failed: one or more ternary projections violate the bit-plane invariant"
    );

    Ok((config, weights))
}
///
/// Returns `(Config, layer_types)` since `layer_types` is needed during weight loading.
///
/// Key metadata fields:
/// - `qwen35.ssm.conv_kernel` — conv kernel size (default 4)
/// - `qwen35.ssm.state_size` — head dim (`ssm_d_state`)
/// - `qwen35.ssm.time_step_rank` — `n_v_heads` (`ssm_dt_rank`)
/// - `qwen35.ssm.group_count` — `n_k_heads` (`ssm_n_group`)
/// - `qwen35.full_attention_interval` — full attention every N layers (default 4)
/// - `qwen35.nextn_predict_layers` — MTP layer count (subtracted from main layers)
#[cfg(feature = "deltanet_inference")]
fn qwen35_deltanet_config_from_gguf_metadata(
    gguf: &GgufFile,
) -> Result<(Config, Vec<DeltaNetLayerType>)> {
    let prefix = "qwen35.";

    let n_embd = gguf
        .metadata_u64(&format!("{prefix}embedding_length"))
        .unwrap_or(1024) as usize;
    let n_layer = gguf
        .metadata_u64(&format!("{prefix}block_count"))
        .unwrap_or(24) as usize;
    let mlp_hidden = gguf
        .metadata_u64(&format!("{prefix}feed_forward_length"))
        .unwrap_or(3072) as usize;
    let n_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count"))
        .unwrap_or(16) as usize;
    let n_kv_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count_kv"))
        .unwrap_or(2) as usize;
    let head_dim = gguf
        .metadata_u64(&format!("{prefix}attention.key_length"))
        .unwrap_or(256) as usize;
    let rms_norm_eps = gguf
        .metadata_f64(&format!("{prefix}attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-6);
    let context_length = gguf
        .metadata_u64(&format!("{prefix}context_length"))
        .unwrap_or(32768) as usize;

    // DeltaNet-specific dimensions
    let conv_kernel = gguf
        .metadata_u64(&format!("{prefix}ssm.conv_kernel"))
        .unwrap_or(4) as usize;
    let state_size = gguf
        .metadata_u64(&format!("{prefix}ssm.state_size"))
        .unwrap_or(128) as usize; // head dim for linear attention
    let n_v_heads = gguf
        .metadata_u64(&format!("{prefix}ssm.time_step_rank"))
        .unwrap_or(16) as usize;
    let n_k_heads = gguf
        .metadata_u64(&format!("{prefix}ssm.group_count"))
        .unwrap_or(16) as usize;

    // Issue 594: partial RoPE dimension count. Qwen3.5 GGUF stores it as
    // `rope.dimension_count` (= 64 for Ternary-Bonsai-27B, head_dim=256).
    // 0 = not present → Config sentinel for full rotation (head_dim).
    let rope_dimension_count = gguf
        .metadata_u64(&format!("{prefix}rope.dimension_count"))
        .unwrap_or(0) as usize;

    // Issue 594: RoPE theta MUST come from the file. Ternary-Bonsai-27B sets
    // `qwen35.rope.freq_base = 1e7`; the `Config::gemma2_2b()` seed below
    // defaults to 1e4, so leaving it unread rotated every full-attention layer
    // (16 of 64) at a 1000× wrong frequency.
    let rope_freq_base = gguf
        .metadata_f64(&format!("{prefix}rope.freq_base"))
        .unwrap_or(10_000.0) as f32;

    // Layer type computation (matches llama.cpp qwen35.cpp logic)
    let full_attn_interval = gguf
        .metadata_u64(&format!("{prefix}full_attention_interval"))
        .unwrap_or(4) as usize;
    let nextn_predict = gguf
        .metadata_u64(&format!("{prefix}nextn_predict_layers"))
        .unwrap_or(0) as usize;
    let n_main = n_layer.saturating_sub(nextn_predict);

    // Layer types for the MAIN stack only. `nextn_predict_layers` counts
    // MTP draft blocks stored as trailing `blk.{n_main}..` layers (llama.cpp
    // folds the model's MTP head into the GGUF as `blk.N.nextn.*` tensors,
    // block_count includes it). The MTP block is NOT a main-stack layer —
    // its input contract is `eh_proj(concat(emb, hidden))`, never the main
    // residual stream — so it must not be typed or loaded as one (Issue 742
    // T9.2: the dbirks dense GGUF carries block_count=65 / nextn=1).
    // Loading it as a regular attention layer would run 65 layers in
    // decode — structurally valid tensor names, semantically garbage.
    let mut layer_types: Vec<DeltaNetLayerType> = Vec::with_capacity(n_main);
    layer_types.extend((0..n_main).map(|i| {
        if !(i + 1).is_multiple_of(full_attn_interval) {
            DeltaNetLayerType::DeltaNet
        } else {
            DeltaNetLayerType::Attention
        }
    }));

    // Get vocab_size from token_embd tensor shape
    let vocab_size = gguf
        .tensor_info("token_embd.weight")
        .and_then(|info| info.shape.last().copied())
        .unwrap_or(248_320);

    let mut config = Config::gemma2_2b(); // Start from a reasonable default
    config.vocab_size = vocab_size;
    config.block_size = context_length;
    config.n_embd = n_embd;
    config.n_head = n_head;
    config.head_dim = head_dim;
    config.mlp_hidden = mlp_hidden;
    config.n_layer = n_main; // main stack only — MTP blocks excluded (nextn)
    config.n_kv_head = n_kv_head;
    config.rms_norm_eps = rms_norm_eps;
    config.rms_norm_offset = false;
    config.post_norm = false;
    config.attn_logit_softcapping = 0.0;
    config.final_logit_softcapping = 0.0;
    config.model_arch = ModelArchitecture::QwenDeltaNet;
    config.use_rope = true;
    // Untied iff the GGUF carries an explicit lm_head (`output.weight`).
    // Bonsai-era Qwen3.5 GGUFs tie; the dbirks dense GGUF does NOT
    // (`tie_word_embeddings: false` upstream + a separate Q6_K output.weight).
    config.tied_embeddings = gguf.tensor_info("output.weight").is_none();

    // DeltaNet-specific config fields
    config.layer_types = layer_types.clone();
    config.deltanet_conv_kernel_size = conv_kernel;
    config.deltanet_state_dim = n_k_heads * state_size * state_size;
    config.deltanet_linear_head_dim = state_size;
    config.deltanet_linear_n_heads = n_k_heads;
    config.deltanet_linear_n_value_heads = n_v_heads;
    config.rope_dimension_count = rope_dimension_count;
    config.rope_theta = rope_freq_base;

    Ok((config, layer_types))
}

// ── Gemma 4 loader (Issue 577) ────────────────────────────────────────────
//
// Loads Gemma 4 unified text models from GGUF. The architecture alternates
// between sliding-window attention layers (5 of every 6) and full-attention
// layers (1 of every 6, at index % 6 == 5) — see `Config::gemma4_12b` for
// the per-layer shape derivation. Tensor names follow llama.cpp's
// `src/models/src/gemma4.cpp`.

/// Load Gemma 4 weights from a GGUF file. Returns the config + weights.
///
/// The returned `Config.gemma4_layer_types` is built from the SWA pattern
// metadata (`gemma4.attention.sliding_window_pattern`); per-layer KV dim
// can then be derived via `kv_dim_for`.
#[cfg(feature = "gemma4_inference")]
pub fn load_gemma4_weights_gguf(
    path: &Path,
) -> Result<(Config, crate::transformer::gemma4::Gemma4TransformerWeights)> {
    let gguf = GgufFile::open(path)?;

    // Validate architecture.
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "gemma4" {
        bail!("expected gemma4 architecture, got '{arch}'");
    }

    let (config, layer_types) = gemma4_config_from_gguf_metadata(&gguf)?;
    let n_layer = config.n_layer;

    // Global weights.
    let wte = gguf.dequant_f16_to_f32("token_embd.weight")?;
    let final_norm = gguf.dequant_f16_to_f32("output_norm.weight")?;
    // output.weight is optional (TIED with token_embd when absent).
    // When present, use it; otherwise reuse wte (tied).
    let _lm_head: Option<Vec<f32>> = match gguf.tensor_info("output.weight") {
        Some(_) => Some(gguf.dequant_f16_to_f32("output.weight")?),
        None => None,
    };

    let mut layers = Vec::with_capacity(n_layer);
    for (i, &layer_type) in layer_types.iter().enumerate().take(n_layer) {
        let ln = gemma4_gguf_layer_names(i);

        // V is optional per `attention_k_eq_v`: if `attn_v.weight` is missing,
        // V = K. Detect + fall back to K's tensor.
        let attn_wv = match gguf.tensor_info(&ln.attn_v) {
            Some(_) => gguf.dequant_f16_to_f32(&ln.attn_v)?,
            None => gguf.dequant_f16_to_f32(&ln.attn_k)?,
        };

        // layer_output_scale is optional (Issue 397). When present (Gemma-4
        // 12B stores ~0.053), it prevents activation explosion from the large
        // norm gammas. Default 1.0 when absent.
        let layer_output_scale: f32 = match gguf.tensor_info(&ln.layer_output_scale) {
            Some(_) => {
                let v = gguf.dequant_f16_to_f32(&ln.layer_output_scale)?;
                v.first().copied().unwrap_or(1.0)
            }
            None => 1.0,
        };

        let layer = crate::transformer::gemma4::Gemma4LayerWeights {
            attn_wq: gguf.dequant_f16_to_f32(&ln.attn_q)?,
            attn_wk: gguf.dequant_f16_to_f32(&ln.attn_k)?,
            attn_wv,
            attn_wo: gguf.dequant_f16_to_f32(&ln.attn_output)?,
            attn_q_norm: gguf.dequant_f16_to_f32(&ln.attn_q_norm)?,
            attn_k_norm: gguf.dequant_f16_to_f32(&ln.attn_k_norm)?,
            gate_proj: gguf.dequant_f16_to_f32(&ln.ffn_gate)?,
            up_proj: gguf.dequant_f16_to_f32(&ln.ffn_up)?,
            down_proj: gguf.dequant_f16_to_f32(&ln.ffn_down)?,
            input_norm: gguf.dequant_f16_to_f32(&ln.attn_norm)?,
            post_attn_norm: gguf.dequant_f16_to_f32(&ln.attn_post_norm)?,
            pre_mlp_norm: gguf.dequant_f16_to_f32(&ln.ffn_norm)?,
            post_mlp_norm: gguf.dequant_f16_to_f32(&ln.ffn_post_norm)?,
            layer_output_scale,
            layer_type,
        };
        layers.push(layer);
    }

    let weights = crate::transformer::gemma4::Gemma4TransformerWeights {
        wte,
        final_norm,
        layers,
    };

    Ok((config, weights))
}

/// Build `Config` from GGUF metadata keys for Gemma 4.
#[cfg(feature = "gemma4_inference")]
pub fn gemma4_config_from_gguf_metadata(
    gguf: &GgufFile,
) -> Result<(Config, Vec<katgpt_core::types::Gemma4LayerType>)> {
    use katgpt_core::types::{Gemma4LayerType, ModelArchitecture};

    let prefix = "gemma4.";

    let context_length = gguf
        .metadata_u64(&format!("{prefix}context_length"))
        .unwrap_or(262_144) as usize;
    let n_embd = gguf
        .metadata_u64(&format!("{prefix}embedding_length"))
        .unwrap_or(3840) as usize;
    let n_layer = gguf
        .metadata_u64(&format!("{prefix}block_count"))
        .unwrap_or(48) as usize;
    let mlp_hidden = gguf
        .metadata_u64(&format!("{prefix}feed_forward_length"))
        .unwrap_or(15_360) as usize;
    let n_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count"))
        .unwrap_or(16) as usize;
    let n_kv_head = gguf
        .metadata_u64(&format!("{prefix}attention.head_count_kv"))
        .unwrap_or(8) as usize;
    let rms_norm_eps = gguf
        .metadata_f64(&format!("{prefix}attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-6);
    // Gemma 4 GGUF convention (from llama.cpp):
    //   `key_length`     = Full-attention (global) head dim  [512 for 12B]
    //   `key_length_swa` = Sliding-attention head dim       [256 for 12B]
    // The naming is counter-intuitive: `key_length` (no suffix) is the FULL
    // attention key length, NOT the sliding one. This was backwards in the
    // initial loader (Issue 395) and caused q_dim_for(Sliding) to compute
    // 16×512=8192 instead of 16×256=4096, panicking the first Sliding layer.
    let head_dim = gguf
        .metadata_u64(&format!("{prefix}attention.key_length_swa"))
        .unwrap_or(256) as usize;
    let global_head_dim = gguf
        .metadata_u64(&format!("{prefix}attention.key_length"))
        // Fall back to head_dim if not present (older converters).
        .unwrap_or(head_dim as u64) as usize;
    let n_global_kv_head = gguf
        .metadata_u64(&format!("{prefix}attention.global_head_count_kv"))
        // Gemma 4 default is 1 (MQA) on full-attention layers.
        .unwrap_or(1) as usize;
    let sliding_window = gguf
        .metadata_u64(&format!("{prefix}attention.sliding_window"))
        .unwrap_or(1024) as usize;
    // `sliding_window_pattern` = every Nth layer is full-attention.
    // Pattern: dense_first=false, so layer at idx `% n_pattern == n_pattern - 1`
    // is full-attention (matches llama.cpp `set_swa_pattern(N, false)`).
    let swa_pattern = gguf
        .metadata_u64(&format!("{prefix}attention.sliding_window_pattern"))
        .unwrap_or(6) as usize;
    let rope_theta = gguf
        .metadata_f64(&format!("{prefix}rope.freq_base"))
        // Sliding-layer theta — may also be keyed as `rope_theta_swa`.
        .or_else(|| gguf.metadata_f64("gemma4.rope_theta_swa"))
        .unwrap_or(10_000.0) as f32;
    let rope_theta_full = gguf
        .metadata_f64(&format!("{prefix}rope.freq_base_train"))
        .or_else(|| gguf.metadata_f64("gemma4.rope_theta"))
        .unwrap_or(1_000_000.0) as f32;
    let final_logit_softcapping = gguf
        .metadata_f64(&format!("{prefix}final_logit_softcapping"))
        .unwrap_or(30.0) as f32;

    // vocab_size from token_embd shape. GGUF stores token_embd as
    // [n_embd, vocab_size] (inner dim first) — use shape.last() per the
    // Gemma 2 precedent.
    let vocab_size = gguf
        .tensor_info("token_embd.weight")
        .and_then(|info| info.shape.last().copied())
        .unwrap_or(262_144);

    // Build per-layer type pattern.
    let layer_types: Vec<Gemma4LayerType> = (0..n_layer)
        .map(|i| {
            if swa_pattern > 0 && i % swa_pattern == swa_pattern.saturating_sub(1) {
                Gemma4LayerType::Full
            } else {
                Gemma4LayerType::Sliding
            }
        })
        .collect();

    // Start from the gemma4_12b preset (canonical values for the canonical
    // model) and override with metadata. This keeps the loader robust to
    // missing keys for fields that are stable across the Gemma 4 family
    // (post-norm, attention_scale=1.0, gelu_pytorch_tanh activation, etc).
    let mut config = Config::gemma4_12b();
    config.vocab_size = vocab_size;
    config.block_size = context_length;
    config.n_embd = n_embd;
    config.n_head = n_head;
    config.head_dim = head_dim;
    config.global_head_dim = global_head_dim;
    config.n_layer = n_layer;
    config.n_kv_head = n_kv_head;
    config.n_global_kv_head = n_global_kv_head;
    config.mlp_hidden = mlp_hidden;
    config.sliding_window = sliding_window;
    config.rms_norm_eps = rms_norm_eps;
    config.rope_theta = rope_theta;
    config.rope_theta_full = rope_theta_full;
    config.final_logit_softcapping = final_logit_softcapping;
    config.gemma4_layer_types = layer_types.clone();
    config.model_arch = ModelArchitecture::Gemma4;

    Ok((config, layer_types))
}

/// GGUF tensor names for a single Gemma 4 layer.
#[cfg(feature = "gemma4_inference")]
pub struct Gemma4GgufLayerNames {
    pub attn_norm: String,
    pub attn_q: String,
    pub attn_k: String,
    pub attn_v: String,
    pub attn_output: String,
    pub attn_q_norm: String,
    pub attn_k_norm: String,
    pub attn_post_norm: String,
    pub ffn_norm: String,
    pub ffn_gate: String,
    pub ffn_up: String,
    pub ffn_down: String,
    pub ffn_post_norm: String,
    pub layer_output_scale: String,
}

#[cfg(feature = "gemma4_inference")]
pub fn gemma4_gguf_layer_names(layer_idx: usize) -> Gemma4GgufLayerNames {
    let i = layer_idx;
    Gemma4GgufLayerNames {
        attn_norm: format!("blk.{i}.attn_norm.weight"),
        attn_q: format!("blk.{i}.attn_q.weight"),
        attn_k: format!("blk.{i}.attn_k.weight"),
        attn_v: format!("blk.{i}.attn_v.weight"),
        attn_output: format!("blk.{i}.attn_output.weight"),
        attn_q_norm: format!("blk.{i}.attn_q_norm.weight"),
        attn_k_norm: format!("blk.{i}.attn_k_norm.weight"),
        attn_post_norm: format!("blk.{i}.post_attention_norm.weight"),
        ffn_norm: format!("blk.{i}.ffn_norm.weight"),
        ffn_gate: format!("blk.{i}.ffn_gate.weight"),
        ffn_up: format!("blk.{i}.ffn_up.weight"),
        ffn_down: format!("blk.{i}.ffn_down.weight"),
        ffn_post_norm: format!("blk.{i}.post_ffw_norm.weight"),
        layer_output_scale: format!("blk.{i}.layer_output_scale.weight"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_from_id() {
        assert_eq!(GgmlType::from_id(0), Some(GgmlType::F32));
        assert_eq!(GgmlType::from_id(1), Some(GgmlType::F16));
        assert_eq!(GgmlType::from_id(12), Some(GgmlType::Q4_K));
        assert_eq!(GgmlType::from_id(42), Some(GgmlType::Q2_0));
        assert_eq!(GgmlType::from_id(99), None);
        // Issue 717: upstream ggml has BF16 = 30 (29 is IQ1_M). This was
        // mapped to 29, so every real BF16 GGUF failed to open. Confirmed
        // against Ternary-Bonsai-27B-dspark-Q4_1.gguf, whose type-30 tensors
        // measure exactly 2.0 bytes/element.
        assert_eq!(GgmlType::from_id(30), Some(GgmlType::BF16));
        assert_eq!(GgmlType::from_id(29), None, "29 is IQ1_M, unsupported");
        assert_eq!(GgmlType::BF16.tensor_bytes(1000), 2000);
    }

    /// `Q4_1` had block info (20 B / 32 weights) but no decoder, so any `Q4_1`
    /// tensor failed at dequant time (Issue 717 — `dspark.markov_head_b`).
    ///
    /// Pins the llama.cpp nibble convention: `qs[j]` low nibble is weight `j`,
    /// high nibble is weight `j + 16`, value = `d * q + m`. A transposed
    /// convention would permute values within each 32-block — silently
    /// corrupting rows rather than erroring.
    #[test]
    fn test_dequantize_row_q4_1_matches_llamacpp_layout() {
        let d = half::f16::from_f32(0.5);
        let m = half::f16::from_f32(-1.0);
        let mut block = Vec::with_capacity(20);
        block.extend_from_slice(&d.to_bits().to_le_bytes());
        block.extend_from_slice(&m.to_bits().to_le_bytes());
        // qs[j] = low nibble j, high nibble (15 - j).
        for j in 0..16u8 {
            block.push((j & 0x0F) | ((15 - j) << 4));
        }
        let mut out = vec![0.0f32; 32];
        dequantize_row_q4_1(&block, &mut out).unwrap();
        for j in 0..16usize {
            // weight j  <- low nibble  = j
            assert_eq!(out[j], 0.5 * j as f32 - 1.0, "low nibble at {j}");
            // weight j+16 <- high nibble = 15 - j
            assert_eq!(
                out[j + 16],
                0.5 * (15 - j) as f32 - 1.0,
                "high nibble at {}",
                j + 16
            );
        }

        // Multi-block: second block must start at element 32, not overlap.
        let mut two = block.clone();
        two.extend_from_slice(&block);
        let mut out2 = vec![0.0f32; 64];
        dequantize_row_q4_1(&two, &mut out2).unwrap();
        assert_eq!(&out2[..32], &out[..]);
        assert_eq!(&out2[32..], &out[..]);

        // Short buffer and non-multiple-of-32 length must error, not panic.
        assert!(dequantize_row_q4_1(&block, &mut vec![0.0; 64]).is_err());
        assert!(dequantize_row_q4_1(&two, &mut [0.0; 33]).is_err());
    }

    /// `Q4_0` = `Q4_1` without the `m` offset (18 B / 32 weights). Pins the same
    /// llama.cpp nibble convention plus the `d * (q - 8)` decoding — the
    /// offset-8 subtraction is the one place a `Q4_0` decoder can silently
    /// diverge (missing it produces all-positive weights).
    #[test]
    fn test_dequantize_row_q4_0_matches_llamacpp_layout() {
        let d = half::f16::from_f32(0.5);
        let mut block = Vec::with_capacity(18);
        block.extend_from_slice(&d.to_bits().to_le_bytes());
        // qs[j] = low nibble j, high nibble (15 - j).
        for j in 0..16u8 {
            block.push((j & 0x0F) | ((15 - j) << 4));
        }
        let mut out = vec![0.0f32; 32];
        dequantize_row_q4_0(&block, &mut out).unwrap();
        for j in 0..16usize {
            // weight j  <- low nibble  = j       → 0.5 * (j - 8)
            assert_eq!(out[j], 0.5 * (j as f32 - 8.0), "low nibble at {j}");
            // weight j+16 <- high nibble = 15 - j → 0.5 * (7 - j)
            assert_eq!(
                out[j + 16],
                0.5 * (15 - j) as f32 - 4.0,
                "high nibble at {}",
                j + 16
            );
        }

        // Multi-block: second block must start at element 32, not overlap.
        let mut two = block.clone();
        two.extend_from_slice(&block);
        let mut out2 = vec![0.0f32; 64];
        dequantize_row_q4_0(&two, &mut out2).unwrap();
        assert_eq!(&out2[..32], &out[..]);
        assert_eq!(&out2[32..], &out[..]);

        // Short buffer and non-multiple-of-32 length must error, not panic.
        assert!(dequantize_row_q4_0(&block, &mut vec![0.0; 64]).is_err());
        assert!(dequantize_row_q4_0(&two, &mut [0.0; 33]).is_err());
    }

    #[test]
    fn test_q4k_tensor_bytes() {
        // 256 weights → 144 bytes
        assert_eq!(GgmlType::Q4_K.tensor_bytes(256), 144);
        // 2304 weights → 9 blocks × 144 = 1296
        assert_eq!(GgmlType::Q4_K.tensor_bytes(2304), 1296);
    }

    #[test]
    fn test_gguf_value_as_u64() {
        let v = GgufValue::U32(42);
        assert_eq!(v.as_u64(), Some(42));
        let v = GgufValue::I32(-1);
        assert_eq!(v.as_u64(), None);
    }

    #[test]
    fn test_gguf_value_as_str() {
        let v = GgufValue::String("gemma2".to_string());
        assert_eq!(v.as_str(), Some("gemma2"));
    }

    #[test]
    fn test_layer_names() {
        let ln = gguf_layer_names(0);
        assert_eq!(ln.attn_q, "blk.0.attn_q.weight");
        assert_eq!(ln.ffn_gate, "blk.0.ffn_gate.weight");
        assert_eq!(ln.post_ffw_norm, "blk.0.post_ffw_norm.weight");
    }

    #[test]
    fn test_llama_layer_names() {
        let ln = llama_gguf_layer_names(0);
        assert_eq!(ln.attn_norm, "blk.0.attn_norm.weight");
        assert_eq!(ln.attn_q, "blk.0.attn_q.weight");
        assert_eq!(ln.attn_k, "blk.0.attn_k.weight");
        assert_eq!(ln.attn_v, "blk.0.attn_v.weight");
        assert_eq!(ln.attn_output, "blk.0.attn_output.weight");
        assert_eq!(ln.ffn_norm, "blk.0.ffn_norm.weight");
        assert_eq!(ln.ffn_gate, "blk.0.ffn_gate.weight");
        assert_eq!(ln.ffn_up, "blk.0.ffn_up.weight");
        assert_eq!(ln.ffn_down, "blk.0.ffn_down.weight");
        // Layer 5
        let ln5 = llama_gguf_layer_names(5);
        assert_eq!(ln5.attn_q, "blk.5.attn_q.weight");
        assert_eq!(ln5.ffn_norm, "blk.5.ffn_norm.weight");
    }

    #[test]
    fn test_qwen35_deltanet_layer_names() {
        let ln = qwen35_deltanet_gguf_names(0);
        assert_eq!(ln.attn_qkv, "blk.0.attn_qkv.weight");
        assert_eq!(ln.attn_gate, "blk.0.attn_gate.weight");
        assert_eq!(ln.ssm_conv1d, "blk.0.ssm_conv1d.weight");
        assert_eq!(ln.ssm_dt, "blk.0.ssm_dt.bias");
        assert_eq!(ln.ssm_a, "blk.0.ssm_a");
        assert_eq!(ln.ssm_beta, "blk.0.ssm_beta.weight");
        assert_eq!(ln.ssm_alpha, "blk.0.ssm_alpha.weight");
        assert_eq!(ln.ssm_norm, "blk.0.ssm_norm.weight");
        assert_eq!(ln.ssm_out, "blk.0.ssm_out.weight");
        assert_eq!(ln.attn_norm, "blk.0.attn_norm.weight");
        assert_eq!(ln.post_attn_norm, "blk.0.post_attention_norm.weight");
        assert_eq!(ln.ffn_gate, "blk.0.ffn_gate.weight");
        assert_eq!(ln.ffn_up, "blk.0.ffn_up.weight");
        assert_eq!(ln.ffn_down, "blk.0.ffn_down.weight");
        // Layer 7
        let ln7 = qwen35_deltanet_gguf_names(7);
        assert_eq!(ln7.attn_qkv, "blk.7.attn_qkv.weight");
        assert_eq!(ln7.ssm_a, "blk.7.ssm_a");
    }

    #[test]
    fn test_qwen35_attention_layer_names() {
        let ln = qwen35_attention_gguf_names(3);
        assert_eq!(ln.attn_q, "blk.3.attn_q.weight");
        assert_eq!(ln.attn_k, "blk.3.attn_k.weight");
        assert_eq!(ln.attn_v, "blk.3.attn_v.weight");
        assert_eq!(ln.attn_output, "blk.3.attn_output.weight");
        assert_eq!(ln.attn_norm, "blk.3.attn_norm.weight");
        assert_eq!(ln.post_attn_norm, "blk.3.post_attention_norm.weight");
        assert_eq!(ln.ffn_gate, "blk.3.ffn_gate.weight");
    }

    #[test]
    #[cfg(feature = "deltanet_inference")]
    fn test_qwen35_layer_type_computation() {
        // Simulate the layer type computation from metadata
        // Qwen3.5-0.8B: 24 layers, full_attention_interval=4, nextn_predict=0
        let n_layer = 24;
        let full_attn_interval = 4;
        let nextn_predict = 0;
        let n_main = n_layer - nextn_predict;

        let layer_types: Vec<DeltaNetLayerType> = (0..n_layer)
            .map(|i| {
                if i < n_main && (i + 1) % full_attn_interval != 0 {
                    DeltaNetLayerType::DeltaNet
                } else {
                    DeltaNetLayerType::Attention
                }
            })
            .collect();

        // Verify: layers 3, 7, 11, 15, 19, 23 are full attention (6 total)
        // All others are DeltaNet (18 total)
        let attn_count = layer_types
            .iter()
            .filter(|&&lt| lt == DeltaNetLayerType::Attention)
            .count();
        let deltanet_count = layer_types
            .iter()
            .filter(|&&lt| lt == DeltaNetLayerType::DeltaNet)
            .count();
        assert_eq!(attn_count, 6);
        assert_eq!(deltanet_count, 18);

        // Verify specific layers
        assert_eq!(layer_types[0], DeltaNetLayerType::DeltaNet); // (0+1)%4=1 ≠ 0
        assert_eq!(layer_types[3], DeltaNetLayerType::Attention); // (3+1)%4=0
        assert_eq!(layer_types[7], DeltaNetLayerType::Attention); // (7+1)%4=0
        assert_eq!(layer_types[11], DeltaNetLayerType::Attention); // (11+1)%4=0
        assert_eq!(layer_types[23], DeltaNetLayerType::Attention); // (23+1)%4=0
    }

    /// Issue 742 T9.2: an MTP-bearing GGUF (`block_count = n_main + nextn`)
    /// must yield EXACTLY `n_main` main-stack layers — the trailing MTP
    /// block (llama.cpp's `blk.{n_main}.nextn.*`) is a draft head, never a
    /// main-stack layer. Loading it as a regular attention layer would run
    /// 65 layers on the dbirks dense model — structurally valid names,
    /// semantically garbage.
    #[test]
    #[cfg(feature = "deltanet_inference")]
    fn test_qwen35_nextn_excludes_mtp_from_main_stack() {
        // dbirks Qwen3.8-27B shape: block_count=65 (64 main + 1 MTP),
        // nextn_predict_layers=1, full_attention_interval=4.
        let n_layer_raw = 65;
        let nextn_predict = 1;
        let n_main = n_layer_raw - nextn_predict;
        assert_eq!(n_main, 64);

        // The fixed computation: layer types over 0..n_main only.
        let layer_types: Vec<DeltaNetLayerType> = (0..n_main)
            .map(|i| {
                if (i + 1) % 4 != 0 {
                    DeltaNetLayerType::DeltaNet
                } else {
                    DeltaNetLayerType::Attention
                }
            })
            .collect();

        assert_eq!(layer_types.len(), 64, "MTP block excluded");
        assert_eq!(
            layer_types
                .iter()
                .filter(|&&lt| lt == DeltaNetLayerType::Attention)
                .count(),
            16,
            "16 full-attention layers (3, 7, ..., 63)"
        );
        assert_eq!(layer_types[63], DeltaNetLayerType::Attention);
        assert_eq!(layer_types[62], DeltaNetLayerType::DeltaNet);
    }

    /// Issue 742 T9.2 real-file gate: the dbirks/Qwen3.8-27B GGUF (either
    /// the `Q4_K_M` deploy artifact or the BF16 parity artifact) parses to a
    /// 64-main-layer untied config, and the MTP tensors remain present in
    /// the file (for the future T3 drafter bring-up).
    /// Skips when the GGUF is absent (`QWEN38_GGUF` overrides the path).
    #[test]
    #[cfg(feature = "deltanet_inference")]
    fn test_qwen35_real_gguf_nextn_and_config() {
        let path = std::path::PathBuf::from(
            std::env::var("QWEN38_GGUF")
                .unwrap_or_else(|_| "F:/models/qwen38-27b-dbirks-Q4_K_M.gguf".to_string()),
        );
        if !path.exists() {
            eprintln!("SKIP: {} not found", path.display());
            return;
        }
        let gguf = GgufFile::open(&path).expect("open qwen38 gguf");
        assert_eq!(gguf.architecture().unwrap_or("?"), "qwen35");

        let (config, layer_types) =
            qwen35_deltanet_config_from_gguf_metadata(&gguf).expect("config from metadata");

        // MTP exclusion: 64 main layers, never 65.
        assert_eq!(config.n_layer, 64, "blk.64 (MTP) excluded from main stack");
        assert_eq!(layer_types.len(), 64);
        assert_eq!(
            layer_types
                .iter()
                .filter(|&&lt| lt == DeltaNetLayerType::Attention)
                .count(),
            16
        );
        assert_eq!(layer_types[0], DeltaNetLayerType::DeltaNet);
        assert_eq!(layer_types[3], DeltaNetLayerType::Attention);
        assert_eq!(layer_types[63], DeltaNetLayerType::Attention);

        // Dense-model config fields (dbirks Qwen3.8-27B).
        assert_eq!(config.vocab_size, 248_320);
        assert_eq!(config.n_embd, 5_120);
        assert_eq!(config.n_head, 24);
        assert_eq!(config.n_kv_head, 4);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.mlp_hidden, 17_408);
        assert_eq!(config.rope_theta, 1e7);
        assert!(
            !config.tied_embeddings,
            "dbirks GGUF has output.weight (untied)"
        );

        // GDN dims: 16 K-heads x 48 V-heads x 128.
        assert_eq!(config.deltanet_linear_n_heads, 16);
        assert_eq!(config.deltanet_linear_n_value_heads, 48);
        assert_eq!(config.deltanet_linear_head_dim, 128);
        assert_eq!(config.rope_dimension_count, 64);

        // The MTP block stays in the file for the T3 drafter — and must NOT
        // collide with any main-stack tensor name the loader reads.
        assert!(gguf.tensor_info("blk.64.nextn.eh_proj.weight").is_some());
        assert!(gguf.tensor_info("blk.64.attn_q.weight").is_some());
        assert!(gguf.tensor_info("blk.63.attn_q.weight").is_some());
    }

    // ── Gemma 4 loader tests (Issue 577) ────────────────────────────────
    #[cfg(feature = "gemma4_inference")]
    #[test]
    fn test_gemma4_layer_names() {
        let ln = gemma4_gguf_layer_names(7);
        assert_eq!(ln.attn_norm, "blk.7.attn_norm.weight");
        assert_eq!(ln.attn_q, "blk.7.attn_q.weight");
        assert_eq!(ln.attn_k, "blk.7.attn_k.weight");
        assert_eq!(ln.attn_v, "blk.7.attn_v.weight");
        assert_eq!(ln.attn_output, "blk.7.attn_output.weight");
        assert_eq!(ln.attn_q_norm, "blk.7.attn_q_norm.weight");
        assert_eq!(ln.attn_k_norm, "blk.7.attn_k_norm.weight");
        assert_eq!(ln.attn_post_norm, "blk.7.post_attention_norm.weight");
        assert_eq!(ln.ffn_norm, "blk.7.ffn_norm.weight");
        assert_eq!(ln.ffn_gate, "blk.7.ffn_gate.weight");
        assert_eq!(ln.ffn_up, "blk.7.ffn_up.weight");
        assert_eq!(ln.ffn_down, "blk.7.ffn_down.weight");
        assert_eq!(ln.ffn_post_norm, "blk.7.post_ffw_norm.weight");
        assert_eq!(ln.layer_output_scale, "blk.7.layer_output_scale.weight");
    }

    /// Verify the Gemma 4 SWA pattern matches llama.cpp's
    /// `set_swa_pattern(n_pattern=6, dense_first=false)`: layers at idx
    /// `% 6 == 5` are Full, the rest Sliding.
    ///
    /// This tests the layer-type construction logic without needing a real
    /// GGUF file (which would be 7+ GB to download). The metadata-driven path
    /// in `gemma4_config_from_gguf_metadata` uses the same logic but reads
    /// `sliding_window_pattern` from GGUF metadata; the default (6) matches.
    #[cfg(feature = "gemma4_inference")]
    #[test]
    fn test_gemma4_swa_pattern_matches_llama_cpp() {
        use katgpt_core::types::Gemma4LayerType;

        // 48-layer model (the real 12B). 8 full-attention layers (every 6th).
        let n_layer = 48usize;
        let swa_pattern = 6usize;
        let layer_types: Vec<Gemma4LayerType> = (0..n_layer)
            .map(|i| {
                if swa_pattern > 0 && i % swa_pattern == swa_pattern.saturating_sub(1) {
                    Gemma4LayerType::Full
                } else {
                    Gemma4LayerType::Sliding
                }
            })
            .collect();

        let full_count = layer_types
            .iter()
            .filter(|t| **t == Gemma4LayerType::Full)
            .count();
        assert_eq!(full_count, 8, "8 full-attention layers expected");

        // Every 6th layer starting at idx 5.
        for &full_idx in &[5, 11, 17, 23, 29, 35, 41, 47] {
            assert_eq!(
                layer_types[full_idx],
                Gemma4LayerType::Full,
                "layer {full_idx} should be Full"
            );
        }
        // The rest are sliding.
        for &sliding_idx in &[0, 1, 2, 3, 4, 6, 12, 46] {
            assert_eq!(
                layer_types[sliding_idx],
                Gemma4LayerType::Sliding,
                "layer {sliding_idx} should be Sliding"
            );
        }
    }

    /// Regression test for Issue 395: verify the GGUF metadata key mapping for
    /// head dimensions. The counter-intuitive convention (confirmed against
    /// the real Gemma-4-12B GGUF + llama.cpp):
    ///   `key_length`     = Full/global head dim  (512 for 12B)
    ///   `key_length_swa` = Sliding head dim      (256 for 12B)
    ///
    /// The initial loader had these swapped, causing `q_dim_for(Sliding`) to
    /// compute 16×512=8192 (Full `q_dim`) and panicking the first Sliding layer's
    /// `W_Q` matmul (the 4096×3840 tensor was accessed as 8192 rows).
    #[cfg(feature = "gemma4_inference")]
    #[test]
    fn test_gemma4_key_length_metadata_mapping() {
        use crate::transformer::gemma4::{kv_dim_for, q_dim_for};
        use katgpt_core::types::Gemma4LayerType;

        // The canonical config encodes the correct values.
        let config = katgpt_core::types::Config::gemma4_12b();
        assert_eq!(config.head_dim, 256, "Sliding head_dim = key_length_swa");
        assert_eq!(
            config.global_head_dim, 512,
            "Full global_head_dim = key_length"
        );
        // The q_dim derivation that the bug broke:
        assert_eq!(
            q_dim_for(&config, Gemma4LayerType::Sliding),
            4096,
            "Sliding q_dim = n_head × head_dim = 16 × 256"
        );
        assert_eq!(
            q_dim_for(&config, Gemma4LayerType::Full),
            8192,
            "Full q_dim = n_head × global_head_dim = 16 × 512"
        );
        // KV dims too.
        assert_eq!(kv_dim_for(&config, Gemma4LayerType::Sliding), 2048);
        assert_eq!(kv_dim_for(&config, Gemma4LayerType::Full), 512);
    }

    /// Build a minimal in-memory GGUF v2 binary with one `Q4_K` tensor.
    ///
    /// Exercises the `dequant_f16_to_f32` `Q4_K` dispatch arm end-to-end
    /// through the real `GgufFile::open` mmap path — the highest-value test
    /// for the Issue 577 T6 unblock (`Q4_K` dequant helper).
    ///
    /// We quantize a known source vector, write it as a `Q4_K` tensor in a
    /// synthesized GGUF file, re-open it, dequantize, and compare against
    /// the in-memory `dequantize_row_q4_k` reference. This proves the
    /// new arm reads the right bytes and dispatches correctly.
    #[test]
    fn test_dequant_q4_k_via_gguf_round_trip() {
        use crate::quant::q4k::{BlockQ4K, QK_K, dequantize_row_q4_k, quantize_row_q4_k};
        use bytemuck::Zeroable;

        // Source: a single Q4_K super-block (256 elements) with a mix of values
        // that exercises the asymmetric quantization (positive + negative range).
        let src: Vec<f32> = (0..QK_K)
            .map(|i| {
                let x = i as f32 / 32.0;
                (x * std::f32::consts::PI).sin() * 2.5
            })
            .collect();

        // Quantize → Q4_K blocks.
        let mut blocks = [BlockQ4K::zeroed()];
        quantize_row_q4_k(&src, &mut blocks);

        // Reference: in-memory dequant.
        let mut ref_out = vec![0.0_f32; QK_K];
        dequantize_row_q4_k(&blocks, &mut ref_out);

        // Serialize the blocks to raw bytes (the GGUF tensor payload).
        let block_bytes: &[u8] = bytemuck::cast_slice(&blocks);

        // Build a minimal GGUF v2 file with one Q4_K tensor.
        let mut buf: Vec<u8> = Vec::new();
        // Header: magic + version + tensor_count + metadata_count
        buf.extend_from_slice(&0x46554747u32.to_le_bytes()); // "GGUF"
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata entries
        // Tensor info: name + n_dims + shape + type_id + offset
        let name = "test_q4k_tensor.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 dim
        buf.extend_from_slice(&(QK_K as u64).to_le_bytes()); // 256 elements
        buf.extend_from_slice(&12u32.to_le_bytes()); // Q4_K type id
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset 0
        // Align to GGUF_DEFAULT_ALIGNMENT (32) for tensor data start.
        let alignment: usize = 32;
        let cursor = buf.len();
        let padding = (alignment - (cursor % alignment)) % alignment;
        buf.extend(std::iter::repeat_n(0u8, padding));
        // Append the Q4_K block bytes.
        buf.extend_from_slice(block_bytes);

        // Write to a temp file and open via GgufFile.
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("riir_engine_q4k_test_{}.gguf", std::process::id()));
        std::fs::write(&path, &buf).expect("write temp GGUF");

        let result = (|| -> Result<()> {
            let gguf = GgufFile::open(&path)?;
            let got = gguf.dequant_f16_to_f32(name)?;
            assert_eq!(got.len(), QK_K, "dequant length mismatch");
            // Q4_K is a lossy quantizer (~4.5 bpw). Allow a per-element
            // tolerance consistent with the q4k.rs test suite (max abs err
            // < 0.2 for a ±2.5 amplitude signal; the existing tests use
            // similar bounds).
            let mut max_err = 0.0_f32;
            for (i, (&g, &r)) in got.iter().zip(ref_out.iter()).enumerate() {
                let err = (g - r).abs();
                if err > max_err {
                    max_err = err;
                }
                // The GGUF path and the in-memory reference both call
                // dequantize_row_q4_k on the same bytes, so they must be
                // bit-identical (both go through the same code path).
                assert!(
                    err < 1e-6,
                    "element {i}: GGUF dequant {g} != reference {r} (err {err})"
                );
            }
            // Sanity: the dequantized values should be close to the source.
            let mut max_src_err = 0.0_f32;
            for (&g, &s) in got.iter().zip(src.iter()) {
                let err = (g - s).abs();
                if err > max_src_err {
                    max_src_err = err;
                }
            }
            assert!(
                max_src_err < 0.3,
                "Q4_K quant error too large: {max_src_err:.4} (expected < 0.3 for ±2.5 signal)"
            );
            Ok(())
        })();

        // Cleanup the temp file regardless of test outcome.
        let _ = std::fs::remove_file(&path);

        result.expect("Q4_K GGUF round-trip failed");
    }

    /// Verify `block_info` + `tensor_bytes` for the `Q2_0` (ternary) format.
    ///
    /// The format is 34 bytes per 128 weights (2.125 bpw) — the same byte
    /// count as `Q8_0` but a 4× larger block size, so the two MUST NOT alias.
    #[test]
    fn test_q2_0_block_info_and_tensor_bytes() {
        assert_eq!(GgmlType::Q2_0.block_info(), Some((34, 128)));
        // 128 weights → 1 block × 34 bytes
        assert_eq!(GgmlType::Q2_0.tensor_bytes(128), 34);
        // 256 weights → 2 blocks × 34 bytes
        assert_eq!(GgmlType::Q2_0.tensor_bytes(256), 68);
        // 1280 weights → 10 blocks × 34 bytes
        assert_eq!(GgmlType::Q2_0.tensor_bytes(1280), 340);
        // Sanity: Q2_0 ≠ Q8_0 despite both being 34-byte blocks.
        assert_ne!(GgmlType::Q2_0.block_info(), GgmlType::Q8_0.block_info());
    }

    /// Build a minimal in-memory GGUF v2 binary with one `Q2_0` tensor + verify
    /// the full mmap → `dequant_f16_to_f32` round-trip.
    ///
    /// Mirrors `test_dequant_q4_k_via_gguf_round_trip` but for the ternary
    /// Ternary-Bonsai format (Plan 333 T2.1/T3.1b). Constructs a block with
    /// all four 2-bit codes (including the +2d fourth state), writes it as a
    /// GGUF tensor, re-opens via `GgufFile::open`, dequantizes, and checks
    /// the output matches the in-memory `dequantize_row_q2_0` reference.
    #[test]
    fn test_dequant_q2_0_via_gguf_round_trip() {
        use crate::quant::q2_0::{BlockQ2_0, Q2_0_BLOCK_SIZE, dequantize_row_q2_0};
        use bytemuck::Zeroable;

        // Build one Q2_0 block with a known pattern: the first 4 weights hit
        // all four codes (0, 1, 2, 3 → -d, 0, +d, +2d), the rest are code 2 (+d).
        let mut block = BlockQ2_0::zeroed();
        block.d = half::f16::from_f32(2.0).to_bits();
        block.qs[0] = 0b11_10_01_00; // codes 0,1,2,3 for weights 0..4
        for b in 1..(Q2_0_BLOCK_SIZE / 4) {
            block.qs[b] = 0b10_10_10_10; // code 2 (+d) for the rest
        }

        // Reference: in-memory dequant.
        let mut ref_out = vec![0.0_f32; Q2_0_BLOCK_SIZE];
        dequantize_row_q2_0(std::slice::from_ref(&block), &mut ref_out);

        // Sanity-check the reference against the hand-computed expected values.
        // d = 2.0; codes 0,1,2,3 → -2.0, 0.0, +2.0, +4.0.
        assert_eq!(ref_out[0], -2.0);
        assert_eq!(ref_out[1], 0.0);
        assert_eq!(ref_out[2], 2.0);
        assert_eq!(ref_out[3], 4.0); // the fourth state, faithfully preserved
        for (j, &w) in ref_out.iter().enumerate().take(Q2_0_BLOCK_SIZE).skip(4) {
            assert_eq!(w, 2.0, "tail weight {j} should be +d");
        }

        // Serialize the block to raw bytes.
        let block_bytes: &[u8] = bytemuck::cast_slice(std::slice::from_ref(&block));

        // Build a minimal GGUF v2 file with one Q2_0 tensor.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes()); // "GGUF"
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata entries
        let name = "test_q2_0_tensor.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // 1 dim
        buf.extend_from_slice(&(Q2_0_BLOCK_SIZE as u64).to_le_bytes()); // 128 elements
        buf.extend_from_slice(&42u32.to_le_bytes()); // Q2_0 type id (GGML_TYPE_Q2_0)
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset 0
        let alignment: usize = 32;
        let cursor = buf.len();
        let padding = (alignment - (cursor % alignment)) % alignment;
        buf.extend(std::iter::repeat_n(0u8, padding));
        buf.extend_from_slice(block_bytes);

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("riir_engine_q2_0_test_{}.gguf", std::process::id()));
        std::fs::write(&path, &buf).expect("write temp GGUF");

        let result = (|| -> Result<()> {
            let gguf = GgufFile::open(&path)?;
            let got = gguf.dequant_f16_to_f32(name)?;
            assert_eq!(got.len(), Q2_0_BLOCK_SIZE, "dequant length mismatch");
            // The GGUF path and the in-memory reference both call
            // dequantize_row_q2_0 on the same bytes, so they must be bit-identical.
            for (i, (&g, &r)) in got.iter().zip(ref_out.iter()).enumerate() {
                assert_eq!(g, r, "element {i}: GGUF dequant {g} != reference {r}");
            }
            // Also exercise `dequant_tensor_row` for a single-row Q2_0 tensor.
            let row = gguf.dequant_tensor_row(name, 0, Q2_0_BLOCK_SIZE)?;
            assert_eq!(row.len(), Q2_0_BLOCK_SIZE);
            for (i, (&g, &r)) in row.iter().zip(ref_out.iter()).enumerate() {
                assert_eq!(g, r, "dequant_tensor_row mismatch at {i}: {g} != {r}");
            }
            Ok(())
        })();

        let _ = std::fs::remove_file(&path);
        result.expect("Q2_0 GGUF round-trip failed");
    }
}
