//! Safetensors weight loader for Gemma 2 models (Plan 087).
//!
//! Supports sharded safetensors (`model.safetensors.index.json`) and
//! single-file (`model.safetensors`) layouts. Dequantizes BF16 → f32,
//! transposes weight matrices from [out, in] to [in, out], and applies
//! `RMSNorm` offset (+1.0) to norm weights (Gemma stores gamma−1).

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

use crate::gemma_layer::{
    GemmaLayerWeights, GemmaLayerWeightsF16, GemmaTransformerWeights, GemmaTransformerWeightsF16,
};
use crate::types::Config;

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

/// Dequantize BF16 (u16) to f32.
///
/// BF16 = upper 16 bits of IEEE 754 f32.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((u32::from(bits)) << 16)
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Parsed tensor metadata from safetensors header.
struct TensorMeta {
    #[allow(dead_code)]
    dtype: String,
    shape: Vec<usize>,
    data_start: usize,
    data_end: usize,
}

/// Weight name mapping for a single Gemma 2 layer.
struct LayerWeightNames {
    attn_wq: String,
    attn_wk: String,
    attn_wv: String,
    attn_wo: String,
    gate_proj: String,
    up_proj: String,
    down_proj: String,
    input_norm: String,
    post_attn_norm: String,
    pre_mlp_norm: String,
    post_mlp_norm: String,
}

fn layer_weight_names(i: usize) -> LayerWeightNames {
    LayerWeightNames {
        attn_wq: format!("model.layers.{i}.self_attn.q_proj.weight"),
        attn_wk: format!("model.layers.{i}.self_attn.k_proj.weight"),
        attn_wv: format!("model.layers.{i}.self_attn.v_proj.weight"),
        attn_wo: format!("model.layers.{i}.self_attn.o_proj.weight"),
        gate_proj: format!("model.layers.{i}.mlp.gate_proj.weight"),
        up_proj: format!("model.layers.{i}.mlp.up_proj.weight"),
        down_proj: format!("model.layers.{i}.mlp.down_proj.weight"),
        input_norm: format!("model.layers.{i}.input_layernorm.weight"),
        post_attn_norm: format!("model.layers.{i}.post_attention_layernorm.weight"),
        pre_mlp_norm: format!("model.layers.{i}.pre_feedforward_layernorm.weight"),
        post_mlp_norm: format!("model.layers.{i}.post_feedforward_layernorm.weight"),
    }
}

// ---------------------------------------------------------------------------
// Safetensors header parser
// ---------------------------------------------------------------------------

/// Parse safetensors file header, returning `(header_json_size, tensor_map)`.
///
/// Format:
/// - Bytes 0..8:  u64 LE → N = JSON header length
/// - Bytes 8..8+N: JSON mapping tensor name → {dtype, shape, `data_offsets`}
/// - Bytes 8+N..:  raw tensor data
fn parse_safetensors_header(data: &[u8]) -> Result<(usize, BTreeMap<String, TensorMeta>)> {
    if data.len() < 8 {
        bail!("safetensors file too small: {len} bytes", len = data.len());
    }

    let header_len = u64::from_le_bytes(
        data[0..8]
            .try_into()
            .context("failed to read header length")?,
    ) as usize;

    let header_end = 8 + header_len;
    if data.len() < header_end {
        bail!(
            "safetensors header truncated: need {need} bytes, have {have}",
            need = header_end,
            have = data.len()
        );
    }

    let header: serde_json::Value = serde_json::from_slice(&data[8..header_end])
        .context("failed to parse safetensors header JSON")?;

    let obj = header
        .as_object()
        .context("safetensors header is not a JSON object")?;

    let mut tensor_map = BTreeMap::new();
    for (name, value) in obj {
        // Skip the special metadata key
        if name == "__metadata__" {
            continue;
        }

        let dtype = value
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let shape: Vec<usize> = value
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();

        let offsets = value
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .with_context(|| format!("missing data_offsets for tensor '{name}'"))?;

        if offsets.len() != 2 {
            bail!(
                "expected 2 data_offsets for tensor '{name}', got {n}",
                n = offsets.len()
            );
        }

        let data_start = offsets[0]
            .as_u64()
            .with_context(|| format!("invalid data_offsets[0] for '{name}'"))?
            as usize;
        let data_end = offsets[1]
            .as_u64()
            .with_context(|| format!("invalid data_offsets[1] for '{name}'"))?
            as usize;

        tensor_map.insert(
            name.clone(),
            TensorMeta {
                dtype,
                shape,
                data_start,
                data_end,
            },
        );
    }

    Ok((header_len, tensor_map))
}

// ---------------------------------------------------------------------------
// Tensor extraction helpers
// ---------------------------------------------------------------------------

/// Add scalar offset to all elements (for `RMSNorm` gamma−1 → gamma).
fn add_offset(mut data: Vec<f32>, offset: f32) -> Vec<f32> {
    for v in &mut data {
        *v += offset;
    }
    data
}

/// Extract and dequantize a BF16 tensor from mmap data.
/// Keeps 2D weights in safetensors layout `[out_features, in_features]`
/// which matches our `matmul(out, weight, input, rows=out, cols=in)` convention.
fn extract_tensor(
    mmap: &[u8],
    header_len: usize,
    meta: &TensorMeta,
    name: &str,
) -> Result<Vec<f32>> {
    let data_offset = 8 + header_len;
    let raw_start = data_offset + meta.data_start;
    let raw_end = data_offset + meta.data_end;
    let raw_data = &mmap[raw_start..raw_end];

    let elements: usize = meta.shape.iter().product();
    let expected_bytes = elements * 2; // BF16 = 2 bytes per element
    if raw_data.len() != expected_bytes {
        bail!(
            "BF16 tensor '{name}' size mismatch: expected {expected} bytes, got {got}",
            expected = expected_bytes,
            got = raw_data.len()
        );
    }

    // Dequantize BF16 → f32.
    //
    // On little-endian hosts, BF16 data is laid out as a contiguous `&[u16]`
    // (after a byte-alignment safe cast via `bytemuck::try_cast_slice`). We
    // still iterate element-by-element for the `bf16_to_f32` shift, but the
    // cast avoids the per-pair `u16::from_le_bytes` + bounds checks of the
    // prior `chunks_exact(2)` iterator path. On unaligned input or big-endian
    // we fall back to the original chunked loop.
    #[cfg(target_endian = "little")]
    let f32_data: Vec<f32> = if let Ok(bits) = bytemuck::try_cast_slice::<_, u16>(raw_data) {
        bits.iter().map(|&b| bf16_to_f32(b)).collect()
    } else {
        raw_data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| bf16_to_f32(u16::from_le_bytes(*chunk)))
            .collect()
    };
    #[cfg(not(target_endian = "little"))]
    let f32_data: Vec<f32> = raw_data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| bf16_to_f32(u16::from_le_bytes(*chunk)))
        .collect();

    // Keep layout as-is — safetensors [out, in] matches matmul(rows=out, cols=in)
    match meta.shape.len() {
        1 => Ok(f32_data),
        2 => Ok(f32_data),
        _ => bail!(
            "unsupported tensor rank {rank} for '{name}'",
            rank = meta.shape.len()
        ),
    }
}

/// Remove a tensor from the map, failing if missing.
fn require_tensor(tensors: &mut BTreeMap<String, Vec<f32>>, name: &str) -> Result<Vec<f32>> {
    tensors
        .remove(name)
        .with_context(|| format!("missing required tensor '{name}'"))
}

// ---------------------------------------------------------------------------
// Shard loading
// ---------------------------------------------------------------------------

/// Load requested tensors from a single safetensors shard file.
fn load_tensors_from_shard(
    shard_path: &Path,
    needed_names: &[String],
) -> Result<BTreeMap<String, Vec<f32>>> {
    let file = std::fs::File::open(shard_path)
        .with_context(|| format!("failed to open shard {path}", path = shard_path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap shard {path}", path = shard_path.display()))?;

    let (header_len, tensor_map) = parse_safetensors_header(&mmap)?;

    let mut result = BTreeMap::new();
    for name in needed_names {
        let meta = tensor_map.get(name).with_context(|| {
            format!(
                "tensor '{name}' not found in shard {path}",
                path = shard_path.display()
            )
        })?;
        let data = extract_tensor(&mmap, header_len, meta, name)?;
        result.insert(name.clone(), data);
    }
    Ok(result)
}

/// Parse the shard index JSON to find which shard each needed tensor is in.
fn parse_shard_index(index_path: &Path, needed: &[String]) -> Result<BTreeMap<String, String>> {
    let index_json = std::fs::read_to_string(index_path)
        .with_context(|| format!("failed to read {path}", path = index_path.display()))?;
    let index: serde_json::Value =
        serde_json::from_str(&index_json).context("failed to parse safetensors index JSON")?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .context("missing weight_map in safetensors index")?;

    let needed_set: HashSet<&String> = needed.iter().collect();
    let mut result = BTreeMap::new();

    for (tensor_name, shard_val) in weight_map {
        if !needed_set.contains(tensor_name) {
            continue;
        }
        let shard_file = shard_val
            .as_str()
            .with_context(|| format!("invalid shard path for tensor '{tensor_name}'"))?;
        result.insert(tensor_name.clone(), shard_file.to_string());
    }
    Ok(result)
}

/// Collect all tensor names needed for a Gemma 2 model.
fn collect_needed_tensor_names(config: &Config) -> Vec<String> {
    let mut names = Vec::with_capacity(2 + config.n_layer * 11);

    names.push("model.embed_tokens.weight".to_string());
    names.push("model.norm.weight".to_string());

    for i in 0..config.n_layer {
        let ln = layer_weight_names(i);
        names.push(ln.attn_wq);
        names.push(ln.attn_wk);
        names.push(ln.attn_wv);
        names.push(ln.attn_wo);
        names.push(ln.gate_proj);
        names.push(ln.up_proj);
        names.push(ln.down_proj);
        names.push(ln.input_norm);
        names.push(ln.post_attn_norm);
        names.push(ln.pre_mlp_norm);
        names.push(ln.post_mlp_norm);
    }
    names.shrink_to_fit();
    names
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Load Gemma 2 weights from a directory containing safetensors files.
///
/// Supports both sharded (`model.safetensors.index.json`) and single-file
/// (`model.safetensors`) layouts. Dequantizes BF16 → f32, transposes weight
/// matrices, and applies `RMSNorm` offset (+1.0) to norm weights.
pub fn load_gemma2_weights(model_dir: &Path, config: &Config) -> Result<GemmaTransformerWeights> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let single_path = model_dir.join("model.safetensors");

    let needed = collect_needed_tensor_names(config);

    // Determine shard mapping: tensor_name → shard_filename
    let shard_map = if index_path.exists() {
        parse_shard_index(&index_path, &needed)?
    } else if single_path.exists() {
        // Single file: all tensors in one shard
        needed
            .iter()
            .map(|name| (name.clone(), "model.safetensors".to_string()))
            .collect()
    } else {
        bail!(
            "no safetensors found in {dir}: expected {index} or {single}",
            dir = model_dir.display(),
            index = index_path.display(),
            single = single_path.display()
        );
    };

    // Group by shard file for efficient loading (one mmap per shard)
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (tensor_name, shard_file) in &shard_map {
        grouped
            .entry(shard_file.clone())
            .or_default()
            .push(tensor_name.clone());
    }

    // Load all tensors from shards
    let mut tensors: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for (shard_file, tensor_names) in &grouped {
        let shard_path = model_dir.join(shard_file);
        let shard_tensors = load_tensors_from_shard(&shard_path, tensor_names)?;
        tensors.extend(shard_tensors);
    }

    // Assemble the weight struct
    let wte = require_tensor(&mut tensors, "model.embed_tokens.weight")?;
    let final_norm = add_offset(require_tensor(&mut tensors, "model.norm.weight")?, 1.0);

    let mut layers = Vec::with_capacity(config.n_layer);
    for i in 0..config.n_layer {
        let n = layer_weight_names(i);
        let layer = GemmaLayerWeights {
            attn_wq: require_tensor(&mut tensors, &n.attn_wq)?,
            attn_wk: require_tensor(&mut tensors, &n.attn_wk)?,
            attn_wv: require_tensor(&mut tensors, &n.attn_wv)?,
            attn_wo: require_tensor(&mut tensors, &n.attn_wo)?,
            gate_proj: require_tensor(&mut tensors, &n.gate_proj)?,
            up_proj: require_tensor(&mut tensors, &n.up_proj)?,
            down_proj: require_tensor(&mut tensors, &n.down_proj)?,
            input_norm: add_offset(require_tensor(&mut tensors, &n.input_norm)?, 1.0),
            post_attn_norm: add_offset(require_tensor(&mut tensors, &n.post_attn_norm)?, 1.0),
            pre_mlp_norm: add_offset(require_tensor(&mut tensors, &n.pre_mlp_norm)?, 1.0),
            post_mlp_norm: add_offset(require_tensor(&mut tensors, &n.post_mlp_norm)?, 1.0),
        };
        layers.push(layer);
    }

    Ok(GemmaTransformerWeights {
        wte,
        final_norm,
        layers,
        #[cfg(feature = "delta_routing")]
        delta_routing_query: (0..config.n_layer)
            .map(|_| vec![0.0; config.n_embd])
            .collect(),
        #[cfg(feature = "delta_routing")]
        delta_routing_norm: (0..config.n_layer)
            .map(|_| vec![1.0; config.n_embd])
            .collect(),
    })
}

// ---------------------------------------------------------------------------
// f16 loader (Plan 095)
// ---------------------------------------------------------------------------

/// Convert f32 vector to f16 vector.
fn f32_to_f16(data: Vec<f32>) -> Vec<half::f16> {
    data.into_iter().map(half::f16::from_f32).collect()
}

/// Load Gemma 2 weights as f16 from a directory containing safetensors files.
///
/// Same as [`load_gemma2_weights`] but stores projections as `half::f16`.
/// `RMSNorm` gammas remain f32 (negligible size: 4 × 2304 per layer).
/// Halves memory bandwidth during inference: ~5.2 GB/token vs ~10.4 GB/token.
pub fn load_gemma2_weights_f16(
    model_dir: &Path,
    config: &Config,
) -> Result<GemmaTransformerWeightsF16> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let single_path = model_dir.join("model.safetensors");

    let needed = collect_needed_tensor_names(config);

    // Determine shard mapping: tensor_name → shard_filename
    let shard_map = if index_path.exists() {
        parse_shard_index(&index_path, &needed)?
    } else if single_path.exists() {
        needed
            .iter()
            .map(|name| (name.clone(), "model.safetensors".to_string()))
            .collect()
    } else {
        bail!(
            "no safetensors found in {dir}: expected {index} or {single}",
            dir = model_dir.display(),
            index = index_path.display(),
            single = single_path.display()
        );
    };

    // Group by shard file for efficient loading (one mmap per shard)
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (tensor_name, shard_file) in &shard_map {
        grouped
            .entry(shard_file.clone())
            .or_default()
            .push(tensor_name.clone());
    }

    // Load all tensors from shards (f32 initially, then convert)
    let mut tensors: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for (shard_file, tensor_names) in &grouped {
        let shard_path = model_dir.join(shard_file);
        let shard_tensors = load_tensors_from_shard(&shard_path, tensor_names)?;
        tensors.extend(shard_tensors);
    }

    // Assemble the f16 weight struct
    let wte = f32_to_f16(require_tensor(&mut tensors, "model.embed_tokens.weight")?);
    let final_norm = add_offset(require_tensor(&mut tensors, "model.norm.weight")?, 1.0);

    let mut layers = Vec::with_capacity(config.n_layer);
    for i in 0..config.n_layer {
        let n = layer_weight_names(i);
        let layer = GemmaLayerWeightsF16 {
            attn_wq: f32_to_f16(require_tensor(&mut tensors, &n.attn_wq)?),
            attn_wk: f32_to_f16(require_tensor(&mut tensors, &n.attn_wk)?),
            attn_wv: f32_to_f16(require_tensor(&mut tensors, &n.attn_wv)?),
            attn_wo: f32_to_f16(require_tensor(&mut tensors, &n.attn_wo)?),
            gate_proj: f32_to_f16(require_tensor(&mut tensors, &n.gate_proj)?),
            up_proj: f32_to_f16(require_tensor(&mut tensors, &n.up_proj)?),
            down_proj: f32_to_f16(require_tensor(&mut tensors, &n.down_proj)?),
            // RMSNorm gammas stay f32
            input_norm: add_offset(require_tensor(&mut tensors, &n.input_norm)?, 1.0),
            post_attn_norm: add_offset(require_tensor(&mut tensors, &n.post_attn_norm)?, 1.0),
            pre_mlp_norm: add_offset(require_tensor(&mut tensors, &n.pre_mlp_norm)?, 1.0),
            post_mlp_norm: add_offset(require_tensor(&mut tensors, &n.post_mlp_norm)?, 1.0),
        };
        layers.push(layer);
    }

    Ok(GemmaTransformerWeightsF16 {
        wte,
        final_norm,
        layers,
        #[cfg(feature = "delta_routing")]
        delta_routing_query: (0..config.n_layer)
            .map(|_| vec![0.0; config.n_embd])
            .collect(),
        #[cfg(feature = "delta_routing")]
        delta_routing_norm: (0..config.n_layer)
            .map(|_| vec![1.0; config.n_embd])
            .collect(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_to_f32() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0f32);
        assert_eq!(bf16_to_f32(0x4000), 2.0f32);
        assert_eq!(bf16_to_f32(0x0000), 0.0f32);
        assert_eq!(bf16_to_f32(0xBF80), -1.0f32);
    }

    #[test]
    fn test_add_offset() {
        let data = vec![0.0f32, 1.0, 2.0];
        let result = add_offset(data, 1.0);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_safetensors_header_too_small() {
        let data = [0u8; 4];
        let result = parse_safetensors_header(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_weight_names() {
        let names = layer_weight_names(0);
        assert_eq!(names.attn_wq, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(names.attn_wk, "model.layers.0.self_attn.k_proj.weight");
        assert_eq!(names.attn_wv, "model.layers.0.self_attn.v_proj.weight");
        assert_eq!(names.attn_wo, "model.layers.0.self_attn.o_proj.weight");
        assert_eq!(names.gate_proj, "model.layers.0.mlp.gate_proj.weight");
        assert_eq!(names.up_proj, "model.layers.0.mlp.up_proj.weight");
        assert_eq!(names.down_proj, "model.layers.0.mlp.down_proj.weight");
        assert_eq!(names.input_norm, "model.layers.0.input_layernorm.weight");
        assert_eq!(
            names.post_attn_norm,
            "model.layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(
            names.pre_mlp_norm,
            "model.layers.0.pre_feedforward_layernorm.weight"
        );
        assert_eq!(
            names.post_mlp_norm,
            "model.layers.0.post_feedforward_layernorm.weight"
        );

        let names5 = layer_weight_names(5);
        assert_eq!(names5.attn_wq, "model.layers.5.self_attn.q_proj.weight");
    }

    #[test]
    fn test_collect_needed_tensor_names_count() {
        let config = Config::gemma2_2b();
        let names = collect_needed_tensor_names(&config);

        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(names.contains(&"model.norm.weight".to_string()));
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.25.post_feedforward_layernorm.weight".to_string()));

        // 2 global + 11 per layer × 26 layers = 288
        assert_eq!(names.len(), 2 + 11 * config.n_layer);
    }
}
