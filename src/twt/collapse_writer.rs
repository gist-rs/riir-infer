//! The Phase-4 collapsed-GGUF writer (riir-infer Issue 022 T4.3) — the
//! model→model compiler that emits a REDUCED-layer ternary GGUF from a
//! parent checkpoint plus a block table:
//!
//! - per block, either a **member passthrough** (the winner layer's
//!   tensors byte-copied verbatim and renamed to the new block index —
//!   zero re-quant, bit-identical payload) or a **merged operator**
//!   (per-tensor payloads supplied by the caller, typically the
//!   `twt::ternarize` arms' output packed to Q2_0 wire, or merged-mean
//!   F32 norms per the pinned norm menu);
//! - the parent's metadata KV list mirrored **in file order** with the
//!   caller's overrides applied (the block count MUST be overridden to
//!   the reduced count — the writer refuses a stale one);
//! - the `twt.*` provenance keys appended (`twt.block_table`,
//!   `twt.parent_weights_blake3`, `twt.arm_codes` + legend, correction
//!   notes) — standard GGUF metadata, readable cross-repo by
//!   riir-train's `scripts/plan402_gguf_probe.py` with no edit.
//!
//! Fail-closed posture: the block table must tile `[0, n_layer)`
//! exactly; a merged plan must supply EVERY tensor suffix the block's
//! first member carries (a missing suffix would silently change the
//! model's shape); every payload must carry exactly its type's byte
//! length. Refusals name the defect.
//!
//! What this module deliberately does NOT do: pick winners (the
//! audition's), dequantize (the caller's, through
//! `twt::ternarize`), or verify decode quality (Phase 5's GOAT gate).

use std::collections::BTreeMap;
use std::io::Write;

use crate::gguf_loader::{GgufFile, GgmlType, GgufValue};
use crate::quant::q2_0::{BlockQ2_0, Q2_0_BLOCK_SIZE};
use crate::types::DeltaNetLayerType;

use super::ternarize::TwtArm;
use super::TwtError;

/// One emitted tensor's payload (NEW bytes — a copy is expressed as
/// [`LayerSource::Member`] and never materializes here).
#[derive(Clone, Debug)]
pub struct TensorOut {
    pub ggml_type: GgmlType,
    /// Dimensions in GGUF file order (`ne[0]` = innermost/cols FIRST —
    /// the reader's `GgufTensorInfo::shape` convention, round-tripped).
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

/// One collapsed block's tensor source.
#[derive(Clone, Debug)]
pub enum LayerSource {
    /// Keep member layer `li` (must lie in `[start, end)`): every parent
    /// `blk.{li}.*` tensor is byte-copied and renamed `blk.{b}.*`.
    Member(usize),
    /// A merged operator: supplied payloads keyed by the tensor SUFFIX
    /// after `blk.{b}.` (e.g. `attn_norm.weight`, `in_proj_qkv.weight`).
    /// The suffix set must EQUAL the parent's for the block — refuse
    /// loud on any difference (a missing suffix changes the model).
    Merged(BTreeMap<String, TensorOut>),
}

/// The collapse plan: the block table + metadata surgery.
#[derive(Clone, Debug)]
pub struct CollapseSpec {
    /// Ascending, contiguous `[start, end)` blocks tiling `[0, n_layer)`.
    pub blocks: Vec<(usize, usize, LayerSource)>,
    /// Replacements for existing parent metadata keys (or appends when
    /// absent). `{arch}.block_count` MUST be overridden to the reduced
    /// count — the writer refuses a plan that forgets it.
    pub metadata_overrides: Vec<(String, GgufValue)>,
    /// The `twt.*` provenance keys, appended after the parent's list.
    pub twt_meta: Vec<(String, GgufValue)>,
}

/// What one emit wrote (the artifact record).
#[derive(Clone, Copy, Debug)]
pub struct CollapsedStats {
    pub bytes_written: u64,
    pub n_tensors: usize,
    pub block_count: usize,
}

/// `twt.block_table` — the block table as a flat U32 array
/// `[start0, end0, start1, end1, …]` (half-open, ascending).
pub fn twt_block_table_value(blocks: &[(usize, usize)]) -> GgufValue {
    GgufValue::Array(
        blocks
            .iter()
            .flat_map(|&(s, e)| [GgufValue::U32(s as u32), GgufValue::U32(e as u32)])
            .collect(),
    )
}

/// `twt.arm_codes` — one [`TwtArm::code`] per emitted block (length =
/// block count; `0` = member passthrough).
pub fn twt_arm_codes_value(arms: &[TwtArm]) -> GgufValue {
    GgufValue::Array(arms.iter().map(|a| GgufValue::U8(a.code())).collect())
}

/// The legend written beside [`twt_layer_types_value`] (a code array
/// without a legend is a guess — same discipline as [`TwtArm::LEGEND`]).
pub const LAYER_TYPES_LEGEND: &str = "twt.layer_types: per collapsed block, \
DeltaNetLayerType discriminants — 0=attention, 1=deltanet";

/// `twt.layer_types` — one [`DeltaNetLayerType`] discriminant per emitted
/// block (length = block count; the enum's own `#[repr(u8)]` values — no
/// second vocabulary).
///
/// Load-bearing for `qwen35` collapses: the stock loader derives layer
/// types from `{arch}.full_attention_interval` INDEX arithmetic, and the
/// parent's interval pattern does not survive renumbering — a collapsed
/// file retyped by the derived pattern silently runs winners through the
/// WRONG forward (Issue 022 T5.0). The explicit array replaces the
/// derivation at load (`gguf_loader::qwen35_deltanet_config_from_gguf_metadata`);
/// this writer REFUSES to emit a `qwen35` collapse without it.
pub fn twt_layer_types_value(layer_types: &[DeltaNetLayerType]) -> GgufValue {
    GgufValue::Array(
        layer_types
            .iter()
            .map(|t| GgufValue::U8(*t as u8))
            .collect(),
    )
}

/// The parent's architecture tag (`general.architecture`) — the prefix
/// of every per-model metadata key.
fn parent_arch(parent: &GgufFile) -> Result<String, TwtError> {
    parent
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| TwtError::GgufWrite(
            "parent GGUF has no general.architecture — not a model file this writer can collapse"
                .to_owned(),
        ))
}

fn parent_block_count(parent: &GgufFile, arch: &str) -> Result<usize, TwtError> {
    let key = format!("{arch}.block_count");
    parent
        .metadata
        .get(&key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| TwtError::GgufWrite(format!("parent GGUF has no {key}")))
}

/// Validate the spec against the parent, build the output tensor list
/// (in emission order), and return `(tensors, block_count)`.
fn plan_tensors<'p, 'q>(
    parent: &'p GgufFile,
    spec: &'q CollapseSpec,
    n_layer: usize,
) -> Result<(Vec<OutTensor<'p, 'q>>, usize), TwtError> {
    // ── tiling ──
    if spec.blocks.is_empty() {
        return Err(TwtError::BadBlockTable { reason: "empty block table" });
    }
    let mut expect = 0usize;
    for (i, &(start, end, _)) in spec.blocks.iter().enumerate() {
        if start != expect {
            return Err(TwtError::BadBlockTable {
                reason: "blocks must be ascending + contiguous from 0",
            });
        }
        if end <= start {
            return Err(TwtError::BadBlockTable { reason: "empty block" });
        }
        if end > n_layer {
            return Err(TwtError::BadBlockTable {
                reason: "block exceeds the parent layer count",
            });
        }
        expect = end;
        match &spec.blocks[i].2 {
            LayerSource::Member(li) => {
                if !(*li >= start && *li < end) {
                    return Err(TwtError::BadBlockTable {
                        reason: "member passthrough outside its own block",
                    });
                }
            }
            LayerSource::Merged(_) => {}
        }
    }
    if expect != n_layer {
        return Err(TwtError::BadBlockTable {
            reason: "block table does not reach n_layer",
        });
    }
    let block_count = spec.blocks.len();

    // ── per-layer tensor sets ──
    let mut out: Vec<OutTensor<'p, 'q>> = Vec::new();

    // Globals first (file order): everything that is not `blk.{i}.…`.
    for info in &parent.tensor_infos {
        if !is_block_tensor(&info.name) {
            out.push(OutTensor::Copy(info));
        }
    }

    for (b, (start, end, src)) in spec.blocks.iter().enumerate() {
        match src {
            LayerSource::Member(li) => {
                let prefix = format!("blk.{li}.");
                for info in &parent.tensor_infos {
                    if let Some(suffix) = info.name.strip_prefix(&prefix) {
                        out.push(OutTensor::Renamed(info, format!("blk.{b}.{suffix}")));
                    }
                }
            }
            LayerSource::Merged(map) => {
                // The suffix set of the block's FIRST member is the
                // completeness oracle (a hybrid stack's GDN vs attention
                // layers differ; the merged op replaces the whole block,
                // so the plan must carry every suffix the block starts
                // with — the caller's block table came from the same
                // partition, so blk.{start} is the honest inventory).
                let prefix = format!("blk.{start}.");
                let mut suffixes: Vec<&str> = parent
                    .tensor_infos
                    .iter()
                    .filter_map(|info| info.name.strip_prefix(&prefix))
                    .collect();
                suffixes.sort_unstable();
                let mut supplied: Vec<&String> = map.keys().collect();
                supplied.sort_unstable();
                if suffixes.len() != supplied.len() {
                    let missing: Vec<String> = suffixes
                        .iter()
                        .filter(|s| !map.contains_key(**s))
                        .map(|s| s.to_string())
                        .collect();
                    return Err(TwtError::IncompletePlan {
                        start: *start,
                        end: *end,
                        missing: missing.join(", "),
                    });
                }
                for (sfx, sup) in suffixes.iter().zip(supplied.iter()) {
                    if sfx != sup {
                        return Err(TwtError::IncompletePlan {
                            start: *start,
                            end: *end,
                            missing: format!("suffix set differs (expected {sfx}, plan has {sup})"),
                        });
                    }
                    // Shape must match the parent's tensor for the suffix.
                    let parent_info = parent
                        .tensor_infos
                        .iter()
                        .find(|info| info.name == format!("blk.{start}.{sfx}"))
                        .expect("suffix came from the same walk");
                    let t = &map[*sup];
                    if t.shape != parent_info.shape {
                        return Err(TwtError::GgufWrite(format!(
                            "blk.{start}.{sfx}: payload shape {:?} != parent shape {:?}",
                            t.shape, parent_info.shape
                        )));
                    }
                    let n_elements: usize = t.shape.iter().product();
                    let want = t.ggml_type.tensor_bytes(n_elements);
                    if t.data.len() != want {
                        return Err(TwtError::GgufWrite(format!(
                            "blk.{start}.{sfx}: payload {} bytes, expected {want} for {:?}",
                            t.data.len(),
                            t.ggml_type
                        )));
                    }
                    out.push(OutTensor::New(
                        format!("blk.{b}.{sfx}"),
                        t.ggml_type,
                        t.shape.clone(),
                        &t.data,
                    ));
                }
            }
        }
    }
    Ok((out, block_count))
}

fn is_block_tensor(name: &str) -> bool {
    name.starts_with("blk.")
        && name[4..]
            .split_once('.')
            .map(|(idx, _)| idx.chars().all(|c| c.is_ascii_digit()) && !idx.is_empty())
            .unwrap_or(false)
}

/// One output tensor: a parent copy (possibly renamed), or new bytes.
/// `'p` borrows the parent (copies), `'q` borrows the plan (new payloads).
enum OutTensor<'p, 'q> {
    /// Byte-identical copy, name unchanged (globals).
    Copy(&'p crate::gguf_loader::GgufTensorInfo),
    /// Byte-identical copy under a new name (member passthrough blocks).
    Renamed(&'p crate::gguf_loader::GgufTensorInfo, String),
    /// Fresh payload (merged-operator tensors).
    New(String, GgmlType, Vec<usize>, &'q [u8]),
}

impl OutTensor<'_, '_> {
    fn name(&self) -> String {
        match self {
            Self::Copy(info) => info.name.clone(),
            Self::Renamed(_, name) => name.clone(),
            Self::New(name, _, _, _) => name.clone(),
        }
    }

    fn ggml_type(&self) -> GgmlType {
        match self {
            Self::Copy(info) | Self::Renamed(info, _) => info.ggml_type,
            Self::New(_, t, _, _) => *t,
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::Copy(info) | Self::Renamed(info, _) => &info.shape,
            Self::New(_, _, shape, _) => shape,
        }
    }

    fn data<'a>(&'a self, parent: &'a GgufFile) -> Result<&'a [u8], TwtError> {
        match self {
            Self::Copy(info) | Self::Renamed(info, _) => parent
                .tensor_slice(&info.name)
                .ok_or_else(|| TwtError::GgufWrite(format!("parent tensor vanished: {}", info.name))),
            Self::New(_, _, _, data) => Ok(data),
        }
    }
}

/// Emit the collapsed GGUF.
///
/// Streaming: header + metadata + tensor infos are written first (the
/// offsets are computed before any payload), then each tensor's bytes in
/// offset order — parent copies stream straight from the mmap, so the
/// peak memory is one tensor's payload, not the model.
pub fn emit_collapsed_gguf<W: Write>(
    parent: &GgufFile,
    spec: &CollapseSpec,
    out: &mut W,
) -> Result<CollapsedStats, TwtError> {
    let arch = parent_arch(parent)?;
    let n_layer = parent_block_count(parent, &arch)?;

    // ── the block-count override is load-bearing ──
    let bc_key = format!("{arch}.block_count");
    let overridden = spec
        .metadata_overrides
        .iter()
        .find(|(k, _)| *k == bc_key)
        .and_then(|(_, v)| v.as_u64());
    let (tensors, block_count) = plan_tensors(parent, spec, n_layer)?;
    match overridden {
        Some(v) if v as usize == block_count => {}
        Some(v) => {
            return Err(TwtError::GgufWrite(format!(
                "{bc_key} override {v} != collapsed block count {block_count}"
            )))
        }
        None => {
            return Err(TwtError::GgufWrite(format!(
                "the plan forgot to override {bc_key} to {block_count} — a collapsed file \
                 advertising the parent's layer count is worse than no file"
            )))
        }
    }

    // ── T5.0 protocol: explicit layer types are REQUIRED for qwen35 ──
    // The stock qwen35 loader types layers by full_attention_interval index
    // arithmetic, which the parent's pattern does not survive renumbering.
    // A collapsed qwen35 file without `twt.layer_types` would load every
    // winner through a derived (and silently WRONG) type — worse than no
    // file, exactly like the stale block_count above.
    let types_key = "twt.layer_types";
    let types_in_meta = match spec.twt_meta.iter().find(|(k, _)| k == types_key) {
        Some((_, GgufValue::Array(v))) => Some(v.len()),
        Some((_, other)) => {
            return Err(TwtError::GgufWrite(format!(
                "{types_key} must be a metadata array, got {other:?}"
            )))
        }
        None => None,
    };
    if arch == "qwen35" {
        let Some(n_types) = types_in_meta else {
            return Err(TwtError::GgufWrite(format!(
                "a collapsed qwen35 file must carry {types_key} (one DeltaNetLayerType \
                 discriminant per block) — the parent's full_attention_interval pattern \
                 does not survive renumbering; see twt_layer_types_value"
            )));
        };
        if n_types != block_count {
            return Err(TwtError::GgufWrite(format!(
                "{types_key} has {n_types} entries for {block_count} collapsed blocks — \
                 every block must be typed"
            )));
        }
        // A stale `qwen35.nextn_predict_layers` shifts the loader's main-stack
        // window and mistypes the LAST block — same class as the stale block
        // count. A collapsed main-stack file is a nextn-0 file: the MTP draft
        // blocks are not collapsible main layers, so the plan must override
        // the key to 0 (and drop the MTP blocks from the table).
        let nextn_key = "qwen35.nextn_predict_layers";
        let parent_nextn = parent.metadata.get(nextn_key).and_then(|v| v.as_u64());
        let plan_nextn = spec
            .metadata_overrides
            .iter()
            .find(|(k, _)| k == nextn_key)
            .and_then(|(_, v)| v.as_u64());
        let stale_nextn = match (parent_nextn, plan_nextn) {
            (Some(_), Some(o)) if o != 0 => Some(o),
            (Some(p), None) if p != 0 => Some(p),
            _ => None,
        };
        if let Some(n) = stale_nextn {
            return Err(TwtError::GgufWrite(format!(
                "{nextn_key} would survive the collapse as {n} — the loader would \
                 subtract it from {bc_key} and mistype the trailing block; override it \
                 to 0 (MTP blocks are not main-stack layers and must not tile the table)"
            )));
        }
    }

    // ── metadata: parent order, overrides applied in place, twt.* appended ──
    let mut kvs: Vec<(String, GgufValue)> = parent.metadata_order.clone();
    for (k, v) in &spec.metadata_overrides {
        match kvs.iter_mut().find(|(ek, _)| ek == k) {
            Some(slot) => slot.1 = v.clone(),
            None => kvs.push((k.clone(), v.clone())),
        }
    }
    kvs.extend(spec.twt_meta.iter().cloned());

    // ── offsets: every tensor aligned to the parent's alignment ──
    let alignment = parent.alignment.max(1);
    let mut infos: Vec<(String, u32, Vec<usize>, u64)> = Vec::with_capacity(tensors.len());
    let mut cursor = 0u64;
    for t in &tensors {
        let ttype = t.ggml_type();
        let shape = t.shape().to_vec();
        let n_elements: usize = shape.iter().product();
        let byte_len = ttype.tensor_bytes(n_elements) as u64;
        cursor = align_up(cursor, alignment);
        infos.push((t.name(), ttype.id(), shape, cursor));
        cursor += byte_len;
    }

    // ── serialize header + KVs + infos into one buffer, then stream ──
    let mut head: Vec<u8> = Vec::with_capacity(1 << 16);
    head.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
    head.extend_from_slice(&3u32.to_le_bytes()); // version 3
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in &kvs {
        write_gguf_string(&mut head, k)?;
        write_gguf_value(&mut head, v)?;
    }
    for (name, ttype, shape, off) in &infos {
        write_gguf_string(&mut head, name)?;
        head.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in shape {
            head.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        head.extend_from_slice(&ttype.to_le_bytes());
        head.extend_from_slice(&off.to_le_bytes());
    }
    // Pad to the alignment before the data section.
    while !head.len().is_multiple_of(alignment as usize) {
        head.push(0);
    }

    out.write_all(&head).map_err(io_err)?;
    let head_len = head.len() as u64;
    let mut written = head_len;

    // ── payloads, in offset order (the infos' order); section position
    //    tracks the data-section-relative cursor ──
    let mut section_pos = 0u64;
    let pad = [0u8; 32];
    for (t, (_, _, _, off)) in tensors.iter().zip(infos.iter()) {
        let need = ((alignment - (section_pos % alignment)) % alignment) as usize;
        if need > 0 {
            out.write_all(&pad[..need]).map_err(io_err)?;
            written += need as u64;
            section_pos += need as u64;
        }
        debug_assert_eq!(section_pos, *off, "offset plan desynced from the write loop");
        let data = t.data(parent)?;
        out.write_all(data).map_err(io_err)?;
        written += data.len() as u64;
        section_pos += data.len() as u64;
    }
    let _ = head_len;

    Ok(CollapsedStats {
        bytes_written: written,
        n_tensors: tensors.len(),
        block_count,
    })
}

fn align_up(v: u64, alignment: u64) -> u64 {
    v.div_ceil(alignment.max(1)) * alignment.max(1)
}

fn io_err(e: std::io::Error) -> TwtError {
    TwtError::GgufWrite(format!("io: {e}"))
}

fn write_gguf_string(buf: &mut Vec<u8>, s: &str) -> Result<(), TwtError> {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
    Ok(())
}

fn write_gguf_value(buf: &mut Vec<u8>, v: &GgufValue) -> Result<(), TwtError> {
    use crate::gguf_loader::meta_type_id;
    buf.extend_from_slice(&meta_type_id(v).to_le_bytes());
    match v {
        GgufValue::U8(x) => buf.push(*x),
        GgufValue::I8(x) => buf.push(*x as u8),
        GgufValue::U16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::U32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::U64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        GgufValue::Bool(x) => buf.push(*x as u8),
        GgufValue::String(s) => write_gguf_string(buf, s)?,
        GgufValue::Array(items) => {
            let Some(first) = items.first() else {
                return Err(TwtError::GgufWrite(
                    "empty metadata array — the GGUF wire needs an element type; use a sentinel value instead"
                        .to_owned(),
                ));
            };
            buf.extend_from_slice(&meta_type_id(first).to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                // Element payload WITHOUT its own type tag.
                write_gguf_value_payload(buf, item)?;
            }
        }
    }
    Ok(())
}

/// The payload half of a value (no leading type id) — array elements.
fn write_gguf_value_payload(buf: &mut Vec<u8>, v: &GgufValue) -> Result<(), TwtError> {
    match v {
        GgufValue::String(s) => write_gguf_string(buf, s),
        GgufValue::Array(_) => Err(TwtError::GgufWrite(
            "nested metadata arrays — not emitted by this writer (the reader's fixture set has none)"
                .to_owned(),
        )),
        other => {
            // Scalars: identical wire form with or without the tag stripped,
            // so serialize via the tagged path into a scratch and strip it.
            let mut scratch = Vec::with_capacity(16);
            write_gguf_value(&mut scratch, other)?;
            buf.extend_from_slice(&scratch[4..]);
            Ok(())
        }
    }
}

/// Convenience for the real lane: pack a materialized ternary operator to
/// Q2_0 wire bytes (the audition driver's merged-tensor payload).
pub fn q2_0_wire_bytes(w: &katgpt_core::TernaryGroupWeights) -> Result<Vec<u8>, TwtError> {
    let mut blocks: Vec<BlockQ2_0> = Vec::with_capacity(w.rows * (w.cols / Q2_0_BLOCK_SIZE));
    crate::quant::q2_0::pack_ternary_group_to_q2_0(w, &mut blocks)
        .map_err(|e| TwtError::GgufWrite(format!("q2_0 pack: {e}")))?;
    Ok(blocks
        .iter()
        .flat_map(|b| bytemuck::bytes_of(b).to_vec())
        .collect())
}
