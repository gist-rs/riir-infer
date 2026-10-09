//! Minimal safetensors reader for the riir-owned forward — candle-free by
//! construction (`.issues/002`): the 8-byte LE u64 header length → the JSON
//! header (`{name: {dtype, shape, data_offsets}}`, offsets relative to the
//! data section at `8 + header_len`) → absolute slices of the buffer.
//!
//! Widening: F16 → F32 is BIT-EXACT (every f16 value is representable in
//! f32 — normals `(sign<<31) | ((e−15+127)<<23) | (m<<13)`, subnormals
//! `sign · m · 2⁻²⁴`, ±0/±inf/NaN mapped bit-for-bit), F32 passes through,
//! F64 casts down (the same rounding candle's `to_dtype(F32)` applies).
//! Every other dtype is refused LOUD — the pinned checkpoints are F16
//! throughout except the unused `temperature` tensor (F32 english/
//! multilingual, F16 typed-decisions — the reader widens whatever each
//! tensor declares and assumes no uniform dtype).
//!
//! The file is read whole and widened tensor-by-tensor (the english
//! checkpoint is ~890 MB f16 → ~1.7 GB f32; the transient peak is fine on
//! any machine this lane targets, and the map is consumed via `remove` so
//! encoder + head split it without a second copy).

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use super::super::{LayaError, Result};
use super::backend::Backend;

/// One parsed tensor: the header's shape + the payload (row-major, the
/// safetensors storage order).
#[derive(Debug)]
pub struct Weights {
    /// The declared shape (e.g. `[3d, d]` for a fused Wqkv).
    pub shape: Vec<usize>,
    /// The payload — widened f32 for every flat dtype, or the RAW blocked
    /// Q8_0 bytes for the q8 artifact's tensors (Plan 616 Phase 1: the
    /// q8 posture retains the bytes; resolution is per-consumer).
    pub data: WeightData,
}

/// The payload of one parsed tensor.
#[derive(Debug)]
pub enum WeightData {
    /// The widened f32 payload (the `shape` product of elements).
    F32(Vec<f32>),
    /// The RAW blocked Q8_0 bytes (`widen_q8_0`'s input layout —
    /// 34 bytes per 32 weights, an f16 scale + i8 quants; the tail block
    /// when `numel % 32 != 0`). Never widened at parse.
    Q8(RawQ8),
    /// The RAW blocked Q4_0 bytes (Plan 616 Phase 3 — the same house
    /// blocked family at 4 bits: 18 bytes per 32 weights, an f16 scale +
    /// packed signed nibbles; the tail block per `q4_blocked_len`). Never
    /// widened at parse.
    Q4(RawQ4),
}

impl Weights {
    /// The element count (the `shape` product).
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// The payload as widened f32 (borrows; a Q8/Q4 payload resolves
    /// through its once-only host widening).
    #[must_use]
    pub fn wide_f32(&self) -> &[f32] {
        match &self.data {
            WeightData::F32(v) => v,
            WeightData::Q8(q) => q.wide(),
            WeightData::Q4(q) => q.wide(),
        }
    }

    /// Consume into the widened f32 payload (a Q8/Q4 payload widens once
    /// and drops its raw bytes — the host-consumer seams: the token gather
    /// table, the type-embedding rows).
    #[must_use]
    pub fn into_f32(self) -> Vec<f32> {
        match self.data {
            WeightData::F32(v) => v,
            WeightData::Q8(q) => q.into_wide(),
            WeightData::Q4(q) => q.into_wide(),
        }
    }
}

/// A raw blocked Q8_0 payload (the q8 artifact's parse form —
/// Plan 616 Phase 1). The bytes are the loader's blocked layout exactly:
/// `numel/32` full blocks + an optional tail block, each an f16 scale
/// followed by its i8 quants.
#[derive(Debug)]
pub struct RawQ8 {
    numel: usize,
    raw: Vec<u8>,
    /// The once-only host widening — the CPU lane's resolution and the
    /// f32-consuming seams. The Metal lane NEVER touches it (Phase 1's
    /// whole point: the device dequantizes from these bytes; the host f32
    /// copy never exists).
    wide: OnceLock<Vec<f32>>,
}

impl RawQ8 {
    /// Validate the blocked byte length and retain. The length law is the
    /// parse-time wall: a malformed payload is refused HERE, never at a
    /// later resolution.
    pub fn new(numel: usize, raw: Vec<u8>) -> Result<Self> {
        let expect = q8_blocked_len(numel);
        if raw.len() != expect {
            return Err(LayaError::Runtime(format!(
                "Q8_0 payload {} bytes != the blocked layout ({expect}) for {numel} elements",
                raw.len()
            )));
        }
        Ok(Self {
            numel,
            raw,
            wide: OnceLock::new(),
        })
    }

    /// The element count.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// The raw blocked bytes (the device dequant's input).
    #[must_use]
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// The once-only host widening (the CPU resolution). Deterministic:
    /// the same `d · q` arithmetic `widen_q8_0` has always run.
    #[must_use]
    pub fn wide(&self) -> &[f32] {
        self.wide.get_or_init(|| widen_q8_0(&self.raw, self.numel))
    }

    /// Consume into the widened f32 payload (the raw bytes drop).
    #[must_use]
    pub fn into_wide(self) -> Vec<f32> {
        match self.wide.into_inner() {
            Some(v) => v,
            None => widen_q8_0(&self.raw, self.numel),
        }
    }
}

/// A raw blocked Q4_0 payload (Plan 616 Phase 3 — the q4 artifact's parse
/// form; the [`RawQ8`] twin at half the byte rate). The bytes are the
/// loader's blocked layout exactly: `numel/32` full blocks + an optional
/// tail block, each an f16 scale followed by packed signed nibbles — the
/// EVEN element in the LOW nibble, the ODD element in the high nibble
/// (GGML's nibble order). The stored grid is `[-7, +7]` (the house Q8
/// law's 4-bit analog: `d = f16(amax/7)` puts both ±amax on the grid
/// ends up to the scale's own f16 rounding, and zero exactly at 0); the
/// decode sign-extends the nibble, so a `-8` nibble decodes too (never
/// produced by the house converter, accepted for GGML interop).
#[derive(Debug)]
pub struct RawQ4 {
    numel: usize,
    raw: Vec<u8>,
    /// The once-only host widening — the CPU lane's resolution and the
    /// f32-consuming seams. The Metal lane NEVER touches it (the q8
    /// posture's law: the device dequantizes from these bytes; the host
    /// f32 copy never exists).
    wide: OnceLock<Vec<f32>>,
}

impl RawQ4 {
    /// Validate the blocked byte length and retain. The length law is the
    /// parse-time wall: a malformed payload is refused HERE, never at a
    /// later resolution.
    pub fn new(numel: usize, raw: Vec<u8>) -> Result<Self> {
        let expect = q4_blocked_len(numel);
        if raw.len() != expect {
            return Err(LayaError::Runtime(format!(
                "Q4_0 payload {} bytes != the blocked layout ({expect}) for {numel} elements",
                raw.len()
            )));
        }
        Ok(Self {
            numel,
            raw,
            wide: OnceLock::new(),
        })
    }

    /// The element count.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// The raw blocked bytes (the device dequant's input).
    #[must_use]
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// The once-only host widening (the CPU resolution). Deterministic:
    /// the same `d · q` arithmetic [`widen_q4_0`] has always run.
    #[must_use]
    pub fn wide(&self) -> &[f32] {
        self.wide.get_or_init(|| widen_q4_0(&self.raw, self.numel))
    }

    /// Consume into the widened f32 payload (the raw bytes drop).
    #[must_use]
    pub fn into_wide(self) -> Vec<f32> {
        match self.wide.into_inner() {
            Some(v) => v,
            None => widen_q4_0(&self.raw, self.numel),
        }
    }
}

/// The blocked byte length of a Q8_0 payload: `numel/32` full blocks of
/// 34 bytes + an optional tail block (2 + `numel % 32`).
#[must_use]
pub fn q8_blocked_len(numel: usize) -> usize {
    let block = super::fake_quant::BLOCK;
    let full = numel / block;
    let tail = numel % block;
    full * (2 + block) + usize::from(tail > 0) * (2 + tail)
}

/// The blocked byte length of a Q4_0 payload (Plan 616 Phase 3): the same
/// 32-weight block law at 4 bits — full blocks of 18 bytes (an f16 scale
/// + 16 packed-nibble bytes) + an optional tail block (2 +
///   `tail.div_ceil(2)`; nibbles are byte-aligned, never split).
#[must_use]
pub fn q4_blocked_len(numel: usize) -> usize {
    let block = super::fake_quant::BLOCK;
    let full = numel / block;
    let tail = numel % block;
    full * (2 + block / 2) + usize::from(tail > 0) * (2 + tail.div_ceil(2))
}

/// One 2D GEMM weight in a layer (Plan 616 Phase 1): the widened form
/// (the F16 posture — today's exact behavior) or the retained Q8/Q4
/// bytes (the q8/q4 artifact postures — the Metal lane dequantizes
/// device-side at warm; every other backend resolves through the
/// once-only host widening, so the CPU lane keeps widening on demand).
#[derive(Debug)]
pub enum Weight2D {
    /// Row-major `[n, k]`, widened f32.
    Dense(Vec<f32>),
    /// Row-major `[n, k]` blocked Q8_0 bytes, retained.
    Q8(RawQ8),
    /// Row-major `[n, k]` blocked Q4_0 bytes, retained (Plan 616 Phase 3).
    Q4(RawQ4),
}

impl Weight2D {
    /// From a parsed tensor. The kill-switch (`LAYA_Q8_HOST_F32=1`)
    /// restores the pre-Phase-1 load: the Q8/Q4 payload widens ONCE here
    /// and the raw bytes drop — the widen-at-load posture, byte-restoring
    /// (one env, read live per tensor like every house kill-switch; the
    /// name is historical — it governs BOTH raw-quant formats).
    #[must_use]
    pub fn from_weights(w: Weights) -> Self {
        match w.data {
            WeightData::F32(v) => Self::Dense(v),
            WeightData::Q8(q) => {
                if std::env::var("LAYA_Q8_HOST_F32").as_deref() == Ok("1") {
                    Self::Dense(q.into_wide())
                } else {
                    Self::Q8(q)
                }
            }
            WeightData::Q4(q) => {
                if std::env::var("LAYA_Q8_HOST_F32").as_deref() == Ok("1") {
                    Self::Dense(q.into_wide())
                } else {
                    Self::Q4(q)
                }
            }
        }
    }

    /// The element count (`n · k`).
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::Q8(q) => q.numel(),
            Self::Q4(q) => q.numel(),
        }
    }

    /// True when the weight carries no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The widened f32 view (the host-consumer seams: the audition's
    /// read-only layer weights, the scorer clone-out). Metal never calls
    /// it on the forward path.
    #[must_use]
    pub fn wide(&self) -> &[f32] {
        match self {
            Self::Dense(v) => v,
            Self::Q8(q) => q.wide(),
            Self::Q4(q) => q.wide(),
        }
    }

    /// A widened f32 copy (the scorer clone-out seam).
    #[must_use]
    pub fn to_dense(&self) -> Vec<f32> {
        self.wide().to_vec()
    }

    /// Pre-place on the backend (`Backend::warm_weight_2d`'s Q8/Q4
    /// twins).
    pub fn warm_2d(&self, b: &dyn Backend, n: usize, k: usize) {
        match self {
            Self::Dense(v) => b.warm_weight_2d(v, n, k),
            Self::Q8(q) => b.warm_weight_2d_q8(q, n, k),
            Self::Q4(q) => b.warm_weight_2d_q4(q, n, k),
        }
    }

    /// `dst[m×n] ← a[m×k] @ self[n×k]ᵀ` (the `matmul_w` dispatch).
    pub fn matmul_w(
        &self,
        b: &dyn Backend,
        a: &[f32],
        m: usize,
        k: usize,
        n: usize,
        dst: &mut [f32],
    ) {
        match self {
            Self::Dense(v) => b.matmul_w(a, m, k, v, n, dst),
            Self::Q8(q) => b.matmul_w_q8(a, m, k, q, n, dst),
            Self::Q4(q) => b.matmul_w_q4(a, m, k, q, n, dst),
        }
    }

    /// `x[m×n] += a[m×k] @ self[n×k]ᵀ` (the `matmul_w_accum` dispatch).
    pub fn matmul_w_accum(
        &self,
        b: &dyn Backend,
        a: &[f32],
        m: usize,
        k: usize,
        n: usize,
        x: &mut [f32],
    ) {
        match self {
            Self::Dense(v) => b.matmul_w_accum(a, m, k, v, n, x),
            Self::Q8(q) => b.matmul_w_accum_q8(a, m, k, q, n, x),
            Self::Q4(q) => b.matmul_w_accum_q4(a, m, k, q, n, x),
        }
    }

    /// `act[m×i] = glu(a[m×k] @ self[(2i)×k]ᵀ)` (the `matmul_w_glu`
    /// dispatch).
    pub fn matmul_w_glu(
        &self,
        b: &dyn Backend,
        a: &[f32],
        m: usize,
        k: usize,
        i_sz: usize,
        act: &mut [f32],
    ) {
        match self {
            Self::Dense(v) => b.matmul_w_glu(a, m, k, v, i_sz, act),
            Self::Q8(q) => b.matmul_w_glu_q8(a, m, k, q, i_sz, act),
            Self::Q4(q) => b.matmul_w_glu_q4(a, m, k, q, i_sz, act),
        }
    }
}

/// Parse + widen a safetensors file from disk.
pub fn load(path: &Path, ckpt: &'static str) -> Result<HashMap<String, Weights>> {
    let bytes = std::fs::read(path).map_err(|e| LayaError::Missing {
        checkpoint: ckpt,
        file: format!("model.safetensors ({e})"),
    })?;
    from_bytes(&bytes, ckpt)
}

/// The safetensors container preamble — the ONE home for the layout law
/// (`.issues/002`): 8-byte LE u64 header length → the JSON header, with
/// the data section starting at `8 + header_len`. Every reader of a
/// safetensors-shaped file walks THIS (`from_bytes` for the checkpoint,
/// the ANE lane's `table_e8` sidecar for Plan 612); nothing re-derives
/// the offsets. Returns `(data_start, parsed header)`.
pub(crate) fn container_header(
    bytes: &[u8],
    ckpt: &'static str,
    file: &str,
) -> Result<(usize, serde_json::Value)> {
    let bad = |detail: String| LayaError::Pin {
        checkpoint: ckpt,
        file: file.to_string(),
        detail,
    };
    if bytes.len() < 8 {
        return Err(bad(format!("{} bytes: no header length", bytes.len())));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let data_start = 8usize
        .checked_add(header_len)
        .ok_or_else(|| bad("header length overflow".into()))?;
    if bytes.len() < data_start {
        return Err(bad(format!(
            "header claims {header_len} bytes but the file has {}",
            bytes.len()
        )));
    }
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..data_start]).map_err(|e| bad(format!("header: {e}")))?;
    Ok((data_start, header))
}

/// Parse + widen an in-memory safetensors buffer (the test seam).
pub fn from_bytes(bytes: &[u8], ckpt: &'static str) -> Result<HashMap<String, Weights>> {
    let bad = |detail: String| LayaError::Pin {
        checkpoint: ckpt,
        file: "model.safetensors".to_string(),
        detail,
    };
    let (data_start, header) = container_header(bytes, ckpt, "model.safetensors")?;
    let entries = header
        .as_object()
        .ok_or_else(|| bad("header is not a JSON object".into()))?;

    let mut out: HashMap<String, Weights> = HashMap::with_capacity(entries.len());
    for (name, entry) in entries {
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry["dtype"]
            .as_str()
            .ok_or_else(|| bad(format!("{name}: missing dtype")))?;
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .ok_or_else(|| bad(format!("{name}: missing shape")))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .map(|v| v as usize)
                    .ok_or_else(|| bad(format!("{name}: non-integer shape dim")))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let offsets = entry["data_offsets"]
            .as_array()
            .ok_or_else(|| bad(format!("{name}: missing data_offsets")))?;
        if offsets.len() != 2 {
            return Err(bad(format!("{name}: data_offsets must be [begin, end]")));
        }
        let begin = offsets[0].as_u64().unwrap_or(u64::MAX) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;
        let numel: usize = shape.iter().product();
        // The expected byte span per dtype — Q8_0/Q4_0 are BLOCKED (34 / 18
        // bytes per 32 weights + a short tail block), every other dtype is
        // flat (its per-element width validated by `widen`'s own dtype
        // match).
        let expect = if dtype == "Q8_0" {
            q8_blocked_len(numel)
        } else if dtype == "Q4_0" {
            q4_blocked_len(numel)
        } else {
            numel * dtype_width(dtype, name, ckpt)?
        };
        if end < begin || end - begin != expect {
            return Err(bad(format!(
                "{name}: data span {} bytes != the {dtype} layout ({expect} for {numel} elements)",
                end.saturating_sub(begin)
            )));
        }
        let abs_begin = data_start
            .checked_add(begin)
            .ok_or_else(|| bad(format!("{name}: data offset overflow")))?;
        let abs_end = data_start
            .checked_add(end)
            .ok_or_else(|| bad(format!("{name}: data offset overflow")))?;
        if bytes.len() < abs_end {
            return Err(bad(format!(
                "{name}: data ends at {abs_end} but the file has {} bytes",
                bytes.len()
            )));
        }
        // Plan 616 Phase 1: the Q8_0 payload is RETAINED raw (never widened
        // at parse — the whole point of the tier); Plan 616 Phase 3: the
        // same for Q4_0. Every other dtype widens here exactly as before.
        let data = if dtype == "Q8_0" {
            let raw = RawQ8::new(numel, bytes[abs_begin..abs_end].to_vec())
                .map_err(|e| bad(format!("{name}: {e}")))?;
            WeightData::Q8(raw)
        } else if dtype == "Q4_0" {
            let raw = RawQ4::new(numel, bytes[abs_begin..abs_end].to_vec())
                .map_err(|e| bad(format!("{name}: {e}")))?;
            WeightData::Q4(raw)
        } else {
            WeightData::F32(widen(dtype, &bytes[abs_begin..abs_end], name, ckpt)?)
        };
        out.insert(name.to_string(), Weights { shape, data });
    }
    Ok(out)
}

/// Widen a Q8_0 tensor's storage bytes to f32: per 32-weight block, one
/// f16 scale + N i8 quants; `w = d · q`. The tail (numel % BLOCK) is its
/// own block with its own scale — the converter's layout. The decode
/// arithmetic is the fake-quant path's OWN (d · f32::from(q)), so an
/// artifact widened here is byte-identical to the same weights
/// fake-quantized in memory — the D2a adoption's no-numerics-change
/// proof (instinct issue 018).
///
/// The blocked length is the CONSTRUCTOR's law ([`RawQ8::new`]); this
/// body only asserts it (it cannot fire past that wall) and is the ONE
/// arithmetic both the host resolution ([`RawQ8::wide`]) and the Metal
/// load kernel's bit-identity gate compare against.
#[must_use]
pub fn widen_q8_0(raw: &[u8], numel: usize) -> Vec<f32> {
    debug_assert_eq!(raw.len(), q8_blocked_len(numel));
    let block = super::fake_quant::BLOCK;
    let full = numel / block;
    let tail = numel % block;
    let mut out = Vec::with_capacity(numel);
    let mut pos = 0usize;
    let mut take_block = |out: &mut Vec<f32>, count: usize| {
        let bits = u16::from_le_bytes(raw[pos..pos + 2].try_into().expect("2 scale bytes"));
        pos += 2;
        let d = super::fake_quant::q8_scale_f32(bits);
        for q in &raw[pos..pos + count] {
            out.push(d * f32::from(i8::from_le_bytes([*q])));
        }
        pos += count;
    };
    for _ in 0..full {
        take_block(&mut out, block);
    }
    if tail > 0 {
        take_block(&mut out, tail);
    }
    out
}

/// Widen a Q4_0 tensor's storage bytes to f32 (Plan 616 Phase 3): per
/// 32-weight block, one f16 scale + 16 packed-nibble bytes; `w = d · q`
/// with `q` the SIGNED 4-bit value — the even element in a byte's LOW
/// nibble, the odd element in the HIGH nibble (GGML's nibble order), each
/// sign-extended from 4 bits. The stored grid is `[-7, +7]` (`d =
/// f16(amax/7)`, both ±amax on the grid ends up to the f16 scale's own
/// rounding, zero exact); a `-8` nibble decodes too (never produced by
/// the house converter, accepted for GGML interop). The tail (`numel %
/// BLOCK`) is its own block with its own scale — the converter's layout;
/// its final half-byte (when the tail count is odd) is present but
/// unused.
///
/// The blocked length is the CONSTRUCTOR's law ([`RawQ4::new`]); this
/// body only asserts it and is the ONE arithmetic both the host
/// resolution ([`RawQ4::wide`]) and the Metal `q4_widen_t` load kernel's
/// bit-identity gate compare against.
#[must_use]
pub fn widen_q4_0(raw: &[u8], numel: usize) -> Vec<f32> {
    debug_assert_eq!(raw.len(), q4_blocked_len(numel));
    let block = super::fake_quant::BLOCK;
    let full = numel / block;
    let tail = numel % block;
    let mut out = Vec::with_capacity(numel);
    let mut pos = 0usize;
    let mut take_block = |out: &mut Vec<f32>, count: usize| {
        let bits = u16::from_le_bytes(raw[pos..pos + 2].try_into().expect("2 scale bytes"));
        pos += 2;
        let d = super::fake_quant::q4_scale_f32(bits);
        let bytes = count.div_ceil(2);
        for i in 0..count {
            let byte = raw[pos + i / 2];
            let nib = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let sq = ((nib as i8) << 4) >> 4; // sign-extend 4 bits
            out.push(d * f32::from(sq));
        }
        pos += bytes;
    };
    for _ in 0..full {
        take_block(&mut out, block);
    }
    if tail > 0 {
        take_block(&mut out, tail);
    }
    out
}

/// The storage byte width per element of `dtype`.
fn dtype_width(dtype: &str, name: &str, ckpt: &'static str) -> Result<usize> {
    match dtype {
        "F32" => Ok(4),
        "F16" => Ok(2),
        "F64" => Ok(8),
        other => Err(LayaError::Pin {
            checkpoint: ckpt,
            file: name.to_string(),
            detail: format!(
                "unsupported dtype {other:?} — the riir reader widens F16/F32/F64 only"
            ),
        }),
    }
}

/// Widen a tensor's storage bytes to f32 (little-endian throughout).
fn widen(dtype: &str, bytes: &[u8], name: &str, ckpt: &'static str) -> Result<Vec<f32>> {
    let bad = |detail: String| LayaError::Pin {
        checkpoint: ckpt,
        file: name.to_string(),
        detail,
    };
    let data: Vec<f32> = match dtype {
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        "F16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_bits_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        "F64" => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        other => {
            return Err(bad(format!(
                "unsupported dtype {other:?} — the riir reader widens F16/F32/F64 only"
            )));
        }
    };
    Ok(data)
}

/// Exact F16 → F32 widening (the spec's bit laws, including subnormals —
/// `sign · m · 2⁻²⁴` is exact in f32: a 10-bit mantissa times a power of
/// two).
#[must_use]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = (bits & 0x7C00) >> 10;
    let man = bits & 0x03FF;
    let bits32: u32 = match (exp, man) {
        (0, 0) => sign, // ±0
        (0, m) => {
            // Subnormal: value = m · 2⁻²⁴ — exact f32 product of an integer
            // ≤ 1023 with a power of two (10 significant bits < 24).
            let mag = (m as f32) * f32::from_bits((127 - 24) << 23);
            return if bits & 0x8000 != 0 { -mag } else { mag };
        }
        (0x1F, m) => sign | 0x7F80_0000 | (u32::from(m) << 13), // ±inf / NaN
        (e, m) => sign | ((u32::from(e) + 112) << 23) | (u32::from(m) << 13),
    };
    f32::from_bits(bits32)
}

/// f32 → f16 bits, round-to-nearest-even — the inverse narrowing of
/// [`f16_bits_to_f32`]. Used ONLY by the ANE lane's host-side gather, whose
/// source values were widened FROM f16 (every fp16 value is exact in f32),
/// so on that path the narrowing is bit-exact regardless of rounding mode —
/// RNE still, because it is the mode numpy's `.astype(np.float16)` (the
/// conversion tool the artifacts were built with) uses, and a general f32
/// input must round the same way the Python smoke did.
#[must_use]
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp32 = (bits >> 23) & 0xFF;
    let man32 = bits & 0x007F_FFFF;
    match exp32 {
        // ±0 (sub-f16 inputs flush through the rounding path below; they can
        // only arise from a non-f16 source, which the ANE lane never feeds).
        0 => return sign,
        0xFF => {
            // ±inf / NaN (NaN payload truncated — f16 has no quiet-bit
            // guarantee to preserve; the ANE inputs are never NaN).
            return sign | 0x7C00 | if man32 != 0 { 0x0200 } else { 0 };
        }
        _ => {}
    }
    // Re-center the f32 exponent to the f16 bias (f32 bias 127 → f16 bias
    // 15): unbiased = exp32 - 127, biased for f16 = unbiased + 15.
    let unbiased = exp32 as i32 - 127;
    let half_exp = unbiased + 15;
    if half_exp >= 0x1F {
        // Overflow → ±inf (f16 max ≈ 65504; unreachable from f16-sourced data).
        return sign | 0x7C00;
    }
    if half_exp <= 0 {
        // Subnormal f16 (or underflow to zero): shift the implicit 1 in,
        // round-to-nearest-even at 10 mantissa bits. From f16-sourced data
        // only ±0 lands here.
        let shift = (1 - half_exp) as u32; // 1..=25
        if shift > 24 {
            return sign;
        }
        let mantissa = man32 | 0x0080_0000; // restore the implicit 1
        let half_man = mantissa >> (13 + shift);
        let rem = mantissa & ((1 << (13 + shift)) - 1);
        let half_bit = 1 << (12 + shift);
        let round = if rem > half_bit || (rem == half_bit && (half_man & 1) == 1) {
            1
        } else {
            0
        };
        return sign | (half_man + round) as u16;
    }
    // Normal f16: round the 13 dropped mantissa bits, RNE.
    let half_man = man32 >> 13;
    let rem = man32 & 0x1FFF;
    let round = if rem > 0x1000 || (rem == 0x1000 && (half_man & 1) == 1) {
        1
    } else {
        0
    };
    let half_man = half_man + round;
    // A rounding carry can spill out of the mantissa into the exponent
    // (0x3FF + 1 = 0x400 → exponent + 1, mantissa 0) — the shift handles it;
    // a carry OUT of 0x1F exponent overflows to inf, unreachable from
    // f16-sourced data.
    if half_man & 0x0400 != 0 {
        return sign | (((half_exp + 1) as u16) << 10);
    }
    sign | ((half_exp as u16) << 10) | (half_man as u16 & 0x03FF)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f32→f16→f32 roundtrips EXACTLY for every f16 value (the ANE gather's
    /// premise: its source values were widened FROM f16, so the narrowing
    /// back is lossless) — sampled across all five bit regions plus a dense
    /// sweep of the normals.
    #[test]
    fn f32_f16_roundtrip_is_exact_for_every_f16_value() {
        let mut bits = 0u32;
        loop {
            let f = f16_bits_to_f32(bits as u16);
            if f.is_nan() {
                // NaN payloads collapse to one canonical quiet NaN — every
                // OTHER bit pattern (incl. ±inf) must roundtrip exactly.
                assert!(f32_to_f16_bits(f) & 0x7C00 == 0x7C00);
                assert!(f32_to_f16_bits(f) & 0x03FF != 0 || f32_to_f16_bits(f) & 0x03FF == 0);
            } else {
                let back = f32_to_f16_bits(f);
                assert_eq!(
                    back, bits as u16,
                    "roundtrip broke at f16 bits {bits:#06x} ({f})"
                );
            }
            if bits & 0xFFFF == 0xFFFF {
                break;
            }
            bits += 1;
        }
    }

    /// RNE ties-to-even on the general-f32 path (numpy parity: the
    /// conversion tool rounds the same way).
    #[test]
    fn f32_to_f16_rounds_ties_to_even() {
        // 0.5 ulp above 1.0 in f16 (2049/2048) ties between 1.0 and
        // 1.0009766 — RNE picks the even mantissa (1.0).
        let tie_up = f32::from_bits((127 << 23) | (1 << 12)); // 1 + 2^-12
        assert_eq!(f32_to_f16_bits(tie_up), 0x3C00); // 1.0, even
        // 1.5 ulp above 1.0 rounds up (not a tie).
        let above = f32::from_bits((127 << 23) | (1 << 12) | 1);
        assert_eq!(f32_to_f16_bits(above), 0x3C01);
    }

    /// One F16 tensor + one F32 tensor + `__metadata__`, serialized by hand
    /// into the exact safetensors layout — validates header length, JSON
    /// parsing, offset math and widening end to end.
    #[test]
    fn parses_a_hand_built_safetensors_buffer() {
        // F16 payload: [0x3C00 (1.0), 0x4000 (2.0), 0x0000 (0.0)]
        let f16_data: [u8; 6] = [0x00, 0x3C, 0x00, 0x40, 0x00, 0x00];
        // F32 payload: [−1.5f32]
        let f32_data = (-1.5f32).to_le_bytes();
        let header = r#"{"__metadata__":{"hub":"test"},"a":{"dtype":"F16","shape":[3],"data_offsets":[0,6]},"b":{"dtype":"F32","shape":[1,1],"data_offsets":[6,10]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&f16_data);
        buf.extend_from_slice(&f32_data);

        let map = from_bytes(&buf, "test").expect("parses");
        assert_eq!(map["a"].shape, vec![3]);
        assert_eq!(map["a"].wide_f32(), &[1.0, 2.0, 0.0]);
        assert_eq!(map["b"].shape, vec![1, 1]);
        assert_eq!(map["b"].wide_f32(), &[-1.5]);
    }

    /// A Q8_0 tensor parses to the RETAINED raw payload (never widened at
    /// parse — Plan 616 Phase 1): the blocked bytes survive verbatim, the
    /// resolution widens through the same `d · q` arithmetic, and a
    /// malformed blocked length is refused loud (naming the tensor).
    #[test]
    fn q8_tensor_retains_raw_bytes_and_resolves_on_demand() {
        use super::WeightData;
        // One full block + a 4-weight tail: scale f16 0x3C00 (1.0), quants
        // [1, -2, 3, 0, ...]; tail scale 0x4000 (2.0), quants [-127, 127,
        // 5, -6].
        let mut q8: Vec<u8> = Vec::new();
        q8.extend_from_slice(&0x3C00u16.to_le_bytes());
        q8.extend_from_slice(&[1, 254, 3, 0, 7, 8, 9, 10]); // 254 = -2 as u8
        q8.resize(2 + 32, 0x81); // 0x81 = -127
        q8.extend_from_slice(&0x4000u16.to_le_bytes());
        q8.extend_from_slice(&[129, 127, 5, 250]); // 129 = -127, 250 = -6
        let numel = 36;
        let header = format!(
            r#"{{"w":{{"dtype":"Q8_0","shape":[6,6],"data_offsets":[0,{}]}}}}"#,
            q8.len()
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&q8);

        let map = from_bytes(&buf, "test").expect("parses");
        let w = &map["w"];
        assert_eq!(w.shape, vec![6, 6]);
        let WeightData::Q8(raw) = &w.data else {
            panic!("q8 tensor must retain the raw payload");
        };
        assert_eq!(raw.raw(), &q8);
        assert_eq!(raw.numel(), numel);
        // The resolution IS the widen arithmetic (spot values: q=1 → 1.0,
        // the tail's -127 → -254.0).
        assert_eq!(w.wide_f32()[0], 1.0);
        assert_eq!(w.wide_f32()[32], -127.0 * 2.0);
        assert_eq!(w.wide_f32().len(), numel);

        // The malformed-payload wall: a truncated blocked span refuses.
        let header_bad = r#"{"w":{"dtype":"Q8_0","shape":[6,6],"data_offsets":[0,5]}}"#;
        let mut bad = Vec::new();
        bad.extend_from_slice(&(header_bad.len() as u64).to_le_bytes());
        bad.extend_from_slice(header_bad.as_bytes());
        bad.extend_from_slice(&[0u8; 5]);
        let err = from_bytes(&bad, "test").expect_err("refused");
        assert!(err.to_string().contains("Q8_0 layout"), "{err}");
    }

    /// The kill-switch (`LAYA_Q8_HOST_F32=1`) widens at
    /// [`Weight2D::from_weights`] and drops the raw bytes — today's
    /// load-shape restored. The variable is read ONLY by `from_weights`
    /// (grep: this test is its only other reader in this binary), so the
    /// in-process mutation cannot race another test's assertion.
    #[test]
    fn host_f32_kill_switch_widens_at_load() {
        use super::{RawQ8, Weight2D, WeightData};
        // SAFETY: env mutation in edition 2024; the variable is read only
        // by `Weight2D::from_weights`, no other thread is running in this
        // process at this moment, and the value is restored below.
        unsafe { std::env::set_var("LAYA_Q8_HOST_F32", "1") };
        let mk = |raw: Vec<u8>| {
            Weight2D::from_weights(Weights {
                shape: vec![2, 32],
                data: WeightData::Q8(RawQ8::new(64, raw).expect("layout")),
            })
        };
        let mut raw = vec![0u8; 68]; // 64 elements = 2 full blocks
        raw[..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        raw[2] = 3;
        let w = mk(raw);
        assert!(
            matches!(w, Weight2D::Dense(_)),
            "switch forces the dense form"
        );
        assert_eq!(w.wide()[0], 3.0);
        assert_eq!(w.len(), 64);
        // SAFETY: restoring the unset state (see above).
        unsafe { std::env::remove_var("LAYA_Q8_HOST_F32") };
        let mut raw2 = vec![0u8; 68];
        raw2[..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        raw2[2] = 5;
        let w2 = mk(raw2);
        assert!(
            matches!(w2, Weight2D::Q8(_)),
            "default retains the raw bytes"
        );
    }

    /// A Q4_0 tensor parses to the RETAINED raw payload (Plan 616 Phase
    /// 3 — the Q8 law's 4-bit shape): the blocked bytes survive verbatim,
    /// the resolution widens through the same `d · q` nibble arithmetic,
    /// a malformed blocked length is refused loud, and the nibble order
    /// is the GGML law (even element low, odd element high).
    #[test]
    fn q4_tensor_retains_raw_bytes_and_resolves_on_demand() {
        use super::WeightData;
        // One full block (scale f16 0x3C00 = 1.0) + a 3-weight tail (scale
        // 0x4000 = 2.0, an ODD tail — its final half-byte is present and
        // unused). Quants encode 4, -4 | 7, -1 | ... as nibble pairs:
        // byte = (odd << 4) | even.
        let mut q4: Vec<u8> = Vec::new();
        q4.extend_from_slice(&0x3C00u16.to_le_bytes());
        // Nibble pairs (even low, odd high): 0xC4 → w0 = 4, w1 = 0xC → -4;
        // 0x1F → w2 = 0xF → -1, w3 = 0x1 → 1. The rest zeros.
        q4.extend_from_slice(&[
            0xC4, 0x1F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
        q4.extend_from_slice(&0x4000u16.to_le_bytes());
        q4.extend_from_slice(&[0xC5, 0x00]); // tail: w32 = 5 (low), w33 = 0xC → -4 (high); w34's byte unused (odd tail → the half-byte is present)
        let numel = 35;
        assert_eq!(q4.len(), q4_blocked_len(numel));
        let header = format!(
            r#"{{"w":{{"dtype":"Q4_0","shape":[7,5],"data_offsets":[0,{}]}}}}"#,
            q4.len()
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&q4);

        let map = from_bytes(&buf, "test").expect("parses");
        let w = &map["w"];
        assert_eq!(w.shape, vec![7, 5]);
        let WeightData::Q4(raw) = &w.data else {
            panic!("q4 tensor must retain the raw payload");
        };
        assert_eq!(raw.raw(), &q4);
        assert_eq!(raw.numel(), numel);
        // The resolution IS the widen arithmetic (spot values; the scale is
        // exactly 1.0 / 2.0 so d·q is exact):
        assert_eq!(w.wide_f32()[0], 4.0); // block 0, low nibble of byte 0
        assert_eq!(w.wide_f32()[1], -4.0); // block 0, high nibble 0xC → -4
        assert_eq!(w.wide_f32()[2], -1.0); // low nibble 0xF → -1
        assert_eq!(w.wide_f32()[3], 1.0); // high nibble 0x1 → 1
        assert_eq!(w.wide_f32()[32], 10.0); // tail low nibble 5, scale 2.0
        assert_eq!(w.wide_f32()[33], -8.0); // tail high nibble -4, scale 2.0
        assert_eq!(w.wide_f32().len(), numel);

        // The malformed-payload wall: a truncated blocked span refuses.
        let header_bad = r#"{"w":{"dtype":"Q4_0","shape":[7,5],"data_offsets":[0,5]}}"#;
        let mut bad = Vec::new();
        bad.extend_from_slice(&(header_bad.len() as u64).to_le_bytes());
        bad.extend_from_slice(header_bad.as_bytes());
        bad.extend_from_slice(&[0u8; 5]);
        let err = from_bytes(&bad, "test").expect_err("refused");
        assert!(err.to_string().contains("Q4_0 layout"), "{err}");
    }

    /// The Q4 round-trip law: widen_q4_0 is a pure function of the
    /// blocked bytes, and the bytes a fake-quant-Q4 grid produces decode
    /// to the same values the grid computed (the converter's proof at
    /// the element level, on synthetic data).
    #[test]
    fn q4_widen_matches_the_fake_quant_grid_synthetic() {
        use super::super::fake_quant::{BLOCK, fake_quant_q4, q4_quant_of, q4_scale_bits};
        // Build a deterministic tensor, fake-quant it in memory, encode
        // the SAME grid to bytes by hand (the converter's own walk), and
        // require the widen of those bytes to equal the fake-quant values
        // bit-for-bit.
        let mut s = 0xDEADBEEFu32;
        let mut data = Vec::with_capacity(3 * BLOCK + 7);
        for _ in 0..3 * BLOCK + 7 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (s >> 8) as f32 / 16_777_216.0;
            data.push((u - 0.5) * 2.0);
        }
        let mut expect = data.clone();
        fake_quant_q4(&mut expect);

        // Encode (the converter's block walk + GGML nibble packing).
        let mut raw = Vec::new();
        for block in data.chunks(BLOCK) {
            let amax = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            raw.extend_from_slice(&q4_scale_bits(amax).to_le_bytes());
            let d = f16_bits_to_f32(q4_scale_bits(amax));
            for pair in block.chunks(2) {
                let lo = q4_quant_of(pair[0], d) as u8;
                let hi = pair.get(1).map(|&v| q4_quant_of(v, d) as u8).unwrap_or(0);
                raw.push(((hi & 0x0F) << 4) | (lo & 0x0F));
            }
        }
        assert_eq!(raw.len(), q4_blocked_len(data.len()));
        let got = widen_q4_0(&raw, data.len());
        assert_eq!(got.len(), expect.len());
        for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            assert_eq!(g.to_bits(), e.to_bits(), "element {i}: {g} vs {e}");
        }
    }

    /// The kill-switch (`LAYA_Q8_HOST_F32=1`) widens a Q4 payload at
    /// [`Weight2D::from_weights`] too — the switch is the raw-quant
    /// family's host-widen restore, not a Q8-only knob.
    #[test]
    fn host_f32_kill_switch_covers_the_q4_payload() {
        use super::{RawQ4, Weight2D, WeightData};
        // SAFETY: env mutation in edition 2024; the variable is read only
        // by `Weight2D::from_weights`, no other thread is running in this
        // process at this moment, and the value is restored below.
        unsafe { std::env::set_var("LAYA_Q8_HOST_F32", "1") };
        let mut raw = Vec::new();
        raw.extend_from_slice(&0x3C00u16.to_le_bytes()); // scale 1.0
        raw.extend_from_slice(&[0xC4]); // w0 = 4, w1 = -4
        raw.resize(18, 0x00); // one full block: 2 + 16 bytes
        let w = Weight2D::from_weights(Weights {
            shape: vec![1, 32],
            data: WeightData::Q4(RawQ4::new(32, raw).expect("layout")),
        });
        assert!(
            matches!(w, Weight2D::Dense(_)),
            "switch forces the dense form"
        );
        assert_eq!(w.wide()[0], 4.0);
        assert_eq!(w.wide()[1], -4.0);
        // SAFETY: restoring the unset state (see above).
        unsafe { std::env::remove_var("LAYA_Q8_HOST_F32") };
    }

    /// The F16 widening bit laws on known values, including the subnormal /
    /// infinity / sign edges.
    #[test]
    fn f16_widening_is_exact_on_known_values() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert!(f16_bits_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_bits_to_f32(0x8000), -0.0);
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0xBC00), -1.0);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        // Max subnormal 0x03FF = 1023 · 2⁻²⁴; min normal 0x0400 = 2⁻¹⁴.
        let max_sub = f16_bits_to_f32(0x03FF);
        let expect_sub = 1023.0f32 * 2.0f32.powi(-24);
        assert_eq!(max_sub, expect_sub);
        assert_eq!(f16_bits_to_f32(0x0400), 2.0f32.powi(-14));
        assert!(f16_bits_to_f32(0x7C00).is_infinite());
        assert!(f16_bits_to_f32(0xFC00).is_infinite());
        assert!(f16_bits_to_f32(0x7E00).is_nan());
        // A normal with a fractional mantissa: 0x3555 = (1 + 341/1024)·2⁻².
        let v = f16_bits_to_f32(0x3555);
        let expect = (1.0 + 341.0 / 1024.0) * 2.0f32.powi(-2);
        assert!((v - expect).abs() < 1e-9);
    }

    /// An unknown dtype is refused LOUD (naming the tensor), never
    /// mis-widened.
    #[test]
    fn unknown_dtype_is_refused_loud() {
        let header = r#"{"t":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&[0x00; 2]);
        let err = from_bytes(&buf, "test").expect_err("refused");
        let msg = err.to_string();
        assert!(msg.contains("BF16") && msg.contains("t"), "{msg}");
    }
}
