//! Multi-host pipeline substrate (Plan 623 Phase 1).
//!
//! The wire + handshake types that let ONE model run across two hosts as
//! layer-range stages: stage 0 runs layers `[0..k)` and hands the hidden
//! activation to stage 1, which runs layers `[k..n)` + final norm + LM head.
//! Output must be byte-identical to the 1-box forward (`tests/
//! plan623_multi_host_g1.rs` — the G1 composition gate).
//!
//! What lives here (feature `multi_host`, opt-in, never default):
//!
//! - [`LayerRange`] — the stage's half-open layer window.
//! - [`FrameDtype`] + [`ActivationFrame`] — the activation wire frame
//!   (16-byte little-endian header + payload; f32 exact posture, bf16
//!   documented-lossy with round-to-nearest-even encoding).
//! - [`StageHandshake`] — the boot-time stage contract; a mismatched pair
//!   refuses LOUD (never a silent wrong-shape stage).
//! - [`HybridStateSegment`] — the per-range [`HybridCache`] state
//!   serializer (KV rows + GDN recurrent/conv state for the range's
//!   layers). Closes Issue 028's out-of-process gap: the in-process
//!   `PhaseHandoff` shares one cache; two processes need this wire form.
//!   Wire size matches the `disaggregated.rs` PhaseHandoff budget: per
//!   attention layer `fill_pos × kv_dim × 2` f32, per DeltaNet layer one
//!   recurrent matrix + one conv window.
//!
//! What does NOT live here: the range-forward compute itself — that is a
//! parameterized loop bound inside the existing forward bodies
//! (`deltanet::ternary_forward`, `transformer::gemma2`), compiled ungated
//! so the full forward and the range stages execute the SAME code and the
//! composition gate proves no drift. Only the public stage entry points
//! are gated behind this feature.
//!
//! # Determinism
//!
//! Every wire encoding is little-endian by field (`to_le_bytes`), never a
//! host-endian struct cast — the frame must decode identically on either
//! side of the pipe. Payload f32/bf16 move as raw LE bits (bytemuck cast on
//! little-endian hosts, the `gguf_loader` precedent; big-endian hosts
//! refuse at compile time). State round-trips are bit-exact; the bf16
//! dtype is the ONLY lossy surface and rides its own per-family retention
//! gate (the lossy-surface law) — f32 is the G1 posture.

use crate::deltanet::HybridCache;
use crate::types::DeltaNetLayerType;

/// A stage's half-open layer window `[lo, hi)`.
///
/// `lo == 0` marks the FIRST stage (it owns the embedding lookup);
/// `hi == n_layer` marks the LAST stage (it owns final norm + LM head).
/// A middle stage (`0 < lo && hi < n_layer`) consumes a hidden vector and
/// produces a hidden vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerRange {
    pub lo: u32,
    pub hi: u32,
}

impl LayerRange {
    /// The full-model range (the 1-box posture).
    pub fn full(n_layer: u32) -> Self {
        Self { lo: 0, hi: n_layer }
    }

    pub fn new(lo: u32, hi: u32) -> Self {
        Self { lo, hi }
    }

    /// Validate against the model's layer count: `0 <= lo < hi <= n_layer`.
    /// The public stage entry points refuse on a bad range — a silent
    /// empty/wrong stage would read as a confident zero-work forward.
    pub fn validate(&self, n_layer: u32) -> Result<(), String> {
        if self.lo >= self.hi {
            return Err(format!(
                "LayerRange [{}, {}) is empty or inverted (lo must be < hi)",
                self.lo, self.hi
            ));
        }
        if self.hi > n_layer {
            return Err(format!(
                "LayerRange [{}, {}) exceeds n_layer {n_layer}",
                self.lo, self.hi
            ));
        }
        Ok(())
    }

    pub fn is_first(&self) -> bool {
        self.lo == 0
    }

    pub fn is_last(&self, n_layer: u32) -> bool {
        self.hi == n_layer
    }

    /// The number of layers this stage runs.
    pub fn len(&self) -> u32 {
        self.hi - self.lo
    }

    /// True when the range is empty/inverted (a `validate` failure class,
    /// checkable without the model).
    pub fn is_empty(&self) -> bool {
        self.lo >= self.hi
    }
}

/// Activation frame payload dtype.
///
/// `F32` is the exact posture (the G1 gate runs on it). `Bf16` is a LOSSY
/// surface: encoding rounds to nearest-even and drops the low 16 mantissa
/// bits; the lossy-surface promotion rule applies (per-family conditional
/// retention, never bit-identity claims).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameDtype {
    F32 = 0,
    Bf16 = 1,
}

impl FrameDtype {
    pub fn from_u8(v: u8) -> Result<Self, String> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::Bf16),
            other => Err(format!("unknown FrameDtype tag {other}")),
        }
    }

    /// Bytes per scalar on the wire.
    pub fn scalar_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

/// f32 → bf16 with round-to-nearest-even (the IEEE bfloat16 conversion).
/// Lossy by construction (16 mantissa bits dropped); NaN/Inf bit-shapes are
/// preserved as NaN/Inf.
pub fn f32_to_bf16_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    // NaN: keep it NaN (force the quiet bit) — rounding could otherwise
    // carry it to Inf.
    if bits & 0x7fff_ffffu32 > 0x7f80_0000u32 {
        return ((bits | 0x0040_0000) >> 16) as u16;
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// bf16 → f32 (exact — a bf16 value IS an f32 value).
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// The 16-byte activation frame header — little-endian on the wire.
///
/// Layout (offset, size): `seq_id` (0, 4) · `pos` (4, 4) · `n_tok` (8, 2) ·
/// `dtype` (10, 1) · `reserved` (11, 1) · `payload_len` (12, 4) = 16 bytes,
/// no padding holes (repr(C), all fields ≤ their natural alignment).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ActivationFrameHeader {
    /// Pipeline stream id (multi-stream decode shares one transport).
    pub seq_id: u32,
    /// The sequence position this frame's activation corresponds to (the
    /// LAST position processed by the upstream stage).
    pub pos: u32,
    /// Token count carried by the payload (1 = the decode fast path).
    pub n_tok: u16,
    /// Payload dtype tag ([`FrameDtype`]).
    pub dtype: u8,
    /// Reserved — must encode as 0 and is refused non-zero on decode
    /// (forward-compatible extension space, fail-closed).
    pub reserved: u8,
    /// Payload byte length (scalar count × dtype scalar bytes).
    pub payload_len: u32,
}

impl ActivationFrameHeader {
    /// The struct's size (no padding) — the fields after the 4-byte magic.
    pub const SIZE: usize = 16;
    /// The magic prefix length; the header fields start at this offset and
    /// the payload at `MAGIC_LEN + SIZE`.
    pub const MAGIC_LEN: usize = 4;
    /// Total wire prefix (magic + header fields) before the payload.
    pub const WIRE_PREFIX: usize = Self::MAGIC_LEN + Self::SIZE;
    pub const MAGIC: [u8; 4] = *b"MAF1";

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&Self::MAGIC);
        out.extend_from_slice(&self.seq_id.to_le_bytes());
        out.extend_from_slice(&self.pos.to_le_bytes());
        out.extend_from_slice(&self.n_tok.to_le_bytes());
        out.extend_from_slice(&[self.dtype, self.reserved]);
        out.extend_from_slice(&self.payload_len.to_le_bytes());
    }

    /// Decode a header from the front of `bytes`; returns the header and
    /// the payload offset ([`Self::WIRE_PREFIX`]).
    pub fn decode(bytes: &[u8]) -> Result<(Self, usize), String> {
        if bytes.len() < Self::WIRE_PREFIX {
            return Err(format!(
                "ActivationFrame: {} bytes < wire prefix {}",
                bytes.len(),
                Self::WIRE_PREFIX
            ));
        }
        if bytes[0..4] != Self::MAGIC {
            return Err("ActivationFrame: bad magic (not an MAF1 frame)".to_string());
        }
        let rd_u32 = |off: usize| {
            u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
        };
        let rd_u16 = |off: usize| u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        let header = Self {
            seq_id: rd_u32(4),
            pos: rd_u32(8),
            n_tok: rd_u16(12),
            dtype: bytes[14],
            reserved: bytes[15],
            payload_len: rd_u32(16),
        };
        if header.reserved != 0 {
            return Err("ActivationFrame: reserved byte non-zero (newer writer?)".to_string());
        }
        Ok((header, Self::WIRE_PREFIX))
    }
}

// The header must stay exactly 16 bytes with no padding holes — a repr(C)
// struct that grew would silently change the wire format.
const _: () = assert!(core::mem::size_of::<ActivationFrameHeader>() == 16);

/// The activation wire frame: [`ActivationFrameHeader`] + raw payload.
///
/// Encoding appends `header || payload` to a caller buffer (amortized
/// alloc); decoding reads into a caller-owned `Vec<f32>` whose capacity is
/// reused across calls — the steady-state decode path allocates nothing
/// (the G4 posture, `snapshot_into` precedent).
pub struct ActivationFrame;

impl ActivationFrame {
    /// Encode `values` (host f32) into `out` as `header || payload`.
    ///
    /// `dtype` picks the wire form: [`FrameDtype::F32`] exact,
    /// [`FrameDtype::Bf16`] round-to-nearest-even (LOSSY). `n_tok` is the
    /// caller's token count for the payload (decode fast path: 1).
    #[cfg(target_endian = "little")]
    pub fn encode_into(
        seq_id: u32,
        pos: u32,
        n_tok: u16,
        values: &[f32],
        dtype: FrameDtype,
        out: &mut Vec<u8>,
    ) {
        let start = out.len();
        let header = ActivationFrameHeader {
            seq_id,
            pos,
            n_tok,
            dtype: dtype as u8,
            reserved: 0,
            payload_len: (values.len() * dtype.scalar_bytes()) as u32,
        };
        header.encode_into(out);
        match dtype {
            FrameDtype::F32 => {
                let bytes: &[u8] = bytemuck::cast_slice(values);
                out.extend_from_slice(bytes);
            }
            FrameDtype::Bf16 => {
                out.reserve(values.len() * 2);
                for &v in values {
                    out.extend_from_slice(&f32_to_bf16_rne(v).to_le_bytes());
                }
            }
        }
        let _ = start;
    }

    /// Decode a frame's payload into `out` (resized in place — capacity
    /// reused across calls; nothing allocates in steady state).
    ///
    /// Returns the decoded header. Refuses: short buffers, unknown dtype
    /// tags, a payload shorter than the header's `payload_len`, and (for
    /// F32) a payload length that is not a multiple of 4.
    #[cfg(target_endian = "little")]
    pub fn decode_into(bytes: &[u8], out: &mut Vec<f32>) -> Result<ActivationFrameHeader, String> {
        let (header, payload_off) = ActivationFrameHeader::decode(bytes)?;
        let dtype = FrameDtype::from_u8(header.dtype)?;
        let payload = bytes
            .get(payload_off..)
            .ok_or("ActivationFrame: payload offset out of bounds")?;
        if payload.len() < header.payload_len as usize {
            return Err(format!(
                "ActivationFrame: payload {} < header payload_len {}",
                payload.len(),
                header.payload_len
            ));
        }
        let payload = &payload[..header.payload_len as usize];
        match dtype {
            FrameDtype::F32 => {
                if !payload.len().is_multiple_of(4) {
                    return Err(format!(
                        "ActivationFrame: f32 payload len {} not a multiple of 4",
                        payload.len()
                    ));
                }
                let vals: &[f32] = bytemuck::cast_slice(payload);
                out.clear();
                out.resize(vals.len(), 0.0);
                out.copy_from_slice(vals);
            }
            FrameDtype::Bf16 => {
                if !payload.len().is_multiple_of(2) {
                    return Err(format!(
                        "ActivationFrame: bf16 payload len {} not a multiple of 2",
                        payload.len()
                    ));
                }
                let n = payload.len() / 2;
                out.clear();
                out.resize(n, 0.0);
                for (i, chunk) in payload.as_chunks::<2>().0.iter().enumerate() {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out[i] = bf16_bits_to_f32(bits);
                }
            }
        }
        Ok(header)
    }
}

/// The boot-time stage contract. Both stages of one pipeline exchange
/// their handshakes and each refuses the pair on ANY mismatch — a silent
/// wrong-shape stage would produce confident garbage (the one failure the
/// multi-host lane must never have).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageHandshake {
    /// Architecture tag, e.g. `"qwen35-deltanet-ternary"` / `"gemma2"`.
    pub arch: String,
    /// blake3 over the geometry metadata (the `disaggregated::
    /// geometry_fingerprint` shape) — every Config field the forward reads.
    pub geometry_fingerprint: [u8; 32],
    /// blake3 of the artifact bytes this stage serves.
    pub artifact_blake3: [u8; 32],
    /// This stage's layer window.
    pub range: LayerRange,
    /// The activation dtype this stage speaks.
    pub dtype: FrameDtype,
}

impl StageHandshake {
    /// Verify `upstream` directly feeds `self` (upstream.hi == self.lo)
    /// and every identity field matches. `n_layer` total layers.
    pub fn verify_pair(&self, upstream: &Self, n_layer: u32) -> Result<(), String> {
        if self.arch != upstream.arch {
            return Err(format!(
                "stage pair arch mismatch: upstream '{}' vs downstream '{}'",
                upstream.arch, self.arch
            ));
        }
        if self.geometry_fingerprint != upstream.geometry_fingerprint {
            return Err("stage pair geometry_fingerprint mismatch".to_string());
        }
        if self.artifact_blake3 != upstream.artifact_blake3 {
            return Err(
                "stage pair artifact_blake3 mismatch — the stages serve DIFFERENT weights"
                    .to_string(),
            );
        }
        if self.dtype != upstream.dtype {
            return Err(format!(
                "stage pair dtype mismatch: upstream {:?} vs downstream {:?}",
                upstream.dtype, self.dtype
            ));
        }
        upstream.range.validate(n_layer)?;
        self.range.validate(n_layer)?;
        if upstream.range.hi != self.range.lo {
            return Err(format!(
                "stage ranges do not tile: upstream [{}, {}) vs downstream [{}, {})",
                upstream.range.lo, upstream.range.hi, self.range.lo, self.range.hi
            ));
        }
        Ok(())
    }
}

/// Per-range [`HybridCache`] state segment — the wire form of the cache
/// slices ONE stage owns.
///
/// Wire layout (v1, all little-endian):
///
/// ```text
/// magic "MHS1" | lo u32 | hi u32 | n_layer u32 | fill_pos u32
/// per layer in [lo..hi):
///   kind u8 (0 = attention, 1 = deltanet) | 3× reserved 0u8
///   attention: key_len u32 || key f32 LE || value_len u32 || value f32 LE
///   deltanet:  rec_len u32 || rec f32 LE || conv_len u32 || conv f32 LE
/// ```
///
/// Attention layers capture rows `[0..fill_pos × kv_dim)` clamped to the
/// physical buffer (a sliding-window ring holds only its last positions —
/// Issue 683); `fill_pos` rides the header so the restore re-arms the
/// tracker exactly. GDN layers capture the full fixed-size recurrent
/// matrix + conv window (they are position-independent). Round-trip is
/// bit-exact: f32 moves as raw LE bits, nothing renormalizes.
pub struct HybridStateSegment;

impl HybridStateSegment {
    const MAGIC: [u8; 4] = *b"MHS1";
    const KIND_ATTENTION: u8 = 0;
    const KIND_DELTANET: u8 = 1;

    /// Encode the cache state for `range`'s layers.
    ///
    /// `layer_types` names every layer of the model (length `n_layer`).
    /// Allocates into `out`
    /// (append; the segment is a cold-path handoff artifact, not a hot
    /// loop — the allocation posture follows `KVSnapshot`, the allocating
    /// variant).
    pub fn encode_into(
        cache: &HybridCache,
        layer_types: &[DeltaNetLayerType],
        range: LayerRange,
        out: &mut Vec<u8>,
    ) -> Result<(), String> {
        let n_layer = layer_types.len() as u32;
        range.validate(n_layer)?;
        if cache.kv_cache.layers.len() != layer_types.len()
            || cache.deltanet_state.recurrent_states.len() != layer_types.len()
        {
            return Err(format!(
                "HybridStateSegment: cache shape ({} kv / {} gdn layers) != layer_types ({})",
                cache.kv_cache.layers.len(),
                cache.deltanet_state.recurrent_states.len(),
                layer_types.len()
            ));
        }
        let fill_pos = cache.kv_cache.fill_pos();
        out.extend_from_slice(&Self::MAGIC);
        out.extend_from_slice(&range.lo.to_le_bytes());
        out.extend_from_slice(&range.hi.to_le_bytes());
        out.extend_from_slice(&n_layer.to_le_bytes());
        out.extend_from_slice(&(fill_pos as u32).to_le_bytes());
        for (i, lt) in layer_types[range.lo as usize..range.hi as usize]
            .iter()
            .enumerate()
        {
            let i = range.lo as usize + i;
            match *lt {
                DeltaNetLayerType::Attention => {
                    let layer = &cache.kv_cache.layers[i];
                    out.extend_from_slice(&[Self::KIND_ATTENTION, 0, 0, 0]);
                    out.extend_from_slice(&(layer.key.len() as u32).to_le_bytes());
                    for v in &layer.key {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    out.extend_from_slice(&(layer.value.len() as u32).to_le_bytes());
                    for v in &layer.value {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                DeltaNetLayerType::DeltaNet => {
                    let rec = &cache.deltanet_state.recurrent_states[i];
                    let conv = &cache.deltanet_state.conv_states[i];
                    out.extend_from_slice(&[Self::KIND_DELTANET, 0, 0, 0]);
                    out.extend_from_slice(&(rec.len() as u32).to_le_bytes());
                    for v in rec {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    out.extend_from_slice(&(conv.len() as u32).to_le_bytes());
                    for v in conv {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        Ok(())
    }

    /// Decode a segment and write it back into `cache` for `range`'s
    /// layers. `cache` must already be shaped for the model (the restore
    /// resizes NOTHING — a shape mismatch refuses). The KV tracker is set
    /// to the header's `fill_pos` (the `set_fill_pos` seam — buffer
    /// contents ride the per-layer writes, the tracker must agree).
    pub fn restore_into(
        bytes: &[u8],
        cache: &mut HybridCache,
        layer_types: &[DeltaNetLayerType],
        range: LayerRange,
    ) -> Result<(), String> {
        let n_layer = layer_types.len() as u32;
        range.validate(n_layer)?;
        let mut cur = 0usize;
        let rd = |cur: &mut usize, n: usize| -> Result<&[u8], String> {
            let end = *cur + n;
            let s = bytes
                .get(*cur..end)
                .ok_or("HybridStateSegment: truncated")?;
            *cur = end;
            Ok(s)
        };
        let magic = rd(&mut cur, 4)?;
        if magic != Self::MAGIC {
            return Err("HybridStateSegment: bad magic (not an MHS1 segment)".to_string());
        }
        let rd_u32 = |cur: &mut usize| -> Result<u32, String> {
            let s = rd(cur, 4)?;
            Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        };
        let wire_lo = rd_u32(&mut cur)?;
        let wire_hi = rd_u32(&mut cur)?;
        let wire_n_layer = rd_u32(&mut cur)?;
        let wire_fill = rd_u32(&mut cur)?;
        if (wire_lo, wire_hi, wire_n_layer) != (range.lo, range.hi, n_layer) {
            return Err(format!(
                "HybridStateSegment: header range [{wire_lo},{wire_hi}) of {wire_n_layer} \
                 != requested [{}, {}) of {n_layer}",
                range.lo, range.hi
            ));
        }
        if cache.kv_cache.layers.len() != layer_types.len() {
            return Err(format!(
                "HybridStateSegment: cache has {} kv layers, model has {}",
                cache.kv_cache.layers.len(),
                layer_types.len()
            ));
        }
        for (i, lt) in layer_types[range.lo as usize..range.hi as usize]
            .iter()
            .enumerate()
        {
            let i = range.lo as usize + i;
            let kind = rd(&mut cur, 4)?[0];
            match (kind, lt) {
                (Self::KIND_ATTENTION, DeltaNetLayerType::Attention) => {
                    let key_len = rd_u32(&mut cur)? as usize;
                    let key_bytes = rd(&mut cur, key_len * 4)?;
                    let value_len = rd_u32(&mut cur)? as usize;
                    let value_bytes = rd(&mut cur, value_len * 4)?;
                    let layer = &mut cache.kv_cache.layers[i];
                    if layer.key.len() != key_len || layer.value.len() != value_len {
                        return Err(format!(
                            "HybridStateSegment: layer {i} kv shape ({},{}) != wire ({key_len},{value_len})",
                            layer.key.len(),
                            layer.value.len()
                        ));
                    }
                    for (dst, chunk) in layer.key.iter_mut().zip(key_bytes.as_chunks::<4>().0) {
                        *dst = f32::from_le_bytes(*chunk);
                    }
                    for (dst, chunk) in layer.value.iter_mut().zip(value_bytes.as_chunks::<4>().0) {
                        *dst = f32::from_le_bytes(*chunk);
                    }
                }
                (Self::KIND_DELTANET, DeltaNetLayerType::DeltaNet) => {
                    let rec_len = rd_u32(&mut cur)? as usize;
                    let rec_bytes = rd(&mut cur, rec_len * 4)?;
                    let conv_len = rd_u32(&mut cur)? as usize;
                    let conv_bytes = rd(&mut cur, conv_len * 4)?;
                    let rec = &mut cache.deltanet_state.recurrent_states[i];
                    let conv = &mut cache.deltanet_state.conv_states[i];
                    if rec.len() != rec_len || conv.len() != conv_len {
                        return Err(format!(
                            "HybridStateSegment: layer {i} gdn shape ({},{}) != wire ({rec_len},{conv_len})",
                            rec.len(),
                            conv.len()
                        ));
                    }
                    for (dst, chunk) in rec.iter_mut().zip(rec_bytes.as_chunks::<4>().0) {
                        *dst = f32::from_le_bytes(*chunk);
                    }
                    for (dst, chunk) in conv.iter_mut().zip(conv_bytes.as_chunks::<4>().0) {
                        *dst = f32::from_le_bytes(*chunk);
                    }
                }
                (kind, lt) => {
                    return Err(format!(
                        "HybridStateSegment: layer {i} kind tag {kind} != model layer type {lt:?}"
                    ));
                }
            }
        }
        // Only AFTER every layer restored cleanly: re-arm the tracker. A
        // refusal mid-way must leave the cache untouched-by-commitment? No
        // — layers before the failure ARE written; the caller treats a
        // refused restore as a dead stage (the handshake law: refuse loud,
        // never continue half-restored). The tracker is set last so a
        // clean restore is the only path that arms it.
        cache.kv_cache.set_fill_pos(wire_fill as usize);
        Ok(())
    }
}
