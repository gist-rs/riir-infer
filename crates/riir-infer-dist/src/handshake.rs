//! Wire form of [`StageHandshake`] — little-endian by field, never a
//! host-endian struct cast (the `multi_host` determinism posture).
//!
//! `arch_len u16 | arch | geometry[32] | artifact[32] | lo u32 | hi u32 | dtype u8`

use riir_infer_core::multi_host::{FrameDtype, LayerRange, StageHandshake};

/// Arch tags are short identifiers; anything longer is a malformed peer.
const MAX_ARCH_LEN: usize = 256;

pub fn encode(h: &StageHandshake, out: &mut Vec<u8>) {
    out.clear();
    let arch = h.arch.as_bytes();
    out.extend_from_slice(&(arch.len() as u16).to_le_bytes());
    out.extend_from_slice(arch);
    out.extend_from_slice(&h.geometry_fingerprint);
    out.extend_from_slice(&h.artifact_blake3);
    out.extend_from_slice(&h.range.lo.to_le_bytes());
    out.extend_from_slice(&h.range.hi.to_le_bytes());
    out.push(h.dtype as u8);
}

pub fn decode(b: &[u8]) -> Result<StageHandshake, String> {
    let take = |at: usize, n: usize| -> Result<&[u8], String> {
        b.get(at..at + n)
            .ok_or_else(|| format!("handshake truncated at byte {at} (+{n}, have {})", b.len()))
    };
    let arch_len = u16::from_le_bytes(take(0, 2)?.try_into().unwrap()) as usize;
    if arch_len > MAX_ARCH_LEN {
        return Err(format!("handshake arch_len {arch_len} > {MAX_ARCH_LEN}"));
    }
    let arch = std::str::from_utf8(take(2, arch_len)?)
        .map_err(|e| format!("handshake arch not utf-8: {e}"))?
        .to_owned();
    let mut at = 2 + arch_len;
    let geometry_fingerprint: [u8; 32] = take(at, 32)?.try_into().unwrap();
    at += 32;
    let artifact_blake3: [u8; 32] = take(at, 32)?.try_into().unwrap();
    at += 32;
    let lo = u32::from_le_bytes(take(at, 4)?.try_into().unwrap());
    let hi = u32::from_le_bytes(take(at + 4, 4)?.try_into().unwrap());
    let dtype = FrameDtype::from_u8(take(at + 8, 1)?[0])?;
    if b.len() != at + 9 {
        return Err(format!(
            "handshake trailing bytes: {} != {}",
            b.len(),
            at + 9
        ));
    }
    if lo >= hi {
        return Err(format!("handshake range [{lo},{hi}) empty or inverted"));
    }
    Ok(StageHandshake {
        arch,
        geometry_fingerprint,
        artifact_blake3,
        range: LayerRange::new(lo, hi),
        dtype,
    })
}
