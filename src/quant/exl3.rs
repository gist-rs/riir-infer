//! EXL3 (trellis-coded, QTIP-variant) weight format — CPU reference
//! dequantization (Issue 001, T3).
//!
//! Format spec: read at pin `turboderp-org/exllamav3 @
//! 6b84a21b6f1e5da3f291b9e1019061f0de788279` — the full record lives in
//! [`.issues/001_exl3_trellis_format_support.md` §10](../../.issues/001_exl3_trellis_format_support.md);
//! this doc restates only what the code needs. The dequantization contract
//! follows the T2 seam decision (§11 of the same issue): this module is the
//! first SAFETENSORS-side quant (no `GgmlType` coupling), the layer struct is
//! zero-copy (borrows over the mmap, dequantizes at use — never eager `f32`),
//! and the codebook markers are carried as raw `u32` words and VERIFIED, not
//! booleanized.
//!
//! # What a dequantization needs (per linear at key `K`)
//!
//! | entry | dtype | shape | role |
//! |---|---|---|---|
//! | `K.trellis` | int16 | `[in/16, out/16, 16·K]` | packed trellis codes |
//! | `K.suh` | fp16 | `[in]` | per-input-channel scale |
//! | `K.svh` | fp16 | `[out]` | per-output-channel scale |
//! | `K.su` / `K.sv` | int16 | `[in/16]` / `[out/16]` | legacy packed ±1 signs (fallback) |
//! | `K.mul1` | int32 | `[1]` | codebook marker, value `0x83DCD12D` |
//! | `K.mcg` | int32 | `[1]` | codebook marker, value `0xCBAC1FED` |
//! | `K.bias` | fp16 | `[out]` | optional bias (not part of dequant) |
//!
//! Bitrate `K` is self-described: `K = trellis.shape[-1] / 16`, integer or
//! half-integer (`ka + 0.5`; half-integer requires the `mul1` marker).
//!
//! # The trellis ring (per 16×16 tile)
//!
//! A tile is a **tail-biting ring of `256·K` bits** (`2·K` u32 words, read
//! little-endian; stream bit `p` = word `p/32` bit `31 − p%32` — MSB-first).
//! Weight at ring position `p` consumes `D(p)` new bits (`K`, or
//! `ka + bit(p mod 16) of 0xAAAA` for half-integer K); its value is
//! `codebook(W)` where `W` = the 16 ring bits ending at `S(p+1)` (the prefix
//! sum), read MSB-first — wrapping cyclically, so the initial states come
//! from the END of the tile's own ring.
//!
//! ⚠ Ring positions are in **tensor-core element order**, not row-major:
//! for ring pos `p = t·8 + j`, the tile element is
//! `(in_off, out_off) = ((t%4)·2 + (j&1) + 8·((j>>1)&1), (t>>2) + 8·((j>>2)&1))`
//! (the quantizer stores indices in that layout; `tensor_core_perm` in
//! `exl3_lib/quantize.py`, `frac_perm` in `frac.cu`).
//!
//! # Incoherence processing
//!
//! `W = diag(suh) · (I⊗H/√128)_in · W_rot · (I⊗H/√128)_out · diag(svh)`
//! where `W_rot[i][j] = codebook(window of tile (i/16, j/16), ring pos …)`
//! and `H` is the Sylvester-128 Hadamard (recursion `[[H,H],[H,−H]]` from
//! `[[1]]` — exllamav3's `hadamard_1.txt` base, verified at the pin).
//!
//! # Numerics note
//!
//! cb0/cb1 decode in exact fp16 semantics (integer ops + one fp16 add) —
//! bit-portable. cb2's final `__hfma` is emulated as
//! `f32_mul_add_round_to_f16` (product exact in f32; one add rounding; one
//! final f16 rounding) — one rounding fewer than separate fp16 mul+add but
//! one more than a true fused op; pinned for T4's real-pack oracle to
//! adjudicate (Issue 001 §6: numeric-error signal, never text equality).

use half::f16;

use crate::safetensors_loader::TensorMeta;

// ── Codebooks ────────────────────────────────────────────────────────────

/// Marker word stored in the `.mul1` tensor (§10.2; `codebook_mul1_mult`).
pub const EXL3_MUL1_MARKER: u32 = 0x83DC_D12D;
/// Marker word stored in the `.mcg` tensor (§10.2; `codebook_mcg_mult`).
pub const EXL3_MCG_MARKER: u32 = 0xCBAC_1FED;

/// The three procedural codebooks (Issue 001 §10.3) — the enum-dispatch
/// shape lives here, at the level where variance actually exists (§11.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exl3Codebook {
    /// "3inst" — default when no marker is present.
    Cb0,
    /// ".mcg" marker present.
    Cb1Mcg,
    /// ".mul1" marker present (the modern default; required for half-K).
    Cb2Mul1,
}

impl Exl3Codebook {
    /// Decode a 16-bit trellis window to one fp16 value — the exact op
    /// sequence of `codebook.cuh`'s `decode_3inst<cb>`.
    pub fn decode_f16(self, code: u16) -> f16 {
        match self {
            Self::Cb0 => {
                let x = (code as u32)
                    .wrapping_mul(89_226_354)
                    .wrapping_add(64_248_484);
                lop3_hadd(x)
            }
            Self::Cb1Mcg => {
                let x = (code as u32).wrapping_mul(EXL3_MCG_MARKER);
                lop3_hadd(x)
            }
            Self::Cb2Mul1 => {
                let x = (code as u32).wrapping_mul(EXL3_MUL1_MARKER);
                // dp4a(x, 0x01010101, 0x6400): byte sum + acc, u32 arithmetic.
                let sum = (x & 0xFF)
                    + ((x >> 8) & 0xFF)
                    + ((x >> 16) & 0xFF)
                    + ((x >> 24) & 0xFF)
                    + 0x6400;
                let h = f16::from_bits(sum as u16); // 0x6400..0x67FF = 1024.0..2047.0
                let inv = f16::from_bits(0x1EEE); // 1/147.7
                let bias = f16::from_bits(0xC931); // (-1024.0 - 510.0) / 147.7
                // __hfma emulation: compute in f32, round once to f16.
                f16::from_f32(f32::from(h) * f32::from(inv) + f32::from(bias))
            }
        }
    }

    /// Resolve the codebook from the RAW marker words (§11.3 condition 3:
    /// the words are verified, never booleanized). `mul1` wins when both
    /// markers are present (matches the quantizer's `cb = 2 if mul1 else …`).
    pub fn from_markers(mcg: Option<u32>, mul1: Option<u32>) -> Result<Self, Exl3Error> {
        if let Some(w) = mul1 {
            if w != EXL3_MUL1_MARKER {
                return Err(Exl3Error::BadMarker {
                    which: "mul1",
                    word: w,
                    expected: EXL3_MUL1_MARKER,
                });
            }
            return Ok(Self::Cb2Mul1);
        }
        if let Some(w) = mcg {
            if w != EXL3_MCG_MARKER {
                return Err(Exl3Error::BadMarker {
                    which: "mcg",
                    word: w,
                    expected: EXL3_MCG_MARKER,
                });
            }
            return Ok(Self::Cb1Mcg);
        }
        Ok(Self::Cb0)
    }
}

/// `lop3(x, 0x8fff8fff, 0x3b603b60, 0x6a)` then fp16-add the two halves:
/// `(x & m) | c` with `m = 0x8fff_8fff`, `c = 0x3b60_3b60`.
fn lop3_hadd(x: u32) -> f16 {
    let y = (x & 0x8fff_8fff) | 0x3b60_3b60;
    f16::from_bits(y as u16) + f16::from_bits((y >> 16) as u16)
}

// ── Bitrate ──────────────────────────────────────────────────────────────

/// Trellis bitrate `K`: integer (`ka`) or half-integer (`ka + 0.5`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exl3K {
    /// Integer part. For half-integer K this is `ka` (K = ka + 0.5).
    pub ka: u8,
    /// Whether K is half-integer.
    pub half: bool,
}

impl Exl3K {
    /// Derive K from the trellis tensor's last-dim word count (`16·K`).
    /// Integer K ⇒ words divisible by 16; half-integer ⇒ ≡ 8 (mod 16).
    pub fn from_words_per_tile(words: usize) -> Option<Self> {
        if words.is_multiple_of(16) {
            let ka = words / 16;
            (1..=8).contains(&ka).then_some(Self { ka: ka as u8, half: false })
        } else if words % 16 == 8 {
            let ka = (words - 8) / 16;
            (1..=7).contains(&ka).then_some(Self { ka: ka as u8, half: true })
        } else {
            None
        }
    }

    /// Bits consumed by ring position `p` (0-based): `K`, or
    /// `ka + bit(p mod 16) of 0xAAAA` for half-integer K (frac.cu `frac_d`).
    #[inline]
    pub fn bits_for_step(self, p: usize) -> u32 {
        if self.half && ((0xAAAAu32 >> (p & 15)) & 1) == 1 {
            self.ka as u32 + 1
        } else {
            self.ka as u32
        }
    }

    /// Ring length in bits per tile: `256·K` (= `words_per_tile() * 16`).
    #[inline]
    pub fn stream_bits_per_tile(self) -> usize {
        if self.half {
            256 * self.ka as usize + 128
        } else {
            256 * self.ka as usize
        }
    }

    /// u16 words per tile: `16·K` (whole for half-integer K).
    #[inline]
    pub fn words_per_tile(self) -> usize {
        self.stream_bits_per_tile() / 16
    }
}

// ── Ring reader + tensor-core element order ─────────────────────────────

/// Ring position → tile element `(in_off, out_off)` (tensor-core order).
///
/// `p = t·8 + j`: `in_off = (t%4)·2 + (j&1) + 8·((j>>1)&1)`,
/// `out_off = (t>>2) + 8·((j>>2)&1)` (`tensor_core_perm` / `frac_perm`).
#[inline]
pub fn ring_pos_to_tile_element(p: usize) -> (usize, usize) {
    let t = p >> 3;
    let j = p & 7;
    let in_off = ((t & 3) * 2) + (j & 1) + 8 * ((j >> 1) & 1);
    let out_off = (t >> 2) + 8 * ((j >> 2) & 1);
    (in_off, out_off)
}

/// Read stream bit `p` (tile-local) from a trellis tile's bytes.
///
/// Convention (both integer and frac paths, verified at the pin): read each
/// consecutive u16 PAIR as one little-endian u32; stream bit `p` = u32
/// `p/32`'s bit `31 − p%32` (MSB-first). `p` wraps mod `ring_bits`.
#[inline]
fn ring_bit(tile_bytes: &[u8], p: usize, ring_bits: usize) -> u32 {
    let p = p % ring_bits;
    let w = p / 32;
    let word = u32::from_le_bytes([
        tile_bytes[w * 4],
        tile_bytes[w * 4 + 1],
        tile_bytes[w * 4 + 2],
        tile_bytes[w * 4 + 3],
    ]);
    (word >> (31 - (p % 32))) & 1
}

/// The 16-bit window ENDING at ring bit `end` (exclusive), MSB-first:
/// window bit 15 = ring bit `end−16`, window bit 0 = ring bit `end−1`
/// (frac.cu `frac_window`: `start = end − 16`, wraps mod ring).
#[inline]
fn window16(tile_bytes: &[u8], end: usize, ring_bits: usize) -> u16 {
    let start = (end + ring_bits - 16) % ring_bits;
    let mut w: u16 = 0;
    for m in 0..16 {
        w |= (ring_bit(tile_bytes, start + m, ring_bits) as u16) << (15 - m);
    }
    w
}

/// Decode one tile's 256 rotated-basis weights into `dst[..256]` (indexed by
/// ring position; use [`ring_pos_to_tile_element`] for placement).
pub fn decode_tile_rot(dst: &mut [f32], tile_bytes: &[u8], k: Exl3K, cb: Exl3Codebook) {
    debug_assert_eq!(dst.len(), 256);
    let ring_bits = k.stream_bits_per_tile();
    let mut s = 0usize; // prefix sum S(p)
    for (p, slot) in dst.iter_mut().enumerate() {
        s += k.bits_for_step(p) as usize;
        let w = window16(tile_bytes, s, ring_bits);
        *slot = f32::from(cb.decode_f16(w));
    }
}

/// LUT variant of [`decode_tile_rot`] over a memoized codebook table.
fn decode_tile_rot_lut(dst: &mut [f32], tile_bytes: &[u8], k: Exl3K, lut: &[f32; 65536]) {
    debug_assert_eq!(dst.len(), 256);
    let ring_bits = k.stream_bits_per_tile();
    let mut s = 0usize; // prefix sum S(p)
    for (p, slot) in dst.iter_mut().enumerate() {
        s += k.bits_for_step(p) as usize;
        let w = window16(tile_bytes, s, ring_bits);
        *slot = lut[w as usize];
    }
}

/// The per-codebook f32 decode table (65536 entries), memoized from
/// [`Exl3Codebook::decode_f16`] — **bit-identical by construction**: the
/// codebook is a pure function of the 16-bit code, so the table IS the
/// same math, precomputed (T7's CPU fast arm).
pub fn codebook_lut(cb: Exl3Codebook) -> &'static [f32; 65536] {
    use std::sync::OnceLock;
    static CB0: OnceLock<Box<[f32; 65536]>> = OnceLock::new();
    static CB1: OnceLock<Box<[f32; 65536]>> = OnceLock::new();
    static CB2: OnceLock<Box<[f32; 65536]>> = OnceLock::new();
    let lock = match cb {
        Exl3Codebook::Cb0 => &CB0,
        Exl3Codebook::Cb1Mcg => &CB1,
        Exl3Codebook::Cb2Mul1 => &CB2,
    };
    lock.get_or_init(|| {
        let mut t = vec![0.0f32; 65536];
        for (c, v) in t.iter_mut().enumerate() {
            *v = f32::from(cb.decode_f16(c as u16));
        }
        t.into_boxed_slice().try_into().unwrap()
    })
}

// ── Sylvester-128 Hadamard ───────────────────────────────────────────────

/// The scaled Sylvester-128 Hadamard `H/√128`, built once.
///
/// exllamav3's `get_hadamard(128)` recurses `[[H,H],[H,−H]]` from the
/// `hadamard_1.txt` base `[[+]]` (verified at the pin) — pure Sylvester.
pub fn sylvester_hadamard_128() -> &'static [[f32; 128]; 128] {
    use std::sync::OnceLock;
    static H: OnceLock<[[f32; 128]; 128]> = OnceLock::new();
    H.get_or_init(|| {
        let mut h = [[0.0f32; 128]; 128];
        h[0][0] = 1.0;
        // Double H_n → H_2n for source sizes n = 1, 2, … 64 (7 doublings: H1 → H128).
        for n in (0..7).map(|e| 1usize << e) {
            // H_{2n} = [[H, H], [H, -H]] from H_n in h[..n][..n].
            for r in 0..n {
                for c in 0..n {
                    let v = h[r][c];
                    h[r][c + n] = v;
                    h[r + n][c] = v;
                    h[r + n][c + n] = -v;
                }
            }
        }
        let scale = 1.0 / (128.0f32).sqrt();
        for row in &mut h {
            for v in row.iter_mut() {
                *v *= scale;
            }
        }
        h
    })
}

// ── Signs ────────────────────────────────────────────────────────────────

/// Unpack legacy packed ±1 signs: bit `i` of LE u16 word `i/16` → `1 − 2·b`.
pub fn unpack_signs(packed: &[u8]) -> Vec<f16> {
    let n = packed.len() * 8;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let word_idx = i / 16;
        let word = u16::from_le_bytes([packed[word_idx * 2], packed[word_idx * 2 + 1]]);
        let bit = (word >> (i % 16)) & 1;
        out.push(f16::from_f32(1.0 - 2.0 * bit as f32));
    }
    out
}

/// Read a fp16 slice from LE bytes.
fn read_f16_slice(bytes: &[u8]) -> Vec<f16> {
    bytes.as_chunks::<2>().0.iter()
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

// ── The zero-copy layer ─────────────────────────────────────────────────

/// Errors from EXL3 layer construction (all loud; §11.3 condition 3).
#[derive(Debug, thiserror::Error)]
pub enum Exl3Error {
    #[error("EXL3 '{which}' marker word 0x{word:08X} != expected 0x{expected:08X}")]
    BadMarker {
        which: &'static str,
        word: u32,
        expected: u32,
    },
    #[error("EXL3 trellis shape invalid: {0}")]
    BadShape(String),
    #[error("EXL3 dims must be 128-divisible (Hadamard blocks): in={in_dim}, out={out_dim}")]
    Not128Divisible { in_dim: usize, out_dim: usize },
    #[error("EXL3 half-integer K={ka}.5 requires the mul1 codebook marker")]
    HalfKWithoutMul1 { ka: u8 },
    #[error("EXL3 tensor '{name}' wrong byte length: expected {expected}, got {got}")]
    BadTensorLen {
        name: &'static str,
        expected: usize,
        got: usize,
    },
}

/// A zero-copy view over one EXL3 quantized linear layer (§11.3 condition 1).
///
/// All byte slices borrow the mmap'd safetensors shard — nothing is copied
/// or expanded at construction; [`Exl3Layer::dequantize_f32`] materializes
/// dense weights at use. `suh`/`svh` (fp16 channel scales) win over the
/// legacy packed `su`/`sv` signs when both spellings are present (the
/// loader-side precedence of `LinearEXL3.__init__`).
pub struct Exl3Layer<'a> {
    trellis: &'a [u8],
    suh: Option<&'a [u8]>,
    svh: Option<&'a [u8]>,
    su: Option<&'a [u8]>,
    sv: Option<&'a [u8]>,
    /// Raw marker words, carried per §11.3 condition 3.
    pub marker_mcg: Option<u32>,
    pub marker_mul1: Option<u32>,
    pub in_features: usize,
    pub out_features: usize,
    pub k: Exl3K,
    pub codebook: Exl3Codebook,
}

impl<'a> Exl3Layer<'a> {
    /// Build from raw byte slices. Validates dims (128-divisible), trellis
    /// length vs shape, K-from-shape, marker words, and the half-K⇒mul1 rule.
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw_parts(
        trellis: &'a [u8],
        suh: Option<&'a [u8]>,
        svh: Option<&'a [u8]>,
        su: Option<&'a [u8]>,
        sv: Option<&'a [u8]>,
        mcg_word: Option<u32>,
        mul1_word: Option<u32>,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self, Exl3Error> {
        if in_features == 0 || out_features == 0 || !in_features.is_multiple_of(128) || !out_features.is_multiple_of(128) {
            return Err(Exl3Error::Not128Divisible { in_dim: in_features, out_dim: out_features });
        }
        // K from the trellis tile word count. The caller passes the packed
        // trellis bytes; words-per-tile comes from the total length only when
        // the tile grid is known, so the CALLER also drives K via trellis
        // shape — here we derive it from length + dims.
        let tiles = (in_features / 16) * (out_features / 16);
        if !trellis.len().is_multiple_of(tiles * 2) {
            return Err(Exl3Error::BadShape(format!(
                "trellis bytes {} not a whole number of {} tiles",
                trellis.len(),
                tiles
            )));
        }
        let words_per_tile = trellis.len() / (tiles * 2);
        let k = Exl3K::from_words_per_tile(words_per_tile)
            .ok_or_else(|| Exl3Error::BadShape(format!("words-per-tile {words_per_tile} implies K outside 1..8.5")))?;

        let codebook = Exl3Codebook::from_markers(mcg_word, mul1_word)?;
        if k.half && codebook != Exl3Codebook::Cb2Mul1 {
            return Err(Exl3Error::HalfKWithoutMul1 { ka: k.ka });
        }

        if let Some(b) = suh
            && b.len() != in_features * 2
        {
            return Err(Exl3Error::BadTensorLen { name: "suh", expected: in_features * 2, got: b.len() });
        }
        if let Some(b) = svh
            && b.len() != out_features * 2
        {
            return Err(Exl3Error::BadTensorLen { name: "svh", expected: out_features * 2, got: b.len() });
        }
        if let Some(b) = su
            && b.len() != in_features / 16 * 2
        {
            return Err(Exl3Error::BadTensorLen { name: "su", expected: in_features / 16 * 2, got: b.len() });
        }
        if let Some(b) = sv
            && b.len() != out_features / 16 * 2
        {
            return Err(Exl3Error::BadTensorLen { name: "sv", expected: out_features / 16 * 2, got: b.len() });
        }
        if suh.is_none() && su.is_none() {
            return Err(Exl3Error::BadShape("neither suh nor su present".into()));
        }
        if svh.is_none() && sv.is_none() {
            return Err(Exl3Error::BadShape("neither svh nor sv present".into()));
        }

        Ok(Self {
            trellis,
            suh,
            svh,
            su,
            sv,
            marker_mcg: mcg_word,
            marker_mul1: mul1_word,
            in_features,
            out_features,
            k,
            codebook,
        })
    }

    /// Full dequantization: `W[in][out]` row-major (in-major), f32.
    ///
    /// `W = diag(suh) · (I⊗H)·W_rot·(I⊗H) · diag(svh)` with `H = H₁₂₈/√128`
    /// applied per 128-block along each axis. Scalar reference — O(in·out·128).
    /// Uses the memoized codebook LUT (bit-identical to the spec ops).
    pub fn dequantize_f32(&self) -> Vec<f32> {
        let (kin, nout) = (self.in_features, self.out_features);
        let mut w_rot = vec![0.0f32; kin * nout];

        // 1. Trellis decode, placed through the tensor-core element order.
        let lut = codebook_lut(self.codebook);
        let words = self.k.words_per_tile();
        let tile_bytes = words * 2;
        let mut tile = [0.0f32; 256];
        for a in 0..kin / 16 {
            for c in 0..nout / 16 {
                let off = (a * (nout / 16) + c) * tile_bytes;
                decode_tile_rot_lut(&mut tile, &self.trellis[off..off + tile_bytes], self.k, lut);
                for (p, &v) in tile.iter().enumerate() {
                    let (r, c_off) = ring_pos_to_tile_element(p);
                    w_rot[(a * 16 + r) * nout + (c * 16 + c_off)] = v;
                }
            }
        }

        // 2. Incoherence: left block-Hadamard, row scales, right block-
        //    Hadamard, column scales.
        let h = sylvester_hadamard_128();
        let suh: Vec<f32> = match self.suh {
            Some(b) => read_f16_slice(b).into_iter().map(f32::from).collect(),
            None => unpack_signs(self.su.unwrap()).into_iter().map(f32::from).collect(),
        };
        let svh: Vec<f32> = match self.svh {
            Some(b) => read_f16_slice(b).into_iter().map(f32::from).collect(),
            None => unpack_signs(self.sv.unwrap()).into_iter().map(f32::from).collect(),
        };

        // left: y = H·x per (128-block of in, out column)
        let mut tmp = vec![0.0f32; kin * nout];
        for ob in 0..kin / 128 {
            for col in 0..nout {
                for r in 0..128 {
                    let mut acc = 0.0;
                    for kk in 0..128 {
                        acc += h[r][kk] * w_rot[(ob * 128 + kk) * nout + col];
                    }
                    tmp[(ob * 128 + r) * nout + col] = acc;
                }
            }
        }
        // row scales
        for r in 0..kin {
            let s = suh[r];
            for col in 0..nout {
                tmp[r * nout + col] *= s;
            }
        }
        // right: y = x·H per (in row, 128-block of out)
        let mut out = vec![0.0f32; kin * nout];
        for row in 0..kin {
            for cb in 0..nout / 128 {
                for c in 0..128 {
                    let mut acc = 0.0;
                    for kk in 0..128 {
                        acc += tmp[row * nout + cb * 128 + kk] * h[kk][c];
                    }
                    out[row * nout + cb * 128 + c] = acc;
                }
            }
        }
        // column scales
        for col in 0..nout {
            let s = svh[col];
            for r in 0..kin {
                out[r * nout + col] *= s;
            }
        }
        out
    }

    /// Parallel dequantization (rayon over DISJOINT row strips) — the T7
    /// CPU fast arm. **Bit-identical to [`Exl3Layer::dequantize_f32`]**:
    /// every stage parallelizes over row ranges whose outputs are disjoint,
    /// and each output element's accumulation order is unchanged, so the
    /// result equals the scalar reference element-for-element (the test
    /// `parallel_matches_scalar_bit_identical` pins this).
    pub fn dequantize_f32_parallel(&self) -> Vec<f32> {
        use rayon::prelude::*;
        let (kin, nout) = (self.in_features, self.out_features);
        let mut w_rot = vec![0.0f32; kin * nout];

        // 1. Trellis decode — parallel over 16-row in-strips (each strip is
        // written only by its own `a` tiles).
        let lut = codebook_lut(self.codebook);
        let words = self.k.words_per_tile();
        let tile_bytes = words * 2;
        let cols_tiles = nout / 16;
        w_rot
            .par_chunks_mut(nout * 16)
            .enumerate()
            .for_each(|(a, strip)| {
                let mut tile = [0.0f32; 256];
                for c in 0..cols_tiles {
                    let off = (a * cols_tiles + c) * tile_bytes;
                    decode_tile_rot_lut(
                        &mut tile,
                        &self.trellis[off..off + tile_bytes],
                        self.k,
                        lut,
                    );
                    for (p, &v) in tile.iter().enumerate() {
                        let (r, c_off) = ring_pos_to_tile_element(p);
                        strip[r * nout + c * 16 + c_off] = v;
                    }
                }
            });

        // 2. Incoherence: left block-Hadamard (parallel over 128-row
        // blocks), row scales, right block-Hadamard (parallel over rows),
        // column scales — inner loop orders unchanged per element.
        let h = sylvester_hadamard_128();
        let suh: Vec<f32> = match self.suh {
            Some(b) => read_f16_slice(b).into_iter().map(f32::from).collect(),
            None => unpack_signs(self.su.unwrap()).into_iter().map(f32::from).collect(),
        };
        let svh: Vec<f32> = match self.svh {
            Some(b) => read_f16_slice(b).into_iter().map(f32::from).collect(),
            None => unpack_signs(self.sv.unwrap()).into_iter().map(f32::from).collect(),
        };

        let mut tmp = vec![0.0f32; kin * nout];
        tmp.par_chunks_mut(nout * 128)
            .enumerate()
            .for_each(|(ob, block)| {
                for col in 0..nout {
                    for r in 0..128 {
                        let mut acc = 0.0;
                        for kk in 0..128 {
                            acc += h[r][kk] * w_rot[(ob * 128 + kk) * nout + col];
                        }
                        block[r * nout + col] = acc;
                    }
                }
            });
        tmp.par_chunks_mut(nout).enumerate().for_each(|(r, row)| {
            let s = suh[r];
            for v in row.iter_mut() {
                *v *= s;
            }
        });

        let mut out = vec![0.0f32; kin * nout];
        out.par_chunks_mut(nout)
            .enumerate()
            .for_each(|(row_i, row)| {
                let src = &tmp[row_i * nout..(row_i + 1) * nout];
                for cb in 0..nout / 128 {
                    for c in 0..128 {
                        let mut acc = 0.0;
                        for kk in 0..128 {
                            acc += src[cb * 128 + kk] * h[kk][c];
                        }
                        row[cb * 128 + c] = acc;
                    }
                }
            });
        out.par_chunks_mut(nout).for_each(|row| {
            for (col, v) in row.iter_mut().enumerate() {
                *v *= svh[col];
            }
        });
        out
    }
}

// ── Metadata detection (loader-side group scan; §11.3 condition 2) ──────

/// One planned EXL3 layer, as detected from safetensors metadata (no bytes
/// touched — the mmap owner slices later using these names + offsets).
#[derive(Clone, Debug)]
pub struct Exl3LayerPlan {
    pub key: String,
    pub in_features: usize,
    pub out_features: usize,
    pub k: Exl3K,
    pub codebook: Exl3Codebook,
    /// (tensor name, byte offset of its data within its shard).
    pub trellis: (String, usize),
    pub suh: Option<(String, usize)>,
    pub svh: Option<(String, usize)>,
    pub su: Option<(String, usize)>,
    pub sv: Option<(String, usize)>,
    pub bias: Option<(String, usize)>,
}

/// Scan safetensors metadata for EXL3 layer groups (the `has_tensor_group`
/// contract: at least one of sv/svh, one of su/suh, and trellis).
///
/// `pub(crate)` (not `pub`) because it consumes the loader's `pub(crate)`
/// `TensorMeta` — the loader owns the metadata seam (§11.3 cond. 2); the
/// mmap owner slices bytes later using the returned names + offsets.
/// Marker words are read from `marker_words` (name → word) when the
/// 1-element I32 markers' data is available; a present-but-unavailable
/// marker defers to [`Exl3Codebook::from_markers`] at byte-slice time.
/// Layers with wrong dtypes, unreadable shapes, or bad marker words are
/// skipped (byte-slice construction refuses loudly).
///
/// `#[allow(dead_code)]` until T4/T5 wire the safetensors loader to call
/// this — reference-module seam, kept compiled by the feature gate + tests.
#[allow(dead_code)]
pub(crate) fn detect_exl3_layers(
    meta: &std::collections::BTreeMap<String, TensorMeta>,
    marker_words: &std::collections::BTreeMap<String, u32>,
) -> Vec<Exl3LayerPlan> {
    let mut keys: Vec<&String> = meta
        .keys()
        .filter(|k| k.ends_with(".trellis"))
        .collect();
    keys.sort();

    let mut plans = Vec::new();
    for trellis_key in keys {
        let base = &trellis_key[..trellis_key.len() - ".trellis".len()];
        let sub = |suffix: &str| meta.get(&format!("{base}.{suffix}"));
        // dtype spelling per the safetensors spec: trellis I16, scales F16,
        // markers I32. A trellis with a foreign dtype is not EXL3 — skip.
        if sub("trellis").is_none_or(|t| t.dtype != "I16") {
            continue;
        }
        if sub("suh").is_none() && sub("su").is_none() {
            continue;
        }
        if sub("svh").is_none() && sub("sv").is_none() {
            continue;
        }
        let Some(t) = sub("trellis") else { continue };
        let (Some(&in_tiles), Some(&out_tiles), Some(&words)) =
            (t.shape.first(), t.shape.get(1), t.shape.last())
        else {
            continue;
        };

        let in_features = in_tiles * 16;
        let out_features = out_tiles * 16;
        let Some(k) = Exl3K::from_words_per_tile(words) else { continue };

        let mcg_word = sub("mcg").filter(|m| m.dtype == "I32").and_then(|_| marker_words.get(&format!("{base}.mcg")).copied());
        let mul1_word = sub("mul1").filter(|m| m.dtype == "I32").and_then(|_| marker_words.get(&format!("{base}.mul1")).copied());
        let Ok(codebook) = Exl3Codebook::from_markers(mcg_word, mul1_word) else {
            continue; // bad marker word — the layer-slice path will refuse loudly
        };

        let loc = |suffix: &str| sub(suffix).map(|m| (format!("{base}.{suffix}"), m.data_start));
        plans.push(Exl3LayerPlan {
            key: base.to_string(),
            in_features,
            out_features,
            k,
            codebook,
            trellis: (trellis_key.clone(), t.data_start),
            suh: loc("suh"),
            svh: loc("svh"),
            su: loc("su"),
            sv: loc("sv"),
            bias: loc("bias"),
        });
    }
    plans
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent known-answer pins: computed at 2026-09-24 with numpy f16
    // (.raw/exl3_pins.py, uv --with numpy) directly from the §10.3 spec ops.
    const PINS: &[(u16, u16, u16, u16)] = &[
        (0x0000, 0x3FEC, 0x3F60, 0xC2E8),
        (0x0001, 0xBB87, 0x3BEE, 0x3921),
        (0x0002, 0x41B6, 0x3010, 0xB72D),
        (0x00FF, 0x41DC, 0xC196, 0x3AB3),
        (0x1234, 0xC1CE, 0x418F, 0xB89E),
        (0x5555, 0xC379, 0xBBCF, 0x35B6),
        (0xBEEF, 0x436B, 0xC363, 0x3124),
        (0xFFFE, 0x41EA, 0xBFAD, 0x3702),
        (0xFFFF, 0xBAF0, 0x3B4D, 0xB936),
    ];

    #[test]
    fn codebook_pins_match_independent_numpy() {
        for &(code, b0, b1, b2) in PINS {
            assert_eq!(Exl3Codebook::Cb0.decode_f16(code).to_bits(), b0, "cb0(0x{code:04X})");
            assert_eq!(Exl3Codebook::Cb1Mcg.decode_f16(code).to_bits(), b1, "cb1(0x{code:04X})");
            assert_eq!(Exl3Codebook::Cb2Mul1.decode_f16(code).to_bits(), b2, "cb2(0x{code:04X})");
        }
    }

    #[test]
    fn codebooks_finite_and_bounded() {
        // Python census over all 65536 codes: cb0/cb1 ∈ ±3.996094,
        // cb2 ∈ [−3.453125, +3.347656], all finite.
        for cb in [Exl3Codebook::Cb0, Exl3Codebook::Cb1Mcg] {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for code in 0..=u16::MAX {
                let v = f32::from(cb.decode_f16(code));
                assert!(v.is_finite());
                lo = lo.min(v);
                hi = hi.max(v);
            }
            assert!((lo + 3.997).abs() < 0.01 && (hi - 3.997).abs() < 0.01, "{cb:?} range {lo}..{hi}");
        }
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for code in 0..=u16::MAX {
            let v = f32::from(Exl3Codebook::Cb2Mul1.decode_f16(code));
            assert!(v.is_finite());
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!((lo + 3.454).abs() < 0.01 && (hi - 3.348).abs() < 0.01, "cb2 range {lo}..{hi}");
    }

    // ── ring reader ──

    /// Reference test-side encoder: pack codes per the spec (MSB-first u32
    /// stream, tensor-core ring positions), no shared code with the reader.
    fn pack_tile(codes: &[u16; 256], k: Exl3K) -> Vec<u8> {
        let ring_bits = k.stream_bits_per_tile();
        assert_eq!(ring_bits % 32, 0);
        let nw = ring_bits / 32;
        let mut words = vec![0u32; nw];
        let mut pos = 0usize;
        for (i, &code) in codes.iter().enumerate() {
            let d = k.bits_for_step(i) as usize;
            for b in (0..d).rev() {
                if (code >> b) & 1 == 1 {
                    words[pos / 32] |= 1u32 << (31 - (pos % 32));
                }
                pos += 1;
            }
        }
        assert_eq!(pos, ring_bits);
        let mut bytes = Vec::with_capacity(nw * 4);
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn ring_roundtrip_integer_k() {
        // Codes round-trip through the packed ring; the decoded window for
        // weight i equals the last 16 bits of (initial tail || codes ≤ i).
        let k = Exl3K { ka: 4, half: false };
        let mut codes = [0u16; 256];
        let mut rng: u32 = 0x243F_6A88;
        for code in codes.iter_mut() {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *code = (rng >> 16) as u16 & 0xF; // 4-bit codes
        }
        let bytes = pack_tile(&codes, k);
        let ring_bits = k.stream_bits_per_tile();

        // The code at position i must be recoverable as the bits the window
        // gained: window(i) ends at S(i+1); window(i-1) ends at S(i).
        let mut s = 0usize;
        let mut prev_window = 0u16;
        for (i, &code) in codes.iter().enumerate() {
            s += k.bits_for_step(i) as usize;
            let w = window16(&bytes, s, ring_bits);
            let d = k.bits_for_step(i);
            let mask = (1u16 << d) - 1;
            assert_eq!(w & mask, code & mask, "code {i} recoverable from window");
            // The window gains exactly the new code bits at the bottom.
            // (Continuity is undefined at i = 0: that window wraps tail-biting.)
            if i > 0 {
                assert_eq!(w >> d, prev_window & ((1u16 << (16 - d)) - 1), "window continuity at {i}");
            }
            prev_window = w;
        }
    }

    #[test]
    fn ring_wraps_tail_biting() {
        // K=2: the first weight's window starts negative → wraps to the END
        // of the tile's ring (the last 14 bits + the first 2 code bits).
        let k = Exl3K { ka: 2, half: false };
        let mut codes = [0u16; 256];
        codes[0] = 0b10;
        codes[255] = 0b11;
        codes[254] = 0b01;
        codes[253] = 0b00;
        let bytes = pack_tile(&codes, k);
        let ring_bits = k.stream_bits_per_tile();
        let w0 = window16(&bytes, k.bits_for_step(0) as usize, ring_bits);
        // window0 = last 14 ring bits (codes 250..255's bits) then code0's 2 bits.
        let tail = window16(&bytes, ring_bits, ring_bits); // bits [R-16, R)
        let expected = (tail << 2) | (codes[0] & 0b11);
        assert_eq!(w0, expected);
    }

    #[test]
    fn half_k_bits_and_words() {
        let k = Exl3K { ka: 3, half: true };
        assert_eq!(k.words_per_tile(), 16 * 3 + 8);
        assert_eq!(k.stream_bits_per_tile(), 256 * 3 + 128);
        // 0xAAAA: even steps 3 bits, odd steps 4.
        assert_eq!(k.bits_for_step(0), 3);
        assert_eq!(k.bits_for_step(1), 4);
        assert_eq!(k.bits_for_step(16), 3);
        assert_eq!(k.bits_for_step(17), 4);
        // Sum over one period-16 block: 8·3 + 8·4 = 56 bits.
        let block: u32 = (0..16).map(|i| k.bits_for_step(i)).sum();
        assert_eq!(block, 56);
        assert_eq!(Exl3K::from_words_per_tile(56), Some(k));
    }

    #[test]
    fn half_k_roundtrip() {
        let k = Exl3K { ka: 2, half: true }; // K = 2.5
        let mut codes = [0u16; 256];
        let mut rng: u32 = 0x5DEE_CE66;
        for (i, code) in codes.iter_mut().enumerate() {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let d = k.bits_for_step(i);
            *code = (rng >> 16) as u16 & ((1 << d) - 1);
        }
        let bytes = pack_tile(&codes, k);
        let ring_bits = k.stream_bits_per_tile();
        let mut s = 0usize;
        for (i, &code) in codes.iter().enumerate() {
            s += k.bits_for_step(i) as usize;
            let w = window16(&bytes, s, ring_bits);
            let d = k.bits_for_step(i);
            assert_eq!(w & ((1u16 << d) - 1), code & ((1 << d) - 1), "half-K code {i}");
        }
    }

    // ── tensor-core element order ──

    #[test]
    fn ring_perm_is_bijective_and_matches_spec() {
        let mut seen = [false; 256];
        for p in 0..256 {
            let (r, c) = ring_pos_to_tile_element(p);
            assert!(r < 16 && c < 16);
            assert!(!seen[r * 16 + c], "bijection broken at p={p}");
            seen[r * 16 + c] = true;
        }
        // Spot-check against tensor_core_perm / frac_perm formulas:
        // p=0 → t=0,j=0 → (r,c) = (0,0); p=1 → (1,0); p=2 → (8+0? j=2 → r0+(0)+(8), c=0)
        assert_eq!(ring_pos_to_tile_element(0), (0, 0));
        assert_eq!(ring_pos_to_tile_element(1), (1, 0));
        assert_eq!(ring_pos_to_tile_element(2), (8, 0));
        assert_eq!(ring_pos_to_tile_element(3), (9, 0));
        assert_eq!(ring_pos_to_tile_element(4), (0, 8));
        assert_eq!(ring_pos_to_tile_element(8), (2, 0)); // t=1,j=0 → r0=2,c0=0
        assert_eq!(ring_pos_to_tile_element(9), (3, 0));
    }

    // ── Hadamard ──

    #[test]
    fn sylvester_hadamard_128_properties() {
        let h = sylvester_hadamard_128();
        // H·Hᵀ = I (H/√128 is its own inverse); symmetry; ±1/√128 entries.
        let inv_sqrt = 1.0 / (128.0f32).sqrt();
        for (r, row) in h.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                assert!((v.abs() - inv_sqrt).abs() < 1e-6, "entry {r},{c} = {v}");
                assert!((v - h[c][r]).abs() < 1e-9, "symmetry at {r},{c}");
            }
        }
        for (r, row) in h.iter().enumerate() {
            for (c, _) in h.iter().enumerate() {
                let mut acc = 0.0;
                for k in 0..128 {
                    acc += row[k] * h[c][k];
                }
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((acc - want).abs() < 1e-4, "H·Hᵀ at {r},{c} = {acc}");
            }
        }
        // First row all + (Sylvester convention).
        assert!(h[0].iter().all(|&v| v > 0.0));
    }

    // ── signs ──

    #[test]
    fn sign_unpack_matches_bit_order() {
        // Word 0 = 0b0000000000000001 → channel 0 = −1, channels 1..15 = +1.
        let packed = [0x01u8, 0x00];
        let s = unpack_signs(&packed);
        assert_eq!(s.len(), 16);
        assert_eq!(f32::from(s[0]), -1.0);
        assert!(s[1..].iter().all(|&v| f32::from(v) == 1.0));
        // Word 0 = 0xFFFF → all −1.
        let packed = [0xFFu8, 0xFF];
        assert!(unpack_signs(&packed).iter().all(|&v| f32::from(v) == -1.0));
    }

    // ── layer construction + full dequant ──

    /// Build a synthetic single-tile-pair layer: in=128, out=128 (8×8 tiles),
    /// zero trellis (all codes 0), unit scales → W = `H·W_rot·H`.
    fn unit_scales(n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n * 2);
        for _ in 0..n {
            v.extend_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        }
        v
    }

    #[test]
    fn layer_validates_dims_markers_and_k() {
        let k = Exl3K { ka: 2, half: false };
        let tiles = (128 / 16) * (128 / 16);
        let trellis = vec![0u8; tiles * k.words_per_tile() * 2];
        let suh = unit_scales(128);
        let svh = unit_scales(128);

        // Good: no markers → Cb0.
        let layer = Exl3Layer::from_raw_parts(&trellis, Some(&suh), Some(&svh), None, None, None, None, 128, 128).unwrap();
        assert_eq!(layer.codebook, Exl3Codebook::Cb0);
        assert_eq!(layer.k, k);

        // Bad marker word → loud refusal (§11.3 condition 3).
        let err = Exl3Layer::from_raw_parts(&trellis, Some(&suh), Some(&svh), None, None, Some(0xDEAD_BEEF), None, 128, 128);
        assert!(matches!(err, Err(Exl3Error::BadMarker { .. })));

        // Half-K without mul1 → loud refusal.
        let kh = Exl3K { ka: 2, half: true };
        let trellis_h = vec![0u8; tiles * kh.words_per_tile() * 2];
        let err = Exl3Layer::from_raw_parts(&trellis_h, Some(&suh), Some(&svh), None, None, None, None, 128, 128);
        assert!(matches!(err, Err(Exl3Error::HalfKWithoutMul1 { .. })));
        // With mul1 → ok.
        let ok = Exl3Layer::from_raw_parts(&trellis_h, Some(&suh), Some(&svh), None, None, None, Some(EXL3_MUL1_MARKER), 128, 128).unwrap();
        assert_eq!(ok.codebook, Exl3Codebook::Cb2Mul1);

        // Non-128-divisible dims → refusal.
        let bad = Exl3Layer::from_raw_parts(&trellis[..], Some(&suh), Some(&svh), None, None, None, None, 112, 128);
        assert!(matches!(bad, Err(Exl3Error::Not128Divisible { .. })));
    }

    #[test]
    fn parallel_matches_scalar_bit_identical() {
        // The T7 CPU fast arm's contract: rayon over disjoint row strips
        // preserves every element's accumulation order → the parallel output
        // equals the scalar reference ELEMENT-FOR-ELEMENT (no tolerance).
        // Covers both scale spellings (suh/svh and legacy su/sv) and two K.
        let mk_rng_bytes = |n: usize, seed: u32| -> Vec<u8> {
            let mut r = seed;
            (0..n)
                .map(|_| {
                    r = r.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (r >> 24) as u8
                })
                .collect()
        };
        // Finite random f16 scales (raw random bytes would carry NaN/Inf
        // bit patterns — legal in the fixture but useless for parity since
        // NaN != NaN; bit comparison below covers that class anyway).
        let mk_f16 = |n: usize, seed: u32| -> Vec<u8> {
            let mut r = seed;
            (0..n)
                .flat_map(|i| {
                    r = r.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    let s = 0.25 + (i % 7) as f32 * 0.125 + ((r >> 28) as f32) * 0.01;
                    f16::from_f32(s).to_bits().to_le_bytes()
                })
                .collect()
        };
        for &(kin, nout, ka, half) in &[(256usize, 128usize, 4u8, false), (128, 384, 2, true)] {
            let k = Exl3K { ka, half };
            let tiles = (kin / 16) * (nout / 16);
            let trellis = mk_rng_bytes(tiles * k.words_per_tile() * 2, 0xBEEF + ka as u32);
            let suh = mk_f16(kin, 7);
            let svh = mk_f16(nout, 91);
            let layer = Exl3Layer::from_raw_parts(
                &trellis,
                Some(&suh),
                Some(&svh),
                None,
                None,
                None,
                Some(EXL3_MUL1_MARKER), // half-K needs mul1
                kin,
                nout,
            )
            .unwrap();
            let a = layer.dequantize_f32();
            let b = layer.dequantize_f32_parallel();
            // Bit-pattern comparison — TRUE bit identity, NaN-payload aware.
            let bits_eq = a
                .iter()
                .zip(&b)
                .all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(bits_eq, "kin={kin} nout={nout} parallel/scalar bit parity broke");
        }
        // Legacy sign spelling exercises the unpack_signs path in both arms.
        let k = Exl3K { ka: 3, half: false };
        let tiles = (128 / 16) * (256 / 16);
        let trellis = mk_rng_bytes(tiles * k.words_per_tile() * 2, 0xC0FFEE);
        let layer = Exl3Layer::from_raw_parts(
            &trellis,
            None,
            None,
            Some(&[0u8; 16]),
            Some(&[0u8; 32]),
            None,
            None,
            128,
            256,
        )
        .unwrap();
        let a = layer.dequantize_f32();
        let b = layer.dequantize_f32_parallel();
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "legacy-sign parallel/scalar bit parity broke"
        );
    }

    #[test]
    fn dequant_matches_reference_composition() {
        // Cross-check the full dequant against a naive in-test composition:
        // every step rebuilt independently (tiles via decode_tile_rot, hads
        // via explicit matrices) and compared elementwise.
        let k = Exl3K { ka: 4, half: false };
        let (kin, nout) = (128, 128);
        let tiles = (kin / 16) * (nout / 16);
        // Random trellis bytes.
        let mut rng: u32 = 0x1234_ABCD;
        let mut trellis = Vec::with_capacity(tiles * k.words_per_tile() * 2);
        for _ in 0..trellis.capacity() {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            trellis.push((rng >> 24) as u8);
        }
        // Random-ish scales: alternating small magnitudes.
        let mk = |n: usize, seed: u32| -> Vec<u8> {
            let mut r = seed;
            let mut v = Vec::with_capacity(n * 2);
            for i in 0..n {
                r = r.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let s = if i % 2 == 0 { 1.0 } else { 0.5 } + ((r >> 28) as f32) * 0.001;
                v.extend_from_slice(&f16::from_f32(s).to_bits().to_le_bytes());
            }
            v
        };
        let suh = mk(kin, 7);
        let svh = mk(nout, 91);

        let layer = Exl3Layer::from_raw_parts(&trellis, Some(&suh), Some(&svh), None, None, None, None, kin, nout).unwrap();
        let got = layer.dequantize_f32();

        // Reference: independent composition.
        let h = sylvester_hadamard_128();
        let mut w_rot = vec![0.0f32; kin * nout];
        let tile_bytes = k.words_per_tile() * 2;
        let mut tile = [0.0f32; 256];
        for a in 0..kin / 16 {
            for c in 0..nout / 16 {
                let off = (a * (nout / 16) + c) * tile_bytes;
                decode_tile_rot(&mut tile, &trellis[off..off + tile_bytes], k, Exl3Codebook::Cb0);
                for (p, &v) in tile.iter().enumerate() {
                    let (r, co) = ring_pos_to_tile_element(p);
                    w_rot[(a * 16 + r) * nout + c * 16 + co] = v;
                }
            }
        }
        let su: Vec<f32> = read_f16_slice(&suh).into_iter().map(f32::from).collect();
        let sv: Vec<f32> = read_f16_slice(&svh).into_iter().map(f32::from).collect();
        // Matrix-scale-aware comparison: the two compositions accumulate the
        // same 16384-term products in different orders, so near-cancellation
        // entries (~1e-4) differ by pure rounding (~3e-7). Compare against the
        // matrix's max magnitude, not per-entry relative error.
        let mut want_mat = vec![0.0f32; kin * nout];
        let mut max_abs = 0.0f32;
        for i in 0..kin {
            for j in 0..nout {
                let mut acc = 0.0f32;
                for kk in 0..128 {
                    for ll in 0..128 {
                        acc += h[i % 128][kk] * w_rot[kk * nout + ll] * h[ll][j % 128];
                    }
                }
                let want = su[i] * acc * sv[j];
                max_abs = max_abs.max(want.abs());
                want_mat[i * nout + j] = want;
            }
        }
        let tol = 1e-4 * max_abs;
        for idx in 0..kin * nout {
            let (g, w) = (got[idx], want_mat[idx]);
            assert!(
                (g - w).abs() <= tol,
                "W[{:3}][{:3}]: got {g}, want {w} (|Δ| > {tol})",
                idx / nout,
                idx % nout
            );
        }
    }

    // ── real-pack oracle (Issue 001 T4a) ─────────────────────────────

    /// Compare the Rust dequant against the independent numpy oracle over
    /// the REAL async0x42/Qwen3-8B-exl3_4.0bpw `layers.0.self_attn.k_proj`
    /// bytes (in=4096, out=1024, K=4, cb0, suh+svh). Requires the range-
    /// fetched fixture at `/tmp/exl3-pack/` + the numpy reference (see
    /// `.raw/exl3_layer_oracle.py`); skips loudly when absent.
    ///
    /// Bar: relative F-norm error < 1e-4 and max-abs < 1e-3·max|W| — the two
    /// implementations accumulate identical math in different orders, so
    /// rounding is ~1e-6 while any layout bug (bit order, permutation,
    /// codebook) lands at O(1).
    #[test]
    #[ignore = "requires /tmp/exl3-pack fixture (Issue 001 T4a; .raw/exl3_layer_oracle.py)"]
    fn real_pack_oracle_k_proj() {
        let dir = std::env::temp_dir().join("exl3-pack");
        let trellis = std::fs::read(dir.join("k_proj.trellis.bin"));
        let (Ok(trellis), Ok(scales)) = (trellis, std::fs::read(dir.join("k_proj.scales.bin")))
        else {
            eprintln!("SKIP: fixture absent — fetch per Issue 001 §12.5");
            return;
        };
        let Some(ref_w) = read_npy_f32(&std::fs::read(dir.join("k_proj_ref_w.npy")).unwrap()) else { eprintln!("SKIP: numpy reference absent");
                return; };

        let (in_f, out_f) = (4096usize, 1024usize);
        assert_eq!(ref_w.len(), in_f * out_f);
        let (suh, svh) = scales.split_at(in_f * 2);
        let layer = Exl3Layer::from_raw_parts(
            &trellis, Some(suh), Some(svh), None, None, None, None, in_f, out_f,
        )
        .expect("real pack layer validates");
        assert_eq!(layer.codebook, Exl3Codebook::Cb0, "this pack predates markers");
        assert_eq!(layer.k, Exl3K { ka: 4, half: false });
        let got = layer.dequantize_f32();

        let mut dot = 0.0f64;
        let (mut n_g, mut n_r, mut max_abs, mut max_w) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (g, r) in got.iter().zip(ref_w.iter()) {
            dot += (*g as f64) * (*r as f64);
            n_g += (*g as f64) * (*g as f64);
            n_r += (*r as f64) * (*r as f64);
            max_abs = max_abs.max((g - r).abs() as f64);
            max_w = max_w.max(r.abs() as f64);
        }
        let rel_frobenius = 1.0 - dot / (n_g.sqrt() * n_r.sqrt());
        eprintln!(
            "T4a k_proj: rel-Frobenius={rel_frobenius:.3e} max_abs_err={max_abs:.3e} (max|W|={max_w:.3e})"
        );
        assert!(rel_frobenius < 1e-4, "relative Frobenius error {rel_frobenius}");
        assert!(max_abs < 1e-3 * max_w, "max abs error {max_abs} vs max|W| {max_w}");
    }

    /// Minimal numpy .npy reader for a flat f32 array (header dict + LE data).
    fn read_npy_f32(bytes: &[u8]) -> Option<Vec<f32>> {
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
            return None;
        }
        let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let data = &bytes[10 + hlen..];
        Some(
            data.as_chunks::<4>().0.iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }
}
