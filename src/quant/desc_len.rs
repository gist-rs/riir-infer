//! Description-length (MDL floor) audit for shipped quant formats — Issue 040.
//!
//! For every discretized weight group in a GGUF checkpoint, compute the
//! two-part MDL floor per arXiv:2509.22445 §B.9.3 (ICLR 2026): the optimal
//! prior over a quantization group's stored symbols is delta masses at the
//! distinct stored values with the empirical frequencies as mixing weights,
//! so the optimal code cost is exactly the histogram entropy
//!
//! ```text
//! floor_bits = N · H(p)      p = empirical histogram of stored symbols
//! slack      = stored_bpw − H(p)   [bits/weight]
//! ```
//!
//! **Two-part honesty (the load-bearing column split):** `N·H(p)` prices the
//! weight VALUES only. The decoder also pays for the side information — the
//! per-block scales (and a stored codebook where the format carries one;
//! none in this GGUF set — every alphabet here is fixed by its format
//! spec). The audit therefore reports `N·H(p)` and
//! `N·H(p) + side_info_bits` as SEPARATE columns from stored bits, and the
//! headline slack is quoted against the side-info-INCLUSIVE floor
//! ([`TensorAudit::honest_slack_bpw`]). Slack quoted without the side-info
//! term overstates headroom for exactly the block formats this repo ships
//! (0.125 bpw of f16 scale on q2_0/PQ2_0, 0.4375–0.625 on the k-quants).
//! Disclosed convention: the frequency table's own description cost is not
//! re-priced (the standard adaptive-coder assumption behind the B.9.3
//! closed form — the decoder learns p from the same counts); scale streams
//! are themselves symbol streams (4/6-bit nibbles) and remain a re-pricing
//! opportunity for a future round.
//!
//! **Prior art, acknowledged:** weight-histogram entropy as an MDL /
//! compressibility probe is published — EntroLLM (arXiv:2505.02380) and Deep
//! Compression (Han et al., 2016). This is an APPLIED audit for our shipped
//! formats, not a primitive claim (Issue 040 header; R011 verdict).
//!
//! **Scope law:** consumes stored symbols ONLY — no dequantization on the
//! audit path (trit/code extraction is symbol recovery, not value
//! reconstruction). f16/f32 and other non-discrete containers are reported
//! as ADVISORY rows, never as floors: their bit histogram is codec-dependent.
//!
//! **Coverage (GGUF):**
//!
//! | format | alphabet K | block | side-info/block | auditable |
//! |---|---|---|---|---|
//! | Q2_0 (PQ2_0) | 4 (2-bit codes) | 128 w / 34 B | 16 (f16 d) | yes |
//! | PTQ1_0 | 3 (trits) | 128 w / 28 B | 16 (f16 d) | yes |
//! | Q2_K | 4 | 256 w / 84 B | 160 | yes |
//! | Q3_K | 8 (2-bit + sign) | 256 w / 110 B | 112 | yes |
//! | Q4_K | 16 | 256 w / 144 B | 128 | yes |
//! | Q5_K | 32 | 256 w / 176 B | 128 | yes |
//! | Q6_K | 64 | 256 w / 210 B | 144 | yes |
//! | Q4_0 / Q4_1 | 16 | 32 w / 18–20 B | 16 / 32 | yes |
//! | Q8_0 / Q8_1 | 256 (i8) | 32 w / 34–36 B | 16 / 32 | yes |
//! | Q5_0 / Q5_1 | — | — | — | advisory: no verified in-repo dequant to pair the bit transcription with |
//! | Q8_K | — | — | — | advisory: block layout not shipped in this loader |
//! | F16/F32/BF16/F64, I8–I64 | — | — | — | advisory: no discrete alphabet |
//!
//! Symbol extraction reuses the module-local verified block layouts
//! (`BlockQ2K` et al., bytemuck-cast over the mmap slice — the same
//! cast-soundness note as `q2k.rs`) and `ptq1_0_element_trit` for the
//! base-3 trit stream. EXL3 (trellis) audit is the named follow-up: it
//! needs the safetensors-side reader (Issue 040 scope note).
//!
//! Not a codec: no re-encoding ships here. The G1 gate (integration test
//! `desc_len_g1_entropy_coder`) bit-matches the floor against a reference
//! rANS coder on synthetic histograms and sampled real groups.

use crate::gguf_loader::{GgmlType, GgufFile};
use crate::quant::ptq1_0::{BlockPtq1_0, PTQ1_0_BLOCK_SIZE, ptq1_0_element_trit};
use crate::quant::q2_0::BlockQ2_0;
use crate::quant::q2k::BlockQ2K;
use crate::quant::q3k::BlockQ3K;
use crate::quant::q4k::BlockQ4K;
use crate::quant::q5k::BlockQ5K;
use crate::quant::q6k::BlockQ6K;
use crate::quant::q8kv::BlockQ8_0;

/// Largest alphabet in the audited set: Q8's 256 raw byte symbols.
pub const MAX_ALPHABET: usize = 256;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from the description-length scan.
#[derive(Debug, thiserror::Error)]
pub enum DescLenError {
    #[error("tensor byte length {len} is not a multiple of the {ty} block size {block_bytes}")]
    BadLength {
        len: usize,
        block_bytes: usize,
        ty: &'static str,
    },
    #[error("tensor data slice misaligned for the {ty} block cast (GGUF data_start must be even)")]
    Misaligned { ty: &'static str },
    #[error("format {ty:?} is advisory-only (no discrete alphabet audit): {reason}")]
    Advisory { ty: GgmlType, reason: &'static str },
}

// ── Fixed-alphabet streaming counter (G4) ───────────────────────

/// Fixed-size streaming symbol counter: one `[u64; 256]` table, zero
/// allocation in the scan loop, `clear()`-reused across tensors (the
/// `lut_grid::WeightHistogram` streaming-counter pattern, specialized to a
/// discrete alphabet — the continuous-binned variant there serves the
/// Lloyd-Max solver and is deliberately not forked).
#[derive(Clone, Debug)]
pub struct SymbolCounts {
    counts: [u64; MAX_ALPHABET],
    total: u64,
}

impl SymbolCounts {
    /// Empty counter over the fixed 256-symbol table (2 KiB).
    pub const fn new() -> Self {
        Self {
            counts: [0; MAX_ALPHABET],
            total: 0,
        }
    }

    /// Reset to empty, reusing the allocation.
    pub fn clear(&mut self) {
        self.counts = [0; MAX_ALPHABET];
        self.total = 0;
    }

    /// Record one symbol (`< MAX_ALPHABET`).
    #[inline]
    pub fn record(&mut self, symbol: usize) {
        debug_assert!(symbol < MAX_ALPHABET, "symbol {symbol} out of alphabet");
        self.counts[symbol] += 1;
        self.total += 1;
    }

    /// Symbols recorded so far.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Distinct symbols observed.
    pub fn distinct(&self) -> u64 {
        self.counts.iter().filter(|&&c| c > 0).count() as u64
    }

    /// Read-only counts view (ascending symbol order, deterministic).
    pub fn counts(&self) -> &[u64; MAX_ALPHABET] {
        &self.counts
    }

    /// Shannon entropy of the recorded distribution, in bits/symbol.
    ///
    /// Deterministic: ascending symbol index, f64 accumulation. Empty
    /// counter → `0.0` (callers gate on `total()` before quoting a floor).
    pub fn entropy_bits(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let n = self.total as f64;
        let mut h = 0.0f64;
        for &c in &self.counts {
            if c == 0 {
                continue;
            }
            let p = c as f64 / n;
            h -= p * p.log2();
        }
        h
    }
}

impl Default for SymbolCounts {
    fn default() -> Self {
        Self::new()
    }
}

// ── Format specs ────────────────────────────────────────────────

/// Per-format audit geometry. The layout self-consistency law
/// (`value_bytes·8 + side_info_bits == block_bytes·8`) is unit-tested for
/// every auditable spec — a layout drift reds at test time, not on a real
/// file.
#[derive(Clone, Copy, Debug)]
pub struct FormatSpec {
    pub ty: GgmlType,
    pub name: &'static str,
    pub block_bytes: usize,
    pub block_weights: usize,
    /// Alphabet size of the stored symbol stream (0 for advisory formats).
    pub alphabet_k: usize,
    /// Bits of side information per block (scales; stored codebooks would
    /// ride here too — none in this format set).
    pub side_info_bits_per_block: u64,
    /// Bytes of value payload per block (the nominal symbol storage).
    pub value_bytes_per_block: usize,
    /// Set → the format gets an advisory row, never a floor.
    pub advisory: Option<&'static str>,
}

impl FormatSpec {
    /// Stored bits per block (the file's actual cost).
    pub const fn stored_bits_per_block(&self) -> u64 {
        (self.block_bytes * 8) as u64
    }

    /// Nominal value-payload bits per block.
    pub const fn value_bits_per_block(&self) -> u64 {
        (self.value_bytes_per_block * 8) as u64
    }

    /// The audit spec for a GGML type.
    pub fn of(ty: GgmlType) -> Self {
        match ty {
            GgmlType::Q2_0 => Self::auditable(ty, "Q2_0", 34, 128, 4, 16, 32),
            GgmlType::PTQ1_0 => Self::auditable(ty, "PTQ1_0", 28, 128, 3, 16, 26),
            GgmlType::Q2_K => Self::auditable(ty, "Q2_K", 84, 256, 4, 160, 64),
            GgmlType::Q3_K => Self::auditable(ty, "Q3_K", 110, 256, 8, 112, 96),
            GgmlType::Q4_K => Self::auditable(ty, "Q4_K", 144, 256, 16, 128, 128),
            GgmlType::Q5_K => Self::auditable(ty, "Q5_K", 176, 256, 32, 128, 160),
            GgmlType::Q6_K => Self::auditable(ty, "Q6_K", 210, 256, 64, 144, 192),
            GgmlType::Q4_0 => Self::auditable(ty, "Q4_0", 18, 32, 16, 16, 16),
            GgmlType::Q4_1 => Self::auditable(ty, "Q4_1", 20, 32, 16, 32, 16),
            GgmlType::Q8_0 => Self::auditable(ty, "Q8_0", 34, 32, 256, 16, 32),
            GgmlType::Q8_1 => Self::auditable(ty, "Q8_1", 36, 32, 256, 32, 32),
            GgmlType::Q5_0 => Self::advisory_spec(
                ty,
                "Q5_0",
                22,
                32,
                "q5_0 bit transcription has no verified in-repo dequant to pair with",
            ),
            GgmlType::Q5_1 => Self::advisory_spec(
                ty,
                "Q5_1",
                24,
                32,
                "q5_1 bit transcription has no verified in-repo dequant to pair with",
            ),
            GgmlType::Q8_K => Self::advisory_spec(
                ty,
                "Q8_K",
                0,
                0,
                "Q8_K block layout not shipped in this loader",
            ),
            GgmlType::F16 | GgmlType::BF16 => Self::advisory_spec(
                ty,
                "F16/BF16",
                2,
                1,
                "floating container — bit histogram is codec-dependent; no floor",
            ),
            GgmlType::F32 => Self::advisory_spec(
                ty,
                "F32",
                4,
                1,
                "floating container — bit histogram is codec-dependent; no floor",
            ),
            GgmlType::F64 => Self::advisory_spec(
                ty,
                "F64",
                8,
                1,
                "floating container — bit histogram is codec-dependent; no floor",
            ),
            GgmlType::I8 => {
                Self::advisory_spec(ty, "I8", 1, 1, "raw integer container; no quant alphabet")
            }
            GgmlType::I16 => {
                Self::advisory_spec(ty, "I16", 2, 1, "raw integer container; no quant alphabet")
            }
            GgmlType::I32 => {
                Self::advisory_spec(ty, "I32", 4, 1, "raw integer container; no quant alphabet")
            }
            GgmlType::I64 => {
                Self::advisory_spec(ty, "I64", 8, 1, "raw integer container; no quant alphabet")
            }
        }
    }

    const fn auditable(
        ty: GgmlType,
        name: &'static str,
        block_bytes: usize,
        block_weights: usize,
        alphabet_k: usize,
        side_info_bits_per_block: u64,
        value_bytes_per_block: usize,
    ) -> Self {
        Self {
            ty,
            name,
            block_bytes,
            block_weights,
            alphabet_k,
            side_info_bits_per_block,
            value_bytes_per_block,
            advisory: None,
        }
    }

    const fn advisory_spec(
        ty: GgmlType,
        name: &'static str,
        block_bytes: usize,
        block_weights: usize,
        reason: &'static str,
    ) -> Self {
        Self {
            ty,
            name,
            block_bytes,
            block_weights,
            alphabet_k: 0,
            side_info_bits_per_block: 0,
            value_bytes_per_block: 0,
            advisory: Some(reason),
        }
    }
}

// ── Scanners ────────────────────────────────────────────────────

/// Scan metadata: complete blocks processed + trailing bytes (normally 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanInfo {
    pub blocks: usize,
    pub remainder_bytes: usize,
}

/// Scan one tensor's stored symbols into `sink`, in file order.
///
/// Zero-allocation: the sink is monomorphized, counters are caller-owned.
/// Block boundaries and bit layouts follow the module-local verified
/// dequant transcriptions (cited inline per format).
pub fn scan_tensor<S: FnMut(usize)>(
    bytes: &[u8],
    ty: GgmlType,
    sink: &mut S,
) -> Result<ScanInfo, DescLenError> {
    let spec = FormatSpec::of(ty);
    if let Some(reason) = spec.advisory {
        return Err(DescLenError::Advisory { ty, reason });
    }
    if !bytes.len().is_multiple_of(spec.block_bytes) {
        return Err(DescLenError::BadLength {
            len: bytes.len(),
            block_bytes: spec.block_bytes,
            ty: spec.name,
        });
    }
    let blocks = bytes.len() / spec.block_bytes;
    match ty {
        GgmlType::Q2_0 => {
            let blks: &[BlockQ2_0] = cast_blocks(bytes, spec.name)?;
            for b in blks {
                for &byte in &b.qs {
                    for k in 0..4 {
                        sink(((byte >> (2 * k)) & 3) as usize);
                    }
                }
            }
        }
        GgmlType::PTQ1_0 => {
            let blks: &[BlockPtq1_0] = cast_blocks(bytes, spec.name)?;
            for b in blks {
                for j in 0..PTQ1_0_BLOCK_SIZE {
                    // trit ∈ {−1,0,+1} → symbol ∈ {0,1,2}
                    sink((ptq1_0_element_trit(b, j) + 1) as usize);
                }
            }
        }
        GgmlType::Q2_K => {
            let blks: &[BlockQ2K] = cast_blocks(bytes, spec.name)?;
            for b in blks {
                for &byte in &b.qs {
                    for k in 0..4 {
                        sink(((byte >> (2 * k)) & 3) as usize);
                    }
                }
            }
        }
        GgmlType::Q3_K => {
            let blks: &[BlockQ3K] = cast_blocks(bytes, spec.name)?;
            // Element map per `q3k::dequantize_row_q3_k` (the verified
            // transcription): q = 2-bit code, the hmask bit folds the sign.
            for b in blks {
                for sb16 in 0..16usize {
                    let half = sb16 >> 3;
                    let sb = sb16 & 1;
                    let j = (sb16 & 7) >> 1;
                    let shift = 2 * j;
                    let qbase = half * 32 + sb * 16;
                    let hbase = sb * 16;
                    let hbit = (half * 4 + j) as u8;
                    for l in 0..16usize {
                        let q = ((b.qs[qbase + l] >> shift) & 3) as usize;
                        let sign = ((b.hmask[hbase + l] >> hbit) & 1) as usize;
                        sink(q | (sign << 2));
                    }
                }
            }
        }
        GgmlType::Q4_K => {
            let blks: &[BlockQ4K] = cast_blocks(bytes, spec.name)?;
            for b in blks {
                for &byte in &b.qs {
                    sink((byte & 0xF) as usize);
                    sink((byte >> 4) as usize);
                }
            }
        }
        GgmlType::Q5_K => {
            let blks: &[BlockQ5K] = cast_blocks(bytes, spec.name)?;
            // Per `q5k::dequantize_row_q5_k`: 4 chunks of 64, the qh
            // high-bit mask advances 2 bits per chunk; ql advances 32 B.
            for b in blks {
                for g in 0..4usize {
                    let u1 = 1u8 << (2 * g);
                    let u2 = u1 << 1;
                    let ql_base = g * 32;
                    for l in 0..32usize {
                        let lo = (b.qs[ql_base + l] & 0x0F) as usize;
                        let hi1 = ((b.qh[l] & u1) != 0) as usize;
                        sink(lo | (hi1 << 4));
                        let hi_nib = (b.qs[ql_base + l] >> 4) as usize;
                        let hi2 = ((b.qh[l] & u2) != 0) as usize;
                        sink(hi_nib | (hi2 << 4));
                    }
                }
            }
        }
        GgmlType::Q6_K => {
            let blks: &[BlockQ6K] = cast_blocks(bytes, spec.name)?;
            // Per `q6k::dequantize_row_q6_k`: two 128-halves advancing
            // ql/qh by 64/32; per half, q1..q4 use ql low/high nibbles +
            // qh 2-bit fields at shifts 0/2/4/6.
            for b in blks {
                let mut ql_off = 0usize;
                let mut qh_off = 0usize;
                for _half in 0..2 {
                    for l in 0..32usize {
                        let ql0 = b.ql[ql_off + l];
                        let ql1 = b.ql[ql_off + l + 32];
                        let qhb = b.qh[qh_off + l];
                        sink(((ql0 & 0x0F) | ((qhb & 3) << 4)) as usize);
                        sink(((ql1 & 0x0F) | (((qhb >> 2) & 3) << 4)) as usize);
                        sink((((ql0 >> 4) & 0x0F) | (((qhb >> 4) & 3) << 4)) as usize);
                        sink((((ql1 >> 4) & 0x0F) | (((qhb >> 6) & 3) << 4)) as usize);
                    }
                    ql_off += 64;
                    qh_off += 32;
                }
            }
        }
        GgmlType::Q8_0 => {
            let blks: &[BlockQ8_0] = cast_blocks(bytes, spec.name)?;
            for b in blks {
                for &q in &b.qs {
                    sink(q as u8 as usize);
                }
            }
        }
        // Legacy formats: layouts match the loader's own verified dequants
        // (`gguf_loader::dequantize_row_q4_0`/`_q4_1` — nibble streams).
        GgmlType::Q4_0 => scan_legacy_nibbles::<18, 2, 16, _>(bytes, sink),
        GgmlType::Q4_1 => scan_legacy_nibbles::<20, 4, 16, _>(bytes, sink),
        GgmlType::Q8_1 => {
            // block_q8_1: d:f16 @0, s:f16 @2, qs:i8[32] @4 (ggml-common.h).
            for blk in bytes.as_chunks::<36>().0 {
                for &b in blk[4..36].iter() {
                    sink(b as usize);
                }
            }
        }
        _ => unreachable!("advisory formats are rejected above"),
    }
    Ok(ScanInfo {
        blocks,
        remainder_bytes: 0,
    })
}

/// Nibble stream over byte-offset legacy blocks (const-generic block
/// geometry: `qs` at `QS_OFF`, `QS_BYTES` long per `BLOCK`-byte block;
/// low nibble first).
#[inline]
fn scan_legacy_nibbles<
    const BLOCK: usize,
    const QS_OFF: usize,
    const QS_BYTES: usize,
    S: FnMut(usize),
>(
    bytes: &[u8],
    sink: &mut S,
) {
    for blk in bytes.as_chunks::<BLOCK>().0 {
        for &b in blk[QS_OFF..QS_OFF + QS_BYTES].iter() {
            sink((b & 0xF) as usize);
            sink((b >> 4) as usize);
        }
    }
}

/// bytemuck cast with a loud misalignment error (the q2k.rs soundness note:
/// GGUF data_start is 32-aligned and every audited stride is even, so the
/// cast succeeds on real files; a violated premise fails LOUD, never UB).
#[inline]
fn cast_blocks<'a, T: bytemuck::Pod>(
    bytes: &'a [u8],
    ty: &'static str,
) -> Result<&'a [T], DescLenError> {
    bytemuck::try_cast_slice(bytes).map_err(|_| DescLenError::Misaligned { ty })
}

/// Collect the full symbol sequence of one tensor (test/tooling lane — the
/// G1 entropy-coder gate feeds this to the reference rANS coder).
pub fn collect_symbols(bytes: &[u8], ty: GgmlType) -> Result<Vec<u8>, DescLenError> {
    let mut seq = Vec::new();
    scan_tensor(bytes, ty, &mut |s: usize| seq.push(s as u8))?;
    Ok(seq)
}

// ── Audits ──────────────────────────────────────────────────────

/// One tensor's audit row.
#[derive(Clone, Debug)]
pub struct TensorAudit {
    pub name: String,
    pub format: &'static str,
    /// Stored symbol count (auditable) or element count (advisory).
    pub n_symbols: u64,
    /// Actual file cost of the tensor payload, bits.
    pub stored_bits: u64,
    /// Nominal value-payload cost, bits (0 for advisory rows).
    pub value_bits_nominal: u64,
    /// Side-information cost (scales), bits (0 for advisory rows).
    pub side_info_bits: u64,
    /// Alphabet size (0 for advisory rows).
    pub alphabet_k: usize,
    /// Distinct symbols observed (0 for advisory rows).
    pub distinct_symbols: u64,
    /// `N·H(p)` — the value-only MDL floor, bits (None for advisory rows).
    pub floor_bits: Option<f64>,
    /// Advisory reason (None for audited rows).
    pub advisory: Option<&'static str>,
}

impl TensorAudit {
    /// Stored bits per symbol.
    pub fn stored_bpw(&self) -> f64 {
        if self.n_symbols == 0 {
            return 0.0;
        }
        self.stored_bits as f64 / self.n_symbols as f64
    }

    /// Value-only floor, bits per symbol (`H(p)`).
    pub fn floor_bpw(&self) -> Option<f64> {
        self.floor_bits.map(|f| f / self.n_symbols as f64)
    }

    /// Side-info bits per symbol.
    pub fn side_info_bpw(&self) -> f64 {
        if self.n_symbols == 0 {
            return 0.0;
        }
        self.side_info_bits as f64 / self.n_symbols as f64
    }

    /// Side-info-INCLUSIVE floor: `H(p) + side_info_bpw` — the honest
    /// two-part floor.
    pub fn honest_floor_bpw(&self) -> Option<f64> {
        self.floor_bpw().map(|h| h + self.side_info_bpw())
    }

    /// `stored_bpw − H(p)` — the value-only slack (the optimistic headline).
    pub fn slack_bpw(&self) -> Option<f64> {
        Some(self.stored_bpw() - self.floor_bpw()?)
    }

    /// `stored_bpw − (H(p) + side_info_bpw)` — the honest slack.
    pub fn honest_slack_bpw(&self) -> Option<f64> {
        Some(self.stored_bpw() - self.honest_floor_bpw()?)
    }
}

/// Audit one tensor from its GGUF byte slice (the counter is reused across
/// calls by the file-level audit).
pub fn audit_tensor_bytes(
    name: &str,
    bytes: &[u8],
    ty: GgmlType,
    counts: &mut SymbolCounts,
) -> Result<TensorAudit, DescLenError> {
    let spec = FormatSpec::of(ty);
    let stored_bits = (bytes.len() * 8) as u64;
    let base = TensorAudit {
        name: name.to_string(),
        format: spec.name,
        n_symbols: 0,
        stored_bits,
        value_bits_nominal: 0,
        side_info_bits: 0,
        alphabet_k: spec.alphabet_k,
        distinct_symbols: 0,
        floor_bits: None,
        advisory: spec.advisory,
    };
    if spec.advisory.is_some() {
        // Advisory row: report the stored cost + an element estimate where
        // the container size is known; never a floor.
        let n = if spec.block_bytes > 0 && spec.block_weights > 0 {
            (bytes.len() / spec.block_bytes * spec.block_weights) as u64
        } else {
            0
        };
        return Ok(TensorAudit {
            n_symbols: n,
            ..base
        });
    }
    counts.clear();
    let mut record = |s: usize| counts.record(s);
    scan_tensor(bytes, ty, &mut record)?;
    let total = counts.total();
    let h = counts.entropy_bits();
    let n_blocks = total / spec.block_weights as u64;
    Ok(TensorAudit {
        n_symbols: total,
        value_bits_nominal: spec.value_bits_per_block() * n_blocks,
        side_info_bits: spec.side_info_bits_per_block * n_blocks,
        distinct_symbols: counts.distinct(),
        floor_bits: Some(h * total as f64),
        ..base
    })
}

/// Whole-file audit: every tensor, one reused counter, file order.
///
/// Read-only over the mmap — the audit path never writes (the G3 gate
/// pins decode-output byte-identity across an audit).
pub fn audit_file(file: &GgufFile) -> Result<ModelAudit, DescLenError> {
    let mut counts = SymbolCounts::new();
    let mut rows = Vec::with_capacity(file.tensor_infos.len());
    for info in &file.tensor_infos {
        let bytes = file
            .tensor_slice(&info.name)
            .expect("tensor_infos entries always have data slices");
        rows.push(audit_tensor_bytes(
            &info.name,
            bytes,
            info.ggml_type,
            &mut counts,
        )?);
    }
    Ok(ModelAudit { rows })
}

/// Model-level audit result: per-tensor rows + rollup helpers.
#[derive(Clone, Debug)]
pub struct ModelAudit {
    pub rows: Vec<TensorAudit>,
}

impl ModelAudit {
    /// Sort rows by honest slack (descending) — most headroom first;
    /// advisory rows (no floor) sink to the end.
    pub fn sort_by_honest_slack(&mut self) {
        self.rows
            .sort_by(|a, b| b.honest_slack_key().total_cmp(&a.honest_slack_key()));
    }

    /// Model rollup over the auditable rows (advisory rows are counted but
    /// contribute no floor).
    pub fn rollup(&self) -> Rollup {
        let mut r = Rollup::default();
        for row in &self.rows {
            r.tensors += 1;
            match row.advisory {
                Some(_) => r.advisory_tensors += 1,
                None => {
                    r.weights += row.n_symbols;
                    r.stored_bits += row.stored_bits;
                    r.value_bits_nominal += row.value_bits_nominal;
                    r.side_info_bits += row.side_info_bits;
                    r.floor_bits += row.floor_bits.unwrap_or(0.0);
                }
            }
        }
        r
    }
}

impl TensorAudit {
    /// Sort key for [`ModelAudit::sort_by_honest_slack`]: honest slack, or
    /// negative infinity for advisory rows (sinks them to the end).
    fn honest_slack_key(&self) -> f64 {
        self.honest_slack_bpw().unwrap_or(f64::NEG_INFINITY)
    }
}

/// Model-level rollup (auditable rows only for the floor columns).
#[derive(Clone, Copy, Debug, Default)]
pub struct Rollup {
    pub tensors: usize,
    pub advisory_tensors: usize,
    pub weights: u64,
    pub stored_bits: u64,
    pub value_bits_nominal: u64,
    pub side_info_bits: u64,
    pub floor_bits: f64,
}

impl Rollup {
    /// Aggregate stored bpw over audited weights.
    pub fn stored_bpw(&self) -> f64 {
        if self.weights == 0 {
            return 0.0;
        }
        self.stored_bits as f64 / self.weights as f64
    }

    /// Aggregate honest floor bpw: `(Σ N·H + Σ side_info) / Σ N`. The
    /// aggregate floor is the SUM of per-tensor floors (the entropy coder is
    /// per-tensor; pooling histograms would understate the true floor).
    pub fn honest_floor_bpw(&self) -> f64 {
        if self.weights == 0 {
            return 0.0;
        }
        (self.floor_bits + self.side_info_bits as f64) / self.weights as f64
    }

    /// Aggregate honest slack bpw.
    pub fn honest_slack_bpw(&self) -> f64 {
        self.stored_bpw() - self.honest_floor_bpw()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable as _;

    // ── layout self-consistency + loader agreement ─────────────

    const AUDITABLE: [GgmlType; 11] = [
        GgmlType::Q2_0,
        GgmlType::PTQ1_0,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q8_0,
        GgmlType::Q8_1,
    ];

    #[test]
    fn every_auditable_spec_sums_to_its_block_bytes() {
        // value payload + side info == stored: a drift in either column
        // reds here, not on a real file.
        for ty in AUDITABLE {
            let spec = FormatSpec::of(ty);
            assert_eq!(
                (spec.value_bytes_per_block * 8) as u64,
                spec.stored_bits_per_block() - spec.side_info_bits_per_block,
                "{ty:?}: value_bytes·8 + side_info != block_bytes·8"
            );
            // Loader block geometry agreement (the same source of truth).
            let (block_bytes, weights) = ty
                .block_info()
                .unwrap_or_else(|| panic!("{ty:?}: loader has no block_info"));
            assert_eq!(block_bytes, spec.block_bytes, "{ty:?} bytes");
            assert_eq!(weights, spec.block_weights, "{ty:?} weights");
            assert!(spec.alphabet_k > 0 && spec.alphabet_k <= MAX_ALPHABET);
        }
    }

    #[test]
    fn advisory_formats_have_no_floor_columns() {
        for ty in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::F64,
            GgmlType::I8,
            GgmlType::I16,
            GgmlType::I32,
            GgmlType::I64,
            GgmlType::Q8_K,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
        ] {
            let spec = FormatSpec::of(ty);
            assert!(spec.advisory.is_some(), "{ty:?} must be advisory");
            let buf = vec![0u8; spec.block_bytes.max(1)];
            let err = scan_tensor(&buf, ty, &mut |_| {}).unwrap_err();
            assert!(matches!(err, DescLenError::Advisory { .. }));
        }
    }

    // ── entropy math ───────────────────────────────────────────

    #[test]
    fn entropy_uniform_is_log2_k_and_degenerate_is_zero() {
        let mut c = SymbolCounts::new();
        for s in [0usize, 1, 2, 3] {
            c.record(s);
        }
        assert!((c.entropy_bits() - 2.0).abs() < 1e-12);

        let mut c = SymbolCounts::new();
        for _ in 0..100 {
            c.record(2);
        }
        assert_eq!(c.entropy_bits(), 0.0);
        assert_eq!(c.distinct(), 1);
    }

    #[test]
    fn entropy_known_value_three_quarters_one_quarter() {
        // H(3/4, 1/4) = 0.811278...
        let mut c = SymbolCounts::new();
        for _ in 0..3 {
            c.record(0);
        }
        c.record(1);
        let expected = -(0.75 * 0.75f64.log2() + 0.25 * 0.25f64.log2());
        assert!((c.entropy_bits() - expected).abs() < 1e-12);
    }

    // ── per-format hand-built block scans ──────────────────────

    #[test]
    fn q2_0_block_scan_counts_four_codes() {
        // One block: d = anything, qs bytes chosen so codes are known.
        // buf[2] = 0b00_01_10_11 → LSB-first symbols 3,2,1,0;
        // buf[3] = 0b01_01_01_01 → four 1s; the remaining 30 qs bytes are
        // zero → 120 extra 0-symbols.
        let mut buf = vec![0u8; 34];
        buf[2] = 0b00_01_10_11; // symbols 3,2,1,0
        buf[3] = 0b01_01_01_01; // four 1s
        let mut c = SymbolCounts::new();
        scan_tensor(&buf, GgmlType::Q2_0, &mut |s| c.record(s)).unwrap();
        assert_eq!(c.total(), 128);
        assert_eq!(c.counts()[0], 121); // 1 + 120 from the zero bytes
        assert_eq!(c.counts()[1], 5); // 1 + four 1s
        assert_eq!(c.counts()[2], 1);
        assert_eq!(c.counts()[3], 1);
        assert_eq!(c.counts()[4..], [0; 252]);
    }

    #[test]
    fn ptq1_0_scan_recovers_the_encoded_trits() {
        // Encode a known trit pattern with the module's reference quantizer
        // (independent traversal), then scan and compare histograms.
        let mut rng = 0x243F6A8885A308D3u64;
        let mut trits = [0i8; 128];
        let mut expected = [0u64; 3];
        for t in &mut trits {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            *t = match rng % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            };
            expected[(*t + 1) as usize] += 1;
        }
        let mut x = [0f32; 128];
        for (x, t) in x.iter_mut().zip(trits) {
            *x = t as f32 * 0.5;
        }
        let block = crate::quant::ptq1_0::quantize_row_ptq1_0_ref(&x);
        let buf = bytemuck::bytes_of(&block).to_vec();
        assert_eq!(buf.len(), 28);
        let mut c = SymbolCounts::new();
        scan_tensor(&buf, GgmlType::PTQ1_0, &mut |s| c.record(s)).unwrap();
        assert_eq!(c.total(), 128);
        assert_eq!(c.counts()[0], expected[0]);
        assert_eq!(c.counts()[1], expected[1]);
        assert_eq!(c.counts()[2], expected[2]);
    }

    #[test]
    fn q3_k_scan_pairs_code_with_sign_bit() {
        // Hand-built block (the pattern of q3k.rs's mapping test):
        // sub-block 0 (sb16=0): half 0, b 0, j 0 → qbase 0, hbase 0, hbit 0.
        //   elems 0..4 → q = 0,1,2,3; elem 0 hi → symbols 0|1<<2=4, 1, 2, 3.
        // sub-block 2 (sb16=2): j 1 → shift 2, qbase 0, hbit 1:
        //   elem 32 → q = 1 at shift 2, hi → symbol 1|1<<2 = 5.
        // sub-block 8 (sb16=8): half 1 → qbase 32, hbit 4:
        //   elem 128 → q = 2, hi → symbol 2|1<<2 = 6.
        let mut blk = BlockQ3K::zeroed();
        blk.qs[0] = 0;
        blk.qs[1] = 1;
        blk.qs[2] = 2;
        blk.qs[3] = 3;
        blk.hmask[0] = 1;
        blk.qs[0] |= 1 << 2;
        blk.hmask[0] |= 1 << 1;
        blk.qs[32] = 2;
        blk.hmask[0] |= 1 << 4;
        let buf = bytemuck::bytes_of(&blk).to_vec();
        assert_eq!(buf.len(), 110);
        let mut c = SymbolCounts::new();
        scan_tensor(&buf, GgmlType::Q3_K, &mut |s| c.record(s)).unwrap();
        assert_eq!(c.total(), 256);
        assert_eq!(c.counts()[4], 1); // elem 0: q0 + hi
        assert_eq!(c.counts()[1], 1);
        assert_eq!(c.counts()[2], 1);
        assert_eq!(c.counts()[3], 1);
        assert_eq!(c.counts()[5], 1); // elem 32: q1 + hi
        assert_eq!(c.counts()[6], 1); // elem 128: q2 + hi
        // everything else zero: 0,1,2,3,4,5,6 = 7 distinct symbols
        assert_eq!(c.distinct(), 7);
    }

    #[test]
    fn q6_k_scan_recovers_the_packed_6bit_codes() {
        // Hand-built: element j+l+0 of half 0 → ql[l] low + qh[l] bits[1:0].
        // Set l=0: ql[0] = 0x05 (q1 low=5), qh[0] bits[1:0] = 0b10 → q1 = 5|(2<<4) = 37.
        // l=0 q2 (element 32): ql[32] low = 3, qh[0] bits[3:2] = 0b01 → q2 = 3|(1<<4) = 19.
        // q3 (element 64): ql[0] high = 7, qh[0] bits[5:4] = 0b11 → q3 = 7|(3<<4) = 55.
        // q4 (element 96): ql[32] high = 1, qh[0] bits[7:6] = 0b00 → q4 = 1.
        let mut blk = BlockQ6K::zeroed();
        blk.ql[0] = 0x75; // low 5, high 7
        blk.ql[32] = 0x13; // low 3, high 1
        blk.qh[0] = 0b00_11_01_10;
        let buf = bytemuck::bytes_of(&blk).to_vec();
        assert_eq!(buf.len(), 210);
        let mut c = SymbolCounts::new();
        scan_tensor(&buf, GgmlType::Q6_K, &mut |s| c.record(s)).unwrap();
        assert_eq!(c.total(), 256);
        assert_eq!(c.counts()[37], 1);
        assert_eq!(c.counts()[19], 1);
        assert_eq!(c.counts()[55], 1);
        assert_eq!(c.counts()[1], 1);
        // Half 1 (ql[64..], qh[32..]) is zeroed → 252 zeros.
        assert_eq!(c.counts()[0], 252);
        assert_eq!(c.distinct(), 5);
    }

    #[test]
    fn q5_k_scan_pairs_nibbles_with_the_chunk_high_bit() {
        // Chunk 0 (g=0): u1 = bit0, u2 = bit1 of qh[l] — byte l holds the
        // high bits for element pair l across all 4 chunks (2 bits/chunk).
        // l=0: ql[0] = 0x2A → lo=10 (qh[0] bit0 set → 10|16=26),
        //                        hi_nib=2 (qh[0] bit1 clear → 2).
        // l=1: ql[1] = 0x1F → lo=15 (qh[1] bit0 clear → 15),
        //                        hi_nib=1 (qh[1] bit1 set → 1|16=17).
        let mut blk = BlockQ5K::zeroed();
        blk.qs[0] = 0x2A;
        blk.qs[1] = 0x1F;
        blk.qh[0] = 0b01; // bit0 set (lo l=0 hi), bit1 clear (hi l=0 clear)
        blk.qh[1] = 0b10; // bit0 clear (lo l=1 clear), bit1 set (hi l=1 set)
        let buf = bytemuck::bytes_of(&blk).to_vec();
        assert_eq!(buf.len(), 176);
        let mut c = SymbolCounts::new();
        scan_tensor(&buf, GgmlType::Q5_K, &mut |s| c.record(s)).unwrap();
        assert_eq!(c.total(), 256);
        assert_eq!(c.counts()[26], 1);
        assert_eq!(c.counts()[2], 1);
        assert_eq!(c.counts()[15], 1);
        assert_eq!(c.counts()[17], 1);
    }

    #[test]
    fn q4k_q2k_nibble_streams_are_multisets() {
        let mut blk4 = BlockQ4K::zeroed();
        blk4.qs[0] = 0xEF; // 15, 14
        blk4.qs[1] = 0x01; // 1, 0
        let buf = bytemuck::bytes_of(&blk4).to_vec();
        let mut c = SymbolCounts::new();
        scan_tensor(&buf, GgmlType::Q4_K, &mut |s| c.record(s)).unwrap();
        assert_eq!(c.total(), 256);
        // The other 126 qs bytes are zero: 252 zero-nibbles + qs[1] high.
        assert_eq!(c.counts()[15], 1);
        assert_eq!(c.counts()[14], 1);
        assert_eq!(c.counts()[1], 1);
        assert_eq!(c.counts()[0], 253);

        let mut blk2 = BlockQ2K::zeroed();
        blk2.qs[0] = 0b11_10_01_00;
        let buf = bytemuck::bytes_of(&blk2).to_vec();
        let mut c = SymbolCounts::new();
        scan_tensor(&buf, GgmlType::Q2_K, &mut |s| c.record(s)).unwrap();
        assert_eq!(c.total(), 256);
        assert_eq!(c.counts()[3], 1);
        // 63 zero bytes × 4 symbols + qs[0]'s own zero code.
        assert_eq!(c.counts()[0], 253);
    }

    #[test]
    fn q8_legacy_streams_count_raw_bytes() {
        let mut blk = BlockQ8_0::zeroed();
        blk.qs[0] = -1i8;
        blk.qs[1] = 127i8;
        let buf = bytemuck::bytes_of(&blk).to_vec();
        let mut c = SymbolCounts::new();
        scan_tensor(&buf, GgmlType::Q8_0, &mut |s| c.record(s)).unwrap();
        assert_eq!(c.total(), 32);
        assert_eq!(c.counts()[255], 1); // −1 as u8
        assert_eq!(c.counts()[127], 1);

        // Q4_0: 18 B blocks, qs at offset 2 — nibbles low-first.
        let mut buf = vec![0u8; 18];
        buf[2] = 0xAB; // 11, 10
        let mut c = SymbolCounts::new();
        scan_tensor(&buf, GgmlType::Q4_0, &mut |s| c.record(s)).unwrap();
        assert_eq!(c.total(), 32);
        assert_eq!(c.counts()[11], 1);
        assert_eq!(c.counts()[10], 1);
        // 15 zero qs bytes → 30 zero-nibbles + buf[2]'s high nibble is 0xA=10
        // (counted), the low 0xB=11 (counted) → zero count = 30.
        assert_eq!(c.counts()[0], 30);
    }

    // ── audit row arithmetic ─────────────────────────────────

    /// Two Q2_0 blocks whose 256 codes are exactly uniform (each of the 32
    /// qs bytes carries codes 3,2,1,0 LSB-first).
    fn uniform_q2_0_two_blocks() -> Vec<u8> {
        let mut buf = Vec::with_capacity(68);
        for _ in 0..2 {
            buf.extend_from_slice(&[0, 0]); // d
            buf.extend(std::iter::repeat_n(0b11_10_01_00, 32));
        }
        buf
    }

    #[test]
    fn uniform_codes_put_the_honest_slack_at_exactly_zero() {
        let buf = uniform_q2_0_two_blocks();
        let mut counts = SymbolCounts::new();
        let row = audit_tensor_bytes("t", &buf, GgmlType::Q2_0, &mut counts).unwrap();
        assert_eq!(row.n_symbols, 256);
        assert_eq!(row.stored_bits, 68 * 8);
        assert_eq!(row.value_bits_nominal, 64 * 8);
        assert_eq!(row.side_info_bits, 32);
        assert_eq!(row.alphabet_k, 4);
        assert_eq!(row.distinct_symbols, 4);
        // H = 2.0 exactly (uniform-4); floor = 512 bits.
        let floor = row.floor_bits.unwrap();
        assert!((floor - 512.0).abs() < 1e-9);
        assert!((row.floor_bpw().unwrap() - 2.0).abs() < 1e-12);
        // stored 2.125 − floor 2.0 = 0.125 value-only slack;
        // honest floor = 2.0 + 0.125 = 2.125 = stored → honest slack 0.
        assert!((row.slack_bpw().unwrap() - 0.125).abs() < 1e-12);
        assert!(row.honest_slack_bpw().unwrap().abs() < 1e-12);
    }

    #[test]
    fn degenerate_codes_pay_the_full_side_info_in_the_honest_floor() {
        // All codes 0 → H = 0: the honest floor is the side info alone
        // (0.125 bpw) and the honest slack is the stored 2.125 minus it.
        let mut buf = Vec::with_capacity(68);
        for _ in 0..2 {
            buf.extend_from_slice(&[0, 0]);
            buf.extend_from_slice(&[0u8; 32]);
        }
        let mut counts = SymbolCounts::new();
        let row = audit_tensor_bytes("t", &buf, GgmlType::Q2_0, &mut counts).unwrap();
        assert_eq!(row.distinct_symbols, 1);
        assert_eq!(row.floor_bits.unwrap(), 0.0);
        assert!((row.honest_floor_bpw().unwrap() - 0.125).abs() < 1e-12);
        assert!((row.honest_slack_bpw().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn advisory_rows_carry_no_floor_and_the_stored_cost() {
        let buf = vec![0u8; 40]; // 10 f32 elements
        let mut counts = SymbolCounts::new();
        let row = audit_tensor_bytes("norm", &buf, GgmlType::F32, &mut counts).unwrap();
        assert!(row.advisory.is_some());
        assert!(row.floor_bits.is_none());
        assert_eq!(row.stored_bits, 320);
        assert_eq!(row.n_symbols, 10);
        assert!(row.honest_slack_bpw().is_none());
    }

    #[test]
    fn malformed_block_length_fails_loud_not_advisory() {
        let buf = vec![0u8; 33]; // not a multiple of 34
        let mut counts = SymbolCounts::new();
        let err = audit_tensor_bytes("t", &buf, GgmlType::Q2_0, &mut counts).unwrap_err();
        assert!(matches!(err, DescLenError::BadLength { .. }));
    }

    #[test]
    fn model_audit_sort_sinks_advisory_and_ranks_by_honest_slack() {
        let mut counts = SymbolCounts::new();
        let uniform = audit_tensor_bytes(
            "uniform",
            &uniform_q2_0_two_blocks(),
            GgmlType::Q2_0,
            &mut counts,
        )
        .unwrap();
        let mut degen_buf = Vec::new();
        for _ in 0..2 {
            degen_buf.extend_from_slice(&[0, 0]);
            degen_buf.extend_from_slice(&[0u8; 32]);
        }
        let degenerate =
            audit_tensor_bytes("degenerate", &degen_buf, GgmlType::Q2_0, &mut counts).unwrap();
        let advisory = audit_tensor_bytes("norm", &[0u8; 8], GgmlType::F32, &mut counts).unwrap();

        let mut model = ModelAudit {
            rows: vec![advisory.clone(), uniform.clone(), degenerate.clone()],
        };
        model.sort_by_honest_slack();
        // degenerate (honest slack 2.0) > uniform (0.0) > advisory sinks last.
        assert_eq!(model.rows[0].name, "degenerate");
        assert_eq!(model.rows[1].name, "uniform");
        assert_eq!(model.rows[2].name, "norm");

        let rollup = model.rollup();
        assert_eq!(rollup.tensors, 3);
        assert_eq!(rollup.advisory_tensors, 1);
        assert_eq!(rollup.weights, 512);
        // stored: 544 bits per q2_0 row × 2 rows = 1088; floor: 512 + 0.
        assert_eq!(rollup.stored_bits, 1088);
        assert!((rollup.floor_bits - 512.0).abs() < 1e-9);
        assert_eq!(rollup.side_info_bits, 64);
        // stored = 1088/512 = 2.125 bpw;
        // honest floor = (512 + 64)/512 = 1.125 bpw (uniform row H=2,
        // degenerate row H=0, side 64 bits pooled over all 512 weights);
        // honest slack = 2.125 − 1.125 = 1.0 bpw.
        assert!((rollup.stored_bpw() - 2.125).abs() < 1e-12);
        assert!((rollup.honest_floor_bpw() - 1.125).abs() < 1e-12);
        assert!((rollup.honest_slack_bpw() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn collect_symbols_reproduces_the_scan_order() {
        let buf = uniform_q2_0_two_blocks();
        let seq = collect_symbols(&buf, GgmlType::Q2_0).unwrap();
        assert_eq!(seq.len(), 256);
        // The uniform helper's qs byte = 0b11_10_01_00 → LSB-first 0,1,2,3.
        assert_eq!(&seq[0..4], &[0, 1, 2, 3]);
    }
}
