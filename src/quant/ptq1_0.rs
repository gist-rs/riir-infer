//! `PTQ1_0` (GGUF type 143) — Prism ternary at group 128, base-3 dense-trit
//! packing (1.75 bpw). Issue 980 T6 / Plan 600 C5.
//!
//! The Bonsai-2 decode pack (`Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5.93 GB)
//! stores every ternary tensor in this format. Wire contract (transcribed
//! from the `PrismML` fork, commit `01fd9521c` — "ggml: add `PTQ1_0`, ternary at
//! group 128"; verified against `ggml-quants.c`'s
//! `quantize_row_ptq1_0_ref` / `dequantize_row_ptq1_0`):
//!
//! ```text
//! block = qs[24] + qh[2] + d:f16        // 28 B per 128 weights = 1.75 bpw
//! ```
//!
//! # Element map (which trit lives where)
//!
//! The base-3 packing is NON-positional: a qs byte carries trits for five
//! elements scattered across the block. Per 128-weight block, with `xi =
//! trit + 1 ∈ {0,1,2}`:
//!
//! - `qs[0..16]` (stage c=16): byte `m` packs elements `{m, m+16, m+32,
//!   m+48, m+64}` — `V = Σ_n xi[m + 16n]·3^(4−n)`, byte = `⌈V·256/243⌉`.
//! - `qs[16..24]` (stage c=8): byte `16+m` packs elements `{80+m, 88+m,
//!   96+m, 104+m, 112+m}` — same 5-trit fold.
//! - `qh[h]` (h ∈ {0,1}): packs elements `{120+h, 122+h, 124+h, 126+h}` —
//!   `V = Σ_m xi[120+h+2m]·3^(3−m)`, then `V·3` (a leading-zero 5th trit so
//!   the ceiling map cannot clip), byte = `⌈V·3·256/243⌉`.
//!
//! Dequant inverts by modular digit extraction: `t = (byte·3ⁿ) mod 256`,
//! `xi = (t·3) >> 8 ∈ {0,1,2}`, weight = `(xi−1)·d`. `n = 0` is the MOST
//! significant trit (the first-packed element).
//!
//! # Losslessness
//!
//! Ternary inputs round-trip exactly (the fork's measured claim — perplexity
//! identical to `PQ2_0` to every printed digit on this model family); our
//! round-trip test pins it. Because every decoded code is already in
//! `{-1, 0, +1}`, the `TernaryGroupWeights` bridge has NO fourth-state
//! rejection path (contrast [`crate::quant::q2_0`]'s code-3 guard).
//!
//! # Consumption
//!
//! [`repack_ptq1_0_to_ternary_group`] decodes straight into the katgpt-core
//! bit-plane substrate — the same container `PQ2_0` repacks into — so the
//! ternary gemv path (and the whole Issue-980 rotated forward) consumes
//! `PTQ1_0` files unchanged. Container cost is 34 B per 128 weights (2.125
//! bpw) against the file's 28 (1.75): the 5.93 GB pack expands to ~7.2 GB in
//! memory, the same footprint class as the `PQ2_0` lane.
//!
//! Same-checkpoint equivalence: `PTQ1_0` and `PQ2_0` encodings of one ternary
//! g128 checkpoint carry identical trits and identical f16 group scales, so
//! the two packs must load to BIT-IDENTICAL containers — pinned by the
//! synthetic both-encodings test; the real-file check rides the G1 lane
//! where both packs live.

use half::f16;

/// Block size: 128 elements per block (group size g128).
pub const PTQ1_0_BLOCK_SIZE: usize = 128;

/// `qs` bytes per block: `(128 − 4·128/64) / 5 = 24` — 120 values at 5 trits/byte.
pub const QS_BYTES: usize = 24;

/// `qh` bytes per block: `128 / 64 = 2` — 8 values at 4 trits/byte.
pub const QH_BYTES: usize = 2;

/// Size of one `PTQ1_0` block in bytes (24 + 2 + 2 = 28; 1.75 bpw).
pub const BLOCK_BYTES: usize = QS_BYTES + QH_BYTES + 2;

/// A `PTQ1_0` quantization block — 128 ternary weights as base-3 trits + one
/// f16 group scale. Layout matches the fork's `block_ptq1_0` exactly for
/// mmap zero-copy compatibility (`qs`, then `qh`, then `d` — `d` LAST,
/// unlike `Q2_0`'s `d`-first).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockPtq1_0 {
    /// 24 B — 120 values, 5 trits per byte (stages c=16 then c=8).
    pub qs: [u8; QS_BYTES],
    /// 2 B — 8 values, 4 trits per byte (each with a leading-zero 5th trit).
    pub qh: [u8; QH_BYTES],
    /// f16 group scale (bit pattern, little-endian in the file).
    pub d: u16,
}
const _: () = assert!(std::mem::size_of::<BlockPtq1_0>() == BLOCK_BYTES);

/// `3ⁿ` for the digit extraction (mod-256 arithmetic wraps — that is the
/// recurrence). `n` never exceeds 4 for `qs` and 3 for `qh`.
const POW3: [u8; 5] = [1, 3, 9, 27, 81];

/// Extract the trit for digit `n` of `byte`: `t = (byte·3ⁿ) mod 256`,
/// `xi = (t·3) >> 8 ∈ {0,1,2}`, trit = `xi − 1 ∈ {−1,0,+1}`.
#[inline]
fn extract_trit(byte: u8, n: usize) -> i8 {
    let t = byte.wrapping_mul(POW3[n]);
    (((t as u16) * 3) >> 8) as i8 - 1
}

/// Locate the owning byte and digit index for element `j` of a block
/// (the inverse of the dequant write order — see the module doc's map).
#[inline]
fn element_byte_trit(block: &BlockPtq1_0, j: usize) -> (u8, usize) {
    debug_assert!(j < PTQ1_0_BLOCK_SIZE);
    if j < 80 {
        // Stage c=16: element m + 16n ← qs[m], digit n.
        (block.qs[j % 16], j / 16)
    } else if j < 120 {
        // Stage c=8: element 80 + m + 8n ← qs[16 + m], digit n.
        let k = j - 80;
        (block.qs[16 + k % 8], k / 8)
    } else {
        // qh: element 120 + h + 2n ← qh[h], digit n.
        let k = j - 120;
        (block.qh[k % 2], k / 2)
    }
}

/// Unscaled trit for element `j` of `block` ∈ `{−1, 0, +1}`.
#[inline]
pub fn ptq1_0_element_trit(block: &BlockPtq1_0, j: usize) -> i8 {
    let (byte, n) = element_byte_trit(block, j);
    extract_trit(byte, n)
}

// ── Dequantize ──────────────────────────────────────────────────

/// Dequantize `PTQ1_0` blocks to f32 values (element order preserved).
///
/// `dst` must be at least `blocks.len() × 128` long; each block writes
/// `dst[i·128 ..][..128]`.
pub fn dequantize_row_ptq1_0(blocks: &[BlockPtq1_0], dst: &mut [f32]) {
    assert!(
        dst.len() >= blocks.len() * PTQ1_0_BLOCK_SIZE,
        "ptq1_0 dequantize: dst.len() {} < {} blocks × 128",
        dst.len(),
        blocks.len()
    );
    for (i, block) in blocks.iter().enumerate() {
        let d = f16::from_bits(block.d).to_f32();
        let out = &mut dst[i * PTQ1_0_BLOCK_SIZE..][..PTQ1_0_BLOCK_SIZE];
        for (e, o) in out.iter_mut().enumerate() {
            *o = ptq1_0_element_trit(block, e) as f32 * d;
        }
    }
}

// ── PTQ1_0 → TernaryGroupWeights bridge (Issue 980 T6) ─────────

/// Errors from the `PTQ1_0` → `TernaryGroupWeights` repack.
#[cfg(feature = "q2_0_ternary_bridge")]
#[derive(Debug, thiserror::Error)]
pub enum Pq1RepackError {
    #[error("too few PTQ1_0 blocks: got {got}, expected {expected}")]
    TooFewBlocks { got: usize, expected: usize },
    #[error("cols ({cols}) is not a multiple of 128")]
    ColsNotMultipleOf128 { cols: usize },
}

/// Repack `PTQ1_0` GGUF blocks into the katgpt-core `TernaryGroupWeights`
/// substrate (Issue 980 T6 — the Phase C decode lane).
///
/// Lossless by construction: every decoded code is already ternary, so the
/// bit-planes hold the file's trits exactly and the f16 group scale copies
/// verbatim — the same container `repack_q2_0_to_ternary_group` produces for
/// the `PQ2_0` encoding of the same checkpoint. Geometry and row order mirror
/// the `Q2_0` bridge (row-major, `cols/128` blocks per row); no fourth-state
/// rejection path exists (the base-3 alphabet has no `+2d` code).
#[cfg(feature = "q2_0_ternary_bridge")]
pub fn repack_ptq1_0_to_ternary_group(
    blocks: &[BlockPtq1_0],
    rows: usize,
    cols: usize,
) -> Result<katgpt_core::TernaryGroupWeights, Pq1RepackError> {
    use katgpt_core::TernaryGroupWeights;

    let expected_blocks = rows * cols / PTQ1_0_BLOCK_SIZE;
    if blocks.len() < expected_blocks {
        return Err(Pq1RepackError::TooFewBlocks {
            got: blocks.len(),
            expected: expected_blocks,
        });
    }
    if !cols.is_multiple_of(PTQ1_0_BLOCK_SIZE) {
        return Err(Pq1RepackError::ColsNotMultipleOf128 { cols });
    }

    let mut out = TernaryGroupWeights::new(rows, cols);
    let blocks_per_row = cols / PTQ1_0_BLOCK_SIZE;

    for row in 0..rows {
        let row_block_base = row * blocks_per_row;
        for g in 0..blocks_per_row {
            let block = &blocks[row_block_base + g];
            // f16 group scale, verbatim.
            out.group_scale[row * out.groups_per_row + g] = f16::from_bits(block.d);

            let b0 = row * out.blocks64 + g * 2;
            let b1 = b0 + 1;
            for j in 0..PTQ1_0_BLOCK_SIZE {
                let bit_idx = if j < 64 { b0 } else { b1 };
                let bit_mask = 1u64 << (j & 63);
                match ptq1_0_element_trit(block, j) {
                    -1 => out.neg_bits[bit_idx] |= bit_mask,
                    0 => {}
                    _ => out.pos_bits[bit_idx] |= bit_mask,
                }
            }
        }
    }

    debug_assert!(
        out.invariant_holds(),
        "pos & neg == 0 invariant violated after PTQ1_0 repack"
    );
    Ok(out)
}

// ── Reference quantizer (round-trip oracle) ─────────────────────

/// Encode one 128-float block (the fork's `quantize_row_ptq1_0_ref`
/// traversal). `x.len()` must be ≥ 128; only the first 128 elements are
/// read. `d = amax` (the reference encoder's choice), trits from
/// `lroundf(x/amax)`, clamped to the ternary alphabet.
///
/// Test-side oracle: the decoder above and this encoder are INDEPENDENT
/// transcriptions (element-map vs staged traversal), so their round-trip is
/// evidence, not tautology.
#[cfg(test)]
pub(crate) fn quantize_row_ptq1_0_ref(x: &[f32]) -> BlockPtq1_0 {
    let mut amax = 0.0f32;
    for &v in &x[..PTQ1_0_BLOCK_SIZE] {
        amax = amax.max(v.abs());
    }
    let d = amax;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut out = BlockPtq1_0 {
        qs: [0; QS_BYTES],
        qh: [0; QH_BYTES],
        d: f16::from_f32(d).to_bits(),
    };
    // Ceiling map: q ∈ [0, 242] → byte = ⌈q·256/243⌉ ∈ [0, 255].
    let ceil256 = |q: usize| (q * 256).div_ceil(243) as u8;

    let xi = |e: usize| (x[e] * id).round().clamp(-1.0, 1.0) as i8 + 1;

    // Stage c=16: bytes 0..16, elements m + 16n (n = 0 most significant).
    for m in 0..16 {
        let mut v: usize = 0;
        for n in 0..5 {
            v = v * 3 + xi(m + 16 * n) as usize;
        }
        out.qs[m] = ceil256(v);
    }
    // Stage c=8: bytes 16..24, elements 80 + m + 8n.
    for m in 0..8 {
        let mut v: usize = 0;
        for n in 0..5 {
            v = v * 3 + xi(80 + m + 8 * n) as usize;
        }
        out.qs[16 + m] = ceil256(v);
    }
    // qh: elements 120 + h + 2m, leading-zero 5th trit.
    for h in 0..2 {
        let mut v: usize = 0;
        for m in 0..4 {
            v = v * 3 + xi(120 + h + 2 * m) as usize;
        }
        out.qh[h] = ceil256(v * 3);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64* — no external rng dep in this module.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn trit(&mut self) -> i8 {
            match self.next() % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            }
        }
    }

    /// The C-shaped staged traversal (dequantize), transcribed independently
    /// of the element map — `dequantize_row_ptq1_0` must agree with it on
    /// ARBITRARY bytes, which is what proves the map derivation.
    fn staged_dequant_block(block: &BlockPtq1_0) -> [f32; 128] {
        let d = f16::from_bits(block.d).to_f32();
        let mut y = [0f32; 128];
        let mut e = 0usize;
        for &c in &[16usize, 8] {
            let base = if c == 16 { 0 } else { 16 };
            for &p3 in &POW3 {
                for m in 0..c {
                    let q = block.qs[base + m].wrapping_mul(p3);
                    let xi = ((q as u16) * 3) >> 8;
                    y[e] = (xi as i8 - 1) as f32 * d;
                    e += 1;
                }
            }
        }
        for &p3 in &POW3[..4] {
            for h in 0..2 {
                let q = block.qh[h].wrapping_mul(p3);
                let xi = ((q as u16) * 3) >> 8;
                y[e] = (xi as i8 - 1) as f32 * d;
                e += 1;
            }
        }
        assert_eq!(e, 128);
        y
    }

    /// Hand-computed known answers (independent anchor — computed on paper
    /// from the fork's C, not generated by either transcription):
    /// qs[0]=188 decodes elements {0,16,32,48,64} = {+1,−1,0,+1,0};
    /// qh[0]=206 decodes elements {120,122,124,126} = {+1,0,−1,+1}.
    #[test]
    fn known_answer_bytes() {
        let mut rng = Lcg(0x9E3779B97F4A7C15);
        let mut trits = [1i8; 128];
        for t in trits.iter_mut() {
            *t = rng.trit();
        }
        for (e, t) in [0usize, 16, 32, 48, 64].iter().zip([1i8, -1, 0, 1, 0]) {
            trits[*e] = t;
        }
        for (e, t) in [120usize, 122, 124, 126].iter().zip([1i8, 0, -1, 1]) {
            trits[*e] = t;
        }
        let x: Vec<f32> = trits.iter().map(|&t| t as f32 * 0.5).collect();
        let block = quantize_row_ptq1_0_ref(&x);
        assert_eq!(block.qs[0], 188, "hand-computed stage-c16 byte 0");
        assert_eq!(block.qh[0], 206, "hand-computed qh byte 0");

        let decoded = staged_dequant_block(&block);
        for (e, &t) in trits.iter().enumerate() {
            assert_eq!(
                (decoded[e] / 0.5).round() as i8,
                t,
                "trit mismatch at element {e}"
            );
        }
    }

    /// Ternary inputs round-trip EXACTLY (the fork's losslessness claim,
    /// pinned): quantize → dequant reproduces the trits bit-for-bit and the
    /// scale is the f16 of amax.
    #[test]
    fn ternary_round_trip_is_exact() {
        let mut rng = Lcg(0xDEADBEEFCAFEF00D);
        for case in 0..256u32 {
            let amax = 0.125f32 * (1.0 + (case % 17) as f32);
            let trits: Vec<i8> = (0..128).map(|_| rng.trit()).collect();
            let x: Vec<f32> = trits.iter().map(|&t| t as f32 * amax).collect();
            let block = quantize_row_ptq1_0_ref(&x);
            assert_eq!(block.d, f16::from_f32(amax).to_bits(), "case {case} scale");

            let mut out = [0f32; 128];
            dequantize_row_ptq1_0(&[block], &mut out);
            let expected = f16::from_f32(amax).to_f32();
            for (e, &t) in trits.iter().enumerate() {
                assert_eq!(out[e], t as f32 * expected, "case {case} element {e}");
            }
        }
    }

    /// The element-map decoder and the staged C traversal must agree on
    /// ARBITRARY bytes (not just valid encodings) — this is the proof that
    /// the non-positional map was derived correctly.
    #[test]
    fn element_map_matches_staged_traversal_on_arbitrary_bytes() {
        let mut rng = Lcg(0x0123456789ABCDEF);
        for _ in 0..512 {
            // qs/qh stay ARBITRARY bytes (the extraction is what's under
            // test); d must be a VALID f16 — random 16-bit patterns hit the
            // NaN exponent and NaN != NaN would fail the compare below.
            let d = f16::from_f32((rng.next() % 1000) as f32 / 997.0);
            let mut block = BlockPtq1_0 {
                qs: [0; 24],
                qh: [0; 2],
                d: d.to_bits(),
            };
            for b in block.qs.iter_mut() {
                *b = rng.next() as u8;
            }
            for b in block.qh.iter_mut() {
                *b = rng.next() as u8;
            }
            let staged = staged_dequant_block(&block);
            let mut mapped = [0f32; 128];
            dequantize_row_ptq1_0(&[block], &mut mapped);
            assert_eq!(staged, mapped, "map vs staged divergence");
        }
    }

    /// The repack lands the exact trits + verbatim scale into the container.
    #[cfg(feature = "q2_0_ternary_bridge")]
    #[test]
    fn repack_matches_trits_and_scale() {
        use katgpt_core::TernaryGroupWeights;

        let mut rng = Lcg(0x5DEECE66D);
        let rows = 2;
        let cols = 256; // 2 blocks per row
        let mut blocks = Vec::new();
        let mut expect = Vec::new();
        for _ in 0..rows * (cols / 128) {
            let trits: Vec<i8> = (0..128).map(|_| rng.trit()).collect();
            let x: Vec<f32> = trits.iter().map(|&t| t as f32 * 0.25).collect();
            let block = quantize_row_ptq1_0_ref(&x);
            expect.push(trits);
            blocks.push(block);
        }
        let tg: TernaryGroupWeights =
            repack_ptq1_0_to_ternary_group(&blocks, rows, cols).expect("repack");
        assert!(tg.invariant_holds());

        for row in 0..rows {
            for g in 0..cols / 128 {
                let block = &blocks[row * (cols / 128) + g];
                assert_eq!(
                    tg.group_scale[row * tg.groups_per_row + g].to_bits(),
                    block.d,
                    "scale must copy verbatim"
                );
                let b0 = row * tg.blocks64 + g * 2;
                let expected = &expect[row * (cols / 128) + g];
                for (j, &want) in expected.iter().enumerate() {
                    let bit_idx = if j < 64 { b0 } else { b0 + 1 };
                    let mask = 1u64 << (j & 63);
                    let pos = (tg.pos_bits[bit_idx] & mask) != 0;
                    let neg = (tg.neg_bits[bit_idx] & mask) != 0;
                    let trit = match (pos, neg) {
                        (true, false) => 1,
                        (false, true) => -1,
                        (false, false) => 0,
                        _ => panic!("pos & neg both set"),
                    };
                    assert_eq!(trit, want, "row {row} group {g} element {j}");
                }
            }
        }
    }

    /// Repack geometry refusals.
    #[cfg(feature = "q2_0_ternary_bridge")]
    #[test]
    fn repack_geometry_refusals() {
        let block = BlockPtq1_0 {
            qs: [0; 24],
            qh: [0; 2],
            d: 0,
        };
        let err = repack_ptq1_0_to_ternary_group(&[], 2, 128).unwrap_err();
        assert!(matches!(
            err,
            Pq1RepackError::TooFewBlocks {
                got: 0,
                expected: 2
            }
        ));
        let err = repack_ptq1_0_to_ternary_group(&[block], 1, 100).unwrap_err();
        assert!(matches!(
            err,
            Pq1RepackError::ColsNotMultipleOf128 { cols: 100 }
        ));
    }

    /// Dequantize dst-length contract.
    #[test]
    #[should_panic(expected = "dst.len()")]
    fn dequantize_refuses_short_dst() {
        let block = BlockPtq1_0 {
            qs: [0; 24],
            qh: [0; 2],
            d: 0,
        };
        let mut out = [0f32; 64];
        dequantize_row_ptq1_0(&[block], &mut out);
    }
}
