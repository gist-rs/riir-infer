//! Issue 040 — the G1 gate for the description-length audit.
//!
//! A reference rANS entropy coder bit-matches the histogram-entropy floor:
//! `collect_symbols` output re-encoded by the coder must round-trip exactly
//! and land within a small constant of `N·H(p)` — the B.9.3 closed form's
//! claim that the optimal code cost IS the histogram entropy, tested
//! against a real (non-ideal, byte-granular) coder.
//!
//! Lanes:
//! 1. Synthetic histograms (uniform / degenerate / skewed) — always run.
//! 2. Sampled real groups from a named GGUF corpus via `RIIR_DESC_LEN_GGUF`
//!    (skips LOUD when unset — a deferral, never a green zero).
#![cfg(feature = "desc_len")]

use riir_infer_core::gguf_loader::{GgmlType, GgufFile};
use riir_infer_core::quant::desc_len::{FormatSpec, SymbolCounts, collect_symbols, scan_tensor};

// ── reference rANS (interleaved-less, single state, 64-bit) ────

/// Frequency table over ≤256 symbols, normalized to `T = 1 << FREQ_BITS`.
struct RansTable {
    freq: [u32; 256],
    cum: [u32; 256],
    symbols: Vec<u8>, // reverse-lookup: cum index → symbol (sorted by cum)
}

impl RansTable {
    /// Build from empirical counts (exact normalization: counts must sum to
    /// a multiple that scales losslessly, or largest-remainder rounding).
    fn from_counts(counts: &[u64; 256], total: u64) -> Self {
        const FREQ_BITS: u32 = 16;
        let t_total = 1u64 << FREQ_BITS;
        let mut freq = [0u32; 256];
        // Largest-remainder apportionment of T over the empirical counts.
        let mut scaled: Vec<(usize, u64)> = counts
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(s, c)| (s, *c * t_total / total.max(1)))
            .collect();
        let mut assigned: u64 = scaled.iter().map(|(_, f)| *f).sum();
        // Remainders: give the leftover units to the largest remainders.
        let mut rem: Vec<(usize, u64)> = counts
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(s, c)| (s, (*c * t_total) % total.max(1)))
            .collect();
        rem.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut rem_iter = rem.into_iter();
        while assigned < t_total {
            match rem_iter.next() {
                Some((s, _)) => {
                    if let Some(e) = scaled.iter_mut().find(|(ss, _)| *ss == s) {
                        e.1 += 1;
                    }
                    assigned += 1;
                }
                None => break,
            }
        }
        while assigned > t_total {
            // Over-assigned (fp-free arithmetic cannot produce this, but keep
            // the loop total for honesty): take one unit from the largest.
            if let Some(e) = scaled.iter_mut().max_by_key(|(_, f)| *f) {
                e.1 -= 1;
            }
            assigned -= 1;
        }
        for (s, f) in &scaled {
            freq[*s] = *f as u32;
        }
        let mut cum = [0u32; 256];
        let mut acc = 0u32;
        let mut symbols = Vec::new();
        for s in 0..256usize {
            cum[s] = acc;
            if freq[s] > 0 {
                symbols.push(s as u8);
            }
            acc += freq[s];
        }
        debug_assert_eq!(acc, t_total as u32);
        Self { freq, cum, symbols }
    }

    #[inline]
    fn symbol_for_slot(&self, slot: u32) -> usize {
        // Linear scan over the distinct symbols (K ≤ 256; test-only lane).
        for &s in &self.symbols {
            let s = s as usize;
            if slot < self.cum[s] + self.freq[s] {
                return s;
            }
        }
        unreachable!("slot {slot} outside the cumulative table");
    }
}

/// Encode `symbols` with static rANS; returns the byte stream (decode order:
/// read from the END). 64-bit state, 16-bit frequencies, byte renorm into
/// the window `[2^48, 2^56)` — the canonical pre-step renorm: flush while
/// `x ≥ f·2^40` (integer division then guarantees the post-step state stays
/// in window: `x/f ≤ 2^40−1` ⇒ `x' < 2^56`; and `x' ≥ (x/f)·2^16 ≥ 2^48`).
fn rans_encode(table: &RansTable, symbols: &[u8]) -> Vec<u8> {
    const T_SHIFT: u32 = 16;
    const WINDOW: u64 = 1 << 56;
    let mut state: u64 = 1 << 48;
    let mut out: Vec<u8> = Vec::new();
    for &s in symbols {
        let s = s as usize;
        let f = table.freq[s] as u64;
        let c = table.cum[s] as u64;
        debug_assert!(f > 0, "symbol {s} has zero frequency");
        // Renorm BEFORE the step: the decoder undoes exactly these bytes.
        while state >= f << 40 {
            out.push((state & 0xFF) as u8);
            state >>= 8;
        }
        // x' = (x / f)·T + (x mod f) + c — stays in [2^48, 2^56).
        state = (state / f) * (1 << T_SHIFT) + (state % f) + c;
        debug_assert!(
            (1 << 48..WINDOW).contains(&state),
            "state {state:#x} out of window"
        );
    }
    // Final flush: the full 7-byte state (the decoder rebuilds from these).
    out.extend_from_slice(&state.to_le_bytes()[..7]);
    out
}

/// Decode `n` symbols; consumes the byte stream from its END (mirroring the
/// encoder's flush order).
fn rans_decode(table: &RansTable, bytes: &[u8], n: usize) -> Vec<u8> {
    const T_SHIFT: u32 = 16;
    const T_MASK: u64 = (1 << T_SHIFT) - 1;
    const WINDOW: u64 = 1 << 56;
    let mut idx = bytes.len();
    let take_byte = |idx: &mut usize| -> u64 {
        *idx -= 1;
        bytes[*idx] as u64
    };
    // Rebuild the final state (7 bytes — the encoder's final flush).
    let mut state: u64 = 0;
    for _ in 0..7 {
        state = (state << 8) | take_byte(&mut idx);
    }
    debug_assert!((1 << 48..WINDOW).contains(&state), "state {state:#x}");
    let mut out = vec![0u8; n];
    // rANS is a stack: forward encode ⇒ REVERSE decode order (the bytes the
    // encoder flushed last decode first).
    for i in (0..n).rev() {
        let slot = (state & T_MASK) as u32;
        let s = table.symbol_for_slot(slot);
        let f = table.freq[s] as u64;
        let c = table.cum[s] as u64;
        state = f * (state >> T_SHIFT) + (slot as u64) - c;
        // Renorm: pull bytes while below the window floor (≤ 2 pulls: the
        // inverse step maps into [f·2^32, 2^56) ⊆ [2^32, 2^56)).
        while state < 1 << 48 {
            state = (state << 8) | take_byte(&mut idx);
        }
        debug_assert!(state < WINDOW, "decoded state {state:#x} over the window");
        out[i] = s as u8;
    }
    debug_assert_eq!(idx, 0, "byte stream not fully consumed");
    out
}

/// The G1 assertion pair: exact round-trip + the size bound
/// `N·H − ε ≤ coder_bits ≤ N·H + C` (C covers the 64-bit final flush +
/// byte-granularity; the deterministic value is pinned per lane).
fn assert_g1(name: &str, symbols: &[u8], counts: &[u64; 256], total: u64, slack_bits: f64) {
    let mut c = SymbolCounts::new();
    for &s in symbols {
        c.record(s as usize);
    }
    // The counts fed to the coder must be the counts the histogram saw.
    assert_eq!(c.total(), total, "{name}: total mismatch");
    assert_eq!(c.counts(), counts, "{name}: counts mismatch");

    let h = c.entropy_bits();
    let table = RansTable::from_counts(counts, total);
    let encoded = rans_encode(&table, symbols);
    let decoded = rans_decode(&table, &encoded, symbols.len());
    assert_eq!(decoded, symbols, "{name}: rANS round-trip broke");

    let coder_bits = encoded.len() as f64 * 8.0;
    let floor_bits = h * total as f64;
    assert!(
        coder_bits >= floor_bits - 1e-6,
        "{name}: coder ({coder_bits:.1} b) beat the entropy floor ({floor_bits:.1} b) — \
         the floor arithmetic is wrong"
    );
    assert!(
        coder_bits <= floor_bits + slack_bits,
        "{name}: coder ({coder_bits:.1} b) exceeds floor + {slack_bits} ({floor_bits:.1} b) — \
         the coder is not approaching the floor"
    );
}

fn histogram_of(symbols: &[u8]) -> ([u64; 256], u64) {
    let mut counts = [0u64; 256];
    let mut total = 0u64;
    for &s in symbols {
        counts[s as usize] += 1;
        total += 1;
    }
    (counts, total)
}

/// Deterministic xorshift64* — no external rng dep in this gate.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

#[test]
fn rans_roundtrips_and_matches_the_floor_on_uniform() {
    // Uniform-4, exactly N = 4·4096 so the apportionment is EXACT
    // (f_i = T/4 each) and the coder's per-symbol cost is exactly −log2 p.
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let symbols: Vec<u8> = (0..16_384).map(|_| (rng.next() % 4) as u8).collect();
    let (counts, total) = histogram_of(&symbols);
    assert_g1("uniform-4", &symbols, &counts, total, 64.0);
    // And the floor sits at the uniform maximum: H(empirical) ≤ 2.0 and
    // within sampling noise of it (exact equality would need exactly 4096
    // of each code — not what an LCG draw guarantees).
    let mut c = SymbolCounts::new();
    for &s in &symbols {
        c.record(s as usize);
    }
    assert!(
        c.entropy_bits() <= 2.0 && c.entropy_bits() > 2.0 - 0.01,
        "uniform-4 empirical H = {} (want ≈2.0, never above)",
        c.entropy_bits()
    );
}

#[test]
fn rans_roundtrips_and_matches_the_floor_on_degenerate() {
    let symbols = vec![2u8; 10_000];
    let (counts, total) = histogram_of(&symbols);
    // H = 0: the coder must emit ONLY the final flush (4 bytes... 8 with the
    // 64-bit state) — the floor arithmetic bound allows exactly that.
    assert_g1("degenerate", &symbols, &counts, total, 64.0);
}

#[test]
fn rans_roundtrips_and_matches_the_floor_on_skewed_ternary() {
    // PTQ1_0-shaped: trits with a realistic {0.1, 0.8, 0.1} skew — the
    // entropy must land well below log2(3) and the coder must track it.
    let mut rng = Lcg(0xDEADBEEFCAFEF00D);
    let mut symbols = Vec::with_capacity(100_000);
    for _ in 0..100_000 {
        let r = rng.next() % 100;
        symbols.push(match r {
            0..=9 => 0,
            10..=89 => 1,
            _ => 2,
        });
    }
    let (counts, total) = histogram_of(&symbols);
    let mut c = SymbolCounts::new();
    for &s in &symbols {
        c.record(s as usize);
    }
    // Skew sanity: H(0.1,0.8,0.1) ≈ 0.922 < log2(3) ≈ 1.585.
    assert!(
        c.entropy_bits() < 1.0,
        "skew construction lost: H = {}",
        c.entropy_bits()
    );
    assert_g1("skewed-trits", &symbols, &counts, total, 256.0);
}

#[test]
fn coder_approaches_the_floor_as_n_grows() {
    // The convergence law: |coder/N − H| must SHRINK with N (the +C constant
    // amortizes; per-symbol slack → 0). Two scales, same distribution.
    let mut rng = Lcg(0x0123456789ABCDEF);
    let gen_symbols = |n: usize, rng: &mut Lcg| -> Vec<u8> {
        (0..n)
            .map(|_| match rng.next() % 8 {
                0..=3 => 0u8,
                4..=6 => 1,
                _ => 2,
            })
            .collect()
    };
    let excess = |symbols: &[u8]| -> f64 {
        let (counts, total) = histogram_of(symbols);
        let mut c = SymbolCounts::new();
        for &s in symbols {
            c.record(s as usize);
        }
        let table = RansTable::from_counts(&counts, total);
        let encoded = rans_encode(&table, symbols);
        encoded.len() as f64 * 8.0 / total as f64 - c.entropy_bits()
    };
    let small = excess(&gen_symbols(10_000, &mut rng));
    let large = excess(&gen_symbols(1_000_000, &mut rng));
    assert!(
        large < small,
        "coder excess did not shrink with N: {small:.6} → {large:.6} bits/symbol"
    );
    assert!(
        large < 0.01,
        "large-N excess {large:.6} bits/symbol too big"
    );
}

#[test]
fn floor_survives_the_audit_roundtrip_through_real_scanners() {
    // Synthetic q2_0 blocks (skewed codes) → collect_symbols → G1 pair.
    // This exercises the AUDIT's own extraction, not just the coder.
    let mut rng = Lcg(0xFEEDFACE12345678);
    let mut buf = Vec::with_capacity(34 * 64);
    for _ in 0..64 {
        buf.extend_from_slice(&[0, 0]);
        for _ in 0..32 {
            // Code distribution: 0 with p≈0.7, 1 with p≈0.2, 2/3 rarely.
            let r = rng.next() % 10;
            let pair = match r {
                0..=6 => (0u8, 0u8),
                7..=8 => (1, 0),
                9 => (2, 1),
                _ => unreachable!(),
            };
            buf.push(pair.0 | (pair.1 << 2) | (pair.0 << 4) | (pair.1 << 6));
        }
    }
    let seq = collect_symbols(&buf, GgmlType::Q2_0).unwrap();
    assert_eq!(seq.len(), 64 * 128);
    let (counts, total) = histogram_of(&seq);
    assert_g1("synthetic-q2_0", &seq, &counts, total, 128.0);
}

// ── real-corpus lane ────────────────────────────────────────────

/// Sampled real groups from the named GGUF corpus (env `RIIR_DESC_LEN_GGUF`).
/// The two largest auditable tensors are scanned in file order and pushed
/// through the same G1 pair — the coder must track the REAL histograms.
#[test]
fn real_corpus_groups_match_the_floor() {
    let Some(path) = std::env::var_os("RIIR_DESC_LEN_GGUF") else {
        eprintln!(
            "SKIP (loud): RIIR_DESC_LEN_GGUF unset — the real-corpus G1 lane needs a \
             named GGUF, e.g. RIIR_DESC_LEN_GGUF=../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf"
        );
        return;
    };
    let path = std::path::PathBuf::from(path);
    let file = GgufFile::open(&path).expect("open RIIR_DESC_LEN_GGUF");
    // Pick the two largest auditable tensors.
    let mut candidates: Vec<_> = file
        .tensor_infos
        .iter()
        .filter(|info| FormatSpec::of(info.ggml_type).advisory.is_none())
        .collect();
    candidates.sort_by_key(|info| std::cmp::Reverse(info.byte_len));
    assert!(
        candidates.len() >= 2,
        "corpus has fewer than 2 auditable tensors"
    );
    let mut named: Vec<&str> = Vec::new();
    for info in candidates.iter().take(2) {
        let name = info.name.as_str();
        if named.contains(&name) {
            continue;
        }
        named.push(name);
        let bytes = file.tensor_slice(name).expect("slice");
        let seq = collect_symbols(bytes, info.ggml_type).unwrap();
        // Sampled group: cap at 4 M symbols for coder runtime (still a real
        // file-order prefix — the histogram is the REAL one for the prefix).
        let full_len = seq.len();
        let cap = full_len.min(4 << 20);
        let seq = &seq[..cap];
        let (counts, total) = histogram_of(seq);
        assert_g1(name, seq, &counts, total, 4096.0);
        // The audit row for the same tensor must quote this histogram's H —
        // only comparable when the whole tensor fit under the cap.
        let mut c = SymbolCounts::new();
        let mut sink = |s: usize| c.record(s);
        scan_tensor(bytes, info.ggml_type, &mut sink).unwrap();
        let mut c_prefix = SymbolCounts::new();
        for &s in seq {
            c_prefix.record(s as usize);
        }
        if cap == full_len {
            assert!(
                (c.entropy_bits() - c_prefix.entropy_bits()).abs() < 1e-12,
                "{name}: audit H vs sequence H diverge"
            );
        }
    }
}
