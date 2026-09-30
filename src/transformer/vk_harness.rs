//! vk_harness — shared measurement helpers for the Issue-013 fitted-V bins
//! (T1 [`crate::transformer::gemma2_vquant`] consumers, T2
//! [`crate::transformer::gemma2_ktov`], and T3's reconstruction lane).
//!
//! Everything here is MEASUREMENT-ONLY and allocation-tolerant (report-time
//! code): NLL/argmax, the frozen-table dump/load artifact (BLAKE3-pinned —
//! T3 reuses T2's 2-hour calibration instead of a second pass), and the
//! Bench-814-shaped NIAH prompt builder.

use std::path::Path;

use anyhow::{bail, Context, Result};

/// `log Σ exp(logits) − logits[target]` in f64 (the teacher-forced NLL the
/// T1 bin established; f64 accumulation over f32 logits).
#[must_use]
pub fn nll(logits: &[f32], target: usize) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = logits.iter().map(|&l| (l as f64 - m).exp()).sum();
    m + z.ln() - logits[target] as f64
}

/// Index of the maximum logit (ties → lowest index; deterministic).
#[must_use]
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0
}

/// Rank of `target` in `logits` (1 = the argmax; ties count adversarially
/// as worse: strictly-greater count + 1).
#[must_use]
pub fn rank(logits: &[f32], target: usize) -> usize {
    let t = logits[target];
    1 + logits.iter().filter(|&&l| l > t).count()
}

// ── The frozen-table artifact ────────────────────────────────────────────

const TABLE_MAGIC: &[u8; 8] = b"RIVKTDMP";

/// Provenance header of a dumped [`FittedTokenTable`].
#[derive(Clone, Debug)]
pub struct TableMeta {
    pub n_layer: usize,
    pub width: usize,
    pub rows: usize,
    pub vocab: usize,
    /// The calibration token count that produced the table (provenance).
    pub cal_tokens: u64,
}

/// Dump a frozen table in token-major order with a COMPACT row remap (rows
/// renumbered 0..tracked in increasing-token order — `row()` semantics are
/// identical; the byte layout is not the original's, which is fine: the
/// artifact is consumed via `row()` only). BLAKE3-pinned; returns the hex.
///
/// `vocab` bounds the token-id enumeration (the table's `row_of_token` is
/// private; a probe over ids 0..vocab recovers the tracked set — any id ≥
/// vocab probes `None`, so a wrong (small) vocab SILENTLY TRUNCATES: pass
/// `config.vocab_size`, and the load-side shape check catches a mismatch
/// against a recorded header).
pub fn dump_fitted_table(
    table: &katgpt_core::fitted_value_table::FittedTokenTable,
    path: &Path,
    cal_tokens: u64,
    vocab: usize,
) -> Result<String> {
    let n_layer = table.n_layer();
    let width = table.width();
    // Recover the tracked set by probing (the compact remap is token-major).
    let mut row_of_token = vec![u32::MAX; vocab];
    let mut tracked: Vec<u32> = Vec::new();
    for (t, r) in row_of_token.iter_mut().enumerate() {
        if table.row(0, t as u32).is_some() {
            *r = tracked.len() as u32;
            tracked.push(t as u32);
        }
    }
    let rows = tracked.len();
    let mut buf: Vec<u8> = Vec::with_capacity(64 + 4 * vocab + 4 * n_layer * rows * width);
    buf.extend_from_slice(TABLE_MAGIC);
    buf.extend_from_slice(&n_layer.to_le_bytes());
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&(rows as u64).to_le_bytes());
    buf.extend_from_slice(&(vocab as u64).to_le_bytes());
    buf.extend_from_slice(&cal_tokens.to_le_bytes());
    for &r in &row_of_token {
        buf.extend_from_slice(&r.to_le_bytes());
    }
    // Data: (layer, token-major row). Layer 0 first, all tracked rows in
    // token order; then layer 1; etc. (the remap order of `row_of_token`).
    for l in 0..n_layer {
        for &t in &tracked {
            let row = table
                .row(l, t)
                .with_context(|| format!("table row ({l}, {t}) vanished mid-dump"))?;
            for v in row {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let digest = blake3::hash(&buf);
    buf.extend_from_slice(digest.as_bytes());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }
    std::fs::write(path, &buf).with_context(|| format!("write {}", path.display()))?;
    Ok(digest.to_hex().to_string())
}

/// Load a table dumped by [`dump_fitted_table`], verifying the BLAKE3 pin
/// and the expected shape. Returns the table + its header.
pub fn load_fitted_table(
    path: &Path,
    expect: Option<(usize, usize)>,
) -> Result<(katgpt_core::fitted_value_table::FittedTokenTable, TableMeta)> {
    let buf = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if buf.len() < 48 + 32 {
        bail!("{}: too small for a table artifact", path.display());
    }
    let body_len = buf.len() - 32;
    let want: [u8; 32] = buf[body_len..]
        .try_into()
        .expect("slice length checked above");
    let got = blake3::hash(&buf[..body_len]);
    if got.as_bytes() != &want {
        bail!(
            "{}: BLAKE3 mismatch (artifact corrupt or tampered)",
            path.display()
        );
    }
    let mut rd: &[u8] = &buf[..body_len];
    let magic: &[u8; 8] = rd[..8].try_into().expect("len checked");
    if magic != TABLE_MAGIC {
        bail!("{}: bad magic", path.display());
    }
    rd = &rd[8..];
    fn u64at(rd: &[u8]) -> (u64, &[u8]) {
        (u64::from_le_bytes(rd[..8].try_into().expect("8 bytes")), &rd[8..])
    }
    let (n_layer, rd) = u64at(rd);
    let (width, rd) = u64at(rd);
    let (rows, rd) = u64at(rd);
    let (vocab, rd) = u64at(rd);
    let (cal_tokens, rd) = u64at(rd);
    let (n_layer, width, rows, vocab) =
        (n_layer as usize, width as usize, rows as usize, vocab as usize);
    if let Some((en, ew)) = expect
        && (n_layer, width) != (en, ew)
    {
        bail!(
            "{}: shape ({n_layer}, {width}) != expected ({en}, {ew})",
            path.display()
        );
    }
    let need_ro = vocab * 4;
    if rd.len() < need_ro {
        bail!("{}: truncated row map", path.display());
    }
    let mut row_of_token = Vec::with_capacity(vocab);
    for i in 0..vocab {
        let b: [u8; 4] = rd[i * 4..i * 4 + 4].try_into().expect("4 bytes");
        row_of_token.push(u32::from_le_bytes(b));
    }
    let rd = &rd[need_ro..];
    let need_data = n_layer * rows * width * 4;
    if rd.len() != need_data {
        bail!(
            "{}: data {} bytes != expected {need_data}",
            path.display(),
            rd.len()
        );
    }
    let mut data = Vec::with_capacity(n_layer * rows * width);
    for c in rd.as_chunks::<4>().0 {
        data.push(f32::from_le_bytes(*c));
    }
    let table = katgpt_core::fitted_value_table::FittedTokenTable::from_rows(
        n_layer, width, row_of_token, data,
    );
    Ok((
        table,
        TableMeta {
            n_layer,
            width,
            rows,
            vocab,
            cal_tokens,
        },
    ))
}

// ── NIAH (the Bench-814 harness shape) ───────────────────────────────────

/// Diverse filler pool — verbatim from katgpt-rs `bench_685` (Bench 814's
/// harness; a single repeated sentence collapses retrieval realism).
pub const FILLER_POOL: [&str; 10] = [
    "The wind moved quietly across the open field where nothing of note had happened for many hours. ",
    "A cart loaded with winter grain creaked along the road past the old stone wall. ",
    "Somewhere beyond the ridge a hawk circled twice and drifted out of sight. ",
    "The innkeeper counted his barrels and marked the tally on a slate by the door. ",
    "Rain had fallen in the night and every leaf still held its bright beads of water. ",
    "Two shepherds argued mildly about the price of wool and then shared a pipe. ",
    "The blacksmith's hammer kept a slow patient rhythm that could be heard for a mile. ",
    "Children chased a dog between the market stalls until the bell rang for noon. ",
    "An old map hung on the tavern wall, its coastlines worn away by many fingers. ",
    "By evening the clouds thinned and the first stars appeared over the eastern hills. ",
];

/// One NIAH trial: filler + one needle + continuation tail (Bench 814's
/// single-key shape; the tail makes a BASE-model-style continuation the
/// scored event, which is deterministic to teacher-force).
#[derive(Clone, Debug)]
pub struct NiahTrial {
    /// `[BOS] + body + tail`, length exactly `seq_len`.
    pub tokens: Vec<usize>,
    /// The password's token subsequence (what the tail asks for).
    pub password_tokens: Vec<usize>,
    /// Position of the last tail token — the forward at this position
    /// produces the answer logits.
    pub answer_pos: usize,
    /// Needle depth (0..1) — where in the haystack the needle sits.
    pub depth: f32,
    /// The password string (e.g. "sunset1037") — the hit check reads it.
    pub password: String,
}

/// Build one trial at exactly `seq_len` tokens (BOS included).
///
/// Token-boundary honesty: the whole prompt text is encoded ONCE and the
/// needle/password ranges are located by prefix-encode diffs, then the
/// password subsequence is VERIFY-DECODED to contain the password (a
/// tokenizer whose Viterbi re-segments across the boundary fails LOUD at
/// build, never silently mid-measurement). Filler overshoots and the token
/// vector is cut inside the post-needle filler only (needle and tail stay
/// byte-aligned to their token ranges).
pub fn build_niah_trial(
    tok: &crate::tokenizer::SentencePieceGgufTokenizer,
    bos: usize,
    seq_len: usize,
    depth: f32,
    password: &str,
) -> Result<NiahTrial> {
    let needle_text = format!("The magic password is {password}. Remember it for later. ");
    let tail_text = " The magic password is".to_string();
    let filler = |chars: usize| -> String {
        let mut s = String::with_capacity(chars + FILLER_POOL[0].len());
        let mut i = 0usize;
        while s.len() < chars {
            s.push_str(FILLER_POOL[i % FILLER_POOL.len()]);
            i += 1;
        }
        s
    };
    // Body char budget: ~4.2 chars/token for this pool (the shrink path
    // below corrects overshoot; the GROW loop here corrects undershoot —
    // the measured pool ratio drifted past 4.2 (4365 chars → 972 tokens at
    // seq 1024, the run that died at `token budget 972 < target 1023`), so
    // a fixed estimate can undershoot and the builder must re-encode, never
    // bail). Bounded: 4 growth iterations is far past convergence.
    let body_target = seq_len - 1;
    let mut body_chars = seq_len * 42 / 10 + 64;
    let (text, toks, filler_a_len) = loop {
        let depth_chars = ((body_chars as f32) * depth.clamp(0.05, 0.9)) as usize;
        let text = format!(
            "{}{needle_text}{}{tail_text}",
            filler(depth_chars),
            filler(body_chars.saturating_sub(depth_chars) + 256)
        );
        let toks = tok.encode(&text);
        if toks.len() >= body_target {
            break (text, toks, filler(depth_chars).len());
        }
        body_chars = body_chars * body_target / toks.len().max(1) + 256;
    };
    // Prefix-diff ranges (p1 = needle start, p2 = needle end, p3 = tail
    // start — all token indices in `toks`, since SP encoding is
    // prefix-stable for this pool: the verify-decode below pins it).
    let p1 = tok.encode(&text[..filler_a_len]).len();
    let p2 = tok.encode(&text[..filler_a_len + needle_text.len()]).len();
    let p3 = tok.encode(&text[..text.len() - tail_text.len()]).len();
    let total = toks.len();
    if p3 < p2 || p2 < p1 || total < p3 + 1 {
        bail!("niah build: prefix-encode ranges not monotone (p1 {p1} p2 {p2} p3 {p3} total {total})");
    }
    // The password subsequence: locate inside the needle by its own
    // prefix-diff (the password text sits after "The magic password is ").
    let pw_off = "The magic password is ".len();
    let pw_start_in_text = filler_a_len + pw_off;
    let pw_end_in_text = pw_start_in_text + password.len();
    let pw1 = tok.encode(&text[..pw_start_in_text]).len();
    let pw2 = tok.encode(&text[..pw_end_in_text]).len();
    if pw2 < pw1 || pw2 > p2 {
        bail!("niah build: password range ({pw1}..{pw2}) outside needle ({p1}..{p2})");
    }
    let password_tokens: Vec<usize> = toks[pw1..pw2].to_vec();
    // VERIFY the subsequence spells the password (the boundary-alignment
    // proof; a re-segmenting tokenizer fails loud here).
    let decoded = tok.decode(&password_tokens);
    let digits = password.chars().filter(|c| c.is_ascii_digit()).collect::<String>();
    if !decoded.replace('▁', " ").contains(&digits) {
        bail!(
            "niah build: password token span decodes to '{decoded}', wants digits '{digits}' — \
             tokenizer re-segmented across the boundary"
        );
    }
    // Shrink to exactly seq_len − 1 (BOS prepends): cut inside the
    // post-needle filler only — the cut region is [p3_adjustable..] before
    // the tail. Cut count = total − (seq_len − 1). (The grow loop above
    // guarantees total ≥ body_target.)
    let drop = total - body_target;
    if p3 - p2 < drop {
        bail!(
            "niah build: post-needle filler ({}) too small to absorb the cut {drop}",
            p3 - p2
        );
    }
    let mut tokens = Vec::with_capacity(seq_len);
    tokens.push(bos);
    tokens.extend_from_slice(&toks[..p2]);
    tokens.extend_from_slice(&toks[p2 + drop..]);
    debug_assert_eq!(tokens.len(), seq_len);
    // Post-cut sanity: the tail must still decode to the tail text.
    let tail_decoded = tok.decode(&tokens[p3 - drop..]);
    if !tail_decoded.contains("magic password is") {
        bail!("niah build: tail mangled by the cut ('{tail_decoded}')");
    }
    Ok(NiahTrial {
        tokens,
        password_tokens,
        answer_pos: seq_len - 1,
        depth,
        password: password.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nll_and_rank_basics() {
        let logits = [1.0f32, 3.0, 2.0, 0.5];
        assert_eq!(argmax(&logits), 1);
        assert_eq!(rank(&logits, 1), 1);
        assert_eq!(rank(&logits, 2), 2);
        assert_eq!(rank(&logits, 3), 4);
        // NLL of a logit at the max: ln(1 + e^0 + e^-1 + e^-2.5)
        let want = (1.0f64 + (-1.0f64).exp() + (-2.0f64).exp() + (-2.5f64).exp()).ln();
        assert!((nll(&logits, 1) - want).abs() < 1e-12);
    }

    #[test]
    fn table_dump_load_round_trip() {
        use katgpt_core::fitted_value_table::FittedTokenTable;
        let dir = std::env::temp_dir().join(format!("vk_harness_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.bin");
        let row_of_token = vec![0u32, u32::MAX, 1];
        let data: Vec<f32> = (0..2 * 2 * 3).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let table = FittedTokenTable::from_rows(2, 3, row_of_token, data);
        let hex = dump_fitted_table(&table, &path, 1234, 3).unwrap();
        assert_eq!(hex.len(), 64);
        let (loaded, meta) = load_fitted_table(&path, Some((2, 3))).unwrap();
        assert_eq!(meta.rows, 2);
        assert_eq!(meta.cal_tokens, 1234);
        for l in 0..2 {
            assert_eq!(loaded.row(l, 0), table.row(l, 0));
            assert_eq!(loaded.row(l, 2), table.row(l, 2));
            assert!(loaded.row(l, 1).is_none());
        }
        // Tamper → loud.
        let mut bytes = std::fs::read(&path).unwrap();
        let n = bytes.len();
        bytes[n - 40] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        assert!(load_fitted_table(&path, None).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
