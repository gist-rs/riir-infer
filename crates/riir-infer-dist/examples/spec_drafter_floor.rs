//! Spec-drafter corpus floors (riir-train Issue 654's v0 calibration rung).
//!
//! The speculative-fill verify path (`spec_fill`) is lossless for ANY
//! [`DraftSource`], so the only question a drafter must answer is
//! acceptance. Before any GPU spend on a trained MTP head, this instrument
//! prices the drafter CLASSES on a real corpus under the runner's EXACT
//! acceptance semantics ([`accepted_prefix`], the shifted alignment:
//! `d_0 == pending`, `d_j == argmax_{j-1}`):
//!
//! 1. `RepeatLastDrafter` — the shipped v0 floor (acceptance ≈ 0 by
//!    construction; `d_0` always matches, nothing past it does).
//! 2. `UnigramModal` — the modal token of the train split. The degenerate
//!    "no context" control: how much of acceptance is just the text's
//!    modal-token rate.
//! 3. `Bigram` — argmax bigram fit on the train split. The no-model-
//!    knowledge ceiling of the class: everything a corpus table can give.
//! 4. `BigramPromptFit` — the bigram table rebuilt from the CURRENT
//!    request's committed prefix only (grows as the stream walks). The
//!    serving-realistic zero-resource drafter: no corpus at boot, each
//!    request self-contained.
//!
//! The corpus's own next token stands in for the target model's greedy
//! argmax (the standard offline target-independent read; the TRUE
//! acceptance against Bonsai-2's own argmaxes is the GPU rung — see
//! riir-train Issue 654 for the hidden-state harness design). Train/read
//! split is frozen (fit on the first `--split` fraction, read on the
//! remainder — the read rows are never in the fit).
//!
//! Usage:
//! ```text
//! cargo run -p riir-infer-dist --example spec_drafter_floor -- \
//!     <model.gguf> <corpus-dir-or-txt> [--k 2,4,8] [--split 0.5] \
//!     [--max-tokens N] [--report PATH]
//! ```
//! Default corpus posture: any dir the shared `corpus_text` loader reads
//! (the reflex canonical dataset pages are the prose posture). Zero GPU,
//! seconds class; the GGUF is opened for the TOKENIZER only and dropped.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_dist::spec_fill::{DraftCtx, DraftSource, RepeatLastDrafter, accepted_prefix};

/// Modal-token drafter: the train split's most frequent token (first-seen
/// tie-break — deterministic). No context at all.
struct UnigramModal {
    token: u32,
}

impl DraftSource for UnigramModal {
    fn propose(&mut self, ctx: &DraftCtx<'_>) -> Vec<u32> {
        // d_0 echoes pending (accepted_prefix law), then the modal token.
        let last = ctx.committed.last().copied().unwrap_or(0);
        let mut out = vec![self.token; ctx.k];
        if ctx.k > 0 {
            out[0] = last;
        }
        out
    }
}

/// Bigram drafter over a frozen table: `argmax P(next | last)`, first-seen
/// tie-break. Out-of-table context falls back to the unigram mode.
#[derive(Clone)]
struct Bigram {
    /// prev -> (best_next, best_count), updated in stream order so the
    /// tie-break is first-seen by construction.
    table: BTreeMap<u32, (u32, u32)>,
    fallback: u32,
}

impl Bigram {
    /// Fit the table + unigram mode on a token slice in one pass.
    fn fit(tokens: &[u32]) -> Self {
        let mut table: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
        let mut uni: BTreeMap<u32, u32> = BTreeMap::new();
        for w in tokens.windows(2) {
            let (prev, next) = (w[0], w[1]);
            *uni.entry(next).or_insert(0) += 1;
            let best = table.entry(prev).or_insert((next, 0));
            if next == best.0 {
                best.1 += 1;
            } else if best.1 == 0 {
                // first pair for this context seeds the row
                *best = (next, 1);
            }
        }
        // Unigram mode: first-seen tie-break over stream order.
        let mut fallback = tokens.first().copied().unwrap_or(0);
        let mut best_n = 0u32;
        for (&t, &n) in uni.iter() {
            if n > best_n {
                best_n = n;
                fallback = t;
            }
        }
        Self { table, fallback }
    }

    fn next_of(&self, prev: u32) -> u32 {
        self.table
            .get(&prev)
            .map(|(n, _)| *n)
            .unwrap_or(self.fallback)
    }
}

impl DraftSource for Bigram {
    fn propose(&mut self, ctx: &DraftCtx<'_>) -> Vec<u32> {
        // The runner's shifted alignment (accepted_prefix) compares d_0
        // against PENDING — a predictive d_0 fails the round instantly. A
        // real drafter echoes pending first, then predicts.
        let last = ctx.committed.last().copied().unwrap_or(0);
        let mut out = Vec::with_capacity(ctx.k);
        out.push(last);
        let mut cur = last;
        for _ in 1..ctx.k {
            cur = self.next_of(cur);
            out.push(cur);
        }
        out
    }
}

/// Per-request bigram: the table is rebuilt from the CURRENT committed
/// prefix only (serving-realistic zero-resource posture — no corpus at
/// boot). Rebuilds incrementally: the table absorbs each newly committed
/// token once, so the amortized cost is O(1) per round.
struct BigramPromptFit {
    table: BTreeMap<u32, (u32, u32)>,
    /// Tokens already absorbed into the table (committed prefix length).
    absorbed: usize,
}

impl BigramPromptFit {
    fn new() -> Self {
        Self {
            table: BTreeMap::new(),
            absorbed: 0,
        }
    }

    fn absorb(&mut self, committed: &[u32]) {
        // New bigrams since the last round: pairs ending at each newly
        // committed position.
        while self.absorbed < committed.len() {
            let i = self.absorbed;
            if i == 0 {
                self.absorbed = 1;
                continue;
            }
            let (prev, next) = (committed[i - 1], committed[i]);
            let best = self.table.entry(prev).or_insert((next, 0));
            if next == best.0 {
                best.1 += 1;
            } else if best.1 == 0 {
                *best = (next, 1);
            }
            self.absorbed += 1;
        }
    }
}

impl DraftSource for BigramPromptFit {
    fn propose(&mut self, ctx: &DraftCtx<'_>) -> Vec<u32> {
        // Same d_0 == pending echo law as Bigram (accepted_prefix).
        self.absorb(ctx.committed);
        let last = ctx.committed.last().copied().unwrap_or(0);
        let mut out = Vec::with_capacity(ctx.k);
        out.push(last);
        let mut cur = last;
        for _ in 1..ctx.k {
            cur = self.table.get(&cur).map(|(n, _)| *n).unwrap_or(cur);
            out.push(cur);
        }
        out
    }
}

/// One drafter's read over the frozen segment at draft count `k`,
/// following the runner's round walk exactly: the cursor advances by the
/// round's committed tokens (full accept commits K new tokens; a partial
/// commits `m` after a re-forward — the corpus sim walks the same
/// increments so the round structure matches serving).
struct RoundStats {
    rounds: usize,
    /// accepted-prefix count per round (the runner's m).
    m_sum: usize,
    full_accepts: usize,
    /// draft hits on j>=1 slots (the honest next-token accuracy denominator).
    draft_slots: usize,
    draft_hits: usize,
    /// histogram of m (index = m, capped at k) — the raw distribution.
    m_hist: Vec<usize>,
    /// rounds where the drafter lost at d_0 (m == 0 — it did not even
    /// echo pending).
    m0: usize,
}

fn evaluate<D: DraftSource>(
    drafter: &mut D,
    tokens: &[u32],
    read_lo: usize,
    k: usize,
) -> RoundStats {
    let mut st = RoundStats {
        rounds: 0,
        m_sum: 0,
        full_accepts: 0,
        draft_slots: 0,
        draft_hits: 0,
        m_hist: vec![0; k + 1],
        m0: 0,
    };
    // The cursor walks the READ segment; a round needs k corpus tokens of
    // lookahead to build the pseudo-argmaxes (cursor+1 ..= cursor+k).
    let mut cursor = read_lo;
    while cursor + k < tokens.len() {
        let pending = tokens[cursor];
        // The runner's committed prefix ends with the pending prediction.
        // The corpus sim's "committed" is tokens[..=cursor] — the drafter
        // only reads its tail, so pass the window (the prompt-fit drafter
        // absorbs it; the frozen drafters ignore it beyond the last token).
        let ctx = DraftCtx {
            committed: &tokens[..=cursor],
            k,
        };
        let drafts = drafter.propose(&ctx);
        // Pseudo-argmaxes under the corpus model: the prediction after
        // consuming draft slot j is the corpus token at cursor+j+1.
        let argmaxes: Vec<u32> = tokens[cursor + 1..=cursor + k].to_vec();
        let m = accepted_prefix(pending, &drafts, &argmaxes);
        st.rounds += 1;
        st.m_sum += m;
        st.m_hist[m] += 1;
        if m == 0 {
            st.m0 += 1;
        }
        if m == k {
            st.full_accepts += 1;
        }
        // Honest next-token accuracy over the j>=1 slots (d_0 is free —
        // it echoes pending; counting it would inflate every drafter).
        // The prefix semantics: slots past the first miss are unscored.
        st.draft_slots += k - 1;
        st.draft_hits += m.saturating_sub(1);
        // Round walk: the cursor advances by the round's commit. Serving
        // commits min-verified tokens; the sim advances by m (the verified
        // prefix) — the same increments the runner's loop produces.
        let advance = m.max(1);
        cursor += advance;
    }
    st
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: spec_drafter_floor <model.gguf> <corpus-dir-or-txt> \
             [--k 2,4,8] [--split 0.5] [--max-tokens N] [--report PATH] \
             [--dump-tokens PREFIX]"
        );
        std::process::exit(2);
    }
    let gguf_path = PathBuf::from(&args[1]);
    let corpus_path = PathBuf::from(&args[2]);
    let mut ks: Vec<usize> = vec![2, 4, 8];
    let mut split: f64 = 0.5;
    let mut max_tokens: usize = 400_000;
    let mut report_path: Option<PathBuf> = None;
    let mut dump_prefix: Option<PathBuf> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--k" => {
                ks = args[i + 1]
                    .split(',')
                    .map(|s| s.parse().context("--k N"))
                    .collect::<Result<Vec<usize>>>()?;
                i += 2;
            }
            "--split" => {
                split = args[i + 1].parse().context("--split F")?;
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = args[i + 1].parse().context("--max-tokens N")?;
                i += 2;
            }
            "--report" => {
                report_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--dump-tokens" => {
                dump_prefix = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => bail!("unknown arg {other}"),
        }
    }
    if !(0.1..=0.9).contains(&split) {
        bail!("--split must sit in [0.1, 0.9]");
    }
    for &k in &ks {
        if k == 0 || k > 64 {
            bail!("--k values must sit in 1..=64 (MAX_VERIFY_K)");
        }
    }

    // ── Tokenizer from the GGUF (weights untouched; the file drops) ──
    let t0 = Instant::now();
    let gguf = GgufFile::open(&gguf_path).context("open gguf for tokenizer")?;
    let tok = SentencePieceGgufTokenizer::from_gguf(&gguf)
        .context("sentencepiece tokenizer from gguf")?;
    drop(gguf);
    println!(
        "# tokenizer: {} ({:.1}s, weights not loaded — this is a token-statistics floor)",
        gguf_path.display(),
        t0.elapsed().as_secs_f32()
    );

    // ── Corpus → one token stream, frozen train/read split ──
    let t1 = Instant::now();
    let text = load_corpus_text(&corpus_path)?;
    let mut tokens: Vec<u32> = tok
        .encode(&text)
        .into_iter()
        .map(|t| u32::try_from(t).unwrap_or(0))
        .collect();
    if tokens.len() < 10_000 {
        bail!(
            "corpus too small: {} tokens (need >= 10k for stable floors)",
            tokens.len()
        );
    }
    tokens.truncate(max_tokens);
    let read_lo = (tokens.len() as f64 * split) as usize;
    println!(
        "# corpus: {} chars -> {} tokens ({:.1}s) from {}",
        text.len(),
        tokens.len(),
        t1.elapsed().as_secs_f32(),
        corpus_path.display()
    );
    println!(
        "# split: fit [0, {}) = {} tokens | frozen read [{}, {}) = {} tokens",
        read_lo,
        read_lo,
        read_lo,
        tokens.len(),
        tokens.len() - read_lo
    );

    // ── riir-train Issue 654 GPU-rung feeder: dump the FROZEN split verbatim ──
    // Writes u32-LE token files so the 4090 capture consumes the byte-identical
    // stream this floor run measured (same tokenizer, same split point). The
    // digests printed here are the cross-check the capture run quotes.
    if let Some(prefix) = dump_prefix {
        let write_side = |p: PathBuf, rows: &[u32]| -> Result<String> {
            let mut bytes = Vec::with_capacity(rows.len() * 4);
            for t in rows {
                bytes.extend_from_slice(&t.to_le_bytes());
            }
            let digest = blake3::hash(&bytes).to_hex();
            std::fs::write(&p, &bytes).with_context(|| format!("write {}", p.display()))?;
            Ok(digest[..16].to_string())
        };
        let fit_p = prefix.with_extension("fit.tokens");
        let read_p = prefix.with_extension("read.tokens");
        let fit_digest = write_side(fit_p.clone(), &tokens[..read_lo])?;
        let read_digest = write_side(read_p.clone(), &tokens[read_lo..])?;
        println!(
            "# dump: {} ({} tok, blake3 {}) | {} ({} tok, blake3 {})",
            fit_p.display(),
            read_lo,
            fit_digest,
            read_p.display(),
            tokens.len() - read_lo,
            read_digest
        );
        return Ok(());
    }

    println!(
        "# box: {} cores, load {:#.2}; read {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        load_avg(),
        Instant::now().duration_since(t0).as_secs_f32()
    );

    // ── Fit the frozen drafters on the train split ──
    let uni_modal = Bigram::fit(&tokens[..read_lo]).fallback;
    let bigram = Bigram::fit(&tokens[..read_lo]);

    // ── Corpus composition probe: what the floors are made of ──
    // The same-token repeat rate explains a repeat-last-class floor; the
    // modal token's decoded form tells artifact (whitespace runs) from
    // language.
    let runs = tokens[..read_lo]
        .windows(2)
        .filter(|w| w[0] == w[1])
        .count();
    let run_rate = runs as f64 / read_lo.saturating_sub(1).max(1) as f64;
    let modal_freq = tokens[..read_lo]
        .iter()
        .filter(|&&t| t == uni_modal)
        .count();
    let modal_rate = modal_freq as f64 / read_lo.max(1) as f64;
    let modal_form = tok.decode(&[uni_modal as usize]);
    let top_rows: Vec<String> = bigram
        .table
        .iter()
        .take(5)
        .map(|(prev, (next, n))| {
            format!(
                "{} -> {} (n={})",
                tok.decode(&[*prev as usize]),
                tok.decode(&[*next as usize]),
                n
            )
        })
        .collect();
    println!(
        "# composition: same-token run rate {:#.3} | modal token {} ({:#.3} of train) \n#   top bigram rows: {}",
        run_rate,
        if modal_form.is_empty() {
            "<undecodable>"
        } else {
            &modal_form
        },
        modal_rate,
        top_rows.join(" | ")
    );

    // ── Evaluate every drafter x k on the frozen read ──
    let mut report: Vec<String> = Vec::new();
    let head = format!(
        "{:<16} {:>3} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "drafter", "K", "rounds", "E[m]", "P(full)", "acc(j>=1)", "tok/rt"
    );
    println!("#\n{head}\n{}", "-".repeat(head.len()));
    report.push(head.clone());

    for &k in &ks {
        let rows: Vec<(&str, RoundStats)> = {
            let v = vec![
                (
                    "repeat_last",
                    evaluate(&mut RepeatLastDrafter, &tokens, read_lo, k),
                ),
                (
                    "unigram_modal",
                    evaluate(&mut UnigramModal { token: uni_modal }, &tokens, read_lo, k),
                ),
                (
                    "bigram_frozen",
                    evaluate(&mut bigram.clone(), &tokens, read_lo, k),
                ),
                (
                    "bigram_promptfit",
                    evaluate(&mut BigramPromptFit::new(), &tokens, read_lo, k),
                ),
            ];
            v
        };
        for (name, st) in &rows {
            let e_m = st.m_sum as f64 / st.rounds.max(1) as f64;
            let p_full = st.full_accepts as f64 / st.rounds.max(1) as f64;
            let acc = st.draft_hits as f64 / st.draft_slots.max(1) as f64;
            let line = format!(
                "{:<16} {:>3} {:>8} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                name,
                k,
                st.rounds,
                e_m,
                p_full,
                acc,
                e_m - 1.0
            );
            println!("{line}");
            let hist: Vec<String> = st
                .m_hist
                .iter()
                .enumerate()
                .map(|(m, c)| format!("m={m}:{c}"))
                .collect();
            println!("    {:<16} m0={} | {}", "", st.m0, hist.join(" "));
            report.push(line);
        }
    }

    // ── Collapsed-unk view: this GGUF's SentencePiece encodes every space
    // as runs of the unk piece (id 0 — 3 per gap measured on ag_news), so
    // the raw stream is ~44% unk runs and every drafter's floor rides that
    // structure. Collapsing each unk run to ONE separator isolates the
    // linguistic bar a content-token drafter must clear. The raw view
    // above stays the wire-true read (the model really sees the runs).
    let mut collapsed: Vec<u32> = Vec::with_capacity(tokens.len());
    let mut prev_unk = false;
    for &t in &tokens {
        let is_unk = t == 0;
        if !(is_unk && prev_unk) {
            collapsed.push(t);
        }
        prev_unk = is_unk;
    }
    let read_lo_c = (collapsed.len() as f64 * split) as usize;
    let uni_modal_c = Bigram::fit(&collapsed[..read_lo_c]).fallback;
    let bigram_c = Bigram::fit(&collapsed[..read_lo_c]);
    println!(
        "#\n# collapsed-unk view: {} tokens (raw {}), read_lo {} — unk runs (id 0) squashed to one separator",
        collapsed.len(),
        tokens.len(),
        read_lo_c
    );
    println!("{head}\n{}", "-".repeat(head.len()));
    for &k in &ks {
        let rows: Vec<(&str, RoundStats)> = vec![
            (
                "repeat_last",
                evaluate(&mut RepeatLastDrafter, &collapsed, read_lo_c, k),
            ),
            (
                "unigram_modal",
                evaluate(
                    &mut UnigramModal { token: uni_modal_c },
                    &collapsed,
                    read_lo_c,
                    k,
                ),
            ),
            (
                "bigram_frozen",
                evaluate(&mut bigram_c.clone(), &collapsed, read_lo_c, k),
            ),
            (
                "bigram_promptfit",
                evaluate(&mut BigramPromptFit::new(), &collapsed, read_lo_c, k),
            ),
        ];
        for (name, st) in &rows {
            let e_m = st.m_sum as f64 / st.rounds.max(1) as f64;
            let p_full = st.full_accepts as f64 / st.rounds.max(1) as f64;
            let acc = st.draft_hits as f64 / st.draft_slots.max(1) as f64;
            let line = format!(
                "{:<16} {:>3} {:>8} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                name,
                k,
                st.rounds,
                e_m,
                p_full,
                acc,
                e_m - 1.0
            );
            println!("{line}");
            report.push(format!("collapsed {line}"));
        }
    }

    if let Some(p) = report_path {
        let mut out = String::from("# spec_drafter_floor JSON-free TSV report\n");
        for l in &report {
            out.push_str(l);
            out.push('\n');
        }
        std::fs::write(&p, out).context("write report")?;
        println!("# report: {}", p.display());
    }
    Ok(())
}

/// 1-minute load average (macOS/Linux) — the box-state line's figure.
fn load_avg() -> f64 {
    #[cfg(target_os = "macos")]
    {
        let mut loads = [0f64; 3];
        // SAFETY: plain FFI over a fixed-size C array out-param.
        unsafe {
            libc_getloadavg(loads.as_mut_ptr(), 3);
        }
        loads[0]
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|s| {
                s.split_whitespace()
                    .next()
                    .and_then(|v| v.parse::<f64>().ok())
            })
            .unwrap_or(0.0)
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn getloadavg(loadavg: *mut f64, nelem: i32) -> i32;
}

#[cfg(target_os = "macos")]
unsafe fn libc_getloadavg(ptr: *mut f64, nelem: i32) -> i32 {
    // SAFETY: plain FFI over a fixed-size C array out-param.
    unsafe { getloadavg(ptr, nelem) }
}
