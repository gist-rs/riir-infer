//! T5.0 GOAT — collapsed-vs-parent top-1 agreement (Issue 022, the T5.1
//! pre-registered budget on the qwen35 lane).
//!
//! Loads the PARENT checkpoint, runs the frozen token stream
//! teacher-forced (per-token forward, chunked with cache resets), records
//! the greedy argmax at every position, DROPS the parent, then repeats
//! with the COLLAPSED checkpoint and compares: agreement = matching
//! argmax positions / scored positions. Peak memory is one model.
//!
//! Pre-registered budget (`.issues/022` T5.1): absolute top-1 agreement
//! ≥ 0.9 — the parent trivially agrees with itself at 1.0, so the bar is
//! absolute, never a ratio.
//!
//! Usage:
//!   cargo run --release --features twt_bonsai --bin twt_goat_agreement -- \
//!     --parent ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!     --collapsed /tmp/twt_collapse_pq2_e005.gguf \
//!     --corpus .raw/twt/audition_calib.txt \
//!     --seq-len 512 --max-tokens 4096
//!
//! Box state: run on AC power; quote the tok/s lines beside the verdict.

#![cfg(feature = "twt_bonsai")]

use std::time::Instant;

use anyhow::{Context, Result, bail};

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, effective_rotary_dim,
};
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary_with_hook;
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;

/// The pre-registered absolute agreement bar (T5.1 lane budget).
const AGREEMENT_BAR: f64 = 0.9;

/// The parent arm's recording, cacheable across invocations (the parent
/// arm is identical for every collapsed point — one 55-min pass, many
/// collapsed arms).
#[derive(serde::Serialize, serde::Deserialize)]
struct ParentCache {
    parent: String,
    corpus: String,
    seq_len: usize,
    max_tokens: usize,
    argmax: Vec<u32>,
    truth: Vec<u32>,
}

struct ArmOut {
    /// Greedy argmax at every scored position (tokens[i] predicts
    /// argmax; scored = all positions except each chunk's first? No —
    /// every position of every chunk scores: token t at position p
    /// produces argmax that should predict token p+1 IN-CHUNK; the
    /// chunk's last position predicts outside the chunk and is scored
    /// against the true next token, which the cache reset makes unfair —
    /// so the LAST position of each chunk is dropped).
    argmax: Vec<u32>,
    /// The true next token at every scored position (for the parent-hit
    /// rate context column).
    truth: Vec<u32>,
    secs: f32,
}

fn run_arm(
    label: &str,
    gguf_path: &std::path::Path,
    chunks: &[Vec<usize>],
    seq_len: usize,
) -> Result<ArmOut> {
    let t0 = Instant::now();
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(gguf_path)
        .with_context(|| format!("load {}", gguf_path.display()))?;
    let load_s = t0.elapsed().as_secs_f32();
    let n_gdn = weights
        .layer_types
        .iter()
        .filter(|&&t| riir_infer_core::types::DeltaNetLayerType::DeltaNet == t)
        .count();
    eprintln!(
        "[goat] {label}: {} layers (gdn {n_gdn}, attn {}) | load {load_s:.1}s",
        config.n_layer,
        config.n_layer - n_gdn,
    );

    // Cap the KV window at the chunk length (the row_logit_floor law).
    config.block_size = seq_len;
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&config);
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];

    let mut argmax: Vec<u32> = Vec::new();
    let mut truth: Vec<u32> = Vec::new();
    let t1 = Instant::now();
    for (ci, chunk) in chunks.iter().enumerate() {
        cache.reset();
        // Score positions 0..len-1 (the last position's prediction lands
        // outside the chunk — dropped, see the struct doc).
        for p in 0..chunk.len() - 1 {
            forward_qwen_deltanet_ternary_with_hook(
                &mut x,
                &weights,
                &mut cache,
                chunk[p],
                p,
                &config,
                &mut scratch,
                &rope_freq,
                None,
                None,
                None,
                None,
            );
            let logits = &x[..config.vocab_size];
            let am = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .expect("non-empty logits");
            argmax.push(am as u32);
            truth.push(chunk[p + 1] as u32);
        }
        if (ci + 1) % 4 == 0 || ci + 1 == chunks.len() {
            let el = t1.elapsed().as_secs_f32();
            let done: usize = chunks[..=ci].iter().map(|c| c.len() - 1).sum();
            eprintln!(
                "[goat] {label}: chunk {}/{} | {} positions | {:.2} tok/s | eta {:.0} min",
                ci + 1,
                chunks.len(),
                done,
                done as f32 / el.max(1e-6),
                (chunks.len() - ci - 1) as f32
                    * (seq_len - 1) as f32
                    / (done as f32 / el.max(1e-6))
                    / 60.0,
            );
        }
    }
    let secs = t1.elapsed().as_secs_f32();
    let positions = argmax.len();
    eprintln!(
        "[goat] {label}: {positions} positions in {secs:.0}s ({:.2} tok/s)",
        positions as f32 / secs.max(1e-6),
    );
    drop(cache);
    drop(scratch);
    drop(weights);
    Ok(ArmOut { argmax, truth, secs })
}

fn main() -> Result<()> {
    let mut parent = None;
    let mut collapsed: Vec<std::path::PathBuf> = Vec::new();
    let mut corpus = None;
    let mut seq_len = 512usize;
    let mut max_tokens = 4096usize;
    let mut cache_path: Option<std::path::PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--parent" => parent = Some(std::path::PathBuf::from(args.next().expect("path"))),
            "--collapsed" => {
                collapsed.push(std::path::PathBuf::from(args.next().expect("path")))
            }
            "--corpus" => corpus = Some(std::path::PathBuf::from(args.next().expect("path"))),
            "--seq-len" => seq_len = args.next().expect("n").parse()?,
            "--max-tokens" => max_tokens = args.next().expect("n").parse()?,
            "--cache" => cache_path = Some(std::path::PathBuf::from(args.next().expect("path"))),
            other => bail!("unknown arg {other}"),
        }
    }
    let parent = parent.context("--parent is required")?;
    if collapsed.is_empty() {
        bail!("at least one --collapsed is required (repeatable — the parent arm runs once)");
    }
    let corpus = corpus.context("--corpus is required")?;
    assert!(seq_len >= 2, "--seq-len must be >= 2 (one scored position min)");

    // ── the frozen token stream (byte-identical for both arms) ──
    let text = load_corpus_text(&corpus)?;
    let tok = {
        let gguf = GgufFile::open(&parent).context("re-open parent for tokenizer")?;
        BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?
    };
    let all = tok.encode(&text);
    let take = all.len().min(max_tokens);
    let tokens: Vec<usize> = all[..take].to_vec();
    let chunks: Vec<Vec<usize>> = tokens.chunks(seq_len).map(<[usize]>::to_vec).collect();
    let scored: usize = chunks.iter().map(|c| c.len() - 1).sum();
    eprintln!(
        "[goat] corpus {} chars → {} tokens → {} chunks × ≤{seq_len} ({scored} scored positions)",
        text.len(),
        tokens.len(),
        chunks.len(),
    );

    // ── arm 1: parent (run once; every collapsed arm follows it) ──
    // Cache: the parent arm is identical for every collapsed point, so a
    // --cache file (params-keyed) replays it. A cache hit prints LOUD —
    // a silent replay would be indistinguishable from a fresh pass.
    let cached = cache_path.as_ref().and_then(|p| {
        let Ok(text) = std::fs::read_to_string(p) else {
            return None;
        };
        let c: ParentCache = serde_json::from_str(&text).ok()?;
        (c.parent == parent.display().to_string()
            && c.corpus == corpus.display().to_string()
            && c.seq_len == seq_len
            && c.max_tokens == max_tokens)
            .then_some(c)
    });
    let (base, base_hit, base_secs) = match cached {
        Some(c) => {
            eprintln!(
                "[goat] parent: CACHE HIT ({parent:?} @ seq {seq_len} × {max_tokens}) — replaying {} recorded positions",
                c.argmax.len()
            );
            let hit = c.argmax.iter().zip(c.truth.iter()).filter(|(a, t)| a == t).count();
            (
                ArmOut { argmax: c.argmax, truth: c.truth, secs: 0.0 },
                hit,
                0.0,
            )
        }
        None => {
            let base = run_arm("parent", &parent, &chunks, seq_len)?;
            let hit = base
                .argmax
                .iter()
                .zip(base.truth.iter())
                .filter(|(a, t)| a == t)
                .count();
            if let Some(p) = &cache_path {
                let c = ParentCache {
                    parent: parent.display().to_string(),
                    corpus: corpus.display().to_string(),
                    seq_len,
                    max_tokens,
                    argmax: base.argmax.clone(),
                    truth: base.truth.clone(),
                };
                std::fs::write(p, serde_json::to_string(&c).unwrap())
                    .with_context(|| format!("write cache {p:?}"))?;
                eprintln!("[goat] parent: cached to {p:?}");
            }
            let secs = base.secs;
            (base, hit, secs)
        }
    };

    let n = base.argmax.len();
    let mut any_pass = false;
    for path in &collapsed {
        let arm2 = run_arm("collapsed", path, &chunks, seq_len)?;
        let arm2_hit = arm2
            .argmax
            .iter()
            .zip(arm2.truth.iter())
            .filter(|(a, t)| a == t)
            .count();

        // ── the verdict ──
        assert_eq!(n, arm2.argmax.len(), "arm position counts diverged");
        let agree = base
            .argmax
            .iter()
            .zip(arm2.argmax.iter())
            .filter(|(a, b)| a == b)
            .count();
        let agreement = agree as f64 / n as f64;
        let first_div = base
            .argmax
            .iter()
            .zip(arm2.argmax.iter())
            .position(|(a, b)| a != b);

        println!("# twt_goat_agreement — T5.0 collapsed-vs-parent (Issue 022, T5.1 budget)");
        println!("# parent:    {}", parent.display());
        println!("# collapsed: {}", path.display());
        println!("# corpus: {} ({} tokens, {} chunks × ≤{seq_len})", corpus.display(), tokens.len(), chunks.len());
        println!("# scored positions: {n}");
        println!("# parent top-1 hit rate (context):    {:.4}", base_hit as f64 / n as f64);
        println!("# collapsed top-1 hit rate:           {:.4}", arm2_hit as f64 / n as f64);
        println!("# parent-vs-collapsed argmax agreement: {:.4} ({agree}/{n})", agreement);
        println!("# first divergence position: {:?}", first_div);
        println!(
            "# arm wall: parent {:.0}s, collapsed {:.0}s",
            base_secs, arm2.secs
        );
        let ok = agreement >= AGREEMENT_BAR;
        any_pass |= ok;
        println!(
            "# verdict: {} (bar {AGREEMENT_BAR} absolute, pre-registered)",
            if ok { "PASS" } else { "FAIL" }
        );
    }
    if !any_pass {
        std::process::exit(1);
    }
    Ok(())
}
