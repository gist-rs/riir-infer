//! Issue 929 T1.2/T1.3 Bonsai-2 leg — loop-alignment probe runner on the REAL
//! Ternary-Bonsai-2-27B (PQ2_0) checkpoint (DiscoLoop, Research 614,
//! arXiv:2607.00341).
//!
//! The katgpt-rs legs (Benches 927+930) proved the instrument on
//! kimi-k3-0.40B but the label axis was dead there (0% task accuracy — the
//! checkpoint's text head is content-untrained; UNDECIDABLE). This runner is
//! the capable-checkpoint leg the issue's T1.3/T1.4 adjudication needs:
//! Bonsai-2-27B is the league model, loaded PACKED (ternary bit-planes,
//! ~7.3 GB resident — `load_qwen_deltanet_ternary_weights_gguf`; a dequant
//! to f32 would need ~108 GB and is never attempted).
//!
//! The loop + probe machinery lives in
//! [`riir_infer_core::deltanet::loop_probe`] (shared with the machinery
//! gates — the Issue-022 T3.1 parity law). This bin adds: the fixture, the
//! prefill, the greedy continuation, and the report. One arm only: α=0
//! reentry (pure measurement, probe-first — the reinject arm is Phase 2,
//! closed unbuilt on a substrate with no correctness axis; re-opens with
//! the signal). Fresh cache+scratch per query — no GDN/KV state leaks
//! across queries.
//!
//! Report-only: CSV grid + JSON line + the T1.4 kill-bar verdict line. The
//! adjudication itself is recorded in the issue, not enforced here.
//!
//! Run (release mandatory; needs the folded-file rotation feature):
//! ```text
//! cargo run --release -p riir-infer-core \
//!   --features issue929_bonsai2_probe --bin issue929_bonsai2_loop_probe
//! ```
//! Model path: `B929_MODEL` env, else
//! `{CARGO_MANIFEST_DIR}/../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf`.
//! Scale knobs (smoke postures): `B929_LOOPS`, `B929_TWO_HOP`,
//! `B929_ONE_HOP`, `B929_FACTS`, `B929_ANSWER_TOKENS`, `B929_BOOTSTRAP`,
//! `B929_MAX_QUERIES`, `B929_OUT` (JSON path).

use anyhow::{Context, Result};
use katgpt_core::loop_alignment_probe::{
    FixtureSpec, bootstrap_auroc_ci, generate_two_hop_fixture,
};
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::forward_qwen_deltanet_ternary;
use riir_infer_core::deltanet::loop_probe::{
    LoopProbeScratch, argmax_of, embed_answer_token, looped_answer_position_probe,
};
use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::Config;

/// Caller-side loop depth at the answer position (loop 0 = the first stack
/// pass over the embedded last token). The kimi legs used K=2; the paper's
/// regime extends to 12 — 4 keeps the CPU bill honest while spanning the
/// confidence-amplification peak (k=2–3) observed in Bench 927.
const LOOPS: usize = 4;
/// Greedy continuation tokens generated for the answer check.
const ANSWER_TOKENS: usize = 8;
/// Bootstrap iterations per reported CI.
const BOOTSTRAP_ITERS: usize = 400;
/// Prompt facts per query (gold facts + same-pool distractors).
const FACTS_PER_PROMPT: usize = 12;
/// Two-hop queries per pool (the hard class).
const TWO_HOP_PER_POOL: usize = 60;
/// One-hop controls per pool (the easy class — AUROC needs both).
const ONE_HOP_PER_POOL: usize = 30;

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// One probe row at one loop of one query.
#[derive(Clone, Copy, Debug)]
struct Row {
    correct: bool,
    cos_argmax: f32,
    margin: f32,
    cos_bridge: f32,
    bridge_top1: bool,
}

/// One fixture query with its tokenized prompt.
struct Query {
    is_two_hop: bool,
    is_ood: bool,
    tokens: Vec<usize>,
    bridge_first: usize,
    answer: String,
}

struct Loaded {
    config: Config,
    weights: QwenDeltaNetTernaryWeights,
    rope_freq: RopeFreqTable,
    tokenizer: BpeTokenizer,
    untied: bool,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let t0 = std::time::Instant::now();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = std::env::var("B929_MODEL").unwrap_or_else(|_| {
        format!("{manifest_dir}/../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf")
    });

    // ── Load (header mmap for tokenizer + tied check, then packed weights) ──
    let gguf = GgufFile::open(std::path::Path::new(&model_path))
        .with_context(|| format!("open {model_path}"))?;
    let untied = gguf.tensor_info("output.weight").is_some();
    let tokenizer = BpeTokenizer::from_gguf(&gguf)?;
    eprintln!(
        "[load] tokenizer vocab={} bos={} eos={} | head: {}",
        tokenizer.vocab_size(),
        tokenizer.bos_id(),
        tokenizer.eos_id(),
        if untied {
            "UNTIED (separate output.weight)"
        } else {
            "TIED (head = wte clone)"
        },
    );

    let (config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(std::path::Path::new(&model_path))
            .with_context(|| format!("load packed ternary weights from {model_path}"))?;
    eprintln!(
        "[load] layers={} hidden={} vocab={} | globals packed {:.2} GB | {:.1}s",
        config.n_layer,
        config.n_embd,
        config.vocab_size,
        (weights.wte.encoded_bytes() + weights.lm_head.encoded_bytes()) as f64 / 1e9,
        t0.elapsed().as_secs_f32(),
    );
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let loaded = Loaded {
        config,
        weights,
        rope_freq,
        tokenizer,
        untied,
    };

    // ── Fixture (paper §2 shape, deterministic; env-scale knobs) ──────────
    let spec = FixtureSpec {
        facts_per_prompt: env_or("B929_FACTS", FACTS_PER_PROMPT),
        two_hop_queries: env_or("B929_TWO_HOP", TWO_HOP_PER_POOL),
        one_hop_queries: env_or("B929_ONE_HOP", ONE_HOP_PER_POOL),
        ..FixtureSpec::default()
    };
    let fixture = generate_two_hop_fixture(spec);
    let two_hop_n = fixture.items.iter().filter(|i| i.is_two_hop).count();
    eprintln!(
        "[fixture] {} queries ({two_hop_n} two-hop; unique comps/pool={})",
        fixture.items.len(),
        fixture.unique_compositions_per_pool,
    );

    let max_queries = env_or("B929_MAX_QUERIES", usize::MAX);
    let queries: Vec<Query> = fixture
        .items
        .iter()
        .take(max_queries)
        .map(|item| Query {
            is_two_hop: item.is_two_hop,
            is_ood: item.is_ood,
            // No BOS prepend — the qwen35 family serves completion-style
            // without a leading BOS (the llama.cpp convention for qwen).
            tokens: loaded.tokenizer.encode(&item.prompt),
            bridge_first: if item.bridge.is_empty() {
                0
            } else {
                loaded
                    .tokenizer
                    .encode(&item.bridge)
                    .first()
                    .copied()
                    .unwrap_or(0)
            },
            answer: item.answer.clone(),
        })
        .collect();
    let queries_max_prompt = queries.iter().map(|q| q.tokens.len()).max().unwrap_or(0);
    eprintln!("[fixture] max prompt {queries_max_prompt} tokens");

    let loops = env_or("B929_LOOPS", LOOPS);
    let answer_tokens = env_or("B929_ANSWER_TOKENS", ANSWER_TOKENS);
    let bootstrap_iters = env_or("B929_BOOTSTRAP", BOOTSTRAP_ITERS);

    let n = loaded.config.n_embd;
    let buf_len = loaded.config.vocab_size.max(n);
    let mut x = vec![0.0f32; buf_len];
    let mut probe_scratch = LoopProbeScratch::new(&loaded.config);
    let mut probe_rows = Vec::new();

    // rows[loop] — one Row per query, query order preserved.
    let mut rows: Vec<Vec<Row>> = vec![Vec::new(); loops];

    for (qi, q) in queries.iter().enumerate() {
        // Fresh cache + scratch per query: the GDN recurrent state and the KV
        // cache must not carry across queries (the Bench-930 discipline).
        let mut cache = HybridCache::with_layer_types(&loaded.config, &loaded.weights.layer_types);
        let mut scratch = HybridForwardScratch::new(&loaded.config);

        // Prefill all but the last prompt token through the STOCK forward
        // (no probing — the paper probes the ANSWER position).
        for (pos, &tok) in q.tokens[..q.tokens.len() - 1].iter().enumerate() {
            forward_qwen_deltanet_ternary(
                &mut x,
                &loaded.weights,
                &mut cache,
                tok,
                pos,
                &loaded.config,
                &mut scratch,
                &loaded.rope_freq,
            );
        }
        let answer_pos = q.tokens.len() - 1;
        let last = *q.tokens.last().expect("non-empty prompt");

        // Embed once, then K weight-shared stack passes with per-loop probes
        // (the shared machinery — same code the gates test).
        embed_answer_token(&mut x, &loaded.weights, last);
        let mut current = looped_answer_position_probe(
            &mut x,
            &loaded.weights,
            &mut cache,
            &mut scratch,
            &loaded.rope_freq,
            &loaded.config,
            answer_pos,
            loops,
            q.bridge_first,
            &mut probe_scratch,
            &mut probe_rows,
        );

        // Greedy continuation from the loop-refined readout, stock forward
        // per token (the loop intervention was answer-position only).
        let mut generated: Vec<usize> = Vec::with_capacity(answer_tokens);
        for i in 0..answer_tokens {
            if i > 0 {
                let logits = forward_qwen_deltanet_ternary(
                    &mut x,
                    &loaded.weights,
                    &mut cache,
                    current,
                    answer_pos + i,
                    &loaded.config,
                    &mut scratch,
                    &loaded.rope_freq,
                );
                current = argmax_of(logits);
            }
            generated.push(current);
        }
        let text = loaded.tokenizer.decode(&generated);
        let correct = text.contains(&q.answer);
        for (k, row) in probe_rows.iter().enumerate() {
            rows[k].push(Row {
                correct,
                cos_argmax: row.cos_alignment,
                margin: row.margin,
                cos_bridge: row.cos_bridge,
                bridge_top1: row.argmax == q.bridge_first,
            });
        }
        if qi < 3 {
            eprintln!(
                "[dbg] q{qi} two_hop={} answer={:?} gen={:?}",
                q.is_two_hop,
                q.answer,
                text.replace('\n', " ")
            );
        }
        if (qi + 1) % 10 == 0 {
            eprintln!(
                "[probe] {}/{} queries ({:.1}s)",
                qi + 1,
                queries.len(),
                t0.elapsed().as_secs_f32()
            );
        }
    }

    // ── Report ───────────────────────────────────────────────────────────
    println!("# issue929 Bonsai-2-27B (PQ2_0) loop-alignment probe");
    println!(
        "# model={model_path} | head={} | fixture: {} queries, facts={}, loops={loops}",
        if loaded.untied { "untied" } else { "tied" },
        queries.len(),
        spec.facts_per_prompt,
    );
    println!("# score cells are point/lb95 (stratified bootstrap, {bootstrap_iters} iters)");
    println!("loop,split,n,acc,bridge_top1,auroc_cos_argmax,auroc_margin,auroc_cos_bridge");
    type SplitFilter = fn(&Query) -> bool;
    let splits: [(&str, SplitFilter); 3] = [
        ("all", |_| true),
        ("two_hop", |q: &Query| q.is_two_hop),
        ("ood", |q: &Query| q.is_ood),
    ];
    let mut any_lb_ge_06 = false;
    let mut any_decidable = false;
    for (k, loop_rows) in rows.iter().enumerate() {
        for (name, keep) in splits {
            let idx: Vec<usize> = queries
                .iter()
                .enumerate()
                .filter(|(_, q)| keep(q))
                .map(|(i, _)| i)
                .collect();
            let cos: Vec<f32> = idx.iter().map(|&i| loop_rows[i].cos_argmax).collect();
            let mg: Vec<f32> = idx.iter().map(|&i| loop_rows[i].margin).collect();
            let cb: Vec<f32> = idx.iter().map(|&i| loop_rows[i].cos_bridge).collect();
            let labels: Vec<bool> = idx.iter().map(|&i| loop_rows[i].correct).collect();
            let acc = labels.iter().filter(|&&l| l).count();
            let bt1 = idx.iter().filter(|&&i| loop_rows[i].bridge_top1).count();
            for scores in [&cos, &mg, &cb] {
                let ci = bootstrap_auroc_ci(scores, &labels, bootstrap_iters, 929);
                if ci.point.is_nan() {
                    continue; // UNDECIDABLE cell — never folded into a verdict
                }
                any_decidable = true;
                if ci.ci_lo >= 0.6 {
                    any_lb_ge_06 = true;
                }
            }
            println!(
                "{k},{name},{},{},{},{},{},{}",
                idx.len(),
                fmt_frac(acc, idx.len()),
                fmt_frac(bt1, idx.len()),
                auroc_cell(&cos, &labels, bootstrap_iters),
                auroc_cell(&mg, &labels, bootstrap_iters),
                auroc_cell(&cb, &labels, bootstrap_iters),
            );
        }
    }
    // The T1.4 kill-bar verdict line (report-only; the issue records it).
    if !any_decidable {
        println!(
            "verdict: UNDECIDABLE — every AUROC cell has an empty class (label axis dead; the kimi-leg failure mode)"
        );
    } else if any_lb_ge_06 {
        println!(
            "verdict: SIGNAL — at least one (loop, split, signal) AUROC 95% CI LB >= 0.6 — T1.3 G1 gate PASSES; Phase 2 GOAT adjudication re-opens"
        );
    } else {
        println!(
            "verdict: KILL — every decidable AUROC 95% CI LB < 0.6 — the T1.4 kill criterion FIRES on a capable checkpoint"
        );
    }
    let acc_last = rows
        .last()
        .map(|r| r.iter().filter(|x| x.correct).count())
        .unwrap_or(0);
    let acc_n = rows.last().map_or(0, |r| r.len());
    println!(
        "json:{{\"model\":\"{model_path}\",\"untied\":{},\"queries\":{},\"loops\":{loops},\"acc_last\":{},\"elapsed_s\":{:.1}}}",
        loaded.untied,
        queries.len(),
        fmt_frac(acc_last, acc_n),
        t0.elapsed().as_secs_f32(),
    );
    if let Ok(out) = std::env::var("B929_OUT") {
        std::fs::write(
            &out,
            format!(
                "{{\"model\":\"{model_path}\",\"untied\":{},\"queries\":{},\"loops\":{loops},\"acc_last\":{},\"elapsed_s\":{:.1}}}\n",
                loaded.untied,
                queries.len(),
                fmt_frac(acc_last, acc_n),
                t0.elapsed().as_secs_f32(),
            ),
        )
        .with_context(|| format!("write {out}"))?;
    }
    Ok(())
}

/// `k/n` as a 4-decimal fraction (0.0 when the set is empty).
fn fmt_frac(k: usize, n: usize) -> String {
    if n == 0 {
        "0.0".to_string()
    } else {
        format!("{:.4}", k as f64 / n as f64)
    }
}

/// "point/lb95" or "nan" when either class is empty.
fn auroc_cell(scores: &[f32], labels: &[bool], iters: usize) -> String {
    let ci = bootstrap_auroc_ci(scores, labels, iters, 929);
    if ci.point.is_nan() {
        "nan".to_string()
    } else {
        format!("{:.4}/{:.4}", ci.point, ci.ci_lo)
    }
}
