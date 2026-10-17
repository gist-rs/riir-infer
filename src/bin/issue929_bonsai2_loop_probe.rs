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
//!
//! `B929_CHAT=1` wraps each query in the qwen3 chat template with the
//! empty-think prefix (`<think>\n\n</think>\n\n`) — the smoke posture that
//! makes the chat/reasoning-tuned checkpoint answer IN the answer window
//! instead of closing the document past `<|im_end|>` and opening its own
//! think block (the 8-query completion-style smoke measured 0/8 two-hop
//! correct; the special tokens ride the GGUF control-token matcher, one id
//! each).
//!
//! `B929_PHRASE=standard` renders fixture facts as `"The {r} of {a} is {b}."`
//! (standard English composition) instead of the default possessive
//! `"{a} is the {r} of {b}."` — the Bench-933 phrasing-inversion A/B knob;
//! labels/bridges/fact windows are phrasing-invariant (same seed → same
//! queries, paired arms). `B929_OUT` (JSON path) carries per-query rows
//! (answer/gen/correct + every probe signal per loop) for paired analysis.
//!
//! Sharding: `B929_SKIP` + `B929_MAX_QUERIES` select a contiguous query
//! range (the fixture is seed-deterministic, so shards partition ONE run);
//! `B929_MERGE=a.json,b.json,...` skips the model entirely and re-reports
//! the grid + verdict from the shards' `B929_OUT` rows, ordered by global
//! query index — the bootstrap is seed-fixed, so a merge of a full
//! partition prints exactly what the unsharded run would.
//!
//! Scoring (think-aware, the full-run protocol's prerequisite): generation
//! stops at `<|im_end|>` (past it is post-answer noise), and the `contains`
//! correctness check reads only the ANSWER window — the tokens after the
//! LAST `</think>`, cut back at a `<think>` re-open. A gold-fact quotation
//! mid-doubt is not an answer, and a generation that opened `<think>` and
//! never closed it scores nothing. Per-query `B929_OUT` rows carry both the
//! raw and the scored text (`gen` / `gen_scored`) plus `stopped_im_end` +
//! `n_gen_tokens`, so a re-report never guesses which surface was scored.

use anyhow::{Context, Result};
use katgpt_core::loop_alignment_probe::{
    FactPhrase, FixtureSpec, bootstrap_auroc_ci, generate_two_hop_fixture,
};
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::forward_qwen_deltanet_ternary;
use riir_infer_core::deltanet::loop_probe::{
    LoopProbeScratch, argmax_of, embed_answer_token, looped_answer_position_probe,
};
use riir_infer_core::deltanet::ternary_forward::qwen_deltanet_ternary_layer_body;
use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::{Config, DeltaNetLayerType};

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

/// The scored answer window of a greedy generation: the tokens after the
/// LAST `</think>`, cut back at a `<think>` re-open inside that tail. A
/// generation that opened `<think>` and never closed it scores nothing (an
/// answer that never landed is not an answer); a generation with no markers
/// at all scores whole (the legacy short-budget posture). When `</think>`
/// is not resolvable as a single vocab id the raw generation is the scored
/// surface (the fallback the header line discloses).
fn scored_tail(
    generated: &[usize],
    think_open_id: Option<usize>,
    think_close_id: Option<usize>,
) -> &[usize] {
    let Some(close_id) = think_close_id else {
        return generated;
    };
    let tail_start = generated
        .iter()
        .rposition(|&t| t == close_id)
        .map_or(0, |p| p + 1);
    let rest = &generated[tail_start..];
    let tail_end = think_open_id
        .and_then(|open_id| rest.iter().position(|&t| t == open_id))
        .map_or(rest.len(), |p| tail_start + p);
    &generated[tail_start..tail_end]
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

/// The split membership the report filters on (shared by the live run and
/// the `B929_MERGE` re-report).
#[derive(Clone, Copy, Debug)]
struct Split {
    is_two_hop: bool,
    is_ood: bool,
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
    if let Ok(paths) = std::env::var("B929_MERGE") {
        return merge(&paths);
    }
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
    // Think-aware scoring ids (None = the marker is not a single vocab
    // entry on this checkpoint — scoring then falls back to the raw
    // generation, disclosed in the header line).
    let im_end_id = loaded.tokenizer.token_to_id("<|im_end|>");
    let think_open_id = loaded.tokenizer.token_to_id("<think>");
    let think_close_id = loaded.tokenizer.token_to_id("</think>");

    // ── Fixture (paper §2 shape, deterministic; env-scale knobs) ──────────
    // B929_PHRASE=standard: the standard-semantics fact template (A/B knob;
    // default = the as-landed possessive phrasing the 927/930/933/934
    // records ride — never redefine the shared instrument mid-adjudication).
    let phrase = match std::env::var("B929_PHRASE") {
        Ok(v) if v == "standard" => FactPhrase::Standard,
        _ => FactPhrase::Possessive,
    };
    let spec = FixtureSpec {
        facts_per_prompt: env_or("B929_FACTS", FACTS_PER_PROMPT),
        two_hop_queries: env_or("B929_TWO_HOP", TWO_HOP_PER_POOL),
        one_hop_queries: env_or("B929_ONE_HOP", ONE_HOP_PER_POOL),
        phrase,
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
    let skip_queries = env_or("B929_SKIP", 0);
    let chat = std::env::var("B929_CHAT").is_ok_and(|v| v == "1");
    let queries: Vec<Query> = fixture
        .items
        .iter()
        .skip(skip_queries)
        .take(max_queries)
        .map(|item| {
            // Chat-template elicitation (B929_CHAT=1): the completion-style
            // axis reads 0% on the chat-tuned checkpoint (the model closes
            // the document and opens <think>). The empty-think prefix is
            // qwen3's non-thinking convention; specials are single ids via
            // the GGUF control-token matcher.
            let prompt = if chat {
                format!(
                    "<|im_start|>system\nAnswer with only the short answer. Do not think.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    item.prompt
                )
            } else {
                item.prompt.clone()
            };
            Query {
                is_two_hop: item.is_two_hop,
                is_ood: item.is_ood,
                // No BOS prepend — the qwen35 family serves completion-style
                // without a leading BOS (the llama.cpp convention for qwen).
                tokens: loaded.tokenizer.encode(&prompt),
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
            }
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
    // Per-query generated text (query order preserved) for the B929_OUT
    // per-query rows — paired A/B analysis reads these.
    let mut gens: Vec<String> = Vec::with_capacity(queries.len());
    let mut gens_scored: Vec<String> = Vec::with_capacity(queries.len());
    let mut stopped: Vec<bool> = Vec::with_capacity(queries.len());
    let mut gen_lens: Vec<usize> = Vec::with_capacity(queries.len());

    for (qi, q) in queries.iter().enumerate() {
        // Fresh cache + scratch per query: the GDN recurrent state and the KV
        // cache must not carry across queries (the Bench-930 discipline).
        // Bounded KV capacity (the 4090 session's measured smoke OOM, 61fc36f):
        // the GGUF's block_size IS the model's full context_length, so the
        // unbounded ctor pre-allocates context-length KV for ALL 64 layers
        // (>100 GiB commit demand) and aborts beside the loaded weights. This
        // run writes positions 0..prompt_len + answer_tokens only (the K
        // stack passes stay AT answer_pos) - cap there plus slack.
        let kv_cap = queries_max_prompt + answer_tokens + 16;
        let mut cache = HybridCache::with_layer_types_bounded(
            &loaded.config,
            &loaded.weights.layer_types,
            kv_cap,
        );
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

        // Label-axis integrity: the loop passes ADVANCE the GDN recurrent/conv
        // state once per pass and overwrite the answer position's KV rows —
        // "that IS the loop" (loop_probe.rs). The reinject arm is Phase 2
        // (closed unbuilt), so the ACCURACY continuation must not read the
        // K-pass state: snapshot post-prefill, probe, restore, then re-run
        // ONE stock pass on the original embedding — the exact state a stock
        // decode of the prompt's last token would hold (K=1 is gate-pinned
        // bit-identical to the stock forward).
        let gdn_snapshot: Vec<(Vec<f32>, Vec<f32>)> = cache
            .deltanet_state
            .recurrent_states
            .iter()
            .zip(&cache.deltanet_state.conv_states)
            .map(|(r, c)| (r.clone(), c.clone()))
            .collect();
        let kv_snapshot = cache
            .kv_cache
            .snapshot(answer_pos + 1, &loaded.config);

        // Embed once, then K weight-shared stack passes with per-loop probes
        // (the shared machinery — same code the gates test).
        embed_answer_token(&mut x, &loaded.weights, last);
        let answer_emb = x[..n].to_vec();
        let probed_first = looped_answer_position_probe(
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

        // Rebuild the clean stock post-answer-token state: pre-loop GDN/KV
        // back, original embedding back, ONE stock stack pass.
        for (li, (r, c)) in gdn_snapshot.iter().enumerate() {
            cache.deltanet_state.recurrent_states[li].copy_from_slice(r);
            cache.deltanet_state.conv_states[li].copy_from_slice(c);
        }
        cache.kv_cache.restore(&kv_snapshot, &loaded.config);
        x[..n].copy_from_slice(&answer_emb);
        for (li, layer_weights) in loaded.weights.layers.iter().enumerate() {
            let is_linear = loaded.weights.layer_types[li] == DeltaNetLayerType::DeltaNet;
            qwen_deltanet_ternary_layer_body(
                &mut x,
                layer_weights,
                is_linear,
                &mut cache.deltanet_state.recurrent_states[li],
                &mut cache.deltanet_state.conv_states[li],
                &mut cache.kv_cache.layers[li],
                answer_pos,
                li,
                &loaded.config,
                &mut scratch,
                &loaded.rope_freq,
                loaded.weights.rotation.as_ref(),
                None,
                None,
                None,
            );
        }
        let mut current = probed_first;

        // Greedy continuation from the loop-refined readout, stock forward
        // per token (the loop intervention was answer-position only). The
        // chat turn's end token stops generation — past it is post-answer
        // noise; ANSWER_TOKENS is an upper bound, not a fixed spend.
        let mut generated: Vec<usize> = Vec::with_capacity(answer_tokens);
        let mut stopped_im_end = false;
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
            if Some(current) == im_end_id {
                stopped_im_end = true;
                break;
            }
            generated.push(current);
        }
        // Think-aware scoring: `contains` reads the ANSWER window, never the
        // whole generation — a gold-fact quotation mid-doubt is not an
        // answer (the scored_tail contract is in the helper's doc).
        let text = loaded.tokenizer.decode(&generated);
        let scored_text = loaded.tokenizer.decode(scored_tail(
            &generated,
            think_open_id,
            think_close_id,
        ));
        gens.push(text.clone());
        gens_scored.push(scored_text.clone());
        stopped.push(stopped_im_end);
        gen_lens.push(generated.len());
        let correct = scored_text.contains(&q.answer);
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
        "# model={model_path} | head={} | fixture: {} queries, facts={}, loops={loops}{}{}{}",
        if loaded.untied { "untied" } else { "tied" },
        queries.len(),
        spec.facts_per_prompt,
        if chat { " | elicitation=chat" } else { "" },
        match phrase {
            FactPhrase::Standard => " | phrase=standard",
            FactPhrase::Possessive => "",
        },
        if think_close_id.is_some() {
            " | score=think-aware"
        } else {
            " | score=raw"
        },
    );
    println!("# score cells are point/lb95 (stratified bootstrap, {bootstrap_iters} iters)");
    let splits: Vec<Split> = queries
        .iter()
        .map(|q| Split {
            is_two_hop: q.is_two_hop,
            is_ood: q.is_ood,
        })
        .collect();
    report_grid(&rows, &splits, bootstrap_iters);
    let acc_last = rows
        .last()
        .map(|r| r.iter().filter(|x| x.correct).count())
        .unwrap_or(0);
    let acc_n = rows.last().map_or(0, |r| r.len());
    let phrase_name = match phrase {
        FactPhrase::Standard => "standard",
        FactPhrase::Possessive => "possessive",
    };
    println!(
        "json:{{\"model\":\"{model_path}\",\"untied\":{},\"queries\":{},\"loops\":{loops},\"phrase\":\"{phrase_name}\",\"acc_last\":{},\"elapsed_s\":{:.1}}}",
        loaded.untied,
        queries.len(),
        fmt_frac(acc_last, acc_n),
        t0.elapsed().as_secs_f32(),
    );
    if let Ok(out) = std::env::var("B929_OUT") {
        // Per-query rows (paired-analysis surface): answer/gen/correct plus
        // the cos-bridge readout at every loop — additive reporting, zero
        // measurement change.
        let per_query: Vec<serde_json::Value> = (0..queries.len())
            .map(|qi| {
                serde_json::json!({
                    "qi": qi,
                    "two_hop": queries[qi].is_two_hop,
                    "ood": queries[qi].is_ood,
                    "answer": queries[qi].answer,
                    "gen": gens.get(qi).cloned().unwrap_or_default(),
                    "gen_scored": gens_scored.get(qi).cloned().unwrap_or_default(),
                    "stopped_im_end": stopped.get(qi).copied().unwrap_or(false),
                    "n_gen_tokens": gen_lens.get(qi).copied().unwrap_or(0),
                    "correct": rows.last().is_some_and(|r| r[qi].correct),
                    "global_qi": skip_queries + qi,
                    "cos_bridge": rows.iter().map(|r| r[qi].cos_bridge).collect::<Vec<_>>(),
                    "cos_argmax": rows.iter().map(|r| r[qi].cos_argmax).collect::<Vec<_>>(),
                    "margin": rows.iter().map(|r| r[qi].margin).collect::<Vec<_>>(),
                    "bridge_top1": rows.iter().map(|r| r[qi].bridge_top1).collect::<Vec<_>>(),
                })
            })
            .collect();
        let payload = serde_json::json!({
            "model": model_path,
            "untied": loaded.untied,
            "queries": queries.len(),
            "loops": loops,
            "phrase": phrase_name,
            "chat": chat,
            "skip": skip_queries,
            "bootstrap": bootstrap_iters,
            "acc_last": fmt_frac(acc_last, acc_n),
            "elapsed_s": t0.elapsed().as_secs_f32(),
            "per_query": per_query,
        });
        std::fs::write(&out, format!("{payload}\n"))
            .with_context(|| format!("write {out}"))?;
    }
    Ok(())
}

/// The per-(loop, split) AUROC grid + the T1.4 kill-bar verdict line
/// (report-only; the issue records it). `rows[loop][query]`, `splits[query]`.
fn report_grid(rows: &[Vec<Row>], splits: &[Split], bootstrap_iters: usize) {
    println!("loop,split,n,acc,bridge_top1,auroc_cos_argmax,auroc_margin,auroc_cos_bridge");
    type SplitFilter = fn(&Split) -> bool;
    let split_defs: [(&str, SplitFilter); 3] = [
        ("all", |_| true),
        ("two_hop", |q: &Split| q.is_two_hop),
        ("ood", |q: &Split| q.is_ood),
    ];
    let mut any_lb_ge_06 = false;
    let mut any_decidable = false;
    for (k, loop_rows) in rows.iter().enumerate() {
        for (name, keep) in split_defs {
            let idx: Vec<usize> = splits
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
    match (any_decidable, any_lb_ge_06) {
        (false, _) => println!(
            "verdict: UNDECIDABLE — every AUROC cell has an empty class (label axis dead; the kimi-leg failure mode)"
        ),
        (true, true) => println!(
            "verdict: SIGNAL — at least one (loop, split, signal) AUROC 95% CI LB >= 0.6 — T1.3 G1 gate PASSES; Phase 2 GOAT adjudication re-opens"
        ),
        (true, false) => println!(
            "verdict: KILL — every decidable AUROC 95% CI LB < 0.6 — the T1.4 kill criterion FIRES on a capable checkpoint"
        ),
    }
}

/// `B929_MERGE`: re-report the grid from shard `B929_OUT` files. Refuses a
/// duplicated or gapped global query range, and shards whose posture
/// (loops / phrase / chat / bootstrap) disagree — a merge across postures
/// would report a run that never happened.
fn merge(paths: &str) -> Result<()> {
    let mut per_query: Vec<(usize, serde_json::Value)> = Vec::new();
    let mut posture: Option<(u64, String, bool, u64)> = None;
    for path in paths.split(',').filter(|p| !p.is_empty()) {
        let text = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parse {path}"))?;
        let this = (
            v["loops"].as_u64().context("loops")?,
            v["phrase"].as_str().context("phrase")?.to_string(),
            v["chat"].as_bool().context("chat")?,
            v["bootstrap"].as_u64().context("bootstrap (pre-merge shard file?)")?,
        );
        match &posture {
            None => posture = Some(this),
            Some(p) if *p != this => anyhow::bail!("{path}: posture {this:?} != {p:?}"),
            Some(_) => {}
        }
        for q in v["per_query"].as_array().context("per_query")? {
            let gqi = q["global_qi"].as_u64().context("global_qi")? as usize;
            per_query.push((gqi, q.clone()));
        }
    }
    let (loops, phrase, chat, bootstrap_iters) = posture.context("B929_MERGE named no files")?;
    per_query.sort_by_key(|(g, _)| *g);
    for (i, (g, _)) in per_query.iter().enumerate() {
        anyhow::ensure!(*g == i, "global query range not a partition: index {i} holds {g}");
    }
    let loops = loops as usize;
    let mut rows: Vec<Vec<Row>> = vec![Vec::with_capacity(per_query.len()); loops];
    let mut splits = Vec::with_capacity(per_query.len());
    let f32_at = |q: &serde_json::Value, key: &str, k: usize| -> Result<f32> {
        Ok(q[key][k].as_f64().with_context(|| format!("{key}[{k}]"))? as f32)
    };
    for (_, q) in &per_query {
        let correct = q["correct"].as_bool().context("correct")?;
        splits.push(Split {
            is_two_hop: q["two_hop"].as_bool().context("two_hop")?,
            is_ood: q["ood"].as_bool().context("ood")?,
        });
        for (k, loop_rows) in rows.iter_mut().enumerate() {
            loop_rows.push(Row {
                correct,
                cos_argmax: f32_at(q, "cos_argmax", k)?,
                margin: f32_at(q, "margin", k)?,
                cos_bridge: f32_at(q, "cos_bridge", k)?,
                bridge_top1: q["bridge_top1"][k].as_bool().context("bridge_top1")?,
            });
        }
    }
    println!(
        "# issue929 B929_MERGE re-report | {} queries from {} shard file(s), loops={loops}{}{}",
        per_query.len(),
        paths.split(',').filter(|p| !p.is_empty()).count(),
        if chat { " | elicitation=chat" } else { "" },
        if phrase == "standard" { " | phrase=standard" } else { "" },
    );
    println!("# score cells are point/lb95 (stratified bootstrap, {bootstrap_iters} iters)");
    report_grid(&rows, &splits, bootstrap_iters as usize);
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
