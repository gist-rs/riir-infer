//! T5.3 perf — collapsed-vs-parent decode (tg) + prefill (pp) at the
//! quality-matched point (Issue 022; the Bench 022 PASS row, ε=0.01
//! agreement 0.9486, is the matched floor — run the collapsed arm(s) at the
//! ε whose agreement you accepted).
//!
//! Three measured phases per arm, all on the SAME frozen prompt head of the
//! agreement corpus (512 tokens — the sweep's chunk length, so the pp rate
//! is comparable with the agreement-harness tok/s column):
//!
//! 1. **pp** — teacher-forced forward over the prompt head; wall → pp tok/s
//!    (the teacher-forced next-token hit rate over the prompt is printed as
//!    context, never a claim).
//! 2. **tg** — greedy free decode: the pp's last argmax seeds the loop, each
//!    generated token feeds back; the decode window is timed separately →
//!    tg tok/s. The generated ids are recorded per arm and the parent-vs-
//!    collapsed free-decode divergence position is printed (chaotic stream
//!    sensitivity at the agreement bar's 5% divergence is EXPECTED — it is
//!    recorded, never scored).
//! 3. **memory law** — per-token KV bytes (attention layers) and fixed GDN
//!    state bytes, computed from the loaded config + layer table, with the
//!    m/L law ASSERTED by integer cross-multiplication: collapsed KV/token
//!    × parent attn layers == parent KV/token × collapsed attn layers
//!    (GDN state likewise). A violation is a loader bug, not a finding.
//!
//! Box-state provenance (load avg, power, powermode, cores, UTC stamp) is
//! printed at start — quote it beside every rate.
//!
//! Usage:
//!   cargo run --release --features twt_bonsai --bin twt_perf_tg_pp -- \
//!     --parent ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!     --collapsed /tmp/twt_collapse_pq2_e0.01.gguf \
//!     --corpus .raw/twt/audition_calib.txt \
//!     --pp-tokens 512 --tg-tokens 128 --out .raw/twt/t53_perf_tg_pp.json
//!
//! The parent arm runs ONCE (first); each --collapsed arm follows it.

#![cfg(feature = "twt_bonsai")]

use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary_with_hook;
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;

/// One measured arm (parent or collapsed).
#[derive(Serialize)]
struct ArmPerf {
    label: String,
    path: String,
    n_layer: usize,
    n_gdn: usize,
    n_attn: usize,
    load_s: f32,
    pp_tokens: usize,
    pp_s: f32,
    pp_tok_per_s: f32,
    /// Teacher-forced next-token hits over the prompt head (context only).
    pp_hits: usize,
    pp_scored: usize,
    tg_tokens: usize,
    tg_s: f32,
    tg_tok_per_s: f32,
    /// Greedy free-decode ids, seed-first (the seed came from the pp tail).
    generated: Vec<u32>,
    /// KV bytes per token (attention layers only, f32 cache).
    kv_bytes_per_token: usize,
    /// Fixed GDN state bytes (recurrence + conv, all GDN layers).
    gdn_state_bytes: usize,
    /// Total KV window bytes at the run's context cap.
    kv_window_bytes: usize,
}

fn argmax_of(x: &[f32], vocab: usize) -> u32 {
    x[..vocab]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .expect("non-empty logits")
}

/// Measurement-only provenance via the shell; a failed read degrades to a
/// printed marker, it never fails the run (provenance is a quote, not a gate).
fn sh(cmd: &str) -> String {
    std::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "(unavailable)".to_owned())
}

fn print_box_state() {
    println!("# box-state (quote beside every rate)");
    println!("# utc: {}", sh("date -u '+%Y-%m-%dT%H:%M:%SZ'"));
    println!("# loadavg: {}", sh("sysctl -n vm.loadavg"));
    println!(
        "# power: {}",
        sh("pmset -g batt | grep -E 'AC Power|Battery Power'")
    );
    println!(
        "# powermode: {}",
        sh("pmset -g | grep -i '^ powermode' || pmset -g | grep -i powermode")
    );
    println!(
        "# cores: {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );
}

fn run_arm(
    label: &str,
    gguf_path: &std::path::Path,
    prompt: &[usize],
    truth: &[usize],
    tg_tokens: usize,
    block_size: usize,
) -> Result<ArmPerf> {
    let t0 = Instant::now();
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(gguf_path)
        .with_context(|| format!("load {}", gguf_path.display()))?;
    let load_s = t0.elapsed().as_secs_f32();
    let n_gdn = weights
        .layer_types
        .iter()
        .filter(|&&t| riir_infer_core::types::DeltaNetLayerType::DeltaNet == t)
        .count();
    let n_attn = config.n_layer - n_gdn;
    eprintln!(
        "[perf] {label}: {} layers (gdn {n_gdn}, attn {n_attn}) | load {load_s:.1}s",
        config.n_layer,
    );

    // Memory law inputs (f32 cache + f32 GDN state, the deployment arithmetic
    // this repo's hybrid path ships).
    let kv_bytes_per_token = n_attn * 2 * config.n_kv_head * config.head_dim * 4;
    let n_v = config.deltanet_linear_n_value_heads;
    let d_k = config.deltanet_linear_head_dim;
    let n_k = config.deltanet_linear_n_heads;
    let conv_dim = (n_k + n_k + n_v) * d_k;
    let gdn_state_bytes =
        n_gdn * (n_v * d_k * d_k + conv_dim * config.deltanet_conv_kernel_size) * 4;
    let kv_window_bytes = kv_bytes_per_token * block_size;

    // The run's own context cap: nothing clamps inside the window (the
    // needle-eviction lane runs far longer contexts through this path).
    config.block_size = block_size;
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&config);
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];

    // ── phase 1: pp (teacher-forced over the prompt head) ──
    let mut pp_hits = 0usize;
    let mut last_argmax = 0u32;
    let tp = Instant::now();
    for (p, &tok) in prompt.iter().enumerate() {
        forward_qwen_deltanet_ternary_with_hook(
            &mut x,
            &weights,
            &mut cache,
            tok,
            p,
            &config,
            &mut scratch,
            &rope_freq,
            None,
            None,
            None,
            None,
        );
        let am = argmax_of(&x, config.vocab_size);
        if p + 1 < prompt.len() && am as usize == truth[p] {
            pp_hits += 1;
        }
        last_argmax = am;
    }
    let pp_s = tp.elapsed().as_secs_f32();
    let pp_scored = prompt.len() - 1;
    eprintln!(
        "[perf] {label}: pp {} tokens in {pp_s:.1}s ({:.2} tok/s) | in-prompt hit {:.4} (context)",
        prompt.len(),
        prompt.len() as f32 / pp_s.max(1e-6),
        pp_hits as f64 / pp_scored.max(1) as f64,
    );

    // ── phase 2: tg (greedy free decode off the pp tail) ──
    let mut generated = Vec::with_capacity(tg_tokens + 1);
    generated.push(last_argmax);
    let mut cur = last_argmax as usize;
    let tt = Instant::now();
    for pos in prompt.len()..prompt.len() + tg_tokens {
        forward_qwen_deltanet_ternary_with_hook(
            &mut x,
            &weights,
            &mut cache,
            cur,
            pos,
            &config,
            &mut scratch,
            &rope_freq,
            None,
            None,
            None,
            None,
        );
        let am = argmax_of(&x, config.vocab_size);
        generated.push(am);
        cur = am as usize;
    }
    let tg_s = tt.elapsed().as_secs_f32();
    eprintln!(
        "[perf] {label}: tg {tg_tokens} decode steps in {tg_s:.1}s ({:.2} tok/s) | gen-seed {}",
        tg_tokens as f32 / tg_s.max(1e-6),
        generated[0],
    );

    drop(cache);
    drop(scratch);
    drop(weights);

    Ok(ArmPerf {
        label: label.to_owned(),
        path: gguf_path.display().to_string(),
        n_layer: config.n_layer,
        n_gdn,
        n_attn,
        load_s,
        pp_tokens: prompt.len(),
        pp_s,
        pp_tok_per_s: prompt.len() as f32 / pp_s.max(1e-6),
        pp_hits,
        pp_scored,
        tg_tokens,
        tg_s,
        tg_tok_per_s: tg_tokens as f32 / tg_s.max(1e-6),
        generated,
        kv_bytes_per_token,
        gdn_state_bytes,
        kv_window_bytes,
    })
}

fn main() -> Result<()> {
    let mut parent: Option<std::path::PathBuf> = None;
    let mut collapsed: Vec<std::path::PathBuf> = Vec::new();
    let mut corpus: Option<std::path::PathBuf> = None;
    let mut pp_tokens = 512usize;
    let mut tg_tokens = 128usize;
    let mut out: Option<std::path::PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--parent" => parent = Some(std::path::PathBuf::from(args.next().expect("path"))),
            "--collapsed" => collapsed.push(std::path::PathBuf::from(args.next().expect("path"))),
            "--corpus" => corpus = Some(std::path::PathBuf::from(args.next().expect("path"))),
            "--pp-tokens" => pp_tokens = args.next().expect("n").parse()?,
            "--tg-tokens" => tg_tokens = args.next().expect("n").parse()?,
            "--out" => out = Some(std::path::PathBuf::from(args.next().expect("path"))),
            other => bail!("unknown arg {other}"),
        }
    }
    let parent = parent.context("--parent is required")?;
    if collapsed.is_empty() {
        bail!("at least one --collapsed is required (repeatable — the parent arm runs once)");
    }
    let corpus = corpus.context("--corpus is required")?;
    assert!(pp_tokens >= 2 && tg_tokens >= 1, "pp >= 2, tg >= 1");

    print_box_state();

    // ── the frozen prompt head (byte-identical for every arm) ──
    let text = load_corpus_text(&corpus)?;
    let gguf = GgufFile::open(&parent).context("re-open parent for tokenizer")?;
    let arch = gguf
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_owned();
    if arch != "qwen35" {
        bail!("twt_perf_tg_pp is the qwen35 (bonsai ternary) lane; arch = {arch}");
    }
    let tok = BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?;
    drop(gguf);
    let all = tok.encode(&text);
    assert!(
        all.len() > pp_tokens,
        "corpus too short: {} tokens < pp {} + 1",
        all.len(),
        pp_tokens
    );
    let prompt: Vec<usize> = all[..pp_tokens].to_vec();
    let truth: Vec<usize> = all[1..=pp_tokens].to_vec();

    // ── parent arm first (the m/L law's reference side) ──
    let base = run_arm(
        "parent",
        &parent,
        &prompt,
        &truth,
        tg_tokens,
        pp_tokens + tg_tokens,
    )?;
    eprintln!(
        "[perf] parent: kv/token {} B (attn {}), gdn state {} B fixed, kv window {} B @ ctx {}",
        base.kv_bytes_per_token,
        base.n_attn,
        base.gdn_state_bytes,
        base.kv_window_bytes,
        pp_tokens + tg_tokens,
    );

    let mut arms = vec![base];
    for path in &collapsed {
        let arm = run_arm(
            "collapsed",
            path,
            &prompt,
            &truth,
            tg_tokens,
            pp_tokens + tg_tokens,
        )?;
        let base_arm = &arms[0];

        // ── the m/L memory laws (integer cross-multiplication, exact) ──
        assert_eq!(
            arm.kv_bytes_per_token * base_arm.n_attn,
            base_arm.kv_bytes_per_token * arm.n_attn,
            "KV-per-token m/L law violated (loader geometry bug, not a finding)"
        );
        assert_eq!(
            arm.gdn_state_bytes * base_arm.n_gdn,
            base_arm.gdn_state_bytes * arm.n_gdn,
            "GDN fixed-state law violated (loader geometry bug, not a finding)"
        );
        // And the collapsed arm's KV/token must be ≤ the parent's (a collapse
        // only ever removes attention layers — never adds them).
        assert!(arm.kv_bytes_per_token <= base_arm.kv_bytes_per_token);

        // Free-decode divergence vs the parent (recorded, never scored).
        let div = base_arm
            .generated
            .iter()
            .zip(arm.generated.iter())
            .position(|(a, b)| a != b);
        eprintln!(
            "[perf] collapsed: kv/token {} B (attn {}), gdn state {} B fixed | free-decode first divergence at generated[{:?}]",
            arm.kv_bytes_per_token, arm.n_attn, arm.gdn_state_bytes, div,
        );

        arms.push(arm);
    }

    // ── the record ──
    println!("# twt_perf_tg_pp — T5.3 collapsed-vs-parent decode/prefill (Issue 022)");
    for a in &arms {
        println!(
            "# {}: {} layers (gdn {}, attn {}) | load {:.1}s | pp {:.2} tok/s ({} tok) | tg {:.2} tok/s ({} steps) | kv/tok {} B | gdn fixed {} B",
            a.label,
            a.n_layer,
            a.n_gdn,
            a.n_attn,
            a.load_s,
            a.pp_tok_per_s,
            a.pp_tokens,
            a.tg_tok_per_s,
            a.tg_tokens,
            a.kv_bytes_per_token,
            a.gdn_state_bytes,
        );
    }

    if let Some(out_path) = &out {
        #[derive(Serialize)]
        struct Record {
            utc: String,
            parent: String,
            corpus: String,
            pp_tokens: usize,
            tg_tokens: usize,
            block_size: usize,
            arms: Vec<ArmPerf>,
        }
        let rec = Record {
            utc: sh("date -u '+%Y-%m-%dT%H:%M:%SZ'"),
            parent: parent.display().to_string(),
            corpus: corpus.display().to_string(),
            pp_tokens,
            tg_tokens,
            block_size: pp_tokens + tg_tokens,
            arms,
        };
        std::fs::write(out_path, serde_json::to_string_pretty(&rec).unwrap())
            .with_context(|| format!("write {}", out_path.display()))?;
        eprintln!("[perf] record → {}", out_path.display());
    }
    Ok(())
}
