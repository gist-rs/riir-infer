//! sink_audit — the Issue-981 T1 census CLI: one pass per checkpoint over the
//! first N tokens, the F-Attn verdict, and the audited KV-eviction policy.
//!
//! ```bash
//! cargo run --release --features sink_audit --bin sink_audit -- \
//!     ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!     --n 512 --out .benchmarks/sink_audit_report.json
//! ```
//!
//! Determinism (G1): the report's `blake3` field is taken over the canonical
//! payload bytes (fixed-precision profile lines), so a re-run over the same
//! checkpoint + corpus + window must reproduce it byte-for-byte.
//!
//! Scope: FULL-ATTENTION layers only. GDN layers have no softmax sink. The
//! verdict's cut points are CHOSEN (0.10 / 0.25) and stated so in the report;
//! an Undecided cell keeps the incumbent posture and is disclosed, never
//! folded. The token-gate reserve is deliberately OUT of scope (Issue 981
//! T2: its sink window + final position is a correctness margin, not an
//! eviction budget).

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::effective_rotary_dim;
use riir_infer_core::deltanet::sink_audit::{SinkCensus, run_census, run_census_ternary};
use riir_infer_core::gguf_loader::{
    GgufFile, GgmlType, load_qwen_deltanet_weights_gguf, load_qwen_deltanet_ternary_weights_gguf,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;

use katgpt_core::kv_sink_window::{
    AUDIT_N_SINK_FLOOR, AUDIT_PROFILE_SINK_MIN, AuditedSinkPolicy, SINKFREE_F_ATTN_MAX,
    SINKHEAVY_F_ATTN_MIN, SinkAuditVerdict, SinkWindowPolicy,
};

const DEFAULT_CORPUS: &str = "../riir-train/data/chat_probe";
const DEFAULT_N: usize = 512;
const DEFAULT_SINKS: usize = 4;
const DEFAULT_WINDOW: usize = 256;

struct Args {
    model: PathBuf,
    corpus: PathBuf,
    n: usize,
    incumbent_sinks: usize,
    incumbent_window: usize,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: PathBuf::from("../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf"),
        corpus: PathBuf::from(DEFAULT_CORPUS),
        n: DEFAULT_N,
        incumbent_sinks: DEFAULT_SINKS,
        incumbent_window: DEFAULT_WINDOW,
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        let mut val = |name: &str| -> String {
            it.next()
                .unwrap_or_else(|| panic!("--{name} needs a value"))
        };
        match f.as_str() {
            "--n" => a.n = val("n").parse().unwrap(),
            "--corpus" => a.corpus = val("corpus").into(),
            "--incumbent-sinks" => a.incumbent_sinks = val("incumbent-sinks").parse().unwrap(),
            "--incumbent-window" => a.incumbent_window = val("incumbent-window").parse().unwrap(),
            "--out" => a.out = Some(val("out").into()),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other if other.starts_with('-') => panic!("unknown arg {other}"),
            other => a.model = other.into(),
        }
    }
    a
}

fn print_help() {
    println!(
        "sink_audit <model.gguf> [--corpus PATH] [--n N] [--incumbent-sinks K] \
         [--incumbent-window W] [--out report.json]\n\n\
         One-time per-checkpoint census (Issue 981): F-Attn profile over the\n\
         first N tokens on the FULL-ATTENTION layers only, the chosen-cut\n\
         verdict, and the audited SinkWindowPolicy. The report JSON embeds the\n\
         BLAKE3 of its canonical payload (G1: same inputs -> same hash)."
    );
}

/// Canonical payload: fixed-precision lines over the profile — the byte
/// string the report's blake3 is taken over. v1 FROZEN: the field set is
/// byte-stable across bodies (the file name already distinguishes the
/// checkpoint); the body lands in the report JSON, never here.
fn canonical_payload(
    model: &str,
    n: usize,
    census: &SinkCensus,
    verdict: SinkAuditVerdict,
) -> Vec<u8> {
    let mut s = String::new();
    s.push_str(&format!(
        "sink_audit v1 model={model} n={} layers={} heads={} m_act={:.6} verdict={verdict:?}\n",
        n, census.profile.layers_scanned, census.profile.heads_scanned, census.profile.m_act
    ));
    for (k, mass) in census.profile.first_n_profile.iter().enumerate() {
        s.push_str(&format!("{k} {:.9}\n", mass));
    }
    s.into_bytes()
}

fn main() -> Result<()> {
    let args = parse_args();

    // Detect the body FIRST: the dense loader mis-shapes a PQ2_0 ternary file
    // (in_proj_qkv row overrun — katgpt-rs Issue 981 follow-up (b)), so the
    // dispatch must happen before any weight load. The globals tensor's dtype
    // is the discriminator (Q2_0/PTQ1_0 = ternary; anything else = dense).
    let probe = GgufFile::open(&args.model)
        .with_context(|| format!("open {}", args.model.display()))?;
    let is_ternary = probe
        .tensor_info("token_embd.weight")
        .map(|info| matches!(info.ggml_type, GgmlType::Q2_0 | GgmlType::PTQ1_0))
        .unwrap_or(false);
    let body = if is_ternary { "ternary" } else { "dense" };

    let corpus_text = load_corpus_text(&args.corpus)
        .with_context(|| format!("load corpus {}", args.corpus.display()))?;
    let tokens = {
        let tok = BpeTokenizer::from_gguf(&probe).context("gpt2 BPE tokenizer from gguf")?;
        tok.encode(&corpus_text)
    };
    if tokens.len() < args.n {
        bail!(
            "corpus encodes {} tokens < the census window {} — shorten --n or grow the corpus",
            tokens.len(),
            args.n
        );
    }
    println!(
        "# corpus: {} | {} tokens encoded (need {}) | body={body}",
        args.corpus.display(),
        tokens.len(),
        args.n
    );

    let t0 = std::time::Instant::now();
    let (config, census) = if is_ternary {
        let (config, weights) = load_qwen_deltanet_ternary_weights_gguf(&args.model)
            .with_context(|| format!("load (ternary) {}", args.model.display()))?;
        println!(
            "# model: qwen35 hybrid | body=ternary | layers={} attn_layers={} n_embd={} n_head={} n_kv_head={} hd={} folded={} | load {:.1}s",
            config.n_layer,
            weights
                .layer_types
                .iter()
                .filter(|&&t| t == riir_infer_core::types::DeltaNetLayerType::Attention)
                .count(),
            config.n_embd,
            config.n_head,
            config.n_kv_head,
            config.head_dim,
            weights.rotation.is_some(),
            t0.elapsed().as_secs_f32()
        );
        let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
        let t1 = std::time::Instant::now();
        let c = run_census_ternary(&weights, &config, &tokens, &rope_freq, args.n)
            .map_err(anyhow::Error::msg)
            .with_context(|| "census pass")?;
        println!("# census pass: {:.1}s (ternary body)", t1.elapsed().as_secs_f32());
        (config, c)
    } else {
        let (config, weights) = load_qwen_deltanet_weights_gguf(&args.model)
            .with_context(|| format!("load {}", args.model.display()))?;
        println!(
            "# model: qwen35 hybrid | body=dense | layers={} attn_layers={} n_embd={} n_head={} n_kv_head={} hd={} | load {:.1}s",
            config.n_layer,
            weights
                .layer_types
                .iter()
                .filter(|&&t| t == riir_infer_core::types::DeltaNetLayerType::Attention)
                .count(),
            config.n_embd,
            config.n_head,
            config.n_kv_head,
            config.head_dim,
            t0.elapsed().as_secs_f32()
        );
        let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
        let t1 = std::time::Instant::now();
        let c = run_census(&weights, &config, &tokens, &rope_freq, args.n)
            .map_err(anyhow::Error::msg)
            .with_context(|| "census pass")?;
        println!("# census pass: {:.1}s (dense body)", t1.elapsed().as_secs_f32());
        (config, c)
    };
    let profile = &census.profile;
    let verdict = profile.verdict();

    println!(
        "# n={} layers={} heads={} observations={}",
        census.n_positions,
        profile.layers_scanned,
        profile.heads_scanned,
        profile.layers_scanned * profile.heads_scanned * census.n_positions
    );
    println!(
        "# f_attn_token0 {:.6} | m_act {:.3} | first-8 profile: {}",
        profile.f_attn_token0,
        profile.m_act,
        profile
            .first_n_profile
            .iter()
            .take(8)
            .map(|m| format!("{m:.6}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    let incumbent = SinkWindowPolicy::new(args.incumbent_sinks, args.incumbent_window);
    match SinkWindowPolicy::from_audit(verdict, profile, incumbent) {
        AuditedSinkPolicy::Audited(policy) => {
            println!(
                "verdict: SinkFree (cut <= {SINKFREE_F_ATTN_MAX}) -> audited pin n_sink={} \
                 (leading mass >= {AUDIT_PROFILE_SINK_MIN}, floor {AUDIT_N_SINK_FLOOR}, \
                 incumbent {}); window {} carried",
                policy.n_sink, incumbent.n_sink, policy.window
            );
        }
        AuditedSinkPolicy::KeepIncumbentSinkHeavy => {
            println!(
                "verdict: SinkHeavy (cut >= {SINKHEAVY_F_ATTN_MIN}) -> incumbent pin \
                 n_sink={} stands. {}",
                incumbent.n_sink,
                SinkAuditVerdict::SinkHeavy
                    .risk_line()
                    .expect("heavy carries a risk line")
            );
        }
        AuditedSinkPolicy::KeepIncumbentUndecided => {
            println!(
                "verdict: Undecided ({SINKFREE_F_ATTN_MAX} < f_attn_token0 < \
                 {SINKHEAVY_F_ATTN_MIN}) -> incumbent pin n_sink={} stands; the cell is \
                 disclosed, never folded",
                incumbent.n_sink
            );
        }
    }

    let payload = canonical_payload(
        args.model
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model"),
        args.n,
        &census,
        verdict,
    );
    let hash = blake3::hash(&payload).to_hex().to_string();
    println!("blake3(canonical payload): {hash}");

    if let Some(out) = args.out.clone() {
        let report = format_report(
            &args,
            &census,
            verdict,
            &hash,
            config.block_size,
            body,
            &payload,
        );
        std::fs::write(&out, report).with_context(|| format!("write {}", out.display()))?;
        println!("# report: {}", out.display());
    }
    Ok(())
}

/// The report JSON: the canonical payload echoed verbatim (the hash's source
/// of truth travels WITH the hash), plus the verdict block.
#[allow(clippy::too_many_arguments)]
fn format_report(
    args: &Args,
    census: &SinkCensus,
    verdict: SinkAuditVerdict,
    hash: &str,
    block_size: usize,
    body: &str,
    payload: &[u8],
) -> String {
    let profile = &census.profile;
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!(
        "  \"schema\": \"sink_audit_v1\",\n  \"model\": {},\n",
        json_str(&args.model.display().to_string())
    ));
    s.push_str(&format!("  \"body\": {},\n", json_str(body)));
    s.push_str(&format!(
        "  \"corpus\": {},\n  \"n_positions\": {},\n  \"layers_scanned\": {},\n  \"heads_scanned\": {},\n",
        json_str(&args.corpus.display().to_string()),
        census.n_positions,
        profile.layers_scanned,
        profile.heads_scanned
    ));
    s.push_str(&format!(
        "  \"f_attn_token0\": {:.9},\n  \"m_act\": {:.6},\n  \"verdict\": {:?},\n",
        profile.f_attn_token0, profile.m_act, verdict
    ));
    s.push_str(&format!(
        "  \"chosen_cuts\": {{\"sink_free_max\": {SINKFREE_F_ATTN_MAX}, \
         \"sink_heavy_min\": {SINKHEAVY_F_ATTN_MIN}, \"profile_sink_min\": \
         {AUDIT_PROFILE_SINK_MIN}, \"n_sink_floor\": {AUDIT_N_SINK_FLOOR}}},\n"
    ));
    s.push_str(&format!(
        "  \"incumbent\": {{\"n_sink\": {}, \"window\": {}}},\n",
        args.incumbent_sinks, args.incumbent_window
    ));
    let policy = match SinkWindowPolicy::from_audit(
        verdict,
        profile,
        SinkWindowPolicy::new(args.incumbent_sinks, args.incumbent_window),
    ) {
        AuditedSinkPolicy::Audited(p) => format!(
            "{{\"n_sink\": {}, \"window\": {}, \"audited\": true}}",
            p.n_sink, p.window
        ),
        AuditedSinkPolicy::KeepIncumbentSinkHeavy => format!(
            "{{\"n_sink\": {}, \"window\": {}, \"audited\": false, \"reason\": \"SinkHeavy\"}}",
            args.incumbent_sinks, args.incumbent_window
        ),
        AuditedSinkPolicy::KeepIncumbentUndecided => format!(
            "{{\"n_sink\": {}, \"window\": {}, \"audited\": false, \"reason\": \"Undecided\"}}",
            args.incumbent_sinks, args.incumbent_window
        ),
    };
    s.push_str(&format!("  \"policy\": {policy},\n"));
    s.push_str(&format!(
        "  \"model_block_size_note\": \"census cache sized to n_positions, not the model's {block_size} (the Issue-929 bounded-cache lesson)\",\n"
    ));
    s.push_str(&format!("  \"blake3\": \"{hash}\",\n"));
    s.push_str("  \"first_n_profile\": [");
    for (i, m) in profile.first_n_profile.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{m:.9}"));
    }
    s.push_str("],\n");
    s.push_str(&format!(
        "  \"canonical_payload_b64\": {}\n}}\n",
        base64_of(payload)
    ));
    s
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Standard base64 (no external dep — a 12-line table encoder keeps this bin
/// dependency-free; the payload is ASCII anyway).
fn base64_of(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    json_str(&out)
}
