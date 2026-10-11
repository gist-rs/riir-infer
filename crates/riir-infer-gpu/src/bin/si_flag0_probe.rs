//! `si_flag0_probe` — riir-infer plan 620 Phase A's bit-identity recording.
//!
//! The scale-invariant decode hook (plan 620) enters `attention_decode_f32`
//! + `_devpos` behind a runtime device flag. The plan's law: a flag-0 launch
//! must reproduce the incumbent kernel's bytes. This bin is the RECORDING
//! instrument for that gate — it is deliberately INDEPENDENT of the si
//! feature so it runs unchanged at the PRE-HOOK commit:
//!
//! 1. At the pre-hook commit (the incumbent build): run, save stdout.
//! 2. At the post-hook commit (flag 0 = the default posture): run again.
//! 3. The two stdout streams must be BYTE-IDENTICAL (same box, same env).
//!
//! What it does: loads the Bonsai-2 PQ2_0 checkpoint once, feeds a fixed
//! 128-token prefill one token at a time (digesting the logits at fixed
//! steps), then greedy-decodes 64 tokens — first on the EAGER lane
//! (`set_input_token` + `forward_token`), then, after a state reset, on the
//! GRAPH lane (`forward_token_graph`, the devpos kernels the G3 harness
//! drives). Every digest is blake3 over the raw f32 logits bytes, rolled
//! incrementally over the greedy steps; the argmax id sequence is printed
//! in full (the league's "argmax STABLE" pin).
//!
//! Determinism posture: fixed prompt, no sampling, no RNG on the decode
//! path, env knobs left at their defaults — two runs of the SAME build are
//! expected byte-identical; two runs of DIFFERENT builds differing anywhere
//! is the bit-identity gate FAILING.
//!
//! 4090 lane (the CUDA box; GPU-exclusive for the record — a correctness
//! recording, not a latency gate, but the box-state law prefers a quiet
//! run anyway).
//!
//! Env: `BONSAI_GGUF` (the checkpoint; defaults to the riir-train league
//! path). Flags: `--gguf <path>` · `--prefill N` (128) · `--greedy N` (64).
//!
//! ```text
//! cargo build --release -p riir-infer-gpu --features ternary_gemv_cuda_raw,cuda_graphs_forward \
//!   --bin si_flag0_probe
//! ./target/release/si_flag0_probe --gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf
//! ```

#![cfg(all(
    feature = "ternary_gemv_cuda_raw",
    feature = "cuda_graphs_forward",
    not(target_os = "macos")
))]
// Dev profile: the refusing main makes the measurement body dead code BY
// DESIGN (it must never run unoptimised) — silence exactly that class, in
// exactly that profile. Release keeps every warning live.
#![cfg_attr(debug_assertions, allow(dead_code))]

use std::path::PathBuf;

use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::ternary_deltanet_gpu_forward_cudarc::TernaryDeltanetGpuForwardCudarc;

const PROMPT_SENTENCE: &str =
    "The merchant walked along the road and counted his coins under the afternoon sun.";

fn main() {
    if let Err(e) = run() {
        eprintln!("si_flag0_probe: FAIL: {e}");
        std::process::exit(1);
    }
}

fn arg_of(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn logits_digest(logits: &[f32]) -> String {
    blake3::hash(bytemuck::cast_slice::<f32, u8>(logits)).to_hex().to_string()
}

/// One greedy run's recording: the argmax ids + the rolled logits digest.
struct GreedyRecord {
    ids: Vec<usize>,
    rolled_digest: String,
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_l = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > best_l {
            best_l = l;
            best = i;
        }
    }
    best
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let gguf = arg_of(&args, "--gguf")
        .map(PathBuf::from)
        .or_else(|| std::env::var("BONSAI_GGUF").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf"));
    let prefill_n: usize = arg_of(&args, "--prefill")
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let greedy_n: usize = arg_of(&args, "--greedy")
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    // ── load once ──
    let (cfg, weights) = load_qwen_deltanet_ternary_weights_gguf(&gguf)
        .map_err(|e| format!("load {}: {e}", gguf.display()))?;
    let gguf_meta = std::fs::metadata(&gguf).map_err(|e| e.to_string())?;
    println!("model={} bytes={}", gguf.display(), gguf_meta.len());
    println!(
        "cfg: layers={} heads={}/{} hd={} block={} vocab={} rotation={}",
        cfg.n_layer,
        cfg.n_head,
        cfg.n_kv_head,
        cfg.head_dim,
        cfg.block_size,
        cfg.vocab_size,
        weights.rotation.is_some()
    );

    let gguf_file = GgufFile::open(&gguf).map_err(|e| e.to_string())?;
    let tok = BpeTokenizer::from_gguf(&gguf_file).map_err(|e| e.to_string())?;
    drop(gguf_file);

    // ── the frozen prefill: the sentence repeated to exactly prefill_n tokens ──
    let mut tokens = vec![tok.bos_id()];
    let sentence = tok.encode(PROMPT_SENTENCE);
    while tokens.len() < prefill_n {
        tokens.extend(sentence.iter().copied());
    }
    tokens.truncate(prefill_n);
    println!("prompt tokens={} bos={}", tokens.len(), tok.bos_id());

    let mut fwd = TernaryDeltanetGpuForwardCudarc::new_graph_ready(&cfg, &weights)
        .map_err(|e| e.to_string())?;

    // ── EAGER lane: prefill with fixed-step digests, then greedy ──
    fwd.reset_state().map_err(|e| e.to_string())?;
    let mut last_logits = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        fwd.set_input_token(t).map_err(|e| e.to_string())?;
        last_logits = fwd.forward_token().map_err(|e| e.to_string())?;
        if i + 1 == 16 || i + 1 == tokens.len() {
            println!(
                "eager prefill digest @{}: {}",
                i + 1,
                logits_digest(&last_logits)
            );
        }
    }
    let eager = greedy_eager(&mut fwd, &last_logits, greedy_n)?;
    println!(
        "eager greedy{greedy_n} ids: {}",
        eager.ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(":")
    );
    println!("eager greedy{greedy_n} rolled digest: {}", eager.rolled_digest);

    // ── GRAPH lane: the same feed + greedy over forward_token_graph ──
    fwd.reset_state().map_err(|e| e.to_string())?;
    let mut graph_ids = Vec::with_capacity(greedy_n);
    let mut hasher = blake3::Hasher::new();
    for (i, &t) in tokens.iter().enumerate() {
        let l = fwd.forward_token_graph(t).map_err(|e| e.to_string())?;
        if i + 1 == 16 || i + 1 == tokens.len() {
            println!("graph prefill digest @{}: {}", i + 1, logits_digest(&l));
        }
        last_logits = l;
    }
    let mut next = argmax(&last_logits);
    graph_ids.push(next);
    while graph_ids.len() < greedy_n {
        let l = fwd.forward_token_graph(next).map_err(|e| e.to_string())?;
        hasher.update(bytemuck::cast_slice::<f32, u8>(&l));
        next = argmax(&l);
        graph_ids.push(next);
    }
    println!(
        "graph greedy{greedy_n} ids: {}",
        graph_ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(":")
    );
    println!(
        "graph greedy{greedy_n} rolled digest: {}",
        hasher.finalize().to_hex().to_string()
    );
    println!(
        "eager==graph ids: {}",
        if graph_ids == eager.ids { "YES" } else { "NO" }
    );
    println!("SI_FLAG0_PROBE done");
    Ok(())
}

/// Greedy over the EAGER lane (`set_input_token` + `forward_token`), the
/// argmax chain seeded from the prefill's final logits.
fn greedy_eager(
    fwd: &mut TernaryDeltanetGpuForwardCudarc,
    first_logits: &[f32],
    steps: usize,
) -> Result<GreedyRecord, String> {
    let mut ids = Vec::with_capacity(steps);
    let mut hasher = blake3::Hasher::new();
    let mut next = argmax(first_logits);
    ids.push(next);
    while ids.len() < steps {
        fwd.set_input_token(next).map_err(|e| e.to_string())?;
        let l = fwd.forward_token().map_err(|e| e.to_string())?;
        hasher.update(bytemuck::cast_slice::<f32, u8>(&l));
        next = argmax(&l);
        ids.push(next);
    }
    Ok(GreedyRecord {
        ids,
        rolled_digest: hasher.finalize().to_hex().to_string(),
    })
}
