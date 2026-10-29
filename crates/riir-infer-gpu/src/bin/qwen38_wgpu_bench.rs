//! Issue 1150 — the Metal dense-qwen3.8 bench + parity harness.
//!
//! Modes:
//! - `--pp 512,2048,...`   prefill cells (median of 3, warm-up first) — the
//!   four-way table row (vs uzu / llama.cpp / MLX on the same GGUF family).
//! - `--decode 32`         decode steps after the prompt (tok/s).
//! - `--decode-ctx 2048`   decode steps after a prompt PADDED to N tokens
//!   (`--decode N` supplies the step count) — the league's decode-at-context
//!   convention (ctx512/ctx2048 cells), not the raw short-prompt shape.
//! - `--walk <text>`       greedy-generate N tokens (`--walk-tokens`), print
//!   the token ids + detokenized text — the ids feed the llama.cpp parity
//!   check on the same GGUF (the G1 acceptance; f32-activation class, module doc).
//!
//! Whole-file cfg keeps non-`qwen38_wgpu` builds an empty binary; the Cargo
//! [[bin]] required-features row is the reader protection.

#![cfg(feature = "qwen38_wgpu")]

use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuForward;

const DEFAULT_GGUF: &str = "/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut gguf = DEFAULT_GGUF.to_string();
    let mut pp: Vec<usize> = vec![];
    let mut decode_steps = 0usize;
    let mut decode_ctx: Option<usize> = None;
    let mut prompt = "The harbor town woke slowly under a grey sky,".to_string();
    let mut ctx = 24_576usize;
    let mut chunk = 512usize;
    let mut walk_tokens = 0usize;
    let mut passes = 3usize;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                i += 1;
                gguf = args[i].clone();
            }
            "--pp" => {
                i += 1;
                pp = args[i]
                    .split(',')
                    .map(|s| s.parse().expect("pp len"))
                    .collect();
            }
            "--decode" => {
                i += 1;
                decode_steps = args[i].parse().expect("decode n");
            }
            "--decode-ctx" => {
                i += 1;
                decode_ctx = Some(args[i].parse().expect("decode-ctx n"));
            }
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--ctx" => {
                i += 1;
                ctx = args[i].parse().expect("ctx");
            }
            "--chunk" => {
                i += 1;
                chunk = args[i].parse().expect("chunk");
            }
            "--walk" => walk_tokens = 64,
            "--walk-tokens" => {
                i += 1;
                walk_tokens = args[i].parse().expect("walk n");
            }
            "--passes" => {
                i += 1;
                passes = args[i].parse().expect("passes");
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    println!("[bench] loading {gguf} (ctx {ctx}, chunk {chunk}) ...");
    let t0 = std::time::Instant::now();
    let mut fwd = Qwen38WgpuForward::new(std::path::Path::new(&gguf), ctx, chunk)
        .unwrap_or_else(|e| panic!("[bench] load failed: {e}"));
    println!("[bench] loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let gg = riir_infer_core::gguf_loader::GgufFile::open(std::path::Path::new(&gguf))
        .unwrap_or_else(|e| panic!("[bench] reopen: {e}"));
    let tok = BpeTokenizer::from_gguf(&gg).unwrap_or_else(|e| panic!("[bench] tokenizer: {e}"));
    let ids: Vec<u32> = tok.encode(&prompt).into_iter().map(|t| t as u32).collect();
    println!(
        "[bench] prompt: {} chars -> {} tokens",
        prompt.len(),
        ids.len()
    );

    // Prefill cells: pad/truncate the prompt to each target by repetition.
    for &target in &pp {
        let mut toks = ids.clone();
        while toks.len() < target {
            toks.extend_from_slice(&ids);
        }
        toks.truncate(target);
        let mut readings = Vec::new();
        for pass in 0..passes + 1 {
            fwd.zero_state();
            let t = std::time::Instant::now();
            fwd.ingest(&toks, 0)
                .unwrap_or_else(|e| panic!("ingest: {e}"));
            let dt = t.elapsed().as_secs_f64();
            let tps = toks.len() as f64 / dt;
            if pass == 0 {
                println!("[pp {target}] warm-up: {dt:.2}s = {tps:.0} tok/s");
                continue;
            }
            readings.push(tps);
            println!("[pp {target}] pass {pass}: {dt:.2}s = {tps:.0} tok/s");
        }
        readings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = readings[readings.len() / 2];
        println!("[pp {target}] MEDIAN {med:.0} tok/s");
    }

    // Decode cell: prefill the prompt then step.
    if decode_steps > 0 {
        fwd.zero_state();
        fwd.ingest(&ids, 0)
            .unwrap_or_else(|e| panic!("ingest: {e}"));
        let mut next = 0u32;
        let t = std::time::Instant::now();
        for s in 0..decode_steps {
            let pos = ids.len() + s;
            next = fwd
                .forward_token(if s == 0 { *ids.last().unwrap() } else { next }, pos)
                .unwrap_or_else(|e| panic!("decode: {e}"));
        }
        let dt = t.elapsed().as_secs_f64();
        println!(
            "[decode] {decode_steps} steps in {dt:.2}s = {:.2} tok/s",
            decode_steps as f64 / dt
        );
    }

    // Decode cell at context depth (the league's decode convention): pad the
    // prompt to N tokens, prefill, then step — tok/s is decode-at-ctx-N, the
    // shape the league's ctx512/ctx2048 decode cells measure.
    if let (Some(ctx_n), &steps) = (decode_ctx, &decode_steps) {
        if steps == 0 {
            eprintln!("[decode-ctx] needs --decode N for the step count");
            std::process::exit(2);
        }
        let mut toks = ids.clone();
        while toks.len() < ctx_n {
            toks.extend_from_slice(&ids);
        }
        toks.truncate(ctx_n);
        fwd.zero_state();
        fwd.ingest(&toks, 0)
            .unwrap_or_else(|e| panic!("decode-ctx ingest: {e}"));
        let mut next = 0u32;
        let t = std::time::Instant::now();
        for s in 0..steps {
            let pos = toks.len() + s;
            next = fwd
                .forward_token(if s == 0 { *toks.last().unwrap() } else { next }, pos)
                .unwrap_or_else(|e| panic!("decode-ctx: {e}"));
        }
        let dt = t.elapsed().as_secs_f64();
        println!(
            "[decode-ctx {ctx_n}] {steps} steps in {dt:.2}s = {:.2} tok/s",
            steps as f64 / dt
        );
    }

    // Greedy walk — the parity artifact.
    if walk_tokens > 0 {
        fwd.zero_state();
        fwd.ingest(&ids, 0)
            .unwrap_or_else(|e| panic!("ingest: {e}"));
        let mut next = 0u32;
        let mut out_ids: Vec<u32> = vec![];
        for s in 0..walk_tokens {
            let pos = ids.len() + s;
            next = fwd
                .forward_token(if s == 0 { *ids.last().unwrap() } else { next }, pos)
                .unwrap_or_else(|e| panic!("walk: {e}"));
            out_ids.push(next);
        }
        println!("[walk] ids: {:?}", out_ids);
        let text: Vec<String> = out_ids
            .iter()
            .map(|&id| tok.decode(&[id as usize]))
            .collect();
        println!("[walk] text: {}", text.join(""));
    }
}
