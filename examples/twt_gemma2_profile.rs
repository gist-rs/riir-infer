//! `twt_gemma2_profile` — Issue 022 T5.1 lane (1): the gemma-2-2b f16
//! control capture (feature `twt_gemma2` = `twt_profile` +
//! `vk_calibration`).
//!
//! The CHEAPEST control for the T5.1 lane order: a dense f16 parent with no
//! quantization interaction, so the S-matrix → partition → collapsed-GGUF →
//! agreement pipeline is adjudicated with every lane constant except depth.
//! The same artifact is riir-train 423 Phase 1's zero-training baseline.
//!
//! Loads the gemma-2 f16 GGUF, walks a fixed token-chunk stream through
//! [`forward_gemma2_f16_tapped`] with the Issue-395 post-layer capture hook
//! (`PostLayerHook` — the post-MLP residual add is exactly the state the
//! TWT S-matrix pools; `NoHook` is bit-identical by monomorphization), pools
//! per-layer states into the TWT S matrices, sweeps the PRE-REGISTERED ε
//! grid through the min-max DP, and adjudicates the Phase-1 kill rule.
//!
//! ⚠ Scope note (recorded, never hidden): this repo's gemma-2 stack does not
//! implement SWA rotation (the `forward_gemma2_f16_tapped` doc pins it) —
//! every layer is full attention below `block_size`. The in-repo control
//! therefore profiles a single-operator stack (the type-split floor is 1)
//! and cannot expose the sliding/global distinction the real model carries
//! above the 4096 window; the issue's ">4096 calibration" green-zero warning
//! applies to windowed stacks, not this lane. Chunk length stays ≤ 4096 so
//! the in-repo semantics are exact for the real model too.
//!
//! Determinism: corpus order + fixed chunk length + canonical fold order —
//! the artifact records the checkpoint path, corpus BLAKE3, chunk length,
//! stride, and row cap needed to reproduce the numbers bit-for-bit.
//!
//! ```text
//! cargo run --release -p riir-infer-core --features twt_gemma2 \
//!   --example twt_gemma2_profile -- \
//!   --gguf ../riir-train/data/gemma-2-2b-it-f16.gguf \
//!   --corpus ../riir-train/data/chat_probe \
//!   --seq-len 1024 --max-tokens 6144 --stride 1 [--out .raw/twt/gemma2_profile.json]
//! ```

use std::io::Write as _;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::gguf_loader::{GgufFile, config_from_gguf_metadata, load_gemma2_f16_direct};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::gemma2_calibration::{
    forward_gemma2_f16_tapped, CalibrationTables,
};
use riir_infer_core::transformer::{ForwardContext, PostLayerHook};
use riir_infer_core::twt::{
    blake3_of, forced_min_blocks, kill_verdict, minmax_partition, Bucket, KillVerdict,
    SMatrixBuilder, PRE_REGISTERED_EPS_GRID,
};
use riir_infer_core::types::kv_dim;
use katgpt_transformer::MultiLayerKVCache;
use std::path::Path;

/// The capture hook: copies the post-layer residual into its own buffer and
/// pushes into the builder (the builder copies on push — no aliasing).
struct Capture<'b> {
    builder: &'b mut SMatrixBuilder,
    row: usize,
    err: Option<String>,
}

impl PostLayerHook for Capture<'_> {
    fn after_layer(&mut self, layer_idx: usize, residual: &mut [f32]) {
        if self.err.is_some() {
            return;
        }
        // Rows are bucketed Prompt: a chunked token stream has no prompt/
        // generation boundary (the chunk IS the unit; the GOAT harness uses
        // the same posture).
        if let Err(e) = self
            .builder
            .push(layer_idx, self.row, residual)
            .and_then(|_| self.builder.set_row_bucket(self.row, Bucket::Prompt))
        {
            self.err = Some(e.to_string());
        }
    }
}

fn main() {
    let mut gguf_path = String::from("../riir-train/data/gemma-2-2b-it-f16.gguf");
    let mut corpus_path = String::from("../riir-train/data/chat_probe");
    let mut seq_len = 1024usize;
    let mut max_tokens = 6144usize;
    let mut out: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf_path = args.next().expect("--gguf needs a path"),
            "--corpus" => corpus_path = args.next().expect("--corpus needs a path"),
            "--seq-len" => seq_len = args.next().expect("--seq-len needs N").parse().unwrap(),
            "--max-tokens" => {
                max_tokens = args.next().expect("--max-tokens needs N").parse().unwrap()
            }
            "--out" => out = Some(args.next().expect("--out needs a path")),
            other => panic!("unknown arg {other}"),
        }
    }
    assert!((2..=4096).contains(&seq_len), "--seq-len must be in [2, 4096] (the in-repo stack is exact only below the sliding window; see the module doc)");
    assert!(seq_len <= max_tokens, "--max-tokens must cover one chunk");

    // ── model + tokenizer from ONE GGUF (the kv_plus_ladder shape) ──
    let t0 = std::time::Instant::now();
    let gguf_path_buf = Path::new(&gguf_path).to_path_buf();
    let gguf = GgufFile::open(&gguf_path_buf).unwrap_or_else(|e| panic!("open gguf: {e}"));
    let arch = gguf.architecture().unwrap_or("unknown");
    assert_eq!(arch, "gemma2", "this lane profiles the gemma2 f16 stack, got {arch}");
    let mut config = config_from_gguf_metadata(&gguf).expect("config");
    let tok = SentencePieceGgufTokenizer::from_gguf(&gguf).expect("tokenizer");
    let corpus_bytes = load_corpus_text(Path::new(&corpus_path))
        .unwrap_or_else(|e| panic!("read corpus: {e}"))
        .into_bytes();
    let weights = load_gemma2_f16_direct(&gguf, &config).expect("weights");
    drop(gguf);
    let n_layers = config.n_layer;
    let dim = config.n_embd;
    println!(
        "# twt_gemma2_profile: gemma-2 f16 | layers={n_layers} hidden={dim} | load {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // ── the frozen token stream → fixed chunks (the GOAT harness posture:
    //    byte-identical tokens for any downstream agreement arm) ──
    let all = tok.encode(std::str::from_utf8(&corpus_bytes).expect("corpus utf-8"));
    let take = all.len().min(max_tokens);
    let tokens: Vec<usize> = all[..take].to_vec();
    let chunks: Vec<Vec<usize>> = tokens.chunks(seq_len).map(<[usize]>::to_vec).collect();
    println!(
        "# corpus: {} chars → {} tokens → {} chunks × {seq_len}",
        corpus_bytes.len(),
        tokens.len(),
        chunks.len()
    );

    // block_size sizes the KV rows + score scratch (the row_logit_floor law:
    // never allocate the advertised context).
    config.block_size = seq_len;

    // The V/K tables are lane baggage (the tapped forward's calibration
    // contract): a top-1 table per layer, fed whatever it observes. The
    // S-matrix never reads it; the cost is one kv_dim-wide accumulate per
    // (layer, token).
    let kvd = kv_dim(&config);
    let mut token_counts = vec![0u64; config.vocab_size];
    for &t in &tokens {
        token_counts[t] += 1;
    }
    let tables = CalibrationTables::from_counts(n_layers, kvd, token_counts, 1);
    let mut tables = tables;

    let new_builder = || SMatrixBuilder::new(n_layers, dim);
    let mut builder = new_builder();

    let mut sequences = 0usize;
    let mut rows_fed = 0u64;
    let mut skipped = 0usize;
    let t1 = std::time::Instant::now();
    for (ci, chunk) in chunks.iter().enumerate() {
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        builder.begin_forward(chunk.len()).unwrap();
        let push_err: Option<String>;
        {
            let mut hook = Capture { builder: &mut builder, row: 0, err: None };
            for (pos, &token) in chunk.iter().enumerate() {
                hook.row = pos;
                forward_gemma2_f16_tapped(
                    &mut ctx, &weights, &mut cache, &mut tables, token, pos, &config, &mut hook,
                );
            }
            push_err = hook.err.take();
        }
        if let Some(e) = push_err {
            eprintln!("[twt] chunk {ci} push failed: {e} — skipped");
            builder = new_builder();
            skipped += 1;
            continue;
        }
        builder.end_forward().unwrap();
        sequences += 1;
        rows_fed += chunk.len() as u64;
        if (ci + 1) % 2 == 0 || ci + 1 == chunks.len() {
            eprintln!(
                "[twt] chunk {}/{} ({sequences} sequences, {rows_fed} rows, {:.2} tok/s)",
                ci + 1,
                chunks.len(),
                rows_fed as f32 / t1.elapsed().as_secs_f32().max(1e-6)
            );
        }
    }

    let sm = builder.finalize().unwrap();
    let s = sm.cosine();
    eprintln!(
        "[twt] S built: {n_layers}×{n_layers} over {sequences} sequences / {} folded rows (dim {dim}, skipped {skipped})",
        sm.rows_folded
    );

    // ε sweep + kill rule — single-operator stack: the type-split floor is 1
    // (nothing forces a split; the kill bar reads the achievable floor, T1.6).
    let types_linear: Vec<bool> = vec![false; n_layers];
    let forced = forced_min_blocks(&types_linear);
    let mut partitions = Vec::with_capacity(PRE_REGISTERED_EPS_GRID.len());
    let mut lines = Vec::new();
    for &eps in PRE_REGISTERED_EPS_GRID.iter() {
        let p = minmax_partition(s, eps).unwrap();
        lines.push(format!(
            "  ε={eps:<5} m={:<3} worst-block={}",
            p.len(),
            p.iter().map(|b| b.len()).max().unwrap_or(0)
        ));
        partitions.push(p);
    }
    let verdict = kill_verdict(&partitions, forced).unwrap();
    println!("checkpoint: {} (layers {n_layers}, hidden {dim})", gguf_path_buf.display());
    println!("type split: {forced} forced block (single-operator stack — all-attention in-repo)");
    println!("ε sweep (cosine meter):");
    for l in &lines {
        println!("{l}");
    }
    println!("kill verdict (T1.6): {verdict:?}");

    if let Some(path) = out {
        if let Some(dir) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let mut f = std::fs::File::create(&path).expect("out file");
        writeln!(f, "{{").unwrap();
        writeln!(f, "  \"checkpoint\": \"{}\",", gguf_path_buf.display()).unwrap();
        writeln!(f, "  \"n_layers\": {n_layers}, \"dim\": {dim},").unwrap();
        writeln!(
            f,
            "  \"sequences\": {sequences}, \"rows_folded\": {},",
            sm.rows_folded
        )
        .unwrap();
        writeln!(f, "  \"skipped\": {skipped},").unwrap();
        writeln!(f, "  \"corpus_blake3\": \"{}\",", blake3_of(&corpus_bytes)).unwrap();
        writeln!(f, "  \"seq_len\": {seq_len}, \"max_tokens\": {max_tokens},").unwrap();
        writeln!(f, "  \"eps_grid\": {PRE_REGISTERED_EPS_GRID:?},").unwrap();
        writeln!(f, "  \"forced_blocks\": {forced},").unwrap();
        writeln!(f, "  \"verdict\": \"{}\",", verdict_word(&verdict)).unwrap();
        // Upper-triangle cosine S, row-major.
        writeln!(f, "  \"S_upper\": [").unwrap();
        let mut first = true;
        for (i, j, d) in s.entries() {
            let sep = if first { "" } else { "," };
            first = false;
            write!(f, "{sep}[{i},{j},{d}]").unwrap();
        }
        writeln!(f, "\n  ]").unwrap();
        writeln!(f, "}}").unwrap();
        eprintln!("[twt] artifact written to {path}");
    }
}

fn verdict_word(v: &KillVerdict) -> &'static str {
    match v {
        KillVerdict::Survives { .. } => "SURVIVES",
        KillVerdict::KillBlockCount { .. } => "KILL_BLOCK_COUNT",
        KillVerdict::KillMiddleBlocks { .. } => "KILL_MIDDLE_BLOCKS",
    }
}
