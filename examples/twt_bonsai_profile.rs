//! `twt_bonsai_profile` — Issue 022 T1.2's REMAINING half: the
//! Bonsai/GDN capture sibling on the ROOT-CRATE ternary forward (feature
//! `twt_profile`), landing the league model (`Ternary-Bonsai-2-27B-PQ2`,
//! the qwen3.5-hybrid ternary GGUF) inside the Phase-1 S-matrix
//! methodology — the laya encoder's half shipped 2026-09-27, this is its
//! root-crate twin.
//!
//! Loads a ternary-hybrid GGUF, walks a plain-text calibration corpus
//! (one prompt per line) through
//! [`forward_qwen_deltanet_ternary_with_capture`] position by position
//! (the existing Issue-594 layer-capture hook — the post-layer residual
//! add is exactly the state the TWT S-matrix pools), pools per-layer
//! states into the TWT S matrices, sweeps the PRE-REGISTERED ε grid
//! through the min-max DP, and adjudicates the Phase-1 kill rule with the
//! GDN/attention type split as the achievable floor.
//!
//! Bit-identity by construction: the capture hook only copies when
//! supplied (`forward_qwen_deltanet_ternary` delegates to the same body),
//! so a capture run computes the identical forward as a plain one.
//!
//! Determinism: corpus order + fixed stride + canonical fold order — the
//! artifact records the checkpoint path, corpus BLAKE3, stride, and row
//! cap needed to reproduce the numbers bit-for-bit.
//!
//! ```text
//! cargo run --release -p riir-infer-core --features twt_profile \
//!   --example twt_bonsai_profile -- --gguf data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!   --calib /path/to/prompts.txt [--svcca] [--stride 4] [--max-rows 4096] \
//!   [--out .raw/twt/bonsai_profile.json]
//! ```

use std::io::Write as _;

use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary_with_capture;
use riir_infer_core::deltanet::forward::{effective_rotary_dim, HybridCache, HybridForwardScratch};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::twt::{
    blake3_of, forced_min_blocks, kill_verdict, minmax_partition, Bucket, KillVerdict,
    SMatrixBuilder, SvccaCfg, PRE_REGISTERED_EPS_GRID,
};
use std::path::Path;

fn main() {
    let mut gguf_path = String::from("../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf");
    let mut calib = String::new();
    let mut svcca = false;
    let mut stride = 4usize;
    let mut max_rows = 4096usize;
    let mut max_prompts = 400usize;
    let mut max_prompt_tokens = 512usize;
    let mut out: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf_path = args.next().expect("--gguf needs a path"),
            "--calib" => calib = args.next().expect("--calib needs a path"),
            "--svcca" => svcca = true,
            "--stride" => stride = args.next().expect("--stride needs N").parse().unwrap(),
            "--max-rows" => max_rows = args.next().expect("--max-rows needs N").parse().unwrap(),
            "--max-prompts" => {
                max_prompts = args.next().expect("--max-prompts needs N").parse().unwrap()
            }
            "--max-prompt-tokens" => {
                max_prompt_tokens = args
                    .next()
                    .expect("--max-prompt-tokens needs N")
                    .parse()
                    .unwrap()
            }
            "--out" => out = Some(args.next().expect("--out needs a path")),
            other => panic!("unknown arg {other}"),
        }
    }
    if calib.is_empty() {
        eprintln!(
            "usage: twt_bonsai_profile --gguf <ternary-hybrid.gguf> --calib <prompts.txt> \
             [--svcca] [--stride N] [--max-rows N] [--max-prompts N] \
             [--max-prompt-tokens N] [--out P]"
        );
        std::process::exit(2);
    }
    let corpus_bytes = std::fs::read(&calib).unwrap_or_else(|e| panic!("read {}: {e}", calib));
    let prompts: Vec<&str> = std::str::from_utf8(&corpus_bytes)
        .expect("calibration corpus must be UTF-8")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .take(max_prompts)
        .collect();
    if prompts.is_empty() {
        panic!("calibration corpus {} has no prompt lines", calib);
    }

    // Weights + tokenizer from ONE GGUF (the loader consumes its handle;
    // the tokenizer re-opens — the act_diagonal_calibration shape).
    let t0 = std::time::Instant::now();
    let gguf_path = Path::new(&gguf_path).to_path_buf();
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&gguf_path)
        .unwrap_or_else(|e| panic!("ternary weights: {e}"));
    let tok = {
        let gguf = GgufFile::open(&gguf_path).unwrap_or_else(|e| panic!("re-open gguf: {e}"));
        BpeTokenizer::from_gguf(&gguf).unwrap_or_else(|e| panic!("tokenizer: {e}"))
    };
    let n_layers = config.n_layer;
    let dim = config.n_embd;
    let types_linear: Vec<bool> = weights
        .layer_types
        .iter()
        .map(|&t| t == riir_infer_core::types::DeltaNetLayerType::DeltaNet)
        .collect();
    println!(
        "# bonsai: {} layers={n_layers} hidden={dim} | gdn/attention split {} GDN layers | load {:.1}s",
        gguf_path.display(),
        types_linear.iter().filter(|&&b| b).count(),
        t0.elapsed().as_secs_f32()
    );

    // Cap the KV window at the longest prompt we will feed (the
    // row_logit_floor law — never allocate the advertised context).
    config.block_size = max_prompt_tokens;
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&config);
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
    // The capture buffer: [n_layers][n_embd], reused per position (the
    // builder copies on push — no aliasing across forwards).
    let mut capture = vec![vec![0.0f32; dim]; n_layers];

    let new_builder = || {
        let mut nb = SMatrixBuilder::new(n_layers, dim);
        if svcca {
            nb = nb.with_svcca(SvccaCfg { stride, cap_rows: max_rows });
        }
        nb
    };
    let mut builder = new_builder();

    let mut sequences = 0usize;
    let mut rows_fed = 0u64;
    let mut skipped = 0usize;
    let t1 = std::time::Instant::now();
    for (pi, prompt) in prompts.iter().enumerate() {
        let ids = tok.encode(prompt);
        let ids = &ids[..ids.len().min(max_prompt_tokens)];
        if ids.len() < 2 {
            skipped += 1;
            continue;
        }
        builder.begin_forward(ids.len()).unwrap();
        let mut push_err: Option<String> = None;
        cache.reset();
        for (pos, &token) in ids.iter().enumerate() {
            forward_qwen_deltanet_ternary_with_capture(
                &mut x,
                &weights,
                &mut cache,
                token,
                pos,
                &config,
                &mut scratch,
                &rope_freq,
                Some(&mut capture),
            );
            for (li, slot) in capture.iter().enumerate() {
                // Prompt-band calibration (the laya pilot posture): every
                // row is bucketed explicitly so the artifact says what it
                // does. Generation-band capture is the follow-up row class
                // the Bucket vocabulary already names.
                if let Err(e) = builder
                    .push(li, pos, slot)
                    .and_then(|_| builder.set_row_bucket(pos, Bucket::Prompt))
                    && push_err.is_none()
                {
                    push_err = Some(e.to_string());
                }
            }
        }
        if let Some(e) = push_err {
            eprintln!("[twt] prompt {pi} push failed: {e} — skipped");
            builder = new_builder();
            skipped += 1;
            continue;
        }
        builder.end_forward().unwrap();
        sequences += 1;
        rows_fed += ids.len() as u64;
        if pi % 10 == 0 {
            eprintln!(
                "[twt] {}/{} prompts ({sequences} sequences, {rows_fed} rows, {:.0} tok/s)",
                pi + 1,
                prompts.len(),
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

    // ε sweep + kill rule — the type split (GDN vs attention) is the
    // achievable floor the kill bar reads, never raw L (T1.6).
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
    println!("checkpoint: {} (layers {n_layers}, hidden {dim})", gguf_path.display());
    println!(
        "type split: {forced} forced blocks (achievable floor — the kill bar reads this, not L)"
    );
    println!("ε sweep (cosine meter):");
    for (gi, l) in lines.iter().enumerate() {
        println!("{l}");
        let blocks = &partitions[gi];
        let layout = blocks
            .iter()
            .map(|b| {
                if b.len() > 1 {
                    format!("{}-{}", b.start, b.end - 1)
                } else {
                    format!("{}", b.start)
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        println!("        blocks: {layout}");
    }
    println!("kill verdict (T1.6): {verdict:?}");

    if let Some(path) = out {
        if let Some(dir) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let mut f = std::fs::File::create(&path).expect("out file");
        writeln!(f, "{{").unwrap();
        writeln!(f, "  \"checkpoint\": \"{}\",", gguf_path.display()).unwrap();
        writeln!(f, "  \"n_layers\": {n_layers}, \"dim\": {dim},").unwrap();
        writeln!(
            f,
            "  \"sequences\": {sequences}, \"rows_folded\": {},",
            sm.rows_folded
        )
        .unwrap();
        writeln!(f, "  \"skipped\": {skipped},").unwrap();
        writeln!(f, "  \"corpus_blake3\": \"{}\",", blake3_of(&corpus_bytes)).unwrap();
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
