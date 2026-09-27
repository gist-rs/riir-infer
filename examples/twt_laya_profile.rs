//! `twt_laya_profile` — the Issue 022 T1.3 capture driver (feature
//! `twt_profile`).
//!
//! Loads a pinned laya checkpoint, walks a plain-text calibration corpus
//! (one prompt per line) through [`Encoder::forward_capture`], pools the
//! per-layer residual states into the TWT S matrices (cosine meter, plus
//! the SVCCA arm with `--svcca`), sweeps the PRE-REGISTERED ε grid
//! through the min-max DP, and adjudicates the Phase-1 kill rule.
//!
//! Determinism: corpus order + fixed stride + canonical fold order — the
//! artifact records everything needed to reproduce the numbers bit-for-bit
//! (checkpoint name, BLAKE3 of the corpus bytes, stride, max rows).
//!
//! ```text
//! cargo run --release -p riir-infer-core --features twt_profile \
//!   --example twt_laya_profile -- --checkpoint english \
//!   --calib /path/to/prompts.txt [--svcca] [--stride 4] [--max-rows 4096] \
//!   [--out .raw/twt/english_profile.json]
//! ```

use std::collections::HashMap;
use std::io::Write as _;

use riir_infer_core::twt::{
    blake3_of, forced_min_blocks, kill_verdict, minmax_partition, Bucket, KillVerdict,
    SMatrixBuilder, SvccaCfg, PRE_REGISTERED_EPS_GRID,
};
use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::backend::Cpu;
use riir_infer_laya::laya::riir::encoder::{CaptureStage, Encoder};
use riir_infer_laya::laya::riir::weights as lane_weights;
use riir_infer_laya::laya::tokenize::Tok;
use riir_infer_laya::laya::weights::{ensure_checkpoint, weights_root};

fn main() {
    let mut checkpoint = Checkpoint::English;
    let mut calib = String::new();
    let mut svcca = false;
    let mut stride = 4usize;
    let mut max_rows = 4096usize;
    let mut out: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--checkpoint" => {
                let v = args.next().expect("--checkpoint needs a value");
                checkpoint = Checkpoint::ALL
                    .into_iter()
                    .find(|c| c.subfolder() == v)
                    .unwrap_or_else(|| {
                        panic!("unknown checkpoint {v:?} (english|multilingual|typed)")
                    });
            }
            "--calib" => calib = args.next().expect("--calib needs a path"),
            "--svcca" => svcca = true,
            "--stride" => stride = args.next().expect("--stride needs N").parse().unwrap(),
            "--max-rows" => max_rows = args.next().expect("--max-rows needs N").parse().unwrap(),
            "--out" => out = Some(args.next().expect("--out needs a path")),
            other => panic!("unknown arg {other}"),
        }
    }
    if calib.is_empty() {
        eprintln!(
            "usage: twt_laya_profile --checkpoint <english|multilingual|typed> --calib <prompts.txt> [--svcca] [--stride N] [--max-rows N] [--out P]"
        );
        std::process::exit(2);
    }
    let corpus_bytes = std::fs::read(&calib).unwrap_or_else(|e| panic!("read {}: {e}", calib));
    let prompts: Vec<&str> = std::str::from_utf8(&corpus_bytes)
        .expect("calibration corpus must be UTF-8")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    if prompts.is_empty() {
        panic!("calibration corpus {} has no prompt lines", calib);
    }

    // Checkpoint: locate → verify (→ download on first use) → parse → load.
    let ckpt_name = checkpoint.subfolder();
    let dir =
        ensure_checkpoint(&weights_root(), checkpoint).unwrap_or_else(|e| panic!("checkpoint fetch: {e}"));
    let (_agent_cfg, enc_cfg) =
        riir_infer_laya::laya::config::load_checkpoint_configs(&dir, ckpt_name)
            .unwrap_or_else(|e| panic!("configs: {e}"));
    let mut map: HashMap<String, lane_weights::Weights> =
        lane_weights::load(&dir.join("model.safetensors"), ckpt_name)
            .unwrap_or_else(|e| panic!("weights: {e}"));
    let encoder = Encoder::from_map(&mut map, enc_cfg.clone(), ckpt_name)
        .unwrap_or_else(|e| panic!("encoder: {e}"));
    let tok = Tok::from_dir(&dir, ckpt_name).unwrap_or_else(|e| panic!("tokenizer: {e}"));
    let backend = Cpu;
    encoder.warm(&backend);

    let (n_layers, dim) = (enc_cfg.layers, enc_cfg.hidden);
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
    for (pi, prompt) in prompts.iter().enumerate() {
        let ids = match tok.encode(prompt) {
            Ok(v) => v,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        if ids.len() < 2 {
            skipped += 1;
            continue;
        }
        builder.begin_forward(ids.len()).unwrap();
        let mut push_err: Option<String> = None;
        let res = encoder.forward_capture(&backend, &ids, &[ids.len()], &mut |stage, row, state| {
            // The S matrix pools LAYER OUTPUTS; the embedding state is the
            // block-input side (Phase 3's audition surface), not a
            // partitionable layer — recorded by the stage vocabulary, not
            // pooled.
            let CaptureStage::AfterLayer(li) = stage else {
                return;
            };
            // Every row of a classification forward is prompt-band; set
            // explicitly so the artifact says what it does.
            if let Err(e) = builder
                .push(li, row, state)
                .and_then(|_| builder.set_row_bucket(row, Bucket::Prompt))
                && push_err.is_none()
            {
                push_err = Some(e.to_string());
            }
        });
        if let Err(e) = res {
            eprintln!("[twt] question {pi} capture failed: {e} — skipped");
            builder = new_builder();
            skipped += 1;
            continue;
        }
        if let Some(e) = push_err {
            eprintln!("[twt] question {pi} push failed: {e} — skipped");
            builder = new_builder();
            skipped += 1;
            continue;
        }
        builder.end_forward().unwrap();
        sequences += 1;
        rows_fed += ids.len() as u64;
        if pi % 50 == 0 {
            eprintln!(
                "[twt] {}/{} prompts ({sequences} sequences, {rows_fed} rows)",
                pi + 1,
                prompts.len()
            );
        }
    }

    let sm = builder.finalize().unwrap();
    let s = sm.cosine();
    eprintln!(
        "[twt] S built: {n_layers}×{n_layers} over {sequences} sequences / {} folded rows (dim {dim}, skipped {skipped})",
        sm.rows_folded
    );

    // ε sweep + kill rule.
    let forced = forced_min_blocks(&enc_cfg.sliding);
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
    println!("checkpoint: {ckpt_name} (layers {n_layers}, hidden {dim})");
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
        writeln!(f, "  \"checkpoint\": \"{ckpt_name}\",").unwrap();
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
