//! T5.0 — the passthrough-collapsed checkpoint emitter (Issue 022).
//!
//! Reads the Phase-1 profile artifact (the BLAKE3-pinned S matrix over the
//! parent's depth), partitions the main stack at a PRE-REGISTERED ε with
//! the crate's own min-max DP, picks each block's winner (the minimax
//! medoid member — the member whose worst cosine distance to the rest of
//! its block is smallest; ties → lowest index), and emits the collapsed
//! GGUF through the landed writer with EVERY block a MEMBER passthrough
//! (the winner's tensors byte-copied, zero re-quant). The explicit
//! `twt.layer_types` array (winner types) is REQUIRED for a qwen35
//! collapse — the writer refuses without it (T5.0a); a gemma2 collapse
//! (T5.1 lane 1, the f16 control) carries the all-attention array as
//! provenance and needs only the renumbered `gemma2.block_count`.
//!
//! Usage (qwen35 league lane):
//!   cargo run --release --features twt_collapse --example twt_collapse_emit -- \
//!     --parent ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!     --profile .raw/twt/bonsai_ultrachat_profile.json \
//!     --eps 0.05 \
//!     --out /tmp/twt_collapse_pq2_e005.gguf
//!
//! Usage (gemma2 f16 control, T5.1 lane 1):
//!   cargo run --release --features twt_collapse --example twt_collapse_emit -- \
//!     --parent ../riir-train/data/gemma-2-2b-it-f16.gguf \
//!     --profile .raw/twt/gemma2_profile.json \
//!     --eps 0.05 \
//!     --out /tmp/twt_collapse_gemma2_e005.gguf
//!
//! The parent-weights BLAKE3 recorded in `twt.parent_weights_blake3` is
//! BLAKE3 over the parent's tensor PAYLOADS concatenated in tensor-infos
//! file order (globals first) — not the whole file bytes, so the key name
//! says exactly what it commits.

use std::io::BufWriter;

use riir_infer_core::gguf_loader::{GgufFile, GgufValue};
use riir_infer_core::twt::collapse_writer::{
    emit_collapsed_gguf, twt_arm_codes_value, twt_block_table_value, twt_layer_types_value,
    CollapseSpec, LAYER_TYPES_LEGEND, LayerSource,
};
use riir_infer_core::twt::partition::{minmax_partition, partition_worst};
use riir_infer_core::twt::smatrix::SMatrix;
use riir_infer_core::twt::ternarize::TwtArm;
use riir_infer_core::types::DeltaNetLayerType;

fn main() {
    let mut parent_path = None;
    let mut profile_path = None;
    let mut out_path = None;
    let mut eps = 0.05f32;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--parent" => parent_path = Some(std::path::PathBuf::from(args.next().expect("--parent needs a path"))),
            "--profile" => profile_path = Some(std::path::PathBuf::from(args.next().expect("--profile needs a path"))),
            "--out" => out_path = Some(std::path::PathBuf::from(args.next().expect("--out needs a path"))),
            "--eps" => eps = args.next().expect("--eps needs a value").parse().expect("--eps must be f32"),
            other => panic!("unknown arg {other}"),
        }
    }
    let parent_path = parent_path.expect("--parent is required");
    let profile_path = profile_path.expect("--profile is required");
    let out_path = out_path.expect("--out is required");

    // ── parent facts (arch-dispatched: qwen35 = the DeltaNet hybrid lane;
    //    gemma2 = the T5.1 lane-1 f16 control — a single-operator stack in
    //    this repo (no SWA), so no index-derived layer typing survives to
    //    break under renumbering and no prism keys exist to renumber) ──
    let parent = GgufFile::open(&parent_path).unwrap_or_else(|e| panic!("open {}: {e}", parent_path.display()));
    let arch = parent
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_owned();
    let (n_layer, block_count_key, parent_types) = match arch.as_str() {
        "qwen35" => {
            let n_layer =
                parent.metadata_u64("qwen35.block_count").expect("qwen35.block_count") as usize;
            let interval = parent
                .metadata_u64("qwen35.full_attention_interval")
                .unwrap_or(4) as usize;
            let nextn = parent
                .metadata_u64("qwen35.nextn_predict_layers")
                .unwrap_or(0) as usize;
            assert_eq!(
                nextn, 0,
                "this lane is main-stack-only; a nextn parent needs its own plan"
            );
            // The same derivation the loader runs (twt.layer_types makes the
            // collapsed file independent of it — this is only the PARENT's truth).
            let types: Vec<DeltaNetLayerType> = (0..n_layer)
                .map(|i| {
                    if (i + 1).is_multiple_of(interval) {
                        DeltaNetLayerType::Attention
                    } else {
                        DeltaNetLayerType::DeltaNet
                    }
                })
                .collect();
            (n_layer, "qwen35.block_count".to_owned(), types)
        }
        "gemma2" => {
            let n_layer =
                parent.metadata_u64("gemma2.block_count").expect("gemma2.block_count") as usize;
            let types = vec![DeltaNetLayerType::Attention; n_layer];
            (n_layer, "gemma2.block_count".to_owned(), types)
        }
        other => panic!("unsupported parent arch {other} (this lane: qwen35 | gemma2)"),
    };
    eprintln!(
        "[twt-emit] parent {} arch {arch}: {n_layer} layers, {} tensors",
        parent_path.display(),
        parent.tensor_infos.len(),
    );

    // ── the profile artifact → S matrix ──
    let profile_text = std::fs::read_to_string(&profile_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", profile_path.display()));
    let profile: serde_json::Value =
        serde_json::from_str(&profile_text).expect("profile json");
    let p_layers = profile["n_layers"].as_u64().expect("n_layers") as usize;
    assert_eq!(p_layers, n_layer, "profile S is for a different layer count");
    let entries = profile["S_upper"].as_array().expect("S_upper");
    let s = SMatrix::from_fn(n_layer, |i, j| {
        // from_fn only reads i < j in fixed row-major order; scan-linear is
        // fine at 2016 entries (the whole matrix is built once).
        entries
            .iter()
            .find_map(|e| {
                let a = e[0].as_u64().unwrap() as usize;
                let b = e[1].as_u64().unwrap() as usize;
                if (a, b) == (i, j) {
                    Some(e[2].as_f64().unwrap() as f32)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| panic!("S_upper missing entry ({i},{j})"))
    });

    // ── the pre-registered ε + the crate's own DP ──
    let blocks = minmax_partition(&s, eps).expect("minmax_partition");
    let worst = partition_worst(&s, &blocks);
    eprintln!(
        "[twt-emit] ε={eps}: {} blocks (depth {:.1}% of {n_layer}), partition worst intra-block distance {worst:.4}",
        blocks.len(),
        100.0 * blocks.len() as f32 / n_layer as f32,
    );

    // ── winner per block: the minimax medoid member ──
    let winners: Vec<usize> = blocks
        .iter()
        .map(|b| {
            let (start, end) = (b.start, b.end);
            if end - start == 1 {
                return start;
            }
            let mut best = start;
            let mut best_worst = f32::INFINITY;
            for m in start..end {
                let w = (start..end)
                    .filter(|&k| k != m)
                    .map(|k| s.get(m, k))
                    .fold(0.0f32, f32::max);
                if w < best_worst {
                    best_worst = w;
                    best = m;
                }
            }
            best
        })
        .collect();

    // ── the collapse spec ──
    let m = blocks.len();
    let spec_blocks: Vec<(usize, usize, LayerSource)> = blocks
        .iter()
        .zip(winners.iter())
        .map(|(b, &w)| (b.start, b.end, LayerSource::Member(w)))
        .collect();
    let collapsed_types: Vec<DeltaNetLayerType> =
        winners.iter().map(|&w| parent_types[w]).collect();
    let pair_list: Vec<(usize, usize)> =
        blocks.iter().map(|b| (b.start, b.end)).collect();

    let mut overrides = vec![(block_count_key, GgufValue::U64(m as u64))];
    if arch == "qwen35"
        && parent
            .metadata_u64("qwen35.nextn_predict_layers")
            .unwrap_or(0)
            > 0
    {
        overrides.push(("qwen35.nextn_predict_layers".to_owned(), GgufValue::U64(0)));
    }

    // Parent-weights BLAKE3: payload bytes in tensor-infos file order.
    let t_hash = std::time::Instant::now();
    let mut hasher = blake3::Hasher::new();
    for info in &parent.tensor_infos {
        hasher.update(parent.tensor_slice(&info.name).expect("tensor bytes"));
    }
    let weights_blake3 = hasher.finalize().to_hex().to_string();
    eprintln!(
        "[twt-emit] parent-weights BLAKE3 {weights_blake3} ({:.1}s)",
        t_hash.elapsed().as_secs_f32()
    );

    let spec = CollapseSpec {
        blocks: spec_blocks,
        metadata_overrides: overrides,
        twt_meta: vec![
            ("twt.block_table".to_owned(), twt_block_table_value(&pair_list)),
            (
                "twt.arm_codes".to_owned(),
                twt_arm_codes_value(&vec![TwtArm::Member; m]),
            ),
            ("twt.arm_legend".to_owned(), GgufValue::String(TwtArm::LEGEND.to_owned())),
            ("twt.layer_types".to_owned(), twt_layer_types_value(&collapsed_types)),
            (
                "twt.layer_types_legend".to_owned(),
                GgufValue::String(LAYER_TYPES_LEGEND.to_owned()),
            ),
            ("twt.parent_weights_blake3".to_owned(), GgufValue::String(weights_blake3.clone())),
            ("twt.partition_eps".to_owned(), GgufValue::F32(eps)),
            (
                "twt.partition_worst".to_owned(),
                GgufValue::F32(worst),
            ),
            (
                "twt.profile_corpus_blake3".to_owned(),
                GgufValue::String(
                    profile["corpus_blake3"].as_str().expect("corpus_blake3").to_owned(),
                ),
            ),
        ],
    };

    // ── emit (streamed from the mmap) ──
    let out_file = std::fs::File::create(&out_path)
        .unwrap_or_else(|e| panic!("create {}: {e}", out_path.display()));
    let mut out = BufWriter::with_capacity(1 << 22, out_file);
    let t_emit = std::time::Instant::now();
    let stats = emit_collapsed_gguf(&parent, &spec, &mut out).expect("emit_collapsed_gguf");
    use std::io::Write;
    out.flush().expect("flush");
    eprintln!(
        "[twt-emit] wrote {} in {:.1}s ({} tensors)",
        out_path.display(),
        t_emit.elapsed().as_secs_f32(),
        stats.n_tensors,
    );

    // ── the human-readable record (stdout — quote this in the bench doc) ──
    println!("# twt_collapse_emit — T5.0 passthrough-collapsed checkpoint");
    println!("# parent: {} ({} layers, {} tensors)", parent_path.display(), n_layer, parent.tensor_infos.len());
    println!("# profile: {} (corpus BLAKE3 {})", profile_path.display(), profile["corpus_blake3"].as_str().unwrap_or("?"));
    println!("# ε = {eps} (pre-registered grid point), partition worst = {worst:.4}");
    println!("# blocks: {} of {n_layer} layers ({:.1}% depth), winner = minimax medoid member", m, 100.0 * m as f32 / n_layer as f32);
    println!("# output: {} ({} bytes, {} tensors)", out_path.display(), stats.bytes_written, stats.n_tensors);
    println!("# parent_weights_blake3: {weights_blake3}");
    println!("block\tspan\twinner\twinner_type\tblock_worst");
    for (i, (b, &w)) in blocks.iter().zip(winners.iter()).enumerate() {
        let bw = (b.start..b.end)
            .filter(|&k| k != w)
            .map(|k| s.get(w, k))
            .fold(0.0f32, f32::max);
        println!(
            "{}\t[{}, {})\t{}\t{:?}\t{bw:.4}",
            i, b.start, b.end, w, collapsed_types[i],
        );
    }
}
