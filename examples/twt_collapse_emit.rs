//! T5.0 — the passthrough-collapsed checkpoint emitter (Issue 022).
//!
//! Three modes:
//!
//! 1. DEFAULT (the no-apply-path lane): reads the Phase-1 profile
//!    artifact (the BLAKE3-pinned S matrix over the parent's depth),
//!    partitions the main stack at a PRE-REGISTERED ε with the crate's
//!    own min-max DP, picks each block's winner (the minimax medoid
//!    member), and emits the collapsed GGUF through the landed writer
//!    with EVERY block a MEMBER passthrough (the winner's tensors
//!    byte-copied, zero re-quant).
//! 2. `--typed`: the same, but the partition is the TYPE-AWARE DP
//!    (`minmax_partition_typed`) — every multi-layer block is homogeneous,
//!    the merge-feasible structure. Winners stay medoid passthroughs (the
//!    equal-depth CONTROL the merged arm is judged against).
//! 3. `--selection <json>`: a merged emit. The block table + winners come
//!    from `twt_bonsai_audition`'s selection table: member winners are
//!    passthroughs, merged winners are RE-MATERIALIZED here at the name
//!    level (dequant members → merge op → arm → wire bytes; norms merged
//!    with the mean; a_log/dt_bias/conv1d byte-copied from the block's
//!    params_member — the same never-average menu the audition applies).
//!
//! The explicit `twt.layer_types` array (winner types) is REQUIRED for a
//! qwen35 collapse — the writer refuses without it (T5.0a); a gemma2
//! collapse (T5.1 lane 1, the f16 control) carries the all-attention array
//! as provenance and needs only the renumbered `gemma2.block_count`.
//!
//! Usage (qwen35 league lane):
//!   cargo run --release --features twt_collapse --example twt_collapse_emit -- \
//!     --parent ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!     --profile .raw/twt/bonsai_ultrachat_profile.json \
//!     --eps 0.05 \
//!     --out /tmp/twt_collapse_pq2_e005.gguf [--typed] [--selection sel.json]
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

use std::collections::BTreeMap;
use std::io::BufWriter;

use riir_infer_core::gguf_loader::{GgufFile, GgmlType, GgufValue};
use riir_infer_core::quant::ptq1_0::repack_ptq1_0_to_ternary_group;
use riir_infer_core::quant::q2_0::repack_q2_0_to_ternary_group;
use riir_infer_core::twt::collapse_writer::{
    TensorOut, emit_collapsed_gguf, q2_0_wire_bytes, twt_arm_codes_value, twt_block_table_value,
    twt_layer_types_value, CollapseSpec, LAYER_TYPES_LEGEND, LayerSource,
};
use riir_infer_core::twt::partition::{minmax_partition, minmax_partition_typed, partition_worst};
use riir_infer_core::twt::smatrix::SMatrix;
use riir_infer_core::twt::ternarize::{
    Materialized, TwtArm, arm_sign_majority, arm_source_quant,
};
use riir_infer_core::types::DeltaNetLayerType;
use riir_infer_core::twt::audition::{merge_mean, merge_rdsc};

/// Dequantize one parent tensor (Q2_0 | PTQ1_0 | F32 | BF16 | F16) to
/// dense f32 (row-major, `rows × cols`).
fn tensor_dense(parent: &GgufFile, name: &str) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    let info = parent
        .tensor_info(name)
        .ok_or_else(|| anyhow::anyhow!("tensor {name} not found"))?;
    let cols = info.shape[0]; // ne[0] = innermost
    let rows: usize = info.shape[1..].iter().product();
    let raw = parent
        .tensor_slice(name)
        .ok_or_else(|| anyhow::anyhow!("tensor {name} has no bytes"))?;
    let container = match info.ggml_type {
        GgmlType::Q2_0 => {
            let mut bytes = Vec::with_capacity(rows * cols / 32 * 34);
            bytes.extend_from_slice(raw);
            let blocks: Vec<riir_infer_core::quant::q2_0::BlockQ2_0> =
                bytemuck::cast_slice(&bytes).to_vec();
            repack_q2_0_to_ternary_group(&blocks, rows, cols)?
        }
        GgmlType::PTQ1_0 => {
            let mut bytes = Vec::with_capacity(rows * cols / 128 * 28);
            bytes.extend_from_slice(raw);
            let blocks: Vec<riir_infer_core::quant::ptq1_0::BlockPtq1_0> =
                bytemuck::cast_slice(&bytes).to_vec();
            repack_ptq1_0_to_ternary_group(&blocks, rows, cols)?
        }
        GgmlType::F32 | GgmlType::BF16 | GgmlType::F16 => {
            return Ok((parent.dequant_f16_to_f32(name)?, rows, cols));
        }
        other => anyhow::bail!("tensor {name} is {other:?} — expected ternary or dense"),
    };
    let dense =
        riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights::dequant_proj_to_dense(
            &container,
        );
    Ok((dense, rows, cols))
}

/// The merged-payload menu for a name suffix (mirrors the audition's
/// per-field construction EXACTLY — a divergence here means the emitted
/// checkpoint is not the auditioned candidate):
/// - `*norm.weight` → merged mean (RMSNorm gammas);
/// - `ssm_a`, `ssm_dt.bias`, `ssm_conv1d.weight` → byte-copy from the
///   block's params_member (never averaged — T3.1);
/// - everything else → the block's arm: ternary projections via
///   dequant→arm→Q2_0 wire; ssm_alpha/beta (the dense escape set) via the
///   dense merge → F32.
enum SuffixMenu {
    NormMean,
    ParamCopy,
    Arm,
}

fn suffix_menu(suffix: &str) -> SuffixMenu {
    if suffix.ends_with("norm.weight") {
        SuffixMenu::NormMean
    } else if suffix == "ssm_a" || suffix == "ssm_dt.bias" || suffix == "ssm_conv1d.weight" {
        SuffixMenu::ParamCopy
    } else {
        SuffixMenu::Arm
    }
}

/// Build one merged block's payloads at the NAME level (the writer's
/// `LayerSource::Merged` map). `op`/`arm`/`dense_op` come from the
/// audition's selection; `params_member` is the never-average anchor.
#[allow(clippy::too_many_lines)]
fn merged_block_payloads(
    parent: &GgufFile,
    start: usize,
    end: usize,
    op_name: &str,
    arm_name: &str,
    dense_op_name: &str,
    params_member: usize,
) -> anyhow::Result<BTreeMap<String, TensorOut>> {
    let merge = |refs: Vec<Vec<f32>>| -> anyhow::Result<Vec<f32>> {
        let refs: Vec<&[f32]> = refs.iter().map(|v| v.as_slice()).collect();
        Ok(match op_name {
            "mean" => merge_mean(refs)?,
            "rdsc" => merge_rdsc(refs)?,
            other => anyhow::bail!("unknown merge op {other}"),
        })
    };
    let dense_merge = |refs: Vec<Vec<f32>>| -> anyhow::Result<Vec<f32>> {
        let refs: Vec<&[f32]> = refs.iter().map(|v| v.as_slice()).collect();
        Ok(match dense_op_name {
            "mean" => merge_mean(refs)?,
            "rdsc" => merge_rdsc(refs)?,
            other => anyhow::bail!("unknown dense op {other}"),
        })
    };

    let first = format!("blk.{start}.");
    let suffixes: Vec<String> = parent
        .tensor_infos
        .iter()
        .filter(|i| i.name.starts_with(&first))
        .map(|i| i.name[first.len()..].to_owned())
        .collect();
    let mut out = BTreeMap::new();
    for suffix in &suffixes {
        let member_names: Vec<String> = (start..end)
            .map(|li| format!("blk.{li}.{suffix}"))
            .collect();
        for name in &member_names {
            anyhow::ensure!(
                parent.tensor_info(name).is_some(),
                "member tensor {name} missing — the merge needs every member's suffix set"
            );
        }
        let info = parent.tensor_info(&member_names[0]).unwrap();
        let shape = info.shape.clone();
        match suffix_menu(suffix) {
            SuffixMenu::NormMean => {
                let dense: Vec<Vec<f32>> = member_names
                    .iter()
                    .map(|n| parent.dequant_f16_to_f32(n))
                    .collect::<anyhow::Result<_>>()?;
                let merged = dense_merge(dense)?;
                out.insert(
                    suffix.clone(),
                    TensorOut {
                        ggml_type: GgmlType::F32,
                        shape,
                        data: merged.iter().flat_map(|f| f.to_le_bytes()).collect(),
                    },
                );
            }
            SuffixMenu::ParamCopy => {
                let src = format!("blk.{params_member}.{suffix}");
                let data = parent
                    .tensor_slice(&src)
                    .ok_or_else(|| anyhow::anyhow!("param tensor {src} missing"))?
                    .to_vec();
                out.insert(
                    suffix.clone(),
                    TensorOut { ggml_type: info.ggml_type, shape, data },
                );
            }
            SuffixMenu::Arm => match info.ggml_type {
                GgmlType::Q2_0 | GgmlType::PTQ1_0 => {
                    let dense: Vec<Vec<f32>> = member_names
                        .iter()
                        .map(|n| tensor_dense(parent, n).map(|(d, _, _)| d))
                        .collect::<anyhow::Result<_>>()?;
                    let rows: usize = shape[1..].iter().product();
                    let cols = shape[0];
                    let materialized = match arm_name {
                        "sign_majority" => {
                            let refs: Vec<&[f32]> = dense.iter().map(|v| v.as_slice()).collect();
                            arm_sign_majority(&refs, rows, cols)?
                        }
                        "source_quant" => {
                            let merged = merge(dense)?;
                            arm_source_quant(&merged, rows, cols)?
                        }
                        other => anyhow::bail!("unknown arm {other}"),
                    };
                    let w = match materialized {
                        Materialized::Ternary(w) => *w,
                        Materialized::DenseF16(_) => {
                            anyhow::bail!("arm A is not a deployable merged payload")
                        }
                    };
                    out.insert(
                        suffix.clone(),
                        TensorOut {
                            ggml_type: GgmlType::Q2_0,
                            shape,
                            data: q2_0_wire_bytes(&w)?,
                        },
                    );
                }
                GgmlType::F32 | GgmlType::BF16 | GgmlType::F16 => {
                    // ssm_alpha/ssm_beta — the dense gate projections
                    let dense: Vec<Vec<f32>> = member_names
                        .iter()
                        .map(|n| tensor_dense(parent, n).map(|(d, _, _)| d))
                        .collect::<anyhow::Result<_>>()?;
                    let merged = dense_merge(dense)?;
                    out.insert(
                        suffix.clone(),
                        TensorOut {
                            ggml_type: GgmlType::F32,
                            shape,
                            data: merged.iter().flat_map(|f| f.to_le_bytes()).collect(),
                        },
                    );
                }
                other => anyhow::bail!("tensor {suffix} is {other:?} — no merge menu"),
            },
        }
    }
    Ok(out)
}

fn main() {
    let mut parent_path = None;
    let mut profile_path = None;
    let mut out_path = None;
    let mut eps = 0.05f32;
    let mut typed = false;
    let mut selection_path: Option<std::path::PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--parent" => parent_path = Some(std::path::PathBuf::from(args.next().expect("--parent needs a path"))),
            "--profile" => profile_path = Some(std::path::PathBuf::from(args.next().expect("--profile needs a path"))),
            "--out" => out_path = Some(std::path::PathBuf::from(args.next().expect("--out needs a path"))),
            "--eps" => eps = args.next().expect("--eps needs a value").parse().expect("--eps must be f32"),
            "--typed" => typed = true,
            "--selection" => {
                selection_path = Some(std::path::PathBuf::from(args.next().expect("--selection needs a path")))
            }
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
    let (n_layer, block_count_key, parent_types, nextn) = match arch.as_str() {
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
            (n_layer, "qwen35.block_count".to_owned(), types, nextn)
        }
        "gemma2" => {
            let n_layer =
                parent.metadata_u64("gemma2.block_count").expect("gemma2.block_count") as usize;
            let types = vec![DeltaNetLayerType::Attention; n_layer];
            (n_layer, "gemma2.block_count".to_owned(), types, 0usize)
        }
        other => panic!("unsupported parent arch {other} (this lane: qwen35 | gemma2)"),
    };
    // The typed DP's flag slice (true = DeltaNet) — same convention as
    // `forced_min_blocks`.
    let parent_flags: Vec<bool> = parent_types
        .iter()
        .map(|&t| t == DeltaNetLayerType::DeltaNet)
        .collect();
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

    // ── the three modes ──
    let selection_text = selection_path
        .as_ref()
        .map(|p| std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display())));
    let selection: Option<serde_json::Value> =
        selection_text.as_deref().map(|t| serde_json::from_str(t).expect("selection json"));

    // (mode 3) merged emit: block table + winners from the audition's
    // selection table; the partition inputs (eps) come from the selection.
    let blocks: Vec<(usize, usize)>;
    let mut spec_blocks: Vec<(usize, usize, LayerSource)>;
    let arms: Vec<TwtArm>;
    if let Some(sel) = &selection {
        eps = sel["eps"].as_f64().expect("selection eps") as f32;
        let sel_blocks = sel["blocks"].as_array().expect("selection blocks");
        let mut pair_list = Vec::with_capacity(sel_blocks.len());
        let mut arm_list = Vec::with_capacity(sel_blocks.len());
        spec_blocks = Vec::with_capacity(sel_blocks.len());
        for b in sel_blocks {
            let (st, en) = (
                b["start"].as_u64().expect("start") as usize,
                b["end"].as_u64().expect("end") as usize,
            );
            pair_list.push((st, en));
            let arm_str = b["arm"].as_str().expect("arm");
            let winner = b["winner"].as_str().expect("winner");
            if arm_str == "member" {
                assert!(winner.starts_with("member:"), "member winner id malformed: {winner}");
                let w: usize = winner["member:".len()..].parse().expect("member idx");
                assert!((st..en).contains(&w), "winner {w} outside block [{st},{en})");
                arm_list.push(TwtArm::Member);
                spec_blocks.push((st, en, LayerSource::Member(w)));
            } else {
                let op = b["op"].as_str().expect("op");
                let dense_op = b["dense_op"].as_str().expect("dense_op");
                let params_member = b["params_member"].as_u64().expect("params_member") as usize;
                eprintln!(
                    "[twt-emit] block [{st},{en}): re-materializing {op}/{arm_str} (dense {dense_op}, params from member {params_member})..."
                );
                let payloads = merged_block_payloads(
                    &parent,
                    st,
                    en,
                    op,
                    arm_str,
                    dense_op,
                    params_member,
                )
                .unwrap_or_else(|e| panic!("merged payloads for [{st},{en}): {e}"));
                arm_list.push(match arm_str {
                    "sign_majority" => TwtArm::SignMajority,
                    "source_quant" => TwtArm::SourceQuant,
                    other => panic!("unknown arm {other}"),
                });
                spec_blocks.push((st, en, LayerSource::Merged(payloads)));
            }
        }
        blocks = pair_list;
        arms = arm_list;
    } else {
        // (modes 1+2) DP partition + minimax-medoid passthrough winners
        let dp = if typed {
            minmax_partition_typed(&s, eps, &parent_flags)
        } else {
            minmax_partition(&s, eps)
        }
        .expect("minmax_partition");
        let dp_worst = partition_worst(&s, &dp);
        eprintln!(
            "[twt-emit] ε={eps}{}: {} blocks (depth {:.1}% of {n_layer}), partition worst intra-block distance {dp_worst:.4}",
            if typed { " TYPED" } else { "" },
            dp.len(),
            100.0 * dp.len() as f32 / n_layer as f32,
        );
        blocks = dp.iter().map(|b| (b.start, b.end)).collect();
        arms = vec![TwtArm::Member; dp.len()];
        spec_blocks = Vec::with_capacity(dp.len());
        for b in &dp {
            let (start, end) = (b.start, b.end);
            if end - start == 1 {
                spec_blocks.push((start, end, LayerSource::Member(start)));
                continue;
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
            spec_blocks.push((start, end, LayerSource::Member(best)));
        }
    }
    let m = blocks.len();

    // winners per block (for layer types + the record): the member source
    // where there is one, else the block's first member (a typed merged
    // block is homogeneous — first member's type IS the block's type).
    let winners: Vec<usize> = spec_blocks
        .iter()
        .map(|&(st, _en, ref src)| match src {
            LayerSource::Member(w) => *w,
            LayerSource::Merged(_) => st,
        })
        .collect();

    // ── the collapse spec ──
    let collapsed_types: Vec<DeltaNetLayerType> =
        winners.iter().map(|&w| parent_types[w]).collect();
    let pair_list: Vec<(usize, usize)> = blocks.clone();
    let block_objs: Vec<riir_infer_core::twt::partition::Block> = blocks
        .iter()
        .map(|&(st, en)| riir_infer_core::twt::partition::Block { start: st, end: en })
        .collect();
    let worst = partition_worst(&s, &block_objs);

    let mut overrides = vec![(block_count_key, GgufValue::U64(m as u64))];
    if nextn > 0 {
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

    // Selection-table BLAKE3 (merged mode): the emit is reproducible only
    // together with the audition artifact it consumes.
    let selection_blake3 = selection_text
        .as_deref()
        .map(|t| blake3::hash(t.as_bytes()).to_string());

    let mut twt_meta = vec![
        ("twt.block_table".to_owned(), twt_block_table_value(&pair_list)),
        ("twt.arm_codes".to_owned(), twt_arm_codes_value(&arms)),
        ("twt.arm_legend".to_owned(), GgufValue::String(TwtArm::LEGEND.to_owned())),
        ("twt.layer_types".to_owned(), twt_layer_types_value(&collapsed_types)),
        (
            "twt.layer_types_legend".to_owned(),
            GgufValue::String(LAYER_TYPES_LEGEND.to_owned()),
        ),
        ("twt.parent_weights_blake3".to_owned(), GgufValue::String(weights_blake3.clone())),
        ("twt.partition_eps".to_owned(), GgufValue::F32(eps)),
        ("twt.partition_worst".to_owned(), GgufValue::F32(worst)),
        (
            "twt.profile_corpus_blake3".to_owned(),
            GgufValue::String(
                profile["corpus_blake3"].as_str().expect("corpus_blake3").to_owned(),
            ),
        ),
        (
            "twt.partition_mode".to_owned(),
            GgufValue::String(
                if selection.is_some() {
                    "typed+auditioned"
                } else if typed {
                    "typed"
                } else {
                    "unconstrained"
                }
                .to_owned(),
            ),
        ),
    ];
    if let Some(h) = &selection_blake3 {
        twt_meta.push(("twt.selection_blake3".to_owned(), GgufValue::String(h.clone())));
    }

    let spec = CollapseSpec {
        blocks: spec_blocks,
        metadata_overrides: overrides,
        twt_meta,
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
    let mode = if selection.is_some() {
        "merged (audition selection)"
    } else if typed {
        "typed medoid passthrough"
    } else {
        "passthrough-collapsed"
    };
    println!("# twt_collapse_emit — T5.0 collapsed checkpoint ({mode})");
    println!("# parent: {} ({} layers, {} tensors)", parent_path.display(), n_layer, parent.tensor_infos.len());
    println!("# profile: {} (corpus BLAKE3 {})", profile_path.display(), profile["corpus_blake3"].as_str().unwrap_or("?"));
    println!("# ε = {eps}, partition mode = {}, partition worst = {worst:.4}", if selection.is_some() || typed { "typed" } else { "unconstrained" });
    if let (Some(sp), Some(h)) = (selection_path.as_ref(), selection_blake3.as_ref()) {
        println!("# selection: {} (blake3 {h})", sp.display());
    }
    println!("# blocks: {} of {n_layer} layers ({:.1}% depth)", m, 100.0 * m as f32 / n_layer as f32);
    println!("# output: {} ({} bytes, {} tensors)", out_path.display(), stats.bytes_written, stats.n_tensors);
    println!("# parent_weights_blake3: {weights_blake3}");
    println!("block\tspan\twinner\twinner_type\tblock_worst\tarm");
    for (i, ((st, en), &w)) in blocks.iter().zip(winners.iter()).enumerate() {
        let bw = (*st..*en)
            .filter(|&k| k != w)
            .map(|k| s.get(w, k))
            .fold(0.0f32, f32::max);
        println!(
            "{i}\t[{st}, {en})\t{w}\t{:?}\t{bw:.4}\t{:?}",
            collapsed_types[i], arms[i],
        );
    }
}
