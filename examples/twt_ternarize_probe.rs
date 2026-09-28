//! `twt_ternarize_probe` — Issue 022 T4.2's operator-level arm-damage
//! probe over REAL Bonsai blocks (dequant-only; the parent forward never
//! runs, so this needs no calibration corpus and no apply path).
//!
//! For each requested block, the probe reads ONE projection tensor per
//! member layer from the parent ternary GGUF (Q2_0 or PTQ1_0 wire —
//! whichever the tensor carries), dequantizes to f32, forms the
//! `merge_mean` operator f̄, materializes the Phase-4 arms, and measures
//! each arm's **materialization damage** against the f32 f̄ over
//! deterministic synthetic inputs:
//!
//! `damage(arm) = Σ_x ‖(W_arm − W̄)x‖² / Σ_x ‖W̄x‖²`
//!
//! Honest reading (the pre-registered law, `twt::ternarize`): the T4.2 κ
//! budget compares END-TO-END mapping errors (arm vs PARENT, surrogate
//! vs PARENT) — the audition's job. This probe is the operator-level
//! HALF: damage is the arm's own price, and the ratio
//! `damage(armC) / (κ−1)·damage(armA)` says whether a block's surrogate
//! error must exceed a floor before arm C can survive the budget at all.
//! Quoted numbers, never a gate.
//!
//! ```text
//! cargo run --release -p riir-infer-core --features twt_collapse \
//!   --example twt_ternarize_probe -- \
//!   --gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!   --blocks 5:8,20:23 [--suffix in_proj] [--n-x 8] [--max-rows 5120]
//! ```
//!
//! `--blocks a:b,c:d` — half-open `[a, b)` member triples (the DP's
//! block table, e.g. from the `twt_bonsai_profile` artifact's sweep
//! table). Blocks shorter than 2 members are refused (there is nothing
//! to merge into).

use std::path::PathBuf;

use riir_infer_core::gguf_loader::{GgufFile, GgmlType};
use riir_infer_core::quant::ptq1_0::repack_ptq1_0_to_ternary_group;
use riir_infer_core::quant::q2_0::repack_q2_0_to_ternary_group;
use riir_infer_core::twt::ternarize::{
    arm_dense_f16, arm_sign_majority, arm_source_quant, budget_ratio, materialization_rel_err,
    KAPPA_BUDGET,
};

/// Dequantize one parent tensor (Q2_0 | PTQ1_0 | F32) to a
/// `TernaryGroupWeights` container, then to dense f32.
fn tensor_to_dense(parent: &GgufFile, name: &str) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    let info = parent
        .tensor_infos
        .iter()
        .find(|i| i.name == name)
        .ok_or_else(|| anyhow::anyhow!("tensor {name} not found"))?;
    let cols = info.shape[0]; // ne[0] = innermost
    let rows: usize = info.shape[1..].iter().product();
    let raw = parent
        .tensor_slice(name)
        .ok_or_else(|| anyhow::anyhow!("tensor {name} has no bytes"))?;
    let container = match info.ggml_type {
        GgmlType::Q2_0 => {
            let n = rows * cols;
            let mut bytes = Vec::with_capacity(n / 34 * 34);
            bytes.extend_from_slice(raw);
            let blocks: Vec<riir_infer_core::quant::q2_0::BlockQ2_0> =
                bytemuck::cast_slice(&bytes).to_vec();
            repack_q2_0_to_ternary_group(&blocks, rows, cols)?
        }
        GgmlType::PTQ1_0 => {
            let n = rows * cols;
            let mut bytes = Vec::with_capacity(n / 128 * 28);
            bytes.extend_from_slice(raw);
            let blocks: Vec<riir_infer_core::quant::ptq1_0::BlockPtq1_0> =
                bytemuck::cast_slice(&bytes).to_vec();
            repack_ptq1_0_to_ternary_group(&blocks, rows, cols)?
        }
        other => anyhow::bail!("tensor {name} is {other:?} — the probe reads ternary projections"),
    };
    let dense =
        riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights::dequant_proj_to_dense(
            &container,
        );
    Ok((dense, rows, cols))
}

fn main() {
    let mut gguf = PathBuf::from("../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf");
    let mut blocks_spec = String::new();
    let mut suffix = String::from("ffn_down");
    let mut n_x = 8usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf = PathBuf::from(args.next().expect("--gguf needs a path")),
            "--blocks" => blocks_spec = args.next().expect("--blocks needs a:b,c:d"),
            "--suffix" => suffix = args.next().expect("--suffix needs a substring"),
            "--n-x" => n_x = args.next().expect("--n-x needs N").parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }
    if blocks_spec.is_empty() {
        panic!("--blocks a:b,c:d is required (half-open member triples from the profile artifact)");
    }

    let parent = GgufFile::open(&gguf).unwrap_or_else(|e| panic!("open {}: {e}", gguf.display()));
    let arch = parent
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_owned();
    eprintln!(
        "[twt-probe] parent {} arch {arch}, {} tensors, κ = {KAPPA_BUDGET}",
        gguf.display(),
        parent.tensor_infos.len()
    );

    // Deterministic synthetic inputs (the probe measures the OPERATOR, not
    // a corpus; the LCG stream is part of the artifact's reproducibility).

    println!("# twt_ternarize_probe — operator-level materialization damage");
    println!("# parent: {}", gguf.display());
    println!("# κ (pre-registered budget multiple) = {KAPPA_BUDGET}");
    println!("# columns: block, tensor, rows×cols, damage_A(f16), damage_B(majority), damage_C(source_quant), ratio_C/A");
    println!("block\ttensor\tshape\tdamage_A\tdamage_B\tdamage_C\tratio_C_A");

    for part in blocks_spec.split(',') {
        let (a, b) = part
            .split_once(':')
            .unwrap_or_else(|| panic!("--blocks entries are a:b, got {part}"));
        let start: usize = a.parse().unwrap();
        let end: usize = b.parse().unwrap();
        let members: Vec<usize> = (start..end).collect();
        assert!(members.len() >= 2, "block [{start},{end}) has < 2 members — nothing to merge");

        // One projection per member (the first tensor matching the suffix).
        let mut member_dense: Vec<Vec<f32>> = Vec::new();
        let mut shape = (0usize, 0usize);
        let mut used_name = String::new();
        for &li in &members {
            let name = parent
                .tensor_infos
                .iter()
                .find(|i| {
                    i.name.starts_with(&format!("blk.{li}."))
                        && i.name.contains(&suffix)
                        && matches!(i.ggml_type, GgmlType::Q2_0 | GgmlType::PTQ1_0)
                })
                .map(|i| i.name.clone())
                .unwrap_or_else(|| panic!("no ternary tensor matching '{suffix}' in blk.{li}.*"));
            let (dense, rows, cols) = tensor_to_dense(&parent, &name).unwrap();
            if member_dense.is_empty() {
                shape = (rows, cols);
                used_name = name;
            } else {
                assert_eq!((rows, cols), shape, "{name} shape drift inside block [{start},{end})");
            }
            member_dense.push(dense);
        }
        let (rows, cols) = shape;
        let refs: Vec<&[f32]> = member_dense.iter().map(|v| v.as_slice()).collect();
        let fbar = riir_infer_core::twt::audition::merge_mean(refs.iter().copied())
            .unwrap_or_else(|e| panic!("merge_mean: {e}"));

        // Deterministic calibration inputs.
        let mut seed = 0xA80BE_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as u32) as f32 / u32::MAX as f32 - 0.5
        };
        let xs: Vec<f32> = (0..n_x * cols).map(|_| next()).collect();
        let mut ya = vec![0f32; rows];
        let mut yb = vec![0f32; rows];

        let arm_a = arm_dense_f16(&refs, rows, cols).expect("arm A");
        let arm_b = arm_sign_majority(&refs, rows, cols).expect("arm B");
        let arm_c = arm_source_quant(&fbar, rows, cols).expect("arm C");

        let d_a = materialization_rel_err(
            &arm_a.to_dense_f32(),
            &fbar,
            &xs,
            rows,
            cols,
            n_x,
            &mut ya,
            &mut yb,
        )
        .expect("arm A damage");
        let d_b = materialization_rel_err(
            &arm_b.to_dense_f32(),
            &fbar,
            &xs,
            rows,
            cols,
            n_x,
            &mut ya,
            &mut yb,
        )
        .expect("arm B damage");
        let d_c = materialization_rel_err(
            &arm_c.to_dense_f32(),
            &fbar,
            &xs,
            rows,
            cols,
            n_x,
            &mut ya,
            &mut yb,
        )
        .expect("arm C damage");
        let ratio = budget_ratio(d_c, d_a);

        println!(
            "[{start},{end})\t{used_name}\t{rows}×{cols}\t{d_a:.3e}\t{d_b:.3e}\t{d_c:.3e}\t{ratio:.1}"
        );
    }
    println!("# reading: damage is the arm's OWN price vs the f32 f̄; the T4.2 κ budget reads");
    println!("# END-TO-END errors (audition), so arm C survives a block only where the surrogate's");
    println!("# own error exceeds roughly damage_C/(κ−1). ratio_C_A > κ−1 ⇒ arm C needs a strong");
    println!("# surrogate on that block; ratio_C_A ≤ κ−1 ⇒ arm C is budget-viable even alone.");
}
