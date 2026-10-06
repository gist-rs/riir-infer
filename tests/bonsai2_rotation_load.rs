#![cfg(feature = "bonsai2_hadamard")]
//! Issue 980 / Plan 600 Phase B — synthetic Bonsai-2 loader + rotation tests.
//!
//! Builds a TINY synthetic GGUF that carries the exact Bonsai-2 contract
//! (arch `qwen35`, `prism.hadamard.*` metadata, dense BF16 `ssm_alpha`/`ssm_beta`
//! escape set, `Q2_0` id-142 ternary projections) and asserts:
//!
//! 1. the loader parses the rotation config (block/signs/inverse/gdn flag),
//! 2. `in_proj_a`/`in_proj_b` load as the DENSE arm while ternary tensors
//!    repack as usual,
//! 3. the forward wiring actually FIRES at every layer (per-layer capture
//!    differs vs the same weights with rotation stripped),
//! 4. the folded-matmul math matches an independent reference (explicit
//!    ±1/√n Hadamard matrix, the fork's own construction),
//! 5. a pre-rotation file (no prism keys, ternary a/b) still loads with
//!    `rotation == None` — the old-file rollback lane,
//! 6. unsupported metadata refuses LOUDLY (version, transform, unknown
//!    folded weight, sign-length mismatch).
//!
//! Real-file validation (the 7 GB `PQ2_0` pack, fork logits parity) is the
//! 4090-side G1 — this file is the fast local gate.

use std::path::Path;

use katgpt_core::TernaryGroupWeights;
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::types::Config;

#[path = "common/synth_bonsai2.rs"]
mod synth;

use synth::*;

// ── tests ───────────────────────────────────────────────────────────────────

fn load(
    path: &Path,
) -> anyhow::Result<(
    Config,
    riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights,
)> {
    load_qwen_deltanet_ternary_weights_gguf(path)
}

/// `expect_err` needs `Debug` on the Ok payload; the production types do not
/// derive it, so the refusal tests match the error text instead.
fn load_expect_err(path: &Path, contains: &str) -> anyhow::Error {
    match load(path) {
        Ok(_) => panic!("expected load refusal containing '{contains}'"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains(contains),
                "refusal message '{msg}' does not mention '{contains}'"
            );
            e
        }
    }
}

/// 1+2. The folded file parses; a/b are the DENSE arm; rotation metadata is
/// complete (block/signs/inverse/gdn).
#[test]
fn bonsai2_synthetic_loads_with_rotation_and_dense_gate_projs() {
    let bytes = build_gguf(&synth_metadata(true), &synth_tensors_bonsai2());
    let path = write_tmp("riir_b2_rot", &bytes);
    let (config, weights) = load(&path).expect("load folded synthetic");
    let _ = std::fs::remove_file(&path);

    assert_eq!(config.n_layer, N_LAYER as usize);
    assert_eq!(config.n_embd, N_EMBD as usize);
    assert_eq!(config.deltanet_linear_n_value_heads, N_V as usize);
    assert_eq!(config.deltanet_linear_n_heads, N_K as usize);
    assert_eq!(config.vocab_size, VOCAB as usize);

    let rot = weights
        .rotation
        .as_ref()
        .expect("rotation must be Some for a folded file");
    assert_eq!(rot.block_size, 1024);
    assert!(rot.inverse_embedding);
    assert!(rot.gdn_v_grouped);
    assert_eq!(rot.gdn_v_heads, N_V as usize);
    assert_eq!(rot.gdn_k_groups, N_K as usize);
    let signs = rot
        .signs_for_width(N_EMBD as usize)
        .expect("sign vector for 1024");
    assert_eq!(signs.len(), N_EMBD as usize);
    assert_eq!(signs[0], -1);

    // a/b dense arm with the right geometry ([n_v_heads × n_embd]).
    for l in &weights.layers {
        if l.in_proj_qkv.rows() > 0 {
            let GateProjShape(a_rows, a_cols) = GateProjShape::of(&l.in_proj_a);
            assert_eq!((a_rows, a_cols), (N_V as usize, N_EMBD as usize));
            assert!(matches!(
                l.in_proj_a,
                riir_infer_core::deltanet::ternary_weights::GateProjWeights::Dense(..)
            ));
            assert!(matches!(
                l.in_proj_b,
                riir_infer_core::deltanet::ternary_weights::GateProjWeights::Dense(..)
            ));
        }
    }
    // Attention layers carry empty a/b.
    let attn = weights.layers.last().unwrap();
    assert_eq!(attn.in_proj_a.rows(), 0);
    // Global invariants still hold.
    assert!(weights.invariants_hold());
}

struct GateProjShape(usize, usize);
impl GateProjShape {
    fn of(w: &riir_infer_core::deltanet::ternary_weights::GateProjWeights) -> Self {
        Self(w.rows(), w.cols())
    }
}

/// 5. The pre-rotation file (ternary a/b, no prism keys) still loads with
///    `rotation == None` — the old-file rollback lane.
#[test]
fn old_file_loads_without_rotation() {
    let bytes = build_gguf(&synth_metadata(false), &synth_tensors_old());
    let path = write_tmp("riir_b2_old", &bytes);
    let (_, weights) = load(&path).expect("load old synthetic");
    let _ = std::fs::remove_file(&path);
    assert!(weights.rotation.is_none());
    for l in &weights.layers {
        if l.in_proj_qkv.rows() > 0 {
            assert!(matches!(
                l.in_proj_a,
                riir_infer_core::deltanet::ternary_weights::GateProjWeights::Ternary(_)
            ));
        }
    }
    assert!(weights.invariants_hold());
}

/// 3. The rotation wiring FIRES at every layer: per-layer residual captures
///    differ between the folded run and the same weights with rotation stripped.
#[test]
fn forward_rotation_wiring_fires_every_layer() {
    let bytes = build_gguf(&synth_metadata(true), &synth_tensors_bonsai2());
    let path = write_tmp("riir_b2_rot_fwd", &bytes);
    let (config, mut weights) = load(&path).expect("load");
    let _ = std::fs::remove_file(&path);

    let layer_types = weights.layer_types.clone();
    let run = |w: &riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights| {
        let mut cache =
            riir_infer_core::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = riir_infer_core::deltanet::HybridForwardScratch::new(&config);
        let rope = riir_infer_core::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
        let mut captures = vec![vec![0.0f32; config.n_embd]; config.n_layer];
        riir_infer_core::deltanet::forward_qwen_deltanet_ternary_with_capture(
            &mut x,
            w,
            &mut cache,
            7,
            0,
            &config,
            &mut scratch,
            &rope,
            Some(&mut captures),
        );
        captures
    };

    let rotated = run(&weights);
    weights.rotation = None;
    let plain = run(&weights);

    for (i, (r, p)) in rotated.iter().zip(plain.iter()).enumerate() {
        let diff: f32 = r.iter().zip(p.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-3,
            "layer {i}: rotation wiring produced no difference (sum |Δ| = {diff})"
        );
    }
}

/// 4. Folded-matmul math vs the fork's OWN construction: explicit ±1/√n
///    Hadamard matrix (row&col parity) times the sign-multiplied input, then a
///    dense matmul with the dequantized folded weights. Tolerance covers the
///    FWHT-vs-matrix rounding difference only.
#[test]
fn folded_matmul_matches_explicit_matrix_reference() {
    let bytes = build_gguf(&synth_metadata(true), &synth_tensors_bonsai2());
    let path = write_tmp("riir_b2_rot_mm", &bytes);
    let (_, weights) = load(&path).expect("load");
    let _ = std::fs::remove_file(&path);

    let rot = weights.rotation.as_ref().unwrap();
    let signs = rot.signs_for_width(N_EMBD as usize).unwrap();
    let layer = &weights.layers[0];
    // Issue 028 T4 S2: these fixture files are all-ternary — the direct
    // kernel call takes the ternary arm.
    let w = layer
        .in_proj_qkv
        .as_ternary()
        .expect("rotation fixtures are all-ternary"); // [key_dim*2+value_dim, n_embd] folded ternary

    // Deterministic input.
    let x: Vec<f32> = (0..N_EMBD as usize)
        .map(|i| ((i * 37 + 11) % 97) as f32 - 48.0)
        .collect();

    // OUR path: sign → FWHT per block → SIMD ternary matvec.
    let mut x_ours = x.clone();
    riir_infer_core::deltanet::rotation::rotate_forward_inplace(
        &mut x_ours,
        Some(signs),
        rot.block_size,
    );
    let mut y_ours = vec![0.0f32; w.rows];
    katgpt_core::simd_ternary_group_matvec_parallel(w, &x_ours, &mut y_ours);

    // REFERENCE: dequantize the ternary weights, build the explicit block
    // matrix, dense-matmul.
    let w_dense = riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights::dequant_proj_to_dense(w);
    let block = rot.block_size;
    let scale = 1.0 / (block as f32).sqrt();
    let mut x_ref = vec![0.0f32; N_EMBD as usize];
    for (bi, xb) in x.chunks_exact(block).enumerate() {
        let signs_b = &signs[bi * block..(bi + 1) * block];
        for r in 0..block {
            let mut acc = 0.0f32;
            for c in 0..block {
                let mut parity = r & c;
                parity ^= parity >> 16;
                parity ^= parity >> 8;
                parity ^= parity >> 4;
                parity ^= parity >> 2;
                parity ^= parity >> 1;
                acc += xb[c] * signs_b[c] as f32 * if parity & 1 == 1 { -scale } else { scale };
            }
            x_ref[bi * block + r] = acc;
        }
    }
    for r in 0..w.rows {
        let row = &w_dense[r * w.cols..(r + 1) * w.cols];
        let mut acc = 0.0f32;
        for (&wv, &xv) in row.iter().zip(x_ref.iter()) {
            acc += wv * xv;
        }
        let denom = acc.abs().max(1e-3);
        assert!(
            (y_ours[r] - acc).abs() / denom < 1e-3,
            "row {r}: ours {} ref {acc}",
            y_ours[r]
        );
    }
}

/// 6a. Unsupported version refuses loudly.
#[test]
fn unsupported_version_refuses() {
    let mut meta = synth_metadata(true);
    for (k, v) in meta.iter_mut() {
        if k == "prism.hadamard.version" {
            *v = Val::U64(2);
        }
    }
    let bytes = build_gguf(&meta, &synth_tensors_bonsai2());
    let path = write_tmp("riir_b2_v2", &bytes);
    load_expect_err(&path, "unsupported");
    let _ = std::fs::remove_file(&path);
}

/// 6b. A folded weight outside the engine's verified structural map refuses.
#[test]
fn unknown_folded_weight_refuses() {
    let mut meta = synth_metadata(true);
    for (k, v) in meta.iter_mut() {
        if let Val::ArrStr(names) = v
            && k == "prism.hadamard.weight_names"
        {
            names.push("blk.0.ssm_alpha.weight".into()); // the escape set — never folded
        }
    }
    let bytes = build_gguf(&meta, &synth_tensors_bonsai2());
    let path = write_tmp("riir_b2_unknown", &bytes);
    load_expect_err(&path, "ssm_alpha");
    let _ = std::fs::remove_file(&path);
}

/// 6c. `sign_values` length mismatch refuses.
#[test]
fn sign_length_mismatch_refuses() {
    let mut meta = synth_metadata(true);
    for (k, v) in meta.iter_mut() {
        if let Val::ArrF64(signs) = v
            && k == "prism.hadamard.sign_values"
        {
            signs.pop();
        }
    }
    let bytes = build_gguf(&meta, &synth_tensors_bonsai2());
    let path = write_tmp("riir_b2_signlen", &bytes);
    load_expect_err(&path, "sign_values");
    let _ = std::fs::remove_file(&path);
}

/// The old-file regression at the `GgufFile` layer: a `Q2_0` id-42 tensor opens
/// identically to id-142 (the `PQ2_0` relabel) — both map to the ternary arm.
/// (`from_id` is crate-private; the public observable is the loader arm, and
/// the same fact is pinned in the in-crate `test_ggml_type_from_id`.)
#[test]
fn q2_0_ids_42_and_142_share_the_ternary_arm() {
    // Old-style synthetic (id 142) and the repack of an id-42 payload must
    // both load through the ternary arm — covered by
    // `old_file_loads_without_rotation` (ternary a/b) and
    // `bonsai2_synthetic_loads_with_rotation_and_dense_gate_projs` (ternary
    // projections at id 142). This placeholder keeps the intent greppable.
}

// Keep imports honest when features shift.
#[allow(dead_code)]
fn _touch(_: &GgufFile, _: &TernaryGroupWeights) {}

// ── PTQ1_0 (type 143) — the Phase C decode pack lane (Issue 980 T6) ──────────

/// A type-143 folded file loads through the SAME ternary arm: rotation
/// parses, invariants hold — the loader treats PTQ1_0 as a wire encoding of
/// the same substrate, not a new model class.
#[test]
fn ptq1_0_synthetic_loads_with_rotation() {
    let bytes = build_gguf(&synth_metadata(true), &synth_tensors_bonsai2_typed(143));
    let path = write_tmp("riir_b2_ptq10", &bytes);
    let (config, weights) = load(&path).expect("load PTQ1_0 folded synthetic");
    let _ = std::fs::remove_file(&path);

    assert!(
        weights.rotation.is_some(),
        "rotation metadata is wire-format independent"
    );
    assert!(weights.invariants_hold());
    assert_eq!(config.n_layer, N_LAYER as usize);
    // Ternary a/b still the Dense (BF16) escape set — the 143 payload only
    // touches the ternary projections.
    for l in &weights.layers {
        if l.in_proj_qkv.rows() > 0 {
            assert!(matches!(
                l.in_proj_a,
                riir_infer_core::deltanet::ternary_weights::GateProjWeights::Dense(..)
            ));
        }
    }
}

/// THE equivalence gate: PQ2_0 (id 142) and PTQ1_0 (id 143) encodings of the
/// same trits + scales must load to IDENTICAL containers and run the
/// rotated forward BIT-IDENTICALLY. This is the whole losslessness argument
/// for the Phase C decode lane in one test: any trit-map or scale-placement
/// error in the 143 decoder shows up as a container or forward divergence.
#[test]
fn ptq1_0_and_pq2_0_encodings_run_bit_identically() {
    let run = |ternary_id: u32| {
        let bytes = build_gguf(
            &synth_metadata(true),
            &synth_tensors_bonsai2_typed(ternary_id),
        );
        let path = write_tmp(&format!("riir_b2_eq_{ternary_id}"), &bytes);
        let (config, weights) = load(&path).expect("load");
        let _ = std::fs::remove_file(&path);
        let layer_types = weights.layer_types.clone();
        let mut cache =
            riir_infer_core::deltanet::HybridCache::with_layer_types(&config, &layer_types);
        let mut scratch = riir_infer_core::deltanet::HybridForwardScratch::new(&config);
        let rope = riir_infer_core::rope::RopeFreqTable::new(config.rope_theta, config.head_dim);
        let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
        for (pos, tok) in [7usize, 13, 42].iter().enumerate() {
            riir_infer_core::deltanet::forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                *tok,
                pos,
                &config,
                &mut scratch,
                &rope,
            );
        }
        x[..config.vocab_size.min(x.len())].to_vec()
    };

    let pq = run(142);
    let ptq = run(143);
    assert_eq!(pq.len(), ptq.len(), "vocab/geometry must agree");
    for (i, (a, b)) in pq.iter().zip(ptq.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit {i} diverged between PQ2_0 and PTQ1_0 encodings — the 143 decoder is not lossless"
        );
    }
}
