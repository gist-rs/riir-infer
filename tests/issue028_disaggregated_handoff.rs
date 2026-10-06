#![cfg(feature = "deltanet_ternary_inference")]
//! Issue 028 T1/T2 — the disaggregated-container + phase-handoff battery.
//!
//! The container quantizes one checkpoint twice (compact decode copy +
//! higher-precision prefill copy); the handoff crosses the prefill→decode
//! boundary in-process over ONE `HybridCache`. These gates:
//!
//! 1. **G1 byte-identity (the plumbing gate):** a same-weights container
//!    (`from_single`) phase-split must be BYTE-IDENTICAL to the
//!    single-checkpoint continuous loop — every logit, by bits, at the
//!    boundary and across decode steps. The consumer-facing
//!    `generate_greedy_disaggregated` must likewise produce the exact token
//!    sequence of `generate_greedy_qwen_deltanet_ternary`.
//! 2. **Boundary-freshness negative control:** resetting the cache at the
//!    boundary MUST diverge — proving the G1 gate can fail (a handoff that
//!    silently dropped state would otherwise read as a pass).
//! 3. **One-file `.pf` container:** the suffixed set loads through the same
//!    loader body; the suffix law is exercised end to end; a plain
//!    single-checkpoint file refuses in `load_single_file` (it would
//!    silently serve the decode copy in both phases while claiming
//!    disaggregation).
//! 4. **Escape-set law (T2):** `ssm_a` / `ssm_dt.bias` / `ssm_alpha` /
//!    `ssm_beta` must be bit-shared between the copies — a diverged one
//!    refuses loudly.
//! 5. **Compat gate:** two files whose geometry metadata differs refuse
//!    (the blake3-over-Debug fingerprint covers every Config field).
//! 6. **Real-artifact arm** (`#[ignore]`, `bonsai2_hadamard`): the 6.7 GB
//!    `Ternary-Bonsai-2-27B-PQ2_0.gguf` pack through the same phase split —
//!    the same-weights byte-identity on the real 27B contract.
//!
//! Fixtures ride the shared synthetic-Bonsai-2 kit
//! (`common/synth_bonsai2.rs`, extracted from `bonsai2_rotation_load.rs`) —
//! one writer, both batteries.

#[path = "common/synth_bonsai2.rs"]
mod synth;

use synth::*;

use riir_infer_core::deltanet::ternary_forward::{
    forward_qwen_deltanet_ternary, generate_greedy_qwen_deltanet_ternary,
};
use riir_infer_core::disaggregated::{
    DisaggregatedTernaryWeights, PREFILL_TENSOR_SUFFIX, PhaseHandoff, generate_greedy_disaggregated,
};
use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_core::types::{Config, DeltaNetLayerType};

/// The synthetic Bonsai-2 contract (unfolded — rotation `None`), one file.
fn synth_single_bytes() -> Vec<u8> {
    build_gguf(&synth_metadata(false), &synth_tensors_bonsai2())
}

/// The one-file disaggregated container: the bare decode set + a byte-equal
/// `.pf`-suffixed prefill set (the same-weights G1 arm).
fn synth_disaggregated_bytes() -> Vec<u8> {
    let mut tensors = synth_tensors_bonsai2();
    let pf: Vec<TensorSpec> = tensors
        .iter()
        .map(|t| TensorSpec {
            name: format!("{}{PREFILL_TENSOR_SUFFIX}", t.name),
            ne: t.ne.clone(),
            ggml_type: t.ggml_type,
            data: t.data.clone(),
        })
        .collect();
    tensors.extend(pf);
    build_gguf(&synth_metadata(false), &tensors)
}

fn load_single(
    path: &std::path::Path,
) -> (
    Config,
    riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights,
) {
    load_qwen_deltanet_ternary_weights_gguf(path).expect("load synthetic single file")
}

/// Same comparator as the production generate loop (the argmax consumer of
/// the byte-identity gate — ties must resolve identically).
fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
        .map_or(0, |(i, _)| i)
}

const PROMPT: [usize; 5] = [7, 13, 42, 100, 5];
const DECODE_STEPS: usize = 4;

/// Arm A — the single-checkpoint continuous loop, capturing the LAST prefill
/// logits + every decode step's logits, bit-exact.
fn run_single_checkpoint(
    config: &Config,
    weights: &riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights,
) -> (Vec<f32>, Vec<Vec<f32>>, Vec<usize>) {
    let layer_types = weights.layer_types.clone();
    let mut cache = riir_infer_core::deltanet::HybridCache::with_layer_types(config, &layer_types);
    let mut scratch = riir_infer_core::deltanet::HybridForwardScratch::new(config);
    let rope = riir_infer_core::rope::RopeFreqTable::new(
        config.rope_theta,
        riir_infer_core::deltanet::effective_rotary_dim(config),
    );
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];

    let mut prefill_last = Vec::new();
    for (pos, tok) in PROMPT.iter().enumerate() {
        let l = forward_qwen_deltanet_ternary(
            &mut x,
            weights,
            &mut cache,
            *tok,
            pos,
            config,
            &mut scratch,
            &rope,
        );
        prefill_last = l[..config.vocab_size].to_vec();
    }

    let mut decode_logits = Vec::new();
    let mut tokens = Vec::new();
    let mut next = argmax(&prefill_last);
    for i in 0..DECODE_STEPS {
        let pos = PROMPT.len() + i;
        let l = forward_qwen_deltanet_ternary(
            &mut x,
            weights,
            &mut cache,
            next,
            pos,
            config,
            &mut scratch,
            &rope,
        );
        let l = l[..config.vocab_size].to_vec();
        next = argmax(&l);
        tokens.push(next);
        decode_logits.push(l);
    }
    (prefill_last, decode_logits, tokens)
}

/// Arm B — the phase split through a `PhaseHandoff`, same capture shape.
fn run_phase_split(
    config: &Config,
    container: &DisaggregatedTernaryWeights,
    layer_types: &[DeltaNetLayerType],
) -> (Vec<f32>, Vec<Vec<f32>>, Vec<usize>) {
    let mut handoff = PhaseHandoff::begin(config, layer_types);
    let prefill_last = handoff
        .prefill(container.prefill(), config, &PROMPT)
        .to_vec();

    let mut decode_logits = Vec::new();
    let mut tokens = Vec::new();
    let mut next = argmax(&prefill_last);
    for _ in 0..DECODE_STEPS {
        let l = handoff
            .decode_step(container.decode(), config, next)
            .to_vec();
        next = argmax(&l);
        tokens.push(next);
        decode_logits.push(l);
    }
    (prefill_last, decode_logits, tokens)
}

fn assert_bits_equal(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length diverged");
    for (i, (u, v)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            u.to_bits(),
            v.to_bits(),
            "{label}: logit {i} diverged ({} vs {})",
            u,
            v
        );
    }
}

/// G1 — the same-weights phase split is byte-identical to the
/// single-checkpoint loop: last prefill logits, every decode step's logits,
/// and the derived token sequence.
#[test]
fn same_weights_phase_split_is_byte_identical_to_single_checkpoint() {
    let path = write_tmp("riir_028_single", &synth_single_bytes());
    let (config, weights) = load_single(&path);
    let _ = std::fs::remove_file(&path);
    let layer_types = weights.layer_types.clone();

    // Arm A borrows the weights; the container then MOVES them (one set,
    // both phases — the from_single semantics).
    let (a_prefill, a_decode, a_tokens) = run_single_checkpoint(&config, &weights);
    let container = DisaggregatedTernaryWeights::from_single(weights);
    let (b_prefill, b_decode, b_tokens) = run_phase_split(&config, &container, &layer_types);

    assert_bits_equal("prefill-boundary logits", &a_prefill, &b_prefill);
    assert_eq!(a_decode.len(), b_decode.len());
    for (step, (a, b)) in a_decode.iter().zip(&b_decode).enumerate() {
        assert_bits_equal(&format!("decode step {step}"), a, b);
    }
    assert_eq!(a_tokens, b_tokens, "greedy tokens must match exactly");
}

/// The consumer-facing driver over `from_single` produces the exact token
/// sequence of the production single-checkpoint generate loop.
#[test]
fn generate_disaggregated_from_single_matches_production_generate() {
    let path = write_tmp("riir_028_gen", &synth_single_bytes());
    let (config, weights) = load_single(&path);
    let _ = std::fs::remove_file(&path);

    let expected = generate_greedy_qwen_deltanet_ternary(&weights, &config, &PROMPT, 8);
    let container = DisaggregatedTernaryWeights::from_single(weights);
    let got = generate_greedy_disaggregated(&container, &config, &PROMPT, 8);
    assert_eq!(expected, got, "token sequences must match exactly");
}

/// Negative control — a handoff that LOSES its state at the boundary must
/// diverge. Resetting the cache after prefill drops the GDN recurrent
/// matrices + the attention KV; if this arm still matched the continuous
/// loop, the G1 gate would be blind to exactly the class it exists for.
#[test]
fn fresh_cache_at_boundary_diverges() {
    let path = write_tmp("riir_028_neg", &synth_single_bytes());
    let (config, weights) = load_single(&path);
    let _ = std::fs::remove_file(&path);
    let layer_types = weights.layer_types.clone();

    let (a_prefill, a_decode, _) = run_single_checkpoint(&config, &weights);
    let container = DisaggregatedTernaryWeights::from_single(weights);

    let mut handoff = PhaseHandoff::begin(&config, &layer_types);
    handoff.prefill(container.prefill(), &config, &PROMPT);
    handoff.cache().reset(); // the defect under test

    // Same boundary token arm A consumed — the ONLY difference is the
    // dropped state.
    let boundary_token = argmax(&a_prefill);
    let first = handoff
        .decode_step(container.decode(), &config, boundary_token)
        .to_vec();
    let diverged = first
        .iter()
        .zip(&a_decode[0])
        .any(|(u, v)| u.to_bits() != v.to_bits());
    assert!(
        diverged,
        "a cache reset at the boundary MUST diverge — the G1 gate would be blind otherwise"
    );
}

/// The one-file `.pf` container: both sets load through the same loader
/// body, the compat fingerprints agree, and the phase split stays
/// byte-identical to the single-checkpoint loop.
#[test]
fn one_file_pf_container_loads_and_runs_byte_identical() {
    let single_path = write_tmp("riir_028_of_single", &synth_single_bytes());
    let (config, weights) = load_single(&single_path);
    let _ = std::fs::remove_file(&single_path);

    let disagg_path = write_tmp("riir_028_of_pair", &synth_disaggregated_bytes());
    let (disagg_config, container) = DisaggregatedTernaryWeights::load_single_file(&disagg_path)
        .expect("one-file container loads");
    let _ = std::fs::remove_file(&disagg_path);

    // Same metadata → same geometry config (modulo the loader's independent
    // derivations, which read identical keys).
    assert_eq!(config.n_layer, disagg_config.n_layer);
    assert_eq!(config.n_embd, disagg_config.n_embd);
    assert_eq!(config.vocab_size, disagg_config.vocab_size);

    let layer_types = weights.layer_types.clone();
    let (a_prefill, a_decode, _) = run_single_checkpoint(&config, &weights);
    let (b_prefill, b_decode, _) = run_phase_split(&config, &container, &layer_types);
    assert_bits_equal("one-file prefill-boundary logits", &a_prefill, &b_prefill);
    for (step, (a, b)) in a_decode.iter().zip(&b_decode).enumerate() {
        assert_bits_equal(&format!("one-file decode step {step}"), a, b);
    }
}

/// A plain single-checkpoint file (no `.pf` set) refuses in
/// `load_single_file` — silently serving the decode copy in both phases
/// while claiming disaggregation is exactly the mis-load this gate prevents.
#[test]
fn single_checkpoint_file_refuses_in_load_single_file() {
    let path = write_tmp("riir_028_refuse", &synth_single_bytes());
    let err = match DisaggregatedTernaryWeights::load_single_file(&path) {
        Ok(_) => panic!("expected refusal for a file with no .pf set"),
        Err(e) => e,
    };
    let _ = std::fs::remove_file(&path);
    let msg = err.to_string();
    assert!(
        msg.contains(".pf"),
        "refusal must name the missing suffix set, got: {msg}"
    );
}

/// The T2 escape law: a prefill copy whose `ssm_a` diverges from the decode
/// copy refuses loudly, naming the tensor.
#[test]
fn escape_set_divergence_refuses() {
    let decode_path = write_tmp("riir_028_esc_dec", &synth_single_bytes());

    let mut prefill_tensors = synth_tensors_bonsai2();
    for spec in prefill_tensors.iter_mut() {
        if spec.name == "blk.0.ssm_a" {
            let vals = vec![2.5f32; spec.data.len() / 4];
            spec.data = f32_payload(&vals);
        }
    }
    let prefill_path = write_tmp(
        "riir_028_esc_pf",
        &build_gguf(&synth_metadata(false), &prefill_tensors),
    );

    let err = match DisaggregatedTernaryWeights::load_pair(&decode_path, &prefill_path) {
        Ok(_) => panic!("expected escape-set refusal"),
        Err(e) => e,
    };
    let _ = std::fs::remove_file(&decode_path);
    let _ = std::fs::remove_file(&prefill_path);
    let msg = err.to_string();
    assert!(
        msg.contains("ssm_a"),
        "refusal must name the diverged escape tensor, got: {msg}"
    );
}

/// The compat gate: two files whose geometry metadata differs (rope base)
/// refuse on the fingerprint — a bonsai-27B decode copy can never pair with
/// a mismatched-geometry prefill copy.
#[test]
fn pair_geometry_mismatch_refuses() {
    let decode_path = write_tmp("riir_028_geo_dec", &synth_single_bytes());

    let mut meta = synth_metadata(false);
    for (k, v) in meta.iter_mut() {
        if k == "qwen35.rope.freq_base" {
            *v = Val::F64(20000.0);
        }
    }
    let prefill_path = write_tmp(
        "riir_028_geo_pf",
        &build_gguf(&meta, &synth_tensors_bonsai2()),
    );

    let err = match DisaggregatedTernaryWeights::load_pair(&decode_path, &prefill_path) {
        Ok(_) => panic!("expected compat refusal"),
        Err(e) => e,
    };
    let _ = std::fs::remove_file(&decode_path);
    let _ = std::fs::remove_file(&prefill_path);
    let msg = err.to_string();
    assert!(
        msg.contains("fingerprints differ"),
        "refusal must name the fingerprint mismatch, got: {msg}"
    );
}

/// Identical-content two-file pairs LOAD (the compat gate's positive arm):
/// same tensors, same metadata, two files → a container whose phases both
/// run byte-identically to the single-checkpoint loop.
#[test]
fn identical_two_file_pair_loads_and_runs_byte_identical() {
    let bytes = synth_single_bytes();
    let decode_path = write_tmp("riir_028_pair_dec", &bytes);
    let prefill_path = write_tmp("riir_028_pair_pf", &bytes);
    let (config, container) = DisaggregatedTernaryWeights::load_pair(&decode_path, &prefill_path)
        .expect("identical pair loads");
    let _ = std::fs::remove_file(&decode_path);
    let _ = std::fs::remove_file(&prefill_path);

    let (_, weights) = {
        let p = write_tmp("riir_028_pair_ref", &synth_single_bytes());
        let out = load_single(&p);
        let _ = std::fs::remove_file(&p);
        out
    };
    let layer_types = weights.layer_types.clone();

    let (a_prefill, a_decode, _) = run_single_checkpoint(&config, &weights);
    let (b_prefill, b_decode, _) = run_phase_split(&config, &container, &layer_types);
    assert_bits_equal("two-file prefill-boundary logits", &a_prefill, &b_prefill);
    for (step, (a, b)) in a_decode.iter().zip(&b_decode).enumerate() {
        assert_bits_equal(&format!("two-file decode step {step}"), a, b);
    }
}

/// The real 27B pack through the phase split (same-weights byte-identity on
/// the production contract). `bonsai2_hadamard` required: the default path
/// is the folded PQ2_0 pack. Run:
/// `cargo test --release --features bonsai2_hadamard --test issue028_disaggregated_handoff real_bonsai -- --ignored --nocapture`
#[test]
#[cfg(feature = "bonsai2_hadamard")]
#[ignore = "loads the real 6.7 GB PQ2_0 pack — run manually (see the doc comment)"]
fn real_bonsai_phase_split_is_byte_identical() {
    let path = std::env::var("BONSAI_PQ2_0_GGUF")
        .unwrap_or_else(|_| "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".to_string());
    let path = std::path::PathBuf::from(path);
    if !path.exists() {
        eprintln!("[skip] checkpoint not found: {}", path.display());
        return;
    }

    let (config, weights) = load_single(&path);
    let layer_types = weights.layer_types.clone();

    let t0 = std::time::Instant::now();
    let (a_prefill, a_decode, a_tokens) = run_single_checkpoint(&config, &weights);
    let container = DisaggregatedTernaryWeights::from_single(weights);
    let (b_prefill, b_decode, b_tokens) = run_phase_split(&config, &container, &layer_types);
    eprintln!(
        "[real-bonsai] single+phase arms: {:?} (prompt {PROMPT:?}, {} decode steps)",
        t0.elapsed(),
        DECODE_STEPS
    );

    assert_bits_equal(
        "real-bonsai prefill-boundary logits",
        &a_prefill,
        &b_prefill,
    );
    for (step, (a, b)) in a_decode.iter().zip(&b_decode).enumerate() {
        assert_bits_equal(&format!("real-bonsai decode step {step}"), a, b);
    }
    assert_eq!(a_tokens, b_tokens);
}

// ── Issue 028 T4 S2 — the dual-format pair (q4-class prefill pack) ─────────

use riir_infer_core::deltanet::ternary_weights::{GateProjWeights, ProjWeights};

/// The dual-format container LOADS and RUNS: decode = `PQ2_0` ternary
/// (id 142), prefill = the same per-layer projections re-quantized to
/// `Q4_K` (type 12, only ε(Q4_K) on the identical ternary content), with
/// globals + a/b + dense fields byte-shared — the S1 pack's exact copy
/// policy at synthetic dims.
///
/// Gates:
///
/// 1. the prefill copy carries `ProjWeights::Q4K` arms on its projections
///    while the decode copy stays all-ternary (a/b stay `GateProjWeights`);
/// 2. the escape-set law passes at load (byte-identical a/b, BF16 on both
///    sides — `verify_escape_set_shared` runs inside `load_pair`);
/// 3. the phase split RUNS through the q4 arms (bitlinear's Q4K path end to
///    end) and the prefill-boundary logits track the all-ternary reference
///    within a loose relative-L2 envelope (the formats differ — NOT a bit
///    identity; the T2 byte-compare law stays on the same-format arms
///    above);
/// 4. decode-phase logits are finite.
#[test]
fn dual_format_q4_prefill_pair_loads_and_runs() {
    let decode_bytes = build_gguf(&synth_metadata(false), &synth_tensors_bonsai2_typed(142));
    let prefill_bytes = build_gguf(&synth_metadata(false), &synth_tensors_bonsai2_q4_prefill());
    let dpath = write_tmp("riir_028_q4_decode", &decode_bytes);
    let ppath = write_tmp("riir_028_q4_prefill", &prefill_bytes);

    // The all-ternary reference (decode copy through the single loop).
    let (config, ref_weights) = load_single(&dpath);
    let (ref_last, _ref_decode, _ref_tokens) = run_single_checkpoint(&config, &ref_weights);
    drop(ref_weights);

    let (config, container) = DisaggregatedTernaryWeights::load_pair(&dpath, &ppath)
        .expect("load the dual-format pair");
    let _ = std::fs::remove_file(&dpath);
    let _ = std::fs::remove_file(&ppath);

    // 1. Arm layout: q4 projections on the prefill side, ternary on decode;
    //    the a/b escape fields stay GateProjWeights (Dense BF16 arm here).
    let pre = container.prefill();
    let dec = container.decode();
    assert!(matches!(
        pre.layers[0].in_proj_qkv,
        ProjWeights::Q4K(..)
    ));
    assert!(matches!(pre.layers[3].attn_wq, ProjWeights::Q4K(..)));
    assert!(matches!(
        dec.layers[0].in_proj_qkv,
        ProjWeights::Ternary(_)
    ));
    assert!(matches!(
        pre.layers[0].in_proj_a,
        GateProjWeights::Dense(..)
    ));
    assert!(dec.layers[0].in_proj_qkv.rows() > 0);
    assert!(pre.layers[0].in_proj_qkv.rows() > 0);

    // 2/3. The phase split through the q4 prefill arms, then the ternary
    //      decode copy — logits track the all-ternary reference.
    let layer_types = dec.layer_types.clone();
    let (pre_last, decode_logits, tokens) = run_phase_split(&config, &container, &layer_types);

    let num: f32 = ref_last
        .iter()
        .zip(&pre_last)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let den: f32 = ref_last.iter().map(|a| a * a).sum::<f32>().max(1e-30);
    let rel_l2 = (num / den).sqrt();
    assert!(
        rel_l2 < 0.05,
        "q4-prefill boundary logits diverge from the ternary reference: relative L2 {rel_l2}"
    );
    eprintln!("[q4-pair] boundary rel L2 vs ternary reference: {rel_l2:.6}");

    // 4. Decode-phase sanity (the decode copy is ternary — its logits are
    //    finite and the greedy loop produced DECODE_STEPS tokens).
    assert_eq!(tokens.len(), DECODE_STEPS);
    for (step, l) in decode_logits.iter().enumerate() {
        assert!(
            l.iter().all(|v| v.is_finite()),
            "decode step {step}: non-finite logit"
        );
    }
}

/// The real dual-format pair — the S1 `Q4_K.pf` pack against the 6.7 GB
/// PQ2_0 decode pack (20.5 GB resident total; the T4 budget row). Proves the
/// LOAD path at production scale (the geometry fingerprint + escape law on
/// the real files) and the q4 host matvec on real shapes. Run:
/// `cargo test --release --features bonsai2_hadamard --test issue028_disaggregated_handoff real_q4 -- --ignored --nocapture`
#[test]
#[cfg(feature = "bonsai2_hadamard")]
#[ignore = "loads the real 20.5 GB pair — run manually (see the doc comment)"]
fn real_q4_prefill_pair_loads_and_prefills() {
    let decode_path = std::env::var("BONSAI_PQ2_0_GGUF")
        .unwrap_or_else(|_| "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".to_string());
    let prefill_path = std::env::var("BONSAI_Q4_PF_GGUF")
        .unwrap_or_else(|_| "../riir-train/data/Ternary-Bonsai-2-27B-Q4_K.pf.gguf".to_string());
    let (decode_path, prefill_path) =
        (std::path::PathBuf::from(decode_path), std::path::PathBuf::from(prefill_path));
    if !decode_path.exists() || !prefill_path.exists() {
        eprintln!(
            "[skip] pair not found: {} / {}",
            decode_path.display(),
            prefill_path.display()
        );
        return;
    }

    let t0 = std::time::Instant::now();
    let (config, container) = DisaggregatedTernaryWeights::load_pair(&decode_path, &prefill_path)
        .expect("load the real dual-format pair");
    eprintln!("[real-q4] load_pair: {:?}", t0.elapsed());

    // Arm layout at real shapes: layer 0 is a DeltaNet layer (attention
    // every 4th — full_attention_interval 4), layer 3 the first attention
    // layer. Their projections carry the q4 arm; the a/b escape fields load
    // from the byte-copied BF16 — the escape check already passed inside
    // load_pair.
    let pre = container.prefill();
    assert!(matches!(pre.layers[0].in_proj_qkv, ProjWeights::Q4K(..)));
    assert!(matches!(pre.layers[3].attn_wq, ProjWeights::Q4K(..)));
    let layer_types = container.decode().layer_types.clone();

    // A SHORT prefill only — the q4 host matvec is the S2 functional arm,
    // not the TTFT deliverable (that is the GPU lane / S3's bench).
    let prompt: &[usize] = &PROMPT;
    let mut handoff = PhaseHandoff::begin(&config, &layer_types);
    let t1 = std::time::Instant::now();
    let last = handoff.prefill(container.prefill(), &config, prompt).to_vec();
    eprintln!(
        "[real-q4] prefill {} tokens: {:?} ({:.3} s/token)",
        prompt.len(),
        t1.elapsed(),
        t1.elapsed().as_secs_f32() / prompt.len() as f32
    );
    assert!(last.iter().all(|v| v.is_finite()), "non-finite prefill logits");
}
