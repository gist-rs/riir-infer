//! Issue 022 T5.4 — the equal-FLOP skip-class arms (ShortGPT removal /
//! hydra-skip class) on the Bonsai ternary lane.
//!
//! The published zero-training removal class must be IN the gate or the
//! collapse lane cannot claim to beat it. This bin:
//!
//! 1. **Glue parity** — the amputation driver's own layer loop with the
//!    EMPTY skip set must reproduce the parent arm's cached argmax
//!    BYTE-IDENTICALLY (the loop glue consumes the shared
//!    [`qwen_deltanet_ternary_layer_body`]; a drift indicts the
//!    instrument, never the checkpoint — the audition's parity law).
//! 2. **ShortGPT selection** — Block Influence (adjacent-layer cosine
//!    over the frozen stream, the paper's pass-1 meter) + greedy
//!    elimination over a seeded next-token-loss scan (the paper's
//!    pass-2 shape at its own calibration granularity — the subset is
//!    pinned, never random).
//! 3. **Amputation agreement** — the T5.0 agreement harness (8 ×
//!    512-token chunks, cache resets, last position unscored) over the
//!    skip set, against the parent cache (params-keyed, LOUD replay).
//!
//! Arms (all at the SAME layer count — equal FLOPs by construction):
//! `--kept-twt <dropped,list>` (the TWT member-emit's dropped set — the
//! certified passing-point construction, replayed in-process) ·
//! `--skip-twt <collapsed.gguf>` (the same set read off the artifact's
//! `twt.block_table`) · `--arm-shortgpt` (ShortGPT: Block Influence +
//! greedy elimination at the TWT kept count) · `--arm-hydra` (the shipped
//! katgpt-rs `hydra_budget` criterion: logit-lens DE profiles transcribed
//! verbatim from `calibrate_profiles`, the |mean_de| ranking over non-
//! backup layers at the TWT kept count; the shipped threshold rule's
//! native skip set recorded as disclosure) · `--arm-random <seed>` (the
//! floor control).
//!
//! The parent cache self-generates when absent (one pass through the REAL
//! forward — never the amputation driver, so the glue-parity gate below
//! stays an independent-path check). The corpus may be a text file OR a
//! directory of HF `page_*.json` files (the `chat_probe` shape — the
//! deterministic, both-boxes-synced source).
//!
//! Run:
//! ```sh
//! cargo run --release --features twt_bonsai --bin twt_skip_amputation -- \
//!     --parent ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!     --corpus ../riir-train/data/chat_probe \
//!     --cache .raw/twt/parent_argmax_cache.json \
//!     --kept-twt 4,12,15 --arm-shortgpt --arm-hydra --arm-random 20261002 \
//!     --out .raw/twt/t54_skip_arms.json
//! ```

use anyhow::{bail, Context, Result};
use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::ternary_forward::{
    forward_qwen_deltanet_ternary_with_hook, qwen_deltanet_ternary_layer_body,
};
use riir_infer_core::deltanet::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::{GgufFile, GgufValue, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use std::io::Write;
use std::time::Instant;

/// Absolute agreement bar (the T5.0 budget — identical shape).
const AGREEMENT_BAR: f64 = 0.9;

const SEQ_LEN: usize = 512;
const MAX_TOKENS: usize = 4096;

struct Loaded {
    config: riir_infer_core::types::Config,
    weights: QwenDeltaNetTernaryWeights,
    rope_freq: RopeFreqTable,
}

fn load_parent(path: &std::path::Path) -> Result<Loaded> {
    let (config, weights) = load_qwen_deltanet_ternary_weights_gguf(path)
        .with_context(|| format!("load {}", path.display()))?;
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    Ok(Loaded { config, weights, rope_freq })
}

/// The amputation forward: the parent's own layer loop with `skip` layers
/// NOT executed (their cache slots untouched — the amputation semantics).
/// The glue mirrors [`forward_qwen_deltanet_ternary_with_hook`]'s body
/// AROUND the layer loop — embed lookup + inverse rotation, the layer
/// loop (the shared per-layer body), then final norm + the rotated LM
/// head — with the loop's skip branch being the only delta. The EMPTY
/// skip set is byte-identical to the real forward, which the glue-parity
/// gate asserts before any arm runs.
fn forward_amputated(
    x: &mut [f32],
    loaded: &Loaded,
    cache: &mut HybridCache,
    scratch: &mut HybridForwardScratch,
    token: usize,
    pos: usize,
    skip: &[bool],
) {
    let n = loaded.config.n_embd;
    let config = &loaded.config;
    let weights = &loaded.weights;
    let rotation = weights.rotation.as_ref();
    // 1. Embedding lookup + inverse rotation (verbatim hook-front glue).
    weights.dequant_wte_row_into(token, &mut x[..n]);
    if let Some(rot) = rotation
        && rot.inverse_embedding
    {
        let signs = rot.signs_for_width(n);
        riir_infer_core::deltanet::rotation::rotate_inverse_inplace(&mut x[..n], signs, rot.block_size);
    }
    // 2. The layer loop — skipped layers leave state AND stream untouched.
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        if skip.get(layer_idx).copied().unwrap_or(false) {
            continue;
        }
        let is_linear = weights.layer_types[layer_idx]
            == riir_infer_core::types::DeltaNetLayerType::DeltaNet;
        qwen_deltanet_ternary_layer_body(
            x,
            layer_weights,
            is_linear,
            &mut cache.deltanet_state.recurrent_states[layer_idx],
            &mut cache.deltanet_state.conv_states[layer_idx],
            &mut cache.kv_cache.layers[layer_idx],
            pos,
            config,
            scratch,
            &loaded.rope_freq,
            rotation,
            None,
            None,
            None,
        );
    }
    // 3. Final norm + the rotated LM head (verbatim hook-tail glue —
    // Issue 980: `output.weight` is folded; the head consumes the rotated
    // copy and logits come out in the primal basis).
    riir_infer_core::types::rmsnorm_with_gamma_eps(&mut x[..n], &weights.final_norm, loaded.config.rms_norm_eps);
    scratch.hidden_copy[..n].copy_from_slice(&x[..n]);
    if let Some(rot) = rotation {
        let signs = rot.signs_for_width(n);
        riir_infer_core::deltanet::rotation::rotate_forward_inplace(
            &mut scratch.hidden_copy[..n],
            signs,
            rot.block_size,
        );
    }
    // bitlinear's None-hook arm: the CPU SIMD matvec (`katgpt_core`'s
    // parallel kernel — the same call the forward's tail makes).
    katgpt_core::simd_ternary_group_matvec_parallel(
        &weights.lm_head,
        &scratch.hidden_copy[..n],
        &mut x[..loaded.config.vocab_size],
    );
}

fn argmax_of(x: &[f32], vocab: usize) -> usize {
    x[..vocab]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .expect("non-empty logits")
}

struct ArmAgreement {
    agree: usize,
    hit: usize,
    first_div: Option<usize>,
    secs: f32,
}

/// The T5.0 agreement harness over an in-process skip set (the loaded
/// config is temporarily set to the chunk window exactly as the
/// agreement bin does).
fn agreement_for_skip(
    loaded: &Loaded,
    chunks: &[Vec<usize>],
    parent_argmax: &[u32],
    skip: &[bool],
    label: &str,
) -> Result<ArmAgreement> {
    let mut cache = HybridCache::with_layer_types(&loaded.config, &loaded.weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&loaded.config);
    let mut x = vec![0.0f32; loaded.config.vocab_size.max(loaded.config.n_embd)];
    let mut argmax: Vec<u32> = Vec::new();
    let mut truth: Vec<u32> = Vec::new();
    let t0 = Instant::now();
    for chunk in chunks {
        cache.reset();
        for p in 0..chunk.len() - 1 {
            forward_amputated(&mut x, loaded, &mut cache, &mut scratch, chunk[p], p, skip);
            argmax.push(argmax_of(&x, loaded.config.vocab_size) as u32);
            truth.push(chunk[p + 1] as u32);
        }
    }
    let secs = t0.elapsed().as_secs_f32();
    let n = argmax.len();
    assert_eq!(n, parent_argmax.len(), "position counts diverged");
    let agree = argmax.iter().zip(parent_argmax.iter()).filter(|(a, b)| a == b).count();
    let hit = argmax.iter().zip(truth.iter()).filter(|(a, b)| a == b).count();
    let first_div = argmax.iter().zip(parent_argmax.iter()).position(|(a, b)| a != b);
    eprintln!(
        "[skip] {label}: {n} positions in {secs:.0}s ({:.2} tok/s) | agreement {:.4} ({agree}/{n}) | hit {:.4} | first div {first_div:?}",
        n as f32 / secs.max(1e-6),
        agree as f64 / n as f64,
        hit as f64 / n as f64,
    );
    Ok(ArmAgreement { agree, hit, first_div, secs })
}

/// The fused selection pass: ONE capture pass serving BOTH selection
/// criteria (the ShortGPT adjacent-layer cosine meter AND the hydra
/// logit-lens DE), so the two arms never pay two full sweeps. The parent
/// forward over the full chunk set with the post-layer capture hook;
/// per position: adjacent cosines accumulate on the raw residuals and the
/// lens DE row is recorded through the EXACT tail glue (RMSNorm →
/// rotation → head-row dot) against the parent's own top token.
struct SelectionPass {
    /// ShortGPT Block Influence: adjacent-layer cosine sums (lower = the
    /// layer changes its input less = more redundant).
    bi: Vec<f64>,
    /// Logit-lens DE matrix `[position][layer]` — the hydra criterion's
    /// raw input: `DE_l(pos) = ⟨W_U[top(pos)]_row, rot(RMSNorm(z^l))⟩`
    /// with top(pos) = the parent's own argmax at pos.
    lens_de: Vec<Vec<f32>>,
}

fn fused_selection_pass(loaded: &Loaded, chunks: &[Vec<usize>]) -> Result<SelectionPass> {
    let n_layers = loaded.config.n_layer;
    let n = loaded.config.n_embd;
    let config = &loaded.config;
    let weights = &loaded.weights;
    let rotation = weights.rotation.as_ref();
    let mut cache = HybridCache::with_layer_types(config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(config);
    let mut x = vec![0.0f32; config.vocab_size.max(n)];
    let mut capture = vec![vec![0.0f32; n]; n_layers];
    let mut cos_sum = vec![0.0f64; n_layers - 1];
    let mut lens_de: Vec<Vec<f32>> = Vec::new();
    let mut wrow = vec![0.0f32; n];
    let mut h = vec![0.0f32; n];
    let mut rows = 0usize;
    let mut lens_tail_logit_delta = 0.0f32;
    let t0 = Instant::now();
    for chunk in chunks {
        cache.reset();
        for (p, &tok) in chunk.iter().enumerate() {
            forward_qwen_deltanet_ternary_with_hook(
                &mut x,
                weights,
                &mut cache,
                tok,
                p,
                config,
                &mut scratch,
                &loaded.rope_freq,
                Some(&mut capture),
                None,
                None,
                None,
            );
            // ShortGPT BI: adjacent-layer cosine on the raw residuals.
            for l in 0..n_layers - 1 {
                let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
                let (a_row, b_row) = (&capture[l], &capture[l + 1]);
                for (a, b) in a_row.iter().zip(b_row.iter()) {
                    let (a, b) = (*a as f64, *b as f64);
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                if na > 0.0 && nb > 0.0 {
                    cos_sum[l] += dot / (na.sqrt() * nb.sqrt());
                }
            }
            // hydra lens DE: the parent's own top token, head-row dot per
            // layer through the exact tail glue (the last layer's dot must
            // land on the top logit up to summation order — disclosed once).
            let top = argmax_of(&x, config.vocab_size);
            weights.dequant_lm_head_row_into(top, &mut wrow);
            let mut row = Vec::with_capacity(n_layers);
            for cap_l in &capture {
                h.copy_from_slice(cap_l);
                riir_infer_core::types::rmsnorm_with_gamma_eps(
                    &mut h,
                    &weights.final_norm,
                    config.rms_norm_eps,
                );
                if let Some(rot) = rotation {
                    let signs = rot.signs_for_width(n);
                    riir_infer_core::deltanet::rotation::rotate_forward_inplace(
                        &mut h,
                        signs,
                        rot.block_size,
                    );
                }
                let de: f32 = wrow.iter().zip(h.iter()).map(|(w, v)| w * v).sum();
                row.push(de);
            }
            if rows == 0 {
                lens_tail_logit_delta = (row[n_layers - 1] - x[top]).abs();
            }
            lens_de.push(row);
            rows += 1;
        }
        eprintln!(
            "[skip] selection pass: {rows} rows ({:.0}s)",
            t0.elapsed().as_secs_f32()
        );
    }
    eprintln!(
        "[skip] selection pass: {rows} rows in {:.0}s | lens tail-vs-logit |Δ| {:.3e}",
        t0.elapsed().as_secs_f32(),
        lens_tail_logit_delta
    );
    Ok(SelectionPass { bi: cos_sum, lens_de })
}

/// katgpt-rs `katgpt-pruners::hydra_budget::calibrate_profiles`, transcribed
/// verbatim — the shipped default-on pruner's own profile formulas (the T5.4
/// arm consumes the SHIPPED criterion; a re-derivation would be a different
/// arm). Returns `(mean_abs_de, backup_frequency, is_erasure)` per layer.
/// Pinned by `hydra_profile_formulas_match_the_shipped_transcription` below.
fn hydra_calibrate_profiles(de_matrix: &[Vec<f32>]) -> Vec<(f32, f32, bool)> {
    let n_layers = de_matrix.first().map_or(0, |r| r.len());
    let n_prompts = de_matrix.len() as f32;
    let mut out = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let (mut sum_abs, mut negative_count, mut backup_count) = (0.0f32, 0usize, 0usize);
        for prompt_de in de_matrix {
            let de = prompt_de.get(l).copied().unwrap_or(0.0);
            sum_abs += de.abs();
            if de < 0.0 {
                negative_count += 1;
            }
            // A layer is a "backup" if it has significant negative DE
            // (indicating it compensates for another layer's damage).
            if de < -0.01 {
                backup_count += 1;
            }
        }
        out.push((
            sum_abs / n_prompts,
            backup_count as f32 / n_prompts,
            negative_count as f32 / n_prompts > 0.5,
        ));
    }
    out
}

/// The parent arm over the full chunk set — the REAL forward
/// ([`forward_qwen_deltanet_ternary_with_hook`]), never the amputation
/// driver: the glue-parity gate must stay an independent-path check even
/// when this bin generates its own cache.
fn run_parent_pass(loaded: &Loaded, chunks: &[Vec<usize>]) -> (Vec<u32>, Vec<u32>, f32) {
    let config = &loaded.config;
    let mut cache = HybridCache::with_layer_types(config, &loaded.weights.layer_types);
    let mut scratch = HybridForwardScratch::new(config);
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
    let mut argmax: Vec<u32> = Vec::new();
    let mut truth: Vec<u32> = Vec::new();
    let t0 = Instant::now();
    for (ci, chunk) in chunks.iter().enumerate() {
        cache.reset();
        for p in 0..chunk.len() - 1 {
            forward_qwen_deltanet_ternary_with_hook(
                &mut x,
                &loaded.weights,
                &mut cache,
                chunk[p],
                p,
                config,
                &mut scratch,
                &loaded.rope_freq,
                None,
                None,
                None,
                None,
            );
            argmax.push(argmax_of(&x, config.vocab_size) as u32);
            truth.push(chunk[p + 1] as u32);
        }
        let done: usize = argmax.len();
        eprintln!(
            "[skip] parent pass: chunk {}/{} | {} positions | {:.2} tok/s",
            ci + 1,
            chunks.len(),
            done,
            done as f32 / t0.elapsed().as_secs_f32().max(1e-6)
        );
    }
    (argmax, truth, t0.elapsed().as_secs_f32())
}

fn main() -> Result<()> {
    let mut parent = None;
    let mut corpus = None;
    let mut cache_path: Option<std::path::PathBuf> = None;
    let mut skip_twt: Option<std::path::PathBuf> = None;
    let mut kept_twt: Vec<usize> = Vec::new();
    let mut arm_shortgpt = false;
    let mut arm_hydra = false;
    let mut arm_random: Option<u64> = None;
    let mut out: Option<std::path::PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--parent" => parent = Some(std::path::PathBuf::from(args.next().expect("path"))),
            "--corpus" => corpus = Some(std::path::PathBuf::from(args.next().expect("path"))),
            "--cache" => cache_path = Some(std::path::PathBuf::from(args.next().expect("path"))),
            "--skip-twt" => skip_twt = Some(std::path::PathBuf::from(args.next().expect("path"))),
            "--kept-twt" => {
                // The dropped-layer list as a comma list (e.g. `4,12,15`) —
                // the amputation set read off the emit's stdout block table
                // (winners). Bypasses the artifact re-read so the arm does
                // not depend on a /tmp file surviving.
                for part in args.next().expect("list").split(',') {
                    kept_twt.push(part.trim().parse().context("kept layer idx")?);
                }
            }
            "--arm-shortgpt" => arm_shortgpt = true,
            "--arm-hydra" => arm_hydra = true,
            "--arm-random" => arm_random = Some(args.next().expect("seed").parse()?),
            "--out" => out = Some(std::path::PathBuf::from(args.next().expect("path"))),
            other => bail!("unknown arg {other}"),
        }
    }
    let parent = parent.context("--parent is required")?;
    let corpus = corpus.context("--corpus is required")?;
    let cache_path = cache_path.context("--cache is required (the parent arm's recording)")?;

    // Box state (the G2 law — accuracy-only arms still record it).
    let loadavg = {
        let out = std::process::Command::new("sysctl").arg("-n").arg("vm.loadavg").output();
        match out {
            Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
            Err(_) => "<unavailable>".to_string(),
        }
    };
    eprintln!("[skip] box: loadavg {loadavg}");

    // ── the frozen token stream (the T5.0 shape; file OR page-directory corpus) ──
    let text = load_corpus_text(&corpus)?;
    let gguf = GgufFile::open(&parent)?;
    let tok = BpeTokenizer::from_gguf(&gguf)?;
    drop(gguf);
    let all = tok.encode(&text);
    let take = all.len().min(MAX_TOKENS);
    let tokens: Vec<usize> = all[..take].to_vec();
    let chunks: Vec<Vec<usize>> = tokens.chunks(SEQ_LEN).map(<[usize]>::to_vec).collect();
    let scored: usize = chunks.iter().map(|c| c.len() - 1).sum();
    eprintln!(
        "[skip] corpus {} chars → {take} tokens → {} chunks × ≤{SEQ_LEN} ({scored} scored)",
        text.len(),
        chunks.len()
    );

    let mut loaded = load_parent(&parent)?;
    // The KV-window cap (the agreement bin's row_logit_floor law — never
    // allocate the advertised context).
    loaded.config.block_size = SEQ_LEN;
    let n_layers = loaded.config.n_layer;
    let parent_display = parent.display().to_string();
    let corpus_display = corpus.display().to_string();

    // ── the parent cache (params-keyed, LOUD replay; self-generates when
    // absent — through the REAL forward, never the amputation driver, so
    // the glue-parity gate below stays an independent-path check) ──
    let (parent_argmax, parent_truth): (Vec<u32>, Vec<u32>) = if cache_path.exists() {
        let cached: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cache_path)?)?;
        let c_parent = cached["parent"].as_str().context("cache: parent")?;
        let c_corpus = cached["corpus"].as_str().context("cache: corpus")?;
        let c_seq = cached["seq_len"].as_u64().context("cache: seq_len")? as usize;
        let c_max = cached["max_tokens"].as_u64().context("cache: max_tokens")? as usize;
        let same_parent = c_parent.ends_with("Ternary-Bonsai-2-27B-PQ2_0.gguf")
            && parent_display.ends_with("Ternary-Bonsai-2-27B-PQ2_0.gguf");
        if !(same_parent && c_corpus == corpus_display && c_seq == SEQ_LEN && c_max == MAX_TOKENS)
        {
            bail!(
                "cache params mismatch: ({c_parent}, {c_corpus}, {c_seq}, {c_max}) vs \
                 ({parent_display}, {corpus_display}, {SEQ_LEN}, {MAX_TOKENS})"
            );
        }
        eprintln!(
            "[skip] parent: CACHE HIT ({parent_display} @ seq {SEQ_LEN} × {MAX_TOKENS}) — replaying {} recorded positions",
            cached["argmax"].as_array().map_or(0, |a| a.len())
        );
        (
            serde_json::from_value(cached["argmax"].clone())?,
            serde_json::from_value(cached["truth"].clone())?,
        )
    } else {
        eprintln!(
            "[skip] parent cache absent — generating over the REAL forward (one {MAX_TOKENS}-position pass)..."
        );
        let (argmax, truth, secs) = run_parent_pass(&loaded, &chunks);
        let hit = argmax.iter().zip(truth.iter()).filter(|(a, t)| a == t).count();
        eprintln!(
            "[skip] parent pass: {} positions in {secs:.0}s ({:.2} tok/s) | hit {:.4}",
            argmax.len(),
            argmax.len() as f32 / secs.max(1e-6),
            hit as f64 / argmax.len() as f64
        );
        let record = serde_json::json!({
            "parent": parent_display, "corpus": corpus_display,
            "seq_len": SEQ_LEN, "max_tokens": MAX_TOKENS,
            "argmax": argmax, "truth": truth,
        });
        std::fs::write(&cache_path, serde_json::to_string(&record)?)
            .with_context(|| format!("write parent cache {}", cache_path.display()))?;
        eprintln!("[skip] parent: cached to {}", cache_path.display());
        (argmax, truth)
    };
    let parent_hit = parent_argmax
        .iter()
        .zip(parent_truth.iter())
        .filter(|(a, b)| a == b)
        .count();
    eprintln!(
        "[skip] parent: {} recorded positions (hit {:.4})",
        parent_argmax.len(),
        parent_hit as f64 / parent_argmax.len() as f64
    );
    assert_eq!(
        parent_argmax.len(),
        scored,
        "cache position count vs the re-derived stream"
    );

    // ── GLUE PARITY: the empty skip set is the parent, byte-identically ──
    {
        let none = vec![false; n_layers];
        let probe_chunks: Vec<Vec<usize>> =
            tokens[..SEQ_LEN].chunks(SEQ_LEN).map(<[usize]>::to_vec).collect();
        let probe = agreement_for_skip(
            &loaded,
            &probe_chunks,
            &parent_argmax[..SEQ_LEN - 1],
            &none,
            "glue-parity (empty skip set)",
        )?;
        assert_eq!(
            probe.agree,
            SEQ_LEN - 1,
            "glue parity FAILED: the empty skip set diverges from the parent forward — \
             the amputation driver's loop glue is wrong; fix the instrument, never the checkpoint"
        );
        eprintln!("[skip] glue parity: OK ({} positions byte-identical)", SEQ_LEN - 1);
    }

    let mut results = serde_json::Map::new();
    let mut token_hasher = blake3::Hasher::new();
    for &t in &tokens {
        token_hasher.update(&t.to_le_bytes());
    }
    let tokens_blake3 = token_hasher.finalize().to_hex()[..16].to_string();
    results.insert(
        "protocol".into(),
        serde_json::json!({
            "issue": "022 T5.4 skip-class arms",
            "parent": parent_display,
            "corpus": corpus_display,
            "tokens_blake3": tokens_blake3,
            "seq_len": SEQ_LEN, "max_tokens": MAX_TOKENS,
            "scored_positions": scored,
            "parent_hit": parent_hit as f64 / parent_argmax.len() as f64,
            "agreement_bar": AGREEMENT_BAR,
            "box_loadavg": loadavg,
        }),
    );

    // ── the TWT arm: the kept set = all layers minus the dropped list ──
    let twt_kept: Vec<usize> = if !kept_twt.is_empty() {
        let mut dropped = kept_twt.clone();
        dropped.sort_unstable();
        let kept: Vec<usize> = (0..n_layers).filter(|l| !dropped.contains(l)).collect();
        let type_name = |l: usize| {
            if loaded.weights.layer_types[l]
                == riir_infer_core::types::DeltaNetLayerType::DeltaNet
            {
                "gdn"
            } else {
                "attn"
            }
        };
        eprintln!(
            "[skip] twt arm: {} kept (dropped {dropped:?} = [{}])",
            kept.len(),
            dropped
                .iter()
                .map(|&l| format!("{l}={}", type_name(l)))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let skip: Vec<bool> = (0..n_layers).map(|i| !kept.contains(&i)).collect();
        let a = agreement_for_skip(&loaded, &chunks, &parent_argmax, &skip, "twt (equal cut)")?;
        results.insert(
            "twt".into(),
            serde_json::json!({
                "kept": kept, "dropped": dropped,
                "agree": a.agree, "hit": a.hit,
                "first_div": a.first_div,
                "agreement": a.agree as f64 / scored as f64,
                "hit_rate": a.hit as f64 / scored as f64,
                "secs": a.secs,
            }),
        );
        kept
    } else if let Some(path) = &skip_twt {
        let file = GgufFile::open(path)?;
        let bt = file
            .metadata
            .get("twt.block_table")
            .context("collapsed artifact carries no twt.block_table")?;
        let arr = match bt {
            GgufValue::Array(v) => v,
            other => bail!("twt.block_table is {other:?}, expected an array"),
        };
        let pairs: Vec<usize> = arr
            .iter()
            .map(|v| v.as_u64().expect("u32 block table entry") as usize)
            .collect();
        assert!(pairs.len().is_multiple_of(2), "block table is (start,end) pairs");
        let mut kept = Vec::new();
        for pair in pairs.chunks(2) {
            let (st, en) = (pair[0], pair[1]);
            assert!(en > st, "empty block in table");
            // A member-emit artifact's multi-layer block would mean a
            // MERGED block — this arm consumes member emits only (the
            // equal-cut comparison is against the passthrough class; the
            // merged class is the audition's own record).
            if en - st == 1 {
                kept.push(st);
            } else {
                bail!(
                    "multi-layer block [{st},{en}) in {path:?}: a merged-block artifact needs \
                     the selection JSON, not the metadata table (this arm consumes member emits)"
                );
            }
        }
        eprintln!("[skip] twt arm: {} kept layers from {}", kept.len(), path.display());
        let skip: Vec<bool> = (0..n_layers).map(|i| !kept.contains(&i)).collect();
        let a = agreement_for_skip(&loaded, &chunks, &parent_argmax, &skip, "twt (equal cut)")?;
        results.insert(
            "twt".into(),
            serde_json::json!({
                "kept": kept, "agree": a.agree, "hit": a.hit,
                "first_div": a.first_div,
                "agreement": a.agree as f64 / scored as f64,
                "hit_rate": a.hit as f64 / scored as f64,
                "secs": a.secs,
            }),
        );
        kept
    } else {
        Vec::new()
    };

    // ── the selection pass: ONE capture pass serves both selection criteria ──
    let selection = if arm_shortgpt || arm_hydra {
        if twt_kept.is_empty() {
            bail!(
                "--arm-shortgpt/--arm-hydra need the TWT kept count (--kept-twt/--skip-twt) — arms run at equal FLOP"
            );
        }
        Some(fused_selection_pass(&loaded, &chunks)?)
    } else {
        None
    };

    // ── the ShortGPT arm: Block Influence + greedy elimination ──
    if arm_shortgpt {
        if twt_kept.is_empty() {
            bail!("--arm-shortgpt needs --skip-twt (the arm runs at the TWT kept count)");
        }
        let k = twt_kept.len();
        let bi = &selection.as_ref().expect("selection pass").bi;
        results.insert(
            "block_influence".into(),
            serde_json::json!(bi.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<f64>>()),
        );
        // Greedy elimination (the ShortGPT pass-2 shape): repeatedly drop
        // the layer whose removal least damages the seeded scan window's
        // next-token hit count, until k layers remain. The scan is the
        // paper's own calibration granularity, pinned + deterministic.
        const SCAN_TOKENS: usize = 256;
        let scan_tokens: Vec<usize> = tokens[..SCAN_TOKENS.min(tokens.len())].to_vec();
        let eval = |alive: &[bool], loaded: &Loaded| -> Result<usize> {
            let mut cache =
                HybridCache::with_layer_types(&loaded.config, &loaded.weights.layer_types);
            let mut scratch = HybridForwardScratch::new(&loaded.config);
            let mut x = vec![0.0f32; loaded.config.vocab_size.max(loaded.config.n_embd)];
            let mut hits = 0usize;
            cache.reset();
            for p in 0..scan_tokens.len() - 1 {
                forward_amputated(
                    &mut x,
                    loaded,
                    &mut cache,
                    &mut scratch,
                    scan_tokens[p],
                    p,
                    alive,
                );
                if argmax_of(&x, loaded.config.vocab_size) == scan_tokens[p + 1] {
                    hits += 1;
                }
            }
            Ok(hits)
        };
        let t_sel = Instant::now();
        let mut alive: Vec<bool> = vec![true; n_layers];
        let mut base_hits = eval(&alive, &loaded)?;
        let mut removed: Vec<usize> = Vec::new();
        let removals = n_layers - k;
        for step in 0..removals {
            let mut best: Option<(usize, usize)> = None;
            for l in 0..n_layers {
                if !alive[l] {
                    continue;
                }
                alive[l] = false;
                let hits = eval(&alive, &loaded)?;
                alive[l] = true;
                eprintln!(
                    "[skip] greedy step {}: drop {l} → hits {hits} (base {base_hits})",
                    step + 1
                );
                if best.is_none_or(|(_, bh)| hits > bh) {
                    best = Some((l, hits));
                }
            }
            let (l, hits) = best.expect("alive layers remain");
            alive[l] = false;
            removed.push(l);
            base_hits = hits;
            eprintln!(
                "[skip] greedy: REMOVED {l} ({} alive, {:.0}s into selection)",
                alive.iter().filter(|&&a| a).count(),
                t_sel.elapsed().as_secs_f32()
            );
        }
        removed.sort_unstable();
        let shortgpt_kept: Vec<usize> = (0..n_layers).filter(|&l| alive[l]).collect();
        let same_set = twt_kept == shortgpt_kept;
        eprintln!("[skip] shortgpt: kept {k} = {shortgpt_kept:?} (dropped {removed:?}) | same set as twt: {same_set}");
        let a = agreement_for_skip(&loaded, &chunks, &parent_argmax, &alive, "shortgpt-greedy (equal cut)")?;
        results.insert(
            "shortgpt".into(),
            serde_json::json!({
                "kept": shortgpt_kept, "removed": removed,
                "same_set_as_twt": same_set,
                "agree": a.agree, "hit": a.hit, "first_div": a.first_div,
                "agreement": a.agree as f64 / scored as f64,
                "hit_rate": a.hit as f64 / scored as f64,
                "secs": a.secs,
                "scan_tokens": SCAN_TOKENS,
            }),
        );
    }

    // ── the hydra arm: the shipped katgpt-rs pruner's criterion at equal FLOP ──
    if arm_hydra {
        let sel = selection.as_ref().expect("selection pass");
        let k = twt_kept.len();
        let profiles = hydra_calibrate_profiles(&sel.lens_de);
        results.insert(
            "hydra_profiles".into(),
            serde_json::json!(profiles
                .iter()
                .map(|(de, b, e)| ((de * 1e4).round() / 1e4, (b * 1e4).round() / 1e4, *e))
                .collect::<Vec<_>>()),
        );
        // The shipped threshold rule (HydraBudgetConfig::default —
        // skip_threshold 0.01, skip_erasure_draft false): skip = non-backup ∧
        // |mean_de| < threshold. Recorded as DISCLOSURE — its native skip
        // count is budget-free, the equal-FLOP gate fixes the count instead.
        const SHIPPED_SKIP_THRESHOLD: f32 = 0.01;
        let native: Vec<usize> = (0..n_layers)
            .filter(|&l| profiles[l].1 <= 0.1 && profiles[l].0.abs() < SHIPPED_SKIP_THRESHOLD)
            .collect();
        // Equal-FLOP translation: the criterion's own importance ranking
        // (|mean_de| ascending, Hydra backups protected) at the TWT kept count.
        let mut eligible: Vec<usize> = (0..n_layers).filter(|&l| profiles[l].1 <= 0.1).collect();
        eligible.sort_by(|&a, &b| profiles[a].0.abs().total_cmp(&profiles[b].0.abs()));
        let removals = n_layers - k;
        if eligible.len() < removals {
            bail!(
                "hydra: only {} non-backup layers eligible for {removals} removals",
                eligible.len()
            );
        }
        let removed: Vec<usize> = eligible[..removals].to_vec();
        let kept: Vec<usize> = (0..n_layers).filter(|l| !removed.contains(l)).collect();
        eprintln!(
            "[skip] hydra: native rule skips {} at threshold {SHIPPED_SKIP_THRESHOLD}; equal-FLOP drop of {removals}: {removed:?} (lowest |mean_de| non-backup)",
            native.len()
        );
        let skip: Vec<bool> = (0..n_layers).map(|i| removed.contains(&i)).collect();
        let a = agreement_for_skip(&loaded, &chunks, &parent_argmax, &skip, "hydra (equal cut)")?;
        results.insert(
            "hydra".into(),
            serde_json::json!({
                "criterion": "katgpt-rs hydra_budget logit-lens DE (calibrate_profiles verbatim)",
                "shipped_rule_native_skips": native,
                "removed": removed, "kept": kept,
                "agree": a.agree, "hit": a.hit, "first_div": a.first_div,
                "agreement": a.agree as f64 / scored as f64,
                "hit_rate": a.hit as f64 / scored as f64,
                "secs": a.secs,
            }),
        );
    }

    // ── the random arm: the floor control at the same count ──
    if let Some(seed) = arm_random {
        if twt_kept.is_empty() {
            bail!("--arm-random needs --skip-twt (the arm runs at the TWT kept count)");
        }
        let k = twt_kept.len();
        // Deterministic LCG (the repo's reproducible-inputs convention).
        let mut s = seed;
        let mut rng = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as usize
        };
        let mut idxs: Vec<usize> = (0..n_layers).collect();
        for i in (1..idxs.len()).rev() {
            let j = rng() % (i + 1);
            idxs.swap(i, j);
        }
        let removed: Vec<usize> = idxs[..n_layers - k].to_vec();
        let skip: Vec<bool> = {
            let mut v = vec![false; n_layers];
            for &l in &removed {
                v[l] = true;
            }
            v
        };
        eprintln!("[skip] random arm (seed {seed}): dropped {removed:?}");
        let a = agreement_for_skip(&loaded, &chunks, &parent_argmax, &skip, "random (floor)")?;
        results.insert(
            "random".into(),
            serde_json::json!({
                "seed": seed, "removed": removed,
                "agree": a.agree, "hit": a.hit, "first_div": a.first_div,
                "agreement": a.agree as f64 / scored as f64,
                "hit_rate": a.hit as f64 / scored as f64,
                "secs": a.secs,
            }),
        );
    }

    if let Some(path) = out {
        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "{}", serde_json::to_string_pretty(&serde_json::Value::Object(results))?)?;
        eprintln!("[skip] record → {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydra_profile_formulas_match_the_shipped_transcription() {
        // katgpt-rs katgpt-pruners/src/hydra_budget.rs :: test_calibrate_profiles
        // — the arm consumes the SHIPPED criterion, so the transcription is
        // pinned against THEIR own known-answer values (plus the backup rule
        // the same formulas imply: DE < -0.01 counts a backup prompt).
        let de_matrix = vec![vec![0.5, -0.3, 0.01], vec![0.4, -0.1, 0.02]];
        let p = hydra_calibrate_profiles(&de_matrix);
        assert_eq!(p.len(), 3);
        // mean |DE| per layer.
        assert!((p[0].0 - 0.45).abs() < 1e-6);
        assert!((p[1].0 - 0.2).abs() < 1e-6);
        assert!((p[2].0 - 0.015).abs() < 1e-6);
        // erasure: majority-negative DE per layer.
        assert!(!p[0].2);
        assert!(p[1].2);
        assert!(!p[2].2);
        // backup frequency: fraction of positions with DE < -0.01.
        assert!((p[0].1 - 0.0).abs() < 1e-6);
        assert!((p[1].1 - 1.0).abs() < 1e-6);
        assert!((p[2].1 - 0.0).abs() < 1e-6);
    }
}
