//! spike_census_t3_kv_ab_ternary — Issue 919 T3 cell 2: the measured-diagonal
//! KV exemption A/B on the DECISIVE cell, Ternary-Bonsai-2-27B-PQ2_0 (qwen35
//! hybrid DeltaNet/attention, Hadamard-folded).
//!
//! Cell 1 (gemma-2-2b) measured G-MAIN PASS-but-thin (+0.0008 ppl over the
//! seeded-random equal-budget control) on a weak-MA model — its headroom was
//! bounded by gemma's small absmax gap. This cell reruns the IDENTICAL
//! protocol (same passages, same families, same gates, same S-by-max
//! selection — the machinery lives in [`riir_infer_core::quant::kvq_harness`])
//! on the model whose T2-measured block ratios hit 140–581×: where per-block
//! absmax should actually hurt, and where the measured diagonal should
//! actually pay.
//!
//! Hybrid-specific facts (the cell-1 protocol, hybrid-executed):
//!
//! - Only the full-attention layers carry KV; the `DeltaNet` layers keep a
//!   fixed recurrent state and never touch the backend. The KV-quant arms
//!   quantize exactly the attention layers' cache (that IS the production
//!   surface — there is no GDN KV to quantize).
//! - The GDN state resets per chunk beside the KV cache (the paired reset is
//!   part of the harness contract; a missed reset would leak cross-chunk
//!   context into every arm equally — but the arms must be clean).
//! - `DiagKvCache` rows are stored only on attention layers, so
//!   `rows_observed` counts attention rows; the exemption-set table carries
//!   the GDN layers' never-read dummies (disclosed in the sidecar).
//!
//! **Pre-registered gates** (identical to cell 1 — recorded before the run):
//!
//! - **G0 (harness):** mean |nll_f16mirror − nll_plain| ≤ 1e-4 per token
//!   (the unit-level pin additionally requires bit-identity on a synthetic
//!   model — `ternary_kvq::tests`).
//! - **G-EQ (equal budget):** `exempt` vs `exempt_rand` at identical bpw by
//!   construction.
//! - **G-MAIN (the T3 question):** `exempt` beats `exempt_rand` on aggregate
//!   eval PPL AND paired mean |Δnll| vs the `f16` arm. A miss is a recorded
//!   NEGATIVE.
//! - **G-487:** `q8` vs `f16` aggregate PPL delta — the model-level cost of
//!   per-block absmax on the strong-MA profile.
//! - **G-SINK (disclosure):** early-position (pos < 8) mean NLL per arm.
//!
//! Per-family retention beside the aggregate (the lossy-surface law).
//!
//! MEASUREMENT LANE: no serving claim; the gates decide. Deterministic
//! (same binary + same model → byte-identical tables; the random arm is
//! seeded splitmix64).
//!
//! Usage:
//! ```text
//! cargo run --release --features spike_census_t3_ternary \
//!     --bin spike_census_t3_kv_ab_ternary -- \
//!     [--gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf] \
//!     [--out /tmp/t3_kv_ab_ternary] [--s 2] [--seq-len 512] [--select max|rms] [--seed 919]
//! ```

use anyhow::{Context, Result};
use katgpt_types::QuantizedKVCache;

use riir_infer_core::deltanet::forward::{
    DeltaNetState, HybridCache, HybridForwardScratch, effective_rotary_dim,
};
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
use riir_infer_core::deltanet::ternary_kvq::{TernaryKvMirror, forward_qwen_deltanet_ternary_qkv};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::quant::kvq_ab::{
    DiagKvCache, ExemptQ8KvCache, Q8AbsmaxKvCache, RawF32KvCache,
};
use riir_infer_core::quant::kvq_harness::{
    self, EARLY_POS, Family, PASSAGES, REPEATS, SplitMix64, build_arm, nll,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::{DeltaNetLayerType, kv_dim};

/// Zero the GDN recurrent + conv states (the paired per-chunk reset beside
/// the KV cache's own `reset`).
fn reset_deltanet_state(state: &mut DeltaNetState) {
    for s in state.recurrent_states.iter_mut() {
        s.fill(0.0);
    }
    for s in state.conv_states.iter_mut() {
        s.fill(0.0);
    }
}

/// Score the eval set on the PLAIN forward (HybridCache).
#[allow(clippy::too_many_arguments)]
fn score_plain(
    families: &[Family],
    x: &mut [f32],
    weights: &riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights,
    cache: &mut HybridCache,
    scratch: &mut HybridForwardScratch,
    rope_freq: &RopeFreqTable,
    config: &riir_infer_core::types::Config,
) -> (Vec<f64>, f64, usize) {
    let mut out = Vec::new();
    let mut early_nll = 0f64;
    let mut early_n = 0usize;
    for f in families {
        for seq in &f.seqs {
            cache.reset();
            for pos in 0..seq.len() - 1 {
                let logits = forward_qwen_deltanet_ternary(
                    x, weights, cache, seq[pos], pos, config, scratch, rope_freq,
                );
                let l = nll(logits, seq[pos + 1]);
                if pos < EARLY_POS {
                    early_nll += l;
                    early_n += 1;
                }
                out.push(l);
            }
        }
    }
    (out, early_nll, early_n)
}

/// Score the eval set on a generic quantized-cache backend through the
/// ternary mirror path. The GDN state resets per chunk beside the backend.
#[allow(clippy::too_many_arguments)]
fn score_quantized<C: QuantizedKVCache>(
    families: &[Family],
    x: &mut [f32],
    weights: &riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights,
    state: &mut DeltaNetState,
    cache: &mut C,
    mirror: &mut TernaryKvMirror,
    scratch: &mut HybridForwardScratch,
    rope_freq: &RopeFreqTable,
    config: &riir_infer_core::types::Config,
) -> (Vec<f64>, f64, usize) {
    let mut out = Vec::new();
    let mut early_nll = 0f64;
    let mut early_n = 0usize;
    for f in families {
        for seq in &f.seqs {
            cache.reset();
            mirror.reset();
            reset_deltanet_state(state);
            for pos in 0..seq.len() - 1 {
                let logits = forward_qwen_deltanet_ternary_qkv(
                    x,
                    weights,
                    state,
                    cache,
                    mirror,
                    seq[pos],
                    pos,
                    config,
                    scratch,
                    rope_freq,
                );
                let l = nll(logits, seq[pos + 1]);
                if pos < EARLY_POS {
                    early_nll += l;
                    early_n += 1;
                }
                out.push(l);
            }
        }
    }
    (out, early_nll, early_n)
}

fn main() -> Result<()> {
    let args = kvq_harness::parse_args(
        "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf",
        "/tmp/t3_kv_ab_ternary",
    )?;

    // ── Load model + tokenizer (the act_diagonal_calibration recipe) ──
    let t0 = std::time::Instant::now();
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&args.gguf)
        .with_context(|| format!("load {}", args.gguf.display()))?;
    let tok = {
        let gguf = GgufFile::open(&args.gguf).context("re-open gguf for tokenizer")?;
        BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?
    };
    let model_name = args
        .gguf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();

    // The row_logit_floor law: cap the cache window at what the run needs —
    // a 262K-context model would otherwise allocate its whole advertised
    // window per attention layer. BEFORE any cache construction.
    config.block_size = args.seq_len;

    let n_attn = weights
        .layer_types
        .iter()
        .filter(|&&lt| lt == DeltaNetLayerType::Attention)
        .count();
    let attn_layers: Vec<usize> = weights
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, lt)| **lt == DeltaNetLayerType::Attention)
        .map(|(i, _)| i)
        .collect();
    println!(
        "# spike_census_t3_kv_ab_ternary: {model_name} | layers={} (gdn {}, attn {n_attn}) \
         n_embd={} kv_heads={} head_dim={} vocab={} | hadamard={} | load {:.1}s",
        config.n_layer,
        config.n_layer - n_attn,
        config.n_embd,
        config.n_kv_head,
        config.head_dim,
        config.vocab_size,
        weights.rotation.is_some(),
        t0.elapsed().as_secs_f32()
    );

    let kvd = kv_dim(&config);
    let n_layer = config.n_layer;
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));

    // ── Families: per-passage cal/eval halves (the shared harness) ──
    let mut cal_tokens: Vec<Vec<usize>> = Vec::new();
    let mut families: Vec<Family> = Vec::new();
    for (pi, p) in PASSAGES.iter().enumerate() {
        let toks = tok.encode(&p.repeat(REPEATS));
        let mid = toks.len() / 2;
        cal_tokens.push(toks[..mid].to_vec());
        let eval: Vec<usize> = toks[mid..].to_vec();
        let seqs: Vec<Vec<usize>> = eval.chunks(args.seq_len).map(<[usize]>::to_vec).collect();
        families.push(Family {
            name: match pi {
                0 => "passage1",
                1 => "passage2",
                2 => "passage3",
                _ => "passage4",
            },
            seqs,
        });
    }
    let cal_total: usize = cal_tokens.iter().map(|t| t.len()).sum();
    let eval_total: usize = families
        .iter()
        .flat_map(|f| &f.seqs)
        .map(|s| s.len().saturating_sub(1))
        .sum();
    println!(
        "# tokens: cal {cal_total} ({} passages × {REPEATS} reps, first half) | eval {eval_total} scored (held-out second half) | seq_len {}",
        PASSAGES.len(),
        args.seq_len
    );

    // ── Pass 1: the KV diagonal over the calibration halves (attention
    //    layers only — GDN layers store nothing) ──
    let t1 = std::time::Instant::now();
    let mut diag = DiagKvCache::new(n_layer, args.seq_len, kvd);
    {
        let mut state = DeltaNetState::new(&config, &weights.layer_types);
        let mut mirror = TernaryKvMirror::new(&config, args.seq_len);
        let mut scratch = HybridForwardScratch::new(&config);
        let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
        let mut done = 0usize;
        let mut next_report = 0usize;
        for toks in &cal_tokens {
            for chunk in toks.chunks(args.seq_len) {
                diag.reset();
                mirror.reset();
                reset_deltanet_state(&mut state);
                for (pos, &t) in chunk.iter().enumerate() {
                    forward_qwen_deltanet_ternary_qkv(
                        &mut x,
                        &weights,
                        &mut state,
                        &mut diag,
                        &mut mirror,
                        t,
                        pos,
                        &config,
                        &mut scratch,
                        &rope_freq,
                    );
                }
                done += chunk.len();
                if done >= next_report {
                    let el = t1.elapsed().as_secs_f32();
                    println!(
                        "# pass1 progress {done}/{cal_total} tokens | {:.0} tok/s | eta {:.0} min",
                        done as f32 / el.max(1e-6),
                        (cal_total - done) as f32 / (done as f32 / el.max(1e-6) + 1e-6) / 60.0
                    );
                    next_report = (done + cal_total / 10).max(done + 1);
                }
            }
        }
    }
    println!(
        "# pass 1 diagonal: {} rows observed in {:.1}s (attention layers only; expected ≈ {n_attn} × cal rows)",
        diag.rows_observed,
        t1.elapsed().as_secs_f32()
    );

    // ── Exemption sets: measured + seeded-random control. The draws run
    //    layer-major k-then-v over ALL layers (a pinned stream); the GDN
    //    layers' sets are never-read dummies (they have no KV to exempt). ──
    let dummy: Vec<usize> = (0..args.s).collect();
    let mut meas_k: Vec<Vec<usize>> = Vec::with_capacity(n_layer);
    let mut meas_v: Vec<Vec<usize>> = Vec::with_capacity(n_layer);
    let mut rand_k: Vec<Vec<usize>> = Vec::with_capacity(n_layer);
    let mut rand_v: Vec<Vec<usize>> = Vec::with_capacity(n_layer);
    let mut rng = SplitMix64(args.seed);
    let mut overlap_k = 0usize;
    let mut overlap_v = 0usize;
    let mut meas_max_k: Vec<Vec<f64>> = Vec::with_capacity(n_layer);
    let mut meas_max_v: Vec<Vec<f64>> = Vec::with_capacity(n_layer);
    for l in 0..n_layer {
        let rk = rng.pick_channels(kvd, args.s);
        let rv = rng.pick_channels(kvd, args.s);
        if weights.layer_types[l] == DeltaNetLayerType::Attention {
            let mk = diag.top_channels(l, true, args.s, args.select_rms);
            let mv = diag.top_channels(l, false, args.s, args.select_rms);
            meas_max_k.push(mk.iter().map(|&(_, v)| v).collect());
            meas_max_v.push(mv.iter().map(|&(_, v)| v).collect());
            let mut mk_set: Vec<usize> = mk.iter().map(|&(c, _)| c).collect();
            let mut mv_set: Vec<usize> = mv.iter().map(|&(c, _)| c).collect();
            // top_channels ranks by magnitude (descending); the backend
            // contract wants ascending channel indices (branch-free zeroing).
            mk_set.sort_unstable();
            mv_set.sort_unstable();
            overlap_k += rk.iter().filter(|c| mk_set.contains(c)).count();
            overlap_v += rv.iter().filter(|c| mv_set.contains(c)).count();
            meas_k.push(mk_set);
            meas_v.push(mv_set);
        } else {
            // Never consulted: the layer has no KV. Fill with the dummy so
            // ExemptQ8KvCache's exactly-S contract holds; disclosed in the
            // sidecar's gdn_layers list.
            meas_max_k.push(Vec::new());
            meas_max_v.push(Vec::new());
            meas_k.push(dummy.clone());
            meas_v.push(dummy.clone());
        }
        rand_k.push(rk);
        rand_v.push(rv);
    }
    println!(
        "# exemption sets (S={} by {}, attention layers only): measured-vs-random overlaps k={overlap_k} v={overlap_v} of {} draws",
        args.s,
        if args.select_rms { "rms" } else { "max" },
        2 * n_attn
    );

    // The measured channel locations (first 4 ATTENTION layers shown; the
    // block index is the Bench-691 poisoned-block coordinate):
    for &l in attn_layers.iter().take(4) {
        println!(
            "#   layer {l}: k={:?} blocks {:?} max {:?} | v={:?} blocks {:?} max {:?}",
            meas_k[l],
            meas_k[l].iter().map(|c| c / 32).collect::<Vec<_>>(),
            meas_max_k[l].iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>(),
            meas_v[l],
            meas_v[l].iter().map(|c| c / 32).collect::<Vec<_>>(),
            meas_max_v[l].iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>(),
        );
    }

    // ── The sidecar (JSON + blake3, the census fixture convention) ──
    std::fs::create_dir_all(&args.out).context("create out dir")?;
    let sidecar_path = args.out.join(format!("{model_name}.kvdiag.json"));
    let gdn_layers: Vec<usize> = (0..n_layer)
        .filter(|&l| weights.layer_types[l] == DeltaNetLayerType::DeltaNet)
        .collect();
    let mut j = String::from("{\n");
    j.push_str(&format!(
        "  \"model\": \"{model_name}\", \"kvd\": {kvd}, \"n_layer\": {n_layer}, \
         \"attention_layers\": {:?}, \"gdn_layers\": {:?}, \
         \"rows_observed\": {}, \"select\": \"{}\", \"s\": {}, \"seed\": {},\n",
        attn_layers,
        gdn_layers,
        diag.rows_observed,
        if args.select_rms { "rms" } else { "max" },
        args.s,
        args.seed
    ));
    let sets_json = |sets: &[Vec<usize>]| -> String {
        sets.iter()
            .map(|s| format!("[{}]", s.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", ")))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let row_json = |rows: &[Vec<f32>]| -> String {
        rows.iter()
            .map(|r| format!("[{}]", r.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(", ")))
            .collect::<Vec<_>>()
            .join(", ")
    };
    j.push_str(&format!(
        "  \"note\": \"gdn_layers carry no KV; their sets are never-read dummies\",\n  \
         \"measured_k\": [{}],\n  \"measured_v\": [{}],\n  \"max_abs_k\": [{}],\n  \"max_abs_v\": [{}]\n}}\n",
        sets_json(&meas_k),
        sets_json(&meas_v),
        row_json(&diag.max_abs_k),
        row_json(&diag.max_abs_v),
    ));
    std::fs::write(&sidecar_path, &j).context("write sidecar")?;
    let digest = blake3::hash(j.as_bytes());
    let blake_path = args.out.join(format!("{model_name}.kvdiag.json.blake3"));
    std::fs::write(
        &blake_path,
        format!(
            "{}  {}\n",
            digest.to_hex(),
            sidecar_path.file_name().unwrap().to_string_lossy()
        ),
    )
    .context("write blake3")?;
    println!("# sidecar: {} (+ .blake3)", sidecar_path.display());

    // ── Pass 2: the arms ──
    let mut scratch = HybridForwardScratch::new(&config);
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];

    let t2 = std::time::Instant::now();
    let mut plain_cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let (plain_nlls, en, en_n) =
        score_plain(&families, &mut x, &weights, &mut plain_cache, &mut scratch, &rope_freq, &config);
    let plain = build_arm("plain", &families, plain_nlls, en, en_n);
    println!(
        "# arm plain: ppl {:.4} ({:.1}s)",
        plain.aggregate_ppl(),
        t2.elapsed().as_secs_f32()
    );

    let t3 = std::time::Instant::now();
    let mut state = DeltaNetState::new(&config, &weights.layer_types);
    let mut mirror = TernaryKvMirror::new(&config, args.seq_len);
    let mut raw_cache = RawF32KvCache::new(n_layer, args.seq_len, kvd);
    let (nlls, en, en_n) = score_quantized(
        &families, &mut x, &weights, &mut state, &mut raw_cache, &mut mirror,
        &mut scratch, &rope_freq, &config,
    );
    let f16 = build_arm("f16", &families, nlls, en, en_n);
    println!("# arm f16: ppl {:.4} ({:.1}s)", f16.aggregate_ppl(), t3.elapsed().as_secs_f32());

    let t4 = std::time::Instant::now();
    let mut q8_cache = Q8AbsmaxKvCache::new(n_layer, args.seq_len, kvd);
    let (nlls, en, en_n) = score_quantized(
        &families, &mut x, &weights, &mut state, &mut q8_cache, &mut mirror,
        &mut scratch, &rope_freq, &config,
    );
    let q8 = build_arm("q8", &families, nlls, en, en_n);
    println!("# arm q8: ppl {:.4} ({:.1}s)", q8.aggregate_ppl(), t4.elapsed().as_secs_f32());

    let t5 = std::time::Instant::now();
    let mut meas_cache =
        ExemptQ8KvCache::new(n_layer, args.seq_len, kvd, meas_k.clone(), meas_v.clone(), args.s);
    let (nlls, en, en_n) = score_quantized(
        &families, &mut x, &weights, &mut state, &mut meas_cache, &mut mirror,
        &mut scratch, &rope_freq, &config,
    );
    let meas = build_arm("exempt", &families, nlls, en, en_n);
    println!("# arm exempt: ppl {:.4} ({:.1}s)", meas.aggregate_ppl(), t5.elapsed().as_secs_f32());

    let t6 = std::time::Instant::now();
    let mut rand_cache =
        ExemptQ8KvCache::new(n_layer, args.seq_len, kvd, rand_k, rand_v, args.s);
    let (nlls, en, en_n) = score_quantized(
        &families, &mut x, &weights, &mut state, &mut rand_cache, &mut mirror,
        &mut scratch, &rope_freq, &config,
    );
    let rand = build_arm("exempt_rand", &families, nlls, en, en_n);
    println!("# arm exempt_rand: ppl {:.4} ({:.1}s)", rand.aggregate_ppl(), t6.elapsed().as_secs_f32());

    // ── Report ──
    let g0 = f16.mean_abs_delta(&plain);
    let bpw_exempt = 8.5 + args.s as f64 / 2.0;
    println!("# ── per-arm table (ppl | Δppl vs f16 | mean |Δnll| vs f16 | early-pos ppl) ──");
    for a in [&plain, &f16, &q8, &meas, &rand] {
        println!(
            "#   {:<12} {:.4} | {:+.4} | {:.5} | {:.4}",
            a.name,
            a.aggregate_ppl(),
            a.aggregate_ppl() - f16.aggregate_ppl(),
            a.mean_abs_delta(&f16),
            a.early_ppl()
        );
    }
    // Per-family mean NLL (NATS — exp() gives the family ppl; printed as
    // nats so the weighted mean reads directly against the aggregate).
    println!("# per-family mean NLL nats (plain | f16 | q8 | exempt | exempt_rand):");
    for ((pf, ff), (qf, (mf, rf))) in plain
        .fam
        .iter()
        .zip(&f16.fam)
        .zip(q8.fam.iter().zip(meas.fam.iter().zip(&rand.fam)))
    {
        println!(
            "#   {:<10} {:.4} | {:.4} | {:.4} | {:.4} | {:.4}",
            pf.0, pf.1, ff.1, qf.1, mf.1, rf.1
        );
    }

    // ── Verdict block (pre-registered gates) ──
    let meas_ppl = meas.aggregate_ppl();
    let rand_ppl = rand.aggregate_ppl();
    let meas_mad = meas.mean_abs_delta(&f16);
    let rand_mad = rand.mean_abs_delta(&f16);
    println!("# ── gates (pre-registered) ──");
    println!(
        "# G0 harness: mean |nll_mirror − nll_plain| = {g0:.3e} ≤ 1e-4 → {}",
        if g0 <= 1e-4 { "PASS" } else { "FAIL" }
    );
    println!(
        "# G-EQ equal budget: both exempt arms at {bpw_exempt:.2} bpw (8.5 + {}/2), identical sidecar shape → PASS by construction",
        args.s
    );
    println!(
        "# G-MAIN measured-diagonal value: exempt {meas_ppl:.4} vs exempt_rand {rand_ppl:.4} (ppl); MAD {meas_mad:.5} vs {rand_mad:.5} → {}",
        if meas_ppl < rand_ppl && meas_mad < rand_mad {
            "PASS — the measured diagonal carries policy value"
        } else {
            "NEGATIVE — no advantage over random exemption at equal budget on this cell"
        }
    );
    println!(
        "# G-487 per-block-absmax cost: q8 {:.4} vs f16 {:.4} (Δ {:+.4}) at 8.5 bpw — the strong-MA cell",
        q8.aggregate_ppl(),
        f16.aggregate_ppl(),
        q8.aggregate_ppl() - f16.aggregate_ppl()
    );
    println!("# done");

    Ok(())
}
