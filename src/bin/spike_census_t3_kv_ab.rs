//! spike_census_t3_kv_ab — Issue 919 T3: the measured-diagonal per-channel
//! KV exemption vs per-block absmax, at equal bit budget, on a real model.
//!
//! The T2 verdict killed the weight-census channel list as an exemption key
//! (P@4 = 0 on both cells) and re-aimed T3 at the MEASURED activation
//! diagonal. This bin measures that arm on the gemma-2-2b-it f16 cell:
//!
//! 1. **Pass 1 (calibration)** — [`DiagKvCache`](riir_infer_core::quant::kvq_ab::DiagKvCache)
//!    records the exact per-(layer, kind, channel) max/RMS diagonal of the
//!    POST-RoPE K and V rows over the calibration half of each fixture
//!    passage, and the top-S channels per (layer, kind) become the
//!    exemption sets. The sets + diagonal maxima are written out as a
//!    BLAKE3-pinned JSON sidecar (the KV-axis sibling of the FFN SPCM
//!    sidecars T2 produced).
//! 2. **Pass 2 (eval)** — the held-out half of each passage, scored under
//!    five KV policies through the SAME forward loop:
//!
//!    | arm | path | bpw |
//!    |---|---|---|
//!    | `plain` | `forward_gemma2_f16` + `MultiLayerKVCache` | 32.0 |
//!    | `f16` | mirror path, [`RawF32KvCache`] | 32.0 (paired baseline) |
//!    | `q8` | mirror path, [`Q8AbsmaxKvCache`] | 8.5 (the Research-487 gap subject) |
//!    | `exempt` | mirror path, [`ExemptQ8KvCache`], measured sets | 8.5 + S/2 |
//!    | `exempt_rand` | mirror path, [`ExemptQ8KvCache`], seeded-random sets | 8.5 + S/2 |
//!
//!    The `f16` arm doubles as the G0 harness control: the quantized-mirror
//!    path at full precision must reproduce the plain forward's NLLs.
//!
//! **Pre-registered gates** (recorded BEFORE the first run; they do not
//! move after numbers exist — the vk_p1_g1 law):
//!
//! - **G0 (harness):** mean |nll_f16mirror − nll_plain| ≤ 1e-4 per token.
//! - **G-EQ (equal budget):** `exempt` vs `exempt_rand` carry identical
//!   bpw by construction (same S, same sidecar shape); any quality delta
//!   is attributable to the channel SET, not the bits.
//! - **G-MAIN (the T3 question):** `exempt` beats `exempt_rand` on
//!   aggregate eval PPL AND on paired mean |Δnll| vs the `f16` arm. A miss
//!   is a recorded NEGATIVE: the measured diagonal adds nothing over
//!   random exemption at equal budget on this cell.
//! - **G-487:** `q8` vs `f16` aggregate PPL delta — the model-level cost
//!   of per-block absmax on a real model (Bench 691 measured the
//!   synthetic mechanism; this is the model-level number).
//! - **G-SINK (disclosure):** early-position (pos < 8) mean NLL per arm —
//!   where MA/sink structure should show if it matters.
//!
//! Per-family retention is reported beside the aggregate (the lossy-surface
//! law: never aggregate ppl alone — a family-conditional regression cannot
//! vanish into the mean).
//!
//! MEASUREMENT LANE: no serving claim; the gates decide. Deterministic
//! (same binary + same model → byte-identical tables; the random arm is
//! seeded splitmix64).
//!
//! Usage:
//! ```text
//! cargo run --release --bin spike_census_t3_kv_ab -- \
//!     --gguf ../riir-train/data/gemma-2-2b-it-f16.gguf \
//!     --out /tmp/t3_kv_ab [--s 2] [--seq-len 512] [--select max|rms] [--seed 919]
//! ```

use anyhow::{Context, Result, bail};
use katgpt_transformer::MultiLayerKVCache;
use katgpt_types::QuantizedKVCache;

use riir_infer_core::gguf_loader::{GgufFile, config_from_gguf_metadata};
use riir_infer_core::quant::kvq_ab::{
    DiagKvCache, ExemptQ8KvCache, Q8AbsmaxKvCache, RawF32KvCache,
};
use riir_infer_core::quant::kvq_harness::{
    self, EARLY_POS, Family, PASSAGES, REPEATS, SplitMix64, build_arm, nll,
};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::ForwardContext;
use riir_infer_core::transformer::gemma2_calibration::load_gemma2_f16_direct;
use riir_infer_core::transformer::gemma2_quantized::{
    QuantizedKvMirror, forward_gemma2_f16_qkv,
};
use riir_infer_core::transformer::forward_gemma2_f16;
use riir_infer_core::types::kv_dim;

/// Score the eval set on the PLAIN forward (MultiLayerKVCache).
fn score_plain(
    families: &[Family],
    ctx: &mut ForwardContext,
    weights: &riir_infer_core::gemma_layer::GemmaTransformerWeightsF16,
    cache: &mut MultiLayerKVCache,
    config: &riir_infer_core::types::Config,
) -> (Vec<f64>, f64, usize) {
    let mut out = Vec::new();
    let mut early_nll = 0f64;
    let mut early_n = 0usize;
    for f in families {
        for seq in &f.seqs {
            cache.reset();
            for pos in 0..seq.len() - 1 {
                let logits = forward_gemma2_f16(ctx, weights, cache, seq[pos], pos, config);
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
/// mirror forward. The early-position share is accumulated at the true
/// in-sequence position (not a global-index modulo).
fn score_quantized<C: QuantizedKVCache>(
    families: &[Family],
    ctx: &mut ForwardContext,
    weights: &riir_infer_core::gemma_layer::GemmaTransformerWeightsF16,
    cache: &mut C,
    mirror: &mut QuantizedKvMirror,
    config: &riir_infer_core::types::Config,
) -> (Vec<f64>, f64, usize) {
    let mut out = Vec::new();
    let mut early_nll = 0f64;
    let mut early_n = 0usize;
    for f in families {
        for seq in &f.seqs {
            cache.reset();
            mirror.reset();
            for pos in 0..seq.len() - 1 {
                let logits = forward_gemma2_f16_qkv(
                    ctx, weights, cache, mirror, seq[pos], pos, config, None,
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
        "../riir-train/data/gemma-2-2b-it-f16.gguf",
        "/tmp/t3_kv_ab",
    )?;

    // ── Load model + tokenizer from ONE open GGUF (the calib-dump shape) ──
    let t0 = std::time::Instant::now();
    let gguf = GgufFile::open(&args.gguf).context("open gguf")?;
    let arch = gguf.architecture().unwrap_or("unknown");
    if arch != "gemma2" {
        bail!("expected gemma2 architecture, got '{arch}'");
    }
    let config = config_from_gguf_metadata(&gguf)?;
    let tok = SentencePieceGgufTokenizer::from_gguf(&gguf)?;
    let weights = load_gemma2_f16_direct(&gguf, &config)?;
    let model_name = args
        .gguf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    println!(
        "# spike_census_t3_kv_ab: {model_name} | layers={} n_embd={} kv_heads={} head_dim={} vocab={} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        config.n_kv_head,
        config.head_dim,
        config.vocab_size,
        t0.elapsed().as_secs_f32()
    );
    drop(gguf);

    let kvd = kv_dim(&config);
    let n_layer = config.n_layer;

    // ── Families: per-passage cal/eval halves ──
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

    // ── Pass 1: the KV diagonal over the calibration halves ──
    let t1 = std::time::Instant::now();
    let mut diag = DiagKvCache::new(n_layer, args.seq_len, kvd);
    {
        let mut ctx = ForwardContext::new(&config);
        let mut mirror = QuantizedKvMirror::new(&config, args.seq_len);
        for toks in &cal_tokens {
            for chunk in toks.chunks(args.seq_len) {
                diag.reset();
                mirror.reset();
                for (pos, &t) in chunk.iter().enumerate() {
                    forward_gemma2_f16_qkv(
                        &mut ctx,
                        &weights,
                        &mut diag,
                        &mut mirror,
                        t,
                        pos,
                        &config,
                        None,
                    );
                }
            }
        }
    }
    println!(
        "# pass 1 diagonal: {} rows observed in {:.1}s",
        diag.rows_observed,
        t1.elapsed().as_secs_f32()
    );

    // ── Exemption sets: measured + seeded-random control ──
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
        let mk = diag.top_channels(l, true, args.s, args.select_rms);
        let mv = diag.top_channels(l, false, args.s, args.select_rms);
        meas_max_k.push(mk.iter().map(|&(_, v)| v).collect());
        meas_max_v.push(mv.iter().map(|&(_, v)| v).collect());
        let mut mk_set: Vec<usize> = mk.iter().map(|&(c, _)| c).collect();
        let mut mv_set: Vec<usize> = mv.iter().map(|&(c, _)| c).collect();
        // top_channels ranks by magnitude (descending); the backend contract
        // wants ascending channel indices (the zeroing pass is branch-free).
        mk_set.sort_unstable();
        mv_set.sort_unstable();
        let rk = rng.pick_channels(kvd, args.s);
        let rv = rng.pick_channels(kvd, args.s);
        overlap_k += rk.iter().filter(|c| mk_set.contains(c)).count();
        overlap_v += rv.iter().filter(|c| mv_set.contains(c)).count();
        meas_k.push(mk_set);
        meas_v.push(mv_set);
        rand_k.push(rk);
        rand_v.push(rv);
    }
    println!(
        "# exemption sets (S={} by {}): measured-vs-random overlaps k={overlap_k} v={overlap_v} of {} draws",
        args.s,
        if args.select_rms { "rms" } else { "max" },
        2 * n_layer
    );

    // The measured channel locations (first 4 layers shown; the block index
    // is the Bench-691 poisoned-block coordinate) with their magnitudes:
    for l in 0..4.min(n_layer) {
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
    let mut j = String::from("{\n");
    j.push_str(&format!(
        "  \"model\": \"{model_name}\", \"kvd\": {kvd}, \"n_layer\": {n_layer}, \
         \"rows_observed\": {}, \"select\": \"{}\", \"s\": {}, \"seed\": {},\n",
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
        "  \"measured_k\": [{}],\n  \"measured_v\": [{}],\n  \"max_abs_k\": [{}],\n  \"max_abs_v\": [{}]\n}}\n",
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
    let mut ctx = ForwardContext::new(&config);

    let t2 = std::time::Instant::now();
    let mut plain_cache = MultiLayerKVCache::new(&config);
    let (plain_nlls, en, en_n) = score_plain(&families, &mut ctx, &weights, &mut plain_cache, &config);
    let plain = build_arm("plain", &families, plain_nlls, en, en_n);
    println!(
        "# arm plain: ppl {:.4} ({:.1}s)",
        plain.aggregate_ppl(),
        t2.elapsed().as_secs_f32()
    );

    let t3 = std::time::Instant::now();
    let mut mirror = QuantizedKvMirror::new(&config, args.seq_len);
    let mut raw_cache = RawF32KvCache::new(n_layer, args.seq_len, kvd);
    let (nlls, en, en_n) = score_quantized(&families, &mut ctx, &weights, &mut raw_cache, &mut mirror, &config);
    let f16 = build_arm("f16", &families, nlls, en, en_n);
    println!("# arm f16: ppl {:.4} ({:.1}s)", f16.aggregate_ppl(), t3.elapsed().as_secs_f32());

    let t4 = std::time::Instant::now();
    let mut q8_cache = Q8AbsmaxKvCache::new(n_layer, args.seq_len, kvd);
    let (nlls, en, en_n) = score_quantized(&families, &mut ctx, &weights, &mut q8_cache, &mut mirror, &config);
    let q8 = build_arm("q8", &families, nlls, en, en_n);
    println!("# arm q8: ppl {:.4} ({:.1}s)", q8.aggregate_ppl(), t4.elapsed().as_secs_f32());

    let t5 = std::time::Instant::now();
    let mut meas_cache =
        ExemptQ8KvCache::new(n_layer, args.seq_len, kvd, meas_k, meas_v, args.s);
    let (nlls, en, en_n) = score_quantized(&families, &mut ctx, &weights, &mut meas_cache, &mut mirror, &config);
    let meas = build_arm("exempt", &families, nlls, en, en_n);
    println!("# arm exempt: ppl {:.4} ({:.1}s)", meas.aggregate_ppl(), t5.elapsed().as_secs_f32());

    let t6 = std::time::Instant::now();
    let mut rand_cache =
        ExemptQ8KvCache::new(n_layer, args.seq_len, kvd, rand_k, rand_v, args.s);
    let (nlls, en, en_n) = score_quantized(&families, &mut ctx, &weights, &mut rand_cache, &mut mirror, &config);
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
        "# G-487 per-block-absmax cost: q8 {:.4} vs f16 {:.4} (Δ {:+.4}) at 8.5 bpw",
        q8.aggregate_ppl(),
        f16.aggregate_ppl(),
        q8.aggregate_ppl() - f16.aggregate_ppl()
    );
    println!("# done");

    Ok(())
}
