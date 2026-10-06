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

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use katgpt_transformer::MultiLayerKVCache;
use katgpt_types::QuantizedKVCache;

use riir_infer_core::gguf_loader::{GgufFile, config_from_gguf_metadata};
use riir_infer_core::quant::kvq_ab::{
    DiagKvCache, ExemptQ8KvCache, Q8AbsmaxKvCache, RawF32KvCache,
};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::ForwardContext;
use riir_infer_core::transformer::gemma2_calibration::load_gemma2_f16_direct;
use riir_infer_core::transformer::gemma2_quantized::{
    QuantizedKvMirror, forward_gemma2_f16_qkv,
};
use riir_infer_core::transformer::forward_gemma2_f16;
use riir_infer_core::types::kv_dim;

/// Calibration passages — byte-identical to the T2 SPCM calibration fixture
/// (`riir-ai` `spike_census_calib_dump.rs`), so the KV diagonal here is
/// cross-referable with the FFN SPCM sidecars. Each passage is one FAMILY
/// (per-family retention law).
const PASSAGES: [&str; 4] = [
    "The harbour city grew along the river because ships could carry grain \
     farther than any cart. Merchants built warehouses near the docks, and \
     the warehouses needed clerks, and the clerks needed houses, and within \
     three generations the mudflats had become a town with a charter, a \
     bell tower, and a stubborn council that argued about bridge tolls. \
     When the railway arrived the council argued again, because a line \
     drawn through the valley would cut the orchards in half, but the \
     orchards were already dying of blight, and the freight fees paid for \
     the new school. The school still stands. Its registers record the \
     names of every child who learned to read within earshot of the \
     goods yard, and the ledgers show that the toll dispute ended only \
     when the bridge collapsed and nobody could afford to rebuild it.",
    "A compiler translates source text into machine instructions through a \
     pipeline of passes. The front end parses the text into an abstract \
     syntax tree and resolves every name to a declaration. The middle end \
     optimizes: common subexpressions are computed once, loops are \
     unrolled or interchanged, and values are kept in registers as long \
     as possible. The back end selects instructions, schedules them \
     against latency, and allocates registers under pressure. Each pass \
     must preserve the semantics of the program, which is why correct \
     compilers are tested against specifications rather than examples. \
     Optimization levels trade compile time for execution speed; at the \
     highest level the compiler may vectorize inner loops, inline \
     functions across module boundaries, and reorder memory operations \
     that the source language promised not to reorder, provided the \
     hardware model says the result cannot be observed.",
    "Mira found the key taped under the third stair, exactly where the \
     letter said, though the tape had yellowed and the key had left a \
     rust shadow on the wood. The door at the end of the corridor had \
     not been opened in years; the air behind it smelled of paper and \
     cold dust. Inside, shelves rose to the ceiling, every one of them \
     full, and in the middle of the room stood a desk with a single \
     drawer. She did not open the drawer at first. She walked the \
     aisles instead, reading spines by the light of her phone: \
     field guides, city directories, ship manifests, and notebooks in \
     a hand that grew more hurried with every decade. The last notebook \
     stopped mid-sentence. Its final page held only an address and the \
     words: ask for the tide table.",
    "\"You said the samples were clean,\" Dana said. She did not look up \
     from the chromatogram. \"I said the blanks were clean,\" Idris \
     replied. \"The blanks and the samples came from the same tray.\" \
     \"Then the tray is compromised.\" \"The tray is fine. The pipette \
     is the problem. Watch: same solution, second pipette, and the \
     peak disappears.\" Dana turned the printout toward him. \"Then why \
     does the peak come back when we run it tomorrow?\" Idris was quiet \
     for a moment. \"Because tomorrow is a different day, and the \
     instrument drifts with the weather, and we are measuring something \
     very small on top of something very noisy.\" \"So we repeat it.\" \
     \"So we repeat it, and we log the barometric pressure, and we stop \
     pretending the number has more digits than the experiment \
     deserves.\"",
];

/// One family = one passage repeated `REPEATS` times, split at the midpoint:
/// the first half calibrates (the diagonal + the exemption sets), the
/// second half scores. Register-matched, held-out.
const REPEATS: usize = 3;

/// Chunk positions ≤ this count are "early" (the sink proxy, vk Gate-4).
const EARLY_POS: usize = 8;

struct Args {
    gguf: PathBuf,
    out: PathBuf,
    s: usize,
    seq_len: usize,
    select_rms: bool,
    seed: u64,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        gguf: PathBuf::from("../riir-train/data/gemma-2-2b-it-f16.gguf"),
        out: PathBuf::from("/tmp/t3_kv_ab"),
        s: 2,
        seq_len: 512,
        select_rms: false,
        seed: 919,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                a.gguf = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--out" => {
                a.out = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--s" => {
                a.s = args[i + 1].parse().context("--s N")?;
                i += 2;
            }
            "--seq-len" => {
                a.seq_len = args[i + 1].parse().context("--seq-len N")?;
                i += 2;
            }
            "--select" => {
                a.select_rms = match args[i + 1].as_str() {
                    "max" => false,
                    "rms" => true,
                    other => bail!("--select max|rms, got {other}"),
                };
                i += 2;
            }
            "--seed" => {
                a.seed = args[i + 1].parse().context("--seed N")?;
                i += 2;
            }
            other => bail!("unknown arg {other}"),
        }
    }
    if a.seq_len == 0 || a.seq_len > 4096 {
        bail!("--seq-len must be in 1..=4096");
    }
    if a.s == 0 || a.s > 32 {
        bail!("--s must be in 1..=32");
    }
    Ok(a)
}

/// Deterministic splitmix64 — the random-control channel sets. No dep; the
/// stream is pinned by the seed and consumed in a fixed order
/// (layer-major, k-then-v), so the sets are byte-reproducible.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `s` strictly ascending distinct channels from `0..kvd`, seeded order.
    fn pick_channels(&mut self, kvd: usize, s: usize) -> Vec<usize> {
        let mut chosen = std::collections::BTreeSet::new();
        while chosen.len() < s {
            chosen.insert((self.next() % kvd as u64) as usize);
        }
        chosen.into_iter().collect()
    }
}

fn nll(logits: &[f32], target: usize) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = logits.iter().map(|&l| (l as f64 - m).exp()).sum();
    m + z.ln() - logits[target] as f64
}

/// One family's eval token stream (the held-out half, chunked at `seq_len`).
struct Family {
    name: &'static str,
    seqs: Vec<Vec<usize>>,
}

/// Per-arm scoring result: the global per-token NLL vector (fixed
/// family-major, seq-major, pos-major order — the paired-delta key), the
/// early-position sink-proxy share, and per-family (ppl, n).
struct ArmScore {
    name: &'static str,
    nlls: Vec<f64>,
    early_nll: f64,
    early_n: usize,
    fam: Vec<(&'static str, f64, usize)>,
}

impl ArmScore {
    fn aggregate_ppl(&self) -> f64 {
        (self.nlls.iter().sum::<f64>() / self.nlls.len().max(1) as f64).exp()
    }

    fn early_ppl(&self) -> f64 {
        (self.early_nll / self.early_n.max(1) as f64).exp()
    }

    /// Paired mean |Δnll| against another arm (position-aligned).
    fn mean_abs_delta(&self, base: &ArmScore) -> f64 {
        self.nlls
            .iter()
            .zip(&base.nlls)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / self.nlls.len().max(1) as f64
    }
}

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

fn build_arm(
    name: &'static str,
    families: &[Family],
    nlls: Vec<f64>,
    early_nll: f64,
    early_n: usize,
) -> ArmScore {
    let mut fam = Vec::with_capacity(families.len());
    let mut idx = 0usize;
    for f in families {
        let n: usize = f.seqs.iter().map(|s| s.len().saturating_sub(1)).sum();
        let fam_nll: f64 = nlls[idx..idx + n].iter().sum();
        fam.push((f.name, fam_nll / n.max(1) as f64, n));
        idx += n;
    }
    ArmScore {
        name,
        nlls,
        early_nll,
        early_n,
        fam,
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;

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
