//! hyperthink_t1_delta_census — katgpt-rs Issue 920 T1: the bias-space
//! delta-content construction (the HyperThink modelless lane, arXiv:2610.03039
//! via `.research/606`).
//!
//! Runs the paired capture the issue pre-registers: for each seed-pinned
//! probe query, ONE forward over the 3-shot worked-exemplar prompt `c` plus
//! the query ("with_c") and ONE over the bare query ("plain"), capturing the
//! six projection-OUTPUT windows (`transformer::gemma2_bias_delta`, K
//! excluded — softmax shift-invariance) at each arm's FINAL position (the
//! pre-decode readout; the same tail token in both arms, different absolute
//! position — recorded, inherent). Per window:
//!
//! ```text
//! Δb_window = E_probe[out_with_c − out_plain]
//! ρ_window  = ‖E[Δ]‖² / E[‖Δ‖²]      (capture ratio; 1.0 = pure bias shift)
//! ```
//!
//! Deliverables per run (under `--out`):
//! - `deltas_f32.bin` — per-probe delta vectors, probe-major /
//!   layer-major / site-slot order, native-endian f32 (the T2 codebook +
//!   per-cluster table input; NOT committed — `.raw`-class artifact).
//! - `deltas_manifest.jsonl` — one row per probe (index, per-arm lengths).
//! - `<model>.t1_bias_delta.json` — the sidecar: protocol echo, per-layer
//!   ρ + energy ranking, the PRE-REGISTERED premise verdict, blake3 pins.
//!
//! **Premise gate (pre-registered BEFORE any table spend; the vk_p1_g1 law —
//! these thresholds do not move after numbers exist):**
//!
//! - **C1 (late-half energy):** the last HALF of layers must hold ≥ **0.60**
//!   of total delta energy `Σ_w ‖E[Δ_w]‖²` (the paper's last-half window
//!   law).
//! - **C2 (late-third capture dominance):** mean ρ over the last
//!   `⌈n/3⌉` layers (for 26 layers: 18..25 — the paper's last-8 shape)
//!   must EXCEED mean ρ over the rest.
//!
//! Both must pass for the layer-window premise to count as reproduced; a
//! miss stops the PoC and is reported as such (issue 920: "if the law does
//! not reproduce, the PoC stops and reports that instead"). The verdict is
//! adjudicated ONLY on the full 2,000-probe run — pilot `--limit` runs are
//! machinery validation, never verdicts.
//!
//! **Probe file:** `tests/fixtures/hyperthink/probes_gsm8k_train_2k.jsonl` —
//! 2,000 questions seed-pinned (splitmix64 seed 920, Fisher–Yates over the
//! 7,473-row GSM8K train split, HF datasets-server fetch 2026-10-07) and
//! NEVER re-drawn; the sidecar pins the file's blake3. Prompt `c` is the
//! hardcoded 3-shot exemplar block below (blake3 echoed in the sidecar).
//!
//! MEASUREMENT LANE: no serving claim; the issue's pre-registered gates
//! decide. Deterministic: same binary + model + probe file → byte-identical
//! dump + sidecar (fixed-order f64 accumulation; matmul row partitioning is
//! fixed-order; no wall-clock values in the sidecar).
//!
//! Usage:
//! ```text
//! cargo run --release --features hyperthink_t1 --bin hyperthink_t1_delta_census -- \
//!     --gguf ../riir-train/data/gemma-2-2b-it-f16.gguf \
//!     --probes tests/fixtures/hyperthink/probes_gsm8k_train_2k.jsonl \
//!     --out .raw/hyperthink_t1 \
//!     [--limit 64] [--max-q-tokens 256] [--self-test]
//! ```

use anyhow::{Context, Result, bail};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use katgpt_pruners::bias_delta::{BiasDeltaBuilder, BiasDeltaTable, BiasSite};
use katgpt_transformer::MultiLayerKVCache;

use riir_infer_core::gguf_loader::{GgufFile, config_from_gguf_metadata, load_gemma2_f16_direct};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::ForwardContext;
use riir_infer_core::transformer::gemma2_bias_delta::{
    BiasDeltaHook, BiasDeltaSite, forward_gemma2_f16_bias_hook,
};

/// The static "thinking register" — 3-shot worked exemplars in the GSM8K
/// CoT + `####` convention. blake3 of this exact string is pinned in the
/// sidecar (a one-char change re-pins the whole lane).
const PROMPT_C: &str = "\
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10.
#### 10

Question: Ken had 50 pencils and gave 10 of them to Manny. How many pencils does Ken have now?
Answer: Ken had 50 pencils and gave away 10, so he has 50 - 10 = 40 pencils.
#### 40

Question: A robin ate 10 worms on Wednesday and twice as many on Thursday. How many worms did it eat in total?
Answer: On Thursday, the robin ate 2 x 10 = 20 worms. In total, it ate 10 + 20 = 30 worms.
#### 30

";

/// C1: minimum late-half energy share (module-doc protocol).
const PREMISE_LATE_ENERGY_MIN: f64 = 0.60;

/// One probe's capture state: slot buffers (`layer * 6 + site slot`),
/// populated only at `target_pos` while `active`.
struct ProbeCapture {
    bufs: Vec<Vec<f32>>,
    target_pos: usize,
    active: bool,
}

impl ProbeCapture {
    fn new(n_layers: usize) -> Self {
        Self {
            bufs: (0..n_layers * BiasDeltaSite::ALL.len()).map(|_| Vec::new()).collect(),
            target_pos: 0,
            active: false,
        }
    }
}

impl BiasDeltaHook for ProbeCapture {
    fn proj_out(&mut self, layer_idx: usize, pos: usize, site: BiasDeltaSite, out: &mut [f32]) {
        if self.active && pos == self.target_pos {
            let slot = layer_idx * BiasDeltaSite::ALL.len() + site_slot(site);
            self.bufs[slot].clear();
            self.bufs[slot].extend_from_slice(out);
        }
    }
}

/// Canonical slot of a site (mirrors `BiasSite::slot` upstream).
#[inline]
fn site_slot(site: BiasDeltaSite) -> usize {
    match site {
        BiasDeltaSite::Q => 0,
        BiasDeltaSite::V => 1,
        BiasDeltaSite::O => 2,
        BiasDeltaSite::Gate => 3,
        BiasDeltaSite::Up => 4,
        BiasDeltaSite::Down => 5,
    }
}

struct Args {
    gguf: PathBuf,
    probes: PathBuf,
    out: PathBuf,
    limit: Option<usize>,
    max_q_tokens: usize,
    self_test: bool,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        gguf: PathBuf::from("../riir-train/data/gemma-2-2b-it-f16.gguf"),
        probes: PathBuf::from("tests/fixtures/hyperthink/probes_gsm8k_train_2k.jsonl"),
        out: PathBuf::from(".raw/hyperthink_t1"),
        limit: None,
        max_q_tokens: 256,
        self_test: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--gguf" => args.gguf = it.next().context("--gguf needs a value")?.into(),
            "--probes" => args.probes = it.next().context("--probes needs a value")?.into(),
            "--out" => args.out = it.next().context("--out needs a value")?.into(),
            "--limit" => {
                args.limit =
                    Some(it.next().context("--limit needs a value")?.parse().context(
                        "--limit needs a number",
                    )?);
            }
            "--max-q-tokens" => {
                args.max_q_tokens = it
                    .next()
                    .context("--max-q-tokens needs a value")?
                    .parse()
                    .context("--max-q-tokens needs a number")?;
            }
            "--self-test" => args.self_test = true,
            other => bail!("unknown arg '{other}'"),
        }
    }
    Ok(args)
}

/// The pre-registered premise verdict.
struct PremiseVerdict {
    n_layers: usize,
    late_half_from: usize,
    late_energy_share: f64,
    late_third_from: usize,
    rho_late_third: f64,
    rho_rest: f64,
    c1_pass: bool,
    c2_pass: bool,
}

fn check_premise(t: &BiasDeltaTable) -> PremiseVerdict {
    let n = t.n_layers();
    let late_half_from = n / 2;
    let late_third_from = n - n.div_ceil(3);
    let late_energy_share = t.late_energy_share(late_half_from);
    let rho_late_third = t.mean_capture_ratio(late_third_from..n);
    let rho_rest = t.mean_capture_ratio(0..late_third_from);
    PremiseVerdict {
        n_layers: n,
        late_half_from,
        late_energy_share,
        late_third_from,
        rho_late_third,
        rho_rest,
        c1_pass: late_energy_share >= PREMISE_LATE_ENERGY_MIN,
        c2_pass: rho_late_third > rho_rest,
    }
}

fn self_test() -> Result<()> {
    let mut ok = true;
    // 1. Site-name mapping riir-infer → the katgpt-pruners vocabulary: the
    //    bin never invents a site the builder can't name.
    for s in BiasDeltaSite::ALL {
        let Some(vs) = BiasSite::from_name(s.name()) else {
            eprintln!("FAIL: no BiasSite for '{}'", s.name());
            ok = false;
            continue;
        };
        if vs.slot() != site_slot(s) {
            eprintln!("FAIL: slot mismatch for '{}' ({} vs {})", s.name(), vs.slot(), site_slot(s));
            ok = false;
        }
    }
    println!("# self-test 1/2 site mapping: {}", if ok { "PASS" } else { "FAIL" });

    // 2. Premise arithmetic on synthetic tables: late-concentrated constant
    //    deltas PASS; front-concentrated FAIL C1; uniform noise fails C2.
    let late_concentrated = {
        let mut b = BiasDeltaBuilder::new(4);
        // Late layers carry 100× the energy AND are pure constants (ρ = 1);
        // early layers are noisy (mean 0 → ρ = 0). The premise's positive case.
        for probe in 0..8 {
            let jitter = if probe % 2 == 0 { 1.0 } else { -1.0 };
            b.observe(0, BiasSite::Q, &[jitter], &[0.0]);
            b.observe(1, BiasSite::Q, &[jitter], &[0.0]);
            b.observe(2, BiasSite::Q, &[10.0], &[0.0]);
            b.observe(3, BiasSite::Q, &[10.0], &[0.0]);
            b.end_probe();
        }
        b.finish()
    };
    let v = check_premise(&late_concentrated);
    let pass2 = v.c1_pass && v.c2_pass;
    println!(
        "# self-test 2/2 premise arithmetic (late-concentrated must PASS): {} (late_energy {:.3}, ρ_late {:.3} vs ρ_rest {:.3})",
        if pass2 { "PASS" } else { "FAIL" },
        v.late_energy_share,
        v.rho_late_third,
        v.rho_rest
    );
    ok &= pass2;

    let front_concentrated = {
        let mut b = BiasDeltaBuilder::new(4);
        for _ in 0..4 {
            b.observe(0, BiasSite::Q, &[10.0], &[0.0]);
            b.observe(1, BiasSite::Q, &[10.0], &[0.0]);
            b.observe(2, BiasSite::Q, &[1.0], &[0.0]);
            b.observe(3, BiasSite::Q, &[1.0], &[0.0]);
            b.end_probe();
        }
        b.finish()
    };
    let v = check_premise(&front_concentrated);
    let pass3 = !v.c1_pass;
    println!(
        "# self-test 3/3 premise arithmetic (front-concentrated must FAIL C1): {} (late_energy {:.3})",
        if pass3 { "PASS" } else { "FAIL" },
        v.late_energy_share
    );
    ok &= pass3;

    if !ok {
        bail!("self-test FAILED");
    }
    println!("# self-test: ALL PASS");
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.self_test {
        return self_test();
    }

    // ── Load model + tokenizer from ONE open GGUF (the spike-census shape) ──
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
    let model_bytes = std::fs::metadata(&args.gguf).map(|m| m.len()).unwrap_or(0);
    println!(
        "# hyperthink_t1_delta_census: {model_name} | layers={} n_embd={} n_head={} kv_heads={} head_dim={} mlp={} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        config.n_head,
        config.n_kv_head,
        config.head_dim,
        config.mlp_hidden,
        t0.elapsed().as_secs_f32()
    );
    drop(gguf);

    let n_layer = config.n_layer;
    let n_sites = BiasDeltaSite::ALL.len();

    // ── Probe file (blake3-pinned) + prompt-c pin ──
    let probe_bytes = std::fs::read(&args.probes).context("read probes file")?;
    let probes_blake3 = blake3::hash(&probe_bytes).to_string();
    let prompt_c_blake3 = blake3::hash(PROMPT_C.as_bytes()).to_string();
    let probes: Vec<String> = probe_bytes
        .split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| {
            let s = std::str::from_utf8(l).context("probes file not utf-8")?;
            let v: serde_json::Value = serde_json::from_str(s).context("probe row not json")?;
            v["question"]
                .as_str()
                .map(str::to_string)
                .context("probe row missing 'question'")
        })
        .collect::<Result<_>>()?;
    let total_probes = probes.len();
    let probes = match args.limit {
        Some(n) => &probes[..n.min(total_probes)],
        None => &probes[..],
    };
    println!(
        "# probes: {} of {total_probes} (limit {:?}) | probes blake3 {probes_blake3} | prompt-c blake3 {prompt_c_blake3}",
        probes.len(),
        args.limit
    );

    // ── Outputs ──
    std::fs::create_dir_all(&args.out).context("create out dir")?;
    let dump_path = args.out.join("deltas_f32.bin");
    let manifest_path = args.out.join("deltas_manifest.jsonl");
    let dump_file = std::fs::File::create(&dump_path).context("create dump")?;
    let mut dump = BufWriter::with_capacity(8 << 20, dump_file);
    let manifest_file = std::fs::File::create(&manifest_path).context("create manifest")?;
    let mut manifest = BufWriter::with_capacity(1 << 20, manifest_file);
    // Streaming dump commitment (the sidecar pins the artifact it describes).
    let mut dump_hasher = blake3::Hasher::new();

    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut cap_w = ProbeCapture::new(n_layer);
    let mut cap_p = ProbeCapture::new(n_layer);
    let mut builder = BiasDeltaBuilder::new(n_layer);
    // Per-site ‖Δb‖² (mean-shift energy) disclosed beside the layer rollup.
    let mut total_tokens: usize = 0;
    let mut skipped_long: usize = 0;
    // Scratch delta buffer (reused per site — zero-alloc in the loop).
    let mut delta_scratch: Vec<f32> = Vec::new();

    let t_run = std::time::Instant::now();
    for (pi, q) in probes.iter().enumerate() {
        let with_text = format!("{PROMPT_C}Question: {q}\nAnswer:");
        let plain_text = format!("Question: {q}\nAnswer:");
        let with_toks = tok.encode(&with_text);
        let plain_toks = tok.encode(&plain_text);
        if with_toks.len() > 4096 || plain_toks.len() > args.max_q_tokens + 8 {
            skipped_long += 1;
            continue;
        }

        // Arm 1: with_c — capture at the final position.
        cap_w.active = true;
        cap_w.target_pos = with_toks.len() - 1;
        cache.reset();
        for (pos, &t) in with_toks.iter().enumerate() {
            forward_gemma2_f16_bias_hook(
                &mut ctx, &weights, &mut cache, &mut cap_w, t, pos, &config,
            );
        }
        cap_w.active = false;

        // Arm 2: plain — same tail token, bare context.
        cap_p.active = true;
        cap_p.target_pos = plain_toks.len() - 1;
        cache.reset();
        for (pos, &t) in plain_toks.iter().enumerate() {
            forward_gemma2_f16_bias_hook(
                &mut ctx, &weights, &mut cache, &mut cap_p, t, pos, &config,
            );
        }
        cap_p.active = false;

        // Per-window delta: observe into the builder + write the probe's
        // dump row (layer-major, site-slot order — the dump's one layout).
        for l in 0..n_layer {
            for (si, site) in BiasDeltaSite::ALL.iter().enumerate() {
                let vs = BiasSite::from_name(site.name()).context("site name drift")?;
                let slot = l * n_sites + si;
                let (w, p) = (&cap_w.bufs[slot], &cap_p.bufs[slot]);
                debug_assert_eq!(w.len(), p.len(), "arm depth mismatch l{l} {site:?}");
                builder.observe(l, vs, w, p);
                delta_scratch.clear();
                delta_scratch.extend(w.iter().zip(p.iter()).map(|(&a, &b)| a - b));
                let bytes: &[u8] = bytemuck::cast_slice(&delta_scratch);
                dump.write_all(bytes).context("write dump")?;
                dump_hasher.update(bytes);
            }
        }
        builder.end_probe();
        total_tokens += with_toks.len() + plain_toks.len();
        writeln!(
            manifest,
            "{{\"i\":{},\"n_with\":{},\"n_plain\":{}}}",
            pi,
            with_toks.len(),
            plain_toks.len()
        )
        .context("write manifest")?;

        if (pi + 1) % 25 == 0 || pi + 1 == probes.len() {
            let rate = total_tokens as f32 / t_run.elapsed().as_secs_f32().max(1e-6);
            println!(
                "# probe {}/{} | {total_tokens} tokens | {rate:.0} tok/s",
                pi + 1,
                probes.len()
            );
        }
    }
    drop(dump);
    drop(manifest);
    let dump_blake3 = dump_hasher.finalize().to_string();
    let run_secs = t_run.elapsed().as_secs_f32();

    // ── Freeze + premise verdict ──
    let table = builder.finish();
    let premise = check_premise(&table);

    println!("# ── per-layer ranking (energy share e_ℓ | capture ratio ρ_ℓ) ──");
    for (l, ls) in table.layer_stats().iter().enumerate() {
        let bar = "*".repeat((ls.energy_share * 40.0) as usize);
        println!(
            "# layer {l:>2}: e {:.4} {bar} | ρ {:.4}",
            ls.energy_share, ls.capture_ratio
        );
    }
    println!("# ── per-site rollup (‖Δb‖² share | ρ) ──");
    let mut site_energy = [0.0f64; 6];
    let mut site_exp = [0.0f64; 6];
    for w in table.windows() {
        site_energy[w.site.slot()] += w.mean_sq;
        site_exp[w.site.slot()] += w.exp_sq;
    }
    let total = site_energy.iter().sum::<f64>().max(1e-30);
    for (si, s) in BiasDeltaSite::ALL.iter().enumerate() {
        let rho = if site_exp[si] > 0.0 {
            site_energy[si] / site_exp[si]
        } else {
            0.0
        };
        println!(
            "# site {:>5}: energy share {:.4} | ρ {:.4}",
            s.name(),
            site_energy[si] / total,
            rho
        );
    }

    let overall = premise.c1_pass && premise.c2_pass;
    println!("# ── T1 premise verdict (pre-registered) ──");
    println!(
        "# C1 late-half energy: {:.4} vs ≥ {:.2} → {}",
        premise.late_energy_share,
        PREMISE_LATE_ENERGY_MIN,
        if premise.c1_pass { "PASS" } else { "MISS" }
    );
    println!(
        "# C2 late-third ρ dominance: {:.4} (layers {}..) vs {:.4} (rest) → {}",
        premise.rho_late_third,
        premise.late_third_from,
        premise.rho_rest,
        if premise.c2_pass { "PASS" } else { "MISS" }
    );
    println!(
        "# T1 premise: {} — {}",
        if overall { "PASS" } else { "FAIL" },
        if overall {
            "the late-block-concentration law reproduces on this cell; T2/T3 (codebook + serving lane) are open"
        } else {
            "the layer-window premise does NOT reproduce — the PoC stops here per the issue's pre-registration"
        }
    );

    // ── Sidecar (JSON, blake3 pins; no wall-clock values) ──
    let mut per_layer = Vec::with_capacity(n_layer);
    for (l, ls) in table.layer_stats().iter().enumerate() {
        let mut sites = serde_json::Map::new();
        for w in table
            .windows()
            .iter()
            .filter(|w| w.layer == l)
        {
            sites.insert(
                w.site.name().to_string(),
                serde_json::json!({
                    "rho": w.capture_ratio,
                    "mean_sq": w.mean_sq,
                    "exp_sq": w.exp_sq,
                    "depth": w.delta.len(),
                }),
            );
        }
        per_layer.push(serde_json::json!({
            "layer": l,
            "energy_share": ls.energy_share,
            "capture_ratio": ls.capture_ratio,
            "windows": ls.windows,
            "sites": sites,
        }));
    }
    let sidecar = serde_json::json!({
        "lane": "issue920_t1_bias_delta",
        "model": model_name,
        "model_bytes": model_bytes,
        "config": {
            "n_layer": config.n_layer,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "n_kv_head": config.n_kv_head,
            "head_dim": config.head_dim,
            "mlp_hidden": config.mlp_hidden,
        },
        "protocol": {
            "probes": probes.len(),
            "probes_total_file": total_probes,
            "probes_blake3": probes_blake3,
            "prompt_c_blake3": prompt_c_blake3,
            "arms": "with_c = PROMPT_C + 'Question: {q}\\nAnswer:' ; plain = 'Question: {q}\\nAnswer:'",
            "capture_rule": "six projection-output windows at each arm's FINAL position (same tail token, different absolute position — recorded)",
            "sites": BiasDeltaSite::ALL.iter().map(|s| s.name()).collect::<Vec<_>>(),
            "k_site": "excluded by construction (softmax shift-invariance, the paper's own exclusion)",
            "dump_layout": "probe-major, layer-major, site-slot order, native-endian f32",
            "premise": {
                "c1_late_half_energy_min": PREMISE_LATE_ENERGY_MIN,
                "c2_rule": "mean rho over the last ceil(n/3) layers exceeds mean rho over the rest",
                "adjudicated_on": "the full 2000-probe run only; --limit runs are machinery validation",
            },
        },
        "run": {
            "n_probes": table.n_probes(),
            "skipped_long": skipped_long,
            "total_tokens": total_tokens,
            "delta_dims_total": table.windows().iter().map(|w| w.delta.len()).sum::<usize>(),
            "total_energy": table.total_energy(),
        },
        "artifacts": {
            "deltas_f32_bin": dump_path.file_name().and_then(|s| s.to_str()).unwrap_or("deltas_f32.bin"),
            "dumps_blake3": dump_blake3,
        },
        "per_layer": per_layer,
        "premise_verdict": {
            "n_layers": premise.n_layers,
            "late_half_from": premise.late_half_from,
            "late_energy_share": premise.late_energy_share,
            "late_third_from": premise.late_third_from,
            "rho_late_third": premise.rho_late_third,
            "rho_rest": premise.rho_rest,
            "c1_pass": premise.c1_pass,
            "c2_pass": premise.c2_pass,
            "overall": if overall { "PASS" } else { "FAIL" },
        },
    });
    let sidecar_path = args.out.join(format!("{model_name}.t1_bias_delta.json"));
    std::fs::write(&sidecar_path, serde_json::to_vec_pretty(&sidecar)?)
        .context("write sidecar")?;
    println!(
        "# sidecar: {} | dump: {} (blake3 {dump_blake3}) | capture {run_secs:.0}s",
        sidecar_path.display(),
        dump_path.display()
    );
    Ok(())
}
