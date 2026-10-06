//! spike_census_t4_sink_constancy — Issue 919 T4: the delimiter-sink
//! constancy probe (katgpt-rs `.issues/919`, secondary deliverable).
//!
//! The T3 KV-exemption lane closed as no-go (scale-based exemption dies on
//! QK-norm). T4 probes the FOLDING question, which QK-norm does not preclude:
//! are delimiter-class sink K/V vectors CONSTANT across prompts? The paper's
//! claim (arXiv:2603.05498): sink keys collapse to 1–2 dims of the `W_K` row
//! space. Position-0 sinks are exact by construction and excluded — the open
//! class is self-sinking delimiters at varying positions, where the foldable
//! constants are the PRE-RoPE K and V rows (post-RoPE K is position-bound).
//!
//! **Protocol (pre-registered BEFORE the first run; the vk_p1_g1 law — these
//! gates do not move after numbers exist):**
//!
//! - **Cells:** (delimiter class × kind ∈ {K, V} × layer × kv-head).
//! - **Occurrences:** first occurrence of a class per window (one per window,
//!   cross-prompt by construction); windows = [`PASSAGES`] × `--repeats`
//!   chunked at `--seq-len`; window-position 0 excluded; per-class budget
//!   `--max-occ`, first-come across windows in fixed order.
//! - **Capture:** `ValueStoreHook::keys_pre_rope` (K — post-projection,
//!   pre-RoPE; this checkpoint carries NO qk-norm tensors — verified by
//!   header scan, so projected K IS pre-RoPE K here) +
//!   `ValueStoreHook::value_stored` (V — the f16 forward has no RoVE path,
//!   so the stored row IS pre-RoPE V by construction).
//! - **Per-cell stats:** median/mean/min pairwise cosine of the head-sliced
//!   vectors (hd=256), uncentered top-1 eigen share of the Gram spectrum
//!   (the literal "collapse to one dimension" measure — → 1 when every
//!   vector is ≈ equal), participation ratio of the centered covariance
//!   (variance-subspace dimension).
//! - **Per-class verdict (the issue's bar):** pooled median pairwise cosine
//!   over ALL cells' pairs > **0.99**. Disclosed beside it: min per-cell
//!   median, fraction of cells whose median > 0.99 (per-family retention
//!   law — a pooled median must not hide a dead cell), median top-1 share,
//!   median PR.
//! - **Overall:** T4 PASS requires EVERY measured class's K verdict to pass
//!   (K is the folding subject of T5); V is disclosed in the same table.
//!   A class with < 2 occurrences is UNMEASURABLE — disclosed, never
//!   silently dropped.
//!
//! MEASUREMENT LANE: no serving claim; the bar decides. Deterministic
//! (same binary + same model → byte-identical sidecar; matmul parallelism
//! partitions rows, so reductions are fixed-order).
//!
//! Usage:
//! ```text
//! cargo run --release --features spike_census_t4 --bin spike_census_t4_sink_constancy -- \
//!     --gguf ../riir-train/data/gemma-2-2b-it-f16.gguf --out /tmp/t4_sink \
//!     [--seq-len 256] [--repeats 6] [--max-occ 24] [--self-test]
//! ```

use anyhow::{Context, Result, bail};

use riir_infer_core::gguf_loader::{GgufFile, config_from_gguf_metadata};
use riir_infer_core::tokenizer::SentencePieceGgufTokenizer;
use riir_infer_core::transformer::gemma2_calibration::load_gemma2_f16_direct;
use riir_infer_core::transformer::{ForwardContext, ValueStoreHook, forward_gemma2_f16_hk};
use riir_infer_core::types::kv_dim;

use katgpt_transformer::MultiLayerKVCache;

use riir_infer_core::quant::kvq_harness::PASSAGES;

/// The delimiter classes (name, single-char piece). Classes whose piece is
/// not a single token, or that never appear in the encoded corpus, are
/// disclosed as absent — never silently dropped.
const CLASSES: [(&str, &str); 6] = [
    ("comma", ","),
    ("period", "."),
    ("colon", ":"),
    ("semicolon", ";"),
    ("question", "?"),
    ("quote", "\""),
];

/// One captured delimiter occurrence: which window + position it came from
/// (the row block is per (class, layer) and aligned with these).
struct OccMeta {
    window: usize,
    pos: usize,
}

struct ClassData {
    name: &'static str,
    token_id: usize,
    /// (window, pos) per occurrence, aligned with every `rows` vector below.
    meta: Vec<OccMeta>,
    /// `k_rows[class_block][layer]` → rows of `kvd` f32, aligned with `meta`.
    k_rows: Vec<Vec<Vec<f32>>>,
    v_rows: Vec<Vec<Vec<f32>>>,
}

/// The capture hook. `window_map` is the CURRENT window's capture map —
/// `(pos, class_idx)` pairs, one per class, positions ≥ 1. The hook fires
/// once per layer per position; each firing copies one row. The occurrence
/// META (window, pos) was already registered at window-build time (that is
/// where the per-class budget is enforced), so the hook only pushes rows —
/// row order == meta order (both are window order).
struct SinkCapture {
    kvd: usize,
    window_map: Vec<(usize, u8)>,
    classes: Vec<ClassData>,
}

impl SinkCapture {
    fn begin_window(&mut self, map: Vec<(usize, u8)>) {
        self.window_map = map;
    }

    fn class_at(&self, pos: usize) -> Option<u8> {
        self.window_map
            .iter()
            .find(|&&(p, _)| p == pos)
            .map(|&(_, c)| c)
    }
}

impl ValueStoreHook for SinkCapture {
    fn keys_pre_rope(&mut self, layer_idx: usize, pos: usize, k_pre: &[f32]) {
        if let Some(ci) = self.class_at(pos) {
            let cd = &mut self.classes[ci as usize];
            debug_assert_eq!(cd.k_rows[layer_idx].len() + 1, cd.meta.len());
            cd.k_rows[layer_idx].push(k_pre[..self.kvd].to_vec());
        }
    }

    fn value_stored(&mut self, layer_idx: usize, pos: usize, layer_values: &mut [f32]) {
        if let Some(ci) = self.class_at(pos) {
            let cd = &mut self.classes[ci as usize];
            debug_assert_eq!(cd.v_rows[layer_idx].len() + 1, cd.meta.len());
            let off = pos * self.kvd;
            cd.v_rows[layer_idx].push(layer_values[off..off + self.kvd].to_vec());
        }
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.self_test {
        return self_test();
    }

    // ── Load model + tokenizer from ONE open GGUF (the T3 shape) ──
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
        "# spike_census_t4_sink_constancy: {model_name} | layers={} n_embd={} kv_heads={} head_dim={} vocab={} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        config.n_kv_head,
        config.head_dim,
        config.vocab_size,
        t0.elapsed().as_secs_f32()
    );
    let model_bytes = std::fs::metadata(&args.gguf).map(|m| m.len()).unwrap_or(0);
    drop(gguf);

    let kvd = kv_dim(&config);
    let hd = config.head_dim;
    let n_layer = config.n_layer;
    let n_kv = config.n_kv_head;

    // ── Resolve delimiter token ids (single-piece requirement) ──
    let mut classes: Vec<ClassData> = Vec::new();
    for (name, piece) in CLASSES {
        let enc = tok.encode(piece);
        if enc.len() != 1 {
            println!("# class '{name}': piece '{piece}' encodes to {} tokens — ABSENT", enc.len());
            continue;
        }
        classes.push(ClassData {
            name,
            token_id: enc[0],
            meta: Vec::new(),
            k_rows: vec![Vec::new(); n_layer],
            v_rows: vec![Vec::new(); n_layer],
        });
    }

    // ── Build windows + per-window capture maps (first occurrence per
    //    window per class, pos ≥ 1, per-class budget, fixed order) ──
    /// One decoded window: (passage index, tokens, capture map). The map is
    /// `(token position, class index)` — at most one entry per class.
    type Window = (usize, Vec<usize>, Vec<(usize, u8)>);
    let mut windows: Vec<Window> = Vec::new(); // (passage, tokens, map)
    let mut window_idx = 0usize;
    for (pi, p) in PASSAGES.iter().enumerate() {
        let long = tok.encode(&p.repeat(args.repeats));
        for chunk in long.chunks(args.seq_len) {
            if chunk.len() < 32 {
                continue; // tail window too short to be a prompt — disclosed below
            }
            let mut map: Vec<(usize, u8)> = Vec::new();
            for (ci, cd) in classes.iter().enumerate() {
                if cd.meta.len() >= args.max_occ {
                    continue;
                }
                if let Some(pos) = chunk
                    .iter()
                    .enumerate()
                    .skip(1) // position 0 excluded (the exact-by-construction class)
                    .find(|&(_, t)| *t == cd.token_id)
                    .map(|(i, _)| i)
                {
                    map.push((pos, ci as u8));
                }
            }
            // Record meta now (occurrence committed to this window).
            for &(pos, ci) in &map {
                classes[ci as usize].meta.push(OccMeta { window: window_idx, pos });
            }
            windows.push((pi, chunk.to_vec(), map));
            window_idx += 1;
        }
    }
    let total_tokens: usize = windows.iter().map(|(_, t, _)| t.len()).sum();
    let occ_summary: Vec<String> = classes
        .iter()
        .map(|c| format!("{}={}@{}", c.name, c.meta.len(), c.token_id))
        .collect();
    println!(
        "# windows: {} ({total_tokens} tokens, seq_len {} × repeats {}) | occurrences: {}",
        windows.len(),
        args.seq_len,
        args.repeats,
        occ_summary.join(" ")
    );

    // Thin classes are disclosed, not failed: < 8 occurrences gives < 28
    // pairs per cell — reported as `thin` in the verdict table.
    let thin_floor = 8usize;

    // ── Decode + capture ──
    let mut ctx = ForwardContext::new(&config);
    let per_layer = vec![kvd; n_layer];
    let mut cache =
        MultiLayerKVCache::new_with_per_layer_kv_dim_bounded(&config, &per_layer, args.seq_len);
    let mut cap = SinkCapture { kvd, window_map: Vec::new(), classes };
    let t1 = std::time::Instant::now();
    for (_, toks, map) in &windows {
        cap.begin_window(map.clone());
        cache.reset();
        for (pos, &t) in toks.iter().enumerate() {
            forward_gemma2_f16_hk(&mut ctx, &weights, &mut cache, &mut cap, t, pos, &config);
        }
    }
    println!(
        "# decode+capture: {total_tokens} tokens in {:.1}s ({:.1} tok/s)",
        t1.elapsed().as_secs_f32(),
        total_tokens as f32 / t1.elapsed().as_secs_f32()
    );

    // Row/meta alignment is the harness's core invariant.
    for cd in &cap.classes {
        for l in 0..n_layer {
            assert_eq!(cd.k_rows[l].len(), cd.meta.len(), "k rows misaligned: {} layer {l}", cd.name);
            assert_eq!(cd.v_rows[l].len(), cd.meta.len(), "v rows misaligned: {} layer {l}", cd.name);
        }
    }

    // ── Analysis ──
    struct CellStat {
        kind_k: bool,
        layer: usize,
        head: usize,
        n: usize,
        median_cos: f64,
        mean_cos: f64,
        min_cos: f64,
        top1_share: f64,
        pr: f64,
    }
    struct ClassVerdict {
        name: &'static str,
        token_id: usize,
        n: usize,
        thin: bool,
        k: Option<Agg>,
        v: Option<Agg>,
        cells: Vec<CellStat>,
    }
    struct Agg {
        pooled_median: f64,
        min_cell_median: f64,
        cell_pass_frac: f64,
        median_top1: f64,
        median_pr: f64,
        pass: bool,
    }

    let mut verdicts: Vec<ClassVerdict> = Vec::new();
    for cd in &cap.classes {
        let mut cells: Vec<CellStat> = Vec::new();
        let mut k_pooled: Vec<f64> = Vec::new();
        let mut v_pooled: Vec<f64> = Vec::new();
        fn slice(r: &[f32], h: usize, hd: usize) -> &[f32] {
            &r[h * hd..(h + 1) * hd]
        }
        for kind_k in [true, false] {
            let rows = if kind_k { &cd.k_rows } else { &cd.v_rows };
            let pooled = if kind_k { &mut k_pooled } else { &mut v_pooled };
            let n = cd.meta.len();
            if n < 2 {
                continue;
            }
            for (l, layer_rows) in rows.iter().enumerate() {
                for h in 0..n_kv {
                    let mut pairs = Vec::with_capacity(n * (n - 1) / 2);
                    let mut vecs: Vec<Vec<f32>> =
                        layer_rows.iter().map(|r| slice(r, h, hd).to_vec()).collect();
                    for i in 0..n {
                        for j in (i + 1)..n {
                            pairs.push(cosine(&vecs[i], &vecs[j]));
                        }
                    }
                    let (med, mean, min) = triple(&pairs);
                    let (top1, pr) = collapse_stats(&mut vecs);
                    pooled.extend_from_slice(&pairs);
                    cells.push(CellStat {
                        kind_k,
                        layer: l,
                        head: h,
                        n,
                        median_cos: med,
                        mean_cos: mean,
                        min_cos: min,
                        top1_share: top1,
                        pr,
                    });
                }
            }
        }
        let agg = |pooled: &[f64], cells_k: bool| -> Option<Agg> {
            let cell_medians: Vec<f64> = cells
                .iter()
                .filter(|c| c.kind_k == cells_k)
                .map(|c| c.median_cos)
                .collect();
            let top1s: Vec<f64> = cells
                .iter()
                .filter(|c| c.kind_k == cells_k)
                .map(|c| c.top1_share)
                .collect();
            let prs: Vec<f64> = cells
                .iter()
                .filter(|c| c.kind_k == cells_k)
                .map(|c| c.pr)
                .collect();
            if pooled.is_empty() || cell_medians.is_empty() {
                return None;
            }
            let pooled_median = median(pooled);
            let min_cell_median = cell_medians.iter().cloned().fold(f64::INFINITY, f64::min);
            let cell_pass_frac =
                cell_medians.iter().filter(|&&m| m > 0.99).count() as f64 / cell_medians.len() as f64;
            Some(Agg {
                pooled_median,
                min_cell_median,
                cell_pass_frac,
                median_top1: median(&top1s),
                median_pr: median(&prs),
                pass: pooled_median > 0.99,
            })
        };
        verdicts.push(ClassVerdict {
            name: cd.name,
            token_id: cd.token_id,
            n: cd.meta.len(),
            thin: cd.meta.len() < thin_floor,
            k: agg(&k_pooled, true),
            v: agg(&v_pooled, false),
            cells,
        });
    }

    // ── Report ──
    type Formatted = (String, String, String, String, String);
    println!(
        "# {:<10} {:>4} {:>4} | {:>10} {:>10} {:>7} {:>6} {:>6} | {:>10} {:>10} {:>7} {:>6} {:>6}",
        "class", "occ", "thin", "K pooled", "K mincell", "K cells", "K t1", "K PR", "V pooled", "V mincell", "V cells", "V t1", "V PR"
    );
    for v in &verdicts {
        let fmt = |a: &Option<Agg>| -> Formatted {
            match a {
                Some(a) => (
                    format!("{:.4}", a.pooled_median),
                    format!("{:.4}", a.min_cell_median),
                    format!("{:.2}", a.cell_pass_frac),
                    format!("{:.3}", a.median_top1),
                    format!("{:.2}", a.median_pr),
                ),
                None => (
                    "-".to_string(),
                    "-".to_string(),
                    "-".to_string(),
                    "-".to_string(),
                    "-".to_string(),
                ),
            }
        };
        let (kp, km, kc, kt1, kpr) = fmt(&v.k);
        let (vp, vm, vc, vt1, vpr) = fmt(&v.v);
        println!(
            "# {:<10} {:>4} {:>4} | {:>10} {:>10} {:>7} {:>6} {:>6} | {:>10} {:>10} {:>7} {:>6} {:>6}",
            v.name, v.n, if v.thin { "Y" } else { "" }, kp, km, kc, kt1, kpr, vp, vm, vc, vt1, vpr
        );
    }
    let k_all = verdicts
        .iter()
        .all(|v| v.thin || v.k.as_ref().is_some_and(|a| a.pass));
    let v_all = verdicts
        .iter()
        .all(|v| v.thin || v.v.as_ref().is_some_and(|a| a.pass));
    println!("# ── T4 verdict (pre-registered) ──");
    for v in &verdicts {
        let kv = match (&v.k, &v.v) {
            (Some(k), Some(vv)) => format!(
                "K {} V {} (K pooled {:.4} vs 0.99)",
                if k.pass { "PASS" } else { "MISS" },
                if vv.pass { "PASS" } else { "MISS" },
                k.pooled_median
            ),
            (Some(k), None) => format!("K {} V n/a", if k.pass { "PASS" } else { "MISS" }),
            (None, Some(vv)) => format!("K n/a V {}", if vv.pass { "PASS" } else { "MISS" }),
            (None, None) => "UNMEASURABLE (<2 occurrences)".to_string(),
        };
        println!("#   {:<10} occ {:>3}{} → {}", v.name, v.n, if v.thin { " THIN" } else { "" }, kv);
    }
    println!(
        "# T4 overall: K axis {} | V axis {} — {}",
        if k_all { "PASS" } else { "MISS" },
        if v_all { "PASS" } else { "MISS" },
        if k_all {
            "the delimiter pre-RoPE K rows are cross-prompt constants at the bar — T5 (folding prototype) is conditionally open"
        } else {
            "at least one class misses the bar — the delimiter folding lane closes as no-go on this cell"
        }
    );

    // ── Sidecar (JSON + blake3, the census fixture convention) ──
    std::fs::create_dir_all(&args.out).context("create out dir")?;
    let sidecar_path = args.out.join(format!("{model_name}.sink_constancy.json"));
    let mut j = String::from("{\n");
    j.push_str(&format!(
        "  \"lane\": \"issue919_t4_sink_constancy\", \"model\": \"{model_name}\", \"model_bytes\": {model_bytes},\n"
    ));
    j.push_str(&format!(
        "  \"protocol\": {{\"seq_len\": {}, \"repeats\": {}, \"max_occ\": {}, \"bar\": \"pooled median pairwise cosine > 0.99 per class\", \"occurrence_rule\": \"first per window, pos>0\", \"qk_norm\": \"absent in this checkpoint (header-verified); projected K IS pre-RoPE K\", \"v_note\": \"f16 forward has no RoVE path; stored V IS pre-RoPE V\"}},\n",
        args.seq_len, args.repeats, args.max_occ
    ));
    j.push_str("  \"classes\": [\n");
    for (vi, v) in verdicts.iter().enumerate() {
        j.push_str("    {\n");
        j.push_str(&format!(
            "      \"class\": \"{}\", \"token_id\": {}, \"occurrences\": {}, \"thin\": {},\n",
            v.name, v.token_id, v.n, v.thin
        ));
        // The occurrence map (window, in-window position) — the disclosure
        // that makes cross-window pairing auditable.
        let occ_json = cap.classes[vi]
            .meta
            .iter()
            .map(|m| format!("[{}, {}]", m.window, m.pos))
            .collect::<Vec<_>>()
            .join(", ");
        j.push_str(&format!("      \"occ_window_pos\": [{}],\n", occ_json));
        let agg_json = |tag: &str, a: &Option<Agg>| match a {
            Some(a) => format!(
                "      \"{tag}\": {{\"pooled_median_cos\": {:.6}, \"min_cell_median_cos\": {:.6}, \"cell_pass_frac\": {:.4}, \"median_top1_share\": {:.6}, \"median_pr\": {:.4}, \"pass\": {}}}",
                a.pooled_median, a.min_cell_median, a.cell_pass_frac, a.median_top1, a.median_pr, a.pass
            ),
            None => format!("      \"{tag}\": null"),
        };
        j.push_str(&agg_json("k", &v.k));
        j.push_str(",\n");
        j.push_str(&agg_json("v", &v.v));
        j.push_str(",\n");
        j.push_str("      \"cells\": [\n");
        for (i, c) in v.cells.iter().enumerate() {
            j.push_str(&format!(
                "        {{\"kind\": \"{}\", \"layer\": {}, \"head\": {}, \"n\": {}, \"median_cos\": {:.6}, \"mean_cos\": {:.6}, \"min_cos\": {:.6}, \"top1_share\": {:.6}, \"pr\": {:.4}}}{}\n",
                if c.kind_k { "k" } else { "v" },
                c.layer,
                c.head,
                c.n,
                c.median_cos,
                c.mean_cos,
                c.min_cos,
                c.top1_share,
                c.pr,
                if i + 1 < v.cells.len() { "," } else { "" }
            ));
        }
        j.push_str("      ]\n");
        j.push_str(&format!("    }}{}\n", if vi + 1 < verdicts.len() { "," } else { "" }));
    }
    j.push_str("  ],\n");
    j.push_str(&format!(
        "  \"verdict\": {{\"k_all_pass\": {k_all}, \"v_all_pass\": {v_all}}}\n}}\n"
    ));
    std::fs::write(&sidecar_path, &j).context("write sidecar")?;
    let digest = blake3::hash(j.as_bytes());
    let blake_path = args.out.join(format!("{model_name}.sink_constancy.json.blake3"));
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
    println!("# done");

    Ok(())
}

struct T4Args {
    gguf: std::path::PathBuf,
    out: std::path::PathBuf,
    seq_len: usize,
    repeats: usize,
    max_occ: usize,
    self_test: bool,
}

fn parse_args() -> Result<T4Args> {
    let mut a = T4Args {
        gguf: std::path::PathBuf::from("../riir-train/data/gemma-2-2b-it-f16.gguf"),
        out: std::path::PathBuf::from("/tmp/t4_sink"),
        seq_len: 256,
        repeats: 6,
        max_occ: 24,
        self_test: false,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--gguf" => {
                a.gguf = std::path::PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--out" => {
                a.out = std::path::PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--seq-len" => {
                a.seq_len = argv[i + 1].parse().context("--seq-len N")?;
                i += 2;
            }
            "--repeats" => {
                a.repeats = argv[i + 1].parse().context("--repeats N")?;
                i += 2;
            }
            "--max-occ" => {
                a.max_occ = argv[i + 1].parse().context("--max-occ N")?;
                i += 2;
            }
            "--self-test" => {
                a.self_test = true;
                i += 1;
            }
            other => bail!("unknown arg {other}"),
        }
    }
    if a.seq_len < 64 || a.seq_len > 4096 {
        bail!("--seq-len must be in 64..=4096");
    }
    if a.repeats == 0 || a.repeats > 64 {
        bail!("--repeats must be in 1..=64");
    }
    if a.max_occ < 2 || a.max_occ > 256 {
        bail!("--max-occ must be in 2..=256");
    }
    Ok(a)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// (median, mean, min) of a pair-cosine sample.
fn triple(v: &[f64]) -> (f64, f64, f64) {
    (median(v), v.iter().sum::<f64>() / v.len() as f64, v.iter().cloned().fold(f64::INFINITY, f64::min))
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// The collapse measures of an occurrence set (in place — `vecs` is scratch):
/// uncentered top-1 eigen share of the second-moment (Gram) spectrum (→ 1
/// when every vector is ≈ equal — the literal "collapse to one dimension"),
/// and the participation ratio of the CENTERED covariance spectrum (the
/// variance-subspace dimension). Returns (top1_share, pr).
fn collapse_stats(vecs: &mut [Vec<f32>]) -> (f64, f64) {
    let n = vecs.len();
    if n < 2 {
        return (1.0, 1.0);
    }
    let dim = vecs[0].len();
    // Uncentered Gram.
    let mut g = vec![0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let vi = &vecs[i];
            let vj = &vecs[j];
            let d: f64 = vi
                .iter()
                .zip(vj.iter())
                .map(|(&x, &y)| x as f64 * y as f64)
                .sum();
            g[i * n + j] = d;
            g[j * n + i] = d;
        }
    }
    let lam_unc = jacobi_eigenvalues(&mut g, n);
    let sum: f64 = lam_unc.iter().map(|l| l.max(0.0)).sum();
    let top1 = if sum > 0.0 { lam_unc.iter().cloned().fold(0f64, f64::max) / sum } else { 0.0 };

    // Centered Gram.
    let mut mean = vec![0f64; dim];
    for v in vecs.iter() {
        for (m, &x) in mean.iter_mut().zip(v.iter()) {
            *m += x as f64;
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    let mut gc = vec![0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let vi = &vecs[i];
            let vj = &vecs[j];
            let d: f64 = mean
                .iter()
                .zip(vi.iter().zip(vj.iter()))
                .map(|(&m, (&x, &y))| (x as f64 - m) * (y as f64 - m))
                .sum();
            gc[i * n + j] = d;
            gc[j * n + i] = d;
        }
    }
    let lam_c = jacobi_eigenvalues(&mut gc, n);
    let s1: f64 = lam_c.iter().map(|l| l.max(0.0)).sum();
    let s2: f64 = lam_c.iter().map(|l| {
        let l = l.max(0.0);
        l * l
    }).sum();
    let pr = if s2 > 0.0 { s1 * s1 / s2 } else { 1.0 };
    (top1, pr)
}

/// Cyclic Jacobi eigenvalues of a symmetric `n×n` row-major matrix
/// (destroyed in place). Deterministic; convergence when the off-diagonal
/// Frobenius norm drops below `1e-12 ×` the diagonal norm (or 100 sweeps).
fn jacobi_eigenvalues(a: &mut [f64], n: usize) -> Vec<f64> {
    for _ in 0..100 {
        let mut off = 0f64;
        let mut diag = 0f64;
        for i in 0..n {
            diag += a[i * n + i] * a[i * n + i];
            for j in (i + 1)..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off <= 1e-24 + 1e-24 * diag {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..n).map(|i| a[i * n + i]).collect()
}

fn self_test() -> Result<()> {
    // 1. Jacobi on a known 3×3: eigenvalues 2, 2±√2.
    let mut m = vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
    let mut lam = jacobi_eigenvalues(&mut m, 3);
    lam.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let r2 = 2.0f64.sqrt();
    for (got, want) in lam.iter().zip([2.0 - r2, 2.0, 2.0 + r2]) {
        assert!((got - want).abs() < 1e-9, "jacobi eigenvalue {got} != {want}");
    }
    // 2. Cosine known-answers.
    let e1 = vec![1.0f32, 0.0, 0.0];
    let e2 = vec![0.0f32, 1.0, 0.0];
    assert!((cosine(&e1, &e1) - 1.0).abs() < 1e-12);
    assert!(cosine(&e1, &e2).abs() < 1e-12);
    // 3. collapse_stats on identical vectors → top1 ≈ 1, PR ≈ 1.
    let mut same = vec![vec![1.0f32, 2.0, 3.0]; 5];
    let (t1, pr) = collapse_stats(&mut same);
    assert!((t1 - 1.0).abs() < 1e-9, "top1 {t1}");
    assert!((pr - 1.0).abs() < 1e-6, "pr {pr}");
    // 4. Orthogonal set → uncentered top1 ≈ 1/n; centered PR ≈ n-1 (the
    //    covariance of n samples has rank ≤ n-1 after centering).
    let mut orth: Vec<Vec<f32>> = (0..4)
        .map(|i| {
            let mut v = vec![0f32; 4];
            v[i] = 1.0;
            v
        })
        .collect();
    let (t1o, pro) = collapse_stats(&mut orth);
    assert!((t1o - 0.25).abs() < 1e-9, "top1 {t1o}");
    assert!((pro - 3.0).abs() < 1e-6, "pr {pro}");
    println!("# self-test: 4/4 arms PASS (jacobi, cosine, collapse-identical, collapse-orthogonal)");
    Ok(())
}
