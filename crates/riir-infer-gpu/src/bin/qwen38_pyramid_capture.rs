//! `qwen38_pyramid_capture` — katgpt-rs Issue 908 / Plan 612 T2.1: the
//! FA-layer Q/K real-tensor capture (the PISA pyramid selection gate's
//! fixture producer). 4090 lane (cudarc; GPU-exclusive by the riir-ai
//! AGENTS rule — the co-resident compute probe REFUSES).
//!
//! Runs the qwen38 dense cudarc whole-model lane over ONE deterministic
//! nested-prefix token stream and, at each configured length mark
//! (default 4096 / 16384 / 32768 / 65536), dumps:
//!
//! * **K** — every KV head's post-RoPE rows `[L, head_dim]` read from the
//!   KV cache via [`Qwen38DenseForward::download_k_cache`] (the exact rows
//!   the decode attention scores against), decoded to f32 (the Issue-753
//!   f16 hatch packs halves into the f32 words — which arm ran is recorded
//!   in the manifest);
//! * **Q** — the post-RoPE query row `[n_head, head_dim]` at every FA
//!   layer for a sampled position set (first / quartiles / tail of each
//!   length), tapped via `forward_token_attn_q_capture`;
//! * **manifest.json** — BLAKE3 of model + prompt + every bin, config
//!   echo, posture, box state, env snapshot.
//!
//! `<out>/full/` holds the generous capture (all FA layers' Q, the target
//! layers' all-head K, prompt.txt, tokens.u32). `<out>/commit/` holds the
//! bounded subset katgpt-rs commits (2 layers × 2 heads, rows capped at
//! `PYCAP_COMMIT_ROWS` via uniform stride + the forced positions, plus a
//! `positions_L*.u32` index so the gate knows exactly which rows it holds
//! — Plan 612 T2.1's size law; the full bins stay in gitignored storage
//! with path + digest in the manifest).
//!
//! The nested-prefix property means ONE 65 536-token eager fill serves
//! every length mark: K is dumped at each mark, Q taps fire once at the
//! union of all marks' sampled positions.
//!
//! ```text
//! # 4090 (PowerShell, detached + logged):
//! QWEN38_KV_DTYPE=f16 cargo run --release -p riir-infer-gpu `
//!   --features ternary_gemv_cuda_raw --bin qwen38_pyramid_capture
//! ```
//!
//! Env: `QWEN38_GGUF` (default `F:/models/qwen38-27b-dbirks-Q4_K_M.gguf`)
//! · `PYCAP_LENGTHS` (comma, ascending; default `4096,16384,32768,65536`)
//! · `PYCAP_K_LAYERS` (K target layers, model layer indices; default
//!   `first,last` FA layer) · `PYCAP_COMMIT_HEADS` (default `0,3`) ·
//! `PYCAP_COMMIT_ROWS` (default `4096`) · `PYCAP_OUT` (default `.pycap`) ·
//! `PYCAP_CTX` (buffer ctx; default the max length) · `PYCAP_GIT_SHA`
//! (recorded as `capture_bin_commit`, else `"unknown"`).

#![cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
// ^ whole-file gate (the bench_663 precedent): on macOS this compiles to an
// empty target; the Cargo.toml required-features row is the reader
// protection for a --features selection without the union.

use std::collections::{BTreeMap, HashSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::qwen38_dense_cudarc::{
    Qwen38DenseConfig, Qwen38DenseForward, Qwen38LayerType, kv_f16_bits,
};

/// Deterministic natural-language paragraph bank (the prompt generator's
/// sole input — no RNG, no seed; the manifest pins the result by BLAKE3).
const PARAGRAPHS: [&str; 24] = [
    "Rivers have always decided where people gather. A dependable current carries silt onto the floodplain, and the silt becomes soil, and the soil becomes bread.",
    "The old surveyors worked from ridge to ridge, sighting across valleys with instruments no heavier than a clock and a lens. Their maps still name the passes.",
    "In the workshop, a seasoned joiner reads the grain before the cut. Timber that looks identical on the rack will fight the plane differently on Tuesday than on Friday.",
    "Harbor pilots memorize the shifting bar. Charts go stale in a single winter storm, so the trade stays oral: an apprentice rides along until the channel lives in the hands.",
    "Markets form where two roads cross, and they outlive the roads. Archaeologists find the market first, then the milestone, then the reason for both.",
    "A good furnace draws air like a lung. The smith controls the color more than the flame, because color is temperature, and temperature is everything the steel will become.",
    "Migrating birds calibrate against fields humans cannot sense. Release a homing pigeon under an overcast and it circles once, then chooses, and is rarely argued with.",
    "The library kept its ledger of borrowed rain. Farmers deposited in the wet years and drew against the ledger when the sky forgot its obligations.",
    "Every bridge is a wager about the river's worst mood. The engineer who designs for the average current has built a monument to the next exceptional flood.",
    "Children learn the shortest path home through their feet, not the map. The body's geometry cuts corners the surveyor would never permit on paper.",
    "Salt roads cross the high passes because salt is light, keeps, and everyone needs it. Empires have followed heavier goods into ruin while salt quietly paid the porters.",
    "The clockmaker's bench is a study in deferred consequences. A mainspring wound one tooth too tight will keep perfect time for exactly as long as it takes to matter.",
    "Orchardists talk to each other through grafts. A variety that never flowered in one valley fruits generously two ridges east, and nobody fully knows why.",
    "The tides keep an older appointment book. Whatever the harbor plans, the water arrives on schedule, and every fitting, every launch, every cargo waits on that ledger.",
    "Stonecutters judge a quarry by its ring. Strike the face and listen: a clear note means the bed is honest, a dull one means the rock is arguing with itself.",
    "Caravan routes are negotiated with water, not distance. A well is worth a week's detour, and the map that ignores this is a document about somewhere else.",
    "The weaver's loom is a small machine for holding tension honestly. Slack anywhere becomes a flaw everywhere, and the cloth remembers every compromise.",
    "Weather stations on the ridge line report a different country than the valley floor. Both are correct, which is why the farmer asks two neighbors and decides alone.",
    "Sailors distinguish forty words for wind the way physicians distinguish pain. The vocabulary exists because the difference once decided who came home.",
    "The aqueduct is a promise written in gradient. Over kilometers the water falls the height of a hand, and the city drinks because the masons told the truth slowly.",
    "Grain stores are calendars. The depth of the bin in March predicts the temper of the council in June, and every ruler learns the arithmetic eventually.",
    "A path across the moor is maintained by feet alone. Nobody builds it, nobody closes it, and detouring around the bog is a knowledge the ground teaches gently at first.",
    "The bell founder mixes ore by ear as much as by weight. The mold is poured once, and the village will hear any impurity for three hundred years.",
    "Ferrymen keep the ledger of the crossing: which bank floods, which fog lies, which customers pay in autumn. The river runs through every column of the book.",
];

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn parse_usize_csv(s: &str) -> Vec<usize> {
    s.split(',').filter_map(|p| p.trim().parse().ok()).collect()
}

fn write_f32(path: &Path, v: &[f32]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for x in v {
        f.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn write_u32(path: &Path, v: &[u32]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for x in v {
        f.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn blake3_file(path: &Path) -> Result<String, String> {
    let mut hasher = blake3::Hasher::new();
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    std::io::copy(&mut f, &mut hasher).map_err(|e| format!("hash {}: {e}", path.display()))?;
    Ok(hasher.finalize().to_hex().to_string())
}

/// Decode one K row (position `p`, kv head `h`) out of the downloaded
/// cache words. f32 arm: direct slice. f16 arm: halves ride the words,
/// low half first (the Issue-753 packing).
fn k_row(
    host: &[f32],
    kv_f16: bool,
    p: usize,
    h: usize,
    hd: usize,
    kvd: usize,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.reserve(hd);
    for d in 0..hd {
        let e = p * kvd + h * hd + d;
        if kv_f16 {
            let w = host[e / 2].to_bits();
            let half = if e.is_multiple_of(2) {
                (w & 0xFFFF) as u16
            } else {
                (w >> 16) as u16
            };
            out.push(kv_f16_bits::f16_to_f32(half));
        } else {
            out.push(host[e]);
        }
    }
}

struct MarkDump {
    json: serde_json::Value,
}

/// Dump one length mark: K full bins (target layers × all KV heads), the
/// bounded commit subset (commit heads; strided + forced + Q positions,
/// with a positions index bin), returning the manifest section.
#[allow(clippy::too_many_arguments)]
fn dump_mark_k(
    fwd: &Qwen38DenseForward,
    cfg: &Qwen38DenseConfig,
    l: usize,
    k_layers: &[usize],
    commit_heads: &[usize],
    commit_rows: usize,
    q_positions: &[usize],
    full: &Path,
    commit: &Path,
) -> Result<MarkDump, String> {
    let hd = cfg.head_dim;
    let kvd = cfg.n_kv_head * hd;
    let kv_f16 = fwd.kv_f16;
    let words = if kv_f16 { l * kvd / 2 } else { l * kvd };

    // commit position set: uniform stride to the row cap + forced + Q
    let stride = l.div_ceil(commit_rows).max(1);
    let mut commit_pos: Vec<usize> = (0..l).step_by(stride).collect();
    for p in [0, l / 4, l / 2, (3 * l) / 4, l - 1]
        .into_iter()
        .chain(q_positions.iter().copied())
    {
        commit_pos.push(p.min(l - 1));
    }
    commit_pos.sort_unstable();
    commit_pos.dedup();

    let mut k_full_files: Vec<serde_json::Value> = Vec::new();
    let mut k_commit_files: Vec<serde_json::Value> = Vec::new();
    let mut sanity: Vec<serde_json::Value> = Vec::new();

    let mut attn_idx = 0usize;
    for (li, lt) in cfg.layer_types.iter().enumerate() {
        if *lt != Qwen38LayerType::Attention {
            continue;
        }
        if k_layers.contains(&li) {
            let host = fwd.download_k_cache(attn_idx, words)?;
            let mut row_buf: Vec<f32> = Vec::with_capacity(hd);
            for h in 0..cfg.n_kv_head {
                // ── full bin: every row [0..l) ──────────────────────────
                let mut rows: Vec<f32> = Vec::with_capacity(l * hd);
                for p in 0..l {
                    k_row(&host, kv_f16, p, h, hd, kvd, &mut row_buf);
                    rows.extend_from_slice(&row_buf);
                }
                let name = format!("k_layer{li}_head{h}_L{l}.f32");
                let path = full.join(&name);
                write_f32(&path, &rows).map_err(|e| format!("write {name}: {e}"))?;
                let digest = blake3_file(&path)?;
                let last_max = rows[l * hd - hd..].iter().fold(0f32, |m, x| m.max(x.abs()));
                let finite = rows.iter().all(|x| x.is_finite());
                if !finite || last_max == 0.0 {
                    return Err(format!(
                        "K sanity FAILED at layer {li} head {h} L={l}: last_row_max_abs={last_max} all_finite={finite}"
                    ));
                }
                if sanity.len() < 4 {
                    sanity.push(serde_json::json!({
                        "layer": li, "head": h,
                        "last_row_max_abs": last_max, "all_finite": finite,
                    }));
                }
                k_full_files.push(serde_json::json!({
                    "file": format!("full/{name}"), "layer": li, "head": h,
                    "rows": l, "row_len_f32": hd, "blake3": digest,
                }));

                // ── commit bin (bounded) + positions index ──────────────
                if commit_heads.contains(&h) {
                    let mut crows: Vec<f32> = Vec::with_capacity(commit_pos.len() * hd);
                    for &p in &commit_pos {
                        k_row(&host, kv_f16, p, h, hd, kvd, &mut row_buf);
                        crows.extend_from_slice(&row_buf);
                    }
                    let cname = format!("k_layer{li}_head{h}_L{l}.f32");
                    let cpath = commit.join(&cname);
                    write_f32(&cpath, &crows).map_err(|e| format!("write {cname}: {e}"))?;
                    let cdigest = blake3_file(&cpath)?;
                    k_commit_files.push(serde_json::json!({
                        "file": format!("commit/{cname}"), "layer": li, "head": h,
                        "rows": commit_pos.len(), "row_len_f32": hd, "blake3": cdigest,
                        "positions_file": format!("commit/positions_L{l}.u32"),
                    }));
                }
            }
        }
        attn_idx += 1;
    }
    // one positions index per length (identical across heads/layers by
    // construction — asserted by writing it once here)
    let ppath = commit.join(format!("positions_L{l}.u32"));
    write_u32(
        &ppath,
        &commit_pos.iter().map(|&p| p as u32).collect::<Vec<_>>(),
    )
    .map_err(|e| format!("write positions: {e}"))?;
    Ok(MarkDump {
        json: serde_json::json!({
            "L": l,
            "q_positions": q_positions,
            "k_full": k_full_files,
            "k_commit": k_commit_files,
            "commit": {
                "stride": stride,
                "rows": commit_pos.len(),
                "positions_file": format!("commit/positions_L{l}.u32"),
                "positions_blake3": blake3_file(&ppath)?,
                "forced_positions": [0, l / 4, l / 2, (3 * l) / 4, l - 1],
            },
            "k_sanity": sanity,
        }),
    })
}

#[allow(clippy::too_many_lines)]
fn run_capture(out: &Path) -> Result<(), String> {
    let t0 = Instant::now();
    let gguf_path = PathBuf::from(env_or(
        "QWEN38_GGUF",
        "F:/models/qwen38-27b-dbirks-Q4_K_M.gguf",
    ));
    let lengths = {
        let mut v = parse_usize_csv(&env_or("PYCAP_LENGTHS", "4096,16384,32768,65536"));
        v.retain(|&l| l > 0);
        v.sort_unstable();
        v.dedup();
        if v.is_empty() {
            return Err("PYCAP_LENGTHS produced no usable lengths".into());
        }
        v
    };
    let max_l = *lengths.last().expect("nonempty");
    let commit_rows: usize = env_or("PYCAP_COMMIT_ROWS", "4096")
        .parse()
        .map_err(|_| "PYCAP_COMMIT_ROWS must be an integer")?;
    let commit_heads = parse_usize_csv(&env_or("PYCAP_COMMIT_HEADS", "0,3"));
    if commit_heads.is_empty() {
        return Err("PYCAP_COMMIT_HEADS produced no heads".into());
    }

    // ── box state probe (GPU exclusivity: the riir-ai AGENTS rule) ────────
    let gpu_info = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader",
        ])
        .output()
        .map_err(|e| format!("nvidia-smi probe failed: {e}"))?;
    let gpu_csv = String::from_utf8_lossy(&gpu_info.stdout).trim().to_string();
    let apps = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ])
        .output()
        .map_err(|e| format!("nvidia-smi compute-apps probe failed: {e}"))?;
    let apps_csv = String::from_utf8_lossy(&apps.stdout).trim().to_string();
    // Windows desktop processes (DWM, explorer, ShellHost, …) hold WDDM
    // contexts that nvidia-smi lists with `[N/A]` memory — the AGENTS GPU
    // rule targets COMPUTE consumers, and a real one reports a parseable
    // dedicated-memory figure. Refuse only those; record the raw list.
    let compute_consumers: Vec<&str> = apps_csv
        .lines()
        .filter(|line| {
            let mem = line.rsplit(',').next().unwrap_or("").trim();
            !mem.is_empty()
                && mem != "[N/A]"
                && mem
                    .trim_end_matches(" MiB")
                    .parse::<u64>()
                    .is_ok_and(|m| m > 0)
        })
        .collect();
    if !compute_consumers.is_empty() {
        return Err(format!(
            "GPU-EXCLUSIVITY refusal: co-resident compute apps:\n{}",
            compute_consumers.join("\n")
        ));
    }
    eprintln!("[pycap] box: {gpu_csv}");
    eprintln!("[pycap] compute apps: none with dedicated memory (raw list: {apps_csv})");

    // ── env snapshot (posture is recorded, never assumed) ─────────────────
    let mut env_snapshot: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| {
            k.starts_with("QWEN38") || k.starts_with("PYCAP") || k == "CUDA_VISIBLE_DEVICES"
        })
        .collect();
    env_snapshot.sort();

    // ── model + tokenizer ──────────────────────────────────────────────────
    eprintln!("[pycap] hashing model {} (one-time)…", gguf_path.display());
    let model_blake3 = blake3_file(&gguf_path)?;
    let model_bytes = gguf_path
        .metadata()
        .map_err(|e| format!("model metadata: {e}"))?
        .len();
    let gguf = GgufFile::open(&gguf_path).map_err(|e| format!("open gguf: {e}"))?;
    let cfg = Qwen38DenseConfig::from_gguf(&gguf)?;
    let tok = BpeTokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer: {e}"))?;
    let fa_layers: Vec<usize> = cfg
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == Qwen38LayerType::Attention)
        .map(|(i, _)| i)
        .collect();
    if fa_layers.len() < 2 {
        return Err(format!(
            "only {} FA layers — nothing to capture",
            fa_layers.len()
        ));
    }
    let k_layers: Vec<usize> = match env_or("PYCAP_K_LAYERS", "first,last").as_str() {
        "first,last" => vec![*fa_layers.first().unwrap(), *fa_layers.last().unwrap()],
        raw => parse_usize_csv(raw)
            .into_iter()
            .filter(|li| cfg.layer_types.get(*li) == Some(&Qwen38LayerType::Attention))
            .collect(),
    };
    if k_layers.is_empty() {
        return Err("PYCAP_K_LAYERS resolved to no FA layers".into());
    }
    eprintln!(
        "[pycap] cfg: n_layer={} n_head={} n_kv_head={} head_dim={} fa_layers={} k_layers={}",
        cfg.n_layer,
        cfg.n_head,
        cfg.n_kv_head,
        cfg.head_dim,
        fa_layers.len(),
        k_layers
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    // ── deterministic prompt stream (nested prefixes of ONE stream) ───────
    let mut tokens: Vec<u32> = Vec::with_capacity(max_l + 1024);
    let mut prompt_text = String::new();
    let mut passes = 0usize;
    while tokens.len() <= max_l {
        for (i, para) in PARAGRAPHS.iter().enumerate() {
            let piece = format!("Pass {passes} paragraph {i}. {para} ");
            prompt_text.push_str(&piece);
            tokens.extend(tok.encode(&piece).into_iter().map(|t| t as u32));
        }
        passes += 1;
    }
    tokens.truncate(max_l);
    if tokens.len() < max_l {
        return Err(format!("prompt bank produced only {} tokens", tokens.len()));
    }
    eprintln!("[pycap] prompt: {max_l} tokens from {passes} passes");

    // ── sampled positions per length; Q taps at the union ────────────────
    let q_positions_of: Vec<(usize, Vec<usize>)> = lengths
        .iter()
        .map(|&l| {
            let mut ps = vec![0usize, l / 4, l / 2, (3 * l) / 4, l - 1];
            ps.sort_unstable();
            ps.dedup();
            (l, ps)
        })
        .collect();
    let tap_union: Vec<usize> = {
        let mut u: Vec<usize> = q_positions_of
            .iter()
            .flat_map(|(_, ps)| ps.iter().copied())
            .collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let tap_pos_set: HashSet<usize> = tap_union.iter().copied().collect();

    // ── output tree ────────────────────────────────────────────────────────
    let full = out.join("full");
    let commit = out.join("commit");
    std::fs::create_dir_all(&full).map_err(|e| format!("mkdir: {e}"))?;
    std::fs::create_dir_all(&commit).map_err(|e| format!("mkdir: {e}"))?;
    std::fs::write(full.join("prompt.txt"), &prompt_text)
        .map_err(|e| format!("write prompt: {e}"))?;
    write_u32(&full.join("tokens.u32"), &tokens).map_err(|e| format!("write tokens: {e}"))?;
    let prompt_blake3 = {
        let mut h = blake3::Hasher::new();
        h.update(prompt_text.as_bytes());
        h.finalize().to_hex().to_string()
    };
    let tokens_blake3 = {
        let mut h = blake3::Hasher::new();
        for t in &tokens {
            h.update(&t.to_le_bytes());
        }
        h.finalize().to_hex().to_string()
    };

    // ── forward: eager nested-prefix fill, Q taps at the union ────────────
    let ctx_len: usize = env_or("PYCAP_CTX", &max_l.to_string())
        .parse()
        .map_err(|_| "PYCAP_CTX must be an integer")?;
    if ctx_len < max_l {
        return Err(format!("PYCAP_CTX {ctx_len} < max length {max_l}"));
    }
    eprintln!("[pycap] loading model + allocating state (ctx {ctx_len})…");
    let mut fwd = Qwen38DenseForward::new(&gguf_path, ctx_len)?;
    let kv_f16 = fwd.kv_f16;
    eprintln!(
        "[pycap] lane up ({:.0}s); kv_dtype={}",
        t0.elapsed().as_secs_f32(),
        if kv_f16 { "f16" } else { "f32" }
    );

    let qd = cfg.n_head * cfg.head_dim;
    let mut q_rows: BTreeMap<usize, Vec<Vec<f32>>> = BTreeMap::new();

    let mut marks: Vec<serde_json::Value> = Vec::new();
    let mut mark_iter = lengths.iter().peekable();

    let fill_start = Instant::now();
    for (pos, &token) in tokens.iter().enumerate() {
        if tap_pos_set.contains(&pos) {
            let mut bufs: Vec<Vec<f32>> = fa_layers.iter().map(|_| vec![0f32; qd]).collect();
            fwd.forward_token_attn_q_capture(token, pos, &fa_layers, &mut bufs)?;
            q_rows.insert(pos, bufs);
            eprintln!(
                "[pycap] pos {pos}/{max_l} q-tap ({:.0}s)",
                fill_start.elapsed().as_secs_f32()
            );
        } else {
            fwd.forward_token(token, pos)?;
        }
        if let Some(&&l) = mark_iter.peek()
            && pos + 1 == l
        {
            mark_iter.next();
            let q_positions = &q_positions_of
                .iter()
                .find(|(pl, _)| *pl == l)
                .map(|(_, ps)| ps.clone())
                .unwrap_or_default();
            let dump = dump_mark_k(
                &fwd,
                &cfg,
                l,
                &k_layers,
                &commit_heads,
                commit_rows,
                q_positions,
                &full,
                &commit,
            )?;
            marks.push(dump.json);
            eprintln!(
                "[pycap] mark L={l} dumped ({:.0}s elapsed)",
                fill_start.elapsed().as_secs_f32()
            );
        }
        if pos % 8192 == 0 && pos > 0 {
            eprintln!(
                "[pycap] pos {pos}/{max_l} ({:.1} tok/s)",
                (pos + 1) as f32 / fill_start.elapsed().as_secs_f32()
            );
        }
    }

    // ── Q bins: full = every FA layer × every tap position; commit = the
    //    K target layers × every tap position (small — commit them all) ──
    let mut q_full_files: Vec<serde_json::Value> = Vec::new();
    let mut q_commit_files: Vec<serde_json::Value> = Vec::new();
    for (&p, rows) in &q_rows {
        for (fi, &li) in fa_layers.iter().enumerate() {
            let name = format!("q_layer{li}_pos{p}.f32");
            let path = full.join(&name);
            write_f32(&path, &rows[fi]).map_err(|e| format!("write {name}: {e}"))?;
            q_full_files.push(serde_json::json!({
                "file": format!("full/{name}"), "layer": li, "position": p,
                "heads": cfg.n_head, "row_len_f32": cfg.head_dim,
                "blake3": blake3_file(&path)?,
            }));
            if k_layers.contains(&li) {
                let cname = format!("q_layer{li}_pos{p}.f32");
                let cpath = commit.join(&cname);
                write_f32(&cpath, &rows[fi]).map_err(|e| format!("write {cname}: {e}"))?;
                q_commit_files.push(serde_json::json!({
                    "file": format!("commit/{cname}"), "layer": li, "position": p,
                    "heads": cfg.n_head, "row_len_f32": cfg.head_dim,
                    "blake3": blake3_file(&cpath)?,
                }));
            }
        }
    }

    // ── manifest ───────────────────────────────────────────────────────────
    let manifest = serde_json::json!({
        "schema": "pyramid_612_capture_v1",
        "created_unix": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "capture_bin_commit": env_or("PYCAP_GIT_SHA", "unknown"),
        "model": {
            "path": gguf_path.display().to_string(),
            "file_bytes": model_bytes,
            "blake3": model_blake3,
            "arch": gguf.architecture().unwrap_or("unknown"),
        },
        "config": {
            "n_embd": cfg.n_embd, "n_layer": cfg.n_layer, "n_head": cfg.n_head,
            "n_kv_head": cfg.n_kv_head, "head_dim": cfg.head_dim,
            "rotary_dim": cfg.rotary_dim, "rope_theta": cfg.rope_theta,
            "vocab_size": cfg.vocab_size,
            "fa_layers": fa_layers,
            "k_layers": k_layers,
            "commit_heads": commit_heads,
        },
        "posture": {
            "fill": "eager forward_token / forward_token_attn_q_capture, nested-prefix single pass",
            "kv_dtype": if kv_f16 { "f16" } else { "f32" },
            "scale_note": "q/k dumped UNSCALED post-RoPE; the attention kernels apply 1/sqrt(head_dim) internally — a positive monotone constant for selection ranking",
            "q_tap": "self.q_normed after forward_attn_layer (the exact buffer the decode attention scores with)",
        },
        "prompt": {
            "blake3_text_full_bank": prompt_blake3,
            "blake3_token_stream": tokens_blake3,
            "tokens_total": tokens.len(),
            "first16": tokens.iter().take(16).collect::<Vec<_>>(),
            "last16": tokens.iter().rev().take(16).rev().collect::<Vec<_>>(),
            "generator": "deterministic 24-paragraph bank cycled with pass/paragraph markers; per-piece BPE concatenation; truncated to max length",
        },
        "box": {
            "gpu_csv": gpu_csv,
            "compute_apps_at_start": apps_csv,
            "exclusivity": "co-resident compute probe REFUSES on any app; empty at start",
            "env": env_snapshot,
        },
        "lengths": marks,
        "q_full": q_full_files,
        "q_commit": q_commit_files,
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    std::fs::write(out.join("manifest.json"), &manifest_bytes)
        .map_err(|e| format!("write manifest: {e}"))?;
    eprintln!(
        "[pycap] DONE in {:.0}s → {}",
        t0.elapsed().as_secs_f32(),
        out.display()
    );
    Ok(())
}

fn main() {
    let out = PathBuf::from(env_or("PYCAP_OUT", ".pycap"));
    if let Err(e) = run_capture(&out) {
        eprintln!("[pycap] FATAL: {e}");
        std::process::exit(1);
    }
}
