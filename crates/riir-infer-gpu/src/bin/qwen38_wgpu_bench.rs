//! Issue 1150 — the Metal dense-qwen3.8 bench + parity harness.
//!
//! Modes:
//! - `--pp 512,2048,...`   prefill cells (median of 3, warm-up first) — the
//!   four-way table row (vs uzu / llama.cpp / MLX on the same GGUF family).
//! - `--decode 32`         decode steps after the prompt (tok/s).
//! - `--decode-ctx 2048`   decode steps after a prompt PADDED to N tokens
//!   (`--decode N` supplies the step count) — the league's decode-at-context
//!   convention (ctx512/ctx2048 cells), not the raw short-prompt shape.
//! - `--walk <text>`       greedy-generate N tokens (`--walk-tokens`), print
//!   the token ids + detokenized text — the ids feed the llama.cpp parity
//!   check on the same GGUF (the G1 acceptance; f32-activation class, module doc).
//! - `--ablate-matrix`     issue 1174: the free-running attribution table —
//!   one model load, per-class interleaved base/ablated pass pairs (no
//!   serialization anywhere; `--ablate-classes` narrows, `--ablate-pairs`
//!   sizes the pair count, default 3). See `qwen38_ablate`.
//! - `--decode-ablate-matrix` issue 1178 T1: the decode twin — the same
//!   free-running pattern over `forward_token` (the decode frame census:
//!   attn/GDN/FFN/embed/small-stages/dispatch+sync shares per step).
//!   Needs `--decode N` for the step count; `--decode-ctx N` runs the
//!   matrix at context depth; `--decode-ablate` narrows the class csv.
//!   See `qwen38_ablate::DECODE_CLASSES`.
//!
//! Whole-file cfg keeps non-`qwen38_wgpu` builds an empty binary; the Cargo
//! [[bin]] required-features row is the reader protection.

#![cfg(feature = "qwen38_wgpu")]

use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuForward;

const DEFAULT_GGUF: &str = "/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut gguf = DEFAULT_GGUF.to_string();
    let mut pp: Vec<usize> = vec![];
    let mut decode_steps = 0usize;
    let mut decode_ctx: Option<usize> = None;
    let mut prompt = "The harbor town woke slowly under a grey sky,".to_string();
    let mut ctx = 24_576usize;
    let mut chunk = 512usize;
    let mut walk_tokens = 0usize;
    let mut passes = 3usize;
    let mut ablate_matrix = false;
    let mut ablate_classes: Option<String> = None;
    let mut ablate_pairs = 3usize;
    let mut decode_ablate_matrix = false;
    let mut decode_ablate_classes: Option<String> = None;
    let mut decode_ablate_pairs = 3usize;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                i += 1;
                gguf = args[i].clone();
            }
            "--pp" => {
                i += 1;
                pp = args[i]
                    .split(',')
                    .map(|s| s.parse().expect("pp len"))
                    .collect();
            }
            "--decode" => {
                i += 1;
                decode_steps = args[i].parse().expect("decode n");
            }
            "--decode-ctx" => {
                i += 1;
                decode_ctx = Some(args[i].parse().expect("decode-ctx n"));
            }
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--ctx" => {
                i += 1;
                ctx = args[i].parse().expect("ctx");
            }
            "--chunk" => {
                i += 1;
                chunk = args[i].parse().expect("chunk");
            }
            "--walk" => walk_tokens = 64,
            "--walk-tokens" => {
                i += 1;
                walk_tokens = args[i].parse().expect("walk n");
            }
            "--passes" => {
                i += 1;
                passes = args[i].parse().expect("passes");
            }
            "--ablate-matrix" => ablate_matrix = true,
            "--ablate-classes" => {
                i += 1;
                ablate_classes = Some(args[i].clone());
            }
            "--ablate-pairs" => {
                i += 1;
                ablate_pairs = args[i].parse().expect("ablate pairs");
            }
            "--decode-ablate-matrix" => decode_ablate_matrix = true,
            "--decode-ablate" => {
                i += 1;
                decode_ablate_classes = Some(args[i].clone());
            }
            "--decode-ablate-pairs" => {
                i += 1;
                decode_ablate_pairs = args[i].parse().expect("decode-ablate pairs");
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    println!("[bench] loading {gguf} (ctx {ctx}, chunk {chunk}) ...");
    let t0 = std::time::Instant::now();
    let mut fwd = Qwen38WgpuForward::new(std::path::Path::new(&gguf), ctx, chunk)
        .unwrap_or_else(|e| panic!("[bench] load failed: {e}"));
    println!("[bench] loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let gg = riir_infer_core::gguf_loader::GgufFile::open(std::path::Path::new(&gguf))
        .unwrap_or_else(|e| panic!("[bench] reopen: {e}"));
    let tok = BpeTokenizer::from_gguf(&gg).unwrap_or_else(|e| panic!("[bench] tokenizer: {e}"));
    let ids: Vec<u32> = tok.encode(&prompt).into_iter().map(|t| t as u32).collect();
    println!(
        "[bench] prompt: {} chars -> {} tokens",
        prompt.len(),
        ids.len()
    );

    if ablate_matrix {
        let pp_target = pp.first().copied().unwrap_or(5120);
        run_ablate_matrix(
            &mut fwd,
            &ids,
            chunk,
            pp_target,
            ablate_classes,
            ablate_pairs,
        );
        return;
    }

    if decode_ablate_matrix {
        if decode_steps == 0 {
            eprintln!("[decode-ablate-matrix] needs --decode N for the step count");
            std::process::exit(2);
        }
        if ablate_matrix || ablate_classes.is_some() {
            eprintln!("[decode-ablate-matrix] refuses --ablate* flags: one scope per run");
            std::process::exit(2);
        }
        run_decode_ablate_matrix(
            &mut fwd,
            &ids,
            decode_ctx,
            decode_steps,
            decode_ablate_classes,
            decode_ablate_pairs,
        );
        return;
    }

    // Prefill cells: pad/truncate the prompt to each target by repetition.
    // Issue 1150: the target rounds UP to a full ingest chunk — a ragged
    // tail runs the remainder through SEQUENTIAL decode (measured 40–42% of
    // a pp 2484 pass: the cell was not pure prefill, and the opponents'
    // cells are). The padded count is disclosed on every line.
    for &target in &pp {
        let padded = target.div_ceil(chunk) * chunk;
        let mut toks = ids.clone();
        while toks.len() < padded {
            toks.extend_from_slice(&ids);
        }
        toks.truncate(padded);
        let mut readings = Vec::new();
        for pass in 0..passes + 1 {
            fwd.zero_state();
            let t = std::time::Instant::now();
            fwd.ingest(&toks, 0)
                .unwrap_or_else(|e| panic!("ingest: {e}"));
            let dt = t.elapsed().as_secs_f64();
            let tps = toks.len() as f64 / dt;
            if pass == 0 {
                println!("[pp {target}→{padded}] warm-up: {dt:.2}s = {tps:.0} tok/s");
                continue;
            }
            readings.push(tps);
            println!("[pp {target}→{padded}] pass {pass}: {dt:.2}s = {tps:.0} tok/s");
        }
        readings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = readings[readings.len() / 2];
        println!("[pp {target}→{padded}] MEDIAN {med:.0} tok/s (padded {padded}, pure prefill)");
    }

    // Decode cell: prefill the prompt then step.
    if decode_steps > 0 {
        fwd.zero_state();
        fwd.ingest(&ids, 0)
            .unwrap_or_else(|e| panic!("ingest: {e}"));
        let mut next = 0u32;
        let t = std::time::Instant::now();
        for s in 0..decode_steps {
            let pos = ids.len() + s;
            next = fwd
                .forward_token(if s == 0 { *ids.last().unwrap() } else { next }, pos)
                .unwrap_or_else(|e| panic!("decode: {e}"));
        }
        let dt = t.elapsed().as_secs_f64();
        println!(
            "[decode] {decode_steps} steps in {dt:.2}s = {:.2} tok/s",
            decode_steps as f64 / dt
        );
    }

    // Decode cell at context depth (the league's decode convention): pad the
    // prompt to N tokens, prefill, then step — tok/s is decode-at-ctx-N, the
    // shape the league's ctx512/ctx2048 decode cells measure.
    if let (Some(ctx_n), &steps) = (decode_ctx, &decode_steps) {
        if steps == 0 {
            eprintln!("[decode-ctx] needs --decode N for the step count");
            std::process::exit(2);
        }
        let mut toks = ids.clone();
        while toks.len() < ctx_n {
            toks.extend_from_slice(&ids);
        }
        toks.truncate(ctx_n);
        fwd.zero_state();
        fwd.ingest(&toks, 0)
            .unwrap_or_else(|e| panic!("decode-ctx ingest: {e}"));
        let mut next = 0u32;
        let t = std::time::Instant::now();
        for s in 0..steps {
            let pos = toks.len() + s;
            next = fwd
                .forward_token(if s == 0 { *toks.last().unwrap() } else { next }, pos)
                .unwrap_or_else(|e| panic!("decode-ctx: {e}"));
        }
        let dt = t.elapsed().as_secs_f64();
        println!(
            "[decode-ctx {ctx_n}] {steps} steps in {dt:.2}s = {:.2} tok/s",
            steps as f64 / dt
        );
    }

    // Greedy walk — the parity artifact.
    if walk_tokens > 0 {
        fwd.zero_state();
        fwd.ingest(&ids, 0)
            .unwrap_or_else(|e| panic!("ingest: {e}"));
        let mut next = 0u32;
        let mut out_ids: Vec<u32> = vec![];
        for s in 0..walk_tokens {
            let pos = ids.len() + s;
            next = fwd
                .forward_token(if s == 0 { *ids.last().unwrap() } else { next }, pos)
                .unwrap_or_else(|e| panic!("walk: {e}"));
            out_ids.push(next);
        }
        println!("[walk] ids: {:?}", out_ids);
        let text: Vec<String> = out_ids
            .iter()
            .map(|&id| tok.decode(&[id as usize]))
            .collect();
        println!("[walk] text: {}", text.join(""));
    }
}

/// Issue 1174 — the free-running attribution matrix. One model load; per
/// class an INTERLEAVED (base, ablated) pass-pair sequence — sequential
/// arms measure the box (the katgpt-rs ab_timing law), so every cell
/// carries its own base arms and the delta is the median of the per-pair
/// walls. `all` runs last: its delta is the pass's launch/dispatch floor.
fn run_ablate_matrix(
    fwd: &mut Qwen38WgpuForward,
    ids: &[u32],
    chunk: usize,
    pp_target: usize,
    classes: Option<String>,
    pairs: usize,
) {
    use riir_infer_gpu::qwen38_ablate::{Ablate, CLASSES};

    // The cell list: every class, or the named subset (`all` cell always
    // runs last — it is the floor cross-check).
    let names: Vec<&str> = match &classes {
        Some(spec) => {
            let a = Ablate::parse(spec).unwrap_or_else(|e| panic!("ablate: {e}"));
            CLASSES.iter().copied().filter(|c| a.skip(c)).collect()
        }
        None => CLASSES.to_vec(),
    };

    let n_gdn = fwd
        .config
        .layer_types
        .iter()
        .filter(|k| **k == riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuLayerType::Deltanet)
        .count();
    let n_attn = fwd.config.layer_types.len() - n_gdn;

    // Pad to the pp target in full chunks (the pp convention — no ragged
    // tail; the padded count is disclosed).
    let padded = pp_target.div_ceil(chunk) * chunk;
    let mut toks = ids.to_vec();
    while toks.len() < padded {
        toks.extend_from_slice(ids);
    }
    toks.truncate(padded);
    let n_chunks = padded / chunk;

    println!(
        "[ablate-matrix] {} classes x {pairs} interleaved pair(s), pp{padded} \
         ({n_chunks} chunks of {chunk}), layers: {n_gdn} gdn + {n_attn} attn — \
         one warm-up pass first",
        names.len()
    );

    // Warm-up: every kernel of the posture pays its first-dispatch cost
    // once, outside any cell (skipping never introduces a new kernel).
    fwd.zero_state();
    let t = std::time::Instant::now();
    fwd.ingest(&toks, 0)
        .unwrap_or_else(|e| panic!("ingest: {e}"));
    println!(
        "[ablate-matrix] warm-up: {:.2}s = {:.0} tok/s",
        t.elapsed().as_secs_f64(),
        toks.len() as f64 / t.elapsed().as_secs_f64()
    );

    let mut base_walls: Vec<f64> = Vec::new();
    let mut rows: Vec<(&str, f64)> = Vec::new(); // (class, Δ ms/chunk)
    for &name in &names {
        let a = Ablate::from_classes([name]).unwrap_or_else(|e| panic!("ablate: {e}"));
        run_ablate_cell(
            fwd,
            name,
            Some(a),
            &toks,
            n_chunks,
            pairs,
            &mut base_walls,
            &mut rows,
        );
    }
    // The launch-floor cell.
    run_ablate_cell(
        fwd,
        "ALL",
        Some(Ablate::all()),
        &toks,
        n_chunks,
        pairs,
        &mut base_walls,
        &mut rows,
    );
    fwd.set_ablate(None);

    // The table. The ALL row is the launch floor, not an additive class —
    // the Σ line must never include it (945's first run printed a 196%
    // Σ that was exactly this bug; the classes-only Σ is the wall share).
    println!(
        "[ablate-matrix] ── free-running per-class costs (ms per chunk, median of {pairs} pair deltas) ──"
    );
    let mut kernel_ms = 0.0f64;
    let mut floor_ms = 0.0f64;
    for (name, delta) in &rows {
        println!("[ablate-matrix]   {name:<16} {delta:8.2} ms/chunk");
        if *name == "ALL" {
            floor_ms = *delta;
        } else {
            kernel_ms += delta;
        }
    }
    let mut w = base_walls.clone();
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let base_med = w[w.len() / 2];
    let base_ms = base_med * 1e3 / n_chunks as f64;
    println!(
        "[ablate-matrix] base wall median {:.1} ms/chunk ({:.0} tok/s) | Σ class deltas \
         {kernel_ms:.1} ms/chunk ({:.1}% of the wall — near-additive attribution above \
         ~90%) | ALL floor cell {floor_ms:.1} ms/chunk (dispatch + per-chunk sync/reads, \
         never kernel execution) | wall − Σ = the unattributed residue",
        base_ms,
        padded as f64 / base_med,
        100.0 * kernel_ms / base_ms
    );
    // The per-layer mapping (the 943 comparison): GDN-only classes divide
    // by the GDN layer count, attn classes by the attn count, shared
    // classes (entry norm / mlp / casts — identical shapes on both kinds)
    // by the layer total; the once-per-chunk classes do not divide.
    let gdn_only = [
        "gemm_qkv_z_ab",
        "beta_decay",
        "conv1d",
        "expand_l2",
        "scan",
        "norm_ssm",
        "z_gate",
        "ssm_out_gemm",
        "residual_gdn",
    ];
    let all_layers = n_gdn + n_attn;
    println!(
        "[ablate-matrix] per-layer mapping (gdn /{n_gdn}, attn /{n_attn}, shared /{all_layers}):"
    );
    for (name, delta) in &rows {
        // ALL is the floor cell, and wte/tail run once per chunk — neither
        // divides by a layer count.
        let per_layer = if *name == "ALL"
            || *name == "wte_upload"
            || *name == "tail_norm"
            || *name == "lm_head_gemm"
        {
            None
        } else if gdn_only.contains(name) {
            Some(*delta / n_gdn as f64)
        } else if name.starts_with("attn_") {
            Some(*delta / n_attn as f64)
        } else {
            Some(*delta / all_layers as f64)
        };
        if let Some(ms) = per_layer {
            println!("[ablate-matrix]   {name:<16} {ms:7.3} ms/layer-instance");
        }
    }
    println!(
        "[ablate-matrix] done — outputs were bit-trashed by design; do not reuse the model state"
    );
}

/// One matrix cell: `pairs` interleaved (base, ablated) pass pairs — the
/// base arm of every pair re-reads the incumbent launch set under the SAME
/// box state, so the paired delta cancels drift (sequential arms measure
/// the box, never the kernel).
#[allow(clippy::too_many_arguments, reason = "the matrix cell's flat inputs")]
fn run_ablate_cell(
    fwd: &mut Qwen38WgpuForward,
    name: &str,
    ablated: Option<riir_infer_gpu::qwen38_ablate::Ablate>,
    toks: &[u32],
    n_chunks: usize,
    pairs: usize,
    base_walls: &mut Vec<f64>,
    rows: &mut Vec<(&'static str, f64)>,
) {
    let mut deltas: Vec<f64> = Vec::new();
    for pair in 0..pairs {
        fwd.zero_state();
        fwd.set_ablate(None);
        let t0 = std::time::Instant::now();
        fwd.ingest(toks, 0)
            .unwrap_or_else(|e| panic!("ingest: {e}"));
        let base = t0.elapsed().as_secs_f64();
        base_walls.push(base);

        fwd.zero_state();
        fwd.set_ablate(ablated.clone());
        let t1 = std::time::Instant::now();
        fwd.ingest(toks, 0)
            .unwrap_or_else(|e| panic!("ingest: {e}"));
        let abl = t1.elapsed().as_secs_f64();
        fwd.set_ablate(None);

        let d = (base - abl) * 1e3 / n_chunks as f64;
        deltas.push(d);
        println!(
            "[ablate {name}] pair {pair}: base {:.2}s ({:.0} tok/s) / \
             abl {:.2}s ({:.0} tok/s) -> Δ {d:+.2} ms/chunk",
            base,
            toks.len() as f64 / base,
            abl,
            toks.len() as f64 / abl
        );
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = deltas[deltas.len() / 2];
    let class: &'static str = riir_infer_gpu::qwen38_ablate::CLASSES
        .iter()
        .copied()
        .find(|c| *c == name)
        .unwrap_or("ALL");
    rows.push((class, med));
}

/// Issue 1178 T1 — the decode frame census: the free-running attribution
/// matrix over `forward_token`. One model load; per class an INTERLEAVED
/// (base, ablated) decode-cell pair sequence — the base arm of every pair
/// re-runs the incumbent launch set under the same box state, so the
/// paired Δ cancels drift. Each arm is a fresh state + untimed prompt
/// ingest, then `steps` timed decode steps. The per-class Δwall is the
/// class's free-running per-step cost; the ALL cell's Δ is the launch
/// floor and its residual wall (base − Δ) is the sync/host floor (the
/// per-token logits read + host argmax the decode loop's value dependency
/// keeps — not instrumented).
#[allow(clippy::too_many_arguments, reason = "the matrix driver's flat inputs")]
fn run_decode_ablate_matrix(
    fwd: &mut Qwen38WgpuForward,
    ids: &[u32],
    ctx_target: Option<usize>,
    steps: usize,
    classes: Option<String>,
    pairs: usize,
) {
    use riir_infer_gpu::qwen38_ablate::{Ablate, DECODE_CLASSES};

    // The cell list: every decode class, or the named subset (`all` always
    // runs last — it is the floor cross-check).
    let names: Vec<&str> = match &classes {
        Some(spec) => {
            let a = Ablate::parse_decode(spec).unwrap_or_else(|e| panic!("decode-ablate: {e}"));
            DECODE_CLASSES
                .iter()
                .copied()
                .filter(|c| a.skip(c))
                .collect()
        }
        None => DECODE_CLASSES.to_vec(),
    };

    // The seed tokens: the raw prompt, or padded to the ctx target (the
    // decode-ctx convention — the scores class reads pos+1 KV rows, so the
    // census splits the ctx-dependence by running at both depths).
    let toks: Vec<u32> = match ctx_target {
        Some(n) => {
            let mut t = ids.to_vec();
            while t.len() < n {
                t.extend_from_slice(ids);
            }
            t.truncate(n);
            t
        }
        None => ids.to_vec(),
    };
    let seed = toks.len();

    let n_gdn = fwd
        .config
        .layer_types
        .iter()
        .filter(|k| **k == riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuLayerType::Deltanet)
        .count();
    let n_attn = fwd.config.layer_types.len() - n_gdn;

    println!(
        "[decode-ablate-matrix] {} classes x {pairs} pair(s), seed {seed} tokens \
         (ctx target {:?}), {steps} steps/cell, layers: {n_gdn} gdn + {n_attn} attn",
        names.len(),
        ctx_target
    );

    // Warm-up: one full decode run — first-dispatch costs pay outside any
    // cell (skipping never introduces a new kernel).
    fwd.zero_state();
    fwd.ingest(&toks, 0)
        .unwrap_or_else(|e| panic!("ingest: {e}"));
    let t = std::time::Instant::now();
    let mut next = *toks.last().unwrap();
    for s in 0..steps {
        next = fwd
            .forward_token(next, seed + s)
            .unwrap_or_else(|e| panic!("decode: {e}"));
    }
    {
        let dt = t.elapsed().as_secs_f64();
        println!(
            "[decode-ablate-matrix] warm-up: {dt:.2}s = {:.2} tok/s",
            steps as f64 / dt
        );
    }

    let mut base_walls: Vec<f64> = Vec::new();
    let mut base_ids: Option<Vec<u32>> = None;
    let mut rows: Vec<(&'static str, f64)> = Vec::new(); // (class, Δ ms/step)
    for &name in &names {
        let a =
            Ablate::from_classes_decode([name]).unwrap_or_else(|e| panic!("decode-ablate: {e}"));
        run_decode_ablate_cell(
            fwd,
            name,
            Some(a),
            &toks,
            steps,
            pairs,
            &mut base_walls,
            &mut base_ids,
            &mut rows,
        );
    }
    // The launch-floor cell.
    run_decode_ablate_cell(
        fwd,
        "ALL",
        Some(Ablate::all_decode()),
        &toks,
        steps,
        pairs,
        &mut base_walls,
        &mut base_ids,
        &mut rows,
    );
    fwd.set_ablate(None);

    // The table. The ALL row is the launch floor, not an additive class —
    // the Σ line must never include it (the 945 law).
    println!(
        "[decode-ablate-matrix] ── free-running per-class costs (ms per decode step, median of {pairs} pair deltas) ──"
    );
    let mut class_ms = 0.0f64;
    let mut floor_ms = 0.0f64;
    for (name, delta) in &rows {
        println!("[decode-ablate-matrix]   {name:<14} {delta:8.3} ms/step");
        if *name == "ALL" {
            floor_ms = *delta;
        } else {
            class_ms += delta;
        }
    }
    let mut w = base_walls.clone();
    w.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let base_med = w[w.len() / 2];
    let base_ms = base_med * 1e3;
    println!(
        "[decode-ablate-matrix] base {:.2} ms/step ({:.2} tok/s) | Σ classes {class_ms:.2} \
         ({:.1}% of the wall — near-additive attribution above ~90%) | ALL floor Δ \
         {floor_ms:.2} ms/step | ALL-cell residual wall {:.2} ms/step (logits read + \
         host argmax + loop — the sync/host floor)",
        base_ms,
        1e3 / base_med,
        100.0 * class_ms / base_ms,
        base_ms - floor_ms,
    );

    // Coarse roll-up (the 1178 T1 buckets) as shares of the base wall.
    let get = |n: &str| {
        rows.iter()
            .find(|(k, _)| *k == n)
            .map(|(_, v)| *v)
            .unwrap_or(0.0)
    };
    let attn = get("d_attn_gemm") + get("d_attn_stage") + get("d_attn_scores");
    let gdn = get("d_gdn_gemm") + get("d_gdn_stage") + get("d_gdn_scan") + get("d_gdn_post");
    let ffn = get("d_ffn");
    let embed = get("d_embed");
    let small = get("d_entry_norm") + get("d_residual") + get("d_out_norm");
    let lm = get("d_lm_head");
    println!("[decode-ablate-matrix] ── coarse shares (of the {base_ms:.2} ms/step base wall) ──");
    for (k, v) in [
        ("attn", attn),
        ("gdn", gdn),
        ("ffn", ffn),
        ("embed", embed),
        ("small-stages", small),
        ("lm_head", lm),
    ] {
        println!(
            "[decode-ablate-matrix]   {k:<14} {v:8.3} ms/step  {:5.1}%",
            100.0 * v / base_ms
        );
    }
    println!(
        "[decode-ablate-matrix] residue (wall − Σ classes): {:.2} ms/step ({:.1}%) — sync + \
         host-loop mass; the ALL floor cross-checks it",
        base_ms - class_ms,
        100.0 * (base_ms - class_ms) / base_ms
    );

    // Per-layer mapping (the 943 comparison): GDN classes divide by the
    // GDN count, attn classes by the attn count, shared classes (entry
    // norm / ffn / residual — identical shapes on both kinds) by the layer
    // total; the once-per-step classes do not divide.
    let all_layers = n_gdn + n_attn;
    println!(
        "[decode-ablate-matrix] per-layer mapping (gdn /{n_gdn}, attn /{n_attn}, shared /{all_layers}):"
    );
    for (name, delta) in &rows {
        let per_layer = if *name == "ALL"
            || *name == "d_embed"
            || *name == "d_out_norm"
            || *name == "d_lm_head"
        {
            None
        } else if name.starts_with("d_gdn_") {
            Some(*delta / n_gdn as f64)
        } else if name.starts_with("d_attn_") {
            Some(*delta / n_attn as f64)
        } else {
            Some(*delta / all_layers as f64)
        };
        if let Some(ms) = per_layer {
            println!("[decode-ablate-matrix]   {name:<14} {ms:7.4} ms/layer-instance");
        }
    }
    println!(
        "[decode-ablate-matrix] done — outputs were bit-trashed by design; do not reuse the model state"
    );
}

/// One decode matrix cell: `pairs` interleaved (base, ablated) decode runs
/// — fresh state + untimed ingest, then `steps` timed forward_token steps.
/// The base arm's first token ids form a drift canary (deterministic
/// kernels must reproduce them across pairs; a divergence names the box,
/// not the class).
#[allow(clippy::too_many_arguments, reason = "the matrix cell's flat inputs")]
fn run_decode_ablate_cell(
    fwd: &mut Qwen38WgpuForward,
    name: &str,
    ablated: Option<riir_infer_gpu::qwen38_ablate::Ablate>,
    toks: &[u32],
    steps: usize,
    pairs: usize,
    base_walls: &mut Vec<f64>,
    base_ids: &mut Option<Vec<u32>>,
    rows: &mut Vec<(&'static str, f64)>,
) {
    let seed = toks.len();
    let mut deltas: Vec<f64> = Vec::new();
    for pair in 0..pairs {
        // Base arm — the incumbent launch set under the current box state.
        fwd.zero_state();
        fwd.set_ablate(None);
        fwd.ingest(toks, 0)
            .unwrap_or_else(|e| panic!("ingest: {e}"));
        let t0 = std::time::Instant::now();
        let mut next = *toks.last().unwrap();
        let mut first_ids: Vec<u32> = Vec::new();
        for s in 0..steps {
            next = fwd
                .forward_token(next, seed + s)
                .unwrap_or_else(|e| panic!("decode: {e}"));
            if first_ids.len() < 8 {
                first_ids.push(next);
            }
        }
        let base = t0.elapsed().as_secs_f64();
        base_walls.push(base);
        match base_ids {
            None => *base_ids = Some(first_ids),
            Some(prev) => {
                if prev != &first_ids {
                    eprintln!(
                        "[decode-ablate {name}] ⚠ base walk ids diverged at pair {pair} — \
                         determinism drift, the box moved; suspect this cell's delta"
                    );
                    *prev = first_ids;
                }
            }
        }

        // Ablated arm — the decode cell skips only d_* classes; the ingest
        // (prefill sites) runs its full launch set either way.
        fwd.zero_state();
        fwd.set_ablate(ablated.clone());
        fwd.ingest(toks, 0)
            .unwrap_or_else(|e| panic!("ingest: {e}"));
        let t1 = std::time::Instant::now();
        let mut next = *toks.last().unwrap();
        for s in 0..steps {
            next = fwd
                .forward_token(next, seed + s)
                .unwrap_or_else(|e| panic!("decode: {e}"));
        }
        let abl = t1.elapsed().as_secs_f64();
        fwd.set_ablate(None);

        let d = (base - abl) * 1e3 / steps as f64;
        deltas.push(d);
        println!(
            "[decode-ablate {name}] pair {pair}: base {:.3}s ({:.2} tok/s) / \
             abl {:.3}s ({:.2} tok/s) -> Δ {d:+.3} ms/step",
            base,
            steps as f64 / base,
            abl,
            steps as f64 / abl
        );
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = deltas[deltas.len() / 2];
    let class: &'static str = riir_infer_gpu::qwen38_ablate::DECODE_CLASSES
        .iter()
        .copied()
        .find(|c| *c == name)
        .unwrap_or("ALL");
    rows.push((class, med));
}
