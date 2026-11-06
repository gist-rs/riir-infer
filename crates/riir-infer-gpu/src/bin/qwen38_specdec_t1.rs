//! Issue 1179 T1 — the head-only specdec acceptance price cell.
//!
//! Measures the uzu DFlash speculator's draft acceptance against OUR
//! Q4_K_M GGUF target: the vendor head (Path A) proposes a 15-token argmax
//! chain per cycle (the DFlash block, CPU f32), the target verifies it in
//! one keyed pass (`qwen38_spec::keyed_accept`), and the cell reports the
//! mean accepted tokens per forward pass (`KeyedAccept::j` — uzu's
//! `tokens_per_forward_pass` definition, byte-checked against their trie
//! walk at pin `1866d189`).
//!
//! **Decision gate (the issue's T1):** mean j ≥ 2.5 head-only continues
//! Path A; below closes it with this record (Path B, the riir-train head,
//! is already planned in parallel).
//!
//! Posture: feature `qwen38_specdec`; the target forward is the promoted
//! league posture (no RIIR_QWEN38_* env set — the DEMOTE-ONLY defaults).
//! Speed is irrelevant here (T1 is acceptance pricing, not a perf cell):
//! verify rows ride serial `forward_token_capture` calls, state rolls back
//! through a full checkpoint/restore when a chain partially mismatches.
//!
//! Correctness gate (G1-T1): in greedy mode the harness first walks
//! `--sanity-steps` tokens PURELY serially, then runs specdec cycles from
//! the same state — the two token streams must be IDENTICAL (the draft
//! only proposes; the target's own samples decide). A divergence is a
//! harness bug (state discipline, capture, or accept wiring), never a
//! lane result.

#![cfg(feature = "qwen38_specdec")]

use std::path::Path;

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuForward;
use riir_infer_gpu::qwen38_spec::{KeyedVerifyPosture, keyed_accept, serial_keyed_step};
use riir_infer_gpu::qwen38_specdec::{DFlashCpu, StFile, parse_vendor_config};

const DEFAULT_GGUF: &str = "/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf";
const DEFAULT_SPEC_DIR: &str = "/Users/katopz/uzu27b/speculator";
/// The Bench-1167 R2 FairnessManifest pin (the speculator artifact row).
const EXPECTED_SPEC_BLAKE3_PREFIX: &str = "f6c67be2";
const CAPTURE_LAYERS: &[usize] = &[5, 19, 33, 47, 61];

/// The vendor-method filler (Bench 1167 R2's `i1179_uzu38_finish.py`
/// FILLER — the league fixture's opening sentence repeated to ~1357
/// tokens).
const VENDOR_FILLER: &str = "Analyze the following system design and enumerate every failure mode you can find, ordered by severity, with the detection signal for each. Be exhaustive: cover consensus, storage, networking, scheduling, and the operator surface. ";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut gguf_path = DEFAULT_GGUF.to_string();
    let mut spec_dir = DEFAULT_SPEC_DIR.to_string();
    let mut mode = "greedy".to_string(); // greedy | stochastic
    let mut cycles = 200usize;
    let mut ctx = 8192usize;
    let mut chunk = 512usize;
    let mut prompt_kind = "fixture".to_string(); // fixture | vendor
    let mut fixture_path = String::new();
    let mut seed = 0u64;
    let mut sanity_steps = 32usize;
    let mut out_path = String::new();
    let mut diag_cycles = 0usize;
    let mut emb_signs_path = String::new();

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                i += 1;
                gguf_path = args[i].clone();
            }
            "--speculator" => {
                i += 1;
                spec_dir = args[i].clone();
            }
            "--mode" => {
                i += 1;
                mode = args[i].clone();
            }
            "--cycles" => {
                i += 1;
                cycles = args[i].parse().expect("cycles");
            }
            "--ctx" => {
                i += 1;
                ctx = args[i].parse().expect("ctx");
            }
            "--chunk" => {
                i += 1;
                chunk = args[i].parse().expect("chunk");
            }
            "--prompt" => {
                i += 1;
                prompt_kind = args[i].clone();
            }
            "--fixture" => {
                i += 1;
                fixture_path = args[i].clone();
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("seed");
            }
            "--sanity-steps" => {
                i += 1;
                sanity_steps = args[i].parse().expect("sanity");
            }
            "--diag" => {
                i += 1;
                diag_cycles = args[i].parse().expect("diag");
            }
            "--emb-signs" => {
                i += 1;
                emb_signs_path = args[i].clone();
            }
            "--out" => {
                i += 1;
                out_path = args[i].clone();
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let posture = if mode == "greedy" {
        KeyedVerifyPosture {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed,
        }
    } else if mode == "stochastic" {
        // Bench-1167 R2 vendor-method convention: T=0.95 top_k=20 top_p=1.0,
        // repetition_penalty 1.0 (their suffix_repetition_length=1 ring is a
        // NO-OP at penalty 1.0 — read from their harness + kernel at the pin).
        KeyedVerifyPosture {
            temperature: 0.95,
            top_k: Some(20),
            top_p: Some(1.0),
            seed,
        }
    } else {
        eprintln!("--mode must be greedy|stochastic");
        std::process::exit(2);
    };

    println!("[t1] gguf {gguf_path}");
    println!("[t1] speculator dir {spec_dir} mode {mode} cycles {cycles} prompt {prompt_kind}");

    // ── artifact pins (A1/A8: verify BEFORE any weight is consumed) ──
    let spec_file = Path::new(&spec_dir).join("model.safetensors");
    let t0 = std::time::Instant::now();
    let digest = blake3_file(&spec_file);
    println!(
        "[t1] speculator blake3 {digest} ({:.1}s)",
        t0.elapsed().as_secs_f32()
    );
    if !digest.starts_with(EXPECTED_SPEC_BLAKE3_PREFIX) {
        eprintln!(
            "[t1] ⛔ speculator blake3 {digest} does not start with the Bench-1167 pin {EXPECTED_SPEC_BLAKE3_PREFIX} — refusing (weights-from-disk pin law)"
        );
        std::process::exit(2);
    }
    let gguf_digest = blake3_file(Path::new(&gguf_path));
    println!("[t1] gguf blake3 {gguf_digest}");

    // ── load the target ──
    let t0 = std::time::Instant::now();
    let mut fwd = Qwen38WgpuForward::new(Path::new(&gguf_path), ctx, chunk)
        .unwrap_or_else(|e| panic!("[t1] target load failed: {e}"));
    println!("[t1] target loaded in {:.1}s", t0.elapsed().as_secs_f32());
    let gg = GgufFile::open(Path::new(&gguf_path)).unwrap_or_else(|e| panic!("[t1] reopen: {e}"));
    let tok = BpeTokenizer::from_gguf(&gg).unwrap_or_else(|e| panic!("[t1] tokenizer: {e}"));

    // ── load the speculator (vendor artifact) ──
    let t0 = std::time::Instant::now();
    let st = StFile::open(&spec_file).unwrap_or_else(|e| panic!("[t1] st: {e}"));
    let cfg_json = std::fs::read_to_string(Path::new(&spec_dir).join("config.json"))
        .unwrap_or_else(|e| panic!("[t1] config.json: {e}"));
    let vcfg = parse_vendor_config(&cfg_json).unwrap_or_else(|e| panic!("[t1] {e}"));
    let mut draft = DFlashCpu::load(&st, &vcfg).unwrap_or_else(|e| panic!("[t1] draft load: {e}"));
    println!(
        "[t1] draft loaded in {:.1}s (layers {} block {} target_layers {:?})",
        t0.elapsed().as_secs_f32(),
        vcfg.draft.layer_count,
        vcfg.draft.block_size,
        vcfg.draft.target_layer_ids,
    );
    if vcfg.draft.target_layer_ids != CAPTURE_LAYERS.to_vec() {
        eprintln!(
            "[t1] ⛔ vendor target_layer_ids {:?} != the harness capture {CAPTURE_LAYERS:?}",
            vcfg.draft.target_layer_ids
        );
        std::process::exit(2);
    }

    // ── lm_head f32 table (the draft's readout = the TARGET's output head) ──
    let t0 = std::time::Instant::now();
    let vocab = fwd.config.vocab_size;
    let e = fwd.config.n_embd;
    let lm_head = dequant_lm_head(&gg, vocab, e);
    println!(
        "[t1] lm_head dequant {} rows in {:.1}s",
        vocab,
        t0.elapsed().as_secs_f32()
    );

    // ── the vendor input-embedding output incoherence signs (OPTIONAL) ──
    // (decoder.embedding.input_embedding.incoherence_signs.output_signs in
    // the vendor checkpoint — a 20 KB ranged GET.) ⚠ MEASURED BOTH WAYS
    // (probe cells): applying the transform made the draft ranks WORSE
    // (near-miss 134-5882 → random 200k+), consistent with the GGUF table
    // already carrying the LOGICAL rows (the converter folded the transform
    // into blk.0 instead). Default OFF; --emb-signs arms it for re-tests.
    let emb_signs: Vec<f32> = if emb_signs_path.is_empty() {
        Vec::new()
    } else {
        let signs_raw =
            std::fs::read(&emb_signs_path).unwrap_or_else(|e| panic!("[t1] emb signs: {e}"));
        let signs_i32: Vec<i32> = bytemuck::cast_slice(&signs_raw).to_vec();
        assert_eq!(signs_i32.len(), e, "emb signs length != n_embd");
        let s: Vec<f32> = signs_i32.iter().map(|&x| x as f32).collect();
        println!("[t1] emb signs loaded ({e} entries) — transform ARMED");
        s
    };

    // ── the prompt ──
    let prompt_text = match prompt_kind.as_str() {
        "fixture" => {
            if fixture_path.is_empty() {
                eprintln!("--prompt fixture needs --fixture <path>");
                std::process::exit(2);
            }
            std::fs::read_to_string(&fixture_path).unwrap_or_else(|e| panic!("[t1] fixture: {e}"))
        }
        "vendor" => {
            // Bench-1167 R2 calibration: one FILLER token count, then reps.
            let per = tok.encode(VENDOR_FILLER).len().max(1);
            let reps = ((1357 + per / 2) / per).max(1);
            VENDOR_FILLER.repeat(reps)
        }
        other => {
            eprintln!("--prompt must be fixture|vendor (got {other})");
            std::process::exit(2);
        }
    };
    let prompt: Vec<u32> = tok
        .encode(&prompt_text)
        .into_iter()
        .map(|t| t as u32)
        .collect();
    println!(
        "[t1] prompt: {} chars -> {} tokens",
        prompt_text.len(),
        prompt.len()
    );
    assert!(
        prompt.len() + cycles * (vcfg.draft.block_size + 4) < ctx,
        "prompt+cycles exceed ctx {ctx}"
    );

    // ── prefill + prompt-feature append (uzu appends EVERY prompt
    //    position's features to the draft KV — stream.rs at the pin) ──
    let t0 = std::time::Instant::now();
    let (feats, last_logits) = fwd
        .ingest_capture(&prompt, 0, CAPTURE_LAYERS)
        .unwrap_or_else(|e| panic!("[t1] prefill: {e}"));
    println!("[t1] prefill done in {:.1}s", t0.elapsed().as_secs_f32());
    let d = vcfg.draft.model_dim;
    let captured = CAPTURE_LAYERS.len();
    let mut packed = vec![0.0f32; prompt.len() * d * captured];
    for (li, f) in feats.iter().enumerate() {
        assert_eq!(f.len(), prompt.len() * d, "capture layer {li} length");
        for t in 0..prompt.len() {
            packed[t * d * captured + li * d..t * d * captured + (li + 1) * d]
                .copy_from_slice(&f[t * d..(t + 1) * d]);
        }
    }
    draft
        .accept(&packed, prompt.len())
        .unwrap_or_else(|e| panic!("[t1] draft prefill accept: {e}"));

    // The first emitted token: the target's sample at the last prompt position.
    let mut pending = serial_keyed_step(&posture, &last_logits, prompt.len());
    let mut emitted_total = 1usize;

    // ── G1-T1 sanity (greedy only): a pure serial walk must equal the
    //    specdec stream. Run the serial walk FIRST from this state, then
    //    rewind and run the specdec cycles, comparing prefixes.
    let mut serial_stream: Vec<u32> = Vec::new();
    if mode == "greedy" && sanity_steps > 0 {
        let cp = fwd.checkpoint_state();
        let mut pos = prompt.len();
        let mut tok_i = pending;
        for _ in 0..sanity_steps {
            let (next, _, _) = fwd
                .forward_token_capture(tok_i, pos, &[])
                .unwrap_or_else(|e| panic!("[t1] serial sanity: {e}"));
            serial_stream.push(next);
            tok_i = next;
            pos += 1;
        }
        fwd.restore_state(&cp);
    }

    // ── the specdec cycle loop ──
    let block = vcfg.draft.block_size;
    let mask_emb_raw = gg
        .dequant_tensor_row("token_embd.weight", vcfg.draft.mask_token_id as usize, e)
        .unwrap_or_else(|e| panic!("[t1] mask embedding: {e}"));
    let mut mask_emb = mask_emb_raw.clone();
    if !emb_signs.is_empty() {
        riir_infer_gpu::qwen38_specdec::apply_embedding_output_incoherence(
            &mut mask_emb,
            &emb_signs,
        );
    }
    let mut js: Vec<usize> = Vec::new();
    let mut specdec_stream: Vec<u32> = Vec::new();
    #[allow(clippy::needless_range_loop)]
    let mut pos = prompt.len(); // the pending token's position (unforwarded)
    let positions_bound = ctx - block - 2;
    let t_loop = std::time::Instant::now();
    for cycle in 0..cycles {
        if pos >= positions_bound {
            println!("[t1] position bound {positions_bound} reached at cycle {cycle}");
            break;
        }
        // Draft: 15 candidates from one block forward. The embedding rows
        // run through the vendor input-embedding's output incoherence
        // transform FIRST (the draft consumes the vendor's transformed
        // lookup rows, not the logical GGUF rows).
        let t_prev_emb_raw = gg
            .dequant_tensor_row("token_embd.weight", pending as usize, e)
            .unwrap_or_else(|e| panic!("[t1] token emb {pending}: {e}"));
        let mut t_prev_emb = t_prev_emb_raw.clone();
        if !emb_signs.is_empty() {
            riir_infer_gpu::qwen38_specdec::apply_embedding_output_incoherence(
                &mut t_prev_emb,
                &emb_signs,
            );
        }
        let hidden = draft
            .draft(&t_prev_emb, &mask_emb)
            .unwrap_or_else(|e| panic!("[t1] draft: {e}"));
        let mut chain = Vec::with_capacity(block - 1);
        for r in 1..block {
            let row = &hidden[r * d..(r + 1) * d];
            chain.push(argmax_row(row, &lm_head, vocab, e));
        }

        // Verify: [pending, chain...] in one keyed pass.
        let rows = chain.len() + 1;
        let cp = fwd.checkpoint_state();
        let mut logits_flat = vec![0.0f32; rows * vocab];
        let mut caps_j: Vec<Vec<Vec<f32>>> = Vec::with_capacity(rows); // per row, per layer, row vecs
        for (r, tok_i) in std::iter::once(pending)
            .chain(chain.iter().copied())
            .enumerate()
        {
            let (_, logits, caps) = fwd
                .forward_token_capture(tok_i, pos + r, CAPTURE_LAYERS)
                .unwrap_or_else(|e| panic!("[t1] verify row {r}: {e}"));
            logits_flat[r * vocab..(r + 1) * vocab].copy_from_slice(&logits);
            caps_j.push(caps);
        }
        // Diagnostic (first `diag_cycles` cycles): target vs draft top-5 per
        // position — near-miss reads as rank~1-10, a structural bug as
        // rank~random. `chosen` = the target's committed token at that
        // position; draft_rank_of_chosen is the near-miss measure.
        if diag_cycles > 0 && cycle < diag_cycles {
            for r in 0..4.min(rows) {
                let trow = &logits_flat[r * vocab..(r + 1) * vocab];
                let mut ttop: Vec<(u32, f32)> = trow
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, v)| (i as u32, v))
                    .collect();
                ttop.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let chosen = posture.sample_row(trow, pos + r + 1);
                let draft_row: Option<&[f32]> = if r < chain.len() {
                    Some(&hidden[(r + 1) * d..(r + 2) * d])
                } else {
                    None
                };
                let mut dline: Vec<(u32, f32)> = Vec::new();
                let mut rank: Option<usize> = None;
                if let Some(row) = draft_row {
                    for (v, lrow) in lm_head.chunks_exact(e).enumerate().take(vocab) {
                        let mut acc = 0.0f32;
                        for i in 0..e {
                            acc += lrow[i] * row[i];
                        }
                        if dline.len() < 5 {
                            dline.push((v as u32, acc));
                            dline.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        } else if acc > dline[4].1 {
                            dline[4] = (v as u32, acc);
                            dline.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        }
                    }
                    let lrow_c = &lm_head[chosen as usize * e..(chosen as usize + 1) * e];
                    let mut s_chosen = 0.0f32;
                    for i in 0..e {
                        s_chosen += lrow_c[i] * row[i];
                    }
                    let mut rk = 1usize;
                    for (v, lrow) in lm_head.chunks_exact(e).enumerate().take(vocab) {
                        if v == chosen as usize {
                            continue;
                        }
                        let mut acc = 0.0f32;
                        for i in 0..e {
                            acc += lrow[i] * row[i];
                        }
                        if acc > s_chosen {
                            rk += 1;
                        }
                    }
                    rank = Some(rk);
                }
                println!(
                    "[t1.diag] cycle {cycle} row {r}: target_top5={:?} chosen={chosen} draft_top5={:?} draft_rank_of_chosen={:?}",
                    &ttop[..5],
                    dline,
                    rank,
                );
                // Cosine(draft_hidden[row r+1], target residual at the same
                // position) — the basis test: high cosine ⇒ the draft hidden
                // is the right vector in the WRONG readout basis; ~0 ⇒ the
                // draft itself is wrong.
                if r < chain.len() {
                    let drow = &hidden[(r + 1) * d..(r + 2) * d];
                    // layer-61 capture of the same position = caps of verify
                    // row r+1, last capture layer slot.
                    if r + 1 < caps_j.len() {
                        let feat = &caps_j[r + 1][CAPTURE_LAYERS.len() - 1];
                        let cos = cosine(drow, feat);
                        println!(
                            "[t1.diag]   basis test: cos(draft_hidden[{}], target_feat_layer61[pos{}]) = {cos:.4}",
                            r + 1,
                            pos + r + 1
                        );
                        // Row-0 case: the draft's OWN row (the real token) vs
                        // the SAME position's target feature — its feature IS
                        // in the draft KV, the strongest conditioning test.
                        if r == 0 && cycle == 0 {
                            let drow0 = &hidden[0..d];
                            let feat0 = &caps_j[0][CAPTURE_LAYERS.len() - 1];
                            println!(
                                "[t1.diag]   row0 test: cos(draft_hidden[0], feat61[pos{}]) = {:.4}; |d0|={:.3} |f0|={:.3}",
                                pos,
                                cosine(drow0, feat0),
                                drow0.iter().map(|v| v * v).sum::<f32>().sqrt(),
                                feat0.iter().map(|v| v * v).sum::<f32>().sqrt(),
                            );
                            // Dump the real inputs for the fast CPU probe.
                            let mut dump = Vec::new();
                            for cap in caps_j.iter().take(4) {
                                for l in cap {
                                    dump.extend_from_slice(bytemuck::cast_slice(l));
                                }
                            }
                            dump.extend_from_slice(bytemuck::cast_slice(&hidden));
                            std::fs::write("/tmp/i1179_cycle0_dump.f32", &dump).ok();
                            println!(
                                "[t1.diag]   dumped /tmp/i1179_cycle0_dump.f32 (4 rows x 5 layers + hidden)"
                            );
                        }
                        // Feature self-consistency: rmsnorm(feat_61) *
                        // output_norm.weight → lm_head should ≈ the target's
                        // logits row (layer 61 of 64 is near-final).
                        if r == 0 && cycle == 0 {
                            let mut normed = feat.clone();
                            let sum_sq: f32 = normed.iter().map(|v| v * v).sum();
                            let inv = 1.0 / (sum_sq / e as f32 + 1e-6).sqrt();
                            for (_i, v) in normed.iter_mut().enumerate() {
                                *v *= inv;
                            }
                            let mut ttop2: Vec<(u32, f32)> = Vec::new();
                            for (v, lrow) in lm_head.chunks_exact(e).enumerate().take(vocab) {
                                let mut acc = 0.0f32;
                                for i in 0..e {
                                    acc += lrow[i] * normed[i];
                                }
                                if ttop2.len() < 5 {
                                    ttop2.push((v as u32, acc));
                                    ttop2.sort_by(|a, b| {
                                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                                    });
                                } else if acc > ttop2[4].1 {
                                    ttop2[4] = (v as u32, acc);
                                    ttop2.sort_by(|a, b| {
                                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                                    });
                                }
                            }
                            println!(
                                "[t1.diag]   feat61 self-check top5={:?} (target row said {:?})",
                                &ttop2[..5],
                                &ttop[..5]
                            );
                        }
                    }
                }
            }
        }
        let acc = keyed_accept(&posture, &logits_flat, vocab, &chain, pos);
        let j = acc.j;
        js.push(j);
        emitted_total += j;
        // The committed tokens this cycle: the matched chain prefix
        // (chain[0..j-1]) then the bonus — the full stream, recorded for
        // the G1-T1 serial-identity compare.
        specdec_stream.extend_from_slice(&chain[..j.saturating_sub(1).min(chain.len())]);
        specdec_stream.push(acc.bonus);

        // Draft KV append: rows 0..j-1 (uzu's accepted_indices = the walk's
        // rows; the LAST row's capture serves the bonus token's feature —
        // the vendor's own approximation, reproduced exactly).
        {
            let mut packed_rows = vec![0.0f32; j * d * captured];
            for r in 0..j {
                for (li, cap) in caps_j[r].iter().enumerate() {
                    packed_rows[r * d * captured + li * d..r * d * captured + (li + 1) * d]
                        .copy_from_slice(cap);
                }
            }
            draft
                .accept(&packed_rows, j)
                .unwrap_or_else(|e| panic!("[t1] draft accept: {e}"));
        }

        // State discipline: full accept (j == rows) leaves the forwarded
        // set exactly the true prefix; a partial accept restores and
        // re-forwards the accepted j rows.
        if j < rows {
            fwd.restore_state(&cp);
            for (r, tok_i) in std::iter::once(pending)
                .chain(chain.iter().take(j - 1).copied())
                .enumerate()
            {
                fwd.forward_token_capture(tok_i, pos + r, &[])
                    .unwrap_or_else(|e| panic!("[t1] re-forward {r}: {e}"));
            }
        }
        pending = acc.bonus;
        pos += j;

        if cycle % 20 == 0 || cycle == cycles - 1 {
            let mean = js.iter().sum::<usize>() as f64 / js.len() as f64;
            println!(
                "[t1] cycle {cycle}: j={j} mean_j={mean:.3} emitted={emitted_total} pos={pos} ctx_len={} ({:.1}s)",
                draft.context_length(),
                t_loop.elapsed().as_secs_f32(),
            );
        }
    }

    // ── G1-T1 verdict: the specdec greedy stream must equal the serial walk
    //    token for token over the compared prefix (the draft only proposes;
    //    the target's own samples decide — any divergence is a harness bug:
    //    state discipline, capture, or accept wiring).
    if mode == "greedy" && sanity_steps > 0 {
        let n = specdec_stream.len().min(serial_stream.len());
        let mismatches: Vec<usize> = (0..n)
            .filter(|&i| specdec_stream[i] != serial_stream[i])
            .collect();
        if !mismatches.is_empty() {
            let i = mismatches[0];
            eprintln!(
                "[t1] ⛔ G1-T1 FAIL at stream {i}: specdec {} vs serial {} ({} mismatches of {n})",
                specdec_stream[i],
                serial_stream[i],
                mismatches.len()
            );
            std::process::exit(3);
        }
        println!("[t1] G1-T1 PASS — serial == specdec over {n} committed tokens");
    }

    let mean = js.iter().sum::<usize>() as f64 / js.len().max(1) as f64;
    let mut hist = [0usize; 17];
    for &j in &js {
        hist[j.min(16)] += 1;
    }
    let row = serde_json::json!({
        "cell": format!("t1_head_only_{}_{}", prompt_kind, mode),
        "mode": mode,
        "prompt_kind": prompt_kind,
        "prompt_tokens": prompt.len(),
        "cycles": js.len(),
        "mean_j": mean,
        "j_histogram": hist,
        "emitted_total": emitted_total,
        "spec_blake3": digest,
        "gguf_blake3": gguf_digest,
        "seed": seed,
        "sampling": if mode == "greedy" { "greedy (keyed argmax, first-index ties)" } else { "T=0.95 top_k=20 top_p=1.0 (keyed gumbel-max; penalty-free)" },
        "capture_layers": CAPTURE_LAYERS,
        "block_size": block,
        "note": "uzu DFlash argmax chain (no weaver tree) over OUR Q4_K_M target; CPU f32 draft; int8-KV rounding not reproduced (orthogonal-basis, disclosed)",
        "ts": chrono_now(),
    });
    let line = serde_json::to_string(&row).expect("json");
    println!("[t1] ROW {line}");
    if !out_path.is_empty() {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out_path)
            .expect("out open");
        writeln!(f, "{line}").expect("out write");
    }
    let _ = &serial_stream;
    let _ = &specdec_stream;
}

fn argmax_row(hidden: &[f32], lm_head: &[f32], vocab: usize, e: usize) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for v in 0..vocab {
        let row = &lm_head[v * e..(v + 1) * e];
        let mut acc = 0.0f32;
        for i in 0..e {
            acc += row[i] * hidden[i];
        }
        if acc > bv {
            bv = acc;
            best = v as u32;
        }
    }
    best
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn dequant_lm_head(gg: &GgufFile, vocab: usize, e: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; vocab * e];
    let t0 = std::time::Instant::now();
    // Per-row through the loader's typed dispatcher (Q6_K for this
    // checkpoint; any supported type works — loud error otherwise).
    for v in 0..vocab {
        let row = gg
            .dequant_tensor_row("output.weight", v, e)
            .unwrap_or_else(|err| panic!("lm_head row {v}: {err}"));
        out[v * e..(v + 1) * e].copy_from_slice(&row);
        if v % 50000 == 0 && v > 0 {
            eprintln!(
                "[t1] lm_head {v}/{vocab} ({:.1}s)",
                t0.elapsed().as_secs_f32()
            );
        }
    }
    out
}

fn blake3_file(path: &Path) -> String {
    use blake3::Hasher;
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("blake3 open {path:?}: {e}"));
    let mut hasher = Hasher::new();
    std::io::copy(&mut f, &mut hasher).expect("blake3 read");
    hasher.finalize().to_hex().to_string()
}

fn chrono_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch-{now}")
}
