//! Plan 622 T1.3 — the CUDA MODEL-LEVEL per-key-bias walk at the LEAGUE
//! posture (the real Bonsai-2 model, FA attention arm, T1.0 row 6): the
//! synth-model walk (`plan622_keybias_walk_cuda`) covers the session wiring
//! and the canonical arm; this gate covers the arm the league measured claim
//! runs — `att_pf_fa` engaged (p ≥ 128, ahd 256, 24:4 GQA), graph capture ON
//! (the shipping default), so arms 2+ REPLAY one capture and the arm flips
//! are CONTENT flips (the no-recapture law, proven at the model level).
//!
//! Same incumbent-vs-None-vs-zero protocol as the synth walk, byte-exact.
//! The model load dominates (~5–7 min); each prefill is sub-second at
//! p = 2048. `#[ignore]`d + `BONSAI_GGUF`-named — run explicitly on an idle
//! 4090 (the GPU-exclusivity rule: no other compute consumer).

#![cfg(all(
    feature = "tokengate",
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill",
    feature = "ternary_attention_batched_prefill",
    not(target_os = "macos")
))]

const PROMPT_LEN: usize = 2048;
/// Session stride headroom over the live key count (the synth walk proves
/// the pad unread with NaN; here the pad is zeros — the FA kernel's clamped
/// index can reach stride-1 only at a full table, never past the causal
/// limit's overwrite).
const STRIDE_PAD: usize = 64;

fn fnv(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        x.to_bits().to_le_bytes().iter().fold(h, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    })
}

fn varying(slot: usize) -> f32 {
    ((slot % 7) as f32 - 3.0) * 0.5
}

fn table_with(
    n_layers: usize,
    n_head: usize,
    stride: usize,
    live: usize,
    att_layers: &[usize],
) -> Vec<f32> {
    let mut t = vec![0.0f32; n_layers * n_head * stride];
    for &l in att_layers {
        for h in 0..n_head {
            for k in 0..live {
                t[(l * n_head + h) * stride + k] = varying((l * n_head + h) * live + k);
            }
        }
    }
    t
}

#[test]
#[ignore = "league-model GPU lane — run explicitly on an idle 4090 (load ~6 min + run)"]
fn plan622_league_walk_kb_session_fa_arms() {
    let Ok(_cl_ctx) = riir_infer_gpu::CubeCLContext::new() else {
        eprintln!("[skip] no GPU (CubeCLContext::new failed)");
        return;
    };
    let Ok(cuda_ctx) = cudarc::driver::safe::CudaContext::new(0) else {
        eprintln!("[skip] no CUDA device");
        return;
    };
    let stream = cuda_ctx.new_stream().expect("cuda stream");

    let model_path = std::env::var("BONSAI_GGUF").unwrap_or_else(|_| {
        "../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".to_string()
    });
    let model_path = std::path::PathBuf::from(model_path);
    eprintln!("[622-league-walk] loading {} …", model_path.display());
    let t0 = std::time::Instant::now();
    let (mut config, weights) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf(&model_path)
            .map_err(|e| format!("load: {e}"))
            .unwrap();
    let gguf = riir_infer_core::gguf_loader::GgufFile::open(&model_path)
        .map_err(|e| format!("reopen: {e}"))
        .unwrap();
    let tok = riir_infer_core::tokenizer::BpeTokenizer::from_gguf(&gguf)
        .map_err(|e| format!("tokenizer: {e}"))
        .unwrap();
    let bos = config.bos_token;
    // Headroom over the p=2048 cell (the DQ-lane clamp — bounds the session
    // stride too).
    config.block_size = config.block_size.min(PROMPT_LEN + STRIDE_PAD);
    let mut fwd = riir_infer_gpu::TernaryDeltanetGpuForward::new(&_cl_ctx, &config, &weights);
    eprintln!(
        "[622-league-walk] forward ready ({:.0}s) · n_layer={} · n_head={} · n_kv={} · ahd={}",
        t0.elapsed().as_secs_f32(),
        config.n_layer,
        config.n_head,
        config.n_kv_head,
        config.head_dim,
    );

    // The shipping knob posture (the recl8 recipe) — and GRAPHS stay at
    // their default (armed): the point of this walk is the content-flip
    // arms replaying one capture.
    use riir_infer_gpu::ternary_deltanet_gpu_forward as tdf;
    tdf::set_prefill_use_gemv(false);
    tdf::set_prefill_seq_rmsnorm(false);
    tdf::set_prefill_zero_scratch(false);
    tdf::set_prefill_batch_elementwise(true);
    tdf::set_prefill_use_cmma16(true);
    tdf::set_prefill_use_cmma_i8(true);
    tdf::set_prefill_cmma_i8_psplit(true);
    riir_infer_gpu::set_prefill_use_cuda_mma(false);
    riir_infer_gpu::set_prefill_use_cuda_ffn(false);
    tdf::set_prefill_chunk_max(4096);
    riir_infer_gpu::prefill_cuda_full::set_prefill_use_cuda(true);

    // One deterministic ~2048-token prompt (the recl8 BPE fill loop).
    let words: Vec<&str> = (0..4096)
        .map(|i| {
            const STEMS: [&str; 8] = [
                "the", "river", "bends", "quiet", "under", "stone", "light", "again",
            ];
            STEMS[i % 8]
        })
        .collect();
    let mut tokens = tok.encode(&words.join(" "));
    if tokens.first() != Some(&bos) {
        tokens.insert(0, bos);
    }
    tokens.truncate(PROMPT_LEN);
    assert_eq!(tokens.len(), PROMPT_LEN);

    let (n_layers, n_head) = (config.n_layer, config.n_head);
    let stride = config.block_size;
    let att_layers: Vec<usize> = weights
        .layer_types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            matches!(t, riir_infer_core::types::DeltaNetLayerType::Attention).then_some(i)
        })
        .collect();
    assert!(!att_layers.is_empty(), "league model carries attention layers");
    eprintln!(
        "[622-league-walk] att layers: {att_layers:?} · stride={stride} · table={} f32",
        n_layers * n_head * stride
    );

    let run = |fwd: &mut riir_infer_gpu::TernaryDeltanetGpuForward| {
        fwd.reset_state();
        fwd.prefill(&tokens)
    };

    // Arm 1 — incumbent (no session).
    let incumbent = run(&mut fwd);
    assert!(incumbent.iter().all(|v| v.is_finite()));

    // Attach: zero table (Attention rows only matter — the DeltaNet rows are
    // never read), disarmed.
    let session = riir_infer_gpu::tokengate_gpu::KeyBiasGpuSession::attach(
        &stream,
        &table_with(n_layers, n_head, stride, PROMPT_LEN, &[]),
        stride,
        n_layers,
        n_head,
    )
    .expect("attach kb session");
    fwd.attach_kb(&stream, session);

    // Arm 2 — flag-off == incumbent (bit) — the re-captured graph (attach
    // invalidated) replays the dead-branch posture bit-identically.
    let flag_off = run(&mut fwd);
    assert_eq!(
        fnv(&flag_off),
        fnv(&incumbent),
        "flag-off moved the league logits — the enabled=0 posture is not bit-identical"
    );

    // Arm 3 — zero-armed == incumbent (bit) — same capture, flag content
    // flip 0→1: the softmax collapses the ±0.0 adds. This is the
    // replay-under-content-flip proof at the league posture.
    fwd.arm_kb(&stream, true).expect("arm");
    let zero_armed = run(&mut fwd);
    assert_eq!(
        fnv(&zero_armed),
        fnv(&incumbent),
        "zero-armed moved the league logits"
    );

    // Arm 4 — moved (Attention rows carry the varying table — written IN
    // PLACE, pointer-stable), run TWICE: both replays of the same capture.
    let moved_table = table_with(n_layers, n_head, stride, PROMPT_LEN, &att_layers);
    fwd.write_kb_table(&stream, &moved_table).expect("write table");
    let moved1 = run(&mut fwd);
    let moved2 = run(&mut fwd);
    assert_eq!(
        fnv(&moved1),
        fnv(&moved2),
        "moved arm non-deterministic run-to-run"
    );
    assert!(moved1.iter().all(|v| v.is_finite()));
    let max_delta = moved1
        .iter()
        .zip(incumbent.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-3,
        "no movement (max Δlogit = {max_delta}) — the FA attention arm did not consume the bias"
    );
    eprintln!(
        "[622-league-walk] moved arm: max Δlogit = {max_delta:.4} (fnv {:016x})",
        fnv(&moved1)
    );

    // Arm 5 — disarm flips back to the incumbent bit-exactly (the flag's
    // other direction, same capture).
    fwd.arm_kb(&stream, false).expect("disarm");
    let disarmed = run(&mut fwd);
    assert_eq!(
        fnv(&disarmed),
        fnv(&incumbent),
        "disarm did not restore the incumbent league logits"
    );

    fwd.detach_kb();
    let detached = run(&mut fwd);
    assert_eq!(fnv(&detached), fnv(&incumbent), "detach changed the logits");

    println!(
        "[622-league-walk] VERDICT: all arms green — flag-off/zero/disarm/detach bit-identical, \
         moved deterministic ×2 with max Δlogit {max_delta:.4}"
    );
}
