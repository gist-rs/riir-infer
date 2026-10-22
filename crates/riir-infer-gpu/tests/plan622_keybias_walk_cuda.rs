//! Plan 622 T1.3 — the CUDA MODEL-LEVEL per-key-bias walk (the 4090 arm of
//! the T1.3 G1 harness): the kb session carried by `TernaryDeltanetGpuForward`
//! and consumed by the whole-prefill CUDA driver's attention launches.
//!
//! The kernel-level teeth live in `plan622_keybias_fa_gate` (the FA launcher,
//! league attention shapes) and the T1.2 in-file decode tests; THIS gate
//! proves the SESSION wiring end to end over a real model forward: attach →
//! `kb_session_args` → the per-layer `window_for` → the launch sites → the
//! logits. The synth Bonsai-2 GGUF (`tests/common/synth_bonsai2.rs`, 4 layers
//! `[DeltaNet ×3, Attention ×1]`, attention head_dim 128) routes through the
//! whole-prefill driver's CANONICAL attention arm (T1.0 row 7 — ahd 128 fails
//! the FA arm's `ahd == 256` gate by construction; the FA-arm model-level walk
//! is the league-model `#[ignore]` lane, `plan622_keybias_walk_league_cuda`).
//!
//! Arms (the incumbent-vs-None-vs-zero protocol, byte-exact):
//! 1. `incumbent` — no session attached.
//! 2. `flag_off` — session attached, `enabled = 0` → bit-identical to (1).
//! 3. `zero_armed` — all-zero live table, armed → bit-identical to (1). The
//!    pad region of EVERY row is NaN, so this arm is simultaneously the
//!    stride-pad-independence tooth: a single pad read poisons the logits.
//! 4. `moved` — a ±1.5 varying table, run TWICE: run-1 == run-2 (the race /
//!    graph-replay determinism guard — when graphs are armed, which is the
//!    default, both runs REPLAY one capture, so this is also the
//!    content-visibility-under-replay proof), != incumbent (movement), all
//!    finite (no pad read).
//! 5. `moved_att_only` — the same varying values on the Attention layer's
//!    rows ONLY, zeros elsewhere → bit-identical to `moved` (the DeltaNet
//!    layers never launch attention; a driver narrowing the WRONG layer's
//!    window shows here first).
//! 6. `disarmed` — flag back to 0 → bit-identical to (1) (the flip-back
//!    under the same capture).

#![cfg(all(
    feature = "tokengate",
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "cubecl_runtime",
    feature = "ternary_deltanet_chunked_prefill",
    feature = "ternary_attention_batched_prefill",
    not(target_os = "macos")
))]

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use riir_infer_gpu::tokengate_gpu::KeyBiasGpuSession;

/// Not tile-aligned (137 % 32 != 0) and long enough to be a real prefill;
/// the synth attention head_dim (128) routes it to the canonical arm
/// regardless of the FA engagement predicate.
const PROMPT_LEN: usize = 137;
/// The session stride's headroom over the live key count — the pad the
/// zero-armed arm proves unread (NaN-filled).
const STRIDE_PAD: usize = 40;

fn fnv(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        x.to_bits().to_le_bytes().iter().fold(h, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    })
}

/// Deterministic varying table values (±1.5, slot-derived — no RNG, no
/// seed drift between runs).
fn varying(slot: usize) -> f32 {
    ((slot % 7) as f32 - 3.0) * 0.5
}

fn table_with(
    n_layers: usize,
    n_head: usize,
    stride: usize,
    live: usize,
    fill: impl Fn(usize, usize, usize) -> f32,
) -> Vec<f32> {
    let mut t = vec![0.0f32; n_layers * n_head * stride];
    for l in 0..n_layers {
        for h in 0..n_head {
            for k in 0..live {
                t[(l * n_head + h) * stride + k] = fill(l, h, k);
            }
            for k in live..stride {
                t[(l * n_head + h) * stride + k] = f32::NAN;
            }
        }
    }
    t
}

#[test]
fn plan622_cuda_model_walk_kb_session_arms() {
    let Ok(_cl_ctx) = riir_infer_gpu::CubeCLContext::new() else {
        eprintln!("[skip] no GPU (CubeCLContext::new failed) — the walk is a 4090-lane gate");
        return;
    };
    // The session's device buffers: a cudarc stream on the SAME device the
    // whole-prefill stack launches on (the si league-harness posture —
    // same-device allocations are one memory space; the session is a si
    // session's twin).
    let Ok(cuda_ctx) = cudarc::driver::safe::CudaContext::new(0) else {
        eprintln!("[skip] no CUDA device (CudaContext::new(0) failed)");
        return;
    };
    let stream = cuda_ctx.new_stream().expect("cuda stream");

    // The synth GGUF → temp file (pid-suffixed — the shared-temp law) → the
    // GPU forward.
    let bytes = synth::build_gguf(&synth::synth_metadata(false), &synth::synth_tensors_bonsai2());
    let path = synth::write_tmp("plan622_kb_walk", &bytes);
    let (config, weights) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf(&path)
            .expect("load synth gguf");
    let mut fwd = riir_infer_gpu::TernaryDeltanetGpuForward::new(&_cl_ctx, &config, &weights);

    // The whole-prefill gate's knob posture (the recl8 e2e recipe —
    // defensive in a fresh process; the gate fails LOUD on any drift, so
    // these setters are what make the walk deterministic across boxes).
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

    let tokens: Vec<usize> = (0..PROMPT_LEN)
        .map(|i| (i * 37 + 11) % config.vocab_size)
        .collect();
    let (n_layers, n_head) = (config.n_layer, config.n_head);
    let stride = PROMPT_LEN + STRIDE_PAD;
    let att_layer = weights
        .layer_types
        .iter()
        .position(|t| matches!(t, riir_infer_core::types::DeltaNetLayerType::Attention))
        .expect("synth model carries an Attention layer");
    eprintln!(
        "[622-walk] synth model: n_layer={n_layers} n_head={n_head} att_layer={att_layer} stride={stride}"
    );

    let run = |fwd: &mut riir_infer_gpu::TernaryDeltanetGpuForward| {
        fwd.reset_state();
        fwd.prefill(&tokens)
    };

    // CUDA health probe — a tiny alloc+copy+sync on the session stream. A
    // sticky CUDA_ERROR_ILLEGAL_ADDRESS poisons the whole context, so WHERE
    // this first fails bisects the boot path vs the prefill path.
    let probe = |tag: &str| {
        let host = [7.0f32, 9.0];
        match stream.clone_htod(&host) {
            Ok(dev) => {
                let _ = stream.synchronize();
                eprintln!("[622-walk] cuda probe {tag}: OK");
                drop(dev);
            }
            Err(e) => eprintln!("[622-walk] cuda probe {tag}: FAIL {e:?}"),
        }
    };
    probe("post-boot");

    // Arm 1 — incumbent: no session attached.
    let incumbent = run(&mut fwd);
    assert!(!incumbent.is_empty());
    assert!(incumbent.iter().all(|v| v.is_finite()));
    probe("post-incumbent");

    // Attach: zero live table + NaN pad, disarmed (the constructor law).
    let session = KeyBiasGpuSession::attach(
        &stream,
        &table_with(n_layers, n_head, stride, PROMPT_LEN, |_, _, _| 0.0),
        stride,
        n_layers,
        n_head,
    )
    .expect("attach kb session");
    assert_eq!(session.slots(), n_layers * n_head);
    assert!(!session.is_armed());
    fwd.attach_kb(&stream, session);

    // Arm 2 — flag-off == incumbent (bit).
    let flag_off = run(&mut fwd);
    assert_eq!(
        fnv(&flag_off),
        fnv(&incumbent),
        "flag-off moved the logits — the enabled=0 posture is not bit-identical"
    );

    // Arm 3 — zero-armed == incumbent (bit), with a NaN pad on every row:
    // simultaneously the pad-independence tooth.
    fwd.arm_kb(&stream, true).expect("arm");
    assert_eq!(fwd.kb_is_armed(), Some(true));
    let zero_armed = run(&mut fwd);
    assert_eq!(
        fnv(&zero_armed),
        fnv(&incumbent),
        "zero-armed moved the logits — a zero-add leaked (the -0.0 class) or the NaN pad was read"
    );

    // Arm 4 — moved (all layers' rows carry the varying table; only the
    // Attention layer's window is ever read), run TWICE.
    let moved_table = table_with(n_layers, n_head, stride, PROMPT_LEN, |l, h, k| {
        varying((l * n_head + h) * PROMPT_LEN + k)
    });
    fwd.write_kb_table(&stream, &moved_table).expect("write table");
    let moved1 = run(&mut fwd);
    let moved2 = run(&mut fwd);
    assert_eq!(
        fnv(&moved1),
        fnv(&moved2),
        "moved arm non-deterministic run-to-run — a race or an unstable read path"
    );
    assert!(moved1.iter().all(|v| v.is_finite()), "non-finite logits under a NaN pad — the pad was read");
    let max_delta = moved1
        .iter()
        .zip(incumbent.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-3,
        "no movement (max Δlogit = {max_delta}) — the whole-prefill CUDA attention arm did not \
         consume the bias: check the gate knobs (gemv/seq_rmsnorm/psplit/chunked/batched) and \
         that the forward routed through prefill_cuda_full at all"
    );
    eprintln!("[622-walk] moved arm: max Δlogit = {max_delta:.4} (fnv {:016x})", fnv(&moved1));

    // Arm 5 — per-layer window selectivity: bias on the Attention layer's
    // rows ONLY must reproduce the all-layers arm bit-exactly.
    let att_only_table = table_with(n_layers, n_head, stride, PROMPT_LEN, |l, h, k| {
        if l == att_layer {
            varying((l * n_head + h) * PROMPT_LEN + k)
        } else {
            0.0
        }
    });
    fwd.write_kb_table(&stream, &att_only_table).expect("write att-only table");
    let moved_att = run(&mut fwd);
    assert_eq!(
        fnv(&moved_att),
        fnv(&moved1),
        "bias on the DeltaNet layers' rows changed the logits — the launch site narrowed the \
         WRONG layer's window (the DeltaNet layers never launch attention)"
    );

    // Arm 6 — disarm-by-content flips back to the incumbent bit-exactly
    // (the same capture, the flag's other direction).
    fwd.arm_kb(&stream, false).expect("disarm");
    let disarmed = run(&mut fwd);
    assert_eq!(
        fnv(&disarmed),
        fnv(&incumbent),
        "disarm did not restore the incumbent logits"
    );

    // Detach — the incumbent posture returns (graphs invalidated; one more
    // run proves the detached driver is the incumbent).
    fwd.detach_kb();
    let detached = run(&mut fwd);
    assert_eq!(fnv(&detached), fnv(&incumbent), "detach changed the logits");

    let _ = std::fs::remove_file(&path);
}
