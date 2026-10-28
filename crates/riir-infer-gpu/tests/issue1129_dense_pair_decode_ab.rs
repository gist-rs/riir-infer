//! Issue 1129 — the `cuda_backend` CubeCL DECODE A/B for the dense-pair fused
//! GEMV (`RIIR_GDN_DENSE_PAIR_FUSED`): the rung is DEFAULT-ON globally since
//! the Bench-1007 Metal GOAT, but its CudaRuntime arm was never measured —
//! the scored 4090 decode lane is the cudarc forward (its K2 already fuses
//! this pair), so the CubeCL portability arm ships an unmeasured default.
//! This cell measures it: `set_dense_pair_fused(false)` (the incumbent
//! two-launch pair) vs `true` (the fused single launch), decode rounds
//! token-by-token on the league model, interleaved ABBA in ONE process with
//! ONE model load — the Issue-1115 pattern applied to the decode path the
//! pair actually serves.
//!
//! Gates:
//!
//! - **G1 (model level)** — per-token FNV over run 1's recorded logits must
//!   be EQUAL across every OFF/ON arm pair (to_bits fold, never tolerance):
//!   the fused kernel is the same per-row math with the same reduction order,
//!   so any difference is a wiring bug. The kernel-level half is the
//!   existing `test_dense_pair_bit_identity` lib gate (same crate, same
//!   feature posture) — run it beside this cell.
//! - **Reach (the vacuous-guard law)** — outputs are bit-identical by
//!   design, so ONLY the counter proves the toggle reached the dispatch: the
//!   OFF arm must advance `dense_pair_launch_count()` by exactly ZERO and
//!   the ON arm by exactly `n_layer * RUNS * (TOKENS + 1)` (one fused launch
//!   per layer per `forward_token` call).
//! - **G2/G3 (paired medians)** — the fusion is a launch-count cut on a
//!   per-dispatch-floor path (Bench 1006/1007), so the expected delta is
//!   positive-or-noise on CUDA (native driver launches are cheaper than the
//!   Metal command-buffer path). A median ratio moving AGAINST (> 1.0)
//!   beyond noise is the demote-scope signal for the cuda posture — verdict
//!   printed, not asserted (the G2 number is the record's).
//!
//! ```bash
//! cargo test -p riir-infer-gpu --release \
//!   --features cuda_backend,ternary_gemv_cuda_raw,ternary_gemm_batched,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard \
//!   --test issue1129_dense_pair_decode_ab -- --ignored --nocapture
//! ```
//!
//! Env: `BONSAI_GGUF` (default the sibling riir-train data path).
//! `RIIR_GDN_DENSE_PAIR_FUSED` / `RIIR_GDN_DENSE_PAIR_WG` /
//! `RIIR_DENSE_PAIR_MLP` must be UNSET: the test drives the arms through
//! their setters and the env's OnceLock would win over them. Run on an IDLE
//! 4090 (the GPU-exclusivity rule: no other compute consumer).

#![cfg(all(
    feature = "cuda_backend",
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill",
    feature = "ternary_attention_batched_prefill",
    not(target_os = "macos")
))]

use std::time::Instant;

/// The decode schedule the Metal A/B pinned: 24 measured tokens, 5 runs
/// (run 0 = warm-up), a per-run token stride keeping every position-carrying
/// metadata fresh (the bench_762 protocol).
const TOKENS: usize = 24;
const RUNS: usize = 5;
const RUN_TOKEN_STRIDE: usize = 997;
const BOS: usize = 1;

fn fnv_tokens(logits: &[Vec<f32>]) -> Vec<u64> {
    logits
        .iter()
        .map(|l| {
            l.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
                x.to_bits().to_le_bytes().iter().fold(h, |h, b| {
                    (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
                })
            })
        })
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    v[v.len() / 2]
}

struct ArmResult {
    /// Run 1's per-token logits (the G1 record).
    logits: Vec<Vec<f32>>,
    ms_per_token: f64,
    /// `dense_pair_launch_count()` delta over the whole arm (all runs).
    pair_launches: u64,
}

fn run_arm(
    fwd: &mut riir_infer_gpu::TernaryDeltanetGpuForward,
    weights: &riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights,
    vocab: usize,
    n_layer: usize,
    pair_on: bool,
) -> ArmResult {
    riir_infer_gpu::gemv_cubecl::set_dense_pair_fused(pair_on);
    let counter_before = riir_infer_gpu::gemv_cubecl::dense_pair_launch_count() as u64;
    let mut logits: Vec<Vec<f32>> = Vec::with_capacity(TOKENS);
    let mut walls = Vec::new();
    for run in 0..RUNS {
        fwd.reset_state();
        fwd.set_input_token(weights, BOS);
        let _ = fwd.forward_token();

        if run == 1 {
            logits.clear(); // record run 1's logits for G1
        }
        let t0 = Instant::now();
        for pos in 1..=TOKENS {
            let token = (BOS + pos + run * RUN_TOKEN_STRIDE) % vocab;
            fwd.set_input_token(weights, token);
            let l = fwd.forward_token();
            if run == 1 {
                logits.push(l);
            }
        }
        if run > 0 {
            walls.push(t0.elapsed().as_secs_f64());
        }
    }
    ArmResult {
        logits,
        ms_per_token: median(&mut walls) / TOKENS as f64 * 1000.0,
        pair_launches: riir_infer_gpu::gemv_cubecl::dense_pair_launch_count() as u64
            - counter_before,
    }
}

/// The per-arm reach assertion (the vacuous guard): OFF moves the counter by
/// exactly ZERO, ON by exactly n_layer per `forward_token` call.
fn assert_reach(tag: &str, arm: &ArmResult, n_layer: usize, pair_on: bool) {
    let expected = if pair_on {
        (n_layer * RUNS * (TOKENS + 1)) as u64
    } else {
        0
    };
    assert_eq!(
        arm.pair_launches, expected,
        "{tag}: dense-pair counter moved {} (expected {expected}) — the toggle did not reach exactly the dense-leg dispatch",
        arm.pair_launches
    );
}

#[test]
#[ignore = "league-model GPU lane — run explicitly on an idle 4090 (load ~6 min + A/B ~4 min)"]
fn issue1129_dense_pair_decode_ab() {
    for (name, what) in [
        (
            "RIIR_GDN_DENSE_PAIR_FUSED",
            "the test drives the fused arm via the setter",
        ),
        (
            "RIIR_GDN_DENSE_PAIR_WG",
            "the test pins wg 256 via the setter",
        ),
        (
            "RIIR_DENSE_PAIR_MLP",
            "the mlp8 posture stays at its default",
        ),
    ] {
        assert!(
            std::env::var(name).is_err(),
            "{name} must be UNSET — {what} and the env's OnceLock would win over it"
        );
    }
    let Ok(ctx) = riir_infer_gpu::CubeCLContext::new() else {
        eprintln!("[skip] no GPU (CubeCLContext::new failed)");
        return;
    };

    let model_path = std::env::var("BONSAI_GGUF")
        .unwrap_or_else(|_| "../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".to_string());
    let model_path = std::path::PathBuf::from(model_path);
    eprintln!("[1129-cuda-decode-ab] loading {} …", model_path.display());
    let t0 = Instant::now();
    let (mut config, weights) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf(&model_path)
            .map_err(|e| format!("load: {e}"))
            .unwrap();
    config.block_size = config.block_size.min(4096);
    let n_layer = config.n_layer;
    let vocab = config.vocab_size;
    let mut fwd = riir_infer_gpu::TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
    eprintln!(
        "[1129-cuda-decode-ab] forward ready ({:.0}s) · n_layer={n_layer} · vocab={vocab}",
        t0.elapsed().as_secs_f32(),
    );

    // The full decode stage mask — every census leg on, the shipping decode
    // posture; the dense pair is one leg of it (GDN_PROJ_DENSE).
    riir_infer_gpu::ternary_deltanet_gpu_forward::set_decode_stage_mask(
        riir_infer_gpu::ternary_deltanet_gpu_forward::DECODE_STAGE_ALL,
    );
    // The wg knob pinned at 256: the micro-bench read the pair FLAT across
    // wg sizes (per-dispatch floor, not grid-bound), so the rung under test
    // is the fusion alone at the incumbent geometry.
    riir_infer_gpu::gemv_cubecl::set_dense_pair_wg(256);

    // ABBA: OFF, ON, ON, OFF — arm order alternates so any monotone box
    // drift lands in BOTH arms' medians.
    let off1 = run_arm(&mut fwd, &weights, vocab, n_layer, false);
    let on1 = run_arm(&mut fwd, &weights, vocab, n_layer, true);
    let on2 = run_arm(&mut fwd, &weights, vocab, n_layer, true);
    let off2 = run_arm(&mut fwd, &weights, vocab, n_layer, false);
    // Leave the promoted default in place (the global posture stays ON).
    riir_infer_gpu::gemv_cubecl::set_dense_pair_fused(true);

    // Reach — the vacuous guard, per arm (OFF exactly zero; ON exactly
    // n_layer per forward_token call).
    assert_reach("off1", &off1, n_layer, false);
    assert_reach("on1", &on1, n_layer, true);
    assert_reach("on2", &on2, n_layer, true);
    assert_reach("off2", &off2, n_layer, false);
    let expected_on = (n_layer * RUNS * (TOKENS + 1)) as u64;

    // G1 — bit-identity at the model level: run 1's per-token FNV equal
    // across every arm pair (to_bits fold, never tolerance).
    let (fnv_off1, fnv_off2, fnv_on1, fnv_on2) = (
        fnv_tokens(&off1.logits),
        fnv_tokens(&off2.logits),
        fnv_tokens(&on1.logits),
        fnv_tokens(&on2.logits),
    );
    assert_eq!(fnv_off1.len(), TOKENS, "off1 recorded logits");
    for i in 0..TOKENS {
        assert_eq!(
            fnv_off1[i], fnv_on1[i],
            "token {i}: off1 vs on1 logits differ — the fused pair is NOT bit-identical on CudaRuntime"
        );
        assert_eq!(
            fnv_off1[i], fnv_off2[i],
            "token {i}: off1 vs off2 (box drift)"
        );
        assert_eq!(fnv_on1[i], fnv_on2[i], "token {i}: on1 vs on2 (box drift)");
    }

    // G1 anchor pin (the bench_734-t5-decode family): FNV-1a over the
    // argmax walk of run 1's ON logits — deterministic token schedule, so a
    // stable cross-session pin for the record.
    let mut walk_hash: u64 = 0xcbf2_9ce4_8422_2325;
    for l in &on1.logits {
        let am = l
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i as u64);
        walk_hash = (walk_hash ^ (am & 0xff)).wrapping_mul(0x100000001b3);
    }

    // G2/G3 — paired medians (each arm has two interleaved samples).
    let mut offs = [off1.ms_per_token, off2.ms_per_token];
    let mut ons = [on1.ms_per_token, on2.ms_per_token];
    let (mo, mn) = (median(&mut offs), median(&mut ons));
    let ratio = mn / mo;
    let delta_pct = (mo - mn) / mo * 100.0;
    let verdict = if ratio > 1.02 {
        "REGRESSION-SIGNAL — the demote-scope question for the cuda posture opens"
    } else if ratio < 0.985 {
        "WIN — the fusion pays on CudaRuntime too"
    } else {
        "NOISE-BAND — the default stands (launch-count savings below the decode-noise floor)"
    };
    println!(
        "[1129-cuda-decode-ab] VERDICT: FNV identical {TOKENS}/{TOKENS} tokens × all arm pairs · \
         pair launches OFF 0/0 ON {expected_on}/{expected_on} · \
         argmax-walk FNV {walk_hash:016x} · \
         ms/token on/off = {ratio:.4}x ({mn:.2} vs {mo:.2}) = {delta_pct:+.2}% — {verdict}"
    );
}
