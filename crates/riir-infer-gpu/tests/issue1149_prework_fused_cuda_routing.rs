//! Issue 1149 — the R2 prework-fused CUDA ROUTING proof: the engagement
//! evidence the N/A close owes. `deltanet_prework_fused` is a CubeCL arm
//! (conv1d + SiLU + q/k-norm + head-expand in ONE dispatch) consumed ONLY
//! inside `ternary_deltanet_gpu_forward` — the toggle is fn-private to that
//! module — while the CudaRuntime whole-prefill
//! (`prefill_cuda_full::whole_prefill_inner`) does its GDN prework through
//! FOUR native cudarc dispatches (`launch_conv1d` + `launch_carry_update` +
//! `launch_beta_decay` + `launch_expand_l2`) and routes the recurrence
//! through gdn_mma / gdn_chunked / L8 / rowpar — structurally unable to
//! consult the toggle. Static routing reads have been wrong before (the
//! T1.5 lesson), so this cell proves it at RUNTIME (the
//! engagement-proof-first protocol):
//!
//! 1. ARM 1 (instrument liveness): the whole-prefill knob OFF → the same
//!    `prefill()` call falls through to the CubeCL body, with the prework
//!    arm SETTER-armed (`set_prefill_prework_fused(true)` — the harness
//!    toggle is authoritative over the env). The prework counter MUST
//!    advance (>0): on this box the CubeCL runtime initializes, the fused
//!    site routes when the CubeCL prefill runs, and the counter counts.
//!    Without this arm a zero on ARM 2 would be indistinguishable from a
//!    blind instrument.
//! 2. ARM 2 (the routing verdict): the whole-prefill knob ON (the 4090
//!    serving lane), prework STILL armed. Every chunk MUST advance the
//!    counter by exactly ZERO — the serving lane never routes the site even
//!    with the arm fully armed. That is the N/A evidence: there is nothing
//!    to promote onto the CudaRuntime default; the macOS scoping is
//!    permanent and the env stays the fall-through lane's only arm.
//!
//! ```bash
//! RIIR_PREFILL_CUDA_GRAPHS=0 cargo test -p riir-infer-gpu --release \
//!   --features cuda_backend,ternary_gemv_cuda_raw,ternary_gemm_batched,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,deltanet_prework_fused,riir-infer-core/bonsai2_hadamard \
//!   --test issue1149_prework_fused_cuda_routing -- --ignored --nocapture
//! ```
//!
//! Env: `BONSAI_GGUF` (default the sibling riir-train data path; ABSOLUTE
//! on the box). `RIIR_PREFILL_CUDA_GRAPHS=0` REQUIRED — a captured graph
//! would freeze one arm's kernel sequence (the D5 law).
//! `RIIR_PREFILL_PREWORK_FUSED` and `RIIR_PREFILL_GDN_L8` must be UNSET —
//! the test drives both through their harness setters (the env wins over
//! them where set). Run on an IDLE 4090 (the GPU-exclusivity rule).
//! NOT a perf cell — no timing verdict is claimed; the counters are the
//! whole measurement.

#![cfg(all(
    feature = "cuda_backend",
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill",
    feature = "ternary_attention_batched_prefill",
    feature = "deltanet_prework_fused",
    not(target_os = "macos")
))]

use riir_infer_gpu::deltanet_prework_fused_cubecl::prework_fused_launches;

fn fnv(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        x.to_bits().to_le_bytes().iter().fold(h, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    })
}

#[test]
#[ignore = "4090 routing-proof GPU lane — run explicitly on an idle 4090 (load minutes + run)"]
fn issue1149_prework_fused_cuda_routing() {
    if std::env::var("RIIR_PREFILL_PREWORK_FUSED").is_ok() {
        panic!(
            "RIIR_PREFILL_PREWORK_FUSED must be UNSET — the test arms the fused kernel via the harness setter (the env wins over it where set)"
        );
    }
    if std::env::var("RIIR_PREFILL_GDN_L8").is_ok() {
        panic!(
            "RIIR_PREFILL_GDN_L8 must be UNSET — the L8 posture is pinned via its setter (the env wins over it)"
        );
    }
    if std::env::var("RIIR_PREFILL_CUDA_GRAPHS").ok().as_deref() != Some("0") {
        panic!(
            "run with RIIR_PREFILL_CUDA_GRAPHS=0 — a captured graph would freeze one arm's kernel sequence (the D5 law)"
        );
    }
    let Ok(_cl_ctx) = riir_infer_gpu::CubeCLContext::new() else {
        eprintln!("[skip] no GPU (CubeCLContext::new failed)");
        return;
    };

    let model_path = std::env::var("BONSAI_GGUF")
        .unwrap_or_else(|_| "../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".to_string());
    let model_path = std::path::PathBuf::from(model_path);
    eprintln!("[1149-routing] loading {} …", model_path.display());
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
    config.block_size = config.block_size.min(4096 + 64);
    let mut fwd = riir_infer_gpu::TernaryDeltanetGpuForward::new(&_cl_ctx, &config, &weights);
    eprintln!(
        "[1149-routing] forward ready ({:.0}s) · n_layer={} · vocab={}",
        t0.elapsed().as_secs_f32(),
        config.n_layer,
        config.vocab_size,
    );

    // The shipping league posture (the bench_947/948 pin set), verbatim
    // from the issue1115 vehicle — this cell varies ONLY the prefill lane
    // knob, never the kernel recipe.
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
    tdf::set_prefill_chunk_max(2048);
    riir_infer_gpu::prefill_cuda_full::set_prefill_gdn_l8(true);

    let words: Vec<&str> = (0..2 * 4096)
        .map(|i| {
            const STEMS: [&str; 8] = [
                "the", "river", "bends", "quiet", "under", "stone", "light", "again",
            ];
            STEMS[i % 8]
        })
        .collect();
    let full_tokens = {
        let mut t = tok.encode(&words.join(" "));
        if t.first() != Some(&bos) {
            t.insert(0, bos);
        }
        t
    };

    // ARM 1 — the CubeCL fall-through lane, prework ARMED: the counter must
    // move (instrument liveness on THIS box, THIS runtime).
    riir_infer_gpu::prefill_cuda_full::set_prefill_use_cuda(false);
    tdf::set_prefill_prework_fused(true);
    let tokens512: Vec<usize> = full_tokens[..512].to_vec();
    fwd.reset_state();
    let l_before = prework_fused_launches();
    let logits_cl = fwd.prefill(&tokens512);
    let l_after_cl = prework_fused_launches();
    let cl_delta = (l_after_cl - l_before) as usize;
    eprintln!(
        "[1149-routing] ARM 1 (cubecl lane, prework armed): p=512 → prework launches +{cl_delta} · fnv {:016x}",
        fnv(&logits_cl)
    );
    assert!(
        cl_delta > 0,
        "ARM 1: the prework counter did not move on the CubeCL lane with the arm setter-armed — the instrument is blind (counter wiring, feature posture, or the fused site never dispatched); this cell proves nothing and must not run"
    );

    // ARM 2 — the 4090 serving lane (the cudarc whole-prefill), prework
    // STILL armed: every chunk must advance the counter by exactly ZERO.
    riir_infer_gpu::prefill_cuda_full::set_prefill_use_cuda(true);
    for &plen in &[2048usize, 4096] {
        let tokens: Vec<usize> = full_tokens[..plen].to_vec();
        fwd.reset_state();
        let before = prework_fused_launches();
        let t = std::time::Instant::now();
        let logits = fwd.prefill(&tokens);
        let dt = t.elapsed().as_secs_f64();
        let after = prework_fused_launches();
        let delta = (after - before) as usize;
        eprintln!(
            "[1149-routing] ARM 2 (whole-prefill lane, prework armed): p={plen} ({dt:.3}s, {:.1} tok/s) → prework launches +{delta} · fnv {:016x}",
            plen as f64 / dt,
            fnv(&logits)
        );
        assert_eq!(
            delta, 0,
            "p{plen}: the whole-prefill lane advanced the prework counter by {delta} with the arm armed — the routing premise of the Issue-1149 N/A is FALSE; adjudicate before any close"
        );
    }

    eprintln!(
        "[1149-routing] VERDICT: the CudaRuntime whole-prefill (the 4090 serving lane) routes ZERO CubeCL prework launches across p=2048 + p=4096 with the arm setter-armed, while the CubeCL fall-through lane advances the same counter — Issue 1149 N/A confirmed at runtime: nothing to promote onto the CudaRuntime default; the macOS scoping is permanent and the env stays the fall-through lane's only arm."
    );

    // Restore the unset postures (the CudaRuntime defaults).
    tdf::set_prefill_prework_fused(false);
    riir_infer_gpu::prefill_cuda_full::set_prefill_use_cuda(false);
}
