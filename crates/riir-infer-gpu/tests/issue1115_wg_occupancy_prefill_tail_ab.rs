//! Issue 1115 — the 4090 prefill-tail lm_head A/B for the GEMV occupancy wg
//! rule (`RIIR_GEMV_WG_OCCUPANCY`): the rule is DEFAULT-ON on the wgpu/Metal
//! runtime (Bench 996) and OPT-IN on the `cuda_backend` CudaRuntime, because
//! the whole-prefill final tail dispatches its lm_head through the SAME
//! `GemvTernaryCubeCL::launch()` (prefill_cuda_full, two sites) and on this
//! platform the unset default resolves OFF — unmeasured. This cell measures
//! it: `set_gemv_wg_occupancy(false)` (today's unset posture) vs `true` (the
//! proposed unset posture), pp512 (the tail-dominated shape) + pp4096 (the
//! league posture), interleaved rounds, one model load.
//!
//! The expected delta is SUB-NOISE — lm_head is ONE dispatch per prefill out
//! of thousands — the cell exists to prove the DEFAULT can flip there without
//! a regression (G3), not to win. G1 is the byte-identity law asserted, never
//! assumed cross-runtime: per-round FNV over the returned logits must be
//! EQUAL across arms. Reach (the vacuous-guard law — outputs are byte-
//! identical, so only a counter proves the toggle reached the dispatch): the
//! arm-on rounds must advance `gemv_wg_occupancy_launch_count()` by exactly
//! ONE per prefill (the final-tail lm_head is the only `launch()` on the
//! whole-prefill path) and the arm-off rounds by ZERO; the model's vocab must
//! sit in the 128-wg bucket (≥ 8192) so the armed dispatch genuinely differs
//! from the incumbent 256 (a smaller vocab makes this cell a 256-vs-256
//! no-op — refuse loud rather than measure a vacuous posture).
//!
//! ```bash
//! RIIR_PREFILL_CUDA_GRAPHS=0 cargo test -p riir-infer-gpu --release \
//!   --features cuda_backend,ternary_gemv_cuda_raw,ternary_gemm_batched,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard \
//!   --test issue1115_wg_occupancy_prefill_tail_ab -- --ignored --nocapture
//! ```
//!
//! Env: `BONSAI_GGUF` (default the sibling riir-train data path).
//! `RIIR_PREFILL_CUDA_GRAPHS=0` REQUIRED — a captured graph would freeze one
//! arm's kernel sequence and replay it against the other (the D5 law). Both
//! `RIIR_GEMV_WG_OCCUPANCY` and `RIIR_PREFILL_GDN_L8` must be UNSET: the test
//! drives both knobs through their harness setters (the env wins over them
//! where set), so a set env would silently re-aim the measured postures.
//! Run on an IDLE 4090 (the GPU-exclusivity rule: no other compute consumer).

#![cfg(all(
    feature = "cuda_backend",
    feature = "ternary_gemv_cuda_raw",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill",
    feature = "ternary_attention_batched_prefill",
    not(target_os = "macos")
))]

/// The two shapes the issue names: the tail-dominated cell and the league
/// posture. pp512 makes the single lm_head dispatch the largest fixed share
/// of the wall; pp4096 is the posture the league rows cite.
const PROMPT_LENS: [usize; 2] = [512, 4096];
/// Rounds per shape, interleaved OFF/ON starting OFF (ABAB…): 3 paired
/// medians each side.
const ROUNDS_PER_SHAPE: usize = 6;
/// Session-stride headroom over the largest prompt (the DQ-lane clamp).
const STRIDE_PAD: usize = 64;

fn fnv(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        x.to_bits().to_le_bytes().iter().fold(h, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    })
}

/// One prefill round at the given arm; returns (fnv, secs, launch-counter
/// delta). The counter delta is the reach proof per round.
fn run_round(
    fwd: &mut riir_infer_gpu::TernaryDeltanetGpuForward,
    tokens: &[usize],
    occupancy_on: bool,
) -> (u64, f64, usize) {
    riir_infer_gpu::gemv_ternary_cubecl::set_gemv_wg_occupancy(occupancy_on);
    let before = riir_infer_gpu::gemv_ternary_cubecl::gemv_wg_occupancy_launch_count();
    fwd.reset_state();
    let t = std::time::Instant::now();
    let logits = fwd.prefill(tokens);
    let dt = t.elapsed().as_secs_f64();
    let after = riir_infer_gpu::gemv_ternary_cubecl::gemv_wg_occupancy_launch_count();
    (fnv(&logits), dt, after - before)
}

#[test]
#[ignore = "league-model GPU lane — run explicitly on an idle 4090 (load ~6 min + run)"]
fn issue1115_wg_occupancy_prefill_tail_ab() {
    if std::env::var("RIIR_GEMV_WG_OCCUPANCY").is_ok() {
        panic!(
            "RIIR_GEMV_WG_OCCUPANCY must be UNSET — the test drives the arm via the setter and the env's OnceLock would win over it"
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

    let model_path = std::env::var("BONSAI_GGUF").unwrap_or_else(|_| {
        "../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".to_string()
    });
    let model_path = std::path::PathBuf::from(model_path);
    eprintln!("[1115-wgocc-ab] loading {} …", model_path.display());
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

    // The armed posture check: the occupancy rule resolves 128 wgs only at
    // m ≥ 8192 — the lm_head's m IS the vocab. Below 4096 the rule is a
    // 256-vs-256 no-op and this cell would measure nothing.
    assert!(
        config.vocab_size >= 8192,
        "model vocab {} < 8192 — the occupancy rule would resolve 256 at the lm_head (a vacuous posture); run the league model",
        config.vocab_size
    );

    let bos = config.bos_token;
    let max_prompt = *PROMPT_LENS.iter().max().unwrap();
    config.block_size = config.block_size.min(max_prompt + STRIDE_PAD);
    let mut fwd = riir_infer_gpu::TernaryDeltanetGpuForward::new(&_cl_ctx, &config, &weights);
    eprintln!(
        "[1115-wgocc-ab] forward ready ({:.0}s) · vocab={} (the lm_head m — the 128-wg bucket) · n_layer={}",
        t0.elapsed().as_secs_f32(),
        config.vocab_size,
        config.n_layer,
    );

    // The shipping league posture (the bench_947/948 pin set, the recl8/622
    // recipe) + the L8 rung pinned ON in BOTH arms (shipping default since
    // 2026-10-24 — pinned explicitly so the A/B varies ONLY the occupancy
    // toggle).
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
    // Chunk at the league's 2048 (NOT max_prompt): a single 4096 chunk with
    // GRAPHS=0 over-commits the 24 GB card (model ~14 GB across the CubeCL
    // + cudarc sides + p=4096 staging) and the driver silently spills to
    // system memory — measured 2026-10-25 as a 100–1000× stage blowup
    // (gemm 200 ms → 110–244 s) with 90–209 s round variance. 2048-chunks
    // are the proven envelope; pp4096 runs as two whole-prefill chunks and
    // the lm_head STILL fires exactly once (the final chunk only — the
    // launch-counter assert pins it).
    tdf::set_prefill_chunk_max(2048);
    riir_infer_gpu::prefill_cuda_full::set_prefill_use_cuda(true);
    riir_infer_gpu::prefill_cuda_full::set_prefill_gdn_l8(true);

    let words: Vec<&str> = (0..2 * max_prompt)
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

    for &plen in PROMPT_LENS.iter() {
        let mut tokens = full_tokens.clone();
        tokens.truncate(plen);
        assert_eq!(tokens.len(), plen);

        let mut fnv_off: Vec<u64> = Vec::new();
        let mut fnv_on: Vec<u64> = Vec::new();
        let mut secs_off: Vec<f64> = Vec::new();
        let mut secs_on: Vec<f64> = Vec::new();

        for round in 0..ROUNDS_PER_SHAPE {
            let on = round % 2 == 1;
            let (f, dt, launches) = run_round(&mut fwd, &tokens, on);
            eprintln!(
                "[1115-wgocc-ab] pp{plen} round {round} wg-occ={on} : {:.1} tok/s ({dt:.3}s) · launches +{launches} · fnv {f:016x}",
                plen as f64 / dt
            );
            // Reach: the whole-prefill final tail is the path's ONLY
            // `GemvTernaryCubeCL::launch()` — one dispatch per prefill, and
            // only the armed arm advances the counter.
            assert_eq!(
                launches,
                usize::from(on),
                "pp{plen} round {round}: the occupancy counter moved {} at arm-on={on} — the toggle did not reach exactly the lm_head dispatch",
                launches
            );
            if on {
                fnv_on.push(f);
                secs_on.push(dt);
            } else {
                fnv_off.push(f);
                secs_off.push(dt);
            }
        }

        // G1 — byte-identity asserted, never assumed cross-runtime: every
        // armed round's logits FNV-equal every incumbent round's.
        for (i, (&a, &b)) in fnv_off.iter().zip(fnv_on.iter()).enumerate() {
            assert_eq!(
                a, b,
                "pp{plen} pair {i}: the occupancy wg rule changed the prefill logits on CudaRuntime — the dispatch-only claim is FALSE at the model level"
            );
        }
        assert_eq!(
            fnv_off.first(),
            fnv_off.last(),
            "pp{plen}: arm-off rounds disagree (box drift) — rerun on a quiet box"
        );

        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (mo, mn) = (med(secs_off), med(secs_on));
        let ratio = mn / mo;
        println!(
            "[1115-wgocc-ab] pp{plen} VERDICT: FNV identical {}/{} pairs · wg-occ on/off = {:.4}x ({:.1} vs {:.1} tok/s median) — sub-noise expected; a ratio moving AGAINST (>1) beyond noise keeps the platform split",
            fnv_on.len(),
            fnv_on.len(),
            ratio,
            plen as f64 / mo,
            plen as f64 / mn,
        );
    }

    // Restore the incumbent (today's CudaRuntime unset posture).
    riir_infer_gpu::gemv_ternary_cubecl::set_gemv_wg_occupancy(false);
}
