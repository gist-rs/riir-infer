//! Per-section wall breakdown of the real Bonsai-2-27B CPU decode — drives
//! `deltanet::profiling::profiled_forward_ternary` (Issue 603's instrumented
//! twin of the production forward) over a teacher-forced corpus head and
//! prints `ForwardProfiler::report()`. A measurement, not a gate: nothing is
//! asserted beyond the model/corpus existing (absent ⇒ UNSEEN panic, never a
//! silent pass).
//!
//! ```bash
//! cargo test --release --features forward_profiling,bonsai2_hadamard[,ternary_q2_resident] \
//!     --test bonsai2_forward_profile -- --ignored --nocapture --test-threads=1
//! ```
//! Env: `BONSAI2_GGUF`, `Q8_GATE_CORPUS`, `PROFILE_TOKENS` (default 32; the
//! first 4 positions are warm-up and excluded).
#![cfg(all(feature = "forward_profiling", feature = "bonsai2_hadamard"))]

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::profiling::{ForwardProfiler, profiled_forward_ternary};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;

const WARMUP: usize = 4;

#[test]
#[ignore = "real 27B model — run with --ignored --release"]
fn bonsai2_decode_section_breakdown() {
    let gguf = std::env::var("BONSAI2_GGUF")
        .unwrap_or_else(|_| "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".into());
    let corpus =
        std::env::var("Q8_GATE_CORPUS").unwrap_or_else(|_| ".raw/twt/audition_calib.txt".into());
    let n_tok: usize = std::env::var("PROFILE_TOKENS").map_or(32, |v| v.parse().expect("usize"));
    assert!(
        std::path::Path::new(&gguf).exists() && std::path::Path::new(&corpus).exists(),
        "UNSEEN, not a pass: model {gguf} or corpus {corpus} missing"
    );
    let text = load_corpus_text(std::path::Path::new(&corpus)).expect("corpus");
    let tok = BpeTokenizer::from_gguf(&GgufFile::open(std::path::Path::new(&gguf)).expect("gguf"))
        .expect("tokenizer");
    let ids = tok.encode(&text);
    assert!(ids.len() > WARMUP + n_tok, "corpus too short: {} tokens", ids.len());

    let (mut config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(std::path::Path::new(&gguf)).expect("load");
    config.block_size = WARMUP + n_tok + 1;
    let rope = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&config);
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];

    let mut warm = ForwardProfiler::default();
    let mut prof = ForwardProfiler::default();
    let t0 = std::time::Instant::now();
    for (pos, &id) in ids.iter().take(WARMUP + n_tok).enumerate() {
        let p = if pos < WARMUP { &mut warm } else { &mut prof };
        if pos == WARMUP {
            eprintln!("[profile] warm-up {WARMUP} tokens {:.1}s", t0.elapsed().as_secs_f64());
        }
        profiled_forward_ternary(&mut x, &weights, &mut cache, id, pos, &config, &mut scratch, &rope, p, None, None, None);
    }
    prof.report();
}
