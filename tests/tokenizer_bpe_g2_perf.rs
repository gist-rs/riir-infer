//! Issue 1118 Finding 1 — G2: encode throughput, new (id-pair) vs old
//! (string-key) merge table. The A/B is COMPILE-TIME (the old map no longer
//! exists in the new tree), so the treatment is process-level alternation:
//! this identical file runs in BOTH trees, the driver alternates the two
//! binaries round by round, and the bench record pairs them per round.
//! Report-only: the printed ns/byte is the deliverable.
//!
//! Runs: `cargo test --release --test tokenizer_bpe_g2_perf -- --ignored --nocapture`

use std::time::Instant;

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::tokenizer::BpeTokenizer;

fn gguf_path() -> String {
    std::env::var("BONSAI2_GGUF")
        .unwrap_or_else(|_| "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".into())
}

/// Deterministic mixed corpus: prose samples + vocab-sourced fuzz (fixed
/// seed) — ~14 KB per pass, every pass identical (the A/B measures the
/// same bytes on both sides).
fn corpus(id_to_token: &[String]) -> String {
    let fixed = [
        "The quick brown fox jumps over the lazy dog while the dog sleeps.",
        "Requirement 42980: all requests MUST settle within 3.5 seconds (99.9%).",
        "fn merge_ranks(&self) -> HashMap<(String, String), usize> { self.table.clone() }",
        "中国人民万岁 / مرحبا بالعالم / Привет мир / ¡Hola señor!",
        "the the the the the the the the the the the the the the the the",
        "2026-10-25T19:00:00Z err=ETIMEDOUT retry=3 backoff=exp",
    ];
    let mut s = fixed.join("\n");
    let mut x = 0x6222_6222_6222_6222u64;
    for _ in 0..9000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let id = (x % id_to_token.len() as u64) as usize;
        s.push_str(&id_to_token[id]);
        if x.is_multiple_of(4) {
            s.push(' ');
        }
    }
    s
}

#[test]
#[ignore = "real 27B model — run with --ignored --release"]
fn encode_throughput_probe() {
    let path = gguf_path();
    assert!(
        std::path::Path::new(&path).exists(),
        "UNSEEN, not a pass: model {path} missing"
    );
    let gguf = GgufFile::open(std::path::Path::new(&path)).expect("gguf opens");
    let tok = BpeTokenizer::from_gguf(&gguf).expect("tokenizer loads");
    let tokens = gguf
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .expect("tokens");
    let id_to_token: Vec<String> = tokens
        .iter()
        .map(|v| v.as_str().expect("token str").to_string())
        .collect();
    let text = corpus(&id_to_token);
    let bytes = text.len() as f64;

    // Warm: page in, HashMap residency, allocator arenas.
    let ids = tok.encode(&text);
    assert!(!ids.is_empty());

    const REPS: usize = 5;
    let mut best = f64::MAX;
    for _ in 0..REPS {
        let t0 = Instant::now();
        for _ in 0..4 {
            let _ = std::hint::black_box(tok.encode(std::hint::black_box(&text)));
        }
        let per = t0.elapsed().as_secs_f64() / 4.0 / bytes;
        best = best.min(per);
    }
    println!(
        "G2   encode {:.1} ns/byte best-of-{REPS} ({} bytes/pass, {} ids, {})",
        best * 1e9,
        bytes as usize,
        ids.len(),
        if std::env::var("A/B_SIDE").as_deref() == Ok("old") {
            "old(string-ranks)"
        } else {
            "new(id-pair-ranks)"
        }
    );
}
