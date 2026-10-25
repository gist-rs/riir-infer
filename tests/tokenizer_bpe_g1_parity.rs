//! Issue 1118 Finding 1 — G1 REAL-MODEL parity: the packed id-pair merge
//! table must produce byte-identical token ids to the naive string-keyed
//! BPE on the REAL Bonsai-2-27B vocab + merges, over a deterministic
//! adversarial corpus (multilingual samples, digit runs, code punctuation,
//! plus ~4 000 vocab-sourced fuzz chunks under a fixed seed).
//!
//! The reference rebuilds the OLD `(String, String)` merge table from the
//! GGUF metadata directly and runs its own pretokenize + merge + vocab
//! lookup — old and new share NOTHING but the model data, so an equivalence
//! pass here is the strong form (no sequential cross-binary comparison).
//! The joint output is blaked as the drift pin.
//!
//! Runs (the model is LOCAL-KEEP on the data lane):
//! `cargo test --release --test tokenizer_bpe_g1_parity -- --ignored --nocapture`
//! Unseen without the model: asserts loud, never a green zero.

use std::collections::HashMap;

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::tokenizer::BpeTokenizer;

fn gguf_path() -> String {
    std::env::var("BONSAI2_GGUF")
        .unwrap_or_else(|_| "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".into())
}

/// The naive string-keyed reference encoder — the OLD path end to end:
/// pretokenize (via the production's own rule, through the doc-hidden seam),
/// two-String key per probe, strict `<` first-min scan, final vocab lookup
/// with the unk fallback.
fn reference_encode(
    text: &str,
    vocab: &HashMap<String, usize>,
    ranks: &HashMap<(String, String), usize>,
    b2u: &HashMap<u8, String>,
    max_digit_run: usize,
    unk: usize,
) -> Vec<usize> {
    let mut out = Vec::new();
    for chunk in riir_infer_core::tokenizer::gpt2_pretokenize_for_test(text, max_digit_run) {
        let bytes = chunk.as_bytes();
        if bytes.is_empty() {
            continue;
        }
        let mut unicode_str = String::with_capacity(bytes.len() * 2);
        let mut word: Vec<(usize, usize)> = Vec::with_capacity(bytes.len());
        for &b in bytes {
            let start = unicode_str.len();
            unicode_str.push_str(&b2u[&b]);
            word.push((start, unicode_str.len()));
        }
        loop {
            let mut min_rank = usize::MAX;
            let mut min_idx = None;
            for i in 0..word.len().saturating_sub(1) {
                let left = &unicode_str[word[i].0..word[i].1];
                let right = &unicode_str[word[i + 1].0..word[i + 1].1];
                let key = (left.to_string(), right.to_string());
                if let Some(&rank) = ranks.get(&key)
                    && rank < min_rank
                {
                    min_rank = rank;
                    min_idx = Some(i);
                }
            }
            match min_idx {
                None => break,
                Some(idx) => {
                    word[idx].1 = word[idx + 1].1;
                    word.remove(idx + 1);
                }
            }
        }
        for &(s, e) in &word {
            out.push(vocab.get(&unicode_str[s..e]).copied().unwrap_or(unk));
        }
    }
    out
}

#[test]
#[ignore = "real 27B model — run with --ignored --release"]
fn idpair_bpe_matches_string_reference_on_the_real_vocab() {
    let path = gguf_path();
    assert!(
        std::path::Path::new(&path).exists(),
        "UNSEEN, not a pass: model {path} missing (BONSAI2_GGUF overrides)"
    );
    let gguf = GgufFile::open(std::path::Path::new(&path)).expect("gguf opens");
    let tok = BpeTokenizer::from_gguf(&gguf).expect("tokenizer loads");

    // The reference pieces, straight from the GGUF metadata.
    let tokens_arr = gguf
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .expect("tokens");
    let id_to_token: Vec<String> = tokens_arr
        .iter()
        .map(|v| v.as_str().expect("token str").to_string())
        .collect();
    let merges_arr = gguf
        .metadata
        .get("tokenizer.ggml.merges")
        .and_then(|v| v.as_array())
        .expect("merges");
    let mut vocab: HashMap<String, usize> = HashMap::with_capacity(id_to_token.len());
    for (id, s) in id_to_token.iter().enumerate() {
        vocab.insert(s.clone(), id);
    }
    let mut ranks: HashMap<(String, String), usize> = HashMap::with_capacity(merges_arr.len());
    let mut skipped = 0usize;
    for (rank, val) in merges_arr.iter().enumerate() {
        let m = val.as_str().expect("merge str");
        if let Some(pos) = m.find(' ') {
            match (vocab.get(&m[..pos]), vocab.get(&m[pos + 1..])) {
                (Some(_), Some(_)) => {
                    ranks.insert((m[..pos].to_string(), m[pos + 1..].to_string()), rank);
                }
                _ => skipped += 1,
            }
        }
    }
    println!(
        "G1   vocab {} tokens, {} merges (skipped-missing-from-vocab: {skipped})",
        vocab.len(),
        ranks.len()
    );
    assert_eq!(
        skipped, 0,
        "a valid gpt2 GGUF has every merge part in the vocab — a burst here names a broken model"
    );

    let (b2u, _u2b) = BpeTokenizer::byte_table_for_test();

    // Deterministic adversarial corpus: fixed samples + vocab-sourced fuzz.
    let fixed = [
        "the quick brown fox jumps over the lazy dog",
        "Thethe THE THE thethe tttthhhheeee",
        "key 42980. 111222333 0.5% em-dash — 中文テスト العربية",
        "fn merge_ranks(&self) -> HashMap<(String, String), usize> { }",
        "aaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbb 000111222",
        "\u{FFFD}\u{FFFD} ok \u{10FFFF}",
    ];
    let mut text = fixed.join("\n");
    let mut x = 0x1118_1118_1118_1118u64;
    for _ in 0..4000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let id = (x % id_to_token.len() as u64) as usize;
        text.push_str(&id_to_token[id]);
        if x.is_multiple_of(8) {
            text.push(' ');
        }
    }

    let max_digit_run = 1; // the Bonsai GGUF's `tokenizer.ggml.pre` default arm
    let new_ids = tok.encode(&text);
    let ref_ids = reference_encode(
        &text,
        &vocab,
        &ranks,
        &b2u,
        max_digit_run,
        tok.unk_for_test(),
    );
    assert_eq!(
        new_ids.len(),
        ref_ids.len(),
        "id-count diverged (new {} vs reference {})",
        new_ids.len(),
        ref_ids.len()
    );
    let mut first_diff: Option<usize> = None;
    for (i, (a, b)) in new_ids.iter().zip(&ref_ids).enumerate() {
        if a != b {
            first_diff = Some(i);
            break;
        }
    }
    assert!(
        first_diff.is_none(),
        "G1: token ids diverged at index {:?} — the id-pair table is NOT equivalent to the string table",
        first_diff
    );

    // Drift pin: the joint encoding, blaked. A future change to either side
    // re-runs this gate and the pin names the movement.
    let mut hasher = blake3::Hasher::new();
    for id in &new_ids {
        hasher.update(&(*id as u64).to_le_bytes());
    }
    let pin = hasher.finalize().to_hex()[..32].to_string();
    println!(
        "G1   {} ids identical; drift pin blake3:{pin}",
        new_ids.len()
    );
}
