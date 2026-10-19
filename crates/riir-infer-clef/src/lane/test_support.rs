//! Test-only support for the lane gates (`#[cfg(test)]`): the scratch
//! dir seam, the lossless in-code tokenizer, and the fixture embedding
//! table writer — shared by the record/host/loader test modules so the
//! fixtures stay one definition (T1.5+).

use std::path::PathBuf;

/// A fresh scratch dir under the cargo target temp (unique per test id
/// via the prefix); the caller removes it when done.
pub(crate) fn scratch(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("clef_lane_{prefix}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// The standard GPT-2 byte→unicode mapping (the same table the
/// crate's own `bytes_char` builds — pub(crate), so re-derived here
/// for the test vocab).
pub(crate) fn byte_unicode_pairs() -> Vec<(u8, char)> {
    let mut bytes: Vec<u8> = (b'!'..=b'~').collect();
    bytes.extend(0xA1..=0xAC);
    bytes.extend(0xAE..=0xFF);
    let mut chars: Vec<char> = bytes.iter().map(|&b| b as char).collect();
    let mut next: u32 = 0;
    for b in 0..=255u8 {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(char::from_u32(256 + next).expect("256+next stays in range"));
            next += 1;
        }
    }
    bytes.into_iter().zip(chars).collect()
}

/// The lossless in-code tokenizer: byte-level BPE over the full
/// 256-char alphabet, empty merge list (each byte its own token), the
/// clef specials as added tokens. `decode(encode(x)) == x` exactly —
/// structural tests pin the record layout through round-trip
/// legibility, no fixture file needed. Built from a tokenizer.json-
/// shaped string (the builder API's vocab type is `Into<AHashMap>`,
/// which would drag ahash into the dep tree).
///
/// Ids: 0..255 = the byte-unicode chars (in [`byte_unicode_pairs`]
/// order), 256+ = the five specials (`<|im_start|>`, `<|im_end|>`,
/// `<think>`, `</think>`, `<|image_pad|>`).
pub(crate) fn lossless_tokenizer() -> crate::lane::ClefTokenizer {
    let vocab: serde_json::Map<String, serde_json::Value> = byte_unicode_pairs()
        .into_iter()
        .enumerate()
        .map(|(id, (_, c))| (c.to_string(), serde_json::Value::from(id as u64)))
        .collect();
    let added: Vec<serde_json::Value> = [
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<|image_pad|>",
    ]
    .iter()
    .enumerate()
    .map(|(i, text)| {
        serde_json::json!({
            "id": 256 + i as u64,
            "content": text,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true,
        })
    })
    .collect();
    let document = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
            "use_regex": false,
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
            "use_regex": false,
        },
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    });
    let inner: tokenizers::Tokenizer = document.to_string().parse().expect("test tokenizer builds");
    crate::lane::ClefTokenizer::from_inner(inner)
}

/// A deterministic pseudo-random f16 payload for byte `b` of vocab row
/// `id` — pure, so assertions can recompute expected values without
/// storing the fixture. Values are f16-representable by construction
/// (small integers / halves).
pub(crate) fn fixture_embedding_bits(id: u32, d: usize) -> u16 {
    // deterministic LCG over (id, d); quantize to a small grid so every
    // value is exact in f16 AND its f32 widening is exact
    let x = (id as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add((d as u64).wrapping_mul(1_442_695_040_888_963_407))
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mantissa = ((x >> 33) % 2048) as u16; // 0..2048
    let exponent = ((x >> 20) % 8) as u16; // small exponents, no inf/nan
    // f16 bits: sign 0, exponent bias 15 → exp field 10..17 stays normal
    (exponent + 10) << 10 | mantissa
}

/// Write a fixture `embeddings.f16` ([`vocab`, `hidden`], values from
/// [`fixture_embedding_bits`]) into `dir` and open it against `config`.
pub(crate) fn write_fixture_embeddings(
    dir: &std::path::Path,
    vocab: u32,
    hidden: u32,
) -> crate::lane::EmbeddingTable {
    let mut bytes = Vec::with_capacity(vocab as usize * hidden as usize * 2);
    for id in 0..vocab {
        for d in 0..hidden as usize {
            bytes.extend_from_slice(&fixture_embedding_bits(id, d).to_le_bytes());
        }
    }
    let path = dir.join("embeddings.f16");
    std::fs::write(&path, &bytes).expect("fixture embeddings written");
    // A minimal valid config-shaped dims record (only vocab/hidden are
    // read by EmbeddingTable::open).
    let config = crate::lane::ClefConfig {
        model: "fixture".into(),
        base_model: String::new(),
        teacher: String::new(),
        hidden_size: hidden,
        head_dim: 1,
        rope_theta: 1_000_000.0,
        pad_id: 0,
        vocab_size: vocab,
        buckets: vec![256],
        max_questions: 16,
        max_options: 96,
        decoder: "Decoder.mlpackage".into(),
        head: "Head.mlpackage".into(),
        tied_embeddings: true,
    };
    crate::lane::EmbeddingTable::open(dir, &config).expect("fixture table opens")
}

/// The fixture-file question form (`records.json`'s `questions` rows)
/// → a typed [`ClefQuestion`]. Shared by the record goldens and the
/// head-packing golden — one parser, one schema.
pub(crate) fn parse_fixture_question(
    value: &serde_json::Value,
) -> (String, crate::lane::record::ClefQuestion) {
    use crate::lane::record::ClefQuestion;
    use std::collections::BTreeMap;
    let id = value["id"].as_str().expect("id").to_string();
    let instructions = value["instructions"].as_str().map(ToOwned::to_owned);
    let kind = value["kind"].as_str().expect("kind");
    let question = match kind {
        "noul" | "choice" => {
            let criteria: BTreeMap<String, String> = value["criteria"]
                .as_object()
                .expect("criteria object")
                .iter()
                .map(|(k, v)| (k.clone(), v.as_str().expect("string").to_string()))
                .collect();
            match kind {
                "noul" => ClefQuestion::Noul {
                    instructions,
                    criteria,
                },
                _ => ClefQuestion::Choice {
                    instructions,
                    criteria,
                },
            }
        }
        "score" => ClefQuestion::Score {
            instructions,
            criteria: value["criteria"]
                .as_array()
                .expect("criteria array")
                .iter()
                .map(|v| v.as_str().expect("string").to_string())
                .collect(),
        },
        other => panic!("unknown fixture kind {other}"),
    };
    (id, question)
}
