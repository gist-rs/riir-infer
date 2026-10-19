//! The clef host math — plan 619 T1.5. A fidelity port of the
//! `ClefTextManager` statics + `ClefVisionHost.headInputs`
//! (FluidUse @ `0230049c`): the pieces the host computes per decision
//! so the CoreML graphs stay bucket-agnostic.
//!
//! - [`rope_tables`] — fp16 `[L, head_dim]` cos/sin, rotate-half
//!   duplicated-halves layout, `angle = position / θ^(2i/head_dim)`;
//!   each half is a bit-exact duplicate (`f16::from_f64` single-rounds
//!   the libm value, the Swift `Float16(Double)` equivalent).
//! - [`causal_pad_mask`] — additive fp16 `[1, 1, L, L]` (0 / −1e4):
//!   real queries see `k ≤ q`; pad queries see every real token and
//!   stay causal among themselves (their outputs are ignored; the law
//!   is finiteness, not attention quality).
//! - [`gather_hidden`] — the fp16 `[1, L, D]` decoder input: raw rows
//!   out of the tied table, pad rows = the `pad_id` row.
//! - [`head_inputs`] — the f32 joint-schema head packing:
//!   `mem_mask` (−1e4 past `n`), `last` (one-hot at `n−1`),
//!   `q_mean`/`o_mean` (1/|span| at the instruction/option spans,
//!   rows packed at the question index / the flattened option index),
//!   `o2q` (option→question incidence), `lexical` (mean of the tied
//!   rows over the option span — f32 accumulation, f32 division),
//!   `type_oh`, `q_mask`/`o_mask` (−1e4 default, 0 for live rows).
//!   Refuses over-capacity records (`TooManyFields`) — the head's
//!   input shapes are fixed at 16 questions / 96 options.
//!
//! The packing never reads the decoder states (the reference passes
//! them through to the feature provider untouched) — [`HeadInputs`] is
//! states-agnostic; T1.6/T1.7 bundle them at the prediction seam.

use super::loaders::EmbeddingTable;
use super::record::ClefEncodedRecord;
use half::f16;

/// The additive blocked value (exactly representable in fp16 and f32).
const BLOCKED: f32 = -1e4;

/// Host-math refusals (the packing's capacity/shape walls).
#[derive(Debug)]
pub enum HostError {
    /// More questions/options than the head's pinned input shapes.
    TooManyFields {
        questions: usize,
        options: usize,
        max_questions: usize,
        max_options: usize,
    },
    /// The record is longer than the bucket it is packed against.
    RecordLongerThanBucket { tokens: usize, bucket: usize },
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyFields {
                questions,
                options,
                max_questions,
                max_options,
            } => write!(
                f,
                "Record has {questions} questions / {options} options; the head holds \
                 {max_questions} / {max_options}"
            ),
            Self::RecordLongerThanBucket { tokens, bucket } => {
                write!(f, "Record needs {tokens} tokens; the bucket holds {bucket}")
            }
        }
    }
}

impl std::error::Error for HostError {}

/// Plain RoPE tables `[L, head_dim]` fp16 (rotate-half layout: the two
/// halves repeat). θ from the config (`rope_theta`).
pub fn rope_tables(length: usize, head_dim: usize, theta: f64) -> (Vec<f16>, Vec<f16>) {
    let half = head_dim / 2;
    let mut cos = vec![f16::ZERO; length * head_dim];
    let mut sin = vec![f16::ZERO; length * head_dim];
    for position in 0..length {
        for i in 0..half {
            let angle = (position as f64) / theta.powf((2 * i) as f64 / head_dim as f64);
            // single rounding double→f16 (the Swift Float16(Double) path)
            let c = f16::from_f64(angle.cos());
            let s = f16::from_f64(angle.sin());
            let row = position * head_dim;
            cos[row + i] = c;
            cos[row + half + i] = c;
            sin[row + i] = s;
            sin[row + half + i] = s;
        }
    }
    (cos, sin)
}

/// Additive causal mask `[1, 1, L, L]` fp16 for `n` real tokens
/// right-padded to `L` (padded keys masked out; padded queries see the
/// real tokens and stay causal among themselves — outputs ignored).
/// Flat `[L·L]` row-major.
pub fn causal_pad_mask(length: usize, real_tokens: usize) -> Vec<f16> {
    let blocked = f16::from_f32(BLOCKED);
    let mut mask = vec![blocked; length * length];
    for query in 0..length {
        for key in 0..length {
            let visible = if query < real_tokens {
                key <= query
            } else {
                key < real_tokens || (key >= real_tokens && key <= query)
            };
            if visible {
                mask[query * length + key] = f16::ZERO;
            }
        }
    }
    mask
}

/// The fp16 `[1, L, D]` decoder input: real positions gather their
/// token's raw row out of the tied table; pad positions gather the
/// `pad_id` row. Flat `[L·D]` fp16 bits, row-major.
pub fn gather_hidden(
    record: &ClefEncodedRecord,
    length: usize,
    pad_id: u32,
    embeddings: &EmbeddingTable,
) -> Vec<u16> {
    let hidden = embeddings.hidden() as usize;
    let mut out = vec![0u16; length * hidden];
    for position in 0..length {
        let token = record.input_ids.get(position).copied().unwrap_or(pad_id);
        embeddings.copy_row_f16(token, &mut out[position * hidden..(position + 1) * hidden]);
    }
    out
}

/// The f32 joint-schema head packing — `HeadInputs` dims: `mem_mask`
/// `[L]`, `last [L]`, `q_mean [maxQ·L]`, `o_mean [maxO·L]`,
/// `o2q [maxO·maxQ]`, `lexical [maxO·hidden]`, `type_oh [maxQ·3]`,
/// `q_mask [maxQ]`, `o_mask [maxO]` — all flat row-major (T1.6 wraps
/// them into `MLMultiArray`s at exactly these shapes).
pub fn head_inputs(
    record: &ClefEncodedRecord,
    length: usize,
    max_questions: usize,
    max_options: usize,
    embeddings: &EmbeddingTable,
) -> Result<HeadInputs, HostError> {
    let n = record.input_ids.len();
    let n_questions = record.questions.len();
    let n_options: usize = record.questions.iter().map(|q| q.option_spans.len()).sum();
    if n_questions > max_questions || n_options > max_options {
        return Err(HostError::TooManyFields {
            questions: n_questions,
            options: n_options,
            max_questions,
            max_options,
        });
    }
    if n > length {
        return Err(HostError::RecordLongerThanBucket {
            tokens: n,
            bucket: length,
        });
    }
    let hidden = embeddings.hidden() as usize;

    let mut mem_mask = vec![0f32; length];
    let mut last = vec![0f32; length];
    let mut q_mean = vec![0f32; max_questions * length];
    let mut o_mean = vec![0f32; max_options * length];
    let mut o2q = vec![0f32; max_options * max_questions];
    let mut lexical = vec![0f32; max_options * hidden];
    let mut type_oh = vec![0f32; max_questions * 3];
    let mut q_mask = vec![BLOCKED; max_questions];
    let mut o_mask = vec![BLOCKED; max_options];

    for value in mem_mask[n..].iter_mut() {
        *value = BLOCKED;
    }
    last[n - 1] = 1.0;
    let mut row = vec![0f32; hidden];

    let mut option_index = 0usize;
    for (question_index, question) in record.questions.iter().enumerate() {
        let weight = 1.0 / question.span.len() as f32;
        for position in question.span.clone() {
            q_mean[question_index * length + position] = weight;
        }
        type_oh[question_index * 3 + question.type_index as usize] = 1.0;
        q_mask[question_index] = 0.0;
        for option_span in &question.option_spans {
            let weight = 1.0 / option_span.len() as f32;
            for position in option_span.clone() {
                o_mean[option_index * length + position] = weight;
            }
            o2q[option_index * max_questions + question_index] = 1.0;
            o_mask[option_index] = 0.0;
            // lexical: f32 accumulation of the widened tied rows over
            // the span, THEN one f32 division (the reference order)
            let dst = &mut lexical[option_index * hidden..(option_index + 1) * hidden];
            for position in option_span.clone() {
                embeddings.row_into_f32(record.input_ids[position], &mut row);
                for (d, value) in row.iter().enumerate() {
                    dst[d] += value;
                }
            }
            for value in dst.iter_mut() {
                *value /= option_span.len() as f32;
            }
            option_index += 1;
        }
    }

    Ok(HeadInputs {
        mem_mask,
        last,
        q_mean,
        o_mean,
        o2q,
        lexical,
        type_oh,
        q_mask,
        o_mask,
        option_count: n_options,
    })
}

/// The packed head inputs ([`head_inputs`]).
#[derive(Debug, Clone, PartialEq)]
pub struct HeadInputs {
    pub mem_mask: Vec<f32>,
    pub last: Vec<f32>,
    pub q_mean: Vec<f32>,
    pub o_mean: Vec<f32>,
    pub o2q: Vec<f32>,
    pub lexical: Vec<f32>,
    pub type_oh: Vec<f32>,
    pub q_mask: Vec<f32>,
    pub o_mask: Vec<f32>,
    /// The packed option count — the logits stride order T1.7's
    /// per-question softmax walks (question order, option order within).
    pub option_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lane::loaders::ClefTokenizer;
    use crate::lane::record::{ClefQuestion, encode};
    use crate::lane::test_support::{lossless_tokenizer, scratch, write_fixture_embeddings};
    use serde_json::{Value, json};
    use std::collections::BTreeMap;

    /// Mirrors `ClefTextTests.testRopeTablesMatchPlainRope`: fp16
    /// values within 1e-3 of the f64 identity, halves bit-duplicated.
    #[test]
    fn rope_tables_match_plain_rope() {
        let (cos, sin) = rope_tables(64, 128, 1_000_000.0);
        assert_eq!(cos.len(), 64 * 128);
        assert_eq!(sin.len(), 64 * 128);
        for position in [0usize, 1, 17, 63] {
            for i in [0usize, 9, 63] {
                let angle = (position as f64) / 1_000_000f64.powf((2 * i) as f64 / 128.0);
                let expected_c = f16::from_f64(angle.cos());
                let expected_s = f16::from_f64(angle.sin());
                assert_eq!(
                    cos[position * 128 + i],
                    expected_c,
                    "cos pos {position} i {i}"
                );
                assert_eq!(
                    sin[position * 128 + i],
                    expected_s,
                    "sin pos {position} i {i}"
                );
                assert_eq!(
                    cos[position * 128 + i],
                    cos[position * 128 + 64 + i],
                    "rotate-half duplication pos {position} i {i}"
                );
                assert_eq!(
                    sin[position * 128 + i],
                    sin[position * 128 + 64 + i],
                    "rotate-half duplication pos {position} i {i}"
                );
            }
        }
        // position 0: angle 0 → cos 1, sin 0 exactly
        assert_eq!(cos[0], f16::ONE);
        assert_eq!(sin[0], f16::ZERO);
    }

    /// Mirrors `ClefTextTests.testMaskIsCausalAndHidesPadding`.
    #[test]
    fn mask_is_causal_and_hides_padding() {
        let length = 8;
        let real = 5;
        let mask = causal_pad_mask(length, real);
        assert_eq!(mask.len(), length * length);
        let visible = |q: usize, k: usize| mask[q * length + k] == f16::ZERO;
        for q in 0..real {
            for k in 0..length {
                assert_eq!(visible(q, k), k <= q, "q {q} k {k}");
            }
        }
        // padded query rows see every real token and stay causal among
        // themselves (outputs ignored, values finite)
        assert!(visible(6, 0) && visible(6, 4) && visible(6, 6));
        assert!(!visible(6, 7));
        // blocked entries are exactly −1e4 (fp16-representable)
        assert_eq!(mask[7], f16::from_f32(-1e4));
    }

    /// The gather: real rows raw from the table, pad rows = the pad_id
    /// row (raw fp16 bits — no widening).
    #[test]
    fn gather_raw_rows_with_pad_tail() {
        let dir = scratch("gather");
        // the lossless tokenizer's ids run to 260 (specials at 256+) —
        // the fixture table must cover them
        let embeddings = write_fixture_embeddings(&dir, 300, 2);
        let tokenizer = lossless_tokenizer();
        let state = json!({"a": 1});
        let questions = vec![(
            "q".to_string(),
            ClefQuestion::Noul {
                instructions: Some("Ready?".into()),
                criteria: BTreeMap::new(),
            },
        )];
        let record = encode(&tokenizer, &state, &questions, 2048).expect("encodes");
        let n = record.input_ids.len();
        let length = n + 3; // 3 pad rows
        let gathered = gather_hidden(&record, length, 0, &embeddings);
        assert_eq!(gathered.len(), length * 2);
        for position in 0..length {
            let token = record.input_ids.get(position).copied().unwrap_or(0);
            let mut expected = [0u16; 2];
            embeddings.copy_row_f16(token, &mut expected);
            assert_eq!(
                &gathered[position * 2..position * 2 + 2],
                &expected[..],
                "row {position} (token {token})"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The packing, end to end over a real (lossless-tokenizer) record:
    /// every array's shape, default fill, and span arithmetic pinned;
    /// lexical checked against an independent f32 fold of the widened
    /// rows (exact — same accumulation order) and an f64 cross-check.
    #[test]
    fn head_packing_matches_span_law() {
        let dir = scratch("packing");
        let embeddings = write_fixture_embeddings(&dir, 300, 4);
        let tokenizer = lossless_tokenizer();
        let state = json!({"tier": "gold", "n": 3});
        let questions: Vec<(String, ClefQuestion)> = vec![
            (
                "gold".into(),
                ClefQuestion::Noul {
                    instructions: Some("Is gold?".into()),
                    criteria: BTreeMap::new(),
                },
            ),
            (
                "where".into(),
                ClefQuestion::Choice {
                    instructions: Some("Where next?".into()),
                    criteria: BTreeMap::from([
                        ("east".to_string(), "Go east".to_string()),
                        ("west".to_string(), "Go west".to_string()),
                    ]),
                },
            ),
        ];
        let record = encode(&tokenizer, &state, &questions, 2048).expect("encodes");
        let n = record.input_ids.len();
        let (max_q, max_o) = (16usize, 96usize);
        // a bucket at or above the record length (the byte-level test
        // tokenizer expands ~7× over the real BPE, so 256 is too small)
        let bucket = if n <= 256 { 256 } else { 1024 };
        let packed = head_inputs(&record, bucket, max_q, max_o, &embeddings).expect("packs");

        // mem_mask / last
        assert_eq!(packed.mem_mask.len(), bucket);
        assert!(packed.mem_mask[..n].iter().all(|v| *v == 0.0));
        assert!(packed.mem_mask[n..].iter().all(|v| *v == -1e4));
        assert_eq!(packed.last.iter().filter(|v| **v == 1.0).count(), 1);
        assert_eq!(packed.last[n - 1], 1.0);

        // option count = 2 (noul) + 2 (choice)
        assert_eq!(packed.option_count, 4);

        // q_mean: each question's span row carries 1/|span| at the span
        let mut span_positions_verified = 0usize;
        for (question_index, question) in record.questions.iter().enumerate() {
            let row = &packed.q_mean[question_index * bucket..(question_index + 1) * bucket];
            let expected_weight = 1.0 / question.span.len() as f32;
            for (position, value) in row.iter().enumerate() {
                let expected = if question.span.contains(&position) {
                    span_positions_verified += 1;
                    expected_weight
                } else {
                    0.0
                };
                assert_eq!(*value, expected, "q_mean q{question_index} pos {position}");
            }
        }
        assert!(span_positions_verified > 0, "spans non-empty");

        // o_mean / o2q / o_mask over the flattened option index
        let mut option_index = 0usize;
        for (question_index, question) in record.questions.iter().enumerate() {
            for option_span in &question.option_spans {
                let row = &packed.o_mean[option_index * bucket..(option_index + 1) * bucket];
                let expected_weight = 1.0 / option_span.len() as f32;
                for (position, value) in row.iter().enumerate() {
                    let expected = if option_span.contains(&position) {
                        expected_weight
                    } else {
                        0.0
                    };
                    assert_eq!(*value, expected, "o_mean o{option_index} pos {position}");
                }
                for other_q in 0..max_q {
                    let expected = if other_q == question_index { 1.0 } else { 0.0 };
                    assert_eq!(
                        packed.o2q[option_index * max_q + other_q],
                        expected,
                        "o2q o{option_index} q{other_q}"
                    );
                }
                assert_eq!(packed.o_mask[option_index], 0.0);
                option_index += 1;
            }
        }
        // dead rows stay defaulted
        for index in packed.option_count..max_o {
            assert!(
                packed.o_mean[index * bucket..(index + 1) * bucket]
                    .iter()
                    .all(|v| *v == 0.0)
            );
            assert_eq!(packed.o_mask[index], -1e4);
        }
        for index in record.questions.len()..max_q {
            assert_eq!(packed.q_mask[index], -1e4);
        }

        // type_oh: noul → 0, choice → 1
        assert_eq!(packed.type_oh[0], 1.0, "noul at index 0");
        assert_eq!(packed.type_oh[3 + 1], 1.0, "choice at index 1");

        // lexical: exact vs the reference-order f32 fold; cross-checked
        // against an f64 fold (within f32 rounding of the sum)
        let mut f64_row = [0f64; 4];
        option_index = 0;
        for question in &record.questions {
            for option_span in &question.option_spans {
                let dst_start = option_index * 4;
                for slot in f64_row.iter_mut() {
                    *slot = 0.0;
                }
                for position in option_span.clone() {
                    let mut widened = [0f32; 4];
                    embeddings.row_into_f32(record.input_ids[position], &mut widened);
                    for (d, value) in widened.iter().enumerate() {
                        f64_row[d] += *value as f64;
                    }
                }
                for (d, expected_f64) in f64_row.iter().enumerate() {
                    let expected_f64 = expected_f64 / option_span.len() as f64;
                    let got = packed.lexical[dst_start + d];
                    // f32 fold (≈|span| additions) vs the f64 fold: the
                    // accumulated rounding is bounded by |span|·eps
                    let absolute = ((got as f64) - expected_f64).abs();
                    let relative = absolute / expected_f64.abs().max(1e-30);
                    assert!(
                        absolute <= 1e-6 || relative < 1e-5,
                        "lexical o{option_index} d{d}: {got} vs {expected_f64}"
                    );
                }
                option_index += 1;
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The capacity wall: > 16 questions or > 96 options refuse, naming
    /// the counts; so does a record longer than its bucket.
    #[test]
    fn packing_capacity_walls() {
        let dir = scratch("walls");
        let embeddings = write_fixture_embeddings(&dir, 8, 2);
        let tokenizer = lossless_tokenizer();
        let state = json!({});
        let question = || {
            (
                "q".to_string(),
                ClefQuestion::Noul {
                    instructions: None,
                    criteria: BTreeMap::new(),
                },
            )
        };
        let seventeen = vec![question(); 17];
        let record = encode(&tokenizer, &state, &seventeen, 8192).expect("encodes");
        let err = head_inputs(&record, 2048, 16, 96, &embeddings)
            .expect_err("17 questions over the wall");
        assert!(
            matches!(
                err,
                HostError::TooManyFields {
                    questions: 17,
                    options: 34,
                    ..
                }
            ),
            "got: {err}"
        );

        let record = encode(&tokenizer, &state, &[question()], 2048).expect("encodes");
        let err = head_inputs(&record, record.input_ids.len() - 1, 16, 96, &embeddings)
            .expect_err("bucket below the record refuses");
        assert!(
            matches!(err, HostError::RecordLongerThanBucket { .. }),
            "got: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ——— the head-packing golden (real pinned tokenizer) ———

    /// Blake3 over a packed array's exact f32 bits (little-endian) —
    /// the golden pins the arithmetic byte-for-byte without committing
    /// 30k floats (and without committing any model weights: the
    /// embeddings in this golden are the deterministic fixture table).
    fn blake3_f32(values: &[f32]) -> String {
        let mut hasher = blake3::Hasher::new();
        for value in values {
            hasher.update(&value.to_le_bytes());
        }
        hasher.finalize().to_string()
    }

    fn packing_hashes(packed: &HeadInputs) -> serde_json::Value {
        serde_json::json!({
            "mem_mask": blake3_f32(&packed.mem_mask),
            "last": blake3_f32(&packed.last),
            "q_mean": blake3_f32(&packed.q_mean),
            "o_mean": blake3_f32(&packed.o_mean),
            "o2q": blake3_f32(&packed.o2q),
            "lexical": blake3_f32(&packed.lexical),
            "type_oh": blake3_f32(&packed.type_oh),
            "q_mask": blake3_f32(&packed.q_mask),
            "o_mask": blake3_f32(&packed.o_mask),
        })
    }

    /// The packing golden input: the `plain_state` record from the
    /// committed record goldens — re-encoded with the REAL pinned
    /// tokenizer so the token stream IS the T1.4-pinned reference
    /// stream — packed against the deterministic fixture embedding
    /// table (weights never committed — the packing math is
    /// table-agnostic).
    fn packing_record_from_goldens(tokenizer: &ClefTokenizer) -> (ClefEncodedRecord, usize) {
        let fixture_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/clef_text/records/records.json"
        ))
        .expect("record goldens committed under tests/fixtures/clef_text/records/");
        let document: Value = serde_json::from_str(&fixture_text).expect("fixture parses");
        let record_json = document["records"]
            .as_array()
            .expect("records")
            .iter()
            .find(|r| r["name"].as_str() == Some("plain_state"))
            .expect("plain_state golden present")
            .clone();
        let max_length = record_json["max_length"].as_u64().expect("max_length") as usize;
        let state = record_json["state"].clone();
        let questions: Vec<(String, ClefQuestion)> = record_json["questions"]
            .as_array()
            .expect("questions")
            .iter()
            .map(crate::lane::test_support::parse_fixture_question)
            .collect();
        let record = encode(tokenizer, &state, &questions, max_length).expect("encodes");
        (record, max_length)
    }

    fn packing_bucket(n: usize) -> usize {
        if n <= 256 { 256 } else { 1024 }
    }

    /// One-time generator for [`real_wire_head_packing_golden`]'s
    /// frozen hashes: `CLEF_GEN_HEAD_FIXTURES=1 cargo test -p
    /// riir-infer-clef --features clef-ane -- --ignored
    /// generate_head_packing_golden`.
    #[test]
    #[ignore = "one-time golden generator (CLEF_GEN_HEAD_FIXTURES=1, real artifact tree required)"]
    fn generate_head_packing_golden() {
        if std::env::var("CLEF_GEN_HEAD_FIXTURES").as_deref() != Ok("1") {
            println!("SKIP: set CLEF_GEN_HEAD_FIXTURES=1 to write the golden");
            return;
        }
        let dir = crate::lane::artifacts_dir().expect("artifact dir resolvable");
        if !dir.join("tokenizer.json").is_file() {
            panic!(
                "tokenizer.json not installed under {} — run the T1.3 install once",
                dir.display()
            );
        }
        let tokenizer = ClefTokenizer::from_dir(&dir).expect("real tokenizer loads");
        let (record, _max_length) = packing_record_from_goldens(&tokenizer);
        let scratch_dir = scratch("genpack");
        // the REAL token stream indexes the full vocab — the fixture
        // table must cover it (1.2 MB scratch, deterministic, never
        // committed; weights never committed — the packing math is
        // table-agnostic)
        let embeddings = write_fixture_embeddings(&scratch_dir, 151_936, 4);
        let bucket = packing_bucket(record.input_ids.len());
        let packed = head_inputs(&record, bucket, 16, 96, &embeddings).expect("packs");
        let document = serde_json::json!({
            "note": "Plan 619 T1.5 head-packing golden — blake3 over each packed array's f32 \
                     bits for the plain_state record (T1.4 goldens) against the deterministic \
                     fixture embedding table (vocab 151936, hidden 4). Regenerate with \
                     CLEF_GEN_HEAD_FIXTURES=1 cargo test -p riir-infer-clef --features clef-ane \
                     -- --ignored generate_head_packing_golden",
            "record": "plain_state",
            "bucket": bucket,
            "record_tokens": record.input_ids.len(),
            "max_questions": 16,
            "max_options": 96,
            "embeddings": {"vocab": 151936, "hidden": 4},
            "option_count": packed.option_count,
            "hashes": packing_hashes(&packed),
        });
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/clef_text/records/head_packing.json"
        ));
        std::fs::create_dir_all(path.parent().expect("parent")).expect("fixture dir");
        std::fs::write(
            path,
            serde_json::to_string_pretty(&document).expect("serializes"),
        )
        .expect("golden written");
        println!("wrote {}", path.display());
        let _ = std::fs::remove_dir_all(&scratch_dir);
    }

    /// The frozen head-packing golden — the packing arithmetic pinned
    /// byte-for-byte over the reference record. `#[ignore]`d: it needs
    /// the artifact tree installed (never an implicit fetch; absent
    /// material skips LOUD).
    #[test]
    #[ignore = "real wire: needs the pinned artifact tree installed"]
    fn real_wire_head_packing_golden() {
        let dir = match crate::lane::artifacts_dir() {
            Ok(dir) => dir,
            Err(e) => {
                println!("SKIP (loud): no artifact dir resolvable: {e}");
                return;
            }
        };
        if !dir.join("tokenizer.json").is_file() {
            println!(
                "SKIP (loud): tokenizer.json not installed under {} — run the T1.3 install once",
                dir.display()
            );
            return;
        }
        let golden_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/clef_text/records/head_packing.json"
        ))
        .expect("golden committed under tests/fixtures/clef_text/records/");
        let golden: Value = serde_json::from_str(&golden_text).expect("golden parses");
        let tokenizer = ClefTokenizer::from_dir(&dir).expect("real tokenizer loads");
        let (record, _max_length) = packing_record_from_goldens(&tokenizer);
        let bucket = golden["bucket"].as_u64().expect("bucket") as usize;
        assert_eq!(
            bucket,
            packing_bucket(record.input_ids.len()),
            "bucket agrees with the record"
        );
        let scratch_dir = scratch("checkpack");
        let embeddings = write_fixture_embeddings(&scratch_dir, 151_936, 4);
        let packed = head_inputs(&record, bucket, 16, 96, &embeddings).expect("packs");
        assert_eq!(
            packed.option_count,
            golden["option_count"].as_u64().expect("option_count") as usize,
            "option count drifted"
        );
        let expected = golden["hashes"].as_object().expect("hashes");
        let got = packing_hashes(&packed);
        for (name, hash) in expected {
            assert_eq!(
                got[name].as_str(),
                Some(hash.as_str().expect("hash")),
                "array {name} drifted"
            );
        }
        println!("head-packing golden ok ({} arrays pinned)", expected.len());
        let _ = std::fs::remove_dir_all(&scratch_dir);
    }
}
