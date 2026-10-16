//! The clef record encoder — plan 619 T1.4. A fidelity port of FluidUse
//! `ClefRecordEncoder` @ `0230049c` (itself a port of Cloudflare's
//! `joint_schema_model.py` `encode_record`): the training-time token
//! stream the student answers inside, plus the instruction/option token
//! spans the head averages over ([`super`] T1.5 packs them).
//!
//! Fidelity contract (every literal below is the reference's):
//! - the fixed system prompt, the `SCHEMA FIELDS` layout, the
//!   `<|im_start|>` / `<think>` prefix and suffix markers;
//! - per field: `FIELD n / ID / TYPE / INSTRUCTION: {raw text}` +
//!   `ALLOWED OPTIONS:` + `OPTION k: {json}` + `END FIELD`; the
//!   instruction renders AS ITSELF (a top-level string is unquoted —
//!   [`super::py_json::render`]); each option renders as the sorted-key
//!   JSON `{"description":…,"option_id":…}`;
//! - option order: `noul` = `[true, false]` (criteria descriptions may
//!   override the defaults), `choice` = sorted option ids, `score` =
//!   `"0".."N"` over the criteria list;
//! - long states truncate FROM THE RIGHT and the field/option spans
//!   shift by `prefix + (possibly truncated) state` — the record is
//!   prefix, state, schema, suffix;
//! - a fixed `prefix + schema + suffix` larger than `max_length`
//!   refuses (`TooLong`); so does an empty option set and — the text
//!   lane's placeholder guard — an input containing the
//!   `<|image_pad|>` special id (the vision splice's token; the text
//!   lane has no splice, so its literal can only be drift).
//!
//! The goldens in `tests/fixtures/clef_text/records/records.json` were
//! generated against the pinned artifact's real `tokenizer.json` (the
//! `#[ignore]`d generator below); the structural tests run always-on
//! over an in-code lossless byte-level tokenizer.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;

use serde_json::Value;

use super::loaders::ClefTokenizer;
use super::py_json;

/// The fixed system prompt (the reference's `ClefRecordEncoder.systemPrompt`).
pub const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";

/// The `noul` default option descriptions (criteria entries override by
/// key; the option ORDER stays `true, false`).
const NOUL_TRUE_DEFAULT: &str = "The proposition is true or the answer is yes.";
const NOUL_FALSE_DEFAULT: &str = "The proposition is false or the answer is no.";

/// A typed question in Clef's schema (the `ClefQuestion` port — the
/// same contract as our `decision_wire` vocabulary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClefQuestion {
    /// Yes / no. `criteria` may describe the `true`/`false` options.
    Noul {
        instructions: Option<String>,
        criteria: BTreeMap<String, String>,
    },
    /// Named options: option id → description. Options are scored in
    /// sorted-id order, like Clef.
    Choice {
        instructions: Option<String>,
        criteria: BTreeMap<String, String>,
    },
    /// Ordered options indexed from 0; the answer is the expected index.
    Score {
        instructions: Option<String>,
        criteria: Vec<String>,
    },
}

impl ClefQuestion {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Noul { .. } => "noul",
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
        }
    }

    /// 0 = noul, 1 = choice, 2 = score (the head's type embedding index).
    pub fn type_index(&self) -> u32 {
        match self {
            Self::Noul { .. } => 0,
            Self::Choice { .. } => 1,
            Self::Score { .. } => 2,
        }
    }

    pub fn instructions(&self) -> Option<&str> {
        match self {
            Self::Noul { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. } => instructions.as_deref(),
        }
    }

    /// `(option id, description)` in Clef's option order.
    pub fn options(&self) -> Vec<(String, Option<String>)> {
        match self {
            Self::Noul { criteria, .. } => {
                let mut merged = BTreeMap::from([
                    ("true".to_string(), NOUL_TRUE_DEFAULT.to_string()),
                    ("false".to_string(), NOUL_FALSE_DEFAULT.to_string()),
                ]);
                // criteria wins on conflict (the reference's merge{_, new in new});
                // keys other than true/false never reach the options.
                merged.extend(criteria.iter().map(|(k, v)| (k.clone(), v.clone())));
                vec![
                    ("true".to_string(), merged.get("true").cloned()),
                    ("false".to_string(), merged.get("false").cloned()),
                ]
            }
            Self::Choice { criteria, .. } => criteria
                .iter()
                .map(|(id, description)| (id.clone(), Some(description.clone())))
                .collect(),
            Self::Score { criteria, .. } => criteria
                .iter()
                .enumerate()
                .map(|(index, description)| (index.to_string(), Some(description.clone())))
                .collect(),
        }
    }
}

/// One encoded question's span bookkeeping (the
/// `ClefEncodedRecord.Question` port).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClefEncodedQuestion {
    pub id: String,
    pub type_index: u32,
    /// The instruction-token span, in `input_ids` coordinates.
    pub span: Range<usize>,
    /// One span per option — the option's JSON tokens (the `OPTION k: `
    /// prefix and the trailing newline are outside).
    pub option_spans: Vec<Range<usize>>,
    pub option_ids: Vec<String>,
}

/// The encoded record: the token stream + the question spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClefEncodedRecord {
    pub input_ids: Vec<u32>,
    pub questions: Vec<ClefEncodedQuestion>,
}

/// The record-level refusals (typed — T1.7's `answer` surfaces these).
#[derive(Debug)]
pub enum RecordError {
    /// `prefix + schema + suffix` alone overflows the largest bucket.
    TooLong { tokens: usize, bucket: usize },
    /// A question with no options (empty `choice` criteria / `score` list).
    EmptyOptions { question: String },
    /// The state or schema carries the `<|image_pad|>` special id — the
    /// vision splice's token; the text lane has no splice, so its
    /// literal can only be drift (the reference refuses the same class).
    ImagePlaceholder { count: usize },
    /// The tokenizer refused a segment.
    Tokenizer(std::io::Error),
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLong { tokens, bucket } => {
                write!(f, "Record needs {tokens} tokens; the largest bucket holds {bucket}")
            }
            Self::EmptyOptions { question } => {
                write!(f, "question {question} has no options")
            }
            Self::ImagePlaceholder { count } => {
                write!(f, "state or schema text contains image placeholder tokens ({count})")
            }
            Self::Tokenizer(e) => write!(f, "clef record: tokenizer: {e}"),
        }
    }
}

impl std::error::Error for RecordError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tokenizer(e) => Some(e),
            _ => None,
        }
    }
}

/// The one segment-encode seam — every token run in the record goes
/// through this (per-segment encode, concatenated: the reference's
/// layout, so BPE never merges across the field boundaries).
fn tokens(tokenizer: &ClefTokenizer, text: &str) -> Result<Vec<u32>, RecordError> {
    tokenizer.encode(text).map_err(RecordError::Tokenizer)
}

/// The option-semantics JSON for one option (sorted keys —
/// `{"description":…,"option_id":…}` when a description exists).
fn option_semantics(option_id: &str, description: Option<&str>) -> Value {
    let mut semantics = serde_json::Map::new();
    semantics.insert("option_id".to_string(), Value::String(option_id.to_string()));
    if let Some(description) = description {
        semantics.insert("description".to_string(), Value::String(description.to_string()));
    }
    Value::Object(semantics)
}

/// The effective instruction text (empty/absent falls back to the field
/// id — the reference's `flatMap { $0.isEmpty ? nil : $0 } ?? id`).
fn effective_instructions(id: &str, question: &ClefQuestion) -> String {
    question
        .instructions()
        .filter(|text| !text.is_empty())
        .map_or_else(|| id.to_string(), ToOwned::to_owned)
}

/// Encode `state` + `questions` into the record the artifact was
/// trained on. `max_length` is the record budget (the largest loaded
/// bucket at the call sites).
pub fn encode(
    tokenizer: &ClefTokenizer,
    state: &Value,
    questions: &[(String, ClefQuestion)],
    max_length: usize,
) -> Result<ClefEncodedRecord, RecordError> {
    let mut schema = tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")?;
    let mut encoded_questions: Vec<ClefEncodedQuestion> = Vec::with_capacity(questions.len());
    for (index, (id, question)) in questions.iter().enumerate() {
        schema.extend(tokens(
            tokenizer,
            &format!(
                "\nFIELD {}\nID: {}\nTYPE: {}\nINSTRUCTION: ",
                index + 1,
                id,
                question.type_name()
            ),
        )?);
        let start = schema.len();
        schema.extend(tokens(
            tokenizer,
            &py_json::render(&Value::String(effective_instructions(id, question))),
        )?);
        let end = schema.len();
        schema.extend(tokens(tokenizer, "\nALLOWED OPTIONS:\n")?);
        let mut option_spans: Vec<Range<usize>> = Vec::new();
        let mut option_ids: Vec<String> = Vec::new();
        let options = question.options();
        if options.is_empty() {
            return Err(RecordError::EmptyOptions { question: id.clone() });
        }
        for (option_index, (option_id, description)) in options.iter().enumerate() {
            schema.extend(tokens(tokenizer, &format!("OPTION {}: ", option_index + 1))?);
            let option_start = schema.len();
            schema.extend(tokens(
                tokenizer,
                &py_json::render(&option_semantics(option_id, description.as_deref())),
            )?);
            option_spans.push(option_start..schema.len());
            option_ids.push(option_id.clone());
            schema.extend(tokens(tokenizer, "\n")?);
        }
        schema.extend(tokens(tokenizer, "END FIELD\n")?);
        encoded_questions.push(ClefEncodedQuestion {
            id: id.clone(),
            type_index: question.type_index(),
            span: start..end,
            option_spans,
            option_ids,
        });
    }

    // The text lane carries no image blocks: the prefix is the plain
    // system/user turn (the vision lane splices `<|image_pad|>` runs
    // between `STATE:` and this newline — never here).
    let prefix = tokens(
        tokenizer,
        &format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"),
    )?;
    let suffix = tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )?;
    let state_ids = tokens(tokenizer, &py_json::render(state))?;

    let fixed = prefix.len() + schema.len() + suffix.len();
    if fixed > max_length {
        return Err(RecordError::TooLong {
            tokens: fixed,
            bucket: max_length,
        });
    }
    let mut state_ids = state_ids;
    if state_ids.len() > max_length - fixed {
        state_ids.truncate(max_length - fixed);
    }
    let offset = prefix.len() + state_ids.len();
    let questions = encoded_questions
        .into_iter()
        .map(|question| ClefEncodedQuestion {
            span: (question.span.start + offset)..(question.span.end + offset),
            option_spans: question
                .option_spans
                .into_iter()
                .map(|span| (span.start + offset)..(span.end + offset))
                .collect(),
            ..question
        })
        .collect();
    let mut input_ids = prefix;
    input_ids.extend_from_slice(&state_ids);
    input_ids.extend(schema);
    input_ids.extend(suffix);

    // The placeholder guard: the tokenizer maps a literal
    // `<|image_pad|>` in the state/schema to its special id, which the
    // vision splice would desync; the text lane has no splice, so any
    // occurrence is drift. Refuse (a vocabulary without the token
    // cannot hit this — the guard is vacuous there, exactly like the
    // text-lane reference, which passes a -1 sentinel id).
    if let Some(image_pad) = tokenizer.token_to_id("<|image_pad|>") {
        let count = input_ids.iter().filter(|&&id| id == image_pad).count();
        if count > 0 {
            return Err(RecordError::ImagePlaceholder { count });
        }
    }

    Ok(ClefEncodedRecord {
        input_ids,
        questions,
    })
}

/// The exact record TEXT the encoder must produce (the structural and
/// golden tests decode the token stream back to this — constructed from
/// the same primitives the encoder uses, independently of tokenization).
#[cfg(test)]
fn expected_record_text(state: &Value, questions: &[(String, ClefQuestion)]) -> String {
    let mut text =
        format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n");
    text.push_str(&py_json::render(state));
    text.push_str("\n\nSCHEMA FIELDS:\n");
    for (index, (id, question)) in questions.iter().enumerate() {
        text.push_str(&format!(
            "\nFIELD {}\nID: {}\nTYPE: {}\nINSTRUCTION: ",
            index + 1,
            id,
            question.type_name()
        ));
        text.push_str(&effective_instructions(id, question));
        text.push_str("\nALLOWED OPTIONS:\n");
        for (option_index, (option_id, description)) in question.options().iter().enumerate() {
            text.push_str(&format!("OPTION {}: ", option_index + 1));
            text.push_str(&py_json::render(&option_semantics(
                option_id,
                description.as_deref(),
            )));
            text.push('\n');
        }
        text.push_str("END FIELD\n");
    }
    text.push_str(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    );
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lane::loaders::ClefTokenizer;
    use serde_json::json;

    /// The standard GPT-2 byte→unicode mapping (the same table the
    /// crate's own `bytes_char` builds — pub(crate), so re-derived
    /// here for the test vocab).
    fn byte_unicode_pairs() -> Vec<(u8, char)> {
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

    /// A lossless in-code tokenizer: byte-level BPE over the full
    /// 256-char alphabet, empty merge list (each byte its own token),
    /// the clef specials as added tokens. `decode(encode(x)) == x`
    /// exactly — the structural tests pin the record layout through
    /// round-trip legibility, no fixture file needed. Built from a
    /// tokenizer.json-shaped string (the builder API's vocab type is
    /// `Into<AHashMap>`, which would drag ahash into the dep tree).
    fn lossless_tokenizer() -> ClefTokenizer {
        let vocab: serde_json::Map<String, Value> = byte_unicode_pairs()
            .into_iter()
            .enumerate()
            .map(|(id, (_, c))| (c.to_string(), Value::from(id as u64)))
            .collect();
        let added: Vec<Value> = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|image_pad|>"]
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
        let inner: tokenizers::Tokenizer = document
            .to_string()
            .parse()
            .expect("test tokenizer builds");
        ClefTokenizer::from_inner(inner)
    }

    fn choice(instructions: Option<&str>, criteria: &[(&str, &str)]) -> ClefQuestion {
        ClefQuestion::Choice {
            instructions: instructions.map(ToOwned::to_owned),
            criteria: criteria
                .iter()
                .map(|(id, description)| (id.to_string(), description.to_string()))
                .collect(),
        }
    }

    fn noul(instructions: Option<&str>) -> ClefQuestion {
        ClefQuestion::Noul {
            instructions: instructions.map(ToOwned::to_owned),
            criteria: BTreeMap::new(),
        }
    }

    fn score(instructions: Option<&str>, criteria: &[&str]) -> ClefQuestion {
        ClefQuestion::Score {
            instructions: instructions.map(ToOwned::to_owned),
            criteria: criteria.iter().map(|level| level.to_string()).collect(),
        }
    }

    /// The whole record decodes back to the exact reference text, and
    /// every span decodes to exactly the instruction / option payload —
    /// the layout law (markers, field numbering, option numbering,
    /// span bookkeeping, state placement) pinned without fixtures.
    #[test]
    fn record_layout_round_trips_to_reference_text() {
        let tokenizer = lossless_tokenizer();
        let state = json!({"tier": "gold", "unread": 3, "ratio": 0.98});
        let questions: Vec<(String, ClefQuestion)> = vec![
            ("tier_check".into(), noul(Some("Is the user gold tier?"))),
            (
                "channel".into(),
                choice(
                    Some("Which channel?"),
                    &[("chat", "Live chat"), ("email", "Email support"), ("phone", "Phone")],
                ),
            ),
            ("mood".into(), score(Some("Rate it."), &["poor", "ok", "good"])),
        ];
        let record = encode(&tokenizer, &state, &questions, 2048).expect("encodes");

        // full-stream legibility
        let decoded = tokenizer.decode(&record.input_ids, false).expect("decodes");
        assert_eq!(decoded, expected_record_text(&state, &questions));

        // span legibility + ordering: each span decodes to its own
        // instruction; spans sit inside the stream, before their option
        // spans, and advance monotonically (the decode-equality assert
        // above already pins the exact positions).
        let mut cursor = 0usize;
        for (encoded, (id, question)) in record.questions.iter().zip(&questions) {
            let span_text = tokenizer
                .decode(&record.input_ids[encoded.span.clone()], false)
                .expect("span decodes");
            assert_eq!(span_text, effective_instructions(id, question), "span of {id}");
            assert!(
                (cursor..encoded.span.end).contains(&encoded.span.start),
                "span ordering"
            );
            cursor = encoded.span.end;
            assert!(
                encoded
                    .option_spans
                    .iter()
                    .all(|span| span.start >= cursor && span.end <= record.input_ids.len()),
                "option spans follow the instruction span"
            );
            let options = question.options();
            for (span, (option_id, description)) in encoded
                .option_spans
                .iter()
                .zip(options.iter())
            {
                let option_text = tokenizer
                    .decode(&record.input_ids[span.clone()], false)
                    .expect("option span decodes");
                assert_eq!(
                    option_text,
                    py_json::render(&option_semantics(option_id, description.as_deref()))
                );
            }
            // END FIELD lands right after the last option span
            let tail = tokenizer
                .decode(
                    &record.input_ids[encoded.option_spans.last().expect("has options").end..],
                    false,
                )
                .expect("tail decodes");
            assert!(tail.starts_with("\nEND FIELD\n"), "got: {tail:?}");
        }

        // type indices: noul 0, choice 1, score 2; option orders
        assert_eq!(record.questions[0].type_index, 0);
        assert_eq!(record.questions[1].type_index, 1);
        assert_eq!(record.questions[2].type_index, 2);
        assert_eq!(record.questions[0].option_ids, vec!["true", "false"]);
        assert_eq!(
            record.questions[1].option_ids,
            vec!["chat", "email", "phone"],
            "choice ids sorted"
        );
        assert_eq!(record.questions[2].option_ids, vec!["0", "1", "2"], "score ids index");
    }

    /// noul criteria override descriptions (never the order); score
    /// renders every level; the head-facing shapes stay exact.
    #[test]
    fn noul_and_score_option_shapes() {
        let overridden = ClefQuestion::Noul {
            instructions: Some("Escalated?".into()),
            criteria: BTreeMap::from([
                ("true".to_string(), "Yes escalate".to_string()),
                ("other".to_string(), "ignored".to_string()),
            ]),
        };
        assert_eq!(
            overridden.options(),
            vec![
                ("true".to_string(), Some("Yes escalate".to_string())),
                (
                    "false".to_string(),
                    Some("The proposition is false or the answer is no.".to_string())
                ),
            ],
            "criteria overrides true, false keeps its default, other keys ignored"
        );

        let levels = score(Some("How confident?"), &["l", "m", "h"]);
        assert_eq!(levels.options().len(), 3);
        assert_eq!(levels.options()[2], ("2".to_string(), Some("h".to_string())));

        // empty option sets refuse (empty choice criteria / score list)
        let tokenizer = lossless_tokenizer();
        let err = encode(
            &tokenizer,
            &json!({}),
            &[("empty".into(), choice(Some("?"), &[]))],
            2048,
        )
        .expect_err("empty choice refuses");
        assert!(
            matches!(err, RecordError::EmptyOptions { ref question } if question == "empty"),
            "got: {err}"
        );
        let err = encode(
            &tokenizer,
            &json!({}),
            &[("empty".into(), score(Some("?"), &[]))],
            2048,
        )
        .expect_err("empty score refuses");
        assert!(matches!(err, RecordError::EmptyOptions { .. }), "got: {err}");
    }

    /// The overflow walls: fixed > max_length refuses with the exact
    /// counts; an oversized STATE truncates from the right (and the
    /// spans shift by the TRUNCATED length); the image-placeholder
    /// literal refuses.
    #[test]
    fn truncation_and_refusal_walls() {
        let tokenizer = lossless_tokenizer();
        let state = json!({"tier": "gold"});
        let questions = vec![("q".into(), noul(Some("Ready?")))];

        // fixed-too-long: max_length below the fixed frame refuses
        let probe =
            encode(&tokenizer, &json!({}), &questions, 1_000_000).expect("empty state fits");
        let fixed = probe.input_ids.len()
            - tokenizer
                .encode(&py_json::render(&json!({})))
                .expect("empty state")
                .len();
        let err = encode(&tokenizer, &state, &questions, fixed - 1)
            .expect_err("fixed over the budget refuses");
        assert!(
            matches!(err, RecordError::TooLong { tokens, bucket } if tokens == fixed && bucket == fixed - 1),
            "got: {err}"
        );

        // exact fit with an EMPTY state passes; any state token
        // truncates away and the record equals an empty-state record at
        // the same budget (the state contributes nothing)
        let truncated = encode(&tokenizer, &state, &questions, fixed).expect("state truncates");
        assert_eq!(truncated.input_ids.len(), fixed, "state cut to zero tokens");
        let empty_at_budget =
            encode(&tokenizer, &json!({}), &questions, fixed).expect("empty state at exact fit");
        assert_eq!(
            truncated.input_ids, empty_at_budget.input_ids,
            "prefix + (no state tokens) + schema + suffix"
        );
        assert_eq!(
            truncated.questions, empty_at_budget.questions,
            "spans shift by the truncated (zero) state"
        );

        // a mid-size budget keeps the FIRST tokens of the state
        // (right truncation drops the tail) and fills the budget exactly;
        // the state lives right after the PREFIX (the frame is prefix +
        // schema + suffix, so the state slot is [prefix_len, ..))
        let budget = fixed + 3;
        let shifted = encode(&tokenizer, &state, &questions, budget).expect("truncates to 3");
        assert_eq!(shifted.input_ids.len(), budget);
        let kept = tokenizer
            .encode(&py_json::render(&state))
            .expect("state")
            .into_iter()
            .take(3)
            .collect::<Vec<_>>();
        let prefix_len = tokenizer
            .encode(&format!(
                "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
            ))
            .expect("prefix")
            .len();
        assert_eq!(
            &shifted.input_ids[prefix_len..prefix_len + 3],
            &kept[..],
            "first 3 survive (right truncation drops the tail)"
        );

        // image placeholder refuses
        let poisoned = json!({"note": "see <|image_pad|> below"});
        let err = encode(&tokenizer, &poisoned, &questions, 2048)
            .expect_err("image placeholder refuses");
        assert!(
            matches!(err, RecordError::ImagePlaceholder { count } if count == 1),
            "got: {err}"
        );
    }

    // ——— the golden fixtures (real pinned tokenizer) ———

    /// One fixture record: input spec + the frozen expectations.
    #[derive(Debug)]
    struct FixtureSpec {
        name: &'static str,
        /// `None` = the generator computes `fixed + 12` (a budget the
        /// state provably overflows) and freezes the value.
        max_length: Option<usize>,
        state: Value,
        questions: Vec<(String, ClefQuestion)>,
        questions_json: Value,
        truncates: bool,
    }

    fn q(id: &str, kind: &str, instructions: Option<&str>, criteria: Value) -> Value {
        serde_json::json!({
            "id": id,
            "kind": kind,
            "instructions": instructions,
            "criteria": criteria,
        })
    }

    fn parse_question(value: &Value) -> (String, ClefQuestion) {
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
                    "noul" => ClefQuestion::Noul { instructions, criteria },
                    _ => ClefQuestion::Choice { instructions, criteria },
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

    /// The six-plus records the plan names: plain state / nested JSON
    /// with floats+bools / multi-field / score levels / noul + unicode
    /// escapes / long-state truncation.
    fn fixture_specs() -> Vec<FixtureSpec> {
        fn build(
            name: &'static str,
            max_length: Option<usize>,
            state: Value,
            questions_json: Vec<Value>,
            truncates: bool,
        ) -> FixtureSpec {
            let questions: Vec<(String, ClefQuestion)> = questions_json
                .iter()
                .map(parse_question)
                .collect();
            FixtureSpec {
                name,
                max_length,
                state,
                questions,
                questions_json: Value::Array(questions_json),
                truncates,
            }
        }
        vec![
            build(
                "plain_state",
                Some(2048),
                serde_json::json!({"tier": "gold", "unread": 3}),
                vec![
                    q("tier_check", "noul", Some("Is the user gold tier?"), serde_json::json!({})),
                    q(
                        "channel",
                        "choice",
                        Some("Which channel?"),
                        serde_json::json!({"email": "Email support", "phone": "Phone support", "chat": "Live chat"}),
                    ),
                ],
                false,
            ),
            build(
                "nested_floats_bools",
                Some(2048),
                serde_json::json!({
                    "user": {"name": "Ada", "age": 29.5, "active": true, "score": 0.98, "tags": ["a", "b"]},
                    "balance": 1250.0,
                    "tiny": 0.0001,
                    "big": 1e15,
                }),
                vec![
                    q(
                        "mood",
                        "score",
                        Some("Rate satisfaction."),
                        serde_json::json!(["very poor", "poor", "ok", "good", "very good"]),
                    ),
                    q("adult", "noul", None, serde_json::json!({})),
                ],
                false,
            ),
            build(
                "multi_field",
                Some(2048),
                serde_json::json!({"ticket": "T-1001", "priority": 2.0, "escalated": false}),
                vec![
                    q(
                        "f1",
                        "noul",
                        Some("Escalated?"),
                        serde_json::json!({"true": "Yes escalate", "false": "Keep as is"}),
                    ),
                    q(
                        "f2",
                        "choice",
                        Some("Pick a lane."),
                        serde_json::json!({"b": "Lane B", "a": "Lane A"}),
                    ),
                    q("f3", "score", None, serde_json::json!(["low", "mid", "high"])),
                    q("f4", "choice", Some("Final"), serde_json::json!({"z": "Zee", "m": "Emm"})),
                ],
                false,
            ),
            build(
                "score_levels",
                Some(2048),
                serde_json::json!({"attempt": 7}),
                vec![q(
                    "confidence",
                    "score",
                    Some("How confident?"),
                    serde_json::json!(["none", "slight", "low", "lean", "mid", "fair", "good", "high", "very high", "certain"]),
                )],
                false,
            ),
            build(
                "noul_unicode_escapes",
                Some(2048),
                serde_json::json!({
                    "note": "héllo — 中文 \n \"quoted\" \\ back\ttab",
                    "amount": 12.25,
                }),
                vec![q("valid", "noul", Some("Valid note?"), serde_json::json!({}))],
                false,
            ),
            // The truncation record: the generator computes fixed+12 so
            // the 80-word state provably overflows and truncates.
            build(
                "long_state_truncation",
                None,
                serde_json::json!({
                    "blob": (0..80).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" "),
                    "keep": "end",
                }),
                vec![q("short", "noul", Some("Short?"), serde_json::json!({}))],
                true,
            ),
        ]
    }

    fn fixtures_path() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clef_text/records/records.json")
    }

    /// One-time generator for [`real_wire_record_goldens`]'s frozen
    /// fixtures: `CLEF_GEN_FIXTURES=1 cargo test -p riir-infer-clef
    /// --features clef-ane -- --ignored generate_record_fixtures`.
    /// Refuses without the real tokenizer — never fabricates a golden.
    #[test]
    #[ignore = "one-time fixture generator (CLEF_GEN_FIXTURES=1, real artifact tree required)"]
    fn generate_record_fixtures() {
        if std::env::var("CLEF_GEN_FIXTURES").as_deref() != Ok("1") {
            println!("SKIP: set CLEF_GEN_FIXTURES=1 to write the fixtures");
            return;
        }
        let dir = crate::lane::artifacts_dir().expect("artifact dir resolvable");
        let tokenizer = ClefTokenizer::from_dir(&dir).expect("real tokenizer loads");
        let mut records = Vec::new();
        for spec in fixture_specs() {
            let max_length = match spec.max_length {
                Some(length) => length,
                None => {
                    // fixed frame (probe over an unbounded budget) + 12
                    let probe = encode(&tokenizer, &json!({}), &spec.questions, usize::MAX)
                        .unwrap_or_else(|e| panic!("record {}: probe: {e}", spec.name));
                    let empty_tokens = tokenizer
                        .encode(&py_json::render(&json!({})))
                        .expect("empty state")
                        .len();
                    probe.input_ids.len() - empty_tokens + 12
                }
            };
            let record = encode(&tokenizer, &spec.state, &spec.questions, max_length)
                .unwrap_or_else(|e| panic!("record {}: {e}", spec.name));
            if spec.truncates {
                assert_eq!(
                    record.input_ids.len(),
                    max_length,
                    "truncation record {} must fill its budget exactly",
                    spec.name
                );
            }
            records.push(serde_json::json!({
                "name": spec.name,
                "max_length": max_length,
                "state": spec.state,
                "questions": spec.questions_json,
                "expected_input_ids": record.input_ids,
                "expected_questions": record.questions.iter().map(|question| serde_json::json!({
                    "id": question.id,
                    "type_index": question.type_index,
                    "span": [question.span.start, question.span.end],
                    "option_spans": question.option_spans.iter()
                        .map(|s| [s.start, s.end]).collect::<Vec<_>>(),
                    "option_ids": question.option_ids,
                })).collect::<Vec<_>>(),
            }));
            println!(
                "captured {} ({} tokens, max_length {max_length})",
                spec.name,
                record.input_ids.len()
            );
        }
        let document = serde_json::json!({
            "note": "Plan 619 T1.4 golden token-id fixtures — generated against the pinned \
                     artifact tokenizer.json (see riir-infer/.plans/619). Regenerate with \
                     CLEF_GEN_FIXTURES=1 cargo test -p riir-infer-clef --features clef-ane \
                     -- --ignored generate_record_fixtures",
            "records": records,
        });
        let path = std::path::Path::new(fixtures_path());
        std::fs::create_dir_all(path.parent().expect("parent")).expect("fixture dir");
        std::fs::write(path, serde_json::to_string_pretty(&document).expect("serializes"))
            .expect("fixtures written");
        println!("wrote {}", path.display());
    }

    /// The real pinned artifact's tokenizer drives the frozen goldens —
    /// exact token ids + spans, generated once by
    /// [`generate_record_fixtures`] and committed under
    /// `tests/fixtures/clef_text/records/`. `#[ignore]`d: it needs the
    /// artifact tree installed (the T1.3 real-wire posture — never an
    /// implicit fetch; absent material skips LOUD).
    #[test]
    #[ignore = "real wire: needs the pinned artifact tree installed"]
    fn real_wire_record_goldens() {
        let dir = match crate::lane::artifacts_dir() {
            Ok(dir) => dir,
            Err(e) => {
                println!("SKIP (loud): no artifact dir resolvable: {e}");
                return;
            }
        };
        if !dir.join("tokenizer.json").is_file() {
            println!(
                "SKIP (loud): tokenizer.json not installed under {} — run the T1.3 \
                 real_wire_install_and_load once",
                dir.display()
            );
            return;
        }
        let fixture_text = std::fs::read_to_string(fixtures_path())
            .expect("goldens committed under tests/fixtures/clef_text/records/");
        let document: Value = serde_json::from_str(&fixture_text).expect("fixture parses");
        let tokenizer = ClefTokenizer::from_dir(&dir).expect("real tokenizer loads");
        for record in document["records"].as_array().expect("records array") {
            let name = record["name"].as_str().expect("name");
            let max_length = record["max_length"].as_u64().expect("max_length") as usize;
            let state = &record["state"];
            let questions: Vec<(String, ClefQuestion)> = record["questions"]
                .as_array()
                .expect("questions")
                .iter()
                .map(parse_question)
                .collect();
            let expected_ids: Vec<u32> = record["expected_input_ids"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|v| v.as_u64().expect("u32 id") as u32)
                .collect();
            let expected_questions: Vec<ClefEncodedQuestion> = record["expected_questions"]
                .as_array()
                .expect("expected questions")
                .iter()
                .map(|question| ClefEncodedQuestion {
                    id: question["id"].as_str().expect("id").to_string(),
                    type_index: question["type_index"].as_u64().expect("type") as u32,
                    span: {
                        let span = question["span"].as_array().expect("span");
                        span[0].as_u64().expect("start") as usize..span[1].as_u64().expect("end") as usize
                    },
                    option_spans: question["option_spans"]
                        .as_array()
                        .expect("option spans")
                        .iter()
                        .map(|span| {
                            let pair = span.as_array().expect("pair");
                            pair[0].as_u64().expect("start") as usize
                                ..pair[1].as_u64().expect("end") as usize
                        })
                        .collect(),
                    option_ids: question["option_ids"]
                        .as_array()
                        .expect("option ids")
                        .iter()
                        .map(|v| v.as_str().expect("string").to_string())
                        .collect(),
                })
                .collect();

            let encoded = encode(&tokenizer, state, &questions, max_length)
                .unwrap_or_else(|e| panic!("record {name}: {e}"));
            assert_eq!(encoded.input_ids, expected_ids, "record {name}: token ids drifted");
            assert_eq!(encoded.questions, expected_questions, "record {name}: spans drifted");

            // legibility against the real BPE: untruncated records
            // decode to the exact reference text; a truncated record
            // keeps the full prefix + the token-cut state lead + the
            // VERBATIM schema+suffix tail, and fills its budget exactly
            // (token truncation cuts mid-word, so the state lead is
            // pinned by the token ids, not by character prefixing).
            let decoded = tokenizer.decode(&encoded.input_ids, false).expect("decodes");
            let full = expected_record_text(state, &questions);
            if decoded != full {
                // the complete-verbatim regions: the prefix (through
                // `STATE:\n`) leads, the schema+suffix tail follows —
                // the state between them is the token-cut middle
                let state_lead_end = full.find("STATE:\n").expect("state lead") + "STATE:\n".len();
                let schema_head = full.find("\n\nSCHEMA FIELDS:\n").expect("schema head in render");
                assert!(
                    decoded.starts_with(&full[..state_lead_end]),
                    "{name}: truncated record must keep the full prefix"
                );
                assert!(
                    decoded.ends_with(&full[schema_head..]),
                    "{name}: truncated record must keep schema + suffix verbatim"
                );
                assert_eq!(
                    expected_ids.len(),
                    max_length,
                    "{name}: truncation must fill the budget exactly"
                );
            } else {
                assert_eq!(decoded, full, "record {name}");
            }
            println!("golden {name} ok ({} tokens)", encoded.input_ids.len());
        }
    }
}
