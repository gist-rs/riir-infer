//! The clef-text lane body (feature `clef-ane`, macOS) — plan 619
//! Phase 1.
//!
//! Landed so far: the skeleton only (T1.1). The task groups below land
//! in order, each with its own gates — never ahead of its substrate:
//!
//! - **T1.2** artifact store — **LANDED** (`lane/artifacts.rs`): the
//!   `ensure_artifacts` law ON the shared host (download-on-demand to
//!   `~/.cache/riir-infer/clef-text-0.6b-coreml/`, env `CLEF_COREML_DIR`
//!   overrides; per-asset sha256 verify BEFORE use; `.verified-<rev>`
//!   marker; refuse loud on mismatch — never a silent re-download). The
//!   9-asset sha256/size manifest is FULL (re-derived from the HF tree
//!   API at the pinned revision, cross-checked vs the FluidUse
//!   prefixes). Weights never committed (repo law).
//! - **T1.3** loaders — **LANDED** (`lane/loaders.rs`): `config.json`
//!   parsed typed + validated (buckets sorted ⊆ {256,512,1024,2048},
//!   max_questions/max_options pinned 16/96, `tied_embeddings` true —
//!   the head's lexical input is the tied rows; dims positive);
//!   `tokenizer.json` through the `tokenizers` crate (exact-reference,
//!   `add_special_tokens = false` — the encoder carries its own
//!   markers); `embeddings.f16` mmapped fp16 `[vocab, hidden]` with the
//!   size wall (bytes == vocab·hidden·2), raw-bit row access for the
//!   decoder gather + bit-exact f32 widening for the head's lexical
//!   mean. `ClefModel::ensure_and_load` = the artifact law + all three
//!   materials in one call.
//! - **T1.4** record encoder — **LANDED** (`lane/record.rs` +
//!   `lane/py_json.rs`): the `ClefRecordEncoder` fidelity port (fixed
//!   system prompt, `SCHEMA FIELDS` layout, per-field
//!   ID/TYPE/INSTRUCTION + `ALLOWED OPTIONS` + `OPTION k` JSON + `END
//!   FIELD`, the `<|im_start|>`/`<think>` prefix and suffix; right
//!   truncation of state with spans shifted by prefix + truncated
//!   state; typed `RecordError` walls — `TooLong` / `EmptyOptions` /
//!   the `<|image_pad|>` placeholder refusal) + the Python-style JSON
//!   rendering (`py_json`: `sort_keys`, `(",",":")` separators,
//!   `ensure_ascii=False`, CPython `repr(float)` — validated against
//!   real-CPython-generated literals) + `ClefQuestion` (noul true/false
//!   defaults with criteria override, choice sorted ids, score
//!   `"0".."N"`). Golden token-id fixtures frozen in
//!   `tests/fixtures/clef_text/records/records.json` — generated
//!   against the REAL pinned `tokenizer.json` (the `#[ignore]`d
//!   generator); structural tests run always-on over an in-code
//!   lossless byte-level BPE tokenizer.
//! - **T1.5** host math — RoPE tables fp16, causal+pad mask, embedding
//!   gather, span-mean head packing (port `ClefTextManager`'s statics).
//! - **T1.6** the multi-function host — compile-cache both packages,
//!   per-bucket `functionName("L{bucket}")` + env-selectable compute
//!   units (`CLEF_COMPUTE_UNITS=ane|gpu|cpu`), one `MLModel` pair per
//!   runtime instance (the actor/thread law, documented at the type).
//! - **T1.7** `ClefText::answer` — encode → bucket select → decoder →
//!   head → per-question softmax over that question's options (the
//!   artifact's own head convention — fidelity, not our sigmoid law).
//!
//! Phase 2 is the parity gate (T2.1 frozen captures, T2.2 the
//! decision-level gate — zero top-1 disagreements outside the near-tie
//! band, counted PER QUESTION — T2.3 the latency bench with the
//! preflight-quoted box state); Phase 3 is the reflex local posture
//! (executed in riir-reflex).

pub mod artifacts;
pub mod loaders;
pub mod py_json;
pub mod record;

pub use artifacts::{
    ARTIFACT_REPO, ARTIFACT_REVISION, ARTIFACTS, artifacts_dir, ensure_artifacts, hf_fetch,
};
pub use loaders::{
    BUCKETS_ALLOWED, ClefConfig, ClefModel, ClefTokenizer, EmbeddingTable, PINNED_MAX_OPTIONS,
    PINNED_MAX_QUESTIONS, ensure_and_load, load_from,
};
pub use py_json::{dump, python_float, render};
pub use record::{
    SYSTEM_PROMPT, ClefEncodedQuestion, ClefEncodedRecord, ClefQuestion, RecordError, encode,
};
