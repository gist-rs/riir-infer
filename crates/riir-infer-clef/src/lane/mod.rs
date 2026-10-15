//! The clef-text lane body (feature `clef-ane`, macOS) — plan 619
//! Phase 1.
//!
//! Landed so far: the skeleton only (T1.1). The task groups below land
//! in order, each with its own gates — never ahead of its substrate:
//!
//! - **T1.2** artifact store — the `ensure_artifacts` law ON the shared
//!   host (download-on-demand to `~/.cache/riir-infer/
//!   clef-text-0.6b-coreml/`, env `CLEF_COREML_DIR` overrides; per-asset
//!   sha256 verify BEFORE use; `.verified-<rev>` marker; refuse loud on
//!   mismatch). Weights never committed (repo law).
//! - **T1.3** loaders — `config.json` (validated), `tokenizer.json` via
//!   the `tokenizers` crate (exact-reference), `embeddings.f16` mmap.
//! - **T1.4** record encoder — `ClefRecordEncoder` + Python-style JSON
//!   rendering (`sort_keys`, `(",",":")`, `ensure_ascii=False`, Python
//!   `repr(float)`); golden token-id fixtures frozen in
//!   `tests/fixtures/clef_text/records/`.
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
