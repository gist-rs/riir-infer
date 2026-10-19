//! The clef runtime — plan 619 T1.6. The multi-function host over the
//! shared [`riir_infer_ane`] machinery: one compiled `.mlmodelc` per
//! package (content-addressed by the bundle digest), a loaded
//! `MLModel` pair per bucket through `functionName("L{bucket}")`, and
//! the two forward legs the answer pipeline (T1.7) composes.
//!
//! **Thread law (the actor-law equivalent, documented at the type):**
//! one `MLModel` pair per [`ClefText`] instance, no shared synchronous
//! prediction — Core ML's sync `prediction` is treated as
//! single-threaded exactly as the reference runtime does (their Swift
//! type is an actor). Drive each instance from one thread.
//!
//! **Pins:** the two bundle digests are blake3-dir-v1 over the
//! assembled `.mlpackage` trees — derived once (2026-10-17) from the
//! T1.2 artifact store's tree, whose every file is individually
//! sha256-pinned at fetch. The dir digests are drift DETECTORS chained
//! to those primary pins: any byte drift in any package file changes
//! the dir digest and refuses the load.
//!
//! **Compute units:** `CLEF_COMPUTE_UNITS=ane|gpu|cpu` selects
//! `CPUAndNeuralEngine` (default) / `CPUAndGPU` / `CPUOnly` — the plan
//! T1.6 posture knob; the placement is disclosed at load, never a
//! silent fallback (the host walks the compute plan on every load).
//! On `cpu` the load is a DISCLOSED SPLIT (issue 048, 2026-10-17):
//! the decoder loads `CPUOnly` — the honest cpu leg — while the head
//! loads `CPUAndNe`, because the head package under `CPUOnly`
//! balloons to ~202 GB at load (a Core ML CPU-planner defect on the
//! head graph; jetsam SIGKILL in every host, a derived
//! single-function package included). The split is named in every
//! per-bucket placement line, `info()`, and the timing example; on a
//! cpu row decode is the honest cpu number, head/total are MIXED
//! numbers, and head-cpu is NOT-MEASURABLE-HERE. `ane`/`gpu` load
//! both packages under the requested posture (no split).

use std::collections::HashMap;

use riir_infer_ane::{ArtifactPin, ComputeUnits, LoadedModel, PlanWalk};

use super::host::{HeadInputs, causal_pad_mask, gather_hidden, head_inputs, rope_tables};
use super::loaders::{ClefConfig, ClefTokenizer, EmbeddingTable};
use super::record::{ClefEncodedRecord, ClefQuestion, encode};

/// The Decoder bundle pin (blake3-dir-v1; 3 files, 889,044,125 bytes —
/// the T1.2 sha256-verified tree, digested 2026-10-17).
const DECODER_PIN: PackagePin = PackagePin {
    digest: "1d6631c174c31148718dea3ebd40d8b3af92c6dab01727614bc914b48ed84eea",
    files: 3,
    bytes: 889_044_125,
};

/// The Head bundle pin (blake3-dir-v1; 3 files, 206,190,612 bytes).
const HEAD_PIN: PackagePin = PackagePin {
    digest: "72ed812d59a568183be2a94fb144f217f0feb834b1f702b4f5dd878068b61aa3",
    files: 3,
    bytes: 206_190_612,
};

struct PackagePin {
    digest: &'static str,
    files: usize,
    bytes: u64,
}

impl PackagePin {
    fn artifact_pin(&self) -> ArtifactPin {
        ArtifactPin {
            digest: self.digest.to_string(),
            files: self.files,
            bytes: self.bytes,
        }
    }
}

/// `CLEF_COMPUTE_UNITS` — `ane` (default) / `gpu` / `cpu`; an unknown
/// value refuses loud (never a silent ANE fallback). `cpu` is a SPLIT
/// posture (issue 048): decoder CpuOnly + head CpuAndNe — see the
/// module docs.
pub fn compute_units_from_env() -> ComputeUnits {
    match std::env::var("CLEF_COMPUTE_UNITS").as_deref() {
        Ok("ane") | Err(_) => ComputeUnits::CpuAndNe,
        Ok("gpu") => ComputeUnits::CpuAndGpu,
        Ok("cpu") => ComputeUnits::CpuOnly,
        Ok(other) => panic!(
            "clef: CLEF_COMPUTE_UNITS={other:?} is not ane|gpu|cpu — refusing (never a silent fallback)"
        ),
    }
}

/// The disclosure wording for a placement verdict ("not measured" for
/// a skipped walk — never a fabricated number).
fn placement_words(placement: riir_infer_ane::Placement) -> String {
    if placement.device_ops == 0 {
        "not-measured".to_string()
    } else {
        format!(
            "ane={} gpu={} cpu={} transitions={}",
            placement.ane_ops, placement.gpu_ops, placement.cpu_ops, placement.transitions
        )
    }
}

/// The loaded clef runtime (see the module docs for the thread law).
pub struct ClefText {
    /// The loaded buckets, ascending (the subset of the config's
    /// buckets this instance loaded).
    pub buckets: Vec<u32>,
    config: ClefConfig,
    decoders: HashMap<u32, LoadedModel>,
    heads: HashMap<u32, LoadedModel>,
    /// Per-bucket RoPE tables (fp16 cos, sin — raw bits), computed at
    /// load — the tables are bucket-shaped inputs, not state (the actor
    /// law keeps this mutation-free after construction).
    rope: HashMap<u32, (Vec<u16>, Vec<u16>)>,
    /// The load-time placement disclosures (one line per bucket).
    pub placements: Vec<String>,
    /// The record tokenizer (the reference manager owns it — T1.7's
    /// `answer` needs it, the plan's exact signature carries only
    /// state + questions).
    tokenizer: ClefTokenizer,
    /// The tied fp16 table (decoder gather + the head's lexical rows).
    embeddings: EmbeddingTable,
    /// The compute-unit posture this instance loaded under (the
    /// `info()` disclosure).
    units: ComputeUnits,
    /// The posture the HEAD packages actually loaded under (issue 048:
    /// a CpuOnly head load balloons ~202 GB, so a `cpu` request loads
    /// the head `CpuAndNe` — differs from `units` only on that split).
    head_units: ComputeUnits,
}

impl ClefText {
    /// Load `buckets` (a subset of the config's, refusing unknown ones)
    /// from the verified artifact tree under `dir`. Compiles each
    /// package once into the content-addressed cache
    /// (`CLEF_ANE_CACHE`, else `~/Library/Caches/riir-infer/clef-ane`).
    pub fn load_from(
        dir: &std::path::Path,
        config: ClefConfig,
        buckets: &[u32],
        units: ComputeUnits,
    ) -> Result<Self, ClefError> {
        // Issue 048 (2026-10-17): the HEAD package under CpuOnly balloons to
        // ~202 GB at load — a Core ML CPU-planner defect on the head graph
        // (jetsam SIGKILL in every host; a derived single-function package
        // included) while the sibling decoder loads + predicts fine CpuOnly.
        // The cpu posture therefore loads the decoder CpuOnly (the honest cpu
        // leg) and the head at its working CpuAndNe — a DISCLOSED split, never
        // a silent fallback; head/total on a cpu row are mixed numbers and
        // head-cpu is NOT-MEASURABLE-HERE.
        let head_units = match units {
            ComputeUnits::CpuOnly => ComputeUnits::CpuAndNe,
            other => other,
        };
        let head_note = if head_units != units {
            "; head loaded CpuAndNe — CpuOnly head load balloons ~202GB \
             (issue 048, Core ML planner defect); head-cpu NOT-MEASURABLE-HERE"
        } else {
            ""
        };
        // The two non-graph materials ride the same verified tree.
        let tokenizer = ClefTokenizer::from_dir(dir)?;
        let embeddings = EmbeddingTable::open(dir, &config)?;
        let decoder_dir = dir.join(&config.decoder);
        let head_dir = dir.join(&config.head);
        let decoder_pin = DECODER_PIN.artifact_pin();
        let head_pin = HEAD_PIN.artifact_pin();
        let mut decoders = HashMap::new();
        let mut heads = HashMap::new();
        let mut rope = HashMap::new();
        let mut placements = Vec::new();
        let mut loaded: Vec<u32> = Vec::new();
        for &bucket in buckets {
            if !config.buckets.contains(&bucket) {
                return Err(ClefError::UnknownBucket {
                    bucket,
                    available: config.buckets.clone(),
                });
            }
            let function = format!("L{bucket}");
            // PlanWalk::Skip: the GB-scale multi-function package's
            // compute-plan walk never completes at load time (measured,
            // 120 s wall) — the placement truth for this lane is the
            // T2.2 parity gate + the T2.3 latency bench, and the
            // SKIPPED placement never reads as a pass.
            let (decoder, decoder_placement) = riir_infer_ane::load_model(
                &decoder_dir,
                &decoder_pin,
                units,
                Some(&function),
                "CLEF_ANE_CACHE",
                "clef-ane",
                "run the plan 619 clef fetcher (ensure_and_load) first",
                PlanWalk::Skip,
            )
            .map_err(ClefError::CoreMl)?;
            let (head, head_placement) = riir_infer_ane::load_model(
                &head_dir,
                &head_pin,
                head_units,
                Some(&function),
                "CLEF_ANE_CACHE",
                "clef-ane",
                "run the plan 619 clef fetcher (ensure_and_load) first",
                PlanWalk::Skip,
            )
            .map_err(ClefError::CoreMl)?;
            let (cos, sin) =
                rope_tables(bucket as usize, config.head_dim as usize, config.rope_theta);
            // fp16 values ride the wire as raw bits (Float16 MLMultiArrays).
            let cos_bits = cos.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let sin_bits = sin.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            placements.push(format!(
                "clef: L{bucket} decoder placement {} (head {}{}) — plan walk skipped \
                 (GB-scale package); placement truth = T2.2 parity + T2.3 latency",
                placement_words(decoder_placement),
                placement_words(head_placement),
                head_note,
            ));
            decoders.insert(bucket, decoder);
            heads.insert(bucket, head);
            rope.insert(bucket, (cos_bits, sin_bits));
            loaded.push(bucket);
        }
        loaded.sort_unstable();
        Ok(Self {
            buckets: loaded,
            config,
            decoders,
            heads,
            rope,
            placements,
            tokenizer,
            embeddings,
            units,
            head_units,
        })
    }

    /// The artifact-law entry: `ensure_and_load` (fetch + verify if
    /// needed) then [`Self::load_from`] over all the config's buckets.
    pub fn ensure_and_load(units: ComputeUnits) -> Result<Self, ClefError> {
        let dir = super::artifacts_dir()?;
        let model = super::ensure_and_load(&dir)?;
        let buckets = model.config.buckets.clone();
        Self::load_from(&dir, model.config, &buckets, units)
    }

    /// The decoder leg: bucket select (first loaded bucket ≥ n), the
    /// T1.5 gather + cached RoPE + mask, one fp16 forward, and the
    /// strided fp16 → f32 states (`[L·D]`, the head's input posture).
    /// Returns `(bucket, states)`.
    pub fn decode_forward(
        &self,
        record: &ClefEncodedRecord,
        embeddings: &EmbeddingTable,
    ) -> Result<(u32, Vec<f32>), ClefError> {
        let n = record.input_ids.len();
        let bucket = self
            .buckets
            .iter()
            .find(|b| (**b) as usize >= n)
            .copied()
            .ok_or(ClefError::TooLong {
                tokens: n,
                bucket: self.buckets.last().copied().unwrap_or(0),
            })?;
        let l = bucket as usize;
        let d = self.config.hidden_size as usize;
        let head_dim = self.config.head_dim as usize;
        let (cos, sin) = &self.rope[&bucket];
        // The fp16 [1, L, D] decoder input: raw rows, pad = pad_id.
        let hidden_u16 = gather_hidden(record, l, self.config.pad_id, embeddings);
        let mask_bits = causal_pad_mask(l, n)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        debug_assert_eq!(cos.len(), l * head_dim);
        let decoder = self
            .decoders
            .get(&bucket)
            .expect("bucket loaded with its decoder");
        let mut states_u16 = vec![0u16; l * d];
        decoder
            .predict_f16(
                &[
                    ("hidden", &hidden_u16, &[1, l, d]),
                    ("cos", cos, &[l, head_dim]),
                    ("sin", sin, &[l, head_dim]),
                    ("mask", &mask_bits, &[1, 1, l, l]),
                ],
                "states",
                &mut states_u16,
            )
            .map_err(ClefError::CoreMl)?;
        // fp16 → f32 widening (bit-exact; the strided read is the
        // host's — the array's own strides rule).
        let states = states_u16
            .iter()
            .map(|bits| f32::from(half::f16::from_bits(*bits)))
            .collect();
        Ok((bucket, states))
    }

    /// The head leg: the T1.5 packing over `states` → one f32 forward →
    /// the packed logits `[Σ options]` (the packing's flattened option
    /// order — T1.7's softmax walk).
    pub fn head_forward(
        &self,
        bucket: u32,
        record: &ClefEncodedRecord,
        states: &[f32],
        embeddings: &EmbeddingTable,
    ) -> Result<(Vec<f32>, HeadInputs), ClefError> {
        let l = bucket as usize;
        let d = self.config.hidden_size as usize;
        debug_assert_eq!(states.len(), l * d, "states must be [L·D]");
        let packed = head_inputs(
            record,
            l,
            self.config.max_questions as usize,
            self.config.max_options as usize,
            embeddings,
        )?;
        let head = self
            .heads
            .get(&bucket)
            .expect("bucket loaded with its head");
        // the head answers the PADDED [maxO] logits array; the packed
        // prefix (option_count entries) is the real answer (the
        // reference reads the same prefix, offset per question)
        let mut logits_full = vec![0f32; self.config.max_options as usize];
        head.predict_f32(
            &[
                ("states", states, &[l, d]),
                ("mem_mask", &packed.mem_mask, &[l]),
                ("last", &packed.last, &[l]),
                (
                    "q_mean",
                    &packed.q_mean,
                    &[self.config.max_questions as usize, l],
                ),
                (
                    "o_mean",
                    &packed.o_mean,
                    &[self.config.max_options as usize, l],
                ),
                (
                    "o2q",
                    &packed.o2q,
                    &[
                        self.config.max_options as usize,
                        self.config.max_questions as usize,
                    ],
                ),
                (
                    "lexical",
                    &packed.lexical,
                    &[self.config.max_options as usize, d],
                ),
                (
                    "type_oh",
                    &packed.type_oh,
                    &[self.config.max_questions as usize, 3],
                ),
                (
                    "q_mask",
                    &packed.q_mask,
                    &[self.config.max_questions as usize],
                ),
                (
                    "o_mask",
                    &packed.o_mask,
                    &[self.config.max_options as usize],
                ),
            ],
            "logits",
            &mut logits_full,
        )
        .map_err(ClefError::CoreMl)?;
        logits_full.truncate(packed.option_count);
        Ok((logits_full, packed))
    }

    pub fn config(&self) -> &ClefConfig {
        &self.config
    }

    /// The lane posture disclosure: model, artifact revision, loaded
    /// buckets, capacities, compute units — one line, for run logs and
    /// the reflex lane's posture string.
    pub fn info(&self) -> String {
        let units_part = if self.head_units != self.units {
            format!(
                "{:?} (head {:?} — issue 048 split; head-cpu not measurable)",
                self.units, self.head_units
            )
        } else {
            format!("{:?}", self.units)
        };
        format!(
            "{} @ {} — buckets {:?} (max_questions {}, max_options {}) compute_units {}",
            self.config.model,
            super::ARTIFACT_REVISION,
            self.buckets,
            self.config.max_questions,
            self.config.max_options,
            units_part,
        )
    }

    /// Decide `questions` about `state`: encode (the record budget is
    /// the largest LOADED bucket — the reference posture) → bucket
    /// select → decoder → head → per-question softmax over that
    /// question's options (the artifact's own head convention —
    /// fidelity, not our sigmoid law; our gates apply to OUR
    /// projections, never this reproduction). Answers follow the
    /// question order; probabilities follow Clef's option order.
    pub fn answer(
        &self,
        state: &serde_json::Value,
        questions: &[(String, ClefQuestion)],
    ) -> Result<Vec<Answer>, ClefError> {
        let max_loaded = self.buckets.last().copied().unwrap_or(0) as usize;
        let record = encode(&self.tokenizer, state, questions, max_loaded)?;
        let (bucket, states) = self.decode_forward(&record, &self.embeddings)?;
        let (logits, packed) = self.head_forward(bucket, &record, &states, &self.embeddings)?;
        Ok(answers_from_logits(&record, &logits, packed.option_count))
    }
}

/// One question's answer: probability per option id in Clef's option
/// order (the `ClefAnswer` port).
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    pub question_id: String,
    pub option_ids: Vec<String>,
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
}

impl Answer {
    /// Highest-probability option id (questions always have at least
    /// one option — the encoder refuses empty ones).
    pub fn choice(&self) -> &str {
        let best = self
            .probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        &self.option_ids[best]
    }

    /// For `noul` questions: probability of `true`.
    pub fn noul(&self) -> Option<f32> {
        self.option_ids
            .iter()
            .position(|id| id == "true")
            .map(|i| self.probabilities[i])
    }

    /// For `score` questions: the expected index (Σ i·pᵢ).
    pub fn score(&self) -> f32 {
        self.probabilities
            .iter()
            .enumerate()
            .map(|(i, p)| i as f32 * p)
            .sum()
    }
}

/// The per-question softmax over the packed logits (peak-subtracted —
/// the reference's expf walk; slice order = the packing's flattened
/// option order). Split out pure for the always-on unit test.
fn answers_from_logits(
    record: &ClefEncodedRecord,
    logits: &[f32],
    option_count: usize,
) -> Vec<Answer> {
    let mut answers = Vec::with_capacity(record.questions.len());
    let mut offset = 0usize;
    for question in &record.questions {
        let count = question.option_ids.len();
        let slice = &logits[offset..offset + count];
        let peak = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = slice.iter().map(|v| (v - peak).exp()).collect();
        let sum: f32 = exps.iter().sum();
        answers.push(Answer {
            question_id: question.id.clone(),
            option_ids: question.option_ids.clone(),
            logits: slice.to_vec(),
            probabilities: exps.iter().map(|e| e / sum).collect(),
        });
        offset += count;
    }
    debug_assert_eq!(offset, option_count);
    answers
}

/// The runtime-level errors (T1.7 composes these with the record and
/// host-math walls).
#[derive(Debug)]
pub enum ClefError {
    /// No loaded bucket fits the record.
    TooLong { tokens: usize, bucket: u32 },
    /// A bucket outside the artifact's config was requested.
    UnknownBucket { bucket: u32, available: Vec<u32> },
    /// The Core ML host refused a load or a prediction.
    CoreMl(riir_infer_ane::Error),
    /// The head packing refused (capacity / shape walls).
    Host(super::host::HostError),
    /// The record encoder refused.
    Record(super::record::RecordError),
    /// Artifact-store / loader I/O.
    Io(std::io::Error),
}

impl std::fmt::Display for ClefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLong { tokens, bucket } => {
                write!(
                    f,
                    "Record needs {tokens} tokens; the largest loaded bucket holds {bucket}"
                )
            }
            Self::UnknownBucket { bucket, available } => {
                write!(
                    f,
                    "clef: no {bucket}-token bucket in the artifact (config carries {available:?})"
                )
            }
            Self::CoreMl(e) => write!(f, "clef: core ml: {e}"),
            Self::Host(e) => write!(f, "clef: host math: {e}"),
            Self::Record(e) => write!(f, "clef: record: {e}"),
            Self::Io(e) => write!(f, "clef: io: {e}"),
        }
    }
}

impl std::error::Error for ClefError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CoreMl(e) => Some(e),
            Self::Host(e) => Some(e),
            Self::Record(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::TooLong { .. } | Self::UnknownBucket { .. } => None,
        }
    }
}

impl From<super::host::HostError> for ClefError {
    fn from(e: super::host::HostError) -> Self {
        Self::Host(e)
    }
}

impl From<super::record::RecordError> for ClefError {
    fn from(e: super::record::RecordError) -> Self {
        Self::Record(e)
    }
}

impl From<std::io::Error> for ClefError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lane::loaders::{ClefTokenizer, load_from};
    use crate::lane::record::{ClefQuestion, encode};
    use crate::lane::test_support::parse_fixture_question;
    use serde_json::{Value, json};
    use std::collections::BTreeMap;

    fn plain_state_record(tokenizer: &ClefTokenizer) -> crate::lane::record::ClefEncodedRecord {
        let fixture_text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/clef_text/records/records.json"
        ))
        .expect("record goldens committed");
        let document: Value = serde_json::from_str(&fixture_text).expect("parses");
        let record_json = document["records"]
            .as_array()
            .expect("records")
            .iter()
            .find(|r| r["name"].as_str() == Some("plain_state"))
            .expect("plain_state golden")
            .clone();
        let questions: Vec<(String, ClefQuestion)> = record_json["questions"]
            .as_array()
            .expect("questions")
            .iter()
            .map(parse_fixture_question)
            .collect();
        encode(
            tokenizer,
            &record_json["state"].clone(),
            &questions,
            record_json["max_length"].as_u64().expect("max_length") as usize,
        )
        .expect("encodes")
    }

    /// The REAL-WIRE T1.6 gate — `#[ignore]`d (needs the artifact tree
    /// and a first CoreML compile of both packages, minutes on a cold
    /// cache; the content-addressed cache makes every later run fast).
    ///
    /// Proves: both packages load through `functionName("L{bucket}")`
    /// at every bucket, the placement is disclosed, and one full
    /// decode→head forward over the reference record yields finite
    /// logits in the packed layout.
    #[test]
    #[ignore = "real wire: needs the pinned artifact tree + a first compile (cached after)"]
    fn real_wire_multifunction_load_and_forward() {
        let dir = match crate::lane::artifacts_dir() {
            Ok(dir) => dir,
            Err(e) => {
                println!("SKIP (loud): no artifact dir resolvable: {e}");
                return;
            }
        };
        if !dir.join("config.json").is_file() {
            println!(
                "SKIP (loud): artifact tree not installed under {} — run the T1.3 install once",
                dir.display()
            );
            return;
        }
        let model = load_from(&dir).expect("three materials load");
        let started = std::time::Instant::now();
        let runtime = ClefText::load_from(
            &dir,
            model.config.clone(),
            &model.config.buckets,
            compute_units_from_env(),
        )
        .expect("loads through the multi-function host");
        println!(
            "DISCLOSURE: loaded buckets {:?} in {:?} (cold compile dominates the first run)",
            runtime.buckets,
            started.elapsed()
        );
        for line in &runtime.placements {
            println!("DISCLOSURE: {line}");
        }
        println!("DISCLOSURE: {}", runtime.info());
        // the reference record forward — bucket select, decode, head
        let record = plain_state_record(&model.tokenizer);
        let (bucket, states) = runtime
            .decode_forward(&record, &model.embeddings)
            .expect("decode forward");
        let l = bucket as usize;
        let d = model.config.hidden_size as usize;
        assert_eq!(states.len(), l * d, "states [L·D]");
        assert!(states.iter().all(|v| v.is_finite()), "states finite");
        let (logits, packed) = runtime
            .head_forward(bucket, &record, &states, &model.embeddings)
            .expect("head forward");
        assert_eq!(logits.len(), packed.option_count, "packed logits layout");
        assert!(logits.iter().all(|v| v.is_finite()), "logits finite");
        println!(
            "DISCLOSURE: L{bucket} forward over {} tokens: {} options, logits[0..4] = {:?}",
            record.input_ids.len(),
            packed.option_count,
            &logits[..logits.len().min(4)]
        );
        // determinism: a second identical forward must agree bit-for-bit
        let (bucket2, states2) = runtime
            .decode_forward(&record, &model.embeddings)
            .expect("second decode");
        assert_eq!(bucket2, bucket);
        assert_eq!(states, states2, "decode must be deterministic");
        let (logits2, _) = runtime
            .head_forward(bucket, &record, &states2, &model.embeddings)
            .expect("second head");
        assert_eq!(logits, logits2, "head must be deterministic");
        // a state string that never crosses into a question token —
        // the softmax-ability of every question's option slice
        let mut offset = 0usize;
        for question in &record.questions {
            let slice = &logits[offset..offset + question.option_ids.len()];
            let peak = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = slice.iter().map(|v| (v - peak).exp()).collect();
            let sum: f32 = exps.iter().sum();
            assert!(sum.is_finite() && sum > 0.0, "softmax per question");
            offset += question.option_ids.len();
        }
        assert_eq!(offset, packed.option_count);
        let _ = json!({}); // (serde_json in scope for the fixture parse)
        let _ = BTreeMap::<String, String>::new();
    }

    /// The T1.7 REAL-WIRE answer gate — the full pipeline through the
    /// plan's exact signature (`answer(state, questions)`): answers
    /// follow the question order, each probability row sums to 1,
    /// choice/noul/score readouts are consistent, and a second call is
    /// bit-identical. `#[ignore]`d (the same real-wire posture).
    #[test]
    #[ignore = "real wire: needs the pinned artifact tree + a first compile (cached after)"]
    fn real_wire_answer_pipeline() {
        let dir = match crate::lane::artifacts_dir() {
            Ok(dir) => dir,
            Err(e) => {
                println!("SKIP (loud): no artifact dir resolvable: {e}");
                return;
            }
        };
        if !dir.join("config.json").is_file() {
            println!(
                "SKIP (loud): artifact tree not installed under {} — run the T1.3 install once",
                dir.display()
            );
            return;
        }
        let runtime = ClefText::load_from(
            &dir,
            crate::lane::ClefConfig::from_dir(&dir).expect("config"),
            &[256, 512],
            compute_units_from_env(),
        )
        .expect("L256+L512 load (the reference default posture)");
        let state = json!({"tier": "gold", "unread": 3});
        let questions: Vec<(String, ClefQuestion)> = vec![
            (
                "tier_check".into(),
                ClefQuestion::Noul {
                    instructions: Some("Is the user gold tier?".into()),
                    criteria: BTreeMap::new(),
                },
            ),
            (
                "channel".into(),
                ClefQuestion::Choice {
                    instructions: Some("Which channel?".into()),
                    criteria: BTreeMap::from([
                        ("email".to_string(), "Email support".to_string()),
                        ("phone".to_string(), "Phone support".to_string()),
                        ("chat".to_string(), "Live chat".to_string()),
                    ]),
                },
            ),
            (
                "mood".into(),
                ClefQuestion::Score {
                    instructions: Some("Rate satisfaction.".into()),
                    criteria: vec![
                        "very poor".into(),
                        "poor".into(),
                        "ok".into(),
                        "good".into(),
                        "very good".into(),
                    ],
                },
            ),
        ];
        let answers = runtime.answer(&state, &questions).expect("answers");
        assert_eq!(answers.len(), 3, "one answer per question, in order");
        for (answer, (id, _)) in answers.iter().zip(&questions) {
            assert_eq!(answer.question_id, *id);
            assert!(answer.logits.iter().all(|v| v.is_finite()));
            let sum: f32 = answer.probabilities.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "{id}: probabilities sum to 1, got {sum}"
            );
            assert_eq!(answer.option_ids.len(), answer.probabilities.len());
        }
        // the typed readouts are consistent with the probabilities
        let noul = &answers[0];
        assert_eq!(noul.option_ids, vec!["true", "false"]);
        assert_eq!(noul.noul(), Some(noul.probabilities[0]));
        let choice = &answers[1];
        assert!(choice.option_ids.contains(&choice.choice().to_string()));
        let best = choice
            .probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(choice.option_ids[best], choice.choice());
        let score = &answers[2];
        assert_eq!(score.option_ids, vec!["0", "1", "2", "3", "4"]);
        let expected: f32 = score
            .probabilities
            .iter()
            .enumerate()
            .map(|(i, p)| i as f32 * p)
            .sum();
        assert!((score.score() - expected).abs() < 1e-6, "score readout");
        println!(
            "DISCLOSURE: answers = {:?}",
            answers
                .iter()
                .map(|a| (a.question_id.clone(), a.choice().to_string()))
                .collect::<Vec<_>>()
        );
        // determinism: a second identical call is bit-identical
        let again = runtime.answer(&state, &questions).expect("answers 2");
        assert_eq!(answers, again, "answer must be deterministic");
    }

    /// The always-on softmax law (no CoreML): peak subtraction keeps
    /// the exponentials finite, each row sums to 1, and the option
    /// order is preserved.
    #[test]
    fn softmax_law_on_packed_logits() {
        let record = crate::lane::record::ClefEncodedRecord {
            input_ids: Vec::new(),
            questions: vec![
                crate::lane::record::ClefEncodedQuestion {
                    id: "a".into(),
                    type_index: 0,
                    span: 0..0,
                    option_spans: Vec::new(),
                    option_ids: vec!["true".into(), "false".into()],
                },
                crate::lane::record::ClefEncodedQuestion {
                    id: "b".into(),
                    type_index: 2,
                    span: 0..0,
                    option_spans: Vec::new(),
                    option_ids: vec!["0".into(), "1".into(), "2".into()],
                },
            ],
        };
        let logits = vec![0.0, 0.0, 10.0, 20.0, 30.0];
        let answers = answers_from_logits(&record, &logits, 5);
        assert_eq!(answers.len(), 2);
        assert_eq!(
            answers[0].probabilities,
            vec![0.5, 0.5],
            "ties split evenly"
        );
        let sum: f32 = answers[1].probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // peak subtraction: huge logits stay finite (10/20/30 in f32
        // would be e^30 ≈ 1e13 — fine, but e^100 would be inf)
        let big = vec![0.0, 100.0, 100.0];
        let record_b = crate::lane::record::ClefEncodedRecord {
            questions: vec![record.questions[1].clone()],
            ..record
        };
        let answers_b = answers_from_logits(&record_b, &big, 3);
        assert!(answers_b[0].probabilities.iter().all(|p| p.is_finite()));
        assert!(
            (answers_b[0].probabilities[1] - 0.5).abs() < 1e-6,
            "shift invariance"
        );
        // option order preserved
        assert_eq!(answers_b[0].option_ids, vec!["0", "1", "2"]);
    }
}
