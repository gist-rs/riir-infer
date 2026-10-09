//! The Apple Neural Engine (ANE) whole-graph encoder backend — feature
//! `laya-riir-ane`, macOS-only, opt-in, never wasm32, never a default.
//!
//! The encoder forward executes ONE Core ML `.mlpackage` per length bucket
//! (the BC1S graphs the consumer repo's offline conversion produced —
//! reflex Plan 002 P0), instead of the per-op [`super::backend::Backend`]
//! op stream the CPU/Metal lanes share. The manifest is the contract: the
//! artifact digest (blake3-dir-v1), the fixed input/output shapes, the
//! per-bucket output tensor NAME, and the conversion-time compute-plan
//! gate (100% ANE-preferred, 0 device transitions under `CPU_AND_NE`) are
//! all pinned there, and this runtime RE-VERIFIES at load:
//!
//! 1. the artifact bytes match the pinned digest (download-on-demand
//!    scope, issue 017 — a swapped artifact refuses, never loads);
//! 2. the plan re-walked HERE under `CPU_AND_NE` reads the same verdict —
//!    every device-bearing op ANE-preferred, 0 device transitions. A
//!    conversion-time-only gate would trust the artifact forever; this
//!    one refuses when the OS re-plans (the no-silent-fallback law).
//!
//! Numerics: the artifact is FP16 end to end — the input is the HOST-side
//! token gather (the vocab-sized gather op is the one op the ANE compiler
//! will not place — the conversion finding that shaped the artifact I/O),
//! the pad tail is filled with the PAD token's row exactly like the
//! reference smoke, `pad_bias` masks the tail (`0.0` attend / `-1e4`
//! masked, per key position, so `[0, n)` can never attend `[n, L)`), and
//! the fp16 output is sliced `[:n]`, widened bit-exactly to f32, and fed
//! to the UNCHANGED f32 [`super::head::Head`]. The lane therefore fails
//! the Metal lane's 1e-3 p-drift bar by nature of fp16 math; its
//! authority is the consumer-side G5-ANE decision-level parity gate
//! (top-1 agreement + the near-tie band, prob err published as
//! observation).
//!
//! Everything Core ML touches runs inside an autoreleasepool (the
//! [`super::super`] agent's `pass_pool` wraps one forward; load wraps its
//! own).
//!
//! Table postures (Plan 612): the host-side gather reads a RESIDENT
//! table whose bytes come in two flavors — fp16 (default, converted
//! from the checkpoint at load) and the `e8` per-row int8 sidecar
//! (`LAYA_ANE_TABLE=e8`, BLAKE3-verified against the manifest at load,
//! dequantized `i8 · scale → f32 → f16 bits` in [`gather_e8`]). The
//! artifact input contract is unchanged bytes either way — the posture
//! is invisible downstream of the gather, and a set-but-unavailable
//! e8 sidecar refuses loud, never falls back.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::rc::Rc;

use block2::RcBlock;
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputePlan, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLModelStructureProgram, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};

use super::super::config::EncoderConfig;
use super::super::{LayaError, Result};
use super::weights::Weights;

/// The compute-unit posture every gate in this module runs under — the
/// conversion gate's `ComputeUnit.CPU_AND_NE` (ANE-preferred, GPU
/// excluded). `MLComputeUnits` is an NS_OPTIONS bit set over an NSInteger.
const COMPUTE_UNITS_CPU_AND_NE: objc2_core_ml::MLComputeUnits =
    objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine;

/// The fp16 stand-in for f32::MIN the conversion baked (the sentinel is a
/// manifest field — `mask_sentinel_fp16` — and the runtime reads it from
/// there rather than restating it).
const FP16_NEG_INF_MASK: f32 = -10_000.0;

/// One manifest entry (one artifact). Fields the runtime CONSUMES only —
/// the manifest carries extras (histograms, box state, tool versions)
/// this side deliberately ignores.
#[derive(Debug, Clone)]
pub struct AneArtifact {
    /// The bucket's fixed sequence length (`bucket_L`).
    pub bucket_l: usize,
    /// The hidden width (`geometry.hidden` — the encoder's `d`).
    pub hidden: usize,
    /// blake3-dir-v1 hex digest of the `.mlpackage` bundle contents.
    pub digest: String,
    /// Pinned file count inside the bundle (digest cross-check).
    pub digest_files: usize,
    /// Pinned byte total inside the bundle (digest cross-check).
    pub digest_bytes: u64,
    /// The per-bucket output tensor name (e.g. `var_5377` — coremltools
    /// names traced ops; the manifest is the only stable home for it).
    pub output_name: String,
    /// Conversion-time gate counts, re-checked against the load-time plan
    /// walk (a count equality is WARNED on mismatch, never fatal — an OS
    /// update may renumber ops; the VERDICT below is the gate).
    pub ane_ops: usize,
    pub device_ops: usize,
    pub transitions: usize,
    /// The fp16 mask sentinel the artifact was converted with.
    pub mask_sentinel: f32,
}

/// The parsed `manifest.json` (committed at `assets/ane/manifest.json` in
/// the consumer repo; key format `<model-dir>/L<bucket>` where the model
/// dir IS [`crate::laya::config::Checkpoint::subfolder`]).
#[derive(Debug, Clone)]
pub struct AneManifest {
    artifacts: HashMap<String, AneArtifact>,
    /// The raw `<model>/table_e8` rows — retained unparsed (the e8
    /// posture's [`Self::table_e8`] parses + validates on demand, so the
    /// default fp16 posture never touches the sidecar schema).
    table_e8_rows: HashMap<String, serde_json::Value>,
}

impl AneManifest {
    /// Parse the committed manifest. Loose JSON: only the consumed fields
    /// are read, so the conversion tool can grow the schema freely.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            LayaError::Runtime(format!(
                "ane manifest unreadable at {}: {e}",
                path.display()
            ))
        })?;
        let raw: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| LayaError::Runtime(format!("ane manifest is not JSON: {e}")))?;
        let arts = raw
            .get("artifacts")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| LayaError::Runtime("ane manifest: missing `artifacts`".into()))?;
        let mut artifacts = HashMap::new();
        let mut table_e8_rows = HashMap::new();
        for (key, a) in arts {
            // The manifest is the conversion tool's file and carries rows
            // beyond this lane's bucket artifacts. `<model>/table_e8` rows
            // are THIS lane's Plan 612 Phase 1 int8 sidecars (the e8
            // posture's consumer lives here) — retained RAW and parsed
            // only when the e8 posture selects them, so the fp16 posture
            // never validates the sidecar schema (the "tool may grow the
            // schema freely" contract). Any other foreign row is skipped,
            // never validated. A malformed ARTIFACT row still errors
            // below.
            if !is_bucket_artifact_key(key) {
                if key
                    .rsplit_once('/')
                    .is_some_and(|(_, last)| last == "table_e8")
                {
                    table_e8_rows.insert(key.clone(), a.clone());
                }
                continue;
            }
            let bucket_l = a
                .get("bucket_L")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| missing(key, "bucket_L"))? as usize;
            let hidden = a
                .pointer("/geometry/hidden")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| missing(key, "geometry.hidden"))? as usize;
            let digest = a
                .pointer("/digest/digest")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| missing(key, "digest.digest"))?
                .to_string();
            let digest_files =
                a.pointer("/digest/files")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| missing(key, "digest.files"))? as usize;
            let digest_bytes = a
                .pointer("/digest/bytes")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| missing(key, "digest.bytes"))?;
            let outputs = a
                .get("outputs")
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| missing(key, "outputs"))?;
            if outputs.len() != 1 {
                return Err(LayaError::Runtime(format!(
                    "ane manifest {key}: expected exactly one output tensor, got {}",
                    outputs.len()
                )));
            }
            let output_name = outputs
                .keys()
                .next()
                .ok_or_else(|| missing(key, "outputs"))?
                .clone();
            let mask_sentinel = a
                .get("mask_sentinel_fp16")
                .and_then(serde_json::Value::as_f64)
                .map(|v| v as f32)
                .unwrap_or(FP16_NEG_INF_MASK);
            let ane_ops =
                a.pointer("/placement/ane_ops")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| missing(key, "placement.ane_ops"))? as usize;
            let device_ops =
                a.pointer("/placement/device_ops")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| missing(key, "placement.device_ops"))? as usize;
            let transitions =
                a.pointer("/placement/transitions")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| missing(key, "placement.transitions"))? as usize;
            artifacts.insert(
                key.clone(),
                AneArtifact {
                    bucket_l,
                    hidden,
                    digest,
                    digest_files,
                    digest_bytes,
                    output_name,
                    ane_ops,
                    device_ops,
                    transitions,
                    mask_sentinel,
                },
            );
        }
        if artifacts.is_empty() {
            return Err(LayaError::Runtime("ane manifest: no artifacts".into()));
        }
        Ok(Self {
            artifacts,
            table_e8_rows,
        })
    }

    /// The entry for one checkpoint + bucket, erroring with the manifest
    /// keys that DO exist (a missing key must name its neighbours).
    pub fn entry(&self, model_dir: &str, bucket: usize) -> Result<&AneArtifact> {
        let key = format!("{model_dir}/L{bucket}");
        self.artifacts.get(&key).ok_or_else(|| {
            let known: Vec<&str> = self
                .artifacts
                .keys()
                .filter(|k| k.starts_with(model_dir))
                .map(String::as_str)
                .collect();
            LayaError::Runtime(format!(
                "ane manifest: no entry {key:?} — known for this model: {known:?}"
            ))
        })
    }

    /// The `<model>/table_e8` row, parsed + validated — Plan 612 Phase 1
    /// (converter `ane_convert.py table --model <model> --table-precision
    /// e8`). Called only by the e8 posture; the fp16 posture never
    /// validates this schema. Required shape: blake3 file digest, table
    /// `[vocab, hidden]` int8, scales `[vocab]` f32, per-row (vocab-axis)
    /// scales — any other axis is a DIFFERENT gather and refuses here.
    pub fn table_e8(&self, model_dir: &str, hidden: usize) -> Result<TableE8> {
        let key = format!("{model_dir}/table_e8");
        let raw = self.table_e8_rows.get(&key).ok_or_else(|| {
            LayaError::Runtime(format!(
                "ane manifest: no {key:?} row — generate the sidecar with \
                 `scripts/ane_convert.py table --model {model_dir} --table-precision e8` \
                 (riir-reflex, Plan 612 Phase 1)"
            ))
        })?;
        let digest = raw
            .pointer("/digest/digest")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| missing(&key, "digest.digest"))?
            .to_string();
        let algo = raw
            .pointer("/digest/algo")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("blake3");
        if algo != "blake3" {
            return Err(LayaError::Runtime(format!(
                "ane manifest {key}: sidecar digest algo {algo:?} unsupported — only blake3"
            )));
        }
        let bytes = raw
            .pointer("/digest/bytes")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| missing(&key, "digest.bytes"))?;
        let shape2 = |field: &str| -> Result<Vec<usize>> {
            raw.pointer(&format!("/{field}/shape"))
                .and_then(serde_json::Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|d| d.as_u64().unwrap_or(0) as usize)
                        .collect::<Vec<_>>()
                })
                .ok_or_else(|| missing(&key, &format!("{field}.shape")))
        };
        let table_shape = shape2("table")?;
        let scales_shape = shape2("scales")?;
        if table_shape.len() != 2 || table_shape[1] != hidden {
            return Err(LayaError::Runtime(format!(
                "ane manifest {key}: table shape {table_shape:?} != [vocab, {hidden}] \
                 (the checkpoint's hidden width)"
            )));
        }
        if scales_shape.len() != 1 || scales_shape[0] != table_shape[0] {
            return Err(LayaError::Runtime(format!(
                "ane manifest {key}: scales shape {scales_shape:?} != [{vocab}] \
                 (one f32 scale per vocab row)",
                vocab = table_shape[0]
            )));
        }
        let axis = raw
            .pointer("/quant/axis")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("per_row_vocab");
        if axis != "per_row_vocab" {
            return Err(LayaError::Runtime(format!(
                "ane manifest {key}: quant axis {axis:?} unsupported — the gather \
                 dequantizes per-row (vocab-axis); {axis:?} is a different consumer"
            )));
        }
        // The sidecar serves every bucket of its checkpoint — the
        // manifest's own bucket rows must all be covered.
        let mut serves = raw
            .get("serves_buckets")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|b| b as usize))
                    .collect::<Vec<_>>()
            });
        if let Some(serves) = serves.as_mut() {
            serves.sort_unstable();
            for bucket in self.artifacts.keys().filter_map(|k| {
                k.strip_prefix(model_dir)
                    .and_then(|r| r.strip_prefix("/L"))
                    .and_then(|b| b.parse::<usize>().ok())
            }) {
                if !serves.contains(&bucket) {
                    return Err(LayaError::Runtime(format!(
                        "ane manifest {key}: serves_buckets {serves:?} does not cover \
                         the checkpoint's bucket {bucket} — regenerate the sidecar"
                    )));
                }
            }
        }
        Ok(TableE8 {
            digest,
            bytes,
            vocab: table_shape[0],
            hidden,
        })
    }
}

/// One validated `<model>/table_e8` manifest row — the fields the e8
/// posture's load-time verification consumes.
#[derive(Debug, Clone)]
pub struct TableE8 {
    /// blake3 hex digest of the sidecar file.
    pub digest: String,
    /// Pinned byte total (cross-checked against the file on disk).
    pub bytes: u64,
    /// Table rows = the checkpoint's vocab.
    pub vocab: usize,
    /// Table cols = the checkpoint's hidden width.
    pub hidden: usize,
}

fn missing(key: &str, field: &str) -> LayaError {
    LayaError::Runtime(format!("ane manifest {key}: missing {field}"))
}

/// `"<model_dir>/L<digits>"` — the ANE-lane artifact key shape. The
/// `<model>/table_e8` sidecar rows are retained separately (the e8
/// posture's [`AneManifest::table_e8`]); anything else in the manifest
/// belongs to another consumer and must not be parsed against the
/// artifact schema.
fn is_bucket_artifact_key(key: &str) -> bool {
    let Some((_, l)) = key.rsplit_once('/') else {
        return false;
    };
    let digits = l.strip_prefix('L').unwrap_or("");
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

/// One loaded + verified artifact: the compiled Core ML model plus the
/// names/shapes one predict needs. Owns its `MLModel` (the compiled
/// bundle on disk is the content-addressed cache, shared across loads).
pub struct AneRuntime {
    model: Retained<MLModel>,
    pub bucket_l: usize,
    pub hidden: usize,
    input_emb: String,
    input_pb: String,
    output: String,
}

/// The load-time plan verdict (the T0.3 gate re-measured at load).
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub device_ops: usize,
    pub ane_ops: usize,
    pub gpu_ops: usize,
    pub cpu_ops: usize,
    pub transitions: usize,
}

impl Placement {
    /// The gate: 100% of device-bearing ops ANE-preferred, zero device
    /// transitions. `false` refuses the load.
    pub fn passes(&self) -> bool {
        self.device_ops > 0 && self.ane_ops == self.device_ops && self.transitions == 0
    }
}

impl AneRuntime {
    /// Verify the artifact digest, compile (content-addressed cache),
    /// load under `CPU_AND_NE`, and re-run the compute-plan gate. Any
    /// failure is a refused load — never a CPU fallback.
    #[allow(deprecated)] // the sync compile API; the async twin needs a
    // semaphore dance for a one-shot load-time call with no win
    pub fn load(artifact_dir: &Path, entry: &AneArtifact) -> Result<Self> {
        Self::verify_digest(artifact_dir, entry)?;
        let compiled = compile_cached(artifact_dir, &entry.digest)?;

        unsafe {
            let url = NSURL::fileURLWithPath_isDirectory(
                &NSString::from_str(&compiled.to_string_lossy()),
                true,
            );
            let config = MLModelConfiguration::new();
            config.setComputeUnits(COMPUTE_UNITS_CPU_AND_NE);
            let model = MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
                .map_err(|e| {
                    LayaError::Runtime(format!(
                        "ane: Core ML load failed for {}: {}",
                        compiled.display(),
                        ns_error_string(&e)
                    ))
                })?;

            let placement = verify_compute_plan(&compiled, &config)?;
            if !placement.passes() {
                return Err(LayaError::Runtime(format!(
                    "ane: compute-plan gate REFUSED for L{} — device_ops={} ane={} gpu={} \
                     cpu={} transitions={} (the artifact must run 100% on the ANE with 0 \
                     device transitions; a CPU/GPU number wearing an ANE label is the \
                     failure mode this gate exists to stop)",
                    entry.bucket_l,
                    placement.device_ops,
                    placement.ane_ops,
                    placement.gpu_ops,
                    placement.cpu_ops,
                    placement.transitions
                )));
            }
            if placement.device_ops != entry.device_ops || placement.ane_ops != entry.ane_ops {
                // Counts drift when the OS re-plans; the VERDICT above is
                // the gate, the count equality is disclosure.
                eprintln!(
                    "ane: L{} plan counts drifted from the manifest (load {} vs \
                     conversion {} device / {} vs {} ane) — verdict unchanged, disclosed",
                    entry.bucket_l,
                    placement.device_ops,
                    entry.device_ops,
                    placement.ane_ops,
                    entry.ane_ops
                );
            }

            Ok(Self {
                model,
                bucket_l: entry.bucket_l,
                hidden: entry.hidden,
                input_emb: "embeddings".to_string(),
                input_pb: "pad_bias".to_string(),
                output: entry.output_name.clone(),
            })
        }
    }

    /// blake3-dir-v1 over the bundle, byte-for-byte the conversion tool's
    /// digest (sorted relative path, `rel \0 data \0` per file). Refuses
    /// on count, size, or digest mismatch — a swapped or stale artifact
    /// is a hard error (issue 017: download-on-demand, digest-pinned).
    fn verify_digest(dir: &Path, entry: &AneArtifact) -> Result<()> {
        if !dir.is_dir() {
            return Err(LayaError::Runtime(format!(
                "ane artifact missing: {} — run the consumer repo's ane_convert.py first \
                 (artifacts are local-only, never committed)",
                dir.display()
            )));
        }
        let mut files: Vec<(String, PathBuf)> = Vec::new();
        collect_files(dir, dir, &mut files)?;
        files.sort_by(|a, b| a.0.cmp(&b.0));
        let total: u64 = files
            .iter()
            .filter_map(|(_, path)| std::fs::metadata(path).ok())
            .map(|m| m.len())
            .sum();
        if files.len() != entry.digest_files || total != entry.digest_bytes {
            return Err(LayaError::Runtime(format!(
                "ane artifact shape drift at {}: {} files / {} bytes vs pinned {} / {}",
                dir.display(),
                files.len(),
                total,
                entry.digest_files,
                entry.digest_bytes
            )));
        }
        let mut h = blake3::Hasher::new();
        for (rel, path) in &files {
            let data = std::fs::read(path).map_err(|e| {
                LayaError::Runtime(format!("ane artifact file unreadable {path:?}: {e}"))
            })?;
            h.update(rel.as_bytes());
            h.update(b"\0");
            h.update(&data);
            h.update(b"\0");
        }
        let digest = h.finalize().to_hex().to_string();
        if digest != entry.digest {
            return Err(LayaError::Runtime(format!(
                "ane artifact digest mismatch at {} — got {}, pinned {} \
                 (the artifact is not the one the manifest verified)",
                dir.display(),
                &digest[..16],
                &entry.digest[..16]
            )));
        }
        Ok(())
    }

    /// One forward: fp16 gather rows `[1, L, d]` + fp16 per-key mask
    /// `[1, 1, 1, L]` in, fp16 hidden `[1, L, d]` out (the caller's
    /// buffer). Fixed shapes — an input whose length ≠ the bucket is a
    /// caller bug and asserts.
    pub fn predict(
        &self,
        emb_f16: &[u16],
        pad_bias_f16: &[u16],
        out_f16: &mut [u16],
    ) -> Result<()> {
        let l = self.bucket_l;
        let d = self.hidden;
        assert_eq!(emb_f16.len(), l * d, "embeddings buffer must be [1, L, d]");
        assert_eq!(
            pad_bias_f16.len(),
            l,
            "pad_bias buffer must be [1, 1, 1, L]"
        );
        assert_eq!(out_f16.len(), l * d, "out buffer must be [1, L, d]");
        unsafe {
            // The arrays borrow the caller's buffers (no-copy init, no-op
            // deallocator); both outlive the prediction call below.
            let emb = multi_array_zero_copy(emb_f16, &[1, l, d])?;
            let pb = multi_array_zero_copy(pad_bias_f16, &[1, 1, 1, l])?;
            let emb_name = NSString::from_str(&self.input_emb);
            let pb_name = NSString::from_str(&self.input_pb);
            let emb_val = MLFeatureValue::featureValueWithMultiArray(&emb);
            let pb_val = MLFeatureValue::featureValueWithMultiArray(&pb);
            // The provider init erases the value type to AnyObject (the
            // ObjC dictionary erasure) — upcast both values once. `cast`
            // is the documented upcast (deprecated name, fine for a root
            // upcast; `cast_unchecked` would add unsafe for nothing).
            #[allow(deprecated)]
            let emb_obj = Retained::cast::<AnyObject>(emb_val);
            #[allow(deprecated)]
            let pb_obj = Retained::cast::<AnyObject>(pb_val);
            let dict: Retained<NSDictionary<NSString, AnyObject>> =
                NSDictionary::from_retained_objects(&[&*emb_name, &*pb_name], &[emb_obj, pb_obj]);
            let features = MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
            .map_err(|e| {
                LayaError::Runtime(format!(
                    "ane: feature provider failed: {}",
                    ns_error_string(&e)
                ))
            })?;
            let out = self
                .model
                .predictionFromFeatures_error(ProtocolObject::from_ref(&*features))
                .map_err(|e| {
                    LayaError::Runtime(format!("ane: prediction failed: {}", ns_error_string(&e)))
                })?;
            let name = NSString::from_str(&self.output);
            let val = out.featureValueForName(&name).ok_or_else(|| {
                LayaError::Runtime(format!(
                    "ane: output {:?} missing from the prediction (manifest/ artifact drift?)",
                    self.output
                ))
            })?;
            let arr = val.multiArrayValue().ok_or_else(|| {
                LayaError::Runtime(format!(
                    "ane: output {:?} is not a multi-array",
                    self.output
                ))
            })?;
            copy_out_f16(&arr, out_f16)?;
            Ok(())
        }
    }
}

/// Gather `input_ids` (pad-filling the tail with the PAD row, exactly the
/// reference smoke's input construction) into the fp16 `[1, L, d]` buffer.
/// The table is fp16 (narrowed once from the checkpoint's f32 at load —
/// bit-exact, every value originated as f16).
fn gather_fp16(table_f16: &[u16], ids: &[u32], pad_id: u32, d: usize, l: usize, buf: &mut [u16]) {
    for s in 0..l {
        let id = if s < ids.len() { ids[s] } else { pad_id };
        let row = id as usize * d;
        buf[s * d..s * d + d].copy_from_slice(&table_f16[row..row + d]);
    }
}

/// The e8 gather (Plan 612 Phase 2): per-token row dequant `i8 · scale →
/// f32 → f16 bits` straight into the SAME `[1, L, d]` scratch the fp16
/// gather fills — the artifact's input contract is unchanged bytes, so
/// the encoder forward is posture-blind downstream of this call. The
/// PAD row is dequantized ONCE per forward (into the first pad slot)
/// and `copy_within`-ed over the rest of the pad tail — never per pad
/// token. Zero-alloc: writes only into `buf`.
fn gather_e8(
    table_i8: &[i8],
    scales: &[f32],
    ids: &[u32],
    pad_id: u32,
    d: usize,
    l: usize,
    buf: &mut [u16],
) {
    let dequant_row = |id: usize, dst: &mut [u16]| {
        let row = id * d;
        let scale = scales[id];
        for (o, q) in dst.iter_mut().zip(table_i8[row..row + d].iter()) {
            *o = super::weights::f32_to_f16_bits(f32::from(*q) * scale);
        }
    };
    for (s, &id) in ids.iter().enumerate() {
        dequant_row(id as usize, &mut buf[s * d..s * d + d]);
    }
    if ids.len() < l {
        // The pad tail: dequantize the PAD row once, then fan out.
        let first = ids.len();
        dequant_row(pad_id as usize, &mut buf[first * d..first * d + d]);
        for s in first + 1..l {
            buf.copy_within(first * d..first * d + d, s * d);
        }
    }
}

/// The resident table posture (Plan 612 Phase 2). fp16 is the default
/// and the always-correct posture; e8 halves the resident table with
/// the dequant moved into the host gather.
#[derive(Debug)]
enum TablePosture {
    /// The fp16 table, converted from the checkpoint at load (the
    /// founding posture — byte-identical behavior, pinned by the
    /// existing gates).
    F16(Vec<u16>),
    /// The Plan 612 Phase 1 int8 sidecar: per-row (vocab-axis) int8
    /// quants + one f32 scale per row, dequantized in [`gather_e8`].
    E8 { table_i8: Vec<i8>, scales: Vec<f32> },
}

/// The env selection ([`TablePosture`] minus its payload — the loader
/// supplies that).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TablePostureSel {
    F16,
    E8,
}

/// The env value → selection. Unset/empty = fp16; `e8` = the int8
/// sidecar; anything else refuses LOUD naming the supported set (the
/// no-silent-fallback law — a typo'd value never demotes to the
/// default).
fn resolve_table_posture(env: Option<&str>) -> Result<TablePostureSel> {
    match env {
        None | Some("") => Ok(TablePostureSel::F16),
        Some("e8") => Ok(TablePostureSel::E8),
        Some(other) => Err(LayaError::Runtime(format!(
            "LAYA_ANE_TABLE={other:?} is unsupported — supported: unset (fp16 default) \
             or \"e8\" (the Plan 612 per-row int8 sidecar)"
        ))),
    }
}

/// One sidecar tensor's `(shape, data slice)` from the parsed header —
/// span length validated against the manifest-derived `expect` (the
/// byte-span IS the dtype layout wall; a wrong dtype cannot produce the
/// right span).
fn tensor_span<'a>(
    entries: &'a serde_json::Map<String, serde_json::Value>,
    data: &'a [u8],
    data_start: usize,
    name: &str,
    expect: usize,
    ckpt: &'static str,
) -> Result<(Vec<usize>, &'a [u8])> {
    let err = |detail: String| LayaError::Config {
        checkpoint: ckpt,
        detail,
    };
    let e = entries.get(name).ok_or_else(|| {
        err(format!(
            "table_e8 header: missing tensor {name:?} (has {:?})",
            entries.keys().collect::<Vec<_>>()
        ))
    })?;
    let dtype = e["dtype"]
        .as_str()
        .ok_or_else(|| err(format!("table_e8 {name}: missing dtype")))?;
    let shape: Vec<usize> = e["shape"]
        .as_array()
        .ok_or_else(|| err(format!("table_e8 {name}: missing shape")))?
        .iter()
        .map(|d| d.as_u64().unwrap_or(0) as usize)
        .collect();
    let offsets = e["data_offsets"]
        .as_array()
        .ok_or_else(|| err(format!("table_e8 {name}: missing data_offsets")))?;
    let begin = offsets.first().and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let end = offsets.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    if end < begin || end - begin != expect {
        return Err(err(format!(
            "table_e8 {name}: data span {} bytes != the {dtype} layout ({expect})",
            end.saturating_sub(begin)
        )));
    }
    Ok((shape, &data[data_start + begin..data_start + end]))
}

/// Load + verify the `<model>/table_e8.safetensors` sidecar (Plan 612
/// Phase 2): BLAKE3 re-checked against the manifest row, byte total
/// cross-checked, geometry cross-checked against the CHECKPOINT's own
/// embedding tensor, scales per-row (vocab-axis) only. A set env with
/// an absent/invalid sidecar refuses LOUD naming the converter — never
/// a silent fallback to fp16 (the no-silent-fallback law).
fn load_table_e8(
    ane_root: &Path,
    model_dir: &str,
    manifest: &AneManifest,
    tok: &Weights,
    hidden: usize,
    ckpt: &'static str,
) -> Result<TablePosture> {
    let cfg_err = |detail: String| LayaError::Config {
        checkpoint: ckpt,
        detail,
    };
    let row = manifest.table_e8(model_dir, hidden)?;
    let path = ane_root.join(model_dir).join("table_e8.safetensors");
    let bytes = std::fs::read(&path).map_err(|e| {
        cfg_err(format!(
            "LAYA_ANE_TABLE=e8: sidecar unreadable at {}: {e} — generate it with \
             `scripts/ane_convert.py table --model {model_dir} --table-precision e8` \
             (riir-reflex, Plan 612 Phase 1); refusing, never a silent fp16 fallback",
            path.display()
        ))
    })?;
    if bytes.len() as u64 != row.bytes {
        return Err(cfg_err(format!(
            "table_e8 sidecar {} bytes != the manifest's pinned {}",
            bytes.len(),
            row.bytes
        )));
    }
    let digest = blake3::hash(&bytes).to_string();
    if digest != row.digest {
        return Err(cfg_err(format!(
            "table_e8 sidecar digest {digest} != the manifest's pinned {} — \
             a stale or swapped sidecar refuses, never loads",
            row.digest
        )));
    }
    let (data_start, header) =
        super::weights::container_header(&bytes, ckpt, "table_e8.safetensors")?;
    let entries = header
        .as_object()
        .ok_or_else(|| cfg_err("table_e8 header is not a JSON object".to_string()))?;
    let (table_shape, table_bytes) = tensor_span(
        entries,
        &bytes,
        data_start,
        "table",
        row.vocab * row.hidden,
        ckpt,
    )?;
    if table_shape != vec![row.vocab, row.hidden] {
        return Err(cfg_err(format!(
            "table_e8 table shape {table_shape:?} != [{}, {}]",
            row.vocab, row.hidden
        )));
    }
    let (scales_shape, scales_bytes) =
        tensor_span(entries, &bytes, data_start, "scales", row.vocab * 4, ckpt)?;
    if scales_shape != vec![row.vocab] {
        return Err(cfg_err(format!(
            "table_e8 scales shape {scales_shape:?} != [{}]",
            row.vocab
        )));
    }
    // The pairing wall: the sidecar must be THIS checkpoint's table —
    // same vocab and hidden as the embedding tensor it replaces.
    if tok.shape != vec![row.vocab, row.hidden] {
        return Err(cfg_err(format!(
            "table_e8 sidecar {:?} does not pair with the checkpoint's embedding {:?} — \
             a wrong-sidecar pairing refuses at load",
            [row.vocab, row.hidden],
            tok.shape
        )));
    }
    let table_i8: Vec<i8> = table_bytes.iter().map(|&b| b as i8).collect();
    let scales: Vec<f32> = scales_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    // Plan 612 Phase 2: the resident halving asserted, never assumed
    // (u16 table → i8 + f32 scales; the f32 scales are the ~0.5%
    // overhead the halving must survive).
    debug_assert!(
        table_i8.len() + 4 * scales.len() < tok.shape[0] * hidden * 2,
        "e8 resident table ({}) must be smaller than the fp16 table ({})",
        table_i8.len() + 4 * scales.len(),
        tok.shape[0] * hidden * 2
    );
    Ok(TablePosture::E8 { table_i8, scales })
}

/// The `pad_bias` row: `0.0` attend for `[0, n)`, the manifest's fp16 mask
/// sentinel for `[n, L)` — per KEY position, so padded keys are invisible
/// to every query (bidirectional attention included).
fn build_pad_bias(n: usize, l: usize, sentinel_f16: u16, buf: &mut [u16]) {
    buf[..n].fill(0);
    buf[n..l].fill(sentinel_f16);
}

/// A loaded ANE encoder stack: the fp16 gather table + one lazy runtime
/// per bucket. [`super::encoder::Encoder`]'s whole-graph counterpart —
/// same `forward(input_ids) -> [n·d] f32` contract, no per-op backend.
pub struct AneEncoder {
    /// Checkpoint name for error rendering (the typed Bucket error).
    ckpt: &'static str,
    d: usize,
    pad_id: u32,
    /// The resident embedding table — fp16 by default, the Plan 612 e8
    /// int8 sidecar under `LAYA_ANE_TABLE=e8` (the gather dequantizes).
    table: TablePosture,
    /// Artifact root (the `assets/ane` directory: `<root>/<model>/L<bucket>.mlpackage`).
    ane_root: PathBuf,
    manifest: AneManifest,
    model_dir: &'static str,
    runtimes: RefCell<HashMap<usize, Rc<AneRuntime>>>,
}

impl AneEncoder {
    /// Assemble from the parsed safetensors map (REMOVES the embedding
    /// table; the head consumes its own names afterwards and the layer
    /// weights are dropped — the artifact holds the encoder fp16, the f32
    /// layer copy would be resident dead weight). `ane_root` is the
    /// artifacts root, `manifest_path` the committed manifest.
    pub fn from_map(
        map: &mut HashMap<String, Weights>,
        cfg: EncoderConfig,
        ckpt: &'static str,
        model_dir: &'static str,
        pad_id: u32,
        ane_root: PathBuf,
        manifest_path: &Path,
    ) -> Result<Self> {
        let manifest = AneManifest::load(manifest_path)?;
        // Shape cross-check against the manifest entries: every bucket the
        // manifest claims for this model must agree with the checkpoint's
        // hidden width (a mismatch is an artifact/checkpoint pairing bug).
        for key in manifest.artifacts.keys() {
            if let Some(rest) = key.strip_prefix(model_dir).filter(|r| r.starts_with("/L")) {
                let _ = rest;
                let entry = manifest.artifacts.get(key).expect("key from map");
                if entry.hidden != cfg.hidden {
                    return Err(LayaError::Config {
                        checkpoint: ckpt,
                        detail: format!(
                            "ane artifact {key} hidden {} != checkpoint hidden {}",
                            entry.hidden, cfg.hidden
                        ),
                    });
                }
            }
        }
        let name = "encoder.embeddings.tok_embeddings.weight";
        let tok = map.remove(name).ok_or_else(|| LayaError::Pin {
            checkpoint: ckpt,
            file: name.to_string(),
            detail: "tensor missing from checkpoint".into(),
        })?;
        // The resident posture comes from `LAYA_ANE_TABLE` (read live at
        // load — the posture IS which bytes are resident). fp16 (unset)
        // widens the checkpoint payload exactly as the founding posture;
        // e8 loads the verified Plan 612 sidecar instead and the payload
        // drops unread (only its SHAPE pairs it with the sidecar).
        let env_sel = std::env::var("LAYA_ANE_TABLE").ok();
        let table = match resolve_table_posture(env_sel.as_deref())? {
            TablePostureSel::F16 => {
                // The ANE posture refuses the q8 variant at load
                // (agent.rs), so this payload is always widened F32 — the
                // resolution below is a compile-shaped no-op on the F16
                // file.
                let table_f16: Vec<u16> = tok
                    .wide_f32()
                    .iter()
                    .map(|v| super::weights::f32_to_f16_bits(*v))
                    .collect();
                TablePosture::F16(table_f16)
            }
            TablePostureSel::E8 => {
                load_table_e8(&ane_root, model_dir, &manifest, &tok, cfg.hidden, ckpt)?
            }
        };
        Ok(Self {
            ckpt,
            d: cfg.hidden,
            pad_id,
            table,
            ane_root,
            manifest,
            model_dir,
            runtimes: RefCell::new(HashMap::new()),
        })
    }

    /// The buckets this checkpoint's manifest carries (e.g. `[64, 128]`).
    pub fn buckets(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .manifest
            .artifacts
            .keys()
            .filter_map(|k| k.strip_prefix(self.model_dir))
            .filter_map(|rest| rest.strip_prefix("/L"))
            .filter_map(|b| b.parse::<usize>().ok())
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Forward `input_ids` to the final-norm'd hidden state `[n, d]`
    /// (f32) — the bucket covering `n` (preloaded by `load_ane`, else lazy), the
    /// pad tail is masked, the fp16 output slices to `n` and widens
    /// bit-exactly. Refuses loud when `n` exceeds the largest bucket (a
    /// longer sequence is a FAILED forward, never a CPU fallback).
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let n = input_ids.len();
        let d = self.d;
        let buckets = self.buckets();
        let Some(&bucket) = buckets.iter().find(|b| **b >= n) else {
            return Err(LayaError::Bucket {
                checkpoint: self.ckpt,
                seq: n,
                max: *buckets.last().unwrap_or(&0),
            });
        };
        let rt = self.runtime_for(bucket)?;
        let l = rt.bucket_l;

        let entry = self.manifest.entry(self.model_dir, bucket)?;
        let sentinel_f16 = super::weights::f32_to_f16_bits(entry.mask_sentinel);
        let mut emb = vec![0u16; l * d];
        let mut pb = vec![0u16; l];
        match &self.table {
            TablePosture::F16(t) => gather_fp16(t, input_ids, self.pad_id, d, l, &mut emb),
            TablePosture::E8 { table_i8, scales } => {
                gather_e8(table_i8, scales, input_ids, self.pad_id, d, l, &mut emb)
            }
        }
        build_pad_bias(n, l, sentinel_f16, &mut pb);
        let mut out_f16 = vec![0u16; l * d];
        rt.predict(&emb, &pb, &mut out_f16)?;

        // Slice [:n] and widen bit-exactly (the head consumes f32).
        let mut hidden = Vec::with_capacity(n * d);
        for row in out_f16[..n * d].chunks_exact(d) {
            for bits in row {
                hidden.push(super::weights::f16_bits_to_f32(*bits));
            }
        }
        Ok(hidden)
    }

    /// Load every bucket's runtime (verify → compile → plan-gate) and run
    /// one pad-only forward through each, so no request pays a first-use
    /// load. Measured on the M3 (riir-reflex Bench 042): each lazy load
    /// was ~0.93 s landing INSIDE a timed call — the 644–891 ms p99 of
    /// every short ANE harness suite, and the p50 of code_fixtures, where
    /// 2 of its 4 servable cases were each the first to reach a bucket.
    /// `load_ane` calls this; [`Self::forward`]'s lazy path stays as the
    /// fallback for a caller that built the encoder some other way.
    pub fn preload(&self) -> Result<()> {
        for bucket in self.buckets() {
            self.forward(&vec![self.pad_id; bucket])?;
        }
        Ok(())
    }

    /// The bucket's runtime, loading (verify → compile → plan-gate) on
    /// first use. The RefCell + Rc make the encoder single-threaded — the
    /// agent's forward is single-threaded by contract (one question at a
    /// time, the capture's posture).
    fn runtime_for(&self, bucket: usize) -> Result<Rc<AneRuntime>> {
        let mut runtimes = self.runtimes.borrow_mut();
        if let Some(rt) = runtimes.get(&bucket) {
            return Ok(Rc::clone(rt));
        }
        let entry = self.manifest.entry(self.model_dir, bucket)?.clone();
        let artifact_dir = self
            .ane_root
            .join(self.model_dir)
            .join(format!("L{}.mlpackage", entry.bucket_l));
        let rt = Rc::new(AneRuntime::load(&artifact_dir, &entry)?);
        runtimes.insert(bucket, Rc::clone(&rt));
        Ok(rt)
    }
}

// ── internals ────────────────────────────────────────────────────────────

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| LayaError::Runtime(format!("ane: readdir {dir:?}: {e}")))?;
    for e in entries {
        let e =
            e.map_err(|err| LayaError::Runtime(format!("ane: readdir entry {dir:?}: {err}")))?;
        let p = e.path();
        if p.is_dir() {
            collect_files(root, &p, out)?;
        } else if p.is_file() {
            let rel = p
                .strip_prefix(root)
                .expect("walk rooted at `root`")
                .to_string_lossy()
                .to_string();
            out.push((rel, p));
        }
    }
    Ok(())
}

/// The standard `.mlpackage` bundle layout — the per-file fetch paths for
/// download-on-demand. coremltools' bundle format is fixed (3 files), but
/// a future layout change cannot corrupt an install: the staged tree must
/// match the pinned file count/bytes/digest BEFORE it is renamed into
/// place, and a short walk fails loud there.
const BUNDLE_FILES: [&str; 3] = [
    "Manifest.json",
    "Data/com.apple.CoreML/model.mlmodel",
    "Data/com.apple.CoreML/weights/weight.bin",
];

/// Fetch-on-first-use for the artifact tree (issue 017's release scope:
/// download-on-demand, digest-pinned — macOS-only artifacts never ship in
/// a release archive). Every manifest entry whose directory is missing is
/// assembled from `<base_url>/<key>.mlpackage/<rel>` into a staging dir,
/// digest-verified, then renamed into place — a bad download never
/// installs. Entries already on disk are LEFT ALONE (the load re-verifies
/// every digest, so a swapped artifact still refuses). `base_url` is the
/// artifact host root (any static file server or HF-style repo layout
/// carrying `<key>.mlpackage/` bundles); `None` refuses with the two
/// remedies when anything is missing.
pub fn ensure_artifacts(ane_root: &Path, base_url: Option<&str>) -> Result<()> {
    let manifest_path = ane_root.join("manifest.json");
    if !manifest_path.exists() {
        return Err(LayaError::Runtime(format!(
            "ane manifest missing at {} — run the consumer repo's scripts/ane_convert.py \
             (offline, one-time) or point LAYA_ANE_ARTIFACTS_DIR at a populated tree",
            manifest_path.display()
        )));
    }
    let manifest = AneManifest::load(&manifest_path)?;
    let mut missing: Vec<String> = Vec::new();
    // Sorted keys: deterministic fetch order, deterministic refusal text.
    let mut keys: Vec<&String> = manifest.artifacts.keys().collect();
    keys.sort();
    for key in keys {
        let dir = ane_root.join(format!("{key}.mlpackage"));
        if dir.is_dir() {
            continue;
        }
        let Some(base) = base_url else {
            missing.push(key.clone());
            continue;
        };
        let entry = manifest.artifacts.get(key).expect("key from the map");
        let staging = ane_root.join(".staging").join(format!("{key}.mlpackage"));
        let _ = std::fs::remove_dir_all(&staging);
        let staged = stage_bundle(&staging, base, key)
            .and_then(|()| AneRuntime::verify_digest(&staging, entry))
            .and_then(|()| {
                let parent = dir.parent().expect("key carries a model dir");
                std::fs::create_dir_all(parent)
                    .map_err(|e| LayaError::Runtime(format!("create {}: {e}", parent.display())))
            })
            .and_then(|()| {
                std::fs::rename(&staging, &dir)
                    .map_err(|e| LayaError::Runtime(format!("install {}: {e}", dir.display())))
            });
        // A failed fetch installs nothing: the staging subtree goes, then
        // the emptied parents (best-effort — a concurrent fetch of a
        // sibling key owns its own subtree).
        if let Err(e) = staged {
            let _ = std::fs::remove_dir_all(&staging);
            for p in [staging.parent(), Some(ane_root.join(".staging")).as_deref()]
                .into_iter()
                .flatten()
            {
                let _ = std::fs::remove_dir(p);
            }
            return Err(e);
        }
        for p in [staging.parent(), Some(ane_root.join(".staging")).as_deref()]
            .into_iter()
            .flatten()
        {
            let _ = std::fs::remove_dir(p);
        }
    }
    if missing.is_empty() {
        return Ok(());
    }
    Err(LayaError::Runtime(format!(
        "ane artifacts missing: {missing:?} — run scripts/ane_convert.py (offline, \
         one-time) or set RIIR_REFLEX_ANE_BASE_URL to an artifact host carrying \
         the `<key>.mlpackage/` bundles (fetch-on-first-use, digest-pinned)"
    )))
}

/// Fetch one bundle's fixed file set into `staging` (per-file `.part` +
/// rename, the weights precedent). The digest gate is the CALLER's —
/// this only assembles bytes from the wire.
fn stage_bundle(staging: &Path, base: &str, key: &str) -> Result<()> {
    for rel in BUNDLE_FILES {
        let url = format!("{base}/{key}.mlpackage/{rel}");
        let dest = staging.join(rel);
        let parent = dest.parent().expect("rel carries a dir");
        std::fs::create_dir_all(parent)
            .map_err(|e| LayaError::Runtime(format!("create {parent:?}: {e}")))?;
        super::super::weights::fetch_file(&url, &dest)?;
    }
    Ok(())
}

/// Compile the `.mlpackage` into a content-addressed `.mlmodelc` cache
/// (recompiles never load a stale artifact: the cache key IS the digest)
/// and return the compiled bundle path. Cache location: `LAYA_ANE_CACHE`,
/// else `~/Library/Caches/riir-infer/laya-ane` (the macOS cache home —
/// this lane is ANE-only; a persistent root survives /tmp cleanup and
/// keeps the compile-once-per-process-tree contract across processes),
/// else a pid-suffixed temp dir when HOME is unset. Trade-off: the
/// persistent root grows one bundle per digest, unbounded — acceptable
/// for compiled model bundles; clear it (or set `LAYA_ANE_CACHE`) to
/// reclaim.
fn compile_cached(artifact_dir: &Path, digest: &str) -> Result<PathBuf> {
    let key = &digest[..16.min(digest.len())];
    let cache_root = match std::env::var_os("LAYA_ANE_CACHE") {
        Some(p) => PathBuf::from(p),
        None => match std::env::var_os("HOME") {
            Some(h) => PathBuf::from(h).join("Library/Caches/riir-infer/laya-ane"),
            None => {
                std::env::temp_dir().join(format!("riir-laya-ane-cache_{}", std::process::id()))
            }
        },
    };
    let final_dir = cache_root.join(format!("mlmodelc-{key}"));
    if final_dir.is_dir() {
        return Ok(final_dir);
    }
    std::fs::create_dir_all(&cache_root)
        .map_err(|e| LayaError::Runtime(format!("ane: cache dir {}: {e}", cache_root.display())))?;
    // Compile into a process-unique staging dir, then atomically claim
    // the final name — two processes racing one digest both compile, one
    // wins the rename, the loser discards (never reads a half-written
    // bundle — the shared-fixed-path lesson, mechanized).
    let staging = cache_root.join(format!("staging-{}-{}", key, std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    let src = NSURL::fileURLWithPath_isDirectory(
        &NSString::from_str(&artifact_dir.to_string_lossy()),
        true,
    );
    // SAFETY: plain ObjC class call; the deprecated sync twin of the async
    // compile API — a one-shot load-time call, not worth a semaphore dance.
    #[allow(deprecated)]
    let compiled = unsafe { MLModel::compileModelAtURL_error(&src) }.map_err(|e| {
        LayaError::Runtime(format!(
            "ane: Core ML compile failed for {}: {}",
            artifact_dir.display(),
            ns_error_string(&e)
        ))
    })?;
    let compiled_path = compiled
        .path()
        .ok_or_else(|| LayaError::Runtime("ane: compiled model has no file path".into()))?
        .to_string();
    move_dir(Path::new(&compiled_path), &final_dir)?;
    let _ = std::fs::remove_dir_all(&staging);
    Ok(final_dir)
}

/// Move `src` into `dst` (rename; same-volume fallback copy), tolerating
/// `dst` having appeared meanwhile (another process won the claim).
fn move_dir(src: &Path, dst: &Path) -> Result<()> {
    if dst.is_dir() {
        let _ = std::fs::remove_dir_all(src);
        return Ok(());
    }
    if std::fs::rename(src, dst).is_ok() {
        return Ok(());
    }
    std::fs::copy(src, dst)
        .map_err(|e| LayaError::Runtime(format!("ane: cache move {src:?} → {dst:?}: {e}")))?;
    let _ = std::fs::remove_dir_all(src);
    Ok(())
}

/// Build a zero-copy fp16 `MLMultiArray` over the caller's buffer (the
/// buffer must outlive the array — in `predict` all three outlive the
/// prediction call, and the arrays drop first).
fn multi_array_zero_copy(data: &[u16], shape: &[usize]) -> Result<Retained<MLMultiArray>> {
    // No-op deallocator: ownership stays with the caller's buffer; the
    // block signature is Apple's `^(void *bytes) { … }`.
    let dealloc = RcBlock::new(|_bytes: NonNull<std::ffi::c_void>| {});
    let shape_arr = number_array(shape);
    let stride_arr = number_array(&contiguous_strides(shape));
    let ptr = NonNull::from(data).cast::<std::ffi::c_void>();
    // SAFETY: `data` covers `shape`'s element count and outlives the
    // returned array (predict drops the arrays before the buffers); the
    // deallocator is a no-op so Core ML never frees caller-owned memory.
    unsafe {
        MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
            MLMultiArray::alloc(),
            ptr,
            &shape_arr,
            MLMultiArrayDataType::Float16,
            &stride_arr,
            Some(&dealloc),
        )
    }
    .map_err(|e| {
        LayaError::Runtime(format!(
            "ane: input multi-array rejected: {}",
            ns_error_string(&e)
        ))
    })
}

fn number_array(values: &[usize]) -> Retained<NSArray<NSNumber>> {
    let nums: Vec<Retained<NSNumber>> = values.iter().map(|v| NSNumber::new_usize(*v)).collect();
    NSArray::from_retained_slice(&nums)
}

/// C-contiguous row-major strides for `shape`.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Read the fp16 output via `getBytesWithHandler` (the documented reader —
/// `dataPointer` is deprecated) into the caller's buffer.
fn copy_out_f16(arr: &MLMultiArray, out: &mut [u16]) -> Result<()> {
    let expected = out.len();
    let got = unsafe { arr.count() } as usize;
    if got != expected {
        return Err(LayaError::Runtime(format!(
            "ane: output count {got} != expected {expected} — artifact shape drift"
        )));
    }
    let sink = out.as_mut_ptr();
    let handler = RcBlock::new(move |bytes: NonNull<std::ffi::c_void>, size: isize| {
        let src = bytes.as_ptr() as *const u16;
        let n = (size as usize) / 2;
        // SAFETY: `sink` points at `expected` elements, live across this
        // call; Core ML hands us the array's own buffer.
        unsafe { std::ptr::copy_nonoverlapping(src, sink, n.min(expected)) };
    });
    // SAFETY: the handler only reads the array's buffer into `out`.
    unsafe {
        arr.getBytesWithHandler(&handler);
    }
    Ok(())
}

/// The T0.3 gate, re-measured at load: walk the ML Program structure of
/// the COMPILED artifact, ask the plan for every operation's preferred
/// device, and count the device-bearing ops + device transitions in
/// program order (exactly the conversion tool's placement table).
fn verify_compute_plan(compiled: &Path, config: &MLModelConfiguration) -> Result<Placement> {
    let (tx, rx) = std::sync::mpsc::channel();
    unsafe {
        let url = NSURL::fileURLWithPath_isDirectory(
            &NSString::from_str(&compiled.to_string_lossy()),
            true,
        );
        let handler = RcBlock::new(move |plan: *mut MLComputePlan, err: *mut NSError| {
            let result = if err.is_null() && !plan.is_null() {
                // SAFETY: a non-null MLComputePlan just handed to the
                // block by Core ML (autoreleased — retained here).
                Ok(Retained::retain_autoreleased(plan.cast::<MLComputePlan>())
                    .expect("non-null plan"))
            } else {
                let msg = if err.is_null() {
                    "null plan".to_string()
                } else {
                    ns_error_string(&*err)
                };
                Err(msg)
            };
            let _ = tx.send(result);
        });
        MLComputePlan::loadContentsOfURL_configuration_completionHandler(&url, config, &handler);
    }
    let plan = rx
        .recv_timeout(std::time::Duration::from_secs(120))
        .map_err(|_| LayaError::Runtime("ane: compute-plan load timed out".into()))?
        .map_err(LayaError::Runtime)?;

    let structure = unsafe { plan.modelStructure() };
    let program = unsafe { structure.program() }
        .ok_or_else(|| LayaError::Runtime("ane: model structure is not an ML Program".into()))?;
    let ops = program_operations(&program)?;

    let mut placement = Placement {
        device_ops: 0,
        ane_ops: 0,
        gpu_ops: 0,
        cpu_ops: 0,
        transitions: 0,
    };
    let mut prev: Option<u8> = None; // 0 ane / 1 gpu / 2 cpu / 3 other
    for op in &ops {
        let usage = unsafe { plan.computeDeviceUsageForMLProgramOperation(op) };
        let Some(usage) = usage else {
            continue; // const / identity — materialized, no device (the P0 rule)
        };
        let device = unsafe { usage.preferredComputeDevice() };
        // The runtime class name is the device kind (the P0 smoke's
        // type-name matching — MLComputeDeviceProtocol carries no kind
        // tag); ProtocolObject reaches the runtime class through its
        // AnyObject view.
        let anyobj: &AnyObject = device.as_ref();
        let class_code = match anyobj.class().name().to_string_lossy().as_ref() {
            "MLNeuralEngineComputeDevice" => 0,
            "MLGPUComputeDevice" => 1,
            "MLCPUComputeDevice" => 2,
            _ => 3,
        };
        placement.device_ops += 1;
        match class_code {
            0 => placement.ane_ops += 1,
            1 => placement.gpu_ops += 1,
            2 => placement.cpu_ops += 1,
            _ => {}
        }
        if prev.is_some_and(|p| p != class_code) {
            placement.transitions += 1;
        }
        prev = Some(class_code);
    }
    Ok(placement)
}

/// `program.functions["main"].block.operations` — the main function's op
/// list in program order.
fn program_operations(
    program: &MLModelStructureProgram,
) -> Result<Vec<Retained<objc2_core_ml::MLModelStructureProgramOperation>>> {
    let main = NSString::from_str("main");
    let functions = unsafe { program.functions() };
    let func = functions
        .objectForKey(&main)
        .ok_or_else(|| LayaError::Runtime("ane: program has no `main` function".into()))?;
    let block = unsafe { func.block() };
    let ops = unsafe { block.operations() };
    Ok(ops.to_vec())
}

fn ns_error_string(e: &NSError) -> String {
    e.localizedDescription().to_string()
}

#[cfg(test)]
mod fetch_tests {
    use super::*;
    use blake3::Hasher;

    /// blake3-dir-v1 over a bundle dir — the exact algorithm
    /// [`AneRuntime::verify_digest`] pins (sorted rel, `rel\0bytes\0`).
    fn digest_of(dir: &Path) -> (String, usize, u64) {
        let mut files: Vec<(String, PathBuf)> = Vec::new();
        collect_files(dir, dir, &mut files).expect("walk");
        files.sort_by(|a, b| a.0.cmp(&b.0));
        let mut total = 0u64;
        let mut h = Hasher::new();
        for (rel, path) in &files {
            let data = std::fs::read(path).expect("read");
            total += data.len() as u64;
            h.update(rel.as_bytes());
            h.update(b"\0");
            h.update(&data);
            h.update(b"\0");
        }
        (h.finalize().to_hex().to_string(), files.len(), total)
    }

    /// One 3-file fixture bundle + the manifest JSON that pins it.
    /// `hidden`/`bucket_L`/`outputs`/`placement` are the manifest loader's
    /// required fields (values are inert for the fetch path).
    fn write_bundle(dir: &Path, payload: &[u8]) {
        for rel in BUNDLE_FILES {
            let p = dir.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, payload).unwrap();
        }
    }

    fn manifest_json(digest: &str, files: usize, bytes: u64) -> String {
        format!(
            r#"{{"artifacts": {{"en/L8": {{
                "bucket_L": 8,
                "geometry": {{"hidden": 4}},
                "digest": {{"algo": "blake3-dir-v1", "digest": "{digest}", "files": {files}, "bytes": {bytes}}},
                "outputs": {{"hidden_state": {{"shape": [1, 8, 4]}}}},
                "mask_sentinel_fp16": -10000.0,
                "placement": {{"ane_ops": 1, "device_ops": 1, "transitions": 0}}
            }}}}}}"#
        )
    }

    fn write_manifest(root: &Path, json: &str) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(root.join("manifest.json"), json).unwrap();
    }

    #[test]
    fn missing_artifact_with_no_base_refuses_naming_both_remedies() {
        let tmp = std::env::temp_dir().join(format!("ane_fetch_t1_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("root");
        write_manifest(&root, &manifest_json("00", 3, 9));
        let err = ensure_artifacts(&root, None).expect_err("refuses");
        let text = err.to_string();
        assert!(text.contains("ane_convert.py"), "got: {text}");
        assert!(text.contains("RIIR_REFLEX_ANE_BASE_URL"), "got: {text}");
        assert!(text.contains("en/L8"), "names the entry: {text}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn missing_manifest_refuses_before_any_fetch() {
        let tmp = std::env::temp_dir().join(format!("ane_fetch_t2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("root");
        std::fs::create_dir_all(&root).unwrap();
        let err = ensure_artifacts(&root, Some("file:///nonexistent")).expect_err("refuses");
        assert!(err.to_string().contains("manifest missing"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn fetch_assembles_verifies_and_installs_atomically() {
        let tmp = std::env::temp_dir().join(format!("ane_fetch_t3_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("root");
        let host = tmp.join("host");
        let bundle = host.join("en/L8.mlpackage");
        write_bundle(&bundle, b"0123456789");
        let (digest, files, bytes) = digest_of(&bundle);
        write_manifest(&root, &manifest_json(&digest, files, bytes));
        let base = format!("file://{}", host.display());
        ensure_artifacts(&root, Some(&base)).expect("installs");
        let installed = root.join("en/L8.mlpackage");
        assert!(installed.is_dir(), "installed");
        // The installed tree passes the SAME gate the load runs.
        let manifest = AneManifest::load(&root.join("manifest.json")).unwrap();
        let entry = manifest.artifacts.get("en/L8").unwrap();
        AneRuntime::verify_digest(&installed, entry).expect("digest");
        // No staging left behind.
        assert!(!root.join(".staging").join("en/L8.mlpackage").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn corrupt_download_fails_the_gate_and_installs_nothing() {
        let tmp = std::env::temp_dir().join(format!("ane_fetch_t4_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("root");
        let host = tmp.join("host");
        let bundle = host.join("en/L8.mlpackage");
        write_bundle(&bundle, b"0123456789");
        let (digest, files, bytes) = digest_of(&bundle);
        write_manifest(&root, &manifest_json(&digest, files, bytes));
        // The served copy diverges from the pin after the manifest was
        // written (a tampered/truncated host).
        std::fs::write(bundle.join(BUNDLE_FILES[1]), b"XXXXXXXXXX").unwrap();
        let base = format!("file://{}", host.display());
        let err = ensure_artifacts(&root, Some(&base)).expect_err("digest gate");
        assert!(err.to_string().contains("digest mismatch"), "got: {err}");
        assert!(!root.join("en/L8.mlpackage").exists(), "nothing installed");
        assert!(
            !root.join(".staging").exists() || {
                let mut it = std::fs::read_dir(root.join(".staging")).unwrap();
                it.next().is_none()
            },
            "staging cleaned"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn present_entries_are_left_alone() {
        let tmp = std::env::temp_dir().join(format!("ane_fetch_t5_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("root");
        let host = tmp.join("host");
        write_bundle(&host.join("en/L8.mlpackage"), b"0123456789");
        let (digest, files, bytes) = digest_of(&host.join("en/L8.mlpackage"));
        write_manifest(&root, &manifest_json(&digest, files, bytes));
        // The entry already on disk — deliberately WRONG vs the pin (the
        // load's own verify catches that; the fetcher must not touch it).
        let present = root.join("en/L8.mlpackage");
        write_bundle(&present, b"zz");
        ensure_artifacts(&root, None).expect("present entry → no missing, no fetch");
        assert!(present.join(BUNDLE_FILES[0]).exists(), "untouched");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn manifest_load_skips_foreign_non_bucket_rows() {
        // The `<model>/table_e8` sidecar rows (this lane's own Plan 612
        // Phase 1 output, reflex 6535b75) share this manifest with their
        // own schema — no `bucket_L`, no `outputs`. The loader retains
        // them RAW (parsed only by the e8 posture's `table_e8`) rather
        // than validating them against the artifact schema, while real
        // `<model>/L<n>` rows still parse and absent buckets still error
        // (the regression that broke the ANE lane at load from
        // 2026-09-26).
        let tmp = std::env::temp_dir().join(format!("ane_fetch_t6_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("root");
        let json = r#"{"artifacts": {
            "en/table_e8": {"path": "assets/ane/en/table_e8.safetensors", "table": {"tensor": "table", "dtype": "int8"}},
            "en/L8": {
                "bucket_L": 8,
                "geometry": {"hidden": 4},
                "digest": {"algo": "blake3-dir-v1", "digest": "00", "files": 3, "bytes": 9},
                "outputs": {"hidden_state": {"shape": [1, 8, 4]}},
                "mask_sentinel_fp16": -10000.0,
                "placement": {"ane_ops": 1, "device_ops": 1, "transitions": 0}
            }
        }}"#;
        write_manifest(&root, json);
        let manifest = AneManifest::load(&root.join("manifest.json"))
            .expect("loads with a foreign non-bucket row present");
        assert!(
            manifest.artifacts.contains_key("en/L8"),
            "artifact row parsed"
        );
        assert!(manifest.entry("en", 8).is_ok(), "bucket entry resolves");
        assert!(
            manifest.entry("en", 128).is_err(),
            "absent bucket still errors"
        );
        assert!(
            !manifest.artifacts.contains_key("en/table_e8"),
            "foreign row not admitted as an artifact"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

/// Plan 612 Phase 2 — the e8 table posture's own gates (pure-Rust: the
/// gather arithmetic, the env contract, and the sidecar load/verify
/// walls against hand-built fixtures; no Core ML, no real artifacts).
#[cfg(test)]
mod posture_tests {
    use super::super::weights::WeightData;
    use super::*;

    const VOCAB: usize = 4;
    /// Wide enough that the per-row f32 scales (4 B/row) do not swamp
    /// the i8 halving — the resident-halving debug assert only holds for
    /// realistic widths (the real checkpoints run d=768/1024).
    const D: usize = 8;

    fn fixture_quants() -> Vec<i8> {
        vec![
            -127, 0, 127, 1, -2, 3, -64, 32, // row 0
            16, 10, -10, 5, 20, -20, 30, -30, // row 1
            -8, 8, -4, 4, -2, 2, -1, 1, // row 2
            7, -7, 6, -6, 5, -5, 4, -4, // row 3 = the PAD row
        ]
    }

    fn fixture_scales() -> Vec<f32> {
        vec![0.5, 2.0, 0.25, 0.5]
    }

    fn widen(bits: u16) -> f32 {
        super::super::weights::f16_bits_to_f32(bits)
    }

    #[test]
    fn gather_e8_dequantizes_per_row_and_fans_the_pad_tail() {
        let table_i8 = fixture_quants();
        let scales = fixture_scales();
        let l = 5;
        let mut buf = vec![0u16; l * D];
        gather_e8(&table_i8, &scales, &[2, 0], 3, D, l, &mut buf);
        let expect: [f32; 24] = [
            -2.0, 2.0, -1.0, 1.0, -0.5, 0.5, -0.25, 0.25, // row 2 × 0.25
            -63.5, 0.0, 63.5, 0.5, -1.0, 1.5, -32.0, 16.0, // row 0 × 0.5
            3.5, -3.5, 3.0, -3.0, 2.5, -2.5, 2.0, -2.0, // pad row (once, fanned ×3)
        ];
        for (got, &want) in buf.iter().zip(expect.iter()) {
            assert_eq!(widen(*got), want, "dequant mismatch");
        }
    }

    #[test]
    fn gather_e8_writes_no_pad_slots_when_ids_fill_the_bucket() {
        let table_i8 = fixture_quants();
        let scales = fixture_scales();
        let l = 2;
        let mut buf = vec![0xABCDu16; l * D];
        gather_e8(&table_i8, &scales, &[2, 0], 3, D, l, &mut buf);
        let expect: [f32; 16] = [
            -2.0, 2.0, -1.0, 1.0, -0.5, 0.5, -0.25, 0.25, // row 2
            -63.5, 0.0, 63.5, 0.5, -1.0, 1.5, -32.0, 16.0, // row 0
        ];
        for (got, &want) in buf.iter().zip(expect.iter()) {
            assert_eq!(widen(*got), want);
        }
    }

    #[test]
    fn resolve_table_posture_matches_the_env_contract() {
        assert_eq!(
            resolve_table_posture(None).unwrap(),
            TablePostureSel::F16,
            "unset = fp16 default"
        );
        assert_eq!(
            resolve_table_posture(Some("")).unwrap(),
            TablePostureSel::F16,
            "empty = fp16 default"
        );
        assert_eq!(
            resolve_table_posture(Some("e8")).unwrap(),
            TablePostureSel::E8
        );
        let err = resolve_table_posture(Some("fp16")).expect_err("a typo refuses loud");
        let text = err.to_string();
        assert!(text.contains("LAYA_ANE_TABLE"), "names the env: {text}");
        assert!(text.contains("e8"), "names the supported set: {text}");
    }

    /// Hand-writes the sidecar container exactly as the converter does
    /// (u64-le header length + JSON header + data) and its manifest row.
    fn write_sidecar(dir: &Path, table_i8: &[i8], scales: &[f32]) -> u64 {
        let mut data = Vec::new();
        for s in scales {
            data.extend_from_slice(&s.to_le_bytes());
        }
        let scales_len = data.len();
        for q in table_i8 {
            data.push(*q as u8);
        }
        let vocab = scales.len();
        let d = table_i8.len() / vocab;
        let header = format!(
            r#"{{"scales":{{"dtype":"F32","shape":[{vocab}],"data_offsets":[0,{scales_len}]}},"table":{{"dtype":"I8","shape":[{vocab},{d}],"data_offsets":[{scales_len},{}]}}}}"#,
            data.len()
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("table_e8.safetensors"), &bytes).unwrap();
        bytes.len() as u64
    }

    fn e8_manifest_json(table_digest: &str, table_bytes: u64, serves: &str) -> String {
        format!(
            r#"{{"artifacts": {{
                "en/L8": {{
                    "bucket_L": 8,
                    "geometry": {{"hidden": {D}}},
                    "digest": {{"algo": "blake3-dir-v1", "digest": "00", "files": 3, "bytes": 9}},
                    "outputs": {{"hidden_state": {{"shape": [1, 8, {D}]}}}},
                    "mask_sentinel_fp16": -10000.0,
                    "placement": {{"ane_ops": 1, "device_ops": 1, "transitions": 0}}
                }},
                "en/table_e8": {{
                    "path": "assets/ane/en/table_e8.safetensors",
                    "bytes": {table_bytes},
                    "digest": {{"algo": "blake3", "digest": "{table_digest}", "files": 1, "bytes": {table_bytes}}},
                    "table": {{"tensor": "table", "dtype": "int8", "shape": [{VOCAB}, {D}]}},
                    "scales": {{"tensor": "scales", "dtype": "float32", "shape": [{VOCAB}]}},
                    "quant": {{"axis": "per_row_vocab"}},
                    "serves_buckets": {serves}
                }}
            }}}}"#
        )
    }

    fn fixture_manifest(root: &Path, serves: &str) -> AneManifest {
        let table_bytes = write_sidecar(&root.join("en"), &fixture_quants(), &fixture_scales());
        let digest =
            blake3::hash(&std::fs::read(root.join("en/table_e8.safetensors")).unwrap()).to_string();
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(
            root.join("manifest.json"),
            e8_manifest_json(&digest, table_bytes, serves),
        )
        .unwrap();
        AneManifest::load(&root.join("manifest.json")).expect("fixture manifest loads")
    }

    fn fixture_tok() -> Weights {
        Weights {
            shape: vec![VOCAB, D],
            data: WeightData::F32(vec![0.0; VOCAB * D]),
        }
    }

    #[test]
    fn sidecar_loads_verified_and_pairs_with_the_checkpoint() {
        let tmp = std::env::temp_dir().join(format!("ane_e8_t1_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("root");
        let manifest = fixture_manifest(&root, "[8]");
        let posture = load_table_e8(&root, "en", &manifest, &fixture_tok(), D, "riir")
            .expect("the verified sidecar loads");
        let TablePosture::E8 { table_i8, scales } = posture else {
            panic!("expected the e8 posture");
        };
        assert_eq!(table_i8, fixture_quants());
        assert_eq!(scales, fixture_scales());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn corrupt_sidecar_refuses_at_the_digest_wall() {
        let tmp = std::env::temp_dir().join(format!("ane_e8_t2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("root");
        let manifest = fixture_manifest(&root, "[8]");
        // Tamper AFTER the manifest pinned the digest.
        let path = root.join("en/table_e8.safetensors");
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        let err = load_table_e8(&root, "en", &manifest, &fixture_tok(), D, "riir")
            .expect_err("a tampered sidecar refuses");
        assert!(err.to_string().contains("digest"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn missing_sidecar_names_the_converter_command() {
        let tmp = std::env::temp_dir().join(format!("ane_e8_t3_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("root");
        let manifest = fixture_manifest(&root, "[8]");
        std::fs::remove_file(root.join("en/table_e8.safetensors")).unwrap();
        let err = load_table_e8(&root, "en", &manifest, &fixture_tok(), D, "riir")
            .expect_err("a set env with no sidecar refuses loud");
        let text = err.to_string();
        assert!(
            text.contains("ane_convert.py"),
            "names the converter: {text}"
        );
        assert!(
            text.contains("--table-precision e8"),
            "names the flag: {text}"
        );
        assert!(
            text.contains("fallback"),
            "says it never falls back: {text}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn missing_manifest_row_names_the_converter_command() {
        let tmp = std::env::temp_dir().join(format!("ane_e8_t4_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("root");
        let _manifest = fixture_manifest(&root, "[8]");
        // A manifest without the table_e8 row (a pre-Phase-1 tree).
        let json = r#"{"artifacts": {"en/L8": {
            "bucket_L": 8, "geometry": {"hidden": 3},
            "digest": {"algo": "blake3-dir-v1", "digest": "00", "files": 3, "bytes": 9},
            "outputs": {"hidden_state": {"shape": [1, 8, 3]}},
            "mask_sentinel_fp16": -10000.0,
            "placement": {"ane_ops": 1, "device_ops": 1, "transitions": 0}
        }}}"#;
        std::fs::write(root.join("manifest.json"), json).unwrap();
        let manifest = AneManifest::load(&root.join("manifest.json")).unwrap();
        let err = load_table_e8(&root, "en", &manifest, &fixture_tok(), D, "riir")
            .expect_err("no row refuses naming the generator");
        assert!(err.to_string().contains("ane_convert.py"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn wrong_checkpoint_pairing_refuses_at_load() {
        let tmp = std::env::temp_dir().join(format!("ane_e8_t5_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("root");
        let manifest = fixture_manifest(&root, "[8]");
        // A different checkpoint's embedding tensor — the pairing wall.
        let tok = Weights {
            shape: vec![VOCAB + 1, D],
            data: WeightData::F32(vec![0.0; (VOCAB + 1) * D]),
        };
        let err = load_table_e8(&root, "en", &manifest, &tok, D, "riir")
            .expect_err("a wrong-sidecar pairing refuses");
        assert!(err.to_string().contains("pair"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn serves_buckets_not_covering_the_checkpoint_refuses() {
        let tmp = std::env::temp_dir().join(format!("ane_e8_t6_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let root = tmp.join("root");
        let manifest = fixture_manifest(&root, "[64, 128]");
        let err = load_table_e8(&root, "en", &manifest, &fixture_tok(), D, "riir")
            .expect_err("a sidecar not serving this checkpoint's bucket refuses");
        assert!(err.to_string().contains("serves_buckets"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
