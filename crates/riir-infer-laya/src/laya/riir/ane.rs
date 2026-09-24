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

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::rc::Rc;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AnyThread;
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
}

impl AneManifest {
    /// Parse the committed manifest. Loose JSON: only the consumed fields
    /// are read, so the conversion tool can grow the schema freely.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| LayaError::Runtime(format!(
            "ane manifest unreadable at {}: {e}",
            path.display()
        )))?;
        let raw: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| LayaError::Runtime(format!("ane manifest is not JSON: {e}")))?;
        let arts = raw
            .get("artifacts")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| LayaError::Runtime("ane manifest: missing `artifacts`".into()))?;
        let mut artifacts = HashMap::new();
        for (key, a) in arts {
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
            let digest_files = a
                .pointer("/digest/files")
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
            let ane_ops = a
                .pointer("/placement/ane_ops")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| missing(key, "placement.ane_ops"))? as usize;
            let device_ops = a
                .pointer("/placement/device_ops")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| missing(key, "placement.device_ops"))? as usize;
            let transitions = a
                .pointer("/placement/transitions")
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
        Ok(Self { artifacts })
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
}

fn missing(key: &str, field: &str) -> LayaError {
    LayaError::Runtime(format!("ane manifest {key}: missing {field}"))
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
                .map_err(|e| LayaError::Runtime(format!(
                    "ane: Core ML load failed for {}: {}",
                    compiled.display(),
                    ns_error_string(&e)
                )))?;

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
            .map(|(p, _)| {
                std::fs::metadata(p)
                    .map(|m| m.len())
                    .unwrap_or_default()
            })
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
    pub fn predict(&self, emb_f16: &[u16], pad_bias_f16: &[u16], out_f16: &mut [u16]) -> Result<()> {
        let l = self.bucket_l;
        let d = self.hidden;
        assert_eq!(emb_f16.len(), l * d, "embeddings buffer must be [1, L, d]");
        assert_eq!(pad_bias_f16.len(), l, "pad_bias buffer must be [1, 1, 1, L]");
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
                    LayaError::Runtime(format!(
                        "ane: prediction failed: {}",
                        ns_error_string(&e)
                    ))
                })?;
            let name = NSString::from_str(&self.output);
            let val = out.featureValueForName(&name).ok_or_else(|| {
                LayaError::Runtime(format!(
                    "ane: output {:?} missing from the prediction (manifest/ artifact drift?)",
                    self.output
                ))
            })?;
            let arr = val.multiArrayValue().ok_or_else(|| {
                LayaError::Runtime(format!("ane: output {:?} is not a multi-array", self.output))
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
    d: usize,
    pad_id: u32,
    table_f16: Vec<u16>,
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
        let tok = map
            .remove(name)
            .ok_or_else(|| LayaError::Pin {
                checkpoint: ckpt,
                file: name.to_string(),
                detail: "tensor missing from checkpoint".into(),
            })?;
        let table_f16: Vec<u16> = tok.data.iter().map(|v| super::weights::f32_to_f16_bits(*v)).collect();
        Ok(Self {
            d: cfg.hidden,
            pad_id,
            table_f16,
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
    /// (f32) — the bucket covering `n` loads lazily on first use, the
    /// pad tail is masked, the fp16 output slices to `n` and widens
    /// bit-exactly. Refuses loud when `n` exceeds the largest bucket (a
    /// longer sequence is a FAILED forward, never a CPU fallback).
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let n = input_ids.len();
        let d = self.d;
        let buckets = self.buckets();
        let Some(&bucket) = buckets.iter().find(|b| **b >= n) else {
            return Err(LayaError::Runtime(format!(
                "ane: sequence length {n} exceeds every bucket this checkpoint's manifest \
                 carries {:?} — the ANE lane refuses, it does not fall back to CPU \
                 (no-silent-fallback law)",
                buckets
            )));
        };
        let rt = self.runtime_for(bucket)?;
        let l = rt.bucket_l;

        let entry = self.manifest.entry(self.model_dir, bucket)?;
        let sentinel_f16 = super::weights::f32_to_f16_bits(entry.mask_sentinel);
        let mut emb = vec![0u16; l * d];
        let mut pb = vec![0u16; l];
        gather_fp16(&self.table_f16, input_ids, self.pad_id, d, l, &mut emb);
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
    let entries =
        std::fs::read_dir(dir).map_err(|e| LayaError::Runtime(format!("ane: readdir {dir:?}: {e}")))?;
    for e in entries {
        let e = e.map_err(|err| LayaError::Runtime(format!("ane: readdir entry {dir:?}: {err}")))?;
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

/// Compile the `.mlpackage` into a content-addressed `.mlmodelc` cache
/// (recompiles never load a stale artifact: the cache key IS the digest)
/// and return the compiled bundle path. Cache location: `LAYA_ANE_CACHE`,
/// else the system temp dir — compile once per process tree, then
/// `MLModel` loads are ms-cheap.
fn compile_cached(artifact_dir: &Path, digest: &str) -> Result<PathBuf> {
    let key = &digest[..16.min(digest.len())];
    let cache_root = match std::env::var_os("LAYA_ANE_CACHE") {
        Some(p) => PathBuf::from(p),
        None => std::env::temp_dir().join("riir-laya-ane-cache"),
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
    let staging = cache_root.join(format!(
        "staging-{}-{}",
        key,
        std::process::id()
    ));
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
        let handler = RcBlock::new(
            move |plan: *mut MLComputePlan, err: *mut NSError| {
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
            },
        );
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
