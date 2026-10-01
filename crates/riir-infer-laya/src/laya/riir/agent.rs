//! The laya runtime, riir-owned backend: load a pinned checkpoint once
//! (our own safetensors reader — candle-free), answer typed questions.
//!
//! `forward_question` is the parity seam (the `tests/laya_riir_parity` gate
//! replays it against the SAME captured reference forwards the candle lane
//! gates on); `system_one` is the envelope surface the harness consumes.
//! Both share ONE forward path with the candle agent's exact semantics —
//! the shared envelope helpers (`option_keys` / `argmax_of` /
//! `confidence_from_probs`) live in [`super::super::types`], one copy for
//! both backends.
//!
//! `LAYA_DEVICE` is HONORED here (`.issues/005`): unset → the build's
//! default posture (Metal on macOS with `laya-riir-metal` compiled — the
//! Plan 001 T4 watchability default; CUDA on non-macOS with
//! `laya-riir-cuda` compiled — the 4090 lane, `.issues/002`; CPU
//! elsewhere), explicit `cpu`/`metal`/`cuda` is honored verbatim, `ane`
//! selects the whole-graph Apple Neural Engine lane (feature
//! `laya-riir-ane`, macOS — Plan 002 P1) and — like `metal`/`cuda` —
//! fails loud when the feature or platform is absent, never a silent
//! fallback. Anything else fails loud too (an env typo must never fall
//! back to CPU — the candle lane's `device_from_env` precedent). The gate
//! law is unchanged: G5 parity must be green at WHICHEVER posture a number
//! is published from (for the ANE lane, that is the consumer-side
//! G5-ANE decision-level gate).

use serde_json::Value;

use super::super::config::{AgentConfig, Checkpoint, load_checkpoint_configs};
use super::super::render::{py_round4, render_options};
use super::super::temps::{Temperatures, softmax32, temp_bucket};
use super::super::tokenize::{InternalQuestion, Tok, build_sequence, to_internal};
use super::super::types::{Answer, Forward, argmax_of, confidence_from_probs, option_keys};
use super::super::weights::ensure_checkpoint;
use super::super::{LayaError, Result};
use super::backend::{Backend, Cpu};
use super::encoder::Encoder;
use super::fake_quant::WeightPosture;
use super::head::{Head, HeadOutput, HeadScratch};

/// The device the riir forward runs on, chosen at load from `LAYA_DEVICE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// The flat-`Vec` CPU backend (the lane's original posture).
    Cpu,
    /// The MSL backend (`laya-riir-metal`, macOS).
    Metal,
    /// The CUDA backend (`laya-riir-cuda`, non-macOS — `.issues/002`).
    Cuda,
    /// The Apple Neural Engine whole-graph lane (`laya-riir-ane`, macOS —
    /// reflex Plan 002 P1). The encoder runs a Core ML artifact; the head
    /// stays on the CPU backend.
    Ane,
    /// The portable CubeCL/wgpu backend (`laya-riir-cubecl`, plan 611) —
    /// the op-layer unification arm. Never a runtime default: selected
    /// explicitly; the A/B (plan 611 S5) decides its fate.
    Cubecl,
}

impl DeviceKind {
    /// Resolve `LAYA_DEVICE` — unset/empty → [`Self::default_device`]
    /// (Metal when this build SHIPS the Metal backend on macOS, CUDA when
    /// it ships the CUDA backend elsewhere — the build's own posture;
    /// CPU when it ships neither; ANE is NEVER a default — the artifact
    /// tree is local-only and the lane is opt-in), `cpu` →
    /// [`DeviceKind::Cpu`] (the explicit opt-out), `metal` →
    /// [`DeviceKind::Metal`], `cuda` → [`DeviceKind::Cuda`], `ane` →
    /// [`DeviceKind::Ane`], `cubecl` → [`DeviceKind::Cubecl`] (plan 611),
    /// anything else is an error (an env typo must fail loud, never fall
    /// back).
    pub fn from_env() -> Result<Self> {
        match std::env::var("LAYA_DEVICE").as_deref() {
            Ok("") | Err(_) => Ok(Self::default_device()),
            Ok("cpu") => Ok(Self::Cpu),
            Ok("metal") => Ok(Self::Metal),
            Ok("cuda") => Ok(Self::Cuda),
            Ok("ane") => Ok(Self::Ane),
            Ok("cubecl") => Ok(Self::Cubecl),
            Ok(other) => Err(LayaError::Config {
                checkpoint: "riir",
                detail: format!(
                    "unknown LAYA_DEVICE {other:?} — expected unset, \"cpu\", \"metal\", \"cuda\", \"ane\" or \"cubecl\""
                ),
            }),
        }
    }

    /// The no-env posture: the device this build SHIPS — Metal where the
    /// Metal backend is compiled and exists (macOS + `laya-riir-metal`),
    /// CUDA where the CUDA backend is compiled (non-macOS +
    /// `laya-riir-cuda` — the 4090 lane, `.issues/002`), CPU everywhere
    /// else. An explicit env value is always honored verbatim — only the
    /// ABSENT choice defaults.
    pub fn default_device() -> Self {
        #[cfg(all(target_os = "macos", feature = "laya-riir-metal"))]
        {
            Self::Metal
        }
        #[cfg(all(not(target_os = "macos"), feature = "laya-riir-cuda"))]
        {
            Self::Cuda
        }
        #[cfg(not(any(
            all(target_os = "macos", feature = "laya-riir-metal"),
            all(not(target_os = "macos"), feature = "laya-riir-cuda"),
        )))]
        {
            Self::Cpu
        }
    }
}

/// One autoreleasepool spanning one forward. The Metal backend creates
/// autoreleased command buffers per op; on a thread with no Cocoa runloop
/// they would otherwise accumulate until thread exit. With the metal
/// feature this is `objc2::rc::autoreleasepool`; without it, identity
/// (one code path for both postures).
#[cfg(all(target_os = "macos", feature = "laya-riir-metal"))]
fn pass_pool<T>(f: impl FnOnce() -> T) -> T {
    objc2::rc::autoreleasepool(|_| f())
}

/// [`pass_pool`] identity form when no Metal backend is compiled.
#[cfg(not(all(target_os = "macos", feature = "laya-riir-metal")))]
fn pass_pool<T>(f: impl FnOnce() -> T) -> T {
    f()
}

/// The packed-pass kill-switch (`RIIR_LAYA_NO_BATCH=1`, read once): the
/// per-question loop is the A/B arm and the bisect posture, never a
/// silent default — only the explicit `1` disables.
fn batch_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("RIIR_LAYA_NO_BATCH").as_deref() == Ok("1"))
}

/// T12 posture (reflex issue 020): the packed driver defers every
/// question's head reads into two drain classes instead of three reads
/// per question. DEFAULT ON since the quiet-box paired A/B
/// (`tests/metal_head_defer_ab.rs`, 24 paired rounds/shape,
/// position-balanced, AC load < 6 preflight): typed 5-q median on/off
/// **0.984 (24/24 wins)**, 5-q short **0.984 (22/24)**, and the 1-q
/// wiring control FLAT (1.002 — `packed_eligible` excludes it, so both
/// postures run the identical loop path). Bit-identical either way
/// (`packed_same_shape` raw-bit gate + `packed_forward_equiv` drift
/// budget). Kill-switch `LAYA_HEAD_DEFER=0` (the `LAYA_METAL_LN_WIDE`
/// spelling) restores the composed per-question forward.
fn head_defer() -> bool {
    static DEFER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DEFER.get_or_init(|| std::env::var("LAYA_HEAD_DEFER").as_deref() != Ok("0"))
}

/// The encoder half of the stack: the per-op lanes share the op-stream
/// [`Encoder`]; the ANE lane swaps in the whole-graph executor. One enum
/// at ONE seam — `forward_internal`'s match is the only place the two
/// shapes meet (the head consumes `[n, d]` f32 either way).
enum EncoderStack {
    Local(Encoder),
    #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
    Ane(super::ane::AneEncoder),
}

impl EncoderStack {
    /// Warm the per-op lanes' weights on the backend (Issue 020 T1). The
    /// ANE variant carries no per-op weights — nothing to place. Without
    /// the ane feature the enum has ONE variant, so the plain `let` is
    /// irrefutable and compiles warning-free.
    fn warm(&self, b: &dyn Backend) {
        #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
        if let EncoderStack::Local(e) = self {
            return e.warm(b);
        }
        #[cfg(not(all(target_os = "macos", feature = "laya-riir-ane")))]
        {
            let EncoderStack::Local(e) = self;
            e.warm(b);
        }
        #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
        let _ = b;
    }
}

/// The [`RiirAgent::encode_question`] output: the encoder's output
/// residual stream (pre-type-emb) plus the head-addressing facts the
/// caller needs to run or train a head over it (riir-train Plan 425's
/// cached-feature lane).
pub struct EncodedQuestion {
    /// The encoder's final hidden states, row-major `[seq_len, d]` —
    /// PRE-type-emb (the type-embedding add is the head's own first op).
    pub hidden: Vec<f32>,
    /// The head's FROZEN representation half: the gathered marker rows
    /// `[k_opts, d]` AFTER the type-emb add + both head layers — the exact
    /// bytes the scorer's LN would consume (the trainer trains the scorer
    /// on these without re-running the encoder or the layers).
    pub marker_rows: Vec<f32>,
    /// The head's marker positions (one per rendered option).
    pub markers: Vec<usize>,
    /// The question type (0 choice / 1 score / 2 noul — the type-emb row).
    pub qtype: usize,
    /// The token count of the forward (`hidden.len() == seq_len * d`).
    pub seq_len: usize,
    /// The encoder's hidden width.
    pub d: usize,
}

/// The q8-artifact posture's disclosure: the decode built no error report
/// (nothing was transformed at load — the artifact IS the quantized
/// form), but the record still names the quantized surface (tensors/
/// elements; the 1D tensors carried F16) with zero error (the stored
/// values ARE the weights).
fn quant_artifact_disclosure(
    raw: &std::collections::HashMap<String, super::weights::Weights>,
) -> super::fake_quant::FakeQuantReport {
    let mut rep = super::fake_quant::FakeQuantReport {
        quantized_tensors: 0,
        quantized_elements: 0,
        blocks: 0,
        skipped_tensors: Vec::new(),
        max_abs_err: 0.0,
        mean_abs_err: 0.0,
        quantized_f16_bytes: 0,
    };
    let mut names: Vec<&String> = raw.keys().collect();
    names.sort_unstable();
    for tensor_name in names {
        let w = &raw[tensor_name];
        if w.shape.len() < 2 {
            rep.skipped_tensors.push(tensor_name.clone());
            continue;
        }
        rep.quantized_tensors += 1;
        let numel = w.numel();
        rep.quantized_elements += numel;
        rep.quantized_f16_bytes += numel as u64 * 2;
        rep.blocks += numel.div_ceil(super::fake_quant::BLOCK);
    }
    rep
}

/// A loaded checkpoint (riir backend): tokenizer + encoder + head +
/// temperature tables, plus the device backend the forward runs on.
pub struct RiirAgent {
    tok: Tok,
    enc: EncoderStack,
    head: Head,
    backend: Box<dyn Backend>,
    /// The posture label (`"cpu"` / `"metal"` / `"ane"`) — stored, not
    /// derived from `backend.name()`, because the ANE posture's HEAD runs
    /// the CPU backend while the lane label must read `ane` (a timing
    /// line can never be mistaken for another posture).
    device_label: &'static str,
    temps: Temperatures,
    cfg: AgentConfig,
    ckpt: &'static str,
    /// The T12 A/B seam (reflex issue 020): `None` = the env posture
    /// ([`head_defer`]); `Some(_)` overrides it for this agent. The paired
    /// A/B harness toggles both postures in ONE process (the fold A/B's
    /// `with_folds` pattern — pairing cancels between-round box drift);
    /// serving paths never touch it.
    head_defer_override: Option<bool>,
    /// The weight-posture disclosure (instinct issue 018 Lane D1/D2a):
    /// `Some` iff this agent loaded under a non-F16 weight posture —
    /// fake-quant (the measured error) or the Q8 artifact (the tensor
    /// surface; values byte-identical to the fake-quant by the
    /// converter's proof). `None` (the shipped posture) discloses
    /// nothing because nothing changed.
    fake_quant: Option<(
        super::fake_quant::WeightPosture,
        super::fake_quant::FakeQuantReport,
    )>,
}

impl RiirAgent {
    /// Load one checkpoint from the weights root (downloading + verifying
    /// against the pins first — never bundled). The safetensors file is
    /// parsed ONCE and split between encoder and head (weights are removed
    /// from the map, no second copy). The device comes from `LAYA_DEVICE`.
    pub fn load(root: &std::path::Path, ckpt: Checkpoint) -> Result<Self> {
        Self::load_with_device(root, ckpt, DeviceKind::from_env()?)
    }

    /// Load one checkpoint on an EXPLICIT device — no env round-trip (an
    /// env value can never silently demote an explicitly requested lane;
    /// the `load_ane` rule). The A/B and parity harnesses construct the
    /// postures they compare in ONE process through this constructor —
    /// `LAYA_DEVICE` cannot express that, and mutating it mid-process
    /// would race every other reader.
    pub fn load_with_device(
        root: &std::path::Path,
        ckpt: Checkpoint,
        device: DeviceKind,
    ) -> Result<Self> {
        Self::load_inner(
            root,
            ckpt,
            None,
            device,
            super::fake_quant::WeightPosture::F16,
        )
    }

    /// Load one checkpoint under an explicit WEIGHT posture (instinct
    /// issue 018 Lane D1): the device comes from `LAYA_DEVICE` exactly as
    /// [`Self::load`], and the checkpoint's >=2D tensors are fake-
    /// quantized Q8_0 at load (quantize-then-dequantize, forward
    /// unchanged). Measurement-only — serving paths never call this; the
    /// posture is disclosed by [`Self::weight_posture`] and the returned
    /// report, never inferred.
    pub fn load_with_posture(
        root: &std::path::Path,
        ckpt: Checkpoint,
        posture: super::fake_quant::WeightPosture,
    ) -> Result<Self> {
        Self::load_inner(root, ckpt, None, DeviceKind::from_env()?, posture)
    }

    /// Load one checkpoint for the ANE posture (Plan 002 P1): the encoder
    /// half executes the digest-pinned Core ML artifacts under `ane_root`
    /// (verified against `manifest_path`), the gather + head stay host-side.
    /// Only meaningful when `LAYA_DEVICE=ane` selected the ANE lane; call
    /// it instead of [`Self::load`] with the same env set.
    #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
    pub fn load_ane(
        root: &std::path::Path,
        ckpt: Checkpoint,
        ane_root: &std::path::Path,
        manifest_path: &std::path::Path,
    ) -> Result<Self> {
        Self::load_inner(
            root,
            ckpt,
            Some((ane_root, manifest_path)),
            DeviceKind::Ane,
            super::fake_quant::WeightPosture::F16,
        )
    }

    fn load_inner(
        root: &std::path::Path,
        ckpt: Checkpoint,
        ane: Option<(&std::path::Path, &std::path::Path)>,
        device: DeviceKind,
        posture: super::fake_quant::WeightPosture,
    ) -> Result<Self> {
        // The caller owns the device choice (env via [`Self::load`], the
        // explicit constructor, or the ANE lane); nothing here re-reads it.
        #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
        let ane_requested = ane.is_some();
        #[cfg(not(all(target_os = "macos", feature = "laya-riir-ane")))]
        let ane_requested = false;
        if device == DeviceKind::Ane && !ane_requested {
            return Err(LayaError::Config {
                checkpoint: ckpt.subfolder(),
                detail: "the ANE lane needs --features laya-riir-ane on macOS, and an \
                         explicit RiirAgent::load_ane(root, ckpt, ane_root, manifest) — \
                         LAYA_DEVICE=ane through the plain constructor is refused \
                         (fail loud, never a silent fallback)"
                    .into(),
            });
        }
        // The derived-artifact postures are STORAGE selections, not
        // in-memory transforms — the env owns them
        // (LAYA_WEIGHTS_VARIANT=q8|q4). An explicit
        // load_with_posture(<artifact posture>) with the env unset would
        // load F16 weights under a quantized label — refused, never a
        // silent mislabel.
        let posture_env_ok = match posture {
            WeightPosture::Q8Artifact => {
                matches!(std::env::var("LAYA_WEIGHTS_VARIANT").as_deref(), Ok("q8"))
            }
            WeightPosture::Q4Artifact => {
                matches!(std::env::var("LAYA_WEIGHTS_VARIANT").as_deref(), Ok("q4"))
            }
            _ => true,
        };
        if !posture_env_ok {
            let env = if posture == WeightPosture::Q4Artifact {
                "q4"
            } else {
                "q8"
            };
            return Err(LayaError::Config {
                checkpoint: ckpt.subfolder(),
                detail: format!(
                    "WeightPosture::{} is selected by LAYA_WEIGHTS_VARIANT={env} \
                     (the storage variant), not by load_with_posture — set the env \
                     or use WeightPosture::F16",
                    posture.label()
                ),
            });
        }
        let dir = ensure_checkpoint(root, ckpt)?;
        let name = ckpt.subfolder();
        let (agent_cfg, enc_cfg) = load_checkpoint_configs(&dir, name)?;

        let tok = Tok::from_dir(&dir, name)?;
        // The storage variant (instinct issue 018 Lane D2a; Plan 616
        // Phase 3 adds q4): LAYA_WEIGHTS_VARIANT=q8|q4 loads the derived
        // artifact (sidecar-verified) instead of the canonical F16 file —
        // the decode arithmetic is the fake-quant path's own, so the
        // numerics are the probe's measured ones with NO in-memory
        // transform. The explicit postures below win over the env where
        // they disagree is a REFUSAL, never a silent pick: FakeQuantQ8 on
        // any artifact would quantize twice.
        let (weights_path, artifact_posture) =
            super::q8_artifact::resolve_weights_posture(&dir, name)?;
        if artifact_posture.is_some() {
            if matches!(posture, WeightPosture::FakeQuantQ8) {
                return Err(LayaError::Config {
                    checkpoint: name,
                    detail: "--fake-quant over a derived quant artifact would quantize TWICE — \
                             the artifact already carries quantized values; \
                             drop --fake-quant or unset LAYA_WEIGHTS_VARIANT"
                        .into(),
                });
            }
            if ane_requested {
                return Err(LayaError::Config {
                    checkpoint: name,
                    detail: "the ANE lane runs the Core ML artifact — a quant weights \
                             variant does not apply (refusing, never a silent ignore)"
                        .into(),
                });
            }
        }
        let mut raw = super::weights::load(&weights_path, name)?;

        // The weight posture (instinct issue 018 Lane D1/D2a): applied BEFORE
        // the encoder/head split so every per-op lane sees the same
        // quantized bytes. The ANE lane refuses — its layer weights live
        // in the Core ML artifact, so a map-level transform would be a
        // silent no-op wearing a quantized label.
        let fake_quant = match posture {
            WeightPosture::F16 => None,
            // The artifact postures load that way above — no in-memory
            // transform here (Q4Artifact rides the same arm).
            WeightPosture::Q8Artifact | WeightPosture::Q4Artifact => None,
            WeightPosture::FakeQuantQ8 if ane_requested => {
                return Err(LayaError::Config {
                    checkpoint: name,
                    detail: "fake-quant does not apply to the ANE lane — its layer weights \
                             live in the Core ML artifact, not this map; refusing rather \
                             than labeling an unquantized forward"
                        .into(),
                });
            }
            WeightPosture::FakeQuantQ8 => {
                let rep = super::fake_quant::fake_quant_q8_map(&mut raw).map_err(|e| {
                    LayaError::Config {
                        checkpoint: name,
                        detail: e,
                    }
                })?;
                Some(rep)
            }
        };
        // A derived-artifact posture's disclosure: the decode built no
        // report (nothing was transformed here), but the record needs the
        // tensor surface — derive it from the loaded map's shape words.
        let fake_quant = if artifact_posture.is_some() {
            Some(quant_artifact_disclosure(&raw))
        } else {
            fake_quant
        };

        let (enc, backend): (EncoderStack, Box<dyn Backend>) = if ane_requested {
            #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
            {
                let Some((ane_root, manifest_path)) = ane else {
                    return Err(LayaError::Config {
                        checkpoint: name,
                        detail: "ANE posture selected but no artifact root/ manifest path — \
                                 call RiirAgent::load_ane"
                            .into(),
                    });
                };
                let enc = super::ane::AneEncoder::from_map(
                    &mut raw,
                    enc_cfg.clone(),
                    name,
                    name,
                    tok.pad,
                    ane_root.to_path_buf(),
                    manifest_path,
                )?;
                // Every bucket loads HERE, not on the first request that
                // reaches it — a lazy load is a ~0.9 s spike inside
                // somebody's request (riir-reflex Bench 042).
                enc.preload()?;
                // The head runs the plain CPU backend (f32, unchanged).
                (EncoderStack::Ane(enc), Box::new(Cpu))
            }
            #[cfg(not(all(target_os = "macos", feature = "laya-riir-ane")))]
            {
                let _ = ane;
                unreachable!("ane_requested is only true behind the ane feature")
            }
        } else {
            let enc = Encoder::from_map(&mut raw, enc_cfg.clone(), name)?;
            let backend: Box<dyn Backend> = match device {
                DeviceKind::Cpu => Box::new(Cpu),
                #[cfg(all(target_os = "macos", feature = "laya-riir-metal"))]
                DeviceKind::Metal => Box::new(super::metal::Metal::new()?),
                #[cfg(not(all(target_os = "macos", feature = "laya-riir-metal")))]
                DeviceKind::Metal => {
                    return Err(LayaError::Config {
                        checkpoint: name,
                        detail: "LAYA_DEVICE=metal needs --features laya-riir-metal on macOS — \
                                 this build has no Metal backend (fail loud, never a silent \
                                 CPU fallback)"
                            .into(),
                    });
                }
                #[cfg(all(not(target_os = "macos"), feature = "laya-riir-cuda"))]
                DeviceKind::Cuda => Box::new(super::cuda::Cuda::new()?),
                #[cfg(not(all(not(target_os = "macos"), feature = "laya-riir-cuda")))]
                DeviceKind::Cuda => {
                    return Err(LayaError::Config {
                        checkpoint: name,
                        detail: "LAYA_DEVICE=cuda needs --features laya-riir-cuda on a \
                                 non-macOS CUDA host — this build has no CUDA backend (fail \
                                 loud, never a silent CPU fallback)"
                            .into(),
                    });
                }
                #[cfg(feature = "laya-riir-cubecl")]
                DeviceKind::Cubecl => Box::new(super::cubecl::CubeclBackend::new()?),
                #[cfg(not(feature = "laya-riir-cubecl"))]
                DeviceKind::Cubecl => {
                    return Err(LayaError::Config {
                        checkpoint: name,
                        detail: "LAYA_DEVICE=cubecl needs --features laya-riir-cubecl — this \
                                 build has no CubeCL backend (fail loud, never a silent \
                                 CPU fallback)"
                            .into(),
                    });
                }
                DeviceKind::Ane => unreachable!("handled above"),
            };
            (EncoderStack::Local(enc), backend)
        };
        // `raw` still holds the head's tensors (the encoder took its own;
        // in the ANE posture only the embedding table left the map). The
        // layer weights are dropped here — the ANE artifact holds them.
        let head = Head::from_map(&mut raw, name, enc_cfg.hidden, enc_cfg.eps)?;
        drop(raw);

        // riir-reflex Issue 020 T1 — device residency is a LOAD cost, not a
        // first-request cost. Without this the ~0.5 GB of f32 projections
        // (english geometry) upload lazily inside `system_one` #1, which is
        // the call the bench times and the call a served client waits on;
        // the torch reference moves its weights inside `load`, before its
        // own handshake. The ANE lane's encoder carries no per-op weights
        // (the artifact holds them fp16); its head still warms. No-op on
        // the CPU backend.
        enc.warm(backend.as_ref());
        head.warm(backend.as_ref());

        let temps = Temperatures::from_config(&agent_cfg);
        let device_label = if ane_requested { "ane" } else { backend.name() };
        let posture_word = if let Some(p) = artifact_posture {
            p
        } else {
            posture
        };
        Ok(Self {
            tok,
            enc,
            head,
            backend,
            device_label,
            temps,
            cfg: agent_cfg,
            ckpt: name,
            head_defer_override: None,
            fake_quant: fake_quant.map(|r| (posture_word, r)),
        })
    }

    /// The checkpoint this agent serves.
    pub fn checkpoint(&self) -> &'static str {
        self.ckpt
    }

    /// The scorer readout's tensors, cloned out for the training lane
    /// (riir-train Plan 425: the banking77 teacher fine-tune initializes
    /// from the checkpoint's own scorer). See [`super::head::ScorerTensors`].
    pub fn scorer_tensors(&self) -> super::head::ScorerTensors {
        self.head.scorer_tensors()
    }

    /// The backend posture this agent runs (`"cpu"` / `"metal"` /
    /// `"ane"`) — gate lines and timing labels print it so a reading can
    /// never be mistaken for the other posture.
    pub fn device(&self) -> &'static str {
        self.device_label
    }

    /// The T12 measurement seam (reflex issue 020): override the packed
    /// head posture for this agent — `Some(true)` deferred (two drain
    /// classes per case), `Some(false)` composed (the per-question
    /// forward), `None` the env default (`LAYA_HEAD_DEFER=1`). Returns
    /// the previous override so a harness can restore it.
    pub fn set_head_defer_override(&mut self, defer: Option<bool>) -> Option<bool> {
        std::mem::replace(&mut self.head_defer_override, defer)
    }

    /// The weight posture this agent loaded under (instinct issue 018
    /// Lane D1/D2a) with its disclosure report — `None` = the shipped
    /// F16 posture, `Some((posture, report))` = fake-quant (the report
    /// names every skipped tensor and the measured error) or the Q8
    /// artifact (the report names the quantized tensor surface; the
    /// values ARE the probe's, byte-identical by the converter's proof).
    pub fn weight_posture(
        &self,
    ) -> Option<(
        super::fake_quant::WeightPosture,
        &super::fake_quant::FakeQuantReport,
    )> {
        self.fake_quant.as_ref().map(|(p, r)| (*p, r))
    }

    /// The largest sequence length the ANE lane can serve (its biggest
    /// manifest bucket) — `None` when this is not the ANE posture. The
    /// consumer-side gate uses it to name + floor the out-of-bucket skips
    /// WITHOUT paying a probe forward.
    #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
    pub fn ane_bucket_max(&self) -> Option<usize> {
        match &self.enc {
            EncoderStack::Local(_) => None,
            EncoderStack::Ane(e) => e.buckets().last().copied(),
        }
    }

    /// Forward one question against `state` — one unpadded sequence, the
    /// reference capture's exact batch shape and the G5 parity seam. Errors
    /// when options exceed the head budget (the reference raises the same
    /// class) or when a marker was truncated away.
    pub fn forward_question(&self, state: &Value, qdef: &Value) -> Result<Forward> {
        let q = to_internal(qdef)?;
        self.forward_internal(state, &q)
    }

    /// The training-seam twin of [`Self::forward_question`] (riir-train
    /// Plan 425: the banking77 teacher fine-tune's cached-feature lane):
    /// everything up to and INCLUDING the encoder forward, NOT the head —
    /// the caller gets the encoder's output residual stream (pre-type-emb:
    /// the type-embedding add is the head's own first op, so the cache is
    /// reusable at any `qtype`) plus the marker positions and `qtype` the
    /// head would consume. Same `build_sequence` budget law and the same
    /// marker-truncation refusal as the forward path, so a cache row and
    /// the teacher pass's row can never disagree about shape. Zero effect
    /// on any existing path — pure addition.
    pub fn encode_question(&self, state: &Value, qdef: &Value) -> Result<EncodedQuestion> {
        let q = to_internal(qdef)?;
        let opts = render_options(&q);
        let (ids, markers) = build_sequence(
            &self.tok,
            state,
            &q,
            self.cfg.max_len,
            self.cfg.head_max_len,
        )?;
        if opts.is_empty() || markers.len() != opts.len() {
            return Err(LayaError::Question(format!(
                "options exceed the head budget: {} rendered, {} markers survived",
                opts.len(),
                markers.len()
            )));
        }
        // The cache lane's whole point: also capture the head's FROZEN
        // representation half (type-emb add + both head layers + marker
        // gather) so the trainer never re-runs the encoder or the layer
        // stack. Same pass/epoch contract as [`Self::forward_internal`]
        // (begin_pass BEFORE the encoder — the metal device slots are
        // epoch-keyed), and BOTH host-read results sync through the
        // backend (under Metal the op stream only drains at a
        // download_into; under CPU these are plain copies).
        let (hidden, marker_rows, d) = pass_pool(|| {
            self.backend.begin_pass();
            let mut h = match &self.enc {
                EncoderStack::Local(e) => e.forward(self.backend.as_ref(), &ids)?,
                #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
                EncoderStack::Ane(_) => {
                    return Err(LayaError::Runtime(
                        "encode_question: the ANE lane is not supported — run the cache lane \
                         on cpu/metal/cuda (no residual-stream seam)"
                            .into(),
                    ));
                }
            };
            let d = match &self.enc {
                EncoderStack::Local(e) => e.hidden_dim(),
                #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
                EncoderStack::Ane(_) => 0,
            };
            let mut sc = HeadScratch::new();
            self.head.layers_forward_rows(
                self.backend.as_ref(),
                &mut h,
                q.qtype,
                &markers,
                &mut sc,
            )?;
            let k = markers.len();
            let mut rows_host = vec![0f32; k * d];
            self.backend.download_into(&sc.rows, &mut rows_host);
            let mut h_host = vec![0f32; h.len()];
            self.backend.download_into(&h, &mut h_host);
            Ok((h_host, rows_host, d))
        })?;
        Ok(EncodedQuestion {
            seq_len: ids.len(),
            markers,
            qtype: q.qtype,
            hidden,
            marker_rows,
            d,
        })
    }

    /// The token twin of [`Self::encode_question`] (riir-train 602: the
    /// static-vector surrogate's corpus-averaging join). Returns the SAME
    /// token stream `encode_question` feeds the encoder — the identical
    /// `to_internal` + `build_sequence` construction with the same budgets —
    /// so a caller can key per-token hidden states by token id with a
    /// row-for-row `ids.len() == seq_len` join and zero drift. Pure
    /// addition; touches no forward path.
    pub fn tokenize_question(&self, state: &Value, qdef: &Value) -> Result<(Vec<u32>, Vec<usize>)> {
        let q = to_internal(qdef)?;
        let (ids, markers) = build_sequence(
            &self.tok,
            state,
            &q,
            self.cfg.max_len,
            self.cfg.head_max_len,
        )?;
        Ok((ids, markers))
    }

    /// The internal-typed variant (avoids re-parsing per row in the test).
    pub fn forward_internal(&self, state: &Value, q: &InternalQuestion) -> Result<Forward> {
        let opts = render_options(q);
        let (ids, markers) =
            build_sequence(&self.tok, state, q, self.cfg.max_len, self.cfg.head_max_len)?;
        if opts.is_empty() || markers.len() != opts.len() {
            return Err(LayaError::Question(format!(
                "options exceed the head budget: {} rendered, {} markers survived",
                opts.len(),
                markers.len()
            )));
        }
        // One autoreleasepool per forward: the Metal backend's per-op
        // autoreleased command buffers drain here instead of accumulating
        // on a thread with no Cocoa runloop (a no-op wrapper without the
        // metal feature — one code path for both postures). The ANE lane's
        // Core ML calls drain their autoreleased temporaries here too.
        let out = pass_pool(|| -> Result<HeadOutput> {
            self.backend.begin_pass();
            let mut hidden = match &self.enc {
                EncoderStack::Local(e) => e.forward(self.backend.as_ref(), &ids)?,
                #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
                EncoderStack::Ane(e) => e.forward(&ids)?,
            };
            let mut sc = HeadScratch::new();
            self.head.forward(
                self.backend.as_ref(),
                &mut hidden,
                q.qtype,
                &markers,
                &mut sc,
            )
        })?;
        let t = self.temps.for_question(q.qtype, markers.len());
        Ok(Self::make_forward(q, out, ids.len(), markers, t))
    }

    /// The shared forward tail: head outputs → temperature-scaled probs →
    /// the [`Forward`] envelope. ONE copy for the per-question path and
    /// the packed path (bit-identical math either way); the temperature is
    /// resolved by the caller so this stays a pure associated fn.
    fn make_forward(
        q: &InternalQuestion,
        out: HeadOutput,
        seq_len: usize,
        markers: Vec<usize>,
        t: f64,
    ) -> Forward {
        let k = markers.len();
        let z: Vec<f32> = out.logits.iter().map(|l| l / t as f32).collect();
        let probs = softmax32(&z);
        let confidence = confidence_from_probs(&probs, k);
        Forward {
            logits: out.logits,
            probs,
            confidence,
            act_probabilities: out.act_probabilities,
            temperature_used: t,
            bucket: temp_bucket(q.qtype, k),
            seq_len,
            markers,
        }
    }

    /// `system_one` over multiple questions (one forward each — the
    /// capture's posture; the reference batches, which is numerically
    /// equivalent modulo padding).
    ///
    /// When the backend can pack (the fused Metal kernel, or CPU) and
    /// `RIIR_LAYA_NO_BATCH=1` is unset, the case's questions run through
    /// ONE packed encoder pass instead ([`Self::system_one_packed`]) —
    /// per-question answers are bit-identical, the pass boundaries
    /// collapse to one per case.
    pub fn system_one(&self, state: &Value, questions: &[(String, Value)]) -> Result<Vec<Answer>> {
        if questions.is_empty() {
            return Ok(Vec::new());
        }
        if self.packed_eligible(questions.len()) {
            return self.system_one_packed(state, questions);
        }
        let mut out = Vec::with_capacity(questions.len());
        for (qid, qdef) in questions {
            let q = to_internal(qdef)?;
            let f = self.forward_internal(state, &q)?;
            out.push(Self::answer_of(qid, &q, f)?);
        }
        Ok(out)
    }

    /// Can this case run the packed pass? The per-question loop is the
    /// answer whenever anything is off: the env kill-switch
    /// (`RIIR_LAYA_NO_BATCH=1` — also the A/B arm), the ANE lane (its
    /// whole-graph encoder is bucket-shaped, one sequence per call), or a
    /// backend that cannot execute the packed attention at this head dim.
    ///
    /// Single-question cases are excluded too (Bench 006 Addendum 7): the
    /// packed pass's per-case overhead — slab copies + the collate pass —
    /// has nothing to amortize at n=1, measured 1.045 median B/A p50 on
    /// the 1-q control suite vs 0.91-0.94 at 5 q/case. The loop path for
    /// one question IS the packed path's per-sequence math, so the gate
    /// costs nothing measurable and the multi-q wins keep their arm.
    fn packed_eligible(&self, question_count: usize) -> bool {
        if batch_disabled() || question_count < 2 {
            return false;
        }
        #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
        if matches!(self.enc, EncoderStack::Ane(_)) {
            return false;
        }
        match &self.enc {
            EncoderStack::Local(e) => self.backend.supports_packed_attention(e.head_dim()),
            #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
            EncoderStack::Ane(_) => false,
        }
    }

    /// The reference's `system_one` shape: all of a case's questions
    /// through ONE forward (`.raw/laya/laya/agent.py:267` — "Evaluate typed
    /// questions across state in a single, parallel forward pass"). Ours
    /// packs instead of padding — exact per-sequence attention, no pad
    /// rows, no attention-mask tensor. The per-row OUTPUTS are not
    /// bit-identical to the per-question loop — the batched GEMMs run
    /// different shapes, so different sgemm instances (reduction orders)
    /// apply; they agree to the [`Encoder::forward_packed`] drift budget
    /// (CPU 1e-5 / Metal 1e-4, `tests/packed_forward_equiv`), and the
    /// answers agree through the rounded envelope plus the raw-bit
    /// equal-shape gate ([`tests/packed_same_shape_gate`], the head
    /// pipeline per question IS shape-identical).
    ///
    /// The head runs PER QUESTION on its own exact-size slab, copied
    /// device-side out of the packed residual ([`Backend::copy_at`]) —
    /// the head's op stream is untouched. Every question's work is
    /// enqueued inside one pass (one `begin_pass`, one chain epoch), so
    /// the first head download drains the whole case and the rest find an
    /// empty pipeline.
    /// The packed path's per-question reads split out of [`Head::forward`]
    /// (issue 020 T12): the per-question streams run in THREE PHASES over
    /// the case (the default; kill-switch `LAYA_HEAD_DEFER=0` restores the
    /// composed form) — 1. every `copy_at` + layers + scorer enqueues; 2.
    /// the scorer-logits + CLS reads (the FIRST read drains the whole
    /// case's stage-1 stream — the other four are plain copies); 3. the act
    /// heads (enqueue, then the act read). Two drain classes per case
    /// instead of the composed form's three reads × questions — same ops,
    /// same order per question, bit-identical outputs. The quiet-box
    /// paired A/B measured typed 5-q median 0.984 (24/24 wins) with the
    /// 1-q wiring control flat.
    fn system_one_packed(
        &self,
        state: &Value,
        questions: &[(String, Value)],
    ) -> Result<Vec<Answer>> {
        // Collate first — the reference's `items` shape: ids + markers per
        // question, checked before any GPU work.
        let mut items: Vec<(String, InternalQuestion, Vec<u32>, Vec<usize>)> =
            Vec::with_capacity(questions.len());
        let mut seqs = Vec::with_capacity(questions.len());
        for (qid, qdef) in questions {
            let q = to_internal(qdef)?;
            let opts = render_options(&q);
            let (ids, markers) = build_sequence(
                &self.tok,
                state,
                &q,
                self.cfg.max_len,
                self.cfg.head_max_len,
            )?;
            if opts.is_empty() || markers.len() != opts.len() {
                return Err(LayaError::Question(format!(
                    "options exceed the head budget: {} rendered, {} markers survived",
                    opts.len(),
                    markers.len()
                )));
            }
            seqs.push(ids.len());
            items.push((qid.clone(), q, ids, markers));
        }
        let total_ids: Vec<u32> = items
            .iter()
            .flat_map(|(_, _, ids, _)| ids.iter().copied())
            .collect();
        pass_pool(|| -> Result<Vec<Answer>> {
            self.backend.begin_pass();
            let hidden = {
                #[cfg(all(target_os = "macos", feature = "laya-riir-ane"))]
                match &self.enc {
                    EncoderStack::Local(e) => {
                        e.forward_packed(self.backend.as_ref(), &total_ids, &seqs)?
                    }
                    EncoderStack::Ane(_) => {
                        unreachable!("packed_eligible excludes the ANE lane")
                    }
                }
                #[cfg(not(all(target_os = "macos", feature = "laya-riir-ane")))]
                {
                    let EncoderStack::Local(e) = &self.enc;
                    e.forward_packed(self.backend.as_ref(), &total_ids, &seqs)?
                }
            };
            let d = hidden.len() / total_ids.len();
            let mut out = Vec::with_capacity(items.len());
            let mut off_rows = 0usize;
            // One slab + one scratch PER QUESTION, all allocated UP FRONT
            // and kept alive for the whole case: within the case's single
            // chain epoch, every logical buffer must own its (host ptr, len)
            // key — per-question alloc/free would malloc-reuse addresses and
            // alias the previous question's device buffers (the HeadScratch
            // doc carries the measured failure).
            let mut slabs: Vec<Vec<f32>> = items
                .iter()
                .map(|(_, _, ids, _)| vec![0f32; ids.len() * d])
                .collect();
            let mut scratches: Vec<HeadScratch> =
                (0..items.len()).map(|_| HeadScratch::new()).collect();
            let head_result = if self.head_defer_override.unwrap_or_else(head_defer) {
                self.packed_head_deferred(
                    &hidden,
                    &mut out,
                    &mut off_rows,
                    &items,
                    &mut slabs,
                    &mut scratches,
                )
            } else {
                self.packed_head_composed(
                    &hidden,
                    &mut out,
                    &mut off_rows,
                    &items,
                    &mut slabs,
                    &mut scratches,
                )
            };
            drop(slabs);
            drop(scratches);
            head_result?;
            Ok(out)
        })
    }

    /// The packed case's answer + capture tail, shared by both head drivers:
    /// the raw-bits parity seam, temperatures and the answer envelope.
    fn packed_capture_answer(
        &self,
        out: &mut Vec<Answer>,
        qid: &str,
        q: &InternalQuestion,
        rows: usize,
        markers: &[usize],
        head_out: HeadOutput,
    ) -> Result<()> {
        // The packed-path raw-bits parity seam (ungated — the gate reads
        // it; a few bytes per question is noise beside a forward).
        let bits: [u32; 2] = [
            head_out.act_probabilities[0].to_bits(),
            head_out.act_probabilities[1].to_bits(),
        ];
        let logit_bits: Vec<u32> = head_out.logits.iter().map(|l| l.to_bits()).collect();
        let mut cap = PACKED_ACT_BITS.lock().unwrap();
        if cap.len() < PACKED_ACT_BITS_CAP {
            cap.push((bits, logit_bits));
        }
        let t = self.temps.for_question(q.qtype, markers.len());
        let f = Self::make_forward(q, head_out, rows, markers.to_vec(), t);
        out.push(Self::answer_of(qid, q, f)?);
        Ok(())
    }

    /// The composed posture (`LAYA_HEAD_DEFER=0`, the pre-T12 default):
    /// each question's composed forward (its own three reads) right after
    /// its slab copy — the op stream byte-identical to the pre-T12 packed
    /// path.
    fn packed_head_composed(
        &self,
        hidden: &[f32],
        out: &mut Vec<Answer>,
        off_rows: &mut usize,
        items: &[(String, InternalQuestion, Vec<u32>, Vec<usize>)],
        slabs: &mut [Vec<f32>],
        scratches: &mut [HeadScratch],
    ) -> Result<()> {
        let d = hidden.len() / items.iter().map(|(_, _, ids, _)| ids.len()).sum::<usize>();
        for ((qid, q, ids, markers), (hq, sc)) in
            items.iter().zip(slabs.iter_mut().zip(scratches.iter_mut()))
        {
            let rows = ids.len();
            self.backend.copy_at(hidden, *off_rows * d, hq, 0, rows * d);
            let head_out = self
                .head
                .forward(self.backend.as_ref(), hq, q.qtype, markers, sc)?;
            *off_rows += rows;
            self.packed_capture_answer(out, qid, q, rows, markers, head_out)?;
        }
        Ok(())
    }

    /// The T12 posture: all streams enqueue first, then one drain class
    /// serves every scorer read, then the act heads. DEFAULT ON since the
    /// quiet-box paired A/B (typed 5-q 0.984, 24/24); `LAYA_HEAD_DEFER=0`
    /// restores the composed form.
    fn packed_head_deferred(
        &self,
        hidden: &[f32],
        out: &mut Vec<Answer>,
        off_rows: &mut usize,
        items: &[(String, InternalQuestion, Vec<u32>, Vec<usize>)],
        slabs: &mut [Vec<f32>],
        scratches: &mut [HeadScratch],
    ) -> Result<()> {
        let d = hidden.len() / items.iter().map(|(_, _, ids, _)| ids.len()).sum::<usize>();
        // Phase 1 — every question's stream enqueues: the slab copy out of
        // the packed residual, then layers + scorer. No read in this loop.
        for ((_, q, ids, markers), (hq, sc)) in
            items.iter().zip(slabs.iter_mut().zip(scratches.iter_mut()))
        {
            let rows = ids.len();
            self.backend.copy_at(hidden, *off_rows * d, hq, 0, rows * d);
            self.head
                .forward_enqueue(self.backend.as_ref(), hq, q.qtype, markers, sc)?;
            *off_rows += rows;
        }
        // Phase 2 — the scorer logits + CLS reads. ONE drain here serves
        // the whole case's stage-1 stream; the remaining questions' reads
        // are copies off an empty pipeline.
        let logit_sets: Vec<Vec<f32>> = items
            .iter()
            .zip(slabs.iter_mut().zip(scratches.iter_mut()))
            .map(|(_, (hq, sc))| self.head.reads_of(self.backend.as_ref(), hq, sc))
            .collect::<Result<Vec<Vec<f32>>>>()?;
        // Phase 3 — the act heads (each enqueues off the same epoch; the
        // first act read drains them all) + answers.
        for (((qid, q, ids, markers), (_hq, sc)), logits) in items
            .iter()
            .zip(slabs.iter_mut().zip(scratches.iter_mut()))
            .zip(logit_sets)
        {
            let rows = ids.len();
            let head_out = self.head.act_of(self.backend.as_ref(), sc, logits)?;
            self.packed_capture_answer(out, qid, q, rows, markers, head_out)?;
        }
        Ok(())
    }

    /// The shared answer envelope: a [`Forward`] → the wire [`Answer`].
    /// ONE copy for the loop and the packed path — the original
    /// `system_one` body, verbatim (`option_keys` errors propagate, as
    /// they always did).
    fn answer_of(qid: &str, q: &InternalQuestion, f: Forward) -> Result<Answer> {
        let keys = option_keys(q)?;
        let argmax = argmax_of(&f.probs);
        let act_probability = py_round4(f.act_probabilities[0] as f64);
        let probabilities = keys
            .iter()
            .zip(f.probs.iter())
            .map(|(k, p)| (k.clone(), py_round4(*p as f64)))
            .collect();
        let (choice, score, noul) = match q.t {
            "choice" => (Some(keys[argmax].clone()), None, None),
            "score" => {
                let exp: f64 = f
                    .probs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * *p as f64)
                    .sum();
                (None, Some(py_round4(exp)), None)
            }
            _ => (None, None, Some(py_round4(f.probs[1] as f64))),
        };
        let confidence = if q.t == "noul" {
            let p1 = f.probs[1] as f64;
            py_round4(p1.max(1.0 - p1))
        } else {
            py_round4(f.confidence)
        };
        Ok(Answer {
            qid: qid.to_string(),
            t: q.t,
            choice,
            score,
            noul,
            probabilities,
            confidence,
            act_probability,
            temperature_used: f.temperature_used,
        })
    }
}

/// The packed-path raw act_probabilities + scorer-logits bits, in answer
/// order — the PARITY SEAM the rounded [`Answer`] envelope cannot carry (a
/// stale `act_in` or an aliased slab can hide inside 4-decimal rounding;
/// neither can hide in f32 bits). Always recorded, capped at 4096 entries
/// (the last survive); [`tests/packed_same_shape_gate.rs`] clears it
/// before its packed run and reads it after. See [`HeadScratch`] for the
/// aliasing hazard this seam caught.
pub static PACKED_ACT_BITS: std::sync::Mutex<Vec<([u32; 2], Vec<u32>)>> =
    std::sync::Mutex::new(Vec::new());
const PACKED_ACT_BITS_CAP: usize = 4096;
