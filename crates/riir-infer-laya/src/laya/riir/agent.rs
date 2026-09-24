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
//! Plan 001 T4 watchability default; CPU elsewhere), explicit `cpu`/`metal`
//! is honored verbatim, `ane` selects the whole-graph Apple Neural Engine
//! lane (feature `laya-riir-ane`, macOS — Plan 002 P1) and — like `metal`
//! — fails loud when the feature or platform is absent, never a silent
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
use super::head::{Head, HeadOutput};

/// The device the riir forward runs on, chosen at load from `LAYA_DEVICE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// The flat-`Vec` CPU backend (the lane's original posture).
    Cpu,
    /// The MSL backend (`laya-riir-metal`, macOS).
    Metal,
    /// The Apple Neural Engine whole-graph lane (`laya-riir-ane`, macOS —
    /// reflex Plan 002 P1). The encoder runs a Core ML artifact; the head
    /// stays on the CPU backend.
    Ane,
}

impl DeviceKind {
    /// Resolve `LAYA_DEVICE` — unset/empty → [`Self::default_device`]
    /// (Metal when this build SHIPS the Metal backend on macOS, CPU
    /// everywhere else — the measured ~2× forward is the arena
    /// watchability default, Plan 001 T4; ANE is NEVER a default — the
    /// artifact tree is local-only and the lane is opt-in), `cpu` →
    /// [`DeviceKind::Cpu`] (the explicit opt-out), `metal` →
    /// [`DeviceKind::Metal`], `ane` → [`DeviceKind::Ane`], anything else
    /// is an error (an env typo must fail loud, never fall back).
    pub fn from_env() -> Result<Self> {
        match std::env::var("LAYA_DEVICE").as_deref() {
            Ok("") | Err(_) => Ok(Self::default_device()),
            Ok("cpu") => Ok(Self::Cpu),
            Ok("metal") => Ok(Self::Metal),
            Ok("ane") => Ok(Self::Ane),
            Ok(other) => Err(LayaError::Config {
                checkpoint: "riir",
                detail: format!(
                    "unknown LAYA_DEVICE {other:?} — expected unset, \"cpu\", \"metal\" or \"ane\""
                ),
            }),
        }
    }

    /// The no-env posture: Metal where the backend is compiled and exists
    /// (macOS + `laya-riir-metal`), CPU everywhere else. An explicit env
    /// value is always honored verbatim — only the ABSENT choice defaults.
    pub fn default_device() -> Self {
        #[cfg(all(target_os = "macos", feature = "laya-riir-metal"))]
        {
            Self::Metal
        }
        #[cfg(not(all(target_os = "macos", feature = "laya-riir-metal")))]
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
}

impl RiirAgent {
    /// Load one checkpoint from the weights root (downloading + verifying
    /// against the pins first — never bundled). The safetensors file is
    /// parsed ONCE and split between encoder and head (weights are removed
    /// from the map, no second copy).
    pub fn load(root: &std::path::Path, ckpt: Checkpoint) -> Result<Self> {
        Self::load_inner(root, ckpt, None)
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
        Self::load_inner(root, ckpt, Some((ane_root, manifest_path)))
    }

    fn load_inner(
        root: &std::path::Path,
        ckpt: Checkpoint,
        ane: Option<(&std::path::Path, &std::path::Path)>,
    ) -> Result<Self> {
        // The explicit ANE constructor owns its posture (no env round-trip:
        // `load_ane` is the ANE lane, whatever `LAYA_DEVICE` says — an env
        // value can never silently demote an explicitly requested lane);
        // the plain constructor resolves the env as before.
        let device = match ane {
            Some(_) => DeviceKind::Ane,
            None => DeviceKind::from_env()?,
        };
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
        let dir = ensure_checkpoint(root, ckpt)?;
        let name = ckpt.subfolder();
        let (agent_cfg, enc_cfg) = load_checkpoint_configs(&dir, name)?;

        let tok = Tok::from_dir(&dir, name)?;
        let mut raw = super::weights::load(&dir.join("model.safetensors"), name)?;

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
        Ok(Self {
            tok,
            enc,
            head,
            backend,
            device_label,
            temps,
            cfg: agent_cfg,
            ckpt: name,
        })
    }

    /// The checkpoint this agent serves.
    pub fn checkpoint(&self) -> &'static str {
        self.ckpt
    }

    /// The backend posture this agent runs (`"cpu"` / `"metal"` /
    /// `"ane"`) — gate lines and timing labels print it so a reading can
    /// never be mistaken for the other posture.
    pub fn device(&self) -> &'static str {
        self.device_label
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
            self.head
                .forward(self.backend.as_ref(), &mut hidden, q.qtype, &markers)
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
        if self.packed_eligible() {
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
    fn packed_eligible(&self) -> bool {
        if batch_disabled() {
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
    /// rows, no attention-mask tensor — so the per-question answers are
    /// bit-identical to the loop, not "modulo padding".
    ///
    /// The head runs PER QUESTION on its own exact-size slab, copied
    /// device-side out of the packed residual ([`Backend::copy_at`]) —
    /// the head's op stream is untouched. Every question's work is
    /// enqueued inside one pass (one `begin_pass`, one chain epoch), so
    /// the first head download drains the whole case and the rest find an
    /// empty pipeline.
    fn system_one_packed(&self, state: &Value, questions: &[(String, Value)]) -> Result<Vec<Answer>> {
        // Collate first — the reference's `items` shape: ids + markers per
        // question, checked before any GPU work.
        let mut items: Vec<(String, InternalQuestion, Vec<u32>, Vec<usize>)> =
            Vec::with_capacity(questions.len());
        let mut seqs = Vec::with_capacity(questions.len());
        for (qid, qdef) in questions {
            let q = to_internal(qdef)?;
            let opts = render_options(&q);
            let (ids, markers) =
                build_sequence(&self.tok, state, &q, self.cfg.max_len, self.cfg.head_max_len)?;
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
            for (qid, q, ids, markers) in &items {
                let rows = ids.len();
                let mut hq = vec![0f32; rows * d];
                self.backend
                    .copy_at(&hidden, off_rows * d, &mut hq, 0, rows * d);
                let head_out = self
                    .head
                    .forward(self.backend.as_ref(), &mut hq, q.qtype, markers)?;
                let t = self.temps.for_question(q.qtype, markers.len());
                let f = Self::make_forward(q, head_out, rows, markers.clone(), t);
                out.push(Self::answer_of(qid, q, f)?);
                off_rows += rows;
            }
            Ok(out)
        })
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
