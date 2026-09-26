# Research 001: FluidInference — On-Device Audio Stack (FluidAudio / mobius / text-processing-rs)

> **Source:** GitHub org [FluidInference](https://github.com/FluidInference) + HF org
> [huggingface.co/FluidInference](https://huggingface.co/FluidInference) (59 models).
> Clones pinned at: FluidAudio `@ 762baf6733ca0f0dbeb1ce335363fc75066bd2c9`
> (2026-09-25, Apache-2.0) · mobius `@ 5beb34007656d16349fec757c379a9beae45c4d5`
> (2026-09-25, Apache-2.0) · text-processing-rs `@ 46ebddcea5e1ff673c2cb5256a1b6b0e6c4ff527`
> (2026-09-21, Apache-2.0). Full-tree grep + quotes re-verified at those shas; clones
> removed after commit (§0.5 hygiene).
> **Date:** 2026-09-26
> **Status:** RECORD — external-org distill, Gain verdict, issues filed
> (this repo `.issues/015`, riir-reflex `.issues/035`, plus one private-sibling
> issue not numbered here)
> **Related Research:** katgpt-rs 147 (Parakeet Context-Trie phrase boosting — the
> decoder-side cousin, already shipped via Plan 446 + GOAT bench_164); riir-reflex 002
> (AgentJev/System-One typed-decision landscape — the CUA-S1 lineage cousin)
> **Classification:** Public (this repo is public; private-sibling specifics are kept
> at the level their own public surfaces document)

---

## TL;DR

FluidInference is an applied-AI lab shipping **fully-local on-device audio AI on Apple
silicon** — ASR, TTS, VAD, speaker diarization, speaker embeddings, and speech
enhancement — with all inference offloaded to the **Apple Neural Engine (ANE) via
CoreML**, deliberately avoiding GPU/MPS for ambient/always-on workloads. Their moat is
not new models (everything is converted open-source: Parakeet, Silero, Kokoro,
pyannote/WeSpeaker, LS-EEND, Sortformer) but the **conversion toolchain (mobius), the
per-stage ANE placement discipline, and the streaming-stateful serving shape**. For us
this is a **new-modality distill**: the workspace ships zero audio substrate today
(census below), while our ANE cost models (`ane_roofline`/`ane_fused_chain`, both
default-on) and CoreML backend (`katgpt-backend`) are exactly the substrate their
converted models would land on.

**Distilled for riir-infer (inference substrate):** an audio lane = *loaders + encoders
+ streaming serving state for public CoreML/ONNX audio models* — same slot quant/
kernels/loaders already occupy, one modality over. Integration verdict (**Gain**), not
novelty: voice-in-games and voice biometrics are both mature fields (§4 below).

---

## 1. What they ship (inventory, at pinned shas)

| Repo | Lang / license | What it is |
|---|---|---|
| **FluidAudio** | Swift / Apache-2.0 (1.85k★) | The runtime SDK: ASR, TTS, VAD, diarization, speaker embeddings, AEC/NS. macOS 14+ / iOS 17+, Apple Silicon, ANE-first |
| **mobius** | Python / Apache-2.0 | Conversion toolkit: `models/{class}/{name}/{destination}` (vad, stt, tts, llm, s2s, emb, computer-use, enhancement, translate, segment-text, speaker-diarization) + `coreml-cli` per-op compute-placement profiler |
| **text-processing-rs** | Rust / Apache-2.0 | Rust port of NVIDIA NeMo text processing: ITN + TN, 7 languages (EN/DE/ES/FR/HI/JA/ZH), **100% NeMo test compatibility (3,011 tests)**, Swift wrapper + wasm-tests dir |
| **HF org** | 59 models | CoreML + OpenVINO-NPU collections: parakeet-tdt-0.6b-v3-coreml (320k downloads), silero-vad-coreml, speaker-diarization-coreml, kokoro/inflect/pocket-tts-coreml, canary-speech-translation-coreml, cua-s1-forms-coreml |

**Model roster + their published numbers** (FluidAudio README/API.md at pinned sha):

- **ASR batch:** Parakeet TDT (v2 en / v3 25-eu-lang / Ultra / Redux ~220MB), ~120–190×
  RTFx on M4 Pro; SenseVoice/Paraformer (zh).
- **ASR streaming:** Parakeet EOU 120m with **end-of-utterance detection**; chunk
  tradeoff table `.ms160`→~8% WER / `.ms320`→~5% WER / `.ms1600` throughput; Nemotron
  streaming with encoder cache; Qwen3-ASR (30+ languages, macOS 15+ **CoreML stateful
  models**).
- **TTS:** Kokoro-82M **ANE-resident in 7 CoreML stages**, 3–11× RTFx, per-stage
  compute assignment; PocketTTS streaming (80 ms frames) **with voice cloning** from a
  short sample; Chatterbox (beta).
- **Diarization — three pipelines, an explicit routing table:** offline pyannote
  Community-1 (powerset segmentation + WeSpeaker + **VBx/PLDA clustering**, 17.7% DER
  AMI @ threshold 0.7); **LS-EEND** streaming end-to-end (8 kHz, 100 ms frames, 20.7%
  DER, up to 10 speakers, step-size knob `.step100ms`–`.step500ms`, 900 ms tentative
  preview); **Sortformer** (NVIDIA, 4 fixed speaker slots, identity-stability-first,
  1.04 s latency preset).
- **Speaker embeddings:** 256-d, L2-normalized (`embedding256`), enrollment API
  (`enrollSpeaker(withAudio:named:)`), slot upsert/remove.
- **VAD:** Silero, 256 ms chunks (4096 samples @ 16 kHz), **recurrent state reuse
  (hidden/cell/context)**, threshold guidance 0.7–0.9 clean / 0.3–0.6 noisy, default
  compute `.cpuAndNeuralEngine`.
- **Enhancement:** LocalVQE 4.8M (AEC + NS + dereverb), 16 ms streaming latency, lifts
  ASR near-end word recall 44.1% → 77.6% on their AEC-Challenge subset (beta, partially
  reproduced table — their own caveat).
- **Decision scoring:** `CuaS1FormsManager` — a CoreML classifier selecting one of
  2–32 supplied actions for a form element; 224-byte context truncation; stable
  softmax + raw-probability retention; serialized actor; "scores do not authorize an
  action" posture. Presumptively the same CUA-S1 lineage as the AgentJev/Jev
  System-One family (riir-reflex Research 002) — lineage UNVERIFIED, F4's T1 is
  the check, with a close-as-footnote exit.

## 2. The five techniques worth keeping

1. **ANE-first small-model serving.** Ambient/always-on workloads target the ANE and
   avoid GPU/MPS entirely. The load-bearing detail is **heterogeneous per-stage
   placement**: Kokoro keeps Albert/PostAlbert/Alignment/Vocoder on
   `cpuAndNeuralEngine` but Prosody/Noise/Tail on `.all` — whole-graph ANE residency
   fails for larger graphs, so the graph is split until each stage fits. This is the
   same shape our `ane_roofline` family-floor capability gate models (2 MB working-set
   cliff, 0.23 ms dispatch floor, per-chip M1–M5 peaks — Plan 379, default-on) — see F5.
2. **Conversion as a repeatable pipeline, not a one-off.** mobius: per-model pyproject
   under `uv`; trace with `.cpuOnly`; target iOS17+; compile once; **profile per-op
   compute placement with `coreml-cli`** (CPU/GPU/ANE assignment, compile time,
   prediction latency) before publishing. The profiler is the missing piece in most
   "I converted it and it got slower" stories.
3. **Streaming-stateful serving state.** Three recurring shapes: (a) CoreML **stateful
   models** (explicit state tensors passed between calls — Qwen3-ASR requires macOS 15+
   for this), (b) recurrent-state reuse (VAD hidden/cell/context), (c) encoder caches
   across chunks (Nemotron). Plus the serving hygiene: EOU debounce (1280 ms),
   sliding-window overlap (14.96 s window / 2 s overlap), `finalizeSession()` flush,
   tentative-vs-finalized segments, and latency knobs (chunk size, LS-EEND step size)
   that trade WER/throughput/latency along one axis.
4. **The diarization decision matrix.** Offline-VBx (best DER, modular, enrollment
   friendly) vs LS-EEND (best streaming latency+capacity) vs Sortformer (identity
   stability, 4-speaker cap) — three pipelines with honestly stated tradeoffs instead
   of one model with flags. Worth stealing verbatim as the "which model serves this
   workload" pattern for any multi-model lane we ship.
5. **Voice identity as a first-class primitive.** `embedding256` + enrollment + slot
   management makes "who is speaking" a servable API rather than a research pipeline —
   the exact shape needed to treat a voiceprint as latent identity (F3).

## 3. Internal census (what we already have — public surfaces only)

- **ANE substrate ships today (public, katgpt-rs):** `katgpt-backend` `InferenceBackend`
  trait with `CpuBackend` / `AneBackend` (CoreML via `coreml-native` + `coreml-proto`
  spec builder + `prost`) / `GpuBackend` (Metal), features `ane`, `gpu_inference`,
  `inference_router` (TriggerGate + router, Plan 176).
- **ANE cost models (public, default-on):** `ane_roofline` (Plan 379, arXiv:2606.22283)
  and `ane_fused_chain` (Plan 439, arXiv:2607.11262). These PREDICT ANE fit — directly
  applicable to their converted models before we convert anything (F5).
- **ANE probe precedent:** `moka_ane` residency probe (katgpt-pruners, Issue 564) —
  "can this topology run on ANE at all" as a cheap gated probe. The audio-lane PoC
  should copy this shape.
- **ONNX-native-model precedent:** `fastembed`/`ort` consumption in `riir-games`
  (quest-trigger embeddings) — the cross-platform fallback path for Linux/wasm voice.
- **Parakeet already a workspace surface (decoder side):** katgpt-rs Research 147 →
  Plan 446 → shipped `phrase_boost.rs` + GOAT bench_164 (Context-Trie phrase boosting
  from parakeet.cpp). FluidInference ships the same model family from the encoder/
  serving side — complementary, not duplicate.
- **The typed-decision cousin:** riir-reflex Research 002 maps the Jev/System-One
  landscape (AgentJev 79.25% / laya-typed 74.45% ours / modelless 31.9%) and already
  flags "CoreML/ONNX ports" as ecosystem arms — `cua-s1-forms-coreml` is one, converted.
- **Audio output exists (editor only):** seal-game-editor consumes `rodio`/`cpal` for
  sfx/bgm playback. Audio INPUT (capture) and inference: **zero hits** — census grep
  `parakeet|diariz|silero|fluidaudio|whisper|speech` over `*.rs` across the workspace
  returned only Research-147/Plan-446/bench text and unrelated prose. **Audio is a new
  modality for the stack.**

## 4. Novelty gate — scored honestly

- **Q1 no prior art? NO.** On-device voice control in games is a mature field
  (KeenASR SDK with Unity plugin; Picovoice voice commands; the Tom Clancy's EndWar
  lineage write-up on gamedeveloper.com; US10926173B2 custom voice utterances for
  in-game character control). Voiceprint authentication is a mature biometric with a
  documented threat landscape (speaker-embedding VAS spoofing surveys; deepfake bypass
  of voice auth). No novelty is claimed anywhere in this note.
- **Q2 new behavior class? Not industry-wide.** New *for our stack* (first audio
  modality), which is adoption, not a new class.
- **Q3 product selling point? Real but derivative:** "fully-local voice chat +
  per-NPC cloned voices + voiceprint soft-trust, no cloud" is a coherent product
  sentence built from their parts.
- **Q4 force multiplier? Yes** — game runtime (voice), riir-auth (voiceprint), reflex
  (arena arm), this repo (substrate lane) — ≥2 pillars.
- **Score: 1 of 4 → Gain.** No Super-GOAT framing is attempted, no "candidate" hedge.

**Fusion-priority ladder check (the healer question):** does this do anything for
riir-clippy? No — the healer's surfaces (corpus, trajectories, selection, benches) have
no audio consumer; voice dictation of lint requests is a stretch, not a surface. The
ladder's #1 (game runtime) is served by F1/F2 below.

## 5. Distillation — fusion candidates

**F1 — riir-infer audio lane (this repo; `.issues/015`).** Load FluidInference's
published CoreML bundles from Rust. Three consumption paths, in evaluation order:
1. **The repo's existing `objc2-core-ml` ANE path** (riir-infer-laya's allowlisted
   `laya-riir-ane` feature, macOS target-scoped) pointed at an external `.mlmodelc`
   (silero-vad-coreml first — smallest, 31k+ downloads). UNVERIFIED whether that
   binding's MLModel compile/load surface accepts external prebuilt bundles — that
   is PoC gate 1.
2. **fluidaudio-rs FFI** (MIT, crates.io) — full pipeline via their Swift bridge;
   costs the Swift toolchain in `build.rs`, macOS 14+ only. Expected to lose to
   path 1 on build hygiene for a public substrate repo.
3. **mobius-convert to ONNX → `ort`** — the cross-platform path (Linux game servers,
   wasm); precedent exists (fastembed).
Every path adds deps that need BOUNDARY.md allowlist rows first — `.issues/015` T0
owns that widening before any code. Feature-gated, default-off, GOAT before any
promotion.

**F2 — game voice (fusion priority #1, downstream of F1).** VAD gate → LocalVQE AEC
(far-end reference = the game's own audio — the echo problem is *structurally present*
in voice chat) → Parakeet ASR → text-processing-rs ITN → text into the existing
chat/decision paths; Kokoro/PocketTTS for NPC voice, voice cloning = per-NPC voice
identity from a short sample; EOU debounce as voice turn-taking. Files as an
issue only after F1 proves the lane.

**F3 — voice identity (routed to a private sibling).** `embedding256` + enrollment
is the audio-domain analog of a behavioral identity signal. The consumer design
(that signal is a SOFT trust input, never a hard authentication factor, because
voice biometrics are spoofable; embeddings stay local with only a bounded confidence
scalar crossing any sync boundary) is scoped in a private-sibling issue — the
public surface of this note deliberately carries no further detail.

**F4 — cua-s1-forms → riir-reflex arena arm (public, `.issues/035`).** Research 002
already maps the System-One landscape; this adds the **CoreML/ANE serving arm** of the
same family ("their stack serves, our Rust measures" — the AgentJev lane pattern,
macOS posture like the metal lane). Lineage (cua-s1-forms = TypeSafe Jev's CUA model
converted by FluidInference) is a hypothesis to verify from the HF card at PoC time.
Their scorer is softmax-native — external lanes are measured as-is; the sigmoid
mandate binds our primitives, not their model.

**F5 — `ane_roofline` × their published benchmarks (cheap, high-value).** Their
Benchmarks.md + `coreml-cli` outputs publish per-model ANE placements and latencies
for conv/RNN/transducer op-mixes. Zero new code: read their tables, check our cost
model's 2 MB cliff + family-floor gate + 0.23 ms dispatch floor against real audio
graphs. Consult BEFORE any conversion effort. Their 7-stage Kokoro split (82M)
shows the per-stage placement discipline; whether whole-graph ANE residency survives
0.6B scale is UNKNOWN — which is exactly why the cost model runs first. The lane's
first target is small models (silero-vad, LocalVQE, EOU-120m), not parakeet-0.6b.

**F6 — text-processing-rs as a direct dependency candidate.** Pure Rust, Apache-2.0,
3011-test NeMo parity, wasm-tests present. Any F2 voice→text→game-event path needs
ITN ("two hundred" → 200); consume, don't port. BOUNDARY/allowlist check at adoption.

## 6. Honest caveats

1. **The repo's existing `objc2-core-ml` external-bundle load is unverified** — the
   PoC's first gate (does the allowlisted `laya-riir-ane` path accept a prebuilt
   `.mlmodelc` it did not export itself?), and the reason F1 path 3 (ONNX) exists.
2. **Their runtime is Apple-only.** fluidaudio-rs drags a Swift toolchain into
   `build.rs` (macOS 14+/iOS 17+); Linux/wasm voice needs the ONNX route; their own
   Windows alternative (fluid-server) is WIP.
3. **Licensing is per-model, not blanket.** FluidAudio/mobius/text-processing-rs are
   Apache-2.0, but **Sortformer is NVIDIA Open Model License** (their README flags
   it); pyannote/WeSpeaker/Silero/Kokoro are permissive. Check NOTICE files
   (text-processing-rs carries NeMo-derived NOTICE + THIRD-PARTY-LICENSES) before any
   shipped surface.
4. **Voiceprint spoofability** bounds F3 to a soft signal by design, not by caution.
5. **ANE fit for transformer ASR is not free** — their own architecture (7-stage
   Kokoro, per-stage placement, EOU-120m at 120M params rather than parakeet-0.6b as
   the streaming model) is consistent with whole-graph ANE residency failing at
   scale. F5's cost-model cross-check exists to avoid repeating that discovery the
   expensive way.

## 7. Verdict

**Gain — integration distill.** The org converts and serves known open models with a
disciplined conversion toolchain and a clean streaming-serving shape; the value here is
a new modality landing on substrate we already ship, not a new primitive. No
Super-GOAT, no novelty claim, no plan yet — three scoped issues gate the plan decision:

| File | Repo | Asks |
|---|---|---|
| `.issues/015_fluidaudio_coreml_audio_lane_poc.md` | riir-infer | the audio lane PoC (F1), opening with a BOUNDARY-widening task |
| `.issues/035_cua_s1_forms_coreml_lane.md` | riir-reflex | CoreML arena arm (F4) |
| *(private sibling)* | riir-auth | voice identity soft-trust PoC (F3) |

**MOAT gate (this repo's row):** "inference-substrate novelty (quant formats, kernels,
**encoders, loaders**) shipped public + upstream-clean" — an audio lane is
encoders+loaders+streaming-serving for public models: in scope, and it widens the
public substrate's modality surface. It stays zero-`riir-*`-dep by construction, but
the EXTERNAL deps each path would add (objc2-core-ml row exists; fluidaudio-rs / ort
rows do not) need BOUNDARY.md allowlist rows first — `.issues/015` T0 owns that.

## 8. Provenance

- Clones: `riir-infer/.raw/{FluidAudio,mobius,text-processing-rs}` at the shas in the
  header; shallow, read-only; **removed after commit** (a finished research task with a
  live `.raw/` entry is an unfinished task). Pre-existing `.raw/` entries from other
  sessions (`exllamav3`, `packs/`) were left untouched.
- Web prior-art sweep: KeenASR (Unity on-device ASR), Picovoice, EndWar voice-control
  retrospective (gamedeveloper.com 2020), US10926173B2 (voice-controlled game
  character), speaker-verification spoofing surveys (VAS threat landscape).
- Internal-first sweep: `.research/` + `.plans/` across the workspace grepped for
  `FluidInference|FluidAudio|parakeet|diariz|silero` — hits only at Research 147 /
  Plan 446 (parakeet.cpp decoder cousin) and reflex Research 002; `*.rs` census as in §3.
