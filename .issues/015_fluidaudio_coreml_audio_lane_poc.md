# Issue 015: FluidAudio/CoreML audio lane PoC — load FluidInference's published models from Rust

**Status:** OPEN — DEFERRED 2026-09-28 (owner directive: audio until **M5 Ultra** is available). T0 boundary stays landed; T1–T6 below deferred with the recon recorded so the pickup is turnkey.
**Date:** 2026-09-26
**Filed from:** [`.research/001_FluidInference_OnDevice_Audio_Stack.md`](../.research/001_FluidInference_OnDevice_Audio_Stack.md) (F1)
**Sources:** FluidAudio `@ 762baf6733ca0f0dbeb1ce335363fc75066bd2c9` (Apache-2.0) ·
mobius `@ 5beb34007656d16349fec757c379a9beae45c4d5` (Apache-2.0) ·
HF `FluidInference/silero-vad-coreml` · text-processing-rs `@ 46ebddcea5e1ff673c2cb5256a1b6b0e6c4ff527` (Apache-2.0)

## Problem

The workspace has zero audio substrate (census in Research 001 §3) while the ANE
substrate to run converted audio models already ships here: `katgpt-backend`'s
`AneBackend` (coreml-native + coreml-proto), `ane_roofline`/`ane_fused_chain` cost
models (default-on). FluidInference publishes ANE-optimized CoreML bundles of open
models (VAD/ASR/TTS/diarization/AEC). The unblocked first step is a PoC that answers:
**can riir-infer load and serve one of their published bundles from pure Rust, and at
what cost vs the alternatives?**

## Deferred state (2026-09-28 recon — turnkey handoff)

Owner directive: defer until M5 Ultra. Recon done this session so the next
pickup starts at T2, not at zero:

- **Target bundle (exact):** `FluidInference/silero-vad-coreml` →
  `silero-vad-unified-256ms-v6.2.1.mlmodelc` — a PRE-COMPILED `.mlmodelc`
  directory (no compile step; direct `MLModel` load), named by FluidAudio
  `ModelNames.VAD.sileroVadFile` (@ `20d4f0bd46d11d7f50a6eb4f7835cfdbd2b4ba14`,
  HEAD — the pinned sha `762baf67` predates it; same wire).
- **Wire contract** (from `VadManager.processUnifiedModel`):
  inputs `audio_input` `[1, 4160]` f32 (64-sample context + 4096 new —
  256 ms @ 16 kHz), `hidden_state` `[1, 128]`, `cell_state` `[1, 128]`;
  outputs `vad_output` (prob, substring-matched — names may carry suffixes),
  `new_hidden_state`, `new_cell_state`. State flow: context = last 64
  samples of the PRIOR chunk, zero-init; LSTM state 128-d carried between
  chunks (the streaming-state shape T1 wanted to exercise).
- **Compute posture:** FluidAudio's default `.cpuAndNeuralEngine`; their
  loader pools the three MLMultiArrays per key (surface churn matters —
  3 allocs/chunk at 256 ms cadence is nothing, but noted).
- **T2 shape (already known):** the generic pieces of
  `crates/riir-infer-laya/src/laya/riir/ane.rs` (compile-cache,
  `MLModel::modelWithContentsOfURL_configuration_error` under
  `CPU_AND_NE`, `verify_compute_plan`) take any bundle URL — the
  digest/manifest coupling is laya-encoder-specific and the audio lane
  bypasses it. The audio lane's plan gate should REPORT the verdict
  (FluidAudio's published bundle is not 100%-ANE/0-transition by
  contract) rather than reuse `Placement::passes()`'s strict laya law.
- First recon commit: `e8e17b8` (landed Issue 023's missed rename in the
  same pass; unrelated to audio).
- **Candidate-model addendum (2026-09-30, Research 005):** FermionResearch
  **Phonon-2** — 164 MB English ASR, 5.21 % WER, ~2.1 bpw five-level
  encoder, a QAT derivative of the same `parakeet-tdt-0.6b-v3` this lane
  targets; beats Parakeet Redux at equal size on their harness. Not the
  first load (English-only vs EN/TH scope; **no CoreML bundle yet** —
  "coming soon"; weights CC-BY-4.0). Re-evaluate when its CoreML runtime
  lands, or measure via `fermion serve` (OpenAI-compatible HTTP) as a
  subprocess comparison lane. Full comparison table:
  [Research 005](../.research/005_Phonon2_Neutrino1_Ternary_ASR_Lane_Intelligence.md) §2.

## PoC plan

Target model: **silero-vad-coreml** (smallest, most downloads, 256 ms chunks,
recurrent state — exercises the streaming-state shape too).

- [ ] **T0 — BOUNDARY first (owner-gated, blocks everything else).** This repo's
      `BOUNDARY.md` §Owns currently reads "The LLM inference substrate" with no audio
      row, and the allowlist carries no row for any audio consumption path. Before any
      code: widen Owns to cover **audio encoders/loaders/streaming serving state**
      (the domain test — "weights, quant, architectures, kernels, loaders" — already
      fits audio loaders; the widening is a row, not a fence move), and add the
      allowlist row for whichever path T2–T5 selects: (1) reuse the EXISTING
      `objc2-core-ml` row (`laya-riir-ane`, macOS target-scoped) — see T2; (2)
      `fluidaudio-rs` + a Swift-toolchain build dep (heavy — expected to be
      discouraged for a public substrate repo); (3) `ort` (cross-platform). Per the
      boundary rule the row lands FIRST, in the same commit as the first code.
      **BOUNDARY HALF DONE 2026-09-28 (owner-gate D8): the AUDIO Owns row landed +
      the `objc2-core-ml` row's condition column names the REUSE (option 1, the
      recorded verdict). T0's remaining half is the row riding the first code
      commit — done by construction: the row is already in the contract, so the
      first code commit cites it. T1–T5 below are the PoC's own tasks.**
- [-] **T1 — consult the cost model FIRST.** Run `ane_roofline` on silero-vad's shape
      (working set vs the 2 MB cliff, dispatch floor). Record the prediction before
      any measurement — Research 001 §6 caveat 5 / F5.
- [-] **T2 — gate 1: can the EXISTING `laya-riir-ane` path load an external
      `.mlmodelc`?** riir-infer-laya already carries an allowlisted macOS ANE feature
      built on `objc2-core-ml` 0.3 (BOUNDARY.md May-depend-on, target-scoped) — the
      question is whether that binding's `MLModel` compile/load surface accepts an
      external prebuilt bundle (it was built for the laya lane's own exports).
      Time-box this; if it fails, document why and move to T4. (katgpt-rs's
      `coreml-native`/`coreml-proto` stack is the sibling spelling of the same idea —
      not a dep candidate here, it is not on this repo's allowlist.)
- [-] **T3 — if T2 passes:** drive 3 verdicts on synthetic audio (silence / tone /
      speech-shaped noise) and check VAD probabilities separate cleanly. Measure
      per-chunk latency vs their published posture (`.cpuAndNeuralEngine`).
- [-] **T4 — fallback path A:** `fluidaudio-rs` FFI (MIT) — full pipeline but drags a
      Swift toolchain into `build.rs`, macOS 14+ only. Evaluate as macOS-only posture;
      requires the T0 allowlist row and is expected to lose to T2 on build hygiene.
- [-] **T5 — fallback path B (cross-platform):** mobius-convert silero-vad to ONNX →
      `ort` (the `fastembed` precedent in riir-games shows ONNX-native consumption).
      This is the Linux-game-server / wasm path; requires the T0 allowlist row.
- [-] **T6 — verdict doc:** consumption-path decision matrix (pure-Rust CoreML vs FFI
      vs ONNX) with measured numbers; land behind a default-off feature gate; GOAT
      gate before any promotion.

## Gates

- G1: model loads (or T2 failure documented with the exact error).
- G2: VAD verdicts correct on the three synthetic classes.
- G3: latency recorded with box state (the standing latency-provenance rule).
- G4: feature-gated, default-off; zero impact on default builds.

## Numbering note (carve-era guard)

Filed on the LOCAL lane (011–014 → **015**, recorded in
`.issues/.highwater_local`). The inherited `998/1003/1004` files are MOVED
riir-ai documents, not local allocations; riir-ai's counter has since advanced
to 1008 — **do not allocate 1005–1008 in this repo** (dual with riir-ai
history; `.issues/.highwater` here still reads 1004 for that reason). See
AGENTS.md §Numbering Discipline for the two-lane rule.

## References

- Research 001 (this repo) — the full distill + fusion map F1–F6.
- katgpt-rs Plan 176 (ane backend), Plan 379 (ane_roofline), Plan 439 (ane_fused_chain).
- FluidAudio API.md (VAD section) at the pinned sha.
