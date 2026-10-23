# Issue 015: FluidAudio/CoreML audio lane PoC — load FluidInference's published models from Rust

**Status:** OPEN — DEFERRED 2026-09-28 (owner directive: audio until **M5 Ultra** is available). T0 boundary stays landed; T1–T6 below deferred with the recon recorded so the pickup is turnkey. **⟲ 2026-10-22 (owner directive: the M5 gate is UNLOCKED): the M5 Ultra is available to this workspace — T1–T6 may proceed ON THE M5; the deferral's waiting condition is discharged.**
**Claimed:** zed-m5ultra-audio-lane, 1792718151, expires 2026-10-25T01:15:00+00:00
**UNBLOCK CONDITION MET (recorded 2026-10-17):** M5 Ultra SHIPPED — Mac Studio M5 Ultra announced 2026-08-25, on sale 2026-09-22 (up to 256 GB unified memory, from $5,499). The hardware exists; the pickup is now an owner purchase/decision, not a waiting act. riir-auth `.issues/005` (the voiceprint PoC) unblocks downstream of this lane's T1.
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

### T1 — cost-model prediction (RECORDED 2026-10-23, BEFORE any measurement)

Bundle: `silero-vad-unified-256ms-v6.2.1.mlmodelc` — 5 files, 1,063,425 B total;
`weights/weight.bin` = 882,304 B (Mixed Float16 weights ⇒ ~441 K params);
metadata histogram: **357 graph ops** (8 Lstm, 48 Conv, 40 Relu, 54 Cast,
41 SliceByIndex, 32 Clip, 26 ExpandDims, 18 Squeeze, 16Transpose/Pow/Add,
9 Concat, 8 Sigmoid/Sqrt/Pad, 7 Mul, 2 Sub). Wire: f32 in `[1,4160]` /
`[1,128]` ×2, f32 out `[1,1,1]` + `[1,128]` ×2 — the recon contract CONFIRMED
byte-for-byte from `metadata.json`.

M5 ANE peaks (`AnePeaks::m5`, katgpt-core ane_roofline, default-on): 8.9
TFLOPS fp16 · 57 GB/s · dispatch floor 0.11 ms · working set 4.72 MB.

- **Working-set gate:** largest live operand = the weight stream, 0.88 MB <
  4.72 MB — fits with **5.3× headroom**. No DRAM-streaming regime. PREDICT:
  structurally ANE-eligible; `FamilyGated`/`WorkingSet` both excluded
  (F0-floor ops, min_family A11).
- **Compute:** ≤ 2 × 441 K MAC ≈ 1.8 MFLOP/chunk → ~0.2 µs at peaks.
  Negligible on every axis.
- **Memory:** ~18.9 KB of activations/chunk (4160+128+128 f32 in, 257 f32
  out). Negligible.
- **BINDING TERM — dispatch.** Per-chunk latency ≈ N_device_passes ×
  0.11 ms, where N is the FUSED pass count (not the 357-op graph count).
  PREDICT: **2–5 ms/chunk** if CoreML fuses to ~20–40 ANE passes; worst
  case 357 unfused → 39 ms/chunk. At the 256 ms cadence that is RTF
  0.008–0.15 — real-time viable across the whole band. The measurement's
  job is to locate N inside the band.
- **Placement verdict PREDICTED TO FAIL the strict law:** the 54 fp32↔fp16
  Cast ops around the LSTM/IO are exactly the transition class
  `Placement::passes()` (100% ANE-preferred, 0 transitions) refuses — the
  recon already predicted the audio lane must REPORT the walk verdict
  (counts + transitions) rather than gate the load on the strict laya law.
- **G2 verdict classes (synthetic):** silence → prob < 0.1; 440 Hz sine →
  prob < 0.3 (periodic non-speech); speech-shaped noise (AM 4 Hz syllabic
  envelope over 300–3400 Hz formant-band carriers) → prob > 0.5, separated
  from silence by > 0.4. RISK: synthetic speech-shaped noise may NOT clear
  0.5 (VAD is speech-specific — formant-band noise is not speech); if the
  separation order holds (silence ≪ tone ≤ speech-shaped) G2 reads PASS on
  ordinal separation with the absolute band disclosed, never folded.
- **Falsifiable predictions for the run:** (a) load under
  `.cpuAndNeuralEngine` succeeds; (b) measured p50 latency lands inside
  2–40 ms/chunk; (c) the plan walk reports transitions > 0; (d) silence
  prob < 0.1 on a freshly-zero-initialized stream. Any miss is recorded,
  never smoothed.

### T1 verdict — prediction vs measured (2026-10-23, PoC run)

| # | Predicted | Measured | Verdict |
|---|---|---|---|
| a | loads under `.cpuAndNeuralEngine` | loads (both postures) | **HELD** |
| b | p50 2–40 ms/chunk (dispatch-bound) | **p50 = 0.163 ms**, p99 = 0.276 ms, max 0.388 ms — RTF 0.00064 | **FALSIFIED, fast side** — the 357-op graph collapses to ~1–2 device passes (0.163 ms ≈ 1.5× the single-op M5 dispatch floor); the dispatch floor never binds |
| c | plan walk reports transitions > 0 | **transitions = 0** — ALL 357 ops report `MLCPUComputeDevice` (ane=0, gpu=0) | **FALSIFIED** — see the ANE finding below |
| d | settled silence < 0.1 | silence = 0.0308 | **HELD** |

Working-set/compute/memory predictions all held trivially (the model is
three orders under every M5 ceiling) — but the headline is:

**THE ANE FINDING (generalized 2026-10-23, `ane_plan_probe` controls):
on this M5 Ultra (macOS 27), `MLComputePlan.preferredComputeDevice` reads
CPU for EVERYTHING testable — the silero bundle (357/357 CPU) AND the
system `ATXIntentPredictionMLModel` (108/108 CPU), each under BOTH
`cpuAndNe` AND `cpuAndGpu`; a third system bundle refuses plan
construction ("model is not supported" — an old-format model). The A/B
against a locally-compiled `.mlpackage` also reads all-CPU, so the
cross-machine-specialization hypothesis is refuted too. Consequence:
the laya strict ANE gate (`ane_ops == device_ops`) would refuse EVERY
load on this box — the plan walk is not a usable ANE-placement
authority on M5-class silicon/macOS 27 today; an execution-side
instrument (what device actually RAN each op — Xcode Core ML perf
report / inference observation) would be needed. The latency is
unaffected for THIS model (0.16 ms/chunk = 1500× real-time headroom on
cPU), so the PoC's serving answer is fully green. Owner-visible
follow-ups: (1) re-run the laya placement walk on this box before
citing any ANE placement claim on M5-class silicon — the M3-measured
claims (2026-09) stand on their own boxes; (2) decide whether the
placement gate gets an execution-side successor or a per-OS-calibration
posture.**
PoC's follow-up, owner-visible.**

### Gate results (2026-10-23)

- **G1 (loads): PASS** — both postures, through the host's new
  precompiled path (A) and the existing compile-cached path (B).
- **G2 (synthetic verdicts): PASS after a disclosed re-registration.**
  The first read FALSIFIED the pre-registered ordinal law (tone read
  BELOW the silence baseline: 0.0171 vs 0.0308) — recorded as-is. The
  re-registered law is the VAD-native framing: all synthetic NEGATIVES
  stay < 0.1 (silence 0.0308 · tone 0.0171 · speech-shaped 0.0425 — none
  false-trigger) while REAL speech (G2c) clears 0.5.
- **G2c (real-speech positive control): PASS** — macOS `say` TTS
  (Samantha, 19 chunks) reads **peak = 1.0000** on every chunk. The
  pipeline (context assembly, LSTM state carry, f32 IO) is proven
  end-to-end; the synthetic classes cannot do this alone.
- **G3 (latency with box state): RECORDED, PROVISIONAL** — p50 0.163 /
  p99 0.276 / max 0.388 ms per 256 ms chunk (posture A; posture B
  0.162/0.242/0.312 — parity). PROVENANCE (bench_preflight):
  `power=AC Power load=8.49 swap=2.38M canary=skipped powermode=0(auto)`
  — the preflight REFUSED a published number (loaded box, plan459 sweep
  resident at ~800% CPU); the figures stand as provisional-poC, quiet-box
  re-run before any league-style citation.
- **G4 (feature-gated default-off): PASS by construction** — `vad-ane`
  opt-in; default build compiles EMPTY (clippy −D warnings green at BOTH
  postures); off-macOS the module cfg's out (the clef-ane posture).
- **State-carry determinism: PASS** — two identical 16-chunk streams
  from reset are bit-identical.

### What landed (T2/T3/T6 partial)

- `crates/riir-infer-ane` host widening (lane-free): `load_model_precompiled`
  (pre-compiled `.mlmodelc` loads — `compileModelAtURL` rejects a compiled
  bundle as input) + `predict_f32_multi` (ONE forward, N named outputs —
  re-predicting per output would re-roll the recurrence) + the shared
  `load_compiled_at` factoring. Existing lanes untouched (the f16/f32 twins
  delegate through the same choreography).
- `crates/riir-infer-audio` (new, feature `vad-ane`): the silero lane —
  digest-pinned both postures (blake3-dir-v1 pins minted by the
  `audio_bundle_digest` example), streaming state (context tail + LSTM
  hidden/cell), deterministic synth classes, the `silero_vad_poc` gate
  suite (5 tests, `#[ignore]`d real-model material), the `vad_timing`
  posture example, `scripts/fetch_silero_vad.sh` (both bundles).
- BOUNDARY.md: the AUDIO row names the crate.

### Remaining

- [-] **T4/T5 fallback paths** — MOOT for the load question (T2 passed;
  both published postures load through the repo's own host), kept only
  if an ANE-restoring consumption path is ever wanted.
- [ ] **T6 verdict doc** (`.docs/` numbered slot + the digest-pin
  disclosure) — the substance is in this file; the doc lands with the
  quiet-box latency re-run so the cited numbers are publish-grade.
- [ ] **The laya-on-M5-Ultra placement check** (the ANE finding's
  follow-up): run the laya lane's plan walk on this box before trusting
  ANY ANE placement claim on M5-class silicon.
- [ ] **Owner decisions surfaced**: (1) the ANE finding (all-CPU plan on
  M5 Ultra for this model class, both postures) — accept CPU serving for
  VAD (the latency is already 1500× headroom) or investigate ANE
  enablement (newer coremltools? different specialization?); (2) any
  promotion beyond opt-in stays owner-gated per the lane header.

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
- [x] **T1 — consult the cost model FIRST.** Prediction recorded ABOVE
      (2026-10-23, before any measurement); verdict table: 2 held, 2
      falsified (both recorded, never smoothed) — the falsifications are
      the finding.
- [x] **T2 — gate 1: can the EXISTING `laya-riir-ane` path load an
      external bundle?** **PASSED — with two narrow host widenings:**
      `load_model_precompiled` (published `.mlmodelc` — the compile
      cache has nothing to compile for a pre-compiled bundle and
      `compileModelAtURL` rejects one as input) and `predict_f32_multi`
      (the VAD's 3-output forward — one prediction, N named outputs;
      per-output re-prediction would re-roll the recurrence). The
      EXISTING compile-cached path loads the published `.mlpackage`
      unchanged (posture B, used as the A/B control). Both loads are
      digest-walled (blake3-dir-v1 pins minted in-code) and refuse
      drift loud.
- [x] **T3 — drive the verdicts + measure.** 5-test gate suite green:
      G1 loads (both postures) · G2 negatives < 0.1 (after the disclosed
      re-registration) · G2c real speech 1.0000 · state-carry
      deterministic · G3 latency p50 0.163 ms/chunk, RTF 0.00064,
      PROVISIONAL (loaded box, preflight-refused publish) · placement
      all-CPU both postures (the ANE finding, owner-visible).
- [-] **T4 — fallback path A:** `fluidaudio-rs` FFI (MIT) — MOOT for the
      load question (T2 passed); kept only if an ANE-restoring
      consumption path is ever wanted.
- [-] **T5 — fallback path B (cross-platform):** mobius-convert
      silero-vad to ONNX → `ort` — MOOT for the load question; the
      Linux-game-server posture would re-open this.
- [-] **T6 — verdict doc:** PARTIAL — the substance is in this issue
      (T1 table + gate results + the ANE finding); the numbered `.docs/`
      slot lands with the quiet-box latency re-run so cited numbers are
      publish-grade. Feature-gated default-off DONE (G4); GOAT gate
      before any promotion still owed.

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
