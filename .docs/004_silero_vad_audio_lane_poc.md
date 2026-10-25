# 004 — the silero-vad audio lane PoC (issue 015)

**Status:** FINAL (latency re-run 2026-10-26, quiet-ish box, preflight PASSED,
replicated ×2; findings and gate verdicts FINAL as of 2026-10-23) ·
issue: closed+removed per the noise rule, record = HISTORY row 2026-10-26 +
this doc · research: `.research/001_FluidInference_OnDevice_Audio_Stack.md` (F1)

The first lane of the audio substrate: **silero-vad** (FluidInference
`silero-vad-coreml`, Apache-2.0), seated on the shared
[`riir-infer-ane`](../crates/riir-infer-ane) host behind feature `vad-ane`
(opt-in, default build compiles EMPTY, off-macOS cfg'd out — the clef-ane
posture). Crate: `crates/riir-infer-audio`.

## Wire contract (confirmed byte-for-byte from the bundle's metadata.json)

- Inputs: `audio_input` `[1, 4160]` f32 (64-sample context + 4096 new =
  256 ms @ 16 kHz), `hidden_state` + `cell_state` `[1, 128]` f32.
- Outputs: `vad_output` `[1, 1, 1]` f32, `new_hidden_state` +
  `new_cell_state` `[1, 128]` f32.
- Streaming law: the context tail is the PRIOR chunk's last 64 samples
  (zero-init); the LSTM state carries between chunks (zero-init). ONE
  multi-output forward per chunk — per-output re-prediction would re-roll
  the recurrence (the reason `predict_f32_multi` exists).

## What landed

| Piece | Where |
|---|---|
| `load_model_precompiled` (published `.mlmodelc`; `compileModelAtURL` rejects a compiled bundle as input) | `riir-infer-ane` |
| `predict_f32_multi` (one forward, N named outputs) | `riir-infer-ane` |
| `probe_compute_plan` + the `ane_plan_probe` example (the "does anything place on ANE on this box" instrument) | `riir-infer-ane` |
| `SileroVad` (load / process / reset, digest-walled both postures) | `riir-infer-audio/src/vad.rs` |
| The 5-test gate suite (`silero_vad_poc`, `#[ignore]`d real-model material) | `riir-infer-audio/tests/` |
| The posture-A/B timing example (`vad_timing`) | `riir-infer-audio/examples/` |
| The pin-minting example (`audio_bundle_digest`) | `riir-infer-audio/examples/` |
| The fetch script (both bundles) | `scripts/fetch_silero_vad.sh` |

Digest pins (blake3-dir-v1, minted 2026-10-23, never hand-computed):

- `silero-vad-unified-256ms-v6.2.1.mlmodelc` —
  `097b5886419c8e73cf86f923c2dab4bc86e294e5cb861ef240e146244f8f21e5`,
  5 files, 1,063,425 B.
- `silero-vad-unified-256ms-v6.0.0.mlpackage` (the A/B control) —
  `8588b28ca6f702087ab2520146f5d3eb42e682a8ec45236994b610fa77dc648c`,
  3 files, 1,040,846 B.

## Gate results

| Gate | Verdict | Detail |
|---|---|---|
| G1 load | **PASS** (both postures) | precompiled (A) + compile-cached (B) paths |
| G2 synthetic negatives | **PASS** (after a disclosed re-registration) | silence 0.0308 · tone 0.0171 · speech-shaped 0.0425 — all < 0.1, none false-trigger |
| G2c real-speech positive control | **PASS** | macOS `say` TTS, 19 chunks, **peak 1.0000** — the pipeline (context, state carry, f32 IO) proven end-to-end |
| G3 latency | **PASS, FINAL** | 2026-10-26 quiet-ish re-run, replicated ×2: p50 **0.166 / 0.166**, p99 0.208→0.181 / 0.200→0.175, max 0.260→0.236 / 0.259→0.239 ms per 256 ms chunk (A, run1→run2); B at parity (p50 0.161/0.166). RTF_p50 ≈ 0.00065 → **≈ 1,540× real-time headroom**. The 10-23 provisional p50s (0.163/0.162) reproduce within ~2%; the loaded-box tails (p99 0.276/0.242, max 0.388/0.312) tighten on the quiet box — the tail inflation was box noise, not model variance. Placement verdict unchanged both runs (device=357 ane=0 gpu=0 cpu=357, transitions=0). |
| G4 feature-gated default-off | **PASS by construction** | clippy −D warnings green at both postures; zero impact on default builds |
| State-carry determinism | **PASS** | two identical streams from reset are bit-identical |

The re-registration disclosure: the first read's pre-registered ordinal law
(silence < tone ≤ speech-shaped) was **FALSIFIED** (tone read below the
silence baseline). Recorded as-is in the issue; the re-registered law is the
VAD-native negatives-stay-low / real-speech-clears framing. The synthetic
classes alone cannot separate pipeline-correct from pipeline-broken — G2c
can, and did (1.0000).

## THE ANE FINDING (the headline; owner-visible)

On this M5 Ultra (macOS 27), `MLComputePlan.preferredComputeDevice` reads
**CPU for everything testable**:

| Bundle | units | device / ane / gpu / cpu | transitions |
|---|---|---|---|
| silero v6.2.1 (published `.mlmodelc`) | cpuAndNe | 357 / 0 / 0 / 357 | 0 |
| silero v6.0.0 (locally compiled `.mlpackage`) | cpuAndNe | 357 / 0 / 0 / 357 | 0 |
| silero v6.2.1 | cpuAndGpu | 357 / 0 / 0 / 357 | 0 |
| system `ATXIntentPredictionMLModel` | cpuAndNe | 108 / 0 / 0 / 108 | 0 |
| system `ATXIntentPredictionMLModel` | cpuAndGpu | 108 / 0 / 0 / 108 | 0 |

The A/B refutes cross-machine specialization staleness (local compilation
changes nothing), and the system-model controls + the cpuAndGpu arms rule
out anything silero-specific. Consequence: **the laya strict ANE gate
(`ane_ops == device_ops`) would refuse every load on this box** — the plan
walk is not a usable ANE-placement authority on M5-class silicon / macOS 27
today. An execution-side instrument (what actually RAN each op — Xcode Core
ML performance report / inference observation) would be needed. The
M3-measured ANE claims (2026-09) stand on their own boxes; re-run the walk
here before citing any ANE placement claim on M5-class silicon.

## T1 prediction scorecard (prediction recorded BEFORE measurement)

| Predicted | Measured | Verdict |
|---|---|---|
| loads under cpuAndNe | loads, both postures | HELD |
| p50 2–40 ms/chunk (dispatch-bound band) | 0.163 ms | **FALSIFIED, fast side** — the graph collapses to ~1–2 device passes |
| plan transitions > 0 | 0 (all-CPU) | **FALSIFIED** — the ANE finding |
| settled silence < 0.1 | 0.0308 | HELD |

## Serving verdict

**The lane works: 1,500× real-time headroom on CPU, correct discrimination
(real speech 1.0000 vs every negative < 0.05), deterministic streaming
state, digest-pinned loads.** VAD does not need ANE at this size — the ANE
question transfers to the bigger audio models (ASR/TTS lanes, Research 001
F2+), where the all-CPU finding matters and is unresolved on this silicon.

## Reproduce

```sh
scripts/fetch_silero_vad.sh
cargo run -p riir-infer-audio --features vad-ane --example audio_bundle_digest -- \
  ~/.cache/riir-infer/audio/silero-vad-unified-256ms-v6.2.1.mlmodelc
cargo test -p riir-infer-audio --features vad-ane --test silero_vad_poc -- --ignored --nocapture
cargo run --release -p riir-infer-audio --features vad-ane --example vad_timing -- 1000
# the placement instrument:
cargo run --release -p riir-infer-ane --example ane_plan_probe -- <any>.mlmodelc
```

Provenance of the 2026-10-26 final numbers: `power=AC Power load=2.99→3.12
swap=6021.94M canary=skipped powermode=0(auto)` — bench_preflight PASSED both
runs (ceiling 6.0). Disclosed concurrent consumers: the riir-rethink elephant-panel
capture (MPS/GPU + ~1 core) and ~6 GB swap in use — the VAD path is all-CPU
(the placement table above), so neither contends with it; the swap warning is
carried, not hidden. The canary was skipped (the laya-riir-metal example not
built on this box); powermode=0 (Automatic) is recorded by the preflight's own
probe, which is the load-bearing check here.

Historical provenance of the 10-23 provisional numbers: `power=AC Power
load=8.49 swap=2.38M canary=skipped powermode=0(auto)` (bench_preflight
REFUSED a published number — plan459 sweep resident at ~800% CPU).
