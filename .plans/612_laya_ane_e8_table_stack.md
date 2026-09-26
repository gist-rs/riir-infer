# Plan 612 — the e8 int8-embedding table stack for the laya ANE lane

**Status:** PROPOSED 2026-09-26 — owner directive ("add plan to create new int8 stack"); grounded in `.research/002_FluidUse_e8_Int8_Embedding_Stack.md` (FluidUse `@ 0a5c85e7` / mobius `@ 5beb3400`, scheme + parity evidence pinned there).

## Goal

Halve the resident embedding-table cost of the laya ANE lane with an `e8` posture:
**linear-symmetric per-channel int8 table + f32 scales, dequantized in the host gather,
ANE artifact byte-identical.** The embedding table is the dominant weight (256k × 768;
~393 MB resident as `Vec<u16>` per checkpoint today, converted from the fp32
safetensors at load). e8 is the ecosystem's only compression variant that survives
parity (their w8/w8e/w6/w4 all fail ANE parity — refused here by the same evidence).
Default posture stays fp16 until the GOAT gate passes; then promote, fp16 stays
env-selectable (demote-the-loser rule).

**Why a weights-layer stack, not artifact precision:** our ANE artifacts EXCLUDE the
table (`ane.rs::AneEncoder::from_map` removes it; the gather feeds `embeddings [1,L,d]`
as the artifact input) — unlike FluidUse's packages, which contain it. So e8 lands as
ONE int8 sidecar per checkpoint serving ALL buckets, and the artifact/manifest/placement
discipline (100%-ANE load re-gate, digest pins) is untouched by construction.

## Non-goals

- **No encoder-weight int8, no sub-8-bit palettes** — their published parity failures
  (`w8`/`w8e`/`w6`/`w4` vs ANE) are the refusal boundary; re-opening one needs its own
  issue with new evidence.
- **No artifact change** — the six BC1S FP16 `.mlpackage`s stay byte-identical; no
  reconversion, no new manifest rows for artifacts.
- **No CPU/Metal lane change in v1** — their G5 bar is 1e-3 p-drift; an int8 table
  changes the table bytes and would fail it. Lane-scoped to the ANE lane (whose
  decision-level gate is the authority); extending to CPU/Metal needs a decision-level
  gate variant first — deferred, noted in Phase 3.
- **No download-slicing win claimed in v1** — the sidecar enables future
  download-on-demand slicing (issue 017's territory) but the ANE lane still consumes
  the shared checkpoint safetensors (the head reads its own names from the same map).

## Phase 0 — technique intake (read-only)

- [ ] Read the dequantize op's axis in THEIR converted mlpackage (`reports/verification-multilingual-L128-*.json` + the L128 package's gather/dequant op metadata @ `5beb3400`) and record it as a PRIOR ONLY in the conversion log — their size arithmetic cannot settle the axis (both options cost < 1 MB of scales against ~197 MB of int8, so 614 → 448 MB is consistent with either).
- [ ] Adopt **per-row (vocab-axis) f32 scales** as OUR default (~256k scales ≈ 1 MB ≈ 0.5% of the table): each token row carries its own range, so one outlier token cannot ruin the others — per-column (768 shared scales) lets a single outlier row set every column's range, which costs accuracy on an embedding table. Their published variant being per-channel does not bind us; our table lives outside the artifact, so we never need to match their axis.
- [ ] Record the refused-variant table (w8/w8e/w6/w4 ANE parity failures) in `assets/ane/conversion_log.md` as the plan's refusal boundary.
- [ ] Confirm in `ane.rs` that the artifact input construction is the ONLY table consumer on the ANE path (no other `table_f16` read) — the byte-identical-artifact claim rests on it.

## Phase 1 — offline quantizer (**riir-reflex-owned**; tracked as riir-reflex `.issues/037`; one tool, one log)

- [ ] Extend `riir-reflex/scripts/ane_convert.py` with `--table-precision e8`: linear-symmetric int8 over `encoder.embeddings.tok_embeddings.weight` with **per-row (vocab-axis) f32 scales**, emitted as `<ane_root>/<model>/table_e8.safetensors` (i8 tensor + scales tensor). The landing commit for this task belongs to riir-reflex — this plan consumes it, it does not edit reflex from here.
- [ ] Manifest rows: `<model>/table_e8` in `assets/ane/manifest.json` — BLAKE3 digest + shapes + scale dtype, the same digest discipline as artifact rows.
- [ ] Determinism golden: two consecutive runs produce byte-identical sidecars (the quantizer is closed-form min/max symmetric — assert it, never assume it).
- [ ] One per checkpoint (english / multilingual / typed): three sidecars, each serving every bucket of its checkpoint.

## Phase 2 — substrate runtime (riir-infer-laya, feature `laya-riir-ane` scope)

- [ ] `AneEncoder` gains the e8 table arm selected by env `LAYA_ANE_TABLE=e8` (default unset = fp16 posture, byte-identical behavior pinned by the existing gates); a set env with an absent sidecar is a LOUD refuse naming the converter command — never a silent fallback.
- [ ] `gather_e8` (per-row scales, the Phase 0 default): read `i8[row*d + col] * scale[row]` → f32 → fp16 bits into the SAME `[1,L,d]` scratch buffer (zero-alloc preserved; the PAD row dequantized once per forward, not per token).
- [ ] Load-time verification: sidecar BLAKE3 re-checked against the manifest + scale shape == hidden width (the no-silent-fallback law, same posture as the artifact digest re-verify).
- [ ] Resident memory: `Vec<u16>` (2 B/elt) → `Vec<i8>` + scales (~1 B/elt) — assert the halving in a debug assert on table byte length.

## Phase 3 — GOAT gate + bench + promotion

- [ ] G1 correctness: `tests/laya_ane_parity.rs` at the e8 posture — top-1 ≥ 99.9% + flips only inside the near-tie band, across all three checkpoints, bucket-covered rows; publish max prob err e8-vs-fp16 beside the fp16 posture's published err (their datum: 0.013 → 0.014/0.015 — expect the same noise class; a larger shift is a RED, not a note).
- [ ] G2 perf: position-balanced p50 forward, e8 vs fp16, same box, `scripts/bench_preflight.sh` provenance line quoted (their claim: latency identical; ours measured, never assumed).
- [ ] G3 no-regression: fp16 posture outputs byte-identical (env unset); CPU/Metal lanes untouched — the riir-infer diff touches only `ane.rs`; the converter change is riir-reflex-owned (`.issues/037`) and lands there.
- [ ] G4 alloc: gather writes into the pre-allocated buffer (the lane's alloc discipline; counting gate where the lane carries one).
- [ ] G5 size axis (this plan's gain axis): resident table bytes u16 → i8+scales published per checkpoint (expected exactly ~50% of the table); sidecar bytes on disk published. **G5 is guaranteed by construction (int8 + scales is always ~50% of fp16) — it is a published measurement, never evidence about quality; promotion rides G1 + G2.**
- [ ] Promotion: e8 becomes the lane DEFAULT iff **G1 + G2 pass** (with G3/G4 holding); else stays opt-in with the failure recorded. The loser (fp16) keeps its env path — demote, never delete.

## Deferred (recorded, not planned here)

- CPU/Metal lanes on the int8 table — needs a decision-level gate variant (their 1e-3 drift bar is lane-authoritative); own issue if a consumer demands it.
- Download-on-demand slicing enabled by the sidecar — issue 017's scope.
- The M5-vs-M3 ANE latency cross-check (their 3.6 ms vs our ~24.5 ms p50 at the same technique) — F5-style read of their `coreml-cli` placement tables beside ours; observation recorded in Research 002 §1.4.
- GLiClass-style joint top-two scoring for the game heads — reflex arena lane (Research 002 §1.5 observation).

## Risks / caveats

- Their e8 numbers are from THEIR conversion pipeline on M5 Pro; our quantizer is independent — the 0.5-pt suite bar and the noise-class Δprob claim must be RE-MEASURED on our artifacts (G1 owns it; their numbers are priors, never evidence).
- Scale axis: per-row (vocab) is our default on accuracy grounds (outlier isolation); Phase 0 records THEIR axis as a prior only — if their per-column choice proves materially more accurate on OUR artifacts, flipping the default is a one-line scale-index change plus a sidecar regen, decided by G1, not by their precedent.
- The `LAYA_ANE_TABLE` env must NOT gate raw sync or correctness-critical paths — it is a memory-precision knob on an opt-in lane; fp16 default is the always-correct posture.
