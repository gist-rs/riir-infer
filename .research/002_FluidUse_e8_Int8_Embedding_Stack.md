# Research 002: FluidUse — the e8 int8-embedding decision stack (FluidInference, decision side)

**Status:** RECORD 2026-09-26 — verdict Gain; plan filed (`.plans/612_laya_ane_e8_table_stack.md`). Companion to Research 001 (the org's AUDIO side); this note owns the DECISION side delta the owner directive points at.

**Filed from:** owner directive 2026-09-26 — "add plan to create new int8 stack".
**Sources (pinned):** FluidUse `@ 0a5c85e780ce0607d28697240f9cb4f58f0bb426` (Apache-2.0) — README + Benchmarks.md · mobius `@ 5beb34007656d16349fec757c379a9beae45c4d5` (Apache-2.0) — `models/computer-use/laya/coreml/quantize.py` + reports · HF `FluidInference/laya-coreml` card. Clones lived in `riir-infer/.raw/{FluidUse,mobius}`, removed after commit (shas above are the provenance).

## TL;DR

FluidInference's FluidUse repo publishes the **decision-model serving stack** for Apple
silicon: laya (our `riir-infer-laya` substrate's own model family) at **3.6 ms warm L128
on CPU+ANE (M5 Pro)**, CUA-S1-FORMS form filling at 0.9 ms, GLiClass Edge at 1.6 ms —
and, load-bearing for the owner directive, a **published post-training int8 precision
stack** with an honest variant matrix: `e8` (int8 embedding table only, linear-symmetric
per-channel, encoder/head stay fp16) is the **only compression variant that passes their
parity gates** — 30% smaller packages, accuracy within 0.5 pt on the full 10-suite
benchmark, identical latency. Encoder-int8 (`w8`/`w8e`) and 6-/4-bit k-means palettes
**fail ANE parity and are deliberately not published**. Our stack already ships the same
lane shape (reflex Plan 002 P0/P1: six BC1S FP16 artifacts, G5-ANE decision gate GREEN
76/76), but with one structural divergence that reshapes where int8 lands for us:

> **Our ANE artifacts EXCLUDE the embedding table.** `ane.rs::AneEncoder::from_map`
> removes `encoder.embeddings.tok_embeddings.weight` from the safetensors map, gathers
> host-side (`gather_fp16`), and feeds `embeddings [1,L,d]` as the artifact INPUT —
> FluidUse's packages CONTAIN the table (393 MB of their 614 MB fp16 bucket). So
> "e8 for us" is **not** an artifact-precision change: it is a **weights-layer stack** —
> one digest-pinned int8 table sidecar per checkpoint (≈50% of the resident table),
> dequantized in the host gather, serving ALL buckets and (future) all lanes.

## 1. What is new since Research 001 / reflex Plan 002

1. **The e8 scheme, pinned from `quantize.py`:** `cto.OpLinearQuantizerConfig(mode=
   "linear_symmetric", dtype="int8", granularity="per_channel")` applied ONLY to weights
   whose child op is `gather` (the embedding table); encoder/head untouched. The trap
   they document and we inherit: the **default weight threshold would also sweep the
   additive attention masks, the RoPE cos/sin tables and the biases into compression,
   which wrecks the outputs on the ANE** — they filter to 2-D `weight_to_fp16` tensors
   feeding `linear`/`gather` ops at `weight_threshold=2048`.
2. **Their published parity findings** (Benchmarks.md, M5 Pro, macOS 27): Core ML fp16
   matches the PyTorch reference on all 10 rebuilt suites (3,899 questions; AG News
   0.935, Emotion 0.537, MASSIVE-20 0.657, spam/phishing 0.993, guardrails 0.808);
   e8 is within 0.5 pt everywhere with Δprob 0.013 → 0.014/0.015 (noise-class shift);
   packages 614 MB → 448–453 MB per bucket (the default 128+512 download 1.32 GB →
   0.93 GB); **latency identical**.
3. **The refused variants are evidence, not gaps:** `w8`/`w8e` (encoder-int8) and
   `w6`/`w4` palettes fail their parity gates on the ANE — the ecosystem's own data says
   **embedding-only int8 is the frontier** for this model class. Our plan refuses
   encoder-int8 and sub-8-bit by default on that evidence.
4. **The cross-chip gap (observation, unresolved, NOT this plan's scope):** their L128
   CPU+ANE warm median is 3.6 ms on an M5 Pro; our Plan 002 P1 measured ~24.5–25.5 ms
   p50 on the M3 Max at the same technique (bucketed whole-graph ANE, 100%-ANE
   placement). Upstream's own M1 Max GPU figure (~27 ms) matches OUR Metal lane
   (~26.9 ms fixtures) — so the residual is chip generation + possibly serving shape.
   An F5-style cross-check (read their `coreml-cli` per-op placement tables beside ours)
   is the cheap follow-up; filed nowhere yet, recorded here as observation.
5. **Tetris/2048 policy landscape (observation for the game-heads lane):** GLiClass
   (32.7M) ranking the heuristic's top-two landings JOINTLY beats laya scoring each
   landing independently (3,667 vs 568 pieces mean; 4.3× less model time per move at
   2048) — joint-pair scoring + call-count reduction is the pattern our Bench-881/882
   game heads do not yet use. Recorded for the reflex arena lane; no file created here.

## 2. Internal census (what already ships — no duplication)

- reflex Plan 002 COMPLETE: six BC1S FP16 artifacts (3 checkpoints × L64/L128),
  100%-ANE/0-transition load re-gate, digest-pinned manifest, G5-ANE decision gate
  (top-1 ≥ 99.9% + near-tie band; 76/76 green, max prob err ≤ 0.0264) — **the gate the
  e8 posture must re-pass is already standing** (`tests/laya_ane_parity.rs`).
- `ane.rs`: `AneEncoder` owns the table (`table_f16: Vec<u16>` converted from the fp32
  safetensors at load) + `gather_fp16` into the artifact's `[1,L,d]` input — **the single
  seam the int8 table slots into**; the ANE graph stays byte-identical by construction.
- reflex `scripts/ane_convert.py` + `assets/ane/{manifest.json,conversion_log.md}` — the
  offline conversion discipline the quantizer extends (one tool, one log; no second
  converter).
- riir-clippy `choice_scorer_poc` already consumed the CUA-S1 lineage (Bench 098,
  REFUTED for healer-rule selection); reflex Issue 035 owns the CUA-S1-FORMS CoreML
  arena arm (T1 lineage DONE 2026-09-26). No new issue needed for that here.
- Novelty: **none claimed** — post-training int8 embedding quantization is standard
  practice (coremltools linear quantization, llama.cpp Q8-class embeddings); this is
  adoption of a published scheme onto our substrate, scored as integration Gain exactly
  like Research 001 (1 of 4 novelty gate — new *for our stack*, not a new class).

## 3. Verdict

**Gain — integration distill.** The actionable item is the e8 weights-layer stack for
the ANE lane (resident-table halving on the lane where serving already wins, with a
standing decision-level gate to re-pass), filed as
**`riir-infer/.plans/612_laya_ane_e8_table_stack.md`** (converter half tracked as
riir-reflex `.issues/037`). One deliberate divergence from their scheme: their variant
is per-channel; **our sidecar defaults to per-row (vocab-axis) f32 scales** (~1 MB,
≈0.5% of the table) on accuracy grounds — per-row isolates an outlier token's range to
its own row, per-column lets one outlier row set every column's range. Their size
arithmetic cannot distinguish the axes (< 1 MB of scales either way); their artifact's
dequant axis is recorded as a prior only (Plan 612 Phase 0). Fusion-priority ladder
check: the healer gains nothing (no healer surface consumes model weights); priority #3
(inference-perf league) is the served surface. MOAT (riir-infer row): quant formats are
explicitly in scope; zero new deps; zero `riir-*` deps preserved.

## 4. Provenance

- Clones (shallow, read-only, removed after commit): `riir-infer/.raw/FluidUse` @
  `0a5c85e7…`, `riir-infer/.raw/mobius` @ `5beb3400…`; pre-existing `.raw/` entries from
  other sessions untouched.
- Web: HF `FluidInference/laya-coreml` model card (artifact/shapes/parity tables);
  GitHub org + FluidUse README fetched 2026-09-26. No novelty claim → the §4 hard gate
  binds nothing here; the scheme's genericity is cited above instead.
- Internal-first sweep: `.research/` + `.plans/` + `src/` grepped for
  `FluidInference|FluidUse|FluidAudio|coreml|neural engine` — controlling prior art:
  Research 001 (this repo), reflex Research 002 + Plan 002 + Issue 035, katgpt-rs ANE
  notes (155/157/223/224/377/427/489). e8/LUT8/int8-embedding: **zero hits** in our
  trees before this note — the stack does not ship yet.
