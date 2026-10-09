# Plan 612 — the e8 int8-embedding table stack for the laya ANE lane

**Status:** PROPOSED 2026-09-26 — owner directive ("add plan to create new int8 stack"); grounded in `.research/002_FluidUse_e8_Int8_Embedding_Stack.md` (FluidUse `@ 0a5c85e7` / mobius `@ 5beb3400`, scheme + parity evidence pinned there). **UPDATE 2026-10-09 (this re-adjudication, evidence-cited): Phases 0 + 1 are DONE** — landed in riir-reflex 2026-09-26 (converter commit `6535b75` "feat(037): e8 table converter arm — per-row int8 sidecar per checkpoint (Plan 612 Phase 1)"; the companion riir-reflex `.issues/037` was filed, landed, and removed post-landing per the noise-reduction rule — the conversion log cites it). **UPDATE 2026-10-09 (late): PHASE 2 LANDED** in riir-infer (this repo) — `ane.rs` gains the e8 consumer arm: `LAYA_ANE_TABLE=e8` env selection (unset = fp16, byte-identical; unknown values refuse loud), the verified sidecar load (BLAKE3 + byte total vs the manifest row, geometry cross-checked against the checkpoint's own embedding tensor, `per_row_vocab` axis enforced, `serves_buckets` coverage enforced, absent/sidecar-less loads name `ane_convert.py` and never fall back), `gather_e8` (per-token `i8 · scale → f32 → f16 bits` into the SAME `[1,L,d]` scratch; PAD row dequantized once per forward then `copy_within`-fanned), and the resident-halving debug assert. The safetensors container preamble was extracted to `weights::container_header` (one home for the 8-byte-header law; `from_bytes` delegates — behavior-preserving, covered by the existing weights tests). 9 new `posture_tests` (gather arithmetic, env contract, digest/pairing/axis/serves/missing walls) + the 71-test lib run green at `--features laya-riir-ane`; default and `--no-default-features` lib tests + clippy green (G3's CPU/Metal-untouched claim holds — the weights extraction is behavior-identical). Real sidecars verified on disk at the pinned byte counts (english/typed 51,778,464; multilingual 197,632,160). **Phases 2–3 remain: Phase 3 only (the GOAT gates) — G1/G2 need a measurement window (Plan 337 load carve; box load 6.6–8.9 all day, sibling census running).** The conversion log (`riir-reflex/assets/ane/conversion_log.md`) carries every Phase-0 record per checkpoint: the axis prior (mobius quantize.py dequantizes PER-CHANNEL — PRIOR only), the per-row (vocab-axis) f32-scale adoption with its accuracy grounds, the refused-variant boundary (w8/w8e/w6/w4), determinism (in-run double-quantize byte-identical + cross-run golden), and artifacts-untouched digests. The manifest rows are rich (`axis: "per_row_vocab"`, blake3 digest, per-tensor shapes/dtypes, source sha256, quant formula). ⛔ **Correction of the earlier scope note:** the `<model>/table_e8` manifest rows skipped by `ane.rs::AneManifest::load` are NOT a foreign "KV-table lane" — they are THIS plan's own Phase-1 output; the loader skip was the correct pre-Phase-2 posture (the consumer did not exist yet; Phase 2 now retains those rows raw and parses them only when the e8 posture selects them). Independent verification this pickup: the mobius clone was re-fetched at the pinned sha and the axis prior re-derived from code (`granularity="per_channel"` on the gather-consumed `[vocab, hidden]` weight → coremltools per-channel = one scale per dim-0 row = per-row) + size arithmetic (english 843,174,305 → 791,749,909 saves 51,424,396 B vs the per-row prediction 51,375,360 B, Δ 0.1% container noise; per-column predicts 51,572,736, Δ −0.3%) — per-row fits closer, corroborating the code leg; clone removed after the read per the standing rule.

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

## Phase 0 — technique intake (read-only) — **DONE 2026-09-26 (riir-reflex landing `6535b75`), re-verified 2026-10-09**

- [x] Read the dequantize op's axis in THEIR converted mlpackage (`reports/verification-multilingual-L128-*.json` + the L128 package's gather/dequant op metadata @ `5beb3400`) and record it as a PRIOR ONLY in the conversion log — their size arithmetic cannot settle the axis (both options cost < 1 MB of scales against ~197 MB of int8, so 614 → 448 MB is consistent with either). **RECORDED** in the conversion log ("axis prior" row per checkpoint): mobius quantize.py @ 5beb3400 dequantizes PER-CHANNEL. 2026-10-09 re-verification strengthened the record: the code leg (`granularity="per_channel"` in `OpLinearQuantizerConfig` on the gather-consumed 2-D weight → one scale per dim-0 row) plus the english conversion's size arithmetic (per-row prediction within 0.1%; per-column −0.3%) — per-row fits closer; the PRIOR-ONLY scoping is unchanged (G1 decides on OUR artifacts).
- [x] Adopt **per-row (vocab-axis) f32 scales** as OUR default (~256k scales ≈ 1 MB ≈ 0.5% of the table): each token row carries its own range, so one outlier token cannot ruin the others — per-column (768 shared scales) lets a single outlier row set every column's range, which costs accuracy on an embedding table. Their published variant being per-channel does not bind us; our table lives outside the artifact, so we never need to match their axis. **EMBODIED in the landed converter** (`scale[r] = max|w[r,:]|/127`, manifest `axis: "per_row_vocab"`) and its grounds recorded in the log.
- [x] Record the refused-variant table (w8/w8e/w6/w4 ANE parity failures) in `assets/ane/conversion_log.md` as the plan's refusal boundary. **RECORDED** ("refused-variant boundary" row per checkpoint).
- [x] Confirm in `ane.rs` that the artifact input construction is the ONLY table consumer on the ANE path (no other `table_f16` read) — the byte-identical-artifact claim rests on it. **CONFIRMED 2026-10-09** (grep): `table_f16` is read only by `gather_fp16` (ane.rs:466/470/490/541/596); the checkpoint-load conversion at :541 is the single producer.

## Phase 1 — offline quantizer (**riir-reflex-owned**; landed as riir-reflex `.issues/037` → removed post-landing; one tool, one log) — **DONE 2026-09-26, converter commit `6535b75`**

- [x] Extend `riir-reflex/scripts/ane_convert.py` with `--table-precision e8`: linear-symmetric int8 over `encoder.embeddings.tok_embeddings.weight` with **per-row (vocab-axis) f32 scales**, emitted as `<ane_root>/<model>/table_e8.safetensors` (i8 tensor + scales tensor). The landing commit for this task belongs to riir-reflex — this plan consumes it, it does not edit reflex from here. **LANDED `6535b75`** (numpy + blake3 only; no coremltools/torch).
- [x] Manifest rows: `<model>/table_e8` in `assets/ane/manifest.json` — BLAKE3 digest + shapes + scale dtype, the same digest discipline as artifact rows. **LANDED** — richer than specced: blake3 digest + per-tensor shapes/dtypes + source sha256 + the full quant scheme (`axis: "per_row_vocab"`, formula, levels, amax bounds) + container format + serves_buckets + artifacts-untouched + determinism.
- [x] Determinism golden: two consecutive runs produce byte-identical sidecars (the quantizer is closed-form min/max symmetric — assert it, never assume it). **ASSERTED** (in-run double quantize payload byte-identical + cross-run rewrite digest match, per checkpoint in the log).
- [x] One per checkpoint (english / multilingual / typed): three sidecars, each serving every bucket of its checkpoint. **LANDED** — english `ebaf93be…` (51,778,464 B), multilingual `0276e83f…` (197,632,160 B), typed `1899dcdb…` (51,778,464 B); each `serves_buckets: [64, 128]`.

## Phase 2 — substrate runtime (riir-infer-laya, feature `laya-riir-ane` scope) — **DONE 2026-10-09 (this pickup; all gates green, no measurement claimed)**

- [x] `AneEncoder` gains the e8 table arm selected by env `LAYA_ANE_TABLE=e8` (default unset = fp16 posture, byte-identical behavior pinned by the existing gates); a set env with an absent sidecar is a LOUD refuse naming the converter command — never a silent fallback. **LANDED**: `resolve_table_posture` (unset/empty → F16, `e8` → E8, anything else refuses naming the supported set) + `load_table_e8` (absent file / absent manifest row / digest mismatch / byte-total mismatch all refuse with `ane_convert.py table --model <model> --table-precision e8` in the message). The fp16 posture never parses the sidecar schema (manifest `table_e8` rows retained RAW, parsed on demand).
- [x] `gather_e8` (per-row scales, the Phase 0 default): read `i8[row*d + col] * scale[row]` → f32 → fp16 bits into the SAME `[1,L,d]` scratch buffer (zero-alloc preserved; the PAD row dequantized once per forward, not per token). **LANDED**: per-token `f32::from(q) * scale → f32_to_f16_bits` into `emb`; PAD row dequantized into the FIRST pad slot then `copy_within`-fanned over the tail; zero alloc (writes only the pre-allocated scratch).
- [x] Load-time verification: sidecar BLAKE3 re-checked against the manifest + scale shape == hidden width (the no-silent-fallback law, same posture as the artifact digest re-verify). **LANDED, stronger than specced**: BLAKE3 + byte total vs the manifest row; table shape `[vocab, hidden]` with hidden == the checkpoint's width; scales `[vocab]`; the sidecar PAIRS with the checkpoint's own embedding tensor (same vocab AND hidden — a wrong-sidecar pairing refuses at load); `quant.axis == per_row_vocab` enforced (any other axis is a different gather); `serves_buckets` must cover every bucket row the manifest carries for the model.
- [x] Resident memory: `Vec<u16>` (2 B/elt) → `Vec<i8>` + scales (~1 B/elt) — assert the halving in a debug assert on table byte length. **LANDED**: `debug_assert!(i8 bytes + 4·scales < 2·vocab·hidden)` at load (the fixture gate proved the assert fires — the toy d=3 fixture violates it, which is WHY the fixture runs d=8; real checkpoints d=768/1024 hold with ~20% headroom for the scale overhead).

**Phase 2 test evidence:** 9 new `posture_tests` (gather per-row dequant + pad-tail fan, no-pad full-bucket case, env contract incl. the typo refuse, verified happy-path load, digest-wall refuse, missing-sidecar and missing-row converter-naming refuses, pairing-wall refuse, serves_buckets-covering refuse) — 71/71 lib tests green at `--features laya-riir-ane`; default + `--no-default-features` lib tests and clippy (all-targets, three postures) green; `cargo fmt` clean (per-file).

## Phase 3 — GOAT gate + bench + promotion — **QUEUED behind box load (Plan 337; the only remaining phase — 2026-10-09)**

- [ ] G1 correctness: `tests/laya_ane_parity.rs` at the e8 posture — top-1 ≥ 99.9% + flips only inside the near-tie band, across all three checkpoints, bucket-covered rows; publish max prob err e8-vs-fp16 beside the fp16 posture's published err (their datum: 0.013 → 0.014/0.015 — expect the same noise class; a larger shift is a RED, not a note). **READY: the three real sidecars are on disk at the pinned byte counts; the env knob needs no reflex-side change (read inside `AneEncoder::from_map`).**
- [ ] G2 perf: position-balanced p50 forward, e8 vs fp16, same box, `scripts/bench_preflight.sh` provenance line quoted (their claim: latency identical; ours measured, never assumed).
- [ ] G3 no-regression: fp16 posture outputs byte-identical (env unset); CPU/Metal lanes untouched — the riir-infer diff touches only `ane.rs`; the converter change is riir-reflex-owned (`.issues/037`) and lands there. **Structural half holds post-Phase-2 (2026-10-09): the F16 arm is the verbatim founding code path (resolve → widen → `gather_fp16` unchanged); the `weights::container_header` extraction is behavior-identical (`from_bytes` delegates; the weights test battery passed untouched); default-posture lib tests + clippy green.** The runtime half (a G5-lane fp16 re-run) rides the same window as G1 — one clean box session proves both postures back-to-back.
- [ ] G4 alloc: gather writes into the pre-allocated buffer (the lane's alloc discipline; counting gate where the lane carries one). **Structural half holds post-Phase-2: `gather_e8` writes only `buf` (plus `copy_within` on it); no allocation inside the gather.**
- [x] G5 size axis (this plan's gain axis): resident table bytes u16 → i8+scales published per checkpoint (expected exactly ~50% of the table); sidecar bytes on disk published. **G5 is guaranteed by construction (int8 + scales is always ~50% of fp16) — it is a published measurement, never evidence about quality; promotion rides G1 + G2. PUBLISHED 2026-10-09 (computed from the PINNED manifest shapes — exact arithmetic, not a box measurement; the load-time debug_assert verifies the resident figure every e8 load):**

  | checkpoint | vocab × hidden | fp16 resident (u16) | e8 resident (i8 + 4 B/row scales) | ratio | sidecar on disk (manifest-pinned) |
  |---|---|---|---|---|---|
  | english | 50,368 × 1,024 | 103,153,664 B | 51,778,304 B | 50.195% | 51,778,464 B (160 B container) |
  | typed | 50,368 × 1,024 | 103,153,664 B | 51,778,304 B | 50.195% | 51,778,464 B |
  | multilingual | 256,000 × 768 | 393,216,000 B | 197,632,000 B | 50.262% | 197,632,160 B |

  Per-checkpoint G1 **recipe (zero consumer-side change — the env is read inside `AneEncoder::from_map`):** `cd ../riir-reflex && LAYA_ANE_TABLE=e8 cargo test --release --features laya-riir-ane --test laya_ane_parity` (decision-level gates vs the frozen fp16 goldens; publishes max prob err e8-vs-golden as observation). The fp16 baseline is the same command without the env. Run both in ONE clean-box window (G1 first, G2 after, preflight quoted).

  **The window runbook (pinned 2026-10-09 23:4x so the session is pure execution — every precondition verified in place: sidecars on disk at pinned bytes + BLAKE3-verified `ebaf93be…`/`1899dcdb…`/`0276e83f…` vs the manifest; census ETA ~06:30 +07):**

  1. **Quiet-box assert first**: `uptime` (1-min load < 6 AND trending down) + `sysctl vm.swapusage` (used near 0, not the 20 GB census pin) + census PID 54022 EXITED with `<model>.t1_bias_delta.json` in `../riir-infer/.raw/hyperthink_t1/` (if the census is still alive, the window is NOT open — reschedule, never share the box).
  2. `../riir-reflex/scripts/bench_preflight.sh` — quote the `PROVENANCE:` line beside every number below.
  3. **G3 runtime half + fp16 baseline (one run)**: `cd ../riir-reflex && cargo test --release --features laya-riir-ane --test laya_ane_parity` (env unset) — byte-vs-golden parity at the fp16 posture across `Checkpoint::ALL`.
  4. **G1**: same command + `LAYA_ANE_TABLE=e8` — top-1 ≥ 99.9%, flips only in the near-tie band, all three checkpoints; record max-prob-err beside the fp16 run's (expect the 0.013 → 0.014/0.015 noise class).
  5. **G2 (alternating-run pairing — the `laya_fixture_timing` instrument is sequential-arms, so position-balancing comes from run ORDER, never from two concurrent processes)**: `scripts/bench_preflight.sh`, then alternate `cargo run --release --features laya-riir-ane --example laya_fixture_timing -- ane english 8` (×4 reps each, fp16 runs 1st + 3rd, e8 runs 2nd + 4th, `LAYA_ANE_TABLE=e8` exported for the even runs only); repeat the block for `typed`; multilingual optional (largest table, slowest load). Verdict = median of paired row_p50 ratios (run2/run1, run4/run3) — a pair contaminated by any load spike is re-run, never averaged in. Bar: e8 p50 within noise of fp16 (their claim: identical; accept ≤ 3% median delta with both pair medians agreeing in sign).
  6. Record all five figures + both PROVENANCE lines into this plan (Phase 3 checkboxes), then the promotion call per the checkbox above. ONE commit (`feat: plan 612 phase 3 — …` or the honest negative), push, update `.highwater` files if a bench record is filed.
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
