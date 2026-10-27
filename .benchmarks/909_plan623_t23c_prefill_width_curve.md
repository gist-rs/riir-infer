# Bench 909 — Plan 623 T2.3c: the prefill_width cost curve under the Issue-1141+ posture

**Status:** RECORD — the single-box grid MEASURED; the 2-box ABBA RAN and DEMOTED the single-box-derived 48 default (64 stands); T2.3c CLOSED at the demotion
**Date:** 2026-10-28 (grid 01:41–01:50; ABBA 02:05–02:25)
**Session:** zed-amo-i623-t24, 1793123917
**Plan:** [`.plans/623_multi_host_pipeline_inference_lane.md`](../.plans/623_multi_host_pipeline_inference_lane.md) T2.3c
**Instrument:** `tests/two_stage_prefill_gpu.rs :: prefill_width_curve_grid` (added this session) — ONE full-range Metal stage, real Bonsai-2 27B, best-of-2 per cell after a warm-up of BOTH kernel sets (first-call Metal compiles would otherwise read as the chunk floor).
**Box:** amo (M5 Ultra, macOS 27.0.1, AC), load 1.5, GPU-exclusive at fire (the two daytime measurement lanes — 886-t5 device gates + the 1122 ktile64 re-cell — had drained; verified `ps` quiet). Commit at fire: riir-infer `fc6f546`.
**Model:** `../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf`, full scope, ctx 264.

## The grid (wall seconds, best-of-2; `ratio` = per-position wall / chunked wall, > 1 means chunked wins)

| P | per-pos | s/tok | w=8 | w=16 | w=32 | w=64 | w=128 |
|---|---|---|---|---|---|---|---|
| 32 | 0.569 | 17.8 ms | 0.805 (0.81×) | 0.488 (**1.17×**) | 0.589 (0.97×) | — | — |
| 48 | 0.849 | 17.7 ms | 1.058 (0.80×) | 0.733 (1.16×) | 0.834 (1.02×) | 0.629 (**1.35×**) | — |
| 64 | 1.135 | 17.7 ms | 1.410 (0.81×) | 0.978 (1.16×) | 1.180 (0.96×) | 0.657 (**1.73×**) | — |
| 96 | 1.710 | 17.8 ms | 2.117 (0.81×) | 1.471 (1.16×) | 1.773 (0.96×) | 1.250 (1.37×) | 0.872 (**1.96×**) |
| 128 | 2.298 | 18.0 ms | 2.831 (0.81×) | 1.967 (1.17×) | 2.371 (0.97×) | 1.325 (1.73×) | 0.977 (**2.35×**) |
| 256 | 4.632 | 18.1 ms | 5.730 (0.81×) | 4.002 (1.16×) | 4.813 (0.96×) | 2.711 (1.71×) | 2.016 (**2.30×**) |

(Raw width cells — the runner's EVENED width is what ships: requested 128 evens to the whole prompt below 128, i.e. the w=P column.)

## Findings

1. **The T2.3b-era model is dead.** The old numbers (floor 0.9 s + 18 ms/tok
   vs 39 ms/tok per position, break-even ≈ 43) do not survive the
   Issue-1141+ kernel stack: per-position dropped **39 → 17.8 ms/token
   (−55%)** while the chunk floor dropped 0.9 → ~0.49 s (−45%). The
   break-even the old model predicted no longer exists in the measured
   range: **chunking wins at EVERY measured P with the right width** —
   even P=32 (w=16, 1.17×).
2. **The width ladder is sharp.** w=8 always loses (0.81× — the floor
   dominates); w=32 is a dead zone (~0.96×); the winners are the widest
   chunk the prompt supports: single-chunk (w ≥ P) or w=128 above. Within
   one chunk the cost grows super-linearly (0.589 @ 32 → 0.977 @ 128) —
   the every-4th-layer full attention's within-chunk quadratic term — but
   it stays under the per-position linear cost past P ≈ 48.
3. **The P=32 kink is real but small**: the single 32-chunk loses to
   per-position (0.965×) while two 16-chunks win (1.166×). The policy
   sidesteps it: below the floor, per-position.
4. **The default width stays 128** — evened, that is one whole-prompt
   chunk at P < 128 (1.35× @ 48, 1.73× @ 64, 1.96× @ 96) and 128-chunks
   above (2.35× @ 128, 2.30× @ 256). No measured P ≥ 48 prefers another
   width.

## The re-tune (LANDED this session)

- `HeadOptions::min_batched` **64 → 48** (`runner.rs`): chunking wins at
  P = 48 (1.35×, single chunk) and the old floor left that on the table;
  P = 32 stays per-position (the policy width there loses). The default
  requested width (128 on `--backend gpu`) is unchanged, now with the
  curve as provenance.
- Pin test `prefill_width_evens_chunks_and_floors_short_prompts` extended:
  the default floor pinned at 48, P=48 → 48 (single chunk), P=47 → 1,
  P=96 → 96 (the evening law at the new floor).
- Scope law: this default only bites when `prefill_chunk > 1` (the GPU
  posture — the bin defaults width 1 on cpu, where chunking only forfeits
  the T2.3a per-position overlap). Flag-off behaviour is unchanged.

## The 2-box ABBA (the task's acceptance cell — ran same session)

amo head (GPU) ⇄ bmo tail (GPU), TB5 192.168.10.x, split 32, `dist_stage`
binary shipped from this build (arm64, system links only), P tokens via
`--prompt`, `--max-new 1 --no-eos` (the prefill wall is the measured
quantity), key-authenticated, clean shutdown every cell:

| P | posture | runs (s) | median |
|---|---|---|---|
| 64 | chunked (evened 64, one chunk) | 1.091 / 0.996 | 1.044 |
| 64 | per-position (warmup 1.050) | 1.000 / 0.932 | 0.966 |
| 256 | chunked (128×2) | 1.828 / 1.899 | 1.864 |
| 256 | per-position | 2.757 / 2.717 | 2.737 |

**P = 256: chunked wins 1.47×** (better than T2.3b's 1.30× — the 1141+
posture improved the batched path further). **P = 64: per-position wins
~8%** — T2.3a's per-position pipelining overlaps the stages every
position, while a whole-prompt chunk serialises head-then-tail; the
single-box kernel win does not survive the wire at this P. The pipeline
crossover sits in (64, 256) — exactly where the standing 64 floor already
sits.

## Verdict (the demotion)

The single-box grid said `min_batched` 48 (chunking wins at every
measured P); **the ABBA — the task's own acceptance cell — REFUTED it
for the posture this runner serves**: a 48 default would chunk P ∈ [48,
64) and LOSE ~8% there on the wire. **`min_batched` stays 64**; the
revert is in the same session as the 48 landing (never pushed to any
consumer posture in between). The default requested width 128 stands on
BOTH postures (the P = 256 cell is its direct confirmation; T2.3b's 1.30×
became 1.47× under the 1141+ stack). The single-box 48 optimum is
recorded for any future 1-box stage consumer — it is real physics, just
not this runner's.

## The re-tune history (this bench's own arc)

- `min_batched` 64 → 48 landed on the single-box grid alone
  (`a970c7f`, no consumer posture ran in between).
- The ABBA ran the same session; 48 demoted back to 64 with the ABBA
  table above as provenance (this commit).
- Pins: `prefill_width_evens_chunks_and_floors_short_prompts` carries the
  64 floor + the evening law; the doc comment carries BOTH curves.

## Honest residue

- **The crossover is bracketed, not pinned**: P ∈ {96, 128, 192} 2-box
  cells would locate it exactly. The floor 64 is safe at both bracket
  ends; nobody requested a tighter bound.
- The grid under-samples P > 256 (league prefill rows go to 2048 — at
  P ≥ 2048 the evened-128 ladder is 16 chunks ≈ 20 s vs per-position
  ≈ 37 s; the win direction holds, the exact ratio is unmeasured).
- G3 (no-regression): full dist suite green at every step (9+2+8+7); the
  G1 bit bars are width-invariant by construction, so no bit-identity
  re-proof was owed.
- Box state at the ABBA: amo load 1.5, bmo load 1.3, both GPUs
  exclusive to the cells, ~15% run-to-run spread across the P = 64 cells
  (TB5 + two GPUs — the medians carry it, the verdict does not hinge on
  the margin).
