# Bench 909 — Plan 623 T2.3c: the prefill_width cost curve under the Issue-1141+ posture

**Status:** RECORD — the grid is MEASURED, the `min_batched` re-tune is LANDED (64 → 48); the 2-box ABBA confirm stays open
**Date:** 2026-10-28 (cells 01:41–01:50)
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

## Honest residue

- **The 2-box ABBA confirm is OPEN** — the task text asks for it and this
  session ran the single-box curve only (amo, Metal, quiet). The bmo leg
  needs the binary shipped + a coordination window; the curve's
  conclusions are per-stage physics (kernel floors + slopes), which the
  2-box cells only re-weight through the wire hop — but the confirm is
  still owed.
- G3 (no-regression) is the full dist suite at the new default: 9+2+8+7
  green + the extended pins (ran post-change). The G1 bit bars are
  width-invariant by construction (the anchor test pins chunked ==
  production at any width), so no bit-identity re-proof was owed.
- The grid under-samples P > 256 (the 2-box cells ran P=256; league
  prefill rows go to 2048 — at P ≥ 2048 the evened-128 ladder is
  16 chunks ≈ 16 × 1.28 s ≈ 20 s vs per-position 2048 × 18 ms ≈ 37 s;
  the win direction holds, the exact ratio is unmeasured).
