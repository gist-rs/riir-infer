# Bench 919 — Issue 1146 rung 1: the prefill chunk-width re-curve through the batched posture

**Status:** RECORD (1-box leg) — width 512 is **1.39× / 1.40× / 1.42×** prefill vs the 128 default at pp512 / pp4096 / pp32768, G1 cross-width PASS (identical generated ids at 128/256/512/1024). The 2-box TB5 leg (the default's original reason) is PENDING a quiet two-box window — the default flip waits on it.
**Date:** 2026-10-28 (cells 16:07–16:35 +07 clean; contamination from 16:36 recorded below)
**Session:** katgpt-rs-71, 1793178290
**Issue:** [`.issues/1146_prefill_throughput_gap_1box.md`](../.issues/1146_prefill_throughput_gap_1box.md) — ladder rung 1 named by [Bench 914](914_issue1146_feature_row_ab_and_prefill_frame_census.md)
**Box:** bmo (Mac17,15, M5 Ultra, 30 cores, 96 GiB), AC power, load 0.1–0.3 at start, no other GPU consumer until 16:36
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` sha256 `3907dc16…62ec1` = the league pin EXACT
**Binary:** `dist_stage --features gpu,bonsai2_hadamard` at riir-infer `0daa149` (the Bench-914 armed posture: chunked deltanet + batched attention prefill), sha256 `40647de5…4c78`

## Rig

`scripts/i1146_width_recurve.sh` — the Bench-911 cell rig (loopback, `--split 32`, both stages
`--ctx P+256`, `--max-new 1`), ONE binary, only `--prefill-chunk W` varies. Each cell runs a warm
pass and then a timed pass. The prompt is a scrambled id sequence (`7 + (i·7919) mod 50000`), not a ramp.
The printed `prefill N tok in Xs` line is the compute wall (boot-free).

## G1 — cross-width token identity

pp1024 + 16 decode tokens at each width: **identical** `generated:` ids at 128/256/512/1024
(`[261, 248046, 198, 248045, …, 8253, 220]`). Width only moves frame boundaries; every chunked
body is already bit-identity-gated against its sequential twin, and this cell is the composition check.

## The curve (timed pass; warm in parentheses)

| prompt | w128 | w256 | w512 | w1024 | w512 vs w128 |
|---|---:|---:|---:|---:|---:|
| pp512 | 3.039 s (3.101) | 2.320 (2.295) | **2.185** (2.252) | — | **1.39×** |
| pp4096 | 25.696 (25.620) | 20.311 (20.257) | **18.400** (18.359) | 18.140 (18.200) | **1.40×** |
| pp32768 | 260.564 (260.724) | 209.835 (210.718) | **183.896 (warm only)** | contaminated | **1.42×** |

tok/s at w512: pp512 234 · pp4096 223 · pp32768 178 (vs 168 / 159 / 126 at w128).

- Warm vs timed agree within ≤ 3% in every clean pair (most < 1%), so the w512 pp32768 **warm**
  cell (16:32–16:35, before any interference) is a valid read.
- w1024 adds only +1.4% over w512 at pp4096 — the curve is flat past 512. The win comes from
  the M=128 → M=512 GEMM width (Bench 914: FFN + DeltaNet batch GEMMs = 90% of the frame), so it
  saturates once the simdgroup GEMM tiles fill.
- The gap vs llama.cpp (Bench 911: 1087 / 1046 / 772 t/s) narrows from 0.13–0.16× to
  **0.21–0.23×**. This is the config-class rung; the remaining ~4.5× is the W-frame kernel rung (ladder 2).

## Contamination (honesty note)

At 16:36 +07 a sibling lane started `bench_809_issue771_flash_cmma` + `perf_rematch.sh … prefill-pp16k`
on bmo. The **pp32768 w512 timed** cell (300.899 s vs its clean warm 183.896 s) and both w1024
pp32768 cells overlapped it. They are discarded, the run was stopped, and the session log keeps them for the record.

## What this does NOT yet decide

The `dist_stage` GPU default `--prefill-chunk 128` was picked as the **2-box** optimum (Bench 909,
through the old sequential posture). A wider frame trades away per-frame head⇄tail overlap: pp512 at w512 is
ONE frame, with no overlap at all. The flip to 512 needs the 2-box TB5 cells (`scripts/i1146_width_2box.sh`:
amo head ⇄ bmo tail, 128 vs 512, ABAB) on a quiet two-box window. Both boxes were held by the
sibling perf lane when this record was written.

## Provenance

- Cells: `.benchmarks/i1146_width/20261028T160722Z/` (session.log, provenance.txt, per-cell head logs).
- Scripts: `scripts/i1146_width_recurve.sh` (1-box), `scripts/i1146_width_2box.sh` (2-box, pending).
