# Bench 921 — Issue 1146 rung 2b H4: hoisted-load tensor-GEMM dequant (`SWEEP_DEQUANT_V2`) — +9% e2e, bit-exact, PROMOTED

**Status:** RECORD — PROMOTED default-ON (`RIIR_METAL_TENSOR_DEQUANT_V2=0` demotes). G1 exact (cold-pool kernel gate max_abs 0 + identical dist token streams); G2 **1.089× (pp512) / 1.092× (pp4096)** at w512, zero contaminated pairs.
**Date:** 2026-10-28 (G1 18:34–18:35, A/B 18:52–18:58 +07)
**Session:** katgpt-rs-71, 1793178290 (measurement + promotion). Kernel variant: katgpt-rs-31, riir-infer `cff3fda`. The source read that ranked it as one of only two kernel differences from llama.cpp: katgpt-rs-bd (riir-infer `ec19235`, `e31b828`).
**Issue:** [`.issues/1146_prefill_throughput_gap_1box.md`](../.issues/1146_prefill_throughput_gap_1box.md) — rung 2b
**Box:** bmo (Mac17,15, M5 Ultra, 96 GiB), AC
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` sha256 `3907dc16…62ec1` (league pin)

## The variant

This is the tensor (matmul2d, ktile64) GEMM's A-tile staging. The incumbent decoded each ternary weight per element: a word load, a scale load and a bounds check every time. `SWEEP_DEQUANT_V2` handles each 16-column work item with **one** pos/neg word load and **one** scale load. A 16-aligned run sits inside one u32 bit-plane word and one 128-column scale group. The bounds check is hoisted to a full-tile fast path, and stores are `half4`. The arithmetic per element is unchanged, `(float)pos − (float)neg` → `× scale` → `half`, so the staged tile is bit-identical by construction. There is no add, so the Metal fma-fusion hazard does not apply. Partial-row and partial-K tiles keep the incumbent scalar body.

## G1

- `issue1146_tensor_range_g1` under the variant (zerocopy arm FIRST, cold pool): `prefill()` / range-full / range-split all **max_abs 0.000**, P=512 w128 (argmax 13) and P=1024 w512 (argmax 261).
- Dist cross-arm: identical 16-token `generated:` streams at w512.
- The promoted default compiles the **identical MSL source string** as the measured `=1` arm (`compose_msl_source` now emits an explicit `#define SWEEP_DEQUANT_V2 0|1` on both branches, and the `.metal` fallback is 1).

## G2 — one binary, env-only A/B (`I1146W_ENV_B`), w512, per-pair contention guard (`c299397`), zero CONTAM

| cell | a (incumbent dequant) | b (v2) | b vs a |
|---|---:|---:|---:|
| pp512 r1 / r2 | 1.747 / 1.803 s | 1.635 / 1.624 s | **1.089×** |
| pp4096 r1 / r2 | 15.250 / 15.238 s | 13.957 / 13.960 s | **1.092×** |

The a arm reproduces Bench 920's tensor w512 values (1.751 / 15.24 s), so the box was clean. Reps agree within 0.1% on pp4096.

**Stacked vs the dist shipping posture before today** (w128, simdgroup GEMM, Bench 919/920 a-arm): pp512 3.041 → 1.630 s (**1.87×, 314 tok/s**), pp4096 25.63 → 13.96 s (**1.84×, 293 tok/s**). Against llama.cpp's 1087 / 1046 tok/s (Bench 911) that is **0.29× / 0.28×**, up from 0.13 / 0.16×.

## Discarded run (honesty)

The first dqv2 A/B (18:36) overlapped a riir-ai `bench_809` league row that started 30 s after the step-start quiet gate. Every later cell ran 4–10× slow (pp512 14–18 s). It was killed, kept at bmo `/tmp/i1146q/dqv2_contam_1836`, and not used. The repair is the width rig's per-pair contention guard (`c299397`), which this record was the first run under.

## Provenance

- Cells: `.benchmarks/i1146_dqv2/` (one binary `dist_v4` at `cff3fda`, sha `afe15466…`, `env_b: RIIR_METAL_TENSOR_DEQUANT_V2=1`).
- Promotion commit: this record's commit.
