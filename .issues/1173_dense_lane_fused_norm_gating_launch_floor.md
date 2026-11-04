# Issue 1173 — dense qwen38 GDN lane: fuse the norm+gating family against the per-kernel launch floor

**Status:** OPEN — bench 943's named lever. Claimed 2026-11-04 (see Claimed line).
**Claimed:** zed-1ee7a33e@amo, 1793797957, expires 2026-11-06T13:12:37+00:00

## Problem

Bench 943 (the GPU-timestamp census) measured the GDN bucket's non-GEMM
share precisely: EVERY small kernel (norms, z-gating, beta/decay,
residuals, gating, expand) reads a size-independent **0.5–0.7 ms launch
floor** per section at pp5120 (the same kernels read 0.01–0.11 ms on an
idle queue at pp512 — the floor is launch/scheduling, not kernel work).
14 sections × ~0.6 ms × 48 GDN layers ≈ **403 ms/chunk** — the whole
non-GEMM share bench 942 attributed. The kernels themselves are
already adequate (µs-scale real work).

## The lever

Fewer launches, not faster kernels. The census gives the exact adjacency
to fuse (each fusion removes one launch floor AND one full
round-trip of the intermediate buffer):

- [ ] **F1 — ssm norm + z-gate** (`norm_ssm` 0.56 + `z_gate` 0.56 → one
  kernel): `RmsNormBatched(rec_b → rec_n_b)` writes, then
  `DeltanetZGating(rec_n_b ⊙ sigmoid(z_b))` reads it back. One fused
  kernel: normalize a row in registers/shared, gate with z, write once.
  Removes 1 launch + one [p × n_v × hd_v] round-trip.
- [ ] **F2 — input norm + f16 cast** (`input_norm` → `gemm_input` cast):
  the half-B posture's `CastF32ToF16` reads what the norm just wrote —
  emit BOTH the f32 residual-stream copy and the f16 twin in one kernel
  (or write only the twin when the f32 copy is dead — check readers per
  posture).
- [ ] **F3 — residual add + next cast** (`residual_gdn`/`mlp_residual` →
  the following `gemm_input` cast): same shape as F2 at the two residual
  sites.
- [ ] **F4 — beta/decay + expand** (adjacent elementwise on the same
  chunk rows): `DeltanetBetaDecayBatched` then
  `ExpandAndL2NormalizeHeadsBatched` — independent outputs today; fusing
  saves one floor, lower priority than F1–F3.

Each fusion: byte-identical G1 vs the unfused pair (same math, one pass),
a pp5120 census cell showing the two floors collapse to one, and the
league cell (pp5120 tok/s) with the pre-registered bar: **≥ +2% e2e per
landed fusion, GOAT-gated before any default flip.**

## Non-goals

- GEMM kernel work (bench 943: write-bandwidth-bound, floor-dominated —
  no arithmetic ceiling to chase).
- The scan kernel (bench 938 + 943: adequate; its floor shares F1's fate
  once the norm fuses).
- Decode-path launches (issue 1170 lever 2, non-serving priority).

## Provenance

Bench 943 (`.benchmarks/943_issue1172_gdn_bucket_gpu_timestamp_census.md`)
measured the floors; the pp512 idle-queue cell proves the kernels are
µs-scale. The instrument (issue 1172) is the acceptance gate: the fused
pair's two census rows must collapse to one.
