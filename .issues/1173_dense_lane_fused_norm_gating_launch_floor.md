# Issue 1173 — dense qwen38 GDN lane: fuse the norm+gating family against the per-kernel launch floor

**Status:** CLOSED — GOAT FAIL, recorded honestly (bench 944). The
lever was executed end-to-end, proven bit-identical, measured sub-bar,
and NOT promoted. The arms stay opt-in for replication.
**Lease closed:** zed-1ee7a33e@amo, epoch 1793797957, claimed
2026-11-04, expired 2026-11-06T13:12:37+00:00 (folded at close).

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

- [x] **F1 — ssm norm + z-gate** (`norm_ssm` 0.56 + `z_gate` 0.56 → one
  kernel): `RmsNormBatched(rec_b → rec_n_b)` writes, then
  `DeltanetZGating(rec_n_b ⊙ sigmoid(z_b))` reads it back. One fused
  kernel: normalize a row in registers/shared, gate with z, write once.
  Removes 1 launch + one [p × n_v × hd_v] round-trip.
- [x] **F2 — input norm + f16 cast** (`input_norm` → `gemm_input` cast):
  the half-B posture's `CastF32ToF16` reads what the norm just wrote —
  emit BOTH the f32 residual-stream copy and the f16 twin in one kernel
  (or write only the twin when the f32 copy is dead — check readers per
  posture).
- [x] **F3 — residual add + next cast** (`residual_gdn`/`mlp_residual` →
  the following `gemm_input` cast): same shape as F2 at the two residual
  sites.
- [-] **F4 — beta/decay + expand** — DEFERRED WITH A CORRECTED PREMISE:
  in the actual dense-prefill dataflow they are NOT adjacent —
  `DeltanetChunkedConv1d` runs between them (beta/decay → conv1d →
  expand), and expand reads `conv_out_b`, not the beta/decay outputs.
  Merging them would change the dataflow, not just the launch count.
  Bench 944 further shows the launch-count lever itself is sub-bar, so
  there is no case to force this fusion.

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

## What landed (commits `5bb84f1` + `2647838` + this close)

- `crates/riir-infer-gpu/src/norms_fused_batched_cubecl.rs` — three
  batched fused kernels + launchers + three promote-only env arms
  (`RIIR_QWEN38_FUSE_SSM_ZGATE` / `_NORM_CAST` / `_RESID_NORM`, exact
  literal `1`; unset = incumbent launch sequence byte-identical). F3
  includes the cross-layer arm (mlp_residual fuses with the NEXT
  layer's input_norm; threaded via the ingest loop).
- `crates/riir-infer-gpu/tests/qwen38_fused_norm_gating_g1.rs` — the
  byte-identity G1: GPU-vs-GPU BITWISE vs the unfused sequences on f32
  + f16 twin + the twin-less posture, 4 shapes each. 3/3 green.
- Wiring in `qwen38_dense_cubecl.rs` (the F2/F3 helpers +
  `gemm_input_fresh` so fused producers' twins never re-cast).
- Bench 944 (`.benchmarks/944_issue1173_fused_norm_gating_ab.md`) —
  the A/B record + the verdict.

## The verdict (bench 944; Claude AGREE-with-revisions 2026-11-04, applied)

- **GOAT FAIL**: steady-pass medians pp5120 — baseline ~971, F1 ~976.5,
  F1+F2 ~981, F1+F2+F3 ~978.5 tok/s. Every per-fusion delta (+0.2–0.6%)
  is within the ~±0.5% run-to-run spread; the family is ≤ ~1% (arm
  ordering not stable). **No measurable per-fusion gain; never default.**
- **Byte-identity held everywhere**: G1 bitwise (3/3) + e2e greedy walks
  digit-exact on BOTH the sequential path (48/48 ids) and the batched
  chunked path (16/16 ids, ~600-token prompt forcing full-chunk ingest).
- **Mechanism finding**: the census's per-section walls do NOT collapse
  under fusion (norm_ssm+z_gate read 1.12 ms fused = 1.12 unfused — the
  fused wall relocated into the following marker's hazard wait). 943's
  "launch floor" is marker-boundary instrumentation cost; free-running
  marginal launch cost ≈ 0.02–0.05 ms/launch. Annotated in place at
  benches 942/943 (append-only dated rows).
- **Instrument law landed en route**: a ZERO-LENGTH buffer binding is
  clamped to the WHOLE buffer by the wgpu/Metal binding layer — the twin
  slot binds a one-element γ view and the twin write is gated by an
  explicit `params[3]` flag, never the slot length (pinned by the G1's
  twin-less arms; the launcher comments carry the law).
- **F4 premise corrected** (deferred): beta/decay and expand are not
  adjacent — conv1d intervenes; see the lever list above.

## Follow-up issues filed (the verdict's list)

- **1174** — the non-serialized attribution instrument (required before
  the next lever from the 942/943 tables).
- **1175** — the zero-length-binding launcher audit
  (`RmsNormQkFusedCubeCL` is safe by accident).
- **1176** — the bare-default ~7 tok/s promotion-debt defect.

## Provenance

Bench 943 (`.benchmarks/943_issue1172_gdn_bucket_gpu_timestamp_census.md`)
measured the floors; the pp512 idle-queue cell proves the kernels are
µs-scale. The instrument (issue 1172) was the acceptance gate — and the
acceptance cell (row collapse) proved unachievable, which is itself the
finding: the walls belong to the instrument, not the launches (bench 944
§the mechanism finding).
