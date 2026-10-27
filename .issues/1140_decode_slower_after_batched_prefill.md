# Issue 1140 — decode tok/s reads lower after a batched (`ternary_gemm_batched`) prefill than after a per-position prefill

**Status:** Open — observation, order-confounded; needs an interleaved A/B before any mechanism is claimed. Filed 2026-10-27 from Plan 623 T2.3b. Unclaimed.

## Observation

Every Plan 623 T2.3b cell read decode lower after a batched prefill than after
a per-position one (Bonsai-2 27B PQ2_0, Metal/CubeCL, `--release`):

| cell | after per-position | after batched | Δ |
|---|---|---|---|
| bmo 1-box, P=256, chunk 256, 47 steps, idle (load 1.0–1.7) | 25.80 / 25.62 tok/s | 24.74 / 24.69 | −4% |
| bmo 1-box, P=200, chunk 100, 15 steps, idle | 26.07 | 24.40 | −6% |
| real 2-box amo⇄bmo TB5, P=256, ABBA order, amo load ~3.8 | 19.66 / 21.33 (chunk 1) | 18.80 / 17.58 (64) · 16.38 / 16.61 (128) · 15.93 / 16.50 (256) | −10..−24%, monotone in chunk |

Instrument: `crates/riir-infer-dist/tests/two_stage_prefill_gpu.rs` (the
`decode after:` line) and `dist_stage --prefill-chunk N` (2-box).

## Why it matters

The production 1-box `TernaryDeltanetGpuForward::prefill` IS the batched
body (the T2.3b anchor proves the stage path bit-identical to it), so if
real, every served prompt that prefills batched decodes a few % slower
afterwards, a cost the Issue-1139 decode scoreboard never sees: its cells
decode from a per-token start.

## Confounds (why this is not yet a finding)

- The 1-box cells are NOT interleaved: the batched arm always runs first in
  the process (a fresh stage after the anchor), the per-position arm second.
- The 2-box cells are ABBA-ordered but amo was a shared, loaded box.
- The decode loops are short (15–47 steps).

## Tasks

- [ ] T1 — interleaved A/B on idle bmo: same process, fresh forwarder per
  arm, alternating order, ≥128 decode steps per arm, box state per the G2 law.
- [ ] T2 — if real: attribute. Suspects, cheapest first: the P-width prefill
  scratch left in the CubeCL memory pool (does `client.memory_cleanup()` after
  prefill restore decode?); the attention-arm / `active_prefill_len` state a
  prefill leaves set; KV-cache residency.
- [ ] T3 — route the fix to the decode campaign's rung set (Issue 1127 family)
  or close as a measured null.
