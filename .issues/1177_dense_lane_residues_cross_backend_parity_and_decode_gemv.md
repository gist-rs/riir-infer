# riir-infer Issue 1177 — dense-lane residues: the cross-backend bit-parity G1 cell (4090) + the parked dense-f16 decode GEMV lever

**Status:** Open — residue pointer filed 2026-11-05 when the parent lanes
closed (1150 + 1170; Claude verdict round 1 AGREE on the closure batch).
Two items the lane still owes, both priced/deferred, neither blocking a
landed surface.
**Claimed:** zed-83e79a9e@amo, 1793857012, expires 2026-11-07T12:36:52+00:00

Lease note: Item 2 price cell on amo Metal; Item 1 stays 4090-queue-gated.

## Item 1 — the cross-backend bit-parity G1 cell (from Issue 1150's acceptance gates)

Issue 1150's G1 gate: Metal forward vs `qwen38_dense_cudarc.rs` (the
~7k-line CUDA twin, the bit-parity oracle) on the SAME qwen3.8-27B
Q4_K_M GGUF — FNV/argmax pins over a fixed token walk, tolerance band
set BEFORE the run (f32 kernel-order classes get the certified-precedent
treatment, never post-hoc bands). **Never run**: the lane's identity
evidence to date is Metal-vs-Metal (benches 938–946 walks digit-exact
within one backend) + the league demote byte-identity (bench 946). The
4090 is the only cudarc host and was BUSY (3 contenders) at filing.
Execute when the 4090 queue frees: build the cudarc forward at the
league env set, run the fixed walk on both backends, register the pins
(FNV + argmax) in this issue. Refusals/classes: the f16-activation
rounding order differs by construction across backends — pre-register
the band from the f32-class kernel-order precedent
(`katgpt-rs closure/bridge_certified`), never post-hoc.

## Item 2 — the parked dense-f16 decode GEMV lever (from Issue 1170 lever 2)

Decode on the dense-f16 posture reads ~1.9–22 tok/s class vs a ~15 tok/s
bandwidth bound per the 1150 audit framing (54 GB f16 at ~800 GB/s); the
`gemv_dense_f16` kernel is one simdgroup-per-row with strided reads,
while the Q6K tail's packed GEMV (bench 939) already proved the
saturating pattern (per-lane partials + one reduction). **Explicitly
parked non-serving in 1170**: the packed posture serves decode; this
lever matters only if the dense posture ever becomes the serving decode
posture or for parity rows. Price before any kernel work: the expected
ceiling is bandwidth-class (~15 tok/s), so the lever buys ~7× on a
non-serving surface — lead with a price cell, never the kernel.

## Provenance

- Parent lanes closed 2026-11-05: riir-infer Issue 1150 (the dense
  forward arc, benches i1150/938–946), riir-infer Issue 1170 (lever 1 =
  half-B, bench 941).
- Claude verdict round 1 AGREE (2026-11-05): one residue issue covering
  both, always cited as "riir-infer Issue 1177" (riir-ai allocated its
  own 1177 the same day — always disambiguate cross-repo).
