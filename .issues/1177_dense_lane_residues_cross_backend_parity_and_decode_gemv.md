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

### Item 2 price cell — FIRST READ + verdict round 1 REVISE (2026-11-05, zed-83e79a9e@amo)

**Status: PARKED pending attribution — the measurements below are real
but the code-vs-box split, the ceiling, and correctness proof are OWED
(Claude verdict round 1 REVISE, session 10bde1fe). Neither "~1.3×" nor
"79% of bound" is claimed.**

Cell ran on amo (M5 Ultra 96 GB, quiet, AC, exclusive GPU, co-tenant
guard passed), bench `qwen38_wgpu_bench --decode 32 --decode-ctx 2048`,
HEAD `d85ee80`+, 2 process runs per arm:

| cell | A: dense-f16 league default (`gemv_dense_f16`) | B: packed incumbent (`DENSE_MM2D=0 ATTN_TENSOR=0 F16ACT=0`) |
|---|---|---|
| raw decode 32-step | 11.67 / 11.67 tok/s | 8.76 / 8.82 tok/s |
| decode-ctx 2048 | 10.02 / 10.03 tok/s | **5.00 / 1.23 tok/s (UNADJUDICATED, n=2)** |
| load | 33.6 s (54 GB f16 materialization) | 9.9–11.1 s (16.8 GB packed) |

Witnesses: `[Issue 1150] dense-f16 matmul2d arm armed` + `[Issue 1170]
half-B armed` + `Q6_K packed tail armed (~2.1 GB resident)` (A) /
`[Issue 1176] DEMOTED` (B). Full logs `/tmp/i1177_price/` (transient).

**Corrections the verdict AGREED are recordable now:**
1. **The i1150 record's "~52 tok/s incumbent (issue-1153 cells)" is a
   mis-citation**: issue 1153 = Plan 623 Phase 3 (Bench 929) = the
   Bonsai ternary 2-box DIST decode row ("ours 2-box s1 52.28–52.59
   tok/s BEATS llama.cpp-RPC 2-box 48.6", amo⇄bmo TB wire, ~7 GB/token
   PQ2_0) — never the qwen3.8 packed single-box Metal decode (16.8
   GB/token). This cell's packed control reads 8.8 raw on amo.
2. **"The f16 posture must never serve decode" is stale post-1176**: the
   league default IS the dense posture (the demote interlocks prove it —
   `DENSE_MM2D=0` alone refuses: ATTN_TENSOR and F16ACT host on the
   dense context), so `gemv_dense_f16` decode is the serving Metal
   decode posture at HEAD.
3. **The 1.86/1.23 baseline is in doubt**: the decode path changed
   post-record (`27edb75`/`5ab9a51` 11-04 — the lm_head 10.7 GB f32
   tail → Q6_K packed GEMV ~2.1 GB + the `forward_token` decode loop;
   `d19223c` 11-05 made lmhead+f16act league default), and `5923713`
   resolved the 10-29 bimodality as a POSTURE split — the 1.86/1.23
   pair may not even be one posture. Co-tenancy cannot yet be credited
   with the 6×.

**OWED before any verdict or lever lane (verdict round 1 REVISE):**
- [ ] Attribution cell: Arm A with `RIIR_QWEN38_LMHEAD_Q6K=0`, then
  F16ACT also demoted (or the interlock refusals recorded) — splits the
  6× into code vs box. First attempt killed by a tool timeout mid-load
  (log stopped at "loaded in 39.1s", f32 tail adds ~5.5 s load).
- [ ] Measured ceiling: streaming-read probe or llama.cpp f16 decode on
  the same box (the ~810 GB/s is ASSUMED; the Q6K commit measured its
  tail GEMV at ~470 GB/s) + recomputed bytes/token for the CURRENT
  posture (f16 layers + 2.1 GB Q6K tail + KV at ctx 2048).
- [ ] Correctness proof: engagement trace that decode actually dispatches
  `gemv_dense_f16` + a greedy-walk digest vs the packed arm (this lane's
  own precedent: `27edb75`'s first run was a fast-but-wrong all-zero
  walk).
- [ ] Packed ctx-2048 instability: UNADJUDICATED at n=2 — 5+ fresh
  processes required before "bimodal" is claimed (the `5923713`
  precedent). Hypothesis named only: depth-coupled (each B run does
  ~4 min of slow packed prefill before the ctx-decode cell; raw packed
  cells are stable 8.76/8.82).
- The re-priced lever lane does NOT open until the attribution + a
  measured ceiling land.

## Provenance

- Parent lanes closed 2026-11-05: riir-infer Issue 1150 (the dense
  forward arc, benches i1150/938–946), riir-infer Issue 1170 (lever 1 =
  half-B, bench 941).
- Claude verdict round 1 AGREE (2026-11-05): one residue issue covering
  both, always cited as "riir-infer Issue 1177" (riir-ai allocated its
  own 1177 the same day — always disambiguate cross-repo).
