# Issue 1168 — decode hold triage: the ffn_gate_up kernel lane is measured-closed; the per-token host-sync serialization is the measured deficit

**Status:** OPEN — triage + probe filed; claim below.

## Summary

Bench 1161 (riir-ai) named `ffn_gate_up` (4.31 ms/token, 28.1% share) as "the decode
defense's named next lever". This issue triages that lever against the measured
record and finds the **kernel-variant form of the lever is already closed**, while a
**different, larger, measured class sits beside it**: the per-token host-sync
serialization in the decode loop.

## The kernel-variant lane — closed by prior art (do NOT re-open without new mechanism)

| closure | record | what it measured |
|---|---|---|
| CubeCL variants FLAT | 1137/1140/1141 (`6277cd2`, bench_1141) | FMA/fork algebra loses ~3× on gate/up (3.28×/2.97×/2.98×); with 1140's cold-rotation method the variant dimension reads variant-flat — **shipping f16scale rowtiled8 is the argmin of the measured family** |
| NOT payload-BW-bound | 1140 | the FFN GEMV stage time does not behave as a payload-bandwidth bound on M5 Ultra — "push closer to DRAM roof" is not an executable rung |
| Raw-MSL field kernel | 1138 → bench_1148 (kernel win, `c429d12`) → Bench 1149 (riir-ai, e2e NO-GO) | the hand-MSL lane wins EVERY aspect at kernel level (byte-field tq2 layout, float4 y, coef/sumy hoist) but loses e2e on the shared in-order queue (T4 flush-first law; max sound batch = ONE kernel) |
| WG sweep on these shapes | riir-ai Bench 996 / Issue 1115 (default-on) | m≥8192 → wg 128 already armed for gate_up m=34816; lm_head flat |
| Launch floor | riir-ai Bench 1008 | gate_up 65 µs/launch classified bandwidth(big-m), own lane, the 1132 non-goal |

A new CubeCL kernel rung for gate_up re-opens a measured-closed lane. Any re-open
needs a NEW mechanism (not a geometry/load-shape variant — that family is
variant-flat), with the 1140 cold-rotation A/B method and preregistered bars.

## The open class — per-token host-sync serialization (~3.1 ms/token, measured twice)

From the Bench 1161 window's own bench_763 census log
(`riir-ai/.benchmarks/rematch/logs/2026-11-03_bench763_census.log`), no new GPU time
required:

- **Phase B** (dispatch-only, one sync per 20 tokens): wall 14.03 + drain 1.32 =
  **15.34 ms/token combined = the GPU-bound floor** → 65.2 tok/s.
- **Phase A** (per-token `read_one` logits sync, no sampling): **18.48 ms/token**.
- The league ours rows (53.46–56.80 tok/s = 17.6–18.7 ms/token) sit at the **Phase A**
  class, not the Phase B floor.
- Same-window opponent: 63.52–63.93 tok/s. **At the Phase B floor (65.2 tok/s) the
  decode hold reads ~0.98× — CLOSED — with zero kernel changes.**

So the ~3.14 ms/token delta between Phase A and Phase B — `read_one`(993 KB logits)
per token: GPU-tail wait + copy + allocation + submission/callback overhead — is the
same size as the entire remaining decode deficit. The prior sync-cost record
(riir-ai Bench 606) formalized round-trip vs amortized measurement but predates the
current pipeline; no record decomposes the current cost.

Floor arithmetic for the fix space: the token chain is serial through sampling
(token i+1's input IS token i's argmax), so per-token logits are needed; the
attackable mass is the **cost of the sync itself**, not its existence:

1. **Cheap rung (probe-gated):** if the fixed sync/submission cost is small and the
   copy dominates → GPU-side argmax/top-1 summary kernel (lm_head epilogue writes a
   4 B summary), read 4 B instead of 993 KB; full logits stay on GPU for the rare
   non-greedy path.
2. **Arc rung (604 lineage):** if the fixed cost dominates → K-deferred host sync
   (GPU-resident greedy decode: argmax + embed-fetch on GPU, host harvests every K
   tokens). Issue 604 attempted the GPU-resident forward and was BLOCKED on the
   CubeCL handle-churn bug family (riir-ai Issue 994 pool poisoning; `pool_poison.rs`
   + `refuse_if_pool_poisoned` now guard it; cubecl is VENDORED at
   `vendor/cubecl-runtime-0.11.0-pre.2` — the sync path itself is a legal patch
   surface). CubeCL now exposes `read_async`/`read_one_tensor_async` — the
   async-readback seam exists regardless.

## Preregistered plan

- **T1 (probe, this issue):** `issue1168_readback_floor_probe.rs` (riir-infer-gpu
  tests, `#[ignore]`, GPU): median `read_one` cost for (a) 4 B, (b) 993 KB, on a
  drained queue; (c) after a tiny GEMV dispatch (tail+copy decomposition); (d) N-deep
  queue + single read (the Phase B amortization reference). Decides rung 1 vs rung 2
  by measurement, not prose.
- **T2 (cheap rung, if T1 says copy-dominated):** GPU argmax summary in the lm_head
  epilogue, opt-in `RIIR_DECODE_LOGITS_SUMMARY=1` (PROMOTE-ONLY, exact-literal arms;
  demote = incumbent byte-identical path). G1: sampled tokens bit-identical (FNV walk
  EXACT on bench_763-style runs); G2: e2e decode ≥ +5% at ctx-class on m5ultra; G4:
  no new steady-state allocations on the hot path.
- **T3 (arc, if T1 says sync-dominated):** K-deferred sync plan (separate `.plans/`
  file before implementation; rides the vendored-cubecl patch surface only with a
  verdict — owner-gated by the source-secrecy posture: patches stay IN-REPO, never
  published).
- **Bars for any promotion:** GOAT gates + reviewer/owner verdict per the env-switch
  law; the league decode rows re-cell (bench_768/831 tg/ctx classes) before any
  default flip; the FNV walk pins must stay EXACT.

## Boundary note

The forward driver (`TernaryDeltanetGpuForward`, `forward_token`/`read_logits`) is
this repo's; the harness that measured the gap (bench_763) and the league lane are
riir-ai's. The probe test lives here (pure cubecl_runtime, no model load). Test-file
precedent for cross-repo issue names: `crates/riir-infer-gpu/tests/issue1115_*`.

## Claim

**Claimed:** zed-4a8f7b63@amo (amo), 1793680541, expires 2026-11-05T04:35Z.
