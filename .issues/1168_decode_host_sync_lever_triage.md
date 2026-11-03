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

- **T1 (probe, DONE 2026-11-03):** `issue1168_readback_floor_probe.rs` (this repo,
  `#[ignore]`, bare client, no model) + `issue1168_decode_loop_decomposition.rs`
  (riir-ai, in-situ, real model). **Results — the T2 band is REFUTED, T3 is the
  arc:**

  | probe | ms |
  |---|---:|
  | (a) read 4 B drained | 0.173 |
  | (b) read 993 KB drained | 0.167 (bytes ≈ free) |
  | (c) 1 tiny GEMV → read | 0.246 |
  | P1 dispatch front (sync-per-token) | 2.455/tok |
  | P1 read (blocks for the whole GPU token) | 15.328/tok |
  | **P1 total (league shape)** | **17.787/tok** |
  | **P2 floor (deep queue)** | **15.315/tok = 65.30 tok/s** |
  | P3 sync every 2nd token | 16.605/tok (≈ floor + half the gap — per-sync drain-refill) |

  **Read of the parts:** the readback itself is CHEAP (0.17–0.25 ms — T2's
  summary-read rung has no mass to recover). The league-shaped loop pays
  **enqueue-front + GPU-work serialized** because the per-token sync drains the
  queue: CPU enqueues (2.455 ms) on an idle GPU, then blocks in `read_one` until
  that token's full GPU work (~15.3 ms) completes. The only VALUE-dependent CPU
  step is the input-embedding upload (set_input = 0.006 ms); every dispatch is
  shape-only. P3 confirms the mass scales with sync frequency.
- **T2 (summary-read rung): REFUTED by T1 — closed with no implementation.** A
  4 B summary read saves ~nothing when the full read costs 0.17 ms.
- **T3 (the arc, OPEN): GPU-resident greedy decode (Issue 604 lineage, now
  unblocked in mechanism):** keep the queue deep ACROSS the token boundary by
  removing the per-token CPU value dependency —
  1. argmax-on-device: a small reduction kernel after lm_head writes the greedy
     token id (u32) DIRECTLY into a device-side id buffer;
  2. embed-gather kernel: input embedding = row gather (id → vocab×n_embd table
     row) on GPU — the CPU never sees logits during greedy decode;
  3. the CPU loop enqueues token i+1's full graph WHILE the GPU processes token
     i (all dispatches shape-only), harvesting output ids every K tokens (or on
     demand); sampled (non-greedy) paths keep the incumbent per-token sync lane;
  4. posture: opt-in env, PROMOTE-ONLY exact-literal arms, incumbent
     byte-identical when off; greedy-only contract documented loud.
  **Prize (preregistered):** P1→P2 = −2.47 ms/token harness-side; league
  ctx-class ≈ −2.5 ms/token ⇒ 53.46 → ~61.7 tok/s ⇒ ratio ~1.03× vs the
  Bench-1161 same-window opponent — **the decode hold closes on the pipeline,
  not on kernels.** G1: sampled tokens bit-identical (greedy argmax = CPU
  argmax of the same logits — verify FNV walk EXACT on the league cells);
  G2: the preregistered league re-cell; G4: no new steady-state hot-path
  allocations. The 604-era blocker (CubeCL handle churn, riir-ai Issue 994
  pool poisoning) is now guarded (`pool_poison.rs` + `refuse_if_pool_poisoned`)
  and cubecl is vendored (`vendor/cubecl-runtime-0.11.0-pre.2`) — patch surface
  exists if the async-read APIs misbehave; patches stay IN-REPO (source
  secrecy), any vendored-cubecl edit is owner-gated.
  **Implementation gate: a `.plans/` file FIRST (queue-depth laws, the K-harvest
  cadence, the non-greedy fallback, the KV/pos bookkeeping), then code.**

## Boundary note

The forward driver (`TernaryDeltanetGpuForward`, `forward_token`/`read_logits`) is
this repo's; the harness that measured the gap (bench_763) and the league lane are
riir-ai's. The probe test lives here (pure cubecl_runtime, no model load). Test-file
precedent for cross-repo issue names: `crates/riir-infer-gpu/tests/issue1115_*`.

## Claim

**Claimed:** zed-4a8f7b63@amo (amo), 1793680541, expires 2026-11-05T04:35Z
**Progress:** T1 + T1b DONE 2026-11-03; T3 arc IMPLEMENTATION LANDED 2026-11-03 (zed-5ae93349@amo, plan 626 T3/T3a): indirect dequant kernel + resident ring loop + argmax-last twin + env arm, hermetic G1 green (bit-identity 3x128), 11/11 sampling kernel tests green, compiles clippy-clean at the full Metal posture; the heavy 27B walk (`plan626_resident_greedy_g1`, --ignored) + league re-cell + promotion OPEN. Measured en route: the incumbent `ArgmaxCubeCL`'s first-index property is butterfly-tree-order luck on arbitrary cross-thread ties (documented in plan 626 T3a; pinned semantics unchanged).
**Bars for any promotion:** GOAT gates + reviewer/owner verdict per the env-switch
law; the league decode rows re-cell (bench_768/831 tg/ctx classes) before any
default flip; the FNV walk pins must stay EXACT.
**Probe box states (DIAGNOSTIC, not published latency):** T1 ran on amo with a
sibling bisect's compile storm (load ~7-9/30) — the (a)/(b)/(c) medians separate
the classes by ≥10×, far above that noise; T1b ran in the same window (load
9.48/30 at print) — P1/P2/P3 separate by 1.5-2.5 ms/token, again ≫ compile noise;
a quiet-window re-run is the promotion-time measurement, not this triage.
