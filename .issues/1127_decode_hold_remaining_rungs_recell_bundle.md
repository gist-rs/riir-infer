# Issue 1127 — decode-hold escalation: the remaining rungs + the league re-cell bundle (carried out of the 1125/1126 closeout)

**Status:** Open — filed 2026-10-26 from the 1125/1126 closeout (the Bench-1004 escalation ladder accounting). Rung 1 (fused-quantize) CLOSED negative-already-shipped (riir-infer HISTORY, census tripwire live); rung 2 (GDN core one-launch) CLOSED measured-negative (Bench 898, opt-in default-OFF); rung 2' (4→3 beta/decay+expand) LANDED direction-positive opt-in (Bench 899). **T1 CLOSED census-negative, T2 CLOSED census-negative, T3 PROMOTED to the active rung (Bench 902, 2026-10-26: the sign_fwht_quantize family is 7.2% of the token at 8–95× over roofline — the only re-cross-class lever measured).** Remaining: the T3 rung (implementation) + the owner-gated re-cell T4. T3 is claimed and in flight (lease below); no second session on it.
**Claimed:** zed-amo-i1127-t1, 1793129600, expires 2026-10-28T17:00:00+00:00

## Where the ladder stands (the accounting, all measured)

riir-ai Bench 1004 (2026-10-26): the decode hold broke to **0.9730×** (ours 91.68 / opp 94.16) — the opponent's +3.84% rode prism b10819→b10826 (#395 fused norm+FWHT+q8_1, #402 o_proj/ffn_down Hadamard⊗q8_1, #398 GDN conv-chain fusion, #396 wider FWHT decode blocks — default-on, PQ2_0-exact).

| rung | verdict | record |
|---|---|---|
| 1. fused quantize-into-rotation (#395 twin) | NEGATIVE already-shipped — the K1–K5 rotation kernels ARE it (Bench-940, default-on 2026-09-19); census `RIIR_QUANTIZE_CENSUS=1` = 0 on the 4090 | 1125 (closed, HISTORY), falsifier `a2ebb37` |
| 2. GDN core one-launch (beta/decay + expand/L2 + recurrence) | MEASURED NEGATIVE −3.5%/−2.1% — fused launch-bound neighbors INTO a compute-bound kernel; opt-in `RIIR_GDN_CORE_FUSED=1` default-OFF | Bench 898, `4132d94` |
| 2'. 4→3 beta/decay + expand/L2 only | DIRECTION-POSITIVE 3/3 paired windows (~+0.6–1.0% e2e, −1.5% GPU busy), G1 exact ×6 windows; opt-in `RIIR_GDN_BETA_EXPAND_FUSED=1`, NOT promoted | Bench 899, `8aabe55` |
| prefill GLU-fused MMA (sibling lane) | kernel probe 1.258×/1.227×, G1 PASS; e2e owed | Bench 897, issue 1124 T2 |

## Tasks

- [x] **T1 (attention-chain fusion census → rung).** CLOSED census-NEGATIVE (Bench 902): the chain (split_qg 0.93 µs + q/k-norm 1.60 µs ×2 + rope 1.63 µs + kv-append 0.89 µs) totals ≈ 6.7 µs per attention layer × 16 layers ≈ 107 µs/token ≈ **1.0% of the token** — a perfect 5→1 fusion recovers ≤ 0.3–0.5% e2e, an order under the +2.71% re-cross bar. The Bench-898 gate reads NO at the census; no rung warranted.
- [x] **T2 (embedding-dequant + inverse-FWHT fusion).** CLOSED census-NEGATIVE (Bench 902): the pair is 1.23 + 4.08 = 5.3 µs/token (0.05%) — nothing.
- [-] **T3 (FWHT decode geometry — the #396 analogue). PROMOTED TO THE ACTIVE RUNG by the Bench-902 census:** the `sign_fwht_quantize` fused family (rmsnorm 408 µs + swiglu 303 µs + gate_sigmoid 77 µs ≈ **788 µs/token ≈ 7.2% of the token**) runs at **8–95× above its traffic roofline** with TIGHT per-launch floors (rmsnorm min 4.77 / med 4.90 µs) = latency/barrier-bound at the 1024-block FWHT geometry, not bandwidth- and not launch-bound (in-graph node scheduling is ns-class). Headroom ≥ 3.5% e2e at a 2× improvement — above the +2.71% bar. The rung: restructure the geometry toward roofline (multi-block reduction à la the Issue-734-T5 `rmsnorm_quantize_mb` leaf-partial pattern, or wider per-block thread geometry), bit-identical reduction order (Issue-940 law), G1 walk/logits FNV EXACT, paired-interleaved A/B, opt-in env, demote hatch. The census already answers the issue's measure-first question: occupancy-bound, NOT launch-bound — a wider block alone is the WRONG direction; parallelism per norm-reduction is the right one.
- [-] **T4 (league re-cell bundle — owner-gated).** The ONE window that decides every staged promotion, riding the riir-ai league runner (`scripts/perf_rematch.sh`, Bonsai-2 through riir-engine, verified-idle protocol, PROVENANCE quoted, ×2 windows where a promotion read is claimed):
  - `RIIR_PREFILL_FFN_GLU=1` — issue 1124 T2 e2e (pp512/pp2048/pp4096 A/B; engagement proof `RIIR_PREFILL_CUDA_FFN_TRACE=1` prints `[721-trace] route=glu`; armed-state VRAM +44.6 MB to measure — a promotion stops building the pair mirrors, halving it);
  - `RIIR_GDN_BETA_EXPAND_FUSED=1` — the Bench-899 rung's promotion read;
  - the decode hold re-crossing 1.0× itself (Bench 1004 RED);
  - the census line (`RIIR_QUANTIZE_CENSUS=1`) prints free at every scored cell — the zero re-checked each bout.
  Demote the loser either way (the incumbent chains stay as `=0` hatches). ⛔ Owner-gated per the standing handoff law — a measured-positive opt-in never self-promotes.
  **Window verdict 2026-10-26 (reviewer sub-agent `e510e3e2`; the claude_code
  verdict channel was auth-unavailable — the Issue-1115-arc fallback):
  WAIT_NEXT_WINDOW, no ad-hoc bout.** Verified first-hand: the 4090 idle (4%
  util / 506 MiB) BUT the bundle cannot re-cross the RED cell this window —
  the re-cross needs +2.71% (91.68 → 94.16) vs the staged arms' +0.6–1.0%
  decode class (GLU is prefill-side, decode runs the GEMV path; its pp rows
  already PASS at 1.1907×/1.0546×); the real re-cross levers are T1–T3 (all
  unimplemented); batching is the league's recorded law (opponent tag
  `c96baf4abe` #390 pending; tags have fired ~daily); and Bench 1004's own
  voided pp4096 cell is the demonstrated cost of squeezing a busy day. The
  window fires on the next opponent scored tag, the next weekly loop cycle,
  or as soon as T1–T3 rung results exist to ride the same window — whichever
  comes first.
- [-] **T5 (Metal-lane int8 consumer — DEFERRED, owned elsewhere).** The m5ultra decode rows run Metal/CubeCL, which has NO int8 consumer at all (`GemvTernaryCubeCL` consumes f32 rows) — the whole fused-rotation+quantize+dp4a posture is CUDA-lane-only, making the Metal int8 consumer question (NEW substrate, not a launch elimination) the real m5ultra decode lever. Recorded as the riir-ai Issue 1113 defense follow-up — do not duplicate here; pick it up in that lane.

## G1 pins (every rung in this issue)

Real-model argmax-walk FNV `98412dc406c4ea19` + logits[::97] FNV `63496f6688c8f742` EXACT at the manifest posture (`B734_ROUNDS=3`). ⚠ The walk FNV is rounds-cumulative: a `B734_ROUNDS=1` run prints `a287e92e8d3d1021` — a different hash is the posture, not drift.

## Non-goals / scope guards

- NOT the lossy state-half successor (owner-gated, Plan 611 / Bench 954) — every rung here is modelless and bit-identical.
- NOT conv1d fusion: true data dependency (in-place conv must complete before any qkv consumer; same-kernel fusion needs a grid barrier) — refused at Bench 898.
- NOT the opponent's #397 multi-column widening — Hopper-only (`pq2_mc_min = HOPPER ? 1 : 3`), sm_89-inert.
- NOT the prefill lane's own remaining work — issue 1124 owns it; only its e2e cell rides T4's window.

**Refs:** riir-ai Bench 1004 (the RED); riir-infer Bench 898 + 899 (the rung records); Bench 897 + issue 1124 (the prefill sibling); 1125/1126 closure records in `../HISTORY.md`; prism #395/#396/#398/#402; riir-refine `.distill/001` (the watch handoff row); riir-ai Issue 1113 (the Metal-lane defense follow-up).
