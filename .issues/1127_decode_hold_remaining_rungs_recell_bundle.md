# Issue 1127 — decode-hold escalation: the remaining rungs + the league re-cell bundle (carried out of the 1125/1126 closeout)

**Status:** Open — filed 2026-10-26 from the 1125/1126 closeout (the Bench-1004 escalation ladder accounting). Rung 1 (fused-quantize) CLOSED negative-already-shipped (riir-infer HISTORY, census tripwire live); rung 2 (GDN core one-launch) CLOSED measured-negative (Bench 898, opt-in default-OFF); rung 2' (4→3 beta/decay+expand) LANDED direction-positive opt-in (Bench 899). **T1: rung LANDED opt-in +0.77% (Bench 900) — both the census and the e2e records kept. T2 CLOSED census-negative (Bench 902). T3 CLOSED measured-negative at the bit-identical boundary (Bench 903: the reclaimable share under the FNV-pin law is ~0.29% e2e; the headroom behind a re-pin is an owner decision).** Remaining: the owner-gated re-cell T4 only — the staged arms (T1 +0.77%, 2' +0.6–1.0%, GLU pp) ride the next league window. All three T1–T3 leases released; no session holds this issue.
**Lease:** none — released 2026-10-27 (T1/T2/T3 all closed with records; the T4 re-cell is owner-gated, never a session claim)
Historical note: the T1 e2e rung was executed in parallel by the shikuwa 4090 lane (Bench 900, `44695f1`+`cf89fb2`) while the amo census lane (Bench 902) ran — the reconciliation (both records kept; the e2e instrument decides e2e claims; in-graph node dispatch is NOT ns-class) lives in Bench 900/902. The Bench-900 nvcc contraction-pairing finding was isolated and reproduced by the Bench-903 V4 bisect arm (the reg-tile, common-mode ±1-ULP).

## Where the ladder stands (the accounting, all measured)

riir-ai Bench 1004 (2026-10-26): the decode hold broke to **0.9730×** (ours 91.68 / opp 94.16) — the opponent's +3.84% rode prism b10819→b10826 (#395 fused norm+FWHT+q8_1, #402 o_proj/ffn_down Hadamard⊗q8_1, #398 GDN conv-chain fusion, #396 wider FWHT decode blocks — default-on, PQ2_0-exact).

| rung | verdict | record |
|---|---|---|
| 1. fused quantize-into-rotation (#395 twin) | NEGATIVE already-shipped — the K1–K5 rotation kernels ARE it (Bench-940, default-on 2026-09-19); census `RIIR_QUANTIZE_CENSUS=1` = 0 on the 4090 | 1125 (closed, HISTORY), falsifier `a2ebb37` |
| 2. GDN core one-launch (beta/decay + expand/L2 + recurrence) | MEASURED NEGATIVE −3.5%/−2.1% — fused launch-bound neighbors INTO a compute-bound kernel; opt-in `RIIR_GDN_CORE_FUSED=1` default-OFF | Bench 898, `4132d94` |
| 2'. 4→3 beta/decay + expand/L2 only | DIRECTION-POSITIVE 3/3 paired windows (~+0.6–1.0% e2e, −1.5% GPU busy), G1 exact ×6 windows; opt-in `RIIR_GDN_BETA_EXPAND_FUSED=1`, NOT promoted | Bench 899, `8aabe55` |
| 2''. conv1d + beta/decay + expand (3→2, the Issue-1133 rung) | MEASURED-NULL 2/4 pairs (max +0.48%, pooled ≈0%), G1 exact everywhere — the dispatch saving is offset by the source-block occupancy loss; opt-in `RIIR_GDN_CONV_MERGE=1` default-OFF, no T4 row | Bench 905 |
| prefill GLU-fused MMA (sibling lane) | kernel probe 1.258×/1.227×, G1 PASS; e2e owed | Bench 897, issue 1124 T2 |

## Tasks

- [x] **T1 (attention-chain fusion census → rung) — BOTH records stand (Bench 902 census + Bench 900 rung).** Census (902): the chain (split_qg 0.93 µs + q/k-norm 1.60 µs ×2 + rope 1.63 µs + kv-append 0.89 µs) totals ≈ 107 µs/token ≈ **1.0% of the token's KERNEL-EXECUTION time** — that part of the record is correct and kept. Rung (900, `44695f1`+`cf89fb2`, landed opt-in): the 5→1 merge measured **+0.77% e2e median, 4/5 paired windows positive** (pairs +0.80/+1.15/−0.12/+1.02/+0.01; G1 walk/logits FNV exact in all 11 windows; kernel-level bit-identity at production/off-shape/full-rotation/KV_F16/devpos≡scalar) — the saving is the graph-NODE-DISPATCH term (~85 µs ≈ 64 nodes × ~1.3 µs) which the kern_sum census cannot see and whose ns-class premise the census ceiling assumed. The census' "≤0.3–0.5%" is a prediction about execution; the A/B is the measurement of e2e — they disagree and both are recorded (Issue-825: the disagreement was the oracle). Ships `RIIR_ATTN_CHAIN_FUSED=1` (default OFF; `=0` demote hatch in BOTH layer functions — the eager scalar path AND the graph-capture devpos path; the first armed window no-op'd on scalar-only wiring and the REACH-absence caught it). Also in 900: nvcc fma-contraction pairing flips across program shapes even for identical expression text — the rope forms are empirical per slot, the bit-identity test is the guard.
- [x] **T2 (embedding-dequant + inverse-FWHT fusion).** CLOSED census-NEGATIVE (Bench 902): the pair is 1.23 + 4.08 = 5.3 µs/token (0.05%) — nothing.
- [x] **T3 (FWHT decode geometry — the #396 analogue). CLOSED measured-negative at the bit-identical boundary (Bench 903, 2026-10-27).** The probe priced the value-preserving fix space: V1 float4-fill bit-clean ×3 shapes, the register-tiled butterfly stages 2+4 DRIFTS (V4 isolated it: 0 i8 + ascale-only = common-mode ±1-ULP — the Bench-900 nvcc contraction-pairing class, reproduced in isolation; dead by the FNV-pin law), the shfl_down reduction tail bit-clean (V3, +6.7% on rmsnorm alone). Shipping-ratio projection: 788 → 756 µs/token = **0.29% e2e** — an order under the +2.71% re-cross bar. The census's 8–95× roofline headroom is real but sits behind the bit-identity wall; reclaiming it is a re-pin (owner) decision, not a modelless rung. V3 was NOT wired — the probe is the record.
- [-] **T4 (league re-cell bundle — owner-gated).** The ONE window that decides every staged promotion, riding the riir-ai league runner (`scripts/perf_rematch.sh`, Bonsai-2 through riir-engine, verified-idle protocol, PROVENANCE quoted, ×2 windows where a promotion read is claimed):
  - `RIIR_PREFILL_FFN_GLU=1` — issue 1124 T2 e2e (pp512/pp2048/pp4096 A/B; engagement proof `RIIR_PREFILL_CUDA_FFN_TRACE=1` prints `[721-trace] route=glu`; armed-state VRAM +44.6 MB to measure — a promotion stops building the pair mirrors, halving it);
  - `RIIR_GDN_BETA_EXPAND_FUSED=1` — the Bench-899 rung's promotion read;
  - `RIIR_ATTN_CHAIN_FUSED=1` — the Bench-900 rung's promotion read (T1, 2026-10-27);
  - the decode hold re-crossing 1.0× itself (Bench 1004 RED);
  - the census line (`RIIR_QUANTIZE_CENSUS=1`) prints free at every scored cell — the zero re-checked each bout.
  Demote the loser either way (the incumbent chains stay as `=0` hatches). ⛔ Owner-gated per the standing handoff law — a measured-positive opt-in never self-promotes.
  **BOUT EXECUTED 2026-10-27 (Bench 1008, riir-ai `e3ffd3004`; riir-infer `c60b34d`):** trigger (3) FIRED (rung results exist; no new scored tag — b10826 newest, `c96baf4abe` #390 untagged/not scored); Claude verdict AGREE (fallback reviewer) → measurement-only, no session claim, incumbent-first. RESULTS: decode arms (T1+2') paired ×3 windows **+1.09/+0.84/+0.005% — direction-positive SUB-BAR**, hold moves 0.9735 → ~0.978 STILL RED (as disclosed up front); **GLU pp after the T1.5 vehicle-gap wiring (`c60b34d` — the arm was wired only into the !folded `dispatch_ffn_block`, the folded league lane had NO v10tglu path; the staged A/B would have been pair-vs-pair): pp2048 paired +6.31/+6.72%, pp4096 +7.27/+6.47%, G1 pins EXACT 4/4, `[1127-glu] REACH` every armed pass — armed league rows 1.28×/1.23×, the widest prefill holds on record. Incumbent matrix re-emitted (decode 0.978 / pp2048 1.202 / pp4096 1.152); census zero re-checked. **The promotion asks ride the owner: (a) `RIIR_PREFILL_FFN_GLU` → default-on (+6.5% prefill, bit-identical, modelless); (b) `RIIR_ATTN_CHAIN_FUSED`+`RIIR_GDN_BETA_EXPAND_FUSED` → default-on (+0.84% pooled, sub-bar — the Bench-899 precedent says opt-in); (c) the decode re-cross levers unchanged (owner re-pin or the next scored tag).**
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
