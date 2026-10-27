# Issue 1136 — rung: gdn_out_gemv (the out-project GEMV class at the gdn chain's exit)

**Status:** CLOSED — ceiling real but G1 structurally dead on the named variant (2026-10-27 ~09:0x +07, 2 verified-idle interleaved windows on amo, launches bit-identical). Ceiling leg: gdn_out_gemv strip +0.524/+0.592 ms/tok (+2.16/+2.46%) @ 10.9/12.3 us x 48 la/tok — marginal over the pre-registered 2% G1-justification gate, census-coherent (W1 0.68 @ 14.2). G1 leg (the issue's ONE named variant, producer-epilogue into the recurrence/outproj-adjacent producer): STRUCTURALLY DEAD — the GEMV's input is the whole v_dim vector through zgate (in-place, multi-workgroup) then the permute-rotate chain (fused 1113 kernel or copy+permute+rotate); no workgroup owns the full row, the b02f379 no-owner class that killed i1134's producer-epilogue; deeper (recurrence-kernel epilogue) worse by the same wall. G1 landmine on any zgate-touching replumb: the Issue-1113 full-K3 variant measured 1-ULP-divergent (fma contraction not reproducible across kernel compilations) — the zgate is pinned bit-untouched. The consumer-prologue (redundant zgate+rot recompute per GEMV workgroup) is zed-m3's PARKED i1134 re-open lever (905 one-named-re-open discipline) — not re-litigated here. CHAIN BUDGET CLOSED BY MEASUREMENT: 905 NULL + i1131 ~0 + i1130 ~0 + i1134 negative + i1135 sub-bar + this (ceiling marginal, G1 dead). Accounting find for the parked lever's economics: the folded posture's whole gdn_outproj exit strips 240 la/tok (not the census Q2_0 posture's 96) at 5.9-6.6 us — pool +1.42..+1.59 ms/tok (+5.87..+6.54%), of which the ~144 staging launches ~ 0.95 ms/tok — that is the quantified staging pool a future un-parked consumer-prologue variant would target. No kernel written (905 disposition). Instrument: riir-ai bench_1139_gdn_outproj_strip_ceiling (3-arm interleave, counter guard, T1.1e row; raw log untracked per the 1008 convention). Session: zed-amo-i1135-attncluster, 1793066834.
The trigger: **gdn_out_gemv 0.68 ms/token @ 48 launches x 14.2 us/launch** (W1/bmo; W2/amo 1.27, same launches). The gdn out-project GEMV is the chain's exit boundary launch class (after zgate 0.14 / rot -0.02 noise-floor).

## The caveat (why this files LAST and A/Bs FIRST)

The gdn chain's launch-class budget has two measured NULLs against it: Bench 905 (conv+expand merge, occupancy NULL) and the i1131 re-price ("the launch-class budget around the GDN chain is exhausted at ~0%" — their words, from the T3/T6 measured closes). Those nulls were at OTHER boundaries (conv→expand; rotation), and outproj sits at the chain EXIT (feeding the residual add-back) — the boundary with the fewest cross-block constraints and a natural epilogue target (fold into the recurrence kernel's epilogue or the residual-add producer, i1134-pattern). But the prior probability is LOW on this evidence; the A/B decides, and a NULL here CLOSES the chain's launch budget question permanently (a useful result — the 905 disposition shape).

## The shape

1. Fold gdn_out_gemv into the recurrence/outproj-adjacent producer's epilogue — same partition rules as i1134 (the G1 reduction-alignment check decides).
2. If the fold misaligns: file the G1 read, close the issue, and the chain-budget question is closed by measurement (do NOT iterate variants blind — the 905 lesson: one named variant per budget; 905's named re-open (512-thread posture) belongs to THEIR lane, not this one).

## The gate

G1 to_bits vs chain (the outproj GEMV is a plain reduction — the fold's reduction-shape risk is the lowest of the three filed rungs), G2 paired 4-pair e2e >= 1.5%, REACH guard, demote hatch. Upper bound: 0.68 ms/tok ~= 4.6% e2e (W1 scale).

## Non-goals

- NOT i1134 (norm pair), NOT i1135 (attention), NOT the ffn band, NOT conv/expand/rotation boundaries (905/1131 own those nulls).

**Refs:** riir-ai `.benchmarks/1008_issue1132_metal_launch_floor_census_w1.md` (W1+W2 + common-mode law), `.benchmarks/905_i1133_conv_merge_ab.md` (the NULL discipline), `.issues/1131` (the chain-budget re-price), `.issues/1132`.

## The close record (2026-10-27 ~09:0x +07, amo, GPU-exclusive, load ~2)

Two verified-idle windows of bench_1139 (3-arm per-run interleave; FULL / no-gdn_outproj-parent / no-gdn_out_gemv; folded league PQ2_0, TOKENS=24, 12 timed runs/arm; launches bit-identical 1254/1014/1206 both windows):

| run | FULL ms/tok | outproj-parent pool (240 la) | outgemv pool (48 la) | gemv us/la |
|---|---|---|---|---|
| 1 | 24.29 | +1.589 (+6.54%) | +0.524 (+2.16%) | 10.9 |
| 2 | 24.19 | +1.415 (+5.87%) | +0.592 (+2.46%) | 12.3 |

- The G1 structural read (above) is the closer: the named variant dies at the no-owner wall regardless of the ceiling number, so further windows were not spent.
- The exit-region pool decomposition feeds the parked i1134 lever: 240 la/tok at ~6 us = the folded posture's staging density (the i1134 undercount, independently reproduced in a second region).
