# Issue 1136 — rung: gdn_out_gemv (the out-project GEMV class at the gdn chain's exit)

**Status:** Open — filed 2026-10-27 ~06:4x +07 from the i1132 census (Bench 1008, W1+W2 reproduced). Unclaimed. **Carries the chain-budget caveat hardest** — read the caveat before claiming.
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
