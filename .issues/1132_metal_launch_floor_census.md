# Issue 1132 — the Metal per-dispatch floor census: whole-graph launch attribution on the folded decode (the 1113 defense's named standing lever (c))

**Status:** CENSUS COMPLETE (2026-10-27 ~06:4x +07) — instrument + W1 + W1-repeat + W2 recorded, G-C2 rank PASS, ladder filed to i1134/i1135/i1136. Lease RELEASED (the lane's deliverables are all landed; the rungs live in their own issues). This file remains the instrument + census reference.
**Lease closed:** zed-amo-1132-launchcensus, epoch 1793032042, expires 2026-10-28T16:00:00+00:00 — deliverables all landed (instrument + W1 + W1-repeat + W2 + G-C2 rank PASS; ladder filed to i1134/i1135/i1136); the holder recorded the release in the Status line; reshaped from the leftover Claimed line per the Issue-943 fold law (the claim read LIVE 9 h past its release — the false-YIELD class).
The claim covers the INSTRUMENT build (CPU-only: the census-bit refinement sites + names + the mask-semantics pin, compile-verified now) and the EXECUTION SLOT (the census RUN); the run phase re-checks the box gates before its first GPU touch (the 649 exclusivity; the sibling Metal training job was live at claim time — the run waits for verified-idle). Disjointness: the i1127 T3 lane (FWHT family, claimed, in flight) and the i1131 T6 scout (CUDA rotation geometry) are DISJOINT from this file and lane; the i1130 concat-split rung is banked (b635a43) with its A/B owned by zed-m3-1130-concat-split — the census INSTRUMENT below must not gate on or flip RIIR_GDN_CONCAT_SPLIT_FUSED (it observes whatever posture is shipping).

**⟲ 2026-10-27 ~01:2x +07 zed-amo-1132-launchcensus: WINDOW 1 RECORDED on bmo (see the box-swap rationale there) — riir-ai `.benchmarks/1008_issue1132_metal_launch_floor_census_w1.md` + raw log; test ok in 51.37 s, tripwire green. G-C1: ffn children 99.7%, gdn_proj 97.7%, joint deltanet+ffn 100.1% (overlap ~= 0); attention 91.6% and gdn_pre 81.4% (launch-accounting-explained; W2 arbitrates). Floor ladder W1: gdn_proj_concat 1.70 ms/tok @ 48x35.4 us (i1130 residue — zed-m3's next rung), norm pairs 2.19 @ 128x~17 us, attn proj+decode 1.10, gdn_out_gemv 0.68 (un-lane'd); gdn_pre_betal2 0.66 is i1133's evidence (disjoint recorded). Rungs NOT filed until W2 reproduces. W2 = amo verified-idle window (post-i645-SFT, ETA ~07:40-09:30 +07) and MUST run the PINNED binary (sha in the record) — a 727-line deltanet-forward refactor is in flight, a tree rebuild would measure a different posture. bmo idle verified at launch (load 1.04); contamination flag for the t04-mlx bmo arms recorded in the bench. The pruned instrument-landed note lives in 3d95ff9/9d33c3a.**

**⟲ 2026-10-27 ~01:2x +07 zed-amo-1132-launchcensus: W1-repeat (intra-box variance read, minutes apart, same pinned binary on bmo, 51.13 s ok): launch counts BIT-IDENTICAL on all 29 rows, top-10 rank identical, common-mode +0.5%, noise floor ~±0.05 ms/class — the W2 rank bar is now QUANTIFIED (see the bench addendum). NOT a G-C2 window; G-C2 remains W1 + W2-on-amo.**

**⟲ 2026-10-27 ~06:4x +07 zed-amo-1132-launchcensus: WINDOW 2 EXECUTED on amo + G-C2 CLOSED — rank PASS (top-10 identical across all four runs, launches bit-identical everywhere); common-mode law quantified (idle +0.5% / load~6 +16% / load~18 +30% — dispatch-wall absorbs host CPU load; cite RATIOS not amo absolutes). W2 canonical (+16%, GPU-exclusive, sibling i645 lane fully exited; load 2.5-6.6); W2b (+30%, refine-build burst) disclosed as the load-sensitivity datapoint. LADDER FILED: i1134 norm-pair epilogue fold (2.19 ms/tok W1 — top un-lane'd class, 1130-pattern, A/B-led), i1135 attn proj+decode (1.10), i1136 gdn_out_gemv (0.68, chain-budget caveat). gdn_pre_betal2 stays i1133's evidence. Lease RELEASED. Record: riir-ai .benchmarks/1008 (W2 section + four-run law).**

## The trigger (measured, not assumed)

- Bench 1007 (the dense-pair rung record): in-pipeline the tiny-GEMV class carries **~74 µs/launch**; the micro read (`bench_dense_pair_geometry`) reads the per-dispatch floor **~105 µs at 2 launches / ~120 µs at 1, FLAT across wg sizes** — geometry-dead, dispatch-count-bound. The promoted rung halved the pair (4.73 to 2.56 ms) but the residue is still 48 launches x ~53 µs — **the largest per-launch GEMV-class item after ffn**.
- Post-promotion the decode dispatches **~1254 launches/token** (Bench 1007 decomposition). At the ~50-100 µs class, the floor-relevant subset is the named standing lever: "the fork's 438 GB/s posture rides fewer, fatter launches" (Bench 1007 addendum).
- The 1005/763 line proved the token is exec-bound (additivity 98.6%) — so launch-floor µs are REAL token time, not bubbles. The 1006 census resolved STAGES (gdn_proj 29.8 pct, then bits 9/10 split to the pair); no instrument yet attributes the token at LAUNCH granularity on Metal.

## The ask (the instrument, then the ladder it names)

Build the Metal-side whole-graph launch census — the wgpu/cubecl analog of the 4090's nsys graph-node trace (Bench 902), which is not executable from macOS:

1. **Per-launch attribution instrument.** Two candidate mechanisms, pick by a 30-min feasibility spike (record the choice + why in the bench record):
   - (a) **Census-bit extension of the 1006 pattern**: refine the DECODE_STAGE_MASK to kernel-class granularity (one bit per launch SITE class: ffn_gate, ffn_up, ffn_down, lm_head, gdn concat, gdn dense pair, attn q/k/v, attn out, conv1d, beta/decay, expand+L2, recurrence, out_proj, norms/splits/copies, rope...) — timing via the existing stage-instrument plumbing (scoped wall reads in-pipeline), mask=ALL bit-identical shipping.
   - (b) **xctrace GPU-encoder trace** of one decode run (the CLI template, not GUI): per-encoder-event µs table, symbolicated by kernel function. Zero production code, but the parse+attribution script is the cost.
   - Law: whichever mechanism, the census is MEASUREMENT-ONLY — mask=ALL (or no-toggling) runs must stay bit-identical (anchors EXACT); the Issue-782-style vacuous guards apply.
2. **The census read**: per launch-site class — count/token x median µs = ms/token table, ranked; the floor-relevant subset (small-m classes) separated from the bandwidth class (big-m). Deliverable: the same shape as Bench 902's table, for Metal.
3. **The ladder it names**: each class with >= ~0.5 ms/token at >=2x over its roofline-class opens a rung candidate (fusion/movement-into-epilogue like 1129/1130, or batching like the fork posture). File each as its own issue with the census row as the trigger; do NOT bundle.

## GOAT gates (for the census itself, and any rung it spawns)

- **G-C1**: the instrument is additive-honest — per-site sums reconcile to the stage total (the 1005 additivity law, >=95%); mask=ALL anchors EXACT x3.
- **G-C2**: the census table reproduces across 2 windows (day-drift common-mode; the ratio/rank is the record, like the re-cell law).
- Any spawned rung follows the standard ladder: bit-identical by construction, G1 to_bits, e2e ABBA promote-if >=1.5%, vacuous-guard counters, demote hatch.

## Non-goals / scope guards

- NOT the FFN big-shape GEMV bandwidth study (118 MB weights at ~79 GB/s effective — a different, bigger lever; the access-pattern rung class). The census will SHOW ffn's per-launch rows, but the bandwidth rung has its own lane.
- NOT the CUDA lane (Bench 900/902 already did the 4090-side graph census; i1131 owns the rotation-geometry scout there).
- NOT the A/B of the banked i1130 rung (lease: zed-m3-1130-concat-split).
- NOT any new kernel work — this issue is the instrument + the named-ladder filing only.

**Refs:** riir-ai `.benchmarks/1007_issue1129_dense_pair_rung.md` (the floor evidence + the standing-lever addendum), `.benchmarks/1006_issue1113_gdn_subclass_census.md` (the census-bit pattern), `.benchmarks/1005_issue1113_decode_execbound_decomposition.md` (exec-bound + additivity law), riir-infer `.issues/1130` (banked rung + lease handoff), `.issues/1131` (the docs-scout shape + BLOCKED-BY-LEASE grammar), riir-infer issue 1127 (the T3 in-flight lane — disjoint), riir-infer issue 1129 (the promoted dense-pair precedent).
