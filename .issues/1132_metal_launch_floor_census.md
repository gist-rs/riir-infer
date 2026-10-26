# Issue 1132 — the Metal per-dispatch floor census: whole-graph launch attribution on the folded decode (the 1113 defense's named standing lever (c))

**Status:** Open — filed 2026-10-26 ~23:3x +07 from the Bench-1007 addendum + the 1113 ledger (docs-only scout, no code touched, no GPU run this session — the box is carrying a sibling Metal training job). Unclaimed for EXECUTION; needs a verified-idle Metal window. The filing preserves the instrument design fully-specified so the next window can execute immediately.
**Claimed:** none — the scout is complete; execution is a fresh claim. Disjointness: the i1127 T3 lane (FWHT family, claimed, in flight) and the i1131 T6 scout (CUDA rotation geometry) are DISJOINT from this file and lane; the i1130 concat-split rung is banked (b635a43) with its A/B owned by zed-m3-1130-concat-split — the census INSTRUMENT below must not gate on or flip RIIR_GDN_CONCAT_SPLIT_FUSED (it observes whatever posture is shipping).

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
