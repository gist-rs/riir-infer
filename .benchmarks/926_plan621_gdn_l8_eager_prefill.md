# Bench 926 — Plan 621 T3/T4/T5: the GDN L8 eager-prefill recurrence (the Yukon lane-geometry class)

**Date:** 2026-10-19 · **Box:** shikuwa / 4090 RTX (sm_89), Windows 11, AC power, i7-13700K ·
**GPU state at bench:** idle compute (3% util desktop residue, 43°C, ~9.7 GiB graphics-resident, no compute apps) ·
**Build:** riir-infer develop @ plan-621 window session (release profile, `--features
ternary_gemv_cuda_raw,ternary_gemm_batched,cubecl_runtime,riir-infer-core/bonsai2_hadamard`) ·
**Model:** `E:/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` (7,206,168,928 B) ·
**Claim:** plan-621 live claim, shikuwa/4090 idle loop, expires 2026-10-20T15:35+07.

## What landed (T3)

`crates/riir-infer-gpu/src/prefill_cuda_deltanet_l8.rs` — the L8 eager recurrence, **ADJACENT-map
port**: K rows/warp × WPC warps/block, group-lane n holds incumbent chains `K·n + m`, register fold
`(p0+p1)+(p2+p3)` (K=4) / `p0+p1` (K=2), then in-group lane xor 1/2/4 (K=4) / 1/2/4/8 (K=2). Six
geometry arms `recl8_pf_k{4,2}w{1,2,4}`. Dispatch seam in `prefill_cuda_full.rs` beside
`gdn_chunked_enabled`: env `RIIR_PREFILL_GDN_L8` (env-wins-over-setter), setter
`set_prefill_gdn_l8` bumps `GV_KNOB_GEN`, admission `hd==128 ∧ p ≥ 17 ∧ rowpar path` (never
composes with chunked/mma — the else-chain routes). **DEFAULT OFF.** Default arm = k4w4 (the T5
winner; arm-independent bit-identity makes this a perf-only pick).

**THE NUMERICS FINDING (plan 621 addendum-d, this session):** the production prefill rowpar
canonical form is `(dot4_f1, UPD_FMA, ps_treeup, SC_RSQ)` — `ps_treeup` is the ASCENDING butterfly
(offsets 1,2,4,8,16). The plan's §4 STRIDED map + T2 oracle modeled the DESCENDING
`fused_rows_hd128` family (the qwen38 decode/verify lane) — that port would have FAILED the
production FNV pin. The adjacent map's fold IS the ps_treeup tree; the T4 gate below is the
mechanical adjudicator.

## T4 — bit-identity gate (G1, bytewise class)

`recl8_bit_identity_vs_incumbent_all_geometries`: all 6 arms × shapes {(48 heads, league), (2, 1
heads)} × p {17, 64, 512} — **state AND output bit-exact vs the production dispatch**
(`launch_recurrence` → recmr r4w4 @ TreeUp), 0 mismatches, pass. Plus the whole-model e2e proof:
the league checkpoint's prefill FNV is IDENTICAL knob-off vs knob-on —
`6921a5c129844b7a` ×4 rounds, argmax 1495 ×4 (both e2e runs, k4w1 and k4w4 defaults).

## T5 — kernel-only bench (interleaved pairs, median of 7, order alternating)

Recurrence-only, Bonsai-2 GDN shape (48 value heads × hd128), release, PROVENANCE above:

| p | incumbent r4w4 | k4w1 | k4w2 | **k4w4** | k2w1 | k2w2 | k2w4 |
|---|---|---|---|---|---|---|---|
| 17 | 4.93 µs/tok | 0.967× | 1.038× | **0.953×** | 1.000× | 1.007× | 0.882× |
| 64 | 1.84 µs/tok | 0.853× | 0.864× | **0.857×** | 1.017× | 1.011× | 0.994× |
| 512 | 0.76 µs/tok | 0.683× | 0.671× | **0.661×** | 1.036× | 1.040× | 1.014× |
| 2048 | 0.62 µs/tok | 0.810× | 0.812× | **0.791×** | 0.953× | 0.949× | 0.934× |
| 4096 | 0.59 µs/tok | 0.807× | 0.808× | **0.785×** | 0.960× | 0.943× | 0.941× |

(ratio = L8/incumbent wall; lower is faster. k4w4 = **1.26× at pp2048, 1.27× at pp4096, 1.51× at
p512**.) K=2 is parity — the 1-shuffle saving does not pay for its doubled k/q load count; the
H200 shape-dependence read holds on our side too. WPC>1 helps (the Bench-896 occupancy lesson
repeats).

## T5 — e2e league cell (the honest half)

`recl8_e2e_league_prefill_ab`: whole-prefill cudarc stack, DQ-lane posture (cmma_i8+psplit,
graphs OFF per the D5 law — a captured graph would freeze one arm's kernel sequence), one
~2048-token prompt, interleaved off/on rounds ×2, Bonsai-2-27B:

- FNV identical across arms (bit-identity e2e ✓, the class proof).
- **e2e tok/s FLAT**: steady-state 63.8 (off) vs 64.0 (on) — ±0.6%, noise. At this posture the
  recurrence stage is ~1.26 ms of a ~490 ms layer (~0.3%) — the GEMMs dominate; the 1.26×
  kernel-only win composes to ≈ nothing at the whole-model cell.
- ⚠ this posture is NOT the league pin posture (65 tok/s here vs the ~4000 tok/s league cells —
  no CUDA-graph capture, fewer arms). The LEAGUE-posture e2e re-scopes to the perf-rematch lane
  (riir-ai), where graphs+full-arm stack may weight the rec stage differently. The A/B lever is
  `RIIR_PREFILL_GDN_L8=1`.

## T6 — verdict

- **G1 PASS** (bytewise; kernel-level all arms + whole-model FNV e2e).
- **G2**: kernel-only PASS (1.26-1.51×, k4w4); **e2e FLAT at the measured cell** — the boundary
  is the honest deliverable.
- **G3 PASS** (default OFF; env-off = the incumbent dispatch untouched — the else-chain routes).
- **G4 PASS** (alloc-free launcher; the vacuous-guard counter ships with the concurrency note).
- **PROMOTION: NO** — no e2e gain demonstrated at the measured cell. The arm ships OPT-IN
  (`RIIR_PREFILL_GDN_L8=1`); a league-posture re-gate (perf-rematch lane) is the re-open path.

## Lessons (this bench's own)

1. **Vocabulary/oracle scope**: T2's "tree-identical" verdict was family-true (descending) and
   production-false — check WHICH lowering the production dispatch uses before trusting a fold-tree
   oracle's transfer. The adjacent map rescues the strong class.
2. **The == counter assert races under parallel libtest threads** (process-global counter, the
   e2e lane's L8 rounds increment it too): failed in the parallel cell, passed serially 3/3 —
   the concurrency class, `>=` + the note is the repair.
3. rustfmt on a shared file: bare `rustfmt` dies on edition-2024 let-chains — format the NEW
   file only, leave shared files to surgical diffs.
