# Issue 1166: the Metal prefill new-kernel follow-up lane — matmul2d tile geometry + the FFN/DeltaNet GEMM residual + H5 cooperative-tensor (crossing 1.0× everywhere)

**Status:** OPEN — lane FILED (no kernel work started); the named residuals are Bench 934's own verdict, quantified against llama.cpp per cell below. Each rung needs its own GOAT; H5 stays census-gated per its standing demotion.
**Date:** 2026-11-02
**Parent:** [`.issues/1146_prefill_throughput_gap_1box.md`](1146_prefill_throughput_gap_1box.md) (CLOSED-class: every 1146 leg measured + promoted; this is the lane 1146's record named as "a NEW-kernel follow-up lane, not an armed cell of this issue")
**Session:** zed-a9badc86@m3

## Where the residual lives (Bench 934 stacked posture, dist, w512, bmo)

| cell | ours | llama.cpp | ratio | residual |
|---|---:|---:|---:|---|
| pp32768 | 774.5 tok/s | 772 | **1.00×** | — (parity; defend, don't chase) |
| pp4096 | 973–977 tok/s | 1046 | **0.93×** | ~7% |
| pp512 | 821–826 tok/s | 1087 | **0.76×** | ~24% |

Bench 925 (promoted-posture census, pre-4c): attention 37% / FFN 34% / DeltaNet projections 29%. The 4(c) arm cut attention ~2× at long context — the census needs a re-run at the promoted+attn posture to re-rank the residuals; the static read says the remainder is **inside the GEMM bodies** (Bench 934: "matmul2d tile geometry at prefill shapes + the FFN/DeltaNet GEMM residual").

## The rungs (each its own GOAT; ranked by expected yield)

- [ ] **Census re-run FIRST** (`bench_1146_prefill_frame_census` at the promoted+attn posture, `I1146C_TENSOR=1` + attn arm + H4 + relaxed): re-rank FFN/DeltaNet/attention shares post-4(c) — never design a kernel against a stale share (the Issue-1113 lever-3 lesson: the chunk-128 census under-predicted the league-chunk share 2.3×).
- [ ] **Rung A — matmul2d tile geometry at prefill shapes**: the zero-copy tensor GEMM's tile/threadgroup config (64×128, 4 simdgroups — the upstream `mul_mm.metal` shape) is tuned for big-M; short-prefill frames (M = 512 tokens × per-projection N/K) may want a second shape entry (per-SKU tuning table is Research 400's institutionalized discipline — M5-class enumerated). Measure per-shape kernel cells first (the 886 calibration-before-assertion rule).
- [ ] **Rung B — FFN/DeltaNet GEMM residual**: whatever the re-census ranks above ~15% share that is NOT matmul2d-geometry (dequant granularity is already fork-equal post-H4 — Bench 920's load-granularity finding; the two-bit-plane delta is minor). Candidates: dispatch batching across the ~7 projections/frame, output blit elimination (H1's static bound was ≤2% — re-read at the promoted posture), f16 B-operand staging (H2, unmeasured at the promoted posture).
- [ ] **Rung C — H5 cooperative-tensor (the absolute-win class, census-gated)**: dequant A straight into a `cooperative_tensor`, feed `matmul2d` from registers — kills the threadgroup staging + the two barriers per K-tile. DEMOTED as a pp32k-gap explainer (upstream runs identical staging — H3/H5 cannot explain their win there), but kept as an ABSOLUTE win: Apple's own recommended custom-format path (WWDC26 Metal-tensors session; Modular's `AppleM5MatMul` — operands DRAM→register because threadgroup staging degrades matmul on Apple silicon). Write it only if the re-census shows dequant/barrier time ≥ the post-H4 residual at the TARGET shapes (short prefill), and mind the Metal fast-math fma fence (`select()`) when moving smem math into registers.
- [ ] **pp512's fixed-cost slice**: 1 frame = worst amortization for per-frame fixed costs (pool zero-init now 8–15 fills/process post-`401af78`, stage handoff ~0.1 ms, embed dequant). Quantify the non-kernel slice at pp512 before crediting it to tile geometry — a config-class rung (frame overlap, Bench 914's fix-direction 3) may be cheaper than any kernel here.

## Acceptance

Any rung: per-shape kernel cell + e2e A/B on the promoted posture (per-pair CONTAM guard, one binary, env-only arm), G1 (tolerance-class arms get the KL instrument at the shipped posture; bit-exact arms assert max_abs 0), decode G3 unchanged. Promotion per the feature-flag discipline; the prefill path is the league's pp rows too, so a win here re-prices riir-ai Issue 1154's ceiling.

## Boundary

Substrate-side: kernels in `riir-infer-gpu`, seams in `riir-infer-dist`. No riir-ai work required; the league-side adoption of the EXISTING arms is riir-ai's Issue 1154 (separate lane — do not block this kernel work on it).
