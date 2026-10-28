# Issue 1149 — R2 prework-fused CUDA cell: the 4090 measured cell owed before the CudaRuntime default moves

**Status:** OPEN — filed 2026-10-28 (amo, the riir-ai Issue-1004 R2 promotion lane). **4090-queued** (the box runs the refine batch lanes; do NOT start a cell without the box-window coordination rule).

**Claimed:** zed-amo-i1004-r2promo, 1793157148, expires 2026-10-30T03:12:28+00:00

(scope: the FILING only — the cell itself is 4090-window work; the lease covers this document, not the GPU task)

## Why

`deltanet_prework_fused` was promoted DEFAULT-ON for the Metal/wgpu posture 2026-10-28 (delegated owner verdict on the m5ultra re-cell evidence: bit-identical ×8, e2e pp2048 1.043× / pp4096 1.045×, G4 ~67 fewer dispatches/layer). The runtime initial value is macOS-scoped (`cfg!(target_os = "macos")`): **the `cuda_backend`/CudaRuntime posture keeps the incumbent chunked-conv + batched-expand pair shipping** — its env default resolves OFF-until-measured, exactly the GEMV-occupancy promotion's scoping discipline (`RIIR_GEMV_WG_OCCUPANCY` promoted per-backend; the CudaRuntime half waited for Issue 1115's measured 4090 A/B).

The kernels are backend-generic CubeCL — the feature now compiles into every non-macOS default build too, but the toggle never routes to it there without the env. Promotion onto the CUDA posture needs its own evidence:

## The cell (what the 4090 window owes)

1. **G1 bit-identity at the league CUDA posture**: the fused kernel's outputs vs the incumbent pair on the 4090, at the Bench-1004 league shapes + the production PQ2_0 file (bit-identical by construction on Metal ×8 — CUDA must reproduce, not assume; the `cuda_backend` compiler path is a different codegen target).
2. **G2 e2e A/B**: `bench_1004_r2_prework_fused_e2e`-class read on the 4090 (the prefill CUDA posture), interleaved arms, ≥1.5% promote-if bar. Note the 4090 prefill lane is the cudarc MMA v10t/v10tglu stack (the GLU-fused arm promoted 2026-10-27) — the CubeCL prework arm may be DEAD WEIGHT on the cudarc posture entirely; the cell must first answer **which dispatch path the CUDA prefill actually routes** (the cudarc whole-prefill arm may not touch the CubeCL prework site at all — in that case the correct verdict is "N/A, the cuda posture never routes this site" and the scoping stays permanent, no promotion owed).
3. **G3 no-regression** on the decode rows (untouched by construction — the fused kernel is prefill-only — but the G1 FNV/argmax anchors must reproduce at the league pins).

## Verdict paths

- **Measured ≥1.5% + bit-identical** → promote the CudaRuntime env default to ON (delete the platform split, the Issue-1115 pattern), one commit + records.
- **Measured dead/N-A** (the cudarc posture never routes the CubeCL prework site) → close this issue with the routing evidence; the macOS scoping stands as permanent, the env stays the only CUDA arm.
- **Measured negative** → the feature stays compile-only on CUDA; record the number.

## Env contract (pinned this promotion, both directions)

`RIIR_PREFILL_PREWORK_FUSED`: unset = platform verdict (macOS ON / elsewhere OFF); `"1"|"on"|"true"` arms the fused kernel on EVERY platform; ANY other set value (`"0"`, empty, typo, non-Unicode) restores the incumbent pair on every platform. `set_prefill_prework_fused` is the harness arm toggle, authoritative over the env. Four-case env-semantics unit pin landed with the promotion (`prework_fused_semantics_tests`).

## Refs

riir-ai `.issues/1004` (the promotion record) · riir-ai `.benchmarks/024_deltanet_prework_fused_ab.md` (addendum) · riir-infer `c1cb377` + Issue 1115 (the GEMV-occupancy per-backend precedent) · bench_1004_r2_prework_fused_ab.rs (the stage A/B) · riir-ai Bench 995 (the Bench-995 promote-if bar).
