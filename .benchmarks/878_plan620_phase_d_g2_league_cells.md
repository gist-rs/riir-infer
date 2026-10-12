# Bench 878 — Plan 620 Phase D: the G2 league cells (si-ON vs flag-off, pp2048 + tg128)

**Date:** 2026-10-13, 04:11–05:37 +07 (cells) · 4090 (Windows, RTX 4090 24 GB), `E:\git\riir-ai` + `E:\git\riir-infer` main checkouts both at origin/develop (riir-ai `5f039e886`, riir-infer `fbd6e6c`).
**Unit:** riir-refine idle loop (m3 lane), plan 620 Phase D item 3 — the last open item of plan 620 (`620_cuda_si_hook_and_gpu_g3_lane.md`).

## Verdict

**G2 PASSES — both cells within ±3% by two orders of margin. The si hook is perf-free within noise on both league postures.**

| Cell | flag-off (incumbent) | si-ON (calibrated, τ=10) | Δ | Gate ±3% |
|---|---|---|---|---|
| tg128 decode (`bench_734_t5_decode`) | **95.30 tok/s** (median of 5; tripwire green) | **95.12 tok/s** (median of 3; tripwire green) | **−0.19%** | PASS |
| pp2048 prefill (`bench_772_t0_gemm_share`) | **4075 tok/s** (median of 3; tripwire green) | **4062 tok/s** (median of 3; tripwire green) | **−0.32%** | PASS |

## The arm wiring (riir-ai `5f039e886`)

ONE build serves both postures — `SI620_G2=1` attaches + arms `SiGpuSession`
(τ=10.0 via `SI620_TAU`, calibrated from the on-box sidecar
`E:\tmp\si622-cal-4090.bin`, `HeadCalibration::digest` prefix
`4d8f95efd6b0ccb2…`, identical in both si cells); the env absent = the
incumbent league posture. The paired A/B rests on the bench-872 flag-0 law
(disarmed = bit-identical incumbent). Feature set (both cells):
`cuda_graphs_forward,bonsai2_hadamard,scale_invariant_attn,ternary_gemm_batched,cubecl_runtime,ternary_gemv,ternary_deltanet_chunked_prefill,prefill_q8_act,prefill_mmq_v2`.

- si-ON decode: `[si620] DECODE ARMED tau=10 calibrated=true block=512 layers=64 heads=24`.
- si-ON prefill: `[si620] PREFILL ARMED … block=2048 …` (the wrapper lane's session — the caller-supplied-stream API exercised exactly as designed at `61eb815`).

## Flag-off evidence (the incumbent posture is intact)

- decode: G1 logits FNV `63496f6688c8f742` — **the league's decode pin EXACT** (league doc 2026-09-24 re-pin cell); argmax-walk FNV `9fd59630b2f93ab1` (this run's argmax-walk seed state; the league pins record `98412dc406c4ea19` under the 8-warmup protocol variant — the walk-hash pairing is per-run-state, the logits FNV is the portable pin).
- pp2048: G1 pins **asserted GREEN** at (2048, q8_v10t, folded) — argmax 248046 / FNV `0f8fddb9ef217548` (the Plan-605-T4 v3 FA pin). A `scale_invariant_attn`-enabled build with the env absent IS the incumbent, proven in the field.
- si-ON outputs differ by design (the armed transform) — decode argmax-walk FNV `3543ad6ba0c59740`; the pp si cell ran with pins skipped (the armed posture's outputs are their own; the harness disclosure line) and the ramp tripwire ARMED (green).

## Box state + provenance

- GPU exclusive at every cell: the Issue-834 in-harness gate REFUSED four earlier attempts (03:47, 03:54) while the 039 lane's `riir_infer_gpu-cd1abd68102c9ada.exe` test batches held the box — the gate did its job, zero contaminated cells; scored cells fired only after two consecutive 60-s quiet ticks (no cargo.exe, GPU ≤6%/372 MiB baseline). The 039 lane owns `wt039infer`; their campaign drained before the scored window.
- nvidia-smi beside the cells: 2–5% util / 372–526 MiB between cells; 99% DURING (the work is real).
- Rounds recorded at `--release`, `--test-threads=1`, Bench-976 clock-warm bursts (8 s scored cells, 30 s decode flag-off retry).

## The honest debris (three classes hit en route)

1. **Decode flag-off ramp** — two attempts tripped the Bench-976 wire (1.46%, 2.70% last-two spreads, mid-run dips not ramps). Valid at 30 s burst × 5 rounds: 95.41/92.92/95.63/94.99/95.30 → median **95.30**. The si-ON cell (8 s × 3) was steady on its first try; no re-run owed.
2. **rustc STATUS_ACCESS_VIOLATION (0xc0000005)** compiling vendored `cubecl-wgpu` (5:10 attempt) — the known flaky-Windows-rustc class; the resume-build re-run passed clean. No code involvement.
3. **Two resolve races with the box's scheduled gitsync** (03:23 lockfile package-collision naming `riir-engine` at the main checkout + a worktree; 05:16 `ternary_group_q8_act` version-selection failure mid-katgpt-rs ff — the sibling's issue-935 commit `0c5ed0fe7` landing under the resolve). Both transient, both resolved by re-running on a quiet box. No worktree/clone scaffolding survived — the scored cells ran from the MAIN checkouts, the league's own home.

## The 038-lane WIP subsumption note (box hygiene, same session)

`E:\git\riir-infer` carried 22 dirty files (+1627/−1157, cycle-40's "038 lane WIP"). The surgical-ff protocol (patch saved to `E:\git\_sync\infer038_wip_2026-10-13.patch` → ff `cd3e327..fbd6e6c` → `git apply --3way`) returned **"No valid patches in input"** — the WIP was already fully present upstream (the sibling's own landed commits: 85345a7 multicol bench, 4db9e15 standby scoping, fd16ab6 039 close, 971b842 pointer-copy). Zero bytes lost; the checkout is clean at origin/develop.

## Consequences

- Plan 620 Phase D item 3 PASSES → plan 620's box work is COMPLETE (A bit-identity 872, B/C harness + G3 873, D cells 878).
- Plan 622 Phase 7 (the engine vehicle) holds through its G2 cells: si-ON is perf-neutral on the league postures; the quality axis stands where bench 873 left it (si/si-cal ppl FAILs reproduce; no promotion) — the owner-gated promotion decision now has BOTH axes measured.
- The Metal parity leg of G2 (the engine's Metal attention hook) stays explicitly out of scope (plan 620 §out-of-scope) — its own unit when claimed.
