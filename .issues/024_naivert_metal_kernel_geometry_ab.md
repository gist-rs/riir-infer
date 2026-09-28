# Issue 024 — NaiveRT-pattern kernel geometry A/Bs on the laya Metal lane

**Status:** OPEN — measurement-gated candidates from `.research/003` (NaiveRT distill); profile first, build only if the gap shows.

**Source:** NaiveAI NaiveRT blog (verify-path mega-kernel; facts + mapping table in `.research/003`). Their two transferable intra-kernel wins: (a) weight fetch overlapped with the dependent RMSNorm inside a fused prologue (their 21–23 μs/step via TMA), (b) small head counts remapped into the MMA n-dimension instead of padding m (their 8 heads/rank → n8 remap).

## Why gated

Neither trick has measured evidence on OUR shapes. This lane is a single-GPU MSL **encoder** forward (seq ~100–320 tokens), not an 8×GPU TP8 FP8 decode path; Apple silicon has no TMA. Both candidates resolve either to a gated win or to a recorded N/A/negative — both endings are acceptable, guessing is not.

## Tasks

- [ ] **T1 — norm-share upper bound.** Run the lane's per-dispatch profiler `LAYA_METAL_PROFILE=1` (`riir-infer-laya/src/laya/riir/metal.rs:362` — every dispatch gets its OWN command buffer, committed + waited, so it reads per-kernel SHARES only and structurally cannot see an in-pass gap inside the shipped single CB) over typed_decisions + banking77 cases — the `typed_case_split` example in riir-reflex is the ready driver (`cargo run --release --features laya-riir-metal --example typed_case_split -- english`). Record RMSNorm's per-kernel share: that share is the UPPER BOUND on what overlapping weight-fetch with the norm (the 024a mechanism) could save. Below the noise floor (±6% idle-class spread, katgpt-rs Issue-833 calibration) → close 024a as measured N/A without building anything. Also record the lane's OWN idle round-to-round spread next to the norm share — the ±6% floor is borrowed from a different lane, and if the Metal lane is noisier the N/A threshold relaxes to the measured spread. Above the floor → T2's fused-vs-unfused interleaved A/B is the only instrument that can show the in-pass boundary gap.
  **BLOCKED 2026-09-28 (this session): the box refuses — `bench_preflight.sh` exit 1 at load 8.28 > MAX_LOAD=6.0 with 1.67 GB swap in use (three sibling sessions active). Instrument verified ready: `typed_case_split` binary builds green at HEAD (abbcbb3 content), preflight arms all pass except load (power=AC, powermode=2, settle 700 min, canary 132.1 µs best-of-5). The m-histogram half (T3) is DERIVED, not timed — that half is load-independent and runs first when this lane resumes; the profile half waits on the same quiet-box window as reflex-site 003 T1.**
- [ ] **T2 (only if T1 shows a gap) — prototype (a):** norm-fused-into-GEMM prologue with threadgroup-staged W tiles (async copy + barrier; no TMA). Position-balanced interleaved A/B, ≥4 rounds, house law; G5 parity byte-identical required before any adoption claim.
- [ ] **T3 — small-m shape histogram.** Instrument/derive the distribution of m at real call sites (per-head ops; any decode/verify shape the lane ever runs). If the encoder lane never hits m < tile M in anger, record the histogram and close 024b as measured N/A.
- [ ] **T4 (only if T3 shows padding tax > noise) — prototype (b):** heads-in-n MMA remap for the narrow instance; same A/B + G5 discipline as T2.
- [ ] **T5 — land the verdict.** Any adopted win: file the kernel_opt rule candidate in riir-clippy (prefetch-before-barrier / heads-in-n) with verbatim quotes from our own diff. Negative/N-A: record raw numbers here and close — the MoE-fusion ending is the honest one.
- [-] **Whole-layer persistent mega-kernel for the encoder — deliberately NOT pursued.** Launch-boundary class already closed by the pass-scoped CB (per-op commit+wait measured 0.59 ms/dispatch = 19× slower) + the fused flash_attn kernel; scale mismatch vs NaiveRT's decode-verify window. Re-open only if T1 shows >10% residual boundary cost after T2.

## Context

- Launch-boundary + K/V-staging classes: already covered (pass-scoped CB; flash_attn threadgroup staging + two-pass normalize; kill-switches `LAYA_METAL_FLASH` / `LAYA_METAL_MPS` / `LAYA_METAL_FOLD_*`). The per-op commit+wait baseline: 0.59 ms/dispatch (`metal.rs:21`), 19× end-to-end on the gate corpus (riir-reflex `.benchmarks/001_phase1_harness.md`, 280 s vs 15 s).
- Reference numbers to beat (both narrative: riir-reflex/AGENTS.md §"The laya-riir lane", Metal passes): typed 5-q median on/off **0.984** post head-defer (T12, riir-infer `40d15dd`, `tests/metal_head_defer_ab.rs`); banking77 **75–76 ms** p50 post flash_attn (fused flash_attn, riir-reflex `bc2a426`; python oracle 70). Stage shares for T1 context: encoder 90.1% of case GPU, sgemm narrow 85.7% of that, flash_attn 7.5% (`typed_case_split`, riir-infer `be46033` / riir-reflex `abbcbb3`).
- Class prior art (no novelty claimed): Hazy Research Megakernel, AutoMegaKernel, Lucebox — see `.research/003` §5.
