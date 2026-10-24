# Bench 887 — Issue 1115: the 4090 prefill-tail lm_head A/B for the GEMV occupancy wg rule (CudaRuntime posture)

**Date:** 2026-10-25 · **Box:** 4090 Windows (shikuwa) · **Posture:** `cuda_backend` CudaRuntime (non-macOS), graphs OFF
**Verdict: G3 CLEAN — the platform split is deleted; `wg_occupancy_platform_default()` is `true` unconditionally (default-ON everywhere).**

## What ran

`tests/issue1115_wg_occupancy_prefill_tail_ab.rs` (landed with this record; `[[test]]` row
`cuda_backend,ternary_gemv_cuda_raw,ternary_gemm_batched,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard`,
`#[ignore]`d — run explicitly on an idle 4090). One model load
(`Ternary-Bonsai-2-27B-PQ2_0.gguf`, vocab **248,320** — the lm_head's m, so the armed rule
resolves the 128-wg bucket, genuinely ≠ the incumbent 256), then interleaved OFF/ON rounds
(`set_gemv_wg_occupancy`, the harness arm — the env is DEMOTE-ONLY and was UNSET), 3 paired
medians per shape, pp512 + pp4096.

## Results (the clean detached run)

| shape | off median (tok/s) | on median (tok/s) | on/off | G1 (FNV across arms) |
|---|---|---|---|---|
| pp512 | 594.8 | 595.6 | **1.0013×** | 3/3 identical |
| pp4096 | 557.6 | 557.6 | **1.0000×** | 3/3 identical |

- Reach (the vacuous-guard law): the occupancy counter advanced **exactly +1 per arm-on
  prefill** (the final-tail lm_head is the whole-prefill path's only
  `GemvTernaryCubeCL::launch()`) and **+0 per arm-off prefill** — 12/12 rounds exact.
- Rounds: pp4096 554.7–558.2 tok/s (±0.3%); pp512 0.86 s class after a 3.2 s init round.
- Dispatch-only held cross-runtime: byte-identical logits at the 128-wg dispatch on CUDA,
  as Metal proved at Bench 996 cell 4.

## Provenance

- **PROVENANCE: power=AC scheme=high load=1.2 swap=5307MB gpu=3 %, 421 MiB canary=skipped**
  (`riir-reflex/scripts/bench_preflight.ps1`, pre-run).
- GPU-exclusive during the run (no compute consumers; Unity/Zed exempt per the standing
  owner call). During-run samples: 16,728 MiB / 99% util / 64–68 °C — no spill.
- The quoted run was a DETACHED quiet-box run (`cargo test` via a hidden process, output
  redirected) — the two foreground runs before it measured the same G1/reach but their
  walls were box-poisoned (see below) and are NOT quoted as timing.

## The measurement trap this cell hit (recorded for the next lane)

Two foreground runs produced 19–45 tok/s pp4096 walls with 90–209 s round swings and a
~1000× gemm-stage blowup (`RIIR_PREFILL_CUDA_TRACE=2`: gemm 200 ms → 110–244 s, every stage
proportionally dead) while G1/reach stayed green. The same binary detached + quiet ran
557.6 tok/s ±0.3%. The signature is VRAM/box-state pressure (24 GB card at 16.7 GB during
the run; foreground job interference), NOT the occupancy rule — the rule's delta stayed
sub-noise in every regime. Lesson: on this box, time the 4090 prefill lanes DETACHED.

Also fixed on the way: a single pp4096 chunk with graphs OFF over-commits and spills —
the test chunks at the league's 2048 (pp4096 = two whole-prefill chunks; the lm_head still
fires exactly once, per the counter assert). The first attempt also silently fell to the
slow CubeCL per-token body: `prefill_cuda_gate_ok()` hard-requires
`ternary_deltanet_chunked_prefill` + `ternary_attention_batched_prefill` — without them
the whole-prefill arm refuses and nothing prints (the green-looking 46 tok/s run was the
fallback path; the launch counter caught it).

## The promotion (same shape as the Metal one)

`wg_occupancy_platform_default()` is now unconditional `true`; the
`#[cfg(any(not(feature = "cuda_backend"), target_os = "macos"))]` /
`#[cfg(all(feature = "cuda_backend", not(target_os = "macos")))]` split is DELETED. The env
stays DEMOTE-ONLY on every platform (unset = ON; any set non-1/on/true value — empty, "0",
a typo — restores the incumbent). Unit-pinned: `the_platform_default_is_on_everywhere` added
(5 tests in `wg_occupancy_semantics_tests`, all green at `--features ternary_gemv,cubecl_runtime`).

References: riir-infer Issue 1115 (the owed cell; closed with this record) · Bench 996 (the
Metal promotion surface) · reviewer verdict round `cd03cea7` (the Metal promotion's
blast-radius correction — this flip removes the split that verdict installed, with the
cell it asked for now measured).
