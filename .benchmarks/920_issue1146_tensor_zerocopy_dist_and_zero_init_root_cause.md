# Bench 920 — Issue 1146 rung 2a: the Metal tensor zero-copy prefill GEMM in the dist lane, plus the root cause of the "first-prefill" corruption (wgpu lazy zero-init)

**Status:** RECORD — PROMOTED into the dist `gpu` row (`54f0dce`, `RIIR_DIST_METAL_TENSOR=0` demotes). G1 bit-exact with a cold pool after the fix (`3bbf275`); G2 **1.21–1.34×** prefill over the simdgroup incumbent at the same width. Stacked on rung 1's width 512, the dist prefill goes from **0.13–0.16× of llama.cpp to ~0.26–0.27×**.
**Date:** 2026-10-28 (bmo cells 16:56–17:41 +07)
**Session:** katgpt-rs-71, 1793178290
**Issue:** [`.issues/1146_prefill_throughput_gap_1box.md`](../.issues/1146_prefill_throughput_gap_1box.md) — ladder rung 2 (the M=128 GEMM classes hold 90% of the frame, [Bench 914](914_issue1146_feature_row_ab_and_prefill_frame_census.md))
**Box:** bmo (Mac17,15, M5 Ultra, 96 GiB), AC
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` sha256 `3907dc16…62ec1` (league pin)

## Why this rung was cheap

riir-ai Plan 641 had already promoted the ktile64 tensor (matmul2d) prefill GEMM ([riir-ai Bench 1009/1011](../../riir-ai/.benchmarks/1009_641_ultra_window_t1_t2_t3_cells.md): 1.16–1.24× e2e, pp2048–pp32k). But only the league **bench harness** called `set_prefill_use_metal_tensor_zerocopy(true)`, and the dist `gpu` row did not even compile `metal_tensor_gemm`. It is the 1141/1146 class a third time: the production posture lived in a harness, not in a serving binary.

## The defect it surfaced: wgpu lazy zero-init (Issue 670 T5, root-caused)

The first dist A/B (opt-in build `cea3fba`) **failed G1**: the generated ids were `[68, 13, 197, 197, …]` against the incumbent's `[261, 248046, 198, …]`. Localised with the new gate `issue1146_tensor_range_g1`:

| order | `prefill()` | range full (w128) | range split @32 |
|---|---|---|---|
| zerocopy OFF first (warm pool) | exact | exact | exact |
| zerocopy ON first (cold pool) | argmax 5968 vs 13, max_abs 8.2 | argmax 80300, max_abs 10.5 | exact (shapes already served) |

**Mechanism:** wgpu zero-initializes buffer memory *lazily*. The first wgpu pass that binds a never-initialized range gets a clear inserted ahead of it. The raw-Metal path blits its GEMM result into a CubeCL pool range behind wgpu's back. When that range is fresh pool memory, the next CubeCL kernel that reads it triggers the clear, and the result becomes zeros. A warm pool (every later prefill) reuses initialized pages. That is exactly Issue 670 T5's "corruption follows case ORDER, same-P warmup recovers", left open in 2026-08 because no e2e gain existed then. It is also why Bench 1009's G1 read exact: its harness ran warm-up passes first.

**Fix (`3bbf275`):** `MetalTensorZeroCopyGemm::ensure_wgpu_initialized`. The first time an output `(pool buffer, offset, bytes)` range is seen, one CubeCL fill is submitted ahead of the raw command buffer on the same queue, so wgpu marks the range initialized. Steady state is a set lookup. A proof-of-mechanism run with an unconditional fill was exact before the keyed version landed. The fill also exposed a latent `FillZerosCubeCL` defect: the 1D grid cap is 65535 workgroups, and a p=1024 FFN output is 69632 of them. The launcher now chunks over offset slices.

**After the fix**, with the ON arm cold and first: max_abs **0.000** in all three postures at P=512/w128, P=1024/w512, P=4096/w512. In the dist lane, the 16-token stream is identical to the incumbent at w128 and w512 (`.benchmarks/i1146_tensor/g1run/`).

## G2 — dist prefill, simdgroup (a) vs tensor zero-copy (b), interleaved per cell

`scripts/i1146_width_recurve.sh` with `I1146W_BIN_B`. The rig is the same as [Bench 919](919_issue1146_prefill_width_recurve.md). The b binary in run1 is the pre-fix opt-in build: same kernels, wrong only on cold-pool ranges. The fix adds a first-use fill only, and the post-fix pair reproduces the run1 ratio.

| cell | a (s) | b (s) | b vs a |
|---|---:|---:|---:|
| pp512 w128 (r1, r2) | 3.030 / 3.052 | 2.274 / 2.272 | **1.34×** |
| pp512 w512 (r1, r2) | 2.259 / 2.238 | 1.750 / 1.752 | **1.28×** |
| pp4096 w128 (r1, r2) | 25.644 / 25.618 | 19.971 / 20.000 | **1.28×** |
| pp4096 w512 (r1) | 18.440 | 15.265 | **1.21×** |
| pp4096 w512 (post-fix binary, g1run r1) | 18.469 | 15.214 | **1.21×** |

Compared with the shipping default (w128, simdgroup), w512 + tensor gives **pp512 3.04 → 1.75 s (1.74×, 292 tok/s)** and **pp4096 25.63 → 15.24 s (1.68×, 269 tok/s)**. Against llama.cpp's 1087 / 1046 tok/s (Bench 911), that is **0.27× / 0.26×**, up from 0.13 / 0.16×.

The gain shrinks as width grows (1.34× at w128 → 1.21× at w512). The tensor kernel's edge is largest where the simdgroup body is most decode-shaped. The two levers stack but are not independent.

## Box state and contamination (honesty)

- run1 (16:56–17:07): a sibling `perf_rematch.sh … prefill-pp16k` lane had processes on bmo from 16:58. Its own quiet-gate (an awk process scan) was **waiting on my cells**, not running the GPU. Every a-arm cell reproduces the clean Bench-919 values within ≤1% (pp4096 w128 25.62–25.64 vs 25.70; pp4096 w512 18.44–18.48 vs 18.40; pp512 w512 2.24–2.26 vs 2.19), so the run is read as uncontended.
- g1run (17:27–17:46) was forced past the quiet gate for the G1 cells (an empty-`LENS` default bug — fixed in `54f0dce` — ran the timing cells too). Its pp512 / pp4096-w128 cells overlapped a sibling `bench_809` pp16k run (a-arm pp512 9–17 s vs 3 s clean) and are **discarded**. Its pp4096 w512 pair (17:39–17:41) matches the clean values and is kept. The run was then killed to release bmo to the sibling lane.
- Not yet measured in dist: pp32768 with tensor. The engine measured the same kernel at pp32k = 1.163× ([riir-ai Bench 1009](../../riir-ai/.benchmarks/1009_641_ultra_window_t1_t2_t3_cells.md)). A confirming dist cell is queued for a quiet window (`RIIR_DIST_METAL_TENSOR=0` arms the A/B from one binary).

## Consequence for the league posture

The engine harness arms the same path, so any **cold** first prefill through it was wrong before `3bbf275`. Warmed league cells were unaffected, so the league numbers stand. The fix lives in `riir-infer-gpu`, the substrate riir-ai consumes, so it is repaired at the source. The riir-ai docs row for Issue 663/670 is updated to point here.

## Provenance

- Cells: `.benchmarks/i1146_tensor/run1/` (a sha `40647de5…`, b sha `42479ae0…`), `.benchmarks/i1146_tensor/g1run/`.
- Gate: `crates/riir-infer-gpu/tests/issue1146_tensor_range_g1.rs` (ignored and real-model; the zerocopy arm runs FIRST by construction).
- Commits: `cea3fba` (opt-in arm), `3bbf275` (zero-init fix + FillZeros grid chunking), `54f0dce` (promotion).
