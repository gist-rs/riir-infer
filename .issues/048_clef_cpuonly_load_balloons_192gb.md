# Issue 048 — clef CpuOnly posture balloons to ~192 GB at model load (jetsam SIGKILL): the plan-619 T2.3 cpu-row blocker is a Core ML planner defect, not box pressure

**Status:** OPEN — filed 2026-10-17 ~09:0x +07 (M3 idle lane; the T2.3 blocker re-diagnosed with kernel-log evidence; bench 884 §CPU's box-pressure diagnosis REFUTED)

## The defect

`clef_text_timing -- cpu` (the CpuOnly posture of the clef-text-0.6b lane) is killed by macOS
jetsam during `ClefText::load_from` — before the `CLEF-TIMING` banner — with a process footprint
of **~192–194 GB** (kernel `memorystatus` evidence below). The box has 64 GB RAM + 9 GB swap; no
box state can satisfy a 190 GB load-time footprint. The kill is **unconditional**, reproducible
on a quiet box (load 7.4, 0 rustc/cargo, 31 GB free RAM at the 08:5x attempts).

Kernel log (both of today's attempts; the four 2026-10-17 03:1x–03:5x attempts recorded in
`.benchmarks/884_plan619_clef_text_ane_latency.md` §CPU are the same defect, misattributed):

```
08:52:14 kernel: memorystatus: killing largest compressed process clef_text_timing [84104] 192577 MB
08:55:04 kernel: memorystatus: killing largest compressed process clef_text_timing [85736] 194688 MB
```

Exit shape: `PIPE_EXIT=137` (SIGKILL), no banner, no ROW — the kill site is the Core ML
load/specialization of the **Decoder.mlpackage** (848 MB, multi-function: one function per
config bucket `L256/L512/L1024/L2048`, fp16, ~0.6B params) under
`MLModelConfiguration.computeUnits = .cpuOnly`.

## The posture differential (same box, same hour, same cached compile)

| posture | compute units | load | result |
|---|---|---|---|
| `ane` | CpuAndNe | **1.74 s** | ROW prints: decode 32.39 / head 3.00 / total 35.39 / answer 35.90 ms @L256 — consistent with the published 35.2 ms row |
| `gpu` | CpuAndGpu | **3.11 s** | ROW prints: decode 35.70 / head 4.25 / total 39.95 / answer 39.98 ms @L256 |
| `cpu` | CpuOnly | **killed ~90 s in** | ~192 GB balloon → jetsam SIGKILL, banner never prints |

The compile cache is posture-independent (key = artifact digest,
`~/Library/Caches/riir-infer/clef-ane/mlmodelc-*`) — all three postures load the SAME compiled
bundle, so the explosion is in Core ML's **CPU-planner specialization at `MLModel` load**, not in
compilation. ~190 GB ≈ 230× the package size; plausibly the CPU planner enumerating
specializations across the multi-function graph. The package was converted ANE-first
(the plan 619 T1 conversion lane); an ANE-shaped graph hitting a pathological CPU-planner path
is a known Core ML issue class.

## Repro

```sh
cargo run --release -p riir-infer-clef --features clef-ane --example clef_text_timing -- cpu --reps 1 --buckets 256
# → PIPE_EXIT=137, ~90 s, banner never prints; kernel log names the footprint
cargo run --release -p riir-infer-clef --features clef-ane --example clef_text_timing -- ane --reps 1 --buckets 256
# → exit 0, load 1.7 s, ROW prints (the control)
```

## What this corrects

- `.benchmarks/884` §CPU: "blocked by box memory pressure … even one bucket pair died, so the
  blocker is the box, not the instrument" — **REFUTED**. The blocker is the instrument's CpuOnly
  load path; the box-pressure framing came from the kills happening to coincide with sibling
  load bursts (and from swapping `exit=$?` through a pipe, which masked SIGKILL as 0).
- My own session characterization (2026-10-17 ~08:2x: "the load-phase jetsam risk that motivated
  the swap bar is absent") — also refuted; the risk is unconditional, not box-state.

## Remedies (in cost order)

1. **Single-function falsifier** — reconvert ONE bucket (e.g. L256 decoder) into its own
   single-function `.mlpackage` (coremltools, the tree's original Python build lane) and load it
   CpuOnly. If it loads, the multi-function package is the trigger and a per-bucket split is the
   fix; if it still balloons, the CPU planner chokes on the ANE-shaped graph itself.
2. **Per-bucket package split** as the standing remedy: ship `Decoder.L<bucket>.mlpackage` per
   bucket (smaller packages, posture-independent win: the GB-scale PlanWalk::Skip workaround
   likely retires too). Wire cost: the loaders' per-function path already keys on `functionName`.
3. **Upstream Core ML feedback** with the kernel-log lines + the posture differential.
4. If 1–3 all fail: close the cpu row **NOT-MEASURABLE-HERE** honestly — their published
   CPU-only decoder@512 = 200 ms (M5 Pro, HF card / Research 012 §Perf) stays the external
   reference floor; the ANE + GPU rows stand as our measurements (fresh re-validation today:
   see the differential table above).

## Non-goals

- No retry-on-a-quieter-box posture: the footprint math (190 GB vs 73 GB RAM+swap) closes that
  door by construction.
- No Python dependency added to the lane: the reconversion rides the existing Python build lane
  that produced the artifact tree (measurement/conversion-lane subprocess oracle per repo law).
