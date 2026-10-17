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

Context correction (2026-10-17 ~09:2x): the artifact tree is **FluidInference's own published
coreml packages** (`FluidInference/clef-text-0.6b-coreml` @ `3c61a609…`, sha256-pinned at
T0.2/T1.2) — NOT our conversion. There is no in-house conversion lane to fix; the Swift
reference runtime (FluidUse `ClefTextManager`, the T2.1 capture) loads these same packages fine
under `cpuAndNeuralEngine`, and their published CPU row (200 ms @512, M5 Pro) implies a working
`.cpuOnly` load on their stack (newer Core ML, or an M5-specific planner). The defect is OUR
stack's CpuOnly planner on THIS package family.

1. **Single-function falsifier** — derive ONE single-function package (e.g. the L256 decoder
   function) via `coremltools` (a THROWAWAY derivative in `.raw/`, never committed, never a
   sha256-pin replacement) and load it CpuOnly. If it loads, the multi-function layout is the
   trigger; if it still balloons, the CPU planner chokes on the ANE-shaped graph itself.
2. **Version probe (cheapest, do FIRST)** — `xcrun coremlcompiler version` / macOS
   `(CoreML.framework)` build against the log; check FluidAudio/FluidUse issues + Apple release
   notes for a known CpuOnly multi-function regression. If a newer macOS/Core ML fixes it, the
   remedy is an OS update, not a fork.
3. **Per-bucket derivative tree** as the standing remedy IF (2) is dry: fork the packages into
   `Decoder.L<bucket>.mlpackage` derivatives (a deliberate trust-boundary change: our own
   sha256 pins would have to move onto the DERIVED tree with the provenance recorded — an
   owner-visible decision, the upstream-trust law at stake).
4. **Upstream report** — FluidAudio/FluidUse issue + Apple Feedback with the kernel-log lines
   and the posture differential.
5. If 1–4 all fail: close the cpu row **NOT-MEASURABLE-HERE** honestly — their published
   CPU-only decoder@512 = 200 ms (M5 Pro, HF card / Research 012 §Perf) stays the external
   reference floor; the ANE + GPU rows stand as our measurements (fresh re-validation today:
   see the differential table above).

## Non-goals

- No retry-on-a-quieter-box posture: the footprint math (190 GB vs 73 GB RAM+swap) closes that
  door by construction.
- No committed derivative packages: any coremltools reconversion is a THROWAWAY in `.raw/`
  (the §0.5 law); promoting a derived tree over the sha256-pinned upstream one is remedy-3, an
  owner-visible trust-boundary change, never a side effect of debugging.
