# Issue 048 — clef CpuOnly posture balloons to ~192 GB at model load (jetsam SIGKILL): the plan-619 T2.3 cpu-row blocker is a Core ML planner defect, not box pressure

**Status:** OPEN — filed 2026-10-17 ~09:0x +07 (M3 idle lane; the T2.3 blocker re-diagnosed with kernel-log evidence; bench 884 §CPU's box-pressure diagnosis REFUTED)
**Claimed:** zed-m3-idle-1017a, 1792220000, expires 2026-10-19T10:00:00+00:00 — remedy 1 (the single-function falsifier via coremltools in `.raw/`, throwaway) + remedy 4 prep (upstream report draft); remedies 3/5 stay owner-gated

## The defect

`clef_text_timing -- cpu` (the CpuOnly posture of the clef-text-0.6b lane) is killed by macOS
jetsam during `ClefText::load_from` — before the `CLEF-TIMING` banner — with a process footprint
of **~192–194 GB** (kernel `memorystatus` evidence below). The box has 64 GB RAM + 9 GB swap; no
box state can satisfy a 190 GB load-time footprint. The kill is **unconditional**, reproducible
on a quiet box (load 7.4, 0 rustc/cargo, 31 GB free RAM at the 08:5x attempts).

Kernel log (24-h census, `log show` predicate `killing largest compressed process clef_text_timing`):
**14 jetsam kills, footprints 192,577–204,183 MB** — the four overnight 03:xx attempts
(197.7–204.2 GB), today's 08:52/08:55 pair (192.6/194.7 GB), and eight more earlier kills the
overnight record never counted. Same defect every time:

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
   remedy is an OS update, not a fork. Probe result (2026-10-17): macOS 26.6.2 (25G83),
   `coremlcompiler` 3600.25.1. External context (Apple Developer Forums, Core ML tag, searched
   same day): a **cluster of multi-function ML Program + load-time-memory regressions across
   the 27 series** — "macOS 27.0 multifunction ML Program recognized but fails to load" (the
   multi-function loading path actively regressing), "iOS 27 ML Program pipeline allocates
   ~2.8 GB at load" (load-time memory explosion; there `.cpuOnly` was the only working posture
   — the inverse of ours), "Core ML memory dramatically higher with Xcode 27 builds" (NE
   residency accounting). None is our exact signature (CpuOnly planner balloon on an ANE-first
   multi-function transformer), but the path we hit sits in an active Apple bug cluster — an
   OS-update remedy is plausible and the upstream report (4) lands in a receptive window.
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

## Evidence addendum (2026-10-17 ~10:2x +07, remedy-1 falsifier lane — the kill site CORRECTED: it is the HEAD, not the Decoder)

Ran remedies 1 + 2's falsifier ladder on this box (throwaway rig in `.raw/clef048_falsifier/`,
never committed; coremltools 9.0 in a uv venv + a Swift harness + a step-printing objc2 probe
example). Every cell, one BALLOON cell each (jetsam confirmed in the kernel log each time):

| # | form | host | functionName | posture | result |
|---|---|---|---|---|---|
| 1 | source Decoder.mlpackage (multi-fn) | Python/coremltools | none | CpuOnly | LOAD OK 6.87 s |
| 2 | derived **Decoder.L256** single-fn package | Python | none | CpuOnly | LOAD OK 2.65 s + PREDICT 0.16 s |
| 3 | source Decoder.mlpackage (multi-fn) | Python | L256 | CpuOnly | LOAD OK 7.27 s |
| 4 | native `compileModelAtURL` product of the Decoder | Swift | L256 | CpuOnly | LOAD OK 4.71 s + PREDICT 0.14 s |
| 5 | the live cache bundle `mlmodelc-1d66…` (what the kills loaded) | Swift | L256 | CpuOnly | LOAD OK 4.76 s + PREDICT 0.14 s |
| 6 | the same cache bundle | **Rust/objc2 probe** | L256 | CpuOnly | LOAD OK 4.94 s (clean) |
| 7 | **Head cache bundle `mlmodelc-72ed…`** | **Rust/objc2 probe** | L256 | CpuOnly | **BALLOON → jetsam SIGKILL 202,249 MB (kernel log, 10:09:55)** |
| 8 | the same head bundle | Swift | L256 | CpuOnly | BALLOON (killed under the 5-min wall, load never returned) |
| 9 | derived **Head.L256** single-fn package (compileModelAtURL product) | Rust/objc2 probe | L256 | CpuOnly | **BALLOON → jetsam SIGKILL 202,363 MB (10:21:56)** |
| 10 | a fresh full native example run (`clef_text_timing -- cpu`) | Rust | L256 | CpuOnly | BALLOON → SIGKILL 202,183 MB (10:04:43) — reproducibility re-confirmed same-day |

**Findings:**

- **The kill site is the HEAD model load, not the Decoder.** Every native cpu run dies on the
  second load (`Head.mlpackage`) — the decoder loads fine first (cells 1–6). The issue's
  original "kill site = Decoder.mlpackage load" premise was an inference from load order, not
  a measurement (the lane prints nothing between the two loads; the banner only prints after
  both). The ~90 s window is the balloon's allocation storm, not the load itself (a healthy
  head load is sub-second: the ane posture loads decoder+head in 1.74 s total).
- **The balloon is language-host-independent**: Rust/objc2 AND Swift both balloon on the head
  (cells 7/8) — the earlier Swift-vs-Rust differential existed only because every early cell
  loaded the DECODER.
- **Remedy 3 (per-bucket derivative tree) is DEAD for the head**: a derived single-function
  `Head.L256.mlpackage` balloons identically (cell 9, 202,363 MB). The trigger is the head
  GRAPH under the CpuOnly planner, not the multi-function layout — so a derivative fork would
  not even buy the cpu row, and the trust-boundary change is moot. (The decoder's derivative
  works, cells 2/6 — but the decoder never needed it.)
- The head graph: 4 functions (L256–L2048), 709 ops each (22 layer_norm, 61 linear, 31 matmul,
  10 softmax, 391 const; CoreML8 block specializations; io e.g. `states [256,1024] fp16 →
  logits [96]`). Nothing in the shapes suggests a 202 GB footprint — planner defect, full stop.
- objc2-core-ml 0.3.2's `MLComputeUnits::CPUOnly = Self(0)` matches Swift's
  `.cpuOnly.rawValue = 0` byte-for-byte (checked both) — the host's config plumbing is NOT a
  suspect; cells 6+7 share it and diverge only on the bundle.

**Remedy consequences (the owner call this addendum sets up):**

- **Option A (the unblock, no trust change): the SPLIT-posture cpu row.** The timing example
  already times decode and head as separate legs over separate `MLModel`s; the DECODER loads
  and predicts fine CpuOnly (cells 4/5/6 + the 0.14–0.16 s L256 predict = the decode leg is
  honestly measurable cpu). Load the head in its working `CpuAndNe` posture even on the cpu
  row, report `decode` as the cpu-leg number and `head/total` labeled mixed
  (head-cpu = NOT-MEASURABLE-HERE; their 200 ms @512 external floor stays the head reference).
- **Option B (remedy 5 as written): close the whole cpu row NOT-MEASURABLE-HERE.**
- Remedy 3 is off the table (above); remedy 2's OS-update watch stays armed (the 27-series
  Apple bug cluster may fix the head graph's planner path).
- Remedy 4 (upstream report) is now DRAFTABLE with a sharp signature: "Head.mlpackage
  (0.6B clef-text student's classification head, fp16, multi-function ML Program, CoreML8
  blocks) → MLModel load with `.cpuOnly` allocates ~202 GB and is jetsam-killed on macOS
  26.6.2 (25G83); same package loads fine under `.cpuAndNeuralEngine`/`.cpuAndGPU`; a
  single-function derivative of the same graph balloons identically; the sibling decoder
  package (same authoring, same layout) loads fine CpuOnly." Filing is an owner act (public
  post under the owner's identity).

## Non-goals

- No retry-on-a-quieter-box posture: the footprint math (190 GB vs 73 GB RAM+swap) closes that
  door by construction.
- No committed derivative packages: any coremltools reconversion is a THROWAWAY in `.raw/`
  (the §0.5 law); promoting a derived tree over the sha256-pinned upstream one is remedy-3, an
  owner-visible trust-boundary change, never a side effect of debugging.
