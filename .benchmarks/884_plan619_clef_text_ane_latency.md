# 884 — Plan 619 T2.3: clef-text-0.6b ANE lane latency (the G2 gate)

**Status:** MEASURED 2026-10-17 — ane + gpu postures complete (both under PASSED preflights at the documented raised ceiling); **cpu posture MEASURED 2026-10-18 via the issue-048 SPLIT (decoder CpuOnly + head CpuAndNe, disclosed; head-cpu NOT-MEASURABLE-HERE — the head's CpuOnly load balloons ~202 GB, a Core ML planner defect; issue 048 resolved, its file removed in the follow-up commit)**. G1 = `4e7e6c1` parity BIT-EXACT (544/544 logits, 0 top-1 disagreements, 102 fixtures) · G2 = this record · G3 = opt-in (`clef-ane` never default, default build compiles empty) · G4 = disclosure (the CoreML host path allocates MLMultiArray inputs — the laya-ane posture; never claimed alloc-free).

## Box state (the preflight law — quoted beside every number)

- `PROVENANCE: power=AC Power load=7.64 swap=7146.38M canary=skipped powermode=2(high)` — the **ane** run (PASSED under `MAX_LOAD=8`, the script's documented ceiling raise; the default ceiling 6.0 was exceeded by sibling sessions, disclosed here rather than waited out because the load class is quoted WITH the numbers).
- `PROVENANCE: power=AC Power load=7.64 swap=7146.38M canary=skipped powermode=2(high)` — the **gpu** run (same posture window).
- **cpu (2026-10-18, the issue-048 SPLIT row):** `PROVENANCE: power=AC Power load=5.62 swap=7134.62M canary=skipped powermode=2(high)` — preflight PASSED immediately before the runs (MAX_LOAD=8 ceiling); the ane control + both cpu subset runs completed inside that quiet window. A post-run re-check REFUSED at load 12.74 (a sibling burst arriving after the window closed) — disclosed: the L2048 cpu p90 tail may carry the arriving burst's noise; p50 is the quoted metric. The old `cpu: PENDING` bullet (load 36.09 refusal, 2026-10-17) described a box-pressure explanation the falsifier ladder later refuted — the blocker was always the CpuOnly head-load defect (issue 048).
- Toolchain: release profile, rustc 1.98.1 (the pinned toolchain), macOS 26.6.2, Apple M3 Max (16 cores), artifacts `3c61a60954478ef385aa42d67d5764642dcc9135` (the T1.2 sha256-pinned tree; the SAME tree the T2.2 parity gate proved bit-exact against the official FluidUse runtime).
- Placement disclosure: every bucket prints `not-measured` — the CoreML compute-plan walk never completes at load on the GB-scale multi-function packages (measured, T1.6); the placement TRUTH is the T2.2 parity + this table (the same artifacts at `cpuAndNeuralEngine` reproduce the official ANE runtime bit-for-bit — 544/544 logits).

## The table (p50 ms, warm=3 reps=20, one posture per process)

Run: `cargo run --release -p riir-infer-clef --features clef-ane --example clef_text_timing -- <ane|gpu|cpu> --reps 20`

### ANE (`cpuAndNeuralEngine` — the serving posture)

| bucket | record tokens | decode p50 | decode p90 | head p50 | head p90 | total p50 | total p90 |
|---|---|---|---|---|---|---|---|
| 256 | 141 | 32.27 | 32.39 | 2.95 | 3.00 | 35.22 | 35.37 |
| 512 | 393 | 82.60 | 82.66 | 4.00 | 4.07 | 86.60 | 86.71 |
| 1024 | 918 | 236.34 | 236.63 | 6.48 | 6.52 | 242.83 | 243.27 |
| 2048 | 1926 | 735.74 | 738.31 | 10.87 | 11.48 | 746.59 | 749.14 |

Whole-pipeline `answer()` on a short record (→ L256): **35.4 ms p50** — encode + bucket select + decode + head + softmax, the user-facing cost of a short decision.

### GPU (`cpuAndGPU`)

| bucket | record tokens | decode p50 | decode p90 | head p50 | head p90 | total p50 | total p90 |
|---|---|---|---|---|---|---|---|
| 256 | 141 | 34.27 | 34.81 | 3.89 | 4.07 | 38.16 | 38.86 |
| 512 | 393 | 68.49 | 68.92 | 4.52 | 4.73 | 73.07 | 73.44 |
| 1024 | 918 | 148.57 | 149.83 | 5.91 | 6.15 | 154.39 | 155.95 |
| 2048 | 1926 | 399.19 | 419.39 | 9.76 | 10.39 | 409.34 | 429.95 |

Whole-pipeline `answer()` (short record → L256): 38.6 ms p50.

### CPU (`cpuOnly`)

**BLOCKED BY A REAL DEFECT — issue 048, NOT box pressure.** The 2026-10-17 03:1x–03:5x jetsam
kills (four attempts, exit 137, ~2m42s in, during the load phase) and two more the same morning
(08:52 + 08:55) are ONE unconditional defect: the **CpuOnly** load balloons the process to
**~193–204 GB** (kernel `memorystatus:
killing largest compressed process clef_text_timing …` — 14 kills in 24 h: the four overnight
03:xx at 197.7–204.2 GB, the two 08:5x at 192.6/194.7 GB) — vs 64 GB RAM + 9 GB swap; no box
state can satisfy it. Reproduced on a QUIET box (load 7.4, 0 rustc/cargo, 31 GB free RAM). The
earlier "box memory pressure" diagnosis on this line is REFUTED; the swap gate below guarded
against a threat that was never the blocker. (The four 03:xx kills were recorded then as
"≈2m42s in" and attributed to sibling load bursts — same defect, wrong attribution.)

**Posture differential (2026-10-17 ~09:0x +07, same box, same cached compile):** `ane` loads
1.74 s → ROW 32.39/3.00/35.39/35.90 ms @L256 (matches the published row — bonus re-validation);
`gpu` loads 3.11 s → ROW 35.70/4.25/39.95/39.98 ms @L256; `cpu` jetsam-killed at ~192 GB.
The compile cache is posture-independent, so the explosion is Core ML's CPU-planner
specialization at MLModel load. Full evidence + remedies: `.issues/048_clef_cpuonly_load_balloons_192gb.md`.

**Kill site CORRECTED (2026-10-17 ~10:2x, the 048 falsifier ladder):** it is the **HEAD**
load, not the Decoder — the decoder loads + predicts fine CpuOnly in every host (Rust probe
4.94 s; Python 2.65–7.27 s; Swift 4.71 s), and the balloon reproduces on the head in BOTH
Rust and Swift (202,249 / killed-under-wall), including on a derived single-function
`Head.L256` package (202,363 MB — so the per-bucket split remedy is DEAD for the head; the
differential table above inherited the wrong kill site from load order, never a measurement).
**LANDED 2026-10-18 — Option A taken: the SPLIT-posture cpu row.** The runtime now loads the
decoder CpuOnly and the head CpuAndNe on the cpu posture, disclosed at load (per-bucket
placement line, `info()`, the example's banner + NOTE). The decode column is the honest
cpu number; head/total/answer are MIXED (the head leg is CpuAndNe); **head-cpu is
NOT-MEASURABLE-HERE** — their published CPU-only head number stays the external reference.
Re-measured under a PASSED preflight (see Box state); the ane control re-ran in the same
window and reproduced the published row (L2048: 735.72/10.94/746.69 vs published
735.74/10.87/746.59 — the unsplit path is byte-identical in behavior).

| bucket | record tokens | decode p50 | decode p90 | head p50 | head p90 | total p50 | total p90 |
|---|---|---|---|---|---|---|---|
| 256 | 141 | 106.96 | 122.86 | 7.20 | 7.36 | 112.90 | 130.22 |
| 512 | 393 | 211.67 | 234.96 | 9.35 | 9.71 | 220.92 | 244.39 |
| 1024 | 918 | 501.65 | 513.46 | 13.11 | 13.24 | 514.85 | 526.57 |
| 2048 | 1926 | 1343.90 | 1646.86 | 18.57 | 19.38 | 1362.57 | 1658.58 |

Whole-pipeline `answer()` (short record → L256): **112.93 ms p50** — MIXED (encode + the
CpuOnly decode + the CpuAndNe head + softmax). Column labels per bucket:
`decode = CpuOnly (the honest cpu leg) · head/total = MIXED`.

- The decode leg's chip story lands clean: ours 211.7 ms @L512 vs their published
  CPU-only decoder @512 = 200 ms (M5 Pro) — **1.06×**, the same chip-generation ratio the
  ANE rows show (1.21–1.26×), now confirmed on the cpu leg too.
- Head-cpu remains NOT-MEASURABLE-HERE pending a Core ML fix (the Apple 27-series
  multi-function/load-memory regression cluster is the re-test trigger — issue 048's
  remedy-2 watch; the issue file itself is removed with this landing, the record lives in
  this section + git history).

## Their published rows beside ours — M5 Pro (the chip difference is DISCLOSED, never pooled)

From the HF card / Research 012 §Perf (their benchmark, their hardware):

| cell | theirs (M5 Pro) | ours (M3 Max) | ratio |
|---|---|---|---|
| ANE decoder+head @L256 | 29 ms | 35.2 ms | 1.21× |
| ANE decoder+head @L512 | 69 ms | 86.6 ms | 1.25× |
| ANE decoder+head @L1024 | 196 ms | 242.8 ms | 1.24× |
| ANE decoder+head @L2048 | 591 ms | 746.6 ms | 1.26× |
| GPU decoder @L512 | 35 ms | 68.5 ms | 1.96× (see the load caveat) |
| CPU-only decoder @L512 | 200 ms | 211.7 ms (issue-048 split row, decode leg CpuOnly) | 1.06× |

- The ANE ratio is **consistent ~1.21–1.26× across all four buckets** — a clean chip-generation story (M5 Pro's newer ANE), not a host overhead smell (which would NOT be bucket-flat; our host adds the gather + packing + softmax legs their manager also pays — the parity gate proves the math is identical, so the delta is the silicon).
- The GPU cell carries a load caveat: ours was taken at load 7.64 with GPU-contention risk (Unity/Zed exempt by owner rule, but sibling GPU jobs were not exclusivity-checked for this posture). The ANE posture is the serving posture and is the number this lane quotes; the GPU row is context.
- Their "3-question 370-tok ticket: 77 ms ANE" cell maps to our L512 row (a 339–393-tok record lands L512): ours ≈ 86.6 ms total → **~1.13×** at a token-comparable cell.

## Method + disclosure

- One posture per process (compute units are baked per `MLModelConfiguration` at load — never a mid-process switch).
- Records: sentence-repeated states landed in each bucket's band `[bucket−96, bucket−8]`; the decode cost is the bucket's padded length L (static CoreML shapes), so any n in the band costs the same. 1 noul question per record.
- Legs timed separately around the SAME forward: `decode_forward` (gather + cached RoPE + mask + fp16 predict + f32 widen) vs `head_forward` (packing + f32 predict + slice); `total = decode + head` (the fix of the elapsed-at-push-time bug — the first ane run's totals printed 0.00 and were discarded, never recorded anywhere).
- Load cost: first ane run paid 203.6 s loading 8 models (ANE specialization, cold-ish); the fixed-instrument rerun loaded in 5.7 s (warm). The table numbers are post-warm (3 warm reps discarded per bucket).
- The `answer()` readout uses a SHORT record, so its flat ~35 ms is the L256 whole-pipeline cost, not a per-bucket answer cost — per-bucket whole costs are the total column + ~1 ms encode.
- G4: the host path allocates (`MLMultiArray` inputs, the strided reads) — same posture as the laya ANE lane; never claimed alloc-free. Per-decision allocations are bounded and posture-independent of the parity result.
- Post-bench check: **the parity gate (`4e7e6c1`) is the correctness authority for every number above** — the same binaries' forward path is bit-exact vs the official runtime, so these latencies are the latency of the RIGHT answer, not of an approximation.
