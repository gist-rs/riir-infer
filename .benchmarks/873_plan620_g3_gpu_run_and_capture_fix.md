# Bench 873: Plan 620 — the G3 GPU run LANDED (the capture-window fix + the base-arm cross-check PASS; si FAILs reproduce, no promotion)

**Status:** RECORD — EXECUTED 2026-10-13 02:20–02:36 (4090, schtask `si620g3fix`,
DONE-EXIT-0). Two results in one run: **(1) the G3 GPU harness now WORKS end to end**
(~8 min for all three arms vs the CPU lane's 12h48m — the FAST re-gate lane plan 620
built is operational); **(2) the cross-check vs the CPU base arm (bench 936) PASSES at
0.07%** — and the si/si-cal ppl FAILs REPRODUCE on the GPU lane, so the no-promotion
verdict stands on both lanes independently.

## The root-cause fix (the b51769c blocker, resolved)

The G3 GPU run's `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` on the first captured launch
was **not si-gated** (b51769c's attribution) — it was the **constructor**: the harness
built the forward via `TernaryDeltanetGpuForwardCudarc::new` (the eager reference lane)
while every graph-lane consumer must use **`new_graph_ready`**. cudarc's multi-stream
event tracking (default ON; `new_graph_ready` calls `disable_event_tracking()`) collects
per-arg wait/record event sets on every kernel launch (`is_managing_stream_synchronization()`
→ `record_event`/`wait` pairs in cudarc's launch path), and the event pair inside the
capture window dies with STREAM_CAPTURE_UNSUPPORTED. The flag0 probe (bench 872) captured
the same body green because it used `new_graph_ready` — the "si-gated" reading was the
constructor delta wearing an si costume. Fix: one line in
`src/bin/si_zero_shot_ppl_gpu.rs` (`new` → `new_graph_ready` + the mechanism comment);
the base arm's bytes are unchanged (bench 872 pinned the graph lane bit-identical at
exactly this construction).

## Box state (provenance — Issue-021 law)

| Axis | Value |
|---|---|
| GPU | RTX 4090 24 GB (sm_89, Ada), KB5121794 post-era |
| GPU util pre-run | 4% / 376 MiB (desktop only) |
| Build | release, `E:\tmp\tgt-si620post` (warm), worktree `E:\git\wt-si620post` @ `fd16ab6` + the one-line harness fix scp'd |
| Features | `cuda_graphs_forward,riir-infer-core/bonsai2_hadamard,scale_invariant_attn` |
| Checkpoint | `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7,206,168,896 bytes) |
| Calibration | `E:\tmp\si622-cal-4090.bin`, 1536 slots, digest `95ef…` (the CPU lane's own sidecar, reproduced on-box at b51769c — SAME digest both lanes) |
| Config | `--arms base,si,si-cal --max-length 2048 --min-length 128 --rungs 16 --score-window 128 --tolerance 0.05 --tau 10.0 --retrieval`; corpus wikitext2_test 2049 tokens (BOS incl.) |
| Wall | 16:00 min total (load 209.6s + forward-up 37.2s + base 159.0s + si 159.4s + si-cal + retrieval probe) |

## The G3 table (GPU lane)

| arm | pooled ppl | worst Δppl/ppl_base | verdict @ 5% |
|---|---|---|---|
| base | **7.2499** | 0.0000 (reference) | — |
| si | 10.5846 | 1.4051 | **FAIL** |
| si-cal | 11.4233 | 1.3350 | **FAIL** |

Retrieval probe (needle-at-head, question-at-tail, 6 greedy): base 0/4 hits; **si 3/4**
(zebra HIT at both lengths + river HIT at 1024); si-cal 2/4 (zebra both lengths). The
needle-vs-bulk trade REPRODUCES the CPU lane's disclosure shape (bench 936: si 3/6
zebra vs base 0/2) — a pattern note, never a claim; the ppl FAIL is the pre-registered
authority.

## The base-arm cross-check (the plan's validation bar) — PASS

| arm | CPU (bench 936, 12h48m) | GPU (this run) | delta |
|---|---|---|---|
| base pooled ppl | 7.2449 | **7.2499** | **+0.069%** |
| si worst Δ | +1.3229 FAIL | +1.4051 FAIL | same verdict |
| si-cal worst Δ | +1.3290 FAIL | +1.3350 FAIL | same verdict |
| retrieval shape | si 3/6 vs base 0/2 | si 3/4 vs base 0/4 | same shape |

The 0.07% base delta is well inside the f32-posture envelope (the plan's bar: "a
divergence beyond that indicts the instrument, not the checkpoint — Issue-022 T3.1
law"); the si FAILs reproduce within the same envelope. **The verdict is now
two-lane-consistent: no si promotion** (the zero-shot pairing fails on both lanes);
the τ-sweep question ("does ANY τ admit a pass, or trained-only?") is the GPU lane's
next enabler — now runnable at ~8 min/arm instead of 12h48m.

## Consequence

1. Plan 620's box work: Phase A (bench 872, three-part GREEN) + the G3 GPU re-run +
   cross-check (this bench) are DONE. Remaining: Phase D's G2 prefill cells
   (pp2048/tg128 ±3% via the league harnesses) — a separate unit per the plan.
2. Plan 622's τ-sweep leg is UNBLOCKED (the GPU lane is operational).
3. The harness constructor law is now recorded: **a graph-lane consumer constructs via
   `new_graph_ready` — `new()` is the eager reference lane and its event tracking
   cannot cross a capture window.**

## Refs

- plan 620 (`riir-infer/.plans/620_cuda_si_hook_and_gpu_g3_lane.md`)
- bench 872 (the Phase-A flag-0 three-part gate) · bench 936's rows via plan 622
  (`katgpt-rs/.plans/622_scale_invariant_attn.md`, the CPU lane record)
- b51769c (the first-exercise findings this bench corrects: si-gated → constructor)
- `src/bin/si_zero_shot_ppl_gpu.rs` (the fix) · `E:\tmp\si620-g3gpu-fix.log` (the run)
