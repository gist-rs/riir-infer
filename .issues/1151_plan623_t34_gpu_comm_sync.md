# Issue 1151 — plan-623 T3.4: GPU↔comm sync — signal-driven readback+send on the decode path

**Status:** OPEN — unit of [Plan 623](../.plans/623_multi_host_pipeline_inference_lane.md) T3.4 (file-level claim lives HERE; the plan file is held by the sibling T3.2/T3.3 claim, one-claim-per-file per the P92 grammar).
**Claimed:** zed-m3-i623-t34gpucomm, 1793255400, expires 2026-10-31T22:30:00+00:00
**Date:** 2026-10-29
**Type:** optimization (multi-host pipeline serving integration)

## The task (plan text)

> **T3.4** GPU↔comm sync: CPU comm thread on `MTLSharedEvent` signal values; no per-
> layer command-buffer commit on the critical path.

## The substrate reality (design notes, measured-where-stated)

The plan text was written against the raw-Metal design assumption; the shipped
substrate is CubeCL-over-wgpu (the shared `init_setup` device/queue, Issue 657),
the same correction class as T3.1's "IPv4-mapped GID" premise. Mechanism facts
read this session:

- CubeCL's wgpu stream batches kernel dispatches into ONE command encoder,
  flushing at `tasks_max` (default 32, `CUBECL_WGPU_MAX_TASKS`) or on
  read/write boundaries — a "per-layer command-buffer commit" does not exist
  per kernel; the commit cadence is the encoder flush.
- `ScheduleTask::Write` (uniform writes) **flushes before every write** — the
  queue-level `write_buffer` cuts in front of encoder work, so value-carrying
  uniforms that miss the Issue-761 metadata cache force flush storms. This is
  the closest live shape to "per-layer commit on the critical path" — measure
  first (`SUBMIT_COUNT`/`FLUSH_COUNT`/`ENQUEUE_WRITE_COUNT` atomics ship in the
  vendored fork, Issue-761 profiling), fix only what the instrument shows.
- `read_one` = staging copy encode + flush + **blocking map wait** — the head's
  inline `step()` serializes: enqueue chain i → block on GPU → readback → send
  → enqueue chain i+1. The GPU idles during each host readback+send.
- `client.read_async` + `cubecl_environment::future::block_on` are the
  completion-signal surface: a read future CREATED right after chain i's
  enqueue encodes its staging copy FIFO-after chain i (before chain i+1), so
  the future resolving IS the "GPU done with chain i" signal value —
  MTLSharedEvent's role in the plan text, delivered through the wgpu/cubecl
  substrate. A literal `MTLSharedEvent` arm is an optional follow-up only if
  the future path shows overhead (raw-hal access precedent: Issue 994).

## The shape (T3.4 deliverable)

1. `riir-infer-gpu`: re-export `{block_on, DynFut}` (the crate already
   direct-deps `cubecl-environment` for the umbrella-shadow workaround);
   `read_hidden_state_fut()` — the staged-copy future, created at enqueue time.
2. `riir-infer-dist` `TernaryGpuStage`: `pending_read` slot;
   `schedule_async_finish()` after `forward_range_dispatch` (creates the
   future — the signal arms); `step_finish` consumes it via `block_on` when
   present, else the legacy sync read. Same kernel sequence, same reads —
   scheduling only.
3. `StageCompute::schedule_async_finish()` (default no-op — the CPU stage
   unchanged); `run_waves` decode arm: step_submit + schedule per stream,
   then finish→encode→send in enqueue order after the loop (prefill frames
   still send inline; outstanding pushed in exact send order — the tail's
   FIFO reply order is unchanged).
4. Demote-only env `RIIR_DIST_ASYNC_FINISH=0` → byte-identical incumbent
   cadence (the repo's env convention).
5. G1: bit-identity (scheduling-only change — same dispatches, FIFO-identical
   chains). G2: 2-box (or 1-box loopback) decode A/B at s=1 and s=2 — the win
   hypothesis is reclaiming the host readback+send gap under the next chain at
   s≥2; s=1 must be no-regression. G4: no new steady-state alloc (the future
   is one boxed allocation per step — same class as the existing DynFut use).

## Threads NOT pulled into this unit

- A literal comm THREAD (network I/O off the main thread) — needs the stage
  state split across threads (Send+Sync surface); only worth it if the T3.5
  scoreboard shows host-side serialization still binding after this lane.
- The tail's logits readback (993 KB) — same primitive applies; adopt there
  only with its own measurement.
- The uniform-write flush-storm fix — measure (the Issue-761 atomics) in this
  unit's bench pass; fix in a follow-up only if the instrument shows a storm.

## Resolution

(landing note goes here)
