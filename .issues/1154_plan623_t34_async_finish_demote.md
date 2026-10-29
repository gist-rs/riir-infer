# Issue 1154 — plan-623 T3.4 residue: demote the async-finish arm's wire default (ABBA-refuted)

**Status:** OPEN — claimed by the T3.5 lane in the same session; the one-line flip + suite re-run rides the Bench-929 landing commit series (kept a separate unit per the promotion discipline: a default-posture change is a code+gate act, never a bench-record rider).
**Claimed:** zed-m3-i971-housekeeping, 1793262200, expires 2026-10-31T23:50:00+00:00
**Date:** 2026-10-29
**Type:** demotion (default-posture flip; the arm stays opt-in)

## The evidence (Bench 929 Row 2, ABBA ×2, amo⇄bmo TB wire, s=2, ±1% spread)

Armed decode window 51.24/51.38 tok/s vs demoted 99.99/99.30 — **−48.5%**. The deferred finish→send drain serializes each pass (both head chains submit before any activation leaves; the two GPUs never overlap); the incumbent inline cadence IS the pipelining. The T3.4 landing's +5% was measured on the M3 loopback rig where both stages share one queue — no cross-GPU pipeline exists there to forfeit. s=1: wash (±0.6%).

## The change

`run_waves`' all-feed deferral gate: default the T3.4 arm OFF for the serving posture — `schedule_async_finish` arms only when `RIIR_DIST_ASYNC_FINISH=1` (the env flips from demote-only to promote-only; exact-literal `1`/`true`/`on`, following the repo's env convention inverted). The loopback rig (both stages one queue) may still arm it for its +5% class; `t34_async_finish_gpu` gains the demoted-default G1 rows alongside the armed rows.

## Acceptance

- The flip compiles at default features; `t34_async_finish_gpu` green at BOTH postures (armed rows keep the existing G1 pins — scheduling-only); dist suites green.
- The demoted-default byte-identity is the incumbent cadence by construction (the arm was scheduling-only).
- Landing note + Bench 929 cross-ref; issue closes with the commit.
