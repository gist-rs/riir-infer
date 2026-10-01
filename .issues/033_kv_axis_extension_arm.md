# Issue 033 — KV-axis extension arm on the DQ phase-matrix runner

**Status:** OPEN (split from Issue 026 T4 at its 2026-10-01 closure)
**Date:** 2026-10-01
**Provenance:** Issue 026 resolved 2026-10-01 (Plan 614, bench 023). Its one
unchecked task is carried here verbatim so the queue keeps surfacing it; the
2×2 activation-matrix core it rode is closed in HISTORY.md.

## Task

- [ ] **KV-axis extension arm.** Three independent axes the phase protocol
  separates: KV *written by* quantized-activation prefill vs KV *stored*
  precision (`q8kv`) vs KV *read* precision at decode. Pin which axis
  dominates on the decode-heavy suite. NOT part of the Plan-614 freeze (its
  scope label is the 2×2 activation matrix); future work on the same runner.

## Context

- The instrument: the `dq_phase_matrix` runner + `dq_fakequant` module
  (Plan 614 T1–T3); record `.benchmarks/023_dq_phase_matrix.md`.
- Master: `.research/004_DQ_Disaggregated_Quantization.md`
  (arXiv:2609.26333 §2.2).
- The 2×2 run read every axis INADMISSIBLE at the frozen corpora (base arith
  0.9583, NIAH pooled 0.9896, window 0.25–0.95) — a gating read of THIS arm
  needs corpora inside the admissibility window (the standing
  pre-registration rule pinned in bench 023).
- Budget note inherited from 026: accuracy evals on M3 or 4090, no training.
