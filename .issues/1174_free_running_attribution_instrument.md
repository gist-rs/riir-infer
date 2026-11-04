# Issue 1174 — a non-serialized attribution instrument: where does the free-running pass actually spend wall?

**Status:** CLOSED (2026-11-04) — delivered: `qwen38_ablate.rs` + the
gated launch sites + `--ablate-matrix` (`e59d59d`); the pp5120 table
landed as bench 945 (`.benchmarks/945_issue1174_free_running_ablation_matrix.md`)
with the verdict flow round 1 REVISE → round 2 AGREE. Headline: the
wall is GEMM-owned (~61%; MLP pair 40%), the GDN dataflow trio 24%,
the launch floor 28 ms/chunk (5.3%) measured directly, and the 943
census's non-GEMM share is marker-boundary cost (the small sections
read 0.0-0.1 ms/layer free-running). Launch reduction is closed as a
lever family. **Lease closed:** zed-ablate@amo, epoch 1793804657,
expires 2026-11-06T15:04:17Z. File removed per the noise-reduction
rule; this record lives in `HISTORY.md` + bench 945.

## Problem

The prefill lane's residual ~403–425 ms/chunk "non-GEMM share" (bench
942 StageTimers, bench 943 census) has **no free-running owner**: both
instruments that attributed it serialize the queue (StageTimers syncs
per stage; census markers carry hazard waits + an event chain), and
bench 944 measured that fusing/removing ~240 launches per chunk buys
only +0.8–1.0% e2e (~4–5 ms of 526). Nobody has attributed the
free-running wall without serialization. The next lever picked from
942/943's tables would repeat the 1173 mistake.

## The ask

An attribution instrument that does NOT serialize:

- **Kernel-skip ablation** (the cheapest honest read): arm one small
  kernel per run to a no-op pass (bit-trashing output ok — the run is
  measurement-only), diff the pp5120 pass wall per skipped kernel
  class. The wall delta IS the kernel's free-running cost; no markers,
  no syncs, no hazards.
- **One timestamp pair per layer** (the finer read): keep the 943
  counter machinery but emit exactly ONE mark pair per GDN layer
  (14× fewer marker boundaries) — the per-layer wall without
  per-section serialization. If per-layer walls sum to ≪ the 943
  section sum, the remainder is boundary cost, closing the question.

Deliverable: a pp5120 table of free-running per-kernel-class costs +
the layer-level wall; the 942/943 caveat rows (added in `2647838`
follow-ups) then point HERE for the honest attribution.

## Non-goals

- Any new optimization lever from the 942/943 tables until this lands.
- Decode-path attribution (issue 1168's lane owns decode).

## Provenance

- Bench 944 (`.benchmarks/944_issue1173_fused_norm_gating_ab.md`) —
  the refutation + the marker-boundary evidence.
- Instrument substrate to reuse: `qwen38_gdn_census.rs` (the
  counter-sample plumbing, the timestamp set law, NS_PER_TICK), the
  env-arm pattern (`RIIR_QWEN38_GDN_CENSUS=1`).
