# Issue 1172 — qwen38 dense lane: the GDN-bucket census (GPU timestamps) — pinning the non-GEMM 30%

**Status:** OPEN — the bench-942 record's named next instrument. Not claimed
(append a `**Claimed:**` line under this Status before implementing).
**Claimed:** zed-986a7538@amo, epoch=1793714997, expires 2026-11-06T18:00:00+00:00

## Problem

Bench 942 left the pass **gdn-dominated: 74%** (~400 ms/chunk × 10 chunks
of a 5.4-5.9 s synchronized pp5120 pass; attention is down to 20%, tail
~1%). The bucket decomposes by arithmetic — GDN GEMMs ≈ 18.8 TFLOP/chunk
at the half-B kernel's measured 53–66 TFLOPS (mlp gate/up 73) ≈ **279 ms
of pure GEMM wall**, leaving **~120 ms/chunk (~30% of the bucket, ~26% of
the pass) in the non-GEMM kernels**: chunked conv1d, the multi-token
recurrence scan, the beta/decay + expand/L2 + z-gating elementwise chains,
and the RMS norms. Bench 938 measured the SCAN alone at ~6% (the
chunked-route OFF/ON delta); the other four classes were never split.

Neither CPU-side deltas (async launches — measures enqueue only) nor
per-section syncs (the ~300 ms idle-queue stall per sync, bench 940's
diagnostic) can measure this honestly. **The instrument is GPU timestamps**:
Metal counter sample buffers on our raw commits + wgpu timestamp queries
inside the CubeCL-launched sections — both report GPU wall, zero queue
distortion.

## Plan

- [ ] Instrument: per-section GPU wall inside `prefill_gdn_layer` for one
  representative layer (qkv/z/a/b GEMMs · conv1d · recurrence ·
  beta/decay+expand+zgate · norms+residual · MLP GEMMs), diagnostic env
  (`RIIR_QWEN38_GDN_CENSUS=1`), summed per chunk in the timer mode.
  DRY check first: wgpu timestamp-query support in this cubecl/wgpu
  version; Metal counter buffers on the raw-commit path (the zero-copy
  modules' command buffers — same pattern as `wait_last`, timestamps
  instead of completion waits).
- [ ] Census run at the promoted posture (pp512 + pp5120): the split
  table, GPU-exclusive, box provenance.
- [ ] Verdict per class: the scan (bench 938: adequate), conv, norms,
  gating — each either (a) priced-lever (kernel work filed separately
  with a pre-registered bar) or (b) measured-adequate (closed, recorded).
- [ ] The GEMM residual too: the census also validates the 279 ms
  arithmetic against the actual GEMM wall (the half-B A/B numbers are
  standalone; in-pass L2 pressure may differ).

## Non-goals

- Any kernel rewrite in this issue (the census files the levers, each
  with its own pre-registered bar).
- The attention path (bench 942 landed it), the decode path (issue 1170
  lever 2), the MLX-residual matmul tuning (league-demand-priced).

## Provenance

Bench 942 (`62f112e`) measured the 74% share + the per-chunk flatness;
the wte sync-stall split (bench 940) is the cautionary tale this
instrument avoids — GPU timestamps, never CPU deltas between async
launches.
