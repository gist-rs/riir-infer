# Issue 1172 — qwen38 dense lane: the GDN-bucket census (GPU timestamps) — pinning the non-GEMM 30%

**Status:** LANDED (2026-11-04, zed-986a7538@amo) — instrument + census
complete; verdicts in `.benchmarks/943` and below. The named lever is
filed forward as `.issues/1173_dense_lane_fused_norm_gating_launch_floor.md`
(fusion, not kernel work — the census's central finding).
**Lease closed:** zed-986a7538@amo, epoch=1793714997, expires 2026-11-06T18:00:00+00:00 (landed `95fea10`/bench 943).

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

- [x] Instrument: per-section GPU wall inside `prefill_gdn_layer` for one
  representative layer (qkv/z/a/b GEMMs · conv1d · recurrence ·
  `beta/decay`+expand+zgate · norms+residual · MLP GEMMs), diagnostic env
  (`RIIR_QWEN38_GDN_CENSUS=1`), summed per chunk in the timer mode.
  DRY check DONE (in-repo, measured — see the record's instrument laws):
  wgpu timestamp queries are NOT the path (cubecl owns its encoders);
  Metal counter sample buffers with the device's "timestamp" counter set
  + pass-attachment sampling on self-committed marker buffers are.
- [x] Census runs at the promoted posture (pp512, pp2560, pp5120): the
  split table, GPU-exclusive, box provenance (bench 943).
- [x] Verdict per class: the scan (0.77 ms = 8.7% of the layer wall;
  matches bench 938's ~6% of the pass — adequate), conv, norms, gating —
  every non-GEMM class reads a size-independent ~0.5–0.7 ms LAUNCH FLOOR
  (the same kernels read 0.01–0.11 ms idle-queue at pp512): the lever is
  fewer launches (filed forward as issue 1173, the fused norm+gating
  family), not faster kernels. GEMMs: write-bandwidth-bound
  (floor-dominated) — the 279 ms arithmetic is refuted, no kernel work
  justified.
- [x] The GEMM residual: the census measured the in-pass GEMM wall
  DIRECTLY (0.55 ms for the 4 GEMMs at pp5120 — 3.4× under bench 941's
  standalone per-GEMM numbers; pp512's idle-queue 8.33 ms is
  dispatch/queue-idle, not FLOPs). The half-B TFLOPS arithmetic is not
  the binding constraint in-pass.

## Non-goals

- Any kernel rewrite in this issue (the census files the levers, each
  with its own pre-registered bar).
- The attention path (bench 942 landed it), the decode path (issue 1170
  lever 2), the MLX-residual matmul tuning (league-demand-priced).

## Provenance

Bench 942 (`62f112e`) measured the 74% share + the per-chunk flatness;
the wte sync-stall split (bench 940) is the cautionary tale this
instrument avoids — GPU timestamps, never CPU deltas between async
launches. Landed: instrument `4ea7adb`, deep-queue marker-chain fix
`95fea10`, census + verdicts bench 943.
