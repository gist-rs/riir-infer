# Issue 1147: T2.4 kernel micro-batching — batch-2 decode step on the tail

**Status:** LANDED 2026-10-28 — `03f0fe6` (Bench 912: G1 PASS ×2, G2 +39% ABBA ×2; kernel-level batch-2 stays the forwarder rung)
**Lease closed:** zed-amo-i623-mb (claimed 2026-10-27 epoch 1793132228 for the T2.4 kernel micro-batching rung — batch-2 decode on the tail with per-stream bit-identity + GPU cells on idle bmo) — released 2026-10-28, work complete (v0 landed + the wire refutation + the demotion)
**Plan:** `.plans/623_multi_host_pipeline_inference_lane.md` T2.4 (the plan file's file-level lease stays t24's T5.1a–c claim — this issue is the micro-batching rung's own claim surface)

## The unit

The multi-stream round-robin (`dist/src/multi.rs`, Bench 908/910) steps each stream's
decode SEQUENTIALLY per pass: s=2 aggregate 109.5 tok/s (2.37× single), saturated at
s=3. Phase 4's launch-bound finding (batch-1 decode COMM_OFF floor −11..−14% from
halving ~90% of weight bytes) says the tail's decode is dominated by per-launch
overhead — so batching 2 streams' decode steps into ONE pass of launches should raise
the s≥2 aggregate beyond the round-robin ceiling (the launches amortise over 2× work).

## The bit-identity anchor (the G1 law for this rung)

Batch across streams in the GRID dimension, never the reduction dimension: each
stream's row arithmetic must keep the EXACT sequential-batch-1 order (same kernels,
same tile shapes, same fold order) so per-stream tokens stay bit-identical to the
round-robin path AND to the 1-box oracle. A reduction-order change is a numerics
change — G1 (byte-identity vs 1-box) forbids it.

## Acceptance

- [x] Fused decode-batch step implemented on the tail (BatchN hint + the
      `step_submit`/`step_finish` seam; `RIIR_DIST_PAIR=0` restores the incumbent
      cadence byte-identically — the A/B arm, default ON after the GOAT read).
      Dispatch-level v0: the launches are not reduced (kernel-level grid-dim
      batch-2 remains the forwarder rung).
- [x] G1: per-stream tokens through the batched tail bit-identical to the
      round-robin tail (CPU captures logit-for-logit) and to the 1-box oracle
      (CPU synth 4/4; real-model bmo loopback GPU `--full-check` PASS ×2 builds).
- [x] GPU cell on idle bmo: s=2 aggregate decode window batched vs the
      round-robin incumbent on the SAME topology — **78.2 vs 56.2 tok/s
      (+38.8–39.5%), ABBA ×2 builds** ([Bench 912](../.benchmarks/912_plan623_issue1147_fused_decode_batch.md);
      the 109.5 2-box row is Bench 908's wire topology — its confirmation cell
      waits on a quiet amo window).
- [x] GOAT verdict + plan/bench records updated: G1 PASS, G2 PASS (+39%), G3
      byte-identity (s=1 + incumbent arm byte-identical, test-pinned), G4
      unchanged (no new frame-path allocations). The incumbent arm keeps its
      demote switch; the paired arm promotes as the default.
