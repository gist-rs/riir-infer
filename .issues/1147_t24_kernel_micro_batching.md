# Issue 1147: T2.4 kernel micro-batching — batch-2 decode step on the tail

**Status:** OPEN
**Claimed:** zed-amo-i623-mb (T2.4 kernel micro-batching: batch-2 decode on the tail with per-stream bit-identity + GPU cells on idle bmo), 1793132228, expires 2026-10-29T20:17:00+00:00
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

- [ ] Batch-2 decode step implemented on the tail (behind its own gate; the round-robin
      sequential step stays the default until the GOAT reads).
- [ ] G1: per-stream tokens through the batched tail bit-identical to the round-robin
      tail and to the 1-box oracle (synth fixture CPU + real-model bmo loopback GPU).
- [ ] GPU cell on idle bmo: s=2 aggregate tok/s batched vs the 109.5 round-robin
      baseline (ABBA, provenance line quoted).
- [ ] GOAT verdict + plan/bench records updated; demote the loser.
