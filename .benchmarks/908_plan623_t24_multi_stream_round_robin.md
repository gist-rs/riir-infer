# Bench 908 — Plan 623 T2.4: multi-stream round-robin (the PP-bubble fill)

**Status:** RECORD — T2.4 code LANDED (CPU-verifiable arms); G2 GPU/2-box cells stay open (box contention)
**Date:** 2026-10-27
**Session:** zed-amo-i623-t24, 1793123917
**Plan:** [`.plans/623_multi_host_pipeline_inference_lane.md`](../.plans/623_multi_host_pipeline_inference_lane.md) T2.4

## What landed

`crates/riir-infer-dist/src/multi.rs` — multi-stream round-robin over ONE
stage link:

- **Wire (additive, single-stream frames untouched):** `BeginStream`
  (seq_id | prompt_len | prefill_chunk, 12 B) + `TokenStream`
  (seq_id | token, 8 B). Activation/prefill frames were ALREADY
  stream-addressed — the Phase-1 `ActivationFrameHeader.seq_id` field the
  single-stream runner encodes as 0.
- **Stream = index into the caller's stage table.** One stage instance per
  stream (per-stream KV/GDN cache + scratch; weights Arc-shared), so the
  table length is the admission cap and `s` streams cost `s` stage states
  per side. Memory is explicit at construction, never implicit.
- **Scheduler:** strict round-robin waves — one work item per non-done
  stream per pass (a prefill chunk or one decode step), then the head
  drains exactly ONE reply. One reply-expected frame outstanding per
  stream; the tail is serial so replies leave it in wire order; the head
  verifies the FIFO route (a reply naming a non-oldest stream refuses
  loud). Prompt chunks of one stream interleave other streams' decode
  steps by construction — exo's prefill-blocks-decode gap cannot form.
- **Tail:** per-stream position cursors (`pos == track.next_pos` per
  stream — the single-stream GLOBAL ordering would refuse a legal
  interleave), per-stream `BeginStream` tracking, duplicate/unknown-seq
  refusals loud.
- **No send/send deadlock** (the T2.3a invariant extends): the head can
  block sending a large chunk only while the tail is still draining it;
  the tail's only sends are 8-byte replies, with a `recv` between them.
- **Bin:** `dist_stage --streams N` (head duplicates the request ×N and
  prints per-stream + aggregate tok/s; tail builds N stages). `--streams 1`
  (default) rides the single-stream runner byte-identically.

## Gates

**G1 — per-stream bit-identity** (`tests/multi_stream_g1.rs`, 7 tests):

- `multi_stream_tokens_bit_identical_per_stream_every_split`: 2 streams
  (different prompts), EVERY split k of the synth Bonsai-2 — each stream's
  token stream == the 1-box oracle, stream 0's per-position tail logits
  bit-identical (`to_bits`), step census exact.
- `one_stream_multi_matches_run_head`: the degenerate 1-stream case ==
  the single-stream runner.
- `stream_b_prefill_interleaves_stream_a_decode`: stream B's last prefill
  chunk lands BETWEEN stream A's decode events in the tail's capture, and
  A keeps decoding after — the interleave pinned in BOTH directions.
- `per_stream_positions_are_independent`: stream 1 restarts at pos 0 while
  stream 0 is deep in its sequence.
- Refusals: activation without `BeginStream` refuses; duplicate
  `BeginStream` refuses.

**Full dist suite green:** 9 lib + 7 multi + 2 planner + 8 two_stage_g1
(single-stream byte-identity intact — the additive frames changed nothing
on the old path). Clippy `-D warnings` (--all-targets), rustfmt clean,
`--features gpu` compiles.

**Real-model loopback smoke (the bin, CPU stages, amo, box under load —
the two GPU measurement lanes were running; indicative):**

```
dist_stage --role tail --model Ternary-Bonsai-2-27B-PQ2_0.gguf --split 32 \
  --listen 127.0.0.1:17623 --streams 2
dist_stage --role head ... --connect ... --streams 2 --prompt 7,13,42,100 \
  --max-new 8 --full-check
stream 0: decode 15.12 tok/s · stream 1: 15.20 tok/s
aggregate: 14 tokens in 0.766s = 18.27 tok/s (2 streams, wall-clock)
1-box cpu oracle: 11.11 tok/s end-to-end
G1 PASS: stream 0 == 1-box stream
G1 PASS: stream 1 == 1-box stream
[tail] served 22 positions, clean shutdown
```

Aggregate **1.64× the 1-box oracle** on real weights (CPU steps are
rayon-parallel, so the two stages' steps genuinely overlap; single-stream
was the serial-sum shape). NOT a G2 cell: CPU backend, contended box,
loopback, n=1 run, no interleave-ordering of arms.

**Loopback synth overlap read** (the G1 test's ratio bar, debug build,
contended box): serial 352.9 ms vs interleaved 284.0 ms = **0.805** (bar
< 0.85 — an overlap-PRESENCE assertion, not a latency claim).

## Honest scope (what T2.4 is NOT yet)

- **Admission is the static request table** — no stream JOINS mid-flight
  yet. The scheduler only asks "which non-done streams have work", so a
  join is a table append at a pass boundary behind a caller-facing seam
  (serve-time shape, later rung).
- **NOT kernel-level micro-batching:** two streams' tokens still compute
  as two `step` calls on the tail. Batching them into one dispatch is a
  forwarder rung — Phase 4 measured batch-1 decode launch-bound, so the
  rung is real, and it is not this module's.
- **Wave granularity:** the drain-one-then-re-rotate loop overlaps sends
  with in-flight replies but the head still computes one step at a time.
- **GPU cells:** one `TernaryGpuStage` per stream = per-stream VRAM state
  (the constructor already refuses over-budget; the admission cap is the
  memory answer). The 2-box GPU ABBA cells (aggregate tok/s vs 1× stream
  vs 1-box, TB5 + tailscale postures) belong to this bench's G2 half +
  T2.7's scoreboard — blocked this window by two live GPU measurement
  lanes on amo (886-t5 device gates + the 1122 ktile64 re-cell) and the
  2-box coordination cost; the CPU-verifiable arms landed first by design.

## Verdict

**LANDED (code + G1)** — the multi-stream protocol is real, bit-identical
per stream, and the bubble fill measures on both the synth fixture and the
real 27B checkpoint over loopback. G2 GPU/2-box cells OPEN, gated on a
quiet amo GPU + bmo coordination.
