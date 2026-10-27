# Bench 912 — Plan 623 T2.4 / issue 1147: the fused decode-batch step (BatchN)

**Status:** LANDED — GOAT G1 PASS ×2 builds, G2 +39% aggregate decode window (ABBA ×2), G3 byte-identity by construction + test-pinned, G4 unchanged (no new allocations on the frame path)
**Session:** zed-amo-i623-mb (1793132228)
**Date:** 2026-10-28
**Plan:** `.plans/623_multi_host_pipeline_inference_lane.md` T2.4 (kernel micro-batching rung, v0) · `.issues/1147_t24_kernel_micro_batching.md`
**Code:** `03f0fe6` — the BatchN hint + `StageCompute::step_submit`/`step_finish` seam (rebuilt `1f6eeb2` pre-rebase; the rebase renumbered BatchN 8→10 under the T5.1a–c landing's VerifyStream/VerifyReply, wire semantics unchanged)

## The rung

The round-robin runner (Bench 908) keeps ≤1 frame in flight per stream, but the TAIL
processes them strictly sequentially WITH a blocking readback between: stream B's GPU
chain is not even submitted until stream A's readback returns and its reply is on the
wire. The GPU idles during [readback wake · reply send · frame recv · hidden upload ·
dispatch] for every stream but the last — the launch-bound gap Phase 4 priced.

**The fused batch (v0, dispatch-level):** when every ready stream in a head pass is a
decode feed, the head sends one `BatchN {count}` hint, then the count one-token
Activation frames. The tail validates ALL rows first (refusals before any compute),
submits EVERY stage's chain, then finishes each in FIFO order. The shared CubeCL
queue (`SHARED_CUBECL_CONTEXT` — one client per process) runs the chains back to
back with no inter-chain idle. `pair_decode=false` (env `RIIR_DIST_PAIR=0`) restores
the single-drain incumbent cadence byte-identically. Kernel-level batch-2 (grid-dim
batching, ~2× fewer launches) stays the forwarder rung — this v0 is its harness.

## Cells — bmo loopback, idle gate, M5 Ultra, macOS 27.0.1

**Box:** bmo (`Mac17,15`, M5 Ultra, 96 GiB), load 0.9–1.3, GPU-quiet gate green at
fire (`ps` compute-consumer gate), release build (`--features gpu,bonsai2_hadamard`),
bin sha256 `a0b042d06e52e82f…` @ `03f0fe6` (the first build `a321eac1…` @ `1f6eeb2`
replicated the same reads). Model `Ternary-Bonsai-2-27B-PQ2_0.gguf`, split 32, GPU
BOTH stages, loopback TCP, `--streams 2`, prompt `7,13,42,100`, `--max-new 32`,
`--no-eos`, ctx 4096. ABBA order.

| cell | pair | decode window (agg) | per-stream | wall agg |
|---|---|---|---|---|
| ab_p1_a | 1 | **77.93 tok/s** | 38.97 / 39.14 | 48.80 |
| ab_p0_b | 0 | 56.17 tok/s | 28.08 / 28.31 | 47.57* |
| ab_p1_b | 1 | **78.33 tok/s** | 39.17 / 39.34 | 48.98 |
| ab_p0_a | 0 | 56.08 tok/s | 28.04 / 28.27 | 40.38 |
| confirm_p1_a (landed code) | 1 | **78.24 tok/s** | 39.12 / 39.30 | 49.77 |
| confirm_p1_b (landed code) | 1 | **78.17 tok/s** | 39.08 / 39.26 | 49.81 |
| confirm_p0_b (landed code) | 0 | 56.18 tok/s | 28.09 / 28.32 | 40.13 |
| confirm_p0_a (landed code) | 0 | 56.24 tok/s | 28.12 / 28.35 | 39.45 |

\* wall-agg on the first incumbent read included a one-off (1.303 s vs 1.53 s — the
decode window is the metric; every window read is tight).

**Verdict: +38.8–39.5% aggregate decode window (p1 77.9–78.3 vs p0 56.1–56.4),
4 paired vs 4 incumbent reads across two builds and both ABBA orders, spread
< 0.7% within each arm.** Per-stream rate rises identically (39.0–39.3 vs
28.0–28.4) — pure bubble fill, no rate partitioning. Against the 1-box GPU
oracle (46.6–46.7 tok/s incl. prefill), the paired pipeline runs **1.68×**
the 1-box rate at s=2 (incumbent 1.20×).

**G1:** `--full-check` under BOTH arms: `G1 PASS: stream 0 == 1-box stream` +
`G1 PASS: stream 1 == 1-box stream` — per-stream tokens BIT-IDENTICAL to the
1-box GPU oracle through the fused batch (×2 builds). CPU synth battery:
`tests/multi_stream_batch.rs` 4/4 (paired == incumbent captures logit-for-logit;
lockstep hint census exact MAX_NEW−1; s=1 byte-identical; six refusals); full
dist suite green at every step (34 pre-rebase, 41 with the T5.1a–c battery),
clippy `-D warnings` clean, rustfmt clean.

## Honest scope

- **Dispatch-level, not kernel-level:** the launches are NOT reduced (each chain
  still enqueues its own ~16-layer kernel sequence); the win is removing the
  inter-chain GPU idle + one host sync per pass. The kernel-level batch-2 rung
  (grid-dim batch over streams, per-row reduction order pinned = bit-identical)
  remains OPEN with this +39% as its incumbent bar and its G1 harness.
- **Loopback cells** (one box, both stages): the 2-box wire topology
  (amo⇄bmo, Bench 908's 109.5 tok/s incumbent) is the next confirmation cell —
  it needs a quiet amo window (sibling lanes active this session) and is
  recorded as the follow-up. The mechanism (tail-side submit fusion) is
  topology-independent; the wire adds RTT to BOTH arms equally.
- Prefill passes are deliberately unpaired (Feed rows only) — chunked prefill
  is already batched (T2.3b + the 1146 feature-row fix).
- The `412c887` prefill-lane feature fix (1146 sibling, same day) is IN this
  build's incumbent arm — the 56 tok/s baseline is the fixed posture.

## Artifacts

- Cells: bmo `/tmp/i623_mb_cells/<ts>/cells/` (per-cell head/tail logs) —
  ephemeral; the numbers above are the record.
- Session script: `/tmp/i623_mb_cells.sh` (GPU-quiet gate, ABBA, per-cell logs).
- bmo build prerequisites discovered this session: bmo has no cmake/brew;
  amo's homebrew cmake Cellar copied to `bmo:/tmp/cmake-444` and put on PATH
  for the sentencepiece-sys build (recorded for future bmo build lanes).
