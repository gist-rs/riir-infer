# Issue 1146: 1-box prefill throughput gap — 6–8× behind the incumbents (T0.3 finding)

**Status:** OPEN — gap MEASURED (Bench 911), cause bracketed (per-frame floor, decode-shaped kernels), no fix landed
**Date:** 2026-10-28
**Found by:** [`.benchmarks/911_plan623_t03_arms.md`](../.benchmarks/911_plan623_t03_arms.md) (Plan 623 T0.3, bmo idle, one session)
**Session:** zed-amo-i623-t24

## The finding

riir-infer's single-box prefill on Bonsai-2-27B (Metal, idle bmo, chunked width 128):

| cell | llama.cpp | MLX (pack) | riir-infer | ours/llama |
|---|---:|---:|---:|---:|
| pp512 | 1087.4 | 1058.1 | 140.2 | **0.13×** |
| pp4096 | 1046.1 | 1026.1 | 166.9 | **0.16×** |
| pp32768 | 772.0 | REFUSED | 96.1 | **0.12×** |

Decode is competitive (0.87–0.98×, same bench). Prefill is not.

## The anatomy (what the measurement brackets)

Per-frame wall at chunk width 128 through the dist pipeline (both 32-layer stages, loopback):

- pp512 (4 frames): 0.914 s/frame → 6.0 ms/token-frame
- pp4096 (32 frames): 0.767 s/frame → 5.7 ms/token-frame (warm-up amortization)
- pp32768 (256 frames): 1.331 s/frame → 10.4 ms/token-frame (KV-growth term: every-4th-layer full attention attends an ever-longer history)

llama.cpp prefills a token in 0.92–1.3 ms at the same widths. Our floor — the CHEAPEST
frame we can produce — costs ~6 ms/token. The gap is structural, not wire: loopback RTT
is ~0.1 ms against 0.77–1.33 s frames (Bench 896), and the 2-box TB5 prefill measured the
same ~137–140 t/s as loopback (Bench 909 ABBA P=256 vs this bench pp512).

**Cause hypothesis (ranked, to be confirmed by a kernel census before any fix):**

1. **Decode-shaped kernels at prefill shapes.** The chunk body runs the batched range
   prefill over the `ternary_gemm_batched` body at M=128 — a tile height sized for the
   decode path. llama.cpp dispatches M=512–32768 rows into big-M Metal MMM kernels; ours
   never leaves the simdgroup-GEMV regime that wins decode.
2. **Per-frame stage serialization + handoff.** Each 128-frame crosses head→tail with a
   handoff; within one stream's prefill the two stages' frame costs do not overlap (the
   T2.3a overlap is per-POSITION, the chunked path is per-FRAME — Bench 909: 2-box chunked
   beat 1-box chunked by only ~8% at P=256).
3. **The GDN scan at M=128** — the recurrence runs per frame; whether a wider-frame scan
   (or a chunked-scan reformulation like llama.cpp's `gated_delta_net` ssm path) pays is
   unmeasured.

## Fix directions (none landed; each needs its own GOAT)

- **W-frame prefill kernels**: an M≥512 prefill tile for the ternary GEMM batch path
  (the Bench-895 spill cliff says regs=128/local=0 is the constraint to design against).
- **Wider chunks**: the width curve (Bench 909) stopped at 128 because within-chunk
  attention is quadratic — but that curve was measured through THIS kernel stack; with
  prefill-shaped kernels the optimum may move. Re-measure, don't assume.
- **Frame overlap**: pipeline two frames (head on frame n+1 while tail drains n) if the
  scheduler permits — it is the serving shape T2.4 already built for streams.

## Acceptance

A prefill GOAT: any landed rung re-runs the T0.3 prefill cells (llama.cpp rows of Bench
911 are the stable incumbent baseline — reproduced bit-close across five weeks) and closes
a measurable share of the 6–8× gap without moving G1 token-exactness or decode t/s
(G3). Promotion follows the standard feature-flag discipline; the prefill path is the
league's pp rows too (riir-ai scoreboard), so a win here is a league lever, not just a
Plan-623 one.

## Boundary

Substrate-side: the kernels live in `riir-infer-gpu`, the range seams in
`riir-infer-dist`. No riir-ai work is required for the fix; the league re-cell after a
win is a riir-ai side effect.
