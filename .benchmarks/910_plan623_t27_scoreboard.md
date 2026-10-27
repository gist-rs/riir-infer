# Bench 910 — Plan 623 T2.7: the G2 scoreboard (PP overhead vs 1-box; vs llama.cpp RPC, mlx-lm, exo)

**Status:** RECORD — the scoreboard is ASSEMBLED from the measured rows (900/904/1139-lineage/908-today); the exo row stays BLOCKED (weight format, Phase 0)
**Date:** 2026-10-28
**Session:** zed-amo-i623-t24, 1793123917
**Plan:** [`.plans/623_multi_host_pipeline_inference_lane.md`](../.plans/623_multi_host_pipeline_inference_lane.md) T2.7
**Model:** Ternary-Bonsai-2-27B — our rows PQ2_0 GGUF; the MLX rows the 2-bit affine+hadamard pack (`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @ `fcba37d21`); llama.cpp rows PQ2_0 on prism-b10787. **Quant postures differ across families — cross-family rows are plan-internal comparability, never an equated quant study** (the Bench-904 caveat, restated because this table invites the misread).
**Hardware:** amo (M5 Ultra) ⇄ bmo (M5 Ultra), one TB5 @ 80 Gb/s (RTT ~0.45 ms) + tailnet; split 32/32 everywhere; GPU stages both sides.

## Our rows (2026-10-27/28, this lane's binary, zed-amo-i623-t24)

| arm | decode t/s | note |
|---|---|---|
| 2-box s1 (our PP, one stream) | 46.0 / 47.7 (ABBA pair) | serial head→tail alternation; the pipeline cost vs 1-box is the wire + sequencing |
| **2-box s2 (our PP + T2.4 round-robin)** | **109.3 / 109.8 aggregate = 2.37× s1** | per-stream 54.6/54.9 — unchanged-to-better; pure bubble fill |
| 2-box s3 | 109.4 aggregate — SATURATED | per-stream 36.5; more streams partition, not add |
| prefill P=256 (2-box, chunk 128) | 1.86 s = **1.47× per-position** | the T2.3c ABBA; P=64 per-position wins ~8% (Bench 909) |
| tailscale posture (Bench 901-era cells, T2.6) | decode −10% vs TB5 | = the RTT; prefill +4% (chunked frames amortise) |

Reference points for the s1 row: the 1-box stage-path decode on idle bmo
read **56.9 tok/s** at the Issue-1144 posture (Issue-1139 T-series) — so
**s1 single-stream PP carries roughly −15% vs a same-engine 1-box**
(consistent with Bench 900's llama.cpp-RPC −20% shape: single-stream PP
over TCP pays the round trip), while **s2 wipes the overhead and doubles
the 1-box number** — the scoreboard's headline: PP was a loss lane only
while it was single-stream.

## Opponent rows (as measured in this plan's Phase 0; postures per source bench)

| family | arm | decode t/s | PP vs its own 1-box | source |
|---|---|---|---|---|
| llama.cpp (RPC-TCP) | 1-box solo bmo | 60.9 | — | Bench 900 |
| llama.cpp (RPC-TCP) | 2-box layer-split | 48.6 | **−20%** | Bench 900 (pp512 −2.7% ≈ free; `-sm tensor` over RPC crashes) |
| mlx-lm (socket PP) | 1-box bmo | 58.1 | — | Bench 904 |
| mlx-lm (socket PP) | 2-box PP | 43.0 | **−26%** | Bench 904 |
| mlx `mx.distributed` ring | 2-node ring p2p | — | **DEADLOCKED** (mlx 0.32.0; ring collectives cost 2–4× over raw TCP) | Bench 904 |
| exo | PP-ring | — | **BLOCKED — weight format** (the pinned research clone refuses PQ2_0, `invalid ggml type 142`; no sanctioned conversion exists — Phase 0 note) | plan §T0.4 |

## The scoreboard verdict

1. **Single-stream PP over TCP loses ~15–26% on every family that
   measured it** (ours −15% vs the 1144-posture 1-box; llama.cpp-RPC
   −20%; mlx-socket −26%). The wire round trip is the tax; nobody's
   transport dodges it.
2. **Round-robin multi-stream is the differentiator**: our s2 = **2.37×
   our s1** — a number NO opponent row has (llama.cpp RPC serves one
   request per slot; mlx socket PP is single-stream; exo un-run). The
   claim is deliberately ours-only: no opponent was measured with ≥2
   concurrent streams, so the honest cross-family comparison stays at
   s1 (where we sit mid-pack, −15% vs our own 1-box, vs −20/−26% for the
   opponents' PP paths).
3. **Prefill**: our chunked 2-box prefill (1.47× per-position at P=256,
   1.86 s) is in the llama.cpp-RPC class (its pp512 was −2.7% vs solo ≈
   free) — chunked frames amortise the RTT the same way.
4. **Reliability rows nobody else publishes**: authenticated frames
   (BLAKE3 MAC + strict counters), loud per-stream position validation,
   the wrong-key/foreign-artifact/unknown-stream refusal battery (7/7),
   and the G1 bit-identity bar at every split.

## Honest scope

- The s2/s3 rows duplicate ONE request ×N (the T2.4 admission is the
  static table) — the serving shape (N distinct requests) is the same
  wire mechanics with different prompts; the aggregate math does not
  change, but the measured rows are the duplicated-request cells.
- The 1-box comparator (56.9, Issue-1144 posture, idle bmo) predates
  today's cells by one kernel rung; the −15% s1 figure carries that
  posture skew (worst case it understates our 1-box, i.e. overstates
  the s1 loss).
- The exo row stays blocked on the weight format — re-opens only when a
  pack loader or a sanctioned stock-arch Bonsai conversion exists (the
  Phase 0 T0.4 note, unchanged).
- MLX rows are bmo-half reads on a clean box (Bench 904 §3); the amo
  half was deferred there (MPS held) — the PP rows are the bmo half's
  throughput, the fair comparison inside that bench.
