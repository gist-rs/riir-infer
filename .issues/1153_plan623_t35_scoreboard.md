# Issue 1153 — plan-623 T3.5: Phase-3 scoreboard rows (incumbents vs ours at the T3.4 posture)

**Status:** RESOLVED 2026-10-29 — [Bench 929](../.benchmarks/929_plan623_t35_scoreboard.md) is the record; the demotion residue filed + landed as [Issue 1154](1154_plan623_t34_async_finish_demote.md) (`6a1c524`); plan T3.5 marked landed.
**Lease closed:** zed-m3-i971-housekeeping, 1793262200, was expires 2026-10-31T23:50:00+00:00 — LANDED the T3.5 scoreboard 2026-10-29 (Bench 929: decode row beats llama.cpp-RPC 2-box +7.6%, PP-overhead gate honest FAIL −7.1%, T3.4 arm wire-refuted → demoted via 1154, prefill overlap bar PASSES at c128, exo chunk heuristic does not transfer, both T3.4 residues declined on the decomposition; the plan-file T3.5 row carries the full landing note).
**Date:** 2026-10-29
**Type:** measurement + record (plan-623 unit; file-level claim lives HERE — the plan file is unclaimed, kept on the unit-issue pattern for one-claim-per-file hygiene while the sibling lane finishes its push)

## The task (plan text)

> **T3.5** Scoreboard rows vs llama.cpp RPC-RDMA and exo/MLX jaccl. (jaccl prose
> comparison exists — Bench 905's 16–22 µs pipelined collective rate vs our 7.7 µs
> ping-pong RTT — but a like-for-like scoreboard row needs the T3.4 serving integration.)

## What "like-for-like" means here (scoping, from the Proposal-042 G-gates + T2.7)

T2.7 (Bench 910) assembled the Phase-2 scoreboard: our PP s1 46–48 t/s (≈ −15% vs the
bmo 1-box 56.9), vs llama.cpp-RPC 2-box −20% (Bench 900), mlx-socket PP −26% (Bench 904),
mlx ring deadlocked, exo blocked on PQ2_0. T3.4 (issue 1151) landed the async-finish
serving arm but measured only the M3 loopback rig (~5% on the s=2 all-feed decode window).
The missing rows are OURS at the T3.4 posture on the real wire:

1. **The wire G1 confirmation** of the T3.4 arm (bit-identity on amo⇄bmo — the landing's
   G1 was M3-loopback; the wire cell is contention-tolerant and runs under the bmo
   co-tenant load).
2. **ABBA armed-vs-demoted decode rows at s=1 and s=2** on the wire (`plan623_wire_cell.sh`
   already drives exactly this; PROVENANCE beside every row — bmo carries the resident
   mammoth capture, bursts disclosed; a clean-window re-read is pinned open if the
   spread demands it).
3. **The 1-box oracle row from the same process** (`--full-check` prints it) — the
   PP-overhead gate (≤5% slower than 1-box) judged same-box, same-load.
4. **The decode-step decomposition** (new instrument, `RIIR_DIST_STEP_TIMING=1`): head
   {submit, finish, send, recv} + tail {recv, fwd, send} wall buckets per 64-step window —
   the honest adjudication of the two T3.4 residues (the literal comm-THREAD split and
   the tail logits async-readback arm are CONDITIONAL follow-ups per issue 1151's scope
   notes: only if the scoreboard shows that leg binding).
5. **The incumbent-row adjudication**: llama.cpp RPC ships TCP-only (no RDMA transport
   exists upstream — the "RPC-RDMA" row records N/A with the T0.4 TCP row as its best);
   exo stays N/A (PQ2_0 unsupported); the jaccl comparison stays prose (Bench 905
   collective-rate) because its serving forms are dead by earlier cells (mlx ring
   deadlock, TP refuted Bench 906) — re-measuring a dead serving form is not a row.

## Non-goals

- No verbs-`Transport` serving swap in this unit: Phase 3 tasked no such row, and the
  swap's justification must come FROM this scoreboard's decomposition (if the TCP RTT
  leg is the binding cost, the transport-swap issue files WITH the measured gap).
- No TP re-cells (Bench 906's refutation stands), no M3-loopback re-reads (T3.4's
  landing carries them; the scoreboard rows are M5-class wire rows).

## Claude verdict (round 1, 2026-10-29, session 584ad1df) — REVISE, applied

The owner asked for source-level research on how exo does 2-box prefill plus a Claude
verdict; exo @ the research pin `bc412e2` cloned to `.raw/exo` (rm at lane end).
Findings + corrections now binding for Bench 929:

1. **COMPUTE-CLAIM CORRECTION (the verdict caught my error):** exo's dummy/placeholder
   iterations run NO model compute (`generate.py:236-247` — the leading/trailing loops
   call only the progress callback). Every rank runs the full prompt through its own
   layer shard — pipeline parallelism, identical total FLOPs to ours. "Ours pays half
   exo's compute" is FALSE and must not appear in the record. The honest difference is
   control/cadence: their blocking `recv_like` + lockstep placeholder slots (MLX's
   collectives need matched send/recv on every rank) vs our async chunk streaming.
   Same systolic schedule both sides; latency ≈ (N+R−1) × chunk-time in both.
2. **Chunk-size rule is the transferable lesson:** `prefill_step_size // min(4,
   world_size)` (`generate.py:194`) is a pipeline-bubble heuristic (idle fraction ≈
   (R−1)/(N+R−1) traded against per-chunk overhead). Their 2-rank chunk = 2048 vs our
   default 128 — and T2.3b's 128–256 optimum predates the Issue-1146 prefill rungs →
   chunk re-cell rides THIS unit (interleaved arms, ≥2 prompt lengths: 512/2048/8192).
3. **Warmth law:** the G1-cell 0.714 s prefill figure is COLD-START (two fresh GPU
   stages' JIT); the 0.131 s oracle was post-warmup — the pair is NOT comparable and is
   withdrawn as a prefill row. Every prefill row in Bench 929 carries its warmth label;
   the in-run oracle is the comparator.
4. **The overlap bar (falsifiable G2):** warm long-prompt 2-box prefill should
   approach ~2× the 1-box oracle minus the bubble (both stages work different chunks
   concurrently). If it lands ≈1×, the head/tail chunk overlap does NOT exist on the
   wire — that is a REAL finding (exo forces overlap with `mx.async_eval`; we must
   measure ours, not assume it).
5. **Scope splits:** a chunk-default CHANGE (if the re-cell's winner ≠ 128) is a
   separate unit (code + G1–G4 + promotion) — never in T3.5's record commit. exo's
   disaggregated prefill (`remote_prefill.py`, msgpack KV chunks over TCP) is prior
   art for a future prefill/decode split lane; out of scope here. The "routed
   TokenStream beats per-token all_gather" claim stays DECODE-ONLY (T2.7-backed).

## Acceptance

Bench record (929) with: the wire G1 line, ABBA rows (armed/demoted × s1/s2) with
PROVENANCE, the 1-box oracle row, the decomposition windows from both roles, the
incumbent adjudication section, and the residue verdicts (thread-split / tail-logits
arm: justified or declined WITH the numbers). Plan 623 T3.5 + AGENTS.md plan-623
paragraph + HISTORY row updated in the landing commit.
