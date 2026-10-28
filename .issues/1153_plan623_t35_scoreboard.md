# Issue 1153 — plan-623 T3.5: Phase-3 scoreboard rows (incumbents vs ours at the T3.4 posture)

**Status:** OPEN
**Claimed:** zed-m3-i971-housekeeping, 1793262200, expires 2026-10-31T23:50:00+00:00
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

## Acceptance

Bench record (929) with: the wire G1 line, ABBA rows (armed/demoted × s1/s2) with
PROVENANCE, the 1-box oracle row, the decomposition windows from both roles, the
incumbent adjudication section, and the residue verdicts (thread-split / tail-logits
arm: justified or declined WITH the numbers). Plan 623 T3.5 + AGENTS.md plan-623
paragraph + HISTORY row updated in the landing commit.
