# Bench 917 — Plan 623 T5.1d: the trained MTP-head consumer — in-posture acceptance + the split-63 seat verdict

**Status:** LANDED 2026-10-28 — consumer shipped, acceptance MEASURED (not inherited), promotion NOT claimed (owner-gated)

- Consumer lane: riir-infer `spec_mtp.rs` (`BmthHead` BMTH v1 reader + `MtpHeadDrafter`) + the runner
  roll seam (`DraftSource::propose_with_roll`, additive; token-level drafters byte-identical — the full
  G1 battery re-ran unchanged, 9/9, plus two new roll-path tests). Head artifact: riir-train Bench 677's
  BMTH v1 (d 5120, vocab 248320, support 2963, blake3 `e9c6f07a…` verified against its sidecar), fetched
  to `../riir-train/data/bonsai_mtp/head/` (amo copy).
- Harness: `examples/mtp_head_acceptance.rs` (`--features bonsai2_hadamard`; required-features row — the
  loud-skip law). In-process 2-stage pair over loopback TCP, real `Ternary-Bonsai-2-27B-PQ2_0.gguf`
  (n_layer 64, n_embd 5120), CPU stages, release.

PROVENANCE: host amo (M5 Ultra, 30 cores, macOS 27.0.1, AC), load avg 2.2 at start; a concurrent
riir-ai B809 metal bench ran during the first free-run cell (SN state, low CPU) — the ACCEPTANCE
numbers are rate-count based (load-insensitive); the tok/s rows are wall-clock and ARE load-sensitive,
quoted as machinery-cost ORDERS only.

## 1. The basis curve (teacher-forced, regime-clean) — THE scoreboard numbers

Same pairing as the fit eval (head(row_t) vs the model's own argmax on the SAME row), 2048 frozen
read-split tokens, the ONLY delta being the row's basis. Post-fit reference: **0.7929** (riir-train
Bench 677, post-final stream, 20k rows).

| seat (head stage owns layers) | per-slot true acceptance | Δ vs fit basis |
|---|---|---|
| post-final (fit basis, all 64) | 0.7929 (Bench 677) | — |
| **split 63** (head 0..63, tail = last layer) | **0.6841** (1401/2048) | −10.9 pt |
| split 48 (head 0..48) | 0.0591 (121/2048) | −73.4 pt |
| split 32 (the production split) | 0.0332 (68/2048) | −76.0 pt |

**A cliff between 63 and 48.** The linear decodability the head exploits lives in the FINAL layers:
removing ONE layer costs 11 points; removing 16 costs 73. The only viable serving seat is
**split 63** — tail = the last layer alone.

Calibration (the read's own sanity rows, split 63 TF): model-argmax consecutive-repeat **0.3957**
(the fit split read 0.4150 — the CPU stage's argmaxes are consistent with the GPU folded runtime's);
model-argmax-in-support **0.9360** (the 2963-id support covers the model's picks; a missed support id
can never be drafted — the 6.4% tail is an absolute ceiling fraction).
Binomial SE at 2048 rows ≈ ±0.010 (95% CI ±2 pt).

**Projected serving E[m] at split 63 on healthy streams:** ≈ 1 + 0.684 ≈ **1.68** at k=2 (the d_0
echo + one trained draft), vs 1.41 for the fit-basis-scaled repeat-last-class baseline (1 + 0.41).
E[m] at k=2 on the FREE run measured 1.104 — see §3 for why that number is stream-polluted.

## 2. The free-run stream is degenerate on raw text (honest caveat on every E[m] below)

The pair's generated stream from the raw ag_news prompt (512 read-split tokens) collapses into a
**period-2 (0, 271) attractor** within ~60 tokens (256-token dump: 14 unique ids, period-2 match
0.906). This is the MODEL's honest greedy behavior, not a consumer defect — the cross-check pins it:

- Teacher-forced argmaxes at positions 504..519 (per-position one-shot steps): `[0, 0, 951, 0, 0, 0,
  5394, 0, 0, 0, 1619, 13446, 291, 0, 0, 0]` — 0-heavy at the SAME positions the free run emits 0.
- The free stream's `1619` at gen position 3 matches TF's argmax at position 514 (1619): the pair's
  state path is bit-consistent with the TF path (chunked prefill == per-position steps, G1).
- The training capture never saw this: it is teacher-forced on the frozen corpus by construction.

Consequence: the free-run acceptance cells below measure the head ON A COLLAPSED STREAM — they are
recorded for completeness and are NOT serving rows. The honest serving-shaped proxy is §1's
teacher-forced number; a chat-shaped-prompt re-read is a named follow-up (§5).

| free-run cell (split 63, prompt 512, amo) | tok/s | E[m] | per-slot > echo |
|---|---|---|---|
| mtp_head, k=2, gen 256 | 0.233 | 1.1039 | 0.1039 |
| repeat_last, k=2, gen 256 | 0.216 | 1.0451 | 0.0451 |
| mtp_head, k=1, gen 64 | 0.238 | 1.0000 | 0.0000 |
| repeat_last, k=1, gen 64 | 0.233 | 1.0000 | 0.0000 |
| ordinary pair (incumbent), gen 256 | 16.195–16.45 | — | — |

## 3. The machinery cost is the SNAPSHOT, and it is ~70× on the CPU pair

Spec pair ≈ 0.23 tok/s vs ordinary pair ≈ 16.2 tok/s at split 63. **k=1 and k=2 read the same
tok/s** — the per-round cost is a FIXED ~4 s and the only k=1 structural cost is ONE snapshot +
one verify frame + one head roll. The hybrid-state snapshot/restore (`HybridStateSegment`,
T5.1b V0 — "snapshot-based, byte-exact by construction, zero forward surgery") is the
whole regression: ~600 positions × 63 layers of KV + GDN state serialized per verify round.
The plan's named fast path — the GDN journal's rank-1 unwind + an attention KV truncate —
is the next rung, plus the snapshot codec itself (a pod-cast memcpy class fix; filed as
riir-infer Issue 1148).

The wall-clock rows are therefore MACHINERY-COST ORDERS ONLY. The acceptance axis (§1) is
count-based and stands independent of them.

## 4. Verdict

- **Consumer lane: LANDED.** Lossless end-to-end (the G1 battery extends over the roll path;
  every partial accept rewinds byte-exact; the MTP drafter on synth weights streams == oracle).
- **Seat verdict: split 63 ONLY** — the head reads 0.684 per-slot there, 0.033 at the production
  split 32. Any deployment claim must name the split.
- **Promotion: NOT claimed** — owner-gated per the feature-flag discipline, and the machinery
  cost (§3) must land its fast path before any tok/s claim is even measurable honestly.
- The trained head beats its baselines IN-POSTURE at split 63 on natural text (0.684 vs the
  0.396 model-repeat floor and vs the 0.095 corpus-repeat floor of this degenerate free run).

## 5. Follow-ups (in yield order)

1. **riir-infer Issue 1148** — the snapshot codec + rewind fast path (journal unwind + KV truncate):
   unblocks ANY tok/s row for the spec lane. **CODEC HALF LANDED SAME-DAY** (the per-f32 append
   loops → bytemuck bulk copies, byte-identical LE wire): the same-process re-cell reads
   **0.86–0.95 tok/s (from 0.23, ~17–18×)**; and at k=2 the mtp_head row now beats repeat_last on
   BOTH axes even on the degenerate stream (tok/s 0.975 vs 0.863, E[m] 1.124 vs 1.061, rounds 170
   vs 180 over 192 tokens) — the trained head is net-positive per-round. The structural half
   (journal unwind + KV truncate — no serialize at all) owns the remaining ~15×.
2. Chat-shaped-prompt acceptance re-read (the dflash chat lane's prompts) — **DONE 2026-10-28, [Bench 924](../.benchmarks/924_mtp_head_chat_transfer.md): the answer SPLITS BY HEAD — the plain head does NOT transfer (0.684 raw → 0.085 chat); the balanced-mode mix head DOES (0.6705 raw / 0.4353 chat, E[m] 1.73 vs the chat repeat floor 1.00, spec 5.35 vs ordinary 7.69 tok/s = 0.695×)**. The chat token file ships as the offline `mtp_chat_tokens` example (dist_stage `chat_turn` template, 8 tiles of 38 hand-written turns, disclosed). Serving-follow-up re-ordered: the re-fit (4) is chat-shaped capture + balanced mode; the plain head is DEAD for serving.
3. The 2-box GPU cell at split 63 (amo ⇄ bmo) once 1148's structural half lands — the
   RTT-amortization axis (rounds/token = 1/E[m]) is the pipeline's real win shape. The drafter
   for that cell is the MIX head (Bench 924), never the plain one.
4. Re-fit consideration: a head fit on the split-63 stream directly (riir-train) would close the
   remaining 10.9-point basis gap; rides owner call after 1148. **Bench 924 sharpens the ask:
   chat-SHAPED capture + BALANCED-mode fit — the balanced head's cross-regime transfer (0.67
   raw / 0.44 chat, un-fit) is the measured evidence; the plain head's 0.684→0.085 collapse
   disqualifies modal-heavy solving for serving.**
