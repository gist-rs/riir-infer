# Issue 1148 — the spec-lane hybrid-state snapshot costs ~4 s/round on the CPU pair (~70× the ordinary decode)

**Status:** OPEN — filed from Bench 917's wall-clock cells; investigation rung, no fix claimed yet

## The finding

Plan 623 T5.1's speculative-fill rewind (T5.1b V0) snapshots and restores the full pipeline state
per verify round via `HybridStateSegment` ("snapshot-based, byte-exact by construction"). On the
CPU pair (amo, split 63, ~600 positions, real Bonsai-2) this reads:

- spec pair: **0.23 tok/s** (k=1 AND k=2 — the per-round cost is FIXED ≈ 4 s)
- ordinary pair (same split, same prompt, same gen): **16.2–16.5 tok/s**

k=1 performs no restore at all (full accept every round) yet reads the same rate — the snapshot
alone (one `stage.snapshot_state(&mut snap)` per round, head stage) is the whole regression. The
state at that posture is ~600 positions of KV (15 attention layers) + 48 GDN conv/recurrence
windows + fill marker ≈ tens of MB; ~4 s means the codec is orders off a memcpy-class floor
(per-element encode, per-layer allocation, or a serde-shaped walk — to profile).

## Why it matters

- The spec lane's scoreboard value is E[m] ≈ 1.68 at split 63 (Bench 917 §1) — real ONLY if a
  verify round costs less than ~1/m ordinary steps. At 4 s/round it can never win, regardless of
  the drafter.
- The fast path is already NAMED in the plan: the GDN journal's rank-1 unwind
  (`deltanet/delta_roll.rs`, opt-in `gdn_delta_roll`, Bench 885: ~64× vs materialized snapshots)
  + an attention KV truncate. This issue is the wiring + the attention half.

## The rungs

1. Profile one `snapshot_state` at the Bench 917 posture (split 63, 600 positions): where do the
   4 s go (encode vs alloc vs copy)?
2. The attention KV truncate seam (`truncate_to`) — the T5.1b plan text already named it as the
   alternative to cursor-overwrite; a rewind then only unwinds GDN (journal, rank-1) and truncates
   KV — no full serialize.
3. Re-run the Bench 917 tok/s cells at the fast posture; the 70× must fall to O(1×) before any
   spec-lane perf claim.

## Ground rule

G1 stays the wall: any fast-rewind path must keep the byte-exact property the current snapshot
holds (the G1 battery's partial-accept == oracle pin is the verdict).
