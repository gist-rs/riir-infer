# 1121 — StabilityTracker: pin the first-observed stage to ordinal 0 (a partial-capture feed silently reports index-space depth)

**Status:** Open — filed 2026-10-26. Hardening finding from the amo idle-cycle read-only smell hunt (`stability.rs` full read, zero-cargo cycle — the elephant window-2 capture held the box). Not started.
**Claimed:** zed-amo-idle-refine-g cont.5, epoch=1792995755, expires 2026-10-28T06:22Z

## Why

`StabilityTracker::observe` (`crates/riir-infer-laya/src/laya/riir/stability.rs`, Issue 1106) pins its capture contract with two `debug_assert!`s: state width, and strictly-increasing stage ordinals. **Nothing pins the FIRST observed ordinal to 0** (`CaptureStage::Embedding`).

The depth semantics depend on it: `deltas[i]` is documented as the transition leaving stage **ordinal** `i`, and `stabilization_depth` returns `Some(i + 1)` in that index space — valid as an ORDINAL only when the feed starts at ordinal 0. A tracker fed a partial/offset capture (e.g. starting at `AfterLayer(30)`) passes both existing guards (30 is trivially "strictly increasing") and returns depth values that are INDEX-space — wrong by the offset, silently.

Measured present-tense evidence: the module's own order-guard test (`out_of_order_stage_trips_the_debug_guard`, L420) starts its tracker at `AfterLayer(1)` — ordinal 1, accepted today. The guard as shipped does not express "capture order starts at the embedding".

## Impact

Calibration-only instrument, never a serving path (the 1106 record carries); the one consumer (reflex `twt_profile` depth-probe) feeds FULL captures by construction (the consumption shape observes every stage of `forward_capture`). No live bug — this is the trap pin for any future consumer that filters stages (windowed lens, per-layer-range probe) and reuses the tracker.

## Fix

1. In `observe`'s first-stage branch (`!self.have_prev`), pin the start:
   ```rust
   debug_assert_eq!(ordinal, 0, "first observed stage must be Embedding (ordinal 0) — a partial-capture feed reports index-space depth");
   ```
2. **Test interaction (the reason this is not a one-liner):** the existing `out_of_order_stage_trips_the_debug_guard` feeds `AfterLayer(1)` FIRST — after the pin it trips the NEW assert (message mismatch → the `should_panic(expected = "capture order")` fails). Rework it to `Embedding → AfterLayer(0) → Embedding` (the last step trips the ORDER guard at message), and add `first_ordinal_must_be_embedding_trips_the_debug_guard` (`AfterLayer(1)` first, expects the new message).
3. Gate: `cargo test -p riir-infer-laya --features laya-riir --lib` (the laya posture — the P43 law: a default-features run compiles the lane empty).

## Tasks

- [ ] T1 — the first-ordinal `debug_assert_eq!` in `observe` + one doc line on `observe`/`stabilization_depth` naming the full-capture precondition.
- [ ] T2 — rework the order-guard test to Embedding-first + add the new guard test (both `#[cfg(debug_assertions)]`).
- [ ] T3 — laya-posture lib test green.
