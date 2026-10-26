# 1122 — History-fed n-gram drafter class (cross-call rollout table) in the speculative_decode family

**Status:** Open — filed 2026-10-26 from the NgramRL distillation (riir-train Research 481 / Plan 472, landed riir-train `8d3c1738`; this issue landed `c277be5`). Fusion idea, novelty TBD. Not started. Zero claims of class novelty: SuffixDecoding and prompt-lookup decoding are published prior art for the n-gram/suffix drafter class.
**Claimed:** zed-amo-sync-1122, 1792979640, expires 2026-10-28T10:00:00Z

## The gap

`crates/riir-infer-gpu/src/speculative_decode/` carries a drafter family: `ngram_drafter.rs` (context-frequency trigram + Issue-742 prompt-lookup verbatim fill), `dspark_drafter.rs` (trained), `regime_router.rs` (selection). `katgpt-rs/crates/katgpt-speculative/src/bigram_markov.rs` (Issue 659) adds the static-corpus head. **Every shipped drafter consumes either the LIVE context or a FIXED corpus. None consumes CROSS-CALL HISTORY** — a persistent, per-prompt-keyed table of what the model itself produced in prior decode calls for the same prompt.

NgramRL (arXiv:2610.32881) demonstrates the source class: recent rollouts of the same prompt are a high-quality, self-updating, model-free draft source (their setting: RL post-training epochs; measured 1.99× SuffixDecoding warm-cache). The signal-diff against both in-stack cousins is the SOURCE, not the lookup machinery:

| Drafter | Source signal | Freshness |
|---|---|---|
| `ngram_drafter` (665/742) | live generated context | within-call |
| `bigram_markov` (659) | static corpus bigrams | build-time, frozen |
| **missing class** | **cross-call history for the same prompt** | **persists across calls, bounded-recent** |

## Why it matters here (consumer map)

- Primary measured consumer: riir-train's GRPO rollout loop (Plan 472 — training-track, the paper's own envelope).
- Serving-side candidates NOT yet claimed (novelty TBD per consumer): any repeated-prompt workload — refine's fixer retry runs (same file, repeated attempts), eval harness re-runs, agent loops with stable prefixes. The regime_router is the natural integration point (a history-fed lane joins the drafter family as one more regime arm).

## Tasks

- [ ] T1 — if/when a serving-side consumer is claimed: `rollout_drafter` arm in `speculative_decode/` (per-prompt BLAKE3-keyed table, bounded-recent FIFO eviction, longest-match n-gram lookup, zero-alloc, never fabricates across prompts — draft 0 on empty history), integrated via `regime_router` as an opt-in lane.
- [ ] T2 — acceptance-rate probe on a repeated-prompt workload BEFORE any wiring (the Issue-665 honest-assessment pattern: measure α on the real workload; the lane is net-negative below the breakeven).
- [ ] T3 — determinism: pinned update/eviction order (Issue-940 law), table reproducible from the committed call log.
