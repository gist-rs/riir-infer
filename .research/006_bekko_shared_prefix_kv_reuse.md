# 006 — Bekko System One: shared-prefix K/V reuse for multi-candidate decision scoring (distill for the laya lane)

**Status:** DISTILLED — pending owner decision (the serving win is a TRAINING-ROUND adoption for the next laya re-train; no inference-time change is safe).

Date: 2026-10-01 · filed from the riir-reflex Bench 103 session (bekko vs reflex gate; the
`--bekko` comparison lane landed there — `.benchmarks/103_bekko_v0_17m_gate/`).

## Source

- `hotchpotch/bekko-system-one` (training + inference code) and the
  `bekko-system-one-v0-{17m,68m,400m}` models, pinned at the card's release revisions:
  17M `2147c3d9d00559bf972616c2589eee967e266f3b`, 68M `6eb1bae2d35066b0d634fabaf8c79beafc6fd9f1`,
  400M `4aeb85b9d4042d75d8b8adf6ff7ba9e4629510ba`. Base: `cross-encoder/ettin-reranker-*`
  (Ettin lineage → ModernBERT). ⚠ The card assigns **no license yet** ("remains to be
  finalized") — the technique is noted from the public architecture description; no code was
  copied.
- Release blog: "Bekko System One: building ultra-small decision models that run in a
  browser" (2026-09-30). Independent external confirmation of the same economics: AgentJev's
  published wide-load table (shared-prefix 298.91 ms vs unshared 609.65 ms at 66 paths /
  33,547 tokens, backbone token-ops 33,547 → 2,551 = 92.4% reduction) — quoted in our own
  typed-decisions landscape footnote since Issue 025.

## The technique (what bekko ships)

A cross-encoder decision model (Noul/Choice/Score — our wire's exact vocabulary) whose
attention is MASK-ASYMMETRIC:

```
{state, instruction}  → encoded bidirectionally ONCE = the shared prefix
   ├─ {candidate A branch} → attends to prefix + its own tokens → pooled → score
   ├─ {candidate B branch} → attends to prefix + its own tokens → pooled → score
   └─ … (prefix NEVER attends candidates; candidates NEVER attend each other)
```

- The prefix K/V at every layer is computed once per unique tokenized prefix within an
  inference microbatch and reused across candidates — the LLM KV-cache trick applied to a
  bidirectional encoder, made VALID by the mask asymmetry (a bidirectional cross-encoder's
  prefix representations depend on the candidate text, so nothing can be cached there).
- Costs: attention work is linear in candidates (vs quadratic for the listwise
  concatenate-everything shape); wall at their published wide-load point: 2.04×, token-ops
  −92.4%. On short inputs (their 5090 table) the win shrinks toward parity — the win lives at
  wide candidate/state loads.
- Accuracy cost: candidate branches cannot attend each other (their blog tried cross-candidate
  attention and could not train an improvement) and the prefix cannot see candidates — a
  representational restriction their training absorbs (Choice still reads 61.32 on S1MB for
  the 400M).
- Per-candidate independence: a candidate can be added/removed without recomputing the
  others' scores — decisions compose.

## The mapping onto riir-infer-laya (grounded in the code, 2026-10-01)

Our lane is **listwise per question**: `laya/tokenize.rs::build_sequence` renders ALL of a
question's options into ONE sequence — `[CLS] {type} question: {instructions} [SEP] [MASK]
opt1 [MASK] opt2 … [MASK] optN` — with the head reading the `[MASK]` positions
(`head_max_len` budget law). Wide option lists (banking77's 77) cost quadratic attention
inside the one sequence, but only ONE forward.

Multi-question cases (typed_decisions: 5 q/case) go through `system_one_packed` — ONE batched
forward, exact per-sequence attention, one drain (the T12 two-phase head reads). BUT: each
question's packed sequence repeats the full state tokens, so the state is re-encoded N times
per case. Packing amortizes GPU dispatch/execution overhead (`packed_eligible`, the 1-q
exclusion, the Bench-006-addendum-7 measurements); it does NOT dedupe the state encode.

**The transferable move is exactly bekko's mask asymmetry, applied at the QUESTION axis:**
train the next laya checkpoint so that the state tokens are masked from attending to the
question/options suffix (state ↔ state + instruction bidirectional; question/option tokens
attend to state + themselves). Then a case's state K/V is computed once and every
question-sequence becomes a short suffix branch against the cached prefix — `system_one`
turns into "1 prefix + N suffixes", and the microbatch dedup extends it across decisions
sharing a state.

## Why this is a training-round adoption, not an inference swap

The shipped laya checkpoints were trained with full bidirectional attention; flipping the
mask at inference shifts every layer's input distribution. The G5 parity gate would flag it
immediately (p-drift vs the frozen captures), and the honest expectation is degradation —
bekko TRAINED under its mask, which is what makes the restriction accuracy-neutral for them.
The adoption path is therefore: (1) riir-train's next laya round adds the prefix-mask as a
training-time config; (2) the round's own eval (holdout + our harness suites via the seat)
certifies accuracy-neutrality; (3) riir-infer-laya gains the prefix-cache forward (state
encoded once per case; per-question suffix branches; the head untouched) with a bit-exact
fallback arm (the mask off ⇒ the current packed path byte-preserved); (4) the encoder-arm
latency class re-measures — the concrete consumer is the encoder lane's serve refusal
(riir-instinct Issue 014: the encoder class reads ~49× the ~0.3 ms/row serve bar; a 5-q typed
case that currently re-encodes the state 5× is the first cell to improve).

## Non-goals / honesty

- This does NOT make laya a bekko replacement or vice versa: bekko-17m/68m measured MIXED on
  our suites (riir-reflex Bench 103 — reflex 0.7058 vs bekko-68m 0.6446 overall, bekko wins
  4/7 suites) and carry no license; our lane stays ours.
- The 2.04×/92.4% figures are their box and their load — quoted as published, never as our
  headroom. Any adoption claims re-measure on our kernels per the league law.
- The bekko training recipe itself (Ettin-reranker init beats encoder-init at 68M; 153
  decision-converted NLP/retrieval subsets; shared-prefix format trained end-to-end) is
  riir-train lane intel — filed there as riir-train Issue 607 (S1MB external eval: laya-typed
  reads 15.00 avg vs Jev 59.59 on the domain benches while beating bekko-17m on the
  generalization view; card-reported, unreproduced).
