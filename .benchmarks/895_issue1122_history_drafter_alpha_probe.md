# 895 — Issue 1122 T2: the cross-call history n-gram drafter's acceptance-rate probe (NO-GO at K=2)

**Verdict:** NO-GO — the history-fed re-roll lane reads α = 0.063 mean / 0.021 median
(16 measurements) against the family's K=2 breakeven of α ≥ 0.58
(`ngram_drafter.rs` module doc); the retry-shaped upper bound reads 0.000.
The lane is net-negative at the family's operating point; T1 wiring is refused.
Record-level honest note: this is the Issue-665 honest-assessment pattern
working as designed — the negative is the deliverable.

## Posture (box state part of every claim)

- Machine: amo (M5 Ultra, 30 cores, macOS 27.0.1), AC plugged, box otherwise
  idle (`ps` showed only the Zed session's own processes; no cargo/GPU
  consumers — CPU-only run, no GPU exclusivity concern).
- Model: `riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` (the league model),
  riir-infer-core CPU reference forward (`forward_qwen_deltanet_ternary`,
  64 layers / n_embd 5120 / vocab 248320), release build,
  `--features deltanet_ternary_inference,bonsai2_hadamard`.
- Grid: 4 real arithmetic word problems (plan-461 collect's template shape,
  qwen3 chat template), 5 samples/prompt at temp 0.7, max_new 96,
  per-(prompt,sample) pinned splitmix64 seeds (master 0x1122) — the run is
  reproducible from the committed driver + seeds alone.
- Wall: 2m54s (user 22m07s — the forward parallelizes across the box).

## Controls (the protocol's own proof)

| Control | Expected | Measured |
|---|---|---|
| Greedy ×2 same prompt, α (per-drafted-token acceptance) | 1.0 else REFUSE | **1.0000** (tpf 2.667) |
| Cold start (empty history) | α = 0 by construction | 0.0000 |

The greedy control passing rules out the Bench-664 class (a broken
protocol reading a false zero): the forward is deterministic, the capture
and estimator agree, and a perfect history reads perfect acceptance on
the very same machinery.

## Results

| Shape | α mean | α median | n | tpf mean |
|---|---|---|---|---|
| Re-roll (same prompt, temp 0.7, history grows 1..4 samples) | **0.0628** | **0.0208** | 16 | 1.083 |
| Retry (operand +1, history = all 5 original samples, CROSS-KEY injected) | **0.0000** | — | 1 | 1.000 |

- Re-roll: an order of magnitude below breakeven. Two stochastic
  re-rolls of the same arithmetic problem at temp 0.7 share essentially
  no 2-gram-extendable continuations: the model's token-level paths
  diverge immediately and rarely re-converge for a full K=2 needle.
- Retry: perturbing ONE operand rewrites the output so completely that
  not one 2-token draft from the unperturbed history matched. Note this
  was measured as a CROSS-KEY upper bound (in deployment the perturbed
  prompt hashes to a different `PromptKey` and the lane drafts nothing
  BY CONSTRUCTION — the table never fabricates across prompts). The true
  deployment read is therefore the same 0.0 with less machinery.
- tpf 1.083 at K=2 means the lane's draft-side ceiling is ~8% more tokens
  per verify forward — before verify-side wall-clock costs (the Bench
  694/695 chain-seam economics: sequential verify makes every committed
  token cost a forward, so sub-breakeven acceptance loses on wall-clock
  twice).

## Why the paper's 1.99× does not transfer

NgramRL (arXiv:2610.32881) measures RL post-training EPOCHS: thousands of
re-rolls of the same prompt with a MOVING policy between rolls — the
history table accumulates hundreds of streams and the policy concentrates
probability mass on its own high-reward paths, which is exactly the
signal a rollout table harvests. A serving-side re-roll grid (5 samples,
one temperature, one fixed policy) has none of that concentration. The
signal-diff is the ENVELOPE, not the lookup machinery — this probe
measured the envelope honestly.

## Re-open conditions (recorded, not implied)

1. A consumer whose workload genuinely re-rolls the same prompt dozens+
   of times under ONE policy (e.g. a GRPO rollout loop — riir-train Plan
   472's envelope — which is the training track, not serving).
2. A measured workload where the re-roll α clears 0.58 at K=2 (re-run
   `rollout_history_probe` on the real logs; the estimator
   (`riir_infer_core::rollout_history`, 9 tests) runs offline from
   `.raw/issue1122/samples.jsonl` without the model).

## Artifacts

- `.raw/issue1122/samples.jsonl` (gitignored): every token stream
  (re-roll grid + greedy control + retry target), keyed by prompt tag +
  sample index + seed.
- `.raw/issue1122/prompts.tsv` (gitignored): the prompt grid.
- Driver: `src/bin/rollout_history_probe.rs` (required-features:
  `deltanet_ternary_inference,bonsai2_hadamard`); estimator:
  `src/rollout_history.rs` (ungated pure token-math, deterministic:
  BTreeMap keys, pinned FIFO recency ties, long→short suffix scan).
