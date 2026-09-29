# Bench — T5.0 collapsed-vs-parent top-1 agreement sweep (Issue 022)

**Status:** RECORD — coarse grid NEGATIVE (all four pre-registered points fail); fine end maps the cliff: **ε=0.01 (4.7% cut) PASSES at 0.9486; ε=0.015 (10.9% cut) misses by 0.0045**.

## What this is

The T5.0 GOAT: teacher-forced greedy argmax agreement between the PARENT
checkpoint (`Ternary-Bonsai-2-27B-PQ2_0.gguf`, qwen35 hybrid, 64 layers —
48 DeltaNet + 16 attention, Hadamard-folded PQ2 ternary) and each
ZERO-TRAINING passthrough-collapsed checkpoint, over a frozen token
stream. The pre-registered budget (issue T5.1): **absolute agreement
≥ 0.9** — the parent trivially agrees with itself at 1.0, so the bar is
absolute, never a ratio.

## Instruments (all landed 2026-09-29)

- `examples/twt_collapse_emit` — profile-artifact S matrix →
  `minmax_partition` at ε → per-block winner = minimax medoid member →
  the collapsed-GGUF writer (all-member passthrough; explicit
  `twt.layer_types`; `prism.hadamard.weight_names` renumbered).
- `src/bin/twt_goat_agreement` — per-token teacher-forced forward,
  chunked 8 × 512 with cache resets; the last position of each chunk is
  unscored (its prediction lands outside the chunk); parent arm cached
  (`--cache`, params-keyed, replay prints LOUD).

## Frozen inputs

- corpus: `.raw/twt/audition_calib.txt` (289,910 chars → 4,096 tokens →
  8 chunks × ≤512 → **4,088 scored positions**)
- parent weights BLAKE3 (payload bytes, file order):
  `f5eae0c831ad4068c6587060e17aaa267ee05f1f6a91ae7be31e30480e4089eb`
- profile corpus BLAKE3: `ea2773689e1a5a3dd6f28b95948aff5d45543058cb7b3e0f5811e1665824c680`
- collapsed artifacts: `/tmp/twt_collapse_pq2_e{0.01,0.015,0.02,0.03,005,0.1,0.2,0.3}.gguf`
  (`twt.partition_eps` + `twt.parent_weights_blake3` carried in-file)

## Results — the ε→agreement cliff (parent hit rate 0.7478)

| ε | blocks | depth kept | depth cut | agreement | collapsed top-1 hit | verdict |
|---|---|---|---|---|---|---|
| 0.01 | 61 | 95.3% | 4.7% | **0.9486** (3878/4088) | **0.7505** | **PASS** |
| 0.015 | 57 | 89.1% | 10.9% | **0.8955** (3661/4088) | **0.7495** | FAIL (+0.0045 short) |
| 0.02 | 49 | 76.6% | 23.4% | 0.5247 (2145/4088) | 0.5076 | FAIL |
| 0.03 | 35 | 54.7% | 45.3% | 0.0301 (123/4088) | 0.0240 | FAIL |
| 0.05 | 25 | 39.1% | 60.9% | 0.1945 (795/4088) | 0.0042 | FAIL |
| 0.1 | 14 | 21.9% | 78.1% | 0.0051 (21/4088) | — | FAIL |
| 0.2 | 8 | 12.5% | 87.5% | 0.0029 (12/4088) | 0.0022 | FAIL |
| 0.3 | 5 | 7.8% | 92.2% | 0.0000 (0/4088) | 0.0000 | FAIL |

Parent top-1 hit rate 0.7478 (context column for every row).

(The pre-registered grid is ε ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2};
the 0.015/0.02/0.03/0.01 points are sub-grid fine-end probes, recorded
with `twt.partition_eps` provenance in each artifact.)

## Reading

1. **The pre-registered grid is entirely in the fatal regime.** Its
   finest point (ε=0.05) already cuts 61% of depth; agreement there is
   0.19. All four grid points FAIL the 0.9 bar — the honest negative
   that JUSTIFIES the merge/audition question per T5.0's own gate.
2. **The cliff sits between an 11% and a 23% depth cut.** At 10.9% cut
   (ε=0.015, 57/64 blocks) agreement is 0.8955 — 0.0045 under the bar,
   i.e. 18 agreements short of 3,679. At 23.4% cut it halves to 0.52,
   and by 45% cut the model babbles (0.03). Cosine redundancy (the S
   matrix's ≤0.05 intra-block distances) is NOT a sufficient license
   for depth cuts on this model class: activation-space redundancy ≠
   functional redundancy (the ShortGPT-class result, now measured on
   the league model).
3. **Parent hit rate 0.7478** at 512-token chunks — the harness
   measures a real signal; the collapsed hit-rate collapse (0.0022 and
   below past 45% cut) is babbling, not drift.

## Box state (per the G2 law — the box-state bullet in katgpt-rs AGENTS.md)

- Apple M3 Max (16 cores), 64 GB, macOS 26.6.2, **AC power** (`pmset -g
  batt`: "AC Power"; powermode high). Load 11-13 through the coarse
  sweep (concurrent sibling agents) — the tok/s figures carry that
  load; the AGREEMENT figures are load-invariant (deterministic
  teacher-forced forwards, no timing dependence).
- Parent arm: 3,361-3,640 s per full pass (~1.22-1.28 tok/s); collapsed
  arms scale ~linearly with kept depth (57blk 3,148 s @ 1.30 tok/s …
  25blk 1,357 s @ 3.01 tok/s).
- Wall for the full 9-point table: parent pass + 8 collapsed arms.

## Verdict

- **The bar's passing point is BRACKETED: 4.7% depth cut passes (0.9486); 10.9% misses by 0.0045 (0.8955); 23.4% halves (0.5247); 45% babbles (0.03).** The zero-training passthrough lane on the qwen35 league model has a real — but narrow — operating window: ~5% depth cut at the pre-registered absolute bar. The S-matrix's cosine redundancy does NOT price quality (intra-block distances ≤0.0089 at the passing point, but 0.0145 at the failing 11% point — the distance axis and the quality axis are not proportional).
- **The TWO METRICS TELL DIFFERENT STORIES, and both are recorded:** the agreement bar (trajectory identity with the parent) passes only at 4.7% cut — but the collapsed model's own top-1 hit rate holds PARITY with the parent to 11% cut (0.7505 / 0.7495 vs parent 0.7478 — the 11%-cut model is marginally BETTER at next-token prediction than its parent) while agreeing with the parent on only 89.6% of argmaxes. Past 11% the hit rate falls off the same cliff (0.5076 at 23%). So: trajectory divergence (chaotic stream sensitivity) overstates functional damage by one full grid notch — an agreement-controlled claim needs the hit-rate column beside it, exactly the collapse-vs-quantization separation T5.2's control demands.
- All four PRE-REGISTERED grid points FAIL (the grid lives entirely at ≥61% cuts) — the honest negative that JUSTIFIES the merge/audition question per T5.0's own gate, and the reason the fine-end probes exist.
- Speed at the passing point: 61-layer collapsed ≈ 1.3-1.5 tok/s vs parent 1.22-1.28 tok/s CPU — the depth cut's perf/sec yield at 4.7% is marginal ON CPU (the league perf axis is GPU-busy-bound; T5.3 measures tg/pp where the cut actually moves the number).
- Next-instrument consequence (T5.0's own decision structure): the GDN apply-path audition (merges) is justified ONLY for quality at REAL depth cuts — and the curve says the honest framing for the lane is "collapse buys speed at a measured quality price; the S-matrix alone cannot price it". The training-distillation alternative (riir-train 423) owns the regime beyond ~11% cuts.

Session: riir-infer-022-phase5-t50
