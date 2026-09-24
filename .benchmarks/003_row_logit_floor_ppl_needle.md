# Bench 003 — row-logit floor on real attention rows: ppl, sink A/B, needle (Issue 011 T2/T3/T4)

**Status:** IN PROGRESS (2026-09-25) · T2 + T4 measured; T3 gemma proxy and MiniCPM5 16K runs measuring · bin: `src/bin/row_logit_floor_ppl.rs` (feature `row_logit_floor`) · primitive: katgpt-core `row_logit_floor` (katgpt-rs Bench 888)

## What this measures

This is katgpt-rs Issue 882 P2's model-bound G1. The sink-exempt row-logit
floor `l̃ = max(l, m_r − w)` is followed by a symmetric b-bit code over
`[m_r − w, m_r]` and a 2^b-entry exp-table softmax. It runs on **every**
attention row of a real decode forward, and is compared against the plain
softmax on the **same** token sequences. Every row reports a paired delta:
per-token |ΔNLL| and top-1 flips against the base arm's argmax. Each row
also reports the mean closed-form envelope its attention rows paid.

Arm grammar: `b<bits>` uses the per-row width `w = ln(n_row/ε)`, ε = 1e-3,
with 4 sinks exempt. `s0` removes the exemption (T4). `n65536` fixes the
width at `ln(65536/ε)` for every row, which applies the 64K code coarsening
to real rows (the T3 proxy).

Box: M3 Max (16 cores, 64 GB), AC power, release build. The box was shared
with sibling compute throughout: loadavg 6 → 21, 12.2 GB free at the t5
launch. **tok/s columns are not a G2 claim**. They are sequential and
load-contaminated. G2 is Bench 888's paired kernel measurement.

## T2 — ppl Δ vs the envelope (gemma-2-2b-it f16, 4096 tokens)

4 chunks of ≤ 1024 tokens (+BOS), KV reset per chunk, 4096 tokens scored.
Mean width 12.84 nats, since rows are shorter than 1024 early in each chunk.

| arm | ppl | Δppl | mean \|ΔNLL\| | top-1 flip | mean env TV | floored | mean w (nats) |
|---|---|---|---|---|---|---|---|
| base | 33.7416 | — | — | — | — | — | — |
| b8 | 33.7528 | +0.033% | 0.00424 | 0.17% | 0.0260 | 3.786% | 12.84 |
| b6 | 33.7640 | +0.066% | 0.01731 | 0.90% | 0.1152 | 3.789% | 12.84 |
| b4 | 33.8306 | +0.264% | 0.07246 | 4.32% | 0.7540 | 3.842% | 12.84 |
| b6s0 | 33.7563 | +0.044% | 0.02341 | 1.34% | 0.1152 | 5.983% | 12.84 |

**Verdict: PASS at 8-bit and 6-bit.**

- **The code term dominates, as Bench 888 predicted.** The floored
  fraction is flat across bits at 3.79–3.84%, so the floor term is
  bit-independent. The error is the code step. Mean |ΔNLL| grows
  **4.08× per 2 bits** (b8 → b6), then **4.19×** (b6 → b4). The code step
  is `w / (2^b − 2)`: 0.0506, 0.207 and 0.917 nats at w = 12.84. Its ratios
  are 4.10× and 4.43×, so the per-token error is **linear in the code
  step**. Mean |ΔNLL| / step is 0.084, 0.084 and 0.079 across the three
  arms, which is first-order behaviour. At 4 bits the ratio sags slightly
  and becomes sublinear. Every figure in this bullet is read off the
  table, not fitted.
- **The envelope is a bound and it holds.** `mean env TV` is the per-row
  closed-form upper bound on the attention-distribution TV. Its units are
  not NLL, so the two columns are not compared as equals. What the table
  shows is that the downstream per-token cost stays 6–10× below the
  per-row bound at every bit width. The bound over-predicts more at low
  bits (4.4× from b8 to b6, 6.5× from b6 to b4, against a measured 4.08×
  and 4.19×). The envelope is worst-case over the row's mass, and it
  grows faster than the step once `e^{h} − 1` leaves its linear regime
  (h is the half step).
- **Read the flips, not the ppl.** This follows the lossy-surface rule
  (riir-ai Issue 750 T3). At 6-bit, 0.90% of next-token argmaxes flip
  while ppl moves 0.066%. 8-bit is the conservative default. 6-bit is
  admissible on ppl, and T3 decides whether it holds at long context.

## T4 — sink exemption A/B on real sinks (same run)

| | b6 (4 sinks exempt) | b6s0 (no exemption) | s0 / exempt |
|---|---|---|---|
| Δppl | +0.066% | **+0.044%** | 0.67× (looks *better*) |
| mean \|ΔNLL\| | 0.01731 | 0.02341 | **1.35×** |
| top-1 flip | 0.90% | 1.34% | **1.49×** |
| floored keys | 3.79% | 5.98% | 1.58× |

**Verdict: the exemption is load-bearing, and aggregate ppl hides it.**
Without the exemption, the sink sets `m_r`, so the floor floors 1.58× as
many context keys. The paired per-token error rises 35% and flips rise
49%. Aggregate ppl *improves*, because per-token NLL errors of both signs
cancel in the mean. This is the lossy-surface rule's failure shape on a
real model: a ppl-only gate would have promoted the defect. It confirms
Bench 888 G1b's synthetic direction (+8.3pp context TV at 6-bit) on real
gemma-2 sinks. The size is smaller here because the 50-nat softcap bounds
how far a sink can pull `m_r` above the context.

## T3 — needle

_Measuring. The gemma-2 proxy (6 × 1024-token passkey prompts, fixed 64K
width) and MiniCPM5-1B at 16K (3 needles, `--decode-floor`) are running._
