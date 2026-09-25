# Bench 003 — row-logit floor on real attention rows: ppl, sink A/B, needle (Issue 011 T2/T3/T4)

**Status:** IN PROGRESS (2026-09-25) · T2 + T4 measured; T3a (gemma 64K-width proxy) + T3b (MiniCPM5 16K) done; T3c (MiniCPM5 64K) measuring · bin: `src/bin/row_logit_floor_ppl.rs` (feature `row_logit_floor`) · primitive: katgpt-core `row_logit_floor` (katgpt-rs Bench 888)

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

### T3a — gemma-2 proxy: the 64K WIDTH on real rows (DONE)

6 passkey prompts × 1024 tokens, a 5-digit key at depth (i + ½)/6 in
tinyshakespeare filler. Every row is floored, and only the 7 answer tokens
per prompt are scored (42 in total). `n65536` gives w = 18.00 nats on every
row, so the 6-bit code step is 0.29 nats, the step a 64K context would pay.
seq-exact is the fraction of prompts whose every answer token is the argmax,
i.e. greedy retrieval succeeds. Log: `/tmp/ri011run/t3.log`. It ran
concurrently with T2, at load 14 → 18.

| arm | answer ppl | Δ | mean \|ΔNLL\| | top-1 flip | seq-exact | floored |
|---|---|---|---|---|---|---|
| base | 1.0345 | — | — | — | 6/6 | — |
| b8n65536 | 1.0347 | +0.018% | 0.00022 | 0.00% | 6/6 | 0.21% |
| b6n65536 | 1.0360 | +0.140% | 0.00144 | 0.00% | 6/6 | 0.21% |
| b6s0n65536 | 1.0345 | −0.004% | 0.00062 | 0.00% | 6/6 | 0.33% |

**m_Y (katgpt-rs Issue 882 P4's instrument, first reading):** the
attention mass on the needle span, from the question + answer rows. It is
measured on the distribution each arm USED.

| arm | m_Y (26 × 8 heads) | top head |
|---|---|---|
| base | 0.1750 | L10H5 0.928 |
| b8n65536 | 0.1750 | L10H5 0.928 |
| b6n65536 | 0.1748 | L10H5 0.929 |
| b6s0n65536 | 0.1757 | L10H5 0.928 |

The per-layer means peak at L8 (0.47), L14/L16 (0.38/0.37), L6 (0.37) and
L12 (0.35), and are ≈ 0 at L0, L2 and L25.

- **Retrieval holds at the 64K code step.** All four arms get 6/6 with 0
  flips. The width is not the binding term at 1K: only 0.2–0.3% of keys
  sit below `m_r − 18`.
- **The floor does not move answer attention.** m_Y moves ≤ 0.0009 and the
  retrieval head L10H5 keeps 0.93 of its mass on the needle. This is P2's
  falsifier, answered negative on this fixture.
- **n = 42 cannot separate s4 from s0.** There are 0 flips everywhere and
  the |ΔNLL| ordering inverts (0.00062 vs 0.00144). That is noise at this
  size, so T4's verdict rests on the T2 table, not here.
- ⚠ This is a proxy. It exercises the 64K code coarsening on real rows,
  NOT the 64K attention dilution, where `n·e^{−w}` meets 64K live keys.
  gemma-2 is 8K with a 4K SWA, so the true T3 needs a long-context
  fixture.

### T3b — MiniCPM5-1B at 16K (`--decode-floor`) (DONE)

This is the first run with real long-context dilution: 16K live keys under
every scored row, on a 131K-context llama-arch model (f32, GQA 16/2, no
softcap). It uses 3 passkey prompts × 16384 tokens and 12 scored answer
tokens. `--decode-floor true` runs the 49 137 prompt rows **once, dense**,
and shares them across arms. Each arm floors only the scored rows, which is
the decode-time low-bit consumer (its prompt KV is dense). So the arms
measure the floor on the retrieval rows. They do **not** measure error
compounding through a floored prefix. Log: `/tmp/ri011run/t5_minicpm16k.log`.
Window 04:24–05:19 (+0700), load 18 → 21, 12.2 GB free at launch, power
source not recorded. The dense prefix ran at 14.96 tok/s.

| arm | answer ppl | Δ | mean \|ΔNLL\| | top-1 flip | seq-exact | mean env TV | floored | mean w |
|---|---|---|---|---|---|---|---|---|
| base | 1.1051 | — | — | — | 3/3 | — | — | — |
| b8 | 1.1051 | +0.000% | 0.00026 | 0.00% | 3/3 | 0.0339 | 8.875% | 16.61 |
| b6 | 1.1026 | −0.232% | 0.00257 | 0.00% | 3/3 | 0.1537 | 8.885% | 16.61 |
| b6s0 | 1.1049 | −0.017% | 0.00091 | 0.00% | 3/3 | 0.1537 | 11.692% | 16.61 |
| b6n65536 | 1.1067 | +0.145% | 0.00241 | 0.00% | 3/3 | 0.1684 | 4.913% | 18.00 |
| b4 | 1.1086 | +0.319% | 0.00613 | 0.00% | 3/3 | 1.1379 | 9.009% | 16.61 |

m_Y (all 24 × 16 heads): base 0.1204, top head L10H1 0.917. b8, b6, b6s0
and b6n65536 all stay within 0.0001 of base, with the same top head.
**b4** moves it to 0.1219 and the top head switches to L15H7 (0.923).

- **Retrieval holds at 16K for every arm**, b4 included: 3/3 exact and 0
  flips over 12 tokens.
- **The floor term grows with n, as `A ≤ n·e^{−w}` predicts.** The floored
  fraction is 8.9% here against 3.8% on gemma-2 rows of ≤ 1K. The
  tv-budget width (16.61 nats at 16K) keeps the envelope bounded.
  `n65536` widens every row to 18 nats and floors about half as many keys
  (4.9%), which is the trade the T3 bullet names.
- **b4 perturbs attention even while retrieval survives.** It is the only
  arm that moves m_Y or the top head. This agrees with T2, where b4 flipped
  4.32% of ppl tokens. 6-bit is the floor for admissibility.
- ⚠ **Power: n = 12 tokens over 3 prompts.** Every Δppl sign is noise at
  this size: b6 reads *better* than base, and b6s0's |ΔNLL| is below b6.
  This row can show that retrieval did not break. It cannot rank arms, and
  it cannot resolve a sub-percent retrieval regression.
- 16K is not the 64K bar. T3c runs the same fixture at 64K.

### T3c — MiniCPM5-1B at 64K (`--decode-floor`)

_Measuring_ (launched 2026-09-25 20:28 +0700, M3 on AC, loadavg ~6.7, 91%
memory free, same binary `b615672`; the llama forward is unchanged through
`9c2d9f6`). Arms: base, b8, b6, b6s0, b4. `n65536` is omitted because it
equals the per-row width at 64K.
