# Bench 932 — Issue 1160 G0: Bonsai-2 PQ2_0 decode top-1 mass census (soft-token collapse probe)

**Status:** DONE (2026-10-31) — verdict **COLLAPSE**: pooled median top-1 mass 0.976 ≥ 0.9 bar ⇒ Issue 1160 closed without the sweep (pre-registered demote-on-wash).

## Setup
- Model `riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf`, Metal (wgpu<msl> + zero-copy raw-Metal GEMM), amo (M5 Ultra).
- `PROVENANCE: power=AC Power load=4.02 swap=3680.00M canary=skipped powermode=0(auto)` (no timing claim here; listed for completeness).
- Instrument: `crates/riir-infer-gpu/tests/issue1160_g0_collapse_probe.rs` — 8 fixed prompts (4 arith = the G1b headline class, 1 code, 2 prose, 1 zh), prefill then greedy eager decode (`set_input_token` + `forward_token`), 64 steps each, n = 512 decode steps. Per step: `p1` = full-softmax top-1 mass (the pre-registered G0 statistic), `H` entropy (nats), `s1` = top-1 weight of the PILOT soft token (top-128, τ = 1.5, renormalised).
- Raw prompts (no chat template); the model opens its own `<|im_end|>\n<think>` turn on the instruction-style ones — the regime it would actually decode in.

## Result

| family | n | p1 q25 | p1 med | p1 q75 | p1 ≥ 0.9 | H med | s1 med (pilot) |
|---|---|---|---|---|---|---|---|
| arith | 256 | 0.960 | **0.999** | 1.000 | 80.1% | 0.008 | **0.985** |
| code | 64 | 0.952 | 0.993 | 0.999 | 81.2% | 0.066 | 0.931 |
| prose | 128 | 0.469 | 0.683 | 0.967 | 35.9% | 0.891 | 0.511 |
| zh | 64 | 0.267 | 0.480 | 0.748 | 7.8% | 1.850 | 0.305 |
| **ALL** | 512 | 0.624 | **0.976** | 1.000 | 60.2% | 0.147 | 0.884 |

## Reading
- The pre-registered bar fires on the pooled median (0.976).
- It fires hardest exactly where the issue's headline gain lives: on the arithmetic class (G1b) the median step is 0.999 top-1, and even the pilot's tempered top-128 soft token puts 0.985 of its weight on the argmax — `E^T s` is the argmax embedding to within ~1.5%. That is LIFT's k=1 collapse regime: the arm cannot differ from the incumbent where it was supposed to help.
- Open-ended text (prose, zh) is NOT spiky (median 0.48–0.68, pilot s1 ≈ 0.31–0.51): a soft token there would carry real width. But the issue's claim and gate are accuracy on generated answers, not open-text quality, and those families have no exact-match gate. Recorded as the reopen condition, not a reason to arm the sweep.
- Prompt-mix caveat: the pooled median depends on the family mix (4/8 arith). The verdict does not: the headline class alone sits far above the bar.

## Decision
Issue 1160 CLOSED (demote-on-wash, G0 collapse). No code on the serving path; the probe stays as a test target (`issue1160_g0_collapse_probe`, `#[ignore]`, required-features row). Reopen only if (a) a dense / non-ternary consumer lane appears, (b) the riir-train Plan-344 LIFT arm lands a TRAINED channel, or (c) someone wants an open-ended-text quality claim, which would need its own gate (prose/zh are the only non-collapsed families measured).
