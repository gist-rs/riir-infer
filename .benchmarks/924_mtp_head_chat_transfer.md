# Bench 924 — MTP heads on chat-shaped prompts: the distribution-transfer read (Bench 917 follow-up 2)

**Date:** 2026-10-28 · **Box:** shikuwa-4090 (Windows 11, i7-13700K 24 threads) ·
**Build:** `cargo run --release` (`riir-infer-dist`, feature `bonsai2_hadamard`), isolated `CARGO_TARGET_DIR=C:/t922chat` ·
**Posture:** the Bench 918 in-process 2-stage CPU pair over loopback TCP, split 63, ctx 4096, prompt 768, gen 192, k 2 ·
**Model:** `riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` (b4f6ab953ef6, the league artifact) ·
**Runs:** 5 harness invocations, 12:25–14:01Z, GPU idle throughout (4%, 416 MiB, no compute apps), no competing cargo.

## Why this run exists

Bench 917's free-run caveat: the raw-text stream degenerates (the model's greedy decode collapses
into a (0, 271) period-2 attractor — a model property), so the head's free-run E[m] there is
stream-polluted, and "the serving-shaped row needs chat-shaped prompts (follow-up 2)". This read
answers follow-up 2 — for ALL THREE head artifacts on disk, one session, one posture.

## The chat token file (new, reproducible)

`crates/riir-infer-dist/examples/mtp_chat_tokens.rs` (this landing) renders 38 hand-written
serving-shaped `(user, assistant)` turns in `dist_stage::chat_turn`'s exact template
(`<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n`), tiled 8× with
parametric variation on the numeric turns (same intents, different instances — substitution BY
TURN INDEX, never substring replacement over free text), BOS once at the head. Tokenized with the
model's own `BpeTokenizer::from_gguf`; output raw u32-LE, the harness's `read_tokens` format.

**Provenance disclosure:** the transcript is hand-written (support / classification / arithmetic /
code / analysis turns — the repo's serving-prompt precedents), NOT model-generated, and the data
file is gitignored (`riir-train/data/bonsai_mtp/tokens.chat.tokens`, 14,075 tokens). This is a
DISTRIBUTION re-read: the prompt distribution is the treatment; assistant replies only need to be
distributed like real assistant output. Regenerate with `--reps 8` (deterministic — no sampling).

## The 3×2 table — all cells this session (raw-TF plain cell = Bench 917's same-day row; raw free-run plain = Bench 918's ck-path re-cell, quoted for the free-run column)

**Teacher-forced per-slot @ split 63, 4000 rows/cell:**

| head \ stream | raw (`tokens.read.tokens`) | chat (`tokens.chat.tokens`) |
|---|---|---|
| plain `head/` (support 2963, modal-argmax TF fit — Bench 677) | 0.6841 *(Bench 917)* | **0.0850** |
| mix `head_mix/` (support 1212, `balanced_mode_fit: true`, amo) | 0.6705 | 0.4353 |
| freerun `head_freerun/` (support 1674, decoded-trajectory fit — Bench 679, **the promotion candidate**) | 0.5465 | 0.3030 |

**Free-run (spec pair, k=2, 768-token window) — E[m] / spec tok/s / vs ordinary:**

| head \ stream | raw | chat |
|---|---|---|
| plain | 1.6754 / 5.032 / 0.66× *(Bench 918)* | 1.0730 / 2.491 / 0.32× |
| mix | 1.6050 / 4.631 / 0.60× | 1.7297 / 5.345 / 0.695× |
| **freerun** | 1.4148 / 3.807 / 0.50× | **1.8725 / 6.207 / 0.808×** |
| repeat_last baseline | 1.2484 / 3.081 | 1.0000 / 2.348 |
| ordinary pair (incumbent) | 7.587–7.72 | 7.681–7.69 |

Calibration rows (chat): model-argmax consecutive-repeat **0.2706** (raw: 0.3878; the fit split's
repeat read 0.4150 — chat drafting is structurally harder); argmax-in-support chat: plain 0.6542 /
mix 0.7560 / freerun 0.7843 (raw: freerun 0.8855).

## The verdict — three findings, one per head

1. **The plain head does NOT transfer** — 0.684 → 0.085 TF (8× collapse), E[m] 1.68 → 1.07 (≈ the
   repeat floor). Its modal-heavy TF-fit solution rides raw-text token statistics. Dead for
   serving chat — and it was already superseded as candidate by Bench 679; nothing here changes
   that, it only removes the last reason to cite its raw rows as serving acceptance.
2. **The balanced-mode mix head TRANSFERS** — 0.6705 / 0.4353 TF, E[m] 1.73 on chat (vs 1.00
   floor, 1.73× per-round). Balanced solving (per-support-token contribution) buys features that
   survive the distribution change. It is the best TF reader on chat — but see finding 3.
3. **The freerun head — the actual promotion candidate — is the best chat DRAFTER: E[m] 1.8725,
   spec 6.207 tok/s = 0.808× of ordinary, the closest-to-parity spec row ever measured in this
   lane.** Its TF read on chat (0.303) is BELOW the mix head's (0.435) while its free-run E[m]
   (1.87) is ABOVE the mix head's (1.73) — TF agreement and draft usefulness DECOUPLE
   cross-regime, exactly as the exposure-bias thesis (riir-train Issue 655) predicts: the freerun
   head was fit along the model's OWN decoded trajectories, the condition the free-run measures
   and TF does not. **TF is a fit-basis metric; E[m] is the scoreboard. Do not rank heads by
   cross-regime TF.**
   - The anomaly worth naming: the freerun head's chat row (1.87 / 6.21) BEATS its own raw row
     (1.41 / 3.81). Its fit stream was the model's decoded raw-text trajectories (Bench 679); on
     the read stream the decode degenerates into the period-2 attractor, whose state distribution
     apparently matches its training data worse than chat does — or chat's higher-entropy states
     suit a trajectory-fit head. Recorded, not explained further; a chat-trajectory re-fit
     (the natural next rung) would answer it.
   - The spec lane's relative deficit also shrinks on chat (0.66× raw → 0.81× chat at the best
     head) while the ordinary-pair bar stayed flat (~7.6–7.7 both streams).

## Gates

- **G1 untouched:** no runtime code changed — the acceptance harness, spec machinery, and heads
  are byte-identical to the Bench 918 posture; the only new code is the OFFLINE tokenizer example.
  The machinery's G1 battery (spec_fill_g1 12/12) is unaffected by a prompt-data file.
- **G3:** `--no-default-features` unaffected (the example compiles feature-free; auto-discovered,
  no required-features needed — it only opens the GGUF for the tokenizer).

## Honest caveats

- The mix head's FIT log lives on amo (`meta.json` host_label "amo"); its raw-read basis is
  quoted from NOTHING here — its meta + capture blake3s are the provenance. The plain-capture
  balanced arm (0.3221 raw, riir-train `fit.log`) is CONTEXT for the balanced objective class,
  not the mix head's own basis.
- n = 4000 teacher-forced rows per cell; free-run cells are 192-token generations (single window
  per row-pair, as in 917/918). The chat file is one hand-written transcript family (8 tiles of
  38 turns) — a real serving corpus would sample wider. The 0.085 collapse is far outside any
  sampling noise; the 1.73-vs-1.87 E[m] ordering (mix vs freerun) is a 0.14 gap on one transcript
  family — a wider chat corpus could reorder them, and the re-fit below supersedes both anyway.
- The heads were NOT re-fit here — this read measures the EXISTING artifacts' transfer only.

## What this changes (the follow-up ladder, re-ordered)

1. **The 654 promotion e2e's chat precondition resolves POSITIVE for the candidate it actually
   names:** the freerun head holds on chat in the metric that matters (E[m] 1.87, 0.81×). The
   owner-gated promotion decision now has the distribution axis measured, not assumed.
2. **The natural next rung (riir-train side): a chat-trajectory re-fit** — capture along the
   model's OWN decoded CHAT-shaped prompts + freerun refit (the Bench 679 recipe applied to the
   chat distribution). The freerun head's chat-beats-its-own-raw anomaly (finding 3) is the
   evidence the recipe transfers. Owner call unchanged (Bench 917 §5.4).
3. **Follow-up 3 (the 2-box linked cell at split 63): the drafter is the FREERUN head** — it
   carries the best E[m] and the best spec/ordinary ratio; a linked cell with plain would
   understate the lane by ~2.4× tok/s.
4. **TF cross-regime reads are disqualified as head-ranking metrics** — this bench is the
   citation. Rank by E[m] at the serving posture.
5. The plain head stays dead (superseded by 655 + collapsed here); the mix head remains the
   balanced-objective existence proof and the fallback drafter.
