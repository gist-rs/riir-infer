# Bench 924 — MTP head on chat-shaped prompts: the distribution-transfer read (Bench 917 follow-up 2)

**Date:** 2026-10-28 · **Box:** shikuwa-4090 (Windows 11, i7-13700K 24 threads) ·
**Build:** `cargo run --release` (`riir-infer-dist`, feature `bonsai2_hadamard`), isolated `CARGO_TARGET_DIR=C:/t922chat` ·
**Posture:** the Bench 918 in-process 2-stage CPU pair over loopback TCP, split 63, ctx 4096, prompt 768, gen 192, k 2 ·
**Model:** `riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` (b4f6ab953ef6, the league artifact) ·
**Runs:** 3 harness invocations, 12:25–13:20Z, GPU idle (4%, 416 MiB, no compute apps), no competing cargo.

## Why this run exists

Bench 917's free-run caveat: the raw-text stream degenerates (the model's greedy decode collapses
into a (0, 271) period-2 attractor — a model property), so the head's free-run E[m] there is
stream-polluted, and "the serving-shaped row needs chat-shaped prompts (follow-up 2)". This read
answers follow-up 2 — and the answer SPLITS BY HEAD:

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

## The 2×2 transfer table — teacher-forced per-slot @ split 63, 4000 rows/cell, ONE session

| head \ stream | raw (`tokens.read.tokens`) | chat (`tokens.chat.tokens`) |
|---|---|---|
| **plain head** `head/` (support 2963, modal-argmax fit) | 0.6841 *(Bench 917, same day, same posture)* | **0.0850** |
| **mix head** `head_mix/` (support 1212, `balanced_mode_fit: true`, amo fit) | **0.6705** *(this run)* | **0.4353** *(this run)* |

Calibration rows (chat cells): model-argmax consecutive-repeat **0.2706** (the fit split's repeat
read 0.4150 — chat drafting is structurally harder), plain-head argmax-in-support 0.6542,
mix-head argmax-in-support **0.7560** (chat) / **0.8788** (raw).

**The split verdict:**

- **The plain head does NOT transfer.** 0.684 → 0.085 off-regime (8× collapse). Its modal-heavy
  solution rides the raw stream's token statistics; chat text almost never visits that regime.
- **The balanced-mode head DOES transfer.** 0.6705 raw / 0.4353 chat — the chat read is *higher
  than any raw read basis the balanced objective class has ever printed* (the plain-capture
  balanced arm read 0.3221 on raw, riir-train `fit.log`). Balanced solving (per-support-token
  equal contribution) buys features that survive the distribution change; modal-heavy solving
  buys accuracy that does not.
- Chat E[m] (free-run, k=2, 768-token multi-turn prompt window): mix head **1.7297** vs
  repeat_last 1.0000 (**1.73× per-round**); spec lane **5.345 tok/s** vs ordinary pair 7.686
  (**0.695×** — the spec pair's relative deficit SHRINKS on chat; it was 0.66× on raw at Bench
  918). The plain head on chat: E[m] 1.0730 (≈ the repeat floor), 2.491 tok/s (0.32×) — net-useless.

## Gates

- **G1 untouched:** no runtime code changed — the acceptance harness, spec machinery, and heads
  are byte-identical to the Bench 918 posture; the only new code is the OFFLINE tokenizer example.
  The machinery's G1 battery (spec_fill_g1 12/12) is unaffected by a prompt-data file.
- **G3:** `--no-default-features` unaffected (the example compiles feature-free; auto-discovered,
  no required-features needed — it only opens the GGUF for the tokenizer).

## Honest caveats

- The mix head's FIT log lives on amo (`meta.json` host_label "amo", created 10-28 03:3xZ); its
  raw-read basis is quoted from NOTHING here — its meta + capture blake3s are the provenance. The
  plain-capture balanced arm (0.3221 raw) is CONTEXT for the balanced objective class, not the
  mix head's own basis.
- n = 4000 teacher-forced rows per cell, one box, one session. The chat file is one hand-written
  transcript family (8 tiles of 38 turns) — a real serving corpus would sample wider. The 0.085
  collapse is far outside any sampling noise; the 0.4353-vs-0.6705 balanced gap is the number a
  chat-shaped re-fit would have to beat, and wider coverage could move it either way.
- The mix head's chat win does NOT make the spec lane serving-positive yet: 5.35 vs 7.69 tok/s
  on a CPU loopback pair still loses e2e; the win shape remains the LINKED posture (real RTT to
  hide, Bench 918's analysis) — and on chat the ordinary-pair bar itself dropped 7.62 → 7.69
  ≈ flat, while the spec lane gained 5.03 → 5.35.

## What this changes (the follow-up ladder, re-ordered)

1. **Follow-up 4 (re-fit) is now concrete and UNGATED-in-shape:** re-capture with CHAT-SHAPED
   prompts and fit BALANCED-MODE — the balanced head's cross-regime robustness is measured
   here; a chat-fit balanced head plausibly clears 0.5+ chat-side. (The re-fit itself still
   rides the owner call recorded in Bench 917 §5.4 — this read is the evidence it asked for.)
2. **The plain head is DEAD for serving** — any promotion path must pin the balanced head (or a
   chat-fit successor). The `head/` artifact's Bench 677 gates read raw-only; they do not
   transfer and must not be cited as serving acceptance.
3. Follow-up 3 (the 2-box linked cell at split 63) now has its drafter: the MIX head, not the
   plain one — a linked-posture cell with the plain head would understate the lane.
4. The 654 promotion e2e's chat precondition: RESOLVED NEGATIVE for the plain head, POSITIVE-shaped
   for the balanced head — the owner-gated promotion decision now has the distribution axis named.
