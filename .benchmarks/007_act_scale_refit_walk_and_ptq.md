# Bench 007 — act-aware ternary scale refit: the born-ternary walk + the dense-parent PTQ lane (Issue 014 T2/T3)

**Status:** RECORD — T2+T3 of [Issue 014](../.issues/014_act_aware_ternary_fit_retention_walk.md)
landed 2026-09-27 (the `act_retention_walk` bin + the `act_ptq_gemma2` bin); T4 = this
doc + the katgpt-rs Issue 886 P1 close.

## What ran

Two lanes, one instrument family (feature `act_scale_refit`):

1. **Born-ternary walk (T2a/T3, the deciding gate).** `act_retention_walk` over
   `riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` (qwen35 hybrid, 64 layers =
   48 DeltaNet + 16 attention at l ≡ 3 (mod 4), Hadamard rotation ACTIVE). Six
   arms × 6 content families (chat_probe pages 0/400/800/1200/1600/1900 — one
   deterministic `.txt` bucket each) × 12 items × 64 tokens = **4,608 scored
   teacher-forced positions per arm**. The T1 diagonal (Bench 005, commitment
   `5b6aead0901e3133cd7ee28512e75122c788ae9d720dec78d099e6b390d9a9d2`) is the
   refit input; every refit re-derives from the SHIPPED payloads (per-arm
   reload — never a prior arm's output). Metrics per family vs the `shipped`
   reference: argmax flips, flips at reference margin > 1.0 (hi-margin), top-8
   retention (ref argmax ∈ arm top-8), reference margin at flips, gold NLL
   (context column only — aggregate PPL is disqualified by the lossy-surface
   law; the conditional walk is the gate).

2. **Dense-parent PTQ lane (T2b, the transfer test).** `act_ptq_gemma2` over
   `riir-train/data/gemma-2-2b-it-f16.gguf` (the house fixture): 8,192 tokens
   of chat_probe through `forward_gemma2_f16_act_tapped` (the Issue-886
   LINEAR-INPUT taps — 4 distinct inputs/layer: attn_in, o_in, ffn_in,
   down_in — the 883 fork's structure re-forked), per-tap `E[x²]` diagonal +
   48 held-out activation vectors per tap (stride 16, min_pos 64). The 7
   per-layer linears are ternarized from their true f16 parents under five
   fits; the metric is Bench 896's own G1 — held-out
   `E‖(W−Ŵ)x‖²/E‖Wx‖²` (sqrt) — at layer level (no full-model ternary gemma
   forward exists; the layer-level read is the honest scope of this lane).

## Box state (the G2 law)

4090 workstation (shikuwa), i7-13700K 16 cores, **CPU lane** for both lanes;
AC power; free RAM 16.2 GiB / 33.3 GiB at walk launch, 8.3 GiB late in the
run (both lanes' resident models); GPU idle (≤ 5%, 507 MiB — neither lane
touches it). The walk ran solo ~18:14–01:00; the gemma lane overlapped its
last ~2.5 h capped at `RAYON_NUM_THREADS=3` (both lanes are correctness
reads, no timing claims). Walk rate 2 tok/s on the reference arm, drifting
to ~0.75 tok/s late (long-run drift, unexplained, does not affect any
verdict column — every arm walked the identical items).

## Results — the born-ternary walk (Ternary-Bonsai-2-27B-PQ2)

Refit payloads vs shipped (25,598,361,600 weights total):

| arm | rule | diagonal | payload changed | scale ratio (med) | refit wall |
|---|---|---|---|---|---|
| shipped | (reference) | — | — | — | — |
| mean_abs | `quantize_from_f32` | — | 14.252% | 0.6683 | 311 s |
| wma_ex2 | `WeightedMeanAbs` | E[x²] | 14.629% | 0.6683 | 347 s |
| ws_ex2 | `WeightedSearch` | E[x²] | **0.001%** (181,268 w) | **1.0593** | 3,405 s |
| ws_uniform | `WeightedSearch` | uniform | **0.000%** (350 w) | **1.0593** | 3,557 s |
| zeroqat | mean-abs codes + multiplier GD (default knobs) | E[x²] | 14.252% (codes) | 0.6683 | 875 s |

The walk (4,608 positions/arm; per-family values in `walk.jsonl`, summarized):

| arm | gold NLL/token | flips / 4,608 | hi-margin flips | top-8 retention |
|---|---|---|---|---|
| shipped (ref) | 2.6836 | — | — | — |
| mean_abs | 8.7791 | 4,457 (96.72%) | 2,191 | 19.49% |
| wma_ex2 | 8.6276 | 4,339 (94.16%) | 2,108 | 21.70% |
| **ws_ex2** | **2.6838** | **31 (0.67%)** | **0** | **100.00%** |
| **ws_uniform** | **2.6836** | **2 (0.04%)** | **0** | **100.00%** |
| zeroqat | 9.0953 | 4,578 (99.35%) | 2,227 | 11.52% |

Every ws flip sits at reference margin < 0.04 (`ref_margin_at_flips_mean`
0.006–0.033) — coin-flip territory; the aggregate gold NLL moves in the 4th
decimal. Per-family rows are uniform (no family hides damage): families read
flips 3–9/768 (ws_ex2) and 0–1/768 (ws_uniform).

## Verdict — born-ternary lane: the shipped payload is the fit family's FIXED POINT

1. **The requant controls are catastrophic — issue trap 2 in its strongest
   form.** The shipped `Q2_0` encoder sets each group scale to **amax**, so
   mean-abs requantization of a born-ternary group collapses the scale to
   `s·nnz/128 ≈ 0.668·s` (measured med 0.6683) — a systematic 33% weight
   shrinkage — and the carry loop, accumulating `s − s′` per nonzero,
   re-derives 14.25% of codes. The model degrades from NLL 2.68 to 8.78 with
   ~97% argmax flips **uniformly across all six families** — the walk's
   positive control, proving the instrument detects payload damage loudly.
   The closed-form act-aware arm (`wma_ex2`) is the same failure (14.629%
   codes, NLL 8.63): the diagonal weights a mean-abs collapse, it does not
   prevent it.
2. **The search variants recover the shipped payload — diagonal-independently.**
   `ws_ex2` and `ws_uniform` both land at scale ratio med 1.0593 with
   ~0% payload delta: on born-ternary values {−s, 0, +s}, reconstruction
   error → 0 at s ≈ s_shipped **for every weighting u**, so the weighted
   search's optimum is the shipped scale regardless of the diagonal (the
   small 5.9% overshoot is the f16-grid's best zero-error point above
   s_wma). The diagonal cannot add information where there is no
   quantization error to redistribute — **T1's "swiglu is near-uniform"
   null prediction generalizes to the whole lane for a mechanical reason**.
3. **The ZeroQAT-class comparator is structurally stationary at its
   insertion point, and its surrogate misaligns with the model when it CAN
   move.** Two recorded facts, both at riir-train default knobs (100 steps,
   lr 0.01, ε 0.01, multiplier init 1.0): (a) on SHIPPED born-ternary codes
   the layer-local surrogate `Σu(w−s·q)²` has gradient exactly 0 at
   s = s_shipped (every nonzero |w| equals s), so the class cannot move a
   born-ternary payload at all — the arm therefore runs at the
   mean-abs-REQUANT insertion point (codes fixed at the requant's, scales
   GD-refined against the shipped weights); (b) at default knobs the GD
   moves the multiplier ≈0% (measured: scale ratio med 0.6683 = the
   requant's own scale — the parabola curvature `C·s²` makes lr 0.01
   glacial), so the codes damage rides through untouched — AND the ~1%
   drift the GD does apply makes the model WORSE than its own starting
   point (NLL 9.0953 vs mean_abs 8.7791, flips 99.35% vs 96.72%): the
   layer-local reconstruction surrogate improved while the end-to-end
   quality degraded — **the surrogate is not aligned with model quality on
   damaged payloads**. The class's verdict on this lane: no-effect at
   defaults, structurally no-margin on born-ternary payloads, and
   negative when its objective moves at all.

**The honest prior (Research 588 §2.6) is CONFIRMED for the born-ternary
lane: the Bench-896 synthetic gains do NOT transfer — clean negative, and
the per-family conditional walk is the evidence (aggregates alone would
have hidden the requant collapse only in the ws arms' favor; here even the
aggregate is loud).**

## Results — the dense-parent PTQ lane (gemma-2-2b f16): the mechanism DOES transfer where parents exist

Held-out `E‖(W−Ŵ)x‖²/E‖Wx‖²` (sqrt), 48 vectors/tap, 26 layers × 7 linears:

| arm | overall | attn family | mlp family | worst tensor |
|---|---|---|---|---|
| mean_abs | 0.6383 | 0.6755 | 0.5857 | l25.attn_wv (1.0926) |
| wma_ex2 | 0.5443 | 0.5522 | 0.5338 | l24.attn_wv (0.8905) |
| **ws_ex2** | **0.4176** | 0.4284 | 0.4030 | l24.attn_wv (0.7779) |
| ws_uniform | 0.5230 | 0.5677 | 0.4574 | l25.attn_wv (1.0572) |
| zeroqat | 0.6382 | 0.6753 | 0.5857 | l25.attn_wv (1.0926) |

- **The act-aware search beats the activation-blind baseline by −34.6%**
  (0.4176 vs 0.6383) and the blind-search control by **−20.1%** (0.4176 vs
  0.5230) — the diagonal genuinely helps **when there are real f32 parents**.
  Bench 896's synthetic ordering transfers exactly (search > closed-form >
  baseline; E[x²] diagonal > uniform), at roughly half the synthetic
  magnitude (−20% here vs −54% there — gemma's activations carry no planted
  20× channels; T1's flatness table is the transfer predictor).
- `zeroqat` at default knobs = `mean_abs` to 4 decimals (0.6382 vs 0.6383):
  on dense parents the surrogate is non-degenerate (unlike born-ternary) but
  the default lr is still ~100× too small for the curvature — the class's
  default knobs are miscalibrated for per-group ternary scales, its second
  recorded fact.
- The worst tensor is `attn_wv` in the last layers for every blind arm —
  the V projection is where ternary PTQ hurts most (consistent with the
  Issue-883 V-work intuition), and the only tensor class the act-aware
  search pulls under 0.78.

## Honest caveats

- **Reduced scale, honestly:** the walk's diagonal is the 8,192-token T1
  artifact (AWQ sampling floor); the walk itself is 4,608 positions/arm over
  6 chat_probe families; the gemma lane calibrates AND evaluates on the same
  8,192-token slice (reconstruction, not generalization — AWQ's own
  protocol, still a reduced-scale read).
- **Families = chat_probe pages** — one corpus, one domain. The artifact +
  digests make every slice reproducible.
- **Family 0 overlaps the T1 calibration slice** (pages 0's first ~2k tokens
  fed the diagonal). Families 1–5 (pages 400–1900) are strictly held out
  relative to the diagonal. No arm's verdict hinges on family 0.
- **Walk rate drift** (2 → 0.75 tok/s across ~7 h) is unexplained; every arm
  walked byte-identical items with identical kernels, so the verdict columns
  (flips/retention/NLL) are unaffected — but no timing claim may cite this
  run.
- **In-process refits; the checkpoint is never written.** The refit payloads
  live and die inside the runs.
- `wte` is out of refit scope by construction (a row lookup, not a matvec —
  no input diagonal exists for it); `in_proj_a`/`in_proj_b` are the
  Issue-980 dense escape set on Bonsai-2 (never ternary).

## Artifacts

- Walk: `.raw/act_diag_005/walk.log` (progress + refit stats),
  `walk.jsonl` (one line per arm: refit stats + per-family walk rows),
  `walk_report.md` (the bin's markdown report).
- Gemma lane: `.raw/act_diag_005/ptq.log` + `ptq_report.md`.
- Families: `.raw/act_diag_005/families/family_{0..5}_page_{0,400,800,1200,1600,1900}.txt`
  (deterministic extraction of the six chat_probe pages).
- Diagonal artifact (T1): `.raw/act_diag_005/diagonal.bin`, commitment
  `5b6aead0901e3133cd7ee28512e75122c788ae9d720dec78d099e6b390d9a9d2`.
- `.raw/` is gitignored; the pipeline is deterministic (fixed slices, fixed
  seeds, no RNG anywhere) and the bins re-derive everything from the
  checkpoint + corpus.

MEASUREMENT-ONLY (Issue 014 P0 law): no serving path is touched; the fits
ship opt-in in katgpt-rs (`act_aware_fit`) and stay there — the born-ternary
verdict is "no consumer on this lane", the dense-parent verdict is
"mechanism verified model-bound; adoption is a PTQ-authoring decision".
