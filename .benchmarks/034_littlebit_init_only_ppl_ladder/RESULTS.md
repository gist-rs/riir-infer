# Bench 034 — LittleBit init-only (Dual-SVID) sub-1-bit PTQ: the PPL ladder verdict — NO-GO for the format+kernel lane

**Status:** RECORD — a measured negative, adjudicated at T5 with the pre-registered
per-target criterion PLUS the absolute-collapse reasoning (below). The deterministic
transform + its pins live on behind `svd_lbit` (default-off) as the
deterministic-initializer record for any future QAT run (riir-train Issue 620).
2026-10-09, M3 Max (loaded throughout — PROVENANCE below). Issue 036 T3/T5.

## Setup (the T0 record, commit `8d3fd7e`)

- Host checkpoint: `gemma-2-2b-it-f16.gguf` (2.6B — deviation from the issue's
  ≤1.3B hint: the only on-disk dense checkpoint with loader+tokenizer+forward
  support here; MiniCPM5-1B absent + arch unsupported).
- Corpus: WikiText-2 raw **test** (HF datasets-server fetch, 4,358 rows, sha256
  `aca2f46735043bcfd0a44eca981d04627b9cdf74c4c9a04bf0856d04066f58fc`), first
  48 non-overlapping 1024-token windows = **49,152 tokens / 49,104 nll terms**,
  seed `0x36`, f64 accumulation.
- Eval: materialize f16(Ŵ) into the live weights → the stock
  `forward_gemma2_f16` causal forward; the sandwich forward itself is NOT
  exercised (that is the T4 kernel lane, gated on GO — never reached).
- Transform: seeded Halko truncated SVD (fastrand, q=1 power iteration, +16
  oversample; ONE k_max basis per tensor serves lbit1 + lbit2-path1 + svd at
  ALL targets — the T0 economy), Dual-SVID init (rank-1 via deterministic
  power iteration on the nonneg magnitudes), staged restack for lbit2.
  Implementation: `b235f4e` (10 unit tests, planner pins vs the paper's
  Appendix D: r=546@0.55/4096², r=133@0.1/4096×11008).

## The ladder (PPL; anchor 18.6698)

| arm | 1.0 | 0.55 | 0.3 | 0.1 |
|---|---|---|---|---|
| anchor (f16) | 18.67 | — | — | — |
| lbit1 (1-path) | 6.92e8 | 2.46e9 | 7.35e9 | 3.13e8 |
| lbit2 (2-path restack) | 4.71e8 | 1.16e9 | 8.29e8 | 3.81e9 |
| plain-SVD low-rank (f16 factors, matched bpw) | 3.47e8 | 1.33e9 | 2.55e9 | 1.35e10 |
| RTN-binary (per-row-mean scaled sign) | **3.45e7** | — (floor ~1.03 bpw, cannot arm) | — | — |

Achieved bpw within ±0.011 of nominal at every cell (the report's per-tensor
table, `lbit_ladder.md` copied here; JSON sidecar alongside). Full pass table
with per-pass transform/ppl seconds + loadavg: `lbit_ladder.md` (also
`.raw/lbit_ladder.json`).

## The verdict

**Per-target, vs the best armable baseline at matched bpw:**

- **1.0: NO-GO.** lbit loses to BOTH real baselines — rtn 3.45e7 is 14-20×
  better than either lbit arm; even plain-SVD beats both. At the ONE target
  where a real training-free method exists, the sophisticated init loses to
  `sign(W)·row-mean`.
- 0.55 / 0.3 / 0.1: lbit2/lbit1 "win" by 13% / 3.1× / 43× over plain low-rank —
  **but those baselines are degenerate there** (rank 63/34/11 low-rank is not a
  method; rtn cannot arm below ~1.03 bpw). Winning among collapsed methods is
  not "dominating the training-free curve".

**The pre-registered auto-close (NO-GO at ≥2 of 4 targets) does NOT fire** — the
mechanical count is 1 of 4. The close is therefore T5's reasoned adjudication,
on three legs:

1. **Absolute collapse everywhere.** The best lbit cell anywhere is 3.13e8 at
   0.1 bpw vs anchor 18.67 — a 7-8.5 order-of-magnitude destruction at every
   target. Nothing init-only here is usable at any bpw; the paper's own
   Appendix-F honesty (their QAT 0.55-bpw model is already hallucination-grade
   at PPL 10.47) puts our init-only 8 orders below THAT.
2. **The only real comparison loses.** See 1.0 above.
3. **The mechanism is understood** (the T1 unit-test finding, `b235f4e`): the
   Dual-SVID magnitude-scale init assumes each factor's |·| is rank-1-like;
   real LLM weight spectra are flat-and-signed, the |V′| rank-1 misfit is
   irreducible, and single-path init does not even refine naive scaled-sign
   binarization on such parents (measured at unit scale, confirmed here at
   model scale).

**T5: NO-GO for the sub-1-bit format + kernel lane.** T4 (±1 sandwich kernel
survey, Samsung repo pin) stays unexecuted by its own precondition ("no lane
until T3 says GO" — GO never came). The paper's headline claim survives as
their QAT story; the init-only point this issue existed to measure is now
**measured**: it is not a lane.

## Corroborating directional result

lbit2 (restack) HELPS at 1.0/0.55/0.3 (e.g. 6.92e8 → 4.71e8 at 1.0) and HURTS
at 0.1 (3.13e8 → 3.81e9) — the paper's own ablation direction (residual
compensation hurts at 0.1 BPW on small models, their §1.4: 60.01 vs 48.51)
reproduces at init-only, 12× amplified.

## What survives

- `src/quant/svd_lbit.rs` behind `svd_lbit` (default-off) — the deterministic
  initializer + planner for any future QAT run; local bit-identity pinned by
  test; cross-box identity holds by construction (std IEEE ops, fixed
  sweep/iteration orders, seeded fastrand; no BLAS anywhere — the T0 Halko
  decision mooted the tie-breaking clause).
- The T1 en-route find (modified Gram-Schmidt needs a RELATIVE degeneracy
  guard or rank-deficient parents destroy orthogonality: exact-rank-8 recon
  0.76 → 4e-7 after the fix) — recorded in code docs.
- riir-train Issue 620 (the recipe/QAT half) is NOT blocked by this verdict —
  but this record is its honest prior: the init it would start from is 7-8
  orders from usable, so the case for QAT-from-Dual-SVID-init (vs QAT from
  scratch/other inits) is now weaker on evidence.

## PROVENANCE (the Issue-021 law, quoted from the run)

- Box: M3 Max, macOS, AC, High Power, 16 cores. **Loaded throughout** — loadavg
  8-68 per pass (sibling sessions + this run's own 10 rayon workers); swap
  26-46 GB in use during the run. **PPL values are deterministic arithmetic and
  load-invariant; every WALL number here is shape-only** (transform seconds,
  ppl seconds — useful only as order-of-magnitude).
- Pipeline validation: the anchor reads **18.67** on wikitext-2-raw test at
  seq 1024 (the it-tuned 2.6B model) — the plausible published band, so the
  loop itself is sound and the catastrophic arms are the transform's, not the
  harness's, signal.
- Wall shape: shared k_max bases 2,875s compute (spread over ~5.5 h wall —
  thrash tax); per-pass ppl 2,685-4,247s; total run ~17 h wall under load.
- Artifacts: `lbit_ladder.md` + `lbit_ladder.json` (copied here; originals in
  `.raw/`, scratch); corpus digest above; gguf = the riir-train data pin.
