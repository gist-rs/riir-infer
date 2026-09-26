# Issue 021 — laya CUDA lane: banking77 repeat-check `determinism_ok = FALSE` at the Bench-052 stratified sample (long-sequence 77-way shape)

**Status:** OPEN — observed 2026-09-27 (riir-reflex Bench 052 §"4090 re-run",
Issue 039 T5). Reported-not-claimed there per the harness's own
`meta.determinism_scoping` law; this repo owns the kernel investigation.

## The observation

riir-reflex `results_4090.json` (harness `634093f`,
`--features laya-riir,laya-riir-cuda`, `LAYA_DEVICE=cuda`, Windows/RTX 4090,
datasets byte-verified against the M3): **banking77 laya-riir·english
`determinism_ok = false`** — the harness's within-run repeat check (first 10
cases answered twice, `render(&answers) != render(&answers2)` —
`riir-reflex/src/harness/runner.rs` `run_laya_checkpoint`) differs on at
least one repeat pair. Every OTHER cuda suite is `true` (14 of 15), the M3
metal run of the same protocol is `true` on all 15, and the measured picks
STILL match metal exactly (accuracy 0.4220 both hosts — the nondeterminism
did not move the recorded pass's picks).

| axis | value |
|---|---|
| shape | banking77 — the long-sequence (~317-token p50), 77-way suite |
| posture | cuda (RTX 4090, Windows), laya-riir·english checkpoint |
| same-run siblings | typed_decisions / massive / ag_news / xnli / emotion / sst5 / prompt / code_fixtures / harness families — ALL `true` on cuda |
| metal control | banking77 metal `true` (M3, same cases, same checkpoint) |
| picks | identical to metal (acc 0.4220 = 0.4220) |

## Why it matters

The lane's determinism discipline (G5 parity, the repeat check) presumes a
forward is a function of its inputs. A cuda reduction whose result depends
on scheduling (atomicAdd ordering, split-K partial-sum order, an
uninitialized/aliased scratch read at the long-sequence shape) breaks that
presumption intermittently and shape-dependently — exactly the class that
passes short-sequence smokes and flips a near-tie argmax in production at
some unlucky tick. The observed repeat-check fire is the FIRST cuda-side
one in the harness record (Bench 045/047/051/052 posture); the stratified
sample's first-10 cases are new cases (the round-robin split changed them),
so this is a new CASE hitting a latent kernel class, not a regression of a
previously-green lane path — the previous 4090 runs checked different
first-10 pairs.

## Candidate mechanisms (to isolate, in order)

1. **flash/split-K reduction order at seq ~317** — any atomic or
   workgroup-race partial accumulation in the score/softmax/context path
   (the long-sequence arm only banking77 reaches).
2. **Uninitialized scratch read** at the ragged long-sequence edge (pad
   region read into the reduction — ulp garbage differs per run).
3. **cuBLAS/cudarc algo heuristics** selecting a nondeterministic algo for
   the big-m shape (banking77 has the widest batch-1 m in the harness).
4. Metal-side non-suspect: same cases are bit-stable on metal, so the
   defect is cuda-arm-local (kernel or backend), not the model math.

## Repro sketch

On the 4090 box: reflex harness at `634093f`,
`--features laya-riir,laya-riir-cuda`, `LAYA_DEVICE=cuda`,
`--head-select --nb-select --datasets-dir <t20k> --suites banking77` —
the repeat check fires on the first-10 double-answer pass (it fired 1/1 in
the T5 run; re-run until characterized). Isolation tooling precedent:
riir-infer Issue 019's repeat-loop/deep probe pattern (per-tensor download
diffs at each layer) aims at this class directly.

## Acceptance

- [ ] T1 — isolate the nondeterministic op (repeat-loop probe at
      banking77's shape; per-layer download diff, the Issue-019 pattern).
- [ ] T2 — fix at the kernel (deterministic reduction order / scratch
      init), or pin the backend to a deterministic algo; NOT a harness-side
      mask.
- [ ] T3 — G5 cuda green + the reflex harness banking77 cuda
      `determinism_ok = true` on a re-run; smoke green.
