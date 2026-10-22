# Issue 1110: β reopen memory on the gemma2_d2f commit path

**Status:** Open — consumer-side extraction from riir-train Research 480 (private repo; arXiv:2610.28543). Opt-in feature; G3 byte-identity when disabled; no promotion path without a consumer (research-scale lane — same posture as Issue 035's closure).

Premise: the decode loop's confidence commits are an **absorbing-kernel posture** — once a position commits it never reopens. The paper's central evidence (restore-capable > prune-only, concentrated on hard instances; ablation: re-corruption + confidence guidance are worth +18 pts on hard Sudoku for masked kernels) penalizes exactly this shape. This issue adds the minimal restore capability the loop can express without changing the trained model.

## Tasks

- [ ] **T1 (β reopen memory — P2).** One (token, p_at_commit) f32 pair per position recorded at commit time (the memory the paper's candidate-confidence needs: p_{i,u} = denoiser probability when u was LAST active). Reopen trigger rides `StabilityTracker`'s existing drift signals (`topk_drift` JSD-bounded, remask-reset lane) — no new detector. Reopened candidates re-enter the candidate pool with priority `p_at_commit^β`; the reopen budget is mean-preserving via the capped-mean bisection law (σ-form, never softmax). Gate: hard-prompt accuracy up / stability-alarm rate down at matched NFE; G3 byte-identity when disabled; G2 latency parity; G4 zero-alloc steady state.
- [ ] **T2 (probe first).** Before T1's GOAT: measure the drift-trigger fire rate and the flip-vs-resample win rate on our decode distribution shape (mirrors katgpt-rs Issue 956 T4). If flip-moves never beat resample-moves on our shape, the row dies honestly at probe cost.
- [-] **T3 (constrained-lane composition — deferred).** Compose with the `fa_constraint` commit-schedule surface (the constrained marginal as the reopen-priority signal) — only if T1 wins AND a structured-output consumer arms (Issue 035's re-grade trigger).
- [x] **T0.** Signal-diff recorded (commit path has no reopen memory today; `SamplerFeatures`/`StabilityTracker` carry the confidence + drift substrate). ← this filing.
