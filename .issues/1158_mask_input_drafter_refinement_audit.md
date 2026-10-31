# Issue 1158: Mask-input drafter refinement audit — the JetSpec++ pass-1 collapse class vs our DFlash2/DSpark lane family

**Status:** OPEN — audit/prophylactic only (no GPU spend, no behavior change). Filed from katgpt-rs Research 641 (arXiv:2610.19792 "JetSpec++", 2026-10-31) with the verdict-reviewer's wider-class grep. Landed: katgpt-rs Research 641 @ `a5d48462b` (katgpt-rs develop); the retrain arm = riir-train Plan 376 T3 @ `b7b5a146` (riir-train develop).
**Claimed:** claude-amo-i1158, 1793440000, expires 2026-11-02T12:00:00+00:00

## The finding

JetSpec++ Table 4's catastrophic control: **a mask-input drafter trained on fully-noised blocks collapses under refinement passes** — acceptance length τ 5.86 → 2.39 at the FIRST refinement pass (it was never trained to read token inputs; mask embeddings at training time mean the J≥1 input distribution is out-of-distribution). Token inputs + clean-prefix curriculum is the training that makes a drafter refinable.

Our serving lane family is mask-input-shaped BY DESIGN (block-denoising drafters denoise from mask tokens — correct for single-pass J0 drafting; katgpt-rs Bench 694's contract note). The hazard is only FUTURE refinement:

| Surface | Repo / file | Shape |
|---|---|---|
| Serving drafter (GGUF contract) | riir-infer `crates/riir-infer-gpu/src/speculative_decode/dspark_drafter.rs:206` — key `dspark.dspark.mask_token_id` | mask-input (DSpark family) |
| Lane config | riir-ai `crates/riir-gpu/src/dflash_config.rs` — `DFlashConfig.mask_token_id` (+ `for_target`) | mask-input (DFlash2 family) |
| DFlash2 bench lane | riir-ai benches 746 / 747 / 759 / 779 / 989 (`mask_id` embed-row reads) | mask-input consumers |
| Trainer family | riir-train trainers reading the same contract key (verdict-reviewer grep 2026-10-31: `dflash_trainer.rs`, `distill_tsd.rs`, `distill_gdsd.rs`, `opsd_loop.rs`, +others) | mask-input training |

## Audit tasks

- [ ] **Inventory**: one table of EVERY surface that reads `mask_token_id` (serving configs + trainers + benches), per repo — confirm the four rows above and complete the trainer list.
- [ ] **Prophylactic assert**: none of these surfaces may be run multi-pass / self-refined as-is. Add a loud refusal (or a pinned doc line at the config) wherever a refinement-pass count COULD be introduced — the guard names this issue.
- [ ] **Pass-1 cliff detector**: pin the signature as a cheap check — if any lane ever adds refinement passes, a τ ladder whose J1 value DROPS below J0 is the mask-input signature (JetSpec++ Tables 4/12); record it as a gate note next to the verify seam.
- [ ] **Cross-link**: the token-input RETRAIN arm is riir-train Plan 376 §JetSpec++ recipe T3 (amended 2026-10-31) — that plan owns the fix; this issue owns the audit of what exists today.

## Non-goals

- Not a bug in the current single-pass posture: J0 mask-denoising drafting is the lane's correct contract and stays untouched.
- No retraining scheduled here (Plan 376 Tier A, ~110–130 GPU-h, is the vehicle — separate gate).
