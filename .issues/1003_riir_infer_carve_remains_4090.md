# Issue 1003 — the riir-infer carve remains on the 4090 box: S6b (riir-train retarget) + S7 (residue re-export flip + dead-dep cleanup)

> **Moved 2026-09-26:** this file lives in `riir-infer/.issues/` now (moved verbatim from riir-ai, same number — riir-ai's ledger keeps 1003 allocated). Prose paths below are written from the riir-ai side at writing time; **`../riir-infer` means THIS repo**, `../riir-ai/*` is the riir-ai checkout.

**Status:** OPEN — **S6b LANDED 2026-09-24 at honest scope (4090 box, riir-train `7e0e7d74`)**: patch rows re-pointed at `../riir-infer/vendor` + the TrainingProvider/genlock adjudications + the measured S6b verdict (12/81 forwards mirrored — the "all mirrored" premise was wrong; the 69 training families stay riir-ai-side BY DESIGN, riir-infer is public per Research 003). The FULL edge-drop franchise is now **owner-gated** (where do the 69 training families live long-term: riir-gpu forever, or migrate into riir-train-gpu). **S7 LANDED 2026-09-24 (riir-ai `498b1732b9`)**: riir-gpu's dead `riir-gpu-async` row removed (measured zero-ref); riir-router measured NOT dead (`dynamic_pair_routing` is DEFAULT-on — the T2 audit's other candidate refuted, row kept); the re-export flip measured COMPLETE by compilation; no CI ripple (riir-train's CI already clones riir-infer). **The issue's 4090-executable content is DONE — what remains is the owner-gated edge-drop decision (S6b franchise) + S8 (gated on reflex P2/T4, not this box).** Filed 2026-09-24 by the M3 session after landing S6a (clippy retarget); executing plan: `../.plans/610_riir_infer_gpu_carve_slice1.md` (campaign: riir-reflex `.issues/008`, mirror `.issues/998`).

## Context — where the carve stands

S1–S5 + S6a are LANDED (plan 610 status line is the live record; reflex 008's P3 section mirrors it). S6a took riir-clippy's ternary lane fully off riir-ai (its inference surface now consumes `../riir-infer` directly; boundary contract clean at 23 repos / 321 edges). The remains:

## S6b — the riir-train retarget (the bigger consumer)

riir-train's workspace carries `riir-engine` (workspace dep, 4 lora features), `riir-gpu`, `riir-gpu-async` and consumes them through a large `riir-gpu/<feature>` forward lattice in riir-train-engine's feature table (maglev_metal_teacher, dflash2_chat_train, mux_training, ciacot_full_integration, ...) plus riir-train-gpu's `riir_gpu::` surface. The 041 target graph: riir-train's remaining riir-ai edges become the genuinely game-shaped ones (riir-games, riir-games-civ, riir-data, riir-router) — the inference edges retarget to `../riir-infer`.

Measured entry points (from the S6a session):

- `riir-train/Cargo.toml` workspace.dependencies rows for riir-engine/riir-gpu/riir-gpu-async (L13-22) + the two patch rows pointing at `../riir-ai/vendor/*` (cubecl-runtime, wgpu-hal — same re-point-to-`../riir-infer/vendor` move S6a made) + the arc-swap genlock row (NOTE: riir-train ALSO path-deps riir-engine directly today, so its genlock row is genuinely load-bearing until its riir-engine edge is adjudicated — do NOT drop it early).
- riir-train-gpu: `speculative_decode = ["riir-gpu/speculative_decode"]` + the whole re-export surface — after S6a, `riir_gpu::speculative_decode` is a re-export of the infer-gpu module, so the retarget is textual at the code level (`riir_gpu::` → `riir_infer_gpu::` where the surface is inference-only) and manifest-level for the feature mirrors (all mirrored in infer-gpu S1–S5). **⛔ MEASURED WRONG 2026-09-24 (the S6b landing): only 12 of 81 forwards are mirrored** (infer-gpu carries 37 features); the 69 unmirrored are training families staying riir-ai-side by design. The facade stays riir-train's single correct surface until the owner decides the training-families home; a partial swap was evaluated and rejected (two vocabularies for one family = churn). Full record: plan 610's S6b row.
- **The one real adjudication: `TrainingProvider`** (riir-train-engine's crate description says "Implements TrainingProvider trait from riir-engine"). 041 Q3's answer is pull-gated: rte is the training engine and stays; the question is only whether the trait re-homes to riir-train-engine (flipping the dep) or stays engine-side (keeping ONE riir-engine edge for the training trait — an acceptable residue if the trait is genuinely training-shaped and cognition-free). MEASURE FIRST: grep the trait's definition + its engine-side dependents; if it is training-only, re-homing is the clean cut; if it shares cognition types, keep the edge and record why in plan 610's S6b row.
- The 4090 angle: every feature mirror the retarget activates compiles the CUDA arms (ternary_gemv_cuda_raw / cudarc family) — the M3 cannot validate those postures; this slice's clippy/check matrix must run on THIS box (the S4a/S4b/S5 precedent).

## S7 — residue re-export flip + dead-dep cleanup (riir-ai side)

- The riir-gpu root's re-export flip: `pub use riir_infer_gpu::<mod>` for EVERY moved module (S1 established the pattern; S7 completes it) — after S6b the consumers are off the old paths, and the flip is what lets the remaining STAYS residue compile against the moved sources.
- The T2-audit dead-dep cleanup: riir-router + riir-gpu-async are zero-reference rows from riir-gpu's side (measured in the T2 audit) — remove the rows the measured graph no longer supports, with the boundary contract + ci_feature_guard as the arbiters.
- The consumer-CI ripple check: after S6b, re-derive which CI clone lists need riir-infer (the slice-1 rider list: riir-ai, riir-clippy, riir-train, riir-chain, riir-mmorpg-examples, seal-remake) — S6b may let riir-train's list shrink its riir-ai usage but riir-infer stays provisioned either way.

## Discipline (the S-series lessons, binding)

1. Fetch origin immediately before EVERY slice commit (the S4b twin-landing — a concurrent session landed its own S4b mid-flight; reconciliation cost a rebase).
2. One slice per commit-set; clippy `-D warnings` at default + cubecl + all-features postures on the 4090 before any push; lib tests serialized (`--test-threads=1`) for the CUDA-family suites (the intra-process CUDA concurrency class, S4b's 3-failure lesson).
3. The boundary contract (`../riir-ai/scripts/ci_boundary_contract.sh`) must read 0 violations + 0 rot at the end of each slice — BOUNDARY rows land in the SAME commit as the dep moves (S6a precedent).
4. Bookkeeping rides the landing commit: plan 610's S6b/S7 rows ticked + this issue closed in the same push; reflex 008's P3 section updated (the campaign mirror).

## Validation floors for S6b (measured baselines to hold)

- riir-train-engine lib at default features: 1,653 passed / 6 ignored (M3, 2026-09-24, post-S6a — the 4090 number may differ by platform rows; record YOUR box state).
- riir-train-gpu at ternary_backward_cudarc: 407/0 (the S4b record).
- The bench_603 rot class: any `GateProjWeights` shape mismatch at --all-features is the f5bfe3f9 recipe (local gate_matvec arm-dispatch), NOT a new bug — check the recipe first.

## Out of scope here

- S8/P5 (op-layer unification) — blocked on reflex's T4 laya move, which waits on that repo's harness sibling WIP; do not start it from this issue.
- The ANE lane (reflex Plan 002) — Apple-silicon only, M3-owned.
