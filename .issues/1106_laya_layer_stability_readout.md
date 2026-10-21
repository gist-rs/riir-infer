# Issue 1106: Laya per-layer stability readout (headless decision-depth primitive)

**Status:** OPEN — substrate half of riir-reflex `.research/011` + reflex Issue 094 (arXiv:2610.17495 distill); calibration-class readout, never a serving path.

**Claimed:** riir-refine-idle-4090, 1792611266, expires 2026-10-23T19:34Z

**Number note:** allocated 1106 and KEPT (pushed numbers are never reused), but it is **not collision-free**: riir-ai's issue counter sits at 1105 (verified locally and on `origin/develop`, 2026-10-21), so 1106 is the next number riir-ai itself will allocate. Skipping cannot outrun a climbing counter — the real protection is that every citation names its repo (`riir-infer Issue 1106`, enforced by the workspace `issue_citation_gate`). The literal reading of this repo's AGENTS.md numbering rule ("reserve 1005–1008, continue from 1006") would have said 1009 — that rule is stale and self-contradictory (riir-ai has long since consumed the whole range); a follow-up may repair the rule text, out of scope here.

## Context

arXiv:2610.17495 trains models to report "decision depth" — the earliest layer from which the final top prediction stays unchanged. The laya encoder lane ships **no LM head** (no `lm_head`/unembed in `crates/riir-infer-laya/src/`), so lens top-1 depth is not directly computable there. The head-free analog is residual-state stabilization: the earliest stage after which the last-position residual stops moving materially.

The hook exists: `encoder.rs::forward_capture` (`twt_profile`) — per-stage residual capture, "bit-identical to a plain forward by construction", cost class "calibration only, never a serving path" (the file's own words — keep both properties).

## Tasks

- [ ] Stability readout on the `twt_profile` capture seam: per-stage last-position residual delta (cosine + L2) → **stabilization depth** = earliest stage after which the delta stays under a pinned band for the remaining stages.
- [ ] Headful variant (only if a served checkpoint ships an LM head): per-layer lens top-1 stability = the paper's true decision depth. Tie rule: first-index per the `float_order` convention; a tie band is DISCLOSED (`tie: true`), never silently resolved — the paper's own tie-indistinguishability finding is the law here.
- [ ] G1: readout parity vs brute-force recompute on a fixture; the capture-vs-plain bit-identity pin stays green (the existing `forward_capture` law).
- [ ] G4: zero-alloc sink path (reuse the `CaptureSink` buffer convention).
- [ ] G3: feature-off byte-identity — the readout compiles away; no serving-path touch.
- [ ] Cross-checkpoint disclosure: publish which served checkpoints expose which variant (laya: headless stability; others per head availability). The readout definition is checkpoint-agnostic.
