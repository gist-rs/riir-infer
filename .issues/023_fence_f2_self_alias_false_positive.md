# Issue 023: fence_gate F2 false positive — self-crate aliases spelled `riir_*` trip the foreign-crate regex
> **Status:** PARTIALLY LANDED 2026-09-28 (idle watch session): the two CLEAN files renamed (`cuda_encoder_repeat_probe.rs`, `cuda_packed_repeat_probe.rs` — `riir_weights` → `lane_weights`, the capture driver's convention; both compile-checked at their required-features), fence red drops 3 → 1. The third (`cubecl_encoder_probe.rs`) STILL CARRIES the sibling session's uncommitted fmt edits — the rename there waits for those hunks to land (committing the file now would sweep a sibling's WIP; the staged-set discipline). Fence stays RED at 1 finding until then — honest, per this file's own rule. **Original deferral text below.**
> **Date:** 2026-09-27

## Symptom

`python3 scripts/fence_gate.py` fails with 3 undefended findings:

```text
⛔ UNDEFENDED crates/riir-infer-laya/tests/cubecl_encoder_probe.rs:45 — riir_weights::load(...)
⛔ UNDEFENDED crates/riir-infer-laya/tests/cuda_encoder_repeat_probe.rs:68 — let mut raw = riir_weights::load(...)
⛔ UNDEFENDED crates/riir-infer-laya/tests/cuda_packed_repeat_probe.rs:51 — let mut raw = riir_weights::load(...)
```

All three files carry the same shape:

```rust
use riir_infer_laya::laya::riir::weights as riir_weights;
...
riir_weights::load(&dir.join("model.safetensors"), name).expect("safetensors");
```

## Root cause

`fence_gate.py` F2 exists to catch FOREIGN workspace crates in code positions
(`riir_(?!infer_core\b|infer_gpu\b|infer_laya\b)[a-z0-9_]+`). The exempt
triple names the repo's own crates — but the regex reads TOKENS, not
resolutions, so a LOCAL ALIAS of an own crate (`as riir_weights`) matches
the foreign pattern at the use sites (`riir_weights::load`).

This is a scanner false positive, NOT a boundary leak: `riir_weights`
resolves to `riir_infer_laya`'s own module. The prose contract ("this
repo's OWN crates are exempt: self-references are the point of the repo")
is correct; the mechanism cannot see through the alias.

## Fix path (the fence must not weaken)

Rename the aliases to a spelling that cannot read as a foreign crate —
the capture driver (`examples/twt_laya_profile.rs`) already uses the
convention:

```rust
use riir_infer_laya::laya::riir::weights as lane_weights;
```

Three files, one import + 1-3 use sites each. Do NOT pin the findings in
`fence_expected.txt` (its header is the contract: "this repo has zero
sanctioned riir-* references" — a pin would convert a false positive into
a sanctioned `riir_*` spelling and invite the next real one). Do NOT
widen the exempt regex (an alias allowlist is a slippery slope — the
rename is the root-cause fix).

## Why deferred

`cubecl_encoder_probe.rs` (and neighbors) carried uncommitted sibling-session
fmt edits at filing time; editing the same lines risks entangling hunks
(the staged-set discipline). The rename lands on a clean tree in a
follow-up commit — the gate stays red until then, which is honest (a red
gate nobody can silently ignore beats a pin that launders the spelling).

## Verification

- Detached-worktree check: `git worktree add --detach /tmp/riir-infer-head HEAD`
  + run the gate → same 3 findings (pre-existing, not working-tree WIP).
- After the rename: gate must print `✓ fence_gate PASSED — 0 undefended`.
