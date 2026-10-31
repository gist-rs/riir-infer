# riir-infer Issue 1162 — riir-infer-gpu: the ternary_gemv-ALONE combo does not compile (pre-existing cfg mismatch, no live consumer)

**Status:** open — found 2026-11-01 during the issue-1161 repair validation (`38df722`); proven PRE-EXISTING at HEAD e2fac9b via a detached worktree run (zero source delta).

## The finding

`cargo clippy -p riir-infer-gpu --no-default-features --features ternary_gemv --lib` fails with 4 errors:

```
error[E0425]: cannot find type `PrefillSite` in this scope   ×2 (ternary_deltanet_gpu_forward.rs:8236/8237 — prefill_project_pair params)
error[E0599]: no method named `prefill_project` found        ×2
```

Root cause shape: `PrefillSite` (ternary_deltanet_gpu_forward.rs:1796) and the `prefill_project*`
method family are gated `#[cfg(feature = "ternary_gemm_batched")]`, but callers/params in the same
module (the `ternary_deltanet_gpu_forward` module itself is gated
`all(cubecl_runtime, ternary_gemv)` — lib.rs:696) reference them without that gate. Every REAL
posture carries the superset (`production`, `dq_phase_bench`, the engine stacks all include
`ternary_gemm_batched`), so this repo's own gates are green and no consumer in the workspace
composes the lone `ternary_gemv` set — a latent posture hole of exactly the issue-1161 class
("any future consumer pulling riir-infer-gpu with a subset feature set will re-meet it").

## Repair (for the owning lane)

Same three options as 1161, adjudicated the same way: the mismatch is INSIDE
ternary_deltanet_gpu_forward.rs, so either gate the referencing sites on
`ternary_gemm_batched` (the cfg their targets already carry), or add
`ternary_gemm_batched` to a sub-gate the referencing impl block already sits behind —
a rustc-guided walk of the 4 sites + their enclosing items; no consumer-edge change
applies (no consumer composes this combo).

## Non-goals

Not fixed by the 1161 repair commit (disjoint files, disjoint cause); recorded here so the
next subset-posture sweep finds the adjudication instead of re-deriving it.
