# i1149 — R2 prework-fused CUDA routing proof (the Issue-1149 N/A cell)

**Status:** CLOSED-NA 2026-10-28 — riir-infer `69e7d8e` (the cell) + this record; run on the 4090
(Shikuwa box, RTX 4090 24 GB, Windows, idle: 5% util / 412 MiB before launch, no other compute
consumer — the sibling M0 run had drained). One frozen read, one box; the cell is re-runnable
(`--ignored`, the command in the test doc).

## Question

`deltanet_prework_fused` was promoted DEFAULT-ON **Metal/wgpu-scoped** 2026-10-28 (riir-ai Issue
1004 / Bench 024 addendum; delegated owner verdict). The runtime env default is
`cfg!(target_os = "macos")` — the `cuda_backend`/CudaRuntime posture kept the incumbent pair
shipping until its own 4090 cell answered the issue's FIRST question: **which dispatch path does
the CUDA prefill actually route?** The issue itself flagged the possible outcome: the CubeCL
prework arm may be dead weight on the cudarc posture entirely.

## Static routing evidence (commit `69e7d8e` review)

- The toggle (`prefill_prework_fused_effective`, `set_prefill_prework_fused`) is **fn-private to
  `ternary_deltanet_gpu_forward.rs`** and consumed nowhere else (workspace grep).
- `prefill_cuda_full::whole_prefill_inner` — the CudaRuntime prefill lane, the only entry every
  serving/verify prefill routes through on non-macOS — does its GDN prework through **four native
  cudarc dispatches**: `launch_conv1d` + `launch_carry_update` + `launch_beta_decay` +
  `launch_expand_l2`, then the recurrence through gdn_mma / gdn_chunked / L8 / rowpar. No CubeCL
  call anywhere in the section; the buffers are `CudaSlice`, a different universe from the
  CubeCL `Handle`s the fused kernel binds.

## Runtime engagement proof (the cell, this record)

Vehicle: `crates/riir-infer-gpu/tests/issue1149_prework_fused_cuda_routing.rs` (`#[ignore]`d,
4090-run, `RIIR_PREFILL_CUDA_GRAPHS=0`, `BONSAI_GGUF` absolute, the issue1115 contract). League
posture pins verbatim from the issue1115 vehicle (chunk 2048, L8 on, cmma recipe). NOT a perf
cell — the launch counters are the whole measurement.

- **ARM 1 (instrument liveness):** whole-prefill knob OFF → the same `prefill()` falls through to
  the CubeCL body (runtime `CubeCL runtime initialized: cuda`), prework setter-armed →
  **prework launches +48** (p=512, one chunk × 48 GDN layers) — the fused site routes, the
  counter counts, on THIS box, THIS runtime.
- **ARM 2 (the routing verdict):** whole-prefill knob ON (the 4090 serving lane), prework STILL
  armed:
  - p=2048 → **prework launches +0**
  - p=4096 (two chunks) → **prework launches +0**

Raw output (abridged):

```
[1149-routing] forward ready (104s) · n_layer=64 · vocab=248320
[1149-routing] ARM 1 (cubecl lane, prework armed): p=512 → prework launches +48 · fnv c57a1cc8740172f7
[1149-routing] ARM 2 (whole-prefill lane, prework armed): p=2048 (8.857s, 231.2 tok/s) → prework launches +0 · fnv c3e9d169021da6e0
[1149-routing] ARM 2 (whole-prefill lane, prework armed): p=4096 (14.454s, 283.4 tok/s) → prework launches +0 · fnv 63e4b3980013c027
[1149-routing] VERDICT: the CudaRuntime whole-prefill (the 4090 serving lane) routes ZERO CubeCL prework launches across p=2048 + p=4096 with the arm setter-armed, while the CubeCL fall-through lane advances the same counter — Issue 1149 N/A confirmed at runtime
```

## Verdict

**N/A — the cuda posture never routes the CubeCL prework site.** There is nothing to promote
onto the CudaRuntime default; issue 1149's verdict path 2 applies. The macOS scoping stands as
**permanent**: unset = ON on macOS / OFF elsewhere, and on cudarc the env is inert for the
serving lane entirely (it arms only the CubeCL fall-through lane, which the box's serving
posture does not route at league shapes). G1/G2/G3 cells as originally filed are subsumed:
no fused-vs-incumbent pair exists on the cudarc lane to compare.

One scope note for the record: `cuda_backend = ["cubecl_runtime", "cubecl/cuda"]`, so a
whole-prefill fall-through ON THE BOX would route the CubeCL body on the CUDA runtime — the env
remains meaningful there, which is exactly why it stays env-only rather than being deleted.

## Refs

riir-infer `.issues/1149` (filed with the promotion; closed by this cell) · riir-ai
`.issues/1004` + `.benchmarks/024_deltanet_prework_fused_ab.md` (the Metal promotion record) ·
riir-infer `c1cb377` + Issue 1115 (the per-backend scoping precedent + the vehicle contract).
