# Issue 017 (2026-09-26) — `gpu_transpose` has no callers, and its docstring asserts an integration nobody performs

**Status:** OPEN — owner adjudication. RENUMBERED from 015 → 017 (2026-09-26): dual allocation with origin's audio-lane PoC 015 (pushed + cited by Research 001, it keeps the number; the 724-T2 rule sends the fewer-cited side); 016 was taken by the CubeCL G5 fail. `.issues/.highwater_local` now reads 017 (it also repairs the stale-015 counter the 016 allocation left behind). Not blocking: riir-train Issue 572 landed
a CubeCL-native transpose beside it (`transpose_cubecl.rs`, `3c4bd92`) rather
than waiting on this.
**Scope:** `crates/riir-infer-gpu/src/gpu_transpose.rs` + `kernels/transpose.wgsl`

---

## The measurement

```
$ git grep -n "gpu_transpose\|GpuTranspose" -- '*.rs' '*.toml'
crates/riir-infer-gpu/src/gpu_transpose.rs:38:pub struct GpuTranspose {
crates/riir-infer-gpu/src/gpu_transpose.rs:46:impl GpuTranspose {
crates/riir-infer-gpu/src/gpu_transpose.rs:203:    ) -> GpuTransposeDispatch {
crates/riir-infer-gpu/src/gpu_transpose.rs:244:        GpuTransposeDispatch {
crates/riir-infer-gpu/src/gpu_transpose.rs:259:        cached: &GpuTransposeDispatch,
crates/riir-infer-gpu/src/gpu_transpose.rs:280:/// [`GpuTranspose::create_dispatch`]; reused across all subsequent steps.
crates/riir-infer-gpu/src/gpu_transpose.rs:286:pub struct GpuTransposeDispatch {
crates/riir-infer-gpu/src/lib.rs:31:pub mod gpu_transpose;
```

Every hit outside the module is the `pub mod` line. The module is a complete,
carefully written tiled WGSL transpose with a `create_dispatch` /
`dispatch_cached` split built for exactly the case where the shape never
changes — and nothing in this workspace calls it.

## Why that matters more than an ordinary dead module

Its own header states the motivation and an integration property:

> the CPU transpose bottleneck for the LM head weight matrix (640 MB at 0.40B
> scale) … was ~674 ms/step on M3 Max

> the target buffer is the same CubeCL-managed pooled memory that
> `WeightBufferSlot` caches.

The first is a real, measured problem. The second is a **claim no caller
demonstrates**: `GpuTranspose` takes raw `wgpu::Buffer`s, and every training and
inference path here moves `cubecl::server::Handle`s. Whether a `Handle` can be
resolved to the `wgpu::Buffer` this API wants — and whether writing through that
`Buffer` is visible to a subsequent CubeCL dispatch on the same client — is
undemonstrated. A docstring that asserts an integration is *stronger* than
silence: the next reader budgets "wire an existing kernel" and discovers the
bridge is the work.

riir-train Issue 572 hit precisely that class of problem one repo over (the
Gemma-2 backward's 9.74 GB of pre-transposed weight handles) and, rather than
verify the bridge under a deadline, landed `transpose_cubecl.rs` — a CubeCL
`#[cube]` kernel with the same tiling, reachable from a `Handle` with no bridge
at all. So the workspace now has **two** tiled transposes in this crate, one of
which is called by nothing.

## What is NOT claimed

- That `gpu_transpose` is wrong. It is unexercised, which is a different and
  weaker statement — no test, gate or lane runs it, so its correctness is
  *unknown*, not *suspect*.
- That the duplication is bad by itself. A raw-`wgpu` transpose is the right
  tool for a consumer that holds `wgpu::Buffer`s; the question is whether such a
  consumer exists or is planned.

## Options (owner call)

1. **Wire it** — find or write the `Handle` ↔ `Buffer` bridge, prove it with a
   test, and give the module a caller. Makes the docstring true.
2. **Delete it** — `git log -p` keeps it recoverable, and `transpose_cubecl.rs`
   covers the use it was written for.
3. **Keep and re-document** — mark it explicitly as the raw-`wgpu` path for
   consumers outside the CubeCL runtime, and strike the pooled-memory sentence
   until something demonstrates it.

`BOUNDARY.md` names "the GPU transpose kernel (+ its WGSL)" as owned here, so
whichever option is taken, the ownership line stays and may want to say which
transpose it means.

Session: riir-train-plan410-572, 1758873600
