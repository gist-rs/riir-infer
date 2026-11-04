# Issue 1175 — audit every CubeCL/wgpu launcher for zero-length buffer bindings (the alias-trample hazard)

**Status:** OPEN — filed from bench 944's instrument law (a
zero-length binding is clamped to the WHOLE buffer by the wgpu/Metal
binding layer).
**Claimed:** zed-ablate@amo, 1793804639, expires 2026-11-06T15:04:17Z

## The hazard

`BufferArg::from_raw_parts(handle, 0)` does NOT produce a kernel-side
slice with `len() == 0`: the binding layer clamps zero-length views to
the whole underlying buffer. Any kernel that branches on a bound
slice's `.len()` therefore sees the FULL buffer length — and if that
slot aliases a live output buffer, the branch fires and tramples the
output through the alias.

**Measured** (issue 1173, caught by the G1's twin-less arm before any
commit): the fused-kernel launcher's 0-length f16 twin slot read as
full-length and the twin write fired through the f32 output alias.
The fix shipped in `norms_fused_batched_cubecl.rs` (`5bb84f1`): bind a
one-element γ view as the unused slot and gate the branch by an
explicit `params[3]` flag — NEVER the slot length.

## Known-live zero-length sites

- `norms_cubecl.rs` `RmsNormQkFusedCubeCL::launch` — passes
  `BufferArg::from_raw_parts(k_handle.clone(), 0)` in the q-loop (and
  the symmetric q-slot in the k-loop). **Safe today by accident**: the
  `rmsnorm_qk_fused_f32` kernel never reads `.len()` on q/k — it
  selects the buffer by `wg < n_q`. Any future edit that adds a
  length check to that kernel silently opens the trample.

## The ask

1. Grep every `from_raw_parts(` call site in `crates/riir-infer-gpu/`
   (and the sibling copies that re-export this crate) for literal
   `0`-length bindings; for each, verify the kernel never length-reads
   that slot. Record the verdict per site in this issue.
2. Replace the risky pattern with the one-element-view + flag shape
   where the slot is ever length-inspected; add a crate-level doc row
   to `norms_cubecl.rs`'s module header (the law + the
   `RmsNormQkFusedCubeCL` caveat) so the next kernel author meets it.
3. Optional hardening: a debug-only launcher helper
   (`buffer_arg_opt(client, handle, opt_len, dummy)`) that encapsulates
   the pattern, so new launchers can't re-introduce the 0-length bind.

## Provenance

- Bench 944 (`.benchmarks/944_issue1173_fused_norm_gating_ab.md`),
  the G1 twin-less arms
  (`tests/qwen38_fused_norm_gating_g1.rs`), commit `5bb84f1`.
- Claude verdict 2026-11-04 (round 1): "file an issue to audit every
  launcher for zero-length binds... a doc note isn't enough."
