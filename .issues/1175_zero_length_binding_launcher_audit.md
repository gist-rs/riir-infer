# Issue 1175 — audit every CubeCL/wgpu launcher for zero-length buffer bindings (the alias-trample hazard)

**Status:** CLOSED (2026-11-04) — the audit ran clean; verdicts below.
**Lease closed:** zed-ablate@amo, epoch 1793804639, expires
2026-11-06T15:04:17Z. File removed per the noise-reduction rule; the
verdicts live in this removed file's git history + `HISTORY.md`.

## Audit verdict (2026-11-04, zed-ablate@amo)

**Population**: 1102 `BufferArg::from_raw_parts(` call sites in
`crates/` (the `std::slice::from_raw_parts` hits are unrelated). The
sibling re-export surfaces (`src/` root, riir-reflex, riir-neuron-db,
riir-refine's own launchers) carry none; ONE literal-zero bind found in
a sibling's own kernel crate — riir-ai's `riir-gpu` (below).

**Literal-zero sites found and fixed** (riir-infer commit `981e0c0`-family):

| site | verdict | action |
|---|---|---|
| `norms_cubecl.rs` q-loop k slot | safe by accident — `rmsnorm_qk_fused_f32` selects the buffer by `params[3]` (n_q) and NEVER reads either slot's `len()` | fixed: binds a ONE-ELEMENT q-γ view (in-bounds by construction, offset 0) |
| `norms_cubecl.rs` k-loop q slot | symmetric, same verdict | fixed: one-element k-γ view |

**Latent bug found by the audit and fixed**: the three fused twin
launchers (`norms_fused_batched_cubecl.rs`) bound their twin-less slot
as a one-element γ view but rode the TWIN layout offset (`r0·dim·2`
bytes) on it — past a γ buffer's end for any CHUNKED twin-less launch
(in-bounds only because the G1 shapes never chunked). Fixed via the
shared `optional_twin_binding` helper (offset 0 on the fallback; the
used arm unchanged); the three duplicated match blocks deduped into it
(DRY). **Validation**: fused G1 3/3 bitwise (twin-less arms included),
norms suite 15/15, full gpu lib suite **314 passed / 0 failed** on
Metal.

**Cross-repo finding (fixed in the sibling, cited per the law)**:
`riir-ai/crates/riir-gpu/src/lora_fused_cubecl.rs` `launch_fused_b`
bound `packed_b` with length 0 and the comment "size not needed for
storage buffers" — the clamp law says the opposite (the bound length IS
the kernel-visible slice). The kernel is index-based (safe by
accident); fixed to the buffer's honest full length — behavior-identical,
4/4 `lora_fused` tests green on Metal. **Sibling commit: riir-ai
`84239dbd2`** (cherry-picked onto origin/develop in a detached worktree
after a concurrent sibling push; the main clone ref-synced). The same
hazard class is already a mined rule face in riir-refine's kernel_opt
corpus (`entries_gpu_06`, the zero-length + over-length binding pair).

**Second-order check (kernels that DO length-read their bound slices)**:
~15 kernels bound themselves by `x.len()` (residual_add, the elementwise
family, rotation kernels, encoder lane). Their launchers all pass
explicit, loop-guarded NONZERO lengths — no zero-length-capable site
found. The module-header law in `norms_cubecl.rs` (issue-1175 section)
now pins the three rules — never a 0-length bind; one-element γ views
at offset 0 for unused slots; route by an explicit params flag, never
slot length — so the next kernel author meets them before the compiler
would.

The debug-only `buffer_arg_opt` helper ask is covered by
`optional_twin_binding` (real, not debug-only — the clamp law is
platform behavior, not a debug assertion) for the Option-slot shape;
the two-loop qk launcher inlines the one-element view with the law
documented in place.

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
