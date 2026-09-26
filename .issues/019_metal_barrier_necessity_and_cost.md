# Issue 019 — is the wgpu-hal Metal barrier (`c0dfa06`) needed, and what does it cost?

**Status:** OPEN — filed 2026-09-26 (session `katgpt-rs-5e`), from the Issue 018
close-out. Blocks nothing; plan 611 S5's A/B should run at BOTH settings of
this question.

## The disagreement (both sides measured or cited, neither overwritten)

Issue 018 was closed by the 018 record (`270740c`) as **two** defect
classes: (A) the in-kernel softmax smem write-after-read, fixed at
`2a34bd3`, and (B) missing inter-dispatch Metal memory barriers, fixed at
`c0dfa06` by emitting `memoryBarrierWithScope` from `transition_buffers` /
`transition_textures` and from `set_bind_group` whenever a compute pass is
open.

The evidence says **A alone was sufficient** for the observed wobble:

- **Isolated A/B** (clean worktree at HEAD `ae7f6b4` with ONLY the `2a34bd3`
  change; no barrier code present; M3, release, AC, load 12.8–16.6): the
  encoder repeat-loop probe had **43/120 fired passes without the fix vs
  0/120 with it** (interleaved, same tree), and G5 cubecl ran **10/10
  bit-identical** at the floor. The base fire rate is ~36%, so 0/120 bounds
  any residual class well below the observed wobble (at the pre-fix rate,
  120 clean passes would happen by chance about 1 time in 10²³).
- **B's localizing evidence is confounded by A.** The deep probe's
  "`L11.scores` diverged with q/k/v clean" downloads `sc.attn.scores` AFTER
  the in-place `softmax_rows`, which is exactly A's corruption site. The
  "fires went 8/8 → 0/18" measurement was taken in a working tree that
  already carried the A fix (in that checkout from 16:07; `c0dfa06` landed
  16:30).
- **B's premise is contested:** "Metal does not order memory across
  dispatches WITHIN one compute pass" holds for `MTLDispatchTypeConcurrent`
  encoders. The vendored fork builds its compute encoders with
  `computeCommandEncoder()` / `computeCommandEncoderWithDescriptor` and
  **never sets a dispatch type**, so they are `MTLDispatchTypeSerial`, where
  each dispatch completes before the next begins. Under that premise the
  emitted barriers are redundant, and the "empty transition bodies" are the
  intended upstream behavior, not a defect.

What is NOT claimed: that B can never fire. A write→read ordering hazard
could exist on an encoder path this analysis did not read, e.g. a
concurrent encoder somewhere, or a different Metal driver.

## Why it matters

`c0dfa06` emits a barrier on every `set_bind_group` inside an open compute
pass, for EVERY wgpu consumer of the fork. That includes the engine decode
lane at ~835 dispatches/token. If the barriers are redundant, that is pure
cost on the hottest wgpu path in the repo.

## Tasks

- [ ] T1 — necessity: G5 cubecl ×10 + the probe repeat loop ×120 at HEAD
      with `c0dfa06` reverted (a worktree; `2a34bd3` kept). 0 fires /
      10/10 bit-identical ⇒ B is not needed for the laya lane.
- [ ] T2 — cost: an interleaved A/B of the engine decode lane (tok/s) and
      the laya cubecl forward (p50), barrier ON vs OFF, box state quoted.
- [ ] T3 — verdict: keep behind a feature/env gate if it costs, or revert if
      T1 is clean and T2 shows cost. Record either way (a documented
      premise, e.g. "serial dispatch type ⇒ no intra-pass barrier", is the
      durable artifact).
