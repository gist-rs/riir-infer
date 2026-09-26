# Issue 019 — is the wgpu-hal Metal barrier (`c0dfa06`) needed, and what does it cost?

**Status:** RESOLVED 2026-09-26 — T3 verdict **REVERTED** (`4205c12`, session
`riir-infer-m3-t7c`): the c0dfa06 barrier emissions are out of the fork; the
premise record below is the durable artifact. T1's isolation was accepted as
controlling (and this session's own timing re-check conceded the confound:
the deep-probe triplet's clean runs 4/5 carried `2a34bd3` — log mtimes
16:07:52 / 16:09:22 vs the sibling's 16:07:41 edit). T2 was left unmeasured —
MOOT after the revert (nothing ships); post-revert G5 ran 22-27 s vs 24-31 s
with the barriers, directionally consistent with dropped driver calls.
Re-validation at HEAD (A only): smoke 9/9, G5 cubecl ×3 bit-identical at the
floor (3.092e-6 / 2.233e-6 / 5.187e-6), clippy `-D` clean. The laya probe
instruments (repeat loop + deep mode) stay — zero prod cost.

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

- [x] T1 — necessity: **DONE by the filing session** (43/120 → 0/120,
      G5 10/10 bit-identical at A-only) and independently conceded by the
      B author on the timing re-check above.
- [x] T2 — cost: **MOOT** (reverted before measurement); the directional
      reading is recorded in the status (G5 wall 22-27 s without vs
      24-31 s with).
- [x] T3 — verdict: **REVERTED** at `4205c12`. Durable artifact: the
      premise — the fork's `begin_compute_pass` never sets a dispatch
      type, Metal defaults to `MTLDispatchTypeSerial`, serial completion
      implies memory visibility ⇒ no intra-pass barrier is required on
      this path. If a concurrent-dispatch-type encoder ever lands (in a
      future cubecl/wgpu bump or a new lane), re-open from this record.
