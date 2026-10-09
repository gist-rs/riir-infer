# Issue 039 — the wgpu `--lib` suite loses the device mid-run (16 failures, pre-existing at HEAD)

**Status:** OPEN — filed 2026-10-10 (4090 box, trainer-held window). Extracted from
issue 1005's T8 status line, where the finding was recorded inline ("⚠ Pre-existing
found: the FULL wgpu `--lib` suite loses the device mid-run (16 failures, reproduced
at HEAD in a clean worktree — follow-up)") without its own file. This is that file.

## The finding

`cargo test -p riir-infer-gpu --features edlm_gpu --lib` under the wgpu (CubeCL
`wgpu<spirv>` / Vulkan) runtime **loses the GPU device partway through the suite**:
16 test failures follow, all consistent with the device being gone rather than the
assertions being wrong.

- **Reproduced at HEAD in a clean worktree** (issue 1005 T8 session, 2026-10-09,
  4090 box) — NOT introduced by the T8 graph/cmma work; it is pre-existing drift in
  the wgpu runtime posture or the suite's device lifetime management.
- The native-CUDA posture is UNAFFECTED as far as recorded: the graph capture lane
  (`cargo test -p riir-infer-gpu --features edlm_gpu --lib graph -- --ignored
  --test-threads=1`, native CUDA) ran green the same session.
- Suspected shape (to verify, not asserted): libtest runs tests on parallel threads;
  the edlm wgpu tests each create/tear down CubeCL contexts against the same
  adapter — concurrent adapter/instance init+drop across threads is a known wgpu
  device-loss class on Windows/Vulkan. `--test-threads=1` is the obvious first
  A/B: serial green + parallel red ⇒ the suite needs a shared-context fixture (one
  lazily-initialized adapter behind a mutex, like the CUDA lane's install-once
  seams) rather than per-test contexts.

## Unblocks

- The edlm_gpu wgpu posture's `--lib` gate cannot be trusted green until this is
  fixed — a 16-failure tail hides real regressions behind device-loss noise.
- Issue 1005 Phase 3+ wgpu-side GOAT lanes (any future interleaved-A/B on
  `wgpu<spirv>`) inherit the flake.

## Run recipe (when a free GPU window exists)

```sh
# reproduce (expect the 16-failure tail):
cargo test -p riir-infer-gpu --features edlm_gpu --lib
# the discriminator (serial ⇒ concurrency class):
cargo test -p riir-infer-gpu --features edlm_gpu --lib -- --test-threads=1
```

⚠ 4090-box discipline: run EXCLUSIVE (no trainer/bench alongside) — the device-loss
class is exactly the one contention can mimic. First run must be a clean-box run or
the reproduction is unreliable.

## Tasks

- [ ] Reproduce at HEAD on a free-4090 window; capture the failure shape (device
      lost vs adapter error vs panic in teardown) — record which of the 16 fail and
      whether the same 16 fail twice in a row.
- [ ] The `--test-threads=1` A/B (serial vs parallel).
- [ ] Fix at the root: shared lazily-initialized adapter/context fixture (or the
      root the A/B points at), not per-test retry retries.
- [ ] Re-run the full `--features edlm_gpu --lib` suite green BOTH postures
      (wgpu + native CUDA).

## Refs

- Issue 1005 status line (the inline finding, 2026-10-09).
- `crates/riir-infer-gpu/src/edlm_cubecl/tests.rs` (the suite in question).
- Note: filed during the plan437 T0.3b AR window (the box's GPU is trainer-held);
  every run above is deferred to a free window — do NOT run them under trainer load.
