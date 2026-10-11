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

The repro binary now carries the device-lost callback (landed 2026-10-11,
see Tasks): at the loss moment stderr prints
`[issue 039] DEVICE LOST: reason=... message=...` — CAPTURE THAT LINE; it
names the real trigger (the hal error string) and is the reproduction's
primary evidence, not the 16 failures that follow it. The callback also
records into the pool_poison state (`poisoned()` → `Some`).

⚠ 4090-box discipline: run EXCLUSIVE (no trainer/bench alongside) — the device-loss
class is exactly the one contention can mimic. First run must be a clean-box run or
the reproduction is unreliable.

## Tasks

- [x] **Pre-run substrate analysis (2026-10-11, CPU-side — SHARPENS the
      hypothesis space before the window run):**
      1. The file's original "suspected shape" (per-test context create/tear
         down across parallel threads) is REFUTED by the shipped substrate:
         `GpuContext::new()` is a process-global `OnceLock` since Issue 714
         (one device, one CubeCL server, one pool — the comment records the
         old 46-live-Vulkan-device leak it fixed) and Issue 719 serializes
         init under `gpu_init_lock` (shared with `CubeCLContext::new`).
         There is NO per-test adapter/device init or drop anywhere in the
         suite. The sharing instead EXPLAINS THE TAIL SHAPE: one loss on
         the shared device → every later test fails on buffers that became
         silent invalid error objects.
      2. The OOM-uncaptured path is RULED OUT by evidence already in the
         tree: `pool_poison::install_uncaptured_handler` (installed on both
         context paths since Issue 994) RECORDS and then RE-PANICS on every
         uncaptured wgpu error. The recorded 16 failures were assertion/
         buffer failures, not panics — so the loss moment did not surface
         through the uncaptured channel.
      3. That leaves the wgpu #10027 channel as the only SILENT loss path
         (consistent with the callback gap found and closed above): a
         driver-refused pipeline creation (CubeCL JIT/autotune kernel
         compile — e.g. over-budget per-thread private memory) maps to
         `DeviceError::Unexpected` → `lose()` → the `DeviceLost` error is
         dropped, no panic, no uncaptured fire. NOTE for interpretation:
         #10027's reporter says autotune compiles candidates at context
         INIT — but CubeCL JIT kernel compilation is also per-shape lazy,
         which fits a MID-SUITE loss (early tests pass; a later test
         compiling a new specialization trips the refusal).
      4. If the `--test-threads=1` A/B reads serial-GREEN, the concurrency
         axis shifts from "concurrent init" (impossible — see 1) to
         concurrent SUBMISSIONS/pool interleaving on the one shared device
         perturbing the driver's compile path. If serial ALSO fails, expect
         it to fail at the SAME test — a deterministic per-test kernel
         compile trigger; the callback's `message=` string then names the
         refused pipeline directly.

- [x] **Callback instrumentation LANDED (2026-10-11, CPU-side):**
      `pool_poison::install_device_lost_handler` — the shared wgpu-30
      `set_device_lost_callback` installer (loud `[issue 039] DEVICE LOST:`
      line + pool_poison record). The GAP was `CubeCLContext::new_uncached`
      (the edlm suite's path): it installed the uncaptured-error handler
      but NOT the device-lost callback — and the wgpu #10027 mechanism
      routes `lose()` through the callback channel ONLY, so device loss was
      100% invisible on that path (the uncaptured handler never fires for
      it). `GpuContext::new_async` (context.rs) already had an inline
      callback — refactored onto the shared helper, which ALSO adds
      poison-state recording there (a post-loss result now refuses at the
      poison reads). Clippy-clean at the edlm_gpu posture; pool_poison unit
      tests 2/2; the staged repro binary in `E:\tmp\infer038` rebuilt with
      the callback.
- [ ] Reproduce at HEAD on a free-4090 window; capture the failure shape (device
      lost vs adapter error vs panic in teardown) — record which of the 16 fail and
      whether the same 16 fail twice in a row. PRIMARY evidence: the
      `[issue 039] DEVICE LOST:` line + its `message=` (the driver's
      originating hal error string).
- [ ] The `--test-threads=1` A/B (serial vs parallel).
- [ ] Fix at the root: shared lazily-initialized adapter/context fixture (or the
      root the A/B points at), not per-test retry retries.
- [ ] Re-run the full `--features edlm_gpu --lib` suite green BOTH postures
      (wgpu + native CUDA).

## Refs

- Issue 1005 status line (the inline finding, 2026-09-09).
- `crates/riir-infer-gpu/src/edlm_cubecl/tests.rs` (the suite in question).
- Note: filed during the plan437 T0.3b AR window (the box's GPU is trainer-held);
  every run above is deferred to a free window — do NOT run them under trainer load.
- **Web-unblock research (2026-10-11, M3 idle-loop unit 9 — the class has a named
  upstream mechanism): gfx-rs/wgpu #10027 "Vulkan: a failed vkCreateComputePipelines
  silently loses the whole device" (opened 2026-08-07).** A VALID but
  driver-uncompilable compute kernel (per-thread private scratch over the driver
  budget — NVIDIA proprietary refuses >16 MiB/thread) fails `vkCreateComputePipelines`;
  `wgpu-hal` maps every non-OOM result to `DeviceError::Unexpected`
  (`map_pipeline_err` → `map_host_device_oom_err` → `get_unexpected_err`),
  `wgpu-core`'s `handle_hal_error` calls `lose()` on `Unexpected`, and the
  `DeviceLost` error is DROPPED silently (no error scope, no uncaptured handler) —
  `create_compute_pipeline` returns a pipeline object as if nothing happened. The
  reporter hit it THROUGH CUBECL: the over-budget kernel was one autotune candidate
  among 31, and the first visible symptom was a later `map_async` answering
  "Buffer with '...' label is invalid" — misleading errors exactly like our
  16-failure tail. **Verified in our tree:** the vendored
  `vendor/wgpu-hal-30.0.0/src/vulkan/mod.rs` carries the cited escalation shape
  (`get_unexpected_err` at L1584/1629/1639/1669) — we are on the exact
  affected version family (wgpu 30). **Implication for the run recipe:** the
  `--test-threads=1` A/B remains the concurrency discriminator, but if SERIAL
  also fails, hypothesis #2 is an over-budget eDLM kernel candidate (CubeCL
  autotune compiles candidates at context init — one driver-refused candidate
  kills the shared device for every later test). Add to the reproduction arm:
  install `wgpu`'s `set_device_lost_callback` (wgpu 30 exposes it; lives in the
  `wgpu`/`wgpu-core` crates, not the vendored hal) in the test fixture so the
  silent loss becomes a loud, attributable reason+message, and check whether any
  edlm_cubecl kernel candidate's scratch exceeds 16 MiB/thread on the 4090's
  proprietary driver. Upstream fix direction (from the issue): map failed
  pipeline creation to a surfaceable `PipelineError` instead of
  `DeviceError::Unexpected` — a vendor-patch candidate for `wgpu-hal-30.0.0` if
  the fixture-level callback proves insufficient. Related upstream: #9511
  (map_async callbacks dropped on device loss), #9029 (hal error escalated
  to device loss too eagerly).
- **Static scratch audit (2026-10-11, CPU-side, same session as the callback
  landing):** the hand-written eDLM kernels carry NO fixed-size per-thread
  private arrays — `edlm_cubecl/mod.rs` + `attention_cubecl/` grep for
  `let mut [T; N]` local arrays returns zero kernel-side hits (the only
  `vec!` matches are host-side test code); every `staging`/`scratch` in the
  lane is DEVICE-buffer staging (`PassStaging`, allocated through the
  compute client = workgroup/device memory, not per-thread private). So
  OUR code has no >16 MiB/thread private face. The remaining exposure is
  the vendored cubecl-0.11.0-pre.2 matmul autotune codegen (where the
  #10027 reporter's over-budget candidate lived) — not statically decidable
  without compiling the candidates; that half is owned at the window by the
  now-installed callback (the hal trigger string names the refused pipeline)
  and stays SECONDARY to the concurrency hypothesis the serial A/B tests
  first.
