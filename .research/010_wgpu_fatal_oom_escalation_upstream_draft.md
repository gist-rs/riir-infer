# Research 010: Upstream defect draft — wgpu-core escalates a recoverable OOM into a full device loss

> **Status:** DRAFT — for the OWNER to file at gfx-rs/wgpu (agents do not file
> upstream). Born-complete from riir-infer Issue 037's root-cause work
> (2026-10-09); every citation re-verified against the real crates.io source
> this session, not transcribed from notes.
> **Classification:** Public (the defect shape + minimal repro; our box
> specifics minimized)

---

## What we observed (the field report)

A long-running GPU test suite (243 tests, one process, one wgpu device) lost
the device mid-run when the BOX's VRAM was globally exhausted (another process
held ~7.6 GiB of 24.5 GiB; the suite's own working set ~16.2 GiB). The trigger
was a marginal, TRANSIENT staging allocation:

1. `StagingBuffer::new` → `device.raw().create_buffer(...)` fails with
   `hal::DeviceError::OutOfMemory` (the driver-level condition: global memory
   exhaustion, not a wgpu-limits rejection).
2. wgpu-core maps it through the FATAL handler → `device.lose("...")` → the
   whole device is invalidated.
3. Every subsequent operation in the process (16 downstream test failures,
   dozens of buffers) reads "invalid object / device is lost".

No test was wrong; the box was briefly out of VRAM. A fresh process recovers.
The observed callback text was exactly `Device is lost`
(`hal::DeviceError::Lost.to_string()`) and the poll thread panicked with
`Error in Device::poll: Parent device is lost`.

## The code shape (wgpu-core 30.0.1, crates.io source)

Two handlers exist side by side (`src/device/resource.rs`):

```rust
// :701 — FATAL for everything
pub fn handle_hal_error(&self, error: hal::DeviceError) -> DeviceError {
    match error {
        hal::DeviceError::OutOfMemory
        | hal::DeviceError::Lost
        | hal::DeviceError::Unexpected => {
            self.lose(&error.to_string());
        }
    }
    DeviceError::from_hal(error)
}

// :712 — OOM is DOWNGRADED to a per-call error
pub fn handle_hal_error_with_nonfatal_oom(&self, error: hal::DeviceError) -> DeviceError {
    match error {
        hal::DeviceError::OutOfMemory => DeviceError::from_hal(error),
        error => self.handle_hal_error(error),
    }
}
```

The non-fatal variant exists and is used on the user-facing allocation path
(`Device::create_buffer`), which is exactly right per the WebGPU spec:
`GPUOutOfMemoryError` is an ordinary per-operation error, NOT a device loss.

But three INTERNAL paths still use the fatal variant where the failure can be a
transient, recoverable OOM:

- `StagingBuffer::new` (`src/resource.rs:1268` `create_buffer` + `:1272`
  `map_buffer`): a TRANSIENT (`MemoryFlags::TRANSIENT`) internal staging
  allocation. Reached from ordinary `queue.write_buffer_with` /
  `Queue::create_staging_buffer` (`src/device/queue.rs:662`, `:699`,
  `:998`, `:1013`) and `CommandEncoder::write_buffer`-class paths.
- `Device::maintain` fence wait (`src/device/resource.rs:896`) and
  `get_fence_value` (`:908`): here the hal error is typically a REAL loss
  (`VK_ERROR_DEVICE_LOST`), so fatal handling is correct — this path is listed
  only for completeness; the fix candidate below does not touch it.

## Why this is a defect (the argument)

- WebGPU semantics treat `out-of-memory` as an operation-scoped error that
  leaves the device valid. `Device::create_buffer` already implements that
  correctly via `handle_hal_error_with_nonfatal_oom`. An internal staging
  buffer is the same driver-level condition arriving through an internal call
  site; escalating it to device loss is inconsistent with the crate's own
  semantics for the identical hal error.
- The consequence in practice is disproportionate: one failed transient
  staging allocation under memory pressure invalidates every resource in the
  process. Real-world processes (test suites, servers, browsers embedding
  wgpu via third-party stacks) sit near VRAM capacity routinely; the failure
  we measured fired at a 23.8/24.5 GiB global commit, inside a 2 MiB-class
  autotune staging allocation.
- The escalation also destroys the diagnostic: a clean
  `QueueWriteError::OutOfMemory` (retryable, reportable) becomes
  "Device is lost" everywhere downstream.

## Fix candidate

Route `StagingBuffer::new`'s two `map_err`s
(`src/resource.rs:1268`, `:1272`) through
`handle_hal_error_with_nonfatal_oom` — mirroring `Device::create_buffer`.
`StagingBuffer::new` already returns `Result<Self, DeviceError>`, and its
callers surface `QueueWriteError::OutOfMemory` / `EncoderError`-class
per-operation errors, so the plumbing needs no signature changes.

Scope note: we deliberately do NOT propose changing the fence-wait path — a
`Lost` from `vkWaitSemaphores` IS a device loss. The defect is narrowly the
OOM arm at internal allocation sites.

## Version / environment of the field report

- wgpu-core 30.0.1 (crates.io), wgpu-hal 30.0.0 (vendored additively by us,
  but the defect needs no vendored code to reproduce), Vulkan
  (`VK_ERROR_DEVICE_LOST` observed at the fence wait AFTER the escalation
  began; the initiating error was the staging OOM), NVIDIA Windows 11
  (24.5 GiB RTX 4090, another compute process co-resident at 7.6 GiB).
- Repro shape (minimal): fill VRAM near capacity from a second process; in
  wgpu, loop `queue.write_buffer_with` (or any staging-buffer path) with
  growing buffer sizes until the marginal staging allocation hits the
  driver's last block → observe full device invalidation instead of a
  per-call `OutOfMemory` error.
- Downstream record (our issue, with the measured VRAM trajectory):
  riir-infer `.issues/037_wgpu_full_suite_device_loss_cascade.md` (resolved
  test-side with an admission gate; the upstream fix would make the gate's
  SKIP-rarely-needed and the RUN-path self-healing).

## Filing checklist for the owner

- [ ] Repo: gfx-rs/wgpu — check wgpu master first (the handlers above are
      30.0.1; if master already routes staging through the non-fatal handler,
      file nothing and note the SHA here).
- [ ] Title sketch: "Device lost (fatal) on a transient staging-buffer OOM —
      internal staging paths escalate recoverable OOM to device loss".
- [ ] Attach the repro shape + this analysis; offer our measured VRAM
      trajectory data if useful.
- [ ] If filed: record the issue URL + SHA here (one line, then delete this
      checklist).
