# Issue 037 — wgpu full-suite device-loss cascade: root cause + VRAM admission gate

**Status:** RESOLVED — guard landed, full suite 242/0 green under trainer co-residency (2026-10-09)

## The symptom

Full `cargo test -p riir-infer-gpu --features cubecl_runtime --lib` (243 tests) lost the
wgpu device mid-run: **16 failures** (5 gemma2_cubecl incl. the issue-714 probe, 10
gemv_q4k, weight_buffer_cache), 226 passed. Reproduced twice (prior session at
`9ece055`, this session at `65deee9`), never caught before because prior sessions ran
filtered subsets.

## The bisect (no single killer test exists)

Serialized VRAM-logged run (`--test-threads=1`, 5 s `nvidia-smi` sampling, 4090 box,
plan437 trainer co-resident at 7.6 GiB):

- Tests 1–102 (everything before the heavy gemma2 tests): green in ~18 s.
- The loss fires **inside `test_cubecl_forward_gpu_matches_forward`** (first heavy
  full-model test), at the marginal `[gemv_autotune] (2048×2304)` benchmark
  allocation — the moment the working set tips over.
- VRAM trajectory: 9.0 → 13.4 GiB (instance build) → dip to 11.7 → **23.8 GiB** at the
  loss second. Test-side ≈ 16.2 GiB; the box's ~24.5 GiB was globally exhausted
  (trainer 7.6 + desktop + suite).
- **No killer test** — the failure is the box's global commit position, not one test's
  logic. The 16 failures are all downstream VICTIMS of one dead shared device.

## The mechanism chain (file:line evidence)

1. Marginal allocation under global VRAM exhaustion → driver-fatal condition on the
   Vulkan device (NVIDIA/Windows; the CUDA trainer was untouched — this is a
   Vulkan-device death, not a TDR).
2. `cubecl-wgpu`'s dedicated poll thread (`cubecl-wgpu-0.11.0-pre.2/src/compute/poll.rs`
   `WgpuPoll::new`) is always inside `device.poll(Wait { timeout: None })` → wgpu-core
   `Device::maintain` → `wait(fence)` → `vkWaitSemaphores` returns
   `VK_ERROR_DEVICE_LOST`.
3. **wgpu-core 30.0.1 escalates**: `Device::handle_hal_error`
   (`wgpu-core-30.0.1/src/device/resource.rs:701`) maps `OutOfMemory | Lost | Unexpected`
   ALL to `self.lose(...)` — and `poll`'s fence-wait path (`:885`, `:896`) uses the
   FATAL variant. Observed exactly: callback `message="Device is lost"`
   (`hal::DeviceError::Lost.to_string()`) + poll-thread panic
   `Error in Device::poll: Parent device is lost` (`WaitIdleError::Device` has no
   `to_poll_error` → `handle_error_fatal`). Note the NON-fatal
   `handle_hal_error_with_nonfatal_oom` exists and is used on the user-facing
   `Device::create_buffer` (`:1104`) — but `StagingBuffer::new` (actual 30.0.1
   lines: `src/resource.rs:1268` create + `:1272` map) and the poll paths are
   fatal. An OOM-class failure at one staging buffer must not be a device death;
   upstream-defect shaped — **draft for the owner to file upstream:
   `.research/010_wgpu_fatal_oom_escalation_upstream_draft.md`** (citations
   re-verified against the crates.io source 2026-10-09).
4. **Issue-676 OnceLock cascade**: one wgpu device per process — every later
   `GpuContext::new()` clones the dead context; every buffer is a silent invalid
   object; `read_one` fails at `map_async` ("Buffer with '' label is invalid"). 16
   victims, fresh process recovers (probe green when run alone).

## The fix (test-side admission gate, three layers)

- **Vendor wgpu-hal** (`vendor/wgpu-hal-30.0.0`, additive API like the Issue-994 patch):
  `Adapter::available_video_memory_bytes()` on Vulkan (`VK_EXT_memory_budget`
  budget−usage, largest DEVICE_LOCAL heap), DX12 (`QueryVideoMemoryInfo`), Metal
  (`recommendedMaxWorkingSetSize − currentAllocatedSize`); plus
  `dx12::global_adapter_video_memory_info()` (DXGI global budget−usage without a wgpu
  adapter). ⚠ Metal arm compiles on macOS only — not verified from this box.
- **`CubeCLContext::available_video_memory()`** — retains the wgpu adapter, probes LIVE.
- **`test_gpu_support::heavy_model_vram_guard`** — wired into the 9 full-model tests
  (4 cubecl + 5 gpu_decode_fusion GOAT). Loud SKIP/RUN lines both paths.
  `RIIR_GPU_FORCE_HEAVY_TESTS=1` overrides.

**Headroom source (the measured part):** NVIDIA/Windows `VK_EXT_memory_budget` reports
PROCESS-local usage and an unshrunk budget (measured 22.8 GiB "free" against 16.9 truly
free under the trainer) — WDDM `Budget` is a commit allowance, not free VRAM. The
Windows guard therefore prefers a bounded `nvidia-smi` NVML query (true global
occupancy), falls back to the in-process probes (honest on Linux/Metal), and FAILS OPEN
when nothing can read the box (never silently narrow coverage).

**Threshold 18 GiB** = measured ~16.2 GiB working set + ~1.8 GiB staging/pool slack;
the measured failure point (16.9 GiB free) clears with margin; an exclusive box
(~23 GiB free) still runs.

## Validation

- Full serialized suite WITH trainer resident: **242 passed / 0 failed / 1 ignored,
  16 s** (was 226/16 in 796 s), 4 SKIP lines, issue-714 probe green ("device alive").
- `RIIR_GPU_FORCE_HEAVY_TESTS=1` single heavy test: RUN, passed (105 s).
- clippy `-D warnings` at default / no-default / cubecl_runtime × all-targets: clean.
- `cargo check --workspace`: clean.
- NOT run here: the exclusive-box RUN path (no exclusive window available — the
  trainer is a standing task); the Metal vendor arm (macOS-only).

## Upstream notes

- wgpu-core fatal-OOM escalation (`handle_hal_error` on `StagingBuffer::new` + poll
  fence-wait) is the deep defect — a candidate upstream issue for gfx-rs/wgpu
  (distill lane: check their main first; wgpu 30.0.1 is current-ish).
- The legacy repro harness `../riir-ai/scripts/issue_714_device_lost_repro.sh` still
  names `riir-gpu` (pre-carve crate name); its posture (serialized + probe) is what
  this issue's method reproduces — leave as-is, it documents the history.

## Conclusion

The "which test kills the device" question was a category error — no single test does.
The suite needs ~16 GiB contiguous commit; the gate now checks the box can grant it
before constructing the model, and the loud SKIP keeps co-resident boxes green while
naming the override.
