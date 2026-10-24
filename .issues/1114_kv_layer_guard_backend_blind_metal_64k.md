# Issue 1114 — the Issue-994 single-buffer KV guard is backend-blind: it refuses a 64K configuration validated clean on Metal

**Status:** OPEN — T1+T2 LANDED 2026-10-24 (`CubeCLContext::is_metal_adapter` + `default_kv_layer_max_bytes(metal)`, Metal = block 65,600 = 268,697,600 B); T3 (raise to 128K) waits on the M5 128K rung, running now. Filed from riir-train Bench 657.
**Claimed:** zed-m3-issue1114-guard, 1792858865, expires 2026-10-26T16:21:05+00:00

Scope of the claim: T1 (the backend-keyed default table + the init-time backend capture) + T2 (the unit tests). T3 stays open — it is gated on riir-train Issue 452's next (128K) rung passing its anchor-ref gate, which is a bench-window act, not this one.

## What was seen

`crate::vram_budget::check_forward_budget` (called from `TernaryDeltanetGpuForward::new*`) refuses any per-layer attention KV buffer above `DEFAULT_KV_LAYER_MAX_BYTES` = 208 MiB (≈ 53,248 positions at kvd 1024) unless `RIIR_GPU_KV_LAYER_MAX_BYTES` is raised. The bound is the midpoint of riir-ai Bench 951's **4090 / `wgpu<spirv>`** bisect (49,216 clean · 57,408 device-lost · 65,472+ silent corruption).

riir-train Bench 657 (2026-10-24, M5 Ultra, 96 GB unified, `wgpu<msl>`, Bonsai-2 PQ2_0): with the bound raised to 1 GiB, the 64K rung at `block_size` 65,600 (0.25 GiB/layer, 8.2 GB total KV) ran with **anchor bit-identity 3/3 and anchor-ref vs an independent 32K artifact exact** (all diffs 0, cos 1.000000, labels and top-1 1.0). On the 4090 the same config corrupted exactly this anchor region (cos 0.18 / 0.42). The guard therefore refuses, on Metal, a configuration measured clean against an external reference.

## Proposed fix

Make the default per backend, each value a MEASURED point, never a guess:

- [x] T1 — key the default on the adapter backend (`wgpu-hal` already reports it; `Adapter::total_video_memory_bytes()` is per backend in the vendored fork). Vulkan/DX12 keep 208 MiB (Bench 951). Metal gets the largest point validated against an external reference: today 65,600 positions ≈ 256.25 MiB/layer (Bench 657). Unknown backends keep 208 MiB.
- [x] T2 — unit test (`kv_guard_default_tests::default_is_keyed_on_backend`; the env override path is unchanged — `parse_env_bytes` still wins before the default is consulted): the backend → default table, plus the env override still winning on every backend.
- [ ] T3 — when a 128K Metal rung passes its anchor-ref gate (riir-train Issue 452's next rung), raise the Metal point to it, with the bench cited at the constant.

## Non-goals

- Lifting the guard on Vulkan. The ≥52K defect there is real and unexplained (riir-ai HISTORY 2026-09-22: pool exhaustion refuted, the 256 MiB page cap refuted).
- Inferring a Metal bound from capacity alone. The 4090 failed well under its capacity, so the value has to be measured.
