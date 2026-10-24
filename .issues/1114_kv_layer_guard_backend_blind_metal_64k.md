# Issue 1114 — the Issue-994 single-buffer KV guard is backend-blind: it refuses a 64K configuration validated clean on Metal

**Status:** OPEN — T1+T2 LANDED 2026-10-24 (`CubeCLContext::is_metal_adapter` + `default_kv_layer_max_bytes(metal)`, Metal = block 65,600 = 268,697,600 B); T3 (raise to 128K) waits on the M5 128K rung, running now. Filed from riir-train Bench 657.
**Lease closed:** zed-m3-issue1114-guard, claimed 2026-10-24T16:21:05+00:00, closed same day — the implementation raced (`cd748a6` by katgpt-rs-a0 pushed 10 min after the claim, the Issue-825 two-sessions-one-issue shape, push = arbiter); this session completed T2's letter (the env-precedence test) on top of the landed shape and released the lane. T3's owner is the M5 bench lane.

Scope of the claim: T1 (the backend-keyed default table + the init-time backend capture) + T2 (the unit tests). T3 stays open — it is gated on riir-train Issue 452's next (128K) rung passing its anchor-ref gate, which is a bench-window act, not this one.

## What was seen

`crate::vram_budget::check_forward_budget` (called from `TernaryDeltanetGpuForward::new*`) refuses any per-layer attention KV buffer above `DEFAULT_KV_LAYER_MAX_BYTES` = 208 MiB (≈ 53,248 positions at kvd 1024) unless `RIIR_GPU_KV_LAYER_MAX_BYTES` is raised. The bound is the midpoint of riir-ai Bench 951's **4090 / `wgpu<spirv>`** bisect (49,216 clean · 57,408 device-lost · 65,472+ silent corruption).

riir-train Bench 657 (2026-10-24, M5 Ultra, 96 GB unified, `wgpu<msl>`, Bonsai-2 PQ2_0): with the bound raised to 1 GiB, the 64K rung at `block_size` 65,600 (0.25 GiB/layer, 8.2 GB total KV) ran with **anchor bit-identity 3/3 and anchor-ref vs an independent 32K artifact exact** (all diffs 0, cos 1.000000, labels and top-1 1.0). On the 4090 the same config corrupted exactly this anchor region (cos 0.18 / 0.42). The guard therefore refuses, on Metal, a configuration measured clean against an external reference.

## Proposed fix

Make the default per backend, each value a MEASURED point, never a guess:

- [x] T1 — key the default on the adapter backend (`wgpu-hal` already reports it; `Adapter::total_video_memory_bytes()` is per backend in the vendored fork). Vulkan/DX12 keep 208 MiB (Bench 951). Metal gets the largest point validated against an external reference: today 65,600 positions ≈ 256.25 MiB/layer (Bench 657). Unknown backends keep 208 MiB.
- [x] T2 — unit test (`kv_guard_default_tests::default_is_keyed_on_backend`; the env override path is unchanged — `parse_env_bytes` still wins before the default is consulted): the backend → default table, plus the env override still winning on every backend.
- [x] T2-letter completion (zed-m3-issue1114-guard, on top of `cd748a6`): T2's letter names the env-override precedence as a TEST, and the landed commit asserted the code path in prose only. `resolve_kv_layer_max_bytes(env, metal)` (const fn, the landed shape's own vocabulary) pins the precedence device-free: env wins verbatim in BOTH directions (tighten as well as raise) on both backend classes; unset falls through to the table. Same commit heals the module's one `assertions_on_constants` residue (the sibling's `METAL > DEFAULT` assert → a compile-time `const {}` pin).

### The collision record (the Issue-825 class, 2026-10-24)

Two sessions implemented this issue concurrently: `katgpt-rs-a0` (`cd748a6`) and `zed-m3-issue1114-guard` (this). The sibling pushed first; its shape stands — bool-keyed `const fn` + both adapter probes defined posture-locally in `cubecl_runtime.rs`, which is the leaner design (no cross-module vocabulary; every non-Metal class takes the same measured default, so a bool is information-sufficient until a second measured point exists — the enum widens THEN, not before). The loser's independent findings, cross-run before withdrawal: (1) the platform-gated check (`check_platform_gated_modules.sh`, aarch64-linux) is what exposed the loser's OWN mid-flight defect — a `GpuBackend` enum living in `vram_budget` left `cubecl_runtime` (feature alone, no `ternary_gemv`) referencing a nonexistent module on the linux target; the landed shape avoids the class by construction, and the lesson stands: **guard-adjacent vocabulary belongs beside the capture, not behind a second module gate**. (2) The landed T2 lacked the env-precedence arm — completed above. Both verified green post-reconciliation: clippy 0 (default + `cubecl_runtime,ternary_gemv`, all-targets), `kv_guard_default_tests` 2/2, rustfmt clean, platform-gated check PASS (11 not-macos modules proven compiled).
- [ ] T3 — when a 128K Metal rung passes its anchor-ref gate (riir-train Issue 452's next rung), raise the Metal point to it, with the bench cited at the constant.

## Non-goals

- Lifting the guard on Vulkan. The ≥52K defect there is real and unexplained (riir-ai HISTORY 2026-09-22: pool exhaustion refuted, the 256 MiB page cap refuted).
- Inferring a Metal bound from capacity alone. The 4090 failed well under its capacity, so the value has to be measured.
