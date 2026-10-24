# Issue 1115 — the GEMV occupancy wg rule is default-ON on the wgpu/Metal runtime only — the 4090 CudaRuntime posture is unmeasured (prefill-tail lm_head) — file the CUDA A/B that promotes it there

**Status:** CLOSED 2026-10-25 — the owed cell measured ([Bench 887](../.benchmarks/887_issue1115_wg_occupancy_cuda_ab.md)): G1 FNV-identical ×6+6 rounds (pp512 + pp4096, CudaRuntime, Bonsai-2 vocab 248320 → the 128-wg bucket), reach exact (+1/±0 across 12/12 rounds via the counter), G3 1.0013× / 1.0000× sub-noise → per this issue's own done-when the platform split is DELETED and `wg_occupancy_platform_default()` is `true` unconditionally; reviewer verdict AGREE (sub-agent `f4e6369c`, 2026-10-25; the request_verdict claude_code channel was auth-unavailable on the box — the independent sub-agent verified the counter semantics, the consumer census, and the harness first-hand). Env-vs-setter deviation, judged faithful by the reviewer: the harness drives `set_gemv_wg_occupancy(false)` vs `(true)` with BOTH envs required UNSET (the same two postures — OFF = the pre-promotion unset resolution — and the env route is unusable in-process: the OnceLock wins over the setter). Unit pin: `the_platform_default_is_on_everywhere` (5/5 green).
**Claimed:** riir-infer-4090-wgocc, epoch 1792863545, expires 2026-10-26T17:39:00Z

## The scoped posture (what landed)

`crates/riir-infer-gpu/src/gemv_ternary_cubecl.rs`:

- `USE_WG_OCCUPANCY` static now defaults TRUE; `wg_occupancy_env_semantics(Ok/Err)`:
  - `Err(NotPresent)` (unset) → `wg_occupancy_platform_default()` — **TRUE on the wgpu
    runtime** (`any(not(feature = "cuda_backend"), target_os = "macos")`, the same split
    `ActiveRuntime` uses), **FALSE on the `cuda_backend` CudaRuntime** (non-macOS).
  - `Ok(v)`: `"1"/"on"/"true"` (case-insensitive) → ON; ANY other set value (empty, "0",
    a typo) → OFF (the kill-switch class — env wins for demotion).
  - `Err(NotUnicode)` → OFF.
- Unit-pinned: `wg_occupancy_semantics_tests` (4 tests: unset/armed/kill-switch/non-Unicode)
  behind `ternary_gemv` (the module's own gate — a default-features lib test compiles it
  to NOTHING; run with `--features ternary_gemv,cubecl_runtime`).
- The 4096–5119 m-bucket is documented as EXTRAPOLATED (only m=5120/6144 measured).

## Why CUDA stayed opt-in — the reviewer's catch

The promotion proposal claimed "the 4090 scored decode lane does NOT consume this
CubeCL `launch()`" — true for decode (raw-cudarc `ternary_gemv_cuda_raw`), FALSE for
the prefill tail: `prefill_cuda_full.rs` dispatches the final lm_head through
`GemvTernaryCubeCL::launch::<ActiveRuntime>` (two sites, ~4298 + ~4574), and on a
`cuda_backend` non-macOS box `ActiveRuntime = CudaRuntime`. A global default flip
would have moved the 4090's prefill-tail lm_head from 256 to 128 threads UNMEASURED
(on Metal that cell was flat; on CUDA nothing is known). Correctness is safe either
way — dispatch-only, byte-identical by construction — but G3 (no-regression) has no
CUDA evidence.

## The owed cell (this issue's done-when)

One 4090 prefill A/B at `RIIR_GEMV_WG_OCCUPANCY=0` vs unset, same window, interleaved:
pp512 (the tail-dominated shape) + pp4096 (the league posture), folded league file.
- lm_head is ONE dispatch per prefill; the expected delta is sub-noise — the cell exists
  to prove the DEFAULT can flip there without a regression, not to win.
- G1 byte-identity both postures (dispatch-only law holds cross-runtime by the same
  construction; assert it, never assume it).
- If G3 clean → flip `wg_occupancy_platform_default()` to `true` unconditionally in the
  same shape as this promotion and delete the platform split. If the lm_head cell moves
  against → keep the split permanently and record the bucket in this file.

## References

- riir-infer Bench 996 (the calibration + e2e GOAT ×2, m5ultra, Metal).
- riir-ai Issue 1113 (the defense campaign this rode).
- Reviewer session `cd03cea7` (2026-10-24): the blast-radius correction + the env
  four-case pin + the extrapolation note — all landed in the same commit as this issue.
