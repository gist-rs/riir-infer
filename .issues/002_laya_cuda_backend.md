# Issue 002 — the CUDA backend for `riir-infer-laya` (the 4090 bench lane)

**Status:** OPEN — in flight (2026-09-24)

## Why

The live arena bench (https://reflex.gist.rs/bench/, `data/bench.json`) shows
the `4090-windows` host running the laya lane **an order of magnitude slower
than the M3 Metal row** — measured p50 (laya rust · english):

| suite | m3 · metal | 4090 · cpu | ratio |
|---|---|---|---|
| ag_news | 38 ms | 460 ms | 12× |
| emotion | 24 ms | 253 ms | 11× |
| banking77 | 87 ms | 1745 ms | 20× |
| typed_decisions | 479 ms | 8127 ms | 17× |

Root cause: `"laya_device": "cpu (LAYA_DEVICE or the no-backend default)"` —
the Metal backend is macOS-target-scoped (`laya-riir-metal`), so a non-macOS
host has exactly one compute posture (CPU `gemm`) and the 4090 GPU sits idle.

## What

A third backend for the lane: **CUDA over cudarc 0.19** (`driver` + `nvrtc`,
runtime PTX compile, `cuda-13030` + `fallback-dynamic-loading`, target-scoped
`not(macos)` — the `riir-infer-gpu` / `riir-gpu` proven declaration,
feature `laya-riir-cuda`), mirroring the Metal backend's architecture:

- ONE forward body unchanged (`.issues/005` law): the backend seam
  (`riir/backend.rs`) gains a `Cuda` impl with the same op semantics;
- device slots: permanent `(ptr, len)` weight cache + epoch-keyed
  `(ptr, len, gen)` chain cache, `begin_pass` invalidation,
  `download_into` host-read barrier with prefix matching — the Metal
  contract verbatim (the lazy-sync correctness argument transfers);
- ONE strided batched sgemm kernel (64×64×32 tile, 512 threads,
  fp32 FMA accumulate) covering `matmul` / `matmul_w` / `matmul_kt` /
  `matmul_kt_heads` / `matmul_heads` via the `(m, n, k, a_rs, a_cs, b_rs,
  b_cs, a_bs, b_bs, c_bs)` stride table — the W operand binds ROW-MAJOR
  `[n, k]` directly (the B-tile loader maps lanes along whichever stride is
  1), so NO device-side transpose cache is needed (unlike Metal's
  simdgroup layout constraint);
- the ~14 elementwise/row kernels ported 1:1 from `MSL_TAIL` semantics
  (CUDA `erff`/`expf`/`sqrtf`, division-form LN inverse, inv-multiply
  softmax normalize);
- attention v1 runs the TRAIT DEFAULT op sequence (split → rope → scale →
  batched kt-scores → mask broadcast → softmax → batched value mix →
  merge) entirely device-side — attention is <6% of forward FLOPs at the
  pinned geometries (projections dominate: ~217 GFLOP/forward at seq 317),
  so a fused flash kernel is a measured-follow-up rung, not the landing
  gate. `needs_window_mask` stays `true` (the default path consumes the
  mask tensor).

## Gates

1. **Op-level** (`tests/cuda_ops_smoke.rs`, `required-features =
   ["laya-riir-cuda"]`): every backend op vs the CPU `ops` free fns on
   known-answer fixtures — the `metal_ops_smoke` pattern including the
   `begin_pass`-per-arm epoch contract and the per-test GPU lock
   (`.issues/015` of the consumer repo, root-caused 2026-09-24).
2. **G5 parity** (consumer-side, `riir-reflex`): `LAYA_DEVICE=cuda cargo
   test --release --features laya-riir-cuda --test laya_riir_parity` —
   top-1 ≥ 99.9%, p-drift ≤ 1e-3 per checkpoint, against the SAME frozen
   captures. No number is published from the posture before this is green
   (the lane's parity law).
3. **Perf gate**: the 4090 laya p50 must beat the 4090 CPU row by ≥ 5× on
   every suite AND land at-or-below the m3 Metal row (banking77-class
   shapes were Metal 87 ms — target ≤ 87 ms; measured before publishing).

## Deliverables

- [ ] `crates/riir-infer-laya/Cargo.toml`: cudarc target-scoped dep +
      `laya-riir-cuda` feature + the smoke-test row
- [ ] `crates/riir-infer-laya/src/laya/riir/cuda.rs` (the backend)
- [ ] `agent.rs`: `DeviceKind::Cuda` + env/default/dispatch wiring
      (`LAYA_DEVICE=cuda` honored verbatim; non-macOS + feature compiled ⇒
      the default posture; absent feature/platform fails LOUD)
- [ ] `tests/cuda_ops_smoke.rs`
- [ ] BOUNDARY.md: the cudarc row for `riir-infer-laya`
- [ ] `riir-reflex`: `laya-riir-cuda` forwarding feature + G5 green at the
      CUDA posture on the 4090
- [ ] The harness 4090 re-run (bench row refresh) + HISTORY records
