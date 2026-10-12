# Bench 876 — gated delta-rule decode step: head-parallel + row-fused, bit-identical

**Status:** DONE 2026-10-13 — PROMOTED (no flag: bit-identical). Follow-up to [Bench 875](875_dense_ab_simd_promotion.md) finding #1.

## What

New `deltanet::forward::gated_deltanet_step_par_inplace`: the 48 V-heads in parallel over rayon, and within a head one pass per state row (decay row → `kv = dot(row,k)` → `δ = β(v−kv)` → `row += δ·k` FMA → `out = dot(row,q)·scale`) instead of four whole-head passes. Same per-element ops in the same order (`simd_outer_product_acc` with `m=1` is the same per-row FMA loop), so it is BIT-IDENTICAL. No `kv_mem`/`delta` scratch: δ is a per-row scalar, so those two dead fields left `DeltaNetLayerScratch`. Wired into the ternary decode (`ternary_forward.rs`), the f32 forward (`forward.rs`) and the profiling mirror (`profiling.rs`); the serial `gated_deltanet_step_inplace` stays as the reference (the allocating `gated_deltanet_step` wrapper and the GOAT test use it).

## Gates (M3 Max, AC, powermode 2)

| gate | result |
|---|---|
| G1 kernel | bit-identical outputs every token + bit-identical state after a 32-token recurrent stream, 48×128×128 (`tests/gdn_step_parallel_goat.rs`) |
| G1 model | real Bonsai-2-27B PQ2_0, `twt_perf_tg_pp` pp 128 / tg 64: generated tokens identical (65/65) to the Bench 875 runs |
| G2 kernel | interleaved median-of-ratios 31×200: 170.2 µs → 69.8 µs per layer (ratio 0.406, rounds 0.382–0.461) → projected 4.8 ms/token |
| G2 model | tg 1.84 / 1.90 tok/s vs 1.86 / 1.85 (Bench 875 build), load avg 5.5: **within run-to-run noise** — 4.8 ms is <1% of a ~540 ms token and sequential runs cannot resolve it. The kernel A/B is the evidence; the model run proves identity + no regression |
| G3 | riir-infer lib 338/338 + `bonsai2_rotation_load`, `issue929_bonsai2_loop_probe_gates`, `issue879_gdn_quant_certification`, `dense_ab_simd_goat` (release, `bonsai2_hadamard,issue929_bonsai2_probe`); clippy clean incl. `forward_profiling` |
| G4 | no allocation (rayon over pre-split chunks; δ on the stack) |

Rayon scaling is 2.4× on 16 cores: per head ≈ 1.5 µs of L1-resident work, so dispatch overhead and memory bandwidth cap it. Fusing the four passes alone buys little because each 64 KB head state already fits the P-core L1.

Re-run: `cargo test --release --features deltanet_ternary_inference --test gdn_step_parallel_goat -- --include-ignored --nocapture --test-threads=1`
