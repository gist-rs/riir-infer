# Bench 877 — Bonsai-2-27B CPU decode 4×: capped rayon split + int8-activation SDOT ternary matvec

**Status:** DONE 2026-10-13 — PROMOTED (`deltanet_ternary_inference` implies `ternary_q8_act`; f32 arm reachable at runtime via `set_ternary_q8_act(false)`). Upstream: katgpt-rs Issue 935 (`0c5ed0fe7`, `170f26ade`).

## Why

`sample` over the `twt_perf_tg_pp` decode put ~97% of busy samples in ONE
kernel — katgpt-types `neon_row_range_hoisted`, the Q2_0_g128 ternary matvec.
Every other decode item (Bench 875/876 follow-ups: conv copy, RoPE table,
rotation signs) is sub-millisecond noise next to it. The kernel was
compute-bound (~18 NEON ops / 8 weights) at ~25× the 2-bit memory floor, and
its rayon split (one chunk per worker) left the 4 E-cores finishing last.

## Changes

1. katgpt-rs: rows-per-task capped at 32 (`par_matvec_chunk`) — bit-identical.
2. katgpt-rs `ternary_group_q8_act`: activations → int8 per 128-group once per
   call, `tbl`+`tst` bit expansion, `sdot` (~7 ops / 16 weights). NEON
   bit-identical to its scalar integer reference.
3. riir-infer: ONE dispatch point `ternary_matvec_parallel` for every forward
   projection + LM head, and the forward mirrors (loop probe, profiler,
   act-taps, ternary-kvq) routed through it so their bit-identity pins hold.

## Kernel A/B (M3 Max, interleaved median-of-ratios, `examples/ternary_group_par_chunk_ab.rs`)

| shape (rows×cols) | shipped | chunk 32 | q8 sdot |
|---|---|---|---|
| 17408×5120 | 1.000 (1487 µs) | 0.712 | **0.202** (303 µs) |
| 5120×17408 | 1.000 (1553 µs) | 0.841 | **0.207** (313 µs) |
| 16384×5120 | 1.000 (1497 µs) | 0.684 | **0.207** (302 µs) |
| 5120×6144 | 1.000 (633 µs) | 0.669 | **0.250** (157 µs) |

## Real model — `Ternary-Bonsai-2-27B-PQ2_0.gguf`, `twt_perf_tg_pp` pp 32 / tg 64 (two arms per run)

| build | pp tok/s | tg tok/s | greedy tokens vs f32 baseline |
|---|---|---|---|
| baseline (before) | 2.08 / 2.14 | 1.96 / 2.03 | — |
| + chunk 32 | 2.78 / 2.80 | 2.46 / 2.47 | 65/65 identical |
| + q8 sdot (promoted) | 7.98 / 7.94 | **7.86 / 7.86** | 65/65 identical |

Load avg at launch: 4.9 / 5.7 / 4.3 (AC power).

## Quality gate — `tests/ternary_q8_act_goat.rs` (one load, two caches, arms alternated per position)

| window | top-1 agree | confident (f32 p ≥ 0.5) | hit f32 / q8 | mean NLL f32 / q8 | wall ratio q8/f32 | free decode 32 |
|---|---|---|---|---|---|---|
| 256 pos | 256/256 | 165/165 | 0.6367 / 0.6367 | 1.7013 / 1.6981 | 0.319 (430 → 139 ms) | identical |
| 512 pos | 510/512 | 348/348 | 0.6426 / 0.6426 | 1.6879 / 1.6685 | 0.326 (423 → 139 ms) | identical |

Bars: top-1 ≥ 0.95, confident ≥ 0.99, NLL Δ ≤ +0.02, hit drop ≤ 0.01, ratio ≤ 0.80.
Both runs at load avg ~15 (a sibling session) — the per-position pairing is the
defence. The NLL moved DOWN both times; the bar is one-sided, recorded as-is.

Re-run: `cargo test --release --features ternary_q8_act,bonsai2_hadamard --test ternary_q8_act_goat -- --ignored --nocapture --test-threads=1`

## Not covered

- x86_64 has no q8 port: the q8 entry point IS the f32 kernel there (no
  regression, no gain). AVX2 `maddubs` port tracked in katgpt-rs Issue 935.
- `transformer/ternary.rs` (the non-DeltaNet ternary lane) still calls the f32
  kernel directly — unvalidated for q8, left alone.
