# Plan 004 — EXL3 T7c-1c: the full-pack bit-exact gate (Issue 001 §17.5)

**Status:** IN FLIGHT — 2026-09-25. T1 landed (gate test
`real_pack_v2_bit_exact_full` + the core `decode_w_rot_f32` extraction);
M3 Metal validation green (core 17 passed incl.
`parallel_matches_scalar_bit_identical`; GPU module 7 passed incl. the
synthetic v2 gate). T2 = the 4090 CUDA run (scheduled task), pending.

T7c-1c (§17.5): "full-pack bit-exact gate (whole pack, 5-class oracle
shape) for v2 before any GEMV work consumes it" — §17.3 gate 1 at
whole-pack scope: **v2-vs-v1 decode BIT-EXACT over EVERY layer of the real
pack, never a sample**, plus both arms against the CPU decode-only
reference (`Exl3Layer::decode_w_rot_f32`). NOT gated on the T7c-2 owner
decision: it protects the LANDED v2 kernel either way, and is the
prerequisite if the owner says proceed.

## Why whole-pack has real value (the coverage gap)

The synthetic fixtures cover K 2/3/4.5/5 × three codebooks; the T7c-1b/1d
benches sampled 5 classes (K 3/4/5). The pack spans **K 3.0–6.0** (§13.2)
— K6 and the half-integers the fixtures never exercised decode HERE for
the first time. A whole-pack gate is the only instrument that closes that
gap.

## Tasks

- [x] T1 — the gate + its CPU reference:
      - core: extract `Exl3Layer::decode_w_rot_f32()` (the decode-only
        stage, rayon over disjoint 16-row strips — bit-identical to
        `dequantize_f32`'s inline stage by per-element independence);
        `dequantize_f32_parallel` now consumes it. The scalar
        `dequantize_f32` stays UNTOUCHED as the independent oracle —
        `parallel_matches_scalar_bit_identical` pins helper == scalar.
      - gpu: `real_pack_v2_bit_exact_full` (`#[ignore]`, `EXL3_PACK_DIR`):
        per layer — v1 readback, v2 readback, bit-compare, CPU
        `decode_w_rot_f32`, bit-compare both; per-K×codebook coverage
        table; coverage floors (≥500 plans, ≥26 G weights, ≥3 classes —
        the T5-measured pack: 573 groups / ~26.0–26.5 G / mixed-K) so a
        loader regression REDS instead of printing a green zero; failure
        rows name the first divergent element + bit patterns. M3 Metal:
        core 17 passed, GPU module 7 passed.
- [ ] T2 — the 4090 CUDA run (pack `E:/git/riir-infer/.raw/packs/qwen38-27b-exl3-4bpw`,
      SCHEDULED TASK — the detached-SSH reaping lesson): box state
      recorded, whole-pack table + coverage in the issue §17.5 T7c-1c row.
- [ ] T3 — commit + push; issue status line updated; plan closed. The
      T7c-2 proceed/pivot/close decision stays the OWNER call.
