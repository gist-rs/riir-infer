# Bench 875 — Bonsai-2 dense a/b gate projections: SIMD dot replaces the serial scalar reduction

**Status:** DONE 2026-10-13 — PROMOTED (unconditional; the scalar arm removed, kept only as the test reference in `tests/dense_ab_simd_goat.rs`).

## What

`GateProjWeights::matvec_into` Dense arm (`src/deltanet/ternary_weights.rs`) — Bonsai 2's BF16→f32 `ssm_alpha`/`ssm_beta`, 48×5120 each, run twice per GDN layer × 48 GDN layers per token (≈23.6M MACs/token). The old arm was a single-accumulator `acc += w*x` loop: no fast-math, so LLVM cannot reassociate it, and it executed as a latency-bound scalar chain. Now one `katgpt_core::simd::simd_dot_f32` per row. Found by a decode hot-path smell hunt (the CPU decode path is otherwise allocation-free; other findings below).

## Gates (M3 Max, AC power, powermode 2)

| gate | result |
|---|---|
| G1 kernel | per-row \|simd − scalar\| ≤ 1e-5·Σ\|w·x\| over 8 inputs × 48 rows (`g1_simd_dense_matches_the_scalar_arm_within_reduction_tolerance`) |
| G1 model | real `Ternary-Bonsai-2-27B-PQ2_0.gguf`, `twt_perf_tg_pp` pp 128 / tg 64, scalar build vs SIMD build: **65/65 generated tokens identical**, in-prompt hit 0.6220 both |
| G2 kernel | interleaved median-of-ratios 31×200: 176.2 µs → 14.5 µs per projection (ratio 0.0826, rounds 0.070–0.086) → projected 15.5 ms/token |
| G2 model | tg 1.77 / 1.83 → 1.86 / 1.85 tok/s (+3%, ≈555 → ≈539 ms/token, matching the projection); pp 1.89 / 1.97 → 2.03 / 2.00 tok/s. Sequential builds; the SIMD build ran at the HIGHER load avg (7.7 vs 5.5) |
| G3 | riir-infer lib 338/338 + `bonsai2_rotation_load` + `issue929_bonsai2_loop_probe_gates` 3/3 (incl. the K=1 bit-identity pin, which compares two paths through the same matvec) + `issue879_gdn_quant_certification`, release, `bonsai2_hadamard,issue929_bonsai2_probe` |
| G4 | no allocation added (per-row dot into the caller's buffer) |

Re-run: `cargo test --release --features deltanet_ternary_inference --test dense_ab_simd_goat -- --include-ignored --nocapture --test-threads=1`

## Remaining smell-hunt findings (not done here, ranked by est. per-token cost)

1. **DONE → [Bench 876](876_gdn_step_head_parallel.md) (bit-identical head-parallel, 2.4× kernel).** Was: `gated_deltanet_step_inplace` (`src/deltanet/forward.rs:243-301`): four passes over each 128×128 head state where one fused per-row pass suffices, and the 48-head loop is serial. Est. 3–6 ms/token. Not bit-identical once fused (reduction order) → needs its own flagged gate.
2. **DONE (riir-infer `39ceb07`)** — `expand_l2_normalized_heads_into`, one shared helper for the three call sites, pinned bit-identical by a lib unit test. Was: L2-norm after the 16→48 head expansion (`ternary_forward.rs:252-270`): 3× redundant; normalize per K-head, then expand. Bit-identical.
3. conv round-trip copy (`ternary_forward.rs:214-228`), causal_conv1d shift + scalar SiLU (`forward.rs:310-340`), RoPE sin/cos per attention layer instead of once per token (`rope.rs:216-222`), i8 rotation signs converted per element (`rotation.rs:139-166`). Each sub-millisecond.
