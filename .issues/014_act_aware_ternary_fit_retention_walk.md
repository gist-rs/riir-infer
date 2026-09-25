# Issue 014 — activation-aware ternary scale fit on a real checkpoint: the per-family conditional retention walk (the model-bound G1 of katgpt-rs Issue 886 P1)

**Status:** OPEN — filed 2026-09-25 from katgpt-rs Issue 886 (the substrate landed there at `0fb2254d9`, katgpt-rs Bench 896; this is its model-bound quality gate). No task started.

## What exists (katgpt-rs `0fb2254d9`, both opt-in)

- **P0 collector** — katgpt-core feature `act_channel_moments`: `ActChannelMoments::new(&widths)` + `observe(layer, x)` / `observe_batch(layer, xs)` accumulate per-input-channel `{Σ|x|, Σx²}` (f64) per linear layer, alloc-free (0.28–0.32 ns/element on the M3). `freeze()` → `ActChannelDiagonal { mean_abs(l), mean_sq(l) }`, BLAKE3-committed over a canonical LE image (`to_bytes` / `from_bytes` verify). Tap the **inputs** of each linear layer (normed hidden → q/k/v/gate/up; attention output → o; SwiGLU product → down). That is the same tapped forward as the Issue-883 V/K pass (`vk_calibration`), at different tap points.
- **P1 fit** — katgpt-types feature `act_aware_fit` (katgpt-core forwards it): `TernaryGroupWeights::quantize_from_f32_act_aware(w, rows, cols, diag: &[f32], fit)`, where `fit` is `WeightedMeanAbs` (closed form; a uniform diagonal gives a payload bit-identical to `quantize_from_f32`) or `WeightedSearch` (21-point grid + weighted LS refit through the real carry loop). The payload is kernel-identical: only the f16 group-scale value moves, so every shipped ternary matvec consumes it as-is.
- **Synthetic G1 (Bench 896)** — measured on PTQ-of-dense Gaussian/Laplace fixtures with planted heavy channels, held-out `E‖(W−Ŵ)x‖²`:
  - The activation-blind search alone gains −20…−27%.
  - The diagonal adds **−54%** on top of that with planted 1%×20 channels, and −14% on a log-normal spread (ternary).
  - A bench-local INT4 reference gains −60% / −26.5%.
  - This verifies the **mechanism**. It is not a deployed-surface gain. Two reasons: the fixture is dense→ternary PTQ (baseline relative error 0.44), and its channels are independent. Neither holds on the born-ternary, Hadamard-rotated Bonsai lane.

## What this issue owns (the gate katgpt-rs cannot run)

- [ ] **T1 — collect the diagonal on a real checkpoint.** Prefer the Ternary-Bonsai lane (`Ternary-Bonsai-2-27B-PQ2_0.gguf`, the DeltaNet ternary forward in `src/deltanet/ternary_forward.rs`). Forward katgpt-core `act_channel_moments` (a `act_diagonal_calibration` feature here, the `vk_calibration` shape).
  - Tap every ternary projection's input. With `bonsai2_hadamard` on, tap the **post-rotation** input, because that is what the ternary matvec sees.
  - Use a small calibration corpus: 16–128 sequences, which is AWQ's sample-efficiency claim. Commit the `ActChannelDiagonal` digest beside the run.
  - Report the diagonal's flatness per layer, e.g. the max/median of `E[x²]` and the top-1% channel share. If rotation has flattened it to near-uniform, T2–T3 are predicted to be null, and saying so up front is part of the result.
- [ ] **T2 — re-author the ternary weights with `act_aware_fit`.** The Bonsai tensors are born-ternary: there are no f32 parents. There are two arms, and neither may be dropped:
  - **(a) Bonsai scale refit.** Dequantize the shipped `Q2_0_g128` tensor to f32, then requantize under {mean-abs (`quantize_from_f32`), `WeightedMeanAbs`, `WeightedSearch`} with the T1 diagonal.
    - ⚠ Mean-abs requantization of an already-ternary group is not the identity. Its mean-abs is `s·nnz/128`, not `s`. So the **as-shipped tensor** is the reference arm, and the mean-abs requant is the G3-class control.
  - **(b) Dense-parent PTQ lane.** Use gemma-2-2b f16 (the house fixture) ternarized by `quantize_from_f32` vs `act_aware_fit`. This is the regime where Bench 896's synthetic numbers are predicted to transfer, if they transfer anywhere.
- [ ] **T3 — G1 by per-family conditional retention walk.** Use the riir-ai Bench 948 pattern: per-family items, argmax flips, top-k retention and min-margin against the reference arm. **Aggregate PPL alone is disqualified** by the lossy-surface law, because aggregates can stay flat while families flip. Arms:
  - the as-shipped / f16 reference;
  - mean-abs (the activation-blind baseline);
  - `WeightedMeanAbs[E x²]`;
  - `WeightedSearch[E x²]`;
  - `WeightedSearch[uniform]`, the blind-search control that separates "search helps" from "the diagonal helps";
  - the **ZeroQAT class**: riir-train `ZeroQatCalibrator` (`riir-train-engine` `zero_qat.rs`, Plan 255 Ph4, feature `zero_qat_calibrate`), which does finite-difference GD on an injected loss at the same group-scale insertion point. Run it at its default knobs, where the run budget allows, as the incumbent comparator. Report its cost (100 steps × 2 loss evaluations per scale) beside the closed-form fit's (one moment pass plus ≤ 23 carry passes per group).

  Record the delta **either sign**. The honest prior (Research 588 §2.6) is small-or-negative at ternary with rotations, and a clean negative is a valid close of Issue 886 P1.
- [ ] **T4 — record.** Write a riir-infer bench doc (next number from `.benchmarks/.highwater`) with box state: free RAM, commit-vs-limit, concurrent jobs, power source and mode. Then close katgpt-rs Issue 886 P1 with the verdict and the hash.

## Traps

1. **Tap after the rotation.** A diagonal collected pre-Hadamard describes channels the ternary matvec never sees.
2. **The born-ternary requantization control is not the shipped tensor.** Compare against both. The first is the G3-class control (what the fit changes); the second is the deployed reference.
3. **`E[x²]` vs `mean|x|`.** Bench 896 found `E[x²]` (imatrix) better in every non-control row. Re-check on real activations rather than inheriting it.
4. **Measure with `--release`**, a paired interleave for any timing, and box state recorded with every figure.
