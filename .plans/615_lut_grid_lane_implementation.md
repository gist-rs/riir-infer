# Plan 615 — Issue 027 implementation: T0 asymmetric encoder + T1 Lloyd-Max grid solver + weight-space G1 eval

**Status:** T0/T1/T4/T5 LANDED (weight-space tables below — solved grid wins every family on both artifacts); T6 model-level gate + T7 kernel pricing open.

Issue: `riir-infer/.issues/027_lut_grid_optimization_lane.md`. Master: `.research/004_DQ_Disaggregated_Quantization.md` §2.3 + A.3 (arXiv:2609.26333).

## Substrate-first verdict (gate run 2026-10-01, before any code)

- **No Lloyd-Max / grid-solver / weight-histogram substrate exists** (vocabulary translation greps: `lloyd|kmeans|grid_opt|centroid|histogram` over riir-infer + katgpt-core — the only `centroid`/`grid` hits are belief-space clustering in katgpt-rs benches and the ANE channel-grid in `ane_prefill`, both unrelated domains). Lane builds new substrate in `src/quant/lut_grid.rs`.
- **Closest existing substrate consumed, not duplicated:**
  - `crates/riir-infer-gpu/src/dq_fakequant.rs::a2_quant_dequant_block` — the A2 ACTIVATION fakequant is already the T0 shape (`d = sign(argmax|a|)·f16(amax/2)`, lowest-index argmax tie-break, `q = clamp(round(a/d), −1, +2)`); its docstring says "027-T0 SHAPE; gates nothing in 027". The T0 WEIGHT encoder mirrors its scale rule bit-exactly so activation/weight lanes share one convention.
  - `src/twt/ternarize.rs::arm_source_quant` — the symmetric reference-encoder pattern (`d = amax`, round-clamp to ternary); T0's control arm.
  - `src/quant/q2_0.rs` — wire decode `(q−1)·d` (code 3 = +2d decodable) + ternary bridge rejection (`UnsupportedFourthState`).
  - `src/gguf_loader.rs::GgufFile` — mmap-backed tensor enumeration for the dense artifacts (`tensor_infos`, `tensor_slice`, `try_dequant_f16_to_f32`).
  - Histogram substrates in `vk_calibration.rs`/`act_retention_walk.rs` are activation-stat shaped (log-ratio/token-count) — not reusable for weight quantization MSE.

## Tasks

- [x] **T1 — plan + substrate gate** (this file; gate verdict above).
- [x] **T2 — `src/quant/lut_grid.rs`: the offline solver core** (UNGATED pure math, lib-tested at default features):
  - `Q2Grid` — 4 levels in d-units `{l0, 0, l2, +2}` (zero + anchor pinned; `l0 < 0 < l2 < 2`); the sign-absorbing scale is an encoder concern, the grid is sign-free.
  - `GRID_T0_UNIFORM = {−1, 0, +1, +2}` — the incumbent.
  - `WeightHistogram` — fixed 4096 bins over `[−2.125, +2.125]`, f64 MASS (each sample weighted by the block's `d²` — see the measured-objective note below).
  - `solve_lloyd_max` — EM from the T0 incumbent; ≤256 iters, stop `|Δ| < 1e−7`; empty-region + ordering guards. `histogram_mse`, `grid_digest` (BLAKE3 commitment).
  - Tests: fixed-point / symmetric-sanity / monotone-vs-T0 / bit-identity / empty + empty-region guards + the bin-pipeline probe (`weighted_histogram_objective_tracks_true_mse`).
- [x] **T3 — encoders + the grid-aware dequant in `src/quant/q2_0.rs`** (UNGATED, wire-format-native):
  - `quantize_row_q2_0_symmetric` — reference RTN; Bonsai byte-identity control (tested: ternary round-trip reproduces the original bytes).
  - `quantize_row_q2_0_asymmetric` — T0 (`d = sign(argmax|w|)·f16(amax/2)`, first-index argmax = the A2 convention; anchor always code 3 — tested).
  - `quantize_row_q2_0_grid` — nearest-level encoder over any `Q2Grid` (decoded-space, ties → lowest code; error-equal to round-clamp on the uniform grid — tested).
  - **`dequantize_row_q2_0_grid` — the Q2_0A decode path.** The base wire decode hard-pins `(c−1)·d`; a non-uniform grid is UNCONSUMABLE without a decoder that knows the grid. This was the session's load-bearing finding (below).
- [x] **T4 — `src/bin/lut_grid_solve.rs`** (feature `lut_grid`, `[[bin]]` row + feature in the same commit): pooled d²-weighted histogram → solve → 3-arm per-family weight-space eval + `--granularity` sweep + `--out` JSON.
- [x] **T5 — run on both dense artifacts** (tables below; grids differ per model ⇒ per-model committed grids, which the BLAKE3 commitment already anticipates).
- [ ] **T6 — model-level G1 (open):** ppl + per-family conditional retention with re-encoded weights (grid-aware decode / Q2_0A reader); the lane's GOAT gate.
- [ ] **T7 — T4 kernel cost** (documented in Issue 027, no code): Q2_0A artifacts need the grid-aware decode kernel (a per-level scale multiply — the LUT form `lut16-pshufb-byte-expansion-at-stage` + one f16 scale), priced against the ternary sign-magnitude fast path. Rides Issue 028.

## Results (weight-space, 2026-10-01, M3 Max, AC, release build)

**Gemma-2-2b-it (2.614B weights, 183 tensors):** grid **l0 = −0.708555, l2 = +0.621162**, 10 iters, commitment `b28b3661…102b380` (full digest in the run log; hist MSE −23.59%).

| family | SNR sym | SNR T0 | SNR solved (Q2_0A) |
|---|---|---|---|
| attn_k | 2.21 | 7.09 | **7.83** |
| attn_output | 2.28 | 7.30 | **7.87** |
| attn_q | 2.23 | 7.14 | **7.83** |
| attn_v | 2.20 | 7.05 | **7.82** |
| ffn_down | 2.20 | 7.06 | **7.82** |
| ffn_gate | 2.26 | 7.26 | **7.87** |
| ffn_up | 2.26 | 7.27 | **7.87** |
| token_embd | 2.17 | 5.92 | **7.16** |

**MiniCPM5-1B (1.081B weights, 170 tensors):** grid **l0 = −0.816637, l2 = +0.708580**, 9 iters, commitment `642e2d0b…4210f0` (hist MSE −17.25%).

| family | SNR sym | SNR T0 | SNR solved (Q2_0A) |
|---|---|---|---|
| attn_k | 2.19 | 6.99 | **7.91** |
| attn_output | 2.31 | 7.46 | **8.19** |
| attn_q | 2.27 | 7.33 | **8.11** |
| attn_v | 2.11 | 6.69 | **7.72** |
| ffn_down | 2.22 | 7.18 | **8.04** |
| ffn_gate | 2.28 | 7.33 | **8.10** |
| ffn_up | 2.29 | 7.35 | **8.12** |
| output (lm_head) | 2.05 | 6.31 | **7.39** |
| token_embd | 2.31 | 7.48 | **8.20** |

Granularity sweep (T0 rule, both models agree in shape):

| group | bpw | gemma SNR | MiniCPM SNR |
|---|---|---|---|
| per-128 | 2.125 | 6.09 dB | 7.21 dB |
| per-64 | 2.250 | 7.24 dB | 7.92 dB |
| per-32 | 2.500 | 8.39 dB | 8.72 dB |

**Weight-space verdicts (proxy — T6 is the GOAT gate):**
1. **T0 beats the symmetric reference by ~+4.9 dB on every family, both models** — the lane's core premise (activating the 4th code state encoder-only, same bytes) is a massive dense-arm win.
2. **Q2_0A (solved grid) beats T0 by +0.5..+1.2 dB per family at matched bpw** — the T2 bar (beat T0, not the ternary encoder) is met at the proxy.
3. **Per-model grids differ materially** (l2: 0.621 vs 0.709) — one global grid would strand ~0.2 dB; commit per-model (the commitment mechanism already anticipates this).
4. Finer scale granularity buys ~+0.7 dB per +0.125 bpw (per-64) — positive at the weight-space proxy; the wire-format cost side is T5's open question at the model level.

## The load-bearing finding: the wire decode pins the level ladder

The first full run reported the solved grid LOSING ~1.3 dB to T0 on every family while the histogram claimed −23.6% — a contradiction. A family-restricted objective diagnostic (hist vs true, both computed from the definition) showed hist ≡ true to 4 digits and the solved grid winning by 13–25% everywhere ⇒ the table's wire path was the defect: `dequantize_row_q2_0` computes `(c−1)·d`, so solved-grid codes decoded against the UNIFORM ladder — the grid never reached the dequant. **A non-uniform grid is a FORMAT variant (Q2_0A), and its real content is the grid-aware decode** — now `dequantize_row_q2_0_grid` (tested: uniform-grid decode ≡ base decode; solved-grid round-trip beats T0 on the solved distribution). The same constraint is exactly why T4's kernel cost is real: the LUT decode kernel must be grid-aware too.

## The measured objective lesson

The solver's histogram was initially UNWEIGHTED (element counts): the solved grid was −15.7% on that histogram yet lost ~0.7 dB on energy-weighted SNR — the objective must match the metric. The extraction now weights each sample by the block's `d²`, making the histogram objective EXACTLY `Σ(w−q)²` (verified: the d²-weighted histogram's per-family restricted objective reproduces the true element MSE to 4 decimal places — the diagnostic table, preserved in the session log).

## Honesty notes

- Weight-space SNR is a PROXY; the GOAT gate is model-level per-family retention (T6). No promotion claim this session.
- The histogram seeds the grid; the wire round-trip eval is the self-consistent check (solved beats T0 through the REAL encode→grid-dequant path, not just on the histogram).
- Bonsai byte-identity control: symmetric re-encode of ternary-valued blocks reproduces the original bytes — a test, not a convention (`symmetric_reencode_of_ternary_blocks_is_byte_identical`).
- The asymmetric re-encode of ternary input reproduces-or-loses — also executable (`asymmetric_reencode_of_ternary_blocks_never_beats`).
