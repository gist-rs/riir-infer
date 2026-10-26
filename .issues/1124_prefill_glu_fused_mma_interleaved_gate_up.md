# Issue 1124: prefill GLU-into-GEMM fusion — interleaved gate/up rows, single accumulator, write-back GLU (llama.cpp #29948 distill)

**Status:** Open — filed 2026-10-26 (4090 box, online-distill lane) from
[llama.cpp #29948](https://github.com/ggml-org/llama.cpp/pull/29948) (merged 2026-10-25,
`c49edf0a`, "CUDA: fuse FFN gate/up matmuls and GLU in MMQ with interleaved gate/up rows").
No plan opens until a session can run the build+bench battery on a quiet box; this issue
stages the distilled technique and the measured gap.

## The opponent move (what merged)

For prompt-processing batch sizes on the quantized GEMM (MMQ) path:

1. **Gate/up weight-row interleaving** — loaders interleave up and gate rows in chunks
   of 8, so each thread holds the up and gate value of the SAME output element (rows
   `r` and `r+8` of the MMA tile). Chunks of 8 (not 16) so one-tile-per-warp configs
   are covered.
2. **ONE accumulator** — the MMA loop, tile config, and `sum[]` are unchanged from the
   stock kernel; the alternative (a second accumulator tile for gate) was REJECTED in
   review against #28702.
3. **GLU in the write-back from registers** — `silu(gate)·up` computed where the
   values are already live; the standalone GLU kernel is DELETED from the launch
   stream. Half the output rows written.
4. **L2 tile-order law** (the load-bearing review finding, from the maintainers'
   measured table): with gate/up weights (~100 MB at the 27B shape) exceeding L2
   (~72 MB on 4090, ~24 MB on GB10), the fused kernel must run **column tiles
   innermost** so concurrently-running blocks share one weight tile through L2 —
   only y (activations, ~12 MB at 2048 tokens) stays resident. Measured at the 27B
   shape: fused launch 21.0 ms → 9.1 ms (Q8_0, GB10) from the reorder alone; the
   row-tiles-innermost order actively LOSES (−5.6% to −20.4%).
5. **Wave-efficiency gate** — fusion only when the plain-tiling grid holds ≥90% wave
   efficiency (the stock tiling-vs-stream-k rule); otherwise the unfused stream-k
   path runs. On 4090 all of pp512/pp2048/pp4096 at ub≥256 pass the gate.

Opponent-measured upside, **RTX 4090 rows from the maintainer's pre-merge verify table**
(llama-8B, pp4096, master → PR): Q8_0 ub512 **1.11×**, ub1024 1.07×, ub2048 1.08×;
IQ4_XS ub512 **1.08×**; Q4_K_S ub512 1.04×. Author's own Ada numbers (Qwen3.8-27B
Q4_K): pp512 +1.3% — and the per-FFN breakdown says on Ada "the gain comes mostly from
removing the GLU kernel rather than from MMA-side savings" (stock pair ~880 µs ≈ fused
~884 µs; the down-side GLU fusion in #30489 measures slightly faster per FFN).

## Our gap (measured at filing)

- `crates/riir-infer-gpu/src/prefill_cuda_ffn.rs` (`RIIR_PREFILL_CUDA_FFN`, cudarc
  mma.sync path): gate GEMM + up GEMM as TWO separate kernels + a STANDALONE
  `deltanet_gating_f32` SwiGLU kernel + down GEMM — 3 launches per FFN; both
  intermediates materialized to global memory.
- `ternary_ffn_fused.rs` (Issue 600, the GEMV/decode path): 3 GEMVs + SwiGLU in one
  command buffer — better (one sync), but SwiGLU is still its OWN kernel between the
  GEMVs; the gate/up pair is not merged into one pass.
- Nothing in the ternary GEMM family (`gemm_ternary_*`) interleaves gate/up rows or
  applies GLU in write-back. Vocabulary check: `interleav`, `gate_up`, `glu`+
  `write` all return zero hits in the GEMM files.

## What would port (the GOAT-shaped task)

Fuse gate+up into ONE ternary GEMM pass with interleaved weight rows and apply
SwiGLU in the epilogue write-back (registers), then down-GEMM — under the existing
bit-identity selectors (the mma path's variant-family lesson: exp/div lowering is
driver-dependent; the fused SwiGLU must reproduce the EXACT op order
`g·sigmoid(g)·up` the FNV pins hold). Targets the pp rows where the opponent just
banked +2–11%.

Honest caveats:

- **Coverage asymmetry, both ways.** The opponent's fusion covers Q4_K/Q5_K/Q6_K/
  Q8_0/IQ4_XS — NOT Q2_0 — so our league cell (Bonsai-2 27B PQ2_0) is NOT directly
  moved by their merge; but THEIR pinned league build predates this anyway. If we
  port the technique, OUR pq2 ternary GEMM gains where theirs does not run. That is
  the closing opportunity, not a threat, for the hold.
- **The promote-bar lesson stands**: our own fused GDN fold-tail + SwiGLU-rotate
  DECODE kernels measured e2e +1.40% — BELOW the promote bar → opt-in (Bench 995).
  Launch-count wins do not automatically survive e2e at decode batch sizes; the
  prefill batch regime (where the whole intermediate is bandwidth) is a different,
  more favorable cell — but the burden of proof is the same: e2e pp A/B or it
  didn't happen.
- Weight-layout constraint: interleaving requires gate and up in the SAME quant
  family/shape (the opponent's #30489 follow-up exists precisely for mixed-type
  FFNs). Our Bonsai quant is uniform per-layer, so the constraint is satisfied by
  construction — verify per-tensor at load.

## Tasks

- [ ] **T1 (POC, env-gated).** Interleaved gate/up loader + single-accumulator MMA
  with SwiGLU-in-write-back for the cudarc `mma.sync` prefill path behind a new env
  (default OFF), gated to the Bench-720 posture (`QuantDiv::Full` + `FoldMode::Fused`
  + `MmaTile::Tm64`, `n % 128 == 0`, `mlp % 128 == 0`). Bit-identity gate: full-model
  FNV pins byte-identical (the fused SwiGLU op order must match the standalone
  kernel's, including the variant-family selection).
- [ ] **T2 (e2e GOAT).** pp512/pp2048/pp4096 A/B vs the incumbent at the league
  posture, verified-idle, PROVENANCE quoted, ×2 windows. Promote bar per the
  Feature Flag Discipline; demote the loser either way.
- [ ] **T3 (kernel_opt rider).** The tile-order law (column-tiles-innermost when
  weight footprint > L2) + the single-accumulator GLU write-back are minable as
  kernel_opt rules — route through the refine mining lane (do NOT mine here; this
  issue is the pointer).
- [-] **T4 (deferred — CubeCL twin).** The same fusion for the CubeCL ternary GEMM
  family (wg/WG-occupancy arms) is a separate cell; the cudarc lane is the league
  posture, do it first.
- [x] **T0.** This filing (technique distilled from #29948 + #30489 + the review
  thread; our gap verified by read; 4090 verify-row numbers quoted from the
  maintainer's pre-merge table).
