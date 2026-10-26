# Issue 1124: prefill GLU-into-GEMM fusion — interleaved gate/up rows, single accumulator, write-back GLU (llama.cpp #29948 distill)

**Status:** T1 LANDED 2026-10-26 (4090, kernel probe 1.26×/1.23×, bit-identity GATE PASS) — T3 DONE 2026-10-26 (mined via refine Batch 287 `d871a0fc`, Plan 292 / Research 325) — T2 (e2e GOAT) armed on the league window; filed 2026-10-26 from
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

- [x] **T1 (POC, env-gated). LANDED 2026-10-26 — shipped BETTER than filed.** The
  filed shape (a bespoke fused kernel with a bespoke accumulator pairing) was
  REPLACED by the truer #29948 port after reading the v11gu prior art: the
  gate/up weight rows are INTERLEAVED in 8-row chunks at mirror-build time
  (`pack_bitplanes_to_q2_interleaved_gate_up` +
  `interleave_gate_up_group_scales`, `prefill_cuda_mma.rs`) so the **v10t body
  runs VERBATIM** — each m16n8 fragment's row halves (c0/c1 at g_id, c2/c3 at
  g_id+8) are then the GATE and UP dots of the SAME hid row, which is exactly
  the single-accumulator trick: no second accumulator set, no Bench-895
  register cliff. The GLU is applied IN the transposed staging pass from
  registers (`GEMM_BODY_V10T_GLU_Q8`, `gemm_ternary_i8_mma_v6_src.rs`): the
  scale multiply hits BOTH projections BEFORE the sigmoid in the standalone
  kernel's op order, `sig = div.full(1.0f + __expf(0.0f - g))` — the Bench-721
  canonical forms hardcoded, so the output is bit-identical to (v10t pair +
  swiglu_ffn_e1_d1) by construction. Env `RIIR_PREFILL_FFN_GLU`
  ("1"/"true"/"on", DEFAULT OFF) + the dispatch posture (fmt route packed
  mirrors, `mlp % 128 == 0`); the fused path skips the gate/up f32 staging
  grows and the SwiGLU launch (one launch per FFN gate/up pair, hid slab
  direct). Failure degradation: mirror build failure / shape mismatch / env
  off → the incumbent pair path byte-identical; a mid-block launch error →
  the whole block falls to CubeCL bit-safe (the existing contract).
  - **Unit gate (G1)**: `v10tglu_matches_v10t_pair_plus_swiglu_bitwise` —
    two shapes (256/256/130 ragged; 128/384/257 third-group) × distinct
    fixtures × the REAL standalone SwiGLU kernel as reference (compiled from
    FFN_CUDA_SRC): **0 bit-diffs**, PASS. Strict fold refuses loud (only
    Fused is instantiated — the FFN block's posture).
  - **Kernel probe (G2, level 0)**: `q8_gemm_v10tglu_perf_ab_ffn`, league FFN
    shapes (m=17408, n=5120), release, interleaved ×9 medians, 4090,
    GPU-quiet at launch (11% util, no compute apps — desktop class only,
    the exempt set): **regs=128, local_bytes=0** (the Bench-895 spill cliff
    dodged); p=2048: 2.502 → 1.989 ms = **1.258×**; p=4096: 5.255 → 4.283 ms
    = **1.227×**. The incumbent cell reads ≈ Bench-895's pair (1.867) +
    SwiGLU (~0.6) ✓ consistent; the win is exactly the filed mechanism
    (delete the SwiGLU pass + halve output slabs), untouched by the
    Bench-895 refutation (which priced the B-stage share, not the write-back).
    Single-window kernel-level probe — the T2 e2e battery owns the promote
    decision. Bench 897.
- [ ] **T2 (e2e GOAT).** pp512/pp2048/pp4096 A/B vs the incumbent at the league
  posture (`RIIR_PREFILL_FFN_GLU=1` vs unset, everything else pinned),
  verified-idle, PROVENANCE quoted, ×2 windows. Promote bar per the
  Feature Flag Discipline; demote the loser either way. VRAM note: the armed
  route builds the interleaved slab ALONGSIDE the pair mirrors (+44.6 MB at
  the 27B FFN shape) — a promotion would stop building the pair mirrors for
  gate/up (the fused slab replaces them), halving that overhead; measure the
  armed-state cost first.
  ⛔ **Vehicle note (measured 2026-10-26, so nobody re-burns this):** the
  riir-infer-internal e2e battery (`tests/issue028_disaggregated_handoff.rs`,
  the real 7.2 GB PQ2_0) drives `riir_infer_core::deltanet::ternary_forward` —
  it NEVER reaches `dispatch_ffn_block` (riir-infer-gpu's
  `ternary_deltanet_gpu_forward::prefill_tokens_chunk`), and its prompt is
  p=5 (under any meaningful FFN batch anyway). The ONLY e2e vehicle for the
  fused FFN block is the league runner (riir-ai `scripts/perf_rematch.sh`,
  Bonsai-2 through riir-engine) — which is why this task is gated on the
  league window and the bout owner's protocol, not runnable from this repo
  alone. Engagement proof at run time: `RIIR_PREFILL_CUDA_FFN_TRACE=1`
  prints the per-layer `[721-trace]` lines when the block dispatches.
- [x] **T3 (kernel_opt rider). DONE 2026-10-26 — already mined; dedup caught the
  duplicate before authoring.** The refine mining lane's Batch 287 (riir-refine
  `d871a0fc`, "the llama.cpp fusion trio P81/P82/P83", Plan 292 / Research 325)
  landed BOTH laws as count-neutral riders: the column-innermost operand-
  footprint order law rides `l2-grouped-tile-schedule` (Batch 287 rider, with
  the maintainer's measured GB10/5090 swing table) and the single-accumulator
  GLU write-back (chunks-of-8 gate/up interleave, unchanged MMA loop, halved
  store loop, the admission faces incl. the stream-K refusal / wave-efficiency
  posture) rides `fuse-gate-up-gemv-activation` (the MMQ prefill instance);
  #30489's mixed-type consumer-fold faces are mined alongside. Verified this
  session by corpus grep before writing anything (the Issue-825 class, caught
  at the dedup step this time).
- [-] **T4 (deferred — CubeCL twin).** The same fusion for the CubeCL ternary GEMM
  family (wg/WG-occupancy arms) is a separate cell; the cudarc lane is the league
  posture, do it first.
- [x] **T0.** This filing (technique distilled from #29948 + #30489 + the review
  thread; our gap verified by read; 4090 verify-row numbers quoted from the
  maintainer's pre-merge table).
