# Bench 898 — Issue 1126 fused decode core (beta/decay + expand/L2 + fused recurrence in one launch): G1 bit-identical everywhere, G2 measured NEGATIVE (−3.5% decode e2e) — stays opt-in `RIIR_GDN_CORE_FUSED=1`, default OFF

**Date:** 2026-10-26 ~20:4x–21:0x +07, shikuwa/4090, verified-idle (551–518 MiB desktop baseline, no compute co-tenants, no cargo contention).
**Trigger:** Issue 1126 implementation (the Bench-1004 escalation ladder's #398-twin rung — the named decode-shape delta after 1125's CUDA-side dedup grep refuted the fused-quantize premise).

## The cells (bench_734_t5_decode ARM 1, manifest posture verbatim, B734_ROUNDS=3, B734_WARM_SECONDS=10, one binary — env flips the dispatch)

| posture | median | tok/s | rounds | G1 argmax-walk FNV | logits FNV |
|---|---|---|---|---|---|
| incumbent (env unset) | 10.787 ms | **92.71** | 92.71/92.37/93.03 | `98412dc406c4ea19` EXACT | `63496f6688c8f742` EXACT |
| armed `RIIR_GDN_CORE_FUSED=1` | 11.283 ms | **88.63** | 89.02/88.17/88.63 | `98412dc406c4ea19` EXACT | `63496f6688c8f742` EXACT |
| incumbent (window 2) | 11.013 ms | **90.80** | — | EXACT | EXACT |
| armed (window 2) | 11.253 ms | **88.86** | 88.86 + ARM-4 re-read | EXACT | EXACT |

**Verdict: G2 FAIL — armed reads −3.5%/−2.1% vs the incumbent's own two windows; the direction is unambiguous (both armed windows ≥ 11.25 ms, both incumbent windows ≤ 11.01 ms), outside the ~2% window spread.**

## G1 (bit-identity) — PASS everywhere, three independent layers

1. **Kernel-level**: `test_recurrence_fused_expand_bit_identical_to_chain` — 5 sequential compounding steps, every state AND output element `.to_bits()`-equal vs the incumbent 3-launch chain (beta_decay → expand_and_l2_normalize → recurrence_fused), at the production geometry (n_k=16, n_v=48, hd=128) + an off-shape (4,8,64) + the zero-head L2-guard edge. The body refactor (the Issue-742 math moved into `recurrence_fused_body_impl<HD, EXPAND_INLINE>`, one instruction stream for both arms) left all 68 pre-existing deltanet tests green.
2. **Real-model walk**: the ARM-1 48-token argmax-walk pin `98412dc406c4ea19` + the logits pin EXACT in BOTH postures, both windows.
3. **The 512-step teacher-forced window (ARM 4)**: armed `lane=f32 argmax-stream FNV dfe85338a90fd217` + every per-64 block hash `a967adc950d35ee7 / 88e97fafdf0742d5 / 2e22e2fdb2d2c3af / c8e877fdf21996f2 …` **bit-identical to the incumbent's** — the fusion is numerically transparent end to end.

## Why it lost (the honest decomposition)

The fusion trades 2 launches/GDN layer (~56/token ≈ 150–170 µs at the in-graph dispatch floor) for redundant per-block compute: each of the 6144 (head,row) warp-blocks per layer re-runs its head's 128-iteration sequential L2 sums (×2 for Q and K) + the softplus/sigmoid — 32-thread blocks doing serial loops, where the replaced expand kernel did the same work as ~18432 parallel elements in wide 256-thread blocks. The added latency + L2 re-reads swamp the dispatch saving: measured +0.47–0.50 ms/token. **The opponent's #398 fuses the CONV chain (a genuinely launch-bound, tiny-work kernel) — not the recurrence — and this A/B is the measurement showing why our recurrence-side fusion is the wrong work at the wrong block shape.** The misread was mine: I fused the launch-bound kernel's NEIGHBORS into a compute-bound kernel instead of fusing launch-bound WITH launch-bound.

## Disposition (the decline law — decline is a correct answer)

- The kernel + test + env gate SHIP opt-in (default OFF — zero production-path change, byte-identical behavior at the default), with this record so nobody re-tries the shape blind (the Bench-995 fused-rotation precedent).
- The `recurrence_fused_body_impl<HD, bool>` refactor STAYS — it is the DRY win (one body, both arms) and is bit-identity-tested both ways.
- **Follow-up rung (NOT tried, named for the next window): 4→3, fuse beta_decay + expand_and_l2_normalize ONLY** (keep the recurrence untouched). Both are launch-bound elementwise kernels; the merged kernel keeps the expand kernel's 256-thread block shape and folds the n_head-wide beta/decay work into its grid — zero added redundancy, ~28 launches/token saved ≈ +0.8% ceiling. Expected inside the ~2% window band: needs the paired-interleaved A/B discipline to certify, never single-window reads.
- The decode-hold defense therefore remains OPEN: this rung consumed, the census's other named levers (attention-chain q/k-norm+rope+append fusion, embedding-dequant+inverse-FWHT fusion, the 4→3 above) unclaimed.

## Provenance

- Model `Ternary-Bonsai-2-27B-PQ2_0.gguf` (sha `3907dc16…`, the league file), `cargo test -p riir-gpu --release --features bonsai2_hadamard --test bench_734_t5_decode` at the manifest-verbatim cmd; riir-infer tip + this change (uncommitted at run time — the A/B is one binary, env-dispatched, so the comparison is internally exact).
- ⚠ Feature-combo note found en route: 1320dda's `cuda_glu_cache` field is cfg-gated `ternary_gemv_cuda_raw + ternary_gemm_batched + not(macos)` but the assignment sites in `gemv_ternary_cubecl.rs` are not — `cargo check -p riir-infer-gpu --features ternary_gemv_cuda_raw` (WITHOUT `ternary_gemm_batched`) fails E0560. The LEAGUE posture is unaffected (riir-gpu defaults pull `ternary_gemm_simdgroup` → `ternary_gemm_batched`); flagged for the i1124 owner to gate the sites (not this issue's file).
