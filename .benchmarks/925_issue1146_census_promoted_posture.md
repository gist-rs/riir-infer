# Bench 925 — Issue 1146: prefill frame census under the PROMOTED posture — attention is now the largest class (37%)

**Status:** RECORD — re-attributes the 1146 frame after the day's GEMM rungs (tensor zero-copy + H4 + relaxed). **Attention = 37% of the pp4096 frame** (Bench 914: 9%, before the 2–3× GEMM gain), FFN GEMMs 34%, DeltaNet projections 29%. Confirms ladder 4 (prefill attention runs on no matrix units) as the next lever.
**Date:** 2026-10-28 21:20–21:31 +07
**Session:** katgpt-rs-71, 1793178290. Census knobs: katgpt-rs-bd `3cde142`.
**Issue:** [`.issues/1146_prefill_throughput_gap_1box.md`](../.issues/1146_prefill_throughput_gap_1box.md)
**Box:** bmo (M5 Ultra, 96 GiB), AC; the runner's GPU-quiet gate passed at start
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` (league pin)

## Posture

`bench_1146_prefill_frame_census` (built at riir-infer `465d173`) with `I1146C_TENSOR=1` (tensor zero-copy GEMM), `RIIR_METAL_TENSOR_RELAXED=1` (relaxed matmul2d), H4 dequant at its default, `I1146C_CHUNK=512 I1146C_FRAMES=8` (prompt 4096). Strip-down Δwall with the bit-identity anchors (never-touched == ALL-first == ALL-after) PASSED. This is the dist serving posture on a 1-box full range.

## The frame (ms/frame, ALL = 911.79 → 7.29 s per 4096 tokens, matching dist's 7.13 s in Bench 923)

| class | Δ ms/frame | share |
|---|---:|---:|
| `-attention` (flash + attn projections, 8 layers) | **334.48** | **37%** |
| `-ffn` (gate/up/down GEMMs + swiglu) | 307.15 | 34% |
| `-deltanet` (in/out projection GEMMs) | 264.63 | 29% |
| `-gdn_prework` (chunked conv) | 71.87 | 8% |
| `-gdn_rec` (multi-token recurrence) | 60.01 | 7% |
| `-gdn_post` | 10.94 | 1% |
| `-embed` | 8.52 | 1% |
| `-gdn_beta` / `-gdn_residual` / `-tail` / `-gdn_rotin` / `-gdn_norm` | ≤ 2.3 each | ~0 |

G-C1 coarse additivity: 915.34 of 911.79 ms attributed (residue −3.55), so the three coarse classes partition the frame. The gdn sub-bits nest inside the deltanet class's chain and are not additional.

**Versus Bench 914** (same census, morning posture, chunk 128, simdgroup GEMM, 873 ms/frame at 1024 tokens): FFN 65% → 34% and DeltaNet projections 25% → 29% of a frame now 2–3× cheaper per token, while **attention went 9% → 37%**. Attention barely moved in absolute terms; it now *dominates* because everything around it got faster. Its share grows further with context (pp32k attends 8× the history).

**Wide-chunk control:** 4×2048 frames = 1.654 ms/token vs 1.781 at 512. Width beyond 512 is worth ~1.08×, so the width lever is exhausted.

## Site-timing rows: an artifact, not a finding

The projection site-timing snapshot shows `gdn_out_proj` and `ffn_down` at max 270–283 ms per dispatch. Every other site is ≤ 11 ms. These rows time the call itself, *including* the `client.flush()` at the top of `MetalTensorZeroCopyGemm::dispatch`, so they absorb whatever CubeCL work was queued just before (the swiglu chain before `ffn_down`, the recurrence chain before `gdn_out_proj`). The strip-down Δ table above is the attribution of record. Do not rank rungs off the site rows.

## What this ranks

1. **Ladder 4, prefill attention (37%)**: the M=32 tiled CubeCL flash arm uses no matrix units. (a) Re-A/B the existing f32 cmma arms: queued (`/tmp/i1146q/attn_cmma`). (b) A matmul2d flash-attention kernel modelled on upstream `fa_tensor.metal` is now the largest single lever.
2. FFN + DeltaNet GEMMs (63% together) are already on the tensor + H4 + relaxed path. Further gains there are kernel-internal (matmul2d tile geometry at prefill shapes).
3. The GDN chain (prework + rec ≈ 15%) is the next class after those.

## Provenance

- Log: `.benchmarks/i1146_census_promoted/census_promoted.log`.
