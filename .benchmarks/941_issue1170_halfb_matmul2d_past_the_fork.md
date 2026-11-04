# Bench 941 — Issue 1170: the half-B matmul2d (staged-activation half×half) — past the fork at the short cells

**Status:** LANDED (2026-11-04, zed-amo-i1170-halfb@amo). The bench-940
follow-up's G2 physics question (does the fork's half×half schedule beat
our mixed half×float device view?) answered YES at the kernel level
(1.31–1.46×), and the wiring landed: the prefill GEMMs ride a half×half
matmul2d over f16 activation twins behind
`RIIR_QWEN38_DENSE_MM2D_F16ACT=1` (promote-only; requires DENSE_MM2D=1).

## Cells (dense-f16 MM2D + Q6_K tail + relaxed + half-B, warm caches, fresh processes, median of 3)

| cell | bench 939 (exact) | bench 940 (relaxed) | **half-B (this)** | vs fork 898 @5k |
|---|---|---|---|---|
| pp512→512 | 274 | 483 | **640 tok/s (0.80 s)** | — |
| pp2484→2560 | 291 | 587 | **725 tok/s (3.53 s)** | **1.24× behind** |
| pp4954→5120 (league cell) | ~276 | 489 | **580 tok/s (8.80 s)** | **1.55× behind** |

League @pp-5k: MLX 1411 (2.43×) / uzu 1157 (1.99×) / fork 898 (1.55×) —
the lane's arc across three records: 67× → 4.7× → 2.9× → **2.4× behind
MLX**. The short cells sit past the fork (725 vs 898 at pp2560); the
residual at long p widens with the attention core's O(p²) share (the attn
bucket is 143–162 ms of a 1.0-1.3 s synchronized pass @p512 and grows
superlinearly — the next census reads it at pp5120 before any kernel work).

## The G2 evidence (kernel A/B, `dense_f16_half_b_ab`, ABBA median, p=512)

| shape | mixed (bench 940) | half-B (this) | ratio |
|---|---:|---:|---:|
| gdn.qkv 10240×5120 | 48.2 | 66.7 | 0.726 |
| gdn.z 6144×5120 | 42.7 | 53.2 | 0.785 |
| ssm.out/attn.o 5120×6144 | 39.7 | 53.6 | 0.756 |
| mlp.gate/up 17408×5120 | 52.9 | **73.3** | 0.710 |
| mlp.down 5120×17408 | 45.4 | 66.2 | 0.687 |
| attn.q 12288×5120 | 55.2 | **76.4** | 0.715 |

mlp.gate/up 73.3 TFLOPS ≈ MLX's effective bar (~74). The fork stages B
into threadgroup memory before the MMA; ours reads the activation as a
device f16 VIEW — no staging barrier, no dequant, the same schedule class
with less work. The remaining ~1.3× to MLX is MLX's own matmul tuning
(persistent-thread-class scheduling) — a different arc if ever priced.

## What landed

- `gemm_dense_f16h_metal_tensor.metal` — the half×half variant kernel
  (same name/bindings as the dense source; drop-in for the pipeline
  constructor; probe + production share one source).
- `new_with_gemm_source` — alternate-MSL constructor (the gemv function
  still compiles from the canonical source; pointer-equal reuse on the
  canonical path).
- `CastF32ToF16CubeCL` (`elementwise_cubecl.rs`) — the staging cast
  (round-to-nearest `half::f16::cast_from`, the same rounding the CPU
  fixture uses).
- Route: the F16ACT env (promote-only, exact literal, refuses without
  DENSE_MM2D and off-macOS/no-feature), four f16 twins (norm_x_b /
  hidden_b / rec_n_b / attn_out_b), `gemm_input` staging (ONE cast per
  producing buffer per chunk — never per GEMM), and the GEMM context
  selection via `gemm_mm_for_posture()` (half-B if armed, else mixed).
  Decode is UNCHANGED (the GEMV keeps f32 activations; the decode GEMV
  gap stays issue 1170 lever 2).
- G1: `dense_f16_half_b_g1` — 4 shape classes, pre-registered 3e-2 band,
  measured **2.1e-7–3.6e-7 rel-to-mag** (the CPU ref shares the f16
  activation rounding, so the band isolates accumulation order). 4/4
  gates pass (exact + relaxed + half-B + gemv); route G1 3/3.

## Numerics class (disclosed)

Activations rounded to f16 at staging (~5e-4 per term) + relaxed MMA when
the relaxed env is armed. **The 512-token greedy walk is BYTE-IDENTICAL
to both prior arms** — md5 `0764d040f09e30264603e762ddcd7c1a`, now
identical across exact / relaxed / half-B (5 runs total, 2× relaxed + 1×
exact from bench 940, 2× half-B). Not a single argmax flip across a 5e-4
logit-scale perturbation. The env stays promote-only; promotion calls
(demote-only flip, lane or posture default) read this record + bench 940
on the lossy-surface discipline — owner-gated like every default flip.

## Provenance

amo, M5 Ultra, AC power, GPU-exclusive at every cell (the co-rig sibling
holds CPU-compile work only); load 2.6–3.5. Code at the commit carrying
this record (probe `74e309f` + this wiring). Model
`~/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf`; logs
`/tmp/i1170_halfb_{arm,walk}.log`.

Session: zed-amo-i1170-halfb@amo, 1793795000 → (fold at record commit).
