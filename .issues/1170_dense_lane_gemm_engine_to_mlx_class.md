# Issue 1170 — dense-lane GEMM engine to MLX-class: the residual ~1.5× (staged-half-B / fork-parity class) + the decode GEMV gap

**Status:** OPEN — the bench-940 follow-up lead. Not claimed (no Claimed
line yet — append one under this Status line before implementing, per the
claim-lease law).

## Problem

Bench 940 (relaxed_precision) took the dense-qwen38 Metal lane to
**pp512 483 / pp2560 587 / pp5120 489 tok/s** — 1.84× the fork (898),
2.37× uzu (1157), 2.89× MLX (1411) at the pp-5k cell. The kernel A/B puts
the relaxed matmul2d at **41.5–53.3 TFLOPS** on the production shapes; MLX's
e2e implies ~74 TFLOPS effective on the same GEMMs — the remaining gap is
**~1.5×, all in the GEMM engine**, and structurally explained:

- llama.cpp's `kernel_mul_mm` STAGES BOTH operands to threadgroup memory
  (A dequanted to half, B converted to half) before the MMA; ours reads A
  (weights, f16) and B (activations, **f32**) as DEVICE views — the mixed
  half×float device path forecloses the cooperative-load/prefetch schedule
  the fork gets (L2-friendly tile walks, no device-memory stalls inside the
  K loop).
- MLX additionally runs its own tuned matmul family (persistent-thread-class
  scheduling) on the same silicon.

Two sub-levers, one arc:

1. **Prefill: staged-half-B matmul2d variant** — convert X once per chunk
   to f16 (one small elementwise kernel, activations are ~10 MB @p512 —
   trivial), then run the matmul2d with BOTH operands f16 device views (or
   staged). Precondition to test: does half×half + cooperative loads beat
   the mixed path's 41–53 TFLOPS materially? The `matmul2d_descriptor`
   accepts it (the ternary kernel's staging precedent is in-crate);
   numerics: activations rounded f16 (lossy class, same discipline as
   bench 923/940: pre-registered band + walk agreement). Expected ceiling
   if it works: fork-parity (×1.4-1.8 on the GEMM buckets → e2e
   ~700-900 @pp512-class cells).
2. **Decode: the f16 GEMV is ~8× off bandwidth** — decode on the f16
   posture reads 1.86 tok/s vs a ~15 tok/s bandwidth bound (54 GB of f16
   weights at ~800 GB/s). The `gemv_dense_f16` kernel is one
   simdgroup-per-row with strided reads — the Q6K tail's packed GEMV
   (bench 939) already proved the pattern that saturates (per-lane partials
   + one reduction). NOTE: decode on this lane is explicitly non-serving
   (the packed posture serves decode at 7 tok/s flat — the audit) — the
   lever matters only if the dense posture ever becomes the serving
   posture, or for parity rows. Price it AFTER lever 1; do not lead with it.

## Acceptance gates

- **G1 (kernel):** pre-registered band vs the f32 CPU reference (the
  relaxed gate's fixture machinery, `qwen38_dense_f16_matmul2d_g1` — extend,
  never fork); determinism pin byte-identical.
- **G2 (kernel):** TFLOPS A/B at the six production shapes
  (`dense_f16_relaxed_ab` pattern — ABBA, median of pair-ratios) — the
  variant must beat 53 TFLOPS on mlp.gate/up to justify the e2e cell.
- **G3 (e2e):** pp512/pp2560/pp4954 cells vs bench 940's 483/587/489;
  512-token walk agreement stats vs both prior arms.
- **G4:** feature-off byte-identical (the selector pattern — a new env,
  promote-only; never fold into `RIIR_QWEN38_DENSE_MM2D_RELAXED`).
- Lossy-surface discipline: per-family deployed-path read before any
  promotion call; owner call on any default flip.

## Non-goals

- The packed (Q4_K) posture's CubeCL GEMM wall (its own lane; the audit's
  incumbent).
- The GDN recurrence scan (bench 938: ~6%, latency-adequate).
- Attention-core kernels at prefill (the attn bucket's non-GEMM ~70 ms at
  pp512 — re-read the shares after lever 1 before touching).

## Provenance

Bench 940 (`bd297c7` + a2d05ce) measured everything above; the kernel A/B
table and the wte sync-stall split are in
[`.benchmarks/940_issue1150_dense_relaxed_precision_2x.md`](../.benchmarks/940_issue1150_dense_relaxed_precision_2x.md).
Machine: amo (M5 Ultra). The lane's env surface after bench 940:
`RIIR_QWEN38_DENSE_MM2D`, `_LMHEAD_Q6K`, `_DENSE_MM2D_RELAXED` (all
promote-only), `_GDN_CHUNKED` (measured negative, opt-in).
