# Bench 944 — Issue 1173: the fused norm+gating family A/B — the launch-floor lever REFUTED at e2e; the census's per-section floors are marker-boundary instrumentation cost

**Status:** LANDED (2026-11-04, zed-1ee7a33e@amo). Instrument commit `5bb84f1`
(fused kernels + G1 + wiring; promote-only env arms, byte-identical
demote). **GOAT VERDICT: FAIL** — the pre-registered bar (≥ +2% e2e per
landed fusion) is met by NO fusion, nor by the family. The arms stay
opt-in, never default.

## The arms (pp5120, promoted league posture, median of 3, steady passes)

Posture: `RIIR_QWEN38_DENSE_MM2D=1 RIIR_QWEN38_LMHEAD_Q6K=1
RIIR_QWEN38_DENSE_MM2D_RELAXED=1 RIIR_QWEN38_DENSE_MM2D_F16ACT=1
RIIR_QWEN38_ATTN_TENSOR=1` (bench 943's promoted posture) + the fusion
arms as promote-only exact literals.

| arm | fusions | all passes | steady median | Δ vs baseline |
|---|---|---|---:|---:|
| baseline | none | 949/970/977, 945/968/972 | ~971 | — |
| F1 | SSM_ZGATE | 963/979/974 | ~976.5 | +0.6% |
| F1+F2 | + NORM_CAST | 982/983, 976/983 | ~981 | +1.0% |
| F1+F2+F3 | + RESID_NORM | 978/978, 985/978 | ~978.5 | +0.8% |

**Noise honesty (verdict-required wording):** the arm ORDERING is not
stable — F1+F2+F3 (978.5) reads BELOW F1+F2 (981), and every per-fusion
delta is the same size as the ~±0.5% run-to-run spread. Within noise,
the honest statement is: **no measurable per-fusion e2e gain; the
family, if it gains at all, is ≤ ~1%** — do not re-arm for a quoted
"+1%". Warm-up pass 1 is always the lowest (first-touch PSO prepare;
the 943 law) — steady passes 2–3 decide. Box: amo (M5 Ultra, macOS
27.2), AC power, thermal clean, load ~2.3, no other GPU compute
consumer.

## Verdict: FAIL — and the mechanism finding is the payload

1. **The launch-floor lever hypothesis is refuted.** Bench 943 named
   the non-GEMM share as a per-kernel launch floor (~0.5–0.7 ms per
   small kernel at pp5120) and issue 1173 pre-registered ≥ +2% e2e per
   fused pair. Measured: removing ~5 launches per GDN layer per chunk
   (family F1+F2+F3 ≈ 240 launches/chunk across 48 layers) buys at
   most ~+1% (within noise). The free-running marginal launch cost is
   **~0.02–0.05 ms per launch**, not 0.55 ms. ⚠ Narrowed claim
   (verdict): this refutes the per-launch-floor lever FOR THE
   NORM-SHAPED PAIRS measured — it does NOT prove the whole ~403
   ms/chunk non-GEMM share is unremovable; that share still has no
   free-running owner (no instrument has attributed it without
   serialization). Issue **1174** files the non-serialized attribution
   instrument (kernel-skip ablation / per-layer timestamp pairs) that
   must land before the next lever is picked from 942/943's tables.
2. **The census's per-section walls do not collapse under fusion —
   they are marker/hazard instrumentation boundaries, not launches.**
   The fused census (all three arms + GDN_CENSUS, pp5120 steady):
   `norm_ssm 0.57 + z_gate 0.55 = 1.12 ms` vs bench 943's unfused
   `0.56 + 0.56 = 1.12` — IDENTICAL. The fused kernel's wall simply
   relocated into the following mark's hazard wait. Section sum 9.7
   ms/group fused vs 8.8 unfused (slightly HIGHER — the extra fused-kernel
   prepare + hazard set). **943's per-section walls are marker/hazard-
   serialized and do not predict free-running savings**; the same
   caveat applies to 942's StageTimers attribution (both instruments
   serialize). Annotated in place at 942/943 (append-only dated rows).
3. **The e2e bar is the binding bar — it held.** This is the
   pre-registration working as designed: the census named the lever,
   the A/B refuted it, the negative is recorded before any promotion.

## What DID land (and stays)

- **Byte-identity, both paths, bitwise**: G1
  (`tests/qwen38_fused_norm_gating_g1`, 3/3) asserts GPU-vs-GPU bitwise
  on f32 + f16 twin + the twin-less posture; e2e greedy walks are
  digit-exact on BOTH the sequential path (48/48 ids) AND the batched
  chunked path (~600-token prompt → full 512-token chunk ingest,
  16/16 ids).
- **Instrument law (the sharp one): a ZERO-LENGTH buffer binding is
  clamped to the WHOLE buffer by the wgpu/Metal binding layer.** The
  twin-less launcher initially bound a 0-length view of the f32 output
  as the f16 slot; `output_h.len()` then read FULL length, the twin
  write fired through the aliased slot and trampled the f32 output
  (caught by the G1's twin-less arm, never shipped). The fix — bind a
  one-element γ view as the unused slot and gate the twin write by an
  explicit `params[3]` flag, never the slot length — is pinned by the
  launcher comments and the twin-less G1 arms. This law applies to ANY
  CubeCL/wgpu kernel that branches on a bound-slice length (the
  `RmsNormQkFusedCubeCL` zero-length precedent is UNSAFE by this
  reading — its 0-len slots are never length-inspected, which is why
  it never fired there).
- The three fused kernels + wiring stay **opt-in**
  (`RIIR_QWEN38_FUSE_SSM_ZGATE` / `_NORM_CAST` / `_RESID_NORM`, exact
  literal `1`, unset = incumbent byte-identical) for replication;
  ≤ ~1% if armed together (within noise — see the honesty note above).
  Never default at this bar. F3's cross-layer wiring (the ingest-loop
  prefetch thread) is part of the same opt-in.

## Follow-up issues filed (the verdict's list)

- **Issue 1174** — the non-serialized attribution instrument
  (kernel-skip ablation / per-layer timestamps) — required before any
  lever is picked from the 942/943 tables.
- **Issue 1175** — the zero-length-binding launcher audit (the hazard
  is a latent bug class; `RmsNormQkFusedCubeCL` is safe by accident).
- **Issue 1176** — the bare-default ~7 tok/s vs ~971 promoted
  promotion-debt defect (promote the dial or write the reason).

## Observations (out of lane, recorded)

- The no-envs DEFAULT posture (no DENSE_MM2D etc.) runs pp5120 at
  **~7 tok/s** — the un-promoted GEMM route is ~100× off the league
  posture. Pre-existing (the 1146 arc's whole point), recorded here so
  a future bare `qwen38_wgpu_bench` run doesn't misread it as a
  regression. The league posture is the measured surface. Now tracked
  as **issue 1176** (the promotion-debt defect; a doc note alone was
  judged insufficient by the verdict).

## Provenance

- Release build `5bb84f1`, isolated target `/tmp/i1173-target`,
  `--features qwen38_wgpu,metal_tensor_gemm`. Cells: median of 3
  passes + a warm-up, one process per arm (env-per-process; arms not
  interleaved — the steady medians are stable across repeat runs:
  baseline 970→968, F123 978→978).
- Model: `/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf`
  (64 layers = 48 GDN / 16 attn).
- Identity walks: `--walk --walk-tokens {48,16}`, fused = all three
  arms. Census run: 943's posture + the three arms + GDN_CENSUS=1.
