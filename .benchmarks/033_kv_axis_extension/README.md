# Bench 033 — the KV-axis extension arm (Issue 033): the KV-store axis is NULL; the activation axis dominates

**Date:** 2026-10-05 · **Box:** shikuwa (4090, RTX 4090 24564 MiB, 23.3 GiB free at start; no
co-resident compute apps — the runner's own exclusivity probe) · **exe:** riir-infer-gpu
`dq_phase_matrix` at `ad683a6` (release, `--no-default-features --features
dq_phase_bench,ternary_gemv_cuda_raw,ternary_gemm_batched`) · **model:**
`Ternary-Bonsai-2-27B-PQ2_0.gguf` blake3 `b4f6ab95…d2550f6` · raw: `report.md` + `run.log`
(this dir; the scheduled-task log with every generation line).

## What ran

The dq_phase_matrix runner extended with Issue 033's KV-STORE axis
(`DQ_KV_GRIDS`): every stored K/V cache row is rounded in place to the grid
at WRITE (the q8kv store spelling) — prefill rows via the CUDA `kv_fill`
site, decode rows via the CubeCL `forward_attention_layer_gpu` site (the
runner's real decode path; the smoke at `8f24e82`'s parent caught the
cudarc-only wiring miss, fixed before this run). Store ≡ read for accuracy
(idempotent quant-dequant — the values entering the attention dot are
round(x) either way); the axis is ONE accuracy experiment. 16 layers carry
KV (attn_layers derived from `weights.layer_types`).

Cells: base×2 + {pf,dec,both}×{a2,a4} + dec_a8 control + {kv,pfkv}×{a8,a4}
= 16 cells × (48 hard arith + 3×32 NIAH). Corpora hardened per the issue's
admissibility requirement: `DQ_ARITH_HARD=1` (ops 3–6, first 1k–99k) pulls
base arith to **0.7500** (v2's 0.9583 was above the 0.95 ceiling);
`DQ_NI_NEEDLES=10`. Frozen defaults stay byte-identical; the corpus blake3
`2dd4544f…ad93e059` is this posture's seal.

## Gates

G-i1 (knob-off counters == 0 incl. the new kv counter) PASS · G-i4 (base
twice byte-stable) PASS · G-i2 (exact launch counts, now with the KV
formula 2×attn_layers per chunk/step: kv cells read 195,744 = 2×16×6117
exactly; pfkv's extra 2,017,024/195,744 prefill-act + kv rows verify)
PASS all 16 cells · positive controls (armed cells move logits vs base)
PASS · dec_a8 control |Δarith| = 0 ≤ 2 PASS.

## The answer — axis dominance (decode-heavy arith, base 0.7500)

| axis | arith acc | Δ vs base | verdict |
|---|---|---|---|
| **kv-store[a8]** (q8kv-class) | **0.7500** | **0.0000, CI [0,0]** | **NULL** |
| **kv-store[a4]** (16-level affine) | 0.7708 | −0.0208, CI [0, 0.0625] | **NULL** |
| interaction pf→pfkv[a4] | 0.8125 | −0.0208, CI [−0.0417, 0.1042] | none |
| prefill-act[a2] | 0.2083 | +0.5417 | catastrophic |
| decode-act[a2] | 0.1250 | +0.6250 | catastrophic |
| pf/dec-act[a4] | 0.7917 / 0.7708 | ≈0 | NULL |

NIAH (base pooled 1.0000): kv-store 1.0000 at BOTH grids — every item
identical to base; prefill-act[a2] collapses it (0.25/0.19/0.13 per length).

**The KV stored/read precision axis is NULL on this model+suite — even
16-level affine per-32 quantization of every K/V row is accuracy-free —
while the ACTIVATION axis dominates catastrophically at a2 and is
measurably free at a4.** For the serving lane this prices the axes for
real: the KV cache of the Qwen3.8-GDN hybrid tolerates a4-class storage
(~4-bit + scales ≈ 20–25% of f32 KV bytes) with zero measured accuracy
cost, and the sensitivity lives in the quantized GEMM activations, not the
cache. (Deployment caveats carried, not measured here: real q8kv kernels
pack/unpack differently than fakequant round-trips; the f32 KV working set
on this runner is block-clamped to 16k+64 slots.)

## Admissibility disclosures

All four decode-heavy rows gate cleanly (base 0.7500 ∈ [0.25, 0.95]). All
prefill-heavy rows read **INADMISSIBLE** by the pre-registered rule — base
NIAH pooled = 1.0000 > 0.95 (the model retrieves perfectly at 10 needles up
to 16k; harder corpora would need longer contexts than the 24 GiB card
fits at this model size). The raw pf_aq[a2] NIAH cells sit in the table —
the damage is real and dramatic, the formal verdict just cannot fire.
Recorded as the standing admissibility boundary of this instrument at this
model.

## Session

shikuwa-4090-i013-followup, 2026-10-05, ~09:40–16:40 (scheduled-task lane,
watchdog armed; two transient rustc 0xc0000005 crashes during the cold
build cleared on retry — recorded for the box's build-reliability history).
