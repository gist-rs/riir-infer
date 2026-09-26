# Bench 870 — Issue 879 Phase 2: ternary recipe/loader audit (arXiv:2609.04098)

> **Moved 2026-09-26:** from riir-ai `.benchmarks/` (same number — riir-infer's bench ledger jumps 4 → 871; the number stays allocated in both). Instrument paths are sibling-relative and still resolve from this repo (`../katgpt-rs`, `../riir-train`).

**Verdict: ALL FOUR ARMS PASS** — the production Ternary-Bonsai-27B serving path
already implements the paper's quantization-survival recipe; no defect found.
Per Issue 879 T2.4: a pass is a one-line note in the bench doc — this doc is
that note, expanded to carry the evidence.

**Status:** RECORD (2026-09-06, CPU-only audit — GGUF introspection + kernel
source reading; no GPU time; instrument `../katgpt-rs/scripts/gguf_header_audit.py`).

## What was audited

Paper: "Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the
Recurrent Half of a Hybrid 27B LLM" (Kozyrev & Maidaoroda 2026-09-03), distilled
in `../katgpt-rs/.research/538_GDN_W4A4_Quantization_Survival_Mechanism.md`.
Its three survival conditions, audited against OUR stack:

| Condition (paper) | Our stack | Verdict |
|---|---|---|
| Gate parameters (`A_log`, `dt_bias`), `conv1d`, norms stay high-precision — post-activation noise on α is catastrophic (0.1% multiplicative → 22% state error) | All F32 (below) | **PASS** |
| Gate projections (`a`, `b`) quantized only as weight matrices; the softplus/exp+sigmoid log-space parameterization compresses ~11% GEMM error → ~2% output error | Ternary (Q2_0) weight matrices, pre-activation only | **PASS** |
| 1/√K query scale present (omission = spurious 1−1/√128 = 91.2% output difference with agreeing states — wrong-instrument signature) | `1/√head_dim` at the retrieve-output read, every variant | **PASS** |

Checkpoint: `../riir-train/data/Ternary-Bonsai-27B-Q2_0.gguf` (gguf v3, arch
`qwen35`, 851 tensors: TYPE_142 × 498, F32 × 353). TYPE_142 = our fork relabel
of Q2_0 ternary — `riir-infer-core/src/gguf_loader.rs` `GgmlType::from_id`:
`42 | 142 => Some(Self::Q2_0)` ("32B packed 2-bit codes per 128 weights …
fork-tip builds refuse id-42 Q2_0, so both ids map here").

## T2.1 — gate parameters / conv1d / norms unquantized: PASS

Header census across **all 64 blocks** (violation grep = empty):

| Tensor | Type | Shape |
|---|---|---|
| `blk.N.ssm_a` (A_log) | **F32** × 48 | [48] |
| `blk.N.ssm_dt.bias` | **F32** × 48 | [48] |
| `blk.N.ssm_conv1d.weight` | **F32** × 48 | [4 × 10240] |
| `blk.N.ssm_norm.weight` | **F32** × 48 | [128] |
| `blk.N.attn_norm` / `post_attention_norm` | **F32** × 128 | [5120] |
| `blk.N.attn_q_norm` / `attn_k_norm` (full-attn) | **F32** × 32 | [256] |
| `output_norm.weight` | **F32** | [5120] |

Loader side (both production paths): `ternary_deltanet_gpu_forward_cudarc.rs`
`upload_layer_weights_cudarc` uploads `a_log`/`dt_bias`/`conv1d_weight`/
`linear_norm`/norms as f32 slices (`CudaSlice<f32>`);
`qwen38_dense_cudarc.rs` `load_weights_gpu` uses `upload_f32` for the same set.
No path quantizes a post-activation gate parameter.

## T2.2 — gate projections quantized only as weight matrices: PASS

`ssm_alpha.weight` / `ssm_beta.weight`: 96 tensors, uniformly [5120 × 48],
TYPE_142 (ternary). They feed `in_proj_a`/`in_proj_b` quantized GEMVs →
`a_raw`/`b_raw` — the ONLY entry point of quantization noise, pre-activation.

Post-activation math is f32 on every execution surface, from the F32 gate
parameters:

```
beta[h]  = sigmoid(b_raw[h])
decay[h] = exp(a_log[h] * softplus_clamped(a_raw[h] + dt_bias[h]))   // ±20 clamp
```

- cudarc decode: `cudarc_kernels/deltanet.rs` `beta_decay_f32` (L96-116).
- cudarc batched rows (prefill/verify): `beta_decay_rows_f32` — "per (row, head)
  the ops are VERBATIM beta_decay_f32 (the a_log/dt_bias weights are shared)".
- cudarc backward: `beta_decay_backward_f32` + CPU-parity gate
  (`test_beta_decay_backward_matches_cpu`, 1e-5, green).
- cubecl (M3 + cubecl routes): `deltanet_beta_decay_f32` +
  `deltanet_beta_decay_batched_f32` — identical clamp + form; batched variant
  documented "bit-identical to p sequential launches".
- `qwen38_dense_cudarc.rs` (Q4_K/Q6_K dense lane): same `launch_beta_decay`
  family, gate params `upload_f32`.

The paper's shield is a property of the LOG-SPACE parameterization — recorded
for the future (paper §9 caveat): any GDN-2/KDA gate work with a
linearly-parameterized decay must re-check this whole table (Issue 879 T2.2
note).

## T2.3 — 1/√K query scale present: PASS

Every GDN recurrence variant applies the scale at the retrieve-output read —
mathematically the paper's query scale (a scalar on `S·q`), with K = head_dim
= 128 = state_size:

- cudarc: `recurrence_f32` (L273), row-parallel (L428), fused hd64 (L531),
  multi-token batched (L611), verify rows (L1064) — `dot * (1/√head_dim)`.
- cubecl: `deltanet_recurrence_f32` (L190), `_rowpar` (L391),
  `_multi_token` (L780) — identical.

The paper's wrong-instrument signature (states agreeing while outputs differ
by 1−1/√128) is structurally impossible here: the scale rides the readout only;
states are never touched by it. Additionally, the G1 cross-backend parity gates
(cudarc ↔ cubecl ↔ CPU, bit-identity/1e-2 class) would catch a missing scale
as a 128× output divergence.

## Honest scoping

- This is a STATIC audit (checkpoint + source), not the dynamic certification
  (Issue 879 T1: lockstep relS(t) plateau + impulse-decay measurement) — T1 is
  what certifies the recipe's headroom is real on OUR inputs; T2 certifies the
  recipe is correctly implemented. Both needed; T2 was the CPU-only half.
- `qwen38_dense_cudarc.rs`'s `upload_quant` accepts Q4_K/Q6_K only — it serves
  the Plan-556 dense lane, NOT the ternary Q2_0 checkpoint (that rides
  `ternary_deltanet_gpu_forward_cudarc.rs`). No conflict; recorded to prevent
  a future "why won't it load the ternary file" session.
- conv state, KV cache, and all activations are f32 end-to-end in the ternary
  forward (the KV f16 option is riir-ai Issue 753's opt-in lane, out of scope
  here — T3 covers the KV × weight-quant interaction separately).
