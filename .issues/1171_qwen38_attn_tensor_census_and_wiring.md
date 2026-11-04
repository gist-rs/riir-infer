# Issue 1171 — qwen38 dense lane: the attention-core long-p census + the tensor-flash attention wiring

**Status:** IN PROGRESS — the bench-941 record's named lead ("the attention
core's O(p²) share at long p — census at pp5120 before any kernel work").

**Claimed:** zed-amo-i1171-attn, 1793820000, expires 2026-11-06T18:00:00Z

## Problem

Bench 941 put the lane at 580 tok/s @pp5120 vs the fork's 898 (1.55×) —
the residual widens with position, and the suspected sink is the attention
core: `QwenAttentionPrefillTiledCubeCL` runs QK/PV as per-lane dot chains
(no matrix units), while the GEMMs now sit at 53–76 TFLOPS. The attn
bucket @p512 (one chunk, kv ≤ 512) is 143–162 ms of a 1.0–1.3 s pass; at
pp5120 the last chunk attends to 5120 KV — ~10× the score volume of chunk
0 — and the bucket grows superlinearly. The estimate says attention could
be ~40% of an 8.8 s pp5120 pass at ~3 TFLOPS-class efficiency. Neither the
share nor the efficiency is measured — the census comes first (the record's
own instruction), the wiring second.

## Substrate (substrate-first: it EXISTS)

`attention_prefill_metal_tensor.rs` — the Issue-1146 ladder-4(c) matmul2d
flash-attention prefill: causal + base_pos, GQA, 1/sqrt(256) scale,
σ(gate) epilogue, **the exact handle contract of the tiled CubeCL arm**
(query/gate/out `[p][n_head][256]`, key/value `[base_pos + p][n_kv][256]`,
f32, HEAD_DIM 256 — this model's head_dim IS 256). Promoted dist-scoped
(`RIIR_DIST_PREFILL_ATTN_TENSOR` default-on there, `=0` demotes);
`routable()` guards the shape class. Tolerance arm (Q/P half, scores/O
f32, K/V f32 from cache) — G1 is a tolerance gate + per-family KL read
(`tests/issue1146_attn_tensor_g1.rs`), NOT bit-identical, the
bench-923/940/941 lossy discipline applies.

## Plan

- [ ] **Census (first, per the bench-941 record):** per-chunk stage rows in
  the lane's timer mode (`chunk_rows` in `StageTimers` + one compact print
  line), then a pp5120 census run at the promoted posture (DENSE_MM2D +
  LMHEAD_Q6K + RELAXED + F16ACT): per-chunk attn bucket growth curve → the
  attention core's share + implied TFLOPS at each position.
- [ ] **Wire (only if the census says the share is material):**
  `MetalTensorAttentionPrefill` into `prefill_attn_layer` behind
  `RIIR_QWEN38_ATTN_TENSOR=1` (promote-only, exact literal; construction
  needs a `MetalTensorZeroCopyGemm` — construct on the same device the
  dense context holds, or extend the module to accept device/queue
  pointers directly; DRY: one construction site, no fork of the module).
  Fallback to the tiled CubeCL arm when `!routable` (base_pos + p < 64 —
  never hit at p=512 chunks, but the guard stays).
- [ ] **G1:** the existing `issue1146_attn_tensor_g1` covers the kernel;
  the lane-level arbiter is the walk (lossy class: the f32→half Q/P
  rounding — expect argmax-stable, NOT byte-identical; pre-register the
  read: id-agreement + divergence point on 512 tokens vs the bench-941
  walk md5 `0764d040…`).
- [ ] **G2/e2e:** pp512 / pp2560 / pp5120 cells vs bench 941's
  640/725/580; per-chunk census again for the share shift.
- [ ] Record (bench by highwater) + fold the lease.

## Non-goals

- The decode attention path (the tensor arm is prefill-only by
  construction).
- The GDN bucket, the GEMM engine (benches 940/941 landed those).
- Any promotion call — owner-gated per the lossy-surface rule, GOAT reads
  this record.
