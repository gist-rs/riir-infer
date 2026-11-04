# Issue 1171 — qwen38 dense lane: the attention-core long-p census + the tensor-flash attention wiring

**Status:** LANDED 2026-11-04 (zed-amo-i1171-attn@amo, bench 942) — both
checkboxes done same session: the census pinned attention at **46.5–48.6%
of the pp5120 pass** (per-chunk 164→847 ms, ~1.5 TFLOPS-class); the
wiring (`RIIR_QWEN38_ATTN_TENSOR=1` promote-only, the Issue-1146 kernel
hosted on the dense context via the new `TensorAttnHost` trait — no
module fork) took the bucket to 1.10 s (20%), per-chunk near-flat
(89→132 ms, +5 ms/512-KV-rows vs +73 before). **pp5120: 580 → 971 tok/s
(1.67×) — PAST THE FORK (898) at its own league cell; MLX 1411 = 1.45×
remains.** Walk md5 `0764d040…` byte-identical across all 4 arms (7
runs). Record:
[`.benchmarks/942_issue1171_tensor_flash_attention_past_the_fork.md`](../.benchmarks/942_issue1171_tensor_flash_attention_past_the_fork.md).
Next ceiling: the GDN bucket (74% of the pass, flat ~400 ms/chunk) — the
layer GEMMs are at half-B 53-66 TFLOPS; the recurrence/conv/norm non-GEMM
share is the unmeasured rest (a per-kernel census inside the gdn bucket
is the next instrument, before any kernel work).

**Lease closed:** zed-amo-i1171-attn, epoch 1793820000 (claimed
2026-11-04 ~18:00Z for the census + wiring arc, folded at the bench-942
landing same day; was expires 2026-11-06T18:00:00Z)

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

- [x] **Census (first, per the bench-941 record):** per-chunk stage rows in
  the lane's timer mode (`chunk_rows` in `StageTimers` + one compact print
  line), then a pp5120 census run at the promoted posture (DENSE_MM2D +
  LMHEAD_Q6K + RELAXED + F16ACT): attention = 46.5–48.6% of the pass,
  per-chunk 164→847 ms, ~1.5 TFLOPS-class — LANDED (the census instrument
  stays in the timer mode; the before/after table is in bench 942).
- [x] **Wire (the census said material):**
  `MetalTensorAttentionPrefill` into `prefill_attn_layer` behind
  `RIIR_QWEN38_ATTN_TENSOR=1` (promote-only, exact literal) via the new
  `TensorAttnHost` trait (the dense context hosts; one tracker; the
  ternary call sites coerce unchanged); `routable()` fallback guard; the
  layer fns propagate errors loud — LANDED.
- [x] **G1:** the kernel's own tolerance gate re-run green (peaked
  argmax 12283/12288, long-cache 23996/24000); the lane walk is
  BYTE-IDENTICAL to all prior arms (md5 `0764d040…`, 7 runs) — the
  pre-registered argmax-stability read exceeded (zero divergence).
- [x] **G2/e2e:** pp512 659 / pp2560 945 / pp5120 971 vs bench 941's
  640/725/580; per-chunk census re-run (the share table above).
- [x] Record: [`.benchmarks/942_issue1171_tensor_flash_attention_past_the_fork.md`](../.benchmarks/942_issue1171_tensor_flash_attention_past_the_fork.md)
  + lease folded.

## Non-goals

- The decode attention path (the tensor arm is prefill-only by
  construction).
- The GDN bucket, the GEMM engine (benches 940/941 landed those).
- Any promotion call — owner-gated per the lossy-surface rule, GOAT reads
  this record.
