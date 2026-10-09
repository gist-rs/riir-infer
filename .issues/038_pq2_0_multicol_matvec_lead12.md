# Issue 038 — the n=3-8 decode band on Ada: adopt the dp4a multi-column mat-vec mechanism (distill Lead 12, prism #306)

**Status:** OPEN — staged distill lead (zero-cargo filing; the CUDA work is a 4090 lane, census-gated per the standing Plan-337 carve until ~2026-10-10 06:30 +07)

## The lead (provenance)

riir-refine distill envelope row 2026-10-09 ~23:5x (`89b19beb`), LEAD 12 (B+, kernel_opt):
PrismML-Eng/llama.cpp `2a42998c5` #306 "cuda: PQ2_0 mat-vec kernel for 3-8 columns on
Ada" — pin `8b0c19c8c` → `2a42998c5`. Measured on **Ternary-Bonsai-2-27B-PQ2_0** (OUR
league model file) on RTX 4070 (sm_89, same Ada class as the 4090):

- batched decode (llama-batched-bench): 4 seq **+17%**, 6 seq **+25%**, 8 seq **+29%**
- kernel cold-L2 (ncu), m=5120 k=17408: n=4 105.6→54.0 µs (**1.96×**), n=8
  136.8→78.1 µs (**1.75×**)
- perplexity byte: `-ub 4` chunk 6.3776→6.3820 — the accumulation-order drift face;
  any adoption here carries the lossy-watch note (per-family conditional retention,
  not aggregate pplx alone)
- tg128 single-seq (1 column) keeps the generic kernel upstream — the league's scored
  tg128 cell does NOT move from this PR; the decode razor (next tagged prism build)
  re-arms separately

## The gap in OUR kernel ladder

`crates/riir-infer-gpu/src/` (the 4090 CUDA arm, cudarc family):

| n (tokens per mat-vec call) | today | note |
|---|---|---|
| 1 | `gemv_ternary_cuda_raw.rs` (dp4a, Issue 608 T2) | the tg128 lane |
| **3–8** | **nothing dedicated — falls between GEMV and MMA** | spec-decode/tree-verify steps + batched decode live here |
| large | `gemm_ternary_i8_mma_cuda_raw.rs` (MMA) | prefill / big batches |

The tree-verify driver (`ternary_tree_verify_driver.rs`) and the batched-dispatch path
score multiple draft tokens per forward — exactly the 3–8 column shape the upstream
kernel fills. Upstream numbers say the generic mmvq loses DRAM throughput from 3
columns on (46% of peak at n=4, 36% at n=8 on Bonsai-2 shapes) — our MMA fallback at
this band pays the same class of underutilization.

## The mechanism atoms (unowned per the corpus dedup)

1. **Permuted activation layout** (`GGML_CUDA_Q8_1_PQ2`): per column, the qs bytes
   permuted inside each 16-element group (position `k*4+m` holds element `m*4+k`),
   then one `half2 (d, int16 sum)` per 32-block. With that layout
   `(code_word >> 2k) & 0x03030303` pairs RAW weight codes with activation bytes —
   **dp4a needs no per-weight decode** (one shift+mask yields four dp4a-ready bytes).
2. **Digit-bias one-subtraction**: `sum(q*(c-1)) = sum(q*c) − sum(q)` — the ternary
   offset bias rides the ACTIVATION's already-stored int sum; no per-weight (c−1)
   decode, no second pass.
3. **Shape-keyed admission**: Ada-only (`cc == GGML_CUDA_CC_ADA_LOVELACE`), plain 2D,
   no ids, not batch-invariant, n∈[3,8], per-ncols template instantiation 3..8; warp =
   4 rows sharing the activation slice; `__launch_bounds__(128, 3)` (register cap →
   3 blocks/SM).
4. **Measure-then-remove-switch discipline** (the #312 class): the env switch existed
   only to measure, removed in the same PR.

## The adoption task (when the 4090 lane is free)

- Port the mechanism to OUR packed-code weight format (the `upload_weights`
  bit-plane→Q2_0-code conversion feeds it directly — the code alphabet is the same
  {0,1,2} ternary family; our activation quant block is 16, upstream's 32 — the
  permutation is block-local, the port is the layout + the four-instruction inner
  product).
- Wire it as the n∈[3,8] dispatch arm between `gemv_ternary_cuda_raw` (n=1) and the
  MMA path; the tree-verify driver + batched decode are the first consumers.
- Gates: the existing task-level gates (Issue 608 T3a class — argmax agreement +
  top-k overlap) at the verify posture + a paired bench
  (`gemv_ternary_cuda_raw` vs new arm vs `gemm_ternary_i8_mma` at n∈{3..8}, the
  alternating-run pairing protocol per plan 612's G2 runbook); GOAT-promotion per the
  standing feature-flag discipline (opt-in first).
- NOT a substitute for the tg128 lane (n=1 unchanged) and NOT the
  prefill/MMA path's replacement.

## Why file now

The snapshot rows scroll; this lane has no standing trigger. The 4090 is currently
0.945× BEHIND on the league decode cell — the n=3-8 band is not that cell, but it is
the verify/batched-decode band the spec-decode lane needs, and the upstream evidence
(our model file, our arch) says the win is real. This issue is the pointer that
survives the snapshot.
