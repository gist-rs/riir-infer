# Issue 038 — the n=3-8 decode band on Ada: adopt the dp4a multi-column mat-vec mechanism (distill Lead 12, prism #306)

**Status:** OPEN — the port is LANDED (`48d598e`, 2026-10-10: kernel + handler + CPU-only NVRTC compile gate, `cargo clippy -p riir-infer-gpu --features ternary_gemv_cuda_raw --all-targets` clean); the GPU gates remain owed on a free 4090 lane (the plan437 T0.1 trainer holds the box — was step 2050/10000 at ~11 s/step when the port landed, ~24 h to completion at that rate; the 06:30 estimate below was the filer's, overtaken by the measured step time)

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

## Progress

- **2026-10-10 — PORT LANDED, `48d598e`** (`gemv_ternary_multicol_cuda_raw.rs`, +lib.rs decls):
  the four mechanism atoms above, ported to our format. Port deltas vs upstream: OUR
  16-element chunks (T3b accuracy posture — upstream quantizes per 32), separate
  `ascale`/`actsum` device buffers (not ggml's interleaved half2), host-side
  quantize+permute that mirrors `TernaryGemmCudaRaw::forward`'s host quantizer
  formula-identically. Warp structure per upstream: 4 rows/warp, all 32 lanes walking
  different chunks (act int4 loaded once per lane, reused across 4 rows in registers),
  clamped-row tight-warp trick, full-warp shfl reduce, `__launch_bounds__(128, 3)`.
  Weight encoding shared with the n=1 handler (`convert_bitplane_to_packed_codes`) —
  the same uploaded bytes feed either handler family.
  - **Bit-parity property** (asserted by `test_multicol_bit_parity_with_n1_kernel`):
    with identical host quantization the multicol output is BIT-IDENTICAL to the
    incumbent n=1 `forward` per token — the int dot is order-exact, the per-block
    float expression has the n=1 shape (`(float)sumi * ws * ascale`), the lane walk
    (stride 32 over 16-element chunks) and the full-warp reduce tree match. Upstream's
    accumulation-order drift face (their pplx byte 6.3776→6.3820) does NOT apply to
    this port by construction.
  - **Landed validation** (CPU-only, run beside the trainer): `cargo clippy -p
    riir-infer-gpu --features ternary_gemv_cuda_raw --all-targets` clean;
    `test_nvrtc_compiles_multicol_src` PASSES — the template/macro source compiles to
    sm_89 PTX with all six entrypoints present (NVRTC is a host-side compiler; no GPU
    touched).
  - **NOT production-wired** — opt-in by construction; nothing calls the handler yet.
- **2026-10-10 — FRESH-EYES REVIEW + CPU gates re-run (pickup session, trainer still holds the GPU):**
  - **Textual parity analysis: mechanism VERIFIED correct.** The permuted layout pairs element `4j+k` with code `4j+k` exactly (int4 `.x/.y/.z/.w` carry elements `{0,4,8,12}/{1,5,9,13}/{2,6,10,14}/{3,7,11,15}`; `t[k]` byte j is the code of element `4j+k` — all 16 covered once); the digit-bias identity `Σ(c−1)q = Σc·q − Σq` holds exactly in int; the accumulation order (lane-strided chunk walk), the per-block float expression shape (`(float)(sumi−qsum) * ws * ascale`, left-assoc — no fusable add, so FMA contraction cannot reorder it), the full-warp `shfl_down` tree, and the wscale/ascale per-chunk reads are all textually identical to `gemv_ternary_row`'s fast path; the uint32 code-word load is 4-byte-aligned BECAUSE `n % 16 == 0` is enforced at upload (that precondition is load-bearing for alignment, not just chunking — worth keeping in mind if anyone loosens it).
  - **Compile posture identical**: both NVRTC units compile with `arch: sm_89` + defaults (no fast-math on either side) — the reassociation parity risk is closed by construction.
  - **CPU-only gates now GREEN in this session's run** (isolated `CARGO_TARGET_DIR`, `-j 6`, GPU untouched): `test_quantize_permute_roundtrip_matches_n1_quantizer` PASS (first execution — the quantizer+permuter matches the n=1 formula element-exactly) + `test_nvrtc_compiles_multicol_src` PASS (re-confirmed). Owed CPU gates: none remaining.
  - **Coverage gap found (fix at the owed GPU run, before the bench):** the parity loop covers tokens {3,4,8} only — entrypoints **n5/n6/n7 compile but never execute in any test**. Extend the parity loop to `[3,4,5,6,7,8]` (seconds of GPU) so every entrypoint executes at least once before wiring. Also noted (non-blocking): `n % 128 != 0` shapes (e.g. n=144) are admitted by the `% 16` precondition and are untested — group_scale padding is consistent by the type's own convention (`cpu_matvec` indexes the same), and production dims are multiples of 128; add one such shape to the tolerance test only if cheap.

### Owed on a free 4090 lane (in order)

0. ~~Extend the parity test loop to tokens `[3,4,5,6,7,8]`~~ — **DONE CPU-side
   2026-10-10** (the loop now spans the full band; execution still owed to the GPU run).
0.5. **STAGED 2026-10-11 (~09:5x–10:2x, pickup session, trainer still holds the box
   — plan437 `stage0_eval` AR arm over 200M tokens):** both GPU-run instruments
   pre-built CPU-side in an isolated `CARGO_TARGET_DIR` so the free window is
   spent MEASURING, not compiling — (a) the unit-gate lib-test binary
   (`ternary_gemv_cuda_raw` posture) and (b) **the step-3 paired bench LANDED**:
   `crates/riir-infer-gpu/examples/multicol_ab.rs` + its `[[example]]` row
   (`required-features = ["ternary_gemv_cuda_raw"]`), clippy-clean, release
   binary built. Three arms at the handler contract — `n1_loop` (n sequential
   `forward` calls) vs `multicol` (one `forward_multicol`) vs `mma` (the
   shipping prefill `launch_prefill_quantize` + `launch_prefill_gemm` pair,
   upload→quantize→gemm→download) — Bonsai-2 down_proj (5120×17408) + a
   sanity-small shape, n∈[3..8], reps order-rotated per rep (plan-612 G2
   adaptation), verdict = median same-rep paired ratios. ⚠ The numbers are
   HANDLER-contract wall times (each arm's public API today: the GEMV arms'
   host quantize + per-call transfers are INSIDE the timed path; the wired
   GPU-resident A/B is step 4's GOAT surface). MMA env knobs unset at run
   time = the plain bitplane shipping call. Run:
   `cargo run --release -p riir-infer-gpu --features ternary_gemv_cuda_raw
   --example multicol_ab` (quote bench-preflight-class provenance in the
   record). Step 2 (task-level argmax gate) correctly sequences AFTER the
   unit gates run green — its design rides the real-weight forward harness;
   the n=1 kernel's own T3a is likewise open (parallel gap, noted).
1. Unit gates: `cargo test -p riir-infer-gpu --features ternary_gemv_cuda_raw --lib
   gemv_ternary_multicol` — bit-parity vs n=1, CPU-reference tolerance (mean_rel < 2%,
   max_rel < 5%), shape rejections, quantizer round-trip (round-trip + nvrtc already
   green CPU-side 2026-10-10). A parity mismatch = FMA
   contraction divergence between the two NVRTC units: fix the expression shape, never
   weaken the assert.
2. Task-level argmax agreement at the verify posture (Issue 608 T3a class).
3. Paired bench: n∈{3..8}, `gemv_ternary_cuda_raw` vs this arm vs
   `gemm_ternary_i8_mma_cuda_raw` (plan 612 G2 alternating-run pairing runbook),
   GPU-exclusive + `scripts/bench_preflight`-class box state.
4. Wire the dispatch arm (tree-verify driver + batched decode, n∈[3,8]) behind the
   standing kill-switch pattern; GOAT-promote per the feature-flag discipline.
