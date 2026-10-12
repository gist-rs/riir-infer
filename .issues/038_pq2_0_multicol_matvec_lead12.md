# Issue 038 — the n=3-8 decode band on Ada: adopt the dp4a multi-column mat-vec mechanism (distill Lead 12, prism #306)

**Status:** OPEN — port LANDED (`48d598e`); steps 1+2 GREEN 2026-10-11; **step-3 bench EXECUTED 2026-10-13 ([bench 035](../.benchmarks/035_multicol_ab_n3_8_band.md))** — multicol beats the n1 incumbent at EVERY cell (median paired ratio 0.971 @ big n=3 → 0.766 @ big n=8; 0.707 → 0.467 small, monotone in n); the MMA arm's handler contract dominates both GEMV arms (2.0–3.7×) — wired GPU-resident A/B stays the GOAT decider. REMAINING: step-4 dispatch wiring (the 039 repro/A-B legs CLOSED 2026-10-12 — see 039)

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
   pre-built CPU-side in an isolated `CARGO_TARGET_DIR` (kept at `E:\tmp\infer038`;
   rm it after the window run) so the free window is spent MEASURING, not
   compiling — (a) the unit-gate lib-test binary
   (`ternary_gemv_cuda_raw` posture; run with
   `CARGO_TARGET_DIR=E:\tmp\infer038 cargo test -p riir-infer-gpu --features
   ternary_gemv_cuda_raw --lib gemv_ternary_multicol`) and (b) **the step-3
   paired bench LANDED**:
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
0.6. **WORKTREE ISOLATION (2026-10-11, staged):** the main riir-infer worktree
   now carries sibling WIP (24 dirty files incl. `lib.rs` +
   `test_gpu_support.rs`); ALL window runs execute from the isolated worktree
   `E:\git\wt038infer` (detached at the verified tip `0767c01`, created by the
   039 staging note — same window serves both issues). All three instruments
   re-warm-built from it CPU-side 2026-10-11 (compile time paid; the window
   spends measuring): unit-gate binary (5 multicol tests listed), the
   `multicol_ab` release example, and the 039 edlm repro binary. Run shape:
   `CARGO_TARGET_DIR=E:\tmp\infer038 cargo test --manifest-path
   E:/git/wt038infer/Cargo.toml -p riir-infer-gpu --features
   ternary_gemv_cuda_raw --lib gemv_ternary_multicol` (the bench likewise with
   `build --release --example multicol_ab`). Cleanup rides the 039 note.
0.7. **STEP-2 GATE AUTHORED (2026-10-11 ~18:0x–18:4x, pickup session, trainer
   still holds the box):** the T3a-class task gate now EXISTS — worktree commit
   `f1e5319` (on `0767c01`): `test_multicol_real_weights_bit_parity_and_argmax`
   in the multicol module, `#[ignore]`d (needs BONSAI_GGUF + exclusive GPU).
   Design: loads the real league GGUF, samples down_proj of layers 0/mid/last
   (real group scales + real ternary patterns — the classes synthetic weights
   hide), runs n=1 loop vs `forward_multicol` at tokens {3,5,8}, asserts
   per-element `to_bits` parity + per-token argmax agreement. Scope note: the
   argmax-vs-CPU-float-reference question is the n1 family's own open T3a
   (the dp4a int8-activation class); THIS gate closes 038's task-level claim —
   swapping n1→multicol changes no task output. Clippy-clean + fmt-clean at
   the posture (the clippy "1 warning" per unit is cargo's hard-link
   diagnostic — E: does not hard-link; zero real lints). Pre-built BOTH
   postures CPU-side: debug unit-gate binary (rebuilt with the gate) + the
   RELEASE test binary `riir_infer_gpu-d576543fa79350cc.exe` (the 27B load
   is minutes in release vs far worse in debug — run the gate at
   `cargo test --release ... -- --ignored`). Window sequence is now:
   unit gates → real-weight gate (release) → `multicol_ab` bench → the 039
   repro/A-B legs. The lib code is unchanged (tests-only edit) — the staged
   `multicol_ab` example + the edlm repro binary stay fingerprint-valid.

1. Unit gates: ~~`cargo test -p riir-infer-gpu --features ternary_gemv_cuda_raw --lib
   gemv_ternary_multicol`~~ — **DONE 2026-10-11 20:3x–20:5x, run beside the trainer**
   (safe: deterministic correctness gates, ~2 GB VRAM vs ~10 free; SM sharing
   changes latency, never results — and bit-parity would break under any
   real corruption): **5/5 GREEN** (worktree `04a4f98`). The FIRST run
   found 2 failures — both **harness defects, not kernel defects** (the
   diagnostic chain is the commit message): (a) G1's float-input CPU
   reference is seed-luck-fragile — a token whose true dot cancels to ~4e-5
   reads mean_rel 1.1e2 against ~1.7e-2 of LEGITIMATE int8 activation
   noise (max_rel 445 = exactly 0.0168/0.000038); fixed by making the
   oracle deterministic: reconstruct x_q = round(x/d)·d per 16-block (the
   quantizer formula the roundtrip test already pins) and gate against
   THAT — measured mean_rel **3.2e-6**, max_rel 1.2e-5 (float-association
   class; 10⁴× tighter than the 2% bar — a STRONGER gate, not a weakened
   one). (b) the shape-rejection probe's `out` buffer was 2·m where the
   output-mismatch arm slices [..4·m−1] — the probe panicked on its own
   slice before reaching the handler; buffer now 4·m. Bit-parity with the
   n=1 kernel held through every run (all tokens 3–8). ⚠ Follow-up noted
   for the Issue-608 lane: the n=1 module's own G1
   (`test_dp4a_matches_cpu_within_int8_tolerance`) carries the SAME
   seed-luck denominator and passes only because 0xCAFEBABE produced no
   cancellation — same one-line oracle fix applies when that lane touches
   it next.
2. Task-level argmax agreement at the verify posture (Issue 608 T3a class):
   ~~**DONE 2026-10-11 ~21:0x–21:2x, run beside the trainer**~~
   (worktree `33d2ef6`) — the gate is `test_multicol_real_weights_bit_parity_and_argmax`,
   now running by DEFAULT in the filtered suite (no `#[ignore]`; skip-loud
   without the GGUF or CUDA — never a green zero, and that skip-green was
   caught live: the first v2 run "passed" in 0.5 s because the
   `../../riir-train` default resolves one level short of the sibling — a
   dq-bin convention the env override always papered over; now `../../../`).
   Design v2 (mmap partial load): `GgufFile::open` parses the header only;
   the three sampled `ffn_down` tensors (0/mid/last, derived from ACTUAL
   tensor presence — never `block_count` metadata, which counts MTP layers)
   are repacked straight from the lazy mmap — ~0.5 GB touched, not the
   full-loader's ~7 GB, so the gate runs BESIDE a trainer safely. **Measured
   on the real league file**: 64 layers, layers 0/32/63 [5120×17408], tokens
   {3,5,8}: bit-parity (`to_bits`) + per-token argmax agreement hold at
   EVERY layer×token combination; both arms consume the identical repacked
   tensor, so the checkpoint's hadamard-fold state is irrelevant to the
   claim. Suite: **6/6 GREEN in 13 s** (worktree `33d2ef6`). Scope note:
   argmax-vs-CPU-float-reference is the n1 family's own open T3a; this gate
   closes 038's claim — swapping n1→multicol changes no task output.
3. Paired bench: n∈{3..8}, `gemv_ternary_cuda_raw` vs this arm vs
   `gemm_ternary_i8_mma_cuda_raw` (plan 612 G2 alternating-run pairing runbook),
   GPU-exclusive + `scripts/bench_preflight`-class box state.
   ~~pending~~ — **DONE 2026-10-13 (the staged free window; [bench 035](../.benchmarks/035_multicol_ab_n3_8_band.md)):**
   full 12-cell grid green, verdict above — mc/n1 wins everywhere (1.03×–2.14×),
   mc/mma loses everywhere at the handler contract (the doc's own caveat: not a wired
   comparison). Provenance quoted in the record (GPU-exclusive: zero CUDA compute
   processes; CPU 3%; 18.8 GiB free; Tailscale; fresh release rebuild 2m15s).
4. Wire the dispatch arm (tree-verify driver + batched decode, n∈[3,8]) behind the
   standing kill-switch pattern; GOAT-promote per the feature-flag discipline.
