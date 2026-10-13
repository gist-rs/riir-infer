# Doc 003 — the PQ2_0 multi-column mat-vec arm (dp4a, n=3–8 band): LANDED, validated, on STANDBY

**Status:** RECORD (standby arm — landed + opt-in + zero production risk; wiring awaits a
future cudarc batched-decode lane, an owner-scoped project). Supersedes
`.issues/038_pq2_0_multicol_matvec_lead12.md` (closed 2026-10-13; the 039 companion —
repro/A-B legs — closed 2026-10-12, file already removed).

## Provenance

riir-refine distill Lead 12 (B+, kernel_opt), envelope row `89b19beb` 2026-10-09:
PrismML-Eng/llama.cpp `2a42998c5` #306 "cuda: PQ2_0 mat-vec kernel for 3-8 columns on Ada",
measured on **Ternary-Bonsai-2-27B-PQ2_0** (our league model) on RTX 4070 (sm_89, same Ada
class as the 4090): batched decode +17%/+25%/+29% at 4/6/8 seqs; cold-L2 ncu m=5120 k=17408:
n=4 105.6→54.0 µs (1.96×), n=8 136.8→78.1 µs (1.75×). Upstream pplx drift byte
(6.3776→6.3820 at `-ub 4`) is THEIR accumulation-order face — our port is bit-identical to
the n=1 kernel by construction (below), so that face does not apply here.

## The landed arm

`crates/riir-infer-gpu/src/gemv_ternary_multicol_cuda_raw.rs` (port commit `48d598e`,
2026-10-10) behind `ternary_gemv_cuda_raw`, **opt-in by construction — nothing calls the
handler in production today**. Mechanism atoms (per the corpus dedup, unowned before):

1. **Permuted activation layout**: per column, qs bytes permuted inside each 16-element
   group (position `k*4+m` holds element `m*4+k`) so `(code_word >> 2k) & 0x03030303`
   pairs raw weight codes with activation bytes — dp4a with no per-weight decode.
2. **Digit-bias one-subtraction**: `sum(q*(c-1)) = sum(q*c) - sum(q)` — the ternary offset
   rides the activation's stored int sum.
3. **Shape-keyed admission**: Ada-only, plain 2D, n∈[3,8], per-ncols template
   instantiation 3..8; 4 rows/warp sharing the activation slice; `__launch_bounds__(128, 3)`.
4. Port deltas vs upstream: OUR 16-element chunks (T3b posture), separate
   `ascale`/`actsum` device buffers, host quantize+permute formula-identical to
   `TernaryGemmCudaRaw::forward`'s host quantizer. Weight encoding shared with the n=1
   handler (`convert_bitplane_to_packed_codes`) — same uploaded bytes feed either family.

## Validation record (all green, 2026-10-11 → 10-13)

- **Unit gates 5/5** (worktree `04a4f98`), incl. `test_multicol_bit_parity_with_n1_kernel`
  (bit-IDENTICAL to the incumbent n=1 `forward` per token — the int dot is order-exact;
  upstream's drift face does not transfer) and the roundtrip test pinning the quantizer
  formula. First-run failures were harness defects (seed-luck-fragile float oracle fixed by
  the deterministic `x_q = round(x/d)·d` reconstruction; a shape-rejection probe buffer),
  not kernel defects.
- **Real-weight task gate 6/6 in 13 s** (worktree `33d2ef6`):
  `test_multicol_real_weights_bit_parity_and_argmax` — mmap partial load of the real league
  GGUF, `ffn_down` of layers 0/32/63 [5120×17408], tokens {3,5,8}: bit-parity (`to_bits`)
  + per-token argmax agreement at every layer×token. Swapping n1→multicol changes no task
  output.
- **Bench 035** (`.benchmarks/035_multicol_ab_n3_8_band.md`, executed 2026-10-13,
  GPU-exclusive, provenance in the record): full 12-cell grid; multicol beats the n1
  incumbent at EVERY cell (median paired ratio 0.971 @ big n=3 → 0.766 @ n=8; 0.707 → 0.467
  small, monotone in n; per-cell 1.03×–2.14×). The MMA arm's handler contract dominates
  both GEMV arms (2.0–3.7×) — handler-contract wall times include host quantize +
  per-call transfers for the GEMV arms; the wired GPU-resident A/B is the future GOAT
  surface.

## Why it is on standby (the step-4 consumer mapping)

The two named consumers do not exist as call sites today: (a) the tree-verify driver
(`ternary_tree_verify_driver`) is the CUBECL lane and already batches T-row projections via
`GemmTernaryBatchedCubeCL`; (b) the cudarc lane's spec verify (`spec_verify_dispatch`,
`SPEC_MAX_K=8`) processes the draft chain SEQUENTIALLY — each forward conditions on the
previous one's state (autoregressive), correctly NOT batchable; (c) a batched MULTI-SEQUENCE
cudarc decode forward (the shape that actually produces n∈[3,8] columns against one weight —
upstream's +17/25/+29% was llama-batched-bench multi-sequence) does not exist on this lane.
**Disposition: landed + opt-in + standby, zero production risk.** The league's scored cell
is tg128 single-seq (n=1) — deliberately untouched by this arm.

## Re-open trigger

A future **cudarc batched-decode lane** (multi-sequence decode forward) or a cudarc
tree-verify — its own owner-scoped project. That project's plan wires the n∈[3,8] dispatch
arm between `gemv_ternary_cuda_raw` (n=1) and the MMA path, behind the standing
kill-switch pattern, with the wired GPU-resident paired A/B as the GOAT decider
(feature-flag discipline: opt-in first, promote on gates).

## Preserved follow-up notes (not this lane's blockers)

- **The n=1 module's own G1** (`test_dp4a_matches_cpu_within_int8_tolerance`) carries the
  same seed-luck-fragile float oracle denominator 038's gate had (passes today because
  `0xCAFEBABE` produced no cancellation). The one-line deterministic-oracle fix
  (reconstruct `x_q = round(x/d)·d` per 16-block) applies when the Issue-608 lane touches
  it next.
- **`n % 128 != 0` shapes** (e.g. n=144) are admitted by the `% 16` upload precondition and
  untested; production dims are multiples of 128; group_scale padding is consistent by the
  type's own convention (`cpu_matvec` indexes the same). Add one such shape to the
  tolerance test only if cheap.
- The `n % 16 == 0` upload precondition is LOAD-BEARING for uint32 code-word 4-byte
  ALIGNMENT, not just chunking — keep it in mind before loosening.
