# Issue 1164: the T5 mixed-mode head mask is VACUOUS at the model level (bench 1157 Cell D)

**Status:** Open — filed from bench 1157 (the T6 league cells, 2026-11-02)
**Claimed:** zed-7710a782, 1793682000, expires 2026-11-04T16:30:00+00:00

## The finding

Bench 1157 Cell D (the runbook's example mixed-mode posture: `RIIR_PREFILL_SPARSE_DENSE_HEADS` = heads 0..=4 dense, τ=0.85 on the rest) produced logits **BIT-IDENTICAL to fully-dense** (FNV 6/6 rounds at BOTH 16K and 32K) at a ~1.00× wall — i.e. the cell measured DENSE, not mixed. If any head beyond 0..=4 had actually run sparse at τ=0.85, the logits would differ from dense exactly as Cell C's do (FNV 0/6). They do not.

The T5 kernel-level G1 remains green (`mixed_mode_equals_the_union_of_single_mode_runs`, `all_dense_mode_ignores_the_mask_and_is_bit_identical_to_m32`) — so the defect class is ABOVE the kernel:

- (a) the whole-prefill batched launcher's mask path never consumes the stored mask (a store-vs-consume seam the single-layer T5 harness doesn't reach), or
- (b) the mask's head indexing doesn't match this model's attention-head layout at the model level (per-layer n_head vs the global index space), or
- (c) the example `0-4` range covers every head the model's attention layers expose (n_head ≤ 5 per attention layer — implausible for the 27B, but unverified in the cell).

## The evidence

`.benchmarks/1157_issue1156_t6_league_cells.md` (riir-ai): Cell D rows — FNV==dense 3/3 @16K + 3/3 @32K, walls ≈ dense walls (63.9–66.3 s @16K vs dense 64.6–66.4 s; 158.9–161.7 s @32K vs dense 158.0–161.5 s), while Cell C (plain τ=0.85, no mask) shows FNV 0/6 with walls −3.5%…−6.5%. Counters moved (the sparse pipeline ran — pool/score/select Δ as C's), so the ARM armed; the per-head MODE WORD is what did not differentiate.

## The fix shape

Reproduce at the model level first: a fixture that runs ONE model prefill with mask 0 vs mask all-ones and asserts the two differ from the unmasked sparse run (the inverse of the T5 anchor: `all_dense_mask ≡ dense` should hold, `partial_mask` must sit BETWEEN). Then walk the launcher path (`prefill_attention_layer_batched` → the T5 mode-word buffer build) and find where the stored mask is (not) consumed. The T5 single-layer harness builds its own mode buffer — that's the seam that bypassed the bug.

## Impact

- Cell D of the T6 runbook is UNMEASURED (not report-only-measured) for mixed-mode perf — the runbook's "REPORT-ONLY perf" verdict for mixed-mode stands but carries NO number until this lands.
- No serving or promotion surface is affected: mixed-mode has no default path and the T6 promotion lane (Cell C) is independent of it.
