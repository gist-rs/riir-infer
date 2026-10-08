# Issue 036: LittleBit-derived init-only sub-1-bit PTQ lane (`svd_lbit`) + ±1 sandwich kernel survey

**Status:** Open — POC, gated; derived from [riir-infer Research 009](../.research/009_LittleBit_Sub1Bit_Factorized_Binarized_Quant.md) (arXiv:2506.13771)

**Date:** 2026-10-08
**Owner:** riir-infer (public substrate; CPU-first, no GPU exclusivity needed)
**Related:** riir-train Issue 620 (the training-recipe half of the same paper)

## Goal

Measure the paper's unmeasured point: **Dual-SVID init-only (no QAT) factorized-binarized PTQ quality** at {1.0, 0.55, 0.3, 0.1} BPW on one small checkpoint, and decide GO/NO-GO on a sub-1-bit format + kernel lane. The paper's skeleton (SVD → sign factors → rank-1 magnitude-scale init → k-stage residual restack) is closed-form linear algebra; every PPL it reports is a QAT output, so the training-free point is genuinely unmeasured in the literature.

The verdict bar is **"dominates the training-free curve at matched BPW"** (RTN-binary; ternary containers at native BPW for context) — never "matches LittleBit".

## Tasks

- [x] **T0** Name the eval host BEFORE T1 starts — this repo has no perplexity harness today (the only `ppl` hit in `src/quant/` is a doc comment quoting the fork's own claim). Decide: reuse an existing riir-train/katgpt-rs perplexity loop, or stand up a minimal PPL loop here (WikiText-2, the small checkpoint's tokenizer); record the choice + cost in this issue before any transform code lands
  **DECIDED 2026-10-08 (before any transform code): stand up a minimal PPL loop HERE** (a measurement bin over existing substrate — nothing new under the hood):
  - **Census:** riir-train HAS `evaluate_perplexity` but it is example-internal (`bonsai_clippy_l4_sft_train.rs`), hard-bound to `QwenDeltaNetTernaryWeights` (Bonsai GDN arch), and a cross-repo dep is boundary-forbidden here anyway; katgpt-rs has no token-PPL loop over GGUF checkpoints. riir-infer already ships every PRIMITIVE the loop needs: `forward_gemma2_f16` (per-token causal forward returning logits), `vk_harness::nll` (f64 logsumexp), `SentencePieceGgufTokenizer`, `corpus_text::load_corpus_text`. The bin is glue, not a harness.
  - **Checkpoint: `gemma-2-2b-it-f16.gguf`** (2.6B — the on-disk house fixture, `act_ptq_gemma2` precedent; verified present in riir-train/data 2026-10-08). **Deviation from the ≤1.3B-class hint, recorded honestly:** MiniCPM5-1B is NOT on disk and its arch is unsupported by this loader; Bonsai-27B is GDN-ternary (wrong shape class, already ~2 bpw); gemma-2-2b is the only dense checkpoint with loader+tokenizer+forward support here. The kill criterion is unaffected: it compares against baselines computed ON THE SAME checkpoint, never against published tables.
  - **Corpus: WikiText-2 raw test**, fetched via the HF datasets-server (reflex `fetch_datasets.sh` pattern), digest-pinned; eval on a FIXED seeded doc subset (~32-48 docs ≈ 35-55k tokens — the paper's full 280k-token read is unnecessary for GO/NO-GO deltas of this size; recorded as the subset size + digest in the bench).
  - **SVD substrate decision (substrate-first, recorded):** katgpt-core's `thin_svd_into` (one-sided Jacobi, public, deterministic) is the RIGHT tool for its design point (≤~64-dim Jacobians) but 10⁴× too slow at gemma2 dims (min-dim 2304, ranks 90-1000: cyclic Jacobi is O(n²) sweeps × 100 GFLOP+ per matrix × 182 matrices). T1 therefore ships a **seeded deterministic randomized (Halko) truncated SVD** in this repo: `fastrand` (allowlisted) fixed-seed Gaussian sketch, q=1 power iteration, oversampling +16; ONE k_max factorization per layer serves all 4 BPW targets AND the plain-SVD baseline (same basis, truncated — internally consistent, disclosed in the bench). Exact per-r SVD stays a documented approximation class, never claimed.
  - **Baselines (computed locally, matched BPW):** plain truncated-SVD low-rank with f16 factors (the SVD-LLM/ASVD class — bits priced as 2·16·r·(dout+din)/(dout·din), rank solved at matched b) + RTN-binary (sign(W) + per-row/col fp16 scales — its floor is ~1.03 bpw so it arms ONLY the 1.0 target) + the FP16 anchor.
  - **Arms (14 PPL passes):** anchor 1 + svd_lbit 1-stage × 4 targets + svd_lbit 2-stage restack × 4 + RTN-binary @1.0 + plain-SVD × 4. Cost: transform ≈ 2-5 min/arm-set (rayon over layers, allowlisted); PPL ≈ rayon-parallel chunked forward, minutes-to-tens-of-minutes per pass on the M3 (loaded box disclosed via PROVENANCE-style note). Kill criterion unchanged (T3).
- [ ] **T1** `quant/svd_lbit.rs`: deterministic transform — truncated SVD → `sign` factors → rank-1-SVD magnitude-scale init (`|U′| ≈ h₀ℓᵤᵀ`, `|V′| ≈ g₀ℓᵥᵀ`, `ℓ₀ = ℓᵤ⊙ℓᵥ`) → optional `--stages k` residual restack (Prop 2 separate-vs-joint law) + BPW budget planner (`rank_for_bpw`/`bpw_for_rank`, pinned vs Appendix D examples: r=546 @0.55/4096², r=133 @0.1/4096×11008)
- [ ] **T2** Determinism gate: same-layer conversion twice locally + once cross-box → byte-identical packed output (pin a fixed-order one-sided Jacobi if BLAS tie-breaking breaks identity); quantize-once-commit-bytes discipline — serving never re-derives
- [ ] **T3** Measurement: init-only PPL ladder on one ≤1.3B-class model at 4 BPW targets vs training-free baselines; record the 0.3→0.1 cliff shape. **Kill criterion (decide the negative before running):** if init-only PPL at a given BPW is worse than the published training-free curve at matched BPW (SVD-LLM/ASVD-class low-rank PTQ; RTN-binary floor) on the same checkpoint, that BPW point is NO-GO; NO-GO at ≥2 of the 4 targets closes the issue as a measured negative
- [ ] **T4** ±1 sandwich kernel survey (no lane until T3 says GO). **Precondition: pin the Samsung repo first** — clone `SamsungLabs/LittleBit` to `.raw/`, record its HEAD sha + license in this issue, read their 1-bit GEMV kernel for the popcount/XNOR shape, then `rm -rf .raw/` per the `.raw` hygiene law. Survey: popcount-XNOR exact integer dots on NEON + x86_64, ns/call vs the ternary LUT path at half the weight bytes; arch-gated arms follow the x86_64 execution-matrix discipline
- [ ] **T5** Verdict + doc: GO → plan the format+kernel lane (feature `lbit`, kill-switch `LBIT_SANDWICH=0`, opt-in posture per the lossy-surface law — never default, per-family conditional retention gate before any promotion); NO-GO → close with the measured numbers (a measured negative is still the deterministic-initializer record for any future QAT run)

## Constraints

- Lossy-surface law: opt-in only, never league-model involvement.
- CPU-first (M3); the 4090 is not needed for T1–T3.
- Zero `riir-*` deps (boundary law); SVD via an allowlisted linalg dep only — check `BOUNDARY.md` allowlist before adding.
