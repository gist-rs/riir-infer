# riir-infer Issue 1156 — Sparse prefill attention evaluation for the pp16k/pp32k league rows

**Status:** Open — POC/evaluation task; profile first; no serving-path change until a quality-gated arm wins its gate.

**Source:** riir-infer Research 019 (Elastic Attention, arXiv:2601.17367) Track A. Training-recipe half lives in riir-train Plan 476 (SECONDARY).

## Why

The league's weakest rows are long-context prefill: pp16k ≈ 0.268× / pp32k ≈ 0.272× vs llama.cpp (m5ultra, T5 re-cells) — despite the ktile64+zerocopy promotion winning pp2048/4096 (1.22×–1.25×) and holding 1.16×–1.22× internally at pp16k/32k. Published training-free sparse prefill attention (XAttention ICML 2025; FlexPrefill arXiv:2502.20766) claims 1.5–13× prefill speedup at 16K+ with block-sparse routing. If attention is a material share of hybrid prefill at 16K, a sparse arm is the one algorithm-class lever on these rows; everything else we have shipped is kernel/GEMM-class (GLU fusion, ktile64, dense mm2d).

⛔ **Order matters:** the dense long-context kernel must be priced first. If our 0.27× deficit is kernel-scaling-class (tiling/occupancy at large S) rather than algorithm-class, sparsity multiplies a losing base — profile BEFORE building any sparse arm.

## Tasks

- [ ] **T1 — Profile.** Decompose pp16k/pp32k prefill wall on the league posture: attention layers vs GDN chunked scan vs FFN GEMMs vs logits. Attention share ≥ ~25% is the go signal for T2; below it, close this issue with the profile as the negative record (the GDN-dominant arch may simply not have the surface area).
- [ ] **T2 — Training-free sparse arm (static posture first).** Block-sparse prefill attention on the attention layers only (XAttention-class antidiagonal block scoring or FlexPrefill-class per-head JSD probe, τ ≈ 0.9); streaming/sink pattern per the BSA config (sink 128, block 64). Static per-model posture — no router, no adaptivity in this arm.
- [ ] **T3 — Per-family ΩMSR sweep instrument.** Offline sweep t ∈ [0,1] over per-family holdouts (needle / QA / summarization / code / continuation-ppl at multiple K); record per-family ΩMSR (last t with retention ≥ 1−ε) into a digest-pinned manifest; planted-knee self-test both directions; per-family conditional retention on every subsequent gate run — never an aggregate-only read (the lossy-surface law; bench-889's continuation-ppl axis is the known killer).
- [ ] **T4 — Modelless regime gate (the elastic extraction).** Boundary-pooled hidden states (first/last 100 tokens, sigmoid — never softmax) → two frozen regime centroids → per-head/per-layer FA-vs-sparse bitmask. Consumes the T3 manifest's budgets. Only built if T2 lands; this is the gate the trained router (riir-train Plan 476) must beat.
- [ ] **T5 — Fused mixed-mode execution.** If T4 ships: per-head mode bitmask as kernel metadata, ONE launch, thread-block branching (BSA/FlexAttention pattern) — never serial head-split dispatch. G1 bitwise: one-launch mixed == union of single-mode runs; degenerate bitmasks byte-identical to the single-mode kernels.
- [ ] **T6 — League cells.** Sparse arm at its measured posture → pp16k/pp32k re-cells on both box classes, `bench_preflight` provenance lines, per-family quality floors from T3. Feature flag opt-in (implies nothing by default); promote only if the GOAT passes on BOTH perf and per-family quality.

## Gates (pre-registered)

- **G1 quality:** per-family conditional retention at the served posture — continuation-ppl band from the bench-889 four-arm record is the pass/fail axis (the axis static tokengate lost); needle/QA floors per family. Aggregate-ppl-flat is NOT a pass.
- **G2 perf:** prefill `--release`, both box classes, league shapes; ≥ the incumbent e2e at equal quality, or a pre-registered quality-for-speed trade accepted per family.
- **G3 byte-identity:** feature-off (and short-context bypass) byte-identical to the incumbent path.
- **G4:** zero-alloc hot path; no new serving deps.

## Notes

- GDN layers are already O(n) — the sparse arm targets the periodic full-attention layers only (~1/4 of layers in the qwen3.8 arch). Routing GDN layers is a category error (their axis is state precision/recall).
- Exact-parity league G1 pins do not apply to a lossy arm — this lane rides the Issue-750-T3 lossy law (opt-in, per-family retention, never default without the GOAT).
- Provenance: Research 019 §2 (Path 0 inventory + FlexPrefill/DuoAttention/InfLLM-v2 positioning).
