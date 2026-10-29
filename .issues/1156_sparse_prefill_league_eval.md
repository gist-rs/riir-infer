# riir-infer Issue 1156 — Sparse prefill attention evaluation for the pp16k/pp32k league rows

**Status:** Open — POC/evaluation task; profile first; no serving-path change until a quality-gated arm wins its gate.
**Claimed:** zed-amo-i1156, 1793294557, expires 2026-10-31T17:22:37+00:00

**Source:** riir-infer Research 019 (Elastic Attention, arXiv:2601.17367) Track A. Training-recipe half lives in riir-train Plan 476 (SECONDARY).

## Why

The league's weakest rows are long-context prefill: pp16k ≈ 0.268× / pp32k ≈ 0.272× vs llama.cpp (m5ultra, T5 re-cells) — despite the ktile64+zerocopy promotion winning pp2048/4096 (1.22×–1.25×) and holding 1.16×–1.22× internally at pp16k/32k. Published training-free sparse prefill attention (XAttention ICML 2025; FlexPrefill arXiv:2502.20766) claims 1.5–13× prefill speedup at 16K+ with block-sparse routing. If attention is a material share of hybrid prefill at 16K, a sparse arm is the one algorithm-class lever on these rows; everything else we have shipped is kernel/GEMM-class (GLU fusion, ktile64, dense mm2d).

⛔ **Order matters:** the dense long-context kernel must be priced first. If our 0.27× deficit is kernel-scaling-class (tiling/occupancy at large S) rather than algorithm-class, sparsity multiplies a losing base — profile BEFORE building any sparse arm.

## Tasks

## Tasks

- [x] **T1 — Profile.** Decompose pp16k/pp32k prefill wall on the league posture: attention layers vs GDN chunked scan vs FFN GEMMs vs logits. Attention share ≥ ~25% is the go signal for T2; below it, close this issue with the profile as the negative record (the GDN-dominant arch may simply not have the surface area).
  **DONE 2026-10-30 (zed-amo-i1156, amo/M5 Ultra) — VERDICT: GO for T2.** Instrument: `bench_734_prefill_stage_decomp` (riir-gpu, subtraction over `PREFILL_STAGE_MASK`, interleaved median-of-ratios, rounds=3, within-round pairing). Model = the folded league file (`BONSAI_GGUF=Ternary-Bonsai-2-27B-PQ2_0.gguf`, sha256 `3907dc1658db1f78…` = the league pin). Measured shares:

  | P | FFN | attention | GDN | all-on median |
  |---|---|---|---|---|
  | 2048 | 61.6% | 18.8% | 15.8% | — (Issue-1113 cell-1, 10-24, quiet-box) |
  | 4096 | 59.5% | 13.4% | 26.6% | — (Issue-1113 cell-1) |
  | **16384** | 56.1% | **24.6%** | 18.6% | 71.46 s (229 tok/s) |
  | **32768** | 46.0% | **39.3%** | 15.2% | 159.03 s (206 tok/s) |

  **Attention share RISES with P** — the 2048→4096 fall (18.8→13.4%) reverses: 24.6% @16K (at the ~25% threshold, within noise of it) and **39.3% @32K (decisively above the go-signal)**. The O(P²) term catches the linear stages; at pp32k a sparse arm attacks ~39% of the wall. Box state: load 1-min 2.68 (an idle `riir-mmorpg-examples-authority` at 1.7% CPU + a sibling's remote-4090 ssh waiter; Zed exempt per owner rule) — NOT verified-idle to the scored-cell standard, but the shares are within-round ratios (common-mode load cancels; the bench_790-validated protocol property), and the two lengths ran in separate processes 33 min apart reading the same ordering. **Second observation for T2's design: the flash arms launched ZERO kernels at both lengths** (`flash arm launches: tiled 0 · m16 0 · cmma 0`, tiled_flash_enabled=true m16_enabled=true) — the measured attention share is the NON-flash attention path serving at 16K/32K; whatever gates the flash arms off at long P, T2's sparse arm interacts with THAT path (or the gate is the finding). Also disclosed: the Issue-864 KV working-set warning (2048 MiB at block_size 16384) — expected, steady-state unaffected. All-on tok/s read ~206–251 across rounds vs the league's 239.95/210.05 scored cells — same class, this is a shares cell, not a scored row. Raw logs: this issue's landing commit message.
  **T2 opens on the 32K row's 39.3% (and the 16K row rides at threshold); the profile stays the T2 design's first constraint — FFN at 46% @32K is still the single largest stage, so T2 competes with the FFN lever for remaining wall, not with a solved baseline.**
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
