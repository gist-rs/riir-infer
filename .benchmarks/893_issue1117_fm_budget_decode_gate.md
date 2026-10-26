# Bench 893 — Issue 1117 T3/T4: the FlashMemory budget decode lane end-to-end on the real Bonsai-2 PQ2_0 — teacher-forced NLL / top-1 vs dense at 32K, ρ ∈ {0.05, 0.10, 0.15, 0.30}

**Status:** COLLECTED 2026-10-27 ~05:37 +07 (the v2 relaunch completed end to end; report.md + report.json at `~/.cache/i1117/32k_rel/`). v1 (killed 10-25) partials stand as the cross-session INDICATIVE rows; the v2 session's dense + ρ 0.15 + ρ 0.30 are one-session self-consistent on corpus v2.
**Date:** 2026-10-25 · **Issue:** [1117](../.issues/1117_flashmemory_budget_decode_consumer.md) T3/T4 · **Code:** riir-infer `51f52f4` (the ternary decode lane + the G1 battery) + katgpt-rs `3d3152fbd` (the incremental last-block centroid substrate) · **Instrument:** `src/bin/flashmemory_decode_gate.rs` · **Predecessor:** riir-train [Bench 671](../../riir-train/.benchmarks/671_issue452_e1_topk_budget_selection_fidelity.md) (the per-layer fidelity oracle this gate compounds), katgpt-rs Issue 962 (the budget selector).

## Question

Bench 671 measured attention-OUTPUT fidelity of block selectors at exactly matched retention — PER LAYER (cos 0.967/0.975/0.986 at ρ 0.10, 32K). Whether per-layer selection survives 16 attention layers compounding through the 64-layer hybrid stack is an end-to-end question: teacher-forced next-token NLL and top-1 agreement vs the dense decode, at the same ρ grid, over a 32K-token slice of real text.

## Rig

`flashmemory_decode_gate`: teacher-forced stepping through the REAL forward — no sampling, both arms consume the same ground-truth token stream (the comparison isolates the attention-read pattern). Dense arm = `forward_qwen_deltanet_ternary` (bounded cache 32770); each ρ arm = `forward_qwen_deltanet_ternary_flashmemory` (block 64, refresh 64, incremental centroids, fresh cache + state per arm). Metrics per position, bucketed at 4096: ΔNLL mean / p95 (tail support disclosed) / max, top-1 agreement, sampled mean selected fraction (every 16 steps, last attention layer), tok/s per arm.

- Model: `Ternary-Bonsai-2-27B-PQ2_0.gguf` — 64 layers (16 full-attention at interval 4), embd 5120, vocab 248,320, 4 KV heads × head_dim 256; loads 3.8 s warm.
- Corpus (v2, relaunch): **nine product-stack repos' AGENTS.md, concatenated in sorted order** — `katgpt-rs riir-ai riir-chain riir-dapps riir-game-sdk riir-infer riir-neuron-db riir-reflex riir-train` (the v1 file list was lost with /tmp and never named in this doc — the recipe gap; v2 names it). 270,109 B → 78,099 BPE tokens (first 32,769 stepped). BLAKE3 `1b4e2e3b6f9430c018753fcffcd97b7551c0b55e4a5bb6102dad860f39d50727` (3,315 lines) — a re-run reproduces the token stream byte-for-byte. v1 record kept below for lineage.
- Corpus (v1, killed run): the workspace's nine repo AGENTS.md files concatenated (328 KB → 94,131 BPE tokens; the first 32,769 tokens stepped). BLAKE3 `3218cadc0193dc1ba6ef946cfdded1da5a5d40be9380260d682e72d4386dab7c` (3669 lines) — file list NOT recoverable (see Status).
- Feature posture: `flashmemory_budget,deltanet_ternary_inference,bonsai2_hadamard`, `--release`.

## Pipeline-validation cell (ctx 1024, ρ 0.10 — completed before the 32K launch)

| arm | mean sel frac | tok/s | ΔNLL mean | ΔNLL p95 (support 52) | ΔNLL max | top-1 agree |
|---|---|---|---|---|---|---|
| ρ 0.1 | 0.1225 | 13.0 | +1.1785 | +5.6960 | +16.2127 | 52.54% |

Dense reference 11.9 tok/s, pooled NLL 3.1537. NOTE the operating point: at 1K context only ~16 blocks exist, so the budget rule's `round(0.1·n) − 1` floors at ZERO complete blocks — the arm attends the CURRENT BLOCK ONLY for most of the run (mean selected fraction 12.25% = the early positions where the current block is most of the context). The 1K cell is the harshest possible posture for the selector (recency-only attention over a docs corpus), not a representative one; the 32K buckets carry the real read.

## Results — 32K

### v2 session (2026-10-26 relaunch, corpus v2, one-session self-consistent — the collected record)

| arm | wall | tok/s | vs dense | mean sel frac | pooled ΔNLL mean | pooled ΔNLL p95 (support) | top-1 agree |
|---|---|---|---|---|---|---|---|
| dense (reference) | 9094.0 s | 3.6 | 1.00× | — | pooled NLL 2.9938 | — | — |
| ρ 0.15 | 6612.0 s | 5.0 | **1.39×** | 0.1467 | +0.4048 | +2.5210 (1639) | 70.56% |
| ρ 0.30 | 11591 s+ | 2.7 | **0.76×** | 0.2968 | +0.2087 | +1.5642 (1639) | 79.91% |

Retention lands nearly exact on target (0.15 → 0.1467, 0.30 → 0.2968) — the budget rule + incremental centroids hold at 32K. Per-bucket (full tables in `~/.cache/i1117/32k_rel/report.md`, committed excerpt below): **quality IMPROVES in later buckets for both arms** — ρ 0.15 ΔNLL mean falls 0.477 → 0.248 and top-1 rises 65.5% → 75.5% across the eight 4K buckets; ρ 0.30 falls 0.232 → 0.107 and rises 76.8% → 84.8%. The Bench-671 depth worry (some rows lose mass at depth) does NOT compound — the early-context buckets are the worst, not the deep ones. The speedup/retention curve now has four points (v1 + v2, cross-session): 1.82× @ 0.05 → 1.33× @ 0.10 → 1.39× @ 0.15 → 0.76× @ 0.30 — the crossover sits near ρ ≈ 0.2; above it the refresh + centroid overhead outpaces the block-read saving.

ρ 0.15 per-bucket (ΔNLL mean / top-1): [0,4K) 0.4770/65.45 · [4K,8K) 0.4852/65.55 · [8K,12K) 0.4944/67.21 · [12K,16K) 0.4504/68.82 · [16K,20K) 0.5542/67.82 · [20K,24K) 0.3100/74.95 · [24K,28K) 0.2189/79.20 · [28K,32K) 0.2483/75.46.
ρ 0.30 per-bucket: 0.2320/76.78 · 0.2429/76.07 · 0.2843/76.73 · 0.2440/78.61 · 0.3123/76.00 · 0.1543/83.50 · 0.0935/86.79 · 0.1065/84.79.

### v1 session (2026-10-25, killed at owner request — corpus v1, INDICATIVE cross-session rows)

| arm | wall | tok/s | mean sel frac (last attn layer, /16 sampling) |
|---|---|---|---|
| dense (reference) | 6435.1 s | 5.1 | — |
| ρ 0.05 | 3517.6 s | 9.3 (**1.82× dense**) | 0.0481 |
| ρ 0.10 | 4827.2 s | 6.8 (**1.33× dense**) | 0.0973 |

## Reading

The per-layer fidelity numbers (riir-train Bench 671) SURVIVE 16 attention layers compounding through the 64-layer hybrid stack at 32K, with the loss quantified and graceful: at ρ 0.15 (85% KV reduction) the sparse decode costs +0.40 pooled ΔNLL and 70.6% top-1 agreement against dense — degraded, measurable, and bucket-monotone-IMPROVING with depth (the opposite failure mode from the one T3 was written to catch). The T4 shape: sparse decode is FASTER than dense up to ρ ≈ 0.2 and SLOWER above it — the operating point for any serving posture is ρ ≤ 0.15, where quality cost and speed gain are both on the favorable side of the crossover. NOT tested here: 128K (the 4×-longer axis, ~16 h-class — deferred with T4 below), and the GPU decode posture (this is the CPU reference the issue scoped). Corpus v2 (not v1) — the arms are self-consistent; absolute NLLs are corpus-specific (dense pooled NLL 2.9938 on v2).

## Box state (v2 session)

M3 Max (16 cores, 64 GB), AC power, battery charged. Launched `nice -n 10` under active sibling agent compiles (load ~16–18 through the run; the run itself owned ~9.3 cores). Dense throughput declined 10.5 → 2.8 tok/s across the run — that is the dense arm's own O(t) attention growth plus load, not a defect; walls read as ratios inside the session. Corpus v2: 78,099 BPE tokens, BLAKE3 `1b4e2e3b…50727` (recipe in §Rig).

## T4 note (the rebuild-vs-incremental cost axis)

The G1 battery (riir-infer `51f52f4`) pins incremental ≡ rebuild bit-identically at ρ 0.5 over 700 steps; the T4 A/B (rebuild-mode vs incremental-mode tok/s at 32K) is NOT in this run — both 32K arms run incremental. The rebuild arm's cost is bounded analytically: a full rebuild is O(seq_len × kvd) per attention layer per step = 32,768 × 1024 × 4 B ≈ 134 M centroid-read MACs/layer/step vs the incremental last-block 64 × 1024 ≈ 65 K — a ~2000× centroid-work ratio that stays under ~4% of the model's ~27 G MACs/step even at full rebuild; the T2.5 record names the measured A/B as follow-up if the lane is ever promoted to a serving posture.

## See also

- riir-infer Issue 1117 — the lane record (T1/T2 landed `51f52f4`; the block-open force-refresh law).
- katgpt-rs Issue 962 — the selector's promotion note (absolute rule stays the selector default until this gate's verdict).
