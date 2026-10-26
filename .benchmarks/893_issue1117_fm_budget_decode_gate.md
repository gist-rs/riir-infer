# Bench 893 — Issue 1117 T3/T4: the FlashMemory budget decode lane end-to-end on the real Bonsai-2 PQ2_0 — teacher-forced NLL / top-1 vs dense at 32K, ρ ∈ {0.05, 0.10, 0.15, 0.30}

**Status:** RELAUNCHED 2026-10-26 ~14:5xZ (zed-m3-idle-relaunch — the same lease, `zed-m3-continue` lineage, expires 2026-10-26T22:22:40Z). The 10-25 state was lost with `/tmp` (bin + corpus + `p1117_corpus/` gone; `/tmp/p1117_32k/` survived EMPTY): the bin was rebuilt (`98fa26d`, which also lands the T3 pause's noted instrument defect — per-64-step stderr progress lines per arm, so an arm is no longer silent for ~1 h) and the corpus REBUILT under a new, explicitly-named recipe below. **The 10-25 corpus was NOT recoverable byte-exactly**: the doc pinned the BLAKE3 but not the file list (recipe gap, now fixed — the recipe is IN the record), and a 935-combo reconstruction over the 10-25-era git trees (line-count + byte-window filtered) found no hash match — the original tree's bytes are not recoverable from git (uncommitted sibling WIP is a plausible cause). Consequence, disclosed: the new run executes **dense + ρ 0.15 + ρ 0.30 in ONE session on the new corpus**, so the ΔNLL / top-1 comparison is self-consistent; the killed session's dense/0.05/0.10 walls stand as the cross-session INDICATIVE rows only (the doc already scoped walls as ratios, never absolutes). Scrath moved OFF /tmp: `~/.cache/i1117/` (this loss is why). RELAUNCH POSTURE: `nice -n 10`, sibling agent compile active (load ~16 at launch — the recorded posture).
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

**Partial (in-flight run, logged `#   wall …` lines):**

| arm | wall | tok/s | mean sel frac (last attn layer, /16 sampling) |
|---|---|---|---|
| dense (reference) | 6435.1 s | 5.1 | — |
| ρ 0.05 | 3517.6 s | 9.3 (**1.82× dense**) | 0.0481 |
| ρ 0.10 | 4827.2 s | 6.8 (**1.33× dense**) | 0.0973 |
| ρ 0.15 | running | | |
| ρ 0.30 | queued | | |

The T4 shape is already legible: the sparse decode is FASTER than dense at 32K (the dense arm's attention reads grow with t_n; the sparse arm reads only the selected blocks — refresh + centroid overhead included), and the speedup DEGRADES with ρ exactly as the retention math predicts (1.82× at 4.8% retained → 1.33× at 9.7%). The quality half (ΔNLL / top-1 per bucket) is the pending half — it lands with the run.

## Reading

PENDING — filled at collection with the per-bucket ΔNLL / top-1 tables from `report.md`.

## Box state

M3 Max (16 cores, 64 GB), AC power, battery 100% charged. The 32K run executed `nice -n 10` with concurrent sibling agent sessions active (load average 14.31 at launch; 93% memory free). tok/s rows are RELATIVE comparisons between arms measured INTERLEAVED-ish (dense first, then each ρ back-to-back) under the same load — the T4 read is the SPARSE:DENSE ratio per session, not an absolute league number (the league decode posture is the 4090 CUDA lane; this is the CPU reference the issue scoped).

## T4 note (the rebuild-vs-incremental cost axis)

The G1 battery (riir-infer `51f52f4`) pins incremental ≡ rebuild bit-identically at ρ 0.5 over 700 steps; the T4 A/B (rebuild-mode vs incremental-mode tok/s at 32K) is NOT in this run — both 32K arms run incremental. The rebuild arm's cost is bounded analytically: a full rebuild is O(seq_len × kvd) per attention layer per step = 32,768 × 1024 × 4 B ≈ 134 M centroid-read MACs/layer/step vs the incremental last-block 64 × 1024 ≈ 65 K — a ~2000× centroid-work ratio that stays under ~4% of the model's ~27 G MACs/step even at full rebuild; the T2.5 record names the measured A/B as follow-up if the lane is ever promoted to a serving posture.

## See also

- riir-infer Issue 1117 — the lane record (T1/T2 landed `51f52f4`; the block-open force-refresh law).
- katgpt-rs Issue 962 — the selector's promotion note (absolute rule stays the selector default until this gate's verdict).
