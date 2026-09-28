# Research 003: NaiveRT — DSA Mega-Kernel Verify Path & the AI-Driven R&D Ledger (NaiveAI)

> **Source:** "Naive-N0.5-Flash: Building Frontier AI with AI" — NaiveAI technical blog, NaiveRT + AutoWM sections (<https://naive.ai/en/research/>); model card `NaiveAI/Naive-N0.5-Flash` (HF, MIT); repo `github.com/NaiveAI-Labs/Naive-N0.5-Flash`. Blog is prose, not an arXiv paper; no `.raw` clone (no kernel source exists to clone — see §5).
> **Date:** 2026-09-28
> **Status:** RECORD — Track A (kernel/serving) = **Gain** (`.issues/024` filed, measurement-gated, nothing promoted); Track B (research-loop process) = **Pass**. Megakernel class novelty: **not claimed** — abundant published + internal prior art (§5).
> **Related Research (this repo):** 001 (FluidInference audio), 002 (FluidUse int8 embedding) — unrelated lanes; this is the first serving-perf note here.
> **Related Research (katgpt-rs):** 112 (mKernel fused multi-node GPU kernels — the megakernel taxonomy), 139 (Kog Monokernel CPU mapping).
> **Related Research (riir-ai):** 036 (Luce Megakernel hybrid DeltaNet/attention — `riir-ai/.research/036_Luce_Megakernel_Hybrid_DeltaNet_Attention.md`).
> **Related Plans:** katgpt-rs 012 (Lucebox distill — already cites the Hazy Research Megakernel blog), 150 (mKernel conceptual alignment, no implementation); riir-ai 182 (Luce layer-wise hybrid), 171 (GPU decode fusion).
> **Classification:** Public (this repo is public; the distill is our own words over a public blog; no NaiveAI code vendored).

---

## TL;DR

NaiveRT is NaiveAI's AI-driven rewrite of the SGLang serving path for their 309B MoE model (15.5B active; hybrid SWA–DSA, GQA4, top-2048 token selection): one DSA layer's **29 kernel launches per decode step** collapse into **one cooperative mega-kernel** (148 CTAs) that also absorbs the preceding RMSNorm and the trailing TP8 all-reduce. Result: 2,122 tok/s single-stream peak on 8 GPUs, 3.4 ms speculative round vs 12.3 ms SGLang. The transferable content for this repo is **not** the megakernel class (known; see §5) but (a) two intra-kernel geometry tricks with plausible analogs on our laya Metal lane — filed as measurement-gated `.issues/024` — and (b) three crisp external specimens of the measurement-honesty laws this workspace already ships. The companion AutoWM section is process-level confirmation of how we already operate; nothing filed from it.

**Distilled for riir-infer (serving substrate):** the per-kernel-boundary costs NaiveRT removed (launch/sync between norm→projection, padded MMA rows for small head counts, un-staged K/V gathers) are exactly the boundary classes our Metal lane already attacks with pass-scoped command buffers, the sgemm instance picker, and the fused flash-attention kernel. Their two remaining tricks are candidate extensions, gated on our stage-level profiler showing the gap at OUR shapes.

---

## 1. What NaiveRT reports (facts kept, with numbers)

- **Baseline (SGLang):** one DSA layer = **29 kernel executions per GPU per decode step**: Q/K/V projections, RoPE, KV-cache writes, indexer GEMM, top-k selection, sparse gathering, attention, output projection — excluding the input RMSNorm and layer-boundary communication.
- **NaiveRT:** ONE cooperative mega-kernel, 148 CTAs in one grid, also taking in the preceding RMSNorm and the trailing TP8 communication. The four context-parallel kernels are **replaced by the new execution topology**, not fused one-for-one — data that crossed kernel boundaries now stays inside the fused path.
- **Weight/compute overlap:** QKV weights do not depend on the norm result, so each of the 124 QKV-projection CTAs issues a **128 KiB TMA weight fetch BEFORE computing RMSNorm locally**, waiting at the barrier only when weights are consumed for the projection. 21–23 μs/step full-model win; the same shared-memory region is reused by the index and attention stages.
- **K/V staging:** selected K/V rows staged in shared memory by split before compute: layer latency 57.0 → 49.5 μs.
- **Tensor Core remap:** with 8 query heads per GPU (64 heads over TP8), a direct m16n8k16 mapping pads the heads 8→16 rows; remapping so the 8 heads occupy the MMA **n8 dimension** eliminates the padded compute.
- **Architecture context (their diagram + model card):** hybrid SWA:DSA = 5:1 (39 SWA + 9 DSA layers of 48; SWA window 128); DSA module = main sparse attention (64 heads; Q head-dim 192, V 128) + **lightning indexer** (16 heads, dim 128): index-Q proj · weight proj · index-K proj → LayerNorm → RoPE-64D + NoPE-64D concat → 128D → FP8 quant → cache append → index sweep over all visible keys → **top-2048, one shared selection set per query** → gather at selected positions. GQA4 replaces the original MLA-based DSA; DSA itself is credited to DeepSeek. Built on the MiMo-V2.5 base; 3.25T tokens of adaptation training; MIT.
- **Process:** 151 documented trials in 3 stages — whole-network engineering 43 trials/28 adopted; real-checkpoint 45/15; W8A8 kernel refinement 63/20. 71 failed or rolled back, 17 explored alternatives/prototypes; thousands of end-to-end runs. **Merge gate: bitwise logits + KV-cache equality, plus end-to-end latency — before any latency change lands.**
- **Results:** peak single-stream **2,122 tok/s on 8 GPUs** (best 1-second window, temp 0.4/top-p 0.95, prefill excluded; the card rounds to "up to 2,000 tok/s Ultrafast"); full speculative round 3.4 ms vs 12.3 ms SGLang same system; verify-only panels end 3.13–3.79 ms.

## 2. The measurement-honesty specimens (why this blog is worth a note here)

Three of their adopt/reject calls are textbook instances of laws this workspace already enforces with instruments:

1. **Short-context bypass** — skipping indexing and reading K/V directly by position: −3 μs at 200 tokens, **+3 μs at 2,200 tokens → rejected**. Our analog: the two-regime/absent-key-regime-gate discipline in the kernel_opt rule corpus (a win that flips sign across the regime is two results, not one).
2. **TMA prefetch** — *slower* in an isolated warm-cache microbenchmark, 21–23 μs/step *faster* end-to-end → kept. Our analog: katgpt-rs AGENTS.md §"A ratio of two SEQUENTIALLY-timed arms measures the BOX" + the bench-target guard (an isolated-kernel reading is not the claim; the deployed path is) + the position-balanced interleaved A/B law our own Metal lane paid for (the cold-GPU sequencing artifact that misread banking77 −16% before collapsing to the honest −3%).
3. **MoE fusion** — seven implementation rounds, every version numerically correct, every version regressed end-to-end (PDL already overlapped the kernel-boundary cost; fusion added synchronization) → humans stopped the direction; MoE kernels stay PDL-chained. Our analog: negative results recorded, not retried until green; G3 no-regression primacy; the fused-attention kernel that was BUILT, MEASURED, and REVERTED the same day when the lane's real sequence lengths refused it.

Their merge gate (bitwise logits + KV equality before ANY latency optimization) is the same shape as this repo's G5 bit-parity gate for the laya lane and the exactness requirements the DFlash2 verify-kernel corpus rules carry.

## 3. Distillation — Track A: kernel/serving → **Gain**

**Pinned claim:** NaiveRT's two remaining intra-kernel tricks — heads→MMA-n remap at small row counts, and weight-fetch overlapped with the dependent norm inside a fused prologue — are *candidate* wins for `riir-infer-laya`'s Metal sgemm/flash_attn lane **only if** the lane's stage-level profile shows boundary gaps or m-padding waste at real encoder shapes. Filed as `.issues/024`; decided by measurement, never by analogy.

| NaiveRT technique | Our lane's state (riir-infer-laya Metal) | Delta / action |
|---|---|---|
| 29 launches → 1 mega-kernel | Launch-boundary class already closed at our scale: per-op commit+wait measured **0.59 ms/dispatch** (`riir-infer-laya/src/laya/riir/metal.rs:21` — "0.59 ms × ~1100 dispatches ≈ 2.5 s/forward"; the **19× end-to-end** figure is riir-reflex `.benchmarks/001_phase1_harness.md`, 280 s vs 15 s on the gate corpus) → ONE pass-scoped command buffer + 1024-encode flush cap; flash_attn fuses qkv-split→rope→scores→window→softmax→PV→merge behind `LAYA_METAL_FLASH=0` kill-switch | Whole-layer persistent kernel deliberately NOT pursued: encoder shapes (seq ~100–320), MSL has no TMA, and the rewrite cost is unjustified while the boundary class is already measured small. Recorded, deferred (`- [-]` in issue 024). |
| CUDA-graph-class boundary costs | riir-ai prefill lane carries `RIIR_PREFILL_CUDA_GRAPHS` (Issues 965/967) on the 4090 side | Same problem, different mechanism (driver-side batching vs boundary removal). No action. |
| Weight TMA fetch before dependent norm | RMSNorm → GEMM are separate dispatches in one pass-scoped CB; no intra-kernel prefetch. Instruments: riir-reflex's `typed_case_split` example (drives the lane under `LAYA_METAL_PROFILE=1`) drains per-stage at real typed_decisions 5-q cases — encoder 90.1% of case GPU, sgemm narrow 85.7% of that, flash_attn 7.5%; head+copy 9.9% (riir-infer `be46033` / riir-reflex `abbcbb3`) | **CANDIDATE (issue 024a):** the lane's own per-dispatch profiler `LAYA_METAL_PROFILE=1` (`riir-infer-laya/src/laya/riir/metal.rs:362` — every dispatch gets its own command buffer, committed + waited) reads per-kernel SHARES only and structurally cannot see an in-pass gap inside the shipped single CB; so measure RMSNorm's per-kernel share as the UPPER BOUND on the prefetch win first; only the T2 fused-vs-unfused A/B can show the in-pass boundary. |
| K/V staged by split in smem | flash_attn already stages K/V tiles through threadgroup memory, two-pass normalize | **COVERED** — no action. |
| heads → MMA n8 remap (8 heads/rank padding m 8→16) | narrow sgemm (32×64, column-twin accs) picked below m ≥ 256; batched attention = one dispatch per op across heads | **CANDIDATE (issue 024b), applicability UNKNOWN:** our m extent is sequence length, not per-rank head count — the padding tax appears only on tiny-m call sites. Measure the real shape histogram first; a legitimate N/A is the expected honest outcome for the encoder lane. |

**League note:** NOT a perf-rematch opponent. NaiveRT is co-designed with one model (TP8, FP8, their own DFlash drafting); it is not a llama.cpp-class general engine that can serve Bonsai-27B / qwen3.8 for a fair `watch_repo` row, and 2,122 tok/s on 8 GPUs is not comparable to our single-box M3/4090 lanes at our model sizes. No `scripts/perf_rematch.sh` row.

## 4. Distillation — Track B: research-loop process → **Pass**

- The operating model (human researchers set objective/compute budget/acceptance criteria; the AI proposes, implements, validates, analyzes, and decides what to keep or roll back; AutoWM: 400 h, 15 major rounds, WorldArena-1 Track 1 **77.43 vs prior leaderboard 73.64**; NaiveRT: the 151-trial ledger) is **the model this workspace already runs** — agent sessions under GOAT gates, owner calls for promotions, per-plan `.benchmarks/` records, necessity/score jsonl ledgers. No new mechanism to adopt; the trial-ledger *reporting* shape was considered and not adopted (our per-plan records + append-only jsonl already carry the substance).
- AutoWM specifics worth one line each: recipe reproduction first; setup changes before method changes (captions rewritten, 2.5K→22.5K clips, VLM-score filtering); **non-monotonic frame scaling (16 > 8, 32 marginal)** — the workspace's "distrust mechanisms inferred from monotone sequences" lesson wearing their numbers; reference-free-metric-guided output selection (multi-timestep → knapsack-DP frame selection → Best-of-N → post-processing) — our analog is riir-instinct's certified-arm selection (best-measured arm under paired LB95), a cousin with no delta worth filing for a text-decision stack that generates no video.
- Training-adjacent check (pre-flight #5): no video/world-model training pipeline exists anywhere in the workspace (riir-train trains LLM specialists/encoders/critics; quest_grammar LoRA; TernaryDraftModel ternary drafter) and no WorldArena consumer exists → AutoWM's training recipe is genuinely out of scope; the redirect stands on this justification.
- **PASS-Redirects:** no workspace `.research/` note carries the measurement-law corpus (the laws live in katgpt-rs/AGENTS.md instruments — timed_region_guard, sequential_ab_timing, bench-target guard — not in notes), so the discoverability record for Track B is THIS note. The kernel-cousin notes (katgpt-rs 112/139, riir-ai 036) are referenced here by number + name so `grep mega|megakernel` lands in both directions.

## 5. Closest cousins (prior art — internal + external)

**Internal (workspace corpus):**
- `katgpt-rs/.research/112_mKernel_Fused_Multi_Node_GPU_Kernels.md` — UCCL mKernel; verdict LOW DIRECT VALUE but ships the **megakernel taxonomy** and names the same vision ("collapsing several fused steps into a single megakernel that spans an entire transformer layer").
- `katgpt-rs/.research/139_Kog_Monokernel_CPU_Conceptual_Mapping.md` — Kog AI Monokernel (CPU analog).
- `riir-ai/.research/036_Luce_Megakernel_Hybrid_DeltaNet_Attention.md` + riir-ai Plan 182 — layer-wise hybrid inference.
- `katgpt-rs/.plans/012_lucebox_distill.md` — Lucebox-Hub: persistent CUDA kernel for all 24 layers, 1.87 tok/J @ RTX 3090; **already cites the Hazy Research Megakernel blog** ("Look Ma, No Bubbles", 2025-05-27).
- `katgpt-rs/.plans/150_mkernel_conceptual_alignment.md` — megakernel = conceptual guide only, no implementation.
- `katgpt-rs/crates/katgpt-attn-match/src/key_selection/highest_attn.rs` — `select_highest_attn_keys` (+ OMP selector, GOAT g4 coverage pins): the modelless-scale analog of the lightning indexer's index→top-k→gather selection path. Different regime (modelless selection vs FP8 trained indexer at 1M context); no action.
- riir-clippy kernel_opt corpus, Batch 178 (cinference DFlash2 distill) — the verify-kernel rule family (draft-block-simdgroup-kv-split, scores-in-registers-PV, warp-bitonic-topk-merge, verify-window-prework-loads-before-stores): the direct domain neighbor. NaiveRT is the same problem one level up (fused drafting + fused verify window).

**External (class-level prior art, §4 searches):** Hazy Research "Look Ma, No Bubbles" Megakernel (2025); AutoMegaKernel (RightNow-AI — auto-tuned megakernel beating CUDA-graphed cuBLAS at batch-1 decode); "Compiling LLMs into a MegaKernel" (Jia). **No novelty is claimed on the megakernel class** — NaiveRT is a strong engineering instance of a known class, differentiated by the DSA/indexer fusion, the TP8 boundary absorption, and the measurement-honesty process record.

## 6. Verdict

| Track | Tier | One-line reasoning |
|---|---|---|
| A — kernel/serving (NaiveRT) | **Gain** | Two concrete, measurement-gated candidates for the laya Metal lane (issue 024); launch-boundary + K/V-staging classes already covered; megakernel class has abundant prior art (internal 112/139/036 + external) → not GOAT, certainly not Super-GOAT. |
| B — research-loop process (AutoWM) | **Pass** | Process confirmation of the operating model we already run; every candidate improvement fails the signal-diff (trial-ledger shape ⊂ existing ledgers; non-monotonic + end-to-end laws shipped with instruments; reference-free selection maps to certified-arm selection with no delta for our domain); training recipe genuinely out of scope (no video pipeline, no consumer). |

**MOAT gate (riir-infer — public substrate moat):** the note strengthens the serving-perf corpus of the public substrate repo with two honest, gated candidates and a distill-source verdict — modest, in-scope, no promotion, no default flip. Track A's candidates live behind measurement, per the feature-flag discipline; nothing ships until the profiler shows the gap.

## 7. Fusion

paper × cousin: NaiveRT's prefetch-before-dependent-compute × our per-stage GPU profiler (`typed_case_split` probe class) = exactly issue 024a's experiment: does the norm→GEMM boundary show a drainable gap at real encoder shapes? If yes beyond the noise floor with G5 byte-identical, it graduates to a plan + a kernel_opt rule candidate; if no, the raw numbers are recorded in the issue — the MoE-fusion ending is the honest one.

**Distill-source verdict for riir-clippy's kernel_opt corpus:** `github.com/NaiveAI-Labs/Naive-N0.5-Flash` was checked (2026-09-28): ONE commit, model files only (`modeling_naive_n05_flash.py`, configs, tokenizer) — **NaiveRT's inference code is not released** (the transformers quick-start path is a vanilla FP8 HF load). A distill batch needs verbatim quotes at a pinned sha from real kernel source; a blog is quote-thin. **Not a mining candidate.** Re-check only if NaiveAI publishes the runtime.
