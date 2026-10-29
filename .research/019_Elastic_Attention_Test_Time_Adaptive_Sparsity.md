# Research 019: Elastic Attention — Test-Time Adaptive Sparsity Ratios

**Status:** Active — substrate issue filed, training plan filed (SECONDARY)

> **Source:** "Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers" — Tang, Qiu, Yang, Hong, Xiang, Liu, Dang, Li, Zhang (Soochow Univ + LCM Lab + Baidu), arXiv:2601.17367 (v2 2026-01-28), CC BY-4.0. Code: github.com/LCM-Lab/Elastic-Attention.
> **Date:** distilled 2026-10-29
> **Related Research:** riir-infer 013 (TokenGate — the in-repo static token-pruning cousin and its bench-889 ppl-axis negative); riir-infer 004 (DQ), 013; katgpt-rs Research 086 (RTPurbo — static head calibration), 362 (HydraHead causal head-importance)
> **Related Plans:** riir-train Plan 476 (the router training gate — SECONDARY, owner-gated), riir-infer Plan 622 (TokenGate lane — the existing static-pruning baseline), riir-infer Issue 1156 (substrate/league evaluation — PRIMARY)
> **Classification:** Public (substrate-level; no serving-posture or product content)

---

## TL;DR

The paper retrofits a lightweight **Attention Router** (~0.27M params/layer: task MLP + router MLP) onto a **frozen** pretrained LLM; the router reads **boundary-pooled** Key hidden states (first/last 100 tokens) and per-input assigns each attention head to **full-attention (FA) or sparse-attention (SA)** mode — so the model's overall sparsity ratio adapts at test time instead of being fixed at training. Training: Gumbel-Sigmoid + STE + temperature annealing + trainable Lagrangian multipliers against non-tight per-task sparsity bounds; 12 h on 8×A800; matches-or-beats the backbone on LongBench-E / RULER / LongBench-V2 at ΩMSR ≈ 0.65–0.8.

**Distilled for this repo (serving substrate):** the transferable content is (a) the **two-regime law** — downstream inputs cluster into sparsity-robust and sparsity-sensitive classes, so a per-input *binary budget decision* suffices (no per-task tuning); (b) the **boundary-pooling discovery** — the routing signal lives in the first/last ~100 tokens, saturates at 100–200, and full-sequence pooling *degrades* routing (middle tokens are noise); (c) the **fused mixed-mode kernel pattern** — per-head mode bitmask passed as metadata, one launch, thread-block branching (prior art: the MIT-HAN-LAB BSA kernel the paper itself uses; FlexAttention's BlockMask); (d) the falsifiable claim that **per-input adaptivity dodges the static-sparsity quality wall** our own `tokengate` lane hit (bench 889: needle-retrieval dominance but continuation-ppl FAIL at K∈{0.5, 0.75}).

---

## 1. Paper Core Findings

1. **Two-regime sparsity sensitivity (§2.2, Table 5).** Progressively converting retrieval heads to streaming-sparse heads: code stays ~98–99% of FA performance at ΩMSR = 1.0; summarization ~93%; in-context ~87–90%; but Single-Doc QA collapses to ~56% and synthetic retrieval to ~41–48% past ΩMSR ≈ 0.3. Tasks map into exactly two classes → the model only needs to decide the class, then apply the matching budget (t = 1.0 robust, t = 0.7 sensitive).
2. **Boundary pooling carries the task signal (§D.1, H.5).** Routing decisions from the first/last 100 tokens saturate by ~100–200; pooling the full sequence *degrades* routing (task signature diluted by document-body variance). Router overhead 0.196 ms, length-invariant.
3. **Head routing behavior (Fig. 6, H.2).** A subset of mid/high-layer heads routes consistently FA (recovers Wu et al.'s retrieval heads); a subset routes consistently SA; a minority switches per task. **Backbone-family polarity:** Qwen3 shows structurally universal (task-agnostic) retrieval heads; Llama-3.1 reallocates per task — static manifests are unsafe to transfer across backbones.
4. **Frozen-backbone training recipe (§3.2, App. F).** Gumbel-Sigmoid relaxation + STE + annealing τ(t) = max(τmin, τinit·e^(−0.6p)); min-max objective = LM CE + λ1·Ldiff + λ2·L2diff, Ldiff = ΩMSR − t with task-dependent **non-tight** bounds; λ's trained by gradient ascent (Lagrangian trick credited to PruLong); backbone frozen; 300 steps, batch 48, seq 64K, ~0.74B tokens.
5. **Mixed-mode execution (App. E).** Serial head-split dispatch (materialize, two kernels, merge) loses to a fused kernel that takes the routing decision as per-head metadata and branches per thread block (BSA kernel). Speedups: FA-SSA ≈ 1.5×, XA-SSA ≈ 3.28× prefill at 256K (ΩESR ≈ 0.995).
6. **Task-MLP geometry (H.1).** The trained router's task representations become near-orthogonal (pairwise cosine ≈ 0) **without task supervision** — the frozen backbone already supplies the separability; the router mostly reads it out.

## 2. Distillation

### 2.1 Path 0 inventory (modelless decomposition)

| Component | Ships here? (coverage) | Modelless-extractable? | Disposition |
|---|---|---|---|
| Per-head FA/SA router (Gumbel+STE+Lagrangian) | **No** — nothing here routes per input (signal-diff §2.2) | **Partial** — a training-free replacement exists: boundary-pooled hidden states → two calibrated regime centroids → one sigmoid gate (`σ(β(⟨x̄,c_rob⟩−⟨x̄,c_sen⟩))`); the paper's own 0.27M-param/≤200-token/near-orthogonal geometry certifies a linear classifier is a priori adequate | modelless gate = the GOAT-floor baseline inside Plan 476; trained arm must beat it |
| Two-regime budget law | Observation published (FlexPrefill Fig. 3, Feb 2025); not a runtime gate here | Yes — offline per-family ΩMSR sweep → digest-pinned manifest (ΩMSR = last t with retention ≥ 1−ε) | riir-infer Issue 1156 (instrument + manifest) |
| Boundary/edge-window pooling | No explicit edge-window pooling consumer found | Yes — closed-form `concat(mean(h[:k]), mean(h[−k:]))`, k calibrated by saturation sweep; pooling-diet diagnostic tells any pooled consumer which class it is | recorded in this note; adopt opportunistically (weak near-term consumer) |
| Offline retrieval scoring S(ℓ,h) (Eq. 9) | **Yes** — `rt_turbo` attention-mass head calibration (katgpt-rs Plan 126) + causal variant (Research 362) | — | covered as *static per-model*; signal-diff below |
| Fused mixed-mode kernel (per-head bitmask, one launch) | No — kernels here are dense FA (`att_pf_fa`, head-gang, ktile64) + GDN; no block-sparse prefill arm | Yes — pure kernel engineering; decision-source-agnostic (the modelless gate writes the same bitmask) | riir-infer Issue 1156 (execution shape if a sparse arm lands) |
| ΩMSR/ΩESR + per-family conditional-retention holdout | No | Yes — sweep + planted-knee self-test + per-family demote tripwire (the lossy-surface law: per-family, never aggregate) | riir-infer Issue 1156 |
| Temperature annealing schedule | — | training machinery, no standalone consumer | **discarded** — reason: a training-loop schedule with no serving-side analog; nothing consumes a sharpening gate here |
| Trained router artifact itself | No | No (needs GD) | riir-train Plan 476 (SECONDARY) |

### 2.2 §3.6 signal-diff on the one coverage dismissal

Claim: "S(ℓ,h) retrieval scoring is covered by `rt_turbo`." Read the cousin: `rt_turbo` (katgpt-rs Plan 126) consumes **offline, per-model** calibration data (attention-mass over a probe set, computed once, frozen as a static head manifest; `retrieval_head_ratio` fixed at load) and applies it identically to every request. The paper's router consumes **per-input** boundary hidden states and re-decides each head per request. Same base signal family (attention behavior), different conditioning axis (per-model vs per-input) — the per-input axis is **not covered**, and the paper's own Qwen3-vs-Llama polarity finding (universal vs reallocating heads) is the modelless test for *when* the uncovered axis is worth its cost (reallocation coefficient ρ̄ = mean pairwise Spearman correlation between families' S(ℓ,h) vectors; ρ̄ high → static manifest is sufficient, routing adds nothing).

### 2.3 Fusion

**Elastic (paper) × TokenGate (riir-infer 013/Plan 622) × RTPurbo (katgpt-rs 086):** our `tokengate` lane is static token-level prefill pruning whose four-arm GOAT (bench 889) **failed the continuation-ppl axis at K∈{0.5, 0.75}** — the recorded static-sparsity quality wall. Elastic's thesis is precisely that per-input adaptivity dodges that wall (task-conditioned bounds, test-time routing). The non-redundant falsifiable question for this stack: **does per-input adaptivity rescue the ppl axis the static lane measurably lost?** Either answer is valuable — a yes upgrades the pruning lanes with an input-conditional gate; a no closes the adaptivity question for the stack at 4B/16K cost. The `tokengate_fourarm` harness is the reusable GOAT rig.

**Modelless router replacement (the strongest extraction):** boundary-pooled hidden state → dot products against two frozen regime centroids → single sigmoid → per-head mode bitmask (sigmoid, never softmax; deterministic; freeze/thaw-carried; BLAKE3-committed like any calibrated snapshot). No Gumbel, no STE, no GD at runtime. The trained router (Plan 476) then has an honest floor to beat — per the Feature Flag Discipline, a trained arm that cannot beat the modelless floor does not promote.

## 3. Verdict

### 3.1 Tier — per track

| Track | Verdict | Reasoning (one line) | Output |
|---|---|---|---|
| **A. Substrate / serving (this repo)** | **GOAT (Gain disposition)** — sparse prefill attention is a real lever for the weakest league rows (pp16k/pp32k ≈ 0.27×), but the training-free sparse baseline (XAttention/FlexPrefill class) is published prior art, not this paper's contribution; this paper's contribution (adaptivity) only matters *after* a static sparse arm exists | riir-infer Issue 1156 (PRIMARY — profile first, then training-free sparse arm, per-family gates) |
| **B. Training recipe (riir-train)** | **Worth filing — gate pre-registered, unexecuted** — recipe is published, LoRA-class lane fit is proven, and the falsifiable question (adaptivity vs the bench-889 ppl wall) is worth one owner-gated window; but no production consumer exists today, the full 64K posture is memory-infeasible on 24 GB, and NO gate has run — GOAT language is reserved for the Phase 2 result | riir-train Plan 476 (SECONDARY — serving-envelope fit defers to Track A; owner-gated 4090 window; says so in its Status) |
| **Super-GOAT** | **Not awarded** | The mechanism is published combination-level work (FlexPrefill 2502.20766 per-head input-adaptive training-free; DuoAttention 2410.10819 static per-head FA/SA; InfLLM-v2 2509.24663 dense-sparse switchable; MoBA 2502.13189 MoE-style attention routing) — this is a distill with scoped deltas, and the product selling point fails (game-decision serving sits at 512–4K, below the sparsity payoff knee; Q3 of the novelty gate fails) | — |

### 3.2 Published prior art (the honest positioning)

- **FlexPrefill** (arXiv:2502.20766, ICLR 2025): per-head, input-adaptive, **training-free** (JSD probe measurement) — but both branches are sparse (no FA mode), and its Fig. 3 already publishes the task-dependent sparsity observation. The paper under review should be read as: FlexPrefill's adaptivity + a trained hidden-state router + an exact-FA escape branch + frozen-backbone retrofit.
- **DuoAttention / RTPurbo / PruLong**: static per-head FA/SA splits (offline calibration / learned-but-fixed). Our `rt_turbo` is this class.
- **InfLLM-v2**: dense–sparse switchable — whole-model, keyed on sequence length, trained-in.
- **MoBA / SeerAttention / NSA**: trained routing at block granularity inside trained models.
- **BSA kernel / FlexAttention**: the mixed-mode fused-kernel pattern is prior art; the paper claims no kernel novelty.
- Concurrent: "Retrieval Heads are Dynamic" (arXiv:2602.11162) and same-group follow-up SwiAttn (arXiv:2603.26380) corroborate the input-dependence premise post-hoc.

### 3.3 MOAT gate

Public-substrate moat (attention kernels + serving policy instruments) — in scope for this repo: the sparse prefill arm, the per-family ΩMSR manifest, the modelless regime gate, the fused-mixed-mode pattern. The trained router recipe routes to riir-train (Plan 476); decision-serving framing would route to riir-reflex and is not applicable here. Game-runtime framing (per-NPC adaptive compute) has no consumer — our cognition kernels are not quadratic — and the render-side limelight already implements the same two-regime adaptive-cost law (region decides existence, rings decide cost); recorded as validation, not a new primitive.

## 4. Provenance

- Paper read in full (arXiv HTML v2, 26 pp) 2026-10-29; no `.raw/` clone needed (verdict rests on the paper, not the code tree; the mining of their fused-kernel diff into the `kernel_opt` corpus is a separate riir-refine distill lane if wanted).
- Panel: No-GD advocate (13-item modelless inventory) + model-based advocate (tiered cost/gate analysis) + two prior-art web searches — one parallel batch (2026-10-29). Advocate-finding discards: one (temperature-annealing schedule — training-loop machinery, no serving consumer; mechanism-level reason above).
- Internal prior-art grep: arXiv ID + "elastic attention" → zero prior distillation; vocabulary sweep (`retrieval head`, `attention router`, `head routing`, `sparsity ratio`) → katgpt-rs Plans 126/131/139/140 + Research 086/362 (the static-calibration cousins).
- Verdict negotiated with the Claude reviewer (§5 of the skill) before commit.
