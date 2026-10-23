# Research 016 — PHOTON (arXiv:2512.20687): Hierarchical Autoregressive Decoding — record-only for the substrate

**Status:** DISTILLED 2026-10-23 — record-only, below GOAT. No authors' PHOTON code or pretrained checkpoints exist publicly (searched 2026-10-23); the stack serves fixed checkpoints and pretrains no hierarchical-arch models, so there is no consumer today. This note records the architecture, the decode shape riir-infer would need IF a PHOTON-class checkpoint ever appears, the training-recipe redirect (justified, not lazy), and one P3 league-metric suggestion. Two cousin notes get PASS-Redirect lines. Zero plans, zero issues filed (the R006 precedent: record the decode shape, no action without a consumer). Verdict path: `request_verdict` claude_code backend auth-unavailable → `spawn_agent` reviewer fallback ran (equal verdict, 1 round); the reviewer independently re-verified the numbers against arXiv v2 and flagged 7 defects (stale highwater, two mis-assembled Table-1 rows, the α=0.0 main-config conflation, a third-party re-implementation the caveat missed, a missing full title, a one-byte edit corruption) — all repaired same session.

> **PASS-Redirects (synthesis):** Ichikawa, Takagi, Nakagawa, Kanazawa, Sakai (Fujitsu / RIKEN AIP / Institute of Science Tokyo / Tokai University) [arXiv:2512.20687 "PHOTON: Hierarchical Autoregressive Modeling for Lightspeed and Memory-Efficient Language Generation"] — distill record-only (ACL 2026 long, `2026.acl-long.1778`; arXiv v2 2026-01-08). Architecture-level L-level chunk hierarchy + RecGen; no checkpoints; no consumer in the 7-repo set.

## TL;DR

PHOTON replaces the Transformer's horizontal token-by-token scan with **vertical, multi-resolution scanning**: a bottom-up hierarchical encoder compresses tokens into low-rate contextual latent streams (per level: a chunker pools `C_l` child states into one, then an autoregressive context encoder runs over the chunk sequence), and a top-down hierarchical decoder reconstructs finer streams with **bounded local causal decoders** (attention span `R_l + C_l`, independent of T), decoded **in parallel across chunks** conditioned on the parent level's latent. **RecGen** (recursive generation) is the decoding schedule that makes decode cost sublinear-in-context: after a one-time hierarchical prefill, only the coarsest level's KV cache grows; each new "meta-context" (`C_≤L` tokens) is sampled by the top-down decoders and the next top-level input is the chunker applied to the decoder's own bottleneck reconstruction — **no bottom-up re-encoding**. Theorem A.6: under *bottleneck recursive consistency* (`X̂^(L−1) = X^(L−1)` at the top interface only), RecGen induces the exact same output distribution as hierarchical generation with re-encoding — the theorem is CONDITIONAL on that consistency; an optional auxiliary loss (per-level cosine distance between encoder states and top-down reconstructions, weight α) encourages the condition, and the headline config runs with the loss OFF (α = 0.0; ablation optimum α ≈ 0.3 for zero-shot accuracy).

Headline (paper's own numbers, ≤1.2B models, DGX H200, 1 epoch Pile-uncopyrighted, ctx 2048): **up to 475× throughput-per-memory (TPM = K tok/s/GiB-KV)** vs vanilla Transformer in the decode-heavy regime (1.2B: TPM 2.56 → 1216.67), with a real quality cost (WikiText PPL 19.68 → 23.79; zero-shot avg over HS/SCiQ/ARCe ~58.8 → ~52.1). RecGen over HierGen adds another 2.2× (PF) / 4.5× (DE) TPM on top of the architecture's gains — the paper's abstract "10³×" is the full stack (architecture × RecGen) against vanilla at 600M DE: 13642.50 / 7.35 ≈ **1,856×**.

## 1. Source core findings (verbatim numbers)

| Claim | Number |
|---|---|
| TPM (1.2B, DE 128-in/2048-out) | vanilla 2.56 → PHOTON 1216.67 K tok/s/GiB (**475×**; the Fujitsu press "475× multi-query throughput per GPU" is this ratio) |
| KV memory per sample, DE | 0.230 → 0.023 GiB (600M); reductions **10.0× / 13.1× / 10.8×** at **600M / 900M / 1.2B** (0.230→0.023, 0.354→0.027, 0.390→0.036) |
| Throughput, DE (K tok/s) | **1.69 → 70.43** (600M, 41.7×); **2.34 → 75.52** (900M, 32.3×); **1.00 → 43.80** (1.2B, 43.8×) |
| Quality cost (1.2B) | WikiText PPL 19.68 → 23.79; HS 45.65→40.70, SCiQ 81.50→69.30, ARCe 49.33→46.25 (avg 58.8 → 52.1) |
| RecGen vs HierGen (600M) | PF TPM 1262.58 → 2828.06 (≈2.2×); DE TPM 3062.17 → 13642.50 (≈4.5×; memory 0.023 → 0.012 GiB) |
| α ablation | PPL stable for α ≤ 0.3, degrades at 0.4; zero-shot peaks at α = 0.3 (non-monotonic — "sensitive to tuning") |
| Chunk-length ablation | (C1,C2)=(2,2) vs (4,4): PPL 24.28 vs 29.91, zero-shot 52.01 vs 48.72, PF TPM 175.45 vs 1262.58 — chunk size is the explicit quality/efficiency Pareto lever |
| Prefill cost | Σ_l O((T/C_≤l)²) instead of O(T²); token-level decode O(1) in T per token (bounded windows) |
| Decode KV traffic | global reads collapse from Σ_l O((T/C_≤l)²) to top-level-only O((T/C_≤L)²); local reads linear in T with fixed window |
| Theorem A.6 | RecGen ≡ HierGen in output distribution, given deterministic streaming + bottleneck recursive consistency (equality needed ONLY at the L→L−1 interface, no assumptions on lower levels) |
| Main-results config | **α = 0.0 — the auxiliary loss is OFF in every headline number** (§3.1: set to zero "to isolate gains from the hierarchical architecture"; C.1 repeats it). The α ≈ 0.3 zero-shot optimum is an ablation finding (B.1), not the operative posture. B.3: the recursive loss decreases even at α = 0 — "the hierarchical architecture itself may encourage self-consistent summaries". This strengthens the Path-0.5 redirect (§3): the loss is optional even INSIDE the architecture. |

## 2. Prior art (honest, searched before any verdict)

**Published — the class is dense; no stack-novelty is possible at mechanism level:**

- **Block Transformer** (Ho et al., NeurIPS 2024, arXiv:2406.02657 — code + pretrained checkpoints public) is the direct parent: single-level global-to-local. PHOTON's deltas: L-level hierarchy (not 1), persistent multi-rate latent streams (not block-structured attention), and RecGen (no bottom-up re-encode) + the consistency theorem.
- The tokenizer-free global-local line: **MEGABYTE** (arXiv:2307.08621), SpaceByte, H-Net (dynamic chunking) — single-stream patch hierarchies, no persistent multi-level decode state.
- Multi-scale encoder-decoder line: Funnel-Transformer, Hourglass, Hierarchical transformers (NAACL 2022) — training-side efficiency; token-level AR decode unchanged, KV still grows linearly.
- KV-efficient inference line: sparse/windowed attention, token dropping — single token-level timeline retained. PHOTON's own related-work table is explicit about this positioning.
- **TPM as a metric**: a simple ratio (throughput per GiB of KV); no metric novelty claimed.

**Internal (workspace grep + note read, 2026-10-23):**

- **katgpt-rs R379 (HGA, chunk→group→token routing)** — the closest hierarchical-attention cousin, but the OPPOSITE approach: inference-time sparse routing over PRETRAINED checkpoints (drop-in, no retraining). PHOTON moves the hierarchy into the architecture (pretraining required). R379's own verdict — "mechanism class already shipped (DashAttention + PFlash + RTPurgo); refinement, not a new class" — plus **R225 MSA GOAT FAILED** and **Plan 612 pyramid_topk G2 iso-quality NEGATIVE** are three workspace data points that coarse-to-fine attention structures must win their own gates here.
- **riir-rethink R006 (NCM, coarse-to-fine generation, arXiv:2610.14162)** — the closest generation-paradigm cousin, with the controlling precedent: paradigm record-only (no text-generation consumer), extractable recipes filed only where a consumer exists. PHOTON gets the same posture, stronger: NCM had public checkpoints; PHOTON has none.
- **riir-infer deltanet/GDN lanes + `delta_roll` journal (Research 007, Issue 049)** — fixed-size recurrent state updated from the current forward's own (k,v): the O(1)-state streaming property PHOTON reaches architecturally with its top-level-only growth. Signal-diff: GDN state is decaying/approximate (recall degrades at long range); PHOTON keeps EXACT attention within chunks and EXACT causal attention at the coarse level — the recall-vs-state-size trade is the actual design axis, and it is the honest reason a PHOTON-arch line could beat a pure-GDN line at long context.
- **katgpt-kv `still_kv` (StillPerceiver / IterativeChunkCompactor, Plans T25-line)** — inference-time KV compaction into latent queries: shrinks KV for EXISTING checkpoints; PHOTON rebuilds the model. Different surface, same goal.
- **Game side (coverage check, no primitive):** the two-brain model already ships the coarse-global + bounded-local shape — think brain = coarse zone-level belief (compact, decaying), info brain = exact raw state; SenseLoD (`crates/katgpt-sense/src/lod.rs`) does coarse-to-fine perception; `latent_functor/zone_gating.rs` gates coarse zones within finer channels. PHOTON is an LM-decoding instance of a pattern the game runtime already has.
- **Healer side (coverage check):** AstChunker → RuleIndex fan-out → `pick_domain` routing already form a chunk→domain hierarchy on the retrieval surface; PHOTON's hierarchy is a generation-time autoregressive state, not a retrieval index. No healer primitive.

## 3. Distillation — Path 0 inventory (one verdict per row)

| PHOTON component | Track | Verdict |
|---|---|---|
| Hierarchical encoder (chunker + AR context encoder, L levels) | (c) trained/architecture | NO consumer — requires from-scratch pretraining; no PHOTON checkpoints exist; league pivots on existing Bonsai/qwen GGUFs. Decode shape recorded (§4). |
| Bounded chunk-local top-down decoders (parallel across chunks, span `R_l + C_l`) | substrate | Recorded. This is the decode shape riir-infer would implement IF a PHOTON-class GGUF/safetensors appears (trigger in §5). |
| RecGen (top-level-only KV growth; decoder-side bottleneck summary feeds STEP) | (c) trained/architecture + theorem | The paper's sharpest primitive. Coverage: GDN/deltanet already ship "state grows O(1), updated from the current computation" natively; `delta_roll` ships state-consistency journaling. No NEW modelless primitive extracted — RecGen is the linear-state property grafted onto chunked exact attention. |
| Recursive consistency loss (per-level cosine, ablation optimum α ≈ 0.3; headline runs α = 0.0) | (c) trained | **Path 0.5 redirect, justified:** the loss is not a standalone recipe — it supervises the top-down reconstruction path of THIS architecture and cannot retrofit any existing checkpoint or pipeline (quest_grammar LoRA, ternary drafter, GRPO/DPO lanes: all inapplicable). It becomes actionable only inside a future PHOTON-arch pretraining run (see the trigger-contingent candidate in §5) — and even there it is optional, since the headline posture ships without it. |
| Theorem A.6 (schedule equivalence under bottleneck consistency) | math | Record-only. The FV pattern (decoding-schedule equivalence under an interface condition, with an auxiliary loss driving the interface) is a nice template; our nearest invariants are freeze/thaw torn-read and `delta_roll` recovered-vs-replayed state equality. No Lean filing — nothing in the runtime depends on it. |
| TPM metric (throughput / GiB KV) | measurement | P3 suggestion for the perf-rematch league's long-context/multi-query cells (§5); owner-gated — the league runs under a pinned fairness manifest, and adding an axis is an owner call. |

## 4. Substrate mapping — what riir-infer would need IF a PHOTON-class checkpoint appears

Recorded so the future session does not re-derive:

- **Arch lane:** L-level hierarchy with per-level chunkers (concat→linear) and AR context encoders; top-down converters (1D conv expand to `R_l` conditioning vectors) + chunk-local causal decoders (LLaMA-block class, bounded mask).
- **Decode:** chunk-parallel sampling within a level (independent chunks share the parent context); RecGen posture = discard encoder-side caches at levels 1..L−1 after prefill, retain `KV^(L)`, update the top level once per meta-context from the chunker applied to the decoder's bottleneck output.
- **Loader:** tokenizer is Llama-class; weights layout would be a new `safetensors`/GGUF arch key (neither exists upstream today — llama.cpp has no PHOTON arch either, so a league-relevant GGUF is not on the near horizon).
- **Trigger:** a public PHOTON/Block-Transformer-class checkpoint in a league-relevant size with a serving path (llama.cpp support or safetensors + our loader). Until then: no code.

## 5. Verdict

**Tier: below GOAT — distillation record-only.** Novelty gate: Q1 no prior art — FAIL (class published + dense; the paper IS the class's current frontier); Q2 new behavior class for the stack — no (nothing our systems can newly do); Q3 product selling point — no (we serve fixed checkpoints); Q4 force multiplier — no. All four NO → no guide, no plan.

- **riir-infer: IN (record + watch).** This note is the file. No issue filed — no consumer exists; the R006 precedent (record the decode shape, no action) controls. Re-open trigger: public PHOTON-class checkpoint + serving path (§4).
- **riir-train: redirect justified (recorded in §3).** Trigger-contingent candidate, NOT a queued item: IF the training line ever revisits ARCHITECTURE choice (the §4 checkpoint trigger has fired, or a serving-relevant reason to pretrain a hierarchical model emerges), the honest comparison is **PHOTON-hybrid vs pure-GDN ternary** — PHOTON's bet is that exact chunk attention + coarse exact attention recovers the long-range recall pure linear state loses. Context that keeps this dormant: the from-scratch test-arch lane (Kimi-K3-0.4B) is retired (Bench 430 — dropped after 3 failures; production Rust-coder is Gemma-2 rank-16, per the verdict reviewer's check), and the workspace's training pivot is Bonsai/ternary serving. R006's R379-line caution applies with full force (coarse-to-fine must win its own gates; MSA and pyramid_topk both failed theirs).
- **P3 suggestion (owner-gated, recorded here):** add a TPM-style column (throughput per GiB of KV, or KV bytes per token measured in-harness) to the perf-rematch league's long-context cells. All current league opponents are vanilla-attention GGUFs, so TPM re-normalizes rather than re-ranks today — value appears only when KV-quant / GQA-variant / hierarchical-arch opponents enter the matrix. Not filed as an issue: league-axis changes are owner calls under the pinned fairness manifest.
- **katgpt-rs / reflex / rethink: no files.** No modelless primitive, no decision-serving surface, no private-tier content.

## 6. Honest caveats

- **All numbers are the paper's own** (H200, Pile-uncopyrighted 1 epoch, ≤1.2B, ctx 2048, LLaMA tokenizer vocab 32k). No independent replication found at search time (2026-10-23); community coverage (Fujitsu press, ACL, Japanese-language write-ups) is expository, not replicating.
- **The quality cost is real and is the trade's point:** ~6.8 pt zero-shot avg drop at 1.2B buys the 475× TPM. The Pareto story ("trade saved bandwidth back for quality via test-time scaling") is asserted, not demonstrated with a concrete mechanism.
- **PPL/zero-shot are teacher-forced (HierGen) only** — the paper itself flags that a likelihood protocol under RecGen is open; production PHOTON serving would generate under RecGen, whose distributional drift from the evaluated posture is unmeasured (Appendix B.3 leaves residual-mismatch → drift characterization to future work).
- **Baseline-fairness caveat:** PHOTON is trained with a parameter-matched (not compute-matched) architecture; vanilla baselines are FLOPs-aligned per the paper, but the 6-module PHOTON stack (4 context decoders + chunkers + converters) carries its own training-stability surface the paper does not ablate (Limitations section concedes: no sensitivity analysis of converter widths; chunk sizes only coarsely).
- **No authors' code/checkpoint release** (searched 2026-10-23) — Block Transformer has both; PHOTON has an ACL camera-ready and a Fujitsu press page. A third-party paper-only re-implementation exists (`github.com/Chiytako/photon-experiment`, surfaced by the verdict reviewer's search; fidelity unverified, no league-scale checkpoints). Until the authors' weights land, this is architecture-reading only.
