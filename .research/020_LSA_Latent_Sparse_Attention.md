# Research 020: LSA — Latent Sparse Attention (Subquadratic Ultra-Long Context)

**Status:** RECORD — distillation only; no plan, no issue filed (the crossover law + the 10M SOTA reference are the durable payload; arming a >256k league cell + bucket-index arm is an owner-gated manifest expansion, and no gain is provable at the league's current ≤32k shapes)

> **Source:** "Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention" — Kubicek, Caciolai, Peng, Sumant, Hoover, Greenlee, Battleday, Urban, Hoefler, Whittington (Prefrontal AI + ETH Zurich), [arXiv:2610.42147](https://arxiv.org/abs/2610.42147) (2026-10-30). Code: github.com/Prefrontal-AI/latent-sparse-attention @ `906be195c1052aaa74f95852ad1917ce1926ba64` (Apache-2.0; clone verified — `src/lsa/core/{ward,hierarchy,index_space,attention}.py` + `core/kernels/cutedsl/{hopper,blackwell}/{ward,bucket_scan,thresholds,top_k}.py`). Blog: prefrontal.ai/news/lsa.
> **Date:** distilled 2026-11-03
> **Related Research:** riir-infer 019 (Elastic — the sparse-arm league lane; its PRIMARY issue 1156 is CLOSED+removed 2026-11-02, promotion lane = issue 1165) · katgpt-rs 379 (HGA — hierarchical routing, GOAT-proxy FAIL) · katgpt-rs 071 (DashAttention — selective adoption) · katgpt-rs 086 (RTPurbo — pre-RoPE low-dim projection) · katgpt-rs 022 (Lighthouse) · katgpt-rs 225 (MSA — failed-GOAT precedent) · katgpt-rs 044 (PFlash)
> **Related Plans/Issues:** riir-train Plan 476 (Elastic router gate — SECONDARY, filed NOT executed, owner-gated 4090 window) · katgpt-rs Plan 397 (HGA — COMPLETE, negative result recorded) · riir-infer issue 1165 (τ=0.85 sparse-prefill promotion, OPEN, claimed zed-7710a782 exp 2026-11-04T16:45Z)
> **Classification:** Public (substrate-level; no serving-posture or product content)

---

## TL;DR

LSA attacks the part of sparse attention that still scales: the **indexer**. At 10M tokens, 98% of a traditional sparse-attention layer's prefill time is spent finding which keys to attend to; a 'free' indexer would add 58× prefill / 8.9× decode on top of today's sparse methods. LSA's move: **cluster keys by content in a learned latent space** (L2-normalized → 4:1 pooled → Ward-agglomerated into ~8-key buckets with unit-norm centroids → cached index), then let each query **score bucket centroids instead of keys** — flat search (O(T)) or a beam-searched tree (O(log T), subquadratic prefill). Two training postures: **D-LSA** distills per-layer index projections W_Q^idx/W_K^idx + a temperature into a frozen pretrained model against head-aggregated attention (forward KL, 352M tokens — a fraction of DSA's 2.1B dense warm-up), and **N-LSA** trains the clustering selection natively from the NTP loss alone with a stop-gradient through selection, no positional encoding, no auxiliary loss.

Headline (H200, 10M tokens): one attention layer prefill **22.7×** faster than GLM-5.3-Flash's own trained sparse attention and **72.4×** faster than Qwen-3.8-27B full attention; decode 4.6× / 42.5×. End-to-end in SGLang: TTFT −2.2× (GLM CP8) / −9.2× (Qwen TP8). Accuracy stays within the baselines' 95% CI on knowledge/reasoning and within 5 pts on RULER to 1M — **exceeding** full attention at 10M (63.9 vs 59.9; PaperTrail 29.1 vs 19.8). N-LSA is the only sparse method in their bake-off that matches full attention at 128k (val loss 1.085 = FA; RULER 32.2 vs FA 34.7, NSA 19.2, PISA 14.6, SWA 3.0) while training 1.6× faster per layer.

**Models named are our league family:** the evaluation targets **Qwen-3.8-27B** (the same hybrid-GDN family as our qwen3.8-27B league model — N-LSA's pretraining topology is literally "every fourth layer attention, the rest Gated DeltaNet, no positional encoding") and GLM-5.3-Flash. This is the closest SOTA reference point our league has for the ultra-long-context tier.

---

## 1. Paper Core Findings

1. **The indexer is the bottleneck at ultra-long context (§2, Fig. 2).** Sparse attention moved the quadratic cost into finding keys; at 10M tokens the indexer is 98% of prefill. Every improvement in the block-sparse family (CSA/QSA/GLM compress 4:1, HiSA/LongCat score blocks then keys, IndexCache reuses across layers) divides the indexer by a constant — work per query still grows linearly.
2. **Content clustering holds accuracy where positional blocks do not (§4.2).** D-LSA (Flat) stays within 5 pts of full attention to 1M and beats it at 10M on RULER; Block-Sparse (Tree) — the positional hierarchy at the most accurate block size they measured — falls to ~63 RULER at 1M and is dropped beyond. Larger blocks fall earlier (App. E.6): a block summary can hide a single relevant key.
3. **Hierarchy costs accuracy; flat content-clusters are the accuracy-preserving shape (§4.2, §5).** D-LSA (Tree) follows Flat on Qwen but loses past 1M on GLM (55.8 vs 66.8 RULER at 10M). Tree decode only wins with batched requests (5.6× at 10M with 16 requests) or subquadratic prefill beyond ~10M. Corroborates our own HGA negative (katgpt-rs Plan 397, G2-proxy FAIL).
4. **The crossover law (App. F.1).** Below 32,768 buckets (~256k tokens at n̄b=8) LSA writes the full score matrix S and takes top-p — selection costs more than it saves. Our league shapes (pp16k/pp32k → 2k–8k pool-4 blocks) are deep inside that dense-wins regime.
5. **Top-k oracle justifies the budget (App. A).** At 128k, oracle top-k=512 keeps 98% of RULER while touching 0.8% of keys; k=4096 with sink+window holds accuracy past 1M.
6. **Engineering that transfers to any indexer lane (App. F):** threshold selection (sample → θ at the nθ-th best → scan-and-compact at θ−δ → fp32 rescore of the shortlist → packed (score‖inverted-id) int64 radix select → best-first expansion to k), fallback rates 5×10⁻⁵ @1M; decode select in 86 µs @10M at 75% HBM bandwidth; bf16 centroids lose ~20/4096 selections vs fp32 with <1e-4 attention-mass delta (fp8 centroids churn 6× — node centroids fp8 fine); bucket cache 246 MiB/layer @10M (vs GLM 331, QSA 642); CP integration replicates the bucket cache per rank and gathers only a ~30% prefix union via a 1-byte/row mask.
7. **The RoPE-free index space (`index_space.py`, pinned sha).** The index lives in "the position-free sub-term of the attention logit" — the non-rotary dims are an exact sub-term of the true score, so clustering groups keys **by content, not position**; model-specific rotary layouts are parameters, never hardcoded prefixes. MLA models take the compressed latent as the index (512 dims where collapsing expanded heads would give 16,384).
8. **PaperTrail (App. D):** a RULER extension embedding fabricated fact-trails (introduce → cite → revise+rename, old value still quoted) inside real 2026 papers at 512k–10M; tier-3 questions (3+ paper trails) fall to ≤19 for every method at 10M. A candidate suite shape for any future long-context lane.

---

## 2. Distillation

### 2.1 Path 0 inventory (modelless decomposition)

| Component | Ships here? (coverage) | Modelless-extractable? | Disposition |
|---|---|---|---|
| Semantic bucket index (pool → Gram-Ward merges → unit-norm centroid cache) | **No** — every sparse arm here scores positional blocks (PFlash, τ=0.85 envelope, HGA) | **Yes** — fully closed-form: Ward merge costs from a Gram matrix (Δ = a_b + a_b′ − a_b∪b′), centroids are normalized sums | design point for a ≥256k tier; **not built** — the paper's own crossover law (finding 4) proves zero gain at league shapes ≤32k |
| Threshold top-k (sample → θ → compact → rescore → packed radix select) | **Partial** — the τ=0.85 envelope scorer selects by threshold at block level (bench 1157); `SpKv` does SIMD top-k KV selection | **Yes** — the packed score‖id radix-select + scan-and-compact epilogue is the reusable kernel shape | recorded; consumption only arrives with the bucket index |
| RoPE-free index space (position-free logit sub-term) | **Partial** — RTPurbo (katgpt-rs 086) is the pre-RoPE low-dim projection cousin; signal-diff §2.2 | **Yes** | recorded |
| Training-free PCA projections (D-LSA (Training-Free)) | No | **Yes** — closed-form principal directions of the layer's keys | the **modelless floor** if a bucket-index arm ever lands (same GOAT-floor convention as Plan 476's sigmoid router floor); paper shows it is close on Qwen, trails on GLM |
| D-LSA distillation (frozen base; train only W^idx_Q,K + τ; KL to head-aggregated attention, top-2048 renormalized target) | No — nothing here trains an indexer against frozen-model attention | No (needs GD) | unexecuted cousin of riir-train Plan 476's lane — **NOT filed** (476 itself is filed-NOT-executed + owner-gated; a second unexecuted indexer-training plan is noise) |
| N-LSA native pretraining (NTP-only, stop-grad selection, no positional encoding) | No | No | **OUT OF SCOPE** — full pretraining is outside the stack's training charter; recorded as the design point for any next-gen league-model question |
| PaperTrail benchmark | No | Yes (mechanical construction) | reference row for a future long-context suite |
| CP bucket-cache replication + 1B-mask prefix-union gather | No (the plan-623 multi-host lane is a different plane) | Yes (engineering) | recorded |

### 2.2 §3.6 signal-diffs on the coverage dismissals

**"Covered by the τ=0.85 envelope sparse prefill (bench 1157 / issue 1165)?"** The shipped arm consumes a per-query cheap envelope score against a τ threshold at **positional block** level, one-shot select, engaged ≥16K (promotion condition 1). LSA consumes a learned/PCA **64-dim index space** scored against **content-clustered bucket centroids**, with hierarchical beam option; buckets are position-invariant (finding 7) and the search is subquadratic. Different bucketing axis, different conditioning, different complexity class — not covered. No contradiction either: the envelope scorer is a ~0.4%-of-16K fixed cost, three orders cheaper than LSA's cluster index, which is exactly why it promotes at 16K while LSA's own crossover sits at ~256k.

**"Covered by RTPurbo (katgpt-rs 086)?"** RTPurbo is a **static per-model calibration** (pre-RoPE low-dim projection + head calibration frozen at load, applied identically to every request). LSA's index space is a **runtime per-context index**: the bucket cache is built per request from the KV cache and scored per query. Conditioning axis: per-model vs per-context. Not covered.

**"HGA (katgpt-rs 379/397) already tried the hierarchy?"** Yes — positional chunk→group→token, G2-proxy FAIL on random-key NIAH, kept opt-in with the negative recorded. LSA's deltas: content clusters (not positional), trained/PCA projections, 10M-scale evaluation. And LSA's **own tree arm loses accuracy past 1M** (finding 3) — this upgrades our HGA negative from "our implementation failed" to a class-level law: *hierarchy trades accuracy for subquadraticity; the flat content-cluster variant is the accuracy-preserving shape.* Future hierarchical attempts should cite both.

### 2.3 Fusion

**LSA × the landed regime router (bench 933, `regime_centroids.json`) × the envelope arm.** The sparse lane now has a per-input regime gate (route flow-side vs sparse-side, calibrated, BLAKE3-frozen). LSA adds two axes the router does not carry: the **within-sparse axis** (what the sparse side selects with — content buckets vs positional blocks) and the **context-length axis** (the ≥~256k crossover). A future tier keys the bucket index off the regime router's sparse side at long context; the PCA variant is its modelless floor.

**LSA × N-LSA × our league family.** N-LSA's pretraining topology (every 4th layer attention, Gated DeltaNet elsewhere, no positional encoding, single KV head, max-over-heads bucket scoring with stop-grad selection) is the same hybrid class as qwen3.8. If the league model family ever gets a native-sparse long-context next generation, this paper is the recipe of record — and its negative controls (NSA/PISA/SWA all fail to match FA at 128k) are the bar to beat.

**The monotone crossover law, three scales.** Healer Bench 126: routing hierarchy over a ~600-rule corpus costs +9.5 ms (NEGATIVE, flat retrieval wins). LSA: below 32k buckets, write-S beats search. HGA: positional tiers fail accuracy at league scale. Same monotone shape everywhere: *hierarchy pays only above a corpus-dependent knee; below it, flat scan + top-k is optimal.* Worth quoting wherever a "let's index it" proposal appears.

---

## 3. Verdict

### 3.1 Tier — per track

| Track | Verdict | Reasoning (one line) | Output |
|---|---|---|---|
| Modelless / substrate (riir-infer) | **Distillation RECORD** (no novelty claim — the class is saturated, §3.3) | Ships + informs the live sparse lane's next tier + carries the crossover law and the 10M SOTA reference; but zero gain is provable at league shapes ≤32k, so no implementation, no flag | this note |
| Self-adaptive | none | no latent-state update surface | — |
| Model-based / training | N-LSA pretraining **out of scope** (full-pretraining charter); D-LSA distillation = unexecuted cousin of Plan 476 | Two unexecuted indexer-training plans would be noise; Plan 476 stays the single owner-gated lane of record and this note is its second citation | reference rows only |

### 3.2 Owner-gated next lane (recorded, not filed)

**Weakest point of this record, named:** its actionability is thin — there is no executable gate today at any shape the league pins, and the 10M posture is unaffordable on our hardware. The note is still the durable artifact because it is the design input for the post-promotion iteration of the live sparse lane (issue 1165's next round at long context) and carries three facts that do not rot with posture: the crossover law, the model-family SOTA reference, and the HGA corroboration.

Arming a **≥256k (or 1M) league cell** plus a content-cluster bucket-index arm would be a league-manifest expansion (changes the fairness posture vs llama.cpp, whose cells are also pinned ≤32k) and needs hardware the stack does not point at that posture (the 10M KV cache is ~10 GiB/layer under CP8 — not affordable on a 24 GB 4090 or Metal boxes). If a long-context consumer materializes (agentic decision serving at 1M+), this note is the design input: flat content-clusters, d(idx)=64, PCA floor first, trained projections only if the floor misses, tree only past ~10M. This is NOT a deferred benchmark — there is nothing to bench until the arm exists, and a gate today is a green zero.

### 3.3 §4 published prior-art landscape (no novelty claimed)

The paper's own Table 1 enumerates the class (token/block/semantic sparse; free/distilled/native; flat/fixed/tree) and a DDG sweep (2026-11-03) confirms saturation beyond it: HSA-UltraLong (ACL 2026, 16M accuracy / 256k perf), DHSA (ICML 26 spotlight, dynamic hierarchical), LycheeCluster (ACL 26 findings, triangle-inequality hierarchical index), Token Sparse Attention (ICML 26), DisSparse (NeurIPS 26), MTraining (MLSys 26 oral, distributed sparse training), Leng et al. ICLR 26 (length generalization in hierarchical sparse attention, arXiv:2510.17196), Unified Sparse Attention with Hierarchical Memory (arXiv:2604.26837). Q1 = NO at class level by construction; the durable value here is the SOTA reference point on **our league's model family** + the crossover/accuracy laws, not mechanism novelty.

### 3.4 Honest caveats

- Headline speedups are H200/B300 (WGMMA/tcgen05, 4.8–8 TB/s HBM). Our lanes are Metal (M3/M5, tuned ktile64/GEMV-occupancy dense stack) and 4090 — the memory-bound decode select would land differently, and the CuTe DSL kernels have no Metal path.
- D-LSA was distilled on GLM-5.3-Flash **from attention that is already sparse**, without finetuning the base model — the fair comparison favors the trained-together baseline, which the paper says openly (≤8 pts RULER at 2M–5M).
- N-LSA is 0.5B, flat-search-only, 128k max — scaling native training to 27B/1M/tree is the authors' own open problem.
