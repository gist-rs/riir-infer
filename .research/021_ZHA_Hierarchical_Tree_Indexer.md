# Research 021: ZHA — Zyphra Hierarchical Attention (Subquadratic Tree-Structured Indexer)

**Status:** RECORD (substrate/league track) + GAIN filed (model-based track → riir-train `.issues/666_zha_reference_query_distill_poc.md`) — no GOAT claim: the class is saturated, the specific combination is the paper's own (published 4 days ago), and zero gain is provable at league shapes by the crossover law. Design rows hand off to Plan 627 (CLAIMED `zed-m3-lsa-distill`, expires 2026-11-10 — yield; this note is input, not competition).

> **Source:** "Subquadratic Attention with a Hierarchical Tree-Structured Indexer" — Lufkin et al. (Zyphra), [arXiv:2610.42112](https://arxiv.org/abs/2610.42112) (v1 2026-10-30, v2 2026-11-02, 42 pp). Blog: zyphra.com/our-work/zyphra-hierarchical-attention. Code: none published (no repo found; no HF page; no citations yet as of 2026-11-03).
> **Date:** distilled 2026-11-03
> **Related Research:** riir-infer 020 (LSA — the sibling record: crossover law, tree-degrades law, 1M owner pull) · riir-infer 019 (Elastic — input-conditional budgets) · katgpt-rs 595/Plan 612 (PISA pyramid — **CLOSED G2 iso-quality NEGATIVE**, Bench 612) · katgpt-rs 379/Plan 397 (HGA — **G2-proxy FAIL**) · katgpt-rs 071/Plan 106 (DashAttention — single-level entmax, default-on) · katgpt-rs 225 (MSA — **GOAT FAILED**) · katgpt-rs 086 (RTPurbo — query-dependent dynamic top-p, 16-dim indexer) · katgpt-rs 044 (PFlash) · katgpt-rs 028 (HLA — the bounded-state endpoint; ZHA is the sublinear-growing-state endpoint)
> **Related Plans/Issues:** riir-infer Plan 627 (GLM-5.3-Flash 1M lane — **filed + claimed**, this note's primary consumer) · riir-infer issue 1165 (τ=0.85 envelope promotion, AGREE — the one sparse WIN on our stack) · riir-train Plan 476 (Elastic router gate — SECONDARY, filed NOT executed; its deferral premise cites Issue 1156, which **CLOSED DONE** `e0984aa8` — premise stale, recorded here, not edited) · riir-train `.issues/666` (the Gain output of this note)
> **Classification:** Public (substrate-level; no serving-posture or product content)

---

## TL;DR

ZHA is the counter-design to the failure mode our own bench record keeps producing. A tree of **recursively compressed key summaries** over the KV cache (chunks → re-chunked summaries → up to 6 levels); a query traverses **top-down with top-p routing** (at each level expand the smallest node set covering a fraction *p* of the **estimated** attention probability); summaries are built from *R* learned **reference queries** (81,920 params on a 35B MoE ≈ 0.00024%) trained by **KL distillation against the frozen dense model's attention** (1B tokens @32K). Claims: top-p set size grows **sublinearly** (power law) with context; **deeper trees often beat shallower at fixed compression** (28-geometry sweep); quality parity at ~1.1% KV touched on reasoning + agentic coding (SWE-bench 67.4 ≈ dense 67.4; LongBench v2 63.4 vs 58.0; RULER-13@256K 96.2/93.3 vs 96.0) while beating MiniMax-MSA's 40B-token indexer with ~20× less training. Roofline: > dense ≥48K ctx, > MSA ≥480K, 32–35× @10M, 373–417× @1B (projected, H200-class).

**Why this note exists:** our stack holds **three in-house negatives** in exactly this class — MSA (R225 GOAT FAIL), HGA two-level (R379 G2-proxy FAIL), PISA pyramid (R595 Bench 612 −0.133 Recall@8 vs single-level exact-LSE) — plus LSA's class law ("hierarchy trades accuracy for subquadraticity"; its own tree arm dies past 1M). All three negatives share two confounds: **untrained/deterministic summaries** (max-pool, mixed-RoPE mean-rule, mean+LSE) and **fixed budgets** (K_c/K_g, Top-K). ZHA removes **both confounds simultaneously** (per-level KL-distilled tangent summaries + mass-adaptive retention). That is the one falsifiable experiment the record has not run: if the ZHA recipe still loses at our shapes, the class law gains a training-robust corroboration; if it wins, the knee moves. The training half is cheap (no model backward — see `.issues/666`), so the experiment is affordable.

---

## 1. Paper core findings

1. **The indexer is the wall; the tree makes *searching* sparse.** Sparse attention moved the quadratic cost into finding keys (LSA 020 finding 1: 98% of prefill at 10M). ZHA's move: never scan the whole index at one resolution — each level eliminates large regions without examining their contents. Per-query routing cost is sublinear provided the number of explored regions grows sublinearly.
2. **Top-p routing** — the smallest set of nodes whose cumulative *estimated* attention probability ≥ *p*, per level. Adaptive budget: concentrated queries explore few branches, diffuse queries more. (Priors per our sweep: Twilight 2502.02770 hierarchical top-p — cited by ZHA; Double-P 2602.05191 two-level top-p — NOT cited; SpotAttention 2606.22874 frozen+KL+dual-top-p FLAT selector — NOT cited; RTPurbo 2605.16928 dynamic query-dependent top-p — our 086 cousin.)
3. **Reference queries, trained tiny.** *R* shared queries per KV group compress each chunk into a tangent score `s = max_r(q·k̂_r/√d + b̂_r)`; `k̂_r = Attn(q̂_r, K_C)`; `b̂_r` = per-chunk entropy bias. KL distillation, base frozen, 1,900 steps ≈ 1B tokens @32K document-masked (App. B-A details agent-fetched: AdamW (0.9, 0.95), clip 1.0, LR 3.2e-2, WSD, R16/KV-group, **query-LoRA ablation NEGATIVE** — plain queries won). At league scale (hybrid-GDN, attention on ~¼ of layers) the trainable surface is ≈ 12k–197k params.
4. **Sublinear growth law** — top-p set size ~ power law in context length (measured on Qwen3.6-35B-A3B over long fiction). As far as the sweep found, **the measurement itself is original to ZHA** ("Softmax is not enough" 2410.01104 is the dispersiveness prior, no measured law). ZHA's own honesty: RoPE-dependent — NoPE (Nemotron) breaks it.
5. **Deep-tree robustness** — at fixed total compression C^L, deeper ≥ shallower (6-level trees fine); "first hierarchical attention variant demonstrated at depth > 1 with close parity to dense" — uncontested in the indexed literature as of 2026-11-03. **Directly challenges** the LSA tree-degrades law (measured at 1M–10M on GLM with positional/untrained trees) — ZHA's accuracy evidence stops at 256K, so the challenge is untested at the scale where the law was measured.
6. **Quality at high sparsity on hard tasks** — RULER-13@256K 96.2 (wide) / 93.3 (deep) vs dense 96.0; LongBench v2 63.4 deep vs 58.0 dense; SWE-bench Verified 67.4 ≈ dense at ~1.1% KV touched; vs MSA's 77.1 RULER despite MSA's 40B-token indexer (10× the training). Data efficiency is the proven claim.
7. **Roofline model** — detailed GPU cost model (routing + attention + bandwidth + launch); crossover vs dense ≈ 48K, vs MSA ≈ 480K; 32–35× @10M; 373–417× @1B. Projection, not measurement, past ~1M.
8. **The "narrow path"** — per-level retained mass compounds (p_eff = p^L): too sparse collapses the active state toward fixed-size (the SSM failure), too dense regresses to quadratic. The routing threshold must evolve with depth and length to track the (sublinear) growth of predictive information. Closed-form consequence: feasibility requires r_l ≥ P^(1/L) per level.

---

## 2. Distillation

### 2.1 Path 0 inventory

| Component | Ships here? (coverage) | Modelless-extractable? | Disposition |
|---|---|---|---|
| Tree of compressed summaries + coarse-to-fine walk | **YES** — `pyramid_topk.rs` (R595/Plan 612: mean-pool exact-subtree summaries + LSE + Top-K, root-seeded, O(gK log N)) + HGA two-level (R379, G2-proxy FAIL) | yes | covered; the deltas are the retention rule + summary construction (rows below) |
| Top-p mass retention per level | **NO** — shipped retention rules: entmax α-support (single-level, Plan 106), Top-K per level (pyramid), τ-threshold (envelope, issue 1165), fixed K_c/K_g (HGA) | **YES** — cumulative-mass cut on per-node scores; closed form; ~small delta on `pyramid_topk`'s retention step | design row for Plan 627's select-rule A/B (third retention rule vs τ and Top-K); **no build now — lane claimed (yield)** |
| Trained reference-query summaries | **NO** — pyramid = mean-pool; `dash_attn` `head_cls` zero-init ⇒ mean-pool; HGA = deterministic mixed-RoPE | **NO** (needs GD) — but **PCA directions of the offline key covariance are the closed-form floor** (020's PCA-floor convention; the No-GD advocate's modelless twin) | → riir-train `.issues/666`; consumer seam = `dash_attn` `ChunkSummaryQuery` (zero-init ⇒ byte-identity G3 free) + Plan 627 PCA-floor-miss fallback |
| KL distill (chunk-pooled target, tangent softmax, **no model backward**) | **Cousin** — D-LSA (020): KL to head-aggregated attention, trains index projections W^idx + τ | no | 3rd citation of the tiny-KL-distill indexer lane (020 was 2nd; PISA's riir-train 586 same family); Plan 476 = lane of record; the no-model-backward property is what makes it PQ2-compatible (Plan 476's deferral era assumed a blocked class) |
| Sublinear growth-law measurement | no | **YES** — offline least-squares β fit over committed attention captures (spike-census sidecars / harness maps); no serving risk | design row: geometry + threshold-schedule input; needs ≥256k captures to fit at the interesting scale (none exist yet — a gate today is a green zero) |
| Per-level schedule p_l = P^(1/L) + tripwires | no | **YES** — closed form; collapse tripwire = support-size variance < ε across queries; dense-regression tripwire = Σ support → N | design row (anti-collapse law for any multi-level arm) |
| Roofline crossover predictor | **Partial** — the 32k-bucket crossover law ships as a recorded law (020 finding 4); no cost-model formula | **YES** — closed-form C(N) = walk bytes + selected-attention bytes vs dense bytes | pre-registration band for any future pp256k/pp1M cell (predict N*, bench must land in-band) |
| KV-touched-% + paired NLL as a reported pair | **Partial** — FNV/argmax digit-exact lanes exist; touched-% not a first-class column | **YES** — protocol only | report-gate row: a sparse latency cell without (latency, touched-%, paired drift) is an unproven claim |

### 2.2 §3.6 signal-diffs on the coverage dismissals

- **vs `pyramid_topk` (R595, the closest shipped tree):** retention rule **Top-K fixed** vs **mass-fraction adaptive**; summary **mean-pool + LSE-over-mean** vs **trained tangent softmax from R shared queries** (ZHA's summaries answer "how relevant is this region to the query", not "what does it contain"); no growth law, no depth schedule, no training. Bench 612's negative tested the untrained-mean-pool arm at iso-quality on 32K+ — ZHA's two deltas are precisely the untested remainder. Not covered.
- **vs `dash_attn` entmax (R71/Plan 106, default-on):** single-level chunk→token; α controls support size globally (not a per-level probability fraction); summaries feed the output softmax via prior-induced bias, ZHA summaries are route-only (strict "summary-keys route, real-keys compute" — HGA's rule). Not covered.
- **vs HGA (R379/Plan 397):** two-level **fixed budgets** + **deterministic mixed-RoPE summaries** → G2-proxy FAIL. ZHA = trained per-level summaries + mass-adaptive retention. The unblock hypothesis — unproven at ≥1M where LSA's tree-degrades law was measured. Not covered; carries the falsifiable claim.
- **vs LSA D-LSA (020):** same distill family (frozen base, KL to attention). Diffs: target = **head-aggregated attention vs chunk-pooled**; parameterization = **index projections W^idx + τ vs R reference queries + entropy bias**; corpus 352M vs 1B tokens. ZHA's own ablation (query-LoRA negative, plain queries) + our zero-init seam favor queries for the PoC.

### 2.3 Fusion — the two live surfaces

- **F1 — Plan 627 (CLAIMED, yield):** the 1M lane's design space gains five rows from this note: (i) top-p as the **third retention rule** A/B vs threshold-τ (shipped winner) and Top-K; (ii) the p_l = P^(1/L) schedule + collapse/dense tripwires; (iii) growth-law β as the sizing input (needs ≥256k captures); (iv) crossover band pre-registration (predict N*, land in-band); (v) the ZHA-trained-tree arm **only if** the flat PCA-floor bucket index misses quality at pp1M — LSA's flat arm is design-of-record and its tree arm lost past 1M; the ZHA recipe is the one confound-free challenger.
- **F2 — riir-train `.issues/666` (filed this session):** the reference-query KL-distill PoC with pre-registered proof path (grad-check vs FD; 30–100M-token gate corpus declared before the run; **two modelless floors to beat** — zero-init mean-pool incumbent and `msa_distill` scorer; per-family conditional retention per the lossy-surface promotion rule + bench-889 continuation-ppl bands; R ∈ {1,4,16} arms). Execution owner-gated (4090 window / bmo queue). Promotion corpus (1B) is a separate, queue-scale document.
- **F3 — the knee law (recorded, no filing):** hierarchy pays only above a corpus-dependent knee — healer ~600 rules (Bench 126 negative), LSA 32k-bucket crossover, ZHA 48K crossover, our three in-house negatives at league shapes. One monotone family. The artifact is the **trigger bound**, not a bench: re-run the healer RuleIndex A/B only when corpus × query-cost crosses the log-depth saving (a green-zero gate today is noise, per 020's own discipline).

### 2.4 Latent reframe (brief)

- **Episodic memory (riir-neuron-db):** tree-of-summaries over sightings + smallest-covering-set retrieval = a consolidation schedule; the AnyRAG escalation predicate falls out free (diffuse top-level mass → exhaustive scan). No documented gap at NPC memory scale — recorded, not filed.
- **NPC zone beliefs:** two-level salience-mass retrieval over zone summaries; the limelight stays the cost tier (HOT/WARM/COLD = how often), the tree adds which-to-think-about (selection); sigmoid drives only, no hard edges. Recorded.
- **DEC operators:** n/a — d = 64+ attention keys violate the curse-of-dimensionality caveat (same verdict as R379 §2.5(f)).
- **HLA relationship:** complementary endpoints of the long-context axis — HLA = bounded O(d²) state; ZHA = sublinear-growing active state over full retained KV. The league lane serves both; no fusion target today.

---

## 3. Verdict — per track

| Track | Verdict | Output |
|---|---|---|
| Modelless / substrate (riir-infer league lane) | **DISTILLATION RECORD** — zero gain provable at league shapes (crossover law; three in-house negatives); the 1M lane is claimed by a sibling session (yield); this note is the design input + the class law's falsifier design | this note; design rows ride Plan 627 through it |
| Self-adaptive | none live | threshold-schedule harvest→freeze→swap (BLAKE3 artifact, atomic swap) recorded in §2.1 only — no consumer |
| Model-based / training | **GAIN** — actionable, cheap, unlocks a deferred lane family | riir-train `.issues/666` (proof path pre-registered; Plan 476 = lane of record — deferral premise stale (Issue 1156 closed DONE `e0984aa8`), recorded here, Plan 476 not edited) |

**Tier discipline:** NOT GOAT (no provable gain at league shapes; the combination is the paper's own), NOT Super-GOAT (Q1 = NO at class level — saturated landscape: NSA/MoBA/InfLLMv2/Quest/MInference/MSA/PISA/HiP/DHSA/HiLS/Leng et al. 2510.17196/HSA-UltraLong/LycheeCluster/LSA; the specific combination was published 2026-10-30 by Zyphra). Novelty claim: none.

**Verdict gate record:** §5 negotiation run 2026-11-03 via the `spawn_agent` reviewer fallback (claude_code backend rate-limited, resets 20:00 ICT) with the same Summary + `#Verdict:` contract — **#Verdict: AGREE** round 1 (session `1bde4956`); reviewer independently re-verified the code facts (zero-init seam, pyramid/msa_distill existence, top-p absence), highwaters, redirect lines, Plan 627/476/1156 live state, and MOAT routing. One figure the reviewer could not re-verify (Plan 106's GOAT 9/9 tally) is read directly from `katgpt-rs/.plans/106` in this session and is not verdict-load-bearing.

**MOAT check (§1.6):** riir-infer public substrate moat — in scope (attention indexer mechanism + the 1M lane is its live consumer). katgpt-rs gets PASS-Redirect lines only (379/071/595); riir-train gets the PoC issue; no game/chain/shard surface carries an actionable row.

---

## 4. §4 published prior-art landscape (agent sweep 2026-11-03; no novelty claimed)

- **top-p × hierarchy:** Twilight (2502.02770, NeurIPS'25 spotlight — cited by ZHA as the top-p prior); **Double-P** (2602.05191, two-level top-p over cluster centroids, unlearned, no growth law — **NOT cited by ZHA**, sharpest per-component prior); **SpotAttention** (2606.22874, frozen model + KL-distilled selector + dual top-p, FLAT one level — NOT cited).
- **log-linear / learned-summary trees:** PISA (2609.31093 — cited by ZHA, dismissed for retrieval degradation; **we ship its modelless pyramid**, R595); HiP (2406.09827 — cited, degrades); HiLS (2607.02980, learned chunk landmarks, single retrieval level); **Landmark Attention** (2305.16300 — NOT cited); HOMER / RAPTOR / MemWalker (distant: compression trees / RAG, never attention indexers).
- **growth law:** "Softmax is not enough" (2410.01104) = the dispersiveness prior; the sublinear top-p-set power-law **measurement** appears original to ZHA.
- **deep-tree robustness:** uncontested as of 2026-11-03; zero citations/follow-ups to ZHA yet (4 days old; Semantic Scholar rate-limited — negative rests on arXiv listing + HF + HN absence).
- ZHA's own citation gaps (noted for completeness, not action): Double-P and Landmark Attention uncited.

---

## 5. Honest caveats

- **ZHA accuracy evidence stops at 256K.** The 10M/1B claims are roofline projections. LSA's tree-degrades law was measured at 1M–10M — ZHA's challenge to it is a hypothesis at exactly that scale, not a refutation.
- **Roofline crossovers (48K / 480K) are H200-class.** Our lanes (Metal ktile64 stack, 4090, bmo) land differently; the crossover predictor must be re-calibrated per box before any pre-registration band is quoted.
- **NoPE risk does not bind today** (league models are RoPE), but ZHA itself loses 10–25 RULER points on NoPE — a gate row if the league ever serves a NoPE family.
- **Advocate estimates are unmeasured** (harvest throughput, decode KV byte-share, bmo hours) — they are T0 rows in `.issues/666`, never quoted as numbers.
- **Plan 627 yield:** claim `zed-m3-lsa-distill` expires 2026-11-10; the design rows hand off through this note. No competing edits were made to Plan 627, Plan 476, or any `.rs` file this session.
- Paper details beyond abstract/blog (App. B-A recipe, LR 3.2e-2, R16, ablations) are agent-fetched from the HTML v2 and attributed as such; the load-bearing facts (1B tokens, ~82K params, top-p routing, growth law, deep-tree sweep, crossovers) are confirmed by the abstract + blog read directly.
