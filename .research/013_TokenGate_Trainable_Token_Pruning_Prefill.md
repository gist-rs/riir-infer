# Research 013: TokenGate — Trainable Token Pruning for Prefill (Substrate Extraction)

> **Source:** "TokenGate: Adapting LLMs for Fast and Memory-Efficient Prefill with Learned Token Pruning" — Patel, Ranjan, Stoica, Zaharia, Guestrin (Stanford/Berkeley), arXiv:2610.26754, 2026-10-19. Code: github.com/token-gate-llms/TokenGate (Apache-2.0).
> **Date:** 2026-10-22
> **Status:** Active — verdicts filed, plans open
> **Related Research:** riir-infer 003 (DSA megakernel verify path), 007 (Triadic GDN tensor state); katgpt-core `ssmax` (the in-substrate name for scalable softmax)
> **Related Plans:** riir-infer Plan 622 (key-bias kernel + pruned-prefill lane — PRIMARY), riir-train Plan 469 (Kimi-K3 arch-test of the training recipe — SECONDARY), riir-infer Issue 1107 (batched-GDN pad-prefix state no-op audit)
> **Classification:** Public (substrate-level extraction; no serving-posture or product content)

---

## TL;DR

TokenGate adds a small gate module at transformer layer L: an MLP scores each token's post-attention residual, top-K keeps the important tokens, and every layer above L processes only those (no KV entries for pruned tokens) — 1.4–11.3× TTFT and 1.5–3.4× peak memory at no accuracy loss. The training trick ("token-gated attention") folds the gate value into attention as an additive per-key logit bias plus a learnable per-query rank-ratio temperature, which makes the non-differentiable top-K trainable end-to-end with plain next-token loss (MoE-router analogy).

**Distilled for this workspace (modelless, inference-time):** the paper's entire inference-time structure is modelless — the only trained artifacts are three small parameter families (one readout row, per-(layer,head) magnitudes, per-(layer,head) temperatures), and the paper's own repo derives two of the three in closed form from pretrained weights (measured: the derived magnitudes land in the same range the learned ones converge to). What we extract: (1) the **per-key × per-head additive attention-logit-bias kernel capability** (absent from every workspace kernel today; subsumes ALiBi, attention sinks, DASH-class selection, and this paper's gate into one kernel parameter), (2) the **exact pruned-prefill serving quartet** (pad-bias buckets, CUDA-graph-per-length, decode-time constant fold, side-stream KV gather — all closed-form with exactness proofs written in the source), (3) the **deterministically-constructed gate overlay** (seq-RMSNorm scores + qk-norm-derived magnitudes + `1 + γ(r−1)` rank cooling with γ=1 = provable exact no-op at K=full) — a legal modelless overlay with an honest, measurable quality band. The training recipe is real and cheap on the arch-test model → riir-train Plan 469.

**Pinned novelty claim (the §1.5 precondition form):** a *per-key×per-head additive attention-logit-bias kernel + exact pruned-prefill serving quartet + deterministically-constructed token-gate overlay* for *the riir-infer qwen35-hybrid serving lane*, consuming *layer-L mixer-output residuals and pretrained qk-norm gains*, distinguished from *TokenGate's trained gate* by *zero training (deterministic construction; floor = random-K, target = FastKV-class selectors; trained-gate accuracy explicitly not claimed)*, from *katgpt-speculative PrefillScorer* by *per-token top-K with bias-steered attention inside the serving path (not block-span draft-side compression)*, and from *MoBA/SeerAttention* by *token-level prefill MLP-compute + KV-memory reduction (not block-level attention sparsity — all tokens still traverse every layer there)*.

---

## 1. Paper Core Findings

- **Forward pass:** layers ≤ L full-context; gate module scores normalized post-attention residuals `ã_L` (MLP → sequence-RMSNorm → importance `b_j`, forced 0 on non-prunable scaffold tokens); top-K keeps K prunable + all non-prunable; layers > L see only kept tokens and write no KV rows for pruned ones.
- **Token-gated attention** (the enabler): `α_ij = softmax(τ_i · (q·k/√d + softplus(γ_ℓh)·b_j))` — gate value as additive per-key per-head bias; per-query temperature `τ = softplus(σ)·log(m_i)/log(max(2, p_i))` where m_i = the query's rank among kept tokens (SSMax-inspired; the code form is `s_eff = 1 + γ·(r−1)`, r = log m / log p, so r ≡ 1 at full context — exact base model for any γ).
- **Training:** LoRA r16 α32 on q/v of all layers + gate params, plain CE only, two budgets per step (K_full + K_prune ~ LogUniform(1, |P|/2), losses averaged; lower forward shared; 1.3×/step, total ≈ 0.4–1.4× SFT). Gate MLP zero-init ⇒ b = 0 ⇒ gated attention degenerates *exactly* to standard attention at start. Muon for gate matrices, AdamW for the rest.
- **Results (Qwen3-1.7B, Qwen3.5-2B hybrid):** 1.4–11.3× TTFT, 1.5–3.4× peak memory at no accuracy loss vs full-context LoRA-SFT; up to 6.4× TTFT over the best training-free baseline (FastKV) at matched accuracy. Hybrid (GDN layers + full attention every 4th): gate at GDN layers reads the GDN mixer output; gated attention only on full-attention layers; GDN state above L built from kept tokens only, layers ≤ L keep full-prompt state; gate at L0 too weak, L≥1 sufficient.
- **Kernel:** FlashAttention-2 patched to accept a per-key per-head additive bias before the online softmax (1.0–1.1× cost); temperature applied as a query multiply; decode folds the (prefill-constant) gate into a per-layer additive mask; pruned forward = one CUDA graph per padded length with −1e9 pad bias; KV gather of layers ≤ L to kept positions on a side stream.

---

## 2. Path 0 Inventory (per-track, condensed)

Full 19-row advocate inventories are summarized; every row ends in exactly one disposition.

| # | Component (paper) | Modelless coverage / extraction | Disposition |
|---|---|---|---|
| 1 | Per-key per-head additive logit bias (kernel) | Kernel parameter, not a weight; forward-only port suffices for serving | → Plan 622 Phase 1 |
| 2 | Rank-ratio cooling law `1 + γ(r−1)` | Closed form; γ=1 is exact no-op at K=full | → Plan 622 Phase 3 |
| 3 | Decode-time gate folding (constant per-key mask) | Pure runtime algebra; key ratios frozen at prefill | → Plan 622 Phase 2 |
| 4 | −1e9 pad-bias bucket exactness | Closed form; bit-exact for real rows at any bucket | → Plan 622 Phase 2 |
| 5 | CUDA-graph-per-padded-length pruned forward | Extends the existing `forward_tokens_graph_walk` opt-in | → Plan 622 Phase 2 |
| 6 | Side-stream KV gather | Stream/event scheduling | → Plan 622 Phase 2 (optional) |
| 7 | Sequence-RMSNorm score normalization | Standalone scoring primitive (normalization, not a weight) | → Plan 622 Phase 3 |
| 8 | QK-norm-derived magnitude table `s_l = ‖g_q∘g_k‖₂/√d` | Derived from pretrained weights; repo-measured to land in the learned range | → Plan 622 Phase 3 |
| 9 | Trained MLP scorer | Hand-designed scorer family (attention-mass tap / hidden-norm / all-ones) with honest quality band: ≥ random-K floor, FastKV-class target, trained-gate accuracy NOT claimed | → Plan 622 Phase 3 |
| 10 | Gate≡0 identity (zero-init ⇒ exact standard attention) | Test law: `None`/all-zero bias byte-identical by dispatch | → Plan 622 G1 gate |
| 11 | Layer choice (L0 insufficient, L≥1 sufficient) | Design constraint on scorer placement; measure, don't copy | → Plan 622 Phase 3 |
| 12 | Log-uniform budget-sampling eval protocol | Benchmark methodology (mandatory no-prune arm; selection-only diagnostics) | → Plan 622 Phase 4 |
| 13 | Hybrid placement rules (mixer-output scoring; bias only on full-attn layers; GDN state split ≤L / >L) | Structural, zero weights; our league family IS this hybrid class | → Plan 622 Phase 2 |
| 14 | Pads-to-front state no-op for batched recurrent layers | Closed-form exact; repo-measured defect class when absent (0.679 vs 1.0 keep-all boundary) | → Issue 1107 |
| 15 | Co-adapted-backbone pruning robustness | NOT shippable modellessly (forbids base-weight mutation); ships as a measurement row (random-K census, ternary-vs-dense) | → Plan 622 Phase 4 (bench row); mechanism discarded — audited below |
| 16 | Kernel-side temperature as query multiply | Folded into #1/#3 | → Plan 622 |
| 17 | Static-address attribute contract for captured graphs | Engineering law for #5 | → Plan 622 Phase 2 |
| 18 | LoRA + gate + CE + Muon + dual-budget training recipe | Training track — real, cheap on the arch-test model (6–12 GPU-h), impractical on the 27B league model (~100–300 GPU-h + missing ternary-MoE backward kernels) | → Plan 469 (27B deferred `- [-]` inside it) |
| 19 | Trained accuracy-at-K tables (incl. 6.4× over training-free) | Outcomes of the trained selector; not claimable modellessly | Discarded — audited below |

---

## 3. Fusion — closest cousins and what the combination produces

- **`katgpt-rs/crates/katgpt-speculative/src/prefill.rs`** — `PrefillScorer` + `compress_prompt`/`block_select` + `attention_sink` keep-always: the workspace's only prompt-compression substrate, unlearned, block-span, draft-side (speculative lane, not wired into riir-infer serving). TokenGate's delta: per-token top-K + bias-steered attention + the exact serving quartet, inside the serving path.
- **`riir-infer/src/transformer/attention_si.rs`** — scale-invariant attention, an affine transform in logit space (per-(layer,head) multiplicative + additive), header names the FlexAttention `score_mod` shape; key-INDEPENDENT. The per-key bias is the missing axis; the sink carve-out precedent already exists there.
- **`katgpt-rs/crates/katgpt-transformer/src/moe.rs`** — DeepSeek-V3 noaux_tc: sigmoid affinities + additive per-expert bias used for SELECTION ONLY (bias excluded from renormalization, G1-pinned), gradients past a non-differentiable top-K. The same mathematical shape as TokenGate's gate, on the expert axis. Token-gated attention is that pattern re-aimed onto tokens.
- **`riir-infer/src/deltanet/kv_evict.rs`** — differential KV eviction with sink-exempt selection on the qwen35-hybrid path, bit-identical no-op at full budget. Sibling machinery; TokenGate prunes before the cache exists instead of after.
- **`riir-infer/src/deltanet/`** — the qwen35-hybrid forward (gated full attention interleaved with DeltaNet, `layer_types` dispatch, `prefill_qwen_deltanet_chunk_into`): exactly TokenGate's host architecture class, already served.
- **`katgpt-core ssmax`** (feature `ssmax_temperature`, default-on) — the substrate's scalable-softmax lane (log(N) global temperature + Lean theorem). TokenGate Eq. 7 is SSMax applied to the kept/original count ratio — a per-query application our ssmax lane does not have.

**Fusion product:** a bias-steered, deterministically-scored prompt-skimming lane on the hybrid serving path, sharing kv_evict's exactness discipline (bit-identity at the inert posture by dispatch, not arithmetic luck) — a capability none of the cousins has: PrefillScorer has no bias steering and never runs in serving; attention_si has no per-key axis; moe.rs steers experts, not tokens; kv_evict cannot reduce prefill compute.

**Published prior art (§4 sweep, 10 searches):** no single published work combines (a) drop-in end-to-end trainable token pruning with no auxiliary loss, (b) inference-time budget generalization, (c) hybrid-attention support. Ingredients each published: MoD/NSA (from-scratch trainable sparsity), MoBA (block-level trainable gate inside attention softmax on pretrained models), SeerAttention/-R (self-distilled block-level gate on frozen models), SSMax (the temperature formula), FlexAttention (score_mod kernel model), FastKV/SlimInfer/DASH/GemFilter (training-free baselines — all four verified training-free; FastKV strongest). TokenGate's own intersection appears genuinely novel in print; the GDN/hybrid extension and the budget-sampling recipe have no published prior art found. Concurrency caveat: 4-day-old paper, zero citations — a citation sweep at a later date closes it.

---

## 4. Verdict — one per track

| Track | Verdict | Reasoning (one line) | Output |
|---|---|---|---|
| **A. Modelless substrate (riir-infer)** | **GOAT (Gain disposition — PROVISIONAL until Plan 622 T4.1 measures the break-even)** | The mechanism is published (paper + Apache-2.0 code) so this is a distill, not a novel class; but the per-key-bias kernel is absent from every workspace kernel, the serving quartet is scorer-agnostic exact wins, and the deterministic-construction overlay is a delta the paper does not ship. The TTFT win is UNPROVEN at our postures: no workspace consumer serves >4k contexts today, and league pp2048/pp4096 sit at the low end of the paper's 1.4–11.3× range — T4.1 must report the (K, length) break-even cells before any gain is claimed | riir-infer Plan 622 (PRIMARY — runs in the serving hot path; feature `tokengate`, never default) |
| **B. Training recipe (riir-train)** | **GOAT** | Recipe is published; applicability is proven on the K3 arch-test lane (~6–12 GPU-h) but no >4k consumer exists in the workspace today (only league benches touch long contexts), and any gate is lossy at K<full ⇒ opt-in forever under the lossy law | riir-train Plan 469 (SECONDARY — serving-envelope fit defers to Track A; says so in its Status) |
| Super-GOAT | **Not awarded** | Requires no prior art for the mechanism itself; TokenGate is the prior art. Our deltas (modelless overlay, in-workspace kernel first) are real but scoped below the trained gate's headline claims | — |

**Serving-envelope fit (TTPO rule):** Track A is primary because its decision layer (pruned prefill) runs inside the serving hot path and needs no training; Track B produces an artifact that would ride Track A's substrate and is ranked secondary. Neither promotes to default: lossy-surface law (the Issue-750-T3 walk precedent — 5/54 argmax flips refused default-on for far smaller divergence) keeps any token-pruning posture opt-in permanently.

**MOAT gate (§1.6):** riir-infer — public substrate moat, fits (kernels/forwards; upstream-clean, no riir-* deps added). riir-train — active moat, fits (training-method plan). No public/private violation: this note and Plan 622 carry only substrate-level, public-safe content; any league-model gate checkpoint (if 469 ever graduates) stays private in riir-train.

---

## 5. GOAT gates (bind Plan 622 and Plan 469)

- **G1 identity:** `bias = None` AND all-zero bias byte-identical to the incumbent path across league shapes (by dispatch, not float luck); bucketed-vs-unpadded byte-identity at every bucket; K=full collapse identity for the overlay (γ any value).
- **G2 perf:** biased call ≤ 1.10× plain at matched shapes; TTFT-vs-K on the league posture with box provenance (`bench_preflight.sh` PROVENANCE line; GPU-exclusivity per the correctness-gate rule).
- **G3 no-regression:** feature-off build byte-identical (the feature is the compile-away off-arm; the env `RIIR_TOKENGATE` is a runtime demote INSIDE the feature); the plan's own posture changes never touch the default path.
- **G4 alloc:** gate forward + top-K gather + r-ratio computation zero-alloc (preallocated scratch).
- **Quality floor (the honest band):** derived scorer ≥ random-K on every acc-vs-K cell (GOAT pass), FastKV-class selectors = the target band measured as a fourth arm on the same seam (reported, non-gating), trained-gate accuracy explicitly not claimed. Four arms mandatory in every table: no-prune ceiling / random-K floor / derived scorer / FastKV-class selector. Per the owner pivot rule the harness runs on the **Bonsai/ternary league posture FIRST** (the seam lands in the shared attention kernels both arches use), qwen35-hybrid second; T4.1 must report the (K, length) break-even cells where quartet overhead exceeds savings.

---

## 6. Discard audit (advocate findings not filed, with reasons)

- **Co-adaptation as a runtime mechanism (advocate row 15):** discarded — the modelless law forbids base-weight mutation, so no runtime analog can exist; the measurable residue (quantization-aware training as a co-adaptation proxy) ships as the Plan 622 Phase 4 random-K census row instead.
- **Trained accuracy-at-K claims (row 19):** discarded — they are outcomes of the trained selector; claiming them modellessly would repeat the exact §3.6 architectural-only-PASS failure. The floor/target band in §5 is the honest substitute.
- **27B league-model training lane:** deferred `- [-]` inside Plan 469 — ~100–300 GPU-h plus missing ternary-MoE backward kernels on this box, and opt-in-only under the lossy law even if it wins; revisit only after a real >4k consumer exists.

---

## 7. Provenance

- Clone: `riir-infer/.raw/TokenGate` @ `6e51aac5be1bd84bfd747f43f3f67ed595072bc2`, Apache-2.0 (upstream repo `token-gate-llms/TokenGate`). Quotes above (qk-norm magnitude derivation and its measured learned-range convergence; `s_eff = 1 + γ(r−1)` code form; `call_layer_padsafe` exactness; decode-fold "Exact, not an approximation"; −1e9 pad-bias exactness) verified at that sha by the panel agents. Clone removed at commit time per the `.raw` lifecycle rule.
- Internal-first grep: no prior workspace `.research/`/`.plans/` mention of arXiv:2610.26754 or TokenGate (the single `riir-refine/.research/108` grep-`-l` hit was verified spurious by direct re-grep).
- Panel: No-GD advocate + model-based advocate + codebase fusion grep + published prior-art search ran in one parallel batch (2026-10-22); verdict negotiated with the Claude reviewer (§5 of the skill) before commit.
