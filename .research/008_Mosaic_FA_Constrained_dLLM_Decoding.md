# 008 — Mosaic: FA-Constrained Exact Posterior Sampling for the dLLM Decode Lane (paper distill + the serving piece the D2F loop is missing)

**Status:** DISTILLED — pending owner decision (GOAT-tier distillation; consumer exists in-tree — the `gemma2_d2f` denoising decode loop — arming is a feature-gated POC filed as local Issue 035).

Date: 2026-10-05 · filed from the M3, user-directed `@research` session (arXiv:2607.07026 + MhDang/mosaic).

## Source

- Paper: Meihua Dang, Stefano Ermon — **"Constrained Decoding for Diffusion Language Models via Efficient Inference over Finite Automata"**, arXiv:2607.07026v1 (2026-07-08, Stanford). NeurIPS 2026 per the repo README.
- Code: `MhDang/mosaic` **@ `36945e4b3c64913e23962e0924a8b40a84c58807`**, MIT (© 2026 Dang, Ermon). Clone was ephemeral under `katgpt-rs/.raw/mosaic`, removed at close; every quote below re-verified at this pin before deletion.
- Internal readouts: `riir-infer/src/transformer/dllm.rs`, `riir-infer/crates/riir-infer-gpu/src/gemma2_d2f/mod.rs`, `../riir-ai/.plans/108_gemma2_d2f_block_causal_decode.md`, `../riir-ai/crates/riir-gpu/tests/goat_250_d2f_quality_gates_gpu.rs`, `../riir-train/.plans/383_dllm_sibling_trainers_lr_fix.md` (existence), `katgpt-rs/crates/katgpt-core/src/legal_token_set/`.

## TL;DR

dLLMs (Dream-7B / LLaDA-8B class) sample **many positions jointly from a fully-factorized mean-field distribution** at each denoising step; AR-style "mask invalid next tokens" is inapplicable. Mosaic gives the **exact** sampler for `p(x⁰ | x^t, constraint) ∝ Πᵢ p_θ(x⁰ᵢ | x^t) · 1[A accepts x⁰]` for any constraint expressible as a finite automaton: view the FA as an HMM whose edges emit their allowed tokens, run forward–backward over the position × state trellis with the LM logits as emissions, and draw a **joint state path** (not per-position marginals — those do not factorize the joint), then tokens given the edges. Greedy = argmax of the token given its sampled edge; sampling = multinomial. Composes with **any** commit/remask schedule because each step re-conditions on committed positions (`forced` tokens) and redraws free positions from the exact posterior given them. Sampling depth reduced from O(L) to **O(log L)** by arithmetic-circuit depth reduction (a segment tree of pairwise log-space matrix products with top-down midpoint conditional sampling). Measured: BFCL-Live Dream-7B greedy 63.9→71.5, sampling 22.3→69.0 (unconstrained collapses), <5% wall-clock overhead (paper-only number — the repo reproduces neither figure; our GOAT gate must measure its own).

## The algorithm (implementer grade, all refs at the pinned sha)

**FA-as-HMM tensors** (`mosaic/sampling/automaton.py:32-42`): `transition (N,E) 0/1` (edge leaves node), `emission (E,V) 0/1` (edge allows token), `edge_index (E,)` (destination), `accept_node (N,)`, plus derived `edge_lookup (N,N)` = the single edge i→j or −1. Every compiled constraint is deterministic: ≤1 edge per (node, token) (automaton.py:213-215) and ≤1 edge per (i,j) pair (parallel.py:224-227) — this is what makes the gather-based flow exact.

**Per-position emission fold** (`automaton.py:314-330`, via `matmul_a_logb` in layers.py:19-32): `e_logits[b,t,e] = log Σᵥ emission[e,v]·exp(lm_logits[b,t,v])` — one `(E,V)×(V,B·L)` log-space matmul, max-shift stabilized. Committed positions become 0/1 indicators (branchless `torch.where`, no host sync).

**Exact joint sampling — why marginals alone are wrong** (`tests/toy.py::exact_distribution`, `tests/test_samplers.py`): independent per-position draws from the constrained marginals almost never produce a jointly-accepted sequence. Both samplers draw a full **state trajectory** first, then tokens per edge; every produced sequence is accepted **by construction** (`test_samples_are_allowed`). TV-distance vs brute-force enumeration < 0.03 at N=20000 over tiny automata, arbitrary L, forced tokens, batched mixed automata.

**Sequential sampler** (`mosaic/sampling/sequential.py`) — textbook HMM forward–backward in edge space, O(L) depth, O(B·L·E) memory (the low-memory fallback). Per-token weight: `token_weight[b,v] = lm[b,k,v] · Σₑ x[b,e] · back_y[k][b,e] · emission[e,v]` (sequential.py:111-112).

**Parallel sampler** (`mosaic/sampling/parallel.py`) — the paper's headline. Bottom-up: pad leaves to a power of two with **identity leaves** (`eye.log()`; end-weights folded into the last leaf's diagonal), then `Q[k+1] = matmul_loga_logb(Q[k][:,0::2], Q[k][:,1::2])` — pairwise log-space matrix products up the tree, per-level max-shifted (`_build_tree`, parallel.py:286-316). Top-down: sample the first/last boundary states from the root, then per level `z_mid ~ softmax(left[begin→·] + right[·→end])` jointly for **every segment at that level in one kernel** (`_forward_sample`, parallel.py:237-283) — divide-and-conquer conditional sampling; total depth log2 L + 1. Marginals (`_compute_marginal_log`, parallel.py:123-215) rebuild the tree without sampling shifts and fill α/β at every segment midpoint in one down-pass: `p(xₜ=v) ∝ p_LM(xₜ=v) · Σₑ αₜ[src e]·emission[e,v]·βₜ₊₁[dst e]`, cross-checked to 1e-9 vs brute force.

**Matcher surface** (`mosaic/matcher.py`, xgrammar-shaped): `propose_x0(block_logits, block_tokens, mask_token_id) → (x0, marginal_log)` per denoising step — `mask_token_id` marks free positions, anything else is committed/forced; `accept_tokens` walks the automaton over the final block; `start_nodes` = the state the accepted prefix reached, `end_log` = acceptance weights, and `finish_within_log(remaining)` (a one-time reverse BFS `steps_to_accept`) conditions a block on "still completable within the length budget" (automaton.py:248-292).

**Complexity** (B batch, L block length, N states, E edges, V vocab): emission fold O(B·L·E·V); edge-flow gather holds **B·L·N²**; tree build **O(B·L·N³)** at O(log L) depth keeping all levels; token draw one B·L×V softmax. The `(E,V)` emission tensor is the big constant (~250 MB fp32 for a BFCL function-call automaton, per `integrations/sglang/README`). **Numerics traps paid for upstream**: fp32 tree products underflow at L=64 with spread logits (fp64 `matmul_dtype` is the escape hatch, tests/test_marginal.py:86-103); NaN→uniform fallbacks with deferred `nan_counts` warnings (never host-sync mid-step); launch latency dominates on GPU (~100 small kernels — the reason `sample_many` packs rows with different automata into shared passes).

**Decoupling**: `mosaic/sampling/*` + `matcher.py` depend only on torch — zero HF/engine imports (`tests/toy.py` builds samplers from a raw graph dict + a 4-token alphabet and validates vs brute force). The Rust-reimplementable core contract is: automaton tensors + `sample(logits (L,V), forced (L,), temperature, start_node, end_log)` + `marginal_log(...)`. It is a **pure modelless inference primitive**.

## Prior-art landscape (why the novelty claim is pinned where it is)

| work | id/yr | mechanism | exact for dLLM? | log-depth? |
|---|---|---|---|---|
| **DINGO** (Suresh et al.) | 2505.24061, 2025 | DP over the FA interleaved with per-step guidance | **yes** — distribution-preserving, but sequential/linear within each step | ✗ |
| Constrained discrete diffusion (CDD) | ~2503.09598, 2025 | likelihood-guided steering | ✗ approximate | ✗ |
| CFG-dLLM / lookahead-then-verify wave | 2025-26 | masking + lookahead + rejection | ✗ approximate | ✗ |
| GFlowNet "Guaranteed Generation" | 2402.10062, 2024 | learned amortized sampler | ✗ (not the base LM's posterior) | ✗ |
| Outlines / DOMINO lineage | 2308.09732+ | AR next-token masking | n/a (AR) | ✗ |
| HMM forward–backward (Rabiner) / parallel smoothers (Särkkä) / VSBR depth reduction | 1989 / 2021 / 1983 | classical ingredients | marginals-only / log-depth marginals-only / tools | partial |

**Pinned claim (§1.5 precondition form):** *joint sampling **exact w.r.t. the mean-field posterior** (the product of per-position LM marginals × the FA indicator — not the model's true joint) of a dLLM under a regular (FA) constraint, parallel across all block positions, at logarithmic sampling depth via segment-tree depth reduction, consuming only per-position logits + the automaton — distinguished from DINGO (exact but sequential, linear depth) and from this workspace's `legal_token_set`/Lodestar (AR next-position legal sets, left-to-right) by the joint-posterior + log-depth delta.* "First exact constrained decoding for dLLMs" would be **false** (DINGO has exactness); the defensible delta is parallel joint sampling + log-depth (the paper reports <5% overhead; unmeasured here — G2 owns the figure, never inherit the paper's number). Caveat carried honestly: the parallel-algorithms axis (a direct "log-depth FFBS" prior) was the thinnest search lane; re-check before any public first-claim.

## The in-stack consumer (why this is Gain, not a redirect)

The workspace ships an **active dLLM track**: `riir-infer/src/transformer/dllm.rs` (feature `dllm`: D2F forwards — `forward_bidirectional` teacher, `forward_block_causal` student, `forward_set_causal`) + `riir-infer-gpu/src/gemma2_d2f/` (CubeCL block-causal **denoising decode loop** `d2f_decode_gemma2`, self-conditioning per Plan 250) + riir-train sibling trainers (Plan 383) + riir-ai re-export + GOAT gates (`goat_250_d2f_quality_gates_gpu.rs`). The decode loop's current step is exactly the paper's setting: per-position temperature-scaled draws with mask-token suppression (`sample_with_confidence`, mod.rs:1442), confidence-based commits via a raw top-1-prob threshold or a learned 7-parameter logistic `DiffusionSampler` (`SamplerFeatures`: top-1 prob + entropy, mask excluded).

What the paper adds to that loop:

1. **`propose_x0` under a constraint** — replace per-position independent draws with the exact joint constrained draw; commit any confidence-ordered subset, remask the rest; exactness survives **any** schedule (the loop's low-confidence rule is one such schedule).
2. **Constrained confidence** — the exact marginal `marginal_log` as a `DiffusionSampler` feature (mosaic's `commit_by="constrained"` axis) instead of raw top-1 prob; mosaic's LLaDA2 loop documents why schema-forced tokens make raw-confidence thresholds over-commit.
3. **Structured-output guarantee** — JSON-schema/function-call grammars as DFAs (mosaic's `grammar/json_schema.py` is the reference compiler shape: type/properties/required/items/enum/anyOf, depth-bounded recursion, char-NFA → token-graph re-alphabetisation).

## Routing + verdicts

- **Tier: GOAT (Gain)** — not Super-GOAT today: Q1 (no prior art in-workspace; published prior art differentiated above), Q2 (new behavior class: constraint-satisfying joint block decode — no incumbent in the loop has it) and Q4 (force multiplier: connects `dllm` forwards × `gemma2_d2f` decode × `DiffusionSampler` commit machinery × future structured-output serving) are solid YES; **Q3 (product selling point) is contingent** — the D2F lane is research-scale today with no product consumer of its text output, so the selling-point sentence carries a conditional clause. Re-grade trigger: the D2F lane gains a structured-output surface (function calling / schema'd text). Named honestly rather than hidden.
- **Per-track:** inference track only. The paper carries no training content; riir-train Plan 383 trainers are untouched. No riir-train deferral exists to justify (§3.5 trivially satisfied — the whole algorithm is modelless inference).
- **Fusion priority ladder:** lands on surface #3 (inference-perf/serving substrate). Lower-priority surfaces recorded, not filed: **game-context reframe** — no consumer; NPC minds/attack FSMs are rule-driven and CGSP self-play action sampling has no sequence-level automaton surface today. **Healer-context reframe** — no consumer; the drafter is corpus retrieval with compile-gated rejection (L2), not generative token sampling.
- **MOAT gate:** `riir-infer` — public inference-substrate novelty, upstream-clean, self-contained. Home = **riir-infer-core** (NOT katgpt-core): riir-infer deliberately carries zero katgpt-rs deps, the consumer is in-tree, and the module is pure modelless math (CSR-style automaton tensors per mosaic's design). Public stays public — the repo is already public.
- **Closest shipped cousins:** `katgpt-core::legal_token_set` (+ `LodestarAutomaton`, `CsrLegalSet`, `ProjectionPlan` Dead/Forced/Restricted/Full) — AR next-position legal sets over a DFA, consumed by `katgpt-forward/cluster_head.rs`, `katgpt-speculative/dd_tree/lodestar.rs`, `katgpt-pruners/lodestar.rs`. **Delta:** left-to-right one-position queries vs joint block posterior under mean-field decode; no masking is "wrong" there and exact here — they are different problems. Second cousin: the D2F decode loop's own confidence machinery (what gets upgraded).

## Fusion

`mosaic sampler × Lodestar-style CSR automata × DiffusionSampler` produces what none has alone: a decode loop that (a) guarantees FA-valid structured output by construction, (b) commits by exact constrained posterior confidence (calibrated against what the constraint forces vs what the model believes), (c) stays bit-compatible with the existing confidence-schedule contract (feature-off = byte-identical). The automaton-tensor layout should follow mosaic's (CSR transition/emission + `edge_lookup`), not the dense `n_states × vocab` δ the legal_token_set docs already call out as the layout to avoid.

## Validation protocol (the GOAT gate the issue must satisfy)

- **G1 correctness:** exactness vs brute-force enumeration on toy automata (mirror mosaic's TV-distance test); constraint satisfaction by construction (every sample accepted, adversarial schemas); parallel ≡ sequential distribution agreement.
- **G2 perf:** per-step overhead vs unconstrained decode at L ∈ {64,128,256}, grammar scales from toy to BFCL-schema; bar: single-digit-% wall-clock (the paper's <5% is unmeasured here — measure our own, cite box state).
- **G3 no regression:** feature-off byte-identical (default arm untouched).
- **G4 alloc:** bounded scratch, zero-alloc steady state per workspace rules; fp64 escape hatch only on the tree-product path.
- **Quality:** grammar-following rate on a structured eval (synthetic first; a real function-calling cell only if/when the lane gains one).

## P0–P3 priority

P0 pure sampler + exactness tests (no model) → P0.5 parallel tree + fp64 hatch → P1 decode-integration arm (opt-in feature, byte-identical off) + constrained-confidence feature for the sampler → P2 schema compiler + bench → P3 GOAT verdict + promote/demote (owner-gated). Filed as `../.issues/035_fa_constrained_dllm_decoding_poc.md`.
