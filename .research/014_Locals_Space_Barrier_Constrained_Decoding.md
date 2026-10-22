# Research 014: Locals / Breaking the Space Barrier — Table-Free Grammar-Constrained Decoding + Mac Batch Serving

> **Source:** Arip Asadulaev, *Breaking the Space Barrier and its Application to Language Model Inference*, [arXiv:2610.09139](https://arxiv.org/abs/2610.09139) (Oct 2026, cs.AI/cs.CC) + the shipping product **Locals** (`github.com/Generated-Computers/Locals`, MIT, cloned at `32cf2feb07cd6e92cde4871347435773393f1812`, ~26k LOC Python + embedded MSL).
> **Date:** 2026-10-22
> **Status:** RECORD
> **Related Research:** 008 (Mosaic — FA-as-HMM constrained decoding for **d**LLMs; the in-repo sibling on the other model class), 006 (bekko shared-prefix KV reuse), 007 (Triadic GDN tensor state — the league arch lane)
> **Classification:** Public (this repo)

---

## TL;DR

Constrained decoding (JSON schema / tool calls) has a hidden structure the complexity-theory literature already named: the automaton×vocabulary-trie product is a **mangrove** (exactly one path to every reachable state), and few-path graphs admit sub-Savitch-space reachability validation. The paper settles Apers–Edenhofer's open question (`STCON_sf` ∈ ReachUL ∩ LOGDCFL ∩ C=L ∩ SC², space O(log²n/log log n) **below Savitch's O(log²n)**) and turns the proof constructions into six exact decode mechanisms: (1) the **walk kernel** — recompute the mask row on the GPU per step (one thread per token walking bytes through the DFA; 7–18 µs vs a 7–15 ms model run) instead of storing a 3|Q|×|V| table (3 MB vs 1.5 GB; 16 grammars served where tables crash at the 15th), (2) **forced chains are free** — deterministic one-live-byte spans ride the same pass (31% of JSON tokens; 226 vs 343 model runs; pass-optimality theorem), (3) **canonical tokenization** of forced text (commit pre-tokenizer blocks, hold the last — non-canonical splits measurably flip answers: "Jean-Luc" → "Just-Luc"), (4) **validity under any token budget** (Bellman–Ford `need(c)` mask, least-restrictive; 90/90 vs 3/90), (5) **recursive schemas on a capped GPU pushdown** (stack-locality: top-2 symbols suffice), (6) **certified parallel field decode** (the language factors at key cut-states; draft fields as batch rows, verify in one pass — token-identical to greedy, 1.8× fewer runs).

**Distilled for this repo (decode-path substrate, modelless):** every mechanism is a decode-path inference primitive with a proof attached — nothing here trains. The engine facts that ride along matter as much as the theory: the test models are **hybrid GDN** (Qwen3.5: 3/4 layers recurrent; single-stream decode falls only 138→118 tok/s from 128→8k context because only 6/24 layers attend), the GDN decode step fuses into ONE Metal kernel, fp16 beats bf16 on M1/M2 (no native bf16; fp16 is *closer to float32*), and the prompt cache persists across restarts (83% of sent tokens served from memory; 138k-token follow-up starts in 2.4 s vs 8 min).

---

## 1. Paper core findings

### 1.1 Theory (App. B)

`STCON_sf = {⟨G,s,t,1^k⟩ : every pair has ≤k walks, and s→t}` — the *total* language: no promise, the algorithm must reject cyclic/exponential-walk graphs in the same bounds. Three constructions, each reused by the engine:

| Construction | The idea | Where it reappears in decoding |
|---|---|---|
| **Budget truncation** (Lemma 1, Thm 8) | With total budget B=nk, a walk of length B has B+1 distinct prefixes — more than the budget. Explore only to depth B; DFS + capped stack terminates on ANY graph. | Capped pushdown for recursive schemas; totality of every bounded search |
| **Certified lift** (Thm 9) | Fingerprint every walk (polynomial hash at α ∈ F_p), lift to (endpoint, length, hash) states; **certify** the lift is reach-unambiguous, *then* count fibers — exact whether or not budgets hold | "Certify, then count" → certified block decoding |
| **Signed counting** (Thm 10) | Σ of squares of ∏(f−j) is a GapL function vanishing exactly on good inputs — one zero test enforcing every budget at once | (certificate form only) |

Structure theory (the part the engine actually consumes): **branch kernel** — between any two vertices at most r−1 ≤ 2k *branch* vertices survive; everything else is a **forced chain** (whoever enters walks it to the end). **Block product** — the language factors at cut vertices every path crosses; r = ∏ of per-block path counts. Finite forbidden-subdivision basis; mangrove normal form (path-count matrix factors into < 4 log₂k mangroves).

### 1.2 The six decoding mechanisms (App. C, each a theorem)

1. **Table-free mask (walk kernel).** Pair every trie node with the state reached from q → an out-tree (mangrove; Prop. 1). So the table row T[·,q] is *recomputable*: one Metal kernel, one thread per token, walk bytes through the 256·|Q| int16 automaton. 7–18 µs/step vs 7–15 ms model run. Grammar cost = automaton (512 B/state) + 2.9 MB packed vocab shared by ALL grammars. GPU never waits for a host mask (host-side masks cost a step: 103 vs 137 tok/s free).
2. **Forced text is free.** A state with one live byte is a forced chain; after each sampled token the whole forced continuation feeds **in the same pass** (a pass over S tokens costs 1.6× S=1 at S=8 — weight-read-bound). Pass-optimality (Thm 19): output identical, and no non-speculative decoder with the same output distribution can use fewer passes. A key costs NO run; a field with r admissible values costs ≤ r−1 (Cor. 1, straight from the branch kernel).
3. **Canonical split (Thm 20).** Pre-tokenizer blocks are cut vertices; commit all but the last, certify the last with a one-byte stable-tail test (quantified only over bytes the constraint allows), and a **boundary-cover** rule (next sampled token must span ≥ ℓ bytes of a held single-token block). Splitting forced text any other way measurably changes answers.
4. **Budget mask (Thm 21).** `need(c)` = fewest tokens to finish validly (Bellman–Ford on the token-step graph); admit a token only if a valid ending still fits. Least-restrictive such mask. Mask inactive until the last 17–38 tokens.
5. **Pushdown with stack locality (Thm 22).** Recursive schemas → deterministic pushdown (LL(1) compile checks); a token's run depends only on exposed real pops → context-free table + small context table indexed by top-2 stack symbols; 64-entry GPU stack; depth cap with explicit error, never an invalid emit.
6. **Certified blocks (Thm 23).** Required-key objects factor at key cut-states (Lemma 12); draft fields as parallel batch rows, replay through the automaton, verify masked argmax in ONE pass, continue from the first rejected position. Token-identical to greedy (36/36 measured), 1.80×/1.42× fewer model runs (2B/4B) — but no wall-time win on the laptop GPU (extra rows aren't free there).

### 1.3 Measured (M2 Pro 16 GB, Qwen3.5-2B/4B 4-bit, vs MLX + llguidance)

- Schema extraction 1.28×/1.18× faster, **character-identical** outputs on the 2B, 45/45 gold facts.
- 3 MB vs up to 1.5 GB per grammar; 16 grammars in ~1 GB vs server death at the 15th; the **typed** tool grammar (1,106 states, every argument name+type fixed) is impossible with tables (824 MB + 332 s build) and 46/46 typed calls at parity speed with the walk kernel.
- 16 coding agents 2.55×; 8 concurrent tool calls 3.1× throughput, every reply parses (reply = a regular language: free text + `<tool_call>` blocks, forced tags = 20–30% of call tokens; a budget-cut reply still closes its call).
- LiveCodeBench batch-8: 476 s vs 1,181 s at 4k cap, similar pass rate; thinking mode 6/20 finish on-engine vs stock OOM at six 16k streams.

---

## 2. The Locals repo (what ships beyond the paper)

MIT. Pinned @ `32cf2feb07cd6e92cde4871347435773393f1812`. Python (uv), MLX runtime, MSL kernels as embedded source strings. `locals/` module map (26k LOC): `regex.py` (Thompson NFA → subset construction, 200k-state budget, trim, Moore-minimize) · `walk.py` (the Metal walk kernel) · `constraint.py`/`forced.py` (forced chains + canonical planner, healing) · `pda.py`/`pdactl.py` (pushdown) · `regions.py` (finite-region exhaustive scoring: enums/booleans/tool names scored as completions in one right-padded pass — exact for what it computes) · `siblings.py` (certified block waves) · `budget.py` · `vocab.py` (packed trie) · `tree.py`/`treedec.py` (tree decomposition utilities) · engine half below.

Engine facts (README + App. D–E, all measured on M-series):

- **Fused GDN decode step** (`gdn_kernel.py`): stock mlx-lm runs ~14 small kernels per linear-attention layer per decode token; one threadgroup per (sequence, value head) does conv+silu → scaled RMS norms → delta-rule recurrence (one simdgroup per 4 rows of the 128×128 state) → gated RMS norm, for up to 16 tokens, matching the stock path "up to summation order".
- **Prompt-lookup × constraint fusion** (`lookup.py`): n-gram drafts from the stream (prompt+output); the draft is walked through the automaton on the host (a draft is ONE path in the step graph — cut at the first disallowed token), then sampled token + forced span + draft feed in ONE pass; masked argmax at every draft position ⇒ **greedy lookup decoding equals greedy decoding** under the constraint. This is the speculative-decode composition the paper's outlook names.
- **Ragged hybrid batch** (`hybrid_batch.py`): continuous batching where each row feeds sampled token + forced span (right-padded into one (B, Lmax) step); attention layers get `right_padding` (rolled in finalize), recurrent layers get per-request lengths so masked positions leave the GDN state unchanged; prompt-trie shared-prefix prefill once + snapshot copy.
- **Prompt cache across restarts**: long system prompts (3k–16k tokens, every coding agent's) read once, kept on disk; 83% of sent tokens from memory; a 48k-token doc follow-up in 0.6 s; prior tool-call turns re-rendered verbatim so whole turns are cache hits (647/675 tokens).
- **Precision**: fp16, not bf16, on M1/M2 — no native bf16 arithmetic there, and fp16 lands *closer to float32* (measured; float32 prefers the fp16 engine's token at 32/35 near-ties vs mlx-lm).
- **Pipeline**: step n+1 enqueued with unevaluated device values (y, q′) while step n runs; forced-span check happens on a functional cache and is discarded when the host reads y. Host graph construction + encoding overlap the GPU.
- **Agent-team lane** (`locals run`): splits a task, runs agents in project copies, tests decide; plus repetition/loop detection and call-count cancels for small models.
- OpenAI + Ollama + Anthropic Messages-compatible server; MLX-format models only (no GGUF).

---

## 3. Distillation — mapping onto this repo

### 3.1 The AR grammar engine (mechanisms 1–5): no in-repo prior art

Workspace greps (both vocab sets): no grammar→automaton→mask pipeline anywhere. Closest hits, read and signal-diffed:

- `katgpt-speculative/src/spec_generator.rs::TokenConstraintPruner` — a boolean `is_valid(depth, token, parents)` **adapter interface**, no automaton, no trie, no GPU mask, no forced chains. It is a consumer *seam* this paper's engine could implement, not coverage.
- `riir-infer/.research/008` (Mosaic) — constrained decoding for **diffusion** LMs: FA-as-HMM forward–backward over the position×state trellis, exact joint posterior sampling. Different model class (dLLM joint denoising vs AR next-token), different mechanism (trellis fold vs per-token byte walk). Complementary: both compile a constraint to a DFA; a future repo that serves both model classes shares the compiler front-end.
- `riir-infer/src/fa_schema.rs` — JSON-reader *configuration* grammar for FA schema instances; vocabulary collision only.
- `anchored_reach.rs` / `certified_frontier.rs` (katgpt-core) — "reach"/"certify" vocabulary collisions (signed blend operator; SafeOpt safe-set). Read, dismissed.

**Fusion candidates if the arm ever opens:**
- Forced spans = **drafts accepted with probability 1** — the one speculative variant that needs no verifier and beats the N-pass lower bound legally. Composes with `katgpt-speculative`'s draft/verify machinery (spec_generator / PeiraPruner) exactly the way `lookup.py` composes lookup drafts with the automaton.
- The **reply-as-regular-language** shape (free text + call grammar, "not containing" construction, forced tags, budget still closes the call) is the right contract for ANY hosted/local model that must emit tool calls — relevant to any hosted-model tool-call lane and to any agent loop that parses model output.
- The **budget mask** answers a question every serving stack gets wrong under `max_tokens`: cut at a state, not at a count.

### 3.2 Prompt cache: the paper is third-party confirmation of a trigger we already wrote

`riir-infer-gpu/src/qwen38_prefix_cache.rs` (Issue 742) already snapshots **whole-prefix GDN recurrent state + conv windows** keyed by token prefix (blake3 filter + exact compare, lineage rule against clobbered KV tails) — the same shape insight the paper states ("the recurrent state exists only at the end of a stored stream"). The doc's own serving-lane trigger names the substrate: when a multi-request lane lands, consume the public `katgpt_kv::radix_prefix::RadixPrefixTree` (+ PagedKVCache chunk-page seam), measured 2.45× hit-rate at equal page budget (Bench 762). The paper adds two deltas we do NOT ship yet:

1. **Cross-restart persistence** — the cache outlives the process (disk-backed; thaw on boot). In our vocabulary this is a **freeze/thaw fusion**: a session/prompt cache whose entries are BLAKE3-committed frozen state, thawed at serve time. Modelless, deterministic, and it lands exactly on the freeze/thaw-over-recompute rule.
2. **Turn-verbatim serving** — re-render the previous assistant turn (tool calls included) so whole turns are prefix hits; 83% of agent traffic from cache.

No novelty claim here (llama.cpp `--prompt-cache` file persistence and the LMCache line are prior art class-wise); the value is the fusion note: radix prefix tree + GDN-state checkpoints + frozen-envelope persistence is a small assembly over substrate we already own.

### 3.3 GDN serving facts (league-adjacent)

- The **fused single-kernel GDN decode step** (14 launches → 1) is a direct mining candidate for our GDN decode paths (`deltanet/`, the league fork's GDN kernels, the chunked-GDN arm). Pattern class: per-(sequence, head) threadgroup ownership + simdgroup-per-4-rows recurrence + activation-dtype rounding at stock-equivalent points for parity.
- **fp16-over-bf16 on M1/M2** — a precision-posture data point for the M-lane (our Metal lane pins fp16 numerics in laya already; the paper's float32-tie evidence is a citable independent measurement).
- **Single-stream decode is context-flat on hybrid GDN** (138→118 tok/s 128→8k; only 6/24 layers attend) — supports the GDN long-context serving positioning already argued in Research 007.
- **League note**: Locals cannot run GGUF (MLX-only), so it is NOT a direct league opponent for our league models — but it is a new M-series data point against mlx-lm/llama.cpp on small models, and its repo is a `distill` candidate (kernel_opt for the MSL kernel patterns; rust_perf for the pipelining/batching host patterns). The distill grading (dedup grep, domain routing, yield estimate) is its own skill run — not performed here.

### 3.4 The theory constructions as generic primitives (fusion ideas, novelty TBD)

- **Budget truncation as a totality certificate**: any bounded search (MCTS rollouts, pruner span walks, graph walkers) gets a termination guarantee — explore to total budget B; prefixes of a too-long walk must collide. Cheap to state, cheap to test, potentially a katgpt-core-able operator for tree searches whose branching is *semantically* few. No consumer today; recorded, not planned.
- **Certify-then-count** (hash-lift, certify unambiguity, then count exactly): a general pattern for dedup/count over paths (trajectory stores, rule-index uniqueness). Same status.
- **Branch kernel as a cognition-budget law**: in a mostly-deterministic transition system, decision points are few (≤ r−1 between any two states) and everything between is forced. The game-side analogue (spend think-ticks only at FSM branch vertices) is a cute reformulation of the salience cadence — recorded as an idea only; the brain pipeline already gates ticks by salience and the forced-chain view adds no mechanism we lack.

---

## 4. Verdict (per-track, never pooled)

| Track | Verdict | Reasoning |
|---|---|---|
| **A. AR grammar-constrained decode engine** (mechanisms 1–5 + reply grammar) | **Gain** — actionable, not yet GOAT | Nothing in-workspace covers it (signal-diffs above), the capability class is real (guaranteed-parse structured output at MB grammar cost), but there is **no live consumer in this repo today**: we serve no local LLM completions from our own engines (the league fork benches; laya scores; reflex serves modelless decisions). A plan opens when a consumer arms — candidates: the league fork's grammar memory in batch serving (llama.cpp GBNF tables vs walk recompute), or a serving lane consuming `RadixPrefixTree` (Issue 742's own trigger). Filed as Issue 1108 with the inventory. |
| **B. Serving/mining (Locals engine facts)** | **Gain** — distill candidate + posture notes | Fused GDN decode kernel, lookup×constraint composition, ragged hybrid batch, fp16-on-M, prompt-trie snapshot — all mining-grade patterns for kernel_opt/rust_perf (via the `distill` skill, not graded here) and posture confirmations for the M-lane. Not a direct league opponent (no GGUF). |
| **C. Complexity constructions** | **Record** — fusion ideas, novelty TBD | §3.4; no consumer, no claim. |

**Pass-redirects:** none (verdict is Gain; cousins 006/007/008 + `TokenConstraintPruner` + `qwen38_prefix_cache` are cited in-place above with signal-diffs).

**PoC status:** none run — no quality-parity claim is made (this is an architectural distill; §3.6 PoC triggers do not fire). Any future implementation of Track A carries its own GOAT gate (outputs token-identical to unconstrained-masked greedy; grammar memory vs GBNF/table baseline; forced-span pass savings vs llguidance-style fast-forward on the same engine).

**MOAT gate:** riir-infer fits — decode-path/kernel substrate novelty, public, upstream-clean (nothing here adds a dep). No public/private tier issue: this note is public-safe; no rethink-class content.

## 5. Actionable follow-ups

See `.issues/1108` — the inventory: (1) Track-A arm behind a feature flag with its consumer trigger named, (2) the freeze/thaw prompt-cache persistence delta folded into the Issue-742 serving-lane trigger, (3) the `distill`-skill run on this repo (kernel_opt/rust_perf) when the mining queue next opens, (4) league-watch note (no watch_repo row — MLX-only, cannot serve the league models).

## Summary

A complexity-theory result (few-path reachability below Savitch) became a working Mac serving engine whose six decode mechanisms are each a proof: recompute the grammar mask on the GPU from the mangrove structure instead of storing tables (3 MB vs 1.5 GB), emit forced text for free in the same pass (31% of tokens), keep the tokenizer's canonical split, stay valid under any token limit, run recursive schemas on a 64-entry GPU stack, and decode independent schema fields in parallel with a certified verify pass — all while the underlying models are hybrid GDN and the engine fuses the GDN decode step into one kernel. For this repo: no in-repo AR grammar engine exists (the closest pieces are an interface seam, a dLLM-side sibling, and a single-stream GDN prefix cache with the radix trigger already written), so the verdict is Gain with the inventory filed and the mining/league posture recorded.
