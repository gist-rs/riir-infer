# Issue 035 — FA-constrained exact posterior sampling for the dLLM decode lane (POC)

**Status:** OPEN — POC filed from Research 008 (arXiv:2607.07026 / `MhDang/mosaic` distill); GOAT-tier, consumer = the `gemma2_d2f` denoising decode loop; owner-gated promote/demote at P3.

## Why

The `dllm` decode path (`riir-infer/src/transformer/dllm.rs` D2F forwards + `riir-infer-gpu/src/gemma2_d2f/` CubeCL block-causal denoising loop) draws **per-position independent samples** each denoising step (`sample_with_confidence`, mask-token suppression) and commits by raw confidence (threshold, or the learned 7-param `DiffusionSampler`). Under any structured-output requirement (JSON schema, function-call grammar, format automaton) that is wrong twice: per-position draws do not respect joint constraints, and raw top-1 confidence over-commits on constraint-forced tokens. Mosaic (arXiv:2607.07026) provides the exact fix: FA-as-HMM forward–backward over the block trellis with the LM logits as emissions, **joint state-path sampling** (per-position marginals provably do not factorize the joint), O(log L) depth via a segment tree of pairwise log-space matrix products, composing with any commit/remask schedule because each step re-conditions on committed positions. Full math, complexity table, numerics traps (fp32 tree-product underflow at L=64 → fp64 escape hatch; NaN→uniform + deferred warnings), and the prior-art delta vs DINGO (2505.24061, exact but sequential/linear-depth) in `../.research/008_Mosaic_FA_Constrained_dLLM_Decoding.md`.

## Non-goals

- Not a trainer change — riir-train Plan 383 sibling trainers untouched.
- Not AR-side work — `katgpt-core::legal_token_set` already owns AR next-position legal sets; this is the mean-field/joint decode problem only.
- Not a default-on flip — the decode loop's default arm stays byte-identical; `fa_constraint` is opt-in until the GOAT gate passes (owner call at P3).
- No new external deps — pure std/core math in `riir-infer-core` (the repo's zero-katgpt-rs-dep posture holds; automaton tensors follow mosaic's CSR-style layout, not dense δ).

## Tasks

### P0 — pure sampler module (no model)
- [ ] `fa_posterior` module in `riir-infer-core`: automaton tensors (CSR `transition`/`emission`, `edge_index`, `accept`, derived `edge_lookup`), determinism invariants asserted (≤1 edge per (node, token) and per (i,j) pair)
- [ ] emission fold: `e_logits[t,e] = log Σᵥ emission[e,v]·exp(logits[t,v])`, max-shift stabilized; committed positions as 0/1 indicators (branchless)
- [ ] sequential sampler: edge-space backward pass + forward ancestral draw; exact **joint** state-path-then-tokens sampling (never per-position marginals)
- [ ] exactness tests vs brute-force enumeration on toy automata (TV distance bar; forced tokens; length budget via reverse-BFS `steps_to_accept` / `finish_within`)
- [ ] zero-alloc steady state audit (bounded scratch per workspace rules)

### P0.5 — parallel sampler (the paper's headline)
- [ ] segment-tree build: identity-leaf power-of-two padding, end-weights folded on the last leaf, pairwise log-space matrix products, per-level max-shift
- [ ] top-down midpoint conditional sampling (`z_mid ~ softmax(left[begin→·] + right[·→end])`, one draw per level)
- [ ] fp64 escape hatch on tree products (fp32 underflows at L=64 with spread logits — measured upstream)
- [ ] parallel ≡ sequential distribution cross-check + marginals (`α/β` midpoint fill) to 1e-9 vs brute force
- [ ] NaN→uniform fallbacks with deferred (end-of-run) warning counters — no host syncs mid-step

### P1 — decode integration (opt-in feature)
- [ ] `fa_constraint` feature (implies `dllm`): `propose_x0` at each `gemma2_d2f` denoising step — forced = committed positions, free = masked; confidence-ordered commit, rest remasked
- [ ] feature-off arm byte-identical (existing `sample_with_confidence` path untouched)
- [ ] constrained-marginal feature for `DiffusionSampler` (the `commit_by="constrained"` axis): `marginal_log` as an extra `SamplerFeatures` input, A/B vs raw top-1 confidence
- [ ] automaton state carry-over across blocks (`start_nodes` from the accepted prefix; block-causal decode)

### P2 — schema front end + bench
- [ ] JSON-schema → char-NFA → token-DFA compiler (type/properties/required/items/enum/anyOf, depth-bounded recursion — mosaic `grammar/json_schema.py` is the reference shape; tokenizer-trie re-alphabetisation for our vocab)
- [ ] bench: per-step overhead vs unconstrained at L ∈ {64,128,256} × grammar scales (toy → function-call schema); box-state provenance per the G2 rule
- [ ] quality: grammar-following rate on a structured eval (synthetic first)

### P3 — GOAT verdict (owner-gated)
- [ ] G1 exactness + by-construction acceptance + parallel≡sequential
- [ ] G2 single-digit-% overhead bar (measure OUR number; the paper's <5% is unmeasured here)
- [ ] G3 feature-off byte-identical
- [ ] G4 bounded alloc + fp64 hatch scoped to tree products only
- [ ] promote/demote decision; re-grade Super-GOAT if the D2F lane gains a product structured-output surface (Research 008 §Routing)

## Refs

- Research note: `../.research/008_Mosaic_FA_Constrained_dLLM_Decoding.md` (all quotes pinned to mosaic @ `36945e4b3c64913e23962e0924a8b40a84c58807`, MIT)
- Upstream: arXiv:2607.07026 (Dang & Ermon); DINGO arXiv:2505.24061 (the exactness-first prior to cite, not to claim over)
- In-stack cousins: `katgpt-core::legal_token_set` (AR-side; different problem), `gemma2_d2f` decode loop (the consumer)
