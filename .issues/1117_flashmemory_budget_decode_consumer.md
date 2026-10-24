# Issue 1117 — run FlashMemory budget (top-ρ) attention in the Bonsai decode loop + the end-to-end quality gate

**Status:** Open — filed 2026-10-25 from riir-train Issue 452 E1 (riir-train Bench 671) + katgpt-rs Issue 962. Not started.

## Why

riir-train Bench 671 measured, on the real Bonsai-2 PQ2_0 forward (Metal, 32K and 128K), attention-OUTPUT fidelity of block selectors at exactly matched retention: the shipped centroid-dot score RANKED per row (keep the top `round(ρ·n) − 1` complete causal blocks + the current block) beats Quest min/max bounds, a recent window and sink+window at every ρ, layer and context. At 128K, ρ 0.10 (90% KV reduction) keeps mean cos 0.980 / 0.990 / 0.991 at layers 19 / 35 / 51. The primitive shipped opt-in in katgpt-rs as `flashmemory_budget_select` (`GqaFlashMemorySelector::with_budget(ρ)`, `aa439587c`).

That is a PER-LAYER fidelity number. Whether it survives 16 attention layers compounding through a 64-layer hybrid stack is an end-to-end question nothing can answer yet: **`forward_attention_layer_flashmemory` (`src/deltanet/flashmemory_gqa/mod.rs`) has no caller outside its own tests** — no decode or prefill path runs FlashMemory attention.

## Tasks

- [ ] **T1 — feature passthrough.** `flashmemory_budget = ["flashmemory_gqa", "katgpt-attn/flashmemory_budget_select"]`; the forward already takes the selector through `GqaBlockSelector`, so the budget arm is a caller-side `with_budget(ρ)`.
- [ ] **T2 — decode-loop wiring.** A decode step for the Bonsai hybrid stack that routes the 16 attention layers through `forward_attention_layer_flashmemory` with a per-layer selector and a per-layer block cache maintained incrementally (rebuild cost is the open perf question — `rebuild_from_keys` recomputes every centroid; an append-only update of the current block is the obvious shape). CPU first (the reference); the GPU forward is a separate lane.
- [ ] **T3 — end-to-end gate (G1).** Teacher-forced over a long-context corpus slice: next-token top-1 agreement and per-token NLL delta vs the dense decode, at ρ ∈ {0.05, 0.10, 0.15, 0.30}, context ≥ 32K. Report per-position-bucket, not only pooled (the Bench 671 worst case — layer 35 min cos 0.85 at 128K — says some rows lose mass).
- [ ] **T4 — perf (G2).** Decode tok/s and KV bytes touched at matched ρ vs dense at 32K/128K, `--release`, box state recorded.

## Non-goals

Paged/offloaded KV (riir-train Issue 452 T5) — this issue reduces attention READS over a resident KV; capacity is the other lane.
