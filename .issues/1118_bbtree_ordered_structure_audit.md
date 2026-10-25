# bbtree-class ordered-structure findings — BPE merge-rank lookup allocation (audit; GOAT-gated tasks)

**Status:** OPEN — findings filed by the bbtree distill session (workspace distill Research 322); Finding 1 IN PROGRESS.
**Claimed:** zed-amo-glm5, 1769426000 — Finding 1 **LANDED 2026-10-25** ([Bench 894](../.benchmarks/894_issue1118_bpe_idpair_ranks.md)): (a) re-key DONE (`HashMap<u64, usize>`, packed id pairs, load-path translation, missing-vocab pairs skipped); (b) G1 PASS — synthetic 4/4 + real-model 25 107 ids identical vs a naive string-keyed reference on the Bonsai vocab (drift pin `blake3:0e9cc4ce…`); (c) G2 PASS — 2.03× encode throughput, 5-round process-level alternation, every round paired same-direction; (d) G4 PASS — 7/11/11 allocs flat across 512× merge iterations, 10 allocs for a 3 KB chunk; (e) G3 green at default/no-default/scale_invariant_attn/deltanet_ternary_inference — all-features blocked by PRE-EXISTING `7b2fed9` breakage (issue 1120, not this change). Finding 1 CLOSED. Findings 2+ (if any) remain unclaimed.

Origin: the cache-conscious ordered-map distill landed in riir-refine (workspace distill Research 322, riir-refine commit `e9444713`), sourced from the bbtree blog + repo (pin below). A 16-repo workspace audit flagged one hot-path finding in this repo. This issue records it as an optimization task; implementation happens LATER, behind its own GOAT gate. Costs below are STATIC ANALYSIS, not measurements — the GOAT gate is what turns the claim into evidence.

## Finding 1 — BPE merge-rank lookups allocate on every encode

- **Site:** `src/tokenizer.rs:278` — `merge_ranks: HashMap<(String, String), usize>`.
- **Current cost (static):** the merge loop allocates TWO heap `String`s per adjacent pair PER MERGE ITERATION (alloc site `src/tokenizer.rs:544-550`; the code's own comment admits "two Strings per pair per iteration"). Every prompt encode pays this: O(pairs × merge-iterations) heap allocations, plus a SipHash probe hashing heap-string contents per lookup (vs a fixed 8-byte key hash).
- **Why it is bbtree-class:** bbtree's core thesis is that small fixed-size keys belong INLINE in the node (one cache line holds 8 8-byte keys), so lookups touch no allocator and ~one cache line. `(u32, u32)` token-id pairs are exactly that shape.

## Proposed approach

Re-key merge ranks by token-id pair. Token ids are stable once the vocab is loaded, so the string-keyed map is only needed during construction.

- [ ] (a) Re-key: `HashMap<(u32, u32), usize>` (or a `u64` key `(a << 32) | b`) replacing the `String`-pair map on the hot path — zero alloc per probe, fixed 8-byte key (bbtree's inline-key sweet spot). Keep the `String`-pair map ONLY for the initial vocab load if the GGUF vocab needs string-keyed construction (build once at load, then translate to id pairs; the load path is cold).
- [ ] (b) **G1 — exact output:** token output BIT-IDENTICAL on golden fixtures. The real-model gate exists and panics-loud rather than skip-green (the gemma-2-2b loud-fail gate discipline) — respect that discipline: the gate must EXECUTE the changed merge loop, not compile it away. Rank ties resolve by the same first-min-rank rule as today; the id-pair key must preserve the exact merge order.
- [ ] (c) **G2 — perf:** encode throughput at realistic prompt lengths (the laya fixture lanes give the harness), `--release`, QUIET BOX, and quote the box-state line (`scripts/bench_preflight.sh`-style PROVENANCE) beside the number.
- [ ] (d) **G4 — alloc axis:** alloc-count assertion — 0 allocations per merge probe after the fix (canary-armed counting allocator or equivalent).
- [ ] (e) **G3 — no regression:** default-feature build + the feature postures the tokenizer serves stay byte-identical and clean under clippy.

## Boundary note

riir-infer is the zero-riir-dep upstream substrate. This change adds NO dependencies (std `HashMap` re-key only) — boundary-clean by construction.

## References

- Blog: <https://bbtree.xiangpeng.systems/post/>
- Source: github.com/XiangpengHao/bbtree @ `8adae97a0ed97e5c98fa2790312eb0aec9719846` (MIT OR Apache-2.0)
- Workspace distill: Research 322 / Plan 289 (riir-refine), landing commit `e9444713`
