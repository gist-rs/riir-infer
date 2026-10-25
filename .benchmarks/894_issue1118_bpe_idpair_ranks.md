# Benchmark 894 — Issue 1118 Finding 1: BPE merge-rank id-pair re-key (G1/G2/G4)

**Status:** GOAT G1 + G2 + G4 PASS; G3 green at the postures this lane claims (all-features blocked by PRE-EXISTING `7b2fed9` breakage — issue 1120, not this change).

**Date:** 2026-10-25 · **Box:** amo (M5 Ultra, 30 cores), release, AC power, load 3.5–7.5 (a sibling training precompute cycled during the runs — the A/B is alternation-treated).
**Change:** `merge_ranks: HashMap<(String, String), usize>` → `HashMap<u64, usize>` — the packed `(left_id << 32) | right_id` key (bbtree inline-key shape, workspace distill Research 322). The merge loop's probe allocates ZERO (was: two heap Strings per adjacent pair per merge iteration, the code's own comment admitting it). Load path: string merges translate through the just-loaded vocab once (cold); missing-vocab pairs skipped — they can never fire in either keying (merge rules are defined over vocab token pairs).

## G1 — exact output

- **Synthetic 4/4** (`src/tokenizer.rs::bpe_idpair_tests`): adversarial chunks vs a naive string-keyed reference built in-test from the SAME vocab+merges — ties, NO_ID paths (non-vocab symbols), long runs, missing-vocab rules, duplicate vocab entries. Both keyings agree everywhere.
- **Real model** (`tests/tokenizer_bpe_g1_parity.rs`, `#[ignore]`, Bonsai-2-27B PQ2): 248 320 tokens, 247 587 merges, **0 skipped**, **25 107 ids byte-identical** between the production id-pair path and the in-test naive string-keyed reference (pretokenize + merge + vocab-lookup all rebuilt reference-side; the only shared piece is the model data). Drift pin: `blake3:0e9cc4cef07728c3b065199a56aad9a9`. ⚠ The gate's first version failed with a TEST-side byte-table reimplementation missing the Latin-1 printable ranges — fixed by consuming the production table through a `byte_table_for_test()` seam; the failure was the reference's, not the change's, and the seam removes the divergence class.

## G2 — encode throughput (process-level alternation)

Compile-time A/B (the old map no longer exists in the new tree) → the established treatment: an identical probe file (`tests/tokenizer_bpe_g2_perf.rs`) in BOTH trees, 5 rounds alternating run order, per-round pairing. Old tree = clean worktree at `8a4f379` + the same probe. Corpus: 106 661 bytes fixed-seed (prose + code + multilingual + 9 000 vocab-sourced fuzz chunks — fuzz-HEAVY, i.e. near the worst case for the old shape; the ratio on pure prose is smaller and quoted as such). Both sides emit 56 556 ids (a second equivalence read).

| round | old ns/byte | new ns/byte | ratio |
|---|---|---|---|
| 1 | 223.6 | 113.5 | 1.97× |
| 2 | 222.9 | 109.1 | 2.04× |
| 3 | 238.5 | 109.9 | 2.17× |
| 4 | 215.2 | 111.6 | 1.93× |
| 5 | 206.7 | 104.0 | 1.99× |

Medians: **old 222.9 / new 109.9 = 2.03×** — every round pairs in the same direction, spread ±7%. Absolute figure carries the load disclosure (sibling precompute cycling; the paired design is load-robust, the ns/byte are not precise).

## G4 — alloc axis (`tests/tokenizer_bpe_g4_alloc.rs`, own counting-allocator target, the fa_g4 precedent)

Single-chunk encode (one pre-token, so per-chunk setup is constant and the only growth axis is merge iterations):

| iterations | allocs |
|---|---|
| 1× ("the") | 7 |
| 64× | 11 |
| 512× (1 024 merge iterations) | **11** |

Flat. The old shape allocated 2 Strings × pairs × iterations — thousands at this depth. Extreme cell: a 3 KB single-chunk encode (1 024 tokens out) = **10 allocs total**. Non-vacuity: an explicit allocation moves the counter (asserted).

## G3 — no regression

- lib 269/269 default (incl. the 4 new parity unit tests); clippy `-D warnings` at default all-targets, `--no-default-features` all-targets, `scale_invariant_attn`, `deltanet_ternary_inference`.
- **Disclosed, not this change:** `--all-features --all-targets` FAILS at HEAD with 5 errors in `spike_census_t3_kv_ab_ternary` + `kv_evict` — confirmed pre-existing on a clean pre-change worktree (they rode in with `7b2fed9`'s SPLIT-handle fix); filed as **issue 1120** for the zerocopy lane.
- The `decode_spin_pool_goat` real-model consumer gate is UNSEEN on this box (its `.raw/twt/audition_calib.txt` corpus absent — the gate asserts loud by design); the tokenizer's real-model surface is covered by G1's own real-vocab run.

## Verdict

GOAT G1+G2+G4 pass → the re-key lands DEFAULT (it is an internal representation change with bit-identical output, not a new primitive: no API surface, no behavioral toggle; the gates ARE the protection). Land: `ad7b5a0` + `54a6616` + `ddc7105`.
