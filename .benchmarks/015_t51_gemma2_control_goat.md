# Bench 015 — Issue 022 T5.1 lane (1): gemma-2-2b f16 control, collapsed-vs-parent GOAT sweep

**Status:** COMPLETE — lane (1) CLOSED as a **clean negative at every real depth cut**; the control reproduces and sharpens the T5.0 bonsai verdict.

## Pre-registration

Budgets were pinned in `.issues/022` BEFORE any measurement (commit `0e0436c`, instrument landed + protocol pre-registered in the same commit):
- absolute top-1 agreement ≥ 0.9 (the T5.1 lane bar; parent self-agrees at 1.0 — never a ratio)
- sweep = pre-registered coarse ε grid first; fine-end bracket (0.01/0.015/0.02/0.03) only where the coarse grid's finest point fails
- hit-rate column reported beside every agreement row (functional quality vs trajectory divergence)
- corpus: `../riir-train/data/chat_probe` (raw text, `load_corpus_text`), frozen 4096-token stream, chunks × ≤1024, 4092 scored positions, teacher-forced greedy argmax, last-position-of-chunk dropped

## Box state

4090 workstation (i7-13700K, 16C/32T), CPU lane, AC. **Concurrent load: T2 (`kv_plus_ladder.exe`, the Issue-013 KV ladder) ran the whole window at ~2 cores** — both lanes are accuracy-class (load-independent arithmetic); throughput degraded ~2× vs solo (3.7 vs ~6 tok/s). Latency numbers are NOT claimed here (the G2 perf axis is T5.3's, and it owns box exclusivity).

## The instrument

- Profile: `examples/twt_gemma2_profile` (feature `twt_gemma2`) — the Issue-395 `PostLayerHook` capture seam added to `forward_gemma2_f16_tapped` (post-MLP residual = the S-matrix state; `NoHook` monomorphizes to the unchanged forward; the four calibration bins updated with the no-op). 6 chunks × 1024 tokens, 26×26 cosine S, profile artifact `.raw/twt/gemma2_profile.json` (corpus BLAKE3 recorded inside).
- Emit: `examples/twt_collapse_emit` extended with the gemma2 arch arm (`gemma2.block_count` renumber, all-Attention layer types, minimax-medoid member winners, member passthrough byte-copies).
- Agreement: `twt_goat_agreement` extended with the gemma2 arm (family read off `general.architecture`; `config_from_gguf_metadata` + `SentencePieceGgufTokenizer` + `load_gemma2_f16_direct` + the production `forward_gemma2_f16`; parent arm cached to `.raw/twt/gemma2_parent_cache.json`).
- Commit: `0e0436c` (instrument + pre-registration).

## Measured (the coarse grid)

Parent top-1 hit rate **0.4746** (context column — healthy signal on a no-BOS raw-text stream).

| ε | blocks | depth | agreement | collapsed hit | verdict |
|---|---|---|---|---|---|
| 0.05 | 26 | 100% | **1.0000** (4092/4092) | 0.4746 | **PASS** (round-trip identity) |
| 0.1 | 17 | 65.4% | 0.2571 | — | FAIL |
| 0.2 | 10 | 38.5% | 0.0831 | — | FAIL |
| 0.3 | 6 | 23.1% | 0.0132 | 0.0022 | FAIL |
| 0.5 | 4 | 15.4% | 0.0049 | 0.0005 | FAIL |
| 0.8 | 3 | 11.5% | 0.0039 | 0.0002 | FAIL |
| 1.2 | 1 | 3.8% | 0.0015 | 0.0002 | FAIL |

Timing: parent arm 4092 positions in 1095 s (3.74 tok/s, shared box); ε=0.05 collapsed arm 1101 s (identical depth). Emit wall: 17–51 s per collapsed file (BLAKE3 over 5.2 GB dominates).

## The fine-end bracket: structurally EMPTY, skipped with the reason

At ε=0.03 the partition is **m=26 with worst-block 0.0000 — zero merges** (identical at ε=0.01/0.015/0.02: all four emitted as full-depth 288-tensor identity checkpoints). Gemma-2's S geometry has **no rung between "no merges" and the 65.4% cut** — the cosine meter cannot express a smaller real cut than the coarse grid's first. The four bracket arms would be byte-identical re-emits of the ε=0.05 identity row (guaranteed 1.0000); they were emitted (artifacts existed) and their agreements NOT run — the ε=0.05 row already measures that exact point.

## Reading (the control-lane verdict)

1. **The pipeline is bit-faithful**: the ε=0.05 identity row reads 1.0000 (4092/4092) — writer byte-copy passthrough + renumbered-metadata reload = exact. The FAILs below are the merges, never the machinery.
2. **The cliff reproduces on a second architecture, and sharper**: bonsai's only passing point was a 4.7% cut (Bench 022-t5); gemma-2 has NO passing real cut at all — the first real cut (65.4% depth) already reads 0.2571, and every deeper cut collapses toward 0.00 with hit rates dying to 0.0002.
3. **Cosine redundancy is not a license for depth cuts on decoder LLMs** — now measured on the hybrid ternary (bonsai) AND a dense f16 (gemma-2) with the same meter, the same DP, and the same writer. The zero-training passthrough lane's quality question is closed: activation-space redundancy ≠ functional redundancy (the ShortGPT-class result), and gemma-2's phase structure (m=10 at ε=0.2 — REAL S-matrix structure, cf. bonsai m=8) does not transfer to merge safety any better than bonsai's did.
4. **Where the lane goes next** (per T5.0's own gate): only a merge/audition instrument that BETS on fine cuts can rescue quality-at-depth — and the two-model evidence says the rescue, if it exists, lives in the apply-path (auditioned merges / distillation, riir-train 423), not in the partition. The control lane is DONE; it hands riir-train 423 its baseline artifact and its negative control.

## The kill-rule footnote (the T5.0 ⚠ class, re-derived live)

The profile's own T1.6 kill verdict printed `KillBlockCount{m_at_max_eps: 1, bar: 1}` — the single-operator stack (forced floor = 1) makes the bar trivially 1, so the rule fires on a technicality at ε=1.2. The rule is measuring "structure beyond the type-split floor", which a single-operator stack cannot have BY CONSTRUCTION; the depth-reduction license on this lane is carried by the S sweep (m=10 at ε=0.2 = real structure) and adjudicated by THIS bench, not by the kill rule. Recorded so the T5.5 Pareto report cites the sweep, never the kill verdict — the same provenance trap the bonsai ⚠ flagged.

## Artifacts

- `.raw/twt/gemma2_profile.json` — the S matrix + corpus BLAKE3 (kept; the regenerable seed for every collapsed file)
- `.raw/twt/gemma2_parent_cache.json` — the parent argmax arm (kept; reusable for any future gemma-2 sweep point)
- Collapsed GGUFs (7 coarse + 4 fine-end identity) — **deleted after measurement** (regenerable from the profile artifact + parent in ~20–90 s each via `twt_collapse_emit`; the bench table + the writer's determinism gates are the record)

## Reproduce

```bash
cargo run --release -p riir-infer-core --features twt_gemma2 --example twt_gemma2_profile -- \
  --gguf ../riir-train/data/gemma-2-2b-it-f16.gguf --corpus ../riir-train/data/chat_probe \
  --seq-len 1024 --max-tokens 6144 --out .raw/twt/gemma2_profile.json
cargo run --release -p riir-infer-core --features twt_collapse --example twt_collapse_emit -- \
  --parent ../riir-train/data/gemma-2-2b-it-f16.gguf --profile .raw/twt/gemma2_profile.json \
  --eps 0.1 --out .raw/twt/gemma2_collapse_e0_1.gguf
cargo run --release -p riir-infer-core --features twt_bonsai --bin twt_goat_agreement -- \
  --parent ../riir-train/data/gemma-2-2b-it-f16.gguf --collapsed .raw/twt/gemma2_collapse_e0_1.gguf \
  --corpus ../riir-train/data/chat_probe --seq-len 1024 --max-tokens 4096 \
  --cache .raw/twt/gemma2_parent_cache.json
```
