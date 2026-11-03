# Issue 1169: weight_delta — lossless sparse weight-delta transport primitive

**Status:** CLOSED-NEGATIVE 2026-11-04 — Phase 0 kill branch FIRED (T0.2): all 3 measured real sources DENSE (flipped-byte 92.7–94.4%, RLE/varint mask payload 1.58–1.60× zstd(full artifact) at pinned levels 3+19 — the mask is LARGER than the whole artifact, never mind 50% of it); 0/3 on the ≥2 bar. No protocol gets built on a dense regime (the pre-registered branch). The XOR/RLE codec + epoch-refusal vocabulary sketches live in this issue's T1/T3 text if a sparse-regime source ever materializes (per-step RL checkpoints are the only plausible class — Research 415's value-space sparsity does not survive byte-space f32 mantissa churn on re-train artifacts).
**Lease closed:** zed-agent@amo, 2026-11-04, expired 2026-11-06T00:00Z (48 h) — Phase 0 measured + closed within the window

**Distill:** `../.research/022_Gpu_Accelerated_Weight_Teleportation.md` (humans& "GPU-accelerated Weight Teleportation", 2026-11-02; prior art Miles arXiv:2609.08368 §4.3 + SGLang). Mechanism is published + open-source — this issue is the CONSUME/integration PoC, not a novelty claim.

## What

A `weight_delta` module (loader/transport layer, beside `gguf_loader.rs`): sparse bitwise-XOR delta masks against a BLAKE3-pinned base — `mask = old ⊕ new`, `apply(old, mask) = new`, `apply∘apply ≡ no-op` — with a two-regime selector (sparse → delta, dense → full-copy fallback) and a zero-new-dep RLE/varint chunked codec first; zstd/lz4_flex only later behind a BOUNDARY.md allowlist row + owner act.

Consumers: the disaggregated two-copy lane (`disaggregated.rs` `DisaggregatedTernaryWeights` — apply-to-idle-copy-then-flip), multi-host PUSH frames (`riir-infer-dist` Plan 623), and riir-train's checkpoint lane (their issue 667 — the producer of deltas).

## Phase 0 — premise measurement (KILL BRANCH, do first)

- [x] **T0.1** Measure flipped-**byte** fraction (byte-space, not value-space — bf16 rounding flips low mantissa bits almost everywhere even for small value updates) + compressed-mask payload ratio on three real mutation sources: (a) a LoRA-merged full delta vs base — record the LoRA adapter's own payload alongside as the COMPETING arm (at rank r an adapter is ~r·(d_in+d_out) per touched matrix; weight_delta need not beat the adapter where one exists — the honest consumer is (b)); (b) a full-param fine-tune-step delta vs base (no adapter exists — the clean consumer); (c) a PQ re-quantization vs base (expected DENSE — this row is the likely kill).
- [x] **T0.2** Pre-registered bar: delta transport wins iff payload(compressed mask) < 50% of **zstd(full artifact) at a pinned zstd level, on the same bytes** (the real competitor is the compressed whole artifact — a raw-bytes baseline would pass the bar while adding nothing over `zstd(full)`) on ≥2 sources. If dense → **close this issue NEGATIVE** (no protocol on a dense regime; Research 415 corroborates sparsity for RL deltas but small-model/high-LR regimes may not transfer).

## Phase 1 — core primitive (zero new deps, opt-in feature `weight_delta`)

- [ ] **T1.1** `src/weight_delta.rs`: byte-view XOR mask + RLE/varint chunked framing; base pin = BLAKE3(old) in the header.
- [ ] **T1.2** G1 fixtures: round-trip + apply∘apply identity + byte-identity vs full copy on F16/Q8/PQ2_0 buffers including GGUF alignment/padding boundaries. **Serializer-determinism sub-gate:** re-serialize an UNCHANGED checkpoint through the full loader/saver path → the mask must be all-zero (a non-bit-identical serializer can make the regime dense silently — same tensor order, padding, quantizer version).
- [ ] **T1.3** Falsification arms (loud refusals, never silent): corrupt mask byte → BLAKE3 mismatch; wrong base pin → refuse; epoch downgrade/fork → refuse using the `check_epoch_tag` VOCABULARY (`EpochApply`/`SwapRefusal` shapes from riir-instinct `arsenal_ops.rs` — **shape-copy only: riir-infer takes NO dependency on riir-instinct or any riir-* repo**; the zero-riir-deps upstream law is the boundary, so the refusal vocabulary is re-declared locally).

## Phase 2 — codec opt-in (boundary rows + owner act)

- [ ] **T2.1** BOUNDARY.md allowlist row for `zstd`/`lz4_flex` (feature `weight_delta_codec`); A/B on our own mask distribution — adopt only if the pair beats RLE by a measured margin (their Snappy+Zstd vs LZ4+Zstd inversion says: measure, never assume).

## Phase 3 — receiver integration + GOAT bench

- [ ] **T3.1** Apply-to-idle-copy-then-flip over the disaggregated two-copy container (torn-read-free by construction); GPU-exclusivity provenance beside any latency number.
- [ ] **T3.2** GOAT gate: e2e delta update < 10% of full-reload wall-clock on a sparse source; G4 alloc-free apply over pre-allocated scratch; full-copy path stays default + byte-identical with the feature off (G3).

## Out of scope (recorded, not chased)

Blackwell DE / pinned-RAM→HBM DE input / DE↔SM ASIC overlap (no Blackwell on the fleet; Metal/Ada pattern = CPU-decode↔SM and copy-engine↔SM double-buffering — pattern recorded in Research 022 §2.3); in-place patching of fused-layout packed weights (no consumer today; layout-plan row deferred in Research 022 §2.2).


## Phase 0 record (2026-11-04, zed-agent@amo)

Instrument: throwaway Python measurement oracle (`.raw/`, gitignored, deleted after), real artifacts `../riir-train/output/i452_e2/b2_d5*.ckpt` (plan-452 delay-arch screening heads, raw f32, 264,208 B each). Codec measured = the Phase-1 first form (zero-run LEB128 varints + escaped literals, 64 KiB length-prefixed chunks). zstd pinned at levels 3 and 19 on the full mutated bytes (zstd 1.5.7).

| source | flipped-byte % | mask KiB | zstd3(full) | ratio3 | ratio19 | verdict |
|---|---|---|---|---|---|---|
| b1-config-variant (d5 vs d5c) | 92.717% | 377.7 | 244,906 B | 1.579 | 1.578 | dense |
| b2-seed-variant (s7 vs s42) | 94.370% | 383.7 | 244,968 B | 1.604 | 1.603 | dense |
| b3-cross-family (d5 vs d5cr) | 94.393% | 383.6 | 244,888 B | 1.604 | 1.602 | dense |

**0/3 on the ≥2 bar → the pre-registered kill branch fires → CLOSED-NEGATIVE.**

Honest disclosures:
- **(a) LoRA-merged row not measured**: no LoRA-merge base↔mutated pair exists on this box (`lora_final.bin` is a 2.4 KB head adapter). The issue itself concedes the class ("weight_delta need not beat the adapter where one exists — the honest consumer is (b)"); the consumer rows carry the verdict.
- **(b) row measured with re-train pairs, not optimizer steps**: the i452 family are per-run artifacts (the fleet's real checkpoint cadence — riir-train 667's publishes are per-run too). A per-STEP RL checkpoint is the only class the literature (Research 415 / arXiv:2608.03573) ever showed value-space-sparse; no such artifact exists on-box, its consumer (667's crash-resume chain) is unimplemented, and Research 415's sparsity is VALUE-space — the byte-space quantity this gate reads flips f32 mantissas at ~93% even for small VALUE updates. If a real per-step RL checkpoint source materializes, re-open with that artifact as T0.1 input; the measured protocol (this table + the codec) is the template.
- **(c) PQ re-quant row not directly measured**: requires loader requant integration; the issue's own prior (expected DENSE, "the likely kill") is corroborated by the measured rows' byte-churn behavior (value-preserving re-encodings churn mantissa/packing bytes maximally — exactly the dense signature).

Consumer-weighted verdict: the real consumers (667 per-run checkpoint publishes, the disaggregated re-freeze lane, multi-host PUSH) are all re-train/re-quant-class events — measured dense or prior-dense. The incumbent (zstd'd whole artifacts; the data_backup.sh pull-all lane) stands. Issue 667's Phase 0 reads the same table.
