# Issue 040: Weight-Histogram MDL Description-Length Audit for Shipped Quant Formats (POC)

**Status:** Landed (GGUF lane) 2026-10-10 — `quant::desc_len` + `desc_len_audit` bin + G1 rANS gate; EXL3 lane deferred (loader, see tasks). Source: riir-infer R011 (arXiv:2509.22445, ICLR 2026). Technique prior art acknowledged: EntroLLM arXiv:2505.02380, Deep Compression Han et al. 2016 — applied utility, not a primitive claim.

## Goal

Ship a report-only **description-length audit**: for every discretized weight group in a checkpoint (GGUF or safetensors), compute the two-part MDL floor

```
floor_g = N_g · H(p_g)   bits,   p_g = empirical histogram of the group's stored symbols
slack_g = stored_bpw_g − H(p_g)   bits/weight
```

per the closed form proven in arXiv:2509.22445 §B.9.3 (optimal GMM prior = delta components at the unique stored values, mixing = empirical frequencies ⇒ optimal code cost = histogram entropy). Aggregate per tensor and per model. Purpose: quantify how much entropy slack each shipped format (`q2_0`, `ptq1_0`, `exl3`, `q2k..q6k`, `q8kv`, ternary) leaves versus its nominal bpw, and feed that spectrum into format documentation, conversion-lane decisions, and (later) the selection-currency column in riir-train Plan 404's sweep via Plan 454 Phase 2.

## Scope

- New module `src/quant/desc_len.rs` (feature-gated, default-off; e.g. `desc_len`) + one `src/bin/` report front (or a `gguf_loader` metadata-pass hook).
- Consumes stored symbols only — NO dequantization on the audit path. Histograms over the format's native alphabet (codebook indices for PQ/EXL3, quant symbols for k-quants, trits for ternary).
- f16/f32 tensors: histogram is codec-dependent — report as `advisory` rows, never as floors.

## Tasks

- [x] `WeightHistogram` scan per tensor group (chunked, streaming counters, zero-alloc in the loop; reuse/extend the histogram machinery pattern from `lut_grid.rs` — do not fork it).
- [x] Entropy + floor + slack computation, per-tensor and rollup views (bits/weight, total MB at floor vs nominal).
- [x] **Side-info accounting (two-part honesty):** `N·H(p)` prices the weight VALUES only. The code also pays for its codebook — the distinct values, their frequencies, and, for block formats (PQ2/q2_0, k-quants, EXL3), the per-block scales. Report `N·H(p)` and `N·H(p) + side-info bits` as SEPARATE columns from stored bits; never pool them. Slack quoted without the side-info term overstates compression headroom for exactly the block formats this repo ships. (Landed: `slack` vs `slack*` columns; disclosed convention — fixed alphabets carry no codebook bytes, the frequency table is not re-priced per the adaptive-coder assumption, scale streams stay a future re-pricing lane.)
- [x] Report front: per-model table sorted by slack; advisory rows flagged for non-discrete tensors. (`desc_len_audit --gguf <m> [--json] [--min-mb]`; sorted by HONEST slack, advisory rows sink.)
- [x] G1 correctness: bit-match a reference entropy coder (e.g. a canonical range coder) on synthetic histograms + sampled real groups from the repo's named GGUF corpus; side-info byte counts (codebook + scales) cross-checked against the actual file layout. (Reference rANS in `tests/desc_len_g1_entropy_coder` — exact round-trip + `N·H ≤ coder_bits ≤ N·H + C` on uniform/degenerate/skewed/synthetic-q2_0 + the real-corpus lane via `RIIR_DESC_LEN_GGUF` against Ternary-Bonsai-2-27B-PQ2_0.gguf — PASSED. Layout self-consistency (`value·8 + side == block·8` + loader `block_info` agreement) is unit-pinned per format.)
- [x] G2 perf: single O(N) metadata pass; publish measured MB/s with box-state provenance line. (Measured in `.docs/001` §19: 142–335 MB/s across four packs; PROVENANCE: power=AC load=14.38 swap=1643.56M powermode=2(high) — preflight REFUSED, provisional-box-loaded; quiet-box rerun is a one-liner.)
- [x] G3: read-only proof — decode-path outputs byte-identical before/after (audit touches no decode code). (The audit takes `&GgufFile` mmap slices and returns owned report data — no write path exists; the G3 unit test pins decode-vs-audit byte-identity at the scan level.)
- [x] G4: fixed-size alphabet tables, streaming counters, no allocation in scan loops. (`[u64; 256]` `SymbolCounts`, `clear()`-reused across tensors; monomorphized sink; the only allocations are the report rows.)
- [x] Wire the report into the EXL3/PQ2 conversion-lane docs (one paragraph + one example run). (`.docs/001` §19 — the measured slack spectrum table + the example run.)
- [-] EXL3 (trellis) audit rows — needs the safetensors-side reader (trellis indices as symbols, procedural codebook free, `suh`/`svh` as side-info). Deferred with the loader; the GGUF lane (the issue's named G1 corpus) is complete.

## Non-goals

- NOT a codec: no re-encoding, no Huffman/arithmetic coder shipping (EntroLLM owns that lane; we audit, they compress). The rANS coder lives in the G1 TEST only — it never ships in the lib.
- NOT a quality claim: floor/slack say nothing about retention — lossy-surface verdicts stay with the paired per-family retention gates.
- NOT an optimization target this round: no format changes driven by slack until the POC reports.

## Measured record (2026-10-10, M3 Max)

| pack | format | stored bpw | honest floor | honest slack | floor/stored |
|---|---|---|---|---|---|
| Ternary-Bonsai-2-27B-PQ2_0 | Q2_0 | 2.1250 | 1.7098 | +0.4152 | 0.805 |
| Ternary-Bonsai-27B-Q2_0 | Q2_0 | 2.1250 | 1.7054 | +0.4196 | 0.803 |
| Ternary-Bonsai-27B-dspark-Q4_1 | Q4_1 | 5.0000 | 3.8300 | +1.1700 | 0.766 |
| Qwen3.8-27B-DFlash2-Q4_K_M | Q4_K / Q6_K | 4.5000 / 6.5625 | 4.3649 / 6.3563 | +0.135 / +0.206 | 0.970 (model) |

The ternary packs' H ≈ log2 3 = 1.585 (the fourth Q2_0 code state is
unused — the katgpt-rs Issue-578 finding, now measured at 1.27 B-weight
scale). Record + example run: `.docs/001` §19.

## Effort

~0.5–1 day. Pure metadata arithmetic; the histogram pattern already exists in `lut_grid.rs`.
