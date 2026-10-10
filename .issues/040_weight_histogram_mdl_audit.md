# Issue 040: Weight-Histogram MDL Description-Length Audit for Shipped Quant Formats (POC)

**Status:** Open — POC task (modelless, report-only). Source: riir-infer R011 (arXiv:2509.22445, ICLR 2026). Technique prior art acknowledged: EntroLLM arXiv:2505.02380, Deep Compression Han et al. 2016 — applied utility, not a primitive claim.

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

- [ ] `WeightHistogram` scan per tensor group (chunked, streaming counters, zero-alloc in the loop; reuse/extend the histogram machinery pattern from `lut_grid.rs` — do not fork it).
- [ ] Entropy + floor + slack computation, per-tensor and rollup views (bits/weight, total MB at floor vs nominal).
- [ ] **Side-info accounting (two-part honesty):** `N·H(p)` prices the weight VALUES only. The code also pays for its codebook — the distinct values, their frequencies, and, for block formats (PQ2/q2_0, k-quants, EXL3), the per-block scales. Report `N·H(p)` and `N·H(p) + side-info bits` as SEPARATE columns from stored bits; never pool them. Slack quoted without the side-info term overstates compression headroom for exactly the block formats this repo ships.
- [ ] Report front: per-model table sorted by slack; advisory rows flagged for non-discrete tensors.
- [ ] G1 correctness: bit-match a reference entropy coder (e.g. a canonical range coder) on synthetic histograms + sampled real groups from the repo's named GGUF corpus; side-info byte counts (codebook + scales) cross-checked against the actual file layout.
- [ ] G2 perf: single O(N) metadata pass; publish measured MB/s with box-state provenance line.
- [ ] G3: read-only proof — decode-path outputs byte-identical before/after (audit touches no decode code).
- [ ] G4: fixed-size alphabet tables, streaming counters, no allocation in scan loops.
- [ ] Wire the report into the EXL3/PQ2 conversion-lane docs (one paragraph + one example run).

## Non-goals

- NOT a codec: no re-encoding, no Huffman/arithmetic coder shipping (EntroLLM owns that lane; we audit, they compress).
- NOT a quality claim: floor/slack say nothing about retention — lossy-surface verdicts stay with the paired per-family retention gates.
- NOT an optimization target this round: no format changes driven by slack until the POC reports.

## Effort

~0.5–1 day. Pure metadata arithmetic; the histogram pattern already exists in `lut_grid.rs`.
