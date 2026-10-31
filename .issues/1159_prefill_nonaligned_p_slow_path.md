# Issue 1159 — prefill at a non-aligned chunk (p=3000) is SLOWER in absolute wall than p=4096

**Status:** OPEN — filed 2026-10-31 as a side-finding of riir-ai Plan 644 / Issue 1147 B4 (riir-infer `02673a9`). Measured only; not diagnosed.

## Evidence

Box: M5 Studio Ultra amo, AC, release, folded `Ternary-Bonsai-2-27B-PQ2_0.gguf`, tensor-zerocopy GPU-only arm of `tests/plan644_channel_split_g1.rs` (single-frame `prefill_range_chunk`, best of 3–4 interleaved passes):

| p | GPU-only wall | tok/s |
|---|---|---|
| 2048 | 5652.6 ms | 362.3 |
| 3000 | **13772.2 ms** | **217.8** |
| 4096 | 12372.7 ms | 331.1 |

A 3000-token chunk costs 1.11× the wall of a 4096-token chunk and runs at 0.60× its throughput, so some site takes a slow path when p is not a multiple of a tile (or of the 2048 block). Every real prompt has a ragged final chunk (`RIIR_PREFILL_CHUNK_MAX` = 4096), so this lands on the common path.

## Tasks

- [ ] **T1 — locate the site.** Run `bench_1146_prefill_frame_census` with `I1146C_EXEC=1` (the sync-bracketed per-site exec timer, `463655f`) at p=3000 against p=2048 and 4096. The site whose ms/token jumps is the slow path. Candidates: the zero-copy matmul2d tiling (`p.div_ceil(nrb)` grid; are the edge tiles handled per element?), the chunked DeltaNet recurrence (chunk-size remainder), and attention prefill tile fallbacks.
- [ ] **T2 — fix or pad.** Either handle the remainder inside the fast kernel, or pad p up to the tile multiple (pad rows are discarded; mind the recurrence, where a pad token must not advance state) and price both.
- [ ] **T3 — gate:** p ∈ {2048, 3000, 4095, 4096}, G1 against the unpadded reference, wall per token monotone-ish in p.

## Related

- riir-ai Plan 644 / Issue 1147 (where it surfaced; the ANE split's 1.059× @3000 is on top of this slow baseline).
