# Issue 1137 — rung: the FFN decode-GEMV kernel-internal memory pipeline (2-bit select-accumulate ALU chain at 26-35% of stream)

**Status:** Open — filed 2026-10-27 ~09:2x +07 from the i1113 probe-(b) close (Bench 1140, instrument + 3 measurements below). Unclaimed.

## The trigger (measured, not assumed)

The launch-floor ladder closed by measurement (i1130 ~0, i1134 negative −2.96..−9.48, i1135 sub-bar 0.89–1.36, i1136 G1-dead) — and the i1113 probe-(b) BW curve (bench_1140, cold-set-rotated, M5 Ultra amo) then showed where the token actually remains:

| shape (payload ~equal) | shipping f16scale µs | GB/s | % of 790 stream class | e2e ms/tok |
|---|---|---|---|---|
| ffn_gate/up (m=17408, n=5120) ×128 | 46.2 | 273 | 34.6% | 5.92 |
| ffn_down (m=5120, n=17408) ×64 | 60.4 | 209 | 26.4% | 3.87 |
| lm_head (m=248320, n=5120) ×1 | 446.3 | 403 | 51.0% | 0.45 |

FFN exec ≈ 10.2 ms of the 24.1 ms token. The kernel (`gemv_ternary_plane_rowtiled8_f16scale`, 8 rows/plane × 8 planes/wg, per-lane word-strided walk, 32-unrolled select+accumulate per word-pair over 8 rows) is **NOT payload-bandwidth-bound**: the f16 arm halves scale bytes and buys only 1.5% over the f32 sibling (a true 273-of-790 BW wall would read ~5%), and all existing variants are within noise (plane-stride 6× worse — input amortization is load-bearing; rowtiled8-f32 ≈ f16scale). The wall is the **per-weight 2-bit decode ALU chain** at issue/latency rate: 32 unrolled iterations × 16 (select + add) per 16 weight-words per lane.

## The census-artifact pricing correction (load-bearing for this rung)

The census wall priced gate/up 65.2 µs vs down 35.9 µs (1.82× "asymmetry"). Isolated, gate/up is FASTER (46.2 vs 60.4 = 0.77×) — down's census share is overlap-hidden, gate/up's near-serial. Any FFN rung prices against EXEC (~10.2 ms/tok; ceiling ~5.9 + 3.9 + 0.45), never against the subtraction wall.

## The shape (candidate kernel families — pick ONE, the 905 one-named-variant discipline)

1. **Tensor-GEMM class for the decode GEMV** — the vendored Issue-771 `metal_tensor_gemm` lineage (smem/tensor tiling exists for prefill); a decode GEMV as a skinny GEMM (m-tiles × n=1 padded) trades the ALU chain for MMA throughput. The G1 risk: the fma-contraction class (Issue-1113 zgate landmine) — to_bits vs the incumbent is the gate, and if 1-ULP divergent, the greedy-token loop-stream gate (Bench 742 law) decides.
2. **The fork q2-K metal dot mapping study** — vendor/prismml-llama.cpp runs the SAME 2-bit weights at 438 GB/s (53% of stream) on this box; read their metal dot kernel's inner loop (vector width, lookup structure, block layout) and map the mechanism onto the plane layout. Study-first (no kernel until the mechanism is named in the record).
3. **dp4a-class on the 4090** (riir-infer Issue 1115's lane) — NOT this rung's body; cross-ref only.

## The gate

G1 to_bits vs `gemv_ternary_plane_rowtiled8_f16scale` at both FFN shapes (the Bench-1140 G1 harness is the oracle); e2e ABBA ≥1.5% promote bar, per-run interleave (the 1130 law), REACH counter, demote hatch. The pricing bar uses EXEC arithmetic (above), and the promote bar stays 1.5% of the token.

## Non-goals

- NOT the wg dimension (Bench 1114 cell 4 swept it; the occupancy rule is default-on).
- NOT LUT/subset-sum (Bench 611 measured the Metal LUT negative).
- NOT launch-count fusion (the i1132 floor ladder: closed NULL/negative at every boundary — i1130/i1134/i1135/i1136).
- NOT the attention/gdn kernels (i1135/i1136 territory, closed).
- NOT lm_head-first: it reads 51% of stream already; the FFN pair is the pool.

**Refs:** riir-ai `.issues/1113` (the ⟲ 2026-10-27 probe-(b) note + the released ffnbw lease), `crates/riir-gpu/tests/bench_1140_ffn_gemv_bw_curve.rs` (the instrument: G1 + BW curve + asymmetry verdict cells), Bench 1008 (the census), Bench 1114 (the roofline + wg sweep), Bench 611 (the LUT negative), riir-infer Issue 771/1115.
