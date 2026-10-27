# Issue 1137 — rung: the FFN decode-GEMV kernel-internal memory pipeline (2-bit select-accumulate ALU chain at 26-35% of stream)

**Status:** CLOSED — NULL, mechanism-already-implemented + uniformly negative on this box (2026-10-27 ~09:4x +07). Sequence as claimed: the fork mapping study landed (below) → substrate-first check then found the mechanism ALREADY in-tree: `gemv_ternary_fma_cubecl` (Plan 563 / Bench 847, the same PR-26980 telescoped algebra cooked to our planes, G0/G1 PASS artifacts, `RIIR_GEMV_FMA` default-OFF with the 847 NO-SHIP verdict and the aspect-selective reopen trigger (a)). bench_1141 then re-measured BOTH arms on THIS box with the 1140 methodology (cold-set rotation, interleaved medians): the FMA kernel loses ~3x on EVERY aspect — gate/up 156.8 vs 47.9 us (3.28x), down 178.3 vs 60.1 (2.97x), lm_head 1416.7 vs 475.0 (2.98x) — the 847 down-win does NOT reproduce on M5; accuracy artifact excellent (max_abs <= 8.8e-6, the 742-law band) but the performance direction is uniformly, decisively negative. Combined with 1140 (shipping kernel NOT payload-BW-bound, variant-flat), the honest disposition: the ALU-chain deficit is real but NO existing kernel family (plane/rowtiled4/rowtiled8/f16scale/FMA-telescoped) reaches past ~35% of stream here — closing the CubeCL-variant dimension of this rung NULL. The residual lane (hand-written MSL kernel outside CubeCL codegen, the fork's 438 GB/s posture is MSL) is a DIFFERENT lane (owner-scoped, not a rung under this issue). Reopen levers: (a) an owner-approved hand-MSL lane with the 847/1141 records as the baseline; (b) a future CubeCL/Metal toolchain generation. Lease RELEASED, work complete. Sequence: fork q2-K mapping study FIRST (read the vendored metal dot kernel, name the 438 GB/s mechanism in the record — reconnaissance informs the G1 design and is not a variant spend), then the ONE named variant (tensor-GEMM class, or the mapped fork mechanism if the study shows the tensor lineage is the wrong family) under the standard ladder: G1 to_bits vs the incumbent f16scale at both FFN shapes, e2e ABBA >=1.5% per-run interleaved, REACH counter, demote hatch. Filed 2026-10-27 ~09:2x +07 from the i1113 probe-(b) close (Bench 1140, instrument + 3 measurements below).
**Lease closed:** adjudication complete 2026-10-27 — the gate ruled GO on reopen lever (a) (delegated verdict session zed-amo-msllane-gate; the claude_code reviewer backend was auth-unavailable, a native subagent stood in and verified the quoted numbers); the lane is filed as `1138_hand_msl_ffn_gemv_lane.md` and the lane claim lives there.
**Closed (2026-10-27, later):** the i1138 lane this ruling opened has itself closed — e2e Bench 1149 MEASURED-NEGATIVE (G1 PASS loop-stream identity, G2 FAIL −8.21% mean: dispatch economy, not math) and the owner-scoped v2 lever (token-batched command buffer) adjudicated **NO-GO** (premise falsified in code: no two field dispatches are adjacent — a CubeCL consumer/producer sits between every pair — so the max sound batch on the shared in-order queue is ONE kernel and the amortization collapses 129:1 → 1:1; sound remainder −0.8..−2.05 ms/tok strictly negative, the 898 law). Lane closed default-OFF (the `RIIR_GEMV_TQ2_FIELD` hatch stays, per-matrix fail-open); re-open ONLY via a CubeCL-channel custom-op escape hatch (owner-gated). Full ruling: riir-ai `.benchmarks/1149_i1138_field_e2e_ab_amo.md` Addendum. The 1138 ledger file is removed per the noise-reduction rule; this line is the surviving cross-link.

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

## The mapping study (2026-10-27 ~10:0x +07, zed-amo-i1137-ffnalu) — the fork mechanism NAMED

Source read: `vendor/prismml-llama.cpp/ggml/src/ggml-metal/kernels/mul_mv.metal` `kernel_mul_mv_tq2_0_f32_impl` (the fork carries a NATIVE ternary type `GGML_TYPE_TQ2_0`; dispatch constants `N_R0_TQ2_0=4` rows/simdgroup, `N_SG_TQ2_0=2` simdgroups/threadgroup = 64 threads, 8 rows/threadgroup; geometry from `ggml-metal-impl.h` + `ggml-metal-device.cpp`).

The mechanism (GGML PR #26980 algebra), four parts:
1. **Layout**: 2-bit codes packed 4-per-byte, row-major byte stream (`qs`), per-block (QK_K=256) f16 super-scale `d`. NOT bit-planes.
2. **Offset-encoding algebra**: each thread owns 4 bytes per 32-value block (`htg = tiisg % 8`); per block it hoists `coef[4]` from the y-vector (y-differences) + `sumy = Σy` ONCE, shared across the 4 rows; the row loop is then `sum = −sumy + Σ_j coef[j]·f_j(v)` with `f_j` the 2-bit fields extracted by exact float floor ops (`floor(v·0.25)` = `v>>2`, etc.) — **pure float ops, no selects, no per-bit integer work**: 1 cvt + 3 floor + 4 fma per byte per row ≈ 3 ops/weight/row vs our 4 int select+add ops/weight/row.
3. **The zero-weight offset fold** (`sum = −sumy + …`): the code offset lands as ONE subtract per block per row instead of per-weight branching.
4. **float4 y loads** (16B) and 4-byte weight reads per row — denser coalescing than our per-lane 16 scattered u32 loads per word-iteration (row-major stride = words_per_row×4B apart, uncoalesced across the 8 rows).

**Mapping to OUR format** (the repack design): our pos/neg planes → byte fields `c = (p − n) + 2 ∈ {1,2,3}` packed 4-per-byte (`v = c0 | c1<<2 | c2<<4 | c3<<6`; the (1,1) alias encodes as 2 — both mean w=0). Then `d = c − 2 ∈ {−1,0,+1}` recovers our weights EXACTLY (value-identical, not approximate), and `dot = Σ s_g (Σ f_j·x_j − 2 Σ x_j)` with the `−2Σx` folding per block exactly like the fork's `sumy`. Repack rides `TernaryHandle::from_weights` as a new arm (upload bytes ≈ same: 4 weights/byte vs 2 planes×1bit).

**G1 class**: NOT to_bits — the offset fold changes f32 accumulation order (pre-authorized in this issue's gate: fall to the Bench-742 law — unit max-abs tolerance + greedy-token loop-stream identity e2e).

**Prize arithmetic** (exec-pricing per the correction): reaching the fork's ~53%-of-stream class at the FFN shapes (~418 GB/s) takes gate/up 46.2→~30 µs and down 60.4→~41 µs → **≈ −3.2 ms/tok ≈ +13% e2e** — the largest single-rung prize left on the decode board; even 45%-of-stream ≈ +8%.

**Next session body (this lease stays armed, expires 2026-10-28T04:00Z)**: (a) the repack arm in `from_weights`; (b) the CubeCL kernel (start at the fork geometry 4 rows/sg × 2 sg, sweep later); (c) the unit gate: max-abs + greedy-token vs the incumbent f16scale (bench_1140's rig builder is the harness base); (d) bench_1141 e2e A/B per the standard ladder (per-run interleave, REACH counter, demote hatch, RIIR_GEMV_TQ2_FIELD toggle default-OFF).

**Refs:** riir-ai `.issues/1113` (the ⟲ 2026-10-27 probe-(b) note + the released ffnbw lease), `crates/riir-gpu/tests/bench_1140_ffn_gemv_bw_curve.rs` (the instrument: G1 + BW curve + asymmetry verdict cells), Bench 1008 (the census), Bench 1114 (the roofline + wg sweep), Bench 611 (the LUT negative), riir-infer Issue 771/1115.

## The close record (2026-10-27 ~09:4x +07, amo, load 5-7, bench_1141 + bench_1140 same methodology)

- Study: `kernel_mul_mv_tq2_0_f32` mechanism named (2-bit byte fields + exact floor extraction + coef/sumy hoisting + float4 y) — then found implemented as `gemv_ternary_fma_cubecl` (Bench 847). The substrate-first law cost one redundant kernel draft, deleted before compile.
- bench_1141 (riir-ai, `ternary_gemv` posture): accuracy max_abs fma-vs-shipping gate/up 2.766e-6 / down 8.814e-6 (742-law PASS); aspect curve in the table above; fma_launch_count 442 vacuous-guard PASS.
- Reconciliation with 847 (M3 Max): sign agreement on gate/up (loses) and lm_head (loses); the down +32% win does not reproduce (2.97x loss). The 847 base readings (12-46 GB/s) were a different regime than the 1140/1141 cold-rotation method (209-403 GB/s) — the current methodology is the stronger instrument (L2-defeated, interleaved, median-of-5).
- The 1140 + 1141 pair closes the "which existing kernel wins the FFN shapes" question EXHAUSTIVELY: shipping f16scale rowtiled8 is the argmin of the measured family on every aspect on this box.
