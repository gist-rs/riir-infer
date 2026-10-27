# Issue 1135 — rung: attention proj+decode launch cluster (proj + attn_decode per-launch floors)

**Status:** CLOSED — ceiling-gate NULL (2026-10-27 ~08:4x +07, 3 verified-idle interleaved windows on amo). The census pool decomposes honest: proj 32 launches/tok x 13.4-20.5 us real floor (pool 0.43-0.66 ms/tok), decode 16 launches/tok ALREADY per-head parallel (grid n_head x workgroup head_dim — the split-K comment's "single-workgroup-per-head" means per head, not total) so the only legal shape (the 648-F9 wq+wkv row-concat merge, -16 launches/tok) recovers at most HALF the proj pool = 0.22-0.33 ms/tok = 0.89-1.36 pct < the 1.5 promote bar in ALL THREE windows, and realized would sit lower (the merged launch absorbs wkv's compute; the same fusion class measured NEGATIVE twice this morning: i1130 -0.8..-1.0, i1134 -2.96..-9.48 — the 898/905 occupancy lesson). Decode-into-proj (candidate 2) dead by structure: 4 dependent stages between (split, qk-norm, rope, kv-append) = the grid-barrier 898 class. No kernel written, per the 905 disposition. Instrument: riir-ai bench_1138_attn_proj_decode_strip_ceiling (3-arm per-run interleave, launch-counter guard, T1.1e required-features row) + raw log 1138_issue1135_attn_cluster_strip_ceiling_amo.raw.log. Census accounting corrected: the filing's "64 launches x ~17 us" over-counted decode 2x (the split-K gate is n_positions > 512; census posture never crosses it — launches bit-identical 1254/1222/1238 across all 3 runs). Re-open lever: if other rungs shrink the token base (24.1 ms/tok folded posture) below ~22 ms, the SAME merge crosses the 1.5 bar arithmetically (0.33/22 = 1.5); also owner-gated batch rider if a future lane touches the proj load path anyway. Lease released, work complete. Session: zed-amo-i1135-attncluster, 1793065034.
The trigger: the attention region's two heaviest substages — **attn:proj 0.56 + attn:attn_decode 0.54 = 1.10 ms/token combined @ 64 launches x ~17 us/launch** (W1/bmo; W2/amo 1.03 + 1.06 = 2.09, same launches, rank stable). Each attention layer (16 of them on the folded league file) pays a proj dispatch and a decode dispatch at ~17 us — the same per-launch floor class as the norm pair (i1134), different region.

## The shape

The attention layer already carries the Issue-640 ATTN_SUBSTAGE instrument (proj / elementwise / kv_append / attn_decode / out_proj / outer_norm), so the A/B reads the census deltas for free. Candidate shapes, in A/B priority order:
1. **proj into the preceding q/k/v-source epilogue** (the i1134 pattern — epilogue fold, work distributes over the producer's threads; no new dependency edge).
2. **attn_decode into the proj epilogue** ONLY if the decode kernel's parallel shape survives the merge (decode is the flash/decode kernel — merging the GEMV-class proj ahead of it must not shrink its thread shape; the 905 occupancy lesson applies directly: if the merged shape loses blocks/threads, expect the NULL and file the read).
3. NOT elementwise/kv_append/outer_norm — the census reads them at the noise floor (0.05-0.21 ms/tok); no dispatch budget there worth a merge.

## The gate

Same discipline as i1134 (and 905): G1 to_bits vs the chain, G2 paired-interleaved 4-pair e2e (>= 1.5% promote bar, 3/4 sign consistency, REACH guard, demote hatch), census bits free. Upper bound if fully recoverable: 1.10 ms/tok ~= 7.4% e2e (W1 scale).

## Non-goals

- NOT the norm pair (i1134), NOT gdn_out_gemv (i1136), NOT the attention BW class, NOT the gdn chain.

**Refs:** riir-ai `.benchmarks/1008_issue1132_metal_launch_floor_census_w1.md` (W1+W2 + common-mode law), `.benchmarks/905_i1133_conv_merge_ab.md` (A/B discipline + the occupancy NULL), `.issues/1132`.

## The close record (2026-10-27 ~08:4x +07, amo, GPU-exclusive, load 4-6 decaying)

Three verified-idle windows of bench_1138 (3-arm per-run interleave, position-balanced, launch-counter guard; folded league PQ2_0, TOKENS=24, 12 timed runs/arm; launches bit-identical every run):

| run | FULL ms/tok | proj pool (32 la) | us/la | decode pool (16 la) | us/la |
|---|---|---|---|---|---|
| 1 | 24.21 | +0.657 (+2.71%) | 20.5 | +0.451 (+1.86%) | 28.2 |
| 2 | 24.05 | +0.430 (+1.79%) | 13.4 | +0.250 (+1.04%) | 15.6 |
| 3 | 24.12 | +0.512 (+2.12%) | 16.0 | +0.527 (+2.19%) | 33.0 |

- Census cross-check: proj implied 13.4-20.5 us/launch vs census W1 17.5 — coherent. Decode 15.6-33.0 straddles the floor — the kernel is floor + thin serial tile work (1 tile at pos<=25, ~11 sync_cube barriers), NOT a geometry problem.
- The folded-posture FULL reads 24.1 ms/tok at 1254 launches/tok (vs the census Q2_0 posture's 16.7 @ 852) — the folded file's staging launches are visible here, the i1134 census-undercount disclosure reproduced independently.
- Why the merge is sub-bar: the wq+wkv concat removes 16 of the 32 proj launches; the strip ceiling prices ALL 32 at 0.43-0.66; half of that is 0.22-0.33 = 0.89-1.36 pct of the 24.1 token. The bar is 1.5. Realized <= ceiling, and the morning's two G2 negatives at this exact class (i1130, i1134-closed-4aed461) both realized BELOW their ceilings (negative), so expected value is worse than the arithmetic.
- The re-open lever is recorded above; nothing else in this cluster moves the needle (elementwise/kv_append/outer_norm are the issue's own non-goals at the noise floor).
