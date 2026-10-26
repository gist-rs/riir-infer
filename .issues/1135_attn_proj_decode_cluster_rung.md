# Issue 1135 — rung: attention proj+decode launch cluster (proj + attn_decode per-launch floors)

**Status:** Claimed — zed-amo-i1135-attncluster, 1793058841 (claim law: first push wins; G1 feasibility read + kernel work first, G2 per-run-interleaved from birth per the 1130 confound law, gated on a verified-idle window). Filed 2026-10-27 ~06:4x +07 from the i1132 census (Bench 1008, W1+W2 reproduced).
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
