# Issue 1134 — rung: norm-pair epilogue fold (input_norm + mid_norm into the producer kernels' epilogues)

**Status:** Open — filed 2026-10-27 ~06:4x +07 from the i1132 census (Bench 1008, W1+W2 reproduced). Unclaimed.
The trigger: the census ranks the norm pair as the top un-lane'd floor class — **input_norm 1.11 + mid_norm 1.08 = 2.19 ms/token combined (11.8% of ALL 14.89), 128 launches x ~17 us/launch** (W1/bmo idle), reproduced W2/amo at 1.88 + 1.67 = 3.55 (same launches, +16% common-mode). The pair wraps every layer (64x input_norm + 64x mid_norm), each its own dispatch at ~17 us — squarely the "fewer, fatter launches" lever.

## The shape (why this is NOT the 905 NULL class)

Bench 905 (i1133 conv-merge) nulled because the merged kernel staged expand work through source-block smem — a cross-block dependency that crushed occupancy (16 blocks vs 72, ~6x serial depth). The norm fold is the OPPOSITE shape: fold each norm into the PRECEDING kernel's EPILOGUE (the promoted Bench-1130 concat-split pattern) — the norm work distributes over the producer kernel's existing threads, no new dependency edge, no staging. Producer kernels: input_norm folds into the previous layer's out_proj/gemv epilogue (or the embedding gather), mid_norm into the preceding attn/gdn block's last kernel. Dispatch saving: 128 x ~17 us ~= 2.19 ms/token upper bound IF fully recoverable — at the 905-calibrated reality (dispatch savings partially eaten), even 30-50% recovery = +4-7% e2e, orders over the 1.5% bar.

## The gate (A/B-led, per the 905 lesson)

- G1 bit-identical: the folded norm must produce to_bits-equal outputs vs the 2-kernel chain (norm math unchanged, only its execution site moves; f32 accumulation order per-element preserved by construction — each thread norms its own elements with the same reduction shape... VERIFY: RMSNorm has a cross-element reduction (rsqrt(mean(x^2))); the epilogue fold must keep the reduction exact — if the producer kernel's partition doesn't own a full norm row per thread-group, the fold is NOT trivial and G1 decides it. The row-parallel partition of the producer gemv may or may not align with norm rows — measure, don't assume).
- G2 paired-interleaved e2e A/B (the 905 4-pair shape, B734_ROUNDS-style warm discipline, ramp tripwire): promote-if >= 1.5% e2e, sign consistency 3/4 minimum, vacuous-guard counters (the fold must actually fire — REACH line per window), demote hatch (=0 byte-identical, posture-gated like RIIR_GDN_CONV_MERGE).
- Census hooks: the norm bits exist (DECODE_STAGE input_norm/mid_norm) — the A/B harness reads their deltas for free.

## Non-goals

- NOT the ffn band (bandwidth lane), NOT the attention cluster (i1135), NOT gdn_out_gemv (i1136), NOT any gdn-chain merge (the 905/1131 budget caution stands there).
- NOT unconditionally: if G1 shows the producer partition misaligns with norm rows, the fold dies at G1 — file the read and close (the 905 disposition shape: the NULL is also a result).

**Refs:** riir-ai `.benchmarks/1008_issue1132_metal_launch_floor_census_w1.md` (W1+W2 tables + the four-run common-mode law), `.benchmarks/905_i1133_conv_merge_ab.md` (the NULL mechanism + the A/B discipline), `.issues/1130` (the promoted epilogue-fold precedent), `.issues/1132` (the census + its lease history).
