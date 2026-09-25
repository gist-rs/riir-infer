# Issue 011 — consume katgpt-core `row_logit_floor` on a low-bit attention path: the ppl + needle@64K half of katgpt-rs Issue 882 P2's G1

**Status:** OPEN — filed 2026-09-25 from katgpt-rs Issue 882 P2 (the primitive landed there, katgpt-rs Bench 888; this is its model-bound quality gate). T1 LANDED (`8fa6e88`); T2 PASS + T4 CONFIRMED (Bench 003, 2026-09-25, M3); T3a (gemma 64K-width proxy) 6/6 and T3b (MiniCPM5 16K, real dilution) 3/3 at every arm; T3c (MiniCPM5 64K) measuring.

## What exists (katgpt-rs, feature `row_logit_floor`, opt-in)

- `floor_row_sink_exempt(row, n_sink, w) -> RowFloor`: the floor `l̃ = max(l, m_r − w)` over non-sink, unmasked keys. It returns `m_r` / `sink_max` / `n_floored`. Sinks and `−∞` are exempt by construction.
- `LogitCodec::new(bits)` + `encode_into` / `exp_lut_into` + `softmax_coded_into`: a symmetric b-bit code over `[m_r − w, m_r]` and a `2^b`-entry exp-table softmax. It measured **−14.8%** against the exp softmax at N=4096 on the M3 CPU.
- `min_width_for_tv(n, ε) = ln(n/ε)`: the row-independent width that guarantees floor TV ≤ ε. This is the recommended default. The range EMA (`RangeEma`) is narrower but gives up the guarantee.
- `LogitCodec::envelope(w, n_floored)`: the closed-form per-row error bound.

## What this issue owns (the gate katgpt-rs cannot run)

- [x] **T1 — wire the floor + code into one attention path.** **LANDED `8fa6e88`** — gemma-2 f16 decode attention (`transformer::attention_floor`, feature `row_logit_floor`) behind `ForwardContext.logit_floor` (`None` ⇒ the plain path, unchanged); fused via katgpt-core `floored_coded_exp_inplace` (katgpt-rs `0adadacdd`, bit-identical to the three-step form). Bin `row_logit_floor_ppl` (ppl mode + `--needle` passkey mode `ed79065`, fixed-width `n65536` arms stand in for the 64K code width). m_Y probe (katgpt-rs 882 P4) rides the needle mode, `dc0e5a9`. Per-row exp-table rebuild NOT hoisted: 26×8×255 ≈ 53K exps/token against ~2.6 GFLOP of matvec, <0.01% — measured-irrelevant, not deferred. Candidates: `riir-infer-laya` `ops::softmax_rows` on the CPU backend, or the decode `attention_forward` default. Feature-gated default-off; width `w = ln(n_ctx/ε)`; `n_sink` from the model's sink convention (the `kv_sink_window` default is 4).
- [x] **T2 — ppl Δ within the envelope's prediction.** **PASS at 8/6-bit ([Bench 003](../.benchmarks/003_row_logit_floor_ppl_needle.md), gemma-2 f16, 4096 tok):** b8 +0.033% ppl / 0.17% top-1 flips, b6 +0.066% / 0.90%, b4 +0.264% / 4.32%. The floored fraction is flat across bits (3.8%), so the code term dominates. |ΔNLL| is linear in the code step `w/(2^b−2)` (0.084 nats/step); the per-row envelope bounds it at every width. Use the house gemma-2 fixture (upstream-faithful: Gemma-2 has no QK-norm, and it softcaps attention logits at 50, which bounds the row range the floor sees; Issue 010's contrary claim was refuted). Run 8-bit and 6-bit. Report ppl Δ beside the mean per-row envelope. Bench 888 measured the **context-conditional** TV at ~0.75% (8-bit) and ~3% (6-bit) on synthetic σ=1 rows. Real rows decide whether 6-bit survives.
- [ ] **T3 — needle@64K at 6-bit ≥ baseline − ε.** T3a proxy PASS (6/6, 0 flips, m_Y moves ≤ 0.0009). T3b 16K PASS (3/3, 0 flips, floored 8.9% vs 3.8% at ≤1K, so the floor term grows with n as predicted; n = 12 tokens, so it cannot rank arms). T3c at 64K is running (`/tmp/ri011run/t6_minicpm64k.log`, launched 20:28). This is the issue's long-context bar. The floor term grows with n (`A ≤ n·e^{−w}`); the tv-budget width compensates via `ln(n/ε)`, which coarsens the code. Measure that trade-off at 64K.
- [x] **T4 — sink exemption A/B on a real model.** **CONFIRMED (Bench 003):** without the exemption, b6s0 floors 1.58× the keys, |ΔNLL| rises 1.35× and flips 1.49×, while aggregate ppl looks BETTER (+0.044% vs +0.066%, sign cancellation). The exemption is load-bearing, and a ppl-only gate would have promoted the defect. Bench 888 G1b pinned the synthetic cost of dropping the exemption: +8.3pp context TV at 6-bit. Confirm on real attention rows with real sinks.

## Traps

1. **Sinks are legitimate outliers** (katgpt-rs Issue 882 trap 3). Without the exemption the floor is set from the sink and floors most of the context.
2. **Read context-conditional error, not joint.** With sinks present, the joint TV is sink-dominated and looks ~100× better than what the context actually sees.
3. **Masked keys must stay `−∞`.** A finite mask sentinel is raised by the floor like any other tail entry.
4. **Measure with `--release`**, paired interleave (the katgpt-rs `tests/common/ab_timing.rs` protocol), and record box state with every figure.

Promotion (katgpt-rs side, default-on) waits on T2 + T3 passing here.
