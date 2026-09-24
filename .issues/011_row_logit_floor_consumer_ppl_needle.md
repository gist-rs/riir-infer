# Issue 011 — consume katgpt-core `row_logit_floor` on a low-bit attention path: the ppl + needle@64K half of katgpt-rs Issue 882 P2's G1

**Status:** OPEN — filed 2026-09-25 from katgpt-rs Issue 882 P2 (the primitive landed there, katgpt-rs Bench 888; this is its model-bound quality gate). Not started.

## What exists (katgpt-rs, feature `row_logit_floor`, opt-in)

- `floor_row_sink_exempt(row, n_sink, w) -> RowFloor`: the floor `l̃ = max(l, m_r − w)` over non-sink, unmasked keys. It returns `m_r` / `sink_max` / `n_floored`. Sinks and `−∞` are exempt by construction.
- `LogitCodec::new(bits)` + `encode_into` / `exp_lut_into` + `softmax_coded_into`: a symmetric b-bit code over `[m_r − w, m_r]` and a `2^b`-entry exp-table softmax. It measured **−14.8%** against the exp softmax at N=4096 on the M3 CPU.
- `min_width_for_tv(n, ε) = ln(n/ε)`: the row-independent width that guarantees floor TV ≤ ε. This is the recommended default. The range EMA (`RangeEma`) is narrower but gives up the guarantee.
- `LogitCodec::envelope(w, n_floored)`: the closed-form per-row error bound.

## What this issue owns (the gate katgpt-rs cannot run)

- [ ] **T1 — wire the floor + code into one attention path.** Candidates: `riir-infer-laya` `ops::softmax_rows` on the CPU backend, or the decode `attention_forward` default. Feature-gated default-off; width `w = ln(n_ctx/ε)`; `n_sink` from the model's sink convention (the `kv_sink_window` default is 4).
- [ ] **T2 — ppl Δ within the envelope's prediction.** Use the house gemma-2 fixture (upstream-faithful: Gemma-2 has no QK-norm, and it softcaps attention logits at 50, which bounds the row range the floor sees; Issue 010's contrary claim was refuted). Run 8-bit and 6-bit. Report ppl Δ beside the mean per-row envelope. Bench 888 measured the **context-conditional** TV at ~0.75% (8-bit) and ~3% (6-bit) on synthetic σ=1 rows. Real rows decide whether 6-bit survives.
- [ ] **T3 — needle@64K at 6-bit ≥ baseline − ε.** This is the issue's long-context bar. The floor term grows with n (`A ≤ n·e^{−w}`); the tv-budget width compensates via `ln(n/ε)`, which coarsens the code. Measure that trade-off at 64K.
- [ ] **T4 — sink exemption A/B on a real model.** Bench 888 G1b pinned the synthetic cost of dropping the exemption: +8.3pp context TV at 6-bit. Confirm on real attention rows with real sinks.

## Traps

1. **Sinks are legitimate outliers** (katgpt-rs Issue 882 trap 3). Without the exemption the floor is set from the sink and floors most of the context.
2. **Read context-conditional error, not joint.** With sinks present, the joint TV is sink-dominated and looks ~100× better than what the context actually sees.
3. **Masked keys must stay `−∞`.** A finite mask sentinel is raised by the floor like any other tail entry.
4. **Measure with `--release`**, paired interleave (the katgpt-rs `tests/common/ab_timing.rs` protocol), and record box state with every figure.

Promotion (katgpt-rs side, default-on) waits on T2 + T3 passing here.
