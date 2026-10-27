# Issue 1138 — the hand-MSL FFN decode-GEMV lane (i1137 reopen lever (a), adjudicated GO)

**Status:** OPEN, CLAIMED — owner gate adjudicated **GO** 2026-10-27 by delegated verdict (session `zed-amo-msllane-gate`; the `claude_code` reviewer backend was auth-unavailable on amo, a native subagent stood in and verified every quoted number against the source records before ruling AGREE). The lane: a hand-written MSL kernel outside CubeCL codegen, per the i1137 mapping study — target the fork's ~53%-of-stream class at the FFN shapes (prize ≈ −3.2 ms/tok ≈ +13% e2e; even 45% ≈ +8%; promote bar +1.5%). Standard ladder throughout; `RIIR_GEMV_TQ2_FIELD` default-OFF.

**Claimed:** zed-amo-msllane-gate, 1793069182, expires 2026-10-29T02:46:22Z

## Tasks

- [ ] Substrate-first: inventory the existing Metal dispatch surface in `riir-ai/crates/riir-gpu` — raw `MTLFunction`/source-compile path vs CubeCL-only — and name the integration point for a hand-MSL kernel; verify no parallel dispatch system already exists (grep 3+ vocabulary variants: msl, metal kernel, new_function_with_source, dispatch threads).
- [ ] Repack arm in `TernaryHandle::from_weights`: pos/neg planes → byte fields `c = (p − n) + 2 ∈ {1,2,3}` packed 4-per-byte (`v = c0 | c1<<2 | c2<<4 | c3<<6`); value-exact recovery `d = c − 2`; upload-bytes parity assert (4 weights/byte vs 2×1-bit planes).
- [ ] MSL kernel `kernel_mul_mv_tq2_field_f32` at the fork geometry (4 rows/simdgroup × 2 simdgroups = 8 rows/threadgroup, 64 threads): per-block `coef[4]` + `sumy` hoist from y, exact float-floor field extraction (`floor(v·0.25)` class — no per-bit selects), float4 y loads, `sum = −sumy + Σ coef[j]·f_j(v)`.
- [ ] Dispatch + toggle: `RIIR_GEMV_TQ2_FIELD` (default OFF, `=0` demote hatch byte-identical to incumbent), one-shot `[tq2_field] REACH:` stderr line (the Bench-768 vacuous-guard law).
- [ ] G1 gate (the Bench-742 law — NOT to_bits, the offset fold reorders f32 accumulation): unit max-abs ≤ the 742 band vs shipping `gemv_ternary_plane_rowtiled8_f16scale` at both FFN shapes (m=17408/n=5120, m=5120/n=17408) + lm_head; greedy-token loop-stream identity e2e on the real model.
- [ ] bench_1141-methodology A/B: cold-set rotation, interleaved medians, aspect curve + e2e ABBA; promote bar +1.5% e2e; record in `.benchmarks/`, promote or keep as demote-hatched opt-in per measured direction. GPU exclusivity check before every window.
- [ ] Docs + close: bench record cross-linked from i1137 (reopen lever (a) → landed/closed), resolution note, remove this file per the noise-reduction rule.

## Non-goals / scope guards

- NOT lm_head-first — it already reads 51% of stream; the FFN pair (26–35%) is the pool.
- NOT any further CubeCL variant — that dimension is closed exhaustively (Bench 1140 + 1141 argmin record: plane/rowtiled4/rowtiled8-f32/f16scale/FMA-telescoped, none past ~35%).
- NOT the launch-floor ladder (i1130/i1134/i1135/i1136 — closed NULL/negative) and NOT the GDN rotation re-pin (i1131 — adjudicated NO-GO same day).
- NOT the CUDA lane (riir-infer-gpu) — Metal on amo only; the 4090 inherits nothing from this record.

## Refs

riir-infer `.issues/1137_ffn_gemv_alu_chain_rung.md` (the close record + the full fork mapping study — the design base), Bench 1140 (the BW-curve instrument, `riir-ai/crates/riir-gpu/tests/bench_1140_ffn_gemv_bw_curve.rs`), Bench 1141 (the FMA aspect probe + cold-rotation methodology), Bench 847 (the CubeCL FMA NO-SHIP record), the Bench-742 law (G1 band + loop-stream identity), the Bench-768 law (REACH guard), vendored fork `riir-ai/.raw/prismml-llama.cpp` (`ggml/src/ggml-metal/kernels/mul_mv.metal`, `kernel_mul_mv_tq2_0_f32_impl`).
