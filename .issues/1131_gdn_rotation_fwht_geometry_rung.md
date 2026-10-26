# Issue 1131 — T6 rung: GDN rotation-kernel FWHT geometry (the Bench-902 census rows OUTSIDE T3's family)

**Status:** RE-PRICED SUB-BAR, execution claim RELEASED (2026-10-27 ~03:0x +07) — the T3 measured close (Bench 903, `546fdd6`) collapses this rung's honest expectation: the bit-identical fix space of the SAME geometry class (norm tree + 1024-butterfly + quantize) measured **~0.29% e2e on the 788 µs T3 family** (float4-fill + shfl-tail bit-clean; the register-tile butterfly DRIFTS ±1-ULP — the nvcc contraction-pairing class, dead by the FNV-pin law). Scaling the same bit-clean transform ratio to this rung's 564 µs/token rotation family (K2+K3): **expected ~0.2–0.4% — below the 899-class staging bar and an order under the +2.71% re-cross**. The census's 66×/7–12× roofline headroom is real but sits behind the same bit-identity wall T3 hit; reclaiming it is a re-pin (owner) decision, not a modelless rung. The finding that survives: the two rotation kernels carry the same barrier-chain floor signature and the same dead/alive transform split — any future lane executes the V1/V3-class variants ONLY (never the reg-tile class). The Issue-1133 conv-merge NULL (Bench 905) independently confirms the launch-class budget around the GDN chain is exhausted at ~0%.
**Lease:** released — shikuwa-4090-i1131t6 held the scout+filing (done) and the post-T3 execution slot; the slot is released unexecuted on the Bench-903 re-pricing (the measured disagreement is the oracle — the ~2.6% premise was a family-scaled extrapolation, the T3 measurement is the family's own number). Re-claim only on an owner re-pin decision.

## The finding (from [Bench 902](.benchmarks/902_i1127_attention_chain_fwht_census.md), the census table rows outside T3's scope)

The 1127 ladder accounting covered every census row EXCEPT the two GDN rotation kernels — they are neither in T3's `sign_fwht_quantize` family sum, nor a closed rung, nor the GDN core (Bench 898/899 territory):

| kernel | per token (total ms / 729) | % of token | measured floor (min/med/avg µs) |
|---|---|---|---|
| `gdn_input_rotabq_f32` | **301.6 µs** | 2.8% | 5.57 / 6.02 / 6.28 |
| `gdn_out_norm_gate_permute_rotq_f32` | **262.5 µs** | 2.4% | 5.25 / 5.34 / 5.47 |

**Combined ≈ 564 µs/token ≈ 5.2% of the ~10.9 ms decode token** — the second-largest non-GEMV block after T3's 788 µs family, with the SAME tight per-launch floor signature (min≈med≈avg = a geometry floor, not noise).

## Geometry verification (source read, `ROTATION_CUDA_SRC` — the barrier-chain class)

- **K2 `gdn_input_rotabq_f32`** (fused rmsnorm + sign + FWHT butterfly + q8 quantize + the dense a/b escape GEMVs, 6→1): grid = `dim/hblock + 2*rows` blocks at 512 threads. At the league posture (dim 5120, hblock 1024): **5 quantize blocks + 2*rows GEMV blocks on a 128-SM 4090 — heavily under-occupied**. Each quantize block runs the FULL barrier chain: 256-thread strided norm reduction (9 `__syncthreads` tree stages) → 1024-wide butterfly (`rot_butterfly_1024`, log2(1024) = 10 barrier stages) → quantize. **~19 sync stages at one-wave grid depth = pure latency.**
- **K3 `gdn_out_norm_gate_permute_rotq_f32`** (per-head RMSNorm + silu(z) gate + head permute + sign + FWHT + quantize, 6→1): grid = `v_dim/hblock` = **5 blocks of 1024 threads (~4% machine occupancy)**, per-head sumsq tree (8 sync stages) → butterfly (10 stages). Same class, worse occupancy.
- This is the SAME reading the census gave T3's family ("latency/barrier-bound at the 1024-block FWHT geometry, not bandwidth- and not launch-bound") — the GDN rotation kernels are that class's other half. The opponent's #396 widened decode FWHT blocks for exactly this family of geometries; our T3 lever applies verbatim.

## Rooflines (honest, per kernel — do NOT quote the census's 8–95× class here)

- **K3**: traffic ≈ x_in 20 KB + z 20 KB + signs 20 KB + out 5 KB + gamma/ascale ~1.5 KB ≈ 66 KB → ~0.08 µs @850 GB/s vs 5.25 µs min = **~66× over roofline** — nearly pure barrier latency; a MB/wider-parallelism restructure has the full T3-class headroom.
- **K2**: traffic ≈ x + gamma + signs 60 KB + qout/ascale ~6 KB + wa/wb 2×rows×20 KB (rows = the ssm a/b escape-set count at the manifest posture — read at execution; ~8–16 ⇒ 320–640 KB) ≈ 0.45–0.83 µs vs 5.57 µs min = **~7–12× over roofline** — latency-dominated but with a REAL compute arm (the dense GEMV rows), so its headroom is lower than K3's.
- **Sub-lever (K2-specific)**: the `2*rows` GEMV blocks each re-run the FULL-dim norm reduction redundantly (9 more sync stages per block, deliberate for bit-identity — "each GEMV block re-runs the full-row rmsnorm reduction redundantly (bit-identical inv_rms, no cross-block dependency at all)"). A leaf-partial single-reduction variant (the Issue-734-T5 `rmsnorm_quantize_mb` pattern) removes the per-GEMV-block barrier chain without a second launch.

## Headroom + compounding (the league math)

At a conservative 2× on the fused-family floor (the same target T3 prices): ~282 µs/token saved ≈ **~2.6% e2e — at the +2.71% re-cross bar nearly solo**, and comfortably compounding:

- T1 attention-chain (landed, opt-in): **+0.77%** (Bench 900, 4/5 pairs positive)
- Bench-899 beta/expand rung (landed, opt-in): **+0.6–1.0%**
- T3 FWHT family (amo, in flight): **≥3.5%** at 2×
- T6 this rung: **~2.6%** at 2×

Staged together the decode re-cross (+2.71% needed, 91.68 → 94.16 tok/s) does not depend on any single rung delivering its ceiling. K3 alone (66× over) is the safest first slice; K2 second (the GEMV arm bounds it).

## The lever (verbatim T3's, plus the DRY base)

Multi-block / wider-parallelism restructuring toward roofline — the Issue-734-T5 `rmsnorm_quantize_mb` leaf-partial pattern; bit-identical reduction order per the Issue-940 law (a sequential fold order pinned at the site; the butterfly's pairing sequence is part of the pinned order). **Reuse whatever kernel shape the T3 rung lands as the DRY base — build the GDN variants on the landed T3 pattern, never a fork of it.** If T3 lands a shared helper (MB leaf-partials + butterfly restructure), T6 consumes it; the same-file layout makes that natural.

## Execution discipline (carried from 1127, unchanged)

- G1: kernel-level bit-identity vs the incumbent per kernel (all postures incl. KV_F16 and the graph/devpos lane) + real-model argmax-walk FNV `98412dc406c4ea19` + logits[::97] FNV `63496f6688c8f742` EXACT at the manifest posture (`B734_ROUNDS=3`; the walk FNV is rounds-cumulative — a B734_ROUNDS=1 run prints `a287e92e8d3d1021`, posture not drift).
- A/B: paired-interleaved windows on the league posture, REACH guard armed from window 1 (the Bench-900 lesson: an armed window that silently no-ops must be CAUGHT, not read as zero-effect).
- Ships opt-in env (`RIIR_GDN_ROT_MB=1` class — name at execution), `=0` demote hatch in BOTH dispatch paths if the graph lane shares the launcher.
- Promotion rides the owner-gated league re-cell (1127 T4 window) — a measured-positive opt-in never self-promotes.

## Non-goals / scope guards

- NOT T3's family (`rmsnorm/swiglu/gate_sigmoid_sign_fwht_quantize`) — amo's live lease, do not touch until it clears.
- NOT the GDN core (`recurrence_f32_fused_hd128`) — Bench 898 measured NEGATIVE (fusing launch-bound neighbors INTO a compute-bound kernel); conv1d stays refused (true data dependency).
- NOT beta/decay+expand — Bench 899 landed that rung (opt-in).
- NOT the GEMV family (72.6% of the token) — near-roofline bandwidth-bound (the league's k4w4/KB-era history); no rung without new physics.
- NOT attention_decode (10.5%) — KV-read-bound at league ctx; launch-class savings only, sub-bar.

**Refs:** Bench 902 (the census this re-reads), [Issue 1127](1127_decode_hold_remaining_rungs_recell_bundle.md) (the ladder + T4 window), Issue 734-T5 (`rmsnorm_quantize_mb` pattern), Bench 898/899/900 (the rung records + discipline), Issue 940 (reduction-order law), prism #396 (the opponent's wider-FWHT-blocks analogue).
