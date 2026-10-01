# Bench 016 — DeltaNet threadgroup-staged multi-token recurrence (riir-ai Issue 1004 R1)

**Status:** STAGE-ISOLATED GOAT; e2e gain BELOW THE NOISE BAND → feature `deltanet_recurrence_smem_staged` stays OPT-IN (2026-10-01). G1 holds end to end (logits FNV `99a0733c45a0e663` = the pinned anchor); G2-e2e is 1.006× @2048 / 1.019× @4096, inside a ±10% round spread.

## What

`DeltanetRecurrenceStagedCubeCL` (`crates/riir-infer-gpu/src/deltanet_recurrence_staged_cubecl.rs`) is the shipping
`DeltanetRecurrenceMultiTokenCubeCL` with `rows` row-planes per cube and k/q (+ the cube's v slice + β/α) for `tb`
tokens staged in threadgroup memory: cooperative coalesced loads, one barrier per block. Each plane still owns one
state row in 4 registers per lane, and its per-token expression sequence, including both `plane_sum`s, is unchanged.

## G1 — bit identity

`deltanet_recurrence_staged_cubecl::tests::staged_is_bit_identical_to_multi_token`: output AND carried state compared
word-for-word against the shipping kernel, 6 shapes × P ∈ {1, 7, 16, 63, 64, 130}, n_head 4, non-zero carried state.
**0 differing words in every cell.** Revert probe: reading v at `tt * rows` (dropping `local_row`) reds it
(`rows=4 tb=16 p=1: 384 output / 49152 state words differ`).

## G2 — stage-isolated A/B (recurrence only)

`tests/bench_1004_r1_staged_recurrence_ab.rs`, Bonsai-2 GDN shape (48 value heads × head_dim 128), synthetic data,
15 interleaved pairs per shape (order alternating), median of per-pair `staged / shipping` time ratios. Every staged
run's output is checked bit-for-bit against the shipping kernel's inside the bench.

PROVENANCE: M3 Max, AC, `powermode 2`, loadavg 4.4–6.9 (Zed + WindowServer + RustDesk; no other GPU job), 2026-10-01.

| P | shipping median | staged (8,16) median | speedup (median ratio) | worst pair |
|---|---|---|---|---|
| 1024 | 6.19 ms | 6.40 ms | **1.097×** | 1.051 (one pair lost) |
| 2048 | 15.17 ms | 13.63 ms | **1.155×** | 0.937 (all won) |
| 4096 | 50.23 ms | 32.00 ms | **1.657×** | 0.726 |
| 16384 † | 518.4 ms | 209.9 ms | **2.53×** | 0.476 |

† first run, 9 pairs, loadavg ~7.2. Production prefill chunks at `prefill_chunk_max()` = 4096, so P ≤ 4096 is the
production range; 16384 shows the scaling only.

Shape sweep (speedup at P=2048 / 4096): (4,16) 0.69 / 0.92 · (8,8) 1.02 / 1.43 · **(8,16) 1.155 / 1.657** ·
(8,24) 1.10 / 1.65 · (16,8) 0.91 / 1.29 · (16,16) 1.00 / 1.47 · (32,8) 1.15 / 1.66. (8,16) is the default: tied
best with (32,8) and (8,24), with the smallest threadgroup footprint of the three (17.0 KB).

**Why it scales:** the shipping kernel's time grows faster than P (6.2 → 15.2 → 50.2 ms for 1K → 2K → 4K). Its
6144 one-plane cubes drift apart in `t`, so the 128 row-cubes of a head stop sharing k/q lines in cache and the
re-reads go to DRAM. Staging reads each k/q element once per 8-row cube instead of once per row.

## G2-e2e — the production prefill, one process

riir-ai `crates/riir-gpu/tests/bench_1004_r1_staged_recurrence_e2e.rs`, pre-rotation Bonsai-1 `Ternary-Bonsai-27B-Q2_0.gguf`
(the Hadamard-folded Bonsai-2 file cannot run Metal prefill yet: the CubeCL body refuses folded weights, Issue 980
T4-ALT, now riir-infer Issue 032). It has the same GDN shape, which is all this kernel sees. Shipping knobs, 5
interleaved rounds after a warm pair. Every round asserts bit-identical logits, and that each arm made (staged) or
skipped (shipping) staged launches (`staged_recurrence_launch_count`).

PROVENANCE: M3 Max, AC, `powermode 2`, loadavg 4.6–5.2, GPU exclusive (a sibling's concurrent run was killed and
this one restarted clean), 2026-10-01 08:44–09:04.

| P | shipping | staged | speedup median (min–max) | logits FNV |
|---|---|---|---|---|
| 2048 | 89.4 tok/s | 94.0 tok/s | 1.006× (0.928–1.128) | `99a0733c45a0e663` — the Issue 1004 G1 pin |
| 4096 | 66.9 tok/s | 68.5 tok/s | 1.019× (0.962–1.117) | `d571603c76f5e75e` |

The medians match the projection below (recurrence ≈ 0.3% of a 2048 prefill and ≈ 1.4% at 4096), but each is
inside the round-to-round spread, so the e2e gain is **not demonstrated**. Verdict: keep opt-in. Re-test when the
prefill gets faster elsewhere (the GEMM wall), which raises the recurrence's share, or at 8–16 rounds in an idle
window.

## Projection (not a measurement)

Recurrence ≈ 15 ms × 48 GDN layers ≈ 0.73 s of a ~19 s Bonsai prefill @2048 → ~0.5% e2e at 2048. At a 16K prompt
(4 × 4096 chunks): ~50 ms → ~32 ms per layer-chunk saves ~3.5 s, a few percent of the 16K prefill. The e2e A/B
decides; this bench does not.
