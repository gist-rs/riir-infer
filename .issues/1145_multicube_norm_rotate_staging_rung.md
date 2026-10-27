# Issue 1145 — rung: multi-cube norm + rotate staging (redundant reduction, double-buffered x)

**Status:** Open — filed 2026-10-27 from Issue 1139 T4.
**Claimed:** zed-amo-i1145-multicube, 1793100861, expires 2026-10-29T11:34:21+00:00

## Trigger (measured, Issue 1139 T4)

The bench_763 census at the 1144 posture (idle bmo, Phase C) finds two Bonsai-2-only
classes on the decode norm sites: `input_norm` 0.80 vs v1 0.46 and `mid_norm` 0.81 vs
0.44. Together that is **+0.71 ms/token**: the `fwht_forward_copy` staging launch after
each norm, 128 launches per token. The obvious fold, Issue 1134's
`residual_add_rmsnorm_rotate_f32`, re-measured today at **17.55 → 21.19 ms/token
(−20%, 3/3, `RIIR_NORM_ROT_FUSED=1`)**. Its single 256-thread workgroup walks the 5
Hadamard chunks serially, so it trades one ~5.5 µs launch for ~33 µs of serial work.

## The shape that could win

Use one launch of `dim / hblock` cubes (5 at Bonsai-2), where each cube:
1. Redundantly runs the incumbent's Phase 1 + tree reduction over the whole vector. It
   uses the same per-lane strided order and the same tree, so every cube gets the
   **same bits** for `inv_rms`, plus the Issue-1143 4-deep hoist. The reads are 2×20 KB,
   cache-resident after the first cube.
2. Normalizes ITS chunk, writes `norm_x[chunk]`, and stages the rotated chunk with the
   Issue-1144 radix-4 schedule into `rot_scratch[chunk]`.

**The blocker** is the in-place residual add `x += residual`. A cube writing its own
chunk of `x` races the other cubes that are still reading `x` for their reductions,
which is the Issue-1136 "no-owner" class. The design therefore needs `x` double-buffered
across the norm site (read `x_a`, write `x_b`, swap per site) in
`ternary_deltanet_gpu_forward.rs`. That is a forward-wide buffer-ownership change, so
audit every `self.x` reader in decode, the Plan-623 stage boundary, and capture.

## Tasks

- [ ] **T1 — price.** Count the `self.x` readers and writers on the decode path and the
  stage-boundary handoff, and decide whether the swap is local to the two norm sites.
- [ ] **T2 — kernel + G1.** to_bits vs the incumbent pair (`ResidualAddRmsNormCubeCL` +
  `launch_forward_copy`) on `x_out`, `norm_x` and `rot_scratch`, at 5120/1024 and an
  off-shape. Keep the Metal fast-math fence where products meet adds in registers
  (memory `metal-fastmath-fma-fence`).
- [ ] **T3 — G2.** Run `decode_readback_probe_gpu` (`--features gpu,bonsai2_hadamard`) on
  idle bmo with interleaved arms and digest `d06bffced92427cb` unchanged. Promote at
  ≥1.5%.

## Non-goals

No reduction-order change, and no single-workgroup serial walk (1134 already measured
it).

**Refs:** `.issues/1139_m5_ultra_decode_gap_vs_llamacpp.md` (T4 census),
`crates/riir-infer-gpu/src/norms_cubecl.rs` (`residual_add_rmsnorm_f32_mlp4`,
`residual_add_rmsnorm_rotate_f32`), `crates/riir-infer-gpu/src/deltanet_rotation_cubecl.rs`
(`fwht_forward_copy_f32_r4`).
