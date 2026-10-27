# Issue 1144 — rung: cut the FWHT butterfly's barrier stages (Bonsai-2 Metal decode rotations)

**Status:** Open — filed 2026-10-27 from Issue 1143 (closed, `85c4da5`). In progress. Bonsai-2 PQ2_0 residual vs llama.cpp on idle bmo: 18.17 vs 16.4 ms/token (1.11×).
**Claimed:** zed-amo-1143-mlp, 1793103600, expires 2026-10-29T12:00:00Z


## Trigger (measured, Issue 1143)

The forward FWHT kernels (`fwht_rotate_forward_f32`, `fwht_forward_copy_f32` in
`crates/riir-infer-gpu/src/deltanet_rotation_cubecl.rs`) run one 256-thread cube per
1024-element Hadamard block, so 5–17 cubes per decode launch, and about 4 launches per
layer (`gdn_out_rot`, `ffn_rotate`, plus the norm staging copies). Issue 1143 hoisted the
fill phase's 4 strided loads per lane (bit-identical) and measured only **+0.9% e2e (3/3,
sub-bar, not shipped)**, so these launches are not load-latency bound. What is left per
launch is **10 `sync_cube` butterfly stages** (log2 1024), each moving 2 smem elements
per thread.

## Tasks

- [ ] **T1 — attribute.** Re-run riir-ai `bench_763` Phase C at the 1143 posture and
  record the per-launch µs of the rotation and staging classes vs the norm class now.
- [ ] **T2 — variant.** Do the low strides in registers: each thread owns 4 contiguous
  elements, so steps 2 and 4 need no barrier. For half < 32, use simdgroup shuffles in
  place of smem round trips. Use the SAME `(a ± b) * inv_sqrt2` per element in the same
  stage order, so it is bit-identical by construction. G1 is the existing CPU-oracle
  tests (`fwht_forward_matches_cpu_bitexact`, `fwht_forward_batched_matches_cpu`,
  `fwht_forward_copy_matches_inplace`) plus a to_bits test against the incumbent.
- [ ] **T3 — G2.** Run `decode_readback_probe_gpu` (`--features gpu,bonsai2_hadamard`) on
  idle bmo with interleaved env arms and digest `d06bffced92427cb` unchanged. Promote at
  ≥1.5%.

## Non-goals

No launch fusion (1134 measured a loss) and no change to the per-element arithmetic.

**Refs:** `.issues/1139_m5_ultra_decode_gap_vs_llamacpp.md` (scoreboard; 1143 record).
