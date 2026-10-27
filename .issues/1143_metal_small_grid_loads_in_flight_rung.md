# Issue 1143 — rung: loads-in-flight (the Issue-1142 pattern) for the remaining small-grid Metal decode kernels (Bonsai-2 rotations, norms, gdn_out_gemv)

**Status:** Open — filed 2026-10-27 from Issue 1139 T3. In progress. Bonsai-2 PQ2_0 residual vs llama.cpp: 19.45 vs 16.4 ms/token (1.19×) on idle bmo.
**Claimed:** zed-amo-1143-mlp, 1793100000, expires 2026-10-29T12:00:00Z


## Trigger (measured, Issue 1139 T2 part 2)

At the composed posture, Bonsai v1 Q2_0 decodes at 16.77 ms/token, which is llama.cpp
parity. Bonsai-2 PQ2_0 adds these classes on top (riir-ai `bench_763` Phase C,
dispatch-only + drain, Bonsai-2 − v1, bmo idle):

| class | Bonsai-2 | v1 | delta | launches/token |
|---|---|---|---|---|
| `gdn_out_rot` + `ffn_rotate` (Hadamard rotations) | 1.20 ms | 0.01 | +1.19 | 96 at ~12 µs |
| `input_norm` + `mid_norm` | 3.17 ms | 2.25 | +0.92 | 128 (same count) |
| `gdn_out_gemv` (all archs, for reference) | 0.71 ms | 0.70 | ≈0 | 48 at ~15 µs |

Issue 1142 showed what the floor-class µs/launch hides on this box. The dense pair's
~52 µs/launch came from **load latency**: few planes, a long strided K walk, one load
in flight. Issuing 8 loads before folding them in the same order recovered
**+9.9% e2e, bit-identical by construction**. Earlier rungs on these classes tried to
**fuse** launches: 1134 (norm into epilogue) measured an e2e LOSS, and 1131/1136 hit the
bit-identity wall on CUDA. None of them tried to raise memory-level parallelism inside
the incumbent kernel while keeping the reduction order. This rung is that axis.

## Tasks

- [ ] **T1 — read the kernels.** For each class (rotation FWHT kernels, the rmsnorm
  kernels on the decode path, `gdn_out_gemv`), record the grid (planes/workgroups at
  decode), the per-lane loop trip count, and the loads in flight per iteration. Only
  classes with a long dependent-load walk on a small grid are candidates. The butterfly
  stages of an FWHT are NOT; its load/store fill phases may be.
- [ ] **T2 — variant per candidate.** Hoist N loads ahead of the unchanged fold order
  (the `gemv_plane_f32_pair_mlp8` shape). Unit G1 must be to_bits EXACT against the
  incumbent at the real decode shapes plus a ragged tail. Ship each variant behind the
  `dense_pair_env_semantics`-style switch.
- [ ] **T3 — G2.** Use riir-infer-dist `decode_readback_probe_gpu` on idle bmo,
  interleaved env arms, with the logits-stream digest unchanged (`d06bffced92427cb` at
  the 1142 posture). Promote a variant if it gains ≥1.5%, demote it otherwise, and
  record the depth sweep as 1142 did.

## Non-goals

No launch fusion (1134's measured loss), no reduction-order change (the bit-identity
wall), no CUDA lane (cudarc decode is a separate funnel).

**Refs:** `.issues/1139_m5_ultra_decode_gap_vs_llamacpp.md` (T2 part 2: census + the 1142
record), `crates/riir-infer-gpu/src/gemv_cubecl.rs` (`gemv_plane_f32_pair_mlp8`),
riir-ai `.benchmarks/1008_issue1132_metal_launch_floor_census_w1.md`.
