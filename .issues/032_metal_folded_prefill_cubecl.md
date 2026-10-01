# Issue 032 — Metal CubeCL prefill for Hadamard-folded (Bonsai-2) models

**Status:** DONE 2026-10-01 — T1-T6 landed (commit cited in HISTORY.md); Metal batch-prefills Bonsai-2 for the first time: **100.9 tok/s @2048 vs 15.7 tok/s token-by-token (6.44×)**, G1 vs folded decode worst rel err ≤ 1.5e-5, pre-rotation pin exact. Unblocks every M3 prefill cell on the production Bonsai-2 PQ2_0 file (riir-ai Plan 602 Phase C / Plan 607 A4 name this as the hard predecessor; neither plan owned the task).

## Why

`TernaryDeltanetGpuForward::prefill_tokens_chunk` (`ternary_deltanet_gpu_forward.rs` ~7159) **panics** on a folded model whenever the cudarc whole-prefill lane is unavailable — on macOS it is compiled out, so **Metal can never batch-prefill Bonsai-2**. The only way a Bonsai-2 prompt reaches the M3 GPU today is token-by-token through the decode path (~24 tok/s AC, Plan 602 C3), against ~90-107 tok/s batched prefill on the pre-rotation file. Every M3 prefill row (riir-ai Issue 1004's Bonsai-2 re-cell, Plan 607 T1/T2/T4) is blocked behind this refusal.

Measured consequence (2026-10-01): riir-ai Issue 1004 R1's full-model staged-recurrence A/B had to run on the pre-rotation Bonsai-1 `Q2_0` file, because the production file cannot reach the prefill body at all.

## Substrate (consume, don't build)

- Rotation kernels already take `[p × width]` batches: `RotationCubeCL::{launch_forward, launch_inverse, launch_forward_copy, launch_gdn_v_permute}` (`deltanet_rotation_cubecl.rs`; signs index `base % width`; a 3-row batched unit test exists).
- Decode eager path (Plan 602 B3) is the per-site spec: embed-inverse · layer-input copy-rotate (`norm_x` stays PRIMAL for dense a/b) · qkv/z read rotated · dense f32 a/b GEMV on PRIMAL input · ssm_out permute+rotate · attn input rotated · attn_out rotate · FFN input rotate · ffn_hidden rotate · lm_head rotated.
- cudarc whole-prefill lane (`prefill_cuda_full.rs`) is the prefill-shaped reference for op order.
- Dense f32: `GemvBatchedCubeCL` exists but serialises the batch inside one plane (6 workgroups for 48 rows) — a token-grid variant keeps the per-(row, token) reduction order of `gemv_plane_f32`, so prefill a/b are bit-identical to decode's a/b per token.

## Tasks

- [x] T1 token-grid dense GEMV (`GemvBatchedCubeCL::launch_token_grid`) + unit test bit-identical to per-token `GemvCubeCL::launch_plane`
- [x] T2 folded wiring in the CubeCL prefill body: embed-inverse, rotated normx scratch (`rotx_b`), qkv/z from `rotx_b`, dense a/b from primal `normx_b`, ssm_out permute+rotate, batched-attention input + attn_out rotate, sequential-attention path, FFN input + hidden rotate, final lm_head rotated
- [x] T3 lift the refusal for the CubeCL body (keep it for any sub-path not wired: e.g. ANE seams already fail open; non-macOS `cuda_ffn` block stays refused on folded)
- [x] T4 G1 (Metal, M3): folded prefill vs folded decode-eager per token (decode-eager is itself G1-gated vs CPU, Plan 602 B4) — top-1 equal, top-20 overlap, max rel err reported; P ∈ {64, 128, 512}
- [x] T5 G3 no regression: pre-rotation `Q2_0` prefill pin `fnv 99a0733c45a0e663` @2048 reproduced exact
- [x] T6 first Bonsai-2 M3 prefill tok/s (unscored, PROVENANCE line; the league cell stays Plan 602 C4 / Ultra)

## Results (M3 Max, AC, powermode 2, loadavg 4.6-5.0, one concurrent agent session; GPU otherwise idle)

- **T1:** `gemv_token_grid_plane_f32` / `GemvBatchedCubeCL::launch_token_grid` — 0 differing words vs per-row `launch_plane` at 37 × 5120 × 48 (`test_gemv_token_grid_bit_identical_to_plane`).
- **T4 G1** (`tests/g1_032_folded_prefill_metal.rs::folded_prefill_matches_folded_decode`, 209 s): every P ∈ {64, 100, 128, 512} — top-1 equal, top-5 1.00, top-20 1.00, worst rel err **1.45e-5 / 5.5e-6 / 6.8e-6 / 1.3e-5** at the last position, and the same bar one decode step after prefill (state handoff: 5.2e-6 / 5.2e-6 / 6.8e-6 / 2.3e-6). P=100 exercises the sequential conv1d fallback. **Revert probe:** dropping the ffn_hidden rotation (site 7) reds it — top-5 0.00, argmax 198 vs 770.
- **T5 G3** (`prerotation_prefill_pin_unchanged`): `Q2_0` P=2048 fnv **99a0733c45a0e663**, argmax 332 — exact.
- **T6** (`folded_prefill_throughput`, unscored): Bonsai-2 PQ2_0 P=2048 batched prefill **100.9 tok/s** (median of 3, 20.27-20.49 s, deterministic fnv `0ec2396fd4627f29`) vs **15.7 tok/s** token-by-token on the same prompt (256-token sample) = **6.44×** time-to-first-token. Same session's pre-rotation `Q2_0` reading for scale: 89.4 tok/s @2048 (riir-ai Issue 1004 e2e, shipping arm) — different lineage, not comparable as a cell.
- **Not done here (by design):** the league cell (riir-ai Plan 602 C4, Ultra-gated) and any perf work on the 8 extra dispatches per layer (the fused-rotation rung, Plan 602 B5's follow-up).

## Gates

- G1 above is a tolerance gate (prefill uses batched ternary GEMMs whose reduction order differs from decode GEMV — the same posture as the pre-rotation prefill-vs-decode relation).
- G3 bit-exact on the pre-rotation file: every folded branch is `rot_tables.is_some()`-keyed, so the old-file path must be byte-for-byte unchanged.

## Refs

riir-ai Plan 602 (Phase C, non-goals), Plan 607 A4, Issue 1004, Issue 980 T4-ALT; `prefill_cuda_full.rs` (cudarc reference); `g1_bonsai2_metal_parity.rs` (decode G1).
