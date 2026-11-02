# Bench 934 — Issue 1146: ladder 4(c) matmul2d flash-attention A/B + the relaxed-OFF pp32k settle

**Status:** RECORD — (placeholder; numbers land when the bmo runner completes)
**Date:** 2026-11-02
**Session:** zed-amo-1146-finish, 1793612565 (lease `3f4f828`)
**Issue:** [`.issues/1146_prefill_throughput_gap_1box.md`](../.issues/1146_prefill_throughput_gap_1box.md)
**Box:** bmo (Mac17,15, M5 Ultra, 30 cores, 96 GiB), AC, load 1.11 at start; runner pid 51651
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` (league pin, 7,206,168,928 B)
**Build:** riir-infer `3f4f828` (clean detached worktree `/tmp/i1146f-wt`, sibling WIP excluded; katgpt-rs `6c7656193`), release, `-p riir-infer-dist --bin dist_stage --features gpu,bonsai2_hadamard`
**Rig:** `scripts/i1146_width_recurve.sh`, one binary both arms, w512, per-pair GPU-quiet guard, `I1146W_REPS=2`

## Phase 1 — ladder 4(c): `RIIR_PREFILL_ATTN_METAL_TENSOR=1` (kernel `672b29c`)

Arm A = the promoted dist posture (tensor zero-copy + H4 + dist-relaxed, CubeCL attention);
Arm B = Arm A + the matmul2d flash-attention attention arm (opt-in selector, env-only).
Engagement proof: arm B logs carry `[Issue 1146] prefill attention = matmul2d tensor arm`;
arm A carries none. G1 pp1024×16tok cross-arm: IDENTICAL 16-token streams.

| cell | a (promoted) | b (+attn tensor) | b vs a |
|---|---:|---:|---:|
| pp512 w512 r1 / r2 | — | — | — |
| pp4096 w512 r1 / r2 | — | — | — |
| pp32768 w512 r1 / r2 | — | — | — |

## Phase 2 — per-family KL of the attention arm (`issue1146_relaxed_kl`, `I1146K_AXIS=attn`)

Ref = CubeCL attention chain, cand = matmul2d attention, tensor GEMM ON in both arms
(the serving posture), RELAXED=1. Launch-counter vacuous guard rides the run.

| family | n | KL mean | KL max (nats) | top1 | ΔNLL mean | NLL(ref) |
|---|---:|---:|---:|---:|---:|---:|
| code | — | — | — | — | — | — |
| math | — | — | — | — | — | — |
| prose | — | — | — | — | — | — |
| zh | — | — | — | — | — | — |

## Phase 3 — relaxed-OFF pp32768 settle (the `401af78` claim)

Arm A = tensor OFF + relaxed OFF (simdgroup incumbent); Arm B = tensor ON, relaxed OFF.
The `401af78` containment fix claim (tensor no longer slower at pp32k once zero-init
refills stopped) gets its guard-clean pairs here.

| cell | a (tensor off) | b (tensor on) | b vs a |
|---|---:|---:|---:|
| pp32768 w512 r1 / r2 | — | — | — |

## Provenance

- Runner log: `bmo:/tmp/i1146f/runner.log`; artifacts synced to `i1146_finish/` on landing.
- Binary sha256s: `bmo:/tmp/i1146f/bin_sha.txt`.
- CONTAM counts: phase 1 `.benchmarks/i1146_finish/` sync.
