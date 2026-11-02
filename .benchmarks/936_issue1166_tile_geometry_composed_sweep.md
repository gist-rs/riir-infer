# Bench 936 — Issue 1166: the second-generation tile-geometry sweep (the ktile64 compositions) — incumbent confirmed, NEGATIVE

**Verdict:** NO PROMOTION — the promoted `ktile64_64x128x4sg` (NRA 64 × NRB 128, K-tile 64, 4 simdgroups; Plan 641 T4, riir-ai Benches 1009/1010) is the **composed-neighborhood optimum**. Every unsearched ktile64 composition is noise-band or far worse. The 1146 residual (0.93× at pp4096) is **not tile geometry**. Issue 1166 rung 1 CLOSED negative; the remaining ground is Issue 1167's (packed-byte dequant load, H5 cooperative-tensor, census re-run).
**Date:** 2026-11-02
**Session:** zed-amo-1166-tiles, 1793712000 (claim `f97dcbc`)
**Instrument:** `crates/riir-infer-gpu/tests/issue1166_tile_geometry_sweep.rs` (`d9f8d9d`, runbook fix `6941f77`)
**Raw log:** [`i1166_tiles/sweep_run.log`](i1166_tiles/sweep_run.log)

## Question

Bench 1113 cell-2 swept 7 tile configs and promoted `ktile64` — but never crossed the
winner with the axes it varied independently: K-tile 128; NRB 256 and NRA 128 AT K-tile 64;
8 simdgroups. All Bench 934/935 numbers were measured ON the winner, so the residual was
either in those compositions or elsewhere. This sweep answers the compositions.

## Posture

- bmo (M5 Ultra, AC), EXCLUSIVE rig slot (queue job `01a19cff-9fda`, the second attempt —
  the first died exit 101: `sentencepiece-sys` needs `cmake`, and bmo's non-interactive rig
  shells lack `/opt/homebrew/bin` on PATH; the script now exports it). Worktree
  `/Users/katopz/git/riir-infer-i1166` @ `6941f77` (bmo's own checkout held the sibling
  timer-build lane — untouched; the run went through an isolated worktree + direct ref push).
- `--release`, full 9-feature `[[test]]` required-features posture
  (`metal_tensor_gemm,cubecl_runtime,ternary_gemv,multi_host,ternary_gemm_batched,
  ternary_gemm_simdgroup,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,
  riir-infer-core/bonsai2_hadamard`), `--test-threads=1 --ignored`.
- G2 methodology: interleaved rotating-start, median over 4 rounds × 2 launches per config;
  G1 runs FIRST and is the gate. End provenance `loadavg1m=2.63` (quiet), start 3.19.
- Shapes = the three Bonsai-2 prefill bottlenecks (Bench 925 census: FFN + DeltaNet
  projections = 63% of the promoted pp4096 frame): ffn_gate/up 17408×5120, ffn_down
  5120×17408, attn_q 12288×5120; P ∈ {512, 2048, 4096}.

## G1 — all 7 configs PASS

`worst_rel=0.000e0` on every config against the CPU dequant-GEMV reference
(config-independent truth) + never-written sentinel. Exact class confirmed.

## G2 — per-launch medians, ratio vs the incumbent

| config | gate/up P512 / 2048 / 4096 | ffn_down P512 / 2048 / 4096 | attn_q P512 / 2048 / 4096 |
|---|---|---|---|
| **ktile64_64x128x4sg (incumbent)** | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| ktile128_64x128x4sg | 1.001 / **1.015** / 1.003 | 0.999 / 1.000 / **1.009** | 1.003 / 1.005 / 0.998 |
| ktile64_64x256x4sg | 0.559 / 0.634 / 0.627 | 0.616 / 0.621 / 0.567 | 0.582 / 0.594 / 0.624 |
| ktile64_128x128x4sg | 0.616 / 0.678 / 0.669 | 0.736 / 0.667 / 0.626 | 0.634 / 0.631 / 0.667 |
| ktile64_64x128x8sg | 0.998 / 0.995 / 0.990 | 1.010 / 0.991 / 0.995 | 0.998 / 0.994 / 0.994 |
| ktile64_64x128x2sg | 0.614 / 0.670 / 0.656 | 0.740 / 0.671 / 0.645 | 0.657 / 0.650 / 0.660 |
| k32_64x128x4sg (floor) | 0.994 / 0.994 / 0.991 | 0.984 / 0.991 / 0.996 | 0.989 / 0.996 / 0.994 |

Incumbent absolutes (bmo): ffn_gate/up 5.44 / 19.96 / 39.32 ms; ffn_down 7.19 / 24.59 /
48.23 ms; attn_q 4.09 / 14.80 / 29.18 ms at P=512/2048/4096.

## Verdict

1. **ktile128 is noise, not a win**: 0.998–1.015× across nine cells, inconsistent in
   direction (best +1.5% once, −0.2% at the longest P) — far below the pre-registered
   ≥1.5% e2e bar WITH the no-long-P-regression requirement it fails outright.
2. **The wide tiles collapse** (~0.56–0.74×): NRA×NRB 64×256 and 128×128 at K64 lose
   threadgroup occupancy that the M=512-frame grid cannot hide. NRB 256 / NRA 128 only
   ever measured at K32 in 1113 — now measured at K64 and REJECTED.
3. **Simdgroup count is not a lever**: 8sg ≈ neutral (0.990–1.010×), 2sg ≈ the wide tiles
   (~0.65×). 4 is right.
4. **K-tile is a weak axis**: the K32 floor sits at 0.984–0.996× — nearly flat.

⇒ **Incumbent stands; nothing to promote, nothing to demote.** The one-tested-axes-at-a-time
winner from 1113 survives its own composition space.

## Consequences

- The 1146 kernel-internal residual is **not the matmul2d tile geometry** — the search moves
  to Issue 1167's rungs: the packed-byte dequant load (rung B: 2 bit-plane loads → 1 packed
  stream, the last assessed structural delta vs upstream) and H5 cooperative-tensor
  (rung C, census-gated), plus their census re-run at the promoted+attn posture.
- `RIIR_METAL_TENSOR_GEOM` stays the manual override; its default is confirmed optimal at
  the prefill bottlenecks — no code change landed (test-only instrument).
- Fleet lesson recorded on both lanes: **rig job shells are non-interactive — a build
  chain needing homebrew tools must export `/opt/homebrew/bin` in the job script**
  (sentencepiece-sys → cmake; 1167's census hit the same wall and requeued with the fix).

Cross-refs: Issue 1166 (CLOSED, this bench); Issue 1167 (the surviving rungs, lease
`2c053ec`); Bench 1113 (the first-generation sweep this composes); Benches 934/935 (the
posture whose numbers this explains).
