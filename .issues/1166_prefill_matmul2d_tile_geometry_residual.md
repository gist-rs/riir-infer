# Issue 1166: prefill matmul2d tile geometry at prefill shapes — the residual ~1.07× at pp4096 / ~1.31× at pp512 (post-1146 stack)

**Status:** OPEN — CLAIMED (zed-amo-1166-tiles, expires 2026-11-04T10:00Z). Rung-1 static read DONE 2026-11-02: the premise re-scoped — the 1113 cell-2 sweep already ran (7 configs) and its winner `ktile64` is the PROMOTED production default (`production_geometry()` → KTILE64, Plan 641 T4, riir-ai Benches 1009/1010; `RIIR_METAL_TENSOR_GEOM=64,128,2,4` is the demote) — so ALL 934/935 numbers were already measured ON ktile64. What was never searched: the ktile64 COMPOSITIONS (K-tile 128; NRB 256 and NRA 128 at K64; 8 simdgroups). Instrument landed: `crates/riir-infer-gpu/tests/issue1166_tile_geometry_sweep.rs` (G1 CPU-reference + sentinel, G2 interleaved rotating-start vs the incumbent; clippy-clean at the full posture, `[[test]]` row armed). Run QUEUED on bmo 2026-11-02 ~15:55Z: exclusive job `01a19cf5-e3ef` (script `/tmp/i1166_run.sh`, worktree `/Users/katopz/git/riir-infer-i1166` @ d9f8d9d, notify zed-a5ed5ffa@amo; bmo's own riir-infer checkout is the sibling timer-build lane — untouched, the run goes through an isolated worktree). ⚠ The instrument's doc-comment runbook originally carried a 3-feature SUBSET of the `[[test]]` required-features — cargo would have SILENTLY SKIPPED the target (green zero); runbook fixed same commit. Both Metal boxes were tenanted at queue time (amo: i659 MTP fit + queued 157-T4; bmo: riir-ai bench_809)
**Date:** 2026-11-02
**Found by:** [`.benchmarks/934_issue1146_attn_matmul2d_ab_pp32k_settle.md`](../.benchmarks/934_issue1146_attn_matmul2d_ab_pp32k_settle.md) verdict §2 + [`.benchmarks/935_issue1146_2box_tb5_width_leg.md`](../.benchmarks/935_issue1146_2box_tb5_width_leg.md)
**Session:** zed-amo-1146-finish, 1793636000

## The finding

After the 1146 stack (tensor zero-copy `54f0dce` + H4 `29be9a0` + relaxed dist-scoped + the
matmul2d attention arm `3499ec4`), the dist prefill reads 0.76× (pp512) / 0.93× (pp4096) /
1.00× (pp32768) of llama.cpp. The gap is now **kernel-internal**: Bench 925's census puts
FFN + DeltaNet GEMMs at 63% of the promoted pp4096 frame, both already ON the
tensor+H4+relaxed path — so the residual is the `gemm_ternary_metal_tensor` matmul2d tile
geometry itself at prefill shapes (w512-class frames), not dispatch, not attention, not the
wire (Bench 935: the 2-box wire is not the floor — pp512/w128 runs FASTER than 1-box).

## Ground held by 1146 (do not re-derive; read the issue arc in HISTORY.md)

- **H1 (per-GEMM submission + copy-back) ≤ ~2%, DEMOTED to last** (static bound, 1146).
- **H2 as first phrased (f32-from-device B) is NOT a fork difference** (1146 source read).
- **H3/H5 (smem staging + 2 barriers/K-tile) CANNOT explain the gap** — upstream llama.cpp
  `19ab5a5` runs IDENTICAL staging and is faster (1146 source read, `mul_mm.metal:20-140`).
- **The two confirmed fork differences are BOTH landed + promoted**: relaxed_precision
  (theirs ON; `6125f9e`) and H4 packed-dequant granularity (theirs `dequantize_pq2_0`,
  4 bytes/16 elems + block scale; ours `cff3fda`). After both, the remaining structural
  delta is **two bit-plane loads vs one packed byte stream per 16 elements** (1146 assessed
  "minor" — UNMEASURED at the new posture).
- **llama.cpp tile**: NRA×NRB 64×128, K-tile 32, 4 simdgroups — same as ours; the fork
  `73bfe6c` variants may differ (fa.metal Q=32/C=64 shapes measured well in 4(c)).

## Candidate rungs (each its own GOAT; order by expected value)

> **RUNG-1 RE-SCOPE (2026-11-02, static read):** rung 1's "sweep the tile geometry"
> premise was PARTIALLY STALE — the sweep exists (1113 cell-2), won, and its winner is
> the promoted default. The re-scoped rung 1 = the SECOND-GENERATION sweep: the
> 1113 list (fork 64×128×K32, coop2 64×64/32×64×2sg, wide 128×128×K32, ktile64
> 64×128×K32, single 32×32×1sg, longp 64×256×K32) never crossed the winner with the
> axes it varied. Instrument: `tests/issue1166_tile_geometry_sweep.rs` (ktile64 as
> the reference arm; challengers ktile128_64x128, ktile64_64x256, ktile64_128x128,
> ktile64_64x128x8sg, ktile64_64x128x2sg; the K32 demote as the floor).

1. **Tile-geometry sweep at prefill shapes** (w512 frames, M=512): NRA/NRB/K-tile/simdgroup
   count against the Bench-895 spill cliff (regs=128/local=0 is the constraint). The fork's
   `kernel_mul_mm` tile matrix vs ours at M=512 is a static read first.
2. **The packed-byte dequant load** (2 planes → 1 packed stream per 16): the last assessed
   structural delta; bit-exact class if the decode math is identical (pack/unpack round-trip).
3. **H5 cooperative-tensor** (dequant → `cooperative_tensor` → matmul2d from registers, no
   threadgroup staging): Apple WWDC26 session 330 + Modular `AppleM5MatMul` name the path;
   1146 kept it as a possible ABSOLUTE win only. Mind the fast-math fma fence (`select()`)
   when smem math moves to registers.
4. **f16-staged B** (halve the B bytes): LOSSY-class — needs the per-family KL instrument
   (`issue1146_relaxed_kl` pattern) + the lossy-surface promotion rule.

## Acceptance

A prefill GOAT on the same rig: the Bench-934 posture (w512, promoted stack) as the arm-A
incumbent; bar ≥1.5% e2e at pp4096 AND no pp32768 regression (the 4(a) lesson); G1
bit-exact for exact-class rungs, the KL instrument for lossy-class; G3 decode untouched.
The league pp rows (riir-ai scoreboard) are the downstream beneficiary — a win is a league
lever.

## Boundary

Substrate-side: `crates/riir-infer-gpu/src/gemm_ternary_metal_tensor.{rs,metal}`. No
riir-ai work required; the league re-cell is a riir-ai side effect.
