# Issue 1112: arm8 whole-prefill CUDA crashes on unfolded (pre-rotation) synth shapes — sticky illegal address, poison surfaces downstream

**Status:** Open — found by the Plan 622 T1.3 synth walk's first 4090 run (2026-10-23, zed-m3-tokengate)
**Claimed:** —

## Symptom

`plan622_keybias_walk_cuda` (synth Bonsai-2 GGUF: 4 layers `[DeltaNet ×3, Attention ×1]`, n_embd 1024, GDN n_k=4/n_v=8/hd=128, attention ahd 128, vocab 256) at the whole-prefill CUDA posture (the recl8 knob recipe, graphs off):

1. `[622-walk] cuda probe post-boot: OK` — the CUDA context is healthy after `TernaryDeltanetGpuForward::new`.
2. During the incumbent `fwd.prefill` the arm8 whole-prefill arm errors:
   - without `CUDA_LAUNCH_BLOCKING`: `[734-arm8] weight-mirror build failed: upload pos: CUDA_ERROR_ILLEGAL_ADDRESS` → `FALLTHROUGH: layer loop error: out mirror` — the sticky poison surfaces at the first cudarc upload AFTER the layer-0 GDN chain (the out_proj mirror build); the mirrors for qkv/z/a/b all uploaded clean before it.
   - with `CUDA_LAUNCH_BLOCKING=1`: the error surfaces synchronously at the offending launch — `FALLTHROUGH: layer loop error: kernel launch failed: CUDA_ERROR_ILLEGAL_ADDRESS` — and the failure string carries NO `gdn/<stage>` prefix (the per-stage naming landed `471f56c`), so the crash is in an UNNAMED launch: the input rmsnorm closure or one of the `gemm_pair` projections (qkv m=2048 / z m=1024 / a m=8 / b m=8 — the i8-MMA `GemmI8MmaError::Launch` spelling), all layer-0.
3. `cuda probe post-incumbent: FAIL` — the context is poisoned for the rest of the process (every later cudarc call dies: the walk's session attach, any kernel).
4. The arm falls through to CubeCL, which produces finite logits — the failure is INVISIBLE to any caller that doesn't touch cudarc again.

## Posture facts

- League (folded Bonsai-2) at the SAME knob recipe is green (the L8 e2e; the league kb walk `plan622_keybias_walk_league_cuda` runs the same recipe plus graphs ON, all arms green 2026-10-23).
- The synth model is UNFOLDED → the layer loop takes the `else` branch at the in_proj site: plain `gemm(qkv) / gemm_pre(z) / gemm_pre(a_c, n_v, n) / gemm_pre(b_c, n_v, n)` — the folded league instead routes a/b through `rot0.kernels.gemm_dense_ab_batched` (which asserts `rows <= 48`). **The plain a/b lane at m = n_v = 8 has likely never been served by arm8 before** — every other width in both models is a 128-multiple (qkv 2048/11264, z/ffn/wo 1024, out 1024).
- The v6 q8 kernel was READ at HEAD for this issue: the OUTPUT stores are row-guarded (`row0 + r < m`) and the weight loads are row-clamped (`(blk_rows + row < m) ? blk_rows + row : m - 1`), and the q8 activation loads are p-clamped — so m=8 is SAFE on paper in `gemm_ternary_i8_mma_v6_src.rs`; the crasher is NOT yet named. Candidates that remain: `launch_prefill_quantize`'s kernel, the `rms` closure's kernel (cudarc_kernels error spelling is IDENTICAL — "kernel launch failed: {s}"), or a weight-side load path this read did not cover (the v6tb variant the fmt route picks, vs the v6 body read).

## What the walk needs to unblock

The synth walk is the kb lane's cheap iteration harness (the league walk is ~2 min/run + 102 s load; the synth walk is seconds). Until this is fixed the synth walk cannot exercise the whole-prefill CUDA arm at all — it currently FAILS at the session attach (the sticky context error), not at any kb assertion.

## Suggested next steps

1. Install compute-sanitizer on the 4090 (the v13.3 partial install lacks it; v12.1 has a bin dir — check `v12.1\bin\compute-sanitizer.exe`, else grab the toolkit component) and memcheck the walk binary — names the exact kernel + thread + address in one run.
2. Name the remaining launches: `gemm_pair` gains a weight tag (one param, ~10 call sites), the `rms` closure maps its error — the same naming treatment the GDN sub-stages got (`471f56c`).
3. Fix at the root per what (1)/(2) name; add a synth-shape regression row to the walk (it IS the regression test — un-#[ignore] it once green).
