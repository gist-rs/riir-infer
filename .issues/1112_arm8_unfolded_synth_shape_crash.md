# Issue 1112: arm8 whole-prefill CUDA crashes on unfolded (pre-rotation) synth shapes — sticky illegal address, poison surfaces downstream

**Status:** CLOSED 2026-10-23 — root-caused + fixed + regression GREEN (`0522312` the fix, `441b78f` the fixture, the walk un-ignored at HEAD+)
**Lease closed:** zed-m3-tokengate, 2026-10-23T04:00Z → closed same session; reshaped 2026-10-23 per the lease-reshape rule (the free-text `Claimed:` form read MALFORMED and made `claim_lease_check` BLIND for every riir-infer item).

## Symptom (as filed)

`plan622_keybias_walk_cuda` (synth Bonsai-2 GGUF: 4 layers `[DeltaNet ×3, Attention ×1]`, n_embd 1024, GDN n_k=4/n_v=8/hd=128, attention ahd 128, vocab 256) at the whole-prefill CUDA posture: probe OK at boot → the incumbent `fwd.prefill` poisons the CUDA context (`CUDA_ERROR_ILLEGAL_ADDRESS`, sticky — every later cudarc call dies) → the arm falls through to CubeCL, which produces finite logits, so the failure is invisible to any caller that doesn't touch cudarc again. With `CUDA_LAUNCH_BLOCKING=1` the error surfaced synchronously at an UNNAMED launch (the then-unnamed rms/gemm family), "all layer-0" was the filing-session's inference.

## Root cause (the naming pass named it: `gemm/a`)

**The crasher was never a kernel.** The synth Bonsai-2-style file is UNFOLDED with **DENSE a/b** (`ssm_alpha`/`ssm_beta` stored BF16 — the escape set). `upload_layer_weights` stores dense a/b as f32 handles (`in_proj_a_f32`/`in_proj_b_f32`) beside **EMPTY dummy `in_proj_a/b` TernaryHandles** (`TernaryGroupWeights::new(0, 0)` — rows 0, 0-word arrays) — the folded whole-prefill lane reads the f32 mirrors (`gemm_dense_ab_batched`) and nothing on a folded path reads the ternary handles. But the whole-prefill **plain lane** (`rotation.is_none()`) mirrored the DUMMIES (`mma_mirror` → 0-word pos/neg/scale CUDA buffers) and launched `gemm_pre(&a_c, normx, a_b, n_v=8, n)` — the v4/v1 kernel reading `pos_bits[rc*32 + s_w4]` on a **0-word allocation** → the illegal address, identically in every kernel generation (RIIR_GEMM_MMA_V2=0 → v1 faults the same), and identically at the FIRST a-projection (layer 0).

Why the league is green and this never fired before: the league model is FOLDED → the layer loop takes the rotation branch (dense escape set via `gemm_dense_ab_batched`, real f32 mirrors); every older unfolded file had TERNARY a/b (real handles). Unfolded + dense-a/b is the Bonsai-2-escape-set file class the synth walk is the first to serve.

The m-tail hypothesis (m = n_v = 8 the first sub-tile m) was **REFUTED empirically**: `gemm_v4_m_tail_sweep_issue1112` (kernel-level, in-file) runs m = 8..320 GREEN in oversized AND exact-sized buffers, both kernel generations.

## En-route posture trap (surfaced after the fix)

With the crash gone the walk still showed zero movement: `graphs_armed()` is DEFAULT-ON, the graph lane implies the gv posture, and the canonical attention arm (ahd != 256) has **no devpos twin** — the graph lane refuses it BY DESIGN (`graph-verify requires mq8 (ahd==256)`), so every prefill fell through to CubeCL (bit-identical arms, zero movement). The walk now pins its posture explicitly: `RIIR_PREFILL_CUDA_GRAPHS=0` — the EAGER whole-prefill arm; the league walk proves the graph-replay posture at ahd 256.

## The fix (`0522312`)

The plain lane gains its dense-a/b branch: the SAME batched dense GEMM the folded lane uses (`RotationKernels::gemm_dense_ab_batched` — no rotation tables involved), via a rotation-free lazily-compiled `RotationKernels` on the stack (`dense_ab_kernels: OnceLock<Option<Arc<_>>>`). The escape set stays full-precision, un-rotated, un-quantized (whitepaper A.2). `n_v > 48` refuses the whole-prefill arm (fall-through to CubeCL, bit-safe) instead of tripping the kernel assert. The pre-rotation ternary-a/b lane is unchanged; per-layer dispatch picks by each layer's own mirrors (a mixed file dispatches per layer).

Also landed: the whole-prefill launch-naming law (`63ab2d2` — every unnamed launch in `run_layers` prefixes its errors: `rms/{rows}x{dim}`, `gemm/<weight>[/quantize|/rot]`, `attn/*`, `ffn/*`, `gdn/out_residual`; a sticky error now names its stage), the clippy-1.98 `err_expect` repair the windows cross lane surfaced, the kernel-level m-tail sweep (`1b5b623`+`d53c8f3`, kept as the launch_gemm sub-tile-m shape regression), the walk's eager posture (`e90d775`) and its ±12 moved fixture + fnv bit-movement tooth (`441b78f`).

## Validation (2026-10-23, 4090)

- The synth walk `plan622_cuda_model_walk_kb_session_arms` **GREEN** un-ignored: incumbent probe OK; flag-off / zero-armed(NaN pad) / disarm / detach all bit-identical; moved deterministic ×2 at **max Δlogit 0.0831** (±12 fixture), bit-movement fnv-confirmed, all-finite; att-layer-only window selectivity bit-identical to all-layers.
- `gemm_v4_m_tail_sweep_issue1112` GREEN (m 8..320, both buffer postures, both kernel generations).
- The FA kb gate + the plan620 si gate + the league walk: regression battery at the fix commit — see the plan 622 T1.3 record (the league walk re-proves the folded path untouched).
- M3: windows-gnu cross lane clippy `-D warnings` GREEN at every step.

## Lessons

1. **A lazy-mirror over a contract-dummy handle needs the consumer to know the contract** — the "empty dummy (rows 0), nothing reads it" invariant lived in a comment two modules away from the crash site. The per-layer mirror dispatch now encodes it in code.
2. **The launch-naming pass paid for itself in one run** — `gemm/a` ended the "unnamed pre-GDN launch" search space instantly; the filing session's "all layer-0" inference was half-wrong (the crash WAS at layer 0's a-gemm, but the m-tail theory built on it was wrong).
3. **compute-sanitizer failed to attach on this box** (injection timeout, app left suspended — both processes had to be killed); the naming pass + kernel-level sweep bisected faster than fighting the injector.
