# Issue 1176 — the bare-default dense-qwen38 prefill runs ~7 tok/s vs ~971 promoted: promotion-debt defect, needs a verdict or a written reason

**Status:** OPEN — observed during bench 944's baseline cells; out of
1173's lane, filed per the verdict. Not claimed (append a
`**Claimed:**` line under this Status before implementing).

## The observation

`qwen38_wgpu_bench --pp 5120` with NO posture envs (the bare default
build of `Qwen38WgpuForward::new`) runs the warm-up pass at
**736 s = 7 tok/s** (~100× off). The promoted league env set
(`RIIR_QWEN38_DENSE_MM2D=1 RIIR_QWEN38_LMHEAD_Q6K=1
RIIR_QWEN38_DENSE_MM2D_RELAXED=1 RIIR_QWEN38_DENSE_MM2D_F16ACT=1
RIIR_QWEN38_ATTN_TENSOR=1`) runs the same cell at **~970 tok/s**
(bench 942/943/944 all reproduce).

## Why this is a defect, not a curiosity

- The 1146 arc promoted every individual arm to DEFAULT-ON **inside
  the env posture** — but the envs themselves are still promote-only
  literals a bare consumer does not set. A consumer that links
  `Qwen38WgpuForward` and serves without the env incantation gets the
  ~100× route. That is promotion debt: the promotion verdicts measured
  the arms, not the default dial.
- A future bench/agent running the bin bare will misread 7 tok/s as a
  regression (bench 944 nearly did — recorded there as an observation).

## The ask (one of two, a decision + record, not necessarily code)

1. **Promote the league env set to true defaults** (the envs flip
   from promote-only to demote-only per the env-switch law), OR
2. **Write the reason it cannot be** into `qwen38_dense_cubecl.rs`'s
   boot path + this issue (e.g. envs exist to keep CUDA/other-platform
   parity lanes selectable; then make the METAL path default-arm and
   the envs demote-only there).

Any promotion move is GOAT-gated per the usual law (the arms each have
their benches; this issue is the dial, not the arms).

## Provenance

- Bench 944 (`.benchmarks/944_issue1173_fused_norm_gating_ab.md`) —
  the observation + the two cells (736 s warm-up vs 970 tok/s
  promoted, same build `5bb84f1`, same box).
- Claude verdict 2026-11-04 (round 1): "either the league env set
  gets promoted to default or the reason it can't be gets written
  down."
