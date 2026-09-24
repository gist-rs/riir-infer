# Issue 003 — CUDA flash attention: the packed-path zeros defect + the fused rung

**Status:** CLOSED (2026-09-25) — the fused kernel landed, all gates green
at the cuda posture, the published-numbers correction landed in the
consumer repo (reflex `.issues/027`); record: `HISTORY.md` §2026-09-25.

## The defect (found from the published bench, not the code)

The `.issues/002` v1 posture ran `attention_forward` through the TRAIT
DEFAULT op sequence on the CUDA backend. That default SLICES host memory
(`backend.rs`: `let qkv = &qkv[qkv_off..]`) — correct on CPU, correct on a
device backend at offset ZERO (the slice IS the parent the device op wrote,
same `(ptr, len)` cache key → hit → device-current), and SILENTLY WRONG at
non-zero offsets: the packed multi-question forward's slice is a NEW key →
`chain_buf` MISS → uploads the host bytes, which under the write-first
discipline are STALE (the parent was written device-side only; the host vec
holds its `resize(.., 0.0)` zeros). **Every multi-question case's attention
ran on zeros.**

Measured evidence (riir-reflex `.benchmarks/026_4090windows_cuda` vs
`018_4090windows_run` CPU, same box, same fixtures):

| suite | CPU 018 | CUDA 026 | |
|---|---|---|---|
| typed_decisions · english | 0.3575 | 0.2690 | corrupted |
| typed_decisions · multilingual | 0.3490 | 0.2690 | corrupted |
| typed_decisions · typed | **0.7445** | 0.2690 | corrupted (−47.5 pt) |
| code_fixtures · english (2 q/case) | 0.5417 | 0.2917 | corrupted |
| ag_news / banking77 / emotion / sst5 / xnli / massive_intent / prompt_inj (1 q/case) | = | = | byte-identical, correct |

M3 Metal (`.benchmarks/025`) matches the CPU numbers on every row — its
fused kernel BINDS offsets device-side, which is exactly why it is immune.

The consumer-side G5 gate passed green at the cuda posture because its
fixture rows are one question per case (single-sequence forwards — offset
zero, the correct path). The gate that catches this class is
`laya_batch_parity` (multi-question floor) — it was not run at the cuda
posture. T5/T6 below close that.

## The fix = the rung: the fused `flash_attn` kernel

Port the Metal lane's one-pass online-softmax flash kernel (MSL_FLASH, the
reflex Issue 020 T10 rung-3 form) to CUDA C. One dispatch per layer over
the packed qkv — split, rope, q-scale, scores, sliding window, softmax and
value mix + head merge in-kernel; the seq² scores parent never exists; the
offsets bind at dispatch (`qkv_off` / `rope_row·hd` / `out_off` are kernel
args, exactly Metal's design) — the packed path becomes the unbatched
kernel's exact math.

- Plain fp32 FMA compute (no tensor cores, no mma): attention is <6% of
  forward FLOPs at the pinned geometries; the win is the eliminated seq²
  scores traffic (~6 touches × `heads·seq²·4B` per layer: ~19 GB/forward at
  seq 1000) plus ~8 dispatches → 1 per layer, plus the windowed FLOP cut
  (~2.4× less attention FLOPs at window 64, seq ≫ window).
- 256 threads, one block per (32-row query block, head); shared staging
  tq[32][65] / tk[64][33] / tv[32][65] / ts[32][33] / tacc[32][65] +
  mrow/lrow/arow[32] = 38 016 B dynamic smem.
- Numerics: `expf` (CUDA's ≤2-ulp precise intrinsic — the `softmax_rows`
  kernel's own class); normalize by reciprocal-multiply at the drain; the
  online rescale α = expf(m_old − m_new) is exactly 1.0f when the max does
  not move. G5 at the cuda posture is the authority.
- Kill-switch `LAYA_CUDA_FLASH=0` → the reference-sequence fallback, which
  now PANICS on non-zero offsets (the Metal guard — converting the silent
  zeros into a loud breach; `supports_packed_attention` answers false there
  so the agent takes the per-question loop and offsets never reach it).
- `needs_window_mask` answers `flash_disabled || hd != 64` (mirroring
  Metal): the encoder stops building the `[seq, seq]` mask tensors on the
  fused path.

## Tasks

- [x] T1: the `flash_attn` CUDA kernel + host override (`attention_forward`
      / `supports_packed_attention` / `needs_window_mask`, `flash_disabled`).
- [x] T2: `cuda_ops_smoke` fused arms (full + sliding + the tile-edge seqs
      1/9/37/64/129 — the metal_ops_smoke mirrors).
- [x] T3: a CUDA arm in `packed_forward_equiv` (non-macOS,
      `laya-riir-cuda`-gated) — the gate class that catches the zeros
      defect at the substrate level.
- [x] T4: gates green on the 4090: cuda_ops_smoke, packed_forward_equiv,
      reflex `laya_batch_parity` at `LAYA_DEVICE=cuda` (THE multi-question
      gate), reflex G5 at the cuda posture.
- [x] T5: fixture-timing A/B (flash vs `LAYA_CUDA_FLASH=0`) — the latency
      record for the rung.
- [x] T6: re-run the corrupted bench suites (typed_decisions ×3 checkpoints
      + code_fixtures) at the cuda posture in riir-reflex; correct the
      published numbers; rider issue there.
- [x] T7: HISTORY record + this issue closed to a record.
