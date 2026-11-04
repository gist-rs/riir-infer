# Bench 942 — Issue 1171: the tensor-flash prefill attention — past the fork at its own league cell

> ⚠ 2026-11-04 (bench 945 / issue 1174 @`e59d59d`): the attribution is
> now measured free-running — the non-GEMM share is **~131 ms/chunk
> (scan + conv + expand + beta_decay)**, not the ~400 this record's
> StageTimers read; the small-kernel family measures ≤ 1-2% of the wall,
> the GEMM family owns 61.5% (mlp gateup+down alone = 41%), and launch
> reduction is closed as a lever. See bench 945's table before picking
> anything from this record.
> ⚠ 2026-11-04 (bench 944 / issue 1173 @`2647838`): this record's
> StageTimers non-GEMM share (~400 ms/chunk) is a SERIALIZED-instrument
> reading — bench 944 measured that removing ~240 launches/chunk buys
> only +0.8–1.0% e2e. The per-section walls do not predict free-running
> savings; see bench 944 + issue 1174 (the non-serialized attribution
> instrument) before picking the next lever from this table.

**Status:** LANDED (2026-11-04, zed-amo-i1171-attn@amo). The bench-941
record's named lead executed in order — census first, wiring second — and
the census was decisive: at pp5120 the attention bucket was **46.5–48.6%
of the pass** (4.45–4.51 s), per-chunk attn climbing 164 → 847 ms at
**~1.5 TFLOPS-class efficiency** (the tiled CubeCL kernel runs QK/PV as
per-lane dot chains, no matrix units). The wiring rode the EXISTING
Issue-1146 substrate (`attention_prefill_metal_tensor.rs` — the matmul2d
flash-attention, the exact handle contract, HEAD_DIM 256), extended with a
`TensorAttnHost` trait so the dense context hosts it (one first-use
tracker per pool; no module fork; the ternary lane's call sites stay
source-compatible).

## Cells (full promoted posture: DENSE_MM2D + Q6K tail + RELAXED + F16ACT + ATTN_TENSOR, median of 3)

| cell | bench 939 | bench 940 | bench 941 | **bench 942 (this)** |
|---|---:|---:|---:|---:|
| pp512→512 | 274 | 483 | 640 | **659 tok/s** |
| pp2484→2560 | 291 | 587 | 725 | **945 tok/s** |
| pp4954→5120 (league) | ~276 | 489 | 580 | **971 tok/s** |

**League row @pp-5k: ours 971 vs fork 898 — 0.92×, PAST THE FORK at its
own cell** (uzu 1157 = 1.19×, MLX 1411 = 1.45× remain). The five-record
arc: 67× → 4.7× → 2.9× → 2.4× → **1.45× behind MLX** in two sessions.

## The census (per-chunk rows, pp5120, 10 chunks) — before/after

| chunk (kv rows) | tiled CubeCL attn ms | tensor attn ms |
|---|---:|---:|
| c0 kv512 | 164–196 | 89–131 |
| c2 kv1536 | 253 | 98 |
| c5 kv3072 | 460–463 | 113 |
| c7 kv4096 | 613–616 | 122 |
| c9 kv5120 | 815–847 | **132** |

Bucket totals: attn **4.50 s (48%) → 1.10 s (20%)**; per-chunk growth
+73 ms/512-KV-rows → **+5 ms/512-KV-rows (14× flatter)**; the attention
core at kv5120 ≈ 40 ms/chunk of actual score work (~1.0 TFLOP →
~25 TFLOPS-class, vs 1.5 before). The pass is now **gdn-dominated (74%)**
— the next ceiling is the GDN bucket's flat ~400 ms/chunk (layer GEMMs at
half-B 53-66 TFLOPS + the recurrence/conv/norm non-GEMM share), NOT
attention.

## What landed

- `TensorAttnHost` trait (`attention_prefill_metal_tensor.rs`) — the host
  services (shared first-use tracker + device/queue pointers) both
  zero-copy contexts implement; `new_with_device` (the lane-side
  constructor); `dispatch` takes `&dyn TensorAttnHost` (the ternary call
  sites coerce unchanged — no fork, DRY).
- `MetalDenseF16ZeroCopyGemm`: `ensure_wgpu_initialized` pub(crate) + the
  trait impl — the qwen38 pool keeps ONE tracker across the dense GEMM
  and the attention arm (the module doc's hazard law).
- Route: `RIIR_QWEN38_ATTN_TENSOR=1` (promote-only, exact literal;
  refuses without DENSE_MM2D and off-macOS/no-feature); the prefill
  attention routes through the tensor arm when `routable` (head_dim 256,
  base_pos + p ≥ 64 — always true at p=512 chunks), else the tiled
  CubeCL arm byte-identical; the layer fns return `Result` (the dispatch
  error propagates loud — no silent fallback).
- Census instrument: per-chunk `(wte/gdn/attn/tail)` rows in the timer
  mode (the aggregate hid the O(p²) curve — now one compact line).

## Numerics class (disclosed)

The tensor arm is a TOLERANCE kernel (Q/P half, scores/O f32, K/V f32
from cache — never bit-identical; the module's own G1 is a tolerance gate
+ argmax read, re-run green this session: peaked fixture argmax
12283/12288, long-cache 23996/24000). **The lane's 512-token greedy walk
is BYTE-IDENTICAL to all three prior arms** — md5 `0764d040f09e30264603e
762ddcd7c1a`, now 7 runs across 4 arms (exact / relaxed / half-B /
half-B+tensor-attn), zero argmax flips. Promotion calls stay owner-gated
per the lossy-surface rule; GOAT reads benches 940/941/942.

Provenance: amo, M5 Ultra, AC power, GPU-exclusive at every cell (the
co-rig sibling holds CPU-compile work only); code at the commit carrying
this record; model `~/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf`; logs
`/tmp/i1171_attn_{tensor_arm,walk1}.log` + the census lines above.

Session: zed-amo-i1171-attn@amo, 1793820000 → (fold at record commit).
