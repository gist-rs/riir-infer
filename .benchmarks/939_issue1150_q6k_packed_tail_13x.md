# Bench 939 — Issue 1150 follow-up 3: the Q6_K packed tail (the tail_logits lane) — 13× e2e on the dense Metal lane

**Status:** LANDED (2026-11-04, zed-986a7538@amo). The dense-qwen38 Metal
lane's dominant stage (tail_logits: 90% of the synchronized pass @pp512,
bench 938) is fixed at the root: the lm_head stays PACKED Q6_K (in-kernel
dequant GEMV, `RIIR_QWEN38_LMHEAD_Q6K=1` promote-only) instead of 10.7 GB of
host-dequanted f32.

## Cells (dense-f16 MM2D posture, warm compile cache, fresh processes, median)

| cell | incumbent f32 tail | Q6_K packed tail | e2e |
|---|---|---|---|
| pp512→512 | 21 tok/s (24.5 s) | **281 tok/s (1.82 s)** | **13.4×** |
| pp2484→2560 | 22 tok/s (116 s) | **298 tok/s (8.6 s)** | **13.4×** |
| pp4954→5120 | ~21 (flat per the audit) | **276 tok/s** | **~12.6×** |
| load time | 39.5 s | **20.2 s** | 2× (the 10.7 GB dequant+upload skipped) |

Stage-timer @pp512 (synchronized): tail_logits **22.8 s (90.3%) → 9 ms
(0.4%)**. League read: the audit's pp-5k-class opponents read MLX 1411 /
uzu 1157 / fork 898 tok/s — this lane was 21-22 (67× behind MLX), now **~280-300
(~4.7× behind)**. The G2 gap closes from ~67× to ~4.7×; the remaining gap is
the layer GEMMs + GDN scan, NOT the tail.

## Physics (the arithmetic that closes)

The packed payload is 2.07 GB (vocab 248,320 × nb 40 blocks × 208 B) vs the
f32 copy's 10.7 GB — 5.2× less traffic. The f32 plane GEMV ran at ~470 GB/s
effective (22.8 s / 10.7 GB); the packed kernel's whole-pass 1.82 s at pp512
implies ~1.1 TB/s-class reads on the M5 Ultra (2.07 GB in under ~2-3 ms of
the pass) — the byte-per-lane payload walk saturates bandwidth where the f32
kernel's strided plane pattern wasted half of it. 5.2× less × ~2.2× better
utilization ≈ the measured 13×.

## What landed

- `gemv_q6k_cubecl.rs` — the CubeCL/Metal Q6_K GEMV: one plane per row,
  per-lane block partials + ONE plane_sum (no per-block trees); host repack
  (`repack_q6k_payload`) splits the 210-byte GGML block into 52 payload
  words (ql/qh/sc — 208 B) + an f32 `d` sidecar (the Q4_K tile GEMM's
  pre-extracted-scale precedent; no in-kernel f16). The kernel mirrors the
  CUDA reference's structure (`gemv_q6k_cuda_raw`, `qwen38_dense_cudarc`'s
  packed `QuantW` lm_head) in the Metal lane's f32-activation class.
- Wiring: `RIIR_QWEN38_LMHEAD_Q6K=1` (exact literal, promote-only) arms
  BOTH the ingest tail and the DECODE tail (the first run's all-zero walk
  caught the decode half — `forward_token`'s own lm_head loop; fixed, walk
  byte-identical after). Off = the incumbent f32 path byte-identical (the
  dequant runs, the packed vecs stay empty).
- Unit gate `gemv_q6k_cubecl_g1` (2/2): quantized realistic rows max_rel
  1.05e-5; sign/bitfield-exhaustive arm 5.26e-7 (i8 sign-extension + all
  four qh 2-bit fields at their extremes) vs the CPU canonical `gemv_q6_k`.

## Numerics class (disclosed)

f32 activations × in-kernel-dequanted Q6_K, f32 accumulate — identical
PRODUCTS to the incumbent's dequant-then-GEMV, different summation order
(1e-5 re-association class). The e2e greedy walk is **byte-identical** to
the incumbent (64 tokens, md5 `79adf65ece6e54380fe5eb78689ae286` both arms).
Promotion bar: this is a lane-structure change (5× less resident memory +
the lane's dominant stage) — the GOAT gate reads this bench; the env stays
promote-only until the owner's default-flip call.

## Session arc (this + bench 938)

bench 938's "GDN chunked route: parity" freed the lane from a bucket misread
and pinned tail_logits as the real bottleneck (90% of the sync pass); bench
939 removes it. The lane went 21 → 281-298 tok/s in one session; next levers
by the remaining shares (sync pass, packed-tail posture): gdn layer 66%,
attn 17.9%, wte upload 15.6% — the layer GEMMs are now the ceiling.

Provenance: amo, M5 Ultra, AC power, load 2.3-2.5, GPU-exclusive at record
cells; code at the commit carrying this record (route `aa33282` + floor fix
`7d5c955` + this lane); model `~/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf`.

Session: zed-986a7538@amo, 1794465837 → 1794548520.
