# Issue 1150 session record — the dense-qwen3.8 Metal lane: what landed, what it measured

**Status:** RECORD — decode + parity G1 LANDED; prefill engine = the named follow-up (2026-10-28)

## Landed (riir-infer, pushed: 465c1d8 → 538a1f2 → 8a17641 → 22d28e3 → 0099e27 → 63c8847)

- **`qwen38_dense_cubecl`** — the full dense-qwen3.8 (qwen35 arch, 48 GDN / 16 attn,
  vocab 248320) forward on CubeCL/wgpu: self-contained config parse (CUDA lane
  untouched), QuantWeight dispatch (Q4_K packed Q4KHandle in-kernel dequant;
  Q6_K host-dequant f32 — the standard Q4_K_M recipe mixes both), host-side
  embeddings + 32768-row-chunked f32 lm_head (Metal per-buffer limits forbid
  5 GB single buffers — measured, twice), full GDN+attention decode, chunked
  prefill (sliced 64-token conv + ordered carry update, multi-token recurrence,
  KV fill, tiled flash), MLP on all four paths.
- **`gemm_q4k_tile_cubecl`** — the Q4_K×f32 tiled GEMM + f32 twin + gather.
  **G1 unit gate GREEN on M3 + bmo** (edge shapes + n=5120, rel < 2e-4 vs the
  dequantized-weight CPU reference).
- **`qwen38_wgpu_bench`** — pp/decode cells + the greedy walk (the parity artifact).

## The G1 that matters — greedy parity vs llama.cpp (bmo, same GGUF, same box)

| engine | raw completion (temp 0, 24 tokens, prompt "The harbor town woke slowly under a grey sky,") |
|---|---|
| **ours (Metal, f32-act)** | `" the kind that promised rain but delivered only a damp chill. The air smelled of salt, diesel, and the faint sweetness of overripe peaches from the market"` |
| **llama.cpp fork 73bfe6cc4** (W4A8 int8-act, /completion raw) | `" the kind that promised rain but delivered only a damp chill. At the edge of the quay, where the water lapped"` |

**12/12 first tokens EXACT** across an entirely different kernel stack — the
module's pre-registered numerics divergence (f32 activations vs the cudarc/
llama int8-activation path) only shows at greedy near-ties from token 13.
The forward is correct end-to-end (boot → 64-layer GDN-hybrid → logits).

## The wall — measured, honest

- Prefill pp2484 = **4 tok/s** (chunk 512): ~216 GFLOP/s aggregate.
- The A/B at the production ffn shape ([17408×5120]×512, bmo release):
  **q4k-tile 614 / matmul-f32 570 / matmul-f16b 547 tok/s-equiv — all ≈110
  GFLOP/s per GEMM.** Every in-crate CubeCL tiled GEMM (mine + the two
  battle-tested ones) sits at the same scalar-kernel wall — the Bonsai
  league's 0.249× Metal prefill row is THIS wall, now measured on dense too.
- Conclusion: a competitive Metal dense prefill needs the **matmul2d (MPP
  cooperative-vector) route** — the ternary zero-copy MSL lane
  (`gemm_ternary_metal_zero_copy`, Issue 656/663/670/1146) is the template;
  the dense-f16 matmul2d variant (weights f16 at load, raw-Metal dispatch on
  the shared queue) is the named follow-up. MLX's 1411 tok/s @10k qwen3.8 row
  is the bar that route must chase.

## Also landed en route

- Q4_K_M files mix Q4_K + Q6_K layer tensors (QuantWeight, the cudarc
  `QuantW{q4}` pattern); vocab reads from token_embd ne-product (shape[0] is
  n_embd — the truncation bug produced a collapse-loop then a clean fix).
- bmo hygiene: llama-server + bench processes killed; `/tmp/i1150` (cargo
  target + llama build) kept for the follow-up lane; cmake via
  `~/Library/Python/3.9/bin` (pip wheel) recorded.

## Follow-up (the next lane)

1. dense-f16 matmul2d MSL kernel + zero-copy dispatch (the speed unlock);
2. decode on the Q4K GEMV path is already fine (bandwidth-bound) — verify tok/s;
3. re-run the bmo four-way pp cells once 1 lands; MLX/fork/uzu rows already recorded.

## Follow-up LANDED — the dense-f16 matmul2d arm (2026-10-29, zed-m3-i1150-densef16)

Step 1 + the f16 load posture, claimed under this issue (lease zed-m3-i1150-densef16):

- **`gemm_dense_f16_metal_tensor.metal` + `gemm_dense_f16_metal_zero_copy.rs`** — the dense
  twin of the ternary zero-copy lane: the SAME matmul2d cooperative path
  (64×128 tiles, 4 simdgroups, K-tile 32) with the A operand as a strided
  **device f16 tensor view** — no bit-plane dequant, no threadgroup staging,
  no staging barrier in the K loop. Plus **`gemv_dense_f16`** (plain kernel,
  one simdgroup per row, coalesced f16 reads, one simd_sum) for decode.
  Shared-device zero-copy choreography mirrored from Issue 663 T5 (flush →
  ensure_wgpu_initialized → fresh raw extraction → dedicated output + blit
  → commit; `covered`/`insert_merged` reused via a `pub(crate)` bump).
- **`QuantWeight::F16`** + the `RIIR_QWEN38_DENSE_MM2D=1` load posture (opt-in
  env, loud-refusing where Metal/the feature is absent; env-off = the packed
  incumbent byte-path): host-dequant once at load → f16 bit patterns → the
  gemm sites route through matmul2d, decode gemv through the plain kernel.
  The 16-bpw memory cost is the posture's disclosed trade (27B ≈ 54 GB — M5
  Ultra class; the M3 cannot hold it, so the unit gates + A/B here are
  kernel-level and the four-way cells are the bmo follow-up).
- **G1 unit gate GREEN (M3)** — `tests/qwen38_dense_f16_matmul2d_g1.rs`:
  matmul2d exact/edge classes (m-edge, p-edge, k-edge, combined ragged,
  p=1, the n=5120 production count) + gemv (m-edge, production shape), all
  vs an f16-rounded CPU reference, rel-to-mag ≤ 1.6e-7 (the band is
  relative to Σ|w·x| — a pure relative band on near-zero dots measures
  summation-order noise, not the kernel).
- **⛔→✓ The deferred-upload race (the session's hard-won finding):** the
  first gate runs measured ALL-ZERO writes across shape classes that are
  provably correct under sync — three fake "shape laws" (m-ragged skip,
  p-floor skip, combined-ragged skip) were minted and then retracted.
  Root cause: `create_from_slice` uploads are DEFERRED writes and a raw
  Metal commit can land before they reach the pool (the Issue-1146
  attention gate's documented hazard; flush orders kernel producers, NOT
  host uploads — the cure is `client.sync()` after uploads). The gate now
  pins the full matrix under sync so any real cooperative-tensor regression
  is loud. Lesson recorded: **a guard minted from a racy negative is a
  constraint on the world that the world never imposed** — re-measure
  negatives under the settled fixture before encoding them as laws.
- **A/B @ [17408×5120]×512 (91.3 GFLOP), release, N=10 back-to-back,
  ABBA×2 median, M3 Max quiet AC:** q4-tile **528 ms (173 GFLOP/s)** vs
  dense-f16 mm2d **10.5 ms (8678 GFLOP/s)** — **50.22× kernel-level**, with
  the loud-zero insurance (wait_last inside the clock + a post-timing
  output readback vs the CPU dot, rel 5.8e-6). 8.7 TFLOP/s is the M3 Max
  fp16 tensor rate — compute-bound, physically coherent. The session
  record's ~110 GFLOP/s wall is measured GONE on the dense path.
- **The wall this unlocks:** the prefill wall (~110 GFLOP/s aggregate, the
  Bonsai league's 0.249× Metal row shares it) is the scalar-dequant GEMM
  class; the matmul2d route replaces it on the dense lane. The e2e prefill
  cells (pp 2484/4954/9856/16392 + decode ctx2048, the four-way same-model
  table vs fork/MLX/uzu) are the next lane — bmo-only (54 GB), co-tenant
  with the mammoth capture (disclosed) or queued behind it.
