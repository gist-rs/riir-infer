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
