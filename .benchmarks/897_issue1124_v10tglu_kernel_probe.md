# Bench 897 — Issue 1124 T1: the v10tglu fused gate+up SwiGLU kernel probe — 1.258×/1.227× at the league FFN shapes, regs=128 spill-free, bit-identity GATE PASS

**Status:** RECORD 2026-10-26 — kernel-level A/B, 4090 (shikuwa), release profile,
interleaved ×9 medians. GPU state at launch: 11% util, 475 MiB used, NO compute
processes (`nvidia-smi --query-compute-apps` = desktop/exempt class only).
Box: i7-13700K + RTX 4090 24 GB, AC power. Kernel-level LEVEL-0 probe — the
e2e GOAT (Issue 1124 T2) owns any promote decision.

## What ran

The Issue-1124 T1 arm: gate/up weight rows interleaved in 8-row chunks at
mirror-build time so the **v10t body runs verbatim** (single accumulator —
each m16n8 fragment's row halves become the gate/up dots of the same hid row),
with SwiGLU applied IN the transposed staging pass (`gemm_i8_mma_tm128v10tglu_q8_fused`,
`gemm_ternary_i8_mma_v6_src.rs`), canonical numerics hardcoded
(`__expf` + `div.full`, the Bench-721 forms, standalone-kernel op order).
League FFN shape m=17408, n=5120, p ∈ {2048, 4096}.

| arm | launches | p=2048 | p=4096 |
|---|---|---|---|
| incumbent: 2× v10t + swiglu_ffn_e1_d1 | 3 | 2.502 ms (1.000×) | 5.255 ms (1.000×) |
| **v10tglu (fused, `RIIR_PREFILL_FFN_GLU=1`)** | **1** | **1.989 ms (1.258×)** | **4.283 ms (1.227×)** |

Occupancy: `regs=128 local_bytes=0` — the v10t register budget held exactly;
zero spill (the Bench-895 spill rule: any local > 0 would have re-read the
launch bounds).

## Verdicts

1. **G1 bit-identity PASS** (the T1 gate):
   `v10tglu_matches_v10t_pair_plus_swiglu_bitwise` — 0 bit-diffs over two
   shapes (256/256/130 ragged; 128/384/257 third-group) against the REAL
   standalone SwiGLU kernel as reference. The interleaved-mirror
   construction is the proof: the v10t MMA/fold op sequence is untouched, so
   each hid element's gate/up dot products are the incumbent's exact
   operations; only the epilogue differs, and it hardcodes the canonical
   forms in the standalone kernel's op order.
2. **The Bench-895 refutation is NOT contradicted — it is bypassed.** Bench
   895 priced the pair kernel's B-stage share (worthless: v10t is not
   B-traffic-bound) and hit the register cliff (fusion +32 regs). This arm
   fuses nothing the incumbent needs twice: the MMA count, tile config, and
   accumulator set are v10t's; the win is the WRITE-BACK side (one slab
   instead of two + the standalone SwiGLU kernel deleted), which Bench 895
   never priced. The measured ~0.5 ms at p=2048 matches the traffic model
   (~570 MB deleted at ~1 TB/s ≈ 0.5 ms).
3. **Lane-level promote bar NOT claimed.** This is a three-launch-segment
   kernel probe, not e2e. T2 (the league pp battery ×2 windows,
   verified-idle) is armed with the env; the promote bar (Feature Flag
   Discipline) reads there.

## Notes for T2

- The armed route builds the interleaved slab alongside the pair mirrors
  (+44.6 MB VRAM at the 27B FFN shape, one-time per model). A promotion
  would stop building the gate/up pair mirrors (the fused slab replaces
  them) — measure the armed-state cost first, then halve it on promote.
- `RIIR_PREFILL_FFN_GLU=1` requires the fmt route (packed mirrors) and
  `mlp % 128 == 0`; the league posture satisfies both.
- The bench fixture weights are synthetic (the v11-test LCG family); the
  identity property is weight-independent by construction.
