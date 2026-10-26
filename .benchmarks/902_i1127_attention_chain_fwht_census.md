# Bench 902 — Issue 1127 T1–T3 attention-chain census (nsys graph-node trace, decode scored posture)

**Status:** COMPLETE — T1 census-NEGATIVE (no rung warranted), T2 census-NEGATIVE, T3 census-POSITIVE (promoted to the active rung). Record closes T1+T2; T3 implementation opens under issue 1127.
**Date:** 2026-10-26 (amo → shikuwa-4090, ssh; zed-amo-i1127-t1)
**Vehicle:** `riir-ai/crates/riir-gpu/tests/bench_734_t5_decode.rs` (`t5_decode_baseline`, the scored decode harness) under Nsight Systems 2026.1.3 with `--cuda-graph-trace=node` — per-node kernel truth inside the captured decode graph.
**Box state:** shikuwa 4090, verified GPU-clear by the wait loop immediately before the run (the 1129 sibling's debug battery had just exited; desktop compute entries only). Windows nsys without admin → CPU sampling disabled (GPU kernel timestamps unaffected). Host wall-clock tok/s from this run are NOT quoted — nsys overhead; only GPU-side per-kernel µs are used. NOT a scored league cell (the league re-cell stays owner-gated, issue 1127 T4).

## The trace

Full battery completed inside the trace: **729 decode tokens** (ARM-1 168 = 3 rounds × (8 warmup + 48 measured); ARM-2 25; ARM-3 24; ARM-4 512 teacher-forced) — every kernel family decomposes as an exact multiple of 729, and the per-token counts pin the posture: `fwht_rotate_inverse_f32` 729 = 1/token (the folded file's inverse-embedding arm live), `attention_decode_f32_devpos` 16/token (16 attention layers), `gdn_*` chain 48/token (48 GDN layers), `swiglu_sign_fwht_quantize` 64/token, `rmsnorm_sign_fwht_quantize` 81/token = 64 FFN inputs + 16 attention inputs + 1 final norm — **n_layer = 64 = 48 GDN + 16 attention**.

## Kernel table (cuda_gpu_kern_sum, aggregated over the 729 tokens)

| kernel | instances | per token | avg µs | med µs | min µs | total ms | % of trace GPU time |
|---|---|---|---|---|---|---|---|
| gemv_ternary_dp4a_multi_persistent | 187353 | 257 | 32.94 | 27.17 | 10.5 | 6172.0 | 72.6 |
| attention_decode_f32_devpos | 11664 | 16 | 76.6 | 57.3 | 8.9 | 893.5 | 10.5 |
| rmsnorm_sign_fwht_quantize_f32 | 59049 | 81 | 5.04 | 4.90 | 4.77 | 297.4 | 3.5 |
| swiglu_sign_fwht_quantize_f32 | 46656 | 64 | 4.74 | 4.61 | 4.51 | 221.3 | 2.6 |
| gdn_input_rotabq_f32 | 34992 | 48 | 6.28 | 6.02 | 5.57 | 219.9 | 2.6 |
| gdn_out_norm_gate_permute_rotq_f32 | 34992 | 48 | 5.47 | 5.34 | 5.25 | 191.4 | 2.3 |
| recurrence_f32_fused_hd128 | 34992 | 48 | 4.79 | 4.58 | 4.29 | 167.7 | 2.0 |
| expand_and_l2_normalize_heads_f32 | 34992 | 48 | 2.44 | 2.40 | 2.37 | 85.3 | 1.0 |
| conv1d_f32 | 34992 | 48 | 1.96 | 1.89 | 1.79 | 68.5 | 0.8 |
| gate_sigmoid_sign_fwht_quantize_f32 | 11664 | 16 | 4.80 | 4.64 | 4.58 | 55.9 | 0.7 |
| beta_decay_f32 | 34992 | 48 | 1.29 | 1.25 | 1.18 | 45.2 | 0.5 |
| rmsnorm_batched_f32 | 23328 | 32 | 1.60 | 1.54 | 1.47 | 37.4 | 0.4 |
| rope_partial_f32_devpos | 11664 | 16 | 1.63 | 1.63 | 1.57 | 19.0 | 0.2 |
| split_qg_f32 | 11664 | 16 | 0.93 | 0.93 | 0.90 | 10.8 | 0.1 |
| kv_cache_append_f32_devpos | 11664 | 16 | 0.89 | 0.90 | 0.86 | 10.4 | 0.1 |
| fwht_rotate_inverse_f32 | 729 | 1 | 4.08 | 3.87 | 3.81 | 3.0 | 0.0 |
| argmax_first_f32 | 729 | 1 | 3.92 | 3.87 | 3.81 | 2.9 | 0.0 |
| dequant_wte_row_f32_devpos | 729 | 1 | 1.23 | 1.22 | 1.12 | 0.9 | 0.0 |

(`attention_decode` avg is position-inflated — 70% of traced tokens ride ARM-4's 512-step walk while the league tg-class cells sit at low positions; the min 8.9 µs is the pos≈8 cell. The position-independent families are exact at the aggregate.)

## Verdicts

**T1 — attention-chain fusion: census-NEGATIVE, closed.** The T1 chain (split_qg → q-norm → k-norm → rope → kv-cache-append) costs **0.93 + 2×1.60 + 1.63 + 0.89 ≈ 6.7 µs per attention layer per token → ≈ 107 µs/token ≈ 1.0% of the ~10.9 ms token**. Every kernel is already sub-2 µs (near the graph-node floor); the Bench-898 launch-bound-vs-compute-bound gate reads NO — the chain's total is too small for fusion to matter: a perfect 5→1 fusion recovers at most ~0.3–0.5% e2e, an order under the +2.71% decode-hold re-cross bar (91.68 → 94.16 tok/s). No rung warranted; the census IS the closure evidence (this is what "census FIRST, rung only if" was for).

**T2 — embedding-dequant + inverse-FWHT fusion: census-NEGATIVE, closed.** The pair costs `dequant_wte_row` 1.23 µs + `fwht_rotate_inverse` 4.08 µs = **5.3 µs/token (0.05%)** — nothing.

**T3 — wider/WIDER-PARALLELISM FWHT decode blocks: census-POSITIVE, the only re-cross-class lever measured.** The `sign_fwht_quantize` fused family runs at **8–95× above its traffic roofline**:

| kernel | per token | µs/launch | traffic/launch | roofline @850 GB/s | over-roof |
|---|---|---|---|---|---|
| rmsnorm_sign_fwht_quantize (n=5120) | 408 µs | 5.04 | ~45 KB | ~53 ns | ~95× |
| swiglu_sign_fwht_quantize (n=17408) | 303 µs | 4.74 | ~226 KB | ~266 ns | ~18× |
| gate_sigmoid_sign_fwht_quantize (~6144) | 77 µs | 4.80 | ~55 KB | ~65 ns | ~74× |

**Total ≈ 788 µs/token ≈ 7.2% of the token** in three kernels whose per-launch distributions are TIGHT (rmsnorm: min 4.77 / med 4.90 / avg 5.04 µs — a geometry floor, not noise; the rare 300+ µs maxes are the load/upload phase). The kernels are latency/barrier-bound at the 1024-block FWHT geometry (the per-SM serialization of the norm reduction + log2(1024) barrier stages), NOT bandwidth-bound and NOT launch-bound (they replay inside the graph where node scheduling is ~ns-class). The opponent's #396 widened decode FWHT blocks for exactly this class. Restructuring toward roofline (multi-block reduction à la the existing Issue-734-T5 `rmsnorm_quantize_mb` leaf-partial pattern, or wider per-block thread geometry, bit-identical reduction order per the Issue-940 law) has **measured headroom ≥ 3.5% e2e at 2× improvement — above the +2.71% re-cross bar**.

## Next

T3 rung opens under issue 1127 (T1/T2 rows close with this record): read the `sign_fwht_quantize` kernel sources (`cudarc_kernels/rotation*`), restructure the geometry multi-block, G1 walk/logits FNV pins EXACT (`98412dc406c4ea19` / `63496f6688c8f742` at `B734_ROUNDS=3`), paired-interleaved A/B, opt-in env, demote hatch.

## Refs

- riir-infer issue 1127 (T1–T3 + the owner-gated re-cell bundle T4) · riir-ai Bench 1004 (the RED) · Benches 898/899 (the rung precedents) · prism b10826 #395/#396/#398/#402 · riir-ai Bench 1006 (the census precedent this run mirrors, Metal side) · issue 734 T5 (`rmsnorm_quantize_mb`, the multi-block reduction pattern)
