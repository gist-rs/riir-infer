# Bench 899 — Issue 1126 follow-up, the 4→3 rung (beta/decay + expand/L2 in ONE launch, recurrence untouched): G1 bit-identical everywhere, G2 direction-positive 3/3 interleaved pairs (~+0.6–1.0% e2e, −1.5% GPU busy) — ships opt-in `RIIR_GDN_BETA_EXPAND_FUSED=1`, default OFF

**Date:** 2026-10-26 ~23:0x–23:5x +07, shikuwa/4090, verified-idle (518 MiB desktop baseline, no compute co-tenants, no cargo contention at start; the bench's own Issue-834 fail-closed exclusivity gate passed every window).
**Trigger:** Bench 898's named follow-up rung — "fuse beta_decay + expand_and_l2_normalize ONLY (keep the recurrence untouched). Both are launch-bound elementwise kernels; the merged kernel keeps the expand kernel's 256-thread block shape… needs the paired-interleaved A/B discipline to certify."

## The change

`beta_decay_expand_fused_f32` (one nvrtc kernel + `launch_beta_decay_expand_fused`): the expand kernel's grid/block shape KEPT (ceil(3·n_v·hd/256) × 256 threads) with the n_head-wide beta/decay work riding the leading grid threads — launch-bound fused with launch-bound, zero added redundancy. Both bodies VERBATIM from their standalone kernels (disjoint write sets → no intra-kernel ordering hazard). Dispatched only in the `!core_fused` branch of `forward_deltanet_layer` behind `RIIR_GDN_BETA_EXPAND_FUSED=1`; the i1126 one-launch core takes precedence when `RIIR_GDN_CORE_FUSED=1`; `=0`/unset = the incumbent 2-launch chain byte-identically. A one-shot `[beta_expand_fused] REACH:` stderr line is the Bench-768 vacuous guard (bit-identical outputs alone cannot prove the fused kernel ran).

## G1 — bit-identity, PASS everywhere

1. **Kernel-level** (`test_beta_decay_expand_fused_bit_identical_to_chain`): beta, decay AND every expanded element `.to_bits()`-equal vs the 2-launch chain at the production geometry (16,48,128), an off-shape (4,8,64), the zero-vector-head L2-guard edge, and all three softplus ±20 branches. All 69 deltanet tests green.
2. **Real model, all 6 windows**: ARM-1 argmax-walk FNV `98412dc406c4ea19`, logits[::97] FNV `63496f6688c8f742`, the ARM-4 512-step stream FNV `dfe85338a90fd217` and all 8 per-64 block hashes — EXACT in both postures, every window.

## G2 — paired-interleaved A/B (3 pairs, one binary, env-dispatched; `B734_ROUNDS=3 B734_WARM_SECONDS=10`; no RAMP-TRIPWIRE; REACH line confirmed in both armed windows that matter)

| pair | incumbent (median tok/s / sync µs) | armed (median tok/s / sync µs) | Δ e2e | Δ sync(GPU) |
|---|---|---|---|---|
| 1 | 90.89 / 10556.0 | 92.72 / 10289.1 | **+2.01%** | −267 µs |
| 2 | 92.28 / 10514.8 | 92.64 / 10385.4 | **+0.39%** | −129 µs |
| 3 | 90.58 / 10784.1 | 91.14 / 10694.1 | **+0.62%** | −90 µs |

- **Direction 3/3 pairs on BOTH axes** (e2e and GPU-busy). Point estimate: **~+0.6–1.0% e2e median** (mean of pair deltas +1.0%, median +0.62%), **−1.5% mean GPU busy** (−162 µs/token of ~10.6 ms).
- **Honesty note:** pair 1's +2.01% overstates — the session showed ±1.3–1.7% between-window common-mode drift (windows 5–6 slowed for BOTH arms; incumbent windows themselves spread 90.58–92.28). Per-pair magnitudes sit inside that band; what certifies here is the SIGN consistency (3/3, p≈0.125 under a symmetric null) plus the mechanistic GPU-busy reduction, not a precise point estimate. Do not quote +2%.
- Why bigger than the +0.8% ceiling: the removed kernel is a grid-1×1, n_head-thread launch — a full serialization stall each token, costlier than the average in-graph node the ceiling assumed.

## Disposition

- **Ships opt-in** (`RIIR_GDN_BETA_EXPAND_FUSED=1`, default OFF — zero production change), with the record. NOT promoted: the league re-cell promotion of anything measured-positive is owner-gated (the standing handoff law); the league runner can arm it via env for the next decode-hold re-cell without a default flip. The incumbent chain stays as the `=0` demote hatch (the loser is kept, not deleted).
- **Decode-hold defense accounting:** the opponent's fusion quartet measured +3.84% 27B decode (riir-ai Bench 1004, the 0.9730× RED). This rung is the CUDA lane's first measured-positive answer (~+1% class); the remaining named levers (attention-chain fusion, embedding-dequant+inverse-FWHT) are the next units — each needs this same paired-interleaved discipline.
- The `recurrence_fused_body_impl` refactor and the i1126 core (measured-negative, Bench 898) are untouched by this rung.

## Provenance

- Model `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7,206,168,928 B, the league file), `bench_734_t5_decode` built at `cargo test -p riir-gpu --release --features bonsai2_hadamard --test bench_734_t5_decode --no-run` from riir-ai (one binary `bench_734_t5_decode-d068f1c19a594faa.exe`, executed directly 6× from the package-root cwd; riir-infer change uncommitted at run time — the A/B is env-dispatched, so the comparison is internally exact).
- Box state at start: GPU 518 MiB / 0% util, 13.2/31.8 GiB RAM, AC plugged, no cargo/rustc/llama co-tenants (tasklist + nvidia-smi checked before window 1; the test's exclusivity gate re-checked every window).
- rustfmt clean on both touched files; clippy `-D warnings` clean at `ternary_gemv_cuda_raw,ternary_gemm_batched`.
