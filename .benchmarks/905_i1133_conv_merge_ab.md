# Bench 905 — Issue 1133 conv+beta/decay+expand merge rung: G1 bit-identical everywhere (kernel + real model, all windows), G2 NULL — 2/4 pairs positive (max +0.48%), pooled ≈ 0% ± 0.5% — ships opt-in `RIIR_GDN_CONV_MERGE=1`, default OFF

**Date:** 2026-10-27 ~01:0x–02:2x +07, shikuwa/4090, verified-idle (490 MiB desktop baseline, no cargo/rustc/llama co-tenants; the bench's own Issue-834 exclusivity gate ran every window).
**Trigger:** the Issue-1133 rung — the last unmerged launch-bound boundary in the armed GDN decode chain (`conv1d → beta_decay_expand_fused → recurrence`): merge the first two into ONE launch, the Bench-899 node-arithmetic (48 layers × ~1.3 µs node dispatch ≈ 62 µs/token ≈ 0.57% e2e predicted).

## The change

`conv_beta_decay_expand_fused_f32` (one nvrtc kernel + `launch_conv_beta_decay_expand_fused`): grid = the conv kernel's own grid (`ceil(conv_dim/256)` × 256 threads); phase 1 per-channel-private (unshifted-window conv — the ascending-k product sequence provably identical to the incumbent's shift-then-dot — in-place qkv write, state shift+append, smem stage, 1:1 V copy, beta/decay on the leading threads), one `__syncthreads()`, phase 3 = the Q/K expand partitioned by SOURCE block (each out-head's source head lives in exactly one block; every `expanded` element has exactly one writer). Dispatched from `forward_deltanet_layer` behind `RIIR_GDN_CONV_MERGE=1` (posture gate: hd ∈ {64,128,256}, (nk·hd)%256==0, (nv·hd)%256==0, nv%nk==0 — misaligned geometries fall back to the chain, the Issue-1124 precedent); precedence `core_fused` > `conv_merge` > `beta_expand` > incumbent; `=0`/unset byte-identical (the demote hatch); one-shot `[conv_merge] REACH:` stderr line (the Bench-768 guard), confirmed in EVERY armed window.

## G1 — bit-identity, PASS everywhere

1. **Kernel-level** (`test_conv_beta_decay_expand_fused_bit_identical_to_chain`): the in-place qkv buffer, the conv_state sliding window, beta, decay AND every expanded element `.to_bits()`-equal vs the 2-launch chain (`launch_conv1d` → `launch_beta_decay_expand_fused`) at the production geometry (16,48,128,ks4), a ks=8 off-shape (4,8,64), the zero-vector-head L2 guard, and all three softplus ±20 branches.
2. **Real model, every window**: ARM-1 argmax-walk FNV `98412dc406c4ea19`, logits[::97] FNV `63496f6688c8f742`, the ARM-4 512-step stream FNV `dfe85338a90fd217` and all 8 per-64 block hashes — EXACT in both postures, every scored window.

## G2 — paired-interleaved A/B (4 pairs, one binary `bench_734_t5_decode-d068f1c19a594faa.exe`, env-dispatched, `B734_ROUNDS=3`, warm 20 s; REACH confirmed in every armed window; NO G1 drift anywhere)

| pair | incumbent (median tok/s / sync µs) | armed (median tok/s / sync µs) | Δ e2e | Δ GPU busy |
|---|---|---|---|---|
| 1 | 91.84 / 10488.2 | 91.97 / 10490.7 | **+0.14%** | +2.5 µs |
| 2 | 91.60 / 10666.0 | 92.04 / 10515.9 | **+0.48%** | −150 µs |
| 3* | 92.11 / 10550.0 (warm 20) | 92.30 / 10637.2 (warm 30) | +0.21%* | +87 µs* |
| 4 | 91.78 / 10485.8 | 90.99 / 10764.7 | **−0.86%** | +279 µs |

\* pair 3's first armed attempt RAMP-TRIPWIRED (1.53% spread) and was re-run at warm 30 s — the paired windows then sit at mixed warm postures, disclosed rather than pooled silently; pair 4 re-ran BOTH windows at warm 30 s (the table row).

- **Verdict: NULL — sign consistency 2/4 FAILS the 899 bar (3/3); pooled point estimate ≈ 0% ± 0.5%, inside the ±1.3–1.7% between-window common-mode.** NOT the 898-class hard negative (−3.5%) — the kernel is exec-clean enough to wash out — but no certified gain, and pair 4's −0.86% with +279 µs GPU busy shows the loss arm is real.
- **Why the predicted +0.57% did not materialize:** the dispatch saving (~62 µs/token) is REAL but is offset by an occupancy loss the conv→expand dependency forces: the Q/K expand must stage through the SOURCE block's smem (cross-block alternatives are grid-barrier territory — the 898 refusal class), which concentrates the expand work into the 16 source blocks (4096 threads) vs the incumbent expand kernel's 72-block/18432-thread shape. The merged kernel's phase-3 serial depth (~768 iterations/thread) is ~6× the incumbent expand's (~128) at 4× lower thread count.

## Dev-iteration record (implementation lessons, kept so the shape is not re-tried blind)

1. **The element-walk serialization** — the first draft dealt phase-3 elements with a `remaining` counter walk: at any walk position exactly ONE thread per block did the 128-iteration sq_sum while 255 decremented; measured **−25% e2e** (68.75 vs 91.81 tok/s). The static decomposition (thread t takes e = t, t+256, …; exact arithmetic over the owned-head ranges) recovered to ~parity. Lesson: a work-list WALK on a GPU kernel is a single-worker shape unless the walk itself is the parallel axis.
2. **Negative-count clamp** — blocks past a section's end produce negative owned-head counts; `total` must clamp them (else a block with real section-1 work can compute a negative total and skip it — caught by reading, not by a test, because the league geometry never straddles).

## Disposition

- **Ships opt-in** (`RIIR_GDN_CONV_MERGE=1`, default OFF — zero production change), with the record and the kernel-level test. NOT promoted; the T4 re-cell staging list does NOT gain a row on this evidence.
- **Named re-open trigger** (the only variant not measured): a 512-thread-block posture halves phase-3 serial depth (~3 elements/thread) under the same partition — worth one A/B cycle ONLY if a future ladder audit finds the launch-class budget still short after T3/T6 land (the expected case: T3 ≥3.5% + T6 ~2.6% make this ~0% rung irrelevant).
- The incumbent chain stays as the `=0` demote hatch; the 899 arm is untouched; the rotation-file lane (T3 lease / T6) is untouched.

## Provenance

- Model `Ternary-Bonsai-2-27B-PQ2_0.gguf` (the league file, resolved via the bench default path), built `cargo test -p riir-gpu --release --features bonsai2_hadamard --test bench_734_t5_decode --no-run` from riir-ai at the uncommitted riir-infer working tree (the A/B is env-dispatched — internally exact); executed directly from the package-root cwd.
- Box state: GPU 490 MiB / ~5% desktop, 16-core i7-13700K, AC plugged; no compute co-tenants at any window (tasklist + nvidia-smi before each; the harness exclusivity gate armed every run). One rustc STATUS_ILLEGAL_INSTRUCTION crash on the FIRST release build, clean on retry (transient; disclosed).
- rustfmt clean on both touched files; clippy `-D warnings` clean on both touched files at `ternary_gemv_cuda_raw,ternary_gemm_batched` (the one remaining −D error in the tree is the T3 lane's in-flight bench903 probe file, not this rung's).
