# Issue 1146: 1-box prefill throughput gap — 6–8× behind the incumbents (T0.3 finding)

**Status:** OPEN — gap MEASURED (Bench 911) → census (914) → rung 1 width 512 = 1.39–1.42× (919, default flip pending the 2-box leg) → rung 2a tensor zero-copy GEMM PROMOTED into the dist `gpu` row (920, `54f0dce`; 1.21–1.34×, G1 exact after the wgpu zero-init root-cause fix `3bbf275`). Stacked: 0.13–0.16× → ~0.26–0.27× of llama.cpp. Remaining ~3.7× = rung 2b (kernel), rung 3 (embed A/B queued)
**Claimed:** zed-amo-1146-prefill, 1793133500, expires 2026-10-29T20:45:00+00:00
**Date:** 2026-10-28
**Found by:** [`.benchmarks/911_plan623_t03_arms.md`](../.benchmarks/911_plan623_t03_arms.md) (Plan 623 T0.3, bmo idle, one session)
**Session:** zed-amo-i623-t24

## The finding

## ⟲ 2026-10-28 ~16:4x +07 katgpt-rs-71 — rung 1 MEASURED (Bench 919): width 512 = 1.39–1.42× prefill, G1 PASS; default flip pending the 2-box leg

The lease above had been idle for 10 h (last commit 9d2e8a8) and no live session matched it, so this
session took the next ladder rung (config-class, no kernel overlap with a W-frame lane). The 1-box
curve on idle bmo, at Bench-914's armed posture: **w512 vs w128 = 1.39× (pp512), 1.40× (pp4096),
1.42× (pp32768)**. w256 sits between them; w1024 is flat (+1.4% at pp4096). G1: identical generated ids at
128/256/512/1024. vs llama.cpp: 0.13–0.16× → **0.21–0.23×**. The `dist_stage` GPU default stays 128 until
the 2-box TB5 cells (`scripts/i1146_width_2box.sh`) show the wider frame does not lose its
head⇄tail overlap on the wire (the reason 128 was chosen, Bench 909). Record: **[Bench 919](../.benchmarks/919_issue1146_prefill_width_recurve.md)**.

- [x] 1-box width curve 128/256/512/1024 × pp512/4096/32768 + G1 cross-width (Bench 919)
- [ ] 2-box TB5 leg 128 vs 512 (ABAB) → flip the `dist_stage` GPU default + `runner.rs` doc if it holds
- [x] Ladder 2a: arm the promoted Metal tensor (ktile64 matmul2d) zero-copy prefill GEMM in dist — PROMOTED `54f0dce` (Bench 920); root-caused + fixed the cold-pool corruption it carried (wgpu lazy zero-init, `3bbf275`, the Issue-670-T5 residual) + `FillZerosCubeCL` grid-cap chunking
- [ ] Ladder 2a confirm: dist pp32768 cell with tensor (`RIIR_DIST_METAL_TENSOR=0` A/B, one binary) in a quiet window
- [ ] Ladder 2b: the remaining kernel gap (~3.7× vs llama.cpp at w512+tensor) — the tensor GEMM's own efficiency at prefill shapes; census it under the promoted posture before choosing a rung — instrument armed `3cde142` (`I1146C_CHUNK=512 I1146C_FRAMES=4 I1146C_TENSOR=1` on `bench_1146_prefill_frame_census`); the census cell is chained behind the measurement queue on amo
  - Rung-2b static candidates (read 2026-10-28 from `gemm_ternary_metal_tensor.metal` + `gemm_ternary_metal_zero_copy.rs`; UNMEASURED — the census decides which, if any, is the floor):
    - **H1 per-GEMM submission tax**: every zero-copy GEMM does `client.flush()` + its own raw command buffer + a blit copy-back of the full P×M f32 output from a dedicated buffer (the aliasing workaround) — one extra write+read of the output per projection, ×~7 projections ×64 layers per frame.
    - H1 static bound (arithmetic, not a measurement): FFN copy-back ≈ 10.5 GB/frame at w512 ≈ 15 ms at 700 GB/s, + ~12 ms submission at ~20 µs × ~576 dispatches, vs ~1905 ms/frame (pp4096 w512+tensor dist) → **≤ ~2%: H1 DEMOTED to last**. The frame runs ~14.5 TFLOPS effective, so the gap is inside the kernel body (H2–H4); the census + a per-shape kernel cell decide among them.
    - **H2 f32 B operand**: activations enter `matmul2d` as f32 straight from device memory (A is half in threadgroup memory) — mixed-precision tiles, 2× the B bytes of an f16-staged operand (llama.cpp stages f16 in threadgroup memory).
    - **H3 redundant dequant**: each token-tile workgroup (NRB=128) re-dequantizes the same weight tile — P/128 = 4× redundancy at w512, and dequant + MMA are serialized by two barriers per K-tile (no double buffer).
    - **H4 scalar dequant**: per element: bit extract + float scale multiply + a bounds check inside the unrolled loop. → variant LANDED `cff3fda` (`RIIR_METAL_TENSOR_DEQUANT_V2=1`, default OFF; bit-exact by construction, V2=0 IR identical to HEAD modulo debug metadata); G1 `issue1146_tensor_range_g1` under the env + w512 A/B queued by the 1146 queue owner (prebuilt amo `/tmp/dist_v4`).
  - **Lead (6125f9e, UNMEASURED):** the fork's tensor `kernel_mul_mm` (vendor/prismml-llama.cpp `mul_mm.metal`) is structurally ours (A dequant → threadgroup half, B f32 from device, 64×128 tile, 4 sg) but compiles `matmul2d` with **relaxed_precision=true**; ours pinned false. Selector `RIIR_METAL_TENSOR_RELAXED=1` (default OFF = incumbent bytes); queue cell: width rig with `I1146W_ENV_B`, w512 pp512+pp4096 — a LOSSY surface: cross-arm id divergence needs a quality read, not an automatic FAIL. H2 as first phrased (f32-from-device B) is NOT a fork difference.
  - **H5 candidate (external evidence, UNMEASURED — the H3 fix shape, census-gated):** dequant A straight into a `cooperative_tensor` and feed `matmul2d` from registers — no threadgroup staging, so the two barriers per K-tile (H3's serialization) go away. Apple's WWDC26 Metal-tensors session names this as the custom-format path (dequant → cooperative tensor → `matmul2d`), and Modular's `AppleM5MatMul` docs state operands load DRAM→register because threadgroup staging degrades matmul on Apple silicon. Redundant per-token-tile dequant (4× at w512) remains, but becomes ALU on registers rather than smem traffic + barriers. Write it only if the census shows dequant/barrier time ≥ the H4 result's residual; mind the Metal fast-math fma fence (`select()`) when moving smem math into registers. Sources: developer.apple.com/videos/play/wwdc2026/330, docs.modular.com/api/mojo/linalg/matmul/gpu/apple/matmul_kernel/AppleM5MatMul/, arXiv 2607.19438 (BaseRT, M5 Neural Accelerators, 3rd-party preprint).
- [ ] Ladder 3: batched embed dequant (55 ms/frame) — CODE LANDED `3f657d0` opt-in `prefill_embed_batched` (`DequantWteRowsCubeCL`, one launch/frame; kernel G1 `issue1146_embed_batched_g1` PASS to_bits); e2e A/B queued (`I1146W_BIN_B` arm of `scripts/i1146_width_recurve.sh`) → promote into the dist `gpu` row if it gains

## ⟲ 2026-10-28 ~06:4x +07 zed-amo-1146-prefill — MEASURED CORRECTION (Bench 914): the A/B + the census

The A/B ran (both postures build, G1 CROSS-POSTURE PASS, incumbent reproduces Bench 911): the
feature-row fix ENGAGED (binary +1.7 MB of batched kernels) but delivered **1.09×** (pp4096
24.579 → 22.504 s) — **the dispatch-storm arithmetic is REFUTED as the floor mechanism**
(CubeCL/Metal submission is µs-class at these shapes, not the cudarc 50–105 µs class). The fix
stays (correct engine-default posture, G1-clean).

The census (the landed instrument, clean run v3) then attributes the 873.36 ms/frame 1-box
floor: **FFN ternary batch GEMMs 569.7 (65%) + DeltaNet ternary batch GEMMs 220.8 (25%)** —
hypothesis 1 (decode-shaped M=128 kernels) CONFIRMED with the alternative eliminated — embed
55.1, attention 80.0, the whole GDN chain ~29.7 (3%; the sequential-chain class is CLOSED as a
floor), tail ~0; coarse additivity +6% (closed). **Wide-chunk control: 4.413 vs 6.823 ms/token
at 512 vs 128 = 1.55× per-token from width alone** (Bench 909's curve re-opens under the
batched posture).

Ladder (each its own GOAT): (1) chunk-width re-curve (config-class, 1.55× already read at 512);
(2) the W-frame ternary GEMM rung (90% of the frame; Bench-895 spill cliff is the design
constraint); (3) batched embed dequant (55 ms, mechanical); (4) GDN chain CLOSED — no rung owed.
Full tables + honesty notes (v1 JIT pollution, v2 host-load pollution — both harness-corrected): **[Bench 914](../.benchmarks/914_issue1146_feature_row_ab_and_prefill_frame_census.md)**.

## ⟲ 2026-10-28 ~03:5x +07 zed-amo-1146-prefill — ROOT CAUSE: the dist gpu row omitted the prefill-lane features

Source-level confirmation chain (no GPU needed — every gate is a compile-time feature):

1. **Without `ternary_deltanet_chunked_prefill`** (not in the dist `gpu` row): the chunked-conv
   static compiles to `false` (the dual-cfg default) → `can_chunk_conv1d=false` → conv1d runs the
   **SEQUENTIAL per-token path** (p dispatches per deltanet layer), AND `chunked_enabled=false` →
   the recurrence runs the **SEQUENTIAL per-token path** (p dispatches per layer).
2. **Without `ternary_attention_batched_prefill`**: `attn_scratch` is `None` → `batched_attn=false`
   → every attention layer runs the **per-token loop** (~9 dispatches × p tokens).
3. **Without `deltanet_prework_fused`** (also not in the row): the fused conv+SiLU+L2+expand
   prework is off — with chunked also off, the sequential conv arm is the only path.
4. Dispatch arithmetic at chunk 128 (24 deltanet + 8 attention layers, Bonsai-2): sequential
   conv 3,072 + sequential recurrence 3,072 + per-token attention ~9,216 + per-token beta/expand
   arm + the batched projections/norms ~500 ≈ **~21,500 dispatches/frame** vs ~500 with the
   prefill features on. At the Bench-1007/1132 dispatch-floor class (~35 µs/launch) that is
   **~0.75 s/frame — the measured Bench-911 floor (0.767–0.914 s) is the dispatch count, to the
   digit.** The gap is NOT decode-shaped GEMM tiles (hypothesis 1 demoted to a secondary, post-fix
   question) — it is a feature-row omission.
5. **Cross-check (the 1141 precedent, one list over):** riir-ai's `riir-gpu` default feature set
   (its Cargo.toml `default =`) carries BOTH `ternary_deltanet_chunked_prefill` AND
   `ternary_attention_batched_prefill` — the league engine runs the batched/chunked bodies; the
   dist lane never did. Issue 1141's note says it in full: "riir-infer-gpu has NO defaults — the
   production posture lived only in riir-ai's riir-gpu defaults". The decode half got fixed
   (1141); the prefill half was missed.

**Fix landed** `412c887`: the dist `gpu` row gains exactly the engine-default prefill pair
(`ternary_deltanet_chunked_prefill` + `ternary_attention_batched_prefill` — NOT prework_fused /
smem_staged, which the engine default also does not carry; those stay their own GOAT lanes).
G1 is gate-covered (chunked-vs-sequential and batched-vs-sequential are bit-identity-gated per
feature; `g1_032_folded_prefill_metal` pins the COMPOSITION on the real folded file); the A/B
(scripts/i1146_ab_session.sh — two worktree builds, the T0.3 cell rig) re-proves e2e token
exactness + measures the closure; decode tg128 is the G3 row.

**The census instrument landed** `95e5068` (the issue's mandated census, retargeted): the
PREFILL_STAGE mask grew the parent-gated deltanet-chain sub-bits + standalone embed/tail bits
(3→12 bits, mask=ALL byte-identical shipping); `bench_1146_prefill_frame_census` strip-down
attributes the POST-fix residue (the frame floor that remains once the dispatch storm is gone —
where the W-frame ternary GEMM rung and the recurrence-scan question actually live now).
Hypothesis 1 (decode-shaped GEMM tiles) re-ranks AGAINST the post-fix residue, not the 6–8× gap.

Original finding text follows.

---

riir-infer's single-box prefill on Bonsai-2-27B (Metal, idle bmo, chunked width 128):

| cell | llama.cpp | MLX (pack) | riir-infer | ours/llama |
|---|---:|---:|---:|---:|
| pp512 | 1087.4 | 1058.1 | 140.2 | **0.13×** |
| pp4096 | 1046.1 | 1026.1 | 166.9 | **0.16×** |
| pp32768 | 772.0 | REFUSED | 96.1 | **0.12×** |

Decode is competitive (0.87–0.98×, same bench). Prefill is not.

## The anatomy (what the measurement brackets)

Per-frame wall at chunk width 128 through the dist pipeline (both 32-layer stages, loopback):

- pp512 (4 frames): 0.914 s/frame → 6.0 ms/token-frame
- pp4096 (32 frames): 0.767 s/frame → 5.7 ms/token-frame (warm-up amortization)
- pp32768 (256 frames): 1.331 s/frame → 10.4 ms/token-frame (KV-growth term: every-4th-layer full attention attends an ever-longer history)

llama.cpp prefills a token in 0.92–1.3 ms at the same widths. Our floor — the CHEAPEST
frame we can produce — costs ~6 ms/token. The gap is structural, not wire: loopback RTT
is ~0.1 ms against 0.77–1.33 s frames (Bench 896), and the 2-box TB5 prefill measured the
same ~137–140 t/s as loopback (Bench 909 ABBA P=256 vs this bench pp512).

**Cause hypothesis (ranked, to be confirmed by a kernel census before any fix):**

1. **Decode-shaped kernels at prefill shapes.** The chunk body runs the batched range
   prefill over the `ternary_gemm_batched` body at M=128 — a tile height sized for the
   decode path. llama.cpp dispatches M=512–32768 rows into big-M Metal MMM kernels; ours
   never leaves the simdgroup-GEMV regime that wins decode.
2. **Per-frame stage serialization + handoff.** Each 128-frame crosses head→tail with a
   handoff; within one stream's prefill the two stages' frame costs do not overlap (the
   T2.3a overlap is per-POSITION, the chunked path is per-FRAME — Bench 909: 2-box chunked
   beat 1-box chunked by only ~8% at P=256).
3. **The GDN scan at M=128** — the recurrence runs per frame; whether a wider-frame scan
   (or a chunked-scan reformulation like llama.cpp's `gated_delta_net` ssm path) pays is
   unmeasured.

## Fix directions (none landed; each needs its own GOAT)

- **W-frame prefill kernels**: an M≥512 prefill tile for the ternary GEMM batch path
  (the Bench-895 spill cliff says regs=128/local=0 is the constraint to design against).
- **Wider chunks**: the width curve (Bench 909) stopped at 128 because within-chunk
  attention is quadratic — but that curve was measured through THIS kernel stack; with
  prefill-shaped kernels the optimum may move. Re-measure, don't assume.
- **Frame overlap**: pipeline two frames (head on frame n+1 while tail drains n) if the
  scheduler permits — it is the serving shape T2.4 already built for streams.

## Acceptance

A prefill GOAT: any landed rung re-runs the T0.3 prefill cells (llama.cpp rows of Bench
911 are the stable incumbent baseline — reproduced bit-close across five weeks) and closes
a measurable share of the 6–8× gap without moving G1 token-exactness or decode t/s
(G3). Promotion follows the standard feature-flag discipline; the prefill path is the
league's pp rows too (riir-ai scoreboard), so a win here is a league lever, not just a
Plan-623 one.

## Boundary

Substrate-side: the kernels live in `riir-infer-gpu`, the range seams in
`riir-infer-dist`. No riir-ai work is required for the fix; the league re-cell after a
win is a riir-ai side effect.
