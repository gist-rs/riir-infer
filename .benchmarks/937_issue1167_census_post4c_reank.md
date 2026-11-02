# Bench 937 — Issue 1167: the prefill frame census re-run at the promoted+attn posture — attention 37%→9%, FFN ~50%, and the fused-prework promotion gap

**Status:** RECORD — census re-run PASS (attempt 4 `01a19d1b-4481`, exit 0, bit-identity anchors held, G-C1 additivity residue −4.35 ms). Re-ranks Bench 925 post-4(c): **attention 37% → 8.9%** (the 4(c) matmul2d arm did its job), **FFN 50.4%** (now the dominant class), DeltaNet 39.0% — of which **gdn_prework 12.1%**, exposed as a PROMOTION GAP (Bench 024's R2 fused prework was promoted default-on Metal 2026-10-28 but `deltanet_prework_fused` never joined `riir-infer-gpu/production`; serving runs the chunked fallback). Paired A/B cell queued (attempt 5 `01a19d26-c556`: census + the feature) — see §Fused-prework cell below for the verdict when it lands.
**Date:** 2026-11-02
**Session:** zed-cadbba9f@amo, 1793635248
**Issue:** [`.issues/1167_metal_prefill_followup_lane_b_c.md`](../.issues/1167_metal_prefill_followup_lane_b_c.md)
**Box:** bmo (M5 Ultra, 96 GiB), AC, Exclusive lane (GPU-quiet by construction — the rig held the box alone)
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` (league pin, absolute `BONSAI2_G1_MODEL`)
**Pin:** worktree `riir-infer@6941f7788fca` (develop at queue time; probe-confirmed present in bmo's clone — its remote-tracking refs are stale at `efeb527` 10-29, objects are not)

## Posture

Bench-925's PROMOTED posture + the 4(c) attention arm: `I1146C_TENSOR=1` (tensor zero-copy GEMM), `RIIR_METAL_TENSOR_RELAXED=1` (relaxed matmul2d, the promoted dist posture), H4 dequant at its default, `RIIR_PREFILL_ATTN_METAL_TENSOR=1` (the 4(c) arm — dist promotes it programmatically default-on; the gpu-level census must arm it explicitly), `I1146C_CHUNK=512 I1146C_FRAMES=8` (pp4096, matching 925's frame shape for Δ-comparability). Features = the `[[test]]` row + `metal_tensor_gemm`. 13 postures × (2 warmup + 5 timed) passes, 418.99 s.

⚠ Posture-validity note: the feature set was checked against `riir-infer-gpu/production` row-by-row — every production rung is in, nothing extra touches the frame path (`multi_host` + `bonsai2_hadamard` are the census's own requirements). The census therefore measured the PRODUCTION prework path — which is exactly what exposed the gap below.

## The census (ms/frame, 8×512, tensor-zerocopy)

| posture | wall | Δ (class cost) | share |
|---|---:|---:|---:|
| -ffn | 278.07 | 286.96 | **50.8%** |
| -deltanet | 342.70 | 222.33 | **39.3%** |
| -attention | 514.41 | 50.62 | **8.9%** |
| -gdn_prework | 496.31 | 68.72 | **12.2%** |
| -gdn_rec | 532.30 | 32.73 | 5.8% |
| -gdn_post | 555.20 | 9.83 | 1.7% |
| -embed | 555.43 | 9.60 | 1.7% |
| -gdn_beta | 563.06 | 1.97 | 0.3% |
| -gdn_norm | 564.20 | 0.83 | 0.1% |
| -gdn_residual | 564.27 | 0.76 | 0.1% |
| -gdn_rotin | 564.79 | 0.24 | <0.1% |
| -tail | 565.17 | −0.14 | ~0 |
| ALL | 565.03 | baseline | — |

G-C1 coarse additivity: 569.38 of 565.03 attributed; residue −4.35 (noise-class). Wide-chunk control: 4×2048 = 1.049 ms/token vs 1.104 at 512 (−5.0%) — the pp512 fixed-cost slice is real and bounded ≈ the small stages (norm/rotin/beta/post/residual/embed ≈ 33 ms ≈ 5.8%).

GEMM-class total: FFN 286.96 + GDN GEMMs ≈107.3 (deltanet Δ minus its measured sub-bits) ≈ **394 ms ≈ 69% of the frame** — but Bench 936 (the 1166 second-generation tile sweep) answered NEGATIVE: the promoted ktile64_64x128x4sg geometry is the composed optimum. The GEMM share is big because the GEMMs are the work, not because they are slow for their shape.

## Re-rank vs Bench 925 (pre-4c: attention 37% / FFN 34% / DN 29%)

1. **Attention CLOSED as a top lever** — 8.9% share; even a 2× attention win moves e2e ~4.5%, and the matmul2d arm already took the 2×.
2. **FFN ~51% is the residual's home, and geometry is exonerated** (936 NEGATIVE) — the in-GEMM levers move to 1167's lane: the packed-byte dequant load (2 bit-planes → 1 packed stream, the last fork structural delta, "minor" per 1146 but never measured at this posture) and f16-staged B (lossy class, needs the KL instrument).
3. **gdn_prework 12.2% is the largest NON-GEMM class — and it is a PROMOTION GAP, not a kernel to write.** Bench 024's R2 fused prework (conv1d+SiLU+L2-norm+expansion in ONE dispatch; 7.3–9.1× stage-isolated; dispatch-overhead-bound shipping chain — 69 kernels/layer at P=2048) was PROMOTED default-on Metal/wgpu 2026-10-28 (m5ultra re-cell 1.043×/1.045× e2e, delegated owner verdict `c928c408`). The landed shape flipped the runtime env default in this repo (`e2edf61`) and joined the feature in **riir-ai's** `riir-gpu` default set — but `deltanet_prework_fused` never joined **riir-infer-gpu's `production` row**, so the serving forward (dist `gpu` → this crate) compiles the fused path OUT and runs the chunked fallback the promotion was measured against. The runtime "demote-only" switch is a no-op without the feature. This is the exact omitted-promoted-rung class the production row's own comment declares "structurally dead" (the 1141 family, third+ fire).
4. **pp512 fixed-cost slice quantified**: ~33 ms/frame of small stages ≈ the 5.0% wide-chunk per-token gain — a config-class rung (frame overlap / chunk-width policy), bounded, below kernel work in priority.
5. **Rung C (H5 cooperative-tensor) re-scoped**: the ATTENTION variant is dead (9% share). The GEMM-staging variant (dequant→register, kill the per-K-tile barriers) stays unpriced — the census cannot separate barrier/staging cost inside the GEMM strips; it needs its own cell before any kernel is written.

## Fused-prework cell (attempt 5, `01a19d26-c556`) — **VERDICT: PASS, joined**

Census re-run with `deltanet_prework_fused` added (the promotion's own posture: runtime default ON on macOS, `p=512 ≤ 65535`, production dims supported per Bench 024 G1). 374.66 s, exit 0, the census's bit-identity anchors HELD (ALL×3 logits bit-exact — the kernel is bit-identical by construction with in-module pins). **ALL wall 565.03 → 500.86 ms/frame (−11.4% e2e)**; the isolation is clean:

| strip | chunked (937) | fused (this cell) | verdict |
|---|---:|---:|---|
| -gdn_prework Δ | 68.72 | **4.62** | **14.9×** on the strip (the stage A/B's 9.13× @P=4096, better at 512 — dispatch-overhead-bound) |
| -attention Δ | 50.62 | 51.48 | unchanged (noise) — clean isolation |
| -ffn Δ | 286.96 | 281.16 | ~noise — clean isolation |
| -deltanet coarse Δ | 222.33 | 158.44 | −28.7% (prework rides inside) |
| wide-chunk ms/token | 1.104 | **0.978** | −11.4%, consistent |

G-C1 additivity residue 1.03. Side observation: `gdn_out_proj` submission wall 1841.75 → 216.72 ms/frame-total — the chunked chain's queue backpressure, gone (the instrument's own warning that submission walls are not exec time, demonstrated).

**Landed:** `deltanet_prework_fused` joined `riir-infer-gpu/production` (the row + the `PRODUCTION_RUNGS` data mirror, one commit; the row stays comment-free — the manifest parser bracket-slices it verbatim and an in-row comment parses as rungs, caught by the both-directions test on first try). The dist posture floor (`gpu_production_posture_floor`) passes; the fused module's bit-identity tests pass at the joined posture; clippy clean on the dist gpu posture (2 pre-existing dead-code warnings in `regime_gate.rs` from the i1163 lane, untouched here).

## Bmo rig-job lessons (2026-11-02, three, all in `.issues/1167`)

1. Non-interactive shells lack `/opt/homebrew/bin` (cmake) → export PATH first (1166 owner, `8edbc616`).
2. `rig run --worktree` sandboxes the repo where `../katgpt-rs` does not resolve (`01a19d04-cbee`) → manual worktree under `/Users/katopz/git/`.
3. Rig jobs carry no SSH keys — `git fetch` from a job dies publickey (`01a19d18-1106`) → pre-create/pin; probe (`01a19d1a-3c02`) before trusting the target clone's refs (its tracking refs were stale at 10-29 while objects were current).

## Boundary

Substrate-side: `crates/riir-infer-gpu` (census instrument + the production row join). No riir-ai work required; the league pp rows re-price via riir-ai Issue 1154 when the promotion gap closes.
