# Issue 1146: 1-box prefill throughput gap — 6–8× behind the incumbents (T0.3 finding)

**Status:** OPEN — gap MEASURED (Bench 911); **ROOT CAUSE IDENTIFIED 2026-10-28 ~03:5x +07** (source-level, feature-gate reading + engine-default cross-check): the dist `gpu` feature row carries the decode posture but NOT the prefill-lane features — the 1141 class one list over. A/B confirmation pending (scripts/i1146_ab_session.sh; the amo window is the 641-T5 lane's right now).
**Claimed:** zed-amo-1146-prefill, 1793133500, expires 2026-10-29T20:45:00+00:00
**Date:** 2026-10-28
**Found by:** [`.benchmarks/911_plan623_t03_arms.md`](../.benchmarks/911_plan623_t03_arms.md) (Plan 623 T0.3, bmo idle, one session)
**Session:** zed-amo-i623-t24

## The finding

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
