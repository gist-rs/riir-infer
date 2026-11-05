# 950 — i1178 T1: decode frame census (bmo, qwen3.8-27B Q4_K_M, Metal)

**Status:** DONE 2026-11-06 (m3-idle-1178t1) — both census cells recorded; instrument summary-print units fixed same session; T2/T3 unclaimed (`.issues/1178`).

Issue 1178 T1: decompose the decode step the way Bench 937 did for prefill — a free-running
per-class decode ablation (`--decode-ablate-matrix`, instrument `7b738c12`) so the uzu-distilled
laws (T0, Research 378 in riir-refine) land on measured shares, not vibes.

## Posture

- HEAD `7b738c12` (worktree `riir-infer-i1178wt` on bmo, detached at the instrument commit).
- Bin `qwen38_wgpu_bench --decode-ablate-matrix`, features `qwen38_wgpu,production` — the
  league DEFAULT decode posture (issue 1176 dense-f16 matmul2d + 1170 half-B + 1171
  tensor-flash + 1150 Q6_K packed tail ~2.1 GB resident, as printed by the arms at boot).
- Model `/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf` — 64 layers (48 GDN / 16 attn),
  vocab 248320, lm_head 8 chunks × 32768 rows; ctx 24576, chunk 512; load 20.7 s / 22.4 s.
- Cells: 13 classes × 3 pairs (short) / × 2 pairs (ctx1344); 256 steps/cell; seed 10 tokens
  (short) / 1344 tokens prefill (ctx1344). Median of pair deltas per class.
- Box: **bmo** (M5 Ultra, 96 GB, macOS, AC), Exclusive lane held (GPU-exclusive), mds
  tripwire passed (mds_stores < 5% CPU after a 5-min drain gate), 82/96 GB free, load 1.5.
  Job `01a1acfb-11af` (2 h timeout), ran ~01:5x–04:0x +07 2026-11-06.

## Attempt 1 — VOIDED (the Spotlight page-out, kept as the lesson)

The first run (job `01a1acf7-5acc`) parked in `read_logits` for 75 min with 44 s CPU and RSS
46 GB → 8.7 GB: an `mds_stores` (Spotlight) indexing burst — triggered by the new worktree
checkout + bundle scp (~01:29 +07) — paged the model out mid-census. Voided per the 1167
posture-trap precedent (never record a page-out posture); job cancelled, mds drained.
Attempt 2 hardened: an mds tripwire gates the job (drain up to 5 min, refuse exit 3) and the
run's own tok/s is the residency proof — attempt 2's warm-ups (12.18 / 10.97 tok/s) reproduce
the Bench-1167 baseline (12.05) within noise; the page-out class reads ~0–1 tok/s.

## Results

### Base walls (256 steps/cell)

| cell | base wall/step | tok/s | warm-up tok/s |
|---|---|---|---|
| short (seed 10 tok) | **82.13 ms/step** (21.03 s) | 12.17 | 12.18 |
| ctx 1344 | **91.16 ms/step** (23.34 s) | 10.97 | 10.97 |

Pair spread is tight: base walls 21.014–21.099 s (0.4%) short; ffn deltas 47.487–47.523
(0.08%) across 3 pairs. Medians sound.

### Per-class costs (ms per decode step, median of pair deltas)

| class | short | ctx1344 |
|---|---|---|
| d_embed | 0.064 | 0.093 |
| d_entry_norm | 0.966 | 1.052 |
| d_gdn_gemm | 16.462 | 16.486 |
| d_gdn_stage | 0.834 | 0.883 |
| d_gdn_scan | 0.450 | 0.459 |
| d_gdn_post | 5.111 | 5.134 |
| d_attn_gemm | 6.480 | 6.551 |
| d_attn_stage | 0.271 | 0.259 |
| d_attn_scores | 1.201 | 10.182 |
| d_ffn | 47.510 | 47.590 |
| d_residual | 0.142 | 0.130 |
| d_out_norm | 0.031 | 0.032 |
| d_lm_head | 2.046 | 2.069 |
| **ALL (floor cell)** | **81.789** | **90.860** |

Additivity: Σ classes 81.57 / ALL 81.79 (99.7%) short; 90.92 / 90.86 (100.1%, pair noise)
ctx — near-additive attribution at both ctx; the ALL floor cross-checks Σ.

**Host floor:** the ALL-ablated cell runs at 0.087 s / 0.083 s per 256 steps → **0.34 / 0.32
ms/step** (logits read + host argmax + loop skeleton) = **0.4% of the wall**. The decode loop
is NOT host-sync-bound at this posture — the census's free-running loop hides host work
entirely (Σ ≈ wall), and the per-step sync crossing prices at ~0.3 ms.

### Coarse shares (recomputed per step — see the print-bug note)

| bucket | short ms/step | % | ctx1344 ms/step | % |
|---|---|---|---|---|
| ffn | 47.510 | **57.8%** | 47.590 | **52.2%** |
| gdn | 22.858 | 27.8% | 22.962 | 25.2% |
| attn | 7.952 | 9.7% | 16.992 | 18.6% |
| lm_head | 2.046 | 2.5% | 2.069 | 2.3% |
| small-stages | 1.138 | 1.4% | 1.214 | 1.3% |
| embed | 0.064 | 0.1% | 0.093 | 0.1% |

The ctx growth is ~all attention scores: 1.201 → 10.182 ms/step (+8.98 against a base +9.03);
every other class moves within noise. Per-layer instances: ffn 0.742 ms × 64, gdn_gemm
0.343 ms × 48, attn_gemm 0.405 ms × 16, attn_scores 0.075 → 0.636 ms/instance short → ctx1344.

## The premise correction

T0/T1 carried "the 116 ms/step (c5d35bb2 posture)". The census posture (current develop,
`7b738c12`) measures **82.13 ms/step** at short ctx: the league arms promoted since c5d35bb2
(issues 1150/1170/1171/1176) already bought **~29% off decode**. The distill gap must be
re-anchored: 82–91 ms/step (short→ctx1344), not 116. The uzu reference ratio re-anchors with
it (the 4.4× class shrinks to ~3.1× against the same reference); Research 378 carries the
uzu-side numbers.

## Law mapping (the T1 deliverable — where the distilled laws land)

- **ffn 57.8% / 52.2%** — the top class at BOTH ctx. The uzu GEMV/quant family lands here:
  K-i2 adjacency (lane-sliced GEMV tiles for W4/W8/W2), the gpu_01 act-quant→next-GEMV face
  (+ the B335 anti-face) — price on the FFN decode GEMV (0.742 ms × 64 instances) first.
- **gdn 27.8% / 25.2%** — the in-proj GEMV dominates the bucket (16.46 of 22.96): same GEMV
  law family; `gdn_post` 5.11 (out-proj + norm) second. Per-instance 0.343 ms.
- **attn 9.7% → 18.6%** — ctx-scaling lives in `attn_scores` (tensor-flash decode class);
  at short ctx the bucket is small, so attn levers are ctx-length-dependent wins.
- **lm_head 2.05 flat** — 8-chunk Q6_K GEMV; small, already packed-tail optimized (1150).
- **small-stages ~1.1 / embed ~0.1** — noise floor; no lever.
- **K-i1 (in-flight chaining)** — the census prices the per-step sync crossing at ~0.3
  ms/step (0.4%): there is NO hidden 90-ms sync mass at this posture. K-i1's value must be
  priced in T2 against the SERVING loop's acceptance boundary (the census loop free-runs with
  synthetic tokens; the serving A/B with G1 greedy-walk digit-exact pins is the arbiter).

## Instrument print-bug note (fixed same session)

The summary line printed the whole-cell wall as "ms/step" (`base 21026.20 ms/step (47.56
tok/s)`), computed tok/s as `1e3/base_med` (total-seconds reciprocal), and divided the
per-step class sums by the per-run total — the printed "%" shares (0.0–0.2%) and "residue
99.6%" were unit-broken. **The raw pair lines and the per-class table were always correct**
(walls in seconds over 256 steps; deltas divided by steps); this record's shares are
recomputed from the raw deltas. Fix: `base_ms` per-step + `tok/s = steps/base_med`
(the `950` note in `qwen38_wgpu_bench.rs`); compile-gated on the bmo worktree (1.3 s).

## Reproduce

```sh
# on bmo (Exclusive; mds quiet), worktree riir-infer-i1178wt at 7b738c12:
cargo build --release -p riir-infer-gpu --features qwen38_wgpu,production --bin qwen38_wgpu_bench
B=target/release/qwen38_wgpu_bench
$B --decode-ablate-matrix --decode 256                                # short cell
$B --decode-ablate-matrix --decode 256 --decode-ctx 1344 --decode-ablate-pairs 2
```

## Files

- Instrument: `7b738c12` (`crates/riir-infer-gpu/src/{bin/qwen38_wgpu_bench.rs,qwen38_ablate.rs}`).
- Print fix: this session's `fix(1178)` commit (units law comment in place).
- Raw logs (bmo): `/tmp/i1178_census_short.log`, `/tmp/i1178_census_ctx.log`.
- Claim/lease: `.issues/1178` (T1 folded; T2/T3 unclaimed — price K-i1/K-i2 on the shares
  above, then the GOAT gate + the one-row `production` act per T3).
