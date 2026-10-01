# Bench 024 — Issue 1004 R2: the fused GDN prework (conv1d + SiLU + q/k L2-norm + head expansion, ONE dispatch)

**Status:** RECORD (R2 BUILT, opt-in `deltanet_prework_fused`; e2e verdict: real but small — four positive medians across two runs, ALL inside the round spread → NOT promoted, opt-in stands; the structural case is the dispatch-count elimination, the league bout owns any promotion)

**Date:** 2026-10-01 · **Box:** M3 Max (AC, powermode 2) · **Load class:** LOADED (loadavg 28–34 — the riir-refine sibling's all-features test suite at ~594% CPU throughout; ratios are interleave-paired so drift lands on both arms, but ABSOLUTE tok/s are load-deflated)

## What was built

`crates/riir-infer-gpu/src/deltanet_prework_fused_cubecl.rs` — `DeltanetPreworkFusedCubeCL` +
kernel `deltanet_prework_fused_f32` (feature `deltanet_prework_fused`, opt-in; runtime toggle
`set_prefill_prework_fused`, default ON when compiled; launch counter `prework_fused_launches`).

The shipping per-layer prefill chain after the qkv projection:

1. `DeltanetChunkedConv1dCubeCL` × P/64 chunks (each = conv + ordered carry-update dispatch),
   writing the intermediate `qkv_conv_b`, then a handle swap;
2. `DeltanetBetaDecayBatchedCubeCL` × 1 (the a/b stream — UNCHANGED, stays separate);
3. `ExpandAndL2NormalizeHeadsBatchedCubeCL` × ceil(P/456) (the Metal grid guard).

→ **~69 dispatches/layer @P=2048.** The fused kernel replaces stages 1+3 with ONE dispatch
(plus the same ordered carry-update, `c = p`) reading the RAW qkv stream once and writing
`qkvx_b` directly: **2 dispatches/layer** (beta/decay is the third). The intermediate buffer
never exists (~160 MB/layer @2048 at Bonsai-2 dims: 80 write + 80 read of `qkv_conv_b`).

Shape: one workgroup per (token, head-slot) — X = 2·n_k + n_v (80 at Bonsai-2: n_k 16, n_v 48,
hd 128, ks 4), Y = p (≤ 65535); 128 threads = head_dim channels; 512 B threadgroup staging of
the silu'd head between the conv and normalize phases. No `p % 64 == 0` constraint (any p —
the old chunked path fell back to P SEQUENTIAL dispatches otherwise).

## G1 — bit-identity (by construction, pinned)

Per output element the expression sequence is the shipping kernels' verbatim: the conv sum
loops `k` ascending over the same window (carry for positions < 0 — the same conv_state
values the per-chunk carry holds), the same SiLU expression, the same ascending `sq_sum` loop
(the expand kernel's exact order) over the same silu'd values (threadgroup round-trip
preserves bits), the same zero-norm guard and multiply, and the inverse of the
`out_head % n_k` broadcast (compact head h writes out-heads h, h+n_k, … — each element once).

- `fused_is_bit_identical_to_the_chain` (in-module): expanded outputs AND carried conv state
  bit-equal across 6 shapes — production dims (16,48,128,ks4) at p ∈ {192, 1, 2}, plus
  (2,4,128) p=200, (4,8,128) p=63, (1,3,128) p=65 (odd, non-multiple-of-64, p<ks−1).
- **e2e logits FNV (the production folded Bonsai-2 PQ2_0 file):**
  - @2048 `0ec2396fd4627f29` (argmax 248046) — byte-for-byte R1's recorded pin.
  - @4096 `f35280cb95f5d306` (argmax 99125) — byte-for-byte R1's recorded pin (and 032's
    folded G1 at 4096).

## G2 — stage-isolated A/B (`tests/bench_1004_r2_prework_fused_ab.rs`)

Prework-only, production dims, 11 interleaved pairs, order alternating, bit-checked every run:

| P | shipping median | fused median | ratio median | range | speedup |
|---:|---:|---:|---:|---|---:|
| 2048 | 11.00 ms | 1.48 ms | 0.137 | 0.125–0.200 | **7.29×** |
| 4096 | 20.40 ms | 2.31 ms | 0.110 | 0.105–0.133 | **9.13×** |

The issue's traffic-only estimate (~0.5% e2e) undercounted the real term: at these sizes the
shipping chain is **dispatch-overhead-bound** (69 kernels × ~150 µs submission/gap ≈ 10 ms of
the 11 ms wall; the traffic is ~0.8 ms of it), not traffic-bound. The fused kernel's 1.48 ms
@2048 ≈ 227 MB moved at ~154 GB/s — a plausible single-kernel rate.

## G2 — e2e A/B (`riir-ai crates/riir-gpu/tests/bench_1004_r2_prework_fused_e2e.rs`)

Production `prefill` over the folded Bonsai-2 PQ2_0 file, one process, interleaved rounds,
order alternating, FNV bit-identity asserted EVERY round. TWO runs (the second the quiet-box
confirmation cell the R1 owed-cell precedent demands):

| Run | Load (1-min) | Rounds | P=2048 | P=4096 |
|---|---:|---:|---|---|
| A (loaded) | 28.8 | 5 | 79.7 → 85.4 tok/s, **1.045×** (1.025–1.080, every round > 1) | 70.5 → 70.8, 1.018× (1.003–1.030) |
| B (quiet-ish) | 5.5 | 8 | 57.6 → 60.8 tok/s, **1.022×** (0.974–1.123) | 59.2 → 61.1, **1.034×** (0.844–1.238) |

Bit-identity: FNV `0ec2396fd4627f29` @2048 / `f35280cb95f5d306` @4096 in BOTH runs —
byte-for-byte R1's recorded pins.

Read honestly: (a) the stage-isolated 7–9× collapses to 2–4% e2e because the prework is a
small slice of the layer (the GEMM wall dominates — the issue's own projection). (b) All
four medians are positive, but run B's spreads cross 1.0 (min 0.974 / 0.844) — by this
repo's own R1 standard ("inside the ±10% round spread = NOT promoted"), the e2e verdict is
**real but small; opt-in stands**. (c) The run-to-run ABSOLUTE flip (79.7 tok/s loaded vs
57.6 quiet @2048 — the LOADED run faster) is the same box-state-dominates-absolutes class
R1 documented across its day (60–103 tok/s @2048); only within-run interleaved ratios are
comparable. (d) The load asymmetry makes mechanistic sense: dispatch elimination pays MORE
under CPU contention (run A's 1.045× vs run B's 1.022× @2048), and can never invert the
sign — the fused arm does strictly less work (fewer dispatches, less traffic) at
bit-identical output.

## G3 — no decode regression

By construction: the decode path's `DeltanetConv1dCubeCL` (single-token, in-place) is
untouched; the fused kernel is wired only into `prefill_tokens_chunk`'s prework branch.

## Posture

- Feature `deltanet_prework_fused` **opt-in** (both repos), toggle default ON when compiled —
  the R1 posture. The quiet-box confirmation cell is DISCHARGED (run B above); by the R1
  "inside the round spread" standard the e2e gain does not promote on its own — promotion
  rides the league loop's bout (never a solo claim — the issue's own law). The structural
case that survives any load: **~67 fewer dispatches/layer, ~160 MB/layer less traffic,
bit-identical output, and the p%64 fallback pathology removed** (odd-p chunks no longer
drop to P sequential dispatches).
- Composes with R1's staged recurrence (independent stages of the same layer).

## Files

- `crates/riir-infer-gpu/src/deltanet_prework_fused_cubecl.rs` (kernel + launcher + tests)
- `crates/riir-infer-gpu/src/deltanet_chunked_cubecl.rs` (carry-update extracted to
  `launch_conv1d_carry_update` — DRY, the chunked launcher now calls it too)
- `crates/riir-infer-gpu/src/ternary_deltanet_gpu_forward.rs` (the prework branch + toggle)
- `crates/riir-infer-gpu/tests/bench_1004_r2_prework_fused_ab.rs` (stage A/B)
- riir-ai: feature forward + re-exports + `bench_1004_r2_prework_fused_e2e.rs`
