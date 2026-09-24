# Plan 003 — EXL3 T7c-1d: the sound decode-bench harness (Issue 001 §17.5)

**Status:** CLOSED — 2026-09-25. T1/T2 landed (commit `9989e9f`); T3 ran
on the 4090 via a SCHEDULED TASK (the detached-SSH reaping lesson) — exit
0, 520.7 s, 15 rows, 14 under the 10% gate; verdicts recorded in the
issue's §17.5 T7c-1d row. T4 this commit.

T7c-1b's reps-differential + readback harness proved too noisy at this
scale (per-chunk readbacks interleave; small decode kernels are
launch-bound; single-pair walls oscillated ±3×). This plan lands the
REPLACEMENT harness and re-measures A1/A2/A3 to the ±5% stability bar.

## The measured constraint that shapes the design

True CUDA-event timing is NOT reachable in cubecl 0.11: cubecl-cuda's raw
`CUstream` is private (`CudaServer.streams`), its `Fence` API only
sync/waits (no elapsed), and `client.profile()` on the CUDA server resolves
through `TimestampProfiler` = HOST system time bracketed by
`block_on(sync)` on both ends (verified in the vendored cubecl-runtime +
cubecl-cuda sources). So the sound reachable primitive is
**sync-bracketed system-time sampling** — sound for kernels ≥ ms scale,
unsound below it, which the harness enforces with a minimum-work floor and
discloses on every row.

## Harness contract (what makes it sound where T7c-1b's was not)

1. **No readback inside the timed region.** One warmup (JIT + allocator),
   then N=30 timed passes per arm; the pass body = `reps` enqueues of the
   decode kernel + ONE final readback OUTSIDE the timed window (the
   decode-only path already supports this shape via reps; the harness
   times pass = reps×kernel-enqueue + sync).
2. **Enough work per pass** (the ≥5 ms floor): reps chosen per layer so a
   pass is launch-overhead-dominated no longer (smallest layer gets the
   largest reps; measured pass wall printed per row).
3. **Robust stats, instability refusal:** report min/median/p90 per arm;
   the headline figure = MIN (peak attained); if (p90 − min)/min > 10%
   the row prints UNSTABLE and the harness refuses to publish a verdict
   for it (never a noisy number wearing a measurement's clothes).
4. **Cross-method agreement gate:** the min-of-N (sampling method) vs the
   reps-differential (the T7c-1b method, still computed) must agree within
   15% per (layer, arm) — two independent samplings agreeing is the
   soundness evidence; a disagreement prints and the row is UNSTABLE.
5. **Interleaving + box state:** arms interleaved per round; box state
   (GPU util/VRAM/RAM/power, concurrent compute check) printed in the
   header — every figure carries it (the G2 rule).
6. **Bit-exactness stays a gate, not a bench product:** v2-vs-v1
   `to_bits` per layer re-asserted each run (§17.3 gate 1 at bench scope).

## Tasks

- [x] T1 — harness fn `bench_arm_stable` + `StableBench` + the
      `TooFastToTime` error (riir-infer-gpu, `exl3_gpu` feature): warmup +
      N=30 samples + min/median/p90 + the ≥5 ms/pass floor + the 10%
      instability gate + the 15% cross-method gate vs the retained
      differential method. The real-pack driver is the
      `real_pack_decode_bench_stable` `#[ignore]`d test (the siblings'
      convention); v2-vs-v1 bit-exactness re-asserted per layer each run.
- [x] T2 — compile + clippy `-D warnings` clean at both postures; the
      M3 Metal lane runs the module suite green (7 passed; the stable
      bench itself is pack-gated and runs on the 4090, T3).
- [x] T3 — the RUN on the 4090 (pack `E:/git/riir-infer/.raw/packs/qwen38-27b-exl3-4bpw`):
      5 classes × 3 arms complete; full table in the issue §17.5 T7c-1d.
      A1≈A2 (extraction confirmed dead), A3 = 1.25–1.29× (the LUT gather),
      wall ≈ 28 Gw/s latency-class. One A1 row UNSTABLE (13.4%, excluded).
- [x] T4 — commit + push; plan closed. T7c-2's proceed/pivot/close is the
      OWNER decision the table feeds — deliberately not taken here.
