# Issue 045 — GDN recurrence costs 2.3× in-model what it costs in isolation

**Status:** OPEN 2026-10-13 — measured, cause unattributed; ~3–5 ms/token (3–5%) on the table, bit-identical fixes only.

## Measurements (M3 Max, AC, `--release`, Bonsai-2-27B PQ2_0, raw-Q2 arm)

| where | µs per layer call | ms/token (48 GDN layers) |
|---|---|---|
| in-model, `bonsai2_forward_profile` `dn_recurrence` | ~171–180 | 8.2–8.6 |
| in-model, an immediate SECOND call on the same state (throwaway probe, not committed) | ~106 | 5.1 |
| isolated `gdn_step_parallel_goat`, one hot state | 72–75 | 3.5 |
| isolated, 48 distinct states cycled (144 MiB > SLC, so cold) | ~80 | 3.8 |

So the kernel is NOT DRAM-bound in isolation: cold vs hot is +7%. The
in-model excess has two parts: ~75 µs/layer that a second call does not pay
(a first-entry effect: state/q/k/v placement after ~150 MB of weights stream
through, or a worker wake-up after the serial conv1d/gates sections), and
~30 µs/layer the second call still pays that the microbench does not.

## Refuted (do not retry)

- **Row-chunked tasks** (`par_chunks_mut` over 16/32/64 state rows instead of
  48 whole heads; bit-identical): **slower** at every chunk size — hot
  +11 / +29 / +40%, cold +9 / +19 / +30% (rows 64/32/16). Per-task overhead
  beats any P/E rebalancing on ~1.5 µs of per-head work. The Issue-935 matvec
  finding does not transfer.
- **Fewer passes per row** (fold decay into the dots, one read + one write):
  changes the rounding order, so it is not bit-identical and needs a quality
  gate. It also targets compute, which isolation shows is not the bottleneck.

## Tasks

- [ ] T1: attribute the first-entry 75 µs. Probes, in order: (a) touch the
      layer's state (prefetch read) inside the preceding `dn_input_proj`
      fork, then time the step; (b) time a no-op `rayon::join` just before
      the step to price the wake-up; (c) `xctrace` CPU counters (L2/SLC miss)
      on the two calls.
- [ ] T2: if (a) wins, ship a bit-identical prefetch, measured with the
      real-model paired GOAT (`ternary_q2_resident_goat`, 3 runs), not the
      profiler alone.
