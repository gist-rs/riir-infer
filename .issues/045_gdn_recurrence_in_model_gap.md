# Issue 045 — GDN recurrence costs 2.3× in-model what it costs in isolation

**Status:** OPEN 2026-10-13 — first-entry cost attributed to rayon worker wake-up (T1b); priced at ~14–16% of each token; prefetch-join (+2.1–2.6%) and an in-rayon keepalive (5.7× slower) both REFUTED. Remaining lever: a spin-barrier pool outside rayon (T4).

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

- [x] T1(b): a no-op `rayon::broadcast` placed just before the step (throwaway
      profiler edit) costs **4.82 ms/token ≈ 100 µs/layer**, and the step
      then drops to **4.86 ms ≈ 101 µs/layer** (from ~175). So the
      first-entry cost is waking the 16 workers, which went to sleep during
      the serial conv1d/gates/expand gap. It is not the state's cache
      placement.
- [-] T1(a) + T2, refuted: `with_state_prefetch` ran steps 4–8 on the caller
      inside `rayon::join` while the workers read the layer's state, to keep
      them awake and warm the state. In-process paired A/B (one load, two
      decode states, switch flipped per position, 64 positions, 3 runs): 0/64
      logit mismatches every run, but ratio **1.023 / 1.026 / 1.021** (on/off;
      off 107–116 ms/tok). It moves the wake-up cost and adds its own.
      Reverted; nothing committed.
- [x] T3 pricing (throwaway profiler edit: a timed no-op `rayon::broadcast`
      before every parallel section): the broadcasts cost **14.2 ms/token**
      and the sections then drop by ~16 ms (recurrence 8.2→3.4, FFN 54→50,
      dn_input_proj 15.8→13.5, dn_out_proj 8.6→5.8, attention −1.9); total
      flat at 98–100 ms. **About 14–16% of each token is parked-worker
      wake-up.**
- [-] T3 inside rayon, refuted: per-token keepalive (`spawn_broadcast` puts
      one job on each worker that spins in `rayon::yield_now` until the
      forward returns; the forward's own worker returns at once). In-process
      paired A/B, one load: bit-identical (0 mismatches), but **5.7× slower**
      (673 vs 97 ms/tok). Adding a 256- or 4096-`spin_loop` backoff between
      idle steal attempts did not help (8.2×, 7.3×), so steal contention on
      the forward thread's deque is NOT the mechanism. Root cause not
      established. Reverted; nothing committed.
- [ ] T4: the remaining lever is a pool outside rayon: a decode-scoped spin-wait barrier
      pool (llama.cpp's threadpool shape) so the workers never park between
      forks inside one token. Every fork after a serial gap pays this, not just
      the recurrence. Price it first: count the forks per token that follow
      a gap long enough to park the workers, using the broadcast probe per
      section. The prize is up to ~14 ms/token (~100 → ~86 ms). It needs the
      katgpt matvec kernels to take a pool handle instead of calling
      `par_chunks_mut` directly, so file it in katgpt-rs as a design issue
      before writing code.
