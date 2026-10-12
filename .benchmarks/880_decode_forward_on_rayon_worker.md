# Bench 880 — the decode forward runs ON a rayon worker (no 17th thread)

**Status:** DONE 2026-10-13 — bit-identical (pure scheduling + a bit-identical conv1d specialization); Bonsai-2-27B CPU decode 117.7 → 100.9 ms/token in the section profiler. Default (no flag: a placement fix, not a lossy primitive).

## Finding

New instrument `tests/bonsai2_forward_profile.rs` (drives `profiled_forward_ternary`,
Issue 603's instrumented twin, over the real model) showed `dn_conv1d` at
6.6 ms/token = **137 µs/layer**, while the same call measured in isolation is
**17 µs/layer** — warm or with a 1 GB cache flush, old generic loop or new.
conv1d does no parallel work, so its fixed cost moving 8× is not compute and
not memory: the calling thread was being time-sliced. Rayon's global pool has
16 workers on 16 cores; after every projection fork they spin-yield
(`swtch_pri` was the #2 symbol in the decode `sample`), and the forward's own
thread is a 17th runnable thread competing with them for every serial section.

Control: `RAYON_NUM_THREADS=15` (one core left for the caller) → conv1d
6.1 → 0.95 ms/token, total 123.6 → 107.1 ms (12 workers: 108.6).

## Fix

`deltanet::on_rayon_worker(f)` — inline when already on a pool thread, else
`rayon::scope(|_| f())`, which (from a non-pool thread) injects the body into
the pool and parks the caller on a latch. `forward_qwen_deltanet_ternary_with_hook`
and its twin `profiled_forward_ternary` wrap their bodies in it; the full
16-worker pool stays (better for the matvecs than 15). Plus
`causal_conv1d_update`: a kernel-4 fixed-array arm (same left-to-right sum, no
FMA), serial — a rayon split over ~10k channels measured 64 µs vs 17 µs serial.

## Numbers (`bonsai2_forward_profile`, raw-Q2 arm, 32 tokens after 4 warm-up, M3 Max AC)

| section (ms/token) | before (load ~3–4) | `RAYON_NUM_THREADS=15` | after (16 workers, load ~2.7) |
|---|---|---|---|
| dn_input_proj | 18.94 | 18.57 | 17.24 |
| dn_conv1d | 6.59 | 0.95 | **1.21** |
| dn_recurrence | 10.18 | 8.12 | 6.90 |
| dn_out_proj | 9.07 | — | 6.35 |
| attn_qkv | 6.09 | — | 5.16 |
| FFN | 58.86 | 57.42 | 55.76 |
| **TOTAL** | **117.65** | 107.08 | **100.87 (9.91 tok/s)** |

Identity: `causal_conv1d_update_bit_identical_to_reference` (verbatim old loop,
serial + Bonsai shape + kernel-3, 3 steps, x and state bits);
`ternary_q2_resident_goat` re-run through the wrapped forward: 64/64 logit
vectors bit-identical planes-vs-raw-Q2, greedy identical, and — at load avg
~10 from sibling sessions — planes 117.0 / raw-Q2 107.8 ms/token vs 126.4 / 116.8
before at load ~2.5. riir-infer lib 341/341.

End-to-end `twt_perf_tg_pp` (default features, bit-plane q8) after the change:
tg 8.55 / 7.89 tok/s at load ~3.4 (was 7.86 / 7.86), then 8.09 / 7.74 and
7.34 / 7.20 as sibling load climbed to 6.7 → 10.1 — the box was not quiet
enough for a clean end-to-end number; the profiler's section split is the
mechanism evidence (a serial section's fixed cost moved 5× from placement alone).

Re-run: `cargo test --release --features forward_profiling,bonsai2_hadamard,ternary_q2_resident --test bonsai2_forward_profile -- --ignored --nocapture --test-threads=1`
