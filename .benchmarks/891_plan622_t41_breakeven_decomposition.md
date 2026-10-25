# Bench 891 — Plan 622 T4.1 (break-even half): the league-posture prefill class decomposition + the pruned-prefill TTFT ceiling arithmetic

**Status:** RECORD — the T4.1 mandatory break-even row is DELIVERED on the measured class decomposition; the measured TTFT-vs-K cells remain gated on the T2.4 GPU vehicle (unchanged). The arithmetic verdict: the pruned-prefill ceiling at the league cells is **~1.5–1.8× TTFT at K ≤ 0.25** (the paper's low end, 1.4×), positive but small; attention is ~linear in p on this engine so **the prunable share does NOT grow with context length** — the paper's long-context win condition does not transfer to this hybrid-GDN stack. The T2.4 vehicle is priced at ~1.5× and left un-built pending the owner's appetite for a lossy opt-in lane whose quality cost T3.4 already refused as a default (T4.2 disposition stands).

**Date:** 2026-10-25 (zed-4090-idle-1025b)
**Plan:** [.plans/622_tokengate_keybias_pruned_prefill_lane.md](../.plans/622_tokengate_keybias_pruned_prefill_lane.md) T4.1 (break-even row)
**Model:** `../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` (folded, prism.hadamard; G1 argmax 248046, fnv `0f8fddb9ef217548` every run incl. the league manifest's warm pin shape).
**Box:** 4090 (Windows, RTX 4090 24 GB, AC, idle: GPU 4% / 382 MiB at fire; idle clock 210 MHz, burst-warmed per round). Commits at fire: riir-ai `9cffde841` + riir-infer `5ded2df` + katgpt-rs `d18d343ce`.
**Instrument:** the incumbent `bench_772_t0_gemm_share` (league manifest ours_cmd verbatim — features `ternary_gemm_batched,cubecl_runtime,ternary_gemv_cuda_raw,ternary_gemv,ternary_deltanet_chunked_prefill,prefill_q8_act,prefill_mmq_v2,bonsai2_hadamard`) at `RIIR_PREFILL_CUDA_GRAPHS=0` + `RIIR_PREFILL_CUDA_TRACE=2` (the issue-772 T2 stage-attribution arm: per-kernel-class host timing with a sync per class — the instrument's documented distortion class). **Both TRACE=2 cells tripped the Bench-976 ramp tripwire (4.11% / 3.99% round spreads) — the scored verdicts are INVALID-ramp by construction and are used for ATTRIBUTION ONLY; no league pin moves.** The untraced eager cell PASSED the tripwire cleanly.

## The cells

| cell | wall | tok/s | verdict |
|---|---|---|---|
| pp2048 graphs-on (league, Bench 999 ref, same day) | 497 ms | 4118 | scored ref (not re-run here) |
| pp2048 eager, untraced | 576 ms | 3554 (3554/3554/3555) | PASS, ramp-clean |
| pp2048 eager + TRACE=2 | ~720 ms | 2841/2729 | INVALID-ramp (attribution only) |
| pp4096 eager + TRACE=2 | ~1.60 s | 2602/2502 | INVALID-ramp (attribution only) |

**The graphs-vs-eager launch-tax slice: 79 ms/chunk at pp2048 (16%)** — the ~1400-launch eager chunk pays ~0.05–0.06 ms/launch of WDDM submit tax that the graph replay deletes. This is the number that makes any eager pruned segment a non-starter and T2.4's graph-per-padded-length capture load-bearing.

## The class decomposition (TRACE=2 stage sums, steady rounds, µs→ms)

| class | p=2048 | p=4096 | ratio | marginal read |
|---|---|---|---|---|
| gemm (MLP+projections, MMA) | 331–334 | 661–665 | **2.005×** | linear (row-scaled) |
| gdn (chunked recurrence) | 316–326 | 743–816 | **2.36×** | superlinear at the 4096 chunk shape |
| attn (16 FA layers, gang layout) | 57.8–59.6 | 117.9–118.5 | **2.02×** | ~LINEAR in p — not the dense p² law |
| rms | 34.2–35.5 | 87.4–88.4 | 2.53× | row-scaled |
| other (elementwise/rope/copies, unclassified) | 682–693 | 1490–1567 | 2.20× | mostly row-scaled |
| classified+other sum | ~1436 | ~3164 | — | inflated by the per-class sync tax (documented); shares, not absolutes, are the signal |

**Shares (p=2048):** gemm 23% · gdn 22% · attn **4.1%** · rms 2.4% · other 48%. At p=4096 the attention share is 3.7% — **flat-to-falling with p.**

## The break-even arithmetic (L = 31, the mid full-attention of 16 at every-4th index; layers > L = 32/64 = every class × 0.5)

Pruned TTFT at keep-fraction K, padded kept rows, one-pass mid-sync steering (the tap D2H ~1–3 ms; no second pass — selection needs only layers ≤ L):

TTFT'(K) ≈ A + B × (kept_pad/2048) + G, with A = B ≈ 248 ms (the graphs layer wall split at L), G ≈ 3 ms:

| K | kept | kept @bucket 128 | TTFT' | speedup | quality cost (T3.4 sel axis) |
|---|---|---|---|---|---|
| 0.25 | 516 | 640 | ~328 ms | **1.51×** | Δppl +0.399 vs ceiling, retrieval 1.000 |
| 0.125 | 260 | 384 | ~298 ms | **1.67×** | Δppl +0.399, retrieval 1.000 |
| 0.0625 | 133 | 256 | ~282 ms | **1.76×** | Δppl +0.786, retrieval 0.75–1.00 (scorer-dependent) |

Peak-memory half: KV saved at layers > L = 8 att layers × kept × kvd(1024) × 4 B ≈ **16.5 MB at K=0.25/ctx 2048** against a ~9–10 GB working set (the MLP scratch dominates; bench 883) — negligible at league lengths; the KV axis only matters at the 32K+ capability boundary, where bench 883's OOM row sits and where this lane's dispatch-identity no-op would have to be overridden explicitly.

**The mandatory break-even row: the league cells (pp2048/pp4096) sit AT the paper's low end (1.4×), not below it** — the win survives bucket-128 padding (+24% on the >L segment) because the >L share being pruned is large (≈50% of layer work is row-scaled above L). But three conditions are load-bearing: (a) the >L segment MUST be graph-captured per padded length (eager, the launch tax alone is ~79 ms — half the entire win); (b) the steering must stay one-pass (a second full pass, as the CPU harness's structure needed, erases the win 2× over); (c) the quality cost of T3.4 rides along (lossy-surface law: opt-in forever, T4.2).

**The T2.4 build verdict (priced, not built):** ~1.5× TTFT at K=0.25 on a lane whose accuracy overlay is already REFUSED as a default (bench 889) — a scoped opt-in systems win. The graph-per-padded-length capture is real engineering on the 4797-line `prefill_cuda_full.rs` (segment split at L, compact KV/gather, pad-bucket capture keyed by length). Left to the owner: build the vehicle for the ~1.5× opt-in, or bank this record as the lane's systems ceiling and stop. The GDN superlinearity at p=4096 (2.36×) is a separate, UNPRUNED observation recorded for the chunked-recurrence lane (the row-count class does not explain it; single chunk both cells).

## Provenance

- Logs: `/tmp/p622_t41/{pp2048_trace2,pp4096_trace2,pp2048_eager_clean}.log` (regenerable; commands = the league manifest ours_cmd verbatim + the env deltas above, run from the riir-ai root).
- The stage instrument is the incumbent TRACE arm — no code was changed for this record (zero-diff measurement).
- RAMP-TRIPWIRE disclosure: both TRACE=2 runs refused their scored verdicts per the Bench-976 amendment (raise `B772_WARM_SECONDS`); the refusal is the instrument working — the stage sums were steady across rounds (±2%) and are used as attribution shares only.
