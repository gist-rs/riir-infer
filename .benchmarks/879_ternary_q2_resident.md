# Bench 879 — raw-Q2 resident projections (`ternary_q2_resident`, katgpt-rs Issue 935 T5)

**Status:** DONE 2026-10-13 — G1 bit-identical, G2 PASS (0.931), kept OPT-IN (GPU loaders refuse non-plane projections; see Verdict). Follow-up landed: the bit-plane repack itself is now word-at-a-time + row-parallel — Bonsai-2-27B load 124.9 → 5.8 s for EVERY consumer (§Repack).

## What

The GGUF `Q2_0` block already stores each weight as a 2-bit code `q = w + 1`.
`ternary_q2_resident` keeps Q2_0 projections in that layout
(`ProjWeights::TernaryQ2`, katgpt-core `TernaryQ2Weights`) and runs the
raw-Q2 int8-activation kernel (`simd_ternary_q2_matvec_q8_parallel`):
`shr`+`and` per code byte against activations permuted once per call, `sdot`
(NEON) / `pmaddubsw` (AVX2), minus the per-group activation sum. Same exact
integer sums and same f32 accumulation order as the bit-plane q8 arm
(`ternary_q8_act`) ⇒ bit-identical output. The loader skips the bit-plane
repack entirely for these tensors.

## Kernel (katgpt-types `ternary_group_par_chunk_ab --features ternary_q2_q8`, M3 Max, AC, load ~2.5)

| shape | f32 shipped | q8 planes | q8 raw q2 | q2 / planes |
|---|---|---|---|---|
| 17408×5120 | 1578.6 µs | 306.2 µs | **263.1 µs** | 0.86 |
| 5120×17408 | 1566.6 µs | 297.1 µs | **246.7 µs** | 0.83 |
| 16384×5120 | 1466.4 µs | 275.5 µs | **227.2 µs** | 0.82 |
| 5120×6144 | 645.8 µs | 147.7 µs | **125.2 µs** | 0.85 |

Single thread the raw-Q2 kernel is **2.0×** the plane kernel (17408×5120:
2661 → 1341 µs); from 8 threads both saturate (~245 µs ≈ 97 GB/s against a
measured ~125 GB/s DRAM streaming roof / ~200 GB/s SLC-resident on this box),
so the parallel gain is bandwidth-capped at ~15%.

## Real model — `tests/ternary_q2_resident_goat.rs` (Bonsai-2-27B PQ2_0, one process, two loads)

| metric | planes q8 | raw-Q2 q8 | bar |
|---|---|---|---|
| logit vectors bit-identical (64 pos, full vocab) | — | **64/64** | 64/64 |
| greedy 32 tokens | — | **identical** | identical |
| decode wall | 126.4 ms/tok | **116.8 ms/tok** | — |
| median paired ratio | — | **0.931** | ≤ 0.95 |
| load | 124.9 s | **14.5 s** | — |
| projections raw-Q2 | 0/400 | 400/400 | > 0 |

Load avg at launch ~2.4–3.1 (AC power). Page cache was warm for BOTH loads
(earlier same-file runs load the second arm in ~125 s too), so the load delta
is the bit-plane repack, not I/O.

Re-run: `cargo test --release --features ternary_q2_resident,bonsai2_hadamard --test ternary_q2_resident_goat -- --ignored --nocapture --test-threads=1`

## Verdict

GOAT on the CPU decode lane (bit-identical, +8% tg, 8.6× load), but **not**
promoted into `deltanet_ternary_inference`: the GPU uploaders
(`q4_refuse`, Metal `register_layer`, cudarc `proj_of`, `vram_budget`) take
`&TernaryGroupWeights` and refuse loud on any non-plane projection, and the
f32 arm (`set_ternary_q8_act(false)`) has no Q2 kernel. Promoting would turn
every GPU build that loads through the same loader into a panic. CPU-only
consumers enable `ternary_q2_resident`; `set_ternary_q2_resident(false)`
before a load restores planes in the same binary.

## Repack (`repack_q2_0_to_ternary_group`) — the broader win

The 125 s load was the bit-plane repack itself (one weight per iteration, one
thread, ~56 MB/s) — paid by every consumer, GPU included. Now 32 codes per
u64 (`lo = w & 0x55…`, `hi = (w>>1) & 0x55…` → pos `hi & !lo`, neg
`!(hi|lo)`, bad `hi & lo`, then a 5-step even-bit compress) with rows on
rayon; a code-3 hit re-scans sequentially so the reported (row, col) is the
same first site as before. Unit gate: equals the per-bit reference on a
37×640 random fixture and reports row 5 over row 30
(`test_repack_word_path_matches_per_bit_reference`). Real model (re-run of
the gate above, load avg ~4.6): **planes load 124.9 → 5.8 s (21.5×)**, raw-Q2
load 14.5 → 2.8 s (its wte/lm_head globals ride the same repack), 64/64
logit vectors still bit-identical against the raw-Q2 arm (which never touches
the repack — an independent oracle), greedy identical, tg ratio 0.928.

katgpt-rs commits: `7032864d0` (ternary_q2_q8 kernel), `2522dc2a9` (AVX2 q8).
