# Bench 879 — raw-Q2 resident projections (`ternary_q2_resident`, katgpt-rs Issue 935 T5)

**Status:** DONE 2026-10-13 — G1 bit-identical, G2 PASS; **PROMOTED** into `deltanet_ternary_inference` after Issue 043 gave the GPU uploaders a plane view (§Promotion), with a fused FFN gate+up call taking the paired tg ratio 0.931 → 0.845–0.880. Follow-up landed earlier: the bit-plane repack is word-at-a-time + row-parallel — Bonsai-2-27B load 124.9 → 5.8 s for EVERY consumer (§Repack).

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

## Promotion (Issue 043, closed 2026-10-13)

The Verdict's blocker was the GPU uploaders: they take `&TernaryGroupWeights`
and refused any non-plane projection. Issue 043 removed it without a second
resident copy on the upload path:

- **Plane views on the Q2 projection** (`TernaryQ2Proj`): `planes()` builds
  once from the same fast repack and stays RESIDENT, so its buffers survive
  moves and the pointer-keyed caches (`gemv_ternary_cubecl`,
  `gemv_ternary_cuda_raw`, keyed on `(pos_bits, neg_bits).as_ptr()`) stay
  valid. `planes_cow()` borrows that view if it exists, else builds a
  TRANSIENT one. One-shot device uploads (`q4_refuse`, ANE `register_layer`,
  cudarc `proj_of`) take the transient view, so no permanent second host
  copy. `vram_budget` prices Q2 from the geometry without allocating.
- **`as_ternary()` answers for Q2** (a Q2 projection IS ternary) through the
  resident view. Every existing hook, act-tap and test consumer works
  unchanged. `as_ternary_mut()` DEMOTES to the plane arm in place, because
  mutating a view while the raw blocks still drive the kernel would desync
  them.
- **f32 arm (T2)**: `set_ternary_q8_act(false)` on a Q2 projection runs the
  bit-plane f32 kernel over the resident view (it used to panic), the exact
  kernel a plane-loaded model runs.
- **Loader guard**: the raw-Q2 layout is declined where its SIMD arm cannot
  run (katgpt-core `q2_fast_path_available`: no aarch64 `dotprod` and no
  x86_64 `avx2`). There the raw-Q2 entry would run its scalar reference,
  while the plane q8 entry falls back to the f32 kernel.
- **Fused gate+up (T4)**: `matvec_pair_into` / `bitlinear_pair`. When both FFN
  projections are Q2 on the q8 arm, katgpt-core's
  `simd_ternary_q2_matvec_q8_parallel_pair` does one activation
  quantize+permute and one fork-join instead of two. It is bit-identical to
  two calls (unit gate, serial + parallel postures). Hooked calls are
  unchanged.

### Measurements (M3 Max 16c, AC power, `--release`)

| check | result |
|---|---|
| unit: plane views = loader repack = katgpt `to_group`; transient never resident; buffer key stable across a move; f32 + q8 arms = plane arms; `as_ternary_mut` demotes | PASS |
| Metal pre-rotation prefill pin @2048 (`g1_032_folded_prefill_metal::prerotation_prefill_pin_unchanged`), Q2-resident load → transient-view uploads | `fnv 99a0733c45a0e663` — **bit-exact** (this run would have panicked in `q4_refuse` before) |
| real-model GOAT before T4 (load avg ~4.3) | 400/400 raw-Q2, 0/64 logit mismatches, greedy identical, ratio **0.934**, load 5.7 → 3.8 s |
| real-model GOAT after T4 ×3 (load avg 3.1 → 10.4, rising: sibling sessions) | 0/64 mismatches + greedy identical every run; ratio **0.880 / 0.862 / 0.845**; q2 102.2 / 104.4 / 106.2 ms/tok against planes 116.5 / 120.7 / 125.4 |
| riir-infer lib, default + `deltanet_ternary_inference,bonsai2_hadamard` | 264 / 342 passed |
| clippy: core (both arms + `forward_profiling`), riir-infer-gpu `ternary_gemv` (± Metal/ANE/batched), cudarc lane typechecked for `aarch64-unknown-linux-gnu` (11 gated modules proven compiled), riir-train-engine, riir-ai riir-engine + riir-gpu | clean |

The T4 gain (~7 ms/token) came out larger than Issue 043's ≤ 1 ms estimate.
The likely cause is that one fork-join per layer is saved, not just the
activation prep, but this was not separately profiled. Per-pair ratios are
quotable; absolute ms moved with the rising load.

Downstream fixes in the same change: riir-train-engine's three exhaustive
`ProjWeights` matches now go through `planes_cow()` / `as_ternary()` /
`as_ternary_mut()` (they could not name the new arm). Four riir-ai riir-gpu
test targets had not compiled since the enum landed; their fixture sites now
call `as_ternary[_mut]()`.

Opt out per process BEFORE a load: `set_ternary_q2_resident(false)`.

katgpt-rs commits: `8a7fff28a` (pair entry + public fast-path probe).
