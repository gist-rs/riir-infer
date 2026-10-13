# Bench 879 — raw-Q2 resident projections (`ternary_q2_resident`, katgpt-rs Issue 935 T5)

**Status:** DONE 2026-10-13 — G1 bit-identical, G2 PASS; **PROMOTED** into `deltanet_ternary_inference` after Issue 043 gave the GPU uploaders a plane view (§Promotion), with fused shared-input calls (FFN gate+up, then `DeltaNet` qkv+z and attention q/k/v via one N-way kernel) taking the paired tg ratio 0.931 → 0.845–0.880 → **0.798–0.833** (§Shared-input fusion). Follow-up landed earlier: the bit-plane repack is word-at-a-time + row-parallel — Bonsai-2-27B load 124.9 → 5.8 s for EVERY consumer (§Repack).

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
| real-model GOAT after T4 ×3 (load avg 3.1 → 10.4, rising — see the load note below) | 0/64 mismatches + greedy identical every run; ratio **0.880 / 0.862 / 0.845**; q2 102.2 / 104.4 / 106.2 ms/tok against planes 116.5 / 120.7 / 125.4 |
| riir-infer lib, default + `deltanet_ternary_inference,bonsai2_hadamard` | 264 / 342 passed |
| clippy: core (both arms + `forward_profiling`), riir-infer-gpu `ternary_gemv` (± Metal/ANE/batched), cudarc lane typechecked for `aarch64-unknown-linux-gnu` (11 gated modules proven compiled), riir-train-engine, riir-ai riir-engine + riir-gpu | clean |

The T4 gain (~7 ms/token) came out larger than Issue 043's ≤ 1 ms estimate;
§Shared-input fusion attributes it at the kernel. Per-pair ratios are
quotable; absolute ms moved with the load.

Downstream fixes in the same change: riir-train-engine's three exhaustive
`ProjWeights` matches now go through `planes_cow()` / `as_ternary()` /
`as_ternary_mut()` (they could not name the new arm). Four riir-ai riir-gpu
test targets had not compiled since the enum landed; their fixture sites now
call `as_ternary[_mut]()`.

Opt out per process BEFORE a load: `set_ternary_q2_resident(false)`.

katgpt-rs commits: `8a7fff28a` (pair entry + public fast-path probe).

## Shared-input fusion (2026-10-13, follow-up)

**Kernel attribution** — `katgpt-types --example ternary_q2_pair_ab`
(17408×5120 gate/up, 16 rayon threads, interleaved rounds, outputs asserted
bit-identical first). Three runs:

| arm | median | ratio vs two calls |
|---|---|---|
| two single calls | 555.6–568.3 µs | 1.000 |
| pair (one prep, one fork-join) | 431.4–446.9 µs | 0.773–0.780 |
| one merged 2R-row matrix (control) | 449.9–456.1 µs | 0.808–0.814 |

121–131 µs saved per FFN × 64 layers = 7.7–8.4 ms/token, which accounts
for the ~7 ms the real-model GOAT showed. The merged control (one prep +
one fork-join over a single matrix) recovers most of the gain, so the saving
is the second prep + fork-join + tail, not the pair's layout. Pair beating
merged by a further ~4% is not explained.

**N-way entry** — `simd_ternary_q2_matvec_q8_parallel_n::<N>` (fixed-size
arrays, zero-alloc, bit-identical to N single calls; unit gate at N=3 with
unequal rows, serial + parallel). The pair is now its N=2 wrapper; after the
change the pair arm reads 0.792–0.796 (control moved 0.81 → 0.82 the same
runs, so pair/merged shifted ~1%, inside noise). riir-infer: `matvec_n_into`
/ `bitlinear_n`, used for `DeltaNet` `in_proj_qkv` + `in_proj_z` (48 layers)
and attention q/k/v (16 layers). The profiler (`forward_profiling`) now calls
the production `bitlinear` / `bitlinear_n` / `bitlinear_pair` instead of its
stale private copy, so it times the same dispatch.

**Real model** (`ternary_q2_resident_goat`, launch load ~2.1–2.5, AC, no
other cargo/rustc running):

| arm | runs | paired ratio | q2 ms/tok |
|---|---|---|---|
| FFN pair only (baseline for this step) | 3 | 0.893 / 0.872 / 0.837 | 104.1 / 103.7 / 109.9 |
| + qkv+z and q/k/v N-way | 3 | **0.833 / 0.808 / 0.798** | 113.4 / 116.9 / 102.5 |

400/400 raw-Q2, 0/64 logit mismatches, greedy identical in all six runs.
The absolute ms do not separate the two rows, because the planes arm moved
just as much (117–131 → 129–145 ms) over back-to-back runs. **Load note:** in
these runs the load average rose from ~2 to ~11 with no other cargo/rustc
process on the box, so the test's own 16-thread decode drives it. The
"sibling sessions" attribution in the T4 row above was therefore at least
partly wrong. Quote the paired ratio, not absolute ms.

**Section breakdown** (`bonsai2_forward_profile`, `ternary_q2_resident`, 32
tokens): total **100.9 ms/token (9.9 tok/s)** — FFN 54.2 (53.7%),
`dn_input_proj` 15.8, `dn_out_proj` 8.6, `dn_recurrence` 8.2,
`attn_qkv` 4.2, `attn_out_proj` 3.4, LM head 3.3, everything else < 1.1
each. About 7 GB of 2-bit weights are streamed per token, so the decode
runs at roughly 70 GB/s of weight traffic. The one large non-weight section
is `dn_recurrence` (8%).

katgpt-rs commit for the N-way entry: `9f3f78bf6`.
## CUDA lane (Issue 044, closed 2026-10-13)

4090 (i7-13700K + RTX 4090, Windows), `si_flag0_probe` (128-token prefill +
64 greedy tokens on the eager AND the graph lane, blake3 logit digests),
built at `995b577` (planes-resident, the parent of the promotion) and at
`efd66be` (Q2-resident + the N-way fused CPU calls), each in a clean detached
worktree with katgpt-rs at `9f3f78bf6`. Build features
`ternary_gemv_cuda_raw,cuda_graphs_forward,riir-infer-core/bonsai2_hadamard`.

| | result |
|---|---|
| stdout pre vs post | **byte-identical** (SHA-256 `7FFAEA98…62BCCB`, 1466 bytes each) |
| eager == graph ids | YES in both builds |
| prefill digest @128 | `7c406a36…c6ce73eb6` both builds |

Admissibility: the probe does not print the representation. The post build
enables `ternary_q2_resident` through `deltanet_ternary_inference`, and the
box's CPU has AVX2, so the loader's fast-path guard keeps raw Q2. That is
inferred from the features and the CPU, not observed in the output. The first
attempt (without `bonsai2_hadamard`) refused to load the Hadamard-folded GGUF
in both builds. The probe's documented build line now names the feature.

## Decode spin pool (katgpt-rs Issue 936 T3, PROMOTED 2026-10-13)

Riir-infer Issue 045 priced it: rayon's workers park in the serial gaps
between a token's forks, and waking them costs ~14–16% of each token. The
`decode_spin_pool` feature (katgpt-rs `bb3b24f2e`, `65574c5d9`, `9c4f7f9dd`)
runs the ternary decode forward on the calling thread with a spin-barrier
pool installed:
- The Q2 single/N-way and bit-plane q8 kernels, the GDN recurrence and the
  attention-heads fork all claim fixed tasks from one CAS'd claim word.
- Workers spin up to 100 µs idle inside the token, then park until the next
  job.
- The pool is sized to P-cores − 1 (11 on the M3 Max).

Entry: `deltanet::on_decode_pool`, used by the ternary forward, its profiler
twin and the ternary-KV forward. The f32 forward keeps `on_rayon_worker`,
because its kernels are not pool-aware.

**Gate** `tests/decode_spin_pool_goat.rs`: one load, two decode states, pool
vs rayon flipped per position, 64 positions, M3 Max AC.

| condition | runs | pool ms/tok | rayon ms/tok | paired ratio | logit mismatches |
|---|---|---|---|---|---|
| no burner, before the idle window (spin forever) | 1 | 70.7 | 106.4 | 0.679 | 0 |
| sibling `cargo test` using ~7 cores, spin forever | 2 | 174–181 | 132–140 | **1.22 / 1.35** | 0 |
| idle window grid, no burner: ∞ / 100 / 30 µs | 1 each | 74.1 / 73.3 / 77.5 | ~100 | 0.740 / 0.735 / 0.757 | 0 |
| idle window grid, 6× `yes` burner: ∞ / 100 / 30 µs | 1 each | 165.8 / 94.6 / 97.7 | ~121 | **1.339** / 0.794 / 0.834 | 0 |
| default (100 µs), no burner | 3 | 71.3 / 71.6 / 72.2 | 98–105 | **0.719 / 0.731 / 0.679** | 0 |
| default (100 µs), 6× burner | 2 | 96.4 / 95.3 | 118–122 | **0.822 / 0.794** | 0 |
| promoted build (feature via `deltanet_ternary_inference`) | 1 | 73.2 | 100.1 | 0.734 | 0 |

So the CPU decode goes ~100 → ~72 ms/token (≈ 10 → ≈ 14 tok/s), bit-identical.
Spinning forever is a trap on a shared box (1.2–1.35× slower); the idle
window is what makes it promotable. Launch load ranged 3–11 across these runs
(sibling sessions plus the test's own threads), so quote the paired ratios.

**Off macOS the pool is inert by default.** `for_this_machine()` gives 0
workers there, and a 0-worker pool is never used: the kernels keep rayon and
`on_decode_pool` keeps Bench 880's rayon-worker posture. Two sessions measured
the microbench on the 4090 box (i7-13700K, 8P+8E, 24 threads, before the idle
window). Spin/rayon at the decode posture was 0.97–0.99 with 7 workers, 1.71–1.75
with 11, 2.55 with 15 and 3.86–3.93 with 23. To opt in, set
`KATGPT_SPIN_POOL_WORKERS`. That configuration is unmeasured with the idle
window.

**End to end** (`twt_perf_tg_pp`, parent model as both arms, 512-token
teacher-forced prefill + 128-step greedy decode at context 512–640, where
attention also takes its pool arm). The arms ran sequentially, not
interleaved, and launch load rose 3.7 → 14, so the paired GOAT ratios above
remain the gate:

| run | pp tok/s | tg tok/s |
|---|---|---|
| pool (default) | 12.69 / 12.41 | **12.72 / 12.75** |
| rayon (`KATGPT_SPIN_POOL_WORKERS=0`) | 8.97 / 8.76 | **8.86 / 8.30** |
| pool again | 12.42 / 12.25 | **12.64 / 12.21** |

Not yet measured: a GPU-hooked decode. The pool is installed there too; its
workers park after 100 µs idle while the GPU works.

