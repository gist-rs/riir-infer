# Bench 940 — Issue 1150 follow-up 4: the dense-lane relaxed_precision — 2× e2e on the layer GEMMs (the post-bench-939 ceiling)

**Status:** LANDED (2026-11-04, zed-amo-i1150-relaxed@amo, `bd297c7`). The
bench-939 remaining ceiling (gdn 66% + attn 17.9% buckets — the layer GEMMs
at ~16 TFLOPS effective) is fixed at the root: the dense matmul2d host
compiles `relaxed_precision = true` behind
`RIIR_QWEN38_DENSE_MM2D_RELAXED=1` (promote-only) — the bench-923 precedent
on the same kernel family (1.96–2.19× in the dist lane; llama.cpp compiles
it true upstream, the prismml fork `73bfe6c` too).

## Cells (dense-f16 MM2D + Q6_K packed tail posture, warm caches, fresh processes, median of 3)

| cell | exact (incumbent) | relaxed | e2e |
|---|---|---|---|
| pp512→512 | 274 tok/s (1.87 s) | **483 tok/s (1.06 s)** | **1.76×** |
| pp2484→2560 | 291 tok/s (8.78 s) | **587 tok/s (4.36 s)** | **2.02×** |
| load | 19.9 s | 20.6 s | — (unchanged; the arm compiles one more pipeline) |

Stage-timer @pp512 (synchronized, shares-comparable): gdn bucket
1329/1359 ms → **700/778 ms (1.85–1.9×)**; attn bucket 339/361 ms →
**182/202 ms (1.85×)**; tail_logits stable 8–13 ms. League read: 281 →
**~480-590** (MLX 1411 / uzu 1157 / fork 898 at pp-5k-class) — the gap
closes from ~4.7× to **~2.4-2.9×**, and the lane now sits ABOVE the fork at
these cells (587 vs 898 at pp-5k is the remaining read; the pp-5k cell
itself is the next bench's first row).

## Kernel-level A/B (`dense_f16_relaxed_ab`, ABBA pairs, median of pair-ratios, p=512 production shapes)

| shape | exact | relaxed | median ratio |
|---|---:|---:|---:|
| gdn.qkv 10240×5120 | 20.4 TFLOPS | 49.3 | 0.412 |
| gdn.z 6144×5120 | 18.1 | 44.9 | 0.404 |
| ssm.out/attn.o 5120×6144 | 17.4 | 41.5 | 0.417 |
| mlp.gate/up 17408×5120 | 22.1 | 52.5 | 0.415 |
| mlp.down 5120×17408 | 18.6 | 46.1 | 0.405 |
| attn.q 12288×5120 | 22.5 | 53.3 | 0.418 |

**2.4× at every production shape** — 17.4–22.5 → 41.5–53.3 TFLOPS. The
exact column also confirms the bench-939 arithmetic (~16-22 TFLOPS across
the GEMM buckets). Tile geometry stays pinned (bench 936: composed
optimum) — relaxed is a schedule change (fp16-rate MMA), not a tile change.

## Numerics class (disclosed)

Pre-registered unit band 3e-2 rel-to-mag (fp16-accumulate physics:
~2.8e-3 per 32-deep tile partial, tiles summing in f32). **Measured:
2.4e-4 on the small-shape classes, 3.2e-5 at the production column count**
— the drift sits at the exact arm's own noise floor (the exact gate bands
2e-4). Determinism pin: two dispatches, byte-identical (relaxed changes
precision, never reduction order). **The 512-token greedy walk is
BYTE-IDENTICAL across both arms** (md5 `0764d040f09e30264603e762ddcd7c1a`,
relaxed ×2 + exact ×1) — not a single argmax flip. This walk also
discharges the verdict's pending ≥512-token walk item. (The bench-939
64-token md5 `79adf65e…` is a different prompt/length — not comparable.)
The env stays promote-only; a lane-default flip is the GOAT read on this
record (same shape as bench 923's dist-scoped promotion: quality axis PASS,
owner call on the default).

## The wte_upload split (new diagnostic, landed in this commit)

The stage-timer's wte_upload read 283–452 ms (15–33% of a synchronized
pass) — now split: **gather 0–1 ms, write 1 ms, sync 282–441 ms**. The
wall is the `client.sync()` itself on an IDLE queue (a cubecl/wgpu sync
stall, variable 100–450 ms), not the upload. **The free-running lane pays
none of it**: the free-running pass (1.06 s) equals the GPU-work sum
(gdn 850 + attn 200 + tail 12 ms) — the per-stage syncs are
diagnostic-mode cost. Consequences: (a) wte_upload's share in EVERY
stage-timer record to date (938, 939, this) is instrument-inflated and
must be read as "sync stall at the boundary", not stage work; (b) the
next timer-mode improvement is replacing the per-stage `client.sync()`
with `wait_last`-class waits or GPU timestamp queries (diagnostic
instrument fix, not a lane fix); (c) wte's TRUE cost is ~2 ms/pass — no
lane lever there.

## What landed

- `gemm_dense_f16_metal_zero_copy.rs`: the selector
  (`set_dense_relaxed_precision` / `production_dense_relaxed_precision`,
  env `RIIR_QWEN38_DENSE_MM2D_RELAXED=1` exact-literal promote-only,
  parsed once + announced), `compose_dense_msl_source` (explicit
  `#define SWEEP_RELAXED` on BOTH branches — the bench-921 law), the
  `relaxed_precision` field + getter, `new_with_relaxed` (explicit
  posture; `new` delegates to the selector — parallel test threads never
  touch the global).
- Route: the armed line prints `relaxed_precision=` (engagement proof —
  the bench log asserts the arm it measured).
- G1 gate: `dense_f16_matmul2d_relaxed_g1` (same fixtures, 3e-2
  pre-registered band + the determinism pin); exact gates switched to
  `new_with_relaxed(false)` (env-immune). 3/3 pass. Route G1 3/3 pass.

Provenance: amo, M5 Ultra, AC power, load 2.6–3.5 (a sibling CPU-compile
cargo on the box throughout — GPU-exclusive at every cell, the sibling
holds no GPU work; ABBA-median absorbs the load noise at the kernel
cells); code at `bd297c7`; model
`~/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf`; logs `/tmp/i1150_{exact,relaxed}_arm.log`,
`/tmp/i1150_{exact,relaxed}_walk*.log`.

Session: zed-amo-i1150-relaxed@amo, 1793786916 → (fold at record commit).
