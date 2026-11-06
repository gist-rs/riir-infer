# Bench 951 — Issue 1178 T2a: the Issue-609 factored Q4_K GEMV routed into the qwen38 decode path — measured FLAT on M5 Ultra

**Status:** RECORD — price cell NEGATIVE (no transfer); feature stays opt-in `q4k_factored_decode_gemv`, never joins production (2026-11-06)

## The cell

T0's distill verdict (Research 378) named the decode deficit class; T2a priced the cheapest mechanism candidate: the decode dispatch site (`QuantWeight::Q4.gemv` in `qwen38_dense_cubecl.rs`) was pinned to the pre-609 batched-plane wrapper (`GemvQ4KBatchedCubeCL` at batch=1) while the Issue-609 factored kernel — lanes pinned to one sub-block, scales lifted out of the element loop, dequant folded into the dot, measured **3.27–3.37× over plane on M3 Metal** (Issue 609 T2) — sat in-tree behind `q4k_rowtiled_gemv`, which the `production` aggregate never joined and the batched wrapper never routed to.

- Arm: opt-in feature `q4k_factored_decode_gemv` (implies `q4k_rowtiled_gemv`); routes the batch-1 decode GEMV through `GemvQ4KCubeCL::launch` (auto-selecting). Flag-off path byte-identical (the same batched call). Commit `c9258157`.

## Result (bmo M5 Ultra, worktree `~/git/riir-infer-i1179wt` @ `c9258157`, release builds OFF/ON from the same tree, sequential legs, GGUF `ce14c955…` = the Bench-1167 same-file pin)

| cell | OFF (batched-plane) | ON (factored via auto-select) |
|---|---|---|
| decode 512 steps, short prompt | **12.06** / 12.06 tok/s | **12.06** / 12.06 tok/s |
| decode @ctx1344 | 10.89 tok/s | 10.87 tok/s |
| pp1536 | 920 tok/s median | 909 tok/s median (drift-band) |
| G1 greedy walk (64 tokens) | `md5 94e5afc27140abc5abccac5dc0848c9c` | **identical md5** — digit-exact |

**Verdict: FLAT.** The 3.27–3.37× M3-class win does not transfer to M5 Ultra at the decode shapes (m=34816 × n=5120 and 5120×34816-class sites). Decode unchanged at 12.06 tok/s; prefill delta inside the cross-process drift band.

## Why (the mechanism reading, for the next candidate)

The M3 win was a lane-utilization/ALU-amortization class on a smaller GPU. On M5 Ultra the FFN GEMV arithmetic reads: ~19.2 GB of Q4_K weight bytes per decode step (64 layers × gate+up+down ≈ 300 MB/layer), ffn share 47.5 ms (Bench 950) ⇒ **~404 GB/s already achieved — ~49% of the ~819 GB/s class** — while uzu's serial-equivalent extracts ~552-570 GB/s (~67-70%). The remaining ~1.4-1.7× is not reachable by kernel SHAPE at the same weight layout: the 144-byte Q4_K block access pattern (d/dmin + 3 scale words + 128 packed bytes per 256 elements, scales re-decoded per row) has an ALU + cache-line amplification cost the factored geometry already minimizes. The next rungs are LAYOUT-class, per uzu's actual law (lane-sliced W4 weight packing at load) or the payload-class f16 lever — **the latter is 1177-item-2's lease (amo, to 2026-11-07), not re-run here**.

## Dispositions

- `q4k_factored_decode_gemv` stays opt-in, OUT of `production` (T3 bar not met — no gain to gate). The feature remains the replication path for the routed variant; deleting it would orphan the route.
- The decode-gap ledger now reads: T2a shape-class FLAT (this record); host floor 0.4% (Bench 950 — sync/serialization class closed at this posture, K-i1 prices near-zero microbench-side, serving-loop pricing still owed if the serving loop ever differs); remaining priced candidates = weight-layout class (lane-sliced repack at load — NEW mechanism, needs its own price cell) + f16 payload class (1177-item-2, amo's lease).

## Raw

bmo `/tmp/t2a_{OFF,ON}_{decode,decode_ctx,pp,walk}.log`; binaries `/tmp/t2a_bench_{OFF,ON}` (same tree, feature-diff only). Session: zed-m3-i1178t2.
