# Bench 906 — Plan 623 T4.1/T4.2: the first real TP cell (two half-model ranks over JACCL)

**Date:** 2026-10-27 (amo rank0 + bmo rank1, TB5 cable, jaccl/RDMA)
**Session:** zed-amo-tp2
**Plan:** [623](../.plans/623_multi_host_pipeline_inference_lane.md) Phase 4
**Model:** `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @ `fcba37d21` (same pack as
Bench 904/905; the custom `plan623_pack_artifact` loader)
**Provenance:** quiet window on both boxes (amo elephant capture ended 04:18Z;
no GPU consumers on either box at run time, ps-verified).

## 0. Verdict line

**TP sharding CORRECT, TP decode REFUTED.** The T4.1/T4.2 cell — column/row
sharding of the pack runtime, `all_sum` after every row-parallel matmul over
jaccl — produces **token-identical output to the unsharded forward** (six
independent produce paths, one 16-token pin) and is replay-deterministic.
But single-stream decode reads **18.2 t/s vs 50.8 (amo 1-box) / 58.1 (bmo
1-box)** = −64/−69%: the Phase-4 arithmetic gate was computed on the
**pipelined** all_sum rate (16–22 µs, Bench 905) while the TP dependency
chain pays the **synchronized per-exchange** cost (**183–277 µs measured**).
128 serialized exchanges/token ≈ +27 ms. Prefill is parity (−1..−9%) —
bulk exchanges amortize. A second, independent negative: **even with comms
free, halving every big matmul yields only −11..−14% decode** — batch-1
decode here is launch-bound, not weight-bandwidth-bound, so the TP memory
thesis does not apply on this model/box class at all.

## 1. The cell (scripts/plan623_mlx_tp2_bench.py)

Both ranks run ALL 64 layers; no socket hop, no token exchange (greedy
argmax over identical full logits on both ranks). One code path, `I623_TP_N`
= 1 (no-comms sanity arm through the same harness) or 2.

| weight | split | comms |
|---|---|---|
| q,k,v (+ fused per-head output gate), mlp gate,up, GDN in_proj_qkv/z | column (out-dim) | none (activations full or naturally local) |
| o_proj, mlp down_proj, GDN out_proj | row (in-dim) | `all_sum` of f16 partials |
| lm_head | column over vocab (124160/rank) | `all_gather` halves in fixed rank order == vocab order |
| embed (inverse-fwht lookup), all RMSNorms, q/k norms, GDN norm/dt_bias/A_log/conv1d | replicated (conv channels follow the head slices) | none |

Design laws the cell pins:

- **Hadamard block-locality** (runtime `fwht` reshapes `(-1, block)` +
  elementwise signflip): an in-dim weight slice is exact iff the slice is
  block-aligned (1024). o_proj/out_proj take 3072 = 3 blocks ✓. The MLP
  intermediate 17408 = **17** blocks has an odd count — the even 8704/8704
  split would cut block 8 across ranks (a partial block cannot be
  transformed independently), so the split is **block-aligned uneven:
  9216 + 8192** (disclosed; ~6% MLP FLOP imbalance).
- **GDN stored layout** (pack-time `vperm`) is key-head-grouped, so every
  GDN rank slice (qkv segments, z, b/a, dt_bias, A_log, conv channels,
  out_proj in-dim) is a contiguous stored-space range.
- **Determinism (Issue 940):** a 2-rank `all_sum` is ONE pairwise add —
  IEEE-commutative, order-free deterministic; `all_gather` concatenates in
  fixed rank order; greedy argmax = first-index over identical full logits.
  For N > 2 the reduction order would need pinning — this cell ships
  n ∈ {1, 2} only.
- The GDN module's own `sharding_group`/`sum_gradients` hooks are left
  unset (training-plane); the allreduce lives in the row wrapper.

## 2. Correctness — the G1 half

`gen_head` (first 16 greedy tokens at PP=512) is **identical** across:

1. the STOCK 1-box forward (mlx-lm modules, unmodified),
2. harness N=1 (amo) — same harness, full widths, no comms,
3. TP-2 run 1, rank0 AND rank1,
4. TP-2 run 2, rank0 AND rank1 (replay determinism),
5. the socket-PP harness of Bench 904 (rank1 head) — a third independent
   implementation landing on the same stream.

Pin: `[29, 197, 197, 1, 197, 197, 1, 197, 197, 1, 197, 197, 1, 197, 197, 1]`
(the ramp prompt is degenerate by construction; the pin is for equality,
not prose). Bit-identity of logits is NOT claimed (row-mat reduction order
changes); the claim is token-exactness + replay determinism, which is what
Plan 623's G1 asks of TP.

## 3. Rows (tg128 median, PP=512 prompt)

| arm | pp512 t/s | tg ms | tg t/s | box(es) |
|---|---|---|---|---|
| stock 1-box (Bench 904, cited) | 357.3 | 17.217 | **58.1** | bmo |
| harness N=1 (this bench) | 335.2 | 19.677 | **50.8** | amo |
| TP-2 run 1 | 324.0/324.4 | 54.91/54.97 | **18.2** | amo/bmo |
| TP-2 run 2 | 350.3/354.0 | 54.94/54.98 | **18.2** | amo/bmo |
| TP-2 smoke (PP64/TG8) | — | 42.8/42.3 | — | amo/bmo |
| TP-2 COMM_OFF diag (PP64/TG8, wrong tokens, timing only) | — | 16.6/15.3 | — | amo/bmo |
| harness N=1 (PP64/TG8) | — | 19.3 | 51.8 | amo |
| socket-PP re-read, quiet window (Bench 904: 23.24 ms / 43.0) | — | 22.1 | 45.2 | bmo half |

## 4. The comm diagnosis — why 128 × ~20 µs became 27 ms

`scripts/plan623_tp_comm_diag.py`, the exact decode payload shape
`(1,1,5120)` f16, jaccl, 30 reps:

| arm | median | reading |
|---|---|---|
| A — back-to-back (probe style, pipelined) | **21.9 µs** | reproduces Bench 905's 16–23 µs — that number is the PIPELINED per-op rate |
| B — synchronized (eval + cold wait each rep) | **183.2 µs** | the honest per-exchange latency: ~8.4× the pipelined rate |
| C — interleaved with independent GPU work (the in-context shape) | **277.4 µs** | the jaccl op blocks the stream; overlap does not materialize for free |
| D — pipelined prefill-class (1,64,5120) | 147.4 µs | bulk amortizes — why TP prefill lands at parity |

**The Bench-905 Phase-4 paper gate multiplied 2·L by the pipelined RTT.**
The cell's arithmetic: 2 row mats × 64 layers = 128 exchanges, strictly
serialized by the residual dependency (mlp needs the full attn sum), plus
one lm_head gather → 128 × 183–277 µs ≈ 23–35 ms/token, matching the
measured comm share (54.9 − 15.3 floor ≈ 27–40 ms). The T4.2 "overlap the
o_proj allreduce with the MLP column matmuls" lever is UNATTAINABLE at the
mlx level — the distributed op blocks the stream (arm C); it would need
our own transport's p2p on a side thread, which is exactly the Phase-3
lane (and mlx p2p itself is deadlocked, Bench 904 §4 + 905).

## 5. The launch-bound finding (independent of comms)

COMM_OFF floor vs N=1, same posture: **bmo 17.2 → 15.3 ms, amo 19.3 →
16.6 ms**. Halving ~90% of the weight bytes buys only 11–14% of the step:
batch-1 decode on this model/box class is **launch-bound** (~240 µs fixed
per layer × 64 ≈ 15 ms of floor), not weight-bandwidth-bound. The TP
memory-halving thesis therefore does not apply at batch-1 even with free
comms. (Same fight as the CUDA decode lane's fusion/graph rungs —
Bench 1004/1008.)

## 6. T0.3 discharges riding this bench (amo quiet window)

- **amo MLX 1-box row:** pp 335.2 / tg 50.8 t/s (19.68 ms) — the Bench-904
  §6 amo deferral is closed; amo is the slower box, bmo's 58.1 stays the
  1-box reference.
- **Clean-window socket-PP re-read:** 22.1 ms / 45.2 t/s vs Bench 904's
  23.24 / 43.0 — within ~5%; the −22% PP decode loss vs bmo 1-box HOLDS in
  the quiet window (not a load artifact).

## 7. Scoreboard (T4.3) + dispositions

| lane | single-stream decode vs bmo 1-box 58.1 |
|---|---|
| llama.cpp-RPC 2-box PP (Bench 900) | 48.6 t/s (−16%) |
| MLX socket 2-box PP (Bench 904/§6) | 43.0–45.2 t/s (−22..−26%) |
| **MLX TP-2 jaccl (this bench)** | **18.2 t/s (−69%)** |
| exo TP-jaccl claim ("up to 1.8×") | **not reproducible on mlx 0.32.0 collectives — reads 0.31× here** |
| llama.cpp `-sm tensor` | unsupported over RPC (Bench 900) |

**Phase-4 disposition: REFUTED for single-stream decode.** The cell is the
deliverable (T4.1/T4.2 built, correct, measured); TP stays a bench-lane
harness, not a serving posture. Plan 623's multi-host single-stream win
path is unchanged: PP with a 1-hop exchange (Phase 2) over the RDMA
transport (Phase 3) + speculative fill (Phase 5). Batch/prefill-class TP
is a possible future cell (bulk amortizes) but is not this lane's goal.

## 8. Upstream filing (E — paste-ready; GitHub auth unavailable on amo)

Owner act: paste to `ml-explore/mlx` issues (zero riir IP; repro scripts
below are self-contained). Cross-ref: #3149 (jaccl consecutive/different-
shape send/recv — ours is the SIMPLER pattern deadlocking on BOTH backends,
and ring reportedly "works" there).

```markdown
**Title:** [BUG] 2-node mx.distributed send/recv (p2p) deadlocks on BOTH
ring and jaccl backends while all collectives work (0.32.0, macOS 26.x,
TB5 RDMA)

**Describe the bug**
On a 2-node setup (the minimal world), `mx.distributed.send`/`recv`
deadlock on BOTH the `ring` and `jaccl` backends, while every collective
(all_sum, all_gather) works on both. With `ring`: both ranks post
(send posted / recv posted), zero bytes cross the wire (netstat queues
empty on all four socket ends), both ranks wait forever in
`mx.core.array::wait()`. With `jaccl`: both ranks spin in
`MeshImpl::recv` -> `tbt_poll_qp_recv` (libthunderboltrdma) with no error
and no delivery. The shared shape (both ranks wedged in recv, nothing on
the wire) points at the distributed-layer p2p path above the transports.

**To Reproduce** (2 nodes, env: MLX_RANK per node + your distributed
config; for jaccl we used MLX_JACCL_COORDINATOR + MLX_IBV_DEVICES)

```python
import os
import mlx.core as mx

rank = int(os.environ.get("MLX_RANK", "0"))
world = mx.distributed.init(backend="ring")  # jaccl deadlocks identically
print(f"[rank{rank}] init ok size={world.size()}", flush=True)

x = mx.ones(4) * (rank + 1.0)
mx.eval(x)

if rank == 0:
    mx.distributed.send(x, dst=1, group=world)
    y = mx.distributed.recv((4,), mx.float32, src=1, group=world)
    mx.eval(y)
    print("[rank0] done", y)
else:
    y = mx.distributed.recv((4,), mx.float32, src=0, group=world)
    mx.eval(y)
    mx.distributed.send(y, dst=0, group=world)
    print("[rank1] done")
```

Collectives on the same 2-node world work on both backends (all_sum 4 B
~16-22 us jaccl / ~325 us ring; 64 MiB 18.5 GB/s jaccl / 3.7 GB/s ring).
The ring backend's own comment (ring.cpp, recv checks sockets in the
opposite order of send "so that they work even with 2 nodes where left
and right is the same neighbor") documents the intended 2-node pairing —
consistent on paper, dead in practice at 0.32.0.

**Expected behavior** send/recv completes on a 2-node world on both
backends.

**Desktop:** macOS 26.x (BuildVersion 26A434 identical on both nodes);
M5 Ultra x2, one Thunderbolt-5 cable (RDMA enabled via rdma_ctl, jaccl
initialized with MLX_JACCL_COORDINATOR + MLX_IBV_DEVICES over rdma_en2);
MLX 0.32.0. Related: #3149 (jaccl consecutive/different-shape p2p wrong
data or hangs; ring reportedly OK there — for us even the single
same-shape exchange deadlocks on ring).
```

## 9. What remains

- Owner: paste the upstream issue (auth); the draft is §8.
- Phase 3 (own verbs transport) proceeds with real numbers; mlx p2p is
  NOT a viable transport for the T4.2-overlap idea or for PP (raw sockets
  stay, ~0.1–0.3 ms of the PP step).
- TP re-open triggers (recorded, not scheduled): a transport with
  side-thread p2p (enables the overlap lever), or a batch/prefill serving
  posture (bulk amortizes, §4 arm D), or N > 2 with pinned reduction order.

## Artifacts

- `906_plan623_t41_tp2_cell.json` — all runs + comm diag + verdicts
- `scripts/plan623_mlx_tp2_bench.py` (rsync twin on bmo) ·
  `scripts/plan623_tp_comm_diag.py` (twin)
- Environment: jaccl env trio as Bench 905; no IP surgery; both boxes idle.
