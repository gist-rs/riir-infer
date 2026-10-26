#!/usr/bin/env python3
"""Plan 623 T0.4 — MLX mx.distributed ring 2-box probe (amo + bmo, TB bridge).

Mirrors the link_probe protocol (4 B / 16 KiB / 1 MiB / 64 MiB) plus the
PP-shaped hop (one qwen3.5-27B hidden vector, 5120 x f16 = 10 KiB) so the
MLX-substrate numbers sit beside Bench 896/900/901 rows.

Run one rank per host (rank 0 on amo, rank 1 on bmo):
  # both hosts, same absolute paths (venv + hostfile are rsync twins):
  MLX_RANK=0 MLX_HOSTFILE=/abs/hostfile.json <venv>/bin/python plan623_mlx_distributed_probe.py
  MLX_RANK=1 MLX_HOSTFILE=/abs/hostfile.json <venv>/bin/python plan623_mlx_distributed_probe.py

hostfile.json (ring backend, one cable):
  [["192.168.10.1:50101"], ["192.168.10.2:50102"]]
"""

import json
import os
import statistics
import sys
import time

import mlx.core as mx

HIDDEN_F16_BYTES = 5120 * 2  # one qwen3.5-27B hidden state, bf16 on the wire
SIZES = [
    ("4B", 1),
    ("16KiB", 4096),
    ("1MiB", 256 * 1024),
    ("hidden10KiB", HIDDEN_F16_BYTES // 4),  # f32 elems, same wire bytes
    ("64MiB", 16 * 1024 * 1024),
]
WARMUP = 5
REPS_SMALL = 50
REPS_BIG = 10


def timed(fn, reps):
    xs = []
    for _ in range(WARMUP):
        mx.eval(fn())
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        xs.append((time.perf_counter() - t0) * 1e6)  # us
    return xs


def stats(xs):
    return {
        "min_us": round(min(xs), 1),
        "median_us": round(statistics.median(xs), 1),
        "max_us": round(max(xs), 1),
    }


def main():
    world = mx.distributed.init(backend="ring")
    rank, size = world.rank(), world.size()
    if size != 2:
        print(f"FATAL: expected 2 ranks, got {size}", file=sys.stderr)
        sys.exit(1)
    other = 1 - rank

    rows = []

    def announce(msg):
        print(f"[rank{rank}] {msg}", flush=True)

    # all_sum at each size (collective latency; 64MiB row doubles as BW class)
    for name, n in SIZES:
        announce(f"op all_sum {name}")
        x = mx.ones(n) * (rank + 1.0)
        reps = REPS_BIG if n >= 16 * 1024 * 1024 else REPS_SMALL

        def op():
            return mx.distributed.all_sum(x, group=world)

        s = stats(timed(op, reps))
        gbps = (2 * n * 4 / 1e9) / (s["median_us"] / 1e6)  # count both directions' worth
        row = {"op": "all_sum", "size": name, **s,
               "algo_gbps": round(gbps, 3) if n >= 256 * 1024 else None}
        rows.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
        del x

    # all_gather at 1MiB/rank (the TP-shaped exchange)
    announce("op all_gather 1MiB")
    x = mx.ones(256 * 1024) * (rank + 1.0)

    def gop():
        return mx.distributed.all_gather(x, group=world)

    s = stats(timed(gop, REPS_SMALL))
    row = {"op": "all_gather", "size": "1MiB", **s}
    rows.append(row)
    if rank == 0:
        print(json.dumps(row), flush=True)
    del x

    # point-to-point ping-pong: DEADLOCKED on this topology in mlx 0.32.0
    # (posted send+recv, zero bytes cross the wire, both ranks wait forever in
    # array::wait). Verdict row records it; the measurement is skipped.
    row = {"op": "pp_rtt", "size": "SKIPPED",
           "verdict": "mlx-0.32.0-ring-p2p-deadlocked-2node"}
    rows.append(row)
    if rank == 0:
        print(json.dumps(row), flush=True)

    if rank == 0:
        out = {
            "probe": "plan623_mlx_distributed_probe",
            "backend": "ring",
            "world_size": size,
            "mlx": mx.__version__ if hasattr(mx, "__version__") else "0.32.0",
            "rows": rows,
        }
        print(json.dumps(out, indent=2))
        dst = os.environ.get("I623_PROBE_OUT")
        if dst:
            with open(dst, "w") as f:
                json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
