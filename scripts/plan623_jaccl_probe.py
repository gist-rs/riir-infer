#!/usr/bin/env python3
"""Plan 623 Phase-3 gate — JACCL (RDMA over TB) probe, the D2-gated rows.

Same measurements as the ring probe (Bench 904 §4) on the jaccl backend:
all_sum at 4 B / 16 KiB / 10 KiB-hidden / 1 MiB / 64 MiB + all_gather 1 MiB,
plus the send/recv ping-pong the ring backend DEADLOCKED on (jaccl has real
p2p — if it works, the deadlock is ring-specific and the PP harness can ride
jaccl p2p instead of raw sockets).

Env (both ranks):
  MLX_RANK=0|1
  MLX_JACCL_COORDINATOR=192.168.10.1:50100   (rank0 listens, over TCP)
  MLX_IBV_DEVICES=/abs/ibv.json              ([[null,"rdma_en2"],["rdma_en2",null]])
  I623_PROBE_OUT=... (rank0)
"""

import json
import os
import statistics
import time

import mlx.core as mx

HIDDEN_F16_BYTES = 5120 * 2
SIZES = [
    ("4B", 1),
    ("16KiB", 4096),
    ("hidden10KiB", HIDDEN_F16_BYTES // 4),
    ("1MiB", 256 * 1024),
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
        xs.append((time.perf_counter() - t0) * 1e6)
    return xs


def stats(xs):
    return {
        "min_us": round(min(xs), 1),
        "median_us": round(statistics.median(xs), 1),
        "max_us": round(max(xs), 1),
    }


def main():
    world = mx.distributed.init(backend="jaccl")
    rank = world.rank()
    print(f"[rank{rank}] jaccl init ok size={world.size()}", flush=True)
    if world.size() != 2:
        raise SystemExit("expected 2 ranks")

    rows = []

    def announce(row):
        if rank == 0:
            print(json.dumps(row), flush=True)

    for name, n in SIZES:
        x = mx.ones(n) * (rank + 1.0)
        reps = REPS_BIG if n >= 16 * 1024 * 1024 else REPS_SMALL
        s = stats(timed(lambda: mx.distributed.all_sum(x, group=world), reps))
        gbps = (2 * n * 4 / 1e9) / (s["median_us"] / 1e6)
        row = {"op": "all_sum", "size": name, **s,
               "algo_gbps": round(gbps, 3) if n >= 256 * 1024 else None}
        rows.append(row)
        announce(row)
        del x

    x = mx.ones(256 * 1024) * (rank + 1.0)
    s = stats(timed(lambda: mx.distributed.all_gather(x, group=world), REPS_SMALL))
    row = {"op": "all_gather", "size": "1MiB", **s}
    rows.append(row)
    announce(row)
    del x

    # p2p ping-pong — the ring-deadlock re-test on jaccl
    x = mx.ones(2560, dtype=mx.bfloat16) * (rank + 1.0)  # 5 KiB, hidden-class
    try:
        def op():
            if rank == 0:
                mx.distributed.send(x, dst=1, group=world)
                return mx.distributed.recv((2560,), mx.bfloat16, src=1, group=world)
            r = mx.distributed.recv((2560,), mx.bfloat16, src=0, group=world)
            mx.distributed.send(r, dst=0, group=world)
            return r

        s = stats(timed(op, REPS_SMALL))
        row = {"op": "pp_rtt", "size": "5KiB-bf16", **s}
    except Exception as e:  # noqa: BLE001 — record the failure shape verbatim
        row = {"op": "pp_rtt", "size": "5KiB-bf16", "error": repr(e)[:200]}
    rows.append(row)
    announce(row)
    del x, y

    if rank == 0:
        out = {"probe": "plan623_jaccl_probe", "backend": "jaccl",
               "mlx": "0.32.0", "rows": rows}
        print(json.dumps(out, indent=2), flush=True)
        dst = os.environ.get("I623_PROBE_OUT")
        if dst:
            with open(dst, "w") as f:
                json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
