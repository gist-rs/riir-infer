#!/usr/bin/env python3
"""Plan 623 TP-cell comm diagnostic — why is in-context all_sum ~210 us when
the Bench-905 probe said 19-23 us for the same payload class?

Arms (all on the exact cell shape (1,1,5120) f16):
  A  back-to-back all_sum, probe style (pipelined)
  B  all_sum with a full mx.eval + sleep spin between reps (synchronized,
     cold pipeline — the decode-step shape)
  C  all_sum with interleaved dummy GPU matmul on the SAME stream (the
     in-context shape: GPU work between comms)
  D  all_sum on the 3-D (1,64,5120) prefill-class shape
"""
import json
import os
import statistics
import time

import mlx.core as mx

import sys
# A non-UTF-8 console (cp874) would kill this script on its first
# non-ASCII glyph; degrade the glyph, never the verdict (katgpt-rs
# Issue 804). A captured stream without reconfigure is left alone.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

REPS = 30


def timeit(fn, reps=REPS, sync_each=False, interleave=None):
    xs = []
    for _ in range(3):
        mx.eval(fn())
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        if interleave is not None:
            out = out + interleave  # same-stream GPU dep between comms
        mx.eval(out)
        xs.append((time.perf_counter() - t0) * 1e6)
        if sync_each:
            time.sleep(0.0005)
    return {
        "min_us": round(min(xs), 1),
        "median_us": round(statistics.median(xs), 1),
        "max_us": round(max(xs), 1),
    }


def main():
    world = mx.distributed.init(backend="jaccl")
    rank = world.rank()
    print(f"[rank{rank}] jaccl init ok size={world.size()}", flush=True)

    x = mx.ones((1, 1, 5120), dtype=mx.float16) * (rank + 1.0)
    big = mx.ones((1, 64, 5120), dtype=mx.float16) * (rank + 1.0)
    dummy_a = mx.ones((512, 5120), dtype=mx.float16)
    dummy_w = mx.ones((5120, 5120), dtype=mx.float16) * 0.001

    rows = {}
    rows["A_pipelined_1x5120_f16"] = timeit(lambda: mx.distributed.all_sum(x, group=world))
    rows["B_synced_1x5120_f16"] = timeit(lambda: mx.distributed.all_sum(x, group=world), sync_each=True)
    rows["C_interleaved_gpu_1x5120_f16"] = timeit(
        lambda: mx.distributed.all_sum(x, group=world), interleave=dummy_a @ dummy_w
    )
    rows["D_pipelined_1x64x5120_f16"] = timeit(lambda: mx.distributed.all_sum(big, group=world))

    if rank == 0:
        out = {"probe": "plan623_tp_comm_diag", "rows": rows}
        print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
