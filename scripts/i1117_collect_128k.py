#!/usr/bin/env python3
"""Issue 1117 T4 - 128K axis collection helper.

The gate (src/bin/flashmemory_decode_gate.rs, arm persistence since f98e921)
persists each arm to <out>/<name>.bin (dense.bin, rho_0.15.bin, rho_0.30.bin)
and, at completion, writes report.md + report.json.  This helper INDEPENDENTLY
recomputes every reported statistic from the bins - the gate's report layer
already had one p95 defect (016272b), so a record-bearing read never trusts it
alone - and prints the 32K-vs-128K comparison table for the bench record.

Bin layout (write_arm_bin, little-endian):
  magic[4]="FMG1" | inputs_digest[32] | expected_steps u64 | complete u8 |
  n_written u64 | mean_selected_fraction f64 | wall_secs f64 |
  n_written x (nll f32 | argmax u64)

The 32K reference (compare baseline) is parsed from 32k_rel/report.json; the
two runs use DIFFERENT corpora (corpus.txt vs corpus_128k.txt - verified not a
prefix extension), so the comparison is same-regime aggregate, never
paired-token.

Usage:
  python3 scripts/i1117_collect_128k.py                 # full collection read
  python3 scripts/i1117_collect_128k.py --selftest      # synthetic-bin verification

Exit 0 = all integrity checks passed; exit 1 = any check failed (including
"run not complete yet" - wait for the gate exit line in run_128k_v2.log).
"""
import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

MAGIC = b"FMG1"
HEADER = struct.Struct("<4s32sQBQdd")  # magic, digest, steps, complete, n, sel, wall
HEADER_SIZE = 69
RECORD = struct.Struct("<fQ")


def read_bin(path):
    b = Path(path).read_bytes()
    if len(b) < HEADER_SIZE:
        raise ValueError("truncated header: %s" % path)
    magic, digest, steps, complete, n, sel, wall = HEADER.unpack(b[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError("bad magic: %s" % path)
    nll = [0.0] * n
    argmax = [0] * n
    off = HEADER_SIZE
    for i in range(n):
        if off + RECORD.size > len(b):
            raise ValueError("truncated records at %d: %s" % (i, path))
        f, a = RECORD.unpack_from(b, off)
        nll[i] = f
        argmax[i] = a
        off += RECORD.size
    return {
        "digest": digest.hex(),
        "steps": steps,
        "complete": bool(complete),
        "n": n,
        "sel": sel,
        "wall": wall,
        "nll": nll,
        "argmax": argmax,
    }


def percentile_p95(values):
    """Nearest-rank p95 + tail support (mirrors percentile_with_support)."""
    s = sorted(values)
    n = len(s)
    idx = max(1, math.ceil(0.95 * n)) - 1
    return s[idx], n - idx


def delta_stats(arm, dense, lo, hi):
    d = [a - b for a, b in zip(arm["nll"][lo:hi], dense["nll"][lo:hi])]
    agree = sum(1 for a, b in zip(arm["argmax"][lo:hi], dense["argmax"][lo:hi]) if a == b)
    n = len(d)
    mean = math.fsum(d) / n
    p95, support = percentile_p95(d)
    mx = max(d)
    return {
        "n": n,
        "mean": mean,
        "p95": p95,
        "support": support,
        "max": mx,
        "agree_pct": 100.0 * agree / n,
    }


def bucket_windows(compare_from, bucket, steps):
    """Mirror of the gate's per-bucket walk (alignment from compare_from)."""
    bstart = compare_from - compare_from % bucket
    out = []
    while bstart < steps:
        bend = min(bstart + bucket, steps)
        lo = max(bstart, compare_from)
        if bend > lo:
            out.append((bstart, bend, lo, bend))
        bstart += bucket
    return out


def close(a, b, tol=5e-4):
    return abs(a - b) <= tol


def selftest():
    tmp = Path("/tmp/i1117_collect_selftest_%d" % os.getpid())
    tmp.mkdir(exist_ok=True)
    steps = 32
    digest = bytes(range(32))
    for name, sel, wall in (("dense", 1.0, 4.0), ("rho_0.15", 0.15, 3.0)):
        nll = [float(i) * 0.25 for i in range(steps)]
        argmax = [i % 7 for i in range(steps)]
        buf = HEADER.pack(MAGIC, digest, steps, 1, steps, sel, wall)
        for f, a in zip(nll, argmax):
            buf += RECORD.pack(f, a)
        (tmp / (name + ".bin")).write_bytes(buf)
    d = read_bin(tmp / "dense.bin")
    r = read_bin(tmp / "rho_0.15.bin")
    assert d["n"] == steps and d["complete"] and d["sel"] == 1.0
    assert d["digest"] == digest.hex()
    st = delta_stats(r, d, 8, 32)
    # both arms carry the same synthetic arrays -> deltas all zero
    assert close(st["mean"], 0.0, 1e-6), st
    assert close(st["max"], 0.0, 1e-6)
    assert st["n"] == 24
    assert st["support"] == 24 - (math.ceil(0.95 * 24) - 1)
    # nearest-rank sanity on a real spread: ceil(0.95*100)-1 = 94
    spread = [float(i) for i in range(100)]
    fake_arm = {"nll": spread, "argmax": [0] * 100}
    fake_dense = {"nll": [0.0] * 100, "argmax": [1] * 100}
    st2 = delta_stats(fake_arm, fake_dense, 0, 100)
    assert close(st2["mean"], 49.5, 1e-9)
    assert st2["p95"] == 94.0 and st2["support"] == 6, st2
    print("selftest OK (parse + stats + nearest-rank support)")
    return 0


def load_report_json(out_dir):
    p = Path(out_dir) / "report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / ".cache/i1117/128k_rel"))
    ap.add_argument("--ref", default=str(Path.home() / ".cache/i1117/32k_rel"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    out = Path(args.out)
    ref = load_report_json(args.ref)
    if ref is None:
        print("missing 32K reference report.json at %s" % args.ref)
        return 1

    fails = []
    names = ["dense", "rho_0.15", "rho_0.30"]
    bins = {}
    for name in names:
        p = out / (name + ".bin")
        if not p.exists():
            fails.append("missing bin %s" % p)
            continue
        try:
            bins[name] = read_bin(p)
        except ValueError as e:
            fails.append(str(e))
    if fails:
        for f in fails:
            print("FAIL: %s" % f)
        print("run not complete - wait for the gate exit line in run_128k_v2.log")
        return 1

    steps = bins["dense"]["steps"]
    for name in names:
        b = bins[name]
        if not b["complete"]:
            fails.append("%s complete=0 (in-flight prefix)" % name)
        if b["n"] != steps or b["steps"] != steps:
            fails.append("%s n=%d steps=%d expected %d" % (name, b["n"], b["steps"], steps))
        if b["digest"] != bins["dense"]["digest"]:
            fails.append("%s digest differs from dense (inputs changed mid-run?)" % name)
    if not close(bins["dense"]["sel"], 1.0, 1e-9):
        fails.append("dense sel frac != 1.0")

    if fails:
        for f in fails:
            print("FAIL: %s" % f)
        return 1

    cfg = load_report_json(out) or {}
    compare_from = cfg.get("compare_from", 0)
    bucket = cfg.get("bucket", 8192)
    print("== integrity: all 3 bins complete, digest-consistent, n=%d, compare_from=%d, bucket=%d" %
          (steps, compare_from, bucket))

    dense = bins["dense"]
    print()
    print("== 128K arms (independent recompute; gate report cross-checked below)")
    print("| arm | sel frac | tok/s | pooled dNLL mean | dNLL p95 (support) | dNLL max | top-1 agree |")
    print("|---|---|---|---|---|---|---|")
    recomputed = {}
    for name in names[1:]:
        rho = name.split("_")[1]
        st = delta_stats(bins[name], dense, compare_from, steps)
        tps = steps / bins[name]["wall"]
        recomputed[name] = st
        print("| rho %s | %.4f | %.1f | %+.4f | %+.4f (%d) | %+.4f | %.2f%% |" %
              (rho, bins[name]["sel"], tps, st["mean"], st["p95"], st["support"],
               st["max"], st["agree_pct"]))
    dense_pooled = math.fsum(dense["nll"][compare_from:]) / (steps - compare_from)
    print("\nDense reference: %.1f tok/s; pooled NLL mean %.4f" %
          (steps / dense["wall"], dense_pooled))

    # Cross-check vs the gate's own report (the layer that had the p95 defect).
    gate = cfg.get("arms") if cfg else None
    if gate:
        print("\n== cross-check vs gate report.json")
        ok = True
        by_rho = {("%.2f" % a["rho"]): a for a in gate}
        for name in names[1:]:
            rho = name.split("_")[1]
            g = by_rho.get(rho)
            if g is None:
                print("  %s: no gate row" % name)
                continue
            st = recomputed[name]
            checks = [
                ("mean", st["mean"], g["delta_nll_mean"]),
                ("p95", st["p95"], g["delta_nll_p95"]),
                ("max", st["max"], g["delta_nll_max"]),
                ("agree", st["agree_pct"], g["top1_agreement_pct"]),
            ]
            for label, mine, theirs in checks:
                if not close(mine, theirs, 2e-3):
                    ok = False
                    print("  MISMATCH %s %s: recomputed %.6f vs gate %.6f" %
                          (name, label, mine, theirs))
        print("  gate cross-check: %s" % ("OK" if ok else "FAILED"))
        if not ok:
            fails.append("gate report cross-check mismatch")

    # The 32K-vs-128K comparison for the record.
    print("\n== 32K vs 128K (different corpora - same-regime comparison, NOT paired-token)")
    ref_dtps = ref["dense_tok_per_s"]
    print("| arm | ctx | sel frac | tok/s | x dense | dNLL mean | top-1 |")
    print("|---|---|---|---|---|---|---|")
    print("| dense | 32K | 1.0 | %.1f | 1.00x | - | - |" % ref_dtps)
    for a in ref["arms"]:
        print("| rho %.2f | 32K | %.4f | %.1f | %.2fx | %+.4f | %.2f%% |" %
              (a["rho"], a["mean_selected_fraction"], a["tok_per_s"],
               a["tok_per_s"] / ref_dtps, a["delta_nll_mean"], a["top1_agreement_pct"]))
    dtps = steps / dense["wall"]
    print("| dense | 128K | 1.0 | %.1f | 1.00x | %.4f (pooled NLL) | - |" % (dtps, dense_pooled))
    for name in names[1:]:
        rho = name.split("_")[1]
        b = bins[name]
        tps = steps / b["wall"]
        st = recomputed[name]
        print("| rho %s | 128K | %.4f | %.1f | %.2fx | %+.4f | %.2f%% |" %
              (rho, b["sel"], tps, tps / dtps, st["mean"], st["agree_pct"]))

    # Per-bucket trend at 128K (position-dependence of the budget arm).
    print("\n== per-bucket at 128K (width %d, from compare_from %d)" % (bucket, compare_from))
    print("| bucket | arm | dNLL mean | dNLL p95 (support) | dNLL max | top-1 |")
    print("|---|---|---|---|---|---|")
    for bs, be, lo, hi in bucket_windows(compare_from, bucket, steps):
        for name in names[1:]:
            st = delta_stats(bins[name], dense, lo, hi)
            print("| [%d, %d) | rho %s | %+.4f | %+.4f (%d) | %+.4f | %.2f%% |" %
                  (bs, be, name.split("_")[1], st["mean"], st["p95"],
                   st["support"], st["max"], st["agree_pct"]))

    if fails:
        for f in fails:
            print("FAIL: %s" % f)
        return 1
    print("\nALL INTEGRITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
