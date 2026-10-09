#!/usr/bin/env python3
"""Plan 617 A5 probe — parity harness for the EXL3 hybrid server.

Fires ONE identical request at BOTH servers (the fp32 reference and the
EXL3 hybrid) and prints per-question deltas: top-1 agreement, p drift
(abs + L1), confidence delta, abstain delta. This is the small-N
functional gate the board run stands on (the pre-registered retention
bar applies to the full 17-suite board, not this probe).

Usage:
  py scripts/plan617_a5_parity_probe.py --n 6 \
     --fp32 http://127.0.0.1:8000 --exl3 http://127.0.0.1:8011 \
     --pool E:/git/riir-reflex/.raw/datasets
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

# Keep this instrument's verdict printable on a non-UTF-8 console
# (katgpt-rs Issue 804 / the 928 drift census): it prints non-ASCII glyphs,
# and print() raises UnicodeEncodeError on e.g. cp874 — the process then dies
# with NO verdict. backslashreplace degrades the glyph visibly and keeps
# ASCII exact, so a verdict line stays greppable. Best-effort: a detached or
# captured stream is left alone rather than made fatal at import.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

# probe cases: real states pulled from suite dumps (deterministic pick by
# seed), one suite per question TYPE so all three decoders are exercised
SUITE_PICKS = [
    ("banking77", "choice"),
    ("ag_news", "choice"),
    ("emotion", "choice"),
    ("thai_wisesight", "choice"),
    ("sst5", "score"),
    ("xnli_en", "noul"),
]

GENERIC_Q = {
    "choice": ("Classify this item into exactly one category.",
               {"a": "the first category fits best",
                "b": "the second category fits best",
                "c": "neither of the two fits"}),
    "score": ("Rate the item on the given scale.",
              [str(i) for i in range(5)]),
    "noul": ("Decide whether the claim about the item is true or false.",
             None),
}


def text_of(row: dict, feats: list[dict]) -> str:
    for f in feats:
        if f.get("type", {}).get("dtype") == "string":
            v = row.get(f["name"])
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def pick_cases(pool: Path, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    for suite, qkind in SUITE_PICKS:
        d = pool / suite
        files = [f for f in sorted(d.glob("train-*.json"))] or [f for f in sorted(d.glob("*.json")) if f.name != "splits.json"]
        rows = []
        for f in files[:2]:
            with open(f, encoding="utf-8") as fh:
                djs = json.load(fh)
            feats = djs.get("features", [])
            rows += [text_of(r.get("row", {}), feats) for r in djs.get("rows", [])]
        rows = [r for r in rows if r]
        for i in range(max(1, n // len(SUITE_PICKS))):
            cases.append({
                "suite": suite, "qkind": qkind,
                "state": rows[rng.randrange(len(rows))][:1200],
            })
    return cases


def question_for(kind: str) -> dict:
    instr, crit = GENERIC_Q[kind]
    if kind == "choice":
        return {"type": "choice", "instructions": instr, "criteria": crit}
    if kind == "score":
        return {"type": "score", "instructions": instr, "criteria": crit}
    return {"type": "noul", "instructions": instr}


def ask(url: str, state: str, qkind: str) -> dict:
    body = json.dumps({
        "state": state,
        "model": "probe",
        "questions": {"q0": question_for(qkind)},
        "permutations": 1,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default="http://127.0.0.1:8000")
    ap.add_argument("--exl3", default="http://127.0.0.1:8011")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n", type=int, default=6, help="total probe cases")
    ap.add_argument("--seed", type=int, default=617)
    args = ap.parse_args()

    cases = pick_cases(Path(args.pool), args.n, args.seed)
    agree = 0
    total = 0
    l1s = []
    print(f"{'suite':16s} {'kind':7s} {'agree':5s} {'maxd':>9s} {'L1':>9s}  fp32_top -> exl3_top")
    for c in cases:
        try:
            a = ask(args.fp32, c["state"], c["qkind"])
            b = ask(args.exl3, c["state"], c["qkind"])
        except Exception as e:
            print(f"{c['suite']:16s} {c['qkind']:7s} ERROR {e}")
            continue
        pa = a["answers"]["q0"]
        pb = b["answers"]["q0"]
        if c["qkind"] == "noul":
            va, vb = pa["noul"], pb["noul"]
            ta = "yes" if va >= 0.5 else "no"
            tb = "yes" if vb >= 0.5 else "no"
            d = abs(va - vb); l1 = d
        else:
            dist_a = pa["probabilities"]; dist_b = pb["probabilities"]
            keys = sorted(dist_a)
            va_ = [dist_a[k] for k in keys]; vb_ = [dist_b[k] for k in keys]
            ta = max(dist_a, key=dist_a.get); tb = max(dist_b, key=dist_b.get)
            d = max(abs(x - y) for x, y in zip(va_, vb_))
            l1 = sum(abs(x - y) for x, y in zip(va_, vb_))
        ok = ta == tb
        agree += ok
        total += 1
        l1s.append(l1)
        print(f"{c['suite']:16s} {c['qkind']:7s} {str(ok):5s} {d:9.5f} {l1:9.5f}  {ta:>8s} -> {tb}")
    print(f"\nprobes: {total}, top-1 agree: {agree}/{total}, "
          f"mean L1: {sum(l1s)/max(1,len(l1s)):.5f}")


if __name__ == "__main__":
    main()
