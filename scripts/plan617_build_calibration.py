#!/usr/bin/env python3
"""Plan 617 A3 — calibration mix for the OpenThai tower EXL3 convert.

The plan requires the calibration mix RECORDED (per-suite calibration
sensitivity is the lossy-law exposure): thai_wisesight + thai_sib200 + the
EN dataset suites, in recorded proportions. exllamav3's converter takes a
packed calibration FILE (-cd/--cal_data): a safetensors with one
"input_ids" tensor of shape [rows, cols] (default 250 x 2048) in the
model's own vocab. This script builds it from the reflex canonical dataset
pool (riir-reflex/.raw/datasets — HF datasets-server /rows JSON dumps).

Mix (of the rows the converter will use): the two thai suites get 50%
combined (the consumer's primary domain), the EN suites share the other
50% in proportion to their row availability — the exact proportions are
printed and pinned into the record. Text field per suite is resolved from
the dump's features (first string feature; wisesight=sibs use "texts").

Usage:
  py scripts/plan617_build_calibration.py \
     --pool E:/git/riir-reflex/.raw/datasets \
     --tokenizer E:/git/riir-infer/.raw/hf/openthai-tower-qwen35-0.8b/tokenizer.json \
     --out E:/git/riir-infer/.raw/packs/openthai_cal.safetensors \
     [--rows 250] [--cols 2048]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

# thai first (consumer domain), then the EN board suites (the 9 dataset
# suites the reflex board publishes; code_fixtures is code-shaped and the
# harness families are synthetic — both excluded from the cal mix).
SUITES = [
    ("thai_wisesight", 3.0),
    ("thai_sib200", 2.0),
    ("ag_news", 1.0),
    ("banking77", 1.0),
    ("emotion", 1.0),
    ("massive_intent_en", 1.0),
    ("prompt_injections", 1.0),
    ("sst5", 1.0),
    ("xnli_en", 1.0),
]


def rows_text(suite: Path) -> list[str]:
    texts: list[str] = []
    for f in sorted(suite.glob("*.json")):
        if f.name == "splits.json":
            continue
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        feats = d.get("features", [])
        str_feats = [x["name"] for x in feats
                     if x.get("type", {}).get("dtype") == "string"]
        if not str_feats:
            continue
        for r in d.get("rows", []):
            row = r.get("row", {})
            for name in str_feats:
                v = row.get(name)
                if isinstance(v, str) and v.strip():
                    texts.append(v.strip())
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=250)
    ap.add_argument("--cols", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=617)
    args = ap.parse_args()

    pool = Path(args.pool)
    tok = Tokenizer.from_file(args.tokenizer)
    rng = random.Random(args.seed)

    # gather + weight
    per_suite: dict[str, list[str]] = {}
    weights: dict[str, float] = {}
    for name, w in SUITES:
        d = pool / name
        if not d.exists():
            print(f"[cal] WARNING: suite dir missing, skipped: {name}")
            continue
        t = rows_text(d)
        if not t:
            print(f"[cal] WARNING: no text rows for {name}")
            continue
        per_suite[name] = t
        weights[name] = w
        print(f"[cal] {name}: {len(t)} texts, weight {w}")

    total_w = sum(weights.values())
    # desired allocation by weight, then cap at each suite's real packing
    # capacity and redistribute the deficit to suites with headroom
    # (sib200's pool is small; the recorded allocation is the availability-
    # capped reality, printed + pinned in the record)
    cap: dict[str, int] = {}
    for name in weights:
        n_tok = sum(len(tok.encode(t).ids) for t in per_suite[name])
        cap[name] = n_tok // args.cols
        if cap[name] < 1:
            cap[name] = 1
    desired = {n: args.rows * w / total_w for n, w in weights.items()}
    alloc = {n: min(int(desired[n]) + (1 if desired[n] % 1 >= 0.5 else 0),
                    cap[n])
             for n in weights}
    for _ in range(4 * args.rows):
        drift = args.rows - sum(alloc.values())
        if drift == 0:
            break
        room = {n: cap[n] - alloc[n] for n in weights if cap[n] > alloc[n]}
        if not room:
            break
        wsum = sum(weights[n] for n in room)
        gave = 0
        for n in sorted(room, key=lambda n: -weights[n]):
            add = min(room[n], max(1, int(drift * weights[n] / wsum + 0.5)))
            add = min(add, drift - gave) if drift > 0 else max(add, drift)
            take = min(room[n], abs(drift))
            if drift > 0:
                take = min(room[n], drift)
                alloc[n] += take
                gave += take
            else:
                take = min(alloc[n], -drift)
                alloc[n] -= take
                gave += take
            if gave >= abs(drift):
                break
    assert sum(alloc.values()) == args.rows, \
        f"cannot reach {args.rows} rows: capacity {sum(min(alloc[n], cap[n]) for n in alloc)}"
    print("[cal] row allocation (desired -> capped):",
          {n: (round(desired[n], 1), alloc[n]) for n in alloc})

    all_rows: list[list[int]] = []
    for name, n_rows in alloc.items():
        texts = per_suite[name]
        rng.shuffle(texts)
        # stream texts through the tokenizer until n_rows rows are full
        buf: list[int] = []
        got = 0
        for t in texts:
            ids = tok.encode(t).ids
            buf.extend(ids)
            while len(buf) >= args.cols:
                all_rows.append(buf[:args.cols])
                buf = buf[args.cols:]
                got += 1
                if got >= n_rows:
                    break
            if got >= n_rows:
                break
        print(f"[cal] {name}: packed {got}/{n_rows} rows")
        assert got == n_rows, f"{name}: only {got}/{n_rows} rows (pool too small)"

    rng.shuffle(all_rows)
    assert len(all_rows) == args.rows
    import torch
    t = torch.tensor(all_rows, dtype=torch.long)
    save_file({"input_ids": t}, args.out, metadata={"format": "pt"})
    import blake3
    b3 = blake3.blake3(Path(args.out).read_bytes()).hexdigest()
    rec = {
        "out": str(args.out),
        "blake3": b3,
        "rows": args.rows,
        "cols": args.cols,
        "seed": args.seed,
        "pool": str(pool),
        "allocation": alloc,
        "weights": weights,
        "suites": sorted(per_suite),
        "note": "thai 50% (wisesight 3 : sib200 2), EN suites share 50% "
                "equally; text-only rows from the reflex canonical pool; "
                "tokenized with the model's own tokenizer; rows shuffled "
                "before the convert consumes them",
    }
    with open(Path(args.out).with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print(f"[cal] wrote {args.out}  shape {tuple(t.shape)}  BLAKE3 {b3}")


if __name__ == "__main__":
    main()
