#!/usr/bin/env python3
"""Plan 623 T0.3/T0.4 — 1-box MLX incumbent bench over the Bonsai-2 2-bit pack.

Loads the hadamard pack through its bundled runtime (stock mlx-lm cannot load
it — PACK-RUNTIME.md), then measures prefill (pp) and decode (tg) throughput
on this box's MLX (Metal). Metrics mirror the llama.cpp solo rows of Bench 900
(pp512 / tg128 t/s) for same-plan comparability; QUANT POSTURES DIFFER
(2-bit affine+hadamard pack vs PQ2_0 GGUF) — disclosed, not equalized.

Greedy sampling (deterministic). Env:
  I623_PACK   pack dir (default ../bonsai2-mlx-2bit beside this repo's .raw)
  I623_PP     prefill tokens (default 512)
  I623_TG     decode tokens  (default 128)
  I623_OUT    optional JSON output path
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

# A non-UTF-8 console (cp874) would kill this script on its first
# non-ASCII glyph; degrade the glyph, never the verdict (katgpt-rs
# Issue 804). A captured stream without reconfigure is left alone.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parent.parent
PACK = Path(os.environ.get("I623_PACK", REPO / ".raw" / "bonsai2-mlx-2bit"))
PP = int(os.environ.get("I623_PP", "512"))
TG = int(os.environ.get("I623_TG", "128"))


def load():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plan623_pack_artifact import load_model  # noqa: E402  (bench-lane copy: schema {1,2})

    return load_model(str(PACK))


def encode_prompt(tok, n_tokens):
    # deterministic prompt: cycle the tokenizer's high-frequency ids
    vocab = tok.get_vocab()
    ids = [i for i in range(min(2000, len(vocab)))]
    if not ids:
        ids = [1]
    return [ids[i % len(ids)] for i in range(n_tokens)]


def main():
    t0 = time.perf_counter()
    model, config = load()
    tok_t0 = time.perf_counter()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(PACK / "tokenizer.json"))
    print(f"load: {tok_t0 - t0:.1f}s", flush=True)

    # ---- prefill (pp) ----
    prompt = mx.array([encode_prompt(tok, PP)], dtype=mx.int32)
    cache = model.make_cache()
    t0 = time.perf_counter()
    logits = model(prompt, cache=cache)
    mx.eval(logits)
    pp_s = time.perf_counter() - t0
    pp_tps = PP / pp_s

    # ---- decode (tg) ----
    token = mx.argmax(logits[:, -1, :], axis=-1)
    rows = []
    for i in range(TG):
        t0 = time.perf_counter()
        logits = model(token[:, None], cache=cache)
        mx.eval(logits)
        dt = time.perf_counter() - t0
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        rows.append(dt * 1e3)
    tg_tps = 1000.0 / statistics.median(rows)

    out = {
        "bench": "plan623_mlx_1box",
        "pack": str(PACK),
        "model_type": config.get("model_type"),
        "mlx_device": str(mx.default_device()),
        "pp_tokens": PP,
        "tg_tokens": TG,
        "pp_tps": round(pp_tps, 1),
        "pp_ms_total": round(pp_s * 1000, 1),
        "tg_tps_median": round(tg_tps, 1),
        "tg_ms_median": round(statistics.median(rows), 3),
        "tg_ms_min": round(min(rows), 3),
        "tg_ms_max": round(max(rows), 3),
    }
    print(json.dumps(out, indent=2))
    dst = os.environ.get("I623_OUT")
    if dst:
        Path(dst).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
