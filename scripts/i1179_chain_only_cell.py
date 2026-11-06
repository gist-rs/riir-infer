#!/usr/bin/env python3
"""Issue 1179 T1 verdict condition (4) — the VENDOR CHAIN-ONLY acceptance cell.

Their engine (uzu 0.6.17) on THEIR target (trymirai/Qwen3.8-27B-M mirai-m),
with the speculator's shapes.json construction_method patched to `Argmax`
(their first-class chain form — dflash_tfm.rs at the pin: tree_budget-1
greedy-sampled chain rows, no Weaver tree). The resulting
tokens_per_forward_pass IS the vendor chain-only acceptance j.

Decides the T1 confound (verdict round 2, condition 4, gate unchanged):
  their chain-only j ~= 1  => the 1.08-1.10 we measure is the CHAIN-vs-WEAVER
                              gap (confound); the T1 gate moves to the weaver
                              form (T2/T3 scope).
  their chain-only j >= 2.5 => a residual bug on our side; keep bisecting.

Posture: greedy (their Greedy sampling), league fixture prompt, 3 passes +
1 warm-up, out cap 512 (acceptance stats don't need the 4096 cap). The
checkpoint is bound via model_by_path (the 0.6.17 registry-row-lag seam).
The Argmax shapes patch must be IN PLACE before this runs — see the sibling
patch_shapes.py (this script only VERIFIES the patch at load-time stats:
a Weaver run reports a weaver-shaped tokens_per_forward_pass distribution;
discriminator recorded per row via the shapes file blake3).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import time


def spec_dump(stats) -> dict:
    try:
        spec = stats.speculator_stats
        if spec is None:
            return {"present": False}
        return {
            "present": True,
            "num_decode_forward_passes": spec.num_decode_forward_passes,
            "tokens_per_forward_pass": round(spec.tokens_per_forward_pass, 3),
        }
    except Exception:
        return {"present": False}


async def amain() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--cap", type=int, default=512)
    args = ap.parse_args()

    fixture_text = open(args.fixture, encoding="utf-8").read()
    out = open(args.out, "a", encoding="utf-8")

    def emit(row) -> None:
        out.write(json.dumps(row) + "\n")
        out.flush()
        print("[row] " + json.dumps(row), flush=True)

    import uzu

    wheel = importlib.metadata.version("uzu")
    print(f"[meta] uzu wheel {wheel}", flush=True)

    cfg = uzu.EngineConfig.create().with_download_manager_type(
        uzu.DownloadManagerType.Universal
    )
    engine = await uzu.Engine.create(cfg)
    model = await engine.model_by_path(args.model_path)
    if model is None:
        print("REFUSE: model_by_path returned None", flush=True)
        return 5
    print(
        f"[meta] identifier={model.identifier} path={model.filesystem_path} "
        f"speculation_capable={model.is_speculation_capable}",
        flush=True,
    )

    instance = await engine.chat_instance(model, uzu.ChatConfig.create())

    for p in range(args.passes + 1):  # pass 0 = warm-up
        session = await engine.chat_with_instance(instance)
        if p == 0:
            cfg0 = uzu.ChatReplyConfig.create().with_token_limit(1)
            msg = uzu.ChatMessage.user().with_text(fixture_text)
            r = await session.reply([msg], cfg0)
            print(
                f"[warm] in_tokens={r[-1].stats.tokens_count_input}",
                flush=True,
            )
            continue
        sm = uzu.SamplingMethod.Greedy()
        cfg_r = (
            uzu.ChatReplyConfig.create()
            .with_token_limit(args.cap)
            .with_sampling_method(sm)
        )
        msg = uzu.ChatMessage.user().with_text(fixture_text)
        r = await session.reply([msg], cfg_r)
        s = r[-1].stats
        emit(
            {
                "leg": "vendor_chain_only",
                "pass": p,
                "in_tokens": s.tokens_count_input,
                "out_tokens": s.tokens_count_output,
                "gen_tok_s": round(s.generate_tokens_per_second, 1),
                "duration_s": round(s.duration, 2),
                "stopped_at": (
                    "limit" if s.tokens_count_output >= args.cap else "eos"
                ),
                "speculator": spec_dump(s),
                "engine": "uzu 0.6.17 chain-only (Argmax shapes patch)",
                "uzu_version": wheel,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
