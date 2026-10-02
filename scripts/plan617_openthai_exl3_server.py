#!/usr/bin/env python3
"""Plan 617 A5 — the OpenThai-EXL3 hybrid server (the cheap falsifier).

Serves the fp32 lane's EXACT wire (`GET /healthz`, `POST /v1/systemone`)
but the tower forward runs over the EXL3 pack (exllamav3) while the
slot-head half stays THEIR code verbatim:

- prompt rendering: their `openthai_systemone.formatting.Formatter`
  (imported from the reflex checkout — the rendering law lives there;
  byte-identical prompt by construction; `permutations=1` pinned, the
  084 disclosed-patch posture for the dtype/numerics statement)
- tower forward: exllamav3 `Model` over the pack, module loop STOPPING
  BEFORE the `logits_output`-capped lm_head (the final norm is the
  second-to-last module — `forward_ls`'s loop minus the head), then the
  hidden states gathered at `<|ts_answer|>` positions
- head: their slot-head math from `modeling.py` verbatim —
  `slot_head(h.to(head.dtype)).float()`, `log_temperature` per-question-
  type temperatures, the option-count validity mask with slot 255 =
  abstain, softmax over (k + abstain)
- decoding: their `SystemOneClient._decode_named` semantics verbatim —
  probabilities renormalized over the k options, `confidence_from_probs`,
  noul/choice/score answer shapes, per-question abstain probability

The reference this server is GATED against (A5's parity + retention bar)
is the fp32 server (`uvicorn openthai_systemone.server:app`,
`OPENTHAI_SYSTEMONE_DTYPE=float32`, `permutations=1`) — same wire, same
rendering, fp32 tower. Numerics differences come ONLY from the packed
tower.

Run (4090 box, inside the vcvars env — see .raw/exl3_serve.bat):
  py scripts/plan617_openthai_exl3_server.py \
     --pack  .raw/packs/openthai-tower-exl3-4.0bpw \
     --head  .raw/packs/openthai-head/openthai_slot_head.safetensors \
     --repo  E:/git/riir-reflex/.raw/openthai-systemone \
     --port  8010
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import ValidationError

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True)
ap.add_argument("--head", required=True)
ap.add_argument("--repo", required=True, help="the openthai-systemone checkout (formatting/types/modeling)")
ap.add_argument("--port", type=int, default=8010)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--model-name", default="openthai-systemone-exl3-4.0bpw")
ap.add_argument("--gpu-split", default=None, help="optional auto float, e.g. '24.0' GiB")
args = ap.parse_args()

REPO = Path(args.repo)
sys.path.insert(0, str(REPO))

from exllamav3 import Cache, Config, Model  # noqa: E402
from openthai_systemone.formatting import Formatter  # noqa: E402
from openthai_systemone.modeling import (  # noqa: E402
    QTYPE_INDEX,
    confidence_from_probs,
)
from openthai_systemone.types import (  # noqa: E402
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)
from safetensors.torch import load_file  # noqa: E402

# ── the tower: the EXL3 pack through exllamav3 ─────────────────────────────
print(f"[a5] loading pack: {args.pack}", flush=True)
config = Config.from_directory(str(args.pack))
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=32768, max_batch_size=1)
kwargs = {}
if args.gpu_split:
    kwargs["autosplit_reserve"] = [float(args.gpu_split) * (1 << 30)]
model.load(**kwargs)
print("[a5] pack loaded", flush=True)

# the head-side trio (bf16 verbatim from the checkpoint) + the wrapper
# keys (answer_token_id / abstain_slot) — the converter's config.json
# carries only the tower, so the wrapper keys come from the extraction's
# head_meta.json (the source config recorded verbatim)
head_dir = Path(args.head).parent
head_meta = json.loads((head_dir / "head_meta.json").read_text(encoding="utf-8"))
src_cfg = head_meta["source_config"]
head_t = load_file(args.head)
slot_head_w = head_t["slot_head.weight"]      # (256, H) bf16
slot_head_b = head_t["slot_head.bias"]        # (256,) bf16
log_temperature = head_t["log_temperature"].float()  # (3,) bf16 -> f32
ABSTAIN_SLOT = src_cfg["abstain_slot"]
ANSWER_ID = src_cfg["answer_token_id"]
assert ABSTAIN_SLOT == 255 and ANSWER_ID == 248082, "wrapper keys drifted"
# hidden-state extraction: the forward_ls module loop minus the head. The
# modules with the logits_output cap are last (lm_head); everything before
# it — including the final RMSNorm — produces the hidden states their
# wrapper reads (out.last_hidden_state).
head_modules = [m for m in model.modules if m.caps.get("logits_output")]
assert len(head_modules) == 1, f"expected exactly one logits_output module, got {len(head_modules)}"
tower_modules = [m for (m, _inst, _idx) in model.fwd_modules]
assert tower_modules[-1] is head_modules[0], "lm_head is not the last fwd module"

from transformers import AutoTokenizer  # noqa: E402

fmt = Formatter(AutoTokenizer.from_pretrained(str(args.pack)))
assert fmt.answer_id == ANSWER_ID, (
    f"tokenizer answer id {fmt.answer_id} != pack config {ANSWER_ID}"
)

DEVICE = "cuda:0"
slot_head_w = slot_head_w.to(DEVICE)
slot_head_b = slot_head_b.to(DEVICE)
log_temperature = log_temperature.to(DEVICE)


@torch.inference_mode()
def tower_hidden(input_ids: torch.Tensor) -> torch.Tensor:
    """(1, T, H) final-normed hidden states, the forward_ls loop minus head."""
    params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": 0,
        # batch_shape is the CACHE GEOMETRY (page-aligned ceiling), not the
        # runtime seq len (prepare_flash_attn asserts a PAGE_SIZE multiple)
        "batch_shape": (1, 32768),
    }
    x = model.prepare_inputs(input_ids, params)
    for m in tower_modules[:-1]:
        x = m.prepare_for_device(x, params)
        x = m.forward(x, params)
    # the single-shot request's recurrent slot is DONE — free it, or every
    # request leaks one of the cache's max_batch_size slots
    for rs in params.get("recurrent_states") or []:
        rs.free()
    return x


@torch.inference_mode()
def slot_logits(hidden: torch.Tensor, answer_positions, option_counts, qtypes):
    """Their slot_logits verbatim (modeling.py), on the gathered states."""
    idx = answer_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    h = torch.gather(hidden, 1, idx)  # (B, Q, H)
    logits = torch.nn.functional.linear(
        h.to(slot_head_w.dtype), slot_head_w, slot_head_b
    ).float()
    t = log_temperature.exp()[qtypes.clamp(min=0)].unsqueeze(-1)
    logits = logits / t
    ar = torch.arange(logits.shape[-1], device=logits.device)
    valid = ar[None, None, :] < option_counts.unsqueeze(-1)
    valid = valid.clone()
    valid[..., ABSTAIN_SLOT] = True
    valid[..., 0] |= option_counts.unsqueeze(-1).squeeze(-1) == 0
    return logits.masked_fill(~valid, float("-inf"))


def run_system_one(state, questions, permutations: int | None) -> SystemOneResponse:
    # their system_one_batch posture law: explicit n, else AUTO (8 cyclic
    # orders when any choice question has >= 11 options, else 1). The lane
    # wire sends neither field, so the fp32 reference (and the 084 pins)
    # ran AUTO — this server mirrors it cell-vs-cell.
    from openthai_systemone.types import Choice, parse_question

    pq = {k: parse_question(v) for k, v in questions.items()}
    choice_k = {qid: len(q.criteria) for qid, q in pq.items() if isinstance(q, Choice)}
    n = permutations
    if n is None:
        n = 8 if any(k >= 11 for k in choice_k.values()) else 1
    n = max(1, min(n, max(choice_k.values(), default=1)))

    # their _cyclic_orders: n distinct cyclic shifts, identity first
    def cyclic_orders(k: int, m: int) -> list[list[int]]:
        m = max(1, min(m, k))
        offsets = sorted({round(j * k / m) % k for j in range(m)})
        return [[(i + off) % k for i in range(k)] for off in offsets]

    orders = {qid: cyclic_orders(k, n) for qid, k in choice_k.items()}
    plans = [{qid: ords[j % len(ords)] for qid, ords in orders.items()}
             for j in range(n)]

    encs = [fmt.encode(state, pq, option_orders=oo) for oo in plans]
    per_q: dict[str, dict[str, float]] = {}
    per_q_abstain: dict[str, float] = {}
    for e in encs:
        ids = torch.tensor([e.input_ids], dtype=torch.long, device=DEVICE)
        hidden = tower_hidden(ids)
        Q = len(e.answer_positions)
        ap_t = torch.tensor([e.answer_positions], dtype=torch.long, device=DEVICE)
        oc_t = torch.tensor([e.option_counts], dtype=torch.long, device=DEVICE)
        qt_t = torch.tensor(
            [[QTYPE_INDEX[s.qtype] for s in e.specs]], dtype=torch.long, device=DEVICE
        )
        logits = slot_logits(hidden, ap_t, oc_t, qt_t)
        probs = logits.softmax(-1).float().cpu()
        for qi, spec in enumerate(e.specs):
            p = probs[0][qi]
            k = len(spec.option_names)
            pk = p[:k] / p[:k].sum().clamp(min=1e-12)
            d = per_q.setdefault(spec.qid, {})
            for name, v in zip(spec.option_names, pk.tolist()):
                d[name] = d.get(name, 0.0) + v / len(encs)
            per_q_abstain[spec.qid] = per_q_abstain.get(spec.qid, 0.0) + \
                float(p[ABSTAIN_SLOT]) / len(encs)

    # their _decode_named, from the base (identity-permutation) encoding
    base = encs[0]
    answers = {}
    for spec in base.specs:
        d = per_q[spec.qid]
        k = len(spec.option_names)
        names = spec.option_names if spec.qtype != "choice" else list(d.keys())
        pk = torch.tensor([d[nm] for nm in names], dtype=torch.float32)
        pk = pk / pk.sum().clamp(min=1e-12)
        conf = confidence_from_probs(pk, k)
        if spec.qtype == "noul":
            answers[spec.qid] = {"noul": float(d["yes"])}
        elif spec.qtype == "choice":
            prob_map = {nm: float(v) for nm, v in zip(names, pk.tolist())}
            best = max(prob_map, key=prob_map.get)
            answers[spec.qid] = {
                "choice": best,
                "probabilities": prob_map,
                "confidence": conf,
                "abstain": per_q_abstain.get(spec.qid),
            }
        else:
            answers[spec.qid] = {
                "score": float((pk * torch.arange(k, dtype=torch.float32)).sum()),
                "legend": {i: (spec.option_descs[i] or str(i)) for i in range(k)},
                "probabilities": {str(i): float(pk[i]) for i in range(k)},
                "confidence": conf,
            }
    return SystemOneResponse(
        model=args.model_name,
        answers=answers,
        usage=Usage(input_tokens=base.n_tokens, permutations=n),
    )


app = FastAPI(title="OpenThai-SystemOne-EXL3", version="0.1.0")


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": args.model_name, "pack": str(args.pack)}


@app.post("/v1/systemone")
def system_one(req: dict):
    try:
        r = SystemOneRequest(**req)
    except ValidationError as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail=e.errors())
    resp = run_system_one(r.state, r.questions, r.permutations)
    return json.loads(resp.model_dump_json())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
