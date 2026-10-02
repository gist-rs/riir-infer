#!/usr/bin/env python3
"""Plan 617 A2 — OpenThai-SystemOne tower extraction surgery (4090 box).

`iapp/OpenThai-SystemOne` @ `5d04bcca` is a custom wrapper arch
(`OpenThaiSystemOneForDecision`, model_type `openthai_systemone`): the
Qwen3.5-0.8B GDN-hybrid tower + a 256-slot classification head
(`slot_head.weight/bias`) + `log_temperature` [3]. exllamav3 1.5.3 does not
know the wrapper arch but DOES know the tower (`Qwen3_5ForCausalLM`,
GatedDeltaNet + interval-4 full attention + interleaved attn output gate).
A1's pre-registered "GDN hybrid = likely converter failure mode" did NOT
fire at the arch level; this extraction is the remaining surgery.

Three deltas, each deliberate and recorded:
1. KEY RENAME: the checkpoint stores the tower under `model.*`
   (the wrapper's own module tree). exllamav3's Qwen3_5Model reads
   `model.language_model.*` (transformers-5.x nesting — note the
   Qwen3Next arch uses `model.*`, but ITS GDN keys are the fused
   in_proj_qkvz/in_proj_ba spelling; this checkpoint carries the
   separate in_proj_qkv/in_proj_z/in_proj_b/in_proj_a spellings that
   only the Qwen3_5 module set reads). Everything else byte-identical.
2. HEAD EXCLUSION: slot_head.weight/bias + log_temperature are NOT
   tower weights; they stay bf16 beside the pack for the hybrid server
   (A5). The extracted dir is a plain causal-LM checkpoint.
3. CONFIG REWRITE: text_config flattened to top level,
   architectures = ["Qwen3_5ForCausalLM"] so exllamav3 dispatches;
   tie_word_embeddings true — the source declares FALSE but ships NO
   lm_head tensor (the wrapper reads hidden states only; it never
   materializes the head). exllamav3 requires an lm_head module, so the
   extraction selects its alt-key path (head bound to embed_tokens —
   unused by the A5 head server either way; documented divergence);
   mtp_num_hidden_layers 0 — the source config declares 1 but ships NO
   MTP tensors (single-shot classification, no draft weights).

Verification arm: EVERY renamed tensor byte-compares against the source
(full compare, not sampled — 0.8B makes it free), the excluded set is
exactly the head trio, and the output safetensors is BLAKE3-recorded.

Usage:
  py scripts/plan617_extract_openthai_tower.py \
     --src  <snapshot dir with model.safetensors + config.json> \
     --out  <extracted tower dir> \
     --head-out <dir for slot_head + log_temperature bf16 safetensors>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import blake3
import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-out", required=True)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    head_out = Path(args.head_out)
    out.mkdir(parents=True, exist_ok=True)
    head_out.mkdir(parents=True, exist_ok=True)

    print(f"[extract] src: {src}")
    tensors = load_file(str(src / "model.safetensors"))
    n_src = len(tensors)
    print(f"[extract] source tensors: {n_src}")

    HEAD_KEYS = {"slot_head.weight", "slot_head.bias", "log_temperature"}
    missing = HEAD_KEYS - set(tensors)
    assert not missing, f"head-side keys absent from source: {missing}"

    # 1+2. rename tower, split head side
    tower: dict = {}
    head: dict = {}
    for k, v in tensors.items():
        if k in HEAD_KEYS:
            head[k] = v
        else:
            assert k.startswith("model."), f"unexpected non-tower key: {k}"
            tower["model.language_model." + k[len("model."):]] = v
    print(f"[extract] tower tensors: {len(tower)}  head-side: {len(head)}")
    assert len(tower) + len(head) == n_src

    # 3. config rewrite
    with open(src / "config.json", encoding="utf-8") as f:
        src_cfg = json.load(f)
    text = dict(src_cfg["text_config"])
    for drop in (
        "_name_or_path", "label2id", "id2label", "output_attentions",
        "output_hidden_states", "problem_type", "return_dict",
        "is_encoder_decoder", "use_cache",
    ):
        text.pop(drop, None)
    text["architectures"] = ["Qwen3_5ForCausalLM"]
    text["tie_word_embeddings"] = True  # documented divergence (see docstring)
    text["mtp_num_hidden_layers"] = 0  # no MTP tensors shipped (see docstring)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(text, f, indent=2, ensure_ascii=False)

    # tokenizer verbatim
    for name in ("tokenizer.json", "tokenizer_config.json"):
        s = src / name
        if s.exists():
            shutil.copyfile(s, out / name)
            print(f"[extract] copied {name}")

    # write outputs
    save_file(tower, str(out / "model.safetensors"), metadata={"format": "pt"})
    save_file(head, str(head_out / "openthai_slot_head.safetensors"),
              metadata={"format": "pt"})
    with open(head_out / "head_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "source": str(src),
            "source_config": src_cfg,
            "head_keys": sorted(head.keys()),
            "shapes": {k: list(v.shape) for k, v in head.items()},
            "note": "slot_head Linear(H=1024 -> 256 slots), log_temperature "
                    "[3] per-question-type; bf16 verbatim from the checkpoint; "
                    "abstain_slot=255, answer_token_id=248082 from wrapper cfg",
        }, f, indent=2)

    # verify: full byte-compare of every tower tensor against source
    back = load_file(str(out / "model.safetensors"))
    assert len(back) == len(tower)
    n_cmp = 0
    for k_src, v_src in tensors.items():
        if k_src in HEAD_KEYS:
            continue
        k_new = "model.language_model." + k_src[len("model."):]
        v_new = back[k_new]
        assert v_new.shape == v_src.shape and v_new.dtype == v_src.dtype, k_src
        assert v_new.view(torch.uint8).eq(v_src.view(torch.uint8)).all(), \
            f"byte mismatch: {k_src}"
        n_cmp += 1
    print(f"[extract] byte-verified {n_cmp}/{n_src - len(HEAD_KEYS)} tower tensors")

    # head-side verify
    hback = load_file(str(head_out / "openthai_slot_head.safetensors"))
    for k, v in head.items():
        assert hback[k].shape == v.shape and hback[k].dtype == v.dtype
    print(f"[extract] head-side tensors reloaded OK: {sorted(head)}")

    b3 = blake3.blake3((out / "model.safetensors").read_bytes()).hexdigest()
    hb3 = blake3.blake3(
        (head_out / "openthai_slot_head.safetensors").read_bytes()).hexdigest()
    print(f"[extract] BLAKE3 tower : {b3}")
    print(f"[extract] BLAKE3 head  : {hb3}")
    rec = {
        "source_snapshot": str(src),
        "extracted_dir": str(out),
        "tower_safetensors_blake3": b3,
        "head_safetensors_blake3": hb3,
        "tower_tensors": len(tower),
        "excluded": sorted(HEAD_KEYS),
        "renames": {"model.*": "model.language_model.*"},
        "config_divergences": [
            "architectures OpenThaiSystemOneForDecision -> Qwen3_5ForCausalLM",
            "tie_word_embeddings false -> true (no lm_head tensor shipped; "
            "exllamav3 alt-key path; head unused by the A5 server)",
            "mtp_num_hidden_layers 1 -> 0 (no MTP tensors shipped)",
        ],
    }
    with open(out / "EXTRACTION_RECORD.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print("[extract] EXTRACTION_RECORD.json written")


if __name__ == "__main__":
    main()
