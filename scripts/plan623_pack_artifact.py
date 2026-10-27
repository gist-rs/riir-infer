"""Plan 623 bench loader — the pack's bundled artifact.py with ONE adaptation:
the schema gate accepts {1, 2}. The Bonsai-2 pack ships config schema_version 2
(additive metadata: hadamard_config / tensor_namespace / gdn_activation_layout /
components / vision keys) while its bundled reader only accepts 1; the loader
reads text_config + modules, unchanged between v1 and v2 (record shapes
verified identical, 402 modules). The PACK itself stays byte-pristine
(files.json sha256 24/24) — this copy lives outside it, in the bench lane."""

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from mlx import nn

# Packed arrives from the PACK's runtime dir at load_model time (path passed in);
# the module-level import in the bundled original cannot resolve from the bench
# lane, so load_model inserts the pack's runtime/ on sys.path first.


def save_model(model, info, tokenizer_path, directory):
    directory = Path(directory)
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, Packed):
            if module.dtype != mx.float16:
                raise ValueError("Only float16 packed artifacts are supported")
            modules.append(
                {
                    "path": name,
                    "block": module.block,
                    "embedding": module.embedding,
                    "dtype": "float16",
                }
            )
    config = {
        "schema_version": 1,
        "model_type": "prism_hadamard_qwen35",
        "requires_runtime": "examples/python/hadamard/artifact.py",
        "text_config": info["config"],
        "modules": modules,
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
    }
    directory.mkdir(parents=True, exist_ok=False)
    mx.save_safetensors(
        str(directory / "model.safetensors"),
        dict(tree_flatten(model.parameters())),
        metadata={"format": "mlx"},
    )
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.copyfile(tokenizer_path, directory / "tokenizer.json")
    (directory / "README.md").write_text(
        "# Experimental folded MLX pack\n\nRequires the accompanying Hadamard-aware loader. Do not load as an ordinary affine Qwen checkpoint: that would omit activation and inverse-embedding transforms. The source numerical and performance reports are separate from serialization integrity.\n", encoding="utf-8"
    )
    return config


def validate_record(original, record, arrays, signs):
    if not isinstance(original, (nn.Linear, nn.Embedding)):
        raise ValueError("Unsupported packed module target")
    if record["embedding"] != isinstance(original, nn.Embedding):
        raise ValueError("Packed module kind mismatch")
    rows, width = original.weight.shape
    if width % 128:
        raise ValueError("Invalid packed width")
    expected = [(rows, width // 16), (rows, width // 128), (rows, width // 128)]
    if [a.shape for a in arrays] != expected or arrays[0].dtype != mx.uint32:
        raise ValueError("Invalid packed tensor shapes or storage dtype")
    for array in arrays[1:]:
        if array.dtype not in (mx.float16, mx.float32, mx.bfloat16):
            raise ValueError("Invalid affine dtype")
        if not mx.all(mx.isfinite(array)).item():
            raise ValueError("Non-finite affine parameters")
    block = record["block"]
    if block:
        if width % block or signs is None or signs.shape != (width,):
            raise ValueError("Invalid transform dimensions")
        if not mx.all((signs == 1) | (signs == -1)).item():
            raise ValueError("Invalid sign values")
    elif signs is not None:
        raise ValueError("Unexpected sign vector")


def load_model(directory):
    directory = Path(directory)
    runtime_dir = str(directory / "runtime")
    if runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)
    from runtime import Packed as _Packed
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("schema_version") not in (1, 2)
        or config.get("model_type") != "prism_hadamard_qwen35"
    ):
        raise ValueError("Unsupported packed model schema")
    model = TextModel(TextModelArgs.from_dict(config["text_config"]))
    raw = mx.load(str(directory / "model.safetensors"))
    # v2 packers serialize the VL-style Model (language_model.* prefix); the
    # v1 reader targets the bare TextModel namespace. Strip the prefix —
    # every non-prefixed key must exist exactly once after the strip.
    stripped = {
        (k[len("language_model."):] if k.startswith("language_model.") else k): v
        for k, v in raw.items()
    }
    dup = len(stripped) != len(raw)
    if dup:
        raise ValueError("language_model prefix strip collided — ambiguous keys")
    weights = stripped
    seen = set()
    for record in config["modules"]:
        if record["path"] in seen:
            raise ValueError("Duplicate packed module")
        seen.add(record["path"])
        parts = record["path"].split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        name = record["path"]
        arrays = [
            weights[name + "." + suffix] for suffix in ("weight", "scales", "biases")
        ]
        if record["dtype"] != "float16":
            raise ValueError("Unsupported activation dtype")
        block = record["block"]
        if block and block not in (512, 1024, 2048, 4096):
            raise ValueError("Unsupported block size")
        signs = weights.get(name + ".signs")
        if block and signs is None:
            raise ValueError("Missing sign vector")
        original = getattr(parent, parts[-1])
        validate_record(original, record, arrays, signs)
        setattr(
            parent,
            parts[-1],
            _Packed(arrays, block, signs, record["embedding"], mx.float16),
        )
    text_keys = [
        (k, v)
        for k, v in weights.items()
        if not k.startswith("vision_tower.")
    ]
    model.load_weights(text_keys, strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, config


def save_and_verify(model, info, tokenizer_path, directory, input_ids):
    import numpy as np

    config = save_model(model, info, tokenizer_path, directory)
    reloaded, _ = load_model(directory)
    x = mx.array([input_ids], dtype=mx.int32)

    def evaluate(m):
        hidden = m.model(x, cache=m.make_cache())
        y = m.lm_head(hidden[:, -1:, :])
        mx.eval(y)
        return np.asarray(y)

    expected = evaluate(model)
    actual = evaluate(reloaded)
    np.testing.assert_array_equal(actual, expected)
    result = {
        "reload_logits_exact": True,
        "checked_logits": actual.size,
        "packed_modules": len(config["modules"]),
    }
    Path(directory, "reload-validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
