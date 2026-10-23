#!/bin/sh
# Fetch the FluidInference silero-vad CoreML bundle (`.issues/015` T2's
# model material) into the lane's models root. Weights never enter the
# repo — this lands under $RIIR_AUDIO_MODELS_DIR (default
# ~/.cache/riir-infer/audio), and the lane's in-code blake3-dir-v1 pin
# refuses any drift at load.
#
# After fetching, mint/verify the pin:
#   cargo run -p riir-infer-audio --features vad-ane \
#     --example audio_bundle_digest -- "$DEST/$BUNDLE"
set -eu

REPO="FluidInference/silero-vad-coreml"
REV="${SILERO_VAD_REV:-main}"
# Both postures (`.issues/015` G1/G1b): the published pre-compiled
# bundle (v6.2.1, posture A) AND the published .mlpackage source
# (v6.0.0, posture B — compiled locally through the host's cache).
BUNDLES="silero-vad-unified-256ms-v6.2.1.mlmodelc silero-vad-unified-256ms-v6.0.0.mlpackage"
DEST="${RIIR_AUDIO_MODELS_DIR:-$HOME/.cache/riir-infer/audio}"

for BUNDLE in $BUNDLES; do
mkdir -p "$DEST/$BUNDLE"

# The recursive tree listing is the file set (the bundle is a directory
# of files on HF).
curl -sL "https://huggingface.co/api/models/$REPO/tree/$REV/$BUNDLE?recursive=true" \
  -o "$DEST/.tree.json"

python3 - "$REPO" "$REV" "$BUNDLE" "$DEST" <<'PY'
import json, os, subprocess, sys
repo, rev, bundle, dest = sys.argv[1:5]
tree = json.load(open(os.path.join(dest, ".tree.json")))
files = [e["path"] for e in tree if e["type"] == "file"]
assert files, "empty tree listing — the API path changed; fix this script"
base = f"https://huggingface.co/{repo}/resolve/{rev}"
for rel in files:
    out = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(["curl", "-sL", "--fail", "-o", out, f"{base}/{rel}"], check=True)
    print("ok", rel)
PY

rm -f "$DEST/.tree.json"
echo "fetched: $DEST/$BUNDLE"
done
echo "verify the pins:"
for BUNDLE in $BUNDLES; do
  echo "  cargo run -p riir-infer-audio --features vad-ane --example audio_bundle_digest -- \"$DEST/$BUNDLE\""
done
