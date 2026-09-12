#!/usr/bin/env bash
# Download the TinyLlama-1.1B-Chat Q4_0 GGUF that 19-llm-engine runs.
#
#   ./scripts/fetch_model.sh              # to ~/models/
#   MODEL_DIR=/content/models ./scripts/fetch_model.sh
#
# 638 MB, Apache-2.0. Kept outside the repository on purpose: never commit
# weights, and never put them in a synced folder (OneDrive, Dropbox, Drive),
# which will upload them. The engine looks in ~/models by default, or wherever
# $CUDA_PORTFOLIO_MODEL points.
set -euo pipefail

NAME="tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/$NAME"
SHA256="da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556"
DIR="${MODEL_DIR:-$HOME/models}"

mkdir -p "$DIR"
dest="$DIR/$NAME"
if [[ -f "$dest" ]] && echo "$SHA256  $dest" | sha256sum -c --status; then
    echo "already present and verified: $dest"
    exit 0
fi

echo "downloading $NAME (638 MB) to $DIR"
curl -fL --retry 3 -o "$dest.part" "$URL"
echo "$SHA256  $dest.part" | sha256sum -c -
mv "$dest.part" "$dest"
echo "ok: $dest"
echo "export CUDA_PORTFOLIO_MODEL=$dest   # if not using ~/models"
