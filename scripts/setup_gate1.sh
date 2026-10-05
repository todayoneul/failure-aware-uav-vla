#!/usr/bin/env bash
# Isolated smoke-test runtime; never modifies the system Python or shell profile.
set -euo pipefail
TASK_ROOT=${UAV_VLA_HOME:-"$HOME/uav-vla-smoke"}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVIDENCE="$REPO_ROOT/outputs/compatibility"
mkdir -p "$TASK_ROOT/tools" "$TASK_ROOT/python" "$TASK_ROOT/cache" "$EVIDENCE"
exec > >(tee "$EVIDENCE/gate1-install.log") 2>&1
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/python"
export UV_CACHE_DIR="$TASK_ROOT/cache"
curl -fL --retry 2 -o "$TASK_ROOT/tools/uv.tar.gz" \
  https://github.com/astral-sh/uv/releases/download/0.12.23/uv-x86_64-unknown-linux-gnu.tar.gz
printf '%s  %s\n' 9167d72b3319674b6303c4cbe071854bba13ebdf3d76b1a7cbdc175471fb66d6 "$TASK_ROOT/tools/uv.tar.gz" | sha256sum -c -
tar -xzf "$TASK_ROOT/tools/uv.tar.gz" -C "$TASK_ROOT/tools"
UV="$TASK_ROOT/tools/uv-x86_64-unknown-linux-gnu/uv"
"$UV" --version
"$UV" python install 3.10.14
"$UV" venv --python 3.10.14 "$TASK_ROOT/gate1"
"$UV" pip install --python "$TASK_ROOT/gate1/bin/python" \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$TASK_ROOT/gate1/bin/python" \
  transformers==4.42.4 accelerate==0.32.1 peft==0.11.1 bitsandbytes==0.48.2 \
  tokenizers==0.19.1 huggingface-hub==0.23.5 safetensors==0.4.5 numpy==1.26.3
"$UV" pip check --python "$TASK_ROOT/gate1/bin/python"
"$UV" pip freeze --python "$TASK_ROOT/gate1/bin/python" > "$EVIDENCE/gate1-freeze.txt"
"$TASK_ROOT/gate1/bin/python" --version
du -sh "$TASK_ROOT/gate1" "$TASK_ROOT/python" "$TASK_ROOT/cache"
