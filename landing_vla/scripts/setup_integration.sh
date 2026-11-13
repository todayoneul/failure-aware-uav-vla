#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${UAV_VLA_HOME:-"$HOME/uav-vla-smoke"}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVIDENCE="$REPO_ROOT/outputs/integration"
export UV_CACHE_DIR="$TASK_ROOT/cache"
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/python"
UV="$TASK_ROOT/tools/uv-x86_64-unknown-linux-gnu/uv"
mkdir -p "$EVIDENCE"
if [[ ! -x "$TASK_ROOT/integration/bin/python" ]]; then
  "$UV" venv --python 3.10.14 "$TASK_ROOT/integration"
fi
"$UV" pip install --python "$TASK_ROOT/integration/bin/python" -r "$REPO_ROOT/configs/integration-requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
"$UV" pip check --python "$TASK_ROOT/integration/bin/python"
"$UV" pip freeze --python "$TASK_ROOT/integration/bin/python" > "$EVIDENCE/integration-freeze.txt"
