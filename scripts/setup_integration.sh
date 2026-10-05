#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/home/gyuhan/uav-vla-smoke
EVIDENCE=/mnt/c/Users/leegy/Desktop/drone/outputs/integration
export UV_CACHE_DIR="$TASK_ROOT/cache"
UV="$TASK_ROOT/tools/uv-x86_64-unknown-linux-gnu/uv"
"$UV" pip freeze --python "$TASK_ROOT/gate1/bin/python" > "$EVIDENCE/gate1-reused-pins.txt"
"$UV" pip install --python "$TASK_ROOT/integration/bin/python" -r "$EVIDENCE/gate1-reused-pins.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
"$UV" pip install --python "$TASK_ROOT/integration/bin/python" -c "$EVIDENCE/gate1-reused-pins.txt" \
  timm==0.9.10 scipy==1.15.3 opencv-python==4.11.0.86 projectairsim==1.0.2
"$UV" pip check --python "$TASK_ROOT/integration/bin/python"
"$UV" pip freeze --python "$TASK_ROOT/integration/bin/python" > "$EVIDENCE/integration-freeze.txt"
