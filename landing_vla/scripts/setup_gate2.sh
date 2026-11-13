#!/usr/bin/env bash
# Reproduce the isolated RPC client stack; no simulator/model downloads here.
set -euo pipefail
TASK_ROOT=${UAV_VLA_HOME:-"$HOME/uav-vla-smoke"}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVIDENCE="$REPO_ROOT/outputs/compatibility"
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/python"
export UV_CACHE_DIR="$TASK_ROOT/cache"
UV="$TASK_ROOT/tools/uv-x86_64-unknown-linux-gnu/uv"
"$UV" venv --python 3.10.14 "$TASK_ROOT/gate2"
"$UV" pip install --python "$TASK_ROOT/gate2/bin/python" \
  numpy==1.26.3 tornado==4.5.3 msgpack==1.1.2 setuptools==75.6.0 wheel==0.45.1
curl -fLsS -o "$TASK_ROOT/tools/msgpack-rpc-python-fix-msgpack-dep.zip" \
  https://raw.githubusercontent.com/XuPeng23/AeroVLA/2c5ae0987a484ab92f00dd9d9ed493cb3e98e492/msgpack-rpc-python-fix-msgpack-dep.zip
printf '%s  %s\n' c0d7df3fe91271ea052384ca7150c7f6730eeed63672168d08a0f27946322197 \
  "$TASK_ROOT/tools/msgpack-rpc-python-fix-msgpack-dep.zip" | sha256sum -c -
"$UV" pip install --python "$TASK_ROOT/gate2/bin/python" \
  "$TASK_ROOT/tools/msgpack-rpc-python-fix-msgpack-dep.zip"
"$UV" pip install --python "$TASK_ROOT/gate2/bin/python" \
  airsim==1.8.1 opencv-contrib-python==4.11.0.86 --no-build-isolation
"$TASK_ROOT/gate2/bin/python" "$REPO_ROOT/scripts/prepare_gate2_client.py"
"$UV" pip check --python "$TASK_ROOT/gate2/bin/python"
"$UV" pip freeze --python "$TASK_ROOT/gate2/bin/python" > "$EVIDENCE/gate2-freeze.txt"
