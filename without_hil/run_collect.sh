#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
SYSTEM_PYTHON="/usr/bin/python3"

export LD_LIBRARY_PATH="/userdata/update/manual_update/lib/galbot_moma/python_env/lib:/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/targets/aarch64-linux/lib:/userdata/lib:/data/galbot/lib"
export PYTHONPATH="/data/galbot/lib${PYTHONPATH:+:$PYTHONPATH}"

check_runtime() {
  "$1" -c 'import yaml, numpy, galbot_sdk.g1' >/dev/null 2>&1
}

if [[ -x "$VENV_PYTHON" ]] && check_runtime "$VENV_PYTHON"; then
  PYTHON="$VENV_PYTHON"
elif [[ -x "$SYSTEM_PYTHON" ]] && check_runtime "$SYSTEM_PYTHON"; then
  PYTHON="$SYSTEM_PYTHON"
  if [[ -x "$VENV_PYTHON" ]]; then
    echo "警告：.venv 不完整，采集将使用 /usr/bin/python3。" >&2
  fi
else
  echo "未找到可用的采集 Python（需要 yaml、numpy 和 galbot_sdk.g1）。" >&2
  echo "采集程序通常可直接使用 /usr/bin/python3，无需虚拟环境。" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
exec "$PYTHON" collector.py "$@"
