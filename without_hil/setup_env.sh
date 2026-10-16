#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if python3 -m venv "$SCRIPT_DIR/.venv" 2>/dev/null; then
  :
elif command -v virtualenv >/dev/null 2>&1; then
  echo "python3-venv/ensurepip 不可用，改用 virtualenv。"
  virtualenv "$SCRIPT_DIR/.venv"
else
  echo "无法创建 .venv：请安装 python3-venv 或 virtualenv。" >&2
  echo "采集本身仍可尝试使用 ./run_collect.sh 运行。" >&2
  exit 1
fi

"$SCRIPT_DIR/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
echo "采集环境已就绪：$SCRIPT_DIR/.venv"
echo "如需转换 LeRobot，请继续执行：$SCRIPT_DIR/.venv/bin/pip install -r $SCRIPT_DIR/requirements-convert.txt"
