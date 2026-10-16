#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"

# The PolicyServer is reached directly.  gRPC otherwise inherits desktop proxy
# variables and may try to tunnel this connection through a dead local proxy.
POLICY_SERVER_HOST="127.0.0.1"
export no_proxy="${no_proxy:+${no_proxy},}${POLICY_SERVER_HOST}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${POLICY_SERVER_HOST}"

# /data/galbot/lib contains the robot SDK, but its libblas is incompatible
# with Ubuntu's system NumPy. Resolve the system BLAS first while still
# keeping the SDK, CUDA and robot libraries available.
export LD_LIBRARY_PATH="/userdata/update/manual_update/lib/galbot_moma/python_env/lib:/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/targets/aarch64-linux/lib:/userdata/lib:/data/galbot/lib"
export PYTHONPATH="/data/galbot/lib${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing project venv: $VENV_PY" >&2
  echo "Recreate it in this directory, then re-run ./run_server.sh" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

# GetActions returns pickled torch.Tensor objects.  Check this here so a
# missing runtime library cannot be misreported later as an empty action chunk.
if ! "$VENV_PY" -c 'import numpy, torch, grpc' >/dev/null 2>&1; then
  echo "Python runtime check failed: numpy, torch or grpc is not importable." >&2
  echo "Use this script (not 'python server.py') so the robot BLAS paths are loaded." >&2
  exit 1
fi

exec "$VENV_PY" server.py "$@"
