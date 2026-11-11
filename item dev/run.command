#!/bin/bash
set -eu

ITEMDEV_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$ITEMDEV_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python was not found. Install Python 3.10 or newer, or set ITEMDEV_PYTHON to an existing Python executable."
  exit 1
fi

itemdev_runtime_is_healthy() {
  command -v "$1" >/dev/null 2>&1 || return 1
  python3 - "$1" <<'PY'
import subprocess
import sys
try:
    check = subprocess.run(
        [sys.argv[1], "-c", "import streamlit, pandas"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
except (OSError, subprocess.TimeoutExpired):
    sys.exit(1)
sys.exit(0 if check.returncode == 0 else 1)
PY
}

ITEMDEV_RUNTIME=""
echo "Checking the local Python environment (up to 10 seconds per candidate)."
if [ -n "${ITEMDEV_PYTHON:-}" ]; then
  if itemdev_runtime_is_healthy "$ITEMDEV_PYTHON"; then
    ITEMDEV_RUNTIME="$ITEMDEV_PYTHON"
  fi
else
  for ITEMDEV_CANDIDATE in "$ITEMDEV_DIR/../.venv/bin/python" python3; do
    if itemdev_runtime_is_healthy "$ITEMDEV_CANDIDATE"; then
      ITEMDEV_RUNTIME="$ITEMDEV_CANDIDATE"
      break
    fi
  done
fi

if [ -z "$ITEMDEV_RUNTIME" ]; then
  echo "No tested Python environment could import Streamlit and pandas within the health-check timeout."
  echo "Use the environment setup instructions in README.md, then run this launcher with ITEMDEV_PYTHON set to that environment."
  echo "Nothing has been installed or changed automatically."
  exit 1
fi

if [ ! -f "$ITEMDEV_DIR/app.py" ]; then
  echo "The application entry point app.py is not present yet."
  exit 1
fi

echo "Starting the local Item Development workspace."
echo "Open http://127.0.0.1:8526 in your browser. Press Control-C here to stop."
exec "$ITEMDEV_RUNTIME" -m streamlit run "$ITEMDEV_DIR/app.py" --server.address 127.0.0.1 --server.port 8526 --server.headless true "$@"
