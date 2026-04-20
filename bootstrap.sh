#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is not installed or not on PATH" >&2
  exit 1
fi

echo "==> Initializing git submodules"
git submodule update --init --recursive

echo "==> Syncing project environment with uv"
uv sync

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: expected virtualenv interpreter at $PYTHON_BIN" >&2
  exit 1
fi

echo "==> Installing editable submodules into .venv"
uv pip install --python "$PYTHON_BIN" -e openpi
uv pip install --python "$PYTHON_BIN" -e openpi/packages/openpi-client
uv pip install --python "$PYTHON_BIN" -e LIBERO

echo "==> Verifying core imports"
"$PYTHON_BIN" - <<'PY'
import importlib.util
modules = ["openpi", "dreamer_v3", "jaxrl2"]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing imports after bootstrap: {', '.join(missing)}")
print("bootstrap complete")
PY

echo
echo "Environment ready. Activate it with: source .venv/bin/activate"
