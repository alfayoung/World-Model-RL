#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run Dreamer-v3 baseline on a LIBERO task.
#
# Usage:
#   bash examples/scripts/run_dreamerv3_libero.sh \
#       --task_id 0 --seed 42 [--dreamer_size size12m] [...]
#
# All flags are forwarded to launch_train_dreamerv3.py.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- GPU / rendering setup (same as other LIBERO scripts) -----------------
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"

# Silence TF GPU chatter
export TF_CPP_MIN_LOG_LEVEL=3

# Point EXP at a sensible default if not already set
if [ -z "${EXP:-}" ]; then
    export EXP="$HOME/exp/dreamerv3"
    mkdir -p "$EXP"
fi

# JAX compilation cache (same as train_sim.py)
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_triton_gemm_any=True"

# ---- Run ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

python -m examples.launch_train_dreamerv3 "$@"
