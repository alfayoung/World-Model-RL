#!/bin/bash
# Run TD-MPC2 (official) on LIBERO-90 task 57.
# Usage: bash run_task57.sh [seed]
set -e
cd /opt/dlami/nvme/cunxin/World-Model-RL-tdmpc2

export DISPLAY=${DISPLAY:-:0}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}
export PYTHONUNBUFFERED=1

SEED=${1:-1}

mkdir -p logs

# train.py must be run from the tdmpc2/ subdirectory (Hydra finds config.yaml there)
cd tdmpc2
uv run train.py \
  --config-name config_libero \
  task=libero-90-57 \
  seed=${SEED} \
  work_dir=/opt/dlami/nvme/cunxin/World-Model-RL-tdmpc2/logs/tdmpc2_task57_seed${SEED}
