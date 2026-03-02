#!/bin/bash
proj_name=DSRL_pi0_Libero
task_id=$1
device_id=${2:-0}

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=$device_id

export PYTHONPATH="$PYTHONPATH:/local_data/cf3331/dsrl_pi0/LIBERO"
export WANDB_DIR="/local_data/cf3331/wandb_cache"
export WANDB_API_KEY="fd108789576fafb9526c8bc9f7e06cb9d3f7d373"

uv run examples/launch_train_sim.py \
--algorithm pixel_sac \
--env libero \
--task_id ${task_id} \
--prefix dsrl_pi0_libero \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 1.0 \
--seed 0 \
--max_steps 500000  \
--eval_interval 10000 \
--log_interval 500 \
--eval_episodes 50 \
--multi_grad_step 20 \
--start_online_updates 500 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 20 \
--hidden_dims 128