#!/bin/bash
# Run TD-MPC2 baseline on a single LIBERO task.
# Usage: bash examples/scripts/run_tdmpc2_libero.sh <task_id> <device_id>

proj_name=TDMPC2_Libero
task_id=${1:-0}
device_id=${2:-0}

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export PYTHONPATH="/opt/dlami/nvme/cunxin/World-Model-RL-env/third_party/LIBERO:${PYTHONPATH}"
# tdmpc2 module lives in the worktree root
export PYTHONPATH="$(pwd):${PYTHONPATH}"

PYTHON=/home/ubuntu/miniconda3/envs/xsim/bin/python

$PYTHON examples/launch_train_tdmpc2.py \
    --env libero \
    --libero_suite libero_90 \
    --task_id ${task_id} \
    --prefix tdmpc2_libero \
    --wandb_project ${proj_name} \
    --batch_size 256 \
    --discount 0.99 \
    --seed 0 \
    --max_steps 500000 \
    --eval_interval 10000 \
    --log_interval 500 \
    --eval_episodes 10 \
    --resize_image 64 \
    --add_states 1 \
    --horizon 5 \
    --num_samples 512 \
    --num_elites 64 \
    --num_pi_trajs 24 \
    --multi_grad_step 1 \
    --start_online_updates 500 \
    --latent_dim 256 \
    --hidden_dim 256 \
    --num_q 5 \
    --consistency_coef 20.0 \
    --reward_coef 0.5 \
    --value_coef 0.1 \
    --device cuda
