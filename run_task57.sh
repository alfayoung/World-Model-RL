#!/bin/bash
set -e
cd /opt/dlami/nvme/cunxin/World-Model-RL-tdmpc2

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p logs

exec uv run python -u \
  examples/launch_train_tdmpc2.py \
  --env libero --libero_suite libero_90 --task_id 57 \
  --prefix tdmpc2_libero --wandb_project TDMPC2_Libero \
  --batch_size 256 --discount 0.99 --seed 0 --max_steps 500000 \
  --eval_interval 10000 --log_interval 500 --eval_episodes 10 \
  --resize_image 64 --add_states 1 \
  --horizon 5 --num_samples 512 --num_elites 64 --num_pi_trajs 24 \
  --multi_grad_step 1 --start_online_updates 500 \
  --latent_dim 256 --hidden_dim 256 --num_q 5 \
  --consistency_coef 20.0 --reward_coef 0.5 --value_coef 0.1 \
  --use_wandb 0 --device cuda
