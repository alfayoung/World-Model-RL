#!/bin/bash
# Script to run Real-World Digital-Twin Double-Q Learning with LIBERO
# Implements the algorithm from IDEA.md

debug=false

proj_name=DSRL_pi0_DoubleQ_Libero
task_id=$1
device_id=$2

echo "I am working in $PWD..."

export WANDB_PROJECT=${proj_name}
export WANDB_API_KEY="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"
export WANDB_MODE=online

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=$PWD/logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export PYTHONPATH="$PWD/LIBERO:$PYTHONPATH"
export WANDB_DIR="$PWD/wandb_cache"

pip install mujoco==3.3.1

# taskid 33, 57


if [ "$debug" = true ] ; then
    debug() {
        echo "Start debugpy on port 5678. Waiting for server to connect..."
        python -m debugpy --listen 5678 --wait-for-client "$@"
    }

    debug examples/launch_train_double_q.py \
    --algorithm twin_pixel_sac \
    --env libero \
    --libero_suite libero_90 \
    --task_id 57 \
    --prefix dsrl_double_q_libero \
    --wandb_project ${proj_name} \
    --batch_size 8 \
    --start_online_updates 10 \
    --discount 0.999 \
    --seed 0 \
    --max_steps 500000 \
    --eval_interval 1 \
    --log_interval 500 \
    --eval_episodes 10 \
    --checkpoint_interval 10000 \
    --multi_grad_step 20 \
    --resize_image 64 \
    --action_magnitude 1.0 \
    --query_freq 20 \
    --hidden_dims 128 \
    --K_seeds 4 \
    --beta_warmup_steps 10000 \
    --beta_max 0.5 \
    --twin_update_freq 1
elif [ $task_id = 57 ] || [ $task_id = 6 ] ; then
    echo "Running easy tasks"
    uv run examples/launch_train_double_q.py \
    --algorithm twin_pixel_sac \
    --env libero \
    --libero_suite libero_90 \
    --task_id ${task_id} \
    --prefix dsrl_double_q_libero \
    --wandb_project ${proj_name} \
    --batch_size 256 \
    --start_online_updates 500 \
    --discount 0.999 \
    --seed 0 \
    --max_steps 500000 \
    --eval_interval 10000 \
    --log_interval 500 \
    --eval_episodes 50 \
    --checkpoint_interval 100000 \
    --multi_grad_step 20 \
    --resize_image 64 \
    --action_magnitude 1.0 \
    --query_freq 20 \
    --hidden_dims 128 \
    --K_seeds 5 \
    --final_K_seeds 1 \
    --k_decay_steps 100000 \
    --beta_warmup_steps 100000 \
    --beta_max 1.0 \
    --twin_update_freq 1
else
    echo "Running hard tasks"
    uv run examples/launch_train_double_q.py \
    --algorithm twin_pixel_sac \
    --env libero \
    --libero_suite libero_90 \
    --task_id ${task_id} \
    --prefix dsrl_double_q_libero_ablate_beta_1e5 \
    --wandb_project ${proj_name} \
    --batch_size 256 \
    --start_online_updates 500 \
    --discount 0.999 \
    --seed 0 \
    --max_steps 500000 \
    --eval_interval 10000 \
    --log_interval 500 \
    --eval_episodes 50 \
    --checkpoint_interval 100000 \
    --multi_grad_step 20 \
    --resize_image 64 \
    --action_magnitude 1.0 \
    --query_freq 20 \
    --hidden_dims 128 \
    --K_seeds 5 \
    --final_K_seeds 1 \
    --k_decay_steps 100000 \
    --beta_warmup_steps 100000 \
    --beta_max 1.0 \
    --twin_update_freq 1
fi