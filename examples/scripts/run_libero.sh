#!/bin/bash
proj_name=DSRL_pi0_Libero
task_id=$1
ref_img_path=${2-""}
main_device_id=$3
vlac_device_id=$4

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$main_device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ $main_device_id -eq $vlac_device_id ]; then
    export CUDA_VISIBLE_DEVICES=$main_device_id
    main_device=0
    vlac_device=0
else
    export CUDA_VISIBLE_DEVICES=$main_device_id,$vlac_device_id
    main_device=0
    vlac_device=1
fi

export PYTHONPATH="$PYTHONPATH:/local_data/cf3331/dsrl_pi0/LIBERO"
export WANDB_DIR="/local_data/cf3331/wandb_cache"
export WANDB_API_KEY="fd108789576fafb9526c8bc9f7e06cb9d3f7d373"

uv run examples/launch_train_sim.py \
--algorithm pixel_sac \
--env libero \
--task_id ${task_id} \
--prefix dsrl_pi0_libero \
--wandb_project ${proj_name} \
--main_device ${main_device} \
--vlac_device ${vlac_device} \
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
--hidden_dims 128 \
--use_vlac_rewards true \
--vlac_model_path /local_data/cf3331/vlac_dsrl/VLAC/checkpoint \
--vlac_batch_num 10 \
--vlac_ref_images_path ${ref_img_path}