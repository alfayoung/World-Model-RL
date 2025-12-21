#!/bin/bash
proj_name=DSRL_pi0_FrankaDroid
device_id=0

export EXP=./logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Fill inFranka Droid camera IDs
export LEFT_CAMERA_ID=""
export RIGHT_CAMERA_ID=""
export WRIST_CAMERA_ID=""

# Fill inpi0 remote host and port
export remote_host=""
export remote_port=""


python3 examples/launch_train_real.py \
--algorithm pixel_sac \
--env franka_droid \
--prefix dsrl_pi0_real \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.99 \
--seed 0 \
--max_steps 500000  \
--eval_interval 2000 \
--log_interval 100 \
--multi_grad_step 30 \
--resize_image 128 \
--action_magnitude 2.5 \
--query_freq 10 \
--hidden_dims 1024 \
--num_qs 2 