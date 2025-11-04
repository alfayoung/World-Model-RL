debug() {
    python -m debugpy --listen 5678 --wait-for-client "$@"
}

step=$1

if [ "$step" == 1 ]; then
    echo "Step 1: Collect simulation demonstration data"
    python scripts/collect_sim_demo.py \
    --save_dir ./sim_demo_data \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /local_data/cf3331/X-Sim/datasets/libero_spatial/ \
    --task_id 0 \
    --demo_id 0 \
    --camera_name agentview \
    --resolution 512 \
    --filter_noops

elif [ "$step" == 2 ]; then
    echo "Step 2: Perform object pose estimation from collected data"
    cd third_party/FoundationPose/
    # Show a dummy image to work around GUI issues
    # On local machine, you have to manually run the following command
    # {rgbd_frames_directory}/rgb/frame_0000.png
    # open with windows drawer to get (min_x, min_y, max_x, max_y)
    
    # ln -s lib -> lib64
    python fp_objects.py \
        --rgbd_frames_directory /local_data/cf3331/X-Sim/sim_demo_data \
        --object_meshes /local_data/cf3331/X-Sim/third_party/LIBERO/libero/libero/assets/stable_scanned_objects/akita_black_bowl/akita_black_bowl.obj \
        --camera_to_robot /local_data/cf3331/X-Sim/sim_demo_data/camera_to_robot.npy \
        --first_frame_bbox 83 275 160 339

else
    echo "Please provide a valid step number (1 or 2)"
    exit 1
fi

