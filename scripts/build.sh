# First activate the installed conda environment
# git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive
# git submodule update --init --recursive
# conda clean -p1I
# conda env create --file environment.yml
# conda activate surfel_splatting

# LIBERO Installation Guide:
# pip install robosuite
# git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
# cd third_party/LIBERO && pip install -e .

step="$1"  # step number
args="$2"  # additional arguments if needed, eg. "interactive" or "panoramic"

dataset_dir="/opt/dlami/nvme/cunxin/World-Model-RL-env/datasets/2dgs_live/libero_spatial_t0_panorama_init_20260413"
# dataset_dir="/local_data/cf3331/X-Sim/datasets/tiger_sfm/"

# Step1: Collect images from the environment
if [ "$step" == 1 ]; then
    echo "Step 1: Collect images from the environment"
    if [ "$args" == "panoramic" ]; then
        python scripts/image_collect.py --task_suite_name libero_spatial \
        --task_id 0 \
        --mode panoramic \
        --camera_name frontview \
        --panoramic_yaw_angles 18 \
        --panoramic_pitch_angles [0] \
        --multiple_radii [0.8,1.0,1.2] \
        --height_variations [0.0] \
        --use_radius_variation True \
        --use_height_variation False \
        --look_at_offset_std 0.0 \
        --output_dir "$dataset_dir/input"
    else
        python scripts/image_collect.py --task_suite_name libero_spatial \
        --task_id 0 \
        --mode interactive \
        --output_dir "$dataset_dir/input"
    fi

# Step2: Inspect the existing sparse model and export a preview PLY
elif [ "$step" == 2 ]; then
    echo "Step 2: Inspect sparse model and export preview PLY"
    # python third_party/2d-gaussian-splatting/convert.py \
    # -s "$dataset_dir"
    # 1. analyze the generated COLMAP model
    colmap model_analyzer --path "$dataset_dir/sparse/0"
    # 2. check sparse/0/points3D.ply if the sparse point cloud is generated correctly
    colmap model_converter \
    --input_path "$dataset_dir/sparse/0/" \
    --output_path "$dataset_dir/points3D.ply" \
    --output_type PLY

# Step3: Train the 2D Gaussian Splatting model
elif [ "$step" == 3 ]; then
    echo "Step 3: Train the 2D Gaussian Splatting model"
    cd third_party/2d-gaussian-splatting
    python train.py \
    -s "$dataset_dir" \
    --iterations 400000 \
    --checkpoint_iterations 100000 200000 300000 400000

# Step4: Render images from the trained model
elif [ "$step" == 4 ]; then
    echo "Step 4: Render images from the trained model"
    model_path=/opt/dlami/nvme/cunxin/World-Model-RL-env/third_party/2d-gaussian-splatting/output/47c817ef-4
    echo "Have you changed the model path in the command below? $model_path"
    read -p "Press enter to confirm"
    python third_party/2d-gaussian-splatting/render.py \
    -m $model_path \
    -s "$dataset_dir" \
    --depth_trunc 5.0 \
    --mesh_res 1024

else
    echo "Invalid step. Please provide step 1, 2, 3, or 4."

fi
