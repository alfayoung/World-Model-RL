You can see each task in [here](/third_party/LIBERO/libero/libero/bddl_files/) defined in bddl.

## Digital Twin Creation
### Part 1. Reconstruction

1. Rescale

We can clearly see the size of the table is (1.0, 1.2, 0.05) in meters from source code.

By using the `Measure` button on the left ![alt text](/docs/assets/image.png), we can infer the size is, for example: (3.35, 4.0).

Press `n` we see the transform and set the ratio is (1.2 / 4.0)

2. Set Rotation

By using the button `Rotate` on the left bar, you can set the rotation of the object.
you may watch the video [here](https://www.youtube.com/watch?v=y6nwGRkL1k4) for tutorial.

3. Set Translation

By using the button `Cursor` on the left bar, select the origin of the object you like. Use Shift+S and select “Cursor to Selected.”

4. Export point cloud

Before exporting, Ctrl+A or Object ‣ Apply ‣ Location / Rotation / Scale / Rotation & Scale to apply transform **locally**.

[Currently Doesn't Work] Cropping Noise using differnece between cube. [here](https://www.youtube.com/watch?v=B6mQoGmNYLc)

### Part 2. Object Tracking

1. Scanning target object

You may directly find them in [here](/third_party/LIBERO/libero/libero/assets) according to task configurations in [here](/third_party/LIBERO/libero/libero/bddl_files/). Retrieve the `.obj` file of the target object.

For example I use `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl`. The target object is [here](/local_data/cf3331/X-Sim/third_party/LIBERO/libero/libero/assets/stable_scanned_objects/akita_black_bowl/akita_black_bowl.obj).

2. Track

Please download demo data from [here](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets). For example in our use case,
```shell
hf download yifengzhu-hf/LIBERO-datasets --repo-type dataset --include "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5" --local-dir .
```
Then run
```shell
bash scripts/track.sh 1
```

Foundation Pose
```shell
# create conda environment
conda create -n foundationpose python=3.9 -y

# activate conda environment
conda activate foundationpose

# gcc
conda install -c conda-forge gcc=11 gxx=11 boost -y

# cuda toolkit
conda install cuda-toolkit -c nvidia/label/cuda-11.8.0 -y

# Install Eigen3 3.4.0 under conda environment
conda install eigen=3.4.0 -c conda-forge -y
sed -i "s#/usr/local/include/#$CONDA_PREFIX/include/#" bundlesdf/mycuda/setup.py

# install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Downgrade setuptools to avoid build issues
pip uninstall setuptools -y && pip install setuptools==69.5.1

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```

```shell
conda install xorg-libx11 xorg-libxcb xorg-libxau xorg-libxdmcp xorg-libxrender xorg-libxext qt-main -c conda-forge -y
ssh -Y user@your_server_address # for interactive access
```

```shell
cp ./real_to_sim/fp_objects.py ./real_to_sim/generate_mask.py  third_party/FoundationPose/ # Run it once to copy necessary files
conda activate foundationpose
bash scripts/track.sh 2
```

## RL training in Digital Twin

### Part 1. Register Code

Register the in the following [directory](/simulation/ManiSkill/mani_skill/assets/xsim/<env>/<task>/)

```shell
cd simulation
python -m scripts.rl_training \
--env_id="<TASK_NAME>" \
--exp_name="<EXPERIMENT_NAME>" \
--num_envs=1024 \
--seed=0 \
--total_timesteps=<TIMESTEPS> \
--num_steps=<STEPS> \
--num_eval_steps=<EVAL_STEPS>
```

### Part 2. 