<div align="center">

# DSRL for π₀: Diffusion Steering via Reinforcement Learning

## [[website](https://diffusion-steering.github.io)]      [[paper](https://arxiv.org/abs/2506.15799)]

</div>


## Overview
This repository provides the official implementation for our paper: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799) (CoRL 2025).

Specifically, it contains a JAX-based implementation of DSRL (Diffusion Steering via Reinforcement Learning) for steering a pre-trained generalist policy, [π₀](https://github.com/Physical-Intelligence/openpi), across various environments, including:

- **Simulation:** Libero, Aloha  
- **Real Robot:** Franka

If you find this repository useful for your research, please cite:

```
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

## UV Installation

Preferred:

```bash
./bootstrap.sh
source .venv/bin/activate
```

Equivalent manual steps:

```bash
# init submodules
git submodule update --init --recursive

uv sync
source .venv/bin/activate

# install openpi
uv pip install --python .venv/bin/python -e openpi
uv pip install --python .venv/bin/python -e openpi/packages/openpi-client

# install Libero
uv pip install --python .venv/bin/python -e LIBERO
```

Notes:

- The submodules are required. On a fresh clone, `uv pip install -e openpi` and `uv pip install -e LIBERO` fail until `git submodule update --init --recursive` has completed.
- If you already have another virtual environment active, bare `uv pip install ...` may target that environment instead of this repo's `.venv`. Using `--python .venv/bin/python` avoids that.
- After the editable installs above, prefer running commands from the activated `.venv` (`python ...`, `bash ...`). `uv run` re-syncs the environment to `pyproject.toml` and `uv.lock`, and can remove manually installed packages that are not declared there, such as `LIBERO`.

## Conda Installation
1. Create a conda environment:
```bash
conda create -n dsrl_pi0 python=3.11.11
conda activate dsrl_pi0
```

2. Init with all submodules
```bash
git submodule update --init --recursive
```

3. Install all packages and dependencies
```bash
pip install -e .
pip install -r requirements.txt
pip install "jax[cuda12]==0.5.0"

# install openpi
pip install -e openpi
pip install -e openpi/packages/openpi-client

# install Libero
pip install -e LIBERO
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu # needed for libero
```

## Training (Simulation)
Libero
```
bash examples/scripts/run_libero.sh
```
Aloha
```
bash examples/scripts/run_aloha.sh
```
### Training Logs
We provide sample W&B runs and logs: https://wandb.ai/mitsuhiko/DSRL_pi0_public

## Training (Real)
For real-world experiments, we use the remote hosting feature from pi0 (see [here](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)) which enables us to host the pi0 model on a higher-spec remote server, in case the robot's client machine is not powerful enough. 

0. Setup Franka robot and install DROID package [[link](https://github.com/droid-dataset/droid.git)]

1. [On the remote server] Host pi0 droid model on your remote server
```
cd openpi && python scripts/serve_policy.py --env=DROID
```
2. [On your robot client machine] Run DSRL
```
bash examples/scripts/run_real.sh
```


## Credits
This repository is built upon [jaxrl2](https://github.com/ikostrikov/jaxrl2) and [PTR](https://github.com/Asap7772/PTR) repositories. 
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at nakamoto\[at\]berkeley\[dot\]edu 
