#! /usr/bin/env python
"""Training script for Real-World Digital-Twin Double-Q Learning."""

import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib
import copy

import jax
from jaxrl2.agents.pixel_sac.twin_pixel_sac_learner import TwinPixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
import gym_aloha
from gym.spaces import Dict, Box

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_double_q import double_q_training_loop
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

# Latent policy evolution visualization
from latent_policy_viz import (
    CanonicalStateManager,
    LatentPolicyTracker,
    LatentEvolutionPlotter,
    ProprioceptiveTracker,
    ProprioceptivePlotter
)
import matplotlib.pyplot as plt

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):
    """Dummy environment for initialization."""

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            if variant.env == 'libero':
                state_dim = 8
            elif variant.env == 'aloha_cube':
                state_dim = 14
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32)


def main(variant):
    """Main training function for Double-Q learning."""
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)

    # Shard batch dimension across all devices
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # Prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)

    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    variant.checkpoint_dir = os.path.join(outputdir, 'checkpoints')
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    if not os.path.exists(variant.checkpoint_dir):
        os.makedirs(variant.checkpoint_dir)
    print('writing to output dir ', outputdir)

    # ========== Save Variant ==========
    import yaml
    variant_save_path = os.path.join(outputdir, 'variant.yaml')
    with open(variant_save_path, 'w') as f:
        yaml.dump(variant, f)
    print(f"Saved variant configuration to {variant_save_path}")

    # ========== Environment Setup ==========
    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[variant.libero_suite]()
        task_id = variant.task_id
        task = task_suite.get_task(task_id)

        # Create TWO separate environments: real and twin
        env_real, task_description = _get_libero_env(task, 256, variant.seed)
        env_twin, _ = _get_libero_env(task, 256, variant.seed)  # TODO: Different seed for twin
        eval_env = env_real

        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 400
        print(f"Task description: {task_description}")

    elif variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )

        env_real = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        env_twin = env_real  # Twin environment
        eval_env = env_real  # Eval environment

        variant.env_max_reward = 4
        variant.max_timesteps = 400
    else:
        raise NotImplementedError()

    print(f"\n{'='*60}")
    print(f"DOUBLE-Q SETUP: Real and Twin Environments Created")
    print(f"{'='*60}\n")

    # ========== WandB Setup ==========
    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = os.environ.get('WANDB_DIR', tempfile.mkdtemp())
    wandb_logger = WandBLogger(
        variant.prefix != '', variant, variant.wandb_project,
        experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name
    )

    # ========== Agent Initialization ==========
    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)

    # Load π₀ diffusion policy
    if variant.env == 'libero':
        config = openpi_config.get_config("pi0_libero")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    elif variant.env == 'aloha_cube':
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise NotImplementedError()

    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi0 policy from %s", checkpoint_dir)

    # Create Twin Pixel SAC agent
    print(f"\n{'='*60}")
    print(f"Initializing TwinPixelSACLearner (Double-Q)")
    print(f"{'='*60}\n")

    agent = TwinPixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    # ========== Replay Buffers ==========
    # Separate buffers for real and twin data
    buffer_size = variant.max_steps // variant.multi_grad_step

    real_replay_buffer = ReplayBuffer(
        dummy_env.observation_space,
        dummy_env.action_space,
        int(buffer_size)
    )
    real_replay_buffer.seed(variant.seed)

    # Twin buffer can be larger since we collect K trajectories per episode
    # Use initial K_seeds for buffer sizing (buffer capacity is fixed at initialization)
    initial_K_seeds = variant.get('K_seeds', 8)
    twin_buffer_size = buffer_size * initial_K_seeds
    twin_replay_buffer = ReplayBuffer(
        dummy_env.observation_space,
        dummy_env.action_space,
        int(twin_buffer_size)
    )
    twin_replay_buffer.seed(variant.seed + 1)

    print(f"\nReplay Buffers:")
    print(f"  Real buffer capacity: {buffer_size}")
    print(f"  Twin buffer capacity: {twin_buffer_size} (sized for initial K={initial_K_seeds})")

    # Proprioceptive trajectory visualization
    if variant.env == 'libero':
        eef_indices = [0, 1, 2]  # End-effector position (x, y, z)
    elif variant.env == 'aloha_cube':
        eef_indices = None  # Use PCA mode
    else:
        eef_indices = [0, 1, 2]

    proprio_mode = variant.get('proprio_viz_mode', 'eef_pos' if variant.env == 'libero' else 'pca')
    proprio_tracker = ProprioceptiveTracker(max_trajectories=variant.get('proprio_viz_max_trajs', 100))
    proprio_plotter = ProprioceptivePlotter(mode=proprio_mode, eef_indices=eef_indices)
    print(f"Initialized proprioceptive trajectory visualization: mode={proprio_mode}, "
          f"max_trajs={variant.get('proprio_viz_max_trajs', 100)}")

    # ========== Double-Q Training Loop ==========
    print(f"\n{'='*60}")
    print(f"STARTING DOUBLE-Q TRAINING")
    print(f"K_seeds curriculum: {variant.get('K_seeds', 8)} → {variant.get('final_K_seeds', 2)} over {variant.get('k_decay_steps', 100000)} steps")
    print(f"beta_warmup_steps: {variant.get('beta_warmup_steps', 5000)}")
    print(f"beta_max: {variant.get('beta_max', 0.5)}")
    print(f"twin_update_freq: {variant.get('twin_update_freq', 1)}")
    print(f"{'='*60}\n")

    double_q_training_loop(
        variant=variant,
        agent=agent,
        env_real=env_real,
        env_twin=env_twin,
        eval_env=eval_env,
        real_replay_buffer=real_replay_buffer,
        twin_replay_buffer=twin_replay_buffer,
        wandb_logger=wandb_logger,
        shard_fn=shard_fn,
        agent_dp=agent_dp,
        proprio_tracker=proprio_tracker,
        proprio_plotter=proprio_plotter
    )
