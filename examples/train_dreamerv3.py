"""Train Dreamer-v3 on LIBERO using frozen π₀ as action executor.

The world-model learner outputs a 32-dim noise vector that is fed into the
frozen OpenPI π₀ diffusion policy to produce the 7-dim LIBERO robot action.
This is identical to the DSRL / PixelSAC action space on main, so the only
variable between the two methods is the learner (world model vs. model-free).

Training follows the trajectory-wise loop from train_utils_sim.py, adapted
for Dreamer-v3's stateful RSSM (which requires episode-level sequences).
"""
from __future__ import annotations

import os
import pathlib
import tempfile

# XLA tuning (same as train_sim.py)
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import math
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download
from openpi_client import image_tools

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from jax.experimental.compilation_cache import compilation_cache

from dreamer_v3.agent import DreamerV3Learner
from dreamer_v3.replay import EpisodeReplayBuffer

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))


# ---------------------------------------------------------------------------
# Environment helpers (copied from train_utils_sim.py)
# ---------------------------------------------------------------------------

def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(variant):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[variant.libero_suite]()
    task = task_suite.get_task(variant.task_id)
    task_description = task.language
    bddl_file = (pathlib.Path(get_libero_path("bddl_files"))
                 / task.problem_folder / task.bddl_file)
    env_args = {
        "bddl_file_name": str(bddl_file),
        "camera_heights": 256,
        "camera_widths":  256,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(variant.seed)
    return env, task_description


def _preprocess_obs(raw_obs: dict, variant, image_size: int) -> dict:
    """Extract (image, state) for the Dreamer-v3 agent."""
    import PIL.Image
    img = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])  # flip
    img = np.array(PIL.Image.fromarray(img).resize((image_size, image_size)))
    state = np.concatenate([
        raw_obs["robot0_eef_pos"],
        _quat2axisangle(raw_obs["robot0_eef_quat"]),
        raw_obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    # Build DSRL-compatible obs dict for agent.sample_actions()
    obs_dict = {
        'pixels': img[np.newaxis, ..., np.newaxis],  # (1, H, W, 3, 1)
        'state':  state[np.newaxis, ..., np.newaxis], # (1, D, 1)
    }
    return obs_dict, img, state


def _build_pi0_obs(raw_obs: dict, task_description: str) -> dict:
    """Build the observation dict expected by π₀."""
    img   = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img   = image_tools.convert_to_uint8(image_tools.resize_with_pad(img,   224, 224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
    state = np.concatenate([
        raw_obs["robot0_eef_pos"],
        _quat2axisangle(raw_obs["robot0_eef_quat"]),
        raw_obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    return {
        "observation/image":       img,
        "observation/wrist_image": wrist,
        "observation/state":       state,
        "prompt":                  task_description,
    }


# ---------------------------------------------------------------------------
# Episode collection (stateful RSSM variant)
# ---------------------------------------------------------------------------

def collect_episode(variant, agent: DreamerV3Learner, env, step: int,
                    pi0_policy, task_description: str, image_size: int):
    """Roll out one full episode and return collected data.

    Returns a dict with:
      images, states, actions, rewards, dones  (for the replay buffer)
      is_success, episode_return, env_steps
    """
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    query_frequency = variant.query_freq

    # Reset RSSM state at episode start
    agent.reset_state()
    raw_obs = env.reset()

    images, states, actions, rewards, dones = [], [], [], [], []
    images_for_pi0 = []

    rng = jax.random.PRNGKey(variant.seed + step)
    is_success = False
    ep_ret = 0.0
    pi0_actions = None  # cached action chunk from π₀

    for t in range(max_timesteps):
        obs_dict, img, state = _preprocess_obs(raw_obs, variant, image_size)

        # ---- Dreamer-v3 action → noise for π₀ ----------------------------
        if step == 0 and t == 0:
            # Initial collection: random noise (matches DSRL warm-up)
            noise_action = np.random.randn(1, variant.action_dim).astype(np.float32)
        else:
            if t % query_frequency == 0:
                noise_action = agent.sample_actions(obs_dict)   # (1, A)

        # ---- Build π₀ call -----------------------------------------------
        if t % query_frequency == 0:
            obs_pi0 = _build_pi0_obs(raw_obs, task_description)
            noise_32 = noise_action[0]                               # (32,)
            noise_1  = noise_32[None, None, :]                       # (1, 1, 32)
            noise_tail = np.zeros((1, 49, 32), dtype=np.float32)
            noise = np.concatenate([noise_1, noise_tail], axis=1)   # (1, 50, 32)
            pi0_actions = pi0_policy.infer(obs_pi0, noise=noise)["actions"]  # (N, 7)

        robot_action = pi0_actions[t % query_frequency]
        raw_next_obs, reward, done, info = env.step(robot_action)

        images.append(img)
        states.append(state)
        actions.append(noise_action[0])  # store the noise action
        rewards.append(float(reward))
        dones.append(bool(done))
        ep_ret += float(reward)

        raw_obs = raw_next_obs
        if done:
            is_success = (reward >= env_max_reward)
            break

    # Final observation (terminal state)
    _, img_final, state_final = _preprocess_obs(raw_obs, variant, image_size)
    images.append(img_final)
    states.append(state_final)

    # Sparse reward reshape (same as DSRL)
    T = len(actions)
    if is_success:
        rewards = [-1.0] * (T - 1) + [0.0]
    else:
        rewards = [-1.0] * T
    dones_sparse = [False] * (T - 1) + [True]

    return {
        'images':         images,
        'states':         states,
        'actions':        actions,
        'rewards':        np.array(rewards, dtype=np.float32),
        'dones':          np.array(dones_sparse, dtype=np.float32),
        'is_success':     is_success,
        'episode_return': ep_ret,
        'env_steps':      T,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def perform_control_eval(variant, agent: DreamerV3Learner, env, step: int,
                         pi0_policy, task_description: str,
                         image_size: int, wandb_logger, num_episodes: int = 10):
    success_rates = []
    ep_returns = []
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    for ep_idx in range(num_episodes):
        agent.reset_state()
        raw_obs = env.reset()
        ep_ret = 0.0
        is_success = False
        pi0_actions = None

        for t in range(max_timesteps):
            obs_dict, _, _ = _preprocess_obs(raw_obs, variant, image_size)
            if t % variant.query_freq == 0:
                noise_action = agent.eval_actions(obs_dict)
                obs_pi0 = _build_pi0_obs(raw_obs, task_description)
                noise_32 = noise_action[0]
                noise_1  = noise_32[None, None, :]
                noise_tail = np.zeros((1, 49, 32), dtype=np.float32)
                noise = np.concatenate([noise_1, noise_tail], axis=1)
                pi0_actions = pi0_policy.infer(obs_pi0, noise=noise)["actions"]

            robot_action = pi0_actions[t % variant.query_freq]
            raw_obs, reward, done, info = env.step(robot_action)
            ep_ret += float(reward)
            if done:
                is_success = (reward >= env_max_reward)
                break

        success_rates.append(float(is_success))
        ep_returns.append(ep_ret)

    metrics = {
        'eval/success_rate':    float(np.mean(success_rates)),
        'eval/episode_return':  float(np.mean(ep_returns)),
    }
    wandb_logger.log(metrics, step=step)
    print(f"[eval @{step}]",
          " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(variant):
    # Prevent TF from grabbing GPU memory.
    tf.config.set_visible_devices([], "GPU")

    seed = variant.seed
    np.random.seed(seed)

    # ---- Experiment setup ---------------------------------------------------
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]
    expname = create_exp_name(variant.prefix, seed=seed)
    if variant.get("suffix", ""):
        expname = expname + f"_{variant.suffix}"
    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    os.makedirs(outputdir, exist_ok=True)
    print('writing to output dir', outputdir)

    # ---- LIBERO env ---------------------------------------------------------
    env, task_description = _get_libero_env(variant)
    variant.task_description = task_description
    variant.env_max_reward = 1.0
    variant.max_timesteps = 400
    variant.action_dim = 32
    image_size = variant.get('image_size', 64)
    print(f"Task: {task_description}")

    # ---- Load frozen π₀ policy ----------------------------------------------
    pi0_config   = openpi_config.get_config("pi0_libero")
    checkpoint   = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    pi0_policy   = policy_config.create_trained_policy(pi0_config, checkpoint)
    print("Loaded π₀ policy from", checkpoint)

    # ---- WandB logger -------------------------------------------------------
    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = os.environ.get('WANDB_DIR', tempfile.mkdtemp())
    wandb_logger = WandBLogger(
        variant.prefix != '',
        variant,
        variant.wandb_project,
        experiment_id=expname,
        output_dir=wandb_output_dir,
        group_name=group_name,
    )

    # ---- Dreamer-v3 agent ---------------------------------------------------
    train_kwargs = variant.get('train_kwargs', {})
    agent = DreamerV3Learner(
        seed=seed,
        image_size=image_size,
        state_dim=8,
        action_dim=32,
        encoder_depth=train_kwargs.get('encoder_depth', 32),
        encoder_embed=train_kwargs.get('encoder_embed', 1024),
        deter=train_kwargs.get('deter', 1024),
        stoch=train_kwargs.get('stoch', 32),
        classes=train_kwargs.get('classes', 32),
        mlp_units=train_kwargs.get('mlp_units', 512),
        mlp_layers=train_kwargs.get('mlp_layers', 3),
        lr=train_kwargs.get('lr', 3e-4),
        discount=train_kwargs.get('discount', variant.get('discount', 0.997)),
        imag_horizon=train_kwargs.get('imag_horizon', 15),
        lam=train_kwargs.get('lam', 0.95),
        kl_free=train_kwargs.get('kl_free', 1.0),
        add_states=bool(variant.add_states),
    )
    print("Initialised DreamerV3Learner")

    # ---- Sequence replay buffer ---------------------------------------------
    replay = EpisodeReplayBuffer(
        capacity=variant.get('replay_capacity', 1_000_000),
        seq_len=train_kwargs.get('batch_length', 64),
        image_size=image_size,
        state_dim=8,
        action_dim=32,
    )

    # ---- Training loop -------------------------------------------------------
    max_steps         = variant.max_steps
    batch_size        = variant.batch_size
    multi_grad_step   = variant.multi_grad_step
    start_updates     = variant.start_online_updates
    eval_interval     = variant.eval_interval
    log_interval      = variant.log_interval
    eval_episodes     = variant.get('eval_episodes', 10)
    checkpoint_interval = variant.get('checkpoint_interval', -1)
    seq_len           = train_kwargs.get('batch_length', 64)

    global_step = 0
    total_env_steps = 0
    episode_count = 0

    wandb_logger.log({'num_online_samples': 0, 'env_steps': 0}, step=0)

    while global_step <= max_steps:
        # ---- Collect one episode -------------------------------------------
        ep_data = collect_episode(
            variant, agent, env, global_step, pi0_policy,
            task_description, image_size)
        episode_count += 1
        total_env_steps += ep_data['env_steps']

        replay.add_episode(
            images  = ep_data['images'],
            states  = ep_data['states'],
            actions = ep_data['actions'],
            rewards = ep_data['rewards'],
            dones   = ep_data['dones'],
        )

        print(f"[ep {episode_count}] return={ep_data['episode_return']:.3f}  "
              f"success={ep_data['is_success']}  "
              f"env_steps={total_env_steps}")

        # ---- Gradient updates -----------------------------------------------
        num_grad_steps = ep_data['env_steps'] * multi_grad_step
        if len(replay) > start_updates:
            for _ in range(num_grad_steps):
                batch = replay.sample(batch_size)
                if batch is None:
                    break

                # Convert to JAX arrays
                jax_batch = {
                    'image':    jnp.array(batch['image']),
                    'state':    jnp.array(batch['state']),
                    'action':   jnp.array(batch['action']),
                    'reward':   jnp.array(batch['reward']),
                    'done':     jnp.array(batch['done']),
                    'is_first': jnp.array(batch['is_first']),
                }
                update_info = agent.update(jax_batch)

                global_step += 1

                if global_step % log_interval == 0:
                    wandb_logger.log(
                        {
                            **{f'training/{k}': v for k, v in update_info.items()},
                            'episode_return': ep_data['episode_return'],
                            'is_success':     int(ep_data['is_success']),
                            'replay_buffer_size': len(replay),
                            'env_steps':      total_env_steps,
                        },
                        step=global_step,
                    )
                    print(f"[step {global_step}]",
                          " | ".join(f"{k}={v:.4f}"
                                     for k, v in update_info.items()))

                if global_step % eval_interval == 0:
                    wandb_logger.log(
                        {'num_online_samples': len(replay),
                         'env_steps': total_env_steps},
                        step=global_step)
                    perform_control_eval(
                        variant, agent, env, global_step,
                        pi0_policy, task_description,
                        image_size, wandb_logger, num_episodes=eval_episodes)

                if (checkpoint_interval > 0 and
                        global_step % checkpoint_interval == 0):
                    agent.save_checkpoint(outputdir, global_step, checkpoint_interval)

                if global_step >= max_steps:
                    break

    wandb_logger.finish()
    env.close()
